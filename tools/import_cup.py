#!/usr/bin/env python3
"""Import one FBC event into FBC_Data.xlsx from the scoring app's CSV export.

Usage:
    python tools/import_cup.py fbc13_archives_export.csv            # update FBC_Data.xlsx in place
    python tools/import_cup.py export.csv --dry-run                  # report only, write nothing
    python tools/import_cup.py export.csv --out /tmp/preview.xlsx    # write elsewhere

What it does
    1. Appends the CSV rows to the bottom of the Archives sheet.
    2. Adds an "FBC <n>" column to the Cups sheet (before Total), marks every
       player 1 / 0 / X from the event's rows, bumps Played, and extends the
       Total / Win% / Lost formulas so they include the new column. A player
       not yet on the Cups sheet gets a new row above Total.
    3. Runs the app's Data Health check on the result and refuses to write
       the workbook if anything is flagged (override with --force).

Why it edits the file the way it does
    The workbook has 28 tabs of formulas, cached values and charts. Libraries
    that load-and-save the whole workbook drop the cached values and can drop
    charts, so this script edits only the XML parts it has to (Archives sheet,
    Cups sheet, RecordBook formulas that point at Cups, the defined names)
    and copies every other part of the file byte for byte. Excel rebuilds
    the calculation chain on next open, so that part is removed.

The CSV is the scoring app's "Export CSV" (Admin footer): a header row and
one row per Archives line, with a blank leading column so a paste lands at
A1. The FBC number, the two captains and the winner are all read from it.
"""
import argparse
import csv
import datetime as dt
import io
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import OrderedDict
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------------------------------------------------------- helpers

def col_to_idx(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n


def idx_to_col(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


REF_RE = re.compile(r'(?<![A-Za-z_!])(\$?)([A-Z]{1,3})(\$?)(\d+)(?![A-Za-z_(\d])')


def shift_refs(text, col_at=None, row_at=None, sheet_prefix=None):
    """Shift A1 references in a formula/ref string: columns >= col_at move
    right by one, rows >= row_at move down by one. With sheet_prefix, only
    references written as <prefix>!A1 (or <prefix>!A1:B2) are shifted."""
    def one(m):
        c_abs, col, r_abs, row = m.groups()
        ci, ri = col_to_idx(col), int(row)
        if col_at and ci >= col_at:
            ci += 1
        if row_at and ri >= row_at:
            ri += 1
        return f"{c_abs}{idx_to_col(ci)}{r_abs}{ri}"

    if sheet_prefix is None:
        return REF_RE.sub(one, text)

    # Only touch refs that follow "<prefix>!" (a range's second half follows a
    # ':' immediately after a shifted first half, so handle the pair together).
    pat = re.compile(re.escape(sheet_prefix) + r'!(\$?[A-Z]{1,3}\$?\d+)(:\$?[A-Z]{1,3}\$?\d+)?')

    def pair(m):
        first = REF_RE.sub(one, m.group(1))
        second = REF_RE.sub(one, m.group(2)) if m.group(2) else ''
        return f"{sheet_prefix}!{first}{second}"

    return pat.sub(pair, text)


def excel_serial(d):
    return (d - dt.datetime(1899, 12, 30)).days


def cell_xml(ref, value, style=None):
    """One <c> element for a literal value; None -> no cell."""
    if value is None or value == '':
        return ''
    s = f' s="{style}"' if style is not None else ''
    if isinstance(value, dt.datetime):
        return f'<c r="{ref}"{s}><v>{excel_serial(value)}</v></c>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{s} t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{s}><v>{value!r}</v></c>'
    return f'<c r="{ref}"{s} t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>'


def parse_value(text):
    """CSV cell -> what Excel would make of a paste: number, date, or text."""
    if text is None or text == '':
        return None
    t = text.strip()
    if re.fullmatch(r'-?\d+', t):
        return int(t)
    if re.fullmatch(r'-?\d*\.\d+', t):
        return float(t)
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', t):
        return dt.datetime.strptime(t, '%Y-%m-%d')
    return t


class SharedStrings:
    def __init__(self, xml):
        self.items = [re.sub(r'<[^>]+>', '', s) for s in re.findall(r'<si>(.*?)</si>', xml, flags=re.S)]
        self.index = {s: i for i, s in enumerate(self.items)}

    def get(self, i):
        return self.items[int(i)]


CELL_RE = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', re.S)


def cell_text(attrs, inner, shared):
    """Displayed value of one cell (string, number or None)."""
    if inner is None:
        return None
    m = re.search(r'<v>(.*?)</v>', inner, flags=re.S)
    if 't="s"' in attrs:
        return shared.get(m.group(1)) if m else None
    if 't="inlineStr"' in attrs:
        return re.sub(r'<[^>]+>', '', inner)
    if 't="str"' in attrs:
        return m.group(1) if m else None
    if m:
        v = m.group(1)
        return float(v) if ('.' in v or 'E' in v) else int(v)
    return None


# ---------------------------------------------------------------- workbook

class Workbook:
    def __init__(self, path):
        self.path = path
        self.zin = zipfile.ZipFile(path)
        self.parts = OrderedDict((i.filename, self.zin.read(i.filename)) for i in self.zin.infolist())
        self.infos = {i.filename: i for i in self.zin.infolist()}
        wb = self.text('xl/workbook.xml')
        rels = self.text('xl/_rels/workbook.xml.rels')
        rid = {}
        for m in re.finditer(r'<Relationship ([^>]*)/>', rels):
            a = m.group(1)
            i = re.search(r'Id="([^"]+)"', a).group(1)
            t = re.search(r'Target="([^"]+)"', a).group(1)
            rid[i] = t
        self.sheet_part = {}
        for name, r in re.findall(r'<sheet [^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wb):
            self.sheet_part[name] = 'xl/' + rid[r].lstrip('/').replace('xl/', '', 1) if not rid[r].startswith('/') else rid[r].lstrip('/')
        self.shared = SharedStrings(self.text('xl/sharedStrings.xml'))

    def text(self, part):
        return self.parts[part].decode('utf8')

    def set_text(self, part, s):
        self.parts[part] = s.encode('utf8')

    def sheet(self, name):
        return self.text(self.sheet_part[name])

    def set_sheet(self, name, s):
        self.set_text(self.sheet_part[name], s)

    def drop_calc_chain(self):
        # Ask Excel to recompute every formula on the next open, so the cached
        # values this script writes are only a courtesy for other readers.
        wbx = self.text('xl/workbook.xml')
        if 'fullCalcOnLoad' not in wbx:
            wbx = re.sub(r'<calcPr ', '<calcPr fullCalcOnLoad="1" ', wbx, count=1)
            self.set_text('xl/workbook.xml', wbx)
        if 'xl/calcChain.xml' not in self.parts:
            return
        del self.parts['xl/calcChain.xml']
        ct = self.text('[Content_Types].xml')
        self.set_text('[Content_Types].xml', re.sub(r'<Override PartName="/xl/calcChain.xml"[^>]*/>', '', ct))
        rels = self.text('xl/_rels/workbook.xml.rels')
        self.set_text('xl/_rels/workbook.xml.rels', re.sub(r'<Relationship [^>]*calcChain[^>]*/>', '', rels))

    def write(self, out_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for name, data in self.parts.items():
                info = self.infos.get(name)
                zi = zipfile.ZipInfo(name, date_time=info.date_time if info else (1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                if info:
                    zi.external_attr = info.external_attr
                z.writestr(zi, data)
        with open(out_path, 'wb') as f:
            f.write(buf.getvalue())


# ---------------------------------------------------------------- archives

def read_csv_rows(csv_path):
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        sys.exit("CSV is empty")
    header = rows[0]
    if header and header[0] == '':
        header = header[1:]
        data = [r[1:] for r in rows[1:]]
    else:
        data = rows[1:]
    data = [r for r in data if any(c.strip() for c in r)]
    return header, data


def archives_header(wb):
    xml = wb.sheet('Archives')
    row1 = re.search(r'<row r="1"[^>]*>(.*?)</row>', xml, flags=re.S).group(1)
    cols = {}
    for m in CELL_RE.finditer(row1):
        cols[col_to_idx(m.group(1))] = cell_text(m.group(3), m.group(4), wb.shared)
    return cols  # {col index: header text}


def append_archives(wb, header, data):
    xml = wb.sheet('Archives')
    sheet_cols = archives_header(wb)
    by_name = {v: k for k, v in sheet_cols.items() if v}
    missing = [h for h in header if h not in by_name]
    if missing:
        sys.exit(f"CSV columns not found in the Archives sheet: {missing}")
    date_col = by_name.get('Date')
    date_style = None
    if date_col:
        m = re.search(r'<c r="%s\d+" s="(\d+)"><v>' % idx_to_col(date_col), xml)
        date_style = m.group(1) if m else None
    entity_col = by_name.get('SingleEntity')
    p1_col, p2_col = by_name.get('Player 1'), by_name.get('Player 2')

    last_row = max(int(r) for r in re.findall(r'<row r="(\d+)"', xml))
    new_rows = []
    r = last_row
    for rec in data:
        r += 1
        vals = {by_name[h]: parse_value(v) for h, v in zip(header, rec)}
        cells = []
        for ci in sorted(vals):
            v = vals[ci]
            if ci == entity_col:
                continue
            style = date_style if ci == date_col and isinstance(v, dt.datetime) else None
            c = cell_xml(f"{idx_to_col(ci)}{r}", v, style)
            if c:
                cells.append(c)
        if entity_col and p1_col:
            p1 = vals.get(p1_col) or ''
            p2 = vals.get(p2_col) or '' if p2_col else ''
            ref = f"{idx_to_col(entity_col)}{r}"
            f = f"${idx_to_col(p1_col)}{r}&amp;${idx_to_col(p2_col)}{r}"
            cells.append(f'<c r="{ref}" t="str"><f>{f}</f><v>{escape(str(p1) + str(p2))}</v></c>')
        cells.sort(key=lambda c: col_to_idx(re.match(r'<c r="([A-Z]+)', c).group(1)))
        new_rows.append(f'<row r="{r}" spans="1:{max(sheet_cols)}" ht="15" customHeight="1" x14ac:dyDescent="0.2">'
                        + ''.join(cells) + '</row>')
    xml = xml.replace('</sheetData>', ''.join(new_rows) + '</sheetData>', 1)
    xml = re.sub(r'<dimension ref="A1:([A-Z]+)\d+"/>', lambda m: f'<dimension ref="A1:{m.group(1)}{r}"/>', xml, count=1)
    wb.set_sheet('Archives', xml)
    return last_row + 1, r


# ---------------------------------------------------------------- event facts

def event_summary(header, data):
    """FBC number, team totals (FTAS once per team), winner, and each player's team."""
    ix = {h: i for i, h in enumerate(header)}
    fbc = {int(float(r[ix['FBC']])) for r in data if r[ix['FBC']]}
    if len(fbc) != 1:
        sys.exit(f"CSV must hold exactly one FBC event, found {sorted(fbc)}")
    fbc = fbc.pop()
    totals, seen_ftas, players = {}, set(), {}
    for r in data:
        team = r[ix['Team']]
        pts = float(r[ix['Points earned']] or 0)
        if r[ix['Singles/Doubles']] == 'FTAS':
            key = (r[ix['UniqueMatchID']], team)
            if key in seen_ftas:
                continue
            seen_ftas.add(key)
        totals[team] = totals.get(team, 0.0) + pts
        for k in ('Player 1', 'Player 2'):
            p = r[ix[k]].strip() if k in ix else ''
            if p:
                players[p] = team
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    if len(ranked) != 2:
        sys.exit(f"expected 2 teams in the CSV, found {list(totals)}")
    tie = ranked[0][1] == ranked[1][1]
    return {'fbc': fbc, 'totals': ranked, 'winner': None if tie else ranked[0][0], 'players': players}


# ---------------------------------------------------------------- cups

def update_cups(wb, ev):
    """Insert the new cup column before Total and fill it in. Returns notes."""
    notes = []
    xml = wb.sheet('Cups')
    sh = wb.shared
    label = f"FBC {ev['fbc']}"

    # Locate the header row (cell 'Player'), Total column, and player rows.
    rows = OrderedDict()
    for m in re.finditer(r'<row r="(\d+)"([^>]*)>(.*?)</row>', xml, flags=re.S):
        rows[int(m.group(1))] = (m.group(2), m.group(3))
    header_row = total_col = played_col = None
    hdr = {}
    for rn, (_, inner) in rows.items():
        for c in CELL_RE.finditer(inner):
            if cell_text(c.group(3), c.group(4), sh) == 'Player':
                header_row = rn
                break
        if header_row:
            for c in CELL_RE.finditer(inner):
                hdr[col_to_idx(c.group(1))] = cell_text(c.group(3), c.group(4), sh)
            break
    if not header_row:
        sys.exit("Cups sheet: no 'Player' header found")
    if label in hdr.values():
        sys.exit(f"Cups sheet already has an '{label}' column")
    name_col = [k for k, v in hdr.items() if v == 'Player'][0]
    total_col = [k for k, v in hdr.items() if v == 'Total'][0]
    played_col = [k for k, v in hdr.items() if v == 'Played'][0]
    pct_col = [k for k, v in hdr.items() if v == '%'][0]
    lost_col = [k for k, v in hdr.items() if v == 'Lost'][0]
    first_cup_col = min(k for k, v in hdr.items() if isinstance(v, str) and re.fullmatch(r'FBC \d+', v))
    new_col = total_col  # insert here; Total and everything right of it shift by one
    L = idx_to_col

    player_rows, total_row = OrderedDict(), None
    for rn, (_, inner) in rows.items():
        if rn <= header_row:
            continue
        cells = {col_to_idx(c.group(1)): (c.group(3), c.group(4)) for c in CELL_RE.finditer(inner)}
        name = cell_text(*cells[name_col], sh) if name_col in cells else None
        if not isinstance(name, str) or not name.strip():
            break
        if name.strip().lower() == 'total':
            total_row = rn
            break
        player_rows[name.strip()] = rn
    if total_row is None:
        sys.exit("Cups sheet: no Total row found")
    last_player_row = max(player_rows.values())

    # Style ids to reuse: header cell, a 1/0 cell, the Win% cell, the name cell.
    def style_of(rn, ci):
        m = re.search(r'<c r="%s%d" s="(\d+)"' % (L(ci), rn), xml)
        return m.group(1) if m else None
    st_hdr = style_of(header_row, first_cup_col)
    st_val = style_of(last_player_row, first_cup_col)
    st_pct = style_of(last_player_row, pct_col)
    st_name = style_of(last_player_row, name_col)
    x_idx = sh.index.get('X')
    x_cell = (lambda ref: f'<c r="{ref}" s="{st_val}" t="s"><v>{x_idx}</v></c>') if x_idx is not None \
        else (lambda ref: cell_xml(ref, 'X', st_val))

    # Every player in the event is marked 1/0; everyone else X. Players new to
    # the Cups sheet get a row inserted above Total.
    marks = {}
    for p, team in ev['players'].items():
        marks[p] = 'X' if ev['winner'] is None else (1 if team == ev['winner'] else 0)
    new_players = [p for p in marks if p not in player_rows]

    # --- 1. shift every cell right of the insertion point, and rows >= Total
    #        down by the number of new players (all in one pass per cell).
    row_shift_at = total_row if new_players else None
    n_new = len(new_players)

    def shift_cell(m):
        col, row, attrs, inner = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        ci = col_to_idx(col)
        if ci >= new_col:
            ci += 1
        if row_shift_at and row >= row_shift_at:
            row += n_new
        if inner is not None:
            inner = re.sub(r'<f([^>]*)>(.*?)</f>', lambda f: '<f' + shift_attr(f.group(1)) + '>' + shift_formula(f.group(2)) + '</f>', inner, flags=re.S)
            inner = re.sub(r'<f([^>]*)/>', lambda f: '<f' + shift_attr(f.group(1)) + '/>', inner)
            return f'<c r="{L(ci)}{row}"{attrs}>{inner}</c>'
        return f'<c r="{L(ci)}{row}"{attrs}/>'

    def shift_formula(text):
        text = shift_refs(text, col_at=new_col, row_at=row_shift_at)
        if row_shift_at:
            text = text.replace('__ROWS__', '')
        return text

    def shift_attr(attrs):
        return re.sub(r'ref="([^"]+)"', lambda a: 'ref="%s"' % shift_refs(a.group(1), col_at=new_col, row_at=row_shift_at), attrs)

    def shift_row(m):
        rn, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        if row_shift_at and rn >= row_shift_at:
            rn += n_new
        attrs = re.sub(r'spans="(\d+):(\d+)"', lambda s: f'spans="{s.group(1)}:{int(s.group(2)) + 1}"', attrs)
        inner = CELL_RE.sub(shift_cell, inner)
        return f'<row r="{rn}"{attrs}>{inner}</row>'

    xml = re.sub(r'<row r="(\d+)"([^>]*)>(.*?)</row>', shift_row, xml, flags=re.S)

    # After the shift the old columns live one to the right.
    total_col2, played_col2, pct_col2, lost_col2 = total_col + 1, played_col + 1, pct_col + 1, lost_col + 1
    prev_cup = L(new_col - 1)
    total_row2 = total_row + n_new

    # --- 2. extend the per-player Total formula SUM(C3:N3) -> SUM(C3:O3)
    xml = re.sub(r'SUM\((\$?)%s(\d+):(\$?)%s(\d+)\)' % (L(first_cup_col), prev_cup),
                 lambda m: f'SUM({m.group(1)}{L(first_cup_col)}{m.group(2)}:{m.group(3)}{L(new_col)}{m.group(4)})', xml)

    # --- 3. insert the new column's cells into each row, then fix cached values
    def values_in(row_inner):
        out = {}
        for c in CELL_RE.finditer(row_inner):
            out[col_to_idx(c.group(1))] = cell_text(c.group(3), c.group(4), sh)
        return out

    def set_v(row_inner, ci, rn, value):
        ref = f"{L(ci)}{rn}"
        return re.sub(r'(<c r="%s"[^>]*>)(.*?)(</c>)' % ref,
                      lambda m: m.group(1) + re.sub(r'<v>.*?</v>', f'<v>{value!r}</v>', m.group(2)) + m.group(3),
                      row_inner, count=1, flags=re.S)

    def insert_cell(row_inner, cell):
        """Place a <c> in column order inside a row."""
        target = col_to_idx(re.match(r'<c r="([A-Z]+)', cell).group(1))
        cells = list(CELL_RE.finditer(row_inner))
        for c in cells:
            if col_to_idx(c.group(1)) > target:
                return row_inner[:c.start()] + cell + row_inner[c.start():]
        return row_inner + cell

    col_sum_rows = {}  # for the Total row cached values
    def edit_rows(m):
        rn, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        ref = f"{L(new_col)}{rn}"
        if rn == header_row:
            inner = insert_cell(inner, cell_xml(ref, label, st_hdr))
        elif rn in player_rows.values():
            name = [p for p, r in player_rows.items() if r == rn][0]
            mark = marks.get(name, 'X')
            inner = insert_cell(inner, x_cell(ref) if mark == 'X' else cell_xml(ref, mark, st_val))
            vals = values_in(inner)
            cups = [vals.get(c) for c in range(first_cup_col, new_col + 1)]
            won = sum(1 for v in cups if v in (1, '1'))
            lost = sum(1 for v in cups if v in (0, '0'))
            played = (vals.get(played_col2) or 0) + (1 if mark != 'X' else 0)
            inner = set_v(inner, total_col2, rn, won)
            inner = set_v(inner, played_col2, rn, played)
            inner = set_v(inner, pct_col2, rn, (won / played) if played else 0)
            inner = set_v(inner, lost_col2, rn, lost)
            for c in range(first_cup_col, lost_col2 + 1):
                v = {total_col2: won, played_col2: played, lost_col2: lost}.get(c, vals.get(c))
                if isinstance(v, (int, float)):
                    col_sum_rows.setdefault(c, {})[rn] = v
        elif rn == total_row2:
            # New column's total: same shared formula as its neighbours.
            si = re.search(r'<c r="%s%d"[^>]*><f t="shared" (?:ref="[^"]+" )?si="(\d+)"' % (L(new_col + 1), rn), inner)
            si = si.group(1) if si else None
            cell = (f'<c r="{ref}" s="{st_val}"><f t="shared" si="{si}"/><v>0</v></c>' if si
                    else f'<c r="{ref}" s="{st_val}"><f>SUM({L(new_col)}{header_row + 1}:{L(new_col)}{last_player_row})</f><v>0</v></c>')
            inner = insert_cell(inner, cell)
            inner = re.sub(r'(<f t="shared" ref=")%s%d:([A-Z]+)%d(")' % (L(first_cup_col), rn, rn),
                           lambda a: f'{a.group(1)}{L(first_cup_col)}{rn}:{L(col_to_idx(a.group(2)))}{rn}{a.group(3)}', inner)
        elif rn > header_row and rn != total_row2:
            inner = insert_cell(inner, f'<c r="{ref}" s="{st_name}"/>') if re.search(r'<c r="%s%d"' % (L(new_col + 1), rn), inner) else inner
        return f'<row r="{rn}"{attrs}>{inner}</row>'

    xml = re.sub(r'<row r="(\d+)"([^>]*)>(.*?)</row>', edit_rows, xml, flags=re.S)

    # --- 4. rows for players new to the Cups sheet (inserted above Total)
    if new_players:
        first_p = header_row + 1
        new_row_xml = []
        for i, p in enumerate(new_players):
            rn = total_row + i
            cells = [f'<c r="{L(name_col)}{rn}" s="{st_name}" t="inlineStr"><is><t>{escape(p)}</t></is></c>']
            for c in range(first_cup_col, new_col):
                cells.append(x_cell(f"{L(c)}{rn}"))
            mark = marks[p]
            cells.append(x_cell(f"{L(new_col)}{rn}") if mark == 'X' else cell_xml(f"{L(new_col)}{rn}", mark, st_val))
            won, played = (1 if mark == 1 else 0), (0 if mark == 'X' else 1)
            cells.append(f'<c r="{L(total_col2)}{rn}" s="{st_val}"><f>SUM({L(first_cup_col)}{rn}:{L(new_col)}{rn})</f><v>{won}</v></c>')
            cells.append(f'<c r="{L(played_col2)}{rn}" s="{st_val}"><v>{played}</v></c>')
            cells.append(f'<c r="{L(pct_col2)}{rn}" s="{st_pct}"><f>{L(total_col2)}{rn}/{L(played_col2)}{rn}</f><v>{(won / played) if played else 0!r}</v></c>')
            cells.append(f'<c r="{L(lost_col2)}{rn}" s="{st_val}"><f>{L(played_col2)}{rn}-{L(total_col2)}{rn}</f><v>{played - won}</v></c>')
            new_row_xml.append(f'<row r="{rn}" spans="1:{lost_col2}" ht="15.75" customHeight="1" x14ac:dyDescent="0.2">' + ''.join(cells) + '</row>')
            player_rows[p] = rn
            for c, v in ((new_col, mark if mark != 'X' else None), (total_col2, won), (played_col2, played), (lost_col2, played - won)):
                if isinstance(v, (int, float)):
                    col_sum_rows.setdefault(c, {})[rn] = v
        xml = re.sub(r'(<row r="%d")' % total_row2, ''.join(new_row_xml) + r'\1', xml, count=1)
        last_player_row = total_row2 - 1
        notes.append(f"added Cups rows for new player(s): {', '.join(new_players)}")

    # --- 5. Total row: make its SUM ranges cover every player row, and refresh
    #        its cached values. (The sheet's ranges stopped one row short of the
    #        last player before this script touched it.)
    def fix_total(m):
        rn, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        if rn != total_row2:
            return m.group(0)
        inner = re.sub(r'SUM\((\$?)([A-Z]+)(\$?)%d:(\$?)([A-Z]+)(\$?)\d+\)' % (header_row + 1),
                       lambda s: f'SUM({s.group(1)}{s.group(2)}{s.group(3)}{header_row + 1}:{s.group(4)}{s.group(5)}{s.group(6)}{last_player_row})', inner)
        for c in CELL_RE.finditer(inner):
            ci = col_to_idx(c.group(1))
            if c.group(4) and '<f' in c.group(4) and ci in col_sum_rows:
                total = sum(v for r_, v in col_sum_rows[ci].items() if header_row < r_ <= last_player_row)
                inner = set_v(inner, ci, rn, total)
        return f'<row r="{rn}"{attrs}>{inner}</row>'
    xml = re.sub(r'<row r="(\d+)"([^>]*)>(.*?)</row>', fix_total, xml, flags=re.S)

    # --- 6. sheet-level ranges: dimension, cols, autoFilter, conditional formats
    xml = re.sub(r'<dimension ref="A1:([A-Z]+)(\d+)"/>',
                 lambda m: f'<dimension ref="A1:{L(col_to_idx(m.group(1)) + 1)}{int(m.group(2)) + n_new}"/>', xml, count=1)

    def fix_cols(m):
        out = []
        for c in re.finditer(r'<col ([^>]*)/>', m.group(1)):
            a = c.group(1)
            lo, hi = int(re.search(r'min="(\d+)"', a).group(1)), int(re.search(r'max="(\d+)"', a).group(1))
            if hi < new_col:
                out.append(f'<col {a}/>')
            elif lo >= new_col:
                out.append('<col ' + re.sub(r'min="\d+"', f'min="{lo + 1}"', re.sub(r'max="\d+"', f'max="{min(hi + 1, 16384)}"', a)) + '/>')
            else:  # span straddles the insertion point: widen it
                out.append('<col ' + re.sub(r'max="\d+"', f'max="{hi + 1}"', a) + '/>')
        # the new column itself, styled like the previous cup column
        prev = re.search(r'<col ([^>]*min="%d"[^>]*)/>' % (new_col - 1), m.group(1)) or \
               [c for c in re.finditer(r'<col ([^>]*)/>', m.group(1))
                if int(re.search(r'min="(\d+)"', c.group(1)).group(1)) <= new_col - 1 <= int(re.search(r'max="(\d+)"', c.group(1)).group(1))]
        prev_attrs = prev.group(1) if hasattr(prev, 'group') else (prev[0].group(1) if prev else 'width="8" customWidth="1"')
        prev_attrs = re.sub(r'(min|max)="\d+"', '', prev_attrs).strip()
        out.append(f'<col min="{new_col}" max="{new_col}" {prev_attrs}/>')
        out.sort(key=lambda s: int(re.search(r'min="(\d+)"', s).group(1)))
        return '<cols>' + ''.join(out) + '</cols>'
    xml = re.sub(r'<cols>(.*?)</cols>', fix_cols, xml, count=1, flags=re.S)

    def fix_range_attr(m):
        ref = shift_refs(m.group(2), col_at=new_col, row_at=row_shift_at)
        # ranges that ended at the row before the last player now reach the last player
        ref = re.sub(r':(\$?[A-Z]+\$?)(\d+)$', lambda s: f':{s.group(1)}{max(int(s.group(2)), last_player_row) if int(s.group(2)) >= header_row + 1 else s.group(2)}', ref)
        return f'{m.group(1)}="{ref}"'
    xml = re.sub(r'(<autoFilter ref|<sortState ref|<sortCondition [^>]*?ref)="([^"]+)"', fix_range_attr, xml)
    def fix_cf(m):
        ref = shift_refs(m.group(2), col_at=new_col, row_at=row_shift_at)
        # the 1/0 colouring rule reaches the new column and any new player row
        ref = re.sub(r'^(\$?[A-Z]+\$?\d+):\$?%s\$?(\d+)$' % prev_cup,
                     lambda s: f'{s.group(1)}:{L(new_col)}{max(int(s.group(2)), last_player_row)}', ref)
        return f'{m.group(1)}="{ref}"'
    xml = re.sub(r'(conditionalFormatting sqref)="([^"]+)"', fix_cf, xml)
    wb.set_sheet('Cups', xml)

    # --- 7. other sheets and defined names that point into Cups
    for name, part in wb.sheet_part.items():
        if name == 'Cups':
            continue
        s = wb.text(part)
        if 'Cups!' not in s:
            continue
        s2 = shift_refs(s, col_at=new_col, row_at=row_shift_at, sheet_prefix='Cups')
        s2 = re.sub(r'(Cups!\$?[A-Z]+\$?%d:\$?[A-Z]+\$?)(\d+)' % (header_row + 1),
                    lambda m: m.group(1) + str(max(int(m.group(2)), last_player_row)), s2)
        if s2 != s:
            wb.set_text(part, s2)
            notes.append(f"updated Cups references in the {name} sheet")
    wbx = wb.text('xl/workbook.xml')
    wbx2 = shift_refs(wbx, col_at=new_col, row_at=row_shift_at, sheet_prefix='Cups')
    wbx2 = re.sub(r'(Cups!\$?[A-Z]+\$?%d:\$?[A-Z]+\$?)(\d+)' % (header_row + 1),
                  lambda m: m.group(1) + str(max(int(m.group(2)), last_player_row)), wbx2)
    if wbx2 != wbx:
        wb.set_text('xl/workbook.xml', wbx2)

    marked = sum(1 for v in marks.values() if v != 'X')
    notes.insert(0, f"Cups: added '{label}' column, marked {marked} players "
                    f"({sum(1 for v in marks.values() if v == 1)} won, {sum(1 for v in marks.values() if v == 0)} lost), others X")
    return notes


# ---------------------------------------------------------------- checks

def health_check(path):
    """Run the app's own validate_data on a workbook. Returns a list of issues."""
    import logging, warnings, contextlib
    warnings.filterwarnings('ignore')
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, REPO)
    with contextlib.redirect_stderr(io.StringIO()):
        import app  # noqa: E402  (streamlit runs in bare mode here)
    mtime = os.path.getmtime(path)
    df = app._load_data_cached(mtime, path)
    cups = app._load_cups_data_cached(mtime, path)
    issues = app.validate_data(df, cups)
    results = [r for r in app.get_fbc_team_results(df)]
    return issues, results, df, cups


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', help="the scoring app's Archives CSV export")
    ap.add_argument('--workbook', default=os.path.join(REPO, 'FBC_Data.xlsx'))
    ap.add_argument('--out', help='write here instead of updating the workbook in place')
    ap.add_argument('--dry-run', action='store_true', help='report what would change; write nothing')
    ap.add_argument('--force', action='store_true', help='write even if Data Health flags issues')
    a = ap.parse_args()

    header, data = read_csv_rows(a.csv)
    ev = event_summary(header, data)
    wb = Workbook(a.workbook)

    existing = health_check(a.workbook)[2]
    if ev['fbc'] in set(existing['FBC'].dropna().astype(int)):
        sys.exit(f"FBC {ev['fbc']} is already in the Archives sheet ({int((existing['FBC'] == ev['fbc']).sum())} rows). "
                 f"Nothing written.")

    first, last = append_archives(wb, header, data)
    notes = update_cups(wb, ev)
    wb.drop_calc_chain()

    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False, dir=os.path.dirname(os.path.abspath(a.workbook)))
    tmp.close()
    wb.write(tmp.name)
    issues, results, df, cups = health_check(tmp.name)
    new_issues = issues  # the pre-import workbook is expected to be clean

    print(f"FBC {ev['fbc']}: {len(data)} rows appended to Archives (rows {first}-{last})")
    for cap, pts in ev['totals']:
        print(f"  {cap}: {pts:g}")
    print(f"  winner: {ev['winner'] or 'TIE'}")
    for n in notes:
        print(f"  {n}")
    ev_res = [r for r in results if r['fbc'] == ev['fbc']]
    if ev_res:
        r = ev_res[0]
        print(f"  app reads it back as: {r['winner']} {r['teams'][0][1]:g} - {r['teams'][1][1]:g} {r['loser']} at {r['location']}")
    if new_issues:
        print(f"\nData Health: {len(new_issues)} issue(s)")
        for i in new_issues:
            print(f"  - {i}")
    else:
        print("\nData Health: all checks pass")

    if a.dry_run:
        os.unlink(tmp.name)
        print("\nDry run: nothing written.")
        return
    if new_issues and not a.force:
        os.unlink(tmp.name)
        sys.exit("\nNot written: fix the issues above (or rerun with --force).")
    out = a.out or a.workbook
    shutil.move(tmp.name, out)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
