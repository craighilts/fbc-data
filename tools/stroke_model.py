"""Stroke-aware match model: the Beats-Card block on the Sandbagger Index tab and the
Stroke Data tab. Called from tools/recompute_analytics.py; numpy only.

Data
  One record per head-to-head match (FTAS, Shamble and 3-man Stableford excluded).
  Each side's handicap is put on an 18-hole basis using, in order of preference:
    1. Recorded handicaps from the per-cup handicap workbooks (tools/data/recorded_handicaps.csv,
       transcribed Oct 2026 from Dropbox: FBC 3-6 results sheets, FBC 7-10, 12, 13 pairing grids).
    2. The Handicaps-tab index x slope/113 under that cup's rules (FBC 1-2, FBC 11, singles 7-13).
  Era rules: Best Ball / Vegas / Hi-Lo use 100% through FBC 12 and 90% at FBC 13 (side value =
  mean of the two players); Pinehurst 60/40; Scramble had no handicaps before FBC 8, 35/15 after;
  singles 100%. Strokes in a match = 18-hole differential x holes / 18. A side is "getting"
  when it receives at least half a stroke in the match.

Beats-Card model
  Stake-weighted ridge logistic regression of each match result on a league-wide getting-strokes
  term plus one rating per player (+1 on side A, -1 on side B). The getting term absorbs the
  league tilt toward the stroke-giving side, so a player's rating is how much better or worse he
  does than his handicap implies. Ratings are reported as win-probability points per match
  (logit x 25, the slope at 50%). Uncertainty: bootstrap over matches (fixed seed).
  A per-player "different when getting strokes" term was tested (Oct 2026) and dropped: it made
  cross-validated predictions worse.

Stroke Data tab
  One row per side per match, with strokes received, source, points and the strokes-only
  expected points (league getting term + per-format value of a stroke).
"""
import os, re
import numpy as np, pandas as pd

BC_C = 0.05          # inverse ridge strength (sklearn-style C); chosen by 5-fold CV, Oct 2026
BC_MIN_MATCHES = 15
BC_BOOT = 400
BC_SEED = 13
HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, 'data', 'recorded_handicaps.csv')
NAME_FIX = {'bussiuclo': 'Bussiculo', 'deoteris': 'DeOteris', 'connolly, b': 'Connolly', 'b. connolly': 'Connolly',
            'connolly, r': 'R. Connolly', 'mcinerny': 'McInerney'}
GRPS = ['Four-ball', 'Pinehurst', 'Scramble', 'Singles']


def _nm(s):
    s = str(s).strip(); return NAME_FIX.get(s.lower(), s)


def fgroup(f):
    if f in ('Best Ball', 'Vegas', 'Hi-Lo'): return 'Four-ball'
    if f == 'Pinehurst': return 'Pinehurst'
    if str(f).startswith('Scramble'): return 'Scramble'
    if f in ('Match Play', 'Singles'): return 'Singles'
    return None


def _index_table(h):
    hdr = h[2]; out = {}
    for r in h[3:]:
        if not r[1]: continue
        for j, c in enumerate(hdr):
            if isinstance(c, str) and c.startswith('FBC') and c[3:].isdigit() and isinstance(r[j], (int, float)):
                out[(str(r[1]).lower(), int(c[3:]))] = float(r[j])
    return out


def _side_from_index(fmt, ix, slope, cup):
    sl = slope if isinstance(slope, (int, float)) and slope == slope else 130.0
    ch = [i * sl / 113.0 for i in ix]
    if len(ch) == 1: return ch[0]
    lo, hi = min(ch), max(ch); g = fgroup(fmt)
    if g == 'Pinehurst': return .6 * lo + .4 * hi
    if g == 'Scramble': return (.35 * lo + .15 * hi) if cup >= 8 else 0.0
    return (.9 if cup >= 13 else 1.0) * (ch[0] + ch[1]) / 2


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def _rec_d18(r, cup):
    fam = r['family']
    if fam == 'scramble' and cup < 8: return 0.0
    if fam in ('pinehurst', 'scramble') and not np.isnan(_f(r['teamA'])) and not np.isnan(_f(r['teamB'])):
        return _f(r['teamA']) - _f(r['teamB'])
    if fam == 'scramble':
        a = sorted([_f(r['hA1']), _f(r['hA2'])]); b = sorted([_f(r['hB1']), _f(r['hB2'])])
        return (.35 * a[0] + .15 * a[1]) - (.35 * b[0] + .15 * b[1])
    if fam == 'singles': return _f(r['hA1']) - _f(r['hB1'])
    return (_f(r['hA1']) + _f(r['hA2'])) / 2 - (_f(r['hB1']) + _f(r['hB2'])) / 2


def _recorded():
    if not os.path.exists(REC): return {}
    R = {}; key = lambda s: '|'.join(sorted(_nm(x).lower() for x in str(s).split('|')))
    for r in pd.read_csv(REC, dtype=str).fillna('').to_dict('records'):
        cup = int(r['FBC']); d = _rec_d18(r, cup)
        if d != d: continue
        A, B = key(r['sideA']), key(r['sideB']); src = r['source_file']
        R.setdefault((cup, r['family'], A, B), (d, src))
        R.setdefault((cup, r['family'], B, A), (-d, src))
    return R


def matches(df, h):
    """One dict per head-to-head match with sides, results, stroke differential and source."""
    IX = _index_table(h); REC_ = _recorded()
    key = lambda ps: '|'.join(sorted(p.lower() for p in ps))
    fams = {'Four-ball': ('fourball', 'hilo'), 'Pinehurst': ('pinehurst',), 'Scramble': ('scramble',), 'Singles': ('singles',)}
    X = df[df['Singles/Doubles'] != 'FTAS']; out = []
    for mid, g in X.groupby('UniqueMatchID', sort=False):
        if len(g) != 2: continue
        a, b = g.iloc[0], g.iloc[1]; grp = fgroup(a['Format'])
        if grp is None: continue
        cup = int(a['FBC'])
        A = [p for p in (a['Player 1'], a['Player 2']) if isinstance(p, str) and p.strip()]
        B = [p for p in (b['Player 1'], b['Player 2']) if isinstance(p, str) and p.strip()]
        d18, src = None, None
        for fam in fams[grp]:
            hit = REC_.get((cup, fam, key(A), key(B)))
            if hit: d18, src = hit; break
        if d18 is None:
            ia = [IX.get((p.lower(), cup)) for p in A]; ib = [IX.get((p.lower(), cup)) for p in B]
            if None in ia or None in ib: continue
            d18 = _side_from_index(a['Format'], ia, a['Slope'], cup) - _side_from_index(a['Format'], ib, a['Slope'], cup)
            src = 'Index x slope (Handicaps tab)'
        stake = float(a['Points at stake']); holes = float(a['Holes'])
        out.append(dict(FBC=cup, mid=mid, fmt=a['Format'], grp=grp, holes=holes, stake=stake,
                        A=A, B=B, teamA=a['Team'], teamB=b['Team'],
                        yA=float(a['W'] + 0.5 * a['T']), d18=float(d18), dm=float(d18) * holes / 18.0, src=src))
    return pd.DataFrame(out)


def _logit_fit(X, y, w, C):
    """Weighted logistic regression, no intercept, L2 penalty 1/(2C) on all coefficients (Newton)."""
    beta = np.zeros(X.shape[1]); lam = 1.0 / C
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (w * (y - p)) - lam * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-10: break
    return beta


def _design(M):
    s = np.where(M.dm >= 0.5, 1.0, np.where(M.dm <= -0.5, -1.0, 0.0))
    return s


def fit(df, h):
    M = matches(df, h)
    w = M.stake.values; y = M.yA.values; s = _design(M)
    # strokes-only expectation (effectively unpenalized)
    Xc = np.column_stack([s] + [np.where(M.grp == g, M.dm, 0.0) for g in GRPS])
    bc = _logit_fit(Xc, y, w, 1e6)
    M['expA'] = 1 / (1 + np.exp(-(Xc @ bc)))
    # beats-card ratings
    players = sorted({p for x in M.A for p in x} | {p for x in M.B for p in x}); P = {p: i for i, p in enumerate(players)}
    SK = np.zeros((len(M), len(players)))
    for k, r in enumerate(M.itertuples()):
        for p in r.A: SK[k, P[p]] += 1
        for p in r.B: SK[k, P[p]] -= 1
    SCALE = 20.0  # keeps the ridge penalty off the league getting term
    X = np.column_stack([s * SCALE, SK])
    beta = _logit_fit(X, y, w, BC_C)[1:]
    rng = np.random.default_rng(BC_SEED); boot = np.zeros((BC_BOOT, len(players)))
    for b in range(BC_BOOT):
        ix = rng.integers(0, len(M), len(M)); boot[b] = _logit_fit(X[ix], y[ix], w[ix], BC_C)[1:]
    n = {p: 0 for p in players}
    for x in list(M.A) + list(M.B):
        for p in x: n[p] += 1
    rows = []
    for p in players:
        if n[p] < BC_MIN_MATCHES: continue
        i = P[p]
        rows.append((p, n[p], round(beta[i] * 25 / 100, 4), round(np.percentile(boot[:, i], 5) * 25 / 100, 4),
                     round(np.percentile(boot[:, i], 95) * 25 / 100, 4), round(float((boot[:, i] > 0).mean()), 4)))
    rows.sort(key=lambda t: (-t[2], t[0]))
    tilt = np.r_[M.yA[s > 0], 1 - M.yA[s < 0]].mean()
    return M, rows, dict(n_matches=len(M), n_recorded=int((~M.src.str.startswith('Index')).sum()), tilt=tilt)


def stroke_rows(M):
    out = []
    for r in M.sort_values(['FBC', 'mid'], key=lambda c: c if c.name == 'FBC' else c.str.replace(r'\d+$', lambda m: m.group(0).zfill(3), regex=True)).itertuples():
        for side, opp, team, y, e, dm, d18 in ((r.A, r.B, r.teamA, r.yA, r.expA, r.dm, r.d18), (r.B, r.A, r.teamB, 1 - r.yA, 1 - r.expA, -r.dm, -r.d18)):
            status = 'Getting' if dm >= 0.5 else ('Giving' if dm <= -0.5 else 'Even')
            out.append((int(r.FBC), r.mid, r.fmt, r.grp, int(r.holes), r.stake, team,
                        side[0], side[1] if len(side) > 1 else None, opp[0], opp[1] if len(opp) > 1 else None,
                        round(d18, 2), round(dm, 2), status, round(y * r.stake, 2), round(e * r.stake, 3),
                        round((y - e) * r.stake, 3), r.src))
    return out


STROKE_HEADERS = ['FBC', 'UniqueMatchID', 'Format', 'Format Group', 'Holes', 'Points at Stake', 'Team',
                  'Player 1', 'Player 2', 'Opponent 1', 'Opponent 2', 'Strokes Received (18)', 'Strokes Received (Match)',
                  'Stroke Status', 'Points Earned', 'Expected Points', 'Points vs Expected', 'Stroke Source']


# ---------------------------------------------------------------------------------------------
# workbook XML writers (called from recompute_analytics.run with the zip parts dict)
from xml.sax.saxutils import escape as _esc


def _col(n):
    s = ''
    while n: n, r = divmod(n - 1, 26); s = chr(65 + r) + s
    return s


def _cell(ref, v, s):
    if v is None or (isinstance(v, float) and v != v): return f'<c r="{ref}" s="{s}"/>'
    if isinstance(v, str): return f'<c r="{ref}" s="{s}" t="inlineStr"><is><t xml:space="preserve">{_esc(v)}</t></is></c>'
    v = float(v); v = int(v) if v.is_integer() else v
    return f'<c r="{ref}" s="{s}"><v>{v}</v></c>'


BC_TITLE = 'Beats-Card Model'
BC_NOTE = ('Win-probability points per match versus what the handicaps imply, using recorded match handicaps and '
           'correcting for the league-wide edge of the stroke-giving side. 90% range from bootstrap; min 15 matches.')
BC_HEADS = ['Player', 'Matches', 'vs Card', '90% Low', '90% High', 'Chance Above']


def write_beats_card(parts, sheet_part, rows):
    """Add/refresh the Beats-Card block in columns I:N of the Sandbagger Index sheet (rows 1, 2, 4, 5+).
    Must run after the Sandbagger table (B:G) has been rewritten."""
    s = parts[sheet_part].decode()
    # drop any previous I:N cells
    s = re.sub(r'<c r="[I-N]\d+"[^>]*?(?:/>|>.*?</c>)', '', s, flags=re.S)
    add = {1: _cell('I1', BC_TITLE, 63), 2: ''.join(_cell(f'{c}2', BC_NOTE if c == 'I' else None, 94) for c in 'IJKLMN'),
           4: ''.join(_cell(f'{c}4', t, 30) for c, t in zip('IJKLMN', BC_HEADS))}
    for i, r in enumerate(rows):
        rn = 5 + i; a = i % 2 == 0
        st = (31, 40, 65, 65, 65, 41) if a else (34, 43, 67, 67, 67, 44)
        add[rn] = ''.join(_cell(f'{c}{rn}', float(v) if not isinstance(v, str) else v, sx) for c, v, sx in zip('IJKLMN', r, st))
    for rn, cells in add.items():
        m = re.search(r'<row r="%d"[^>]*?(?:/>|>.*?</row>)' % rn, s, re.S)
        if m:
            x = m.group(0)
            if x.endswith('/>'): x = x[:-2] + '></row>'
            x = x.replace('</row>', cells + '</row>')
            x = re.sub(r' spans="[^"]*"', ' spans="2:14"', x)
            s = s.replace(m.group(0), x, 1)
        else:
            prev = [int(n) for n in re.findall(r'<row r="(\d+)"', s) if int(n) < rn]
            row = f'<row r="{rn}" spans="2:14" ht="15" customHeight="1" x14ac:dyDescent="0.2">{cells}</row>'
            if prev:
                pm = re.search(r'<row r="%d"[^>]*?(?:/>|>.*?</row>)' % max(prev), s, re.S)
                s = s[:pm.end()] + row + s[pm.end():]
            else:
                s = s.replace('<sheetData>', '<sheetData>' + row, 1)
    s = re.sub(r'(<row r="2"[^>]*?) ht="[\d.]+"', r'\1 ht="40"', s, count=1)
    last = max(int(n) for n in re.findall(r'<row r="(\d+)"', s))
    s = re.sub(r'<dimension ref="[^"]+"/>', f'<dimension ref="B1:N{last}"/>', s)
    cols = '<col min="8" max="8" width="3" customWidth="1"/><col min="9" max="9" width="13" customWidth="1"/>' \
           '<col min="10" max="10" width="9" customWidth="1"/><col min="11" max="13" width="10" customWidth="1"/>' \
           '<col min="14" max="14" width="12" customWidth="1"/>'
    s = re.sub(r'<col min="(?:8|9|10|11|14)" [^>]*/>', '', s)
    s = s.replace('</cols>', cols + '</cols>', 1)
    if '<mergeCell ref="I2:N2"/>' not in s:
        s = re.sub(r'<mergeCells count="(\d+)">', lambda m: f'<mergeCells count="{int(m.group(1)) + 1}"><mergeCell ref="I2:N2"/>', s, count=1)
    parts[sheet_part] = s.encode()


STROKE_SHEET = 'Stroke Data'


def write_stroke_sheet(parts, infos, rows, after='Sandbagger Index'):
    """Create or replace the Stroke Data worksheet (header in row 1, one row per side per match)."""
    import zipfile
    wbx = parts['xl/workbook.xml'].decode(); rels = parts['xl/_rels/workbook.xml.rels'].decode()
    m = re.search(r'<sheet name="%s" sheetId="\d+"(?: state="\w+")? r:id="(rId\d+)"/>' % re.escape(STROKE_SHEET), wbx)
    if m:
        rid = m.group(1)
        tgt = re.search(r'Id="%s"[^>]*Target="([^"]+)"' % rid, rels).group(1)
        part = 'xl/' + tgt.lstrip('/').replace('xl/', '')
    else:
        rids = [int(x) for x in re.findall(r'Id="rId(\d+)"', rels)]; rid = f'rId{max(rids) + 1}'
        nums = [int(x) for x in re.findall(r'worksheets/sheet(\d+)\.xml', rels)]; part = f'xl/worksheets/sheet{max(nums) + 1}.xml'
        sid = max(int(x) for x in re.findall(r'sheetId="(\d+)"', wbx)) + 1
        anchor = re.search(r'<sheet name="%s" [^>]*/>' % re.escape(_esc(after)), wbx).group(0)
        wbx = wbx.replace(anchor, anchor + f'<sheet name="{STROKE_SHEET}" sheetId="{sid}" r:id="{rid}"/>', 1)
        rels = rels.replace('</Relationships>', f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="{part[3:]}"/></Relationships>')
        ct = parts['[Content_Types].xml'].decode()
        ct = ct.replace('</Types>', f'<Override PartName="/{part}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        parts['[Content_Types].xml'] = ct.encode()
        if 'docProps/app.xml' in parts:
            ap = parts['docProps/app.xml'].decode()
            ap = re.sub(r'(<vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>)(\d+)', lambda x: x.group(1) + str(int(x.group(2)) + 1), ap)
            ap = re.sub(r'<vt:vector size="(\d+)" baseType="lpstr">', lambda x: f'<vt:vector size="{int(x.group(1)) + 1}" baseType="lpstr">', ap)
            ap = ap.replace(f'<vt:lpstr>{_esc(after)}</vt:lpstr>', f'<vt:lpstr>{_esc(after)}</vt:lpstr><vt:lpstr>{STROKE_SHEET}</vt:lpstr>', 1)
            parts['docProps/app.xml'] = ap.encode()
        parts['xl/workbook.xml'] = wbx.encode(); parts['xl/_rels/workbook.xml.rels'] = rels.encode()
        zi = zipfile.ZipInfo(part, date_time=infos[0].date_time); zi.compress_type = zipfile.ZIP_DEFLATED
        infos.append(zi)
    # column styles: text / int / 2-dec
    kinds = ['i', 't', 't', 't', 'i', 'd', 't', 't', 't', 't', 't', 'd', 'd', 't', 'd', 'd', 'd', 't']
    sty = {('t', 0): 31, ('t', 1): 34, ('i', 0): 40, ('i', 1): 43, ('d', 0): 42, ('d', 1): 45}
    widths = [6, 28, 18, 12, 7, 9, 10, 12, 12, 12, 12, 11, 11, 11, 9, 10, 10, 44]
    n = len(STROKE_HEADERS); lastc = _col(n)
    out = [f'<row r="1" spans="1:{n}">' + ''.join(_cell(f'{_col(j + 1)}1', t, 30) for j, t in enumerate(STROKE_HEADERS)) + '</row>']
    for i, r in enumerate(rows):
        rn = i + 2
        out.append(f'<row r="{rn}" spans="1:{n}">' + ''.join(
            _cell(f'{_col(j + 1)}{rn}', (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v), sty[(kinds[j], i % 2)])
            for j, v in enumerate(r)) + '</row>')
    cols = ''.join(f'<col min="{j + 1}" max="{j + 1}" width="{wd}" customWidth="1"/>' for j, wd in enumerate(widths))
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
           f'<dimension ref="A1:{lastc}{len(rows) + 1}"/>'
           '<sheetViews><sheetView showGridLines="0" workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
           '<selection pane="bottomLeft" activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
           '<sheetFormatPr defaultRowHeight="15"/>'
           f'<cols>{cols}</cols><sheetData>' + ''.join(out) + '</sheetData>'
           '<pageMargins left="0.75" right="0.75" top="1" bottom="1" header="0.5" footer="0.5"/></worksheet>')
    parts[part] = xml.encode()
    return part
