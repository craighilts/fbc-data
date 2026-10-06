import streamlit as st
import pandas as pd
import numpy as np
import html
import os
import re
import contextlib
import anthropic

# Page config
st.set_page_config(
    page_title="FBC Stats",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palette — matches .streamlit/config.toml. Player 1 / Team 1 is green, the other side
# is a warm clay so comparisons never read as "good vs bad" red.
COLORS = {
    'primary': '#1E5B43',      # Augusta-ish green
    'primary_soft': '#E6EFE9',
    'accent': '#B8913A',       # muted brass (trophy) gold
    'rival': '#A4552F',        # clay — the "other side" in comparisons
    'rival_soft': '#F5E9E2',
    'bg': '#FAFAF7',
    'surface': '#FFFFFF',
    'border': '#E2E5DE',
    'text': '#18221D',
    'muted': '#5F6B64',
    'win': '#1E7A4C',
    'loss': '#B42318',
    'tie': '#9A6B12',
}

st.markdown(f"""
<style>
    /* ---------- layout ---------- */
    .block-container {{ padding-top: 2rem; max-width: 1180px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* ---------- masthead ---------- */
    .fbc-masthead {{
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; flex-wrap: wrap;
        padding: 0 0 1.1rem; margin-bottom: 0.25rem;
        border-bottom: 1px solid {COLORS['border']};
    }}
    .fbc-brand {{ display: flex; align-items: center; gap: 0.85rem; }}
    .fbc-crest {{
        width: 46px; height: 46px; border-radius: 50%;
        background: {COLORS['primary']}; color: #fff;
        display: grid; place-items: center; flex-shrink: 0;
        font: 700 0.85rem/1 'Source Serif 4', serif; letter-spacing: 0.5px;
        box-shadow: inset 0 0 0 2px {COLORS['primary']}, inset 0 0 0 3.5px {COLORS['accent']};
    }}
    .fbc-title {{
        font: 700 1.55rem/1.15 'Source Serif 4', serif; color: {COLORS['text']};
        margin: 0; letter-spacing: -0.2px;
    }}
    .fbc-sub {{ color: {COLORS['muted']}; font-size: 0.85rem; margin-top: 2px; }}
    .fbc-latest {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
        border-radius: 999px; padding: 0.4rem 0.9rem; font-size: 0.82rem; color: {COLORS['muted']};
    }}
    .fbc-latest b {{ color: {COLORS['text']}; font-weight: 600; }}
    .fbc-latest .dot {{
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: {COLORS['accent']}; margin-right: 6px; vertical-align: 1px;
    }}

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {COLORS['border']}; }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.7rem 0 0.6rem; font-weight: 500; color: {COLORS['muted']};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] p {{ font-size: 0.95rem; }}
    .stTabs [aria-selected="true"] {{ color: {COLORS['text']}; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {COLORS['primary']}; height: 2px; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* ---------- section headings ---------- */
    .section-header {{
        font: 600 1.2rem/1.3 'Source Serif 4', serif; color: {COLORS['text']};
        margin: 1.9rem 0 0.15rem; padding: 0; letter-spacing: -0.1px;
    }}
    .section-header.h4 {{ font-size: 1.05rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
    .section-note {{ color: {COLORS['muted']}; font-size: 0.85rem; margin: 0 0 0.7rem; }}

    /* ---------- KPI tiles ---------- */
    .kpi-grid {{
        display: grid; gap: 0.75rem; margin: 0.75rem 0 0.5rem;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }}
    .kpi {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 0.85rem 1rem 0.9rem;
    }}
    .kpi-label {{
        font-size: 0.72rem; font-weight: 600; color: {COLORS['muted']};
        text-transform: uppercase; letter-spacing: 0.6px;
    }}
    .kpi-value {{
        font-size: 1.6rem; font-weight: 700; color: {COLORS['text']};
        line-height: 1.2; margin-top: 0.3rem; font-variant-numeric: tabular-nums;
        letter-spacing: -0.4px; overflow-wrap: anywhere;
    }}
    .kpi-note {{ font-size: 0.8rem; color: {COLORS['muted']}; margin-top: 0.15rem; }}
    .kpi.p1 {{ border-top: 3px solid {COLORS['primary']}; }}
    .kpi.p2 {{ border-top: 3px solid {COLORS['rival']}; }}

    /* ---------- versus / prediction ---------- */
    .vs-bar {{ display: flex; height: 12px; border-radius: 999px; overflow: hidden; margin: 0.4rem 0 0.3rem; }}
    .vs-bar span:first-child {{ background: {COLORS['primary']}; }}
    .vs-bar span:last-child {{ background: {COLORS['rival']}; }}
    .vs-legend {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: {COLORS['muted']}; }}
    .vs-legend b {{ color: {COLORS['text']}; font-variant-numeric: tabular-nums; }}
    .side-label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: -0.4rem; }}
    .side-label.p1 {{ color: {COLORS['primary']}; }}
    .side-label.p2 {{ color: {COLORS['rival']}; }}

    .legend-chips {{ display: flex; gap: 0.5rem; flex-wrap: wrap; font-size: 0.8rem; color: {COLORS['muted']}; margin: 0 0 0.6rem; }}
    .legend-chips span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
    .legend-chips i {{ display: inline-block; width: 18px; height: 18px; border-radius: 4px; font-style: normal;
        font-size: 0.7rem; font-weight: 700; text-align: center; line-height: 18px; }}

    @media (max-width: 640px) {{
        .block-container {{ padding-top: 1rem; }}
        .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem; }}
        .kpi {{ padding: 0.7rem 0.8rem; }}
        .kpi-value {{ font-size: 1.25rem; }}
        .fbc-title {{ font-size: 1.25rem; }}
        .fbc-latest {{ border-radius: 10px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 1rem; }}
    }}
</style>
""", unsafe_allow_html=True)


def kpi_tiles(items, variant=None):
    """Render a responsive row of stat tiles. items: (label, value[, note]) tuples."""
    cls = f"kpi {variant}" if variant else "kpi"
    tiles = []
    for item in items:
        label, value = item[0], item[1]
        note = item[2] if len(item) > 2 and item[2] else ''
        note_html = f"<div class='kpi-note'>{html.escape(str(note))}</div>" if note else ''
        tiles.append(f"<div class='{cls}'><div class='kpi-label'>{html.escape(str(label))}</div>"
                     f"<div class='kpi-value'>{html.escape(str(value))}</div>{note_html}</div>")
    st.markdown(f"<div class='kpi-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)


def section(title, note=None, level=3):
    # A div (not <h3>) so Streamlit doesn't attach hover anchor-link icons
    st.markdown(f"<div class='section-header h{level}' role='heading' aria-level='{level}'>"
                f"{html.escape(title)}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<p class='section-note'>{note}</p>", unsafe_allow_html=True)


def vs_bar(left_label, left_pct, right_label, right_pct):
    """Two-sided probability bar (left = green side, right = clay side)."""
    st.markdown(f"""
    <div class="vs-bar"><span style="width:{left_pct:.1f}%"></span><span style="width:{right_pct:.1f}%"></span></div>
    <div class="vs-legend"><span>{html.escape(left_label)} <b>{left_pct:.0f}%</b></span>
    <span><b>{right_pct:.0f}%</b> {html.escape(right_label)}</span></div>
    """, unsafe_allow_html=True)


# Column helpers: keep numbers numeric so grid header-click sorting works
# (pre-formatted "%"-strings sort alphabetically: "9.1%" ranked above "50.0%").
def pct_column(label='Win %', bar=False):
    if bar:
        return st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0, max_value=100)
    return st.column_config.NumberColumn(label, format="%.1f%%")


def to_pct(series):
    return (series.astype(float) * 100).round(1)


def show_table(data, column_config=None, fit=False, **kwargs):
    """st.dataframe with house defaults. fit=True sizes the grid to show every row
    without an inner scrollbar (rows are 33px at the theme's 15px base font, plus the
    header and border)."""
    if fit:
        n = len(data.data) if hasattr(data, 'data') else len(data)
        kwargs['height'] = 33 * (n + 1) + 3
    st.dataframe(data, hide_index=True, width='stretch', column_config=column_config, **kwargs)

_FBC_COL_PATTERN = re.compile(r'^FBC\s+(\d+)$')


def _fbc_columns(df):
    """Return FBC column names from a DataFrame, ordered by cup number."""
    matches = [(int(_FBC_COL_PATTERN.match(c).group(1)), c)
               for c in df.columns if isinstance(c, str) and _FBC_COL_PATTERN.match(c)]
    matches.sort(key=lambda x: x[0])
    return matches


def _data_file_mtime():
    """Modification time of the data file — passed to cached loaders so the
    st.cache_data key changes (and the cache busts) when the Excel is edited
    while the app is running."""
    return os.path.getmtime('FBC_Data.xlsx')


@st.cache_data
def _load_cups_data_cached(file_mtime, path='FBC_Data.xlsx'):
    """Load and process the Cups data showing which players won each cup."""
    cups_raw = pd.read_excel(path, sheet_name='Cups', header=None)

    # Find the header row by locating the cell whose value is 'Player'.
    header_row_idx = None
    player_col_idx = None
    for r in range(cups_raw.shape[0]):
        for c in range(cups_raw.shape[1]):
            cell = cups_raw.iat[r, c]
            if isinstance(cell, str) and cell.strip() == 'Player':
                header_row_idx, player_col_idx = r, c
                break
        if header_row_idx is not None:
            break
    if header_row_idx is None:
        raise ValueError("Could not find 'Player' header in Cups sheet")

    # Map header cells in that row to the source column indices we care about.
    fbc_cols = []  # (cup_number, source_col_idx)
    total_col = played_col = pct_col = lost_col = None
    for c in range(cups_raw.shape[1]):
        cell = cups_raw.iat[header_row_idx, c]
        if not isinstance(cell, str):
            continue
        s = cell.strip()
        m = _FBC_COL_PATTERN.match(s)
        if m:
            fbc_cols.append((int(m.group(1)), c))
        elif s == 'Total':
            total_col = c
        elif s == 'Played':
            played_col = c
        elif s == '%':
            pct_col = c
        elif s == 'Lost':
            lost_col = c
    fbc_cols.sort(key=lambda x: x[0])

    # Walk rows below the header; stop at the first blank, 'Total', or note row.
    player_rows = []
    for r in range(header_row_idx + 1, cups_raw.shape[0]):
        player_val = cups_raw.iat[r, player_col_idx]
        if not isinstance(player_val, str) or not player_val.strip():
            break
        name = player_val.strip()
        if name.lower() == 'total':
            break
        fbc_vals = [cups_raw.iat[r, src_c] for _, src_c in fbc_cols]
        if not fbc_vals or all(pd.isna(v) for v in fbc_vals):
            break  # note / descriptive row with no per-cup data

        row_dict = {'Player': name}
        for num, src_c in fbc_cols:
            row_dict[f'FBC {num}'] = cups_raw.iat[r, src_c]
        row_dict['Total'] = cups_raw.iat[r, total_col] if total_col is not None else None
        row_dict['Played'] = cups_raw.iat[r, played_col] if played_col is not None else None
        row_dict['Win%'] = cups_raw.iat[r, pct_col] if pct_col is not None else None
        row_dict['Lost'] = cups_raw.iat[r, lost_col] if lost_col is not None else None
        player_rows.append(row_dict)

    columns = ['Player'] + [f'FBC {n}' for n, _ in fbc_cols] + ['Total', 'Played', 'Win%', 'Lost']
    cups_df = pd.DataFrame(player_rows, columns=columns)

    # Derive Total/Played/Lost/Win% from the per-cup 1/0/X cells rather than trusting
    # the sheet's summary columns: those are Excel formulas, and their cached values
    # disappear whenever the file is saved by a tool other than Excel (reading them
    # then yields NaN). The per-cup cells are literal values and always reliable.
    fbc_names = [f'FBC {n}' for n, _ in fbc_cols]
    def _count(row, accepted):
        return sum(1 for c in fbc_names if row[c] in accepted)
    cups_df['Total'] = cups_df.apply(lambda r: _count(r, (1, '1')), axis=1)
    cups_df['Lost'] = cups_df.apply(lambda r: _count(r, (0, '0')), axis=1)
    cups_df['Played'] = cups_df['Total'] + cups_df['Lost']
    # NaN (not a crash) for a player with no cups played yet, e.g. a roster placeholder
    cups_df['Win%'] = cups_df['Total'] / cups_df['Played'].where(cups_df['Played'] > 0)

    return cups_df


def load_cups_data():
    return _load_cups_data_cached(_data_file_mtime())


@st.cache_data
def _load_cup_info_cached(file_mtime, path='FBC_Data.xlsx'):
    """Load the Cup Info sheet: one manually entered row per cup.

    Returns {fbc_number: {'rain': True/False/None, 'notes': str}}. Columns are found
    by header text ('FBC', 'Rain', 'Notes'), not position. Rain is 'Yes'/'No'; anything
    else (including blank) is treated as unknown. A missing sheet returns {} so the
    rest of the app keeps working.
    """
    try:
        raw = pd.read_excel(path, sheet_name='Cup Info', header=None)
    except ValueError:  # sheet not present
        return {}

    header_row = fbc_col = rain_col = notes_col = None
    for r in range(raw.shape[0]):
        labels = {str(raw.iat[r, c]).strip(): c for c in range(raw.shape[1])
                  if isinstance(raw.iat[r, c], str)}
        if 'FBC' in labels and 'Rain' in labels:
            header_row, fbc_col = r, labels['FBC']
            rain_col, notes_col = labels['Rain'], labels.get('Notes')
            break
    if header_row is None:
        return {}

    info = {}
    for r in range(header_row + 1, raw.shape[0]):
        fbc = pd.to_numeric(raw.iat[r, fbc_col], errors='coerce')
        if pd.isna(fbc):
            continue
        rain_val = raw.iat[r, rain_col]
        rain_txt = rain_val.strip().lower() if isinstance(rain_val, str) else ''
        rain = True if rain_txt == 'yes' else (False if rain_txt == 'no' else None)
        note = raw.iat[r, notes_col] if notes_col is not None else None
        info[int(fbc)] = {'rain': rain,
                          'notes': note.strip() if isinstance(note, str) else ''}
    return info


def load_cup_info():
    try:
        return _load_cup_info_cached(_data_file_mtime())
    except Exception:
        return {}


def _rain_label(rain):
    return 'Yes' if rain is True else ('No' if rain is False else '')


def get_cups_summary(cups_df):
    """Get summary statistics about cup wins."""
    fbc_cols = _fbc_columns(cups_df)
    summary = []
    for _, row in cups_df.iterrows():
        player = row['Player']
        total_wins = row['Total'] if pd.notna(row['Total']) else 0
        total_played = row['Played'] if pd.notna(row['Played']) else 0
        win_pct = row['Win%'] if pd.notna(row['Win%']) else 0

        # Count individual cup results
        cup_results = []
        for num, col in fbc_cols:
            result = row.get(col, 'X')
            if result == 1 or result == '1':
                cup_results.append(f"FBC {num}: Won")
            elif result == 0 or result == '0':
                cup_results.append(f"FBC {num}: Lost")
            # X means didn't participate

        summary.append({
            'Player': player,
            'Cups Won': int(total_wins),
            'Cups Played': int(total_played),
            'Cup Win%': win_pct,
            'Cup Results': cup_results
        })

    return sorted(summary, key=lambda x: x['Cups Won'], reverse=True)

@st.cache_data
def _load_data_cached(file_mtime, path='FBC_Data.xlsx'):
    """Load and process the FBC data."""
    df = pd.read_excel(path, sheet_name='Archives')

    # Clean up the data
    df = df.dropna(subset=['Player 1'])
    df = df[df['Player 1'].apply(lambda x: isinstance(x, str))]

    # Dates typed as text (rather than real Excel dates) would leave this column with
    # mixed types and break every sort and .dt access downstream. Coerce them to NaT;
    # validate_data reports the affected rows.
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Derive Singles/Doubles column if missing from older data files
    if 'Singles/Doubles' not in df.columns:
        df['Singles/Doubles'] = df.apply(
            lambda row: 'Doubles' if pd.notna(row.get('Player 2')) else
                        ('Singles' if pd.notna(row.get('Singles Opponent')) else 'FTAS'),
            axis=1
        )

    return df


def load_data():
    return _load_data_cached(_data_file_mtime())


def get_player_stats(df, player):
    """Calculate career stats for a player."""
    # Get matches where player participated (as Player 1 or Player 2)
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)].copy()

    if len(player_matches) == 0:
        return None

    total_matches = len(player_matches)
    wins = player_matches['W'].sum()
    losses = player_matches['L'].sum()
    ties = player_matches['T'].sum()
    points = player_matches['Points earned'].sum()
    win_pct = (wins + 0.5 * ties) / total_matches if total_matches > 0 else 0

    # Events attended
    events = player_matches['FBC'].nunique()

    return {
        'matches': total_matches,
        'wins': int(wins),
        'losses': int(losses),
        'ties': int(ties),
        'points': points,
        'win_pct': win_pct,
        'events': events,
        'record': f"{int(wins)}-{int(losses)}-{int(ties)}"
    }

def get_player_by_event(df, player):
    """Get player's record broken down by FBC event."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)].copy()

    if len(player_matches) == 0:
        return pd.DataFrame()

    event_stats = player_matches.groupby('FBC').agg({
        'W': 'sum',
        'L': 'sum',
        'T': 'sum',
        'Points earned': 'sum',
        'Geographic Location': 'first'
    }).reset_index()

    event_stats['Matches'] = event_stats['W'] + event_stats['L'] + event_stats['T']
    event_stats['Win%'] = (event_stats['W'] + 0.5 * event_stats['T']) / event_stats['Matches']
    event_stats['Record'] = event_stats.apply(lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1)

    event_stats = event_stats.rename(columns={
        'FBC': 'Event',
        'Geographic Location': 'Location',
        'Points earned': 'Points'
    })

    return event_stats[['Event', 'Location', 'Record', 'Win%', 'Points', 'Matches']].sort_values('Event')


# Display-only spellings for the Archives 'Team' (captain) column. FBC 7 had co-captains
# entered as "Ferrin/Jax"; everywhere else Jackson is "Jackson".
TEAM_DISPLAY_ALIASES = {'Jax': 'Jackson'}


def team_label(team):
    if not isinstance(team, str):
        return team
    return '/'.join(TEAM_DISPLAY_ALIASES.get(t.strip(), t.strip()) for t in team.split('/'))


def fmt_pts(x):
    """24.0 -> '24', 24.5 -> '24.5'."""
    return f"{x:g}" if x is not None and pd.notna(x) else ''

def get_partner_performance(df, player):
    """Get player's record with each doubles partner."""
    if 'Singles/Doubles' not in df.columns:
        return pd.DataFrame()
    doubles_matches = df[df['Singles/Doubles'] == 'Doubles'].copy()

    # Matches where player is Player 1
    as_p1 = doubles_matches[doubles_matches['Player 1'] == player].copy()
    as_p1['Partner'] = as_p1['Player 2']

    # Matches where player is Player 2
    as_p2 = doubles_matches[doubles_matches['Player 2'] == player].copy()
    as_p2['Partner'] = as_p2['Player 1']

    all_matches = pd.concat([as_p1, as_p2])

    if len(all_matches) == 0:
        return pd.DataFrame()

    partner_stats = all_matches.groupby('Partner').agg({
        'W': 'sum',
        'L': 'sum',
        'T': 'sum',
        'Points earned': 'sum'
    }).reset_index()

    partner_stats['Matches'] = partner_stats['W'] + partner_stats['L'] + partner_stats['T']
    partner_stats['Win%'] = (partner_stats['W'] + 0.5 * partner_stats['T']) / partner_stats['Matches']
    partner_stats['Record'] = partner_stats.apply(lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1)

    partner_stats = partner_stats.rename(columns={'Points earned': 'Points'})

    return partner_stats[['Partner', 'Record', 'Win%', 'Points', 'Matches']].sort_values('Matches', ascending=False)

@st.cache_data
def get_all_partnership_stats(df):
    """Calculate statistics for all doubles partnerships.

    Returns a list of partnership records sorted by wins (descending).
    Each partnership is identified by the two player names (in alphabetical order).
    """
    doubles_matches = df[df['Singles/Doubles'] == 'Doubles'].copy()

    if len(doubles_matches) == 0:
        return []

    # Create a canonical partnership key (alphabetically sorted names)
    doubles_matches['Partnership'] = doubles_matches.apply(
        lambda row: tuple(sorted([row['Player 1'], row['Player 2']]))
        if pd.notna(row['Player 1']) and pd.notna(row['Player 2']) else None,
        axis=1
    )

    # Remove rows without valid partnerships
    doubles_matches = doubles_matches[doubles_matches['Partnership'].notna()]

    # Group by partnership and calculate stats
    partnership_stats = doubles_matches.groupby('Partnership').agg({
        'W': 'sum',
        'L': 'sum',
        'T': 'sum',
        'Points earned': 'sum',
        'FBC': 'nunique'  # Number of events played together
    }).reset_index()

    partnership_stats['Matches'] = partnership_stats['W'] + partnership_stats['L'] + partnership_stats['T']
    partnership_stats['Win%'] = (partnership_stats['W'] + 0.5 * partnership_stats['T']) / partnership_stats['Matches']
    partnership_stats['Record'] = partnership_stats.apply(
        lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1
    )

    # Convert to list of dicts for easier handling
    results = []
    for _, row in partnership_stats.iterrows():
        p1, p2 = row['Partnership']
        results.append({
            'Partner1': p1,
            'Partner2': p2,
            'Partnership': f"{p1} & {p2}",
            'Wins': int(row['W']),
            'Losses': int(row['L']),
            'Ties': int(row['T']),
            'Record': row['Record'],
            'Win%': row['Win%'],
            'Matches': int(row['Matches']),
            'Points': row['Points earned'],
            'Events': int(row['FBC'])
        })

    # Sort by wins (descending), then by win% (descending)
    return sorted(results, key=lambda x: (x['Wins'], x['Win%']), reverse=True)

@st.cache_data
def get_opponent_pairs_never_partnered(df):
    """Calculate all opponent pairs and identify those who have never been doubles partners.

    An 'opponent pair' is any two players who appeared on opposite sides in a match:
    - Singles: Player 1 vs their Singles Opponent
    - Doubles: each player on one team vs each player on the opposing team
      (FTAS rows carry no opponent columns, so the tiebreaker is not counted)

    Returns a list of dicts sorted by opponent match count (descending), filtered to
    only include pairs that have NEVER been doubles partners.
    """
    from collections import Counter

    opponent_pair_counts = Counter()

    for _, row in df.iterrows():
        p1 = row.get('Player 1')
        p2 = row.get('Player 2')
        opp1 = row.get('Opponent1')
        opp2 = row.get('Opponent2')
        singles_opp = row.get('Singles Opponent')
        match_type = row.get('Singles/Doubles')

        if match_type == 'Singles':
            opponent = singles_opp if pd.notna(singles_opp) else opp1
            if pd.notna(p1) and pd.notna(opponent) and isinstance(p1, str) and isinstance(opponent, str):
                pair = tuple(sorted([p1, opponent]))
                opponent_pair_counts[pair] += 1
        else:
            # Doubles: each player on one side vs each on the other (FTAS rows list no opponents)
            team = [p for p in [p1, p2] if pd.notna(p) and isinstance(p, str)]
            opponents = [p for p in [opp1, opp2] if pd.notna(p) and isinstance(p, str)]
            for t in team:
                for o in opponents:
                    pair = tuple(sorted([t, o]))
                    opponent_pair_counts[pair] += 1

    # Get all doubles partnership pairs (players who have been on the same team)
    doubles_df = df[df['Singles/Doubles'] == 'Doubles']
    partnership_pairs = set()
    for _, row in doubles_df.iterrows():
        p1 = row.get('Player 1')
        p2 = row.get('Player 2')
        if pd.notna(p1) and pd.notna(p2) and isinstance(p1, str) and isinstance(p2, str):
            partnership_pairs.add(tuple(sorted([p1, p2])))

    # Filter to pairs that have NEVER been doubles partners
    never_partnered = []
    for pair, count in opponent_pair_counts.items():
        if pair not in partnership_pairs:
            never_partnered.append({
                'Player1': pair[0],
                'Player2': pair[1],
                'Pair': f"{pair[0]} & {pair[1]}",
                'OpponentMatches': count
            })

    never_partnered.sort(key=lambda x: x['OpponentMatches'], reverse=True)
    return never_partnered

def get_specific_partnership_stats(df, player1, player2):
    """Get the record for a specific doubles partnership."""
    doubles_matches = df[df['Singles/Doubles'] == 'Doubles'].copy()

    # Find matches where both players were partners (in either order)
    partnership_matches = doubles_matches[
        ((doubles_matches['Player 1'] == player1) & (doubles_matches['Player 2'] == player2)) |
        ((doubles_matches['Player 1'] == player2) & (doubles_matches['Player 2'] == player1))
    ]

    if len(partnership_matches) == 0:
        return None

    wins = int(partnership_matches['W'].sum())
    losses = int(partnership_matches['L'].sum())
    ties = int(partnership_matches['T'].sum())
    points = partnership_matches['Points earned'].sum()
    matches = len(partnership_matches)
    events = partnership_matches['FBC'].nunique()

    return {
        'Partner1': player1,
        'Partner2': player2,
        'Partnership': f"{player1} & {player2}",
        'Wins': wins,
        'Losses': losses,
        'Ties': ties,
        'Record': f"{wins}-{losses}-{ties}",
        'Win%': (wins + 0.5 * ties) / matches if matches > 0 else 0,
        'Matches': matches,
        'Points': points,
        'Events': events
    }

def get_aggregate_group_doubles_stats(df, player_group):
    """Calculate aggregate doubles stats where BOTH partners are in the specified group.

    Args:
        df: DataFrame with match data
        player_group: List of player names to consider as a group

    Returns:
        Dictionary with aggregate stats, individual partnership breakdowns, and excluded matches info
    """
    doubles_matches = df[df['Singles/Doubles'] == 'Doubles'].copy()

    if len(doubles_matches) == 0 or len(player_group) < 2:
        return None

    # Normalize player group to lowercase for matching
    player_group_lower = [p.lower() for p in player_group]

    # Find matches where BOTH Player 1 AND Player 2 are in the group
    qualifying_matches = doubles_matches[
        (doubles_matches['Player 1'].str.lower().isin(player_group_lower)) &
        (doubles_matches['Player 2'].str.lower().isin(player_group_lower))
    ]

    # Also track matches where only ONE partner is in the group (for transparency)
    one_in_group = doubles_matches[
        ((doubles_matches['Player 1'].str.lower().isin(player_group_lower)) &
         (~doubles_matches['Player 2'].str.lower().isin(player_group_lower))) |
        ((~doubles_matches['Player 1'].str.lower().isin(player_group_lower)) &
         (doubles_matches['Player 2'].str.lower().isin(player_group_lower)))
    ]

    if len(qualifying_matches) == 0:
        return {
            'group': player_group,
            'total_wins': 0,
            'total_losses': 0,
            'total_ties': 0,
            'total_matches': 0,
            'record': '0-0-0',
            'win_pct': 0.0,
            'total_points': 0.0,
            'partnerships': [],
            'excluded_matches': len(one_in_group),
            'excluded_note': f"Excluded {len(one_in_group)} matches where only one partner was in the group"
        }

    # Calculate aggregate totals
    total_wins = int(qualifying_matches['W'].sum())
    total_losses = int(qualifying_matches['L'].sum())
    total_ties = int(qualifying_matches['T'].sum())
    total_matches = len(qualifying_matches)
    total_points = qualifying_matches['Points earned'].sum()
    win_pct = (total_wins + 0.5 * total_ties) / total_matches if total_matches > 0 else 0

    # Get breakdown by individual partnerships within the group
    partnerships = []
    player_group_normalized = sorted([p for p in player_group])

    for i in range(len(player_group_normalized)):
        for j in range(i + 1, len(player_group_normalized)):
            p1, p2 = player_group_normalized[i], player_group_normalized[j]
            pstats = get_specific_partnership_stats(df, p1, p2)
            if pstats and pstats['Matches'] > 0:
                partnerships.append(pstats)

    return {
        'group': player_group,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_ties': total_ties,
        'total_matches': total_matches,
        'record': f"{total_wins}-{total_losses}-{total_ties}",
        'win_pct': win_pct,
        'total_points': total_points,
        'partnerships': partnerships,
        'excluded_matches': len(one_in_group),
        'excluded_note': f"Excluded {len(one_in_group)} matches where only one partner was in the group"
    }


def get_head_to_head(df, player):
    """Get player's head-to-head record against all opponents."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)].copy()

    if len(player_matches) == 0:
        return pd.DataFrame()

    # Get all opponents faced
    opponents = []
    for _, row in player_matches.iterrows():
        opp1 = row.get('Opponent1')
        opp2 = row.get('Opponent2')
        singles_opp = row.get('Singles Opponent')

        if pd.notna(singles_opp) and singles_opp != player:
            opponents.append({
                'Opponent': singles_opp,
                'W': row['W'],
                'L': row['L'],
                'T': row['T']
            })
        else:
            if pd.notna(opp1) and opp1 != player:
                opponents.append({
                    'Opponent': opp1,
                    'W': row['W'],
                    'L': row['L'],
                    'T': row['T']
                })
            if pd.notna(opp2) and opp2 != player:
                opponents.append({
                    'Opponent': opp2,
                    'W': row['W'],
                    'L': row['L'],
                    'T': row['T']
                })

    if not opponents:
        return pd.DataFrame()

    opp_df = pd.DataFrame(opponents)
    opp_stats = opp_df.groupby('Opponent').agg({
        'W': 'sum',
        'L': 'sum',
        'T': 'sum'
    }).reset_index()

    opp_stats['Matches'] = opp_stats['W'] + opp_stats['L'] + opp_stats['T']
    opp_stats['Win%'] = (opp_stats['W'] + 0.5 * opp_stats['T']) / opp_stats['Matches']
    opp_stats['Record'] = opp_stats.apply(lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1)

    return opp_stats[['Opponent', 'Record', 'Win%', 'Matches']].sort_values('Matches', ascending=False)

def get_course_performance(df, player):
    """Get player's record by course."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)].copy()

    if len(player_matches) == 0:
        return pd.DataFrame()

    course_stats = player_matches.groupby('Course').agg({
        'W': 'sum',
        'L': 'sum',
        'T': 'sum',
        'Points earned': 'sum'
    }).reset_index()

    course_stats['Matches'] = course_stats['W'] + course_stats['L'] + course_stats['T']
    course_stats['Win%'] = (course_stats['W'] + 0.5 * course_stats['T']) / course_stats['Matches']
    course_stats['Record'] = course_stats.apply(lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1)

    course_stats = course_stats.rename(columns={'Points earned': 'Points'})

    # Most-played courses first: sorting by Win% put one-off 1-0-0 rounds at the top
    return course_stats[['Course', 'Record', 'Win%', 'Points', 'Matches']].sort_values(
        ['Matches', 'Win%'], ascending=[False, False])

@st.cache_data
def get_leaderboard(df):
    """Calculate overall leaderboard using vectorized groupby."""
    p1 = df[['Player 1', 'W', 'L', 'T', 'Points earned', 'FBC']].rename(columns={'Player 1': 'Player'})
    p2 = df[df['Player 2'].notna()][['Player 2', 'W', 'L', 'T', 'Points earned', 'FBC']].rename(columns={'Player 2': 'Player'})
    stacked = pd.concat([p1, p2], ignore_index=True)
    stacked = stacked[stacked['Player'].apply(lambda x: isinstance(x, str))]

    agg = stacked.groupby('Player').agg(
        Points=('Points earned', 'sum'),
        W=('W', 'sum'),
        L=('L', 'sum'),
        T=('T', 'sum'),
        Events=('FBC', 'nunique')
    ).reset_index()

    agg['Matches'] = agg['W'] + agg['L'] + agg['T']
    agg['Win%'] = (agg['W'] + 0.5 * agg['T']) / agg['Matches']
    agg['Record'] = agg.apply(lambda r: f"{int(r['W'])}-{int(r['L'])}-{int(r['T'])}", axis=1)
    agg['Pts/Event'] = agg['Points'] / agg['Events']

    return agg[['Player', 'Points', 'Record', 'Win%', 'Matches', 'Events', 'Pts/Event']].sort_values('Points', ascending=False).reset_index(drop=True)

def extract_fbc_number(question):
    """Extract FBC event number from a question if mentioned (e.g. 'FBC 11', 'FBC11').

    FBC events are referred to by arabic numerals only.
    """
    match = re.search(r'fbc\s*(\d+)', question.lower())
    return int(match.group(1)) if match else None

def extract_player_names(question, all_players):
    """Extract player names mentioned in a question.

    Special handling for the two Connollys in the data:
      - 'Connolly' = Brett Connolly (the more common one)
      - 'R. Connolly' = Rick Connolly
    When the user says just 'Connolly', we assume Brett unless Rick is indicated.
    """
    question_lower = question.lower()
    mentioned = []

    # Connolly disambiguation constants
    connolly_brett = 'Connolly'
    connolly_rick = 'R. Connolly'
    has_both_connollys = connolly_brett in all_players and connolly_rick in all_players

    rick_mentioned = False
    brett_mentioned = False
    bare_connolly = False

    if has_both_connollys:
        # Detect Rick Connolly references (word-boundary matching to avoid e.g. "trick")
        rick_mentioned = bool(re.search(r'\brick\b', question_lower)) or \
                         bool(re.search(r'\br\.?\s*connolly', question_lower)) or \
                         bool(re.search(r'connolly,?\s*r\b', question_lower))

        # Detect Brett Connolly references
        brett_mentioned = bool(re.search(r'\bbrett\b', question_lower))

        # Detect "both Connollys" references
        both_connollys = bool(re.search(r'\bconnollys\b', question_lower)) or \
                         bool(re.search(r'\bboth\s+connolly', question_lower))

        # Check if "connolly" appears standalone (not just as part of a Rick-style reference).
        # Remove Rick-style references to see if a bare "connolly" remains.
        cleaned = re.sub(r'\brick\s+connolly\b', '', question_lower)
        cleaned = re.sub(r'\br\.?\s*connolly\b', '', cleaned)
        cleaned = re.sub(r'\bconnolly,?\s*r\b', '', cleaned)
        bare_connolly = 'connolly' in cleaned

    for player in all_players:
        if player.lower() in question_lower:
            # Special handling for Connolly disambiguation
            if has_both_connollys and player == connolly_brett:
                if both_connollys:
                    pass  # Keep Brett
                elif rick_mentioned and not brett_mentioned and not bare_connolly:
                    # "connolly" substring matched but user only means Rick — skip Brett
                    continue
            mentioned.append(player)

    # Add players referenced by first name only (won't match via substring)
    if has_both_connollys:
        if both_connollys:
            if connolly_brett not in mentioned:
                mentioned.append(connolly_brett)
            if connolly_rick not in mentioned:
                mentioned.append(connolly_rick)
        else:
            if rick_mentioned and connolly_rick not in mentioned:
                mentioned.append(connolly_rick)
            if brett_mentioned and connolly_brett not in mentioned:
                mentioned.append(connolly_brett)

    return mentioned

def extract_course_names(question, all_courses):
    """Extract course names mentioned in a question."""
    question_lower = question.lower()
    mentioned = []
    for course in all_courses:
        if isinstance(course, str) and course.lower() in question_lower:
            mentioned.append(course)
    # Also check for partial matches (e.g., "Pebble Beach" in "Pebble Beach Golf Links")
    for course in all_courses:
        if isinstance(course, str):
            # Check if any significant word from the question matches the course
            words = question_lower.split()
            for word in words:
                if len(word) > 4 and word in course.lower() and course not in mentioned:
                    mentioned.append(course)
    return mentioned

def calculate_player_stats_for_subset(df, players=None):
    """Calculate stats for all players (or a subset) in a given DataFrame."""
    p1 = df[['Player 1', 'W', 'L', 'T', 'Points earned']].rename(columns={'Player 1': 'Player'})
    p2 = df[df['Player 2'].notna()][['Player 2', 'W', 'L', 'T', 'Points earned']].rename(columns={'Player 2': 'Player'})
    stacked = pd.concat([p1, p2], ignore_index=True)
    stacked = stacked[stacked['Player'].apply(lambda x: isinstance(x, str))]

    if players is not None:
        stacked = stacked[stacked['Player'].isin(players)]

    if stacked.empty:
        return []

    agg = stacked.groupby('Player').agg(
        Points=('Points earned', 'sum'),
        Wins=('W', 'sum'),
        Losses=('L', 'sum'),
        Ties=('T', 'sum')
    ).reset_index()

    agg['Matches'] = agg['Wins'] + agg['Losses'] + agg['Ties']
    agg['Win%'] = (agg['Wins'] + 0.5 * agg['Ties']) / agg['Matches']
    agg['Record'] = agg.apply(lambda r: f"{int(r['Wins'])}-{int(r['Losses'])}-{int(r['Ties'])}", axis=1)

    result = []
    for _, row in agg.iterrows():
        result.append({
            'Player': row['Player'],
            'Points': row['Points'],
            'Wins': int(row['Wins']),
            'Losses': int(row['Losses']),
            'Ties': int(row['Ties']),
            'Matches': int(row['Matches']),
            'Win%': row['Win%'],
            'Record': row['Record']
        })
    return sorted(result, key=lambda x: x['Points'], reverse=True)

@st.cache_data
def get_all_individual_fbc_performances(df):
    """Calculate per-player, per-FBC event performance stats using vectorized groupby."""
    p1 = df[['Player 1', 'W', 'L', 'T', 'Points earned', 'FBC', 'Geographic Location']].rename(columns={'Player 1': 'Player'})
    p2 = df[df['Player 2'].notna()][['Player 2', 'W', 'L', 'T', 'Points earned', 'FBC', 'Geographic Location']].rename(columns={'Player 2': 'Player'})
    stacked = pd.concat([p1, p2], ignore_index=True)
    stacked = stacked[stacked['Player'].apply(lambda x: isinstance(x, str))]

    event_locations = df.groupby('FBC')['Geographic Location'].first().to_dict()

    agg = stacked.groupby(['Player', 'FBC']).agg(
        Wins=('W', 'sum'),
        Losses=('L', 'sum'),
        Ties=('T', 'sum'),
        Points=('Points earned', 'sum')
    ).reset_index()

    agg['Matches'] = agg['Wins'] + agg['Losses'] + agg['Ties']
    agg['Win%'] = (agg['Wins'] + 0.5 * agg['Ties']) / agg['Matches']
    agg['Record'] = agg.apply(lambda r: f"{int(r['Wins'])}-{int(r['Losses'])}-{int(r['Ties'])}", axis=1)
    agg['PPM'] = agg['Points'] / agg['Matches']
    agg['Location'] = agg['FBC'].map(event_locations).fillna('Unknown')
    agg['FBC'] = agg['FBC'].astype(int)

    result = []
    for _, row in agg.iterrows():
        result.append({
            'Player': row['Player'],
            'FBC': row['FBC'],
            'Location': row['Location'],
            'Points': row['Points'],
            'Wins': int(row['Wins']),
            'Losses': int(row['Losses']),
            'Ties': int(row['Ties']),
            'Matches': int(row['Matches']),
            'Record': row['Record'],
            'Win%': row['Win%'],
            'PPM': row['PPM']
        })
    return result

def _team_event_totals(event):
    """Sum each team's points for one FBC event, scoring the FTAS correctly.

    Most formats are entered as one row per team per match, so a plain sum is right.
    The FTAS (Full Team Alternate Shot) is the exception: it is a single sudden-death
    tiebreaker worth only 0.5 points TO THE WINNING TEAM, but it is recorded as one
    row PER PLAYER (e.g. 11 rows per team) so that each player's individual record
    reflects it. Summing those rows would count the FTAS ~11x and can even flip the
    event winner (it did for FBC 4). So for FTAS we collapse each team's rows for that
    match to a single team result (the shared per-player value) instead of summing.
    See https://www.freddiebcup.com/ftas-rules — "0.5 points to the winning team."
    """
    is_ftas = (event['Singles/Doubles'].astype(str) == 'FTAS')
    if 'Format' in event.columns:
        is_ftas = is_ftas | (event['Format'].astype(str) == 'FTAS')

    totals = {}
    # Non-FTAS rows: straightforward sum per team
    normal = event[~is_ftas]
    for team, pts in normal.groupby('Team')['Points earned'].sum().items():
        totals[team] = totals.get(team, 0.0) + float(pts)
    # FTAS rows: one shared team result per match (use the mean of the team's rows,
    # which equals the single team point since every player on a team shares the result)
    ftas = event[is_ftas]
    if len(ftas) > 0:
        for (mid, team), grp in ftas.groupby(['UniqueMatchID', 'Team']):
            totals[team] = totals.get(team, 0.0) + float(grp['Points earned'].mean())

    return pd.Series(totals).sort_values(ascending=False)


def get_fbc_team_results(df, fbc_num=None):
    """Compute team standings, captains, and margins of victory for each FBC event.

    In the FBC, every match row carries a 'Team' value which is the name of that
    team's CAPTAIN (teams are named after their captain). Each team's final point
    total is the points it earned across the event (with the FTAS tiebreaker scored
    once, not per player — see _team_event_totals). The winning team has the most
    points; the margin of victory is the gap to the runner-up.

    Returns a list of dicts (one per event), each with:
      fbc, location, teams (captain -> points, sorted high to low),
      winner (captain), loser (captain, runner-up), margin, num_teams, tie.
    """
    if 'Team' not in df.columns:
        return []

    work = df[df['FBC'].notna() & df['Team'].notna()]
    if fbc_num is not None:
        work = work[work['FBC'] == fbc_num]

    results = []
    for fbc in sorted(work['FBC'].dropna().unique()):
        event = work[work['FBC'] == fbc]
        totals = _team_event_totals(event)
        if len(totals) == 0:
            continue
        location = event['Geographic Location'].dropna().iloc[0] if event['Geographic Location'].notna().any() else 'Unknown'
        # Captain Size (e.g. "Under 6'") is the draft theme for each team, when present
        sizes = {}
        for cap in totals.index:
            cs = event[event['Team'] == cap]['Captain Size'].dropna()
            if len(cs) > 0:
                sizes[cap] = str(cs.iloc[0])
        winner = totals.index[0]
        runner_up = totals.index[1] if len(totals) >= 2 else None
        margin = float(totals.iloc[0] - totals.iloc[1]) if len(totals) >= 2 else 0.0
        results.append({
            'fbc': int(fbc),
            'location': location,
            'teams': [(cap, float(pts)) for cap, pts in totals.items()],
            'sizes': sizes,
            'winner': winner,
            'loser': runner_up,
            'margin': margin,
            'num_teams': len(totals),
            'tie': margin == 0.0 and len(totals) >= 2,
        })
    return results


def get_cup_results_table(df):
    """Build a per-event summary table of every FBC cup result.

    One row per FBC event with: winning captain, losing captain, location,
    each team's point total, the margin of victory, and the highest individual
    point total at that event (the event's top scorer). Team totals use the
    FTAS-correct scoring from get_fbc_team_results. The top-scorer figure is the
    player's total Points earned at the event (which, like all individual stats,
    includes their 0.5 FTAS share).
    """
    results = get_fbc_team_results(df)
    cup_info = load_cup_info()
    rows = []
    for r in results:
        event = df[df['FBC'] == r['fbc']]
        info = cup_info.get(r['fbc'], {})
        stats = calculate_player_stats_for_subset(event)  # sorted by Points desc
        if stats:
            top_pts = max(s['Points'] for s in stats)
            top_players = [s['Player'] for s in stats if abs(s['Points'] - top_pts) < 1e-9]
            top_label = f"{', '.join(top_players)} ({top_pts:.1f})"
        else:
            top_label = ""
        loser_pts = r['teams'][1][1] if len(r['teams']) >= 2 else None
        rows.append({
            'FBC': r['fbc'],
            'Location': r['location'],
            'Winning Captain': r['winner'],
            'Losing Captain': r['loser'] or '',
            'Winner Total': round(r['teams'][0][1], 1),
            'Loser Total': round(loser_pts, 1) if loser_pts is not None else None,
            'Margin': round(r['margin'], 1),
            'Highest Individual': top_label,
            'Rain': _rain_label(info.get('rain')),
            'Notes': info.get('notes', ''),
        })
    return pd.DataFrame(rows)


@st.cache_data
def get_streak_records(df):
    """Compute per-player match streaks (win / unbeaten / losing / current).

    Matches are ordered by FBC event first, then Date and match number — FBC number is
    the reliable event sequence (rain-delayed makeup matches, like FBC 8's, can carry
    dates later than the following event, and date typos shouldn't scramble streaks).
    Returns a dict with 'win', 'unbeaten', 'loss' (each a list sorted by streak length)
    and 'current' (active streaks for players who played the most recent event).
    """
    work = df[df['FBC'].notna()].copy()
    work['_mn'] = pd.to_numeric(work['Match #'], errors='coerce').fillna(999)
    work = work.sort_values(['FBC', 'Date', '_mn'])
    latest_fbc = int(work['FBC'].max())

    # Build each player's chronological result sequence: (result, fbc)
    sequences = {}
    for _, row in work.iterrows():
        res = 'W' if row['W'] == 1 else ('T' if row['T'] == 1 else 'L')
        fbc = int(row['FBC'])
        for p in [row['Player 1'], row.get('Player 2')]:
            if isinstance(p, str):
                sequences.setdefault(p, []).append((res, fbc))

    def longest_run(seq, accept):
        """Longest consecutive run of results satisfying accept(); returns (len, start_fbc, end_fbc)."""
        best = (0, None, None)
        cur_len, cur_start = 0, None
        for res, fbc in seq:
            if accept(res):
                if cur_len == 0:
                    cur_start = fbc
                cur_len += 1
                if cur_len > best[0]:
                    best = (cur_len, cur_start, fbc)
            else:
                cur_len = 0
        return best

    win_list, unbeaten_list, loss_list, current_list = [], [], [], []
    for player, seq in sequences.items():
        for target, accept in [(win_list, lambda r: r == 'W'),
                               (unbeaten_list, lambda r: r != 'L'),
                               (loss_list, lambda r: r == 'L')]:
            length, start, end = longest_run(seq, accept)
            if length > 0:
                span = f"FBC {start}" if start == end else f"FBC {start}–{end}"
                target.append({'Player': player, 'Streak': length, 'Span': span})

        # Current (active) streak: trailing run of same result type — only meaningful
        # for players who actually played the most recent event
        if seq[-1][1] == latest_fbc:
            last_res = seq[-1][0]
            cur = 0
            for res, _ in reversed(seq):
                if res == last_res:
                    cur += 1
                else:
                    break
            label = {'W': 'Won', 'L': 'Lost', 'T': 'Tied'}[last_res]
            current_list.append({'Player': player, 'Streak': f"{label} last {cur}", 'Length': cur, 'Type': last_res})

    win_list.sort(key=lambda x: x['Streak'], reverse=True)
    unbeaten_list.sort(key=lambda x: x['Streak'], reverse=True)
    loss_list.sort(key=lambda x: x['Streak'], reverse=True)
    current_list.sort(key=lambda x: x['Length'], reverse=True)
    return {'win': win_list, 'unbeaten': unbeaten_list, 'loss': loss_list, 'current': current_list}


@st.cache_data
def get_biggest_match_wins(df):
    """Most lopsided match-play wins, parsed from 'Result' values like '8&7'."""
    work = df[(df['FBC'].notna()) & (df['W'] == 1)].copy()
    rows = []
    for _, r in work.iterrows():
        res = r.get('Result')
        if not isinstance(res, str):
            continue
        m = re.match(r'^\s*(\d+)\s*&\s*(\d+)\s*$', res)
        if not m:
            continue
        up, togo = int(m.group(1)), int(m.group(2))
        winners = f"{r['Player 1']}/{r['Player 2']}" if pd.notna(r.get('Player 2')) else str(r['Player 1'])
        if pd.notna(r.get('Opponent1')):
            losers = f"{r['Opponent1']}/{r['Opponent2']}" if pd.notna(r.get('Opponent2')) else str(r['Opponent1'])
        else:
            losers = str(r.get('Singles Opponent', ''))
        rows.append({'Margin': f"{up}&{togo}", 'Winner': winners, 'Loser': losers,
                     'FBC': int(r['FBC']), 'Format': r.get('Format', ''),
                     '_sort': (up, togo)})
    rows.sort(key=lambda x: x['_sort'], reverse=True)
    for r in rows:
        del r['_sort']
    return rows


@st.cache_data
def get_consecutive_cup_wins(cups_df):
    """Longest run of consecutive cups WON (among cups played; skipped events don't break a run)."""
    results = []
    fbc_cols = _fbc_columns(cups_df)
    for _, row in cups_df.iterrows():
        played = [(num, row[col]) for num, col in fbc_cols
                  if row.get(col) in (0, 1, '0', '1')]
        best, cur, best_span, cur_start = 0, 0, '', None
        for num, val in played:
            if val in (1, '1'):
                if cur == 0:
                    cur_start = num
                cur += 1
                if cur > best:
                    best = cur
                    best_span = f"FBC {cur_start}" if cur_start == num else f"FBC {cur_start}–{num}"
            else:
                cur = 0
        if best > 0:
            results.append({'Player': row['Player'], 'Consecutive Cups Won': best, 'Span': best_span})
    return sorted(results, key=lambda x: x['Consecutive Cups Won'], reverse=True)


@st.cache_data
def get_perfect_events(df, min_matches=5):
    """Players who got through an entire FBC event without losing a match."""
    perfect = [p for p in get_all_individual_fbc_performances(df)
               if p['Losses'] == 0 and p['Matches'] >= min_matches]
    return sorted(perfect, key=lambda x: (x['Wins'], x['Points']), reverse=True)


# Matches the date-outlier check should accept despite falling outside their
# event's normal window: long-delayed makeup matches that really were played
# years later. FBC8-S-Hilts-Mangold is the FBC 8 makeup finally played
# 2026-06-20 at Old Barnwell (Hilts d. Mangold 5&4).
DATE_CHECK_EXEMPT_MATCHES = {'FBC8-S-Hilts-Mangold'}


# Values the app recognises in the Singles/Doubles column. Matching is exact and
# case-sensitive everywhere (filters, team totals), so 'ftas' or 'singles' would be
# scored as an ordinary match.
VALID_MATCH_TYPES = {'Singles', 'Doubles', 'FTAS'}

# Archives columns every match row needs, with the reason shown in the Data Health
# message so the fix is obvious. All existing rows have all of these filled in.
REQUIRED_ROW_FIELDS = [
    ('UniqueMatchID', "pairs the two sides of a match and scores the FTAS once per team; "
                      "blank IDs silently drop the FTAS from team totals"),
    ('Team', "names the captain; team totals and the cup winner are computed from it"),
    ('Date', "must be a real Excel date, not text; used to order matches and check dates"),
    ('Geographic Location', "labels the event everywhere it appears"),
    ('Course', "drives the By Course stats and the Match Predictor"),
]


@contextlib.contextmanager
def _health_check(issues, name):
    """Run one Data Health check; if it raises, record that instead of aborting the rest."""
    try:
        yield
    except Exception as e:
        issues.append(f"Check '{name}' could not run: {e}")


@st.cache_data
def validate_data(df, cups_df=None):
    """Run structural integrity checks on the Archives data (and, when given, the Cups sheet).

    Catches the kinds of entry errors that have actually occurred or that a dry run of
    adding a new cup showed would go unnoticed: stray note rows, blank required columns
    (UniqueMatchID, Team, Date, Location, Course), mis-cased Singles/Doubles values,
    one-sided matches, bad W/L/T flags, extra teams in an event, opponent-name typos,
    inconsistent FTAS rows, Cups-sheet names that don't match the Archives spelling, and
    an Archives event with no matching Cups column.
    Each check is isolated, so one failing check reports itself rather than hiding the rest.
    Returns a list of issue strings; an empty list means all checks pass.
    """
    issues = []

    # 0. Rows with a player but no FBC number: usually a note typed into the Player 1
    # column. Every other check skips them, but they appear as a phantom player.
    with _health_check(issues, 'stray rows'):
        stray = df[df['FBC'].isna() & df['Player 1'].apply(lambda x: isinstance(x, str))]
        for _, r in stray.iterrows():
            issues.append(f"Archives: row with Player 1 = '{r['Player 1']}' has no FBC number — "
                          f"it shows up as a phantom player; delete the row or fill in the FBC")

    work = df[df['FBC'].notna()].copy()
    # Tolerate text in numeric columns (a typed '1' or a stray space): coerce, and let
    # the checks below flag whatever no longer adds up.
    for col in ['W', 'L', 'T', 'Points earned']:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')
    if 'Date' in work.columns:
        work['Date'] = pd.to_datetime(work['Date'], errors='coerce')

    # 1. Every row should have exactly one of W/L/T set
    with _health_check(issues, 'W/L/T flags'):
        wlt = work['W'].fillna(0) + work['L'].fillna(0) + work['T'].fillna(0)
        for idx in work[wlt != 1].index:
            r = work.loc[idx]
            issues.append(f"FBC {int(r['FBC'])}: row for {r['Player 1']} ({r.get('UniqueMatchID', '?')}) "
                          f"has W+L+T != 1")

    # 2. Unusual Points earned values
    with _health_check(issues, 'Points earned'):
        valid_pts = {0.0, 0.5, 1.0, 2.0}
        bad_pts = work[~work['Points earned'].fillna(-1).isin(valid_pts)]
        for _, r in bad_pts.iterrows():
            issues.append(f"FBC {int(r['FBC'])}: unusual Points earned ({r['Points earned']}) "
                          f"for {r['Player 1']} ({r.get('UniqueMatchID', '?')})")

    for fbc in sorted(work['FBC'].unique()):
        event = work[work['FBC'] == fbc]
        label = f"FBC {int(fbc)}"

        # 3. Required columns filled in on every row
        with _health_check(issues, f'{label} required columns'):
            for col, why in REQUIRED_ROW_FIELDS:
                if col not in event.columns:
                    continue
                n = int(event[col].isna().sum())
                if n:
                    issues.append(f"{label}: {n} row(s) with blank {col} — {why}")

        # 3b. Singles/Doubles must be one of the three exact values
        with _health_check(issues, f'{label} Singles/Doubles values'):
            bad = event.loc[~event['Singles/Doubles'].isin(VALID_MATCH_TYPES), 'Singles/Doubles']
            for val, n in bad.fillna('(blank)').value_counts().items():
                issues.append(f"{label}: Singles/Doubles value '{val}' on {n} row(s) — must be exactly "
                              f"Singles, Doubles or FTAS (case-sensitive); anything else is scored as a "
                              f"normal match, so a mis-cased FTAS counts once per player instead of once per team")

        # 4. Exactly two teams per event
        with _health_check(issues, f'{label} teams'):
            teams = event['Team'].dropna().unique().tolist()
            if len(teams) != 2:
                issues.append(f"{label}: expected 2 teams, found {len(teams)} ({', '.join(map(str, teams))})")

        # (Players CAN legitimately appear under both Team labels within an event —
        # FBC 5 and FBC 9 had mixed-pair sessions — so no cross-team check here.)
        roster = {}
        for _, r in event.iterrows():
            for p in [r['Player 1'], r.get('Player 2')]:
                if isinstance(p, str):
                    roster.setdefault(p, set()).add(r['Team'])

        # 5. Every non-FTAS match should have rows for both teams
        with _health_check(issues, f'{label} match pairing'):
            nonftas = event[event['Singles/Doubles'] != 'FTAS']
            for mid, grp in nonftas.groupby('UniqueMatchID'):
                if grp['Team'].nunique() < 2:
                    issues.append(f"{label}: match {mid} only has rows for one team "
                                  f"({grp['Team'].iloc[0]}) — missing the opposing row?")

        # 6. Date outliers — all rows in an event should fall within ~1 year of the
        # event's typical date (FBC 8's 13-month makeup window passes; a 2004-for-2014
        # year typo gets flagged)
        with _health_check(issues, f'{label} dates'):
            if event['Date'].notna().any():
                modal_year = int(event['Date'].dt.year.mode().iloc[0])
                stray = event[((event['Date'].dt.year - modal_year).abs() > 1) &
                              (~event['UniqueMatchID'].isin(DATE_CHECK_EXEMPT_MATCHES))]
                for _, r in stray.iterrows():
                    issues.append(f"{label}: suspicious date {r['Date'].date()} for {r['Player 1']} "
                                  f"({r.get('UniqueMatchID', '?')}) — event is mostly {modal_year}")

        # 7. Opponent names should match players who appear in this event (typo catch)
        with _health_check(issues, f'{label} opponent names'):
            participants = set(roster.keys())
            opp_names = set()
            for col in ['Opponent1', 'Opponent2', 'Singles Opponent']:
                if col in event.columns:
                    opp_names |= {v for v in event[col].dropna() if isinstance(v, str)}
            for name in sorted(opp_names - participants):
                issues.append(f"{label}: opponent name '{name}' never appears as a player this event "
                              f"— possible misspelling")

    # 8. FTAS rows: every player on a side should carry the same Points earned (0.5 for
    # the winning team, 0 for the losers). Team totals use the group mean, so a stray
    # 0 among 0.5s would silently shave the team score.
    with _health_check(issues, 'FTAS consistency'):
        ftas = work[work['Singles/Doubles'] == 'FTAS']
        for (fbc, mid, team), grp in ftas.groupby(['FBC', 'UniqueMatchID', 'Team']):
            if grp['Points earned'].nunique(dropna=False) > 1:
                vals = sorted(float(v) for v in grp['Points earned'].fillna(-1).unique())
                issues.append(f"FBC {int(fbc)}: FTAS rows for team {team} ({mid}) have mixed "
                              f"Points earned {vals} — every player on a side should show the same value")

    # 9. Cups sheet names must match the Archives spelling exactly, every Archives
    # player needs a Cups row, and every Archives event needs a Cups column — otherwise
    # cup stats and the Ask Claude context refer to one person by two names, leave
    # them out, or never count the newest cup at all.
    if cups_df is not None and 'Player' in cups_df.columns:
        with _health_check(issues, 'Cups sheet'):
            archive_players = {p for p in pd.concat([work['Player 1'], work['Player 2']]).dropna()
                               if isinstance(p, str)}
            cups_players = {p.strip() for p in cups_df['Player'].dropna() if isinstance(p, str)}
            unknown = sorted(cups_players - archive_players)
            if unknown:
                issues.append("Cups sheet: names not found in Archives: " + ", ".join(unknown) +
                              " — rename to the Archives spelling")
            missing = sorted(archive_players - cups_players)
            if missing:
                issues.append("Cups sheet: Archives players with no Cups row: " + ", ".join(missing))
            cup_events = {num for num, _ in _fbc_columns(cups_df)}
            for num in sorted({int(x) for x in work['FBC'].unique()} - cup_events):
                issues.append(f"Cups sheet: no 'FBC {num}' column — the header must read exactly "
                              f"'FBC {num}' (with the space); that cup is not being counted")

    return issues

def prepare_data_context(df, question, cups_df=None):
    """Prepare relevant FBC data context based on the question."""
    # Get all players and courses for reference
    all_players = sorted(set(df['Player 1'].dropna().unique()) |
                        set(df[df['Player 2'].notna()]['Player 2'].unique()))
    all_players = [p for p in all_players if isinstance(p, str)]
    all_courses = df['Course'].dropna().unique().tolist()
    all_events = sorted([int(e) for e in df['FBC'].dropna().unique() if pd.notna(e)])

    # Check if question is about cups/championships
    question_lower = question.lower()
    is_cups_question = any(word in question_lower for word in ['cup', 'cups', 'champion', 'championship', 'won', 'winning team', 'title'])

    # Check if question is specifically about singles or doubles
    is_singles_question = any(word in question_lower for word in ['singles', 'single', 'one on one', '1v1', 'individual'])
    is_doubles_question = any(word in question_lower for word in ['doubles', 'double', 'partner', 'partners', 'team', 'teams', 'pairing'])

    # Check if question is about partnerships/doubles teams
    is_partnership_question = any(phrase in question_lower for phrase in [
        'partnership', 'partnerships', 'doubles team', 'doubles teams', 'best team',
        'best partners', 'best pair', 'best pairing', 'as partners', 'together',
        'paired with', 'teamed with', 'partner with', 'duo', 'tandem'
    ]) or ('and' in question_lower and any(word in question_lower for word in ['record', 'wins', 'partner']))

    # Check if question is about aggregate doubles record for a group of players
    # e.g., "What is the aggregate doubles record when any two of Hilts, Lynch, Connolly are partners?"
    is_aggregate_group_question = any(phrase in question_lower for phrase in [
        'aggregate', 'combined record', 'any two of', 'any pair of', 'any pairing of',
        'group record', 'together as partners', 'any combination of', 'when any of',
        'among these players', 'between these players', 'within this group'
    ]) and is_doubles_question

    # Check if question is about team captains, team scores, or margins of victory
    is_team_margin_question = any(phrase in question_lower for phrase in [
        'captain', 'captains', 'captained', 'margin', 'margins', 'biggest win',
        'closest', 'blowout', 'team score', 'team scores', 'team total', 'team totals',
        'team points', 'final score', 'won by', 'beat by', 'lost by', 'point spread',
        'led their team', 'led his team', 'led the team', 'winning team', 'losing team',
        'team result', 'team results', 'team standings', 'how much did', 'dominant'
    ])

    # Check if question is about individual FBC performance (per-player, per-event)
    is_performance_question = any(phrase in question_lower for phrase in [
        'best fbc', 'worst fbc', 'best performance', 'worst performance', 'best cup',
        'worst cup', 'single fbc', 'single cup', 'single event', 'one fbc', 'one cup',
        'individual performance', 'cup performance', 'fbc performance', 'event performance',
        'top performance', 'top 10', 'top ten', 'best ever', 'worst ever', 'all time best',
        'all time worst', 'most points in a', 'most points at', 'highest points',
        'rank all', 'rank his', 'rank her', 'from best to worst', 'from worst to best',
        'performances from', 'best single', 'worst single'
    ])

    # Extract entities from the question
    fbc_num = extract_fbc_number(question)
    mentioned_players = extract_player_names(question, all_players)
    mentioned_courses = extract_course_names(question, all_courses)

    context_parts = []
    context_parts.append("FBC (Freddie B Cup) Golf Tournament Data\n" + "="*50)

    # If a specific FBC event is mentioned, provide complete data for that event
    if fbc_num is not None:
        event_df = df[df['FBC'] == fbc_num]
        if len(event_df) > 0:
            location = event_df['Geographic Location'].iloc[0] if pd.notna(event_df['Geographic Location'].iloc[0]) else "Unknown"
            courses = event_df['Course'].dropna().unique().tolist()

            context_parts.append(f"\n\nFBC {fbc_num} - {location}")
            context_parts.append(f"Courses: {', '.join(str(c) for c in courses)}")
            context_parts.append(f"Total matches: {len(event_df)}")

            # Match type breakdown
            match_types = event_df['Singles/Doubles'].value_counts().to_dict()
            context_parts.append(f"Match types: {', '.join(f'{k}: {v}' for k, v in match_types.items())}")

            # Calculate and show complete leaderboard for this event
            context_parts.append(f"\nFBC {fbc_num} COMPLETE LEADERBOARD:")
            event_stats = calculate_player_stats_for_subset(event_df)
            for i, stat in enumerate(event_stats, 1):
                context_parts.append(f"  {i}. {stat['Player']}: {stat['Points']:.1f} pts, Record: {stat['Record']}, Win%: {stat['Win%']:.1%}")

            # Show all matches with details
            context_parts.append(f"\nALL FBC {fbc_num} MATCHES ({len(event_df)} total):")
            for _, row in event_df.iterrows():
                p1 = row.get('Player 1', '')
                p2 = row.get('Player 2', '')
                opp1 = row.get('Opponent1', '')
                opp2 = row.get('Opponent2', '')
                course = row.get('Course', '')
                wlt = row.get('W/L/T', '')
                result = row.get('Result', '')
                match_type = row.get('Singles/Doubles', '')
                format_type = row.get('Format', '')
                pts = row.get('Points earned', 0)

                if pd.notna(p2) and p2:
                    players = f"{p1}/{p2}"
                else:
                    players = str(p1)

                if pd.notna(opp2) and opp2:
                    opponents = f"{opp1}/{opp2}"
                else:
                    opponents = str(opp1) if pd.notna(opp1) else ''

                context_parts.append(f"  {players} vs {opponents} | {match_type}/{format_type} | {wlt} {result} | {pts:.1f} pts | {course}")

    # If specific players are mentioned, provide their complete stats
    if mentioned_players:
        for player in mentioned_players:
            player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]
            if len(player_matches) == 0:
                continue

            context_parts.append(f"\n\n{player.upper()}'S COMPLETE STATS:")

            # Overall stats
            stats = calculate_player_stats_for_subset(player_matches, [player])[0]
            context_parts.append(f"  Overall: {stats['Points']:.1f} pts, Record: {stats['Record']}, Win%: {stats['Win%']:.1%}, {stats['Matches']} matches")

            # By match type
            context_parts.append(f"\n  By Match Type:")
            for match_type in ['Doubles', 'Singles', 'FTAS']:
                type_matches = player_matches[player_matches['Singles/Doubles'] == match_type]
                if len(type_matches) > 0:
                    type_stats = calculate_player_stats_for_subset(type_matches, [player])
                    if type_stats:
                        s = type_stats[0]
                        context_parts.append(f"    {match_type}: {s['Record']}, {s['Points']:.1f} pts, {s['Win%']:.1%}")

            # By FBC event
            context_parts.append(f"\n  By FBC Event:")
            for fbc in sorted(player_matches['FBC'].dropna().unique()):
                fbc_matches = player_matches[player_matches['FBC'] == fbc]
                fbc_stats = calculate_player_stats_for_subset(fbc_matches, [player])
                if fbc_stats:
                    s = fbc_stats[0]
                    context_parts.append(f"    FBC {int(fbc)}: {s['Record']}, {s['Points']:.1f} pts")

            # Head-to-head records
            context_parts.append(f"\n  Head-to-Head Records:")
            h2h = get_head_to_head(df, player)
            if not h2h.empty:
                for _, row in h2h.head(15).iterrows():
                    context_parts.append(f"    vs {row['Opponent']}: {row['Record']} ({row['Matches']} matches)")

    # If specific courses are mentioned, provide performance data
    if mentioned_courses:
        for course in mentioned_courses:
            course_matches = df[df['Course'] == course]
            if len(course_matches) == 0:
                continue

            context_parts.append(f"\n\nPERFORMANCE AT {course.upper()}:")
            context_parts.append(f"  Total matches played: {len(course_matches)}")

            # Stats by player at this course
            course_stats = calculate_player_stats_for_subset(course_matches)
            context_parts.append(f"\n  Player stats at this course:")
            for stat in course_stats[:15]:
                context_parts.append(f"    {stat['Player']}: {stat['Record']}, {stat['Win%']:.1%}")

    # Always include overall context
    context_parts.append(f"\n\nOVERALL FBC CONTEXT:")
    context_parts.append(f"  Total FBC events: {len(all_events)} ({min(all_events)}-{max(all_events)})")
    context_parts.append(f"  Total matches in database: {len(df)}")
    context_parts.append(f"  Total players: {len(all_players)}")
    context_parts.append(f"  Players: {', '.join(all_players)}")

    # Overall leaderboard
    context_parts.append(f"\n  LIFETIME LEADERBOARD - ALL MATCHES (Top 20):")
    overall_stats = calculate_player_stats_for_subset(df)
    for i, stat in enumerate(overall_stats[:20], 1):
        context_parts.append(f"    {i}. {stat['Player']}: {stat['Points']:.1f} pts, {stat['Record']}, {stat['Win%']:.1%}")

    # SINGLES LEADERBOARD - Filter by 'Singles/Doubles' column
    # This column contains 'Singles', 'Doubles', or 'FTAS'
    singles_df = df[df['Singles/Doubles'] == 'Singles']
    context_parts.append(f"\n  SINGLES ONLY LEADERBOARD (Top 20):")
    context_parts.append(f"  (These stats are ONLY from singles matches - {len(singles_df)} total singles matches)")
    singles_stats = calculate_player_stats_for_subset(singles_df)
    for i, stat in enumerate(singles_stats[:20], 1):
        context_parts.append(f"    {i}. {stat['Player']}: {stat['Wins']} wins, {stat['Record']}, {stat['Win%']:.1%}, {stat['Points']:.1f} pts")

    # DOUBLES LEADERBOARD
    doubles_df = df[df['Singles/Doubles'] == 'Doubles']
    context_parts.append(f"\n  DOUBLES ONLY LEADERBOARD (Top 20):")
    context_parts.append(f"  (These stats are ONLY from doubles matches - {len(doubles_df)} total doubles matches)")
    doubles_stats = calculate_player_stats_for_subset(doubles_df)
    for i, stat in enumerate(doubles_stats[:20], 1):
        context_parts.append(f"    {i}. {stat['Player']}: {stat['Wins']} wins, {stat['Record']}, {stat['Win%']:.1%}, {stat['Points']:.1f} pts")

    # DOUBLES PARTNERSHIP LEADERBOARD
    context_parts.append(f"\n  DOUBLES PARTNERSHIP LEADERBOARD (Top 25):")
    context_parts.append(f"  (Stats for each unique pair of players who played doubles together)")
    partnership_stats = get_all_partnership_stats(df)
    for i, pstat in enumerate(partnership_stats[:25], 1):
        context_parts.append(f"    {i}. {pstat['Partnership']}: {pstat['Wins']} wins, {pstat['Record']}, {pstat['Win%']:.1%}, {pstat['Matches']} matches, {pstat['Events']} events")

    # OPPONENT PAIRS NEVER PARTNERED
    never_partnered = get_opponent_pairs_never_partnered(df)
    context_parts.append(f"\n  OPPONENT PAIRS NEVER PARTNERED (Top 30):")
    context_parts.append(f"  (Pairs who have faced each other as opponents across singles/doubles but have NEVER been doubles partners)")
    context_parts.append(f"  (OpponentMatches = total matches where they were on opposite sides)")
    for i, npair in enumerate(never_partnered[:30], 1):
        context_parts.append(f"    {i}. {npair['Pair']}: {npair['OpponentMatches']} opponent matches, never partnered in doubles")

    # INDIVIDUAL FBC PERFORMANCE LEADERBOARD (per-player, per-event)
    all_performances = get_all_individual_fbc_performances(df)

    # Sort by points for best single FBC performances
    best_by_points = sorted(all_performances, key=lambda x: (-x['Points'], -x['Win%']))
    context_parts.append(f"\n  BEST INDIVIDUAL FBC PERFORMANCES - BY POINTS (Top 30):")
    context_parts.append(f"  (Each player's performance at each FBC event they attended)")
    for i, perf in enumerate(best_by_points[:30], 1):
        context_parts.append(f"    {i}. {perf['Player']} at FBC {perf['FBC']} ({perf['Location']}): {perf['Points']:.1f} pts, {perf['Record']}, {perf['Win%']:.1%}, {perf['Matches']} matches")

    # Sort by win percentage (min 3 matches) for best win rate performances
    qualified_perfs = [p for p in all_performances if p['Matches'] >= 3]
    best_by_winpct = sorted(qualified_perfs, key=lambda x: (-x['Win%'], -x['Points']))
    context_parts.append(f"\n  BEST INDIVIDUAL FBC PERFORMANCES - BY WIN% (min 3 matches, Top 25):")
    for i, perf in enumerate(best_by_winpct[:25], 1):
        context_parts.append(f"    {i}. {perf['Player']} at FBC {perf['FBC']} ({perf['Location']}): {perf['Win%']:.1%}, {perf['Record']}, {perf['Points']:.1f} pts, {perf['Matches']} matches")

    # Worst performances (by points, then win%)
    worst_by_points = sorted([p for p in all_performances if p['Matches'] >= 3], key=lambda x: (x['Points'], x['Win%']))
    context_parts.append(f"\n  WORST INDIVIDUAL FBC PERFORMANCES (min 3 matches, Bottom 20):")
    for i, perf in enumerate(worst_by_points[:20], 1):
        context_parts.append(f"    {i}. {perf['Player']} at FBC {perf['FBC']} ({perf['Location']}): {perf['Points']:.1f} pts, {perf['Record']}, {perf['Win%']:.1%}, {perf['Matches']} matches")

    # If specific players mentioned, show ALL their FBC performances
    if mentioned_players:
        for player in mentioned_players:
            player_perfs = [p for p in all_performances if p['Player'] == player]
            if player_perfs:
                # Sort by points (best first)
                player_perfs_by_pts = sorted(player_perfs, key=lambda x: (-x['Points'], -x['Win%']))
                context_parts.append(f"\n  {player.upper()}'S FBC PERFORMANCES RANKED (Best to Worst by Points):")
                for i, perf in enumerate(player_perfs_by_pts, 1):
                    context_parts.append(f"    {i}. FBC {perf['FBC']} ({perf['Location']}): {perf['Points']:.1f} pts, {perf['Record']}, {perf['Win%']:.1%}, {perf['Matches']} matches")

    # If asking about individual FBC performances, add emphasis
    if is_performance_question:
        context_parts.append(f"\n  *** IMPORTANT: The question asks about INDIVIDUAL FBC PERFORMANCES. ***")
        context_parts.append(f"  *** Use the BEST/WORST INDIVIDUAL FBC PERFORMANCES leaderboards above. ***")
        context_parts.append(f"  *** Each entry represents one player's stats at one specific FBC event. ***")

    # If specific players are mentioned, check if they've been partners
    if len(mentioned_players) >= 2:
        context_parts.append(f"\n  SPECIFIC PARTNERSHIP RECORDS FOR MENTIONED PLAYERS:")
        # Check all pairs of mentioned players
        for i in range(len(mentioned_players)):
            for j in range(i + 1, len(mentioned_players)):
                p1, p2 = mentioned_players[i], mentioned_players[j]
                pstats = get_specific_partnership_stats(df, p1, p2)
                if pstats:
                    context_parts.append(f"    {pstats['Partnership']}: {pstats['Wins']} wins, {pstats['Record']}, {pstats['Win%']:.1%}, {pstats['Matches']} matches together")
                else:
                    context_parts.append(f"    {p1} & {p2}: Never partnered in doubles")

        # Calculate aggregate doubles record for the group (only matches where BOTH partners are in the group)
        aggregate_stats = get_aggregate_group_doubles_stats(df, mentioned_players)
        if aggregate_stats and aggregate_stats['total_matches'] > 0:
            context_parts.append(f"\n  AGGREGATE DOUBLES RECORD FOR THIS GROUP:")
            context_parts.append(f"  (Only includes matches where BOTH partners are in the group: {', '.join(mentioned_players)})")
            context_parts.append(f"    Combined Record: {aggregate_stats['record']}")
            context_parts.append(f"    Total Wins: {aggregate_stats['total_wins']}")
            context_parts.append(f"    Total Losses: {aggregate_stats['total_losses']}")
            context_parts.append(f"    Total Ties: {aggregate_stats['total_ties']}")
            context_parts.append(f"    Total Matches: {aggregate_stats['total_matches']}")
            context_parts.append(f"    Win Percentage: {aggregate_stats['win_pct']:.1%}")
            context_parts.append(f"    Total Points: {aggregate_stats['total_points']:.1f}")
            if aggregate_stats['excluded_matches'] > 0:
                context_parts.append(f"    Note: {aggregate_stats['excluded_note']}")

            # Show breakdown by partnership
            if aggregate_stats['partnerships']:
                context_parts.append(f"\n  Breakdown by Partnership (within group):")
                for pstat in aggregate_stats['partnerships']:
                    context_parts.append(f"    {pstat['Partnership']}: {pstat['Record']}, {pstat['Win%']:.1%}, {pstat['Matches']} matches")
        elif aggregate_stats:
            context_parts.append(f"\n  AGGREGATE DOUBLES RECORD FOR THIS GROUP:")
            context_parts.append(f"  No doubles matches found where both partners are in the group: {', '.join(mentioned_players)}")

    # If asking about aggregate group doubles, add emphasis
    if is_aggregate_group_question and len(mentioned_players) >= 2:
        context_parts.append(f"\n  *** IMPORTANT: The question asks about AGGREGATE DOUBLES RECORD for a group. ***")
        context_parts.append(f"  *** Use the AGGREGATE DOUBLES RECORD section above. ***")
        context_parts.append(f"  *** This ONLY includes matches where BOTH Player 1 AND Player 2 are in the specified group. ***")
        context_parts.append(f"  *** Matches where only one partner is in the group are EXCLUDED. ***")

    # If asking about partnerships, add emphasis
    if is_partnership_question:
        context_parts.append(f"\n  *** IMPORTANT: The question asks about DOUBLES PARTNERSHIPS. Use the DOUBLES PARTNERSHIP LEADERBOARD above. ***")
        context_parts.append(f"  *** A partnership is two players who played together as a team in doubles matches. ***")

    # If specifically asking about singles or doubles, add extra emphasis
    if is_singles_question:
        context_parts.append(f"\n  *** IMPORTANT: The question asks about SINGLES matches. Use the SINGLES ONLY LEADERBOARD above. ***")
        context_parts.append(f"  *** Singles matches are where 'Singles/Doubles' column equals 'Singles' ***")
    if is_doubles_question:
        context_parts.append(f"\n  *** IMPORTANT: The question asks about DOUBLES matches. Use the DOUBLES ONLY LEADERBOARD above. ***")
        context_parts.append(f"  *** Doubles matches are where 'Singles/Doubles' column equals 'Doubles' ***")

    # Team results, captains, and margins of victory (derived from the 'Team' column,
    # which names each team after its captain). Always include the compact summary so
    # Claude can answer captain / margin / team-score questions for any event.
    team_results = get_fbc_team_results(df)
    if team_results:
        context_parts.append(f"\n\nTEAM RESULTS, CAPTAINS & MARGINS OF VICTORY:")
        context_parts.append("  (Each FBC is a team event. The 'Team' is named after its CAPTAIN.")
        context_parts.append("   A team's score = total Points earned by its players. The winning captain's")
        context_parts.append("   team has the most points; margin of victory = winner's points minus runner-up's.)")
        cup_info = load_cup_info()
        # Sort the summary by margin so 'biggest/closest margin' questions are easy to read
        for r in sorted(team_results, key=lambda x: x['margin'], reverse=True):
            info = cup_info.get(r['fbc'], {})
            weather = {True: " | rain: yes", False: " | rain: no"}.get(info.get('rain'), "")
            note = f" | note: {info['notes']}" if info.get('notes') else ""
            scores = ', '.join(f"{cap} {pts:.1f}" for cap, pts in r['teams'])
            if r['tie']:
                outcome = f"TIED at {r['teams'][0][1]:.1f} (no margin)"
            else:
                outcome = f"won by {r['winner']} over {r['loser']} by {r['margin']:.1f} pts"
            multi = f" [{r['num_teams']} teams]" if r['num_teams'] > 2 else ""
            context_parts.append(f"    FBC {r['fbc']} ({r['location']}): {outcome}{multi} | team scores: {scores}{weather}{note}")
        rained = sorted(f for f, v in cup_info.items() if v.get('rain') is True)
        dry = sorted(f for f, v in cup_info.items() if v.get('rain') is False)
        if rained or dry:
            context_parts.append(
                "  WEATHER (per cup, from the Cup Info tab; not recorded per round): "
                f"rained at FBC {', '.join(map(str, rained)) or 'none'}; "
                f"no rain at FBC {', '.join(map(str, dry)) or 'none'}.")

        # Detailed roster for a specifically mentioned event or team/margin questions
        detail_fbc = fbc_num if fbc_num is not None else None
        if detail_fbc is not None:
            detail = [r for r in team_results if r['fbc'] == detail_fbc]
            for r in detail:
                context_parts.append(f"\n  FBC {r['fbc']} TEAM BREAKDOWN ({r['location']}):")
                for cap, pts in r['teams']:
                    size = f" — draft theme: {r['sizes'][cap]}" if cap in r['sizes'] else ""
                    tag = " (WINNING CAPTAIN)" if cap == r['winner'] and not r['tie'] else ""
                    context_parts.append(f"    Captain {cap}: {pts:.1f} pts{tag}{size}")
                    roster = df[(df['FBC'] == r['fbc']) & (df['Team'] == cap)]
                    members = sorted(set(roster['Player 1'].dropna()) |
                                     set(roster[roster['Player 2'].notna()]['Player 2'].dropna()))
                    members = [m for m in members if isinstance(m, str)]
                    if members:
                        context_parts.append(f"      Roster: {', '.join(members)}")

        if is_team_margin_question:
            max_margin = max(r['margin'] for r in team_results)
            biggest = [r for r in team_results if abs(r['margin'] - max_margin) < 1e-9]
            decided = [r for r in team_results if not r['tie']]
            closest = min(decided, key=lambda x: x['margin']) if decided else None
            context_parts.append(f"\n  *** IMPORTANT: The question is about CAPTAINS / TEAM SCORES / MARGIN OF VICTORY. ***")
            context_parts.append(f"  *** Use the TEAM RESULTS section above. The 'Team' value is the CAPTAIN's name. ***")
            if len(biggest) == 1:
                b = biggest[0]
                context_parts.append(f"  *** Biggest margin of victory: FBC {b['fbc']} — captain {b['winner']} "
                                     f"won by {b['margin']:.1f} pts ({b['teams'][0][1]:.1f} to {b['teams'][1][1]:.1f}). ***")
            else:
                tied = '; '.join(f"FBC {b['fbc']} (captain {b['winner']}, {b['teams'][0][1]:.1f}-{b['teams'][1][1]:.1f})" for b in biggest)
                context_parts.append(f"  *** Biggest margin of victory: a TIE at {max_margin:.1f} pts between {tied}. ***")
            if closest is not None:
                context_parts.append(f"  *** Closest finish: FBC {closest['fbc']} — captain {closest['winner']} "
                                     f"won by just {closest['margin']:.1f} pts. ***")

    # Add Cups data if available and relevant
    if cups_df is not None and (is_cups_question or mentioned_players):
        context_parts.append(f"\n\nCUP CHAMPIONSHIPS DATA:")
        context_parts.append("(1 = won the cup, 0 = lost the cup, X = did not participate)")

        # Full cups leaderboard
        context_parts.append(f"\n  CUP WINS LEADERBOARD:")
        cups_summary = get_cups_summary(cups_df)
        for i, player_cups in enumerate(cups_summary, 1):
            if player_cups['Cups Played'] > 0:
                context_parts.append(f"    {i}. {player_cups['Player']}: {player_cups['Cups Won']} cups won out of {player_cups['Cups Played']} played ({player_cups['Cup Win%']:.1%})")

        # Detailed cup results for mentioned players
        if mentioned_players:
            for player in mentioned_players:
                names_lower = cups_df['Player'].str.lower()
                player_cup_data = cups_df[names_lower == player.lower()]
                if len(player_cup_data) == 0:
                    # Fall back to a prefix match ("Connolly" -> "Connolly, B") before a
                    # substring match, so "Connolly" does not pick up "R. Connolly".
                    player_cup_data = cups_df[names_lower.str.startswith(player.lower(), na=False)]
                if len(player_cup_data) == 0:
                    player_cup_data = cups_df[names_lower.str.contains(player.lower(), na=False, regex=False)]

                if len(player_cup_data) > 0:
                    row = player_cup_data.iloc[0]
                    context_parts.append(f"\n  {player.upper()}'S CUP HISTORY:")
                    for fbc_num, col in _fbc_columns(cups_df):
                        result = row.get(col, 'X')
                        if result == 1 or result == '1':
                            context_parts.append(f"    FBC {fbc_num}: WON (on winning team)")
                        elif result == 0 or result == '0':
                            context_parts.append(f"    FBC {fbc_num}: LOST (on losing team)")
                        else:
                            context_parts.append(f"    FBC {fbc_num}: Did not participate")
                    total = row.get('Total', 0)
                    played = row.get('Played', 0)
                    total = 0 if pd.isna(total) else total
                    played = 0 if pd.isna(played) else played
                    context_parts.append(f"    TOTAL: {int(total)} cups won out of {int(played)} played")

    return '\n'.join(context_parts)

def ask_claude(question, df, cups_df=None, history=None):
    """Send a question to Claude with relevant FBC data context.

    history is a list of (question, answer) tuples from earlier in the conversation,
    so follow-ups like "what about in singles?" resolve against prior turns.
    """
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    history = history or []

    # Load cups data if not provided
    if cups_df is None:
        try:
            cups_df = load_cups_data()
        except Exception:
            cups_df = None

    # Prepare context based on the question. For follow-ups, include the last couple
    # of user questions so entity extraction (players, events, courses) still finds
    # names that were only mentioned earlier in the conversation.
    extraction_text = ' '.join(q for q, _ in history[-2:]) + ' ' + question if history else question
    data_context = prepare_data_context(df, extraction_text, cups_df)

    system_prompt = """You are an expert analyst for the FBC (Freddie B Cup), a golf match play tournament between friends.
You have access to historical match data and should answer questions about player statistics, head-to-head records,
course performance, tournament history, and CUP CHAMPIONSHIPS (which team won each FBC event).

IMPORTANT: The data provided includes pre-calculated statistics and complete match records. Use these directly -
do not try to recalculate from raw data. When asked about points, wins, or records, cite the exact numbers from
the leaderboard or stats provided.

CRITICAL - SINGLES vs DOUBLES DISTINCTION:
- The data includes THREE separate leaderboards: ALL MATCHES, SINGLES ONLY, and DOUBLES ONLY
- When asked about "singles wins", "singles record", or "singles performance", ONLY use the SINGLES ONLY LEADERBOARD
- When asked about "doubles wins", "doubles record", or "doubles performance", ONLY use the DOUBLES ONLY LEADERBOARD
- When asked about overall/total stats without specifying match type, use the ALL MATCHES leaderboard
- FTAS (Full Team Alternate Shot) is a separate format - not singles or doubles. It is a single
  sudden-death tiebreaker hole worth 0.5 points to the winning TEAM (once). In INDIVIDUAL stats,
  every player on the FTAS-winning team is credited 0.5 in their personal point total - that is
  the official convention, so individual totals legitimately include that share.

DOUBLES PARTNERSHIPS:
- The DOUBLES PARTNERSHIP LEADERBOARD shows statistics for each unique pair of players who have played doubles together
- Use this leaderboard when asked about "best doubles team", "partnership records", "who has won the most as partners", etc.
- A partnership is identified by two player names (e.g., "Hilts & Lynch") and shows their combined record when playing as teammates
- Note: Individual doubles stats (DOUBLES ONLY LEADERBOARD) are DIFFERENT from partnership stats - individual stats count each player separately, while partnership stats count the team's record together

OPPONENT PAIRS NEVER PARTNERED:
- The OPPONENT PAIRS NEVER PARTNERED leaderboard shows pairs of players who have faced each other as opponents but have NEVER been doubles partners
- "OpponentMatches" counts every match where the two players were on opposite sides (singles: direct opponents; doubles: on opposing teams; the FTAS tiebreaker is not counted)
- These pairs have zero doubles matches together as partners
- Use this when asked about "who has played the most against each other without being partners", "rivals who never teamed up", etc.

INDIVIDUAL FBC PERFORMANCES (Per-Player, Per-Event):
- The BEST/WORST INDIVIDUAL FBC PERFORMANCES leaderboards show each player's performance at each specific FBC event
- Each entry represents ONE player at ONE FBC tournament (e.g., "Grise at FBC 7")
- Use these when asked about "best single FBC performance", "top 10 individual Cup performances", "who had the best FBC ever"
- Also use when asked about a specific player's best/worst FBC or to rank all their FBC performances
- Data includes: points earned, win-loss-tie record, win percentage, and number of matches at that event
- When a player is mentioned, their individual FBC performances are listed from best to worst

PLAYER NAME DISAMBIGUATION - TWO CONNOLLYS:
- "Connolly" in the data refers to BRETT Connolly
- "R. Connolly" in the data refers to RICK Connolly
- If the user asks about "Brett Connolly", "Brett", or just "Connolly" without further context, use the "Connolly" data (Brett)
- If the user asks about "Rick Connolly", "Rick", "Connolly R", or "R. Connolly", use the "R. Connolly" data (Rick)
- When presenting stats for either Connolly, always clarify which one you mean (e.g., "Connolly (Brett)" or "R. Connolly (Rick)")
- If both Connollys appear in a leaderboard or result set, label them clearly so the user can tell them apart

TEAM CAPTAINS, TEAM SCORES & MARGINS OF VICTORY:
- Each FBC is a team competition. Every team is CAPTAINED by a player and is NAMED AFTER that captain
  (the "Team" value in the data IS the captain's name, e.g. team "Delneky" was captained by Delneky)
- The TEAM RESULTS, CAPTAINS & MARGINS OF VICTORY section gives, for every FBC event: each team's
  total points, which captain won, and the margin of victory (winner's points minus runner-up's points)
- Use this section for questions about captains, "which captain led their team to victory", team scores,
  team point totals, biggest/smallest margin of victory, blowouts, closest finishes, and who beat whom by how much
- A team's score = the sum of Points earned by all players on that team that event
- When a specific FBC is mentioned, a TEAM BREAKDOWN with each captain's roster is also provided
- Most FBCs have exactly 2 teams; a few have 3 (noted with "[N teams]") — for those, the margin shown is winner minus runner-up
- Do NOT say captain or margin data is unavailable — it is in the TEAM RESULTS section

For CUP questions: A "cup win" means the player was on the winning TEAM for that FBC event. This is different
from individual match wins. The Cups data shows team championship results.

This may be a multi-turn conversation. When the user asks a follow-up (e.g. "what about in singles?"
or "and at FBC 10?"), resolve it against the earlier turns — the players, events, or topics they
were just discussing.

Be concise but thorough. Always cite the specific data that supports your answer."""

    # Rebuild the conversation: prior turns as plain Q/A (their data contexts are not
    # resent), then the current question with fresh data context attached.
    messages = []
    for q, a in history[-6:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({
        "role": "user",
        "content": f"""Here is the FBC tournament data relevant to your question:

{data_context}

Question: {question}

Please answer based on the data provided above. Cite specific statistics."""
    })

    message = client.beta.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=8000,
        # Cache the static system prompt so repeat questions in a session are cheaper/faster
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        # If a safety classifier declines the question, the API retries it on a
        # fallback model within the same call instead of returning nothing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if message.stop_reason == "refusal":
        return "Claude declined to answer that question. Please try rephrasing it."

    # The response is a list of content blocks; current models can return a
    # thinking block ahead of the text, so pick the text block explicitly
    # rather than assuming it is first.
    text_blocks = [b.text for b in message.content if b.type == "text"]
    if not text_blocks:
        return "Claude returned no answer for that question. Please try rephrasing it."
    return "\n\n".join(text_blocks)

def get_direct_h2h(df, player1, player2):
    """Get direct head-to-head record between two specific players."""
    # Find matches where player1 was on one side and player2 on the opposing side
    p1_matches = df[(df['Player 1'] == player1) | (df['Player 2'] == player1)]

    h2h_matches = p1_matches[
        (p1_matches['Opponent1'] == player2) |
        (p1_matches['Opponent2'] == player2) |
        (p1_matches['Singles Opponent'] == player2)
    ]

    if len(h2h_matches) == 0:
        return {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0}

    wins = h2h_matches['W'].sum()
    losses = h2h_matches['L'].sum()
    ties = h2h_matches['T'].sum()

    return {
        'wins': int(wins),
        'losses': int(losses),
        'ties': int(ties),
        'matches': len(h2h_matches)
    }

def get_stats_by_format(df, player):
    """Get player's record broken down by format (Singles, Doubles, FTAS)."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]

    format_stats = {}
    for fmt in ['Singles', 'Doubles', 'FTAS']:
        fmt_matches = player_matches[player_matches['Singles/Doubles'] == fmt]
        if len(fmt_matches) > 0:
            wins = fmt_matches['W'].sum()
            losses = fmt_matches['L'].sum()
            ties = fmt_matches['T'].sum()
            total = len(fmt_matches)
            format_stats[fmt] = {
                'wins': int(wins),
                'losses': int(losses),
                'ties': int(ties),
                'matches': total,
                'win_pct': (wins + 0.5 * ties) / total if total > 0 else 0
            }
        else:
            format_stats[fmt] = {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'win_pct': 0}

    return format_stats

def get_best_worst_courses(df, player, top_n=3):
    """Get player's best and worst courses by win percentage."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]

    course_stats = []
    for course in player_matches['Course'].dropna().unique():
        course_matches = player_matches[player_matches['Course'] == course]
        if len(course_matches) >= 2:  # At least 2 matches for meaningful stats
            wins = course_matches['W'].sum()
            losses = course_matches['L'].sum()
            ties = course_matches['T'].sum()
            total = len(course_matches)
            win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
            course_stats.append({
                'course': course,
                'wins': int(wins),
                'losses': int(losses),
                'ties': int(ties),
                'matches': total,
                'win_pct': win_pct
            })

    if not course_stats:
        return [], []

    sorted_courses = sorted(course_stats, key=lambda x: x['win_pct'], reverse=True)
    best = sorted_courses[:top_n]
    worst = sorted_courses[-top_n:][::-1] if len(sorted_courses) >= top_n else sorted_courses[::-1]

    return best, worst

def get_best_partners(df, player, top_n=3):
    """Get player's best doubles partners by win percentage."""
    doubles = df[df['Singles/Doubles'] == 'Doubles']

    # Find all partners
    as_p1 = doubles[doubles['Player 1'] == player].copy()
    as_p1['Partner'] = as_p1['Player 2']

    as_p2 = doubles[doubles['Player 2'] == player].copy()
    as_p2['Partner'] = as_p2['Player 1']

    all_partner_matches = pd.concat([as_p1, as_p2])

    if len(all_partner_matches) == 0:
        return []

    partner_stats = []
    for partner in all_partner_matches['Partner'].dropna().unique():
        partner_matches = all_partner_matches[all_partner_matches['Partner'] == partner]
        if len(partner_matches) >= 2:  # At least 2 matches
            wins = partner_matches['W'].sum()
            losses = partner_matches['L'].sum()
            ties = partner_matches['T'].sum()
            total = len(partner_matches)
            win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
            partner_stats.append({
                'partner': partner,
                'wins': int(wins),
                'losses': int(losses),
                'ties': int(ties),
                'matches': total,
                'win_pct': win_pct
            })

    sorted_partners = sorted(partner_stats, key=lambda x: (x['win_pct'], x['matches']), reverse=True)
    return sorted_partners[:top_n]

def get_recent_form(df, player, n_matches=10):
    """Get player's recent form (last n matches)."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]
    if 'Date' in df.columns:
        player_matches = player_matches.sort_values('Date')
    recent = player_matches.tail(n_matches)

    if len(recent) == 0:
        return {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'win_pct': 0}

    wins = recent['W'].sum()
    losses = recent['L'].sum()
    ties = recent['T'].sum()
    total = len(recent)

    return {
        'wins': int(wins),
        'losses': int(losses),
        'ties': int(ties),
        'matches': total,
        'win_pct': (wins + 0.5 * ties) / total if total > 0 else 0
    }

def get_player_course_stats(df, player, course):
    """Get player's performance at a specific course."""
    player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]
    course_matches = player_matches[player_matches['Course'] == course]

    if len(course_matches) == 0:
        return None

    wins = course_matches['W'].sum()
    losses = course_matches['L'].sum()
    ties = course_matches['T'].sum()
    total = len(course_matches)

    return {
        'wins': int(wins),
        'losses': int(losses),
        'ties': int(ties),
        'matches': total,
        'win_pct': (wins + 0.5 * ties) / total if total > 0 else 0
    }

def get_partner_chemistry(df, player1, player2):
    """Get the record when two players are partners in doubles."""
    doubles = df[df['Singles/Doubles'] == 'Doubles']

    # Find matches where both players were on the same team
    team_matches = doubles[
        ((doubles['Player 1'] == player1) & (doubles['Player 2'] == player2)) |
        ((doubles['Player 1'] == player2) & (doubles['Player 2'] == player1))
    ]

    if len(team_matches) == 0:
        return None

    wins = team_matches['W'].sum()
    losses = team_matches['L'].sum()
    ties = team_matches['T'].sum()
    total = len(team_matches)

    return {
        'wins': int(wins),
        'losses': int(losses),
        'ties': int(ties),
        'matches': total,
        'win_pct': (wins + 0.5 * ties) / total if total > 0 else 0
    }

def predict_match(df, player1, player2, course=None, is_doubles=False, partner1=None, partner2=None):
    """Predict match outcome based on historical data."""
    factors = []
    p1_score = 50.0  # Start at 50-50

    # Factor 1: Overall win percentage
    p1_stats = get_player_stats(df, player1)
    p2_stats = get_player_stats(df, player2)

    if p1_stats and p2_stats:
        p1_overall = p1_stats['win_pct']
        p2_overall = p2_stats['win_pct']
        overall_diff = (p1_overall - p2_overall) * 30  # Weight: up to +/- 15%
        p1_score += overall_diff
        factors.append({
            'factor': 'Overall Win %',
            'p1_value': f"{p1_overall:.1%}",
            'p2_value': f"{p2_overall:.1%}",
            'edge': player1 if p1_overall > p2_overall else (player2 if p2_overall > p1_overall else 'Even'),
            'impact': abs(overall_diff)
        })

    # Factor 2: Head-to-head record
    h2h = get_direct_h2h(df, player1, player2)
    if h2h['matches'] > 0:
        h2h_pct = (h2h['wins'] + 0.5 * h2h['ties']) / h2h['matches']
        h2h_diff = (h2h_pct - 0.5) * 40  # Weight: up to +/- 20%
        p1_score += h2h_diff
        factors.append({
            'factor': 'Head-to-Head',
            'p1_value': f"{h2h['wins']}-{h2h['losses']}-{h2h['ties']}",
            'p2_value': f"{h2h['losses']}-{h2h['wins']}-{h2h['ties']}",
            'edge': player1 if h2h['wins'] > h2h['losses'] else (player2 if h2h['losses'] > h2h['wins'] else 'Even'),
            'impact': abs(h2h_diff)
        })

    # Factor 3: Recent form
    p1_recent = get_recent_form(df, player1, 10)
    p2_recent = get_recent_form(df, player2, 10)

    if p1_recent['matches'] > 0 and p2_recent['matches'] > 0:
        recent_diff = (p1_recent['win_pct'] - p2_recent['win_pct']) * 20  # Weight: up to +/- 10%
        p1_score += recent_diff
        factors.append({
            'factor': 'Recent Form (Last 10)',
            'p1_value': f"{p1_recent['wins']}-{p1_recent['losses']}-{p1_recent['ties']} ({p1_recent['win_pct']:.1%})",
            'p2_value': f"{p2_recent['wins']}-{p2_recent['losses']}-{p2_recent['ties']} ({p2_recent['win_pct']:.1%})",
            'edge': player1 if p1_recent['win_pct'] > p2_recent['win_pct'] else (player2 if p2_recent['win_pct'] > p1_recent['win_pct'] else 'Even'),
            'impact': abs(recent_diff)
        })

    # Factor 4: Course performance (if course specified)
    if course:
        p1_course = get_player_course_stats(df, player1, course)
        p2_course = get_player_course_stats(df, player2, course)

        if p1_course and p2_course:
            course_diff = (p1_course['win_pct'] - p2_course['win_pct']) * 20  # Weight: up to +/- 10%
            p1_score += course_diff
            factors.append({
                'factor': f"Course ({course if len(course) <= 20 else course[:20] + '…'})",
                'p1_value': f"{p1_course['wins']}-{p1_course['losses']}-{p1_course['ties']} ({p1_course['win_pct']:.1%})",
                'p2_value': f"{p2_course['wins']}-{p2_course['losses']}-{p2_course['ties']} ({p2_course['win_pct']:.1%})",
                'edge': player1 if p1_course['win_pct'] > p2_course['win_pct'] else (player2 if p2_course['win_pct'] > p1_course['win_pct'] else 'Even'),
                'impact': abs(course_diff)
            })

    # Factor 5: Partner chemistry (for doubles)
    if is_doubles and partner1 and partner2:
        team1_chem = get_partner_chemistry(df, player1, partner1)
        team2_chem = get_partner_chemistry(df, player2, partner2)

        if team1_chem and team2_chem:
            chem_diff = (team1_chem['win_pct'] - team2_chem['win_pct']) * 20  # Weight: up to +/- 10%
            p1_score += chem_diff
            factors.append({
                'factor': 'Partner Chemistry',
                'p1_value': f"{team1_chem['wins']}-{team1_chem['losses']}-{team1_chem['ties']} ({team1_chem['win_pct']:.1%})",
                'p2_value': f"{team2_chem['wins']}-{team2_chem['losses']}-{team2_chem['ties']} ({team2_chem['win_pct']:.1%})",
                'edge': f"{player1}/{partner1}" if team1_chem['win_pct'] > team2_chem['win_pct'] else (f"{player2}/{partner2}" if team2_chem['win_pct'] > team1_chem['win_pct'] else 'Even'),
                'impact': abs(chem_diff)
            })

    # Clamp probability between 15% and 85%
    p1_score = max(15, min(85, p1_score))
    p2_score = 100 - p1_score

    return {
        'p1_prob': p1_score,
        'p2_prob': p2_score,
        'factors': factors,
        'favorite': player1 if p1_score > 50 else (player2 if p2_score > 50 else 'Toss-up')
    }

def format_pct(val):
    """Format percentage for display."""
    return f"{val:.1%}"

def _record_rows(items, name_key, name_label):
    """Turn best/worst course or partner dicts into a small display table."""
    return pd.DataFrame([{
        name_label: it[name_key],
        'Record': f"{it['wins']}-{it['losses']}-{it['ties']}",
        'Win%': round(it['win_pct'] * 100, 1),
    } for it in items])


CUP_CELL = {'1': 'W', '0': 'L', 'X': '·'}


def _cup_cell_style(v):
    if v == 'W':
        return f"background-color: {COLORS['primary_soft']}; color: {COLORS['win']}; font-weight: 600; text-align: center;"
    if v == 'L':
        return f"color: {COLORS['loss']}; text-align: center;"
    return "color: #B7BDB6; text-align: center;"


def render_masthead(df):
    results = get_fbc_team_results(df)
    n_players = len(set(df['Player 1'].dropna()) | set(df['Player 2'].dropna()))
    sub = f"{len(results)} cups · {df['UniqueMatchID'].nunique():,} matches · {n_players} players"
    latest = ''
    if results:
        r = results[-1]
        w, l = r['teams'][0][1], (r['teams'][1][1] if len(r['teams']) > 1 else None)
        verb = 'halved with' if r['tie'] else 'def.'
        latest = (f"<div class='fbc-latest'><span class='dot'></span>Latest · <b>FBC {r['fbc']}</b>, "
                  f"{html.escape(str(r['location']))} — Team {html.escape(team_label(r['winner']))} {verb} "
                  f"Team {html.escape(team_label(r['loser'] or ''))}, <b>{fmt_pts(w)}–{fmt_pts(l)}</b></div>")
    st.markdown(f"""
    <div class="fbc-masthead">
        <div class="fbc-brand">
            <div class="fbc-crest">FBC</div>
            <div><div class="fbc-title">The Freddie B Cup</div><div class="fbc-sub">{sub}</div></div>
        </div>
        {latest}
    </div>
    """, unsafe_allow_html=True)


def main():
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    render_masthead(df)

    # Get list of all players
    all_players = sorted(set(df['Player 1'].dropna().unique()) |
                        set(df[df['Player 2'].notna()]['Player 2'].unique()))
    all_players = [p for p in all_players if isinstance(p, str)]

    # Default comparison picks: the career points leaders (instead of alphabetical,
    # which opened every comparison on long-retired players)
    leaderboard_base = get_leaderboard(df)
    leaders = [p for p in leaderboard_base['Player'] if p in all_players]
    def _default_idx(rank):
        return all_players.index(leaders[rank]) if len(leaders) > rank else 0

    # Load cups data
    try:
        cups_df = load_cups_data()
    except Exception as e:
        cups_df = None
        st.warning(f"Could not load Cups data: {e}")

    # Create tabs
    tab1, tab2, tab3, tab_records, tab4, tab5, tab6 = st.tabs([
        "Players", "Leaderboard", "Cups", "Records",
        "Tale of the Tape", "Match Predictor", "Ask Claude"
    ])

    with tab1:
        default_idx = all_players.index("Connolly") if "Connolly" in all_players else 0
        selected_player = st.selectbox("Player", options=all_players, index=default_idx, width=320)

        if selected_player:
            stats = get_player_stats(df, selected_player)

            if stats:
                section(f"{selected_player} — career")
                kpi_tiles([
                    ('Record', stats['record'], 'W-L-T'),
                    ('Win %', f"{stats['win_pct']:.1%}"),
                    ('Points', f"{stats['points']:.1f}"),
                    ('Matches', stats['matches']),
                    ('Events', stats['events']),
                ])

                subtab1, subtab2, subtab3, subtab4 = st.tabs([
                    "By event", "Partners", "Head-to-head", "By course"
                ])

                with subtab1:
                    event_df = get_player_by_event(df, selected_player)
                    if not event_df.empty:
                        event_df = event_df.rename(columns={'Event': 'FBC'})
                        event_df['Win%'] = to_pct(event_df['Win%'])
                        show_table(event_df, {
                            'FBC': st.column_config.NumberColumn('FBC', format="%d", width='small'),
                            'Win%': pct_column(bar=True),
                            'Points': st.column_config.NumberColumn('Points', format="%.1f"),
                        }, fit=True)
                    else:
                        st.info("No event data available.")

                with subtab2:
                    partner_df = get_partner_performance(df, selected_player)
                    if not partner_df.empty:
                        st.markdown("<p class='section-note'>Doubles record with each partner.</p>", unsafe_allow_html=True)
                        partner_df['Win%'] = to_pct(partner_df['Win%'])
                        show_table(partner_df, {'Win%': pct_column(bar=True),
                                                'Points': st.column_config.NumberColumn('Points', format="%.1f")})
                    else:
                        st.info("No doubles partner data available.")

                with subtab3:
                    h2h_df = get_head_to_head(df, selected_player)
                    if not h2h_df.empty:
                        st.markdown("<p class='section-note'>Record against each opponent, singles and doubles combined.</p>", unsafe_allow_html=True)
                        h2h_df['Win%'] = to_pct(h2h_df['Win%'])
                        show_table(h2h_df, {'Win%': pct_column(bar=True)})
                    else:
                        st.info("No head-to-head data available.")

                with subtab4:
                    course_df = get_course_performance(df, selected_player)
                    if not course_df.empty:
                        course_df['Win%'] = to_pct(course_df['Win%'])
                        show_table(course_df, {'Win%': pct_column(bar=True),
                                               'Points': st.column_config.NumberColumn('Points', format="%.1f")})
                    else:
                        st.info("No course data available.")
            else:
                st.warning("No stats found for this player.")

    with tab2:
        section("Career leaderboard", "Every match ever played. Click any column header to re-sort.")

        lb_data = leaderboard_base.copy()  # raw numbers for the highlight tiles
        sort_col = st.segmented_control(
            "Rank by", options=['Points', 'Win%', 'Matches', 'Pts/Event'],
            default='Points', key="lb_sort",
        ) or 'Points'

        # Rate stats need a sample: a 1-0-0 cameo shouldn't outrank a 77-match career
        min_matches = 20
        if sort_col in ('Win%', 'Pts/Event'):
            lb_data['_q'] = lb_data['Matches'] >= min_matches
            leaderboard = lb_data.sort_values(['_q', sort_col], ascending=False).drop(columns='_q')
            st.caption(f"Players with fewer than {min_matches} matches are listed after everyone else.")
        else:
            leaderboard = lb_data.sort_values(sort_col, ascending=False)
        leaderboard = leaderboard.reset_index(drop=True)
        leaderboard.insert(0, 'Rank', range(1, len(leaderboard) + 1))
        leaderboard['Win%'] = to_pct(leaderboard['Win%'])

        qualified = lb_data[lb_data['Matches'] >= 20]
        top_points = lb_data.nlargest(1, 'Points').iloc[0]
        top_matches = lb_data.nlargest(1, 'Matches').iloc[0]
        tiles = [('Most points', top_points['Player'], f"{top_points['Points']:.1f} pts")]
        if not qualified.empty:
            top_winpct = qualified.nlargest(1, 'Win%').iloc[0]
            tiles.append(('Best win % (20+ matches)', top_winpct['Player'], f"{top_winpct['Win%']:.1%}"))
        tiles.append(('Most matches', top_matches['Player'], f"{top_matches['Matches']} matches"))
        kpi_tiles(tiles)

        show_table(leaderboard, {
            'Rank': st.column_config.NumberColumn('#', width='small'),
            'Player': st.column_config.TextColumn('Player'),
            'Points': st.column_config.NumberColumn('Points', format="%.1f"),
            'Record': st.column_config.TextColumn('Record (W-L-T)'),
            'Win%': pct_column(bar=True),
            'Matches': st.column_config.NumberColumn('Matches'),
            'Events': st.column_config.NumberColumn('Events'),
            'Pts/Event': st.column_config.NumberColumn('Pts / event', format="%.2f"),
        }, fit=True)

    with tab3:
        if cups_df is not None:
            cups_summary = get_cups_summary(cups_df)
            if cups_summary:
                top_winner = cups_summary[0]
                tiles = [('Most cups won', top_winner['Player'], f"{top_winner['Cups Won']} cups")]
                qualified = [p for p in cups_summary if p['Cups Played'] >= 5]
                if qualified:
                    best_pct = max(qualified, key=lambda x: x['Cup Win%'])
                    tiles.append(('Best cup win % (5+ played)', best_pct['Player'], f"{best_pct['Cup Win%']:.1%}"))
                most_played = max(cups_summary, key=lambda x: x['Cups Played'])
                tiles.append(('Most cups played', most_played['Player'], f"{most_played['Cups Played']} cups"))
                kpi_tiles(tiles)

            # ----- Cup Results by Event (team scores, captains, margins) -----
            section("Results by event",
                    "Teams are named for their captain. Scores count the FTAS tiebreaker once "
                    "(0.5 to the winning team). Top scorer is the event's highest individual points total.")
            try:
                cup_results = get_cup_results_table(df)
                cup_results['Winner'] = cup_results['Winning Captain'].map(team_label)
                cup_results['Runner-up'] = cup_results['Losing Captain'].map(team_label)
                cup_results['Score'] = cup_results.apply(
                    lambda r: f"{fmt_pts(r['Winner Total'])} – {fmt_pts(r['Loser Total'])}", axis=1)
                cup_results = cup_results.sort_values('FBC', ascending=False)
                # Rain stays in the Ask Claude context but is left out of this table
                show_table(
                    cup_results[['FBC', 'Location', 'Winner', 'Runner-up', 'Score', 'Margin',
                                 'Highest Individual']],
                    {
                        'FBC': st.column_config.NumberColumn('FBC', format="%d", width='small'),
                        'Margin': st.column_config.NumberColumn('Margin', format="%.1f", width='small'),
                        'Highest Individual': st.column_config.TextColumn('Top scorer'),
                    },
                    fit=True,
                )
                # Notes are sparse, so list them under the table rather than as a mostly-empty column
                noted = cup_results[cup_results['Notes'].astype(str).str.strip() != '']
                for _, r in noted.iterrows():
                    st.caption(f"**FBC {r['FBC']}:** {r['Notes']}")
            except Exception as e:
                st.warning(f"Could not build the cup results table: {e}")

            section("Results by player", "Was each player on the winning side?")
            st.markdown(f"""
            <div class="legend-chips">
              <span><i style="background:{COLORS['primary_soft']};color:{COLORS['win']}">W</i>won the cup</span>
              <span><i style="color:{COLORS['loss']};border:1px solid {COLORS['border']}">L</i>lost</span>
              <span><i style="color:#B7BDB6;border:1px solid {COLORS['border']}">·</i>didn't play</span>
            </div>""", unsafe_allow_html=True)

            display_df = cups_df.copy()
            display_df = display_df.sort_values(['Total', 'Win%'], ascending=False)
            display_df['Win%'] = to_pct(display_df['Win%'].fillna(0))

            # Per-cup 1/0/X cells -> W / L / · (uniform strings also keep Arrow happy:
            # mixed int/str columns fail serialization and spam the server log).
            fbc_col_names = [c for _, c in _fbc_columns(display_df)]
            for c in fbc_col_names:
                display_df[c] = display_df[c].map(
                    lambda v: CUP_CELL.get(str(int(v)) if v in (0, 1, 0.0, 1.0) else str(v).strip().upper(), '·')
                    if pd.notna(v) else '·')
            display_df = display_df[['Player'] + fbc_col_names + ['Total', 'Played', 'Win%']]
            styled = display_df.style.map(_cup_cell_style, subset=fbc_col_names)
            cfg = {c: st.column_config.TextColumn(c.replace('FBC ', ''), width=40) for c in fbc_col_names}
            cfg.update({
                'Player': st.column_config.TextColumn('Player', pinned=True, width=110),
                'Total': st.column_config.NumberColumn('Won'),
                'Played': st.column_config.NumberColumn('Played'),
                'Win%': pct_column(),
            })
            show_table(styled, cfg, fit=True)
        else:
            st.error("Cups data could not be loaded.")

    with tab_records:
        streaks = get_streak_records(df)
        team_results = get_fbc_team_results(df)

        # Headline cup records
        if team_results:
            max_margin = max(r['margin'] for r in team_results)
            biggest = [r for r in team_results if abs(r['margin'] - max_margin) < 1e-9]
            decided = [r for r in team_results if not r['tie']]
            closest = min(decided, key=lambda x: x['margin']) if decided else None
            tiles = [('Biggest cup margin', ' & '.join(team_label(b['winner']) for b in biggest),
                      f"{max_margin:.1f} pts · " + ', '.join(f"FBC {b['fbc']}" for b in biggest))]
            if closest:
                tiles.append(('Closest cup', team_label(closest['winner']),
                              f"by {closest['margin']:.1f} at FBC {closest['fbc']}"))
            if streaks['win']:
                top_streak = streaks['win'][0]
                tiles.append(('Longest win streak', top_streak['Player'],
                              f"{top_streak['Streak']} matches · {top_streak['Span']}"))
            kpi_tiles(tiles)

        streak_cfg = {'Streak': st.column_config.NumberColumn('Matches', width='small')}
        col1, col2 = st.columns(2, gap="large")
        with col1:
            section("Longest win streaks", level=4)
            show_table(pd.DataFrame(streaks['win'][:10]), streak_cfg)
        with col2:
            section("Longest unbeaten streaks", level=4)
            show_table(pd.DataFrame(streaks['unbeaten'][:10]), streak_cfg)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            section("Longest losing streaks", level=4)
            show_table(pd.DataFrame(streaks['loss'][:10]), streak_cfg)
        with col2:
            section("Active streaks", "Heading into the next cup.", level=4)
            active = [s for s in streaks['current'] if s['Length'] >= 2 and s['Type'] != 'T']
            if active:
                show_table(pd.DataFrame(active[:10])[['Player', 'Streak']])
            else:
                st.info("No active streaks of 2+ matches.")

        section("Most lopsided match wins", level=4)
        blowouts = get_biggest_match_wins(df)
        if blowouts:
            show_table(pd.DataFrame(blowouts[:10]), {'FBC': st.column_config.NumberColumn('FBC', format="%d")})
        else:
            st.info("No match-play margins recorded.")

        section("Perfect events", "No losses across 5+ matches at one cup.", level=4)
        perfect = get_perfect_events(df)
        if perfect:
            pdf_ = pd.DataFrame(perfect)[['Player', 'FBC', 'Location', 'Record', 'Points']]
            show_table(pdf_, {'FBC': st.column_config.NumberColumn('FBC', format="%d"),
                              'Points': st.column_config.NumberColumn('Points', format="%.1f")})
        else:
            st.info("Nobody has finished an event unbeaten (5+ matches) — yet.")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            section("Most consecutive cups won", "Skipped cups don't break a run.", level=4)
            if cups_df is not None:
                consec = get_consecutive_cup_wins(cups_df)
                show_table(pd.DataFrame(consec[:10]),
                           {'Consecutive Cups Won': st.column_config.NumberColumn('Cups', width='small')})
            else:
                st.info("Cups data unavailable.")
        with col2:
            section("Best partnerships", "Minimum 5 matches together.", level=4)
            partnerships = [p for p in get_all_partnership_stats(df) if p['Matches'] >= 5]
            partnerships.sort(key=lambda x: (x['Win%'], x['Matches']), reverse=True)
            if partnerships:
                pp = pd.DataFrame(partnerships[:10])[['Partnership', 'Record', 'Win%', 'Matches']]
                pp['Win%'] = to_pct(pp['Win%'])
                show_table(pp, {'Win%': pct_column()})

    with tab4:
        section("Tale of the Tape", "Compare any two players across every metric.")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='side-label p1'>Player 1</div>", unsafe_allow_html=True)
            tape_player1 = st.selectbox("Player 1", options=all_players, index=_default_idx(0),
                                        key="tape_p1", label_visibility="collapsed")
        with col2:
            st.markdown("<div class='side-label p2'>Player 2</div>", unsafe_allow_html=True)
            tape_player2 = st.selectbox("Player 2", options=all_players, index=_default_idx(1),
                                        key="tape_p2", label_visibility="collapsed")

        if tape_player1 and tape_player2 and tape_player1 != tape_player2:
            p1_stats = get_player_stats(df, tape_player1)
            p2_stats = get_player_stats(df, tape_player2)

            if p1_stats and p2_stats:
                section("Career", level=4)
                col1, col2 = st.columns(2, gap="large")
                for col, name, s_, variant in ((col1, tape_player1, p1_stats, 'p1'),
                                               (col2, tape_player2, p2_stats, 'p2')):
                    with col:
                        kpi_tiles([
                            ('Record', s_['record']),
                            ('Win %', f"{s_['win_pct']:.1%}"),
                            ('Points', f"{s_['points']:.1f}", f"{s_['events']} events"),
                        ], variant=variant)

                section("Direct head-to-head", level=4)
                h2h = get_direct_h2h(df, tape_player1, tape_player2)
                if h2h['matches'] > 0:
                    total = h2h['matches']
                    p1_share = (h2h['wins'] + 0.5 * h2h['ties']) / total * 100
                    kpi_tiles([
                        (f"{tape_player1} wins", h2h['wins']),
                        ('Halved', h2h['ties'], f"{total} meetings"),
                        (f"{tape_player2} wins", h2h['losses']),
                    ])
                    vs_bar(tape_player1, p1_share, tape_player2, 100 - p1_share)
                else:
                    st.info(f"{tape_player1} and {tape_player2} have never faced each other directly.")

                section("By format", level=4)
                p1_formats = get_stats_by_format(df, tape_player1)
                p2_formats = get_stats_by_format(df, tape_player2)

                format_data = []
                for fmt in ['Singles', 'Doubles', 'FTAS']:
                    p1_f = p1_formats.get(fmt, {})
                    p2_f = p2_formats.get(fmt, {})
                    format_data.append({
                        'Format': fmt,
                        f'{tape_player1}': f"{p1_f.get('wins', 0)}-{p1_f.get('losses', 0)}-{p1_f.get('ties', 0)} ({p1_f.get('win_pct', 0):.1%})" if p1_f.get('matches', 0) > 0 else "—",
                        f'{tape_player2}': f"{p2_f.get('wins', 0)}-{p2_f.get('losses', 0)}-{p2_f.get('ties', 0)} ({p2_f.get('win_pct', 0):.1%})" if p2_f.get('matches', 0) > 0 else "—",
                        'Edge': tape_player1 if p1_f.get('win_pct', 0) > p2_f.get('win_pct', 0) else (tape_player2 if p2_f.get('win_pct', 0) > p1_f.get('win_pct', 0) else 'Even')
                    })
                show_table(pd.DataFrame(format_data))

                pct_cfg = {'Win%': pct_column()}
                col1, col2 = st.columns(2, gap="large")
                for col, name in ((col1, tape_player1), (col2, tape_player2)):
                    with col:
                        best, worst = get_best_worst_courses(df, name)
                        section(f"{name}: best courses", level=4)
                        if best:
                            show_table(_record_rows(best, 'course', 'Course'), pct_cfg)
                        else:
                            st.info("Not enough course data.")
                        if worst and worst != best:
                            section(f"{name}: toughest courses", level=4)
                            show_table(_record_rows(worst, 'course', 'Course'), pct_cfg)
                        section(f"{name}: best partners", level=4)
                        partners = get_best_partners(df, name)
                        if partners:
                            show_table(_record_rows(partners, 'partner', 'Partner'), pct_cfg)
                        else:
                            st.info("No doubles partner data.")

        elif tape_player1 == tape_player2:
            st.warning("Please select two different players to compare.")

    with tab5:
        section("Match Predictor", "Win probability from career form, head-to-head and recent results.")

        match_type = st.segmented_control("Match type", ["Singles", "Doubles"], default="Singles",
                                          key="pred_match_type") or "Singles"
        all_courses = sorted([c for c in df['Course'].dropna().unique() if isinstance(c, str)])

        def _render_prediction(prediction, left, right, left_col, right_col):
            section("Prediction", level=4)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                kpi_tiles([(left, f"{prediction['p1_prob']:.0f}%", 'favorite' if prediction['p1_prob'] > 50 else '')], variant='p1')
            with col2:
                kpi_tiles([(right, f"{prediction['p2_prob']:.0f}%", 'favorite' if prediction['p2_prob'] > 50 else '')], variant='p2')
            vs_bar(left, prediction['p1_prob'], right, prediction['p2_prob'])

            section("What's driving it", level=4)
            if prediction['factors']:
                factors_df = pd.DataFrame(prediction['factors']).rename(columns={
                    'factor': 'Factor', 'p1_value': left_col, 'p2_value': right_col, 'edge': 'Edge'})
                show_table(factors_df[['Factor', left_col, right_col, 'Edge']])
            else:
                st.info("Not enough historical data to analyze factors.")

        if match_type == "Singles":
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("<div class='side-label p1'>Player 1</div>", unsafe_allow_html=True)
                pred_p1 = st.selectbox("Player 1", options=all_players, index=_default_idx(0),
                                       key="pred_singles_p1", label_visibility="collapsed")
            with col2:
                st.markdown("<div class='side-label p2'>Player 2</div>", unsafe_allow_html=True)
                pred_p2 = st.selectbox("Player 2", options=all_players, index=_default_idx(1),
                                       key="pred_singles_p2", label_visibility="collapsed")

            pred_course = st.selectbox("Course (optional)", options=["Any course"] + all_courses, key="pred_course_singles")
            pred_course = None if pred_course == "Any course" else pred_course

            if pred_p1 != pred_p2:
                if st.button("Predict match", type="primary", key="predict_singles"):
                    prediction = predict_match(df, pred_p1, pred_p2, course=pred_course)
                    _render_prediction(prediction, pred_p1, pred_p2, pred_p1, pred_p2)
            else:
                st.warning("Please select two different players.")

        else:  # Doubles
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("<div class='side-label p1'>Team 1</div>", unsafe_allow_html=True)
                pred_d1_p1 = st.selectbox("Team 1 — player A", options=all_players, index=_default_idx(0), key="pred_d1_p1")
                pred_d1_p2 = st.selectbox("Team 1 — player B", options=all_players, index=_default_idx(3), key="pred_d1_p2")
            with col2:
                st.markdown("<div class='side-label p2'>Team 2</div>", unsafe_allow_html=True)
                pred_d2_p1 = st.selectbox("Team 2 — player A", options=all_players, index=_default_idx(1), key="pred_d2_p1")
                pred_d2_p2 = st.selectbox("Team 2 — player B", options=all_players, index=_default_idx(2), key="pred_d2_p2")

            pred_course_d = st.selectbox("Course (optional)", options=["Any course"] + all_courses, key="pred_course_doubles")
            pred_course_d = None if pred_course_d == "Any course" else pred_course_d

            team1 = {pred_d1_p1, pred_d1_p2}
            team2 = {pred_d2_p1, pred_d2_p2}

            if len(team1) == 2 and len(team2) == 2 and not team1.intersection(team2):
                if st.button("Predict match", type="primary", key="predict_doubles"):
                    prediction = predict_match(df, pred_d1_p1, pred_d2_p1, course=pred_course_d,
                                              is_doubles=True, partner1=pred_d1_p2, partner2=pred_d2_p2)
                    _render_prediction(prediction,
                                       f"{pred_d1_p1} & {pred_d1_p2}", f"{pred_d2_p1} & {pred_d2_p2}",
                                       f"{pred_d1_p1}/{pred_d1_p2}", f"{pred_d2_p1}/{pred_d2_p2}")
            else:
                st.warning("Please select 4 different players (no player can be on both teams or appear twice).")

    with tab6:
        section("Ask Claude",
                "Ask anything about FBC history — player stats, head-to-heads, courses, trends. "
                "Follow-ups work: ask about a player, then “what about in singles?”")

        # Initialize session state
        if 'claude_history' not in st.session_state:
            st.session_state.claude_history = []  # list of {'q': ..., 'a': ...}
        if 'claude_question' not in st.session_state:
            st.session_state.claude_question = ""
        if 'submit_question' not in st.session_state:
            st.session_state.submit_question = False

        example_questions = [
            "Who has won the most cups?",
            "Which captain won by the biggest margin of victory?",
            "What's Hilts and Lynch's record as partners?",
            "Who has the most singles wins?",
            "Who had the most points at FBC 11?",
            "What was the closest cup ever?"
        ]

        if not st.session_state.claude_history:
            picked = st.pills("Try one", example_questions, key="claude_examples")
            if picked:
                st.session_state.claude_question = picked
                st.session_state.submit_question = True
                st.session_state.pop("claude_examples", None)
                st.rerun()

        # Conversation so far (renders Claude's markdown natively)
        for turn in st.session_state.claude_history:
            with st.chat_message("user"):
                st.markdown(turn['q'])
            with st.chat_message("assistant"):
                st.markdown(turn['a'])

        # Form for question submission (Enter key or button both work)
        in_conversation = len(st.session_state.claude_history) > 0
        with st.form("claude_question_form", clear_on_submit=True, border=False):
            user_question = st.text_input(
                "Your question" if not in_conversation else "Ask a follow-up",
                placeholder="e.g., Who has the best overall win percentage?" if not in_conversation
                            else "e.g., What about in doubles?",
                key="question_input"
            )
            form_submitted = st.form_submit_button("Ask", type="primary")

        # Resolve what to ask (typed question, or auto-submit from an example)
        question_to_ask = None
        if form_submitted:
            if user_question.strip():
                question_to_ask = user_question.strip()
            else:
                st.warning("Please enter a question.")
        elif st.session_state.submit_question:
            st.session_state.submit_question = False
            question_to_ask = st.session_state.claude_question

        if question_to_ask:
            if "ANTHROPIC_API_KEY" not in st.secrets:
                st.error("Anthropic API key not configured. Please add ANTHROPIC_API_KEY to your Streamlit secrets.")
            else:
                with st.chat_message("user"):
                    st.markdown(question_to_ask)
                with st.spinner("Claude is analyzing the FBC data..."):
                    try:
                        hist = [(t['q'], t['a']) for t in st.session_state.claude_history]
                        response = ask_claude(question_to_ask, df, cups_df, history=hist)
                        st.session_state.claude_history.append({'q': question_to_ask, 'a': response})
                        st.rerun()
                    except anthropic.AuthenticationError:
                        st.error("Invalid API key. Please check your ANTHROPIC_API_KEY in Streamlit secrets.")
                    except Exception as e:
                        st.error(f"Error getting response from Claude: {str(e)}")

        if st.session_state.claude_history:
            if st.button("Start a new conversation", key="clear_chat", type="tertiary"):
                st.session_state.claude_history = []
                st.rerun()

        with st.expander("API key setup"):
            st.markdown("""
            Ask Claude needs an Anthropic API key in Streamlit secrets:

            - **Recommended:** `~/.streamlit/secrets.toml` (keeps the key out of the Dropbox-synced folder)
            - Or `.streamlit/secrets.toml` next to `app.py`

            Add the line `ANTHROPIC_API_KEY = "your-api-key-here"`. Get a key at https://console.anthropic.com/
            """)

    # Data health check — surfaces entry errors (one-sided matches, phantom teams,
    # name typos) so they get caught right after new FBC data is entered.
    st.divider()
    try:
        issues = validate_data(df, cups_df)
        if issues:
            with st.expander(f"Data health: {len(issues)} issue(s) found — click to review", expanded=False):
                for issue in issues:
                    st.warning(issue)
        else:
            st.caption("Data health: all integrity checks pass "
                       "(required columns filled, two-sided matches, valid W/L/T, exact "
                       "Singles/Doubles values, two teams per event, no name mismatches, "
                       "a Cups column for every event).")
    except Exception as e:
        st.caption(f"Data health check could not run: {e}")

if __name__ == "__main__":
    main()
