import streamlit as st
import pandas as pd
import numpy as np
import anthropic

# Page config
st.set_page_config(
    page_title="FBC Stats",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Golf-themed color scheme
COLORS = {
    'primary': '#1B4D3E',      # Dark green (masters green)
    'secondary': '#2E8B57',    # Sea green
    'accent': '#FFD700',       # Gold
    'light': '#90EE90',        # Light green
    'bg': '#F5F5DC',           # Beige
    'text': '#1B4D3E',
    'win': '#2E8B57',
    'loss': '#DC143C',
    'tie': '#DAA520'
}

# Custom CSS for mobile-friendly design
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    .main-header h1 {{
        margin: 0;
        font-size: 2rem;
    }}
    .stat-card {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid {COLORS['primary']};
    }}
    .stat-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
    }}
    .section-header {{
        color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['accent']};
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }}
    .win {{ color: {COLORS['win']}; font-weight: bold; }}
    .loss {{ color: {COLORS['loss']}; font-weight: bold; }}
    .tie {{ color: {COLORS['tie']}; font-weight: bold; }}
    .dataframe {{
        font-size: 0.9rem !important;
    }}
    @media (max-width: 768px) {{
        .stat-value {{ font-size: 1.4rem; }}
        .stat-label {{ font-size: 0.75rem; }}
        .main-header h1 {{ font-size: 1.5rem; }}
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.5rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 5px 5px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_cups_data():
    """Load and process the Cups data showing which players won each cup."""
    cups_raw = pd.read_excel('FBC_Data.xlsx', sheet_name='Cups')

    # The data starts at row 1 (0-indexed), with headers in row 0
    # Column 1 is Player, columns 2-13 are FBC 1-12, then Total, Played, %, Lost
    cups_df = cups_raw.iloc[1:31].copy()  # Rows 1-30 contain player data
    cups_df.columns = ['Drop', 'Player', 'FBC 1', 'FBC 2', 'FBC 3', 'FBC 4', 'FBC 5', 'FBC 6',
                       'FBC 7', 'FBC 8', 'FBC 9', 'FBC 10', 'FBC 11', 'FBC 12',
                       'Total', 'Played', 'Win%', 'Lost']
    cups_df = cups_df.drop(columns=['Drop'])
    cups_df = cups_df[cups_df['Player'].notna()]

    # Convert Win% to float
    cups_df['Win%'] = pd.to_numeric(cups_df['Win%'], errors='coerce')
    cups_df['Total'] = pd.to_numeric(cups_df['Total'], errors='coerce')
    cups_df['Played'] = pd.to_numeric(cups_df['Played'], errors='coerce')
    cups_df['Lost'] = pd.to_numeric(cups_df['Lost'], errors='coerce')

    return cups_df

def get_cups_summary(cups_df):
    """Get summary statistics about cup wins."""
    summary = []
    for _, row in cups_df.iterrows():
        player = row['Player']
        total_wins = row['Total'] if pd.notna(row['Total']) else 0
        total_played = row['Played'] if pd.notna(row['Played']) else 0
        win_pct = row['Win%'] if pd.notna(row['Win%']) else 0

        # Count individual cup results
        cup_results = []
        for i in range(1, 13):
            result = row.get(f'FBC {i}', 'X')
            if result == 1 or result == '1':
                cup_results.append(f"FBC {i}: Won")
            elif result == 0 or result == '0':
                cup_results.append(f"FBC {i}: Lost")
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
def load_data():
    """Load and process the FBC data."""
    df = pd.read_excel('FBC_Data.xlsx', sheet_name='Archives')

    # Clean up the data
    df = df.dropna(subset=['Player 1'])
    df = df[df['Player 1'].apply(lambda x: isinstance(x, str))]

    # Standardize player names (fix inconsistencies)
    name_map = {
        'Shivelli': 'Shively',
        'DeOteris': 'Deoteris',
        'Connolly, R': 'R. Connolly'
    }
    df['Player 1'] = df['Player 1'].replace(name_map)
    df['Player 2'] = df['Player 2'].replace(name_map)
    df['Opponent1'] = df['Opponent1'].replace(name_map)
    df['Opponent2'] = df['Opponent2'].replace(name_map)
    df['Singles Opponent'] = df['Singles Opponent'].replace(name_map)

    return df

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

def get_partner_performance(df, player):
    """Get player's record with each doubles partner."""
    doubles_matches = df[df['Singes/Doubles'] == 'Doubles'].copy()

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

    return course_stats[['Course', 'Record', 'Win%', 'Points', 'Matches']].sort_values('Win%', ascending=False)

def get_leaderboard(df):
    """Calculate overall leaderboard."""
    all_players = set(df['Player 1'].dropna().unique()) | set(df[df['Player 2'].notna()]['Player 2'].unique())

    leaderboard = []
    for player in all_players:
        if not isinstance(player, str):
            continue
        stats = get_player_stats(df, player)
        if stats:
            leaderboard.append({
                'Player': player,
                'Points': stats['points'],
                'Record': stats['record'],
                'Win%': stats['win_pct'],
                'Matches': stats['matches'],
                'Events': stats['events'],
                'Pts/Event': stats['points'] / stats['events'] if stats['events'] > 0 else 0
            })

    return pd.DataFrame(leaderboard).sort_values('Points', ascending=False).reset_index(drop=True)

def extract_fbc_number(question):
    """Extract FBC event number from a question if mentioned."""
    import re
    # Match patterns like "FBC 11", "FBC11", "fbc 11", "FBC XI", etc.
    patterns = [
        r'fbc\s*(\d+)',  # FBC 11, FBC11
        r'fbc\s*(xi+|iv|v?i{0,3})\b',  # Roman numerals
    ]
    question_lower = question.lower()
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            num = match.group(1)
            # Convert roman numerals if needed
            roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
                        'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
                        'xi': 11, 'xii': 12, 'xiii': 13}
            if num in roman_map:
                return roman_map[num]
            try:
                return int(num)
            except ValueError:
                continue
    return None

def extract_player_names(question, all_players):
    """Extract player names mentioned in a question."""
    question_lower = question.lower()
    mentioned = []
    for player in all_players:
        if player.lower() in question_lower:
            mentioned.append(player)
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
    """Calculate stats for all players in a subset of matches."""
    if players is None:
        players = set(df['Player 1'].dropna().unique()) | set(df[df['Player 2'].notna()]['Player 2'].unique())
        players = [p for p in players if isinstance(p, str)]

    stats = []
    for player in players:
        player_matches = df[(df['Player 1'] == player) | (df['Player 2'] == player)]
        if len(player_matches) == 0:
            continue

        wins = player_matches['W'].sum()
        losses = player_matches['L'].sum()
        ties = player_matches['T'].sum()
        points = player_matches['Points earned'].sum()
        matches = len(player_matches)

        stats.append({
            'Player': player,
            'Points': points,
            'Wins': int(wins),
            'Losses': int(losses),
            'Ties': int(ties),
            'Matches': matches,
            'Record': f"{int(wins)}-{int(losses)}-{int(ties)}",
            'Win%': (wins + 0.5 * ties) / matches if matches > 0 else 0
        })

    return sorted(stats, key=lambda x: x['Points'], reverse=True)

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
            match_types = event_df['Singes/Doubles'].value_counts().to_dict()
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
                match_type = row.get('Singes/Doubles', '')
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
                type_matches = player_matches[player_matches['Singes/Doubles'] == match_type]
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
    context_parts.append(f"\n  LIFETIME LEADERBOARD (Top 20):")
    overall_stats = calculate_player_stats_for_subset(df)
    for i, stat in enumerate(overall_stats[:20], 1):
        context_parts.append(f"    {i}. {stat['Player']}: {stat['Points']:.1f} pts, {stat['Record']}, {stat['Win%']:.1%}")

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
                player_cup_data = cups_df[cups_df['Player'].str.lower() == player.lower()]
                if len(player_cup_data) == 0:
                    # Try partial match
                    player_cup_data = cups_df[cups_df['Player'].str.lower().str.contains(player.lower(), na=False)]

                if len(player_cup_data) > 0:
                    row = player_cup_data.iloc[0]
                    context_parts.append(f"\n  {player.upper()}'S CUP HISTORY:")
                    for fbc_num in range(1, 13):
                        result = row.get(f'FBC {fbc_num}', 'X')
                        if result == 1 or result == '1':
                            context_parts.append(f"    FBC {fbc_num}: WON (on winning team)")
                        elif result == 0 or result == '0':
                            context_parts.append(f"    FBC {fbc_num}: LOST (on losing team)")
                        else:
                            context_parts.append(f"    FBC {fbc_num}: Did not participate")
                    total = row.get('Total', 0)
                    played = row.get('Played', 0)
                    context_parts.append(f"    TOTAL: {int(total)} cups won out of {int(played)} played")

    return '\n'.join(context_parts)

def ask_claude(question, df, cups_df=None):
    """Send a question to Claude with relevant FBC data context."""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Load cups data if not provided
    if cups_df is None:
        try:
            cups_df = load_cups_data()
        except Exception:
            cups_df = None

    # Prepare context based on the question
    data_context = prepare_data_context(df, question, cups_df)

    system_prompt = """You are an expert analyst for the FBC (Freddie B Cup), a golf match play tournament between friends.
You have access to historical match data and should answer questions about player statistics, head-to-head records,
course performance, tournament history, and CUP CHAMPIONSHIPS (which team won each FBC event).

IMPORTANT: The data provided includes pre-calculated statistics and complete match records. Use these directly -
do not try to recalculate from raw data. When asked about points, wins, or records, cite the exact numbers from
the leaderboard or stats provided.

For CUP questions: A "cup win" means the player was on the winning TEAM for that FBC event. This is different
from individual match wins. The Cups data shows team championship results.

Be concise but thorough. Always cite the specific data that supports your answer."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"""Here is the FBC tournament data relevant to your question:

{data_context}

Question: {question}

Please answer based on the data provided above. Cite specific statistics."""
            }
        ]
    )

    return message.content[0].text

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
        fmt_matches = player_matches[player_matches['Singes/Doubles'] == fmt]
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
    doubles = df[df['Singes/Doubles'] == 'Doubles']

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
    doubles = df[df['Singes/Doubles'] == 'Doubles']

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
        h2h_pct = h2h['wins'] / h2h['matches'] if h2h['matches'] > 0 else 0.5
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
                'factor': f'Course ({course[:20]}...)',
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

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⛳ FBC Statistics Dashboard</h1>
        <p style="margin:0;opacity:0.9;">Freddie B Cup</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Get list of all players
    all_players = sorted(set(df['Player 1'].dropna().unique()) |
                        set(df[df['Player 2'].notna()]['Player 2'].unique()))
    all_players = [p for p in all_players if isinstance(p, str)]

    # Load cups data
    try:
        cups_df = load_cups_data()
    except Exception as e:
        cups_df = None
        st.warning(f"Could not load Cups data: {e}")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Player Stats", "🏆 Leaderboard", "🏅 Cups",
        "⚔️ Tale of the Tape", "🎯 Match Predictor", "🤖 Ask Claude"
    ])

    with tab1:
        # Player selection
        col1, col2 = st.columns([2, 1])
        with col1:
            # Default to Connolly if available
            default_idx = all_players.index("Connolly") if "Connolly" in all_players else 0
            selected_player = st.selectbox(
                "Select a Player",
                options=all_players,
                index=default_idx
            )

        if selected_player:
            stats = get_player_stats(df, selected_player)

            if stats:
                # Career stats cards
                st.markdown(f"<h3 class='section-header'>{selected_player}'s Career Stats</h3>", unsafe_allow_html=True)

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['record']}</div>
                        <div class="stat-label">Record</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['win_pct']:.1%}</div>
                        <div class="stat-label">Win %</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['points']:.1f}</div>
                        <div class="stat-label">Points</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['matches']}</div>
                        <div class="stat-label">Matches</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['events']}</div>
                        <div class="stat-label">Events</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Sub-tabs for detailed stats
                subtab1, subtab2, subtab3, subtab4 = st.tabs([
                    "📅 By Event", "👥 Partners", "🎯 Head-to-Head", "🏌️ By Course"
                ])

                with subtab1:
                    st.markdown("<h4 class='section-header'>Record by FBC Event</h4>", unsafe_allow_html=True)
                    event_df = get_player_by_event(df, selected_player)
                    if not event_df.empty:
                        event_df['Win%'] = event_df['Win%'].apply(format_pct)
                        st.dataframe(
                            event_df,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No event data available.")

                with subtab2:
                    st.markdown("<h4 class='section-header'>Doubles Partner Performance</h4>", unsafe_allow_html=True)
                    partner_df = get_partner_performance(df, selected_player)
                    if not partner_df.empty:
                        partner_df['Win%'] = partner_df['Win%'].apply(format_pct)
                        st.dataframe(
                            partner_df,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No doubles partner data available.")

                with subtab3:
                    st.markdown("<h4 class='section-header'>Head-to-Head Record</h4>", unsafe_allow_html=True)
                    h2h_df = get_head_to_head(df, selected_player)
                    if not h2h_df.empty:
                        h2h_df['Win%'] = h2h_df['Win%'].apply(format_pct)
                        st.dataframe(
                            h2h_df,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No head-to-head data available.")

                with subtab4:
                    st.markdown("<h4 class='section-header'>Performance by Course</h4>", unsafe_allow_html=True)
                    course_df = get_course_performance(df, selected_player)
                    if not course_df.empty:
                        course_df['Win%'] = course_df['Win%'].apply(format_pct)
                        st.dataframe(
                            course_df,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No course data available.")
            else:
                st.warning("No stats found for this player.")

    with tab2:
        st.markdown("<h3 class='section-header'>🏆 Overall Leaderboard</h3>", unsafe_allow_html=True)

        # Sorting options
        sort_col = st.selectbox(
            "Sort by",
            options=['Points', 'Win%', 'Matches', 'Events', 'Pts/Event'],
            index=0
        )

        leaderboard = get_leaderboard(df)
        leaderboard = leaderboard.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Add rank column
        leaderboard.insert(0, 'Rank', range(1, len(leaderboard) + 1))

        # Format percentages
        leaderboard['Win%'] = leaderboard['Win%'].apply(format_pct)
        leaderboard['Pts/Event'] = leaderboard['Pts/Event'].apply(lambda x: f"{x:.2f}")

        # Display leaderboard with highlighting
        st.dataframe(
            leaderboard,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Rank': st.column_config.NumberColumn('Rank', width='small'),
                'Player': st.column_config.TextColumn('Player', width='medium'),
                'Points': st.column_config.NumberColumn('Points', format="%.1f"),
                'Record': st.column_config.TextColumn('Record', width='small'),
                'Win%': st.column_config.TextColumn('Win%', width='small'),
                'Matches': st.column_config.NumberColumn('Matches', width='small'),
                'Events': st.column_config.NumberColumn('Events', width='small'),
                'Pts/Event': st.column_config.TextColumn('Pts/Event', width='small'),
            }
        )

        # Quick stats
        st.markdown("<h4 class='section-header'>Quick Stats</h4>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        lb_data = get_leaderboard(df)

        with col1:
            top_points = lb_data.nlargest(1, 'Points').iloc[0]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{top_points['Player']}</div>
                <div class="stat-label">Most Points ({top_points['Points']:.1f})</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Filter for players with at least 20 matches for meaningful win%
            qualified = lb_data[lb_data['Matches'] >= 20]
            if not qualified.empty:
                top_winpct = qualified.nlargest(1, 'Win%').iloc[0]
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{top_winpct['Player']}</div>
                    <div class="stat-label">Best Win% ({top_winpct['Win%']:.1%})</div>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            top_matches = lb_data.nlargest(1, 'Matches').iloc[0]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{top_matches['Player']}</div>
                <div class="stat-label">Most Matches ({top_matches['Matches']})</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h3 class='section-header'>🏅 Cup Championships</h3>", unsafe_allow_html=True)

        if cups_df is not None:
            st.markdown("""
            This shows which players were on the **winning team** at each FBC event.
            - **1** = On winning team
            - **0** = On losing team
            - **X** = Did not participate
            """)

            # Summary stats
            cups_summary = get_cups_summary(cups_df)

            col1, col2, col3 = st.columns(3)

            # Most cups won
            if cups_summary:
                top_winner = cups_summary[0]
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{top_winner['Player']}</div>
                        <div class="stat-label">Most Cups ({top_winner['Cups Won']})</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Best cup win percentage (min 5 cups played)
                qualified = [p for p in cups_summary if p['Cups Played'] >= 5]
                if qualified:
                    best_pct = max(qualified, key=lambda x: x['Cup Win%'])
                    with col2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{best_pct['Player']}</div>
                            <div class="stat-label">Best Cup Win% ({best_pct['Cup Win%']:.1%})</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Most cups played
                most_played = max(cups_summary, key=lambda x: x['Cups Played'])
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{most_played['Player']}</div>
                        <div class="stat-label">Most Cups Played ({most_played['Cups Played']})</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<h4 class='section-header'>Cup Results by Player</h4>", unsafe_allow_html=True)

            # Create display dataframe
            display_df = cups_df.copy()

            # Format Win% as percentage
            display_df['Win%'] = display_df['Win%'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")

            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['Total', 'Win%', 'Played', 'Player'],
                index=0,
                key="cups_sort"
            )

            if sort_by == 'Player':
                display_df = display_df.sort_values('Player')
            elif sort_by == 'Win%':
                display_df = display_df.sort_values(cups_df['Win%'], ascending=False)
            else:
                display_df = display_df.sort_values(sort_by, ascending=False)

            # Show the table
            st.dataframe(
                display_df[['Player', 'FBC 1', 'FBC 2', 'FBC 3', 'FBC 4', 'FBC 5', 'FBC 6',
                           'FBC 7', 'FBC 8', 'FBC 9', 'FBC 10', 'FBC 11', 'FBC 12',
                           'Total', 'Played', 'Win%']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.error("Cups data could not be loaded.")

    with tab4:
        st.markdown("<h3 class='section-header'>⚔️ Tale of the Tape</h3>", unsafe_allow_html=True)
        st.markdown("Compare any two players head-to-head across all metrics.")

        # Player selection
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='text-align: center; color: {COLORS['primary']}; font-weight: bold;'>PLAYER 1</div>", unsafe_allow_html=True)
            tape_player1 = st.selectbox("Select Player 1", options=all_players, index=0, key="tape_p1")
        with col2:
            st.markdown(f"<div style='text-align: center; color: {COLORS['loss']}; font-weight: bold;'>PLAYER 2</div>", unsafe_allow_html=True)
            # Default to a different player
            default_p2_idx = 1 if len(all_players) > 1 else 0
            tape_player2 = st.selectbox("Select Player 2", options=all_players, index=default_p2_idx, key="tape_p2")

        if tape_player1 and tape_player2 and tape_player1 != tape_player2:
            st.markdown("---")

            # Get stats for both players
            p1_stats = get_player_stats(df, tape_player1)
            p2_stats = get_player_stats(df, tape_player2)

            if p1_stats and p2_stats:
                # Overall Records - Side by Side
                st.markdown("<h4 class='section-header'>Overall Career Records</h4>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2, 1, 2])

                with col1:
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: {COLORS['primary']};">
                        <div class="stat-value">{p1_stats['record']}</div>
                        <div class="stat-label">{tape_player1}</div>
                        <div style="margin-top: 0.5rem;">
                            <span style="color: {COLORS['primary']};">{p1_stats['win_pct']:.1%} Win</span> |
                            {p1_stats['points']:.1f} pts | {p1_stats['events']} events
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem; font-size: 2rem; font-weight: bold;">
                        VS
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: {COLORS['loss']};">
                        <div class="stat-value">{p2_stats['record']}</div>
                        <div class="stat-label">{tape_player2}</div>
                        <div style="margin-top: 0.5rem;">
                            <span style="color: {COLORS['loss']};">{p2_stats['win_pct']:.1%} Win</span> |
                            {p2_stats['points']:.1f} pts | {p2_stats['events']} events
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Direct Head-to-Head
                st.markdown("<h4 class='section-header'>Direct Head-to-Head</h4>", unsafe_allow_html=True)
                h2h = get_direct_h2h(df, tape_player1, tape_player2)

                if h2h['matches'] > 0:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {COLORS['win'] if h2h['wins'] > h2h['losses'] else COLORS['primary']};">
                            <div class="stat-value">{h2h['wins']}</div>
                            <div class="stat-label">Wins vs {tape_player2}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 1.2rem; color: {COLORS['tie']};">{h2h['ties']} Ties</div>
                            <div style="font-size: 0.9rem; color: #666;">{h2h['matches']} meetings</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {COLORS['win'] if h2h['losses'] > h2h['wins'] else COLORS['loss']};">
                            <div class="stat-value">{h2h['losses']}</div>
                            <div class="stat-label">Wins vs {tape_player1}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"{tape_player1} and {tape_player2} have never faced each other directly.")

                # Win % by Format
                st.markdown("<h4 class='section-header'>Win % by Format</h4>", unsafe_allow_html=True)
                p1_formats = get_stats_by_format(df, tape_player1)
                p2_formats = get_stats_by_format(df, tape_player2)

                format_data = []
                for fmt in ['Singles', 'Doubles', 'FTAS']:
                    p1_f = p1_formats.get(fmt, {})
                    p2_f = p2_formats.get(fmt, {})
                    format_data.append({
                        'Format': fmt,
                        f'{tape_player1}': f"{p1_f.get('wins', 0)}-{p1_f.get('losses', 0)}-{p1_f.get('ties', 0)} ({p1_f.get('win_pct', 0):.1%})" if p1_f.get('matches', 0) > 0 else "N/A",
                        f'{tape_player2}': f"{p2_f.get('wins', 0)}-{p2_f.get('losses', 0)}-{p2_f.get('ties', 0)} ({p2_f.get('win_pct', 0):.1%})" if p2_f.get('matches', 0) > 0 else "N/A",
                        'Edge': tape_player1 if p1_f.get('win_pct', 0) > p2_f.get('win_pct', 0) else (tape_player2 if p2_f.get('win_pct', 0) > p1_f.get('win_pct', 0) else 'Even')
                    })

                st.dataframe(pd.DataFrame(format_data), hide_index=True, use_container_width=True)

                # Best/Worst Courses
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"<h4 class='section-header'>{tape_player1}'s Courses</h4>", unsafe_allow_html=True)
                    p1_best, p1_worst = get_best_worst_courses(df, tape_player1)
                    if p1_best:
                        st.markdown("**Best Courses:**")
                        for c in p1_best:
                            st.markdown(f"- {c['course']}: {c['wins']}-{c['losses']}-{c['ties']} ({c['win_pct']:.1%})")
                    if p1_worst and p1_worst != p1_best:
                        st.markdown("**Worst Courses:**")
                        for c in p1_worst:
                            st.markdown(f"- {c['course']}: {c['wins']}-{c['losses']}-{c['ties']} ({c['win_pct']:.1%})")

                with col2:
                    st.markdown(f"<h4 class='section-header'>{tape_player2}'s Courses</h4>", unsafe_allow_html=True)
                    p2_best, p2_worst = get_best_worst_courses(df, tape_player2)
                    if p2_best:
                        st.markdown("**Best Courses:**")
                        for c in p2_best:
                            st.markdown(f"- {c['course']}: {c['wins']}-{c['losses']}-{c['ties']} ({c['win_pct']:.1%})")
                    if p2_worst and p2_worst != p2_best:
                        st.markdown("**Worst Courses:**")
                        for c in p2_worst:
                            st.markdown(f"- {c['course']}: {c['wins']}-{c['losses']}-{c['ties']} ({c['win_pct']:.1%})")

                # Best Partners
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"<h4 class='section-header'>{tape_player1}'s Best Partners</h4>", unsafe_allow_html=True)
                    p1_partners = get_best_partners(df, tape_player1)
                    if p1_partners:
                        for p in p1_partners:
                            st.markdown(f"- {p['partner']}: {p['wins']}-{p['losses']}-{p['ties']} ({p['win_pct']:.1%})")
                    else:
                        st.info("No doubles partner data.")

                with col2:
                    st.markdown(f"<h4 class='section-header'>{tape_player2}'s Best Partners</h4>", unsafe_allow_html=True)
                    p2_partners = get_best_partners(df, tape_player2)
                    if p2_partners:
                        for p in p2_partners:
                            st.markdown(f"- {p['partner']}: {p['wins']}-{p['losses']}-{p['ties']} ({p['win_pct']:.1%})")
                    else:
                        st.info("No doubles partner data.")

        elif tape_player1 == tape_player2:
            st.warning("Please select two different players to compare.")

    with tab5:
        st.markdown("<h3 class='section-header'>🎯 Match Predictor</h3>", unsafe_allow_html=True)
        st.markdown("Predict match outcomes based on historical performance data.")

        # Match type selection
        match_type = st.radio("Match Type", ["Singles", "Doubles"], horizontal=True, key="pred_match_type")

        if match_type == "Singles":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>PLAYER 1</div>", unsafe_allow_html=True)
                pred_p1 = st.selectbox("Select Player 1", options=all_players, index=0, key="pred_singles_p1")
            with col2:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>PLAYER 2</div>", unsafe_allow_html=True)
                default_idx = 1 if len(all_players) > 1 else 0
                pred_p2 = st.selectbox("Select Player 2", options=all_players, index=default_idx, key="pred_singles_p2")

            # Optional course selection
            all_courses = sorted([c for c in df['Course'].dropna().unique() if isinstance(c, str)])
            pred_course = st.selectbox("Course (optional)", options=["Any Course"] + all_courses, key="pred_course_singles")
            pred_course = None if pred_course == "Any Course" else pred_course

            if pred_p1 != pred_p2:
                if st.button("Predict Match", type="primary", key="predict_singles"):
                    prediction = predict_match(df, pred_p1, pred_p2, course=pred_course)

                    st.markdown("---")
                    st.markdown("<h4 class='section-header'>Prediction</h4>", unsafe_allow_html=True)

                    # Visual probability bar
                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        color1 = COLORS['win'] if prediction['p1_prob'] > 50 else COLORS['primary']
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {color1};">
                            <div class="stat-value" style="color: {color1};">{prediction['p1_prob']:.0f}%</div>
                            <div class="stat-label">{pred_p1}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown("""
                        <div style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 1.5rem;">⚡</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        color2 = COLORS['win'] if prediction['p2_prob'] > 50 else COLORS['loss']
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {color2};">
                            <div class="stat-value" style="color: {color2};">{prediction['p2_prob']:.0f}%</div>
                            <div class="stat-label">{pred_p2}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability bar
                    st.markdown(f"""
                    <div style="background: linear-gradient(to right, {COLORS['primary']} {prediction['p1_prob']:.0f}%, {COLORS['loss']} {prediction['p1_prob']:.0f}%);
                                height: 30px; border-radius: 15px; margin: 1rem 0;">
                    </div>
                    """, unsafe_allow_html=True)

                    # Factors breakdown
                    st.markdown("<h4 class='section-header'>Analysis Factors</h4>", unsafe_allow_html=True)

                    if prediction['factors']:
                        factors_df = pd.DataFrame(prediction['factors'])
                        factors_df = factors_df.rename(columns={
                            'factor': 'Factor',
                            'p1_value': pred_p1,
                            'p2_value': pred_p2,
                            'edge': 'Edge',
                            'impact': 'Impact'
                        })
                        st.dataframe(factors_df[['Factor', pred_p1, pred_p2, 'Edge']], hide_index=True, use_container_width=True)
                    else:
                        st.info("Not enough historical data to analyze factors.")
            else:
                st.warning("Please select two different players.")

        else:  # Doubles
            st.markdown("**Team 1**")
            col1, col2 = st.columns(2)
            with col1:
                pred_d1_p1 = st.selectbox("Player 1A", options=all_players, index=0, key="pred_d1_p1")
            with col2:
                default_idx = 1 if len(all_players) > 1 else 0
                pred_d1_p2 = st.selectbox("Player 1B", options=all_players, index=default_idx, key="pred_d1_p2")

            st.markdown("**Team 2**")
            col1, col2 = st.columns(2)
            with col1:
                default_idx = 2 if len(all_players) > 2 else 0
                pred_d2_p1 = st.selectbox("Player 2A", options=all_players, index=default_idx, key="pred_d2_p1")
            with col2:
                default_idx = 3 if len(all_players) > 3 else 0
                pred_d2_p2 = st.selectbox("Player 2B", options=all_players, index=default_idx, key="pred_d2_p2")

            # Optional course selection
            all_courses = sorted([c for c in df['Course'].dropna().unique() if isinstance(c, str)])
            pred_course_d = st.selectbox("Course (optional)", options=["Any Course"] + all_courses, key="pred_course_doubles")
            pred_course_d = None if pred_course_d == "Any Course" else pred_course_d

            # Validate no duplicate players
            team1 = {pred_d1_p1, pred_d1_p2}
            team2 = {pred_d2_p1, pred_d2_p2}

            if len(team1) == 2 and len(team2) == 2 and not team1.intersection(team2):
                if st.button("Predict Match", type="primary", key="predict_doubles"):
                    # For doubles, we combine factors from both team members
                    prediction = predict_match(df, pred_d1_p1, pred_d2_p1, course=pred_course_d,
                                              is_doubles=True, partner1=pred_d1_p2, partner2=pred_d2_p2)

                    st.markdown("---")
                    st.markdown("<h4 class='section-header'>Prediction</h4>", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        color1 = COLORS['win'] if prediction['p1_prob'] > 50 else COLORS['primary']
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {color1};">
                            <div class="stat-value" style="color: {color1};">{prediction['p1_prob']:.0f}%</div>
                            <div class="stat-label">{pred_d1_p1} / {pred_d1_p2}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown("""
                        <div style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 1.5rem;">⚡</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        color2 = COLORS['win'] if prediction['p2_prob'] > 50 else COLORS['loss']
                        st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {color2};">
                            <div class="stat-value" style="color: {color2};">{prediction['p2_prob']:.0f}%</div>
                            <div class="stat-label">{pred_d2_p1} / {pred_d2_p2}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability bar
                    st.markdown(f"""
                    <div style="background: linear-gradient(to right, {COLORS['primary']} {prediction['p1_prob']:.0f}%, {COLORS['loss']} {prediction['p1_prob']:.0f}%);
                                height: 30px; border-radius: 15px; margin: 1rem 0;">
                    </div>
                    """, unsafe_allow_html=True)

                    # Factors breakdown
                    st.markdown("<h4 class='section-header'>Analysis Factors</h4>", unsafe_allow_html=True)

                    if prediction['factors']:
                        factors_df = pd.DataFrame(prediction['factors'])
                        factors_df = factors_df.rename(columns={
                            'factor': 'Factor',
                            'p1_value': f"{pred_d1_p1}/{pred_d1_p2}",
                            'p2_value': f"{pred_d2_p1}/{pred_d2_p2}",
                            'edge': 'Edge',
                            'impact': 'Impact'
                        })
                        st.dataframe(factors_df[['Factor', f"{pred_d1_p1}/{pred_d1_p2}", f"{pred_d2_p1}/{pred_d2_p2}", 'Edge']], hide_index=True, use_container_width=True)
                    else:
                        st.info("Not enough historical data to analyze factors.")
            else:
                st.warning("Please select 4 different players (no player can be on both teams or appear twice).")

    with tab6:
        st.markdown("<h3 class='section-header'>Ask Claude About FBC Data</h3>", unsafe_allow_html=True)

        st.markdown("""
        Ask any question about FBC tournament data - player stats, head-to-head records,
        course performance, historical trends, and more!
        """)

        # Initialize session state
        if 'claude_question' not in st.session_state:
            st.session_state.claude_question = ""
        if 'submit_question' not in st.session_state:
            st.session_state.submit_question = False
        if 'claude_response' not in st.session_state:
            st.session_state.claude_response = None
        if 'last_question' not in st.session_state:
            st.session_state.last_question = ""

        # Example questions
        st.markdown("**Try these example questions:**")

        example_questions = [
            "Who has won the most cups?",
            "Who has the best record against Connolly?",
            "What's Hilts' win percentage at Pebble Beach?",
            "Who is the best doubles partner for Hilts?",
            "Who had the most points at FBC 11?",
            "How many cups has Lynch won?"
        ]

        # Create columns for example question buttons
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.claude_question = question
                    st.session_state.submit_question = True
                    st.rerun()

        st.markdown("---")

        # Text input for custom questions
        user_question = st.text_input(
            "Your question:",
            value=st.session_state.claude_question,
            placeholder="e.g., Who has the best overall win percentage?",
            key="question_input"
        )

        # Sync text input back to session state
        if user_question != st.session_state.claude_question:
            st.session_state.claude_question = user_question

        # Submit button
        submit_clicked = st.button("Ask Claude", type="primary", use_container_width=True)

        # Determine if we should submit (either button clicked or auto-submit from example)
        should_submit = submit_clicked or st.session_state.submit_question

        # Reset the auto-submit flag
        if st.session_state.submit_question:
            st.session_state.submit_question = False

        # Get the question to use
        question_to_ask = st.session_state.claude_question

        if should_submit and question_to_ask.strip():
            # Check if API key is configured
            if "ANTHROPIC_API_KEY" not in st.secrets:
                st.error("Anthropic API key not configured. Please add ANTHROPIC_API_KEY to your Streamlit secrets.")
            else:
                with st.spinner("Claude is analyzing the FBC data..."):
                    try:
                        # Get response from Claude (data filtering happens inside)
                        response = ask_claude(question_to_ask, df, cups_df)
                        st.session_state.claude_response = response
                        st.session_state.last_question = question_to_ask

                    except anthropic.AuthenticationError:
                        st.error("Invalid API key. Please check your ANTHROPIC_API_KEY in Streamlit secrets.")
                        st.session_state.claude_response = None
                    except Exception as e:
                        st.error(f"Error getting response from Claude: {str(e)}")
                        st.session_state.claude_response = None
        elif should_submit:
            st.warning("Please enter a question.")

        # Display the response (persists across reruns)
        if st.session_state.claude_response and st.session_state.last_question:
            st.markdown("---")
            st.markdown(f"**Question:** {st.session_state.last_question}")
            st.markdown("**Claude's Answer:**")
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 10px;
                        border-left: 4px solid {COLORS['primary']}; margin-top: 1rem;">
                {st.session_state.claude_response}
            </div>
            """, unsafe_allow_html=True)

        # Info about API key setup
        with st.expander("How to set up your API key"):
            st.markdown("""
            To use the Ask Claude feature, you need to add your Anthropic API key to Streamlit secrets:

            **For local development:**
            1. Create a file `.streamlit/secrets.toml` in your project directory
            2. Add: `ANTHROPIC_API_KEY = "your-api-key-here"`

            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Click "Secrets" in the sidebar
            3. Add: `ANTHROPIC_API_KEY = "your-api-key-here"`

            Get your API key at: https://console.anthropic.com/
            """)

if __name__ == "__main__":
    main()
