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

def prepare_data_context(df):
    """Prepare a summary of the FBC data for Claude."""
    # Get all players
    all_players = sorted(set(df['Player 1'].dropna().unique()) |
                        set(df[df['Player 2'].notna()]['Player 2'].unique()))
    all_players = [p for p in all_players if isinstance(p, str)]

    # Get leaderboard
    leaderboard = get_leaderboard(df)

    # Get list of courses
    courses = df['Course'].dropna().unique().tolist()

    # Get list of FBC events
    events = df['FBC'].dropna().unique().tolist()
    events = sorted([int(e) for e in events if pd.notna(e)])

    # Build context string
    context = f"""FBC (Freddie B Cup) Golf Tournament Data Summary:

PLAYERS ({len(all_players)} total): {', '.join(all_players)}

LEADERBOARD (Top 15 by points):
"""
    for i, row in leaderboard.head(15).iterrows():
        context += f"  {row['Player']}: {row['Points']:.1f} pts, Record: {row['Record']}, Win%: {row['Win%']:.1%}, {row['Matches']} matches\n"

    context += f"\nCOURSES ({len(courses)} total): {', '.join(str(c) for c in courses[:30])}"
    if len(courses) > 30:
        context += f"... and {len(courses) - 30} more"

    context += f"\n\nFBC EVENTS: {', '.join(str(e) for e in events)}"

    context += "\n\nDETAILED MATCH DATA (recent 100 matches):\n"
    recent_matches = df.tail(100)
    for _, row in recent_matches.iterrows():
        p1 = row.get('Player 1', '')
        p2 = row.get('Player 2', '')
        opp1 = row.get('Opponent1', '')
        opp2 = row.get('Opponent2', '')
        course = row.get('Course', '')
        result = row.get('W/L/T', '')
        fbc = row.get('FBC', '')
        match_type = row.get('Singes/Doubles', '')

        if pd.notna(p2) and p2:
            players = f"{p1}/{p2}"
        else:
            players = str(p1)

        if pd.notna(opp2) and opp2:
            opponents = f"{opp1}/{opp2}"
        else:
            opponents = str(opp1) if pd.notna(opp1) else ''

        context += f"  FBC {int(fbc) if pd.notna(fbc) else '?'}: {players} vs {opponents} at {course} - {result} ({match_type})\n"

    return context

def ask_claude(question, data_context):
    """Send a question to Claude with FBC data context."""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    system_prompt = """You are an expert analyst for the FBC (Freddie B Cup), a golf match play tournament between friends.
You have access to historical match data and should answer questions about player statistics, head-to-head records,
course performance, and tournament history. Be concise but thorough. If you need to calculate statistics,
show your work briefly. Use the data provided to give accurate answers."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"""Here is the FBC tournament data:

{data_context}

Question: {question}"""
            }
        ]
    )

    return message.content[0].text

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

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Player Stats", "🏆 Leaderboard", "🤖 Ask Claude"])

    with tab1:
        # Player selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_player = st.selectbox(
                "Select a Player",
                options=all_players,
                index=0
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
        st.markdown("<h3 class='section-header'>Ask Claude About FBC Data</h3>", unsafe_allow_html=True)

        st.markdown("""
        Ask any question about FBC tournament data - player stats, head-to-head records,
        course performance, historical trends, and more!
        """)

        # Example questions
        st.markdown("**Try these example questions:**")

        example_questions = [
            "Who has the best record against Connolly?",
            "What's Hilts' win percentage at Pebble Beach?",
            "Who is the best doubles partner for Hilts?",
            "Which player has improved the most over time?",
            "What course does Gallagher perform best at?",
            "Who has the best record in singles matches?"
        ]

        # Create columns for example question buttons
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.claude_question = question

        st.markdown("---")

        # Initialize session state for the question if not exists
        if 'claude_question' not in st.session_state:
            st.session_state.claude_question = ""

        # Text input for custom questions
        user_question = st.text_input(
            "Your question:",
            value=st.session_state.claude_question,
            placeholder="e.g., Who has the best overall win percentage?",
            key="question_input"
        )

        # Update session state when text input changes
        if user_question != st.session_state.claude_question:
            st.session_state.claude_question = user_question

        # Submit button
        if st.button("Ask Claude", type="primary", use_container_width=True):
            if user_question.strip():
                # Check if API key is configured
                if "ANTHROPIC_API_KEY" not in st.secrets:
                    st.error("Anthropic API key not configured. Please add ANTHROPIC_API_KEY to your Streamlit secrets.")
                else:
                    with st.spinner("Claude is analyzing the FBC data..."):
                        try:
                            # Prepare data context
                            data_context = prepare_data_context(df)

                            # Get response from Claude
                            response = ask_claude(user_question, data_context)

                            # Display response
                            st.markdown("---")
                            st.markdown("**Claude's Answer:**")
                            st.markdown(f"""
                            <div style="background: white; padding: 1.5rem; border-radius: 10px;
                                        border-left: 4px solid {COLORS['primary']}; margin-top: 1rem;">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)

                        except anthropic.AuthenticationError:
                            st.error("Invalid API key. Please check your ANTHROPIC_API_KEY in Streamlit secrets.")
                        except Exception as e:
                            st.error(f"Error getting response from Claude: {str(e)}")
            else:
                st.warning("Please enter a question.")

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
