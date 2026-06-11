# FBC Stats Dashboard

A Streamlit web app for the **Freddie B Cup** — displaying historical match stats, leaderboards, cup championship results, head-to-head comparisons, a match predictor, and an AI-powered Q&A interface.

---

## Running the App

### First-time setup

1. Make sure Python 3.10+ is installed.
2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

3. Set up your Anthropic API key (required for the Ask Claude tab):
   - Create a folder called `.streamlit` in the same directory as `app.py`
   - Inside it, create a file called `secrets.toml` with:

```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

Get a key at [console.anthropic.com](https://console.anthropic.com/).

### Running

```bash
cd "FBC Data & Apps"
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Updating the Data

All match data lives in **`FBC_Data.xlsx`**. The app reads this file every time it loads — no other steps needed after saving the spreadsheet.

### Archives sheet — match results

This is the main data sheet. Each row is one match (from one team's perspective).

**Key columns to fill in for new matches:**

| Column | What to enter | Notes |
|---|---|---|
| `FBC` | FBC event number (e.g. 13) | Integer |
| `Date` | Match date | Format: YYYY-MM-DD or MM/DD/YYYY |
| `Geographic Location` | City / region (e.g. "Scottsdale") | Used for event labels |
| `Course` | Full course name | Spell consistently — used for course stats |
| `Singles/Doubles` | `Singles`, `Doubles`, or `FTAS` | Case-sensitive |
| `Format` | Match format (e.g. `Match Play`, `Best Ball`) | |
| `Player 1` | First player on the team | |
| `Player 2` | Second player (Doubles/FTAS only) | Leave blank for Singles |
| `Singles Opponent` | Opponent name (Singles only) | Leave blank for Doubles/FTAS |
| `Opponent1` | First opponent (Doubles/FTAS) | Leave blank for Singles |
| `Opponent2` | Second opponent (Doubles/FTAS) | Leave blank for Singles |
| `W` | `1` if this team won, `0` otherwise | |
| `L` | `1` if this team lost, `0` otherwise | |
| `T` | `1` if this match tied, `0` otherwise | Exactly one of W/L/T must be 1 per row |
| `Points earned` | Points awarded: `1.0` win, `0.5` tie, `0.0` loss | Some formats award `2.0` for bonus wins |

**How doubles matches are entered:**

Each doubles match is entered as **two rows** — one for each team:

```
Row 1: Player1=Hilts, Player2=Lynch,  Opponent1=Grise, Opponent2=Connolly, W=1, L=0, T=0, Points=1.0
Row 2: Player1=Grise, Player2=Connolly, Opponent1=Hilts, Opponent2=Lynch,  W=0, L=1, T=0, Points=0.0
```

**How singles matches are entered:**

Each singles match is entered as **two rows**:

```
Row 1: Player1=Hilts, Singles Opponent=Grise, W=1, L=0, T=0, Points=1.0
Row 2: Player1=Grise, Singles Opponent=Hilts, W=0, L=1, T=0, Points=0.0
```

### Cups sheet — championship results

This sheet tracks which team won each FBC event. Update after each FBC:

- `1` = player was on the winning team
- `0` = player was on the losing team
- `X` = player did not participate

### Player name rules

Player names must be spelled **exactly the same** every time (canonical spellings: `Deoteris`,
`R. Connolly`). The app does **not** auto-correct spellings — a typo creates a new "player".
The Data Health check at the bottom of the app flags opponent names that don't match any
player in the event, which catches most typos.

If a new player joins, just use their name consistently and it will appear automatically in all stats and dropdowns.

### FTAS entry convention

The FTAS tiebreaker is entered as **one row per player** (so each player's individual record
reflects it): every player on the winning team gets `Points earned = 0.5`, losers get `0`.
The app knows to count the FTAS only **once** (0.5 to the winning team) when computing team
totals and margins — do not worry that the per-player rows appear to "overcount".

---

## File Structure

```
FBC Data & Apps/
├── app.py               # Main Streamlit app
├── FBC_Data.xlsx        # All match data (edit this to update stats)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .streamlit/
    └── secrets.toml     # API key (not committed to git)
```

---

## Tabs

| Tab | Description |
|---|---|
| Player Stats | Career stats, by-event breakdown, partner records, head-to-head, course performance |
| Leaderboard | Overall rankings sortable by points, win%, matches, events |
| Cups | Cup results by event (captains, team scores, margins, top scorer) and by player |
| Records | Win/unbeaten/losing streaks, active streaks, lopsided wins, perfect events, consecutive cups |
| Tale of the Tape | Side-by-side comparison of any two players |
| Match Predictor | Win probability based on historical stats |
| Ask Claude | Conversational Q&A powered by Claude AI — follow-up questions supported |

A **Data Health** check runs at the bottom of every page — it verifies match structure
(two sides per match, valid W/L/T flags, two teams per event, no opponent-name typos)
and surfaces any entry errors after new FBC data is added.
