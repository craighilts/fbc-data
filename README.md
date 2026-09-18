# FBC Stats Dashboard

A Streamlit web app for the **Freddie B Cup** — displaying historical match stats, leaderboards, cup championship results, head-to-head comparisons, a match predictor, and an AI-powered Q&A interface.

---

## Running the App

### First-time setup

1. Make sure Python 3.10+ is installed.
2. Create a virtual environment **outside Dropbox** and install dependencies. This
   folder syncs to Dropbox, and Dropbox's "online-only" space saving evicts thousands
   of package files from disk, which makes every launch hang for minutes while they
   re-download. Keep the real environment at `~/.venvs/fbc-stats` and leave a `venv`
   symlink in this folder so the `./venv/bin/python` commands in these docs keep working:

```bash
python3 -m venv ~/.venvs/fbc-stats
~/.venvs/fbc-stats/bin/python -m pip install -r requirements.txt
ln -s ~/.venvs/fbc-stats venv        # run from inside "FBC Data & Apps"
```

3. Set up your Anthropic API key (required for the Ask Claude tab):
   - Put it in `~/.streamlit/secrets.toml` (recommended — keeps the key out of the
     Dropbox-synced folder), or in a `.streamlit/secrets.toml` next to `app.py`:

```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

Get a key at [console.anthropic.com](https://console.anthropic.com/).

### Running

```bash
cd "FBC Data & Apps"
./venv/bin/python -m streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Updating the Data

All match data lives in **`FBC_Data.xlsx`**. The app reads this file every time it loads — no other steps needed after saving the spreadsheet.

### Importing a cup with Claude (the fast path)

After an event, the scoring app's Admin footer has an **Export CSV** button that writes
the event's rows in the exact Archives layout. Hand that file to Claude in a Claude Code
session on this repository with a prompt like:

> Here is the FBC 13 export from the scoring app. Run `tools/import_cup.py` on it,
> show me the Data Health result, then commit, push and open a pull request.

`tools/import_cup.py` appends the rows to the Archives sheet, adds the `FBC 13` column
to the Cups sheet (marking every player 1 / 0 / X, bumping Played, and extending the
Total / Win% / Lost formulas), adds a Cups row for any first-time player, and then runs
the app's Data Health check. It refuses to write the workbook if anything is flagged.
It edits only the parts of the file it has to, so every other tab, formula and chart is
untouched; Excel recalculates everything on the next open.

```bash
python tools/import_cup.py fbc13_archives_export.csv --dry-run   # report only
python tools/import_cup.py fbc13_archives_export.csv             # update FBC_Data.xlsx
```

Export only once the scoreboard shows every match reported: unreported matches are
simply left out, and nothing downstream can tell. Still by hand afterwards, and only
because they live in workbook tabs the app does not read: the Handicaps column, the
Ratings row, the Difficulty Graph row, and Captain Size on the Cups tab.

### Archives sheet — match results

This is the main data sheet. Each row is one match (from one team's perspective).

**Key columns to fill in for new matches:**

| Column | What to enter | Notes |
|---|---|---|
| `Match #` | Match number within the event (1, 2, 3 …) | Orders matches within an event for streak calculations |
| `FBC` | FBC event number (e.g. 13) | Integer |
| `UniqueMatchID` | `FBC13-D1`, `FBC13-D2` … for doubles; `FBC13-S-Cole-Shively` (both surnames, alphabetical) for singles; `FBC13-FT` for every FTAS row | **Required.** Pairs the two sides of a match and lets the app score the FTAS once per team instead of once per player |
| `Date` | Match date | Must be a real Excel date cell, not text |
| `Geographic Location` | City / region (e.g. "Scottsdale, AZ") | Used for event labels |
| `Course` | Full course name | Spell consistently — used for course stats |
| `Singles/Doubles` | `Singles`, `Doubles`, or `FTAS` | **Exactly** these spellings — `ftas` or `singles` is scored as an ordinary match |
| `Format` | Match format (e.g. `Match Play`, `Best Ball`) | |
| `Team` | The captain's surname (e.g. `Lynch`) | **Required.** Team totals and the cup winner are computed from it |
| `Captain Size` | `Over 6'`, `Under 6'` or `Mix` | Optional; shown on the Cups tab |
| `Player 1` | First player on the team | |
| `Player 2` | Second player (Doubles/FTAS only) | Leave blank for Singles |
| `Singles Opponent` | Opponent name (Singles only) | Leave blank for Doubles/FTAS |
| `Opponent1` | First opponent (Doubles/FTAS) | Leave blank for Singles |
| `Opponent2` | Second opponent (Doubles/FTAS) | Leave blank for Singles |
| `W` | `1` if this team won, `0` otherwise | |
| `L` | `1` if this team lost, `0` otherwise | |
| `T` | `1` if this match tied, `0` otherwise | Exactly one of W/L/T must be 1 per row |
| `Points earned` | Points awarded: `1.0` win, `0.5` tie, `0.0` loss | Some formats award `2.0` for bonus wins |

The numeric `W`, `L`, `T` columns are what the app counts records from (the letter `W/L/T`
column is for reading). `SingleEntity` is the only unused column. Do not type notes into the
`Player 1` column of a spare row — a row with a name but no `FBC` number becomes a phantom
player in every dropdown.

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

- Add a column headed **exactly** `FBC 13` (with the space). The app finds cup columns by that
  label and silently ignores anything else, and Data Health will flag an event with no column.
- `1` = player was on the winning team
- `0` = player was on the losing team
- `X` = player did not participate
- A first-time player gets a row **inside** the existing list, above the `Total` row. Rows below
  `Total` are not read.

### Player name rules

Player names must be spelled **exactly the same** every time (canonical spellings: `DeOteris`,
`R. Connolly`). The app does **not** auto-correct spellings — a typo creates a new "player".
The Data Health check at the bottom of the app flags opponent names that don't match any
player in the event, which catches most typos.

If a new player joins, just use their name consistently and it will appear automatically in all stats and dropdowns.

### FTAS entry convention

The FTAS tiebreaker is entered as **one row per player** (so each player's individual record
reflects it): every player on the winning team gets `Points earned = 0.5`, losers get `0`.
Every FTAS row carries `Singles/Doubles = FTAS` and `UniqueMatchID = FBC13-FT` (for FBC 13).
The app knows to count the FTAS only **once** (0.5 to the winning team) when computing team
totals and margins — do not worry that the per-player rows appear to "overcount". Not every
event has one (FBC 12 did not); just leave it out when it wasn't played.

---

## File Structure

```
FBC Data & Apps/
├── app.py               # Main Streamlit app
├── FBC_Data.xlsx        # All match data (edit this to update stats)
├── fbc13-scoring.jsx    # FBC 13 live scoring app (React source)
├── fbc13-worker.js      # Generated Cloudflare Worker — the deployed scoring app
├── fbc13-build/         # Build tooling for the worker (see FBC13-DEPLOY.md)
├── FBC13-DEPLOY.md      # How to deploy/update the scoring app
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── venv -> ~/.venvs/fbc-stats   # symlink; the real environment lives outside Dropbox
└── .streamlit/
    └── secrets.toml     # API key (optional here; ~/.streamlit/secrets.toml also works)
```

## FBC 13 Live Scoring App

`fbc13-scoring.jsx` is a mobile scoring app for the October 2–3, 2026 event
(PGA Frisco). It deploys as a Cloudflare Worker (`fbc13-worker.js`) with a D1
database for the shared scoreboard — **not** as a claude.ai artifact (public
artifacts have no shared storage and require sign-in; that's what sank the
Old Barnwell test run). See **FBC13-DEPLOY.md** for deploy, testing, and
update instructions. Its admin CSV export matches the Archives format below.

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

A **Data Health** check runs at the bottom of every page and surfaces entry errors after new
FBC data is added. It checks for: stray rows with a name but no FBC number; blank
`UniqueMatchID`, `Team`, `Date`, `Geographic Location` or `Course`; `Singles/Doubles` values
other than the three exact spellings; two sides per match; valid W/L/T flags and Points
earned; two teams per event; date typos; opponent-name typos; consistent FTAS rows; Cups-sheet
names that match the Archives spelling; and a Cups column for every event in Archives. Each
check runs independently, so one bad column cannot hide the other results.
