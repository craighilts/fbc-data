[README.md](https://github.com/user-attachments/files/24968315/README.md)
# FBC Tracker

A command-line tool for managing and analyzing FBC (golf tournament) data.

## Quick Start

```bash
# Navigate to the project folder
cd fbc-tracker

# Run commands
python fbc.py stats              # Your career stats
python fbc.py stats Connolly     # Any player's stats
python fbc.py leaderboard        # Overall standings
python fbc.py fbc 11             # FBC 11 summary
python fbc.py partners           # Your doubles partner performance
python fbc.py record Hilts Grise # Head-to-head record
python fbc.py search Pinehurst   # Search by course/player/location
python fbc.py courses            # Your performance by course
python fbc.py add                # Add a new match (interactive)
```

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `stats [player]` | Career statistics | `python fbc.py stats Jackson` |
| `leaderboard` | Overall FBC standings | `python fbc.py leaderboard` |
| `fbc [num]` | Event summary | `python fbc.py fbc 12` |
| `partners [player]` | Doubles partner stats | `python fbc.py partners` |
| `record [p1] [p2]` | Head-to-head record | `python fbc.py record Hilts Lynch` |
| `search [term]` | Search matches | `python fbc.py search "Sea Island"` |
| `courses [player]` | Performance by course | `python fbc.py courses` |
| `add` | Add new match | `python fbc.py add` |
| `pending` | View pending matches | `python fbc.py pending` |

## Using Claude Code to Extend This

Once you have Claude Code installed, you can ask it to add new features conversationally:

```
> claude add a command to show my win% trend over time as an ASCII chart

> claude add the ability to export my stats to a PDF

> claude create a "rivals" command that shows my record against each player

> claude add filtering by date range to the search command
```

### Installing Claude Code

1. Open your terminal
2. Run: `npm install -g @anthropic-ai/claude-code`
3. Navigate to this folder: `cd fbc-tracker`
4. Start Claude Code: `claude`

Then you can have a conversation about modifying this code.

## File Structure

```
fbc-tracker/
├── fbc.py                      # Main CLI tool
├── FBCArchives_Aug25_v2.xlsx   # Your FBC data
├── pending_matches.csv         # New matches (created when you add)
└── README.md                   # This file
```

## Data Notes

- The tool reads from `FBCArchives_Aug25_v2.xlsx` (Archives sheet)
- New matches are saved to `pending_matches.csv` to preserve your original data
- Player names are case-sensitive (use: Hilts, not hilts)

## Requirements

- Python 3.8+
- pandas
- openpyxl

Install with: `pip install pandas openpyxl`
