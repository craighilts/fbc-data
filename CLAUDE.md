# FBC Data — notes for Claude sessions

Streamlit app (`app.py`) over `FBC_Data.xlsx`, deployed from `main` at
freddiebcup.streamlit.app. The app reads only the **Archives** and **Cups** sheets;
the other 26 tabs are Excel formulas the owner maintains by hand.

## Adding a cup's results

Use `tools/import_cup.py` with the scoring app's "Export CSV" file. Do not paste
rows in by hand and do not write your own importer.

```bash
python tools/import_cup.py <export>.csv --dry-run   # report first
python tools/import_cup.py <export>.csv             # then write
```

It appends the Archives rows, adds the Cups column, runs the app's Data Health
check, and refuses to write if anything is flagged. Then commit, push, and open a
pull request; the owner merges. See README "Importing a cup with Claude".

## Editing the workbook

Never load-and-save `FBC_Data.xlsx` with openpyxl, pandas or similar: that drops
every cached formula value and can drop charts across the workbook. Edit the
specific XML parts inside the zip and copy everything else byte for byte
(`tools/import_cup.py` shows the pattern). After any workbook change, run the app
headless and confirm the Data Health check passes:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('/abs/path/app.py', default_timeout=300); at.run()
assert not at.exception and not at.warning   # warnings are Data Health issues
```

## Conventions the app depends on

- Player names are exact and case-sensitive; Brett is `Connolly`, Rick is `R. Connolly`.
- `Singles/Doubles` is exactly `Singles`, `Doubles` or `FTAS`.
- `UniqueMatchID`: `FBC13-D1`… (doubles), `FBC13-S-Cole-Shively` (singles, surnames
  alphabetical), `FBC13-FT` (every FTAS row). The FTAS is one row per player and is
  scored once per team by grouping on this ID.
- Team totals, cup winners and margins come from the `Team` column (captain surname)
  and the numeric `W`, `L`, `T` flags.
- Cups sheet columns are found by the exact header `FBC <n>`.
