# Serie A BTTS Predictor

A Python project for predicting football match outcomes, with a baseline logistic
regression model for whether both teams will score (BTTS).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

Initialize and populate the database, then build the pre-match features:

```powershell
python -m src.main init-db
python -m src.main import-serie-a
python -m src.main build-features
```

Train and evaluate the BTTS model using a chronological 80/20 split:

```powershell
python -m src.main train-btts
```

Train the advanced BTTS experiment with rolling attack/defense form,
BTTS-specific history, venue form, shots, over/under odds, model comparison,
and a chronological in-season holdout:

```powershell
python -m src.main train-advanced-btts
```

Run the week-safe walk-forward BTTS backtest:

```powershell
python -m src.main train-weekly-btts
```

This command selects a feature family and weekly probability-rank rule on the three
seasons before the test season. Week 1 uses only earlier seasons. After every later
week is predicted and scored, its completed results update the online models before
the next week's prediction. All fixtures in one week are feature-built as a batch,
preventing outcomes from that week from leaking into another fixture in the same
week. Match-level candidate probabilities, final predictions, and the per-week
accuracy summary are saved under `artifacts/predictions/`.

Run the equivalent calendar-month walk-forward backtest:

```powershell
python -m src.main train-monthly-btts
```

Every fixture in a month is predicted before any result from that month is learned.
After the month closes, its outcomes update the online models for the next month.
Feature-family and probability-rank rules are calibrated only on the three seasons
before the latest-season holdout. Monthly predictions and summaries are written to
separate `monthly_btts_*.csv` artifacts, leaving the weekly run unchanged.

Optional open BTTS modelling enrichment is available from the CC-BY-4.0 Global
Football Data Lake. Download and match its BTTS labels, xG, detailed match stats,
coaches, formations, and confirmed starter lists:

```powershell
python -m src.main import-open-btts
python -m src.main train-weekly-btts
```

The xG and match-stat values are post-match facts. They are never used for their own
fixture; only rolling values from completed prior matchweeks become model features.
Confirmed lineups, formations, and coaches are treated as pre-match inputs, so this
version represents a prediction made after official team news is available.

The local SQLite database, virtual environment, and generated model artifacts are
excluded from version control.

## Additional leagues

Download and import Serie A, Premier League, Bundesliga, La Liga, and Ligue 1:

```powershell
python -m src.main download-leagues --start-year 2019 --end-year 2025
python -m src.main import-leagues
```

The advanced model uses competition-specific rolling form and validation-selected
decision thresholds.

## Match weeks

Populate the nullable `matches.match_week` column from OpenFootball's public
matchday schedules:

```powershell
python -m src.main import-match-weeks
```

The importer supports `I1`, `E0`, `D1`, `SP1`, and `F1`, and safely matches
rescheduled fixtures by club pairing rather than assuming every ten consecutive
matches belong to one round. Filter or sort with, for example:

```sql
SELECT * FROM matches
WHERE competition = 'I1' AND season = '2025/2026' AND match_week = 12
ORDER BY date, id;
```

OpenFootball currently has no Italian JSON schedules before 2013/14, so older
Serie A rows retain a null `match_week`.

## Optional Sportmonks enrichment

The enrichment pipeline supports historical rolling xG, pre-match BTTS odds,
confirmed starting lineups, lineup continuity, sidelined-player counts, fixture
weather, managers, and player transfers. The advanced model also derives promoted
team and previous-season strength features directly from the match database.
Configure the token outside the repository:

```powershell
$env:SPORTMONKS_API_TOKEN="your-token"
python -m src.main import-sportmonks 2019-08-01 2026-05-31 --league-ids 384
python -m src.main import-sportmonks-transfers 2019-06-01 2026-05-31
python -m src.main enrichment-stats
python -m src.main train-advanced-btts
```

Sportmonks package and competition coverage determines which fields are returned.
BTTS odds use market ID 14 and require an additional API request per fixture.
Current-match xG is never used as a pre-match feature; it is added only to rolling
history for later fixtures. Manager changes and transfer counts are calculated only
from events dated before each match, so future information cannot leak into training.

Use `--no-fetch-btts-odds` on `import-sportmonks` if your package does not include
odds or if you want to avoid one extra odds request per fixture.
