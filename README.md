# Serie A BTTS Predictor

A Python project for predicting football match outcomes, with a baseline logistic
regression model for whether both teams will score (BTTS).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Docker and Render deployment

Build and run the same container locally with Docker Compose:

```bash
docker compose up --build
```

Open <http://localhost:10000>. `docker compose down` stops the service while
keeping its named database volume for the next run.

For Render, commit `Dockerfile`, `.dockerignore`, `render.yaml`,
`football_predictor.db`, and the weekly prediction CSV, then push the branch to
your Git provider. In the Render dashboard, choose **New > Blueprint**, connect
the repository, and apply its `render.yaml`. The Blueprint builds the Docker
image, checks `/api/health`, and attaches a 1 GB persistent disk at
`/app/storage`. It uses Render's Frankfurt region and its smallest paid web
service size (`0.5c-512mb`).

The service uses `FOOTBALL_PREDICTOR_DB=/app/storage/football_predictor.db`.
On the first boot, `src.webapp.start` copies the image's
`football_predictor.db` seed to that disk. Later rounds, reviews, and
calibration data are written to the disk copy and survive deploys and restarts.
Existing disk data is never replaced automatically by a newer image seed.

Render persistent disks require a paid web-service plan and limit the service
to one instance. A free instance can run this image only with ephemeral SQLite
data, so user rounds and reviews would be lost after a restart or redeploy.

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

Train separate home/away scoring models and apply the scoring-probability-gap
rule. BTTS Yes requires a gap of at least 20 percentage points and both teams'
scoring probabilities to be strictly above 50%; optional joint-probability,
upper-gap, and team-ranking safeguards are selected on a separate
chronological validation slice before the final holdout is evaluated. Extra
safeguards remain disabled unless they improve validation accuracy by at least
one percentage point, limiting fragile threshold overfitting:

```powershell
python -m src.main train-team-scoring
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

The experimental branches and comparable holdout results are documented in
[`docs/btts_experiments.md`](docs/btts_experiments.md). The combined accuracy branch
also provides calibrated probabilities, a Dixon-Coles benchmark, optional no-quota
and opening-market-residual policies, a closing-market benchmark, and a block-
bootstrap uncertainty audit.

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

The deployment seed database is included in version control; other SQLite
databases, the virtual environment, and generated model artifacts are excluded.

## Predict an upcoming fixture

Build a new pre-match feature row from completed results strictly before the
fixture date, retrain odds-free models on that history, and predict 1/X/2, double
chance, BTTS, over/under 2.5, and each team to score:

```powershell
python -m src.main predict-upcoming "Udinese" "Como" 2026-08-22 --competition I1
```

Unlike the older `predict-match` historical test command, `predict-upcoming` does
not reuse an earlier head-to-head feature row and does not require bookmaker odds.
Matches on the prediction date and later are excluded from both feature generation
and model training. The command reports the most recent result date it used so the
data cutoff is visible in every prediction.

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
