# Touchline web game

Touchline turns the historical football archive into a ten-match human-versus-model
challenge. The first version supports both BTTS and exact-score rounds.

## Run it

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.webapp.server
```

Open <http://127.0.0.1:8000>. To use another address or port:

```powershell
.\.venv\Scripts\python.exe -m src.webapp.server --host 0.0.0.0 --port 8080
```

No Node installation or additional Python package is required.

## Game rules

- Choose a league before starting each round. All ten random completed fixtures come
  from that league and one season (the latest eligible season for that league).
- The browser receives current-season stats up to the fixture date and the four prior
  seasons. It also receives the last five pre-match results and earlier head-to-heads.
- A full league table shows every club in the selected fixture's season, including
  clubs that have not played yet. It uses only completed matches strictly before the
  fixture date and highlights the home and away teams. Rankings use points, goal
  difference, goals scored, then alphabetical order for remaining ties. These are
  reconstructed standings, without official league-specific tiebreaks or deductions.
- Results, AI picks, and AI confidence stay server-side until all ten human picks are
  locked and the round is revealed.
- A correct prediction is worth one point. Exact-score mode has no partial credit.
- If the human wins the round, unique fixtures where the human was correct and the AI
  was wrong become learning examples. A fixture is never learned twice.

For BTTS, human-edge examples shift the model's decision threshold by at most eight
percentage points. For exact scores, they apply capped home/away goal corrections.
The before-and-after calibration is shown after a win. Game state and learning examples
are stored in the existing SQLite database. `web_game_rounds` stores the selected
league, locked AI picks, human picks, scores, and calibration snapshots.
`web_game_results` stores one row per fixture per completed round, including both
predictions, the actual score and outcome, correctness flags, and completion time.
All ten result rows and the round totals are saved in one transaction, including
losing and tied rounds. Repeated reveal requests do not duplicate results. Saved
scores survive later corrections to historical match data. `web_learning_examples`
continues to retain only unique human advantages from won rounds.

Existing databases are upgraded automatically at server startup. Older completed
rounds receive individual result records; older mixed-league rounds remain readable.
The API requires a `competition` code when creating a round, for example:

```json
{"mode": "btts", "competition": "I1"}
```

`GET /api/meta` lists eligible leagues and seasons. Restart the server after updating
the application to load the new routes and apply the additive database migration.

The BTTS opponent uses `weekly_btts_predictions_2025_2026.csv` when present. If that
artifact is unavailable, Touchline selects the latest complete matchweek season and
uses a pre-match expected-goals fallback. Exact-score calls always use the expected-goals
model, calculated only from matches before the challenged fixture.
