from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import uuid
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import BASE_DIR, DATABASE_PATH


LEAGUE_NAMES = {
    "D1": "Bundesliga",
    "E0": "Premier League",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga",
}

REVIEW_FACTORS = {
    "form": "Recent form",
    "tactics": "Tactical matchup",
    "squad": "Squad and availability",
    "venue": "Home or away effect",
    "motivation": "Motivation and context",
    "intuition": "Football intuition",
}


class GameError(Exception):
    """A safe error that can be returned to the browser."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ModelPick:
    prediction: str
    probability: float | None
    threshold: float | None
    expected_home: float | None = None
    expected_away: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "threshold": self.threshold,
            "expected_home": self.expected_home,
            "expected_away": self.expected_away,
        }


class GameService:
    """Owns historical round selection, scoring, and human-edge learning."""

    ROUND_SIZE = 10

    def __init__(
        self,
        database_path: Path | str | None = None,
        predictions_path: Path | str | None = None,
    ) -> None:
        self.database_path = Path(database_path or DATABASE_PATH)
        self.predictions_path = Path(
            predictions_path
            or BASE_DIR
            / "artifacts"
            / "predictions"
            / "weekly_btts_predictions_2025_2026.csv"
        )
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.database_path}. Import match data first."
            )
        self._prediction_rows = self._load_prediction_rows()
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_tables(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS web_game_rounds (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    season TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fixture_ids_json TEXT NOT NULL,
                    model_picks_json TEXT NOT NULL,
                    human_picks_json TEXT NOT NULL DEFAULT '{}',
                    calibration_before_json TEXT NOT NULL,
                    calibration_after_json TEXT,
                    human_score INTEGER,
                    ai_score INTEGER,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS web_learning_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    match_id INTEGER NOT NULL,
                    competition TEXT NOT NULL,
                    human_prediction TEXT NOT NULL,
                    ai_prediction TEXT NOT NULL,
                    actual TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(mode, match_id)
                );

                CREATE INDEX IF NOT EXISTS ix_web_rounds_status
                    ON web_game_rounds(status);
                CREATE INDEX IF NOT EXISTS ix_web_learning_mode
                    ON web_learning_examples(mode);

                CREATE TABLE IF NOT EXISTS web_game_results (
                    round_id TEXT NOT NULL REFERENCES web_game_rounds(id),
                    match_id INTEGER NOT NULL REFERENCES matches(id),
                    human_prediction TEXT NOT NULL,
                    ai_prediction TEXT NOT NULL,
                    actual_prediction TEXT NOT NULL,
                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,
                    human_correct INTEGER NOT NULL CHECK(human_correct IN (0, 1)),
                    ai_correct INTEGER NOT NULL CHECK(ai_correct IN (0, 1)),
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (round_id, match_id)
                );

                CREATE TABLE IF NOT EXISTS web_game_reviews (
                    round_id TEXT NOT NULL,
                    match_id INTEGER NOT NULL,
                    human_factor TEXT NOT NULL,
                    review_text TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    PRIMARY KEY (round_id, match_id),
                    FOREIGN KEY (round_id, match_id)
                        REFERENCES web_game_results(round_id, match_id)
                        ON DELETE CASCADE
                );
                """
            )
            columns = {row['name'] for row in connection.execute('PRAGMA table_info(web_game_rounds)')}
            if 'competition' not in columns:
                connection.execute('ALTER TABLE web_game_rounds ADD COLUMN competition TEXT')
            learning_columns = {
                row['name']
                for row in connection.execute(
                    'PRAGMA table_info(web_learning_examples)'
                )
            }
            for column_name in ('human_factor', 'human_review', 'reviewed_at'):
                if column_name not in learning_columns:
                    connection.execute(
                        f'ALTER TABLE web_learning_examples ADD COLUMN {column_name} TEXT'
                    )
            # Preserve older games, including mixed-league rounds. Backfill their
            # results once so completed games also have individual result records.
            for game in connection.execute('SELECT * FROM web_game_rounds').fetchall():
                fixtures = self._matches_by_ids(connection, json.loads(game['fixture_ids_json']))
                competitions = {row['competition'] for row in fixtures.values()}
                if game['competition'] is None and len(competitions) == 1:
                    connection.execute('UPDATE web_game_rounds SET competition = ? WHERE id = ?',
                                       (next(iter(competitions)), game['id']))
                if game['status'] == 'complete':
                    self._store_results(connection, game, fixtures, game['completed_at'])

    def _load_prediction_rows(self) -> dict[int, dict[str, str]]:
        if not self.predictions_path.exists():
            return {}
        with self.predictions_path.open(encoding="utf-8", newline="") as handle:
            return {
                int(row["match_id"]): row
                for row in csv.DictReader(handle)
                if row.get("match_id")
            }

    def healthcheck(self) -> dict[str, str]:
        """Confirm that the live database is readable for platform probes."""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT 1 FROM matches LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("The match database is empty.")
        return {"status": "ok"}

    def metadata(self) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            completed = connection.execute(
                """
                SELECT COUNT(*) AS rounds,
                       COALESCE(SUM(human_score), 0) AS human,
                       COALESCE(SUM(ai_score), 0) AS ai,
                       COALESCE(SUM(CASE WHEN human_score > ai_score THEN 1 ELSE 0 END), 0) AS wins
                FROM web_game_rounds WHERE status = 'complete'
                """
            ).fetchone()
            available = self._eligible_matches(connection)
            recent_round_rows = connection.execute(
                """
                SELECT r.id, r.mode, r.season, r.competition,
                       r.human_score, r.ai_score, r.completed_at,
                       COUNT(v.match_id) AS reviews_completed
                FROM web_game_rounds r
                LEFT JOIN web_game_reviews v ON v.round_id = r.id
                WHERE r.status = 'complete'
                GROUP BY r.id
                ORDER BY r.completed_at DESC
                LIMIT 8
                """
            ).fetchall()
        seasons = sorted({row["season"] for row in available}, reverse=True)
        leagues = []
        for code, name in LEAGUE_NAMES.items():
            league_rows = [row for row in available if row['competition'] == code]
            league_seasons = sorted({row['season'] for row in league_rows}, reverse=True)
            league_seasons = [season for season in league_seasons
                              if sum(row['season'] == season for row in league_rows) >= self.ROUND_SIZE]
            if league_seasons:
                leagues.append({'code': code, 'name': name, 'seasons': league_seasons,
                                'matches_available': sum(row['season'] == league_seasons[0] for row in league_rows)})
        return {
            "round_size": self.ROUND_SIZE,
            "seasons": seasons,
            "leagues": leagues,
            "matches_available": len(available),
            "archive_matches": self._archive_match_count(),
            "modes": ["btts", "score"],
            "review_factors": [
                {"value": value, "label": label}
                for value, label in REVIEW_FACTORS.items()
            ],
            "recent_rounds": [
                {
                    "id": row["id"],
                    "mode": row["mode"],
                    "season": row["season"],
                    "competition": row["competition"],
                    "competition_name": LEAGUE_NAMES.get(
                        row["competition"], "Mixed leagues"
                    ),
                    "human_score": row["human_score"],
                    "ai_score": row["ai_score"],
                    "completed_at": row["completed_at"],
                    "reviews_completed": row["reviews_completed"],
                    "reviews_required": self.ROUND_SIZE,
                    "review_complete": row["reviews_completed"] == self.ROUND_SIZE,
                }
                for row in recent_round_rows
            ],
            "record": {
                "rounds": completed["rounds"],
                "wins": completed["wins"],
                "human_points": completed["human"],
                "ai_points": completed["ai"],
            },
            "calibration": {
                "btts": self._calibration_profile("btts"),
                "score": self._calibration_profile("score"),
            },
        }

    def _archive_match_count(self) -> int:
        with closing(self._connect()) as connection, connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM matches
                    WHERE full_time_home_goals IS NOT NULL
                      AND full_time_away_goals IS NOT NULL
                    """
                ).fetchone()[0]
            )

    def _eligible_matches(self, connection: sqlite3.Connection) -> list[sqlite3.Row]:
        if self._prediction_rows:
            placeholders = ",".join("?" for _ in self._prediction_rows)
            return connection.execute(
                f"""
                SELECT id, competition, season, match_week, date,
                       home_team, away_team, full_time_home_goals,
                       full_time_away_goals
                FROM matches
                WHERE id IN ({placeholders})
                  AND full_time_home_goals IS NOT NULL
                  AND full_time_away_goals IS NOT NULL
                ORDER BY date, competition, id
                """,
                tuple(self._prediction_rows),
            ).fetchall()

        season = connection.execute(
            """
            SELECT season
            FROM matches
            WHERE full_time_home_goals IS NOT NULL
              AND full_time_away_goals IS NOT NULL
              AND match_week IS NOT NULL
            GROUP BY season
            HAVING COUNT(*) >= ?
            ORDER BY season DESC LIMIT 1
            """,
            (self.ROUND_SIZE,),
        ).fetchone()
        if not season:
            return []
        return connection.execute(
            """
            SELECT id, competition, season, match_week, date,
                   home_team, away_team, full_time_home_goals,
                   full_time_away_goals
            FROM matches
            WHERE season = ? AND match_week IS NOT NULL
              AND full_time_home_goals IS NOT NULL
              AND full_time_away_goals IS NOT NULL
            ORDER BY date, competition, id
            """,
            (season["season"],),
        ).fetchall()

    def create_round(
        self,
        mode: str = "btts",
        season: str | None = None,
        seed: int | None = None,
        competition: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"btts", "score"}:
            raise GameError("Mode must be either 'btts' or 'score'.")
        if not isinstance(competition, str) or competition not in LEAGUE_NAMES:
            raise GameError('Choose a league before starting a round.')

        with closing(self._connect()) as connection, connection:
            eligible = self._eligible_matches(connection)
            eligible = [row for row in eligible if row['competition'] == competition]
            if season is None:
                seasons = sorted({row['season'] for row in eligible}, reverse=True)
                season = next((s for s in seasons if sum(row['season'] == s for row in eligible) >= self.ROUND_SIZE), None)
            eligible = [row for row in eligible if row['season'] == season]
            if len(eligible) < self.ROUND_SIZE:
                raise GameError("There are not enough completed fixtures in this league and season for a round.")

            rng = random.Random(seed) if seed is not None else random.SystemRandom()
            fixtures = rng.sample(eligible, self.ROUND_SIZE)
            fixtures.sort(key=lambda row: (row["date"], row["competition"], row["id"]))
            calibration = self._calibration_profile(mode, connection)
            model_picks = {
                str(row["id"]): self._model_pick(row, mode, calibration, connection).as_dict()
                for row in fixtures
            }
            round_id = uuid.uuid4().hex
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                INSERT INTO web_game_rounds (
                    id, mode, season, status, fixture_ids_json,
                    model_picks_json, human_picks_json,
                    calibration_before_json, created_at, competition
                ) VALUES (?, ?, ?, 'playing', ?, ?, '{}', ?, ?, ?)
                """,
                (
                    round_id,
                    mode,
                    fixtures[0]["season"],
                    json.dumps([row["id"] for row in fixtures]),
                    json.dumps(model_picks),
                    json.dumps(calibration),
                    now,
                    competition,
                ),
            )

        return self.get_round(round_id)

    def get_round(self, round_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            game = self._get_game_row(connection, round_id)
            fixture_ids = json.loads(game["fixture_ids_json"])
            fixtures = self._matches_by_ids(connection, fixture_ids)
            human_picks = json.loads(game["human_picks_json"])
            model_picks = json.loads(game["model_picks_json"])
            complete = game["status"] == "complete"
            results = {row['match_id']: row for row in connection.execute(
                'SELECT * FROM web_game_results WHERE round_id = ?', (round_id,)
            )} if complete else {}
            reviews = {
                row['match_id']: row
                for row in connection.execute(
                    'SELECT * FROM web_game_reviews WHERE round_id = ?',
                    (round_id,),
                )
            } if complete else {}

            serialized = []
            for position, match_id in enumerate(fixture_ids, start=1):
                match = fixtures[match_id]
                item = self._public_fixture(match, position, connection)
                item["human_prediction"] = human_picks.get(str(match_id))
                if complete:
                    saved = results[match_id]
                    item.update({
                        'actual_score': f"{saved['home_goals']}–{saved['away_goals']}",
                        'actual_prediction': saved['actual_prediction'],
                        'human_correct': bool(saved['human_correct']),
                        'ai_correct': bool(saved['ai_correct']),
                        'model': {'locked': False, **model_picks[str(match_id)]},
                        'human_review': (
                            {
                                'factor': reviews[match_id]['human_factor'],
                                'text': reviews[match_id]['review_text'],
                                'submitted_at': reviews[match_id]['submitted_at'],
                            }
                            if match_id in reviews
                            else None
                        ),
                    })
                else:
                    item["model"] = {"locked": True}
                serialized.append(item)

            response: dict[str, Any] = {
                "id": game["id"],
                "mode": game["mode"],
                "season": game["season"],
                "competition": game['competition'],
                "competition_name": LEAGUE_NAMES.get(game['competition'], 'Mixed leagues'),
                "status": game["status"],
                "round_size": self.ROUND_SIZE,
                "picks_made": len(human_picks),
                "fixtures": serialized,
                "calibration_before": json.loads(game["calibration_before_json"]),
            }
            if complete:
                human_score = int(game["human_score"])
                ai_score = int(game["ai_score"])
                response.update(
                    {
                        "human_score": human_score,
                        "ai_score": ai_score,
                        "outcome": (
                            "human" if human_score > ai_score else "ai" if ai_score > human_score else "draw"
                        ),
                        "completed_at": game["completed_at"],
                        "calibration_after": json.loads(game["calibration_after_json"]),
                        "reviews_completed": len(reviews),
                        "reviews_required": len(fixture_ids),
                        "review_complete": len(reviews) == len(fixture_ids),
                    }
                )
            return response

    def save_pick(self, round_id: str, match_id: int, prediction: Any) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            game = self._get_game_row(connection, round_id)
            if game["status"] != "playing":
                raise GameError("This round has already been completed.", 409)
            fixture_ids = json.loads(game["fixture_ids_json"])
            if match_id not in fixture_ids:
                raise GameError("That fixture does not belong to this round.")
            normalized = self._normalize_prediction(game["mode"], prediction)
            picks = json.loads(game["human_picks_json"])
            picks[str(match_id)] = normalized
            connection.execute(
                "UPDATE web_game_rounds SET human_picks_json = ? WHERE id = ?",
                (json.dumps(picks), round_id),
            )
        return {"saved": True, "match_id": match_id, "prediction": normalized, "picks_made": len(picks)}

    def complete_round(self, round_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            game = self._get_game_row(connection, round_id)
            if game["status"] == "complete":
                return self.get_round(round_id)
            fixture_ids = json.loads(game["fixture_ids_json"])
            human_picks = json.loads(game["human_picks_json"])
            if len(human_picks) != len(fixture_ids):
                raise GameError("Make a prediction for all ten fixtures before revealing the result.")
            fixtures = self._matches_by_ids(connection, fixture_ids)
            model_picks = json.loads(game["model_picks_json"])
            human_score = 0
            ai_score = 0
            for match_id in fixture_ids:
                match = fixtures[match_id]
                actual = self._actual_prediction(match, game["mode"])
                human = human_picks[str(match_id)]
                model = model_picks[str(match_id)]["prediction"]
                human_correct = human == actual
                ai_correct = model == actual
                human_score += int(human_correct)
                ai_score += int(ai_correct)
            calibration_after = self._calibration_profile(game["mode"], connection)
            completed_at = datetime.now(UTC).isoformat()
            self._store_results(connection, game, fixtures, completed_at)
            connection.execute(
                """
                UPDATE web_game_rounds
                SET status = 'complete', human_score = ?, ai_score = ?,
                    calibration_after_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    human_score,
                    ai_score,
                    json.dumps(calibration_after),
                    completed_at,
                    round_id,
                ),
            )
        return self.get_round(round_id)

    def save_review(
        self,
        round_id: str,
        match_id: int,
        review_text: Any,
        human_factor: Any,
    ) -> dict[str, Any]:
        """Save one qualitative review and learn after the full review is complete."""
        text_value = str(review_text).strip()
        factor_value = str(human_factor).strip().lower()
        if factor_value not in REVIEW_FACTORS:
            raise GameError("Choose the main human factor behind this review.")
        if len(text_value) < 10:
            raise GameError("Write at least 10 characters about this match.")
        if len(text_value) > 2000:
            raise GameError("Keep each match review under 2,000 characters.")

        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            game = self._get_game_row(connection, round_id)
            if game["status"] != "complete":
                raise GameError("Finish the round before writing match reviews.", 409)
            fixture_ids = json.loads(game["fixture_ids_json"])
            if match_id not in fixture_ids:
                raise GameError("That fixture does not belong to this round.")
            if not connection.execute(
                """
                SELECT 1 FROM web_game_results
                WHERE round_id = ? AND match_id = ?
                """,
                (round_id, match_id),
            ).fetchone():
                raise GameError("The saved match result could not be found.", 409)
            submitted_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                INSERT INTO web_game_reviews (
                    round_id, match_id, human_factor, review_text, submitted_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(round_id, match_id) DO UPDATE SET
                    human_factor = excluded.human_factor,
                    review_text = excluded.review_text,
                    submitted_at = excluded.submitted_at
                """,
                (round_id, match_id, factor_value, text_value, submitted_at),
            )
            reviews_completed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM web_game_reviews WHERE round_id = ?",
                    (round_id,),
                ).fetchone()[0]
            )
            review_complete = reviews_completed == len(fixture_ids)
            if review_complete:
                self._finalize_reviewed_learning(connection, game)
                calibration_after = self._calibration_profile(
                    game["mode"], connection
                )
                connection.execute(
                    """
                    UPDATE web_game_rounds
                    SET calibration_after_json = ? WHERE id = ?
                    """,
                    (json.dumps(calibration_after), round_id),
                )
            else:
                calibration_after = json.loads(game["calibration_after_json"])

        return {
            "saved": True,
            "match_id": match_id,
            "review": {
                "factor": factor_value,
                "text": text_value,
                "submitted_at": submitted_at,
            },
            "reviews_completed": reviews_completed,
            "reviews_required": len(fixture_ids),
            "review_complete": review_complete,
            "learning_applied": review_complete
            and int(game["human_score"]) > int(game["ai_score"]),
            "calibration_after": calibration_after,
        }

    def _finalize_reviewed_learning(
        self, connection: sqlite3.Connection, game: sqlite3.Row
    ) -> None:
        """Attach reviewed human wins to the adaptive model exactly once."""
        if int(game["human_score"]) <= int(game["ai_score"]):
            return
        rows = connection.execute(
            """
            SELECT r.match_id, r.human_prediction, r.ai_prediction,
                   r.actual_prediction, m.competition,
                   v.human_factor, v.review_text, v.submitted_at
            FROM web_game_results r
            JOIN web_game_reviews v
              ON v.round_id = r.round_id AND v.match_id = r.match_id
            JOIN matches m ON m.id = r.match_id
            WHERE r.round_id = ? AND r.human_correct = 1 AND r.ai_correct = 0
            """,
            (game["id"],),
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO web_learning_examples (
                mode, match_id, competition, human_prediction,
                ai_prediction, actual, created_at,
                human_factor, human_review, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mode, match_id) DO UPDATE SET
                human_factor = excluded.human_factor,
                human_review = excluded.human_review,
                reviewed_at = excluded.reviewed_at
            """,
            [
                (
                    game["mode"],
                    row["match_id"],
                    row["competition"],
                    row["human_prediction"],
                    row["ai_prediction"],
                    row["actual_prediction"],
                    row["submitted_at"],
                    row["human_factor"],
                    row["review_text"],
                    row["submitted_at"],
                )
                for row in rows
            ],
        )

    def _store_results(self, connection, game, fixtures, completed_at):
        """Snapshot all calls, including losses and ties, in the round transaction."""
        human_picks = json.loads(game['human_picks_json'])
        model_picks = json.loads(game['model_picks_json'])
        for match_id, match in fixtures.items():
            human = human_picks[str(match_id)]
            ai = model_picks[str(match_id)]['prediction']
            actual = self._actual_prediction(match, game['mode'])
            connection.execute(
                '''INSERT OR IGNORE INTO web_game_results (
                       round_id, match_id, human_prediction, ai_prediction,
                       actual_prediction, home_goals, away_goals,
                       human_correct, ai_correct, completed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (game['id'], match_id, human, ai, actual,
                 match['full_time_home_goals'], match['full_time_away_goals'],
                 int(human == actual), int(ai == actual), completed_at),
            )

    def _get_game_row(self, connection: sqlite3.Connection, round_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM web_game_rounds WHERE id = ?", (round_id,)
        ).fetchone()
        if not row:
            raise GameError("Round not found.", 404)
        return row

    def _matches_by_ids(
        self, connection: sqlite3.Connection, fixture_ids: list[int]
    ) -> dict[int, sqlite3.Row]:
        placeholders = ",".join("?" for _ in fixture_ids)
        rows = connection.execute(
            f"SELECT * FROM matches WHERE id IN ({placeholders})", fixture_ids
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _normalize_prediction(self, mode: str, prediction: Any) -> str:
        if mode == "btts":
            value = str(prediction).lower()
            if value not in {"yes", "no"}:
                raise GameError("BTTS prediction must be 'yes' or 'no'.")
            return value
        if not isinstance(prediction, dict):
            raise GameError("A score prediction needs home and away goal values.")
        try:
            home = int(prediction["home"])
            away = int(prediction["away"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GameError("Home and away goals must be whole numbers.") from exc
        if not (0 <= home <= 9 and 0 <= away <= 9):
            raise GameError("Goal predictions must be between 0 and 9.")
        return f"{home}-{away}"

    def _public_fixture(
        self, match: sqlite3.Row, position: int, connection: sqlite3.Connection
    ) -> dict[str, Any]:
        return {
            "id": match["id"],
            "position": position,
            "competition": match["competition"],
            "competition_name": LEAGUE_NAMES.get(match["competition"], match["competition"]),
            "season": match["season"],
            "match_week": match["match_week"],
            "date": match["date"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_code": self._team_code(match["home_team"]),
            "away_code": self._team_code(match["away_team"]),
            "intel": self._fixture_intel(match, connection),
            "standings": self._standings(match, connection),
        }

    def _standings(self, match: sqlite3.Row, connection: sqlite3.Connection) -> dict[str, Any]:
        # Season membership may be known in advance; results must precede the
        # fixture date. Include clubs that have yet to play with zero statistics.
        members = connection.execute(
            '''SELECT home_team AS team FROM matches WHERE competition = ? AND season = ?
               UNION SELECT away_team AS team FROM matches WHERE competition = ? AND season = ?''',
            (match['competition'], match['season'], match['competition'], match['season']),
        ).fetchall()
        table = {
            row['team']: dict(team=row['team'], played=0, won=0, drawn=0, lost=0,
                              goals_for=0, goals_against=0, goal_difference=0, points=0,
                              highlight='home' if row['team'] == match['home_team'] else
                                        'away' if row['team'] == match['away_team'] else None)
            for row in members
        }
        prior = connection.execute(
            '''SELECT home_team, away_team, full_time_home_goals, full_time_away_goals
               FROM matches WHERE competition = ? AND season = ? AND date < ?
                 AND full_time_home_goals IS NOT NULL AND full_time_away_goals IS NOT NULL''',
            (match['competition'], match['season'], match['date']),
        ).fetchall()
        for row in prior:
            for team, gf, ga in (
                (row['home_team'], row['full_time_home_goals'], row['full_time_away_goals']),
                (row['away_team'], row['full_time_away_goals'], row['full_time_home_goals']),
            ):
                entry = table[team]
                entry['played'] += 1
                entry['won'] += int(gf > ga)
                entry['drawn'] += int(gf == ga)
                entry['lost'] += int(gf < ga)
                entry['goals_for'] += gf
                entry['goals_against'] += ga
                entry['goal_difference'] += gf - ga
                entry['points'] += 3 if gf > ga else 1 if gf == ga else 0
        rows = sorted(table.values(), key=lambda row: (
            -row['points'], -row['goal_difference'], -row['goals_for'], row['team'].casefold()
        ))
        for position, row in enumerate(rows, 1):
            row['position'] = position
        return {'season': match['season'], 'before_date': match['date'], 'rows': rows}

    @staticmethod
    def _team_code(team: str) -> str:
        words = [part for part in team.replace("-", " ").split() if part]
        return ("".join(word[0] for word in words[:3]) if len(words) > 1 else team[:3]).upper()

    def _fixture_intel(
        self, match: sqlite3.Row, connection: sqlite3.Connection
    ) -> dict[str, Any]:
        return {
            "home": self._team_intel(
                match["home_team"], match["competition"], match["season"], match["date"], connection
            ),
            "away": self._team_intel(
                match["away_team"], match["competition"], match["season"], match["date"], connection
            ),
            "head_to_head": self._head_to_head(match, connection),
        }

    def _team_intel(
        self,
        team: str,
        competition: str,
        target_season: str,
        cutoff_date: str,
        connection: sqlite3.Connection,
    ) -> dict[str, Any]:
        start_year = int(target_season.split("/")[0])
        seasons = [f"{year}/{year + 1}" for year in range(start_year, start_year - 5, -1)]
        placeholders = ",".join("?" for _ in seasons)
        rows = connection.execute(
            f"""
            SELECT season, date, home_team, away_team,
                   full_time_home_goals, full_time_away_goals
            FROM matches
            WHERE competition = ? AND season IN ({placeholders})
              AND date < ? AND (home_team = ? OR away_team = ?)
              AND full_time_home_goals IS NOT NULL
              AND full_time_away_goals IS NOT NULL
            ORDER BY date DESC, id DESC
            """,
            (competition, *seasons, cutoff_date, team, team),
        ).fetchall()
        by_season: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_season[row["season"]].append(row)
        summaries = [
            self._summarize_team_rows(team, season, by_season.get(season, []))
            for season in seasons
        ]
        return {
            "team": team,
            "seasons": summaries,
            "recent": [self._recent_result(team, row) for row in rows[:5]],
        }

    @staticmethod
    def _summarize_team_rows(
        team: str, season: str, rows: list[sqlite3.Row]
    ) -> dict[str, Any]:
        wins = draws = losses = gf = ga = btts = 0
        for row in rows:
            home = row["home_team"] == team
            goals_for = row["full_time_home_goals"] if home else row["full_time_away_goals"]
            goals_against = row["full_time_away_goals"] if home else row["full_time_home_goals"]
            gf += goals_for
            ga += goals_against
            btts += int(goals_for > 0 and goals_against > 0)
            wins += int(goals_for > goals_against)
            draws += int(goals_for == goals_against)
            losses += int(goals_for < goals_against)
        played = len(rows)
        return {
            "season": season.replace("/", "–"),
            "played": played,
            "record": f"{wins}-{draws}-{losses}",
            "points_per_game": round((wins * 3 + draws) / played, 2) if played else None,
            "goals_for_per_game": round(gf / played, 2) if played else None,
            "goals_against_per_game": round(ga / played, 2) if played else None,
            "btts_rate": round(btts / played, 3) if played else None,
        }

    @staticmethod
    def _recent_result(team: str, row: sqlite3.Row) -> dict[str, Any]:
        home = row["home_team"] == team
        goals_for = row["full_time_home_goals"] if home else row["full_time_away_goals"]
        goals_against = row["full_time_away_goals"] if home else row["full_time_home_goals"]
        opponent = row["away_team"] if home else row["home_team"]
        result = "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"
        return {
            "result": result,
            "score": f"{goals_for}–{goals_against}",
            "opponent": opponent,
            "venue": "H" if home else "A",
        }

    def _head_to_head(
        self, match: sqlite3.Row, connection: sqlite3.Connection
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT date, home_team, away_team,
                   full_time_home_goals, full_time_away_goals
            FROM matches
            WHERE competition = ? AND date < ?
              AND ((home_team = ? AND away_team = ?)
                OR (home_team = ? AND away_team = ?))
              AND full_time_home_goals IS NOT NULL
              AND full_time_away_goals IS NOT NULL
            ORDER BY date DESC, id DESC LIMIT 5
            """,
            (
                match["competition"],
                match["date"],
                match["home_team"],
                match["away_team"],
                match["away_team"],
                match["home_team"],
            ),
        ).fetchall()
        return [
            {
                "date": row["date"],
                "home": row["home_team"],
                "away": row["away_team"],
                "score": f"{row['full_time_home_goals']}–{row['full_time_away_goals']}",
            }
            for row in rows
        ]

    def _model_pick(
        self,
        match: sqlite3.Row,
        mode: str,
        calibration: dict[str, Any],
        connection: sqlite3.Connection,
    ) -> ModelPick:
        expected_home, expected_away = self._expected_goals(match, connection)
        if mode == "score":
            home = self._poisson_mode(expected_home + calibration["home_goal_bias"])
            away = self._poisson_mode(expected_away + calibration["away_goal_bias"])
            return ModelPick(
                prediction=f"{home}-{away}",
                probability=None,
                threshold=None,
                expected_home=round(expected_home, 2),
                expected_away=round(expected_away, 2),
            )

        saved = self._prediction_rows.get(match["id"])
        if saved:
            probability = float(saved["btts_probability"])
            threshold = float(saved["decision_threshold"]) + calibration["threshold_shift"]
            prediction = "yes" if probability >= threshold else "no"
        else:
            probability = (1 - math.exp(-expected_home)) * (1 - math.exp(-expected_away))
            threshold = 0.5 + calibration["threshold_shift"]
            prediction = "yes" if probability >= threshold else "no"
        return ModelPick(
            prediction=prediction,
            probability=round(probability, 6),
            threshold=round(threshold, 6),
            expected_home=round(expected_home, 2),
            expected_away=round(expected_away, 2),
        )

    def _expected_goals(
        self, match: sqlite3.Row, connection: sqlite3.Connection
    ) -> tuple[float, float]:
        prior = connection.execute(
            """
            SELECT home_team, away_team, full_time_home_goals, full_time_away_goals
            FROM matches
            WHERE competition = ? AND date < ?
              AND full_time_home_goals IS NOT NULL
              AND full_time_away_goals IS NOT NULL
            ORDER BY date DESC, id DESC LIMIT 800
            """,
            (match["competition"], match["date"]),
        ).fetchall()
        if not prior:
            return 1.35, 1.1
        league_home = sum(row["full_time_home_goals"] for row in prior[:200]) / min(200, len(prior))
        league_away = sum(row["full_time_away_goals"] for row in prior[:200]) / min(200, len(prior))
        home_rows = [row for row in prior if row["home_team"] == match["home_team"]][:20]
        away_rows = [row for row in prior if row["away_team"] == match["away_team"]][:20]
        home_for = self._mean([row["full_time_home_goals"] for row in home_rows], league_home)
        home_against = self._mean([row["full_time_away_goals"] for row in home_rows], league_away)
        away_for = self._mean([row["full_time_away_goals"] for row in away_rows], league_away)
        away_against = self._mean([row["full_time_home_goals"] for row in away_rows], league_home)
        expected_home = 0.45 * home_for + 0.35 * away_against + 0.20 * league_home
        expected_away = 0.45 * away_for + 0.35 * home_against + 0.20 * league_away
        return max(0.15, min(4.5, expected_home)), max(0.15, min(4.5, expected_away))

    @staticmethod
    def _mean(values: list[int], fallback: float) -> float:
        return sum(values) / len(values) if values else fallback

    @staticmethod
    def _poisson_mode(expected: float) -> int:
        return max(0, min(9, int(math.floor(max(0.0, expected)))))

    @staticmethod
    def _actual_prediction(match: sqlite3.Row, mode: str) -> str:
        home = match["full_time_home_goals"]
        away = match["full_time_away_goals"]
        if mode == "btts":
            return "yes" if home > 0 and away > 0 else "no"
        return f"{home}-{away}"

    def _result_payload(
        self,
        match: sqlite3.Row,
        model: dict[str, Any],
        human_picks: dict[str, str],
    ) -> dict[str, Any]:
        mode = "btts" if model["probability"] is not None else "score"
        actual = self._actual_prediction(match, mode)
        human = human_picks[str(match["id"])]
        return {
            "actual_score": f"{match['full_time_home_goals']}–{match['full_time_away_goals']}",
            "actual_prediction": actual,
            "human_correct": human == actual,
            "ai_correct": model["prediction"] == actual,
            "model": {
                "locked": False,
                "prediction": model["prediction"],
                "probability": model["probability"],
                "threshold": model["threshold"],
                "expected_home": model["expected_home"],
                "expected_away": model["expected_away"],
            },
        }

    def _calibration_profile(
        self, mode: str, connection: sqlite3.Connection | None = None
    ) -> dict[str, Any]:
        owns_connection = connection is None
        if owns_connection:
            connection = self._connect()
        assert connection is not None
        rows = connection.execute(
            """
            SELECT actual, ai_prediction, human_factor
            FROM web_learning_examples WHERE mode = ?
            """,
            (mode,),
        ).fetchall()
        if owns_connection:
            connection.close()
        samples = len(rows)
        ramp = min(samples / 8, 1.0)
        if mode == "btts":
            yes = sum(row["actual"] == "yes" for row in rows)
            no = samples - yes
            direction = (no - yes) / samples if samples else 0.0
            return {
                "learning_examples": samples,
                "yes_edges": yes,
                "no_edges": no,
                "threshold_shift": round(max(-0.08, min(0.08, direction * 0.08 * ramp)), 4),
                "human_factors": self._factor_counts(rows),
            }

        home_residuals: list[int] = []
        away_residuals: list[int] = []
        for row in rows:
            actual_home, actual_away = (int(value) for value in row["actual"].split("-"))
            ai_home, ai_away = (int(value) for value in row["ai_prediction"].split("-"))
            home_residuals.append(actual_home - ai_home)
            away_residuals.append(actual_away - ai_away)
        home_bias = self._mean(home_residuals, 0.0) * 0.5 * ramp
        away_bias = self._mean(away_residuals, 0.0) * 0.5 * ramp
        return {
            "learning_examples": samples,
            "home_goal_bias": round(max(-0.75, min(0.75, home_bias)), 3),
            "away_goal_bias": round(max(-0.75, min(0.75, away_bias)), 3),
            "human_factors": self._factor_counts(rows),
        }

    @staticmethod
    def _factor_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
        counts = {factor: 0 for factor in REVIEW_FACTORS}
        for row in rows:
            factor = row["human_factor"]
            if factor in counts:
                counts[factor] += 1
        return counts
