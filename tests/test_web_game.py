import csv
import gc
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.webapp.game import GameError, GameService


class WebGameTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.database_path = root / "test.db"
        self.predictions_path = root / "predictions.csv"
        self._build_database()
        self._build_predictions()
        self.service = GameService(self.database_path, self.predictions_path)

    def tearDown(self):
        # sqlite Row/cursor objects can be finalized one GC cycle after a test on
        # Windows, where an open handle prevents TemporaryDirectory cleanup.
        del self.service
        gc.collect()
        self.tempdir.cleanup()

    def _build_database(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE matches (
                    id INTEGER PRIMARY KEY,
                    competition TEXT NOT NULL,
                    season TEXT NOT NULL,
                    match_week INTEGER,
                    date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    full_time_home_goals INTEGER,
                    full_time_away_goals INTEGER
                )
                """
            )
            match_id = 1
            for start_year in range(2021, 2025):
                for index in range(4):
                    connection.execute(
                        "INSERT INTO matches VALUES (?, 'E0', ?, ?, ?, ?, ?, 0, 0)",
                        (
                            match_id,
                            f"{start_year}/{start_year + 1}",
                            index + 1,
                            f"{start_year}-08-{index + 1:02d}",
                            f"Team {index}",
                            f"Team {(index + 1) % 4}",
                        ),
                    )
                    match_id += 1
            for index in range(10):
                connection.execute(
                    "INSERT INTO matches VALUES (?, 'E0', '2025/2026', ?, ?, ?, ?, 2, 1)",
                    (
                        100 + index,
                        index + 1,
                        f"2025-{8 + index // 4:02d}-{index + 1:02d}",
                        f"Team {index % 4}",
                        f"Team {(index + 1) % 4}",
                    ),
                )

    def _build_predictions(self):
        fieldnames = [
            "match_id",
            "btts_probability",
            "decision_threshold",
            "predicted_btts",
        ]
        with self.predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for match_id in range(100, 110):
                writer.writerow(
                    {
                        "match_id": match_id,
                        "btts_probability": 0.25,
                        "decision_threshold": 0.5,
                        "predicted_btts": 0,
                    }
                )

    def test_playing_round_hides_results_and_model_prediction(self):
        game = self.service.create_round("btts", competition="E0", seed=7)

        self.assertEqual(len(game["fixtures"]), 10)
        self.assertEqual(game["status"], "playing")
        self.assertNotIn("actual_score", game["fixtures"][0])
        self.assertEqual(game["fixtures"][0]["model"], {"locked": True})
        self.assertEqual(len(game["fixtures"][0]["intel"]["home"]["seasons"]), 5)

    def test_winning_round_is_scored_and_recalibrates_btts_threshold(self):
        game = self.service.create_round("btts", competition="E0", seed=3)
        for fixture in game["fixtures"]:
            self.service.save_pick(game["id"], fixture["id"], "yes")

        result = self.service.complete_round(game["id"])

        self.assertEqual(result["outcome"], "human")
        self.assertEqual(result["human_score"], 10)
        self.assertEqual(result["ai_score"], 0)
        self.assertEqual(result["calibration_after"]["learning_examples"], 10)
        self.assertEqual(result["calibration_after"]["threshold_shift"], -0.08)
        self.assertIn("actual_score", result["fixtures"][0])
        self.assertEqual(result["fixtures"][0]["model"]["prediction"], "no")

    def test_same_fixture_is_never_learned_twice(self):
        for seed in (1, 2):
            game = self.service.create_round("btts", competition="E0", seed=seed)
            for fixture in game["fixtures"]:
                self.service.save_pick(game["id"], fixture["id"], "yes")
            self.service.complete_round(game["id"])

        self.assertEqual(
            self.service.metadata()["calibration"]["btts"]["learning_examples"], 10
        )

    def test_cannot_reveal_an_incomplete_round(self):
        game = self.service.create_round("score", competition="E0", seed=5)
        self.service.save_pick(
            game["id"], game["fixtures"][0]["id"], {"home": 2, "away": 1}
        )

        with self.assertRaises(GameError):
            self.service.complete_round(game["id"])

    def test_exact_score_round_reveals_predictions_and_goal_calibration(self):
        game = self.service.create_round("score", competition="E0", seed=6)
        for fixture in game["fixtures"]:
            self.service.save_pick(
                game["id"], fixture["id"], {"home": 2, "away": 1}
            )

        result = self.service.complete_round(game["id"])

        self.assertEqual(result["human_score"], 10)
        self.assertEqual(result["outcome"], "human")
        self.assertEqual(result["fixtures"][0]["actual_prediction"], "2-1")
        self.assertIsNone(result["fixtures"][0]["model"]["probability"])
        self.assertGreater(result["calibration_after"]["learning_examples"], 0)
        self.assertIn("home_goal_bias", result["calibration_after"])

    def test_rejects_invalid_prediction(self):
        game = self.service.create_round("btts", competition="E0", seed=4)

        with self.assertRaises(GameError):
            self.service.save_pick(game["id"], game["fixtures"][0]["id"], "maybe")

    def test_league_selection_is_required_and_round_never_mixes_leagues(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute('''INSERT INTO matches
                SELECT id + 200, 'I1', season, match_week, date, home_team, away_team,
                       full_time_home_goals, full_time_away_goals FROM matches WHERE id >= 100''')
        for match_id in range(300, 310):
            self.service._prediction_rows[match_id] = dict(self.service._prediction_rows[100])
        meta = self.service.metadata()
        self.assertEqual({league['code'] for league in meta['leagues']}, {'E0', 'I1'})
        self.assertTrue(all(league['matches_available'] == 10 for league in meta['leagues']))
        for mode in ('btts', 'score'):
            game = self.service.create_round(mode, competition='I1')
            self.assertEqual(game['competition'], 'I1')
            self.assertEqual(game['competition_name'], 'Serie A')
            self.assertEqual({f['competition'] for f in game['fixtures']}, {'I1'})
            self.assertEqual(len(game['fixtures']), 10)
            with closing(sqlite3.connect(self.database_path)) as connection:
                self.assertEqual(connection.execute('SELECT competition FROM web_game_rounds WHERE id = ?',
                                                    (game['id'],)).fetchone()[0], 'I1')
        for code in (None, '', 'INVALID', 'D1'):
            with self.subTest(competition=code), self.assertRaises(GameError):
                self.service.create_round(competition=code)

    def test_standings_include_unplayed_teams_and_exclude_fixture_date_and_future(self):
        game = self.service.create_round(competition='E0')
        first = game['fixtures'][0]
        self.assertEqual(len(first['standings']['rows']), 4)
        self.assertTrue(all(row['played'] == 0 for row in first['standings']['rows']))
        self.assertEqual({row['highlight'] for row in first['standings']['rows'] if row['highlight']}, {'home', 'away'})
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            # Prior-season results must not affect this season's standings.
            connection.execute('UPDATE matches SET full_time_home_goals = 99 WHERE id < 100')
            # Same-day and future goals must not enter this fixture's table.
            connection.execute('UPDATE matches SET full_time_home_goals = 99 WHERE id >= 100')
        again = self.service.get_round(game['id'])['fixtures'][0]['standings']
        self.assertEqual(first['standings'], again)

    def test_standings_points_goal_difference_and_ranking(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.executemany('''INSERT INTO matches VALUES (?, 'E0', '2025/2026', 1,
                                   '2025-07-20', ?, ?, ?, ?)''', [
                (200, 'Team 0', 'Team 1', 3, 0),
                (201, 'Team 2', 'Team 3', 2, 0),
                (202, 'Team 0', 'Team 2', 1, 1),
            ])
        first = self.service.create_round(competition='E0')['fixtures'][0]
        rows = first['standings']['rows']
        self.assertEqual([r['team'] for r in rows], ['Team 0', 'Team 2', 'Team 3', 'Team 1'])
        self.assertEqual([r['points'] for r in rows], [4, 4, 0, 0])
        self.assertEqual([r['goal_difference'] for r in rows], [3, 2, -2, -3])
        self.assertEqual([r['position'] for r in rows], [1, 2, 3, 4])
        self.assertEqual((rows[0]['played'], rows[0]['won'], rows[0]['drawn'], rows[0]['lost']), (2, 1, 1, 0))
        self.assertEqual((rows[0]['goals_for'], rows[0]['goals_against']), (4, 1))

    def test_all_results_persist_for_wins_losses_and_ties_and_repeated_reveals(self):
        for probability, pick, outcome in ((.25, 'yes', 'human'), (.75, 'no', 'ai'), (.25, 'no', 'draw')):
            with self.subTest(outcome=outcome):
                for row in self.service._prediction_rows.values():
                    row['btts_probability'] = str(probability)
                game = self.service.create_round(competition='E0')
                for fixture in game['fixtures']:
                    self.service.save_pick(game['id'], fixture['id'], pick)
                result = self.service.complete_round(game['id'])
                self.assertEqual(result['outcome'], outcome)
                self.assertEqual(result, self.service.complete_round(game['id']))
                with closing(sqlite3.connect(self.database_path)) as connection:
                    saved = connection.execute('''SELECT human_prediction, ai_prediction, actual_prediction,
                        home_goals, away_goals, human_correct, ai_correct FROM web_game_results WHERE round_id = ?''',
                        (game['id'],)).fetchall()
                self.assertEqual(len(saved), 10)
                self.assertTrue(all(row[0] == pick and row[2:5] == ('yes', 2, 1) for row in saved))
                self.assertEqual(sum(row[5] for row in saved), result['human_score'])
                self.assertEqual(sum(row[6] for row in saved), result['ai_score'])
                restarted = GameService(self.database_path, self.predictions_path)
                self.assertEqual(restarted.get_round(game['id']), result)

    def test_result_snapshot_survives_historical_data_corrections(self):
        game = self.service.create_round('score', competition='E0')
        for fixture in game['fixtures']:
            self.service.save_pick(game['id'], fixture['id'], {'home': 2, 'away': 1})
        result = self.service.complete_round(game['id'])
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute('UPDATE matches SET full_time_home_goals = 0 WHERE id >= 100')
        restarted = GameService(self.database_path, self.predictions_path)
        loaded = restarted.get_round(game['id'])
        self.assertEqual(loaded['human_score'], result['human_score'])
        self.assertTrue(all(f['actual_prediction'] == '2-1' for f in loaded['fixtures']))
        self.assertTrue(all(f['human_correct'] for f in loaded['fixtures']))

    def test_old_completed_rounds_are_backfilled_without_duplicates(self):
        game = self.service.create_round(competition='E0')
        for fixture in game['fixtures']:
            self.service.save_pick(game['id'], fixture['id'], 'yes')
        self.service.complete_round(game['id'])
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute('DELETE FROM web_game_results')
            connection.execute('ALTER TABLE web_game_rounds DROP COLUMN competition')
        for _ in range(2):
            restarted = GameService(self.database_path, self.predictions_path)
            loaded = restarted.get_round(game['id'])
            self.assertEqual(loaded['competition'], 'E0')
            self.assertEqual(loaded['human_score'], 10)
            with closing(sqlite3.connect(self.database_path)) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM web_game_results').fetchone()[0], 10)

    def test_failed_result_write_rolls_back_entire_completion(self):
        game = self.service.create_round(competition='E0')
        for fixture in game['fixtures']:
            self.service.save_pick(game['id'], fixture['id'], 'yes')
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute('''CREATE TRIGGER simulate_disk_failure BEFORE INSERT ON web_game_results
                                  BEGIN SELECT RAISE(ABORT, 'test failure'); END''')
        with self.assertRaises(sqlite3.IntegrityError):
            self.service.complete_round(game['id'])
        self.assertEqual(self.service.get_round(game['id'])['status'], 'playing')
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM web_game_results').fetchone()[0], 0)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM web_learning_examples').fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
