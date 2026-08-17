import unittest

import numpy as np
import pandas as pd

from src.models.train_weekly_btts import (
    _adaptive_score,
    _build_weekly_features,
    _fast_weekly_model,
    _update_weekly_model,
    _weekly_rank_predictions,
    _weekly_metrics,
)


class WeeklyBttsFeatureTests(unittest.TestCase):
    def test_adaptive_score_uses_only_completed_week_scores(self):
        validation = {
            "minimum_week_accuracy": 0.4,
            "weeks_at_target": 10,
            "mean_week_accuracy": 0.55,
            "overall_accuracy": 0.56,
        }
        score = _adaptive_score([0.60, 0.40, 0.70], validation)

        self.assertEqual(score[0], 0.40)
        self.assertEqual(score[1], 2)
        self.assertAlmostEqual(score[2], 0.5666666667)

    def test_online_model_learns_only_after_explicit_week_update(self):
        X = pd.DataFrame({"form": [0.0, 0.2, 0.8, 1.0]})
        y = pd.Series([0, 0, 1, 1])
        model = _fast_weekly_model()
        model.fit(X, y)
        before = model.predict_proba(pd.DataFrame({"form": [0.9]}))[0, 1]

        _update_weekly_model(
            model,
            pd.DataFrame({"form": [0.9, 1.0]}),
            pd.Series([0, 0]),
        )
        after = model.predict_proba(pd.DataFrame({"form": [0.9]}))[0, 1]

        self.assertLess(after, before)

    def test_rank_predictions_do_not_use_targets(self):
        frame = pd.DataFrame(
            {
                "season": ["A"] * 4,
                "match_week": [1] * 4,
                "competition": ["I1"] * 4,
                "target_btts": [1, 0, 1, 0],
            }
        )
        predictions, thresholds = _weekly_rank_predictions(
            frame, np.array([0.1, 0.8, 0.4, 0.7]), 0.5
        )

        self.assertEqual(predictions.tolist(), [0, 1, 0, 1])
        self.assertTrue(np.allclose(thresholds, 0.7))

    def test_weekly_target_is_counted_per_week_not_overall(self):
        frame = pd.DataFrame(
            {
                "season": ["A"] * 10,
                "match_week": [1] * 5 + [2] * 5,
                "target_btts": [1] * 10,
            }
        )
        predictions = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
        metrics = _weekly_metrics(frame, predictions)

        self.assertEqual(metrics["weeks_at_target"], 1)
        self.assertEqual(metrics["weeks_total"], 2)

    def test_current_week_results_are_not_visible_to_same_week(self):
        rows = [
            {
                "id": 1,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 1,
                "date": "2025-08-20",
                "home_team": "A",
                "away_team": "B",
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
            },
            {
                "id": 2,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 1,
                "date": "2025-08-21",
                "home_team": "C",
                "away_team": "A",
                "full_time_home_goals": 2,
                "full_time_away_goals": 0,
            },
            {
                "id": 3,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 2,
                "date": "2025-08-27",
                "home_team": "A",
                "away_team": "C",
                "full_time_home_goals": 2,
                "full_time_away_goals": 1,
            },
        ]
        for row in rows:
            row.update(
                {
                    "home_shots": None,
                    "away_shots": None,
                    "home_shots_on_target": None,
                    "away_shots_on_target": None,
                    "odds_home_win": 2.0,
                    "odds_draw": 3.0,
                    "odds_away_win": 4.0,
                    "odds_over_25": 2.0,
                    "odds_under_25": 1.8,
                    "open_home_xg": 1.5,
                    "open_away_xg": 0.5,
                    "open_home_shots_inside_box": 8,
                    "open_away_shots_inside_box": 5,
                    "open_home_shots_outside_box": 4,
                    "open_away_shots_outside_box": 3,
                    "open_home_blocked_shots": 2,
                    "open_away_blocked_shots": 1,
                    "open_home_coach_name": "Coach Home",
                    "open_away_coach_name": "Coach Away",
                    "open_home_formation": "4-3-3",
                    "open_away_formation": "4-4-2",
                    "open_home_starter_ids": "[1,2,3,4,5,6,7,8,9,10,11]",
                    "open_away_starter_ids": "[1,2,3,4,5,6,7,8,9,10,12]",
                }
            )

        features = _build_weekly_features(pd.DataFrame(rows))
        week_one = features[features["match_week"] == 1]
        week_two = features[features["match_week"] == 2].iloc[0]

        self.assertEqual(week_one.iloc[0]["home_games_5"], 0)
        self.assertEqual(week_one.iloc[1]["away_games_5"], 0)
        self.assertEqual(week_two["home_games_5"], 2)
        self.assertEqual(week_two["away_games_5"], 1)
        self.assertTrue(pd.isna(week_one.iloc[0]["home_xg_for_5"]))
        self.assertEqual(week_two["home_xg_for_5"], 1.0)
        self.assertEqual(week_one.iloc[0]["home_lineup_known"], 1)
        self.assertTrue(pd.isna(week_one.iloc[0]["home_lineup_continuity"]))
        self.assertAlmostEqual(week_two["home_lineup_continuity"], 10 / 11)

    def test_current_month_results_are_not_visible_to_same_month(self):
        rows = [
            {
                "id": 1,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 1,
                "date": "2025-08-02",
                "home_team": "A",
                "away_team": "B",
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
            },
            {
                "id": 2,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 2,
                "date": "2025-08-24",
                "home_team": "C",
                "away_team": "A",
                "full_time_home_goals": 2,
                "full_time_away_goals": 0,
            },
            {
                "id": 3,
                "competition": "I1",
                "season": "2025/2026",
                "match_week": 3,
                "date": "2025-09-06",
                "home_team": "A",
                "away_team": "C",
                "full_time_home_goals": 2,
                "full_time_away_goals": 1,
            },
        ]
        defaults = {
            "home_shots": None,
            "away_shots": None,
            "home_shots_on_target": None,
            "away_shots_on_target": None,
            "odds_home_win": 2.0,
            "odds_draw": 3.0,
            "odds_away_win": 4.0,
            "odds_over_25": 2.0,
            "odds_under_25": 1.8,
        }
        for row in rows:
            row.update(defaults)

        features = _build_weekly_features(
            pd.DataFrame(rows), batch_by="month"
        )
        august = features[features["prediction_month"] == "2025-08"]
        september = features[
            features["prediction_month"] == "2025-09"
        ].iloc[0]

        self.assertEqual(august.iloc[0]["home_games_5"], 0)
        self.assertEqual(august.iloc[1]["away_games_5"], 0)
        self.assertEqual(september["home_games_5"], 2)
        self.assertEqual(september["away_games_5"], 1)


if __name__ == "__main__":
    unittest.main()
