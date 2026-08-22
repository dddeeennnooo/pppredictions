import unittest
from datetime import date

import pandas as pd

from src.models.predict_upcoming import (
    build_training_data,
    build_upcoming_features,
)


def _matches(rows):
    return pd.DataFrame(
        [
            {
                "id": index,
                "date": match_date,
                "home_team": home,
                "away_team": away,
                "full_time_home_goals": home_goals,
                "full_time_away_goals": away_goals,
            }
            for index, (
                match_date,
                home,
                away,
                home_goals,
                away_goals,
            ) in enumerate(rows, start=1)
        ]
    )


class UpcomingPredictionFeatureTests(unittest.TestCase):
    def test_cutoff_excludes_match_day_and_future_results(self):
        matches = _matches(
            [
                ("2026-08-01", "A", "B", 2, 0),
                ("2026-08-22", "A", "C", 0, 4),
                ("2026-08-30", "B", "A", 1, 1),
            ]
        )

        training, state = build_training_data(matches, date(2026, 8, 22))

        self.assertEqual(len(training), 1)
        self.assertEqual(len(state.team_matches["A"]), 1)
        self.assertEqual(state.team_matches["A"][0].goals_for, 2)

    def test_same_day_results_are_not_visible_to_each_other(self):
        matches = _matches(
            [
                ("2026-08-01", "A", "B", 2, 0),
                ("2026-08-01", "C", "A", 1, 3),
            ]
        )

        training, _ = build_training_data(matches, date(2026, 8, 2))

        self.assertEqual(training.iloc[0]["home_points_5"], 1.0)
        self.assertEqual(training.iloc[1]["away_points_5"], 1.0)

    def test_future_feature_uses_latest_completed_form(self):
        matches = _matches(
            [
                ("2026-08-01", "A", "B", 2, 0),
                ("2026-08-08", "C", "A", 1, 1),
            ]
        )
        _, state = build_training_data(matches, date(2026, 8, 22))

        features = build_upcoming_features(
            state, "A", "B", date(2026, 8, 22)
        )

        self.assertEqual(features["home_points_5"], 2.0)
        self.assertEqual(features["home_goals_for_5"], 1.5)
        self.assertEqual(features["away_goals_against_5"], 2.0)
        self.assertEqual(features["home_rest_days"], 14.0)


if __name__ == "__main__":
    unittest.main()
