import unittest

import numpy as np
import pandas as pd

from src.models.train_dixon_coles_btts import DixonColesGoalModel


class DixonColesBttsTests(unittest.TestCase):
    def _matches(self):
        rows = []
        for index in range(40):
            rows.extend(
                [
                    {
                        "date": f"2024-{index // 4 + 1:02d}-{index % 4 + 1:02d}",
                        "home_team": "Attack",
                        "away_team": "Defence",
                        "home_goals": 2 + index % 2,
                        "away_goals": index % 2,
                    },
                    {
                        "date": f"2024-{index // 4 + 1:02d}-{index % 4 + 1:02d}",
                        "home_team": "Defence",
                        "away_team": "Attack",
                        "home_goals": 0,
                        "away_goals": 2,
                    },
                ]
            )
        return pd.DataFrame(rows)

    def test_score_matrix_is_a_probability_distribution(self):
        model = DixonColesGoalModel().fit(self._matches())
        matrix = model.score_matrix("Attack", "Defence")

        self.assertAlmostEqual(float(matrix.sum()), 1.0)
        self.assertTrue(np.all(matrix >= 0))

    def test_predictions_are_valid_for_seen_and_promoted_teams(self):
        model = DixonColesGoalModel().fit(self._matches())

        for fixture in (("Attack", "Defence"), ("Promoted", "Attack")):
            probability = model.predict_btts(*fixture)
            self.assertGreaterEqual(probability, 0)
            self.assertLessEqual(probability, 1)


if __name__ == "__main__":
    unittest.main()
