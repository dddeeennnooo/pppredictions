import unittest

import numpy as np

from src.models.train_team_scoring import (
    _btts_rule_predictions,
    _select_supplemental_rule,
)


class TeamScoringRuleTests(unittest.TestCase):
    def test_gap_below_twenty_points_is_always_btts_no(self):
        predictions = _btts_rule_predictions(
            [0.80, 0.80, 0.95],
            [0.61, 0.60, 0.60],
        )

        self.assertEqual(predictions.tolist(), [0, 1, 1])

    def test_btts_yes_requires_both_team_probabilities_above_fifty_percent(self):
        predictions = _btts_rule_predictions(
            [0.50, 0.80, 0.80],
            [0.80, 0.50, 0.59],
        )

        self.assertEqual(predictions.tolist(), [0, 0, 1])

    def test_supplemental_rule_cannot_override_mandatory_gap_rule(self):
        selected = _select_supplemental_rule(
            np.array([1, 0, 1, 0]),
            np.array([0.80, 0.90, 0.75, 0.85]),
            np.array([0.65, 0.60, 0.55, 0.70]),
            np.array([False, False, False, False]),
        )
        predictions = _btts_rule_predictions(
            [0.80],
            [0.65],
            minimum_joint_probability=selected["minimum_joint_probability"],
            upper_gap_no_threshold=selected["upper_gap_no_threshold"],
        )

        self.assertEqual(predictions.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
