import unittest

import numpy as np

from src.betting.btts_selection import (
    devig_two_way,
    select_accuracy_threshold,
    select_value_bets,
)


class BttsSelectionTests(unittest.TestCase):
    def test_accuracy_threshold_resolves_every_match_without_quota(self):
        targets = np.array([0, 0, 0, 1])
        probabilities = np.array([0.40, 0.45, 0.55, 0.60])
        result = select_accuracy_threshold(
            targets, probabilities, thresholds=[0.5, 0.58]
        )

        self.assertEqual(result["threshold"], 0.58)
        self.assertEqual(result["accuracy"], 1.0)

    def test_devigged_probabilities_sum_to_one(self):
        yes, no = devig_two_way([1.90, 2.10], [1.90, 1.80])

        self.assertTrue(np.allclose(yes + no, 1.0))

    def test_value_policy_can_pass_on_both_sides(self):
        selections = select_value_bets(
            probabilities=[0.50, 0.70, 0.30],
            yes_odds=[1.90, 2.00, 1.80],
            no_odds=[1.90, 1.80, 2.00],
            minimum_edge=0.03,
        )

        self.assertEqual(selections.tolist(), [-1, 1, 0])


if __name__ == "__main__":
    unittest.main()
