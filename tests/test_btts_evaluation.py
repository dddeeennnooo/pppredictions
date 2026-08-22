import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtesting.btts_evaluation import (
    audit_prediction_frame,
    cached_dataset,
    rolling_season_splits,
)


class BttsEvaluationTests(unittest.TestCase):
    def test_rolling_splits_never_train_on_test_or_future_season(self):
        frame = pd.DataFrame(
            {
                "season": ["A", "B", "C", "D", "E", "F"],
                "date": pd.date_range("2020-01-01", periods=6, freq="365D"),
            }
        )
        splits = rolling_season_splits(frame, minimum_train_seasons=3, test_seasons=2)

        self.assertEqual(splits, [(["A", "B", "C", "D"], "E"), (["A", "B", "C", "D", "E"], "F")])

    def test_audit_reports_block_bootstrap_interval(self):
        frame = pd.DataFrame(
            {
                "season": ["A"] * 8,
                "match_week": [1] * 4 + [2] * 4,
                "competition": ["I1"] * 8,
                "target_btts": [1, 0, 1, 0, 1, 1, 0, 0],
                "predicted_btts": [1, 0, 1, 0, 1, 0, 0, 0],
                "btts_probability": [0.7, 0.3, 0.6, 0.4, 0.6, 0.6, 0.3, 0.2],
            }
        )
        result = audit_prediction_frame(frame)

        self.assertEqual(result["matches"], 8)
        self.assertIn("lower_95", result["accuracy_lift"])
        self.assertIn("upper_95", result["accuracy_lift"])

    def test_cached_dataset_calls_builder_once(self):
        calls = []

        def builder():
            calls.append(1)
            return pd.DataFrame({"value": [1, 2]})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            cached_dataset(path, builder)
            cached_dataset(path, builder)

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
