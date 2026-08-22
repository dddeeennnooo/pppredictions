import unittest

import numpy as np
from sklearn.metrics import log_loss

from src.models.probability_calibration import (
    PlattProbabilityCalibrator,
    expected_calibration_error,
)


class ProbabilityCalibrationTests(unittest.TestCase):
    def test_platt_calibration_reduces_deliberate_overconfidence(self):
        raw = np.array([0.01, 0.05, 0.10, 0.90, 0.95, 0.99] * 40)
        targets = np.array([0, 1, 0, 1, 0, 1] * 40)
        calibrated = PlattProbabilityCalibrator().fit(raw, targets).transform(raw)

        self.assertLess(log_loss(targets, calibrated), log_loss(targets, raw))
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))

    def test_expected_calibration_error_is_zero_for_balanced_exact_bins(self):
        targets = np.array([0, 1, 0, 1])
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])

        self.assertAlmostEqual(
            expected_calibration_error(targets, probabilities, bins=1), 0.0
        )

    def test_calibrator_preserves_probability_order(self):
        raw = np.linspace(0.05, 0.95, 100)
        targets = (raw > 0.55).astype(int)
        calibrated = PlattProbabilityCalibrator().fit(raw, targets).transform(raw)

        self.assertTrue(np.all(np.diff(calibrated) >= 0))


if __name__ == "__main__":
    unittest.main()
