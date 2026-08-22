from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


EPSILON = 1e-6


def _as_probabilities(values) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    return np.clip(probabilities, EPSILON, 1 - EPSILON)


class PlattProbabilityCalibrator:
    """Calibrate binary probabilities using a logistic model on their logits."""

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self.model = LogisticRegression(C=regularization, max_iter=2000)

    def fit(self, probabilities, targets) -> "PlattProbabilityCalibrator":
        values = _as_probabilities(probabilities)
        logits = np.log(values / (1 - values)).reshape(-1, 1)
        self.model.fit(logits, np.asarray(targets, dtype=int))
        return self

    def transform(self, probabilities) -> np.ndarray:
        values = _as_probabilities(probabilities)
        logits = np.log(values / (1 - values)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


def expected_calibration_error(
    targets, probabilities, bins: int = 10
) -> float:
    """Return equal-frequency expected calibration error."""
    y = np.asarray(targets, dtype=int).reshape(-1)
    p = _as_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError("targets and probabilities must have the same length")
    if len(y) == 0:
        raise ValueError("at least one observation is required")

    order = np.argsort(p)
    groups = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(group) * abs(float(y[group].mean()) - float(p[group].mean()))
            for group in groups
            if len(group)
        )
        / len(y)
    )
