from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score


def select_accuracy_threshold(
    targets,
    probabilities,
    thresholds=None,
) -> dict[str, float]:
    """Choose a binary threshold on validation data, resolving every match."""
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = (
        np.arange(0.35, 0.651, 0.01)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )
    scored = [
        (float(accuracy_score(y, p >= threshold)), float(threshold))
        for threshold in candidates
    ]
    accuracy, threshold = max(
        scored, key=lambda item: (item[0], -abs(item[1] - 0.5))
    )
    return {"threshold": threshold, "accuracy": accuracy}


def devig_two_way(yes_odds, no_odds) -> tuple[np.ndarray, np.ndarray]:
    """Convert two-way decimal odds to proportional fair probabilities."""
    yes = np.asarray(yes_odds, dtype=float)
    no = np.asarray(no_odds, dtype=float)
    raw_yes = 1 / yes
    raw_no = 1 / no
    total = raw_yes + raw_no
    return raw_yes / total, raw_no / total


def select_value_bets(
    probabilities,
    yes_odds,
    no_odds,
    minimum_edge: float = 0.03,
) -> np.ndarray:
    """Return 1 for Yes, 0 for No, and -1 when neither side has enough EV."""
    p_yes = np.asarray(probabilities, dtype=float)
    yes = np.asarray(yes_odds, dtype=float)
    no = np.asarray(no_odds, dtype=float)
    yes_edge = p_yes * yes - 1
    no_edge = (1 - p_yes) * no - 1
    selections = np.full(len(p_yes), -1, dtype=int)
    selections[(yes_edge >= minimum_edge) & (yes_edge >= no_edge)] = 1
    selections[(no_edge >= minimum_edge) & (no_edge > yes_edge)] = 0
    return selections
