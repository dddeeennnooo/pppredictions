from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def rolling_season_splits(
    frame: pd.DataFrame,
    minimum_train_seasons: int = 4,
    test_seasons: int = 3,
) -> list[tuple[list[str], str]]:
    """Return chronological expanding-window train/test season splits."""
    starts = frame.assign(date=pd.to_datetime(frame["date"]))
    ordered = list(starts.groupby("season")["date"].min().sort_values().index)
    first_test = max(minimum_train_seasons, len(ordered) - test_seasons)
    return [(ordered[:index], ordered[index]) for index in range(first_test, len(ordered))]


def expected_calibration_error(targets, probabilities, bins: int = 10) -> float:
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
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


def block_bootstrap_accuracy_lift(
    frame: pd.DataFrame,
    baseline_class: int = 1,
    block_columns: tuple[str, ...] = ("season", "match_week"),
    resamples: int = 20_000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap whole match-week blocks to retain within-week dependence."""
    scored = frame.copy()
    scored["model_correct"] = (
        scored["predicted_btts"].astype(int) == scored["target_btts"].astype(int)
    ).astype(int)
    scored["baseline_correct"] = (
        scored["target_btts"].astype(int) == baseline_class
    ).astype(int)
    blocks = (
        scored.groupby(list(block_columns))
        .agg(
            difference=("model_correct", "sum"),
            baseline=("baseline_correct", "sum"),
            matches=("model_correct", "size"),
        )
        .reset_index(drop=True)
    )
    blocks["difference"] -= blocks["baseline"]
    values = blocks[["difference", "matches"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices]
    lifts = samples[:, :, 0].sum(axis=1) / samples[:, :, 1].sum(axis=1)
    observed = float(values[:, 0].sum() / values[:, 1].sum())
    lower, upper = np.quantile(lifts, [0.025, 0.975])
    return {"lift": observed, "lower_95": float(lower), "upper_95": float(upper)}


def audit_prediction_frame(
    frame: pd.DataFrame, baseline_class: int = 1
) -> dict[str, object]:
    y = frame["target_btts"].to_numpy(dtype=int)
    prediction = frame["predicted_btts"].to_numpy(dtype=int)
    probability = frame["btts_probability"].to_numpy(dtype=float)
    interval = block_bootstrap_accuracy_lift(frame, baseline_class=baseline_class)
    return {
        "matches": len(frame),
        "accuracy": float(accuracy_score(y, prediction)),
        "baseline_accuracy": float(accuracy_score(y, np.full(len(y), baseline_class))),
        "accuracy_lift": interval,
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)),
        "calibration_error": expected_calibration_error(y, probability),
        "by_competition": frame.assign(correct=prediction == y)
        .groupby("competition")
        .agg(matches=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .to_dict("records"),
    }


def cached_dataset(
    cache_path: Path,
    builder,
    refresh: bool = False,
) -> pd.DataFrame:
    """Cache deterministic feature frames for repeated model experiments."""
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    frame = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame
