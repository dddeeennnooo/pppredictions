from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.importers.serie_a_importer import (
    _parse_date,
    load_raw_league_csv_files,
)


def devig(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    inverse = 1 / values
    return inverse / inverse.sum()


def direct_btts_market_probability(yes_odds: float, no_odds: float) -> float:
    return float(devig([yes_odds, no_odds])[0])


def implied_btts_from_closing_markets(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    over_25_odds: float,
    under_25_odds: float,
) -> float:
    """Infer independent-Poisson BTTS probability from closing 1X2 and O/U."""
    home_target, draw_target, away_target = devig(
        [home_odds, draw_odds, away_odds]
    )
    over_target = devig([over_25_odds, under_25_odds])[0]
    total_rate = brentq(
        lambda rate: 1 - poisson.cdf(2, rate) - over_target,
        0.05,
        10.0,
    )

    def outcome_error(home_share: float) -> float:
        home_rate = total_rate * home_share
        away_rate = total_rate * (1 - home_share)
        goals = np.arange(11)
        matrix = np.outer(
            poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate)
        )
        matrix /= matrix.sum()
        home = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away = float(np.triu(matrix, 1).sum())
        return (
            (home - home_target) ** 2
            + (draw - draw_target) ** 2
            + (away - away_target) ** 2
        )

    share = minimize_scalar(
        outcome_error, bounds=(0.03, 0.97), method="bounded"
    ).x
    home_rate = total_rate * share
    away_rate = total_rate * (1 - share)
    return float((1 - np.exp(-home_rate)) * (1 - np.exp(-away_rate)))


def _first_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame:
            result = result.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return result


def build_closing_market_benchmark() -> pd.DataFrame:
    frame = load_raw_league_csv_files().copy()
    frame["date"] = frame["Date"].map(_parse_date)
    frame["close_home"] = _first_available(frame, ["AvgCH", "PSCH", "B365CH"])
    frame["close_draw"] = _first_available(frame, ["AvgCD", "PSCD", "B365CD"])
    frame["close_away"] = _first_available(frame, ["AvgCA", "PSCA", "B365CA"])
    frame["close_over_25"] = _first_available(
        frame, ["AvgC>2.5", "PSC>2.5", "B365C>2.5"]
    )
    frame["close_under_25"] = _first_available(
        frame, ["AvgC<2.5", "PSC<2.5", "B365C<2.5"]
    )
    required = [
        "close_home",
        "close_draw",
        "close_away",
        "close_over_25",
        "close_under_25",
    ]
    frame = frame.dropna(subset=required + ["FTHG", "FTAG"]).copy()
    frame = frame[(frame[required] > 1).all(axis=1)]
    frame["target_btts"] = ((frame["FTHG"] > 0) & (frame["FTAG"] > 0)).astype(int)
    frame["market_btts_probability"] = frame.apply(
        lambda row: implied_btts_from_closing_markets(
            row.close_home,
            row.close_draw,
            row.close_away,
            row.close_over_25,
            row.close_under_25,
        ),
        axis=1,
    )
    return frame


def evaluate_closing_market_benchmark() -> dict:
    frame = build_closing_market_benchmark()
    season_starts = frame.groupby("season")["date"].min().sort_values()
    test_season = season_starts.index[-1]
    test = frame[frame["season"] == test_season]
    y = test["target_btts"].to_numpy(dtype=int)
    p = test["market_btts_probability"].to_numpy()
    prediction = (p >= 0.5).astype(int)
    return {
        "test_season": test_season,
        "rows": len(test),
        "accuracy": float(accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "by_competition": test.assign(correct=prediction == y)
        .groupby("competition")
        .agg(matches=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .to_dict("records"),
    }
