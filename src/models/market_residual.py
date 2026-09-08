from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, poisson


EPSILON = 1e-6
_TOTAL_RATE_GRID = np.linspace(0.05, 10.0, 5000)
_OVER_25_GRID = 1 - poisson.cdf(2, _TOTAL_RATE_GRID)


def opening_market_btts_proxy(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    over_25_probability: float,
) -> float:
    """Fast independent-Poisson BTTS proxy from de-vigged opening markets."""
    values = np.asarray(
        [home_probability, draw_probability, away_probability, over_25_probability],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return np.nan
    total_rate = float(
        np.interp(
            np.clip(over_25_probability, _OVER_25_GRID[0], _OVER_25_GRID[-1]),
            _OVER_25_GRID,
            _TOTAL_RATE_GRID,
        )
    )
    decisive_total = home_probability + away_probability
    if decisive_total <= 0:
        return np.nan
    conditional_home = np.clip(home_probability / decisive_total, 0.01, 0.99)
    goal_difference = float(norm.ppf(conditional_home) * np.sqrt(total_rate))
    goal_difference = np.clip(goal_difference, -0.95 * total_rate, 0.95 * total_rate)
    home_rate = (total_rate + goal_difference) / 2
    away_rate = (total_rate - goal_difference) / 2
    return float((1 - np.exp(-home_rate)) * (1 - np.exp(-away_rate)))


def logit_market_blend(
    model_probabilities,
    market_probabilities,
    model_weight: float,
) -> np.ndarray:
    """Blend a statistical model with the market prior in log-odds space."""
    if not 0 <= model_weight <= 1:
        raise ValueError("model_weight must be between zero and one")
    model = np.clip(np.asarray(model_probabilities, dtype=float), EPSILON, 1 - EPSILON)
    market = np.clip(
        np.asarray(market_probabilities, dtype=float), EPSILON, 1 - EPSILON
    )
    missing = ~np.isfinite(market)
    market[missing] = model[missing]
    return expit(model_weight * logit(model) + (1 - model_weight) * logit(market))
