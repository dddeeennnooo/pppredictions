from __future__ import annotations

from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sqlalchemy import text

from src.config import BASE_DIR
from src.database.connection import get_session


MODEL_PATH = BASE_DIR / "artifacts" / "models" / "dixon_coles_btts_model.pkl"
PREDICTION_PATH = (
    BASE_DIR / "artifacts" / "predictions" / "dixon_coles_btts_predictions.csv"
)


@dataclass
class DixonColesGoalModel:
    half_life_days: float = 365.0
    regularization: float = 0.08

    def fit(self, matches: pd.DataFrame, cutoff=None) -> "DixonColesGoalModel":
        frame = matches.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self.teams_ = sorted(set(frame["home_team"]) | set(frame["away_team"]))
        self.team_index_ = {team: index for index, team in enumerate(self.teams_)}
        n_teams = len(self.teams_)
        home_index = frame["home_team"].map(self.team_index_).to_numpy()
        away_index = frame["away_team"].map(self.team_index_).to_numpy()
        home_goals = frame["home_goals"].to_numpy(dtype=float)
        away_goals = frame["away_goals"].to_numpy(dtype=float)
        reference = pd.Timestamp(cutoff) if cutoff is not None else frame["date"].max()
        age = np.maximum((reference - frame["date"]).dt.days.to_numpy(), 0)
        weights = np.exp(-np.log(2) * age / self.half_life_days)

        def objective(parameters: np.ndarray) -> float:
            attack = parameters[:n_teams]
            defence = parameters[n_teams : 2 * n_teams]
            attack = attack - attack.mean()
            defence = defence - defence.mean()
            intercept, home_advantage, rho = parameters[-3:]
            home_rate = np.exp(
                np.clip(
                    intercept
                    + home_advantage
                    + attack[home_index]
                    - defence[away_index],
                    -3,
                    3,
                )
            )
            away_rate = np.exp(
                np.clip(
                    intercept + attack[away_index] - defence[home_index],
                    -3,
                    3,
                )
            )
            tau = np.ones(len(frame))
            zero_zero = (home_goals == 0) & (away_goals == 0)
            zero_one = (home_goals == 0) & (away_goals == 1)
            one_zero = (home_goals == 1) & (away_goals == 0)
            one_one = (home_goals == 1) & (away_goals == 1)
            tau[zero_zero] = 1 - home_rate[zero_zero] * away_rate[zero_zero] * rho
            tau[zero_one] = 1 + home_rate[zero_one] * rho
            tau[one_zero] = 1 + away_rate[one_zero] * rho
            tau[one_one] = 1 - rho
            if np.any(tau <= 0):
                return 1e12
            log_likelihood = (
                home_goals * np.log(home_rate)
                - home_rate
                - gammaln(home_goals + 1)
                + away_goals * np.log(away_rate)
                - away_rate
                - gammaln(away_goals + 1)
                + np.log(tau)
            )
            penalty = self.regularization * (
                np.sum(attack**2) + np.sum(defence**2)
            )
            return float(-(weights * log_likelihood).sum() + penalty)

        initial = np.zeros(2 * n_teams + 3)
        initial[-3] = np.log(max((home_goals.mean() + away_goals.mean()) / 2, 0.2))
        initial[-2] = 0.15
        initial[-1] = -0.05
        bounds = (
            [(-2.0, 2.0)] * (2 * n_teams)
            + [(-1.5, 1.5), (-0.7, 1.0), (-0.2, 0.2)]
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")
        parameters = result.x
        self.attack_ = parameters[:n_teams] - parameters[:n_teams].mean()
        defence = parameters[n_teams : 2 * n_teams]
        self.defence_ = defence - defence.mean()
        self.intercept_, self.home_advantage_, self.rho_ = parameters[-3:]
        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home = self.team_index_.get(home_team)
        away = self.team_index_.get(away_team)
        home_attack = self.attack_[home] if home is not None else 0.0
        home_defence = self.defence_[home] if home is not None else 0.0
        away_attack = self.attack_[away] if away is not None else 0.0
        away_defence = self.defence_[away] if away is not None else 0.0
        return (
            float(np.exp(self.intercept_ + self.home_advantage_ + home_attack - away_defence)),
            float(np.exp(self.intercept_ + away_attack - home_defence)),
        )

    def score_matrix(
        self, home_team: str, away_team: str, max_goals: int = 10
    ) -> np.ndarray:
        home_rate, away_rate = self.expected_goals(home_team, away_team)
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
        matrix[0, 0] *= 1 - home_rate * away_rate * self.rho_
        matrix[0, 1] *= 1 + home_rate * self.rho_
        matrix[1, 0] *= 1 + away_rate * self.rho_
        matrix[1, 1] *= 1 - self.rho_
        matrix = np.clip(matrix, 0, None)
        return matrix / matrix.sum()

    def predict_btts(self, home_team: str, away_team: str) -> float:
        return float(self.score_matrix(home_team, away_team)[1:, 1:].sum())


def _load_completed_matches() -> pd.DataFrame:
    session = get_session()
    try:
        return pd.read_sql(
            text(
                """
                SELECT competition, season, date, home_team, away_team,
                       full_time_home_goals AS home_goals,
                       full_time_away_goals AS away_goals
                FROM matches
                WHERE match_week IS NOT NULL
                  AND full_time_home_goals IS NOT NULL
                  AND full_time_away_goals IS NOT NULL
                ORDER BY date, id
                """
            ),
            session.bind,
            parse_dates=["date"],
        )
    finally:
        session.close()


def _fit_by_competition(
    history: pd.DataFrame, cutoff, half_life_days: float
) -> dict[str, DixonColesGoalModel]:
    models = {}
    for competition, league in history.groupby("competition"):
        models[competition] = DixonColesGoalModel(
            half_life_days=half_life_days
        ).fit(league, cutoff=cutoff)
    return models


def _predict_frame(
    models: dict[str, DixonColesGoalModel], fixtures: pd.DataFrame
) -> np.ndarray:
    return np.array(
        [
            models[row.competition].predict_btts(row.home_team, row.away_team)
            for row in fixtures.itertuples(index=False)
        ]
    )


def train_dixon_coles_btts_model() -> dict:
    matches = _load_completed_matches()
    matches["target_btts"] = (
        (matches["home_goals"] > 0) & (matches["away_goals"] > 0)
    ).astype(int)
    season_starts = matches.groupby("season")["date"].min().sort_values()
    seasons = list(season_starts.index)
    test_season = seasons[-1]
    validation_seasons = seasons[-4:-1]

    candidates = []
    for half_life in (180.0, 365.0, 730.0, 1460.0):
        probabilities = []
        targets = []
        for season in validation_seasons:
            fixtures = matches[matches["season"] == season]
            history = matches[matches["date"] < fixtures["date"].min()]
            models = _fit_by_competition(history, fixtures["date"].min(), half_life)
            probabilities.extend(_predict_frame(models, fixtures))
            targets.extend(fixtures["target_btts"].astype(int))
        probabilities = np.asarray(probabilities)
        targets = np.asarray(targets)
        for threshold in np.arange(0.40, 0.651, 0.01):
            candidates.append(
                {
                    "half_life_days": half_life,
                    "threshold": float(threshold),
                    "accuracy": float(accuracy_score(targets, probabilities >= threshold)),
                    "log_loss": float(log_loss(targets, probabilities)),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (row["accuracy"], -row["log_loss"], -abs(row["threshold"] - 0.5)),
    )

    test = matches[matches["season"] == test_season].copy()
    history = matches[matches["date"] < test["date"].min()]
    models = _fit_by_competition(
        history, test["date"].min(), selected["half_life_days"]
    )
    probabilities = _predict_frame(models, test)
    predictions = (probabilities >= selected["threshold"]).astype(int)
    targets = test["target_btts"].to_numpy(dtype=int)
    test["btts_probability"] = probabilities
    test["predicted_btts"] = predictions
    test["correct"] = (predictions == targets).astype(int)

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(PREDICTION_PATH, index=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump({"models": models, "selection": selected}, file)

    return {
        "test_season": test_season,
        "validation_seasons": validation_seasons,
        "test_rows": len(test),
        "half_life_days": selected["half_life_days"],
        "threshold": selected["threshold"],
        "validation_accuracy": selected["accuracy"],
        "test_accuracy": float(accuracy_score(targets, predictions)),
        "test_log_loss": float(log_loss(targets, probabilities)),
        "test_brier_score": float(brier_score_loss(targets, probabilities)),
        "baseline_accuracy": float(max(targets.mean(), 1 - targets.mean())),
        "predictions_path": str(PREDICTION_PATH),
        "model_path": str(MODEL_PATH),
    }
