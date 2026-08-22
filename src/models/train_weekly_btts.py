from __future__ import annotations

from collections import defaultdict
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.config import BASE_DIR
from src.database.connection import Base, engine, get_session
from src.models.probability_calibration import (
    PlattProbabilityCalibrator,
    expected_calibration_error,
)


MODEL_PATH = BASE_DIR / "artifacts" / "models" / "weekly_btts_model.pkl"
PREDICTION_DIR = BASE_DIR / "artifacts" / "predictions"
PREDICTION_PATH = PREDICTION_DIR / "weekly_btts_predictions.csv"
SUMMARY_PATH = PREDICTION_DIR / "weekly_btts_summary.csv"
WINDOWS = (5, 10, 20)
COMPETITIONS = ("I1", "E0", "D1", "SP1", "F1")

METADATA_COLUMNS = {
    "match_id",
    "competition",
    "season",
    "date",
    "match_week",
    "prediction_month",
    "home_team",
    "away_team",
    "target_btts",
}


def _mean(records: list[dict], key: str) -> float:
    values = [record[key] for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else np.nan


def _add_history_features(
    features: dict,
    prefix: str,
    history: list[dict],
    windows: tuple[int, ...] = WINDOWS,
) -> None:
    for window in windows:
        recent = history[-window:]
        features[f"{prefix}_games_{window}"] = min(len(history), window)
        for key in (
            "goals_for",
            "goals_against",
            "scored",
            "conceded",
            "btts",
            "clean_sheet",
            "failed_to_score",
            "total_goals",
            "over_25",
            "shots_for",
            "shots_against",
            "shots_on_target_for",
            "shots_on_target_against",
            "xg_for",
            "xg_against",
            "shots_inside_box_for",
            "shots_inside_box_against",
            "shots_outside_box_for",
            "shots_outside_box_against",
            "blocked_shots_for",
            "blocked_shots_against",
            "penalties_for",
            "penalties_against",
            "corners_for",
            "corners_against",
            "yellow_cards",
            "red_cards",
            "possession",
            "fouls_for",
            "fouls_against",
            "offsides_for",
            "offsides_against",
            "pass_accuracy",
        ):
            features[f"{prefix}_{key}_{window}"] = _mean(recent, key)


def _market_probabilities(row) -> tuple[float, float, float]:
    odds = (row.odds_home_win, row.odds_draw, row.odds_away_win)
    if any(pd.isna(value) or value <= 0 for value in odds):
        return np.nan, np.nan, np.nan
    raw = np.array([1 / value for value in odds])
    probabilities = raw / raw.sum()
    return tuple(float(value) for value in probabilities)


def _over_probability(row) -> float:
    if (
        pd.isna(row.odds_over_25)
        or pd.isna(row.odds_under_25)
        or row.odds_over_25 <= 0
        or row.odds_under_25 <= 0
    ):
        return np.nan
    raw_over = 1 / row.odds_over_25
    raw_under = 1 / row.odds_under_25
    return float(raw_over / (raw_over + raw_under))


def _match_record(row, home: bool) -> dict:
    if home:
        goals_for = int(row.full_time_home_goals)
        goals_against = int(row.full_time_away_goals)
        shots_for = row.home_shots
        shots_against = row.away_shots
        shots_on_target_for = row.home_shots_on_target
        shots_on_target_against = row.away_shots_on_target
        xg_for = getattr(row, "open_home_xg", np.nan)
        xg_against = getattr(row, "open_away_xg", np.nan)
        shots_inside_box_for = getattr(
            row, "open_home_shots_inside_box", np.nan
        )
        shots_inside_box_against = getattr(
            row, "open_away_shots_inside_box", np.nan
        )
        shots_outside_box_for = getattr(
            row, "open_home_shots_outside_box", np.nan
        )
        shots_outside_box_against = getattr(
            row, "open_away_shots_outside_box", np.nan
        )
        blocked_shots_for = getattr(row, "open_home_blocked_shots", np.nan)
        blocked_shots_against = getattr(row, "open_away_blocked_shots", np.nan)
    else:
        goals_for = int(row.full_time_away_goals)
        goals_against = int(row.full_time_home_goals)
        shots_for = row.away_shots
        shots_against = row.home_shots
        shots_on_target_for = row.away_shots_on_target
        shots_on_target_against = row.home_shots_on_target
        xg_for = getattr(row, "open_away_xg", np.nan)
        xg_against = getattr(row, "open_home_xg", np.nan)
        shots_inside_box_for = getattr(
            row, "open_away_shots_inside_box", np.nan
        )
        shots_inside_box_against = getattr(
            row, "open_home_shots_inside_box", np.nan
        )
        shots_outside_box_for = getattr(
            row, "open_away_shots_outside_box", np.nan
        )
        shots_outside_box_against = getattr(
            row, "open_home_shots_outside_box", np.nan
        )
        blocked_shots_for = getattr(row, "open_away_blocked_shots", np.nan)
        blocked_shots_against = getattr(row, "open_home_blocked_shots", np.nan)

    side = "home" if home else "away"
    other = "away" if home else "home"

    def open_value(team_side: str, metric: str):
        return getattr(row, f"open_{team_side}_{metric}", np.nan)

    def numeric(team_side: str, metric: str):
        value = open_value(team_side, metric)
        return None if pd.isna(value) else float(value)

    starter_value = open_value(side, "starter_ids")
    try:
        starter_ids = tuple(sorted(int(value) for value in json.loads(starter_value)))
    except (TypeError, ValueError, json.JSONDecodeError):
        starter_ids = ()

    total_goals = goals_for + goals_against
    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "scored": int(goals_for > 0),
        "conceded": int(goals_against > 0),
        "btts": int(goals_for > 0 and goals_against > 0),
        "clean_sheet": int(goals_against == 0),
        "failed_to_score": int(goals_for == 0),
        "total_goals": total_goals,
        "over_25": int(total_goals > 2),
        "shots_for": None if pd.isna(shots_for) else float(shots_for),
        "shots_against": None if pd.isna(shots_against) else float(shots_against),
        "shots_on_target_for": (
            None if pd.isna(shots_on_target_for) else float(shots_on_target_for)
        ),
        "shots_on_target_against": (
            None
            if pd.isna(shots_on_target_against)
            else float(shots_on_target_against)
        ),
        "xg_for": None if pd.isna(xg_for) else float(xg_for),
        "xg_against": None if pd.isna(xg_against) else float(xg_against),
        "shots_inside_box_for": (
            None if pd.isna(shots_inside_box_for) else float(shots_inside_box_for)
        ),
        "shots_inside_box_against": (
            None
            if pd.isna(shots_inside_box_against)
            else float(shots_inside_box_against)
        ),
        "shots_outside_box_for": (
            None if pd.isna(shots_outside_box_for) else float(shots_outside_box_for)
        ),
        "shots_outside_box_against": (
            None
            if pd.isna(shots_outside_box_against)
            else float(shots_outside_box_against)
        ),
        "blocked_shots_for": (
            None if pd.isna(blocked_shots_for) else float(blocked_shots_for)
        ),
        "blocked_shots_against": (
            None if pd.isna(blocked_shots_against) else float(blocked_shots_against)
        ),
        "penalties_for": numeric(side, "penalties"),
        "penalties_against": numeric(other, "penalties"),
        "corners_for": numeric(side, "corners"),
        "corners_against": numeric(other, "corners"),
        "yellow_cards": numeric(side, "yellow_cards"),
        "red_cards": numeric(side, "red_cards"),
        "possession": numeric(side, "possession"),
        "fouls_for": numeric(side, "fouls"),
        "fouls_against": numeric(other, "fouls"),
        "offsides_for": numeric(side, "offsides"),
        "offsides_against": numeric(other, "offsides"),
        "pass_accuracy": numeric(side, "pass_accuracy"),
        "coach": (
            None
            if pd.isna(open_value(side, "coach_name"))
            else str(open_value(side, "coach_name"))
        ),
        "formation": (
            None
            if pd.isna(open_value(side, "formation"))
            else str(open_value(side, "formation"))
        ),
        "starter_ids": starter_ids,
    }


def _add_current_team_context(
    features: dict,
    prefix: str,
    history: list[dict],
    coach_value,
    formation_value,
    starter_value,
) -> None:
    coach = None if pd.isna(coach_value) else str(coach_value)
    formation = None if pd.isna(formation_value) else str(formation_value)
    try:
        starters = set(int(value) for value in json.loads(starter_value))
    except (TypeError, ValueError, json.JSONDecodeError):
        starters = set()
    prior = history[-1] if history else None
    prior_coach = prior.get("coach") if prior else None
    prior_formation = prior.get("formation") if prior else None
    prior_starters = set(prior.get("starter_ids", ())) if prior else set()

    features[f"{prefix}_coach_known"] = int(coach is not None)
    features[f"{prefix}_coach_changed"] = (
        int(coach != prior_coach)
        if coach is not None and prior_coach is not None
        else np.nan
    )
    if coach is None:
        features[f"{prefix}_coach_tenure"] = np.nan
    else:
        tenure = 0
        for record in reversed(history):
            if record.get("coach") != coach:
                break
            tenure += 1
        features[f"{prefix}_coach_tenure"] = tenure

    features[f"{prefix}_formation_known"] = int(formation is not None)
    features[f"{prefix}_formation_changed"] = (
        int(formation != prior_formation)
        if formation is not None and prior_formation is not None
        else np.nan
    )
    common_formations = (
        "4-3-3",
        "4-2-3-1",
        "4-4-2",
        "3-5-2",
        "3-4-3",
        "3-4-2-1",
        "4-1-4-1",
    )
    for known_formation in common_formations:
        safe_name = known_formation.replace("-", "_")
        features[f"{prefix}_formation_{safe_name}"] = int(
            formation == known_formation
        )

    features[f"{prefix}_lineup_known"] = int(bool(starters))
    if starters and prior_starters:
        returning = len(starters & prior_starters)
        features[f"{prefix}_returning_starters"] = returning
        features[f"{prefix}_lineup_continuity"] = returning / len(starters)
        features[f"{prefix}_lineup_changes"] = len(starters - prior_starters)
    else:
        features[f"{prefix}_returning_starters"] = np.nan
        features[f"{prefix}_lineup_continuity"] = np.nan
        features[f"{prefix}_lineup_changes"] = np.nan


def _previous_season_features(matches: pd.DataFrame) -> dict[tuple, dict]:
    season_starts = matches.groupby(["competition", "season"])["date"].min()
    previous: dict[tuple, dict] = {}

    for competition in sorted(matches["competition"].unique()):
        seasons = sorted(
            matches.loc[matches["competition"] == competition, "season"].unique(),
            key=lambda season: season_starts[(competition, season)],
        )
        for index in range(1, len(seasons)):
            prior_season = seasons[index - 1]
            current_season = seasons[index]
            prior = matches[
                (matches["competition"] == competition)
                & (matches["season"] == prior_season)
            ]
            stats: dict[str, dict[str, float]] = defaultdict(
                lambda: {"games": 0, "points": 0, "gf": 0, "ga": 0}
            )
            for row in prior.itertuples(index=False):
                home = stats[row.home_team]
                away = stats[row.away_team]
                home["games"] += 1
                away["games"] += 1
                home["gf"] += row.full_time_home_goals
                home["ga"] += row.full_time_away_goals
                away["gf"] += row.full_time_away_goals
                away["ga"] += row.full_time_home_goals
                if row.full_time_home_goals > row.full_time_away_goals:
                    home["points"] += 3
                elif row.full_time_away_goals > row.full_time_home_goals:
                    away["points"] += 3
                else:
                    home["points"] += 1
                    away["points"] += 1

            ranked = sorted(
                stats,
                key=lambda team: (
                    stats[team]["points"],
                    stats[team]["gf"] - stats[team]["ga"],
                    stats[team]["gf"],
                ),
                reverse=True,
            )
            denominator = max(len(ranked) - 1, 1)
            previous[(competition, current_season)] = {
                "teams": set(ranked),
                "rank": {
                    team: rank / denominator for rank, team in enumerate(ranked)
                },
                "ppg": {
                    team: stats[team]["points"] / stats[team]["games"]
                    for team in ranked
                },
            }
    return previous


def _build_weekly_features(
    matches: pd.DataFrame, batch_by: str = "week"
) -> pd.DataFrame:
    """Build features in period batches so a match never sees its period."""
    if batch_by not in {"week", "month"}:
        raise ValueError("batch_by must be either 'week' or 'month'.")
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["prediction_month"] = matches["date"].dt.strftime("%Y-%m")
    previous_seasons = _previous_season_features(matches)
    team_history: dict[tuple, list[dict]] = defaultdict(list)
    home_history: dict[tuple, list[dict]] = defaultdict(list)
    away_history: dict[tuple, list[dict]] = defaultdict(list)
    head_to_head: dict[tuple, list[dict]] = defaultdict(list)
    league_history: dict[str, list[dict]] = defaultdict(list)
    season_history: dict[tuple, list[dict]] = defaultdict(list)
    elo: dict[tuple, float] = defaultdict(lambda: 1500.0)
    rows: list[dict] = []

    season_starts = matches.groupby(["competition", "season"])["date"].min()
    season_keys = sorted(season_starts.index, key=lambda key: season_starts[key])

    for competition, season in season_keys:
        season_matches = matches[
            (matches["competition"] == competition)
            & (matches["season"] == season)
        ]
        period_column = (
            "match_week" if batch_by == "week" else "prediction_month"
        )
        period_values = season_matches[period_column]
        if batch_by == "week":
            periods = sorted(period_values.astype(int).unique())
        else:
            periods = sorted(period_values.unique())
        for period in periods:
            batch = season_matches[
                season_matches[period_column] == period
            ].sort_values(["date", "id"])

            # Compute every row before adding any result from this period.
            for row in batch.itertuples(index=False):
                home_key = (competition, row.home_team)
                away_key = (competition, row.away_team)
                season_key = (competition, season)
                pair_key = (competition, *sorted((row.home_team, row.away_team)))
                p_home, p_draw, p_away = _market_probabilities(row)
                prior = previous_seasons.get(season_key)
                home_was_present = prior and row.home_team in prior["teams"]
                away_was_present = prior and row.away_team in prior["teams"]
                home_elo = elo[home_key]
                away_elo = elo[away_key]

                features = {
                    "match_id": row.id,
                    "competition": competition,
                    "season": season,
                    "date": row.date,
                    "match_week": int(row.match_week),
                    "prediction_month": row.prediction_month,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "target_btts": int(
                        row.full_time_home_goals > 0
                        and row.full_time_away_goals > 0
                    ),
                    "week_number": int(row.match_week),
                    "market_home_probability": p_home,
                    "market_draw_probability": p_draw,
                    "market_away_probability": p_away,
                    "market_strength_gap": abs(p_home - p_away),
                    "market_over_25_probability": _over_probability(row),
                    "has_over_25_market": int(
                        not pd.isna(row.odds_over_25)
                        and not pd.isna(row.odds_under_25)
                    ),
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_gap": home_elo - away_elo,
                    "home_promoted": (
                        int(not home_was_present) if prior is not None else np.nan
                    ),
                    "away_promoted": (
                        int(not away_was_present) if prior is not None else np.nan
                    ),
                    "home_prior_rank": (
                        prior["rank"].get(row.home_team, 1.05)
                        if prior is not None
                        else np.nan
                    ),
                    "away_prior_rank": (
                        prior["rank"].get(row.away_team, 1.05)
                        if prior is not None
                        else np.nan
                    ),
                    "home_prior_ppg": (
                        prior["ppg"].get(row.home_team, 0.0)
                        if prior is not None
                        else np.nan
                    ),
                    "away_prior_ppg": (
                        prior["ppg"].get(row.away_team, 0.0)
                        if prior is not None
                        else np.nan
                    ),
                    "season_matches_played": len(season_history[season_key]),
                }
                for known_competition in COMPETITIONS:
                    features[f"competition_{known_competition}"] = int(
                        competition == known_competition
                    )

                _add_history_features(features, "home", team_history[home_key])
                _add_history_features(features, "away", team_history[away_key])
                _add_history_features(
                    features, "home_venue", home_history[home_key], (5, 10)
                )
                _add_history_features(
                    features, "away_venue", away_history[away_key], (5, 10)
                )
                _add_history_features(
                    features, "head_to_head", head_to_head[pair_key], (5,)
                )
                _add_history_features(
                    features, "league", league_history[competition], (100, 380)
                )
                _add_current_team_context(
                    features,
                    "home",
                    team_history[home_key],
                    getattr(row, "open_home_coach_name", np.nan),
                    getattr(row, "open_home_formation", np.nan),
                    getattr(row, "open_home_starter_ids", np.nan),
                )
                _add_current_team_context(
                    features,
                    "away",
                    team_history[away_key],
                    getattr(row, "open_away_coach_name", np.nan),
                    getattr(row, "open_away_formation", np.nan),
                    getattr(row, "open_away_starter_ids", np.nan),
                )
                for window in (20, 50, 100):
                    recent = season_history[season_key][-window:]
                    features[f"season_btts_rate_{window}"] = _mean(recent, "btts")
                    features[f"season_over_25_rate_{window}"] = _mean(
                        recent, "over_25"
                    )
                for window in WINDOWS:
                    home_inputs = (
                        features[f"home_goals_for_{window}"],
                        features[f"away_goals_against_{window}"],
                    )
                    away_inputs = (
                        features[f"away_goals_for_{window}"],
                        features[f"home_goals_against_{window}"],
                    )
                    home_available = [
                        value for value in home_inputs if not pd.isna(value)
                    ]
                    away_available = [
                        value for value in away_inputs if not pd.isna(value)
                    ]
                    home_expected = (
                        float(np.mean(home_available))
                        if home_available
                        else np.nan
                    )
                    away_expected = (
                        float(np.mean(away_available))
                        if away_available
                        else np.nan
                    )
                    features[f"home_expected_goals_{window}"] = home_expected
                    features[f"away_expected_goals_{window}"] = away_expected
                    features[f"poisson_btts_probability_{window}"] = (
                        (1 - np.exp(-home_expected))
                        * (1 - np.exp(-away_expected))
                        if not pd.isna(home_expected)
                        and not pd.isna(away_expected)
                        else np.nan
                    )
                rows.append(features)

            # Only now may this period's outcomes become history for the next one.
            for row in batch.itertuples(index=False):
                home_key = (competition, row.home_team)
                away_key = (competition, row.away_team)
                season_key = (competition, season)
                pair_key = (competition, *sorted((row.home_team, row.away_team)))
                home_record = _match_record(row, True)
                away_record = _match_record(row, False)
                team_history[home_key].append(home_record)
                team_history[away_key].append(away_record)
                home_history[home_key].append(home_record)
                away_history[away_key].append(away_record)
                head_to_head[pair_key].append(home_record)
                league_history[competition].append(home_record)
                season_history[season_key].append(home_record)

                expected_home = 1 / (
                    1 + 10 ** ((elo[away_key] - (elo[home_key] + 65)) / 400)
                )
                if row.full_time_home_goals > row.full_time_away_goals:
                    actual_home = 1.0
                elif row.full_time_home_goals < row.full_time_away_goals:
                    actual_home = 0.0
                else:
                    actual_home = 0.5
                change = 20 * (actual_home - expected_home)
                elo[home_key] += change
                elo[away_key] -= change

    return pd.DataFrame(rows).sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)


def build_weekly_btts_dataset(batch_by: str = "week") -> pd.DataFrame:
    Base.metadata.create_all(bind=engine)
    session = get_session()
    query = text(
        """
        SELECT matches.id AS id, competition, season, match_week, date,
               home_team, away_team,
               full_time_home_goals, full_time_away_goals,
               home_shots, away_shots,
               home_shots_on_target, away_shots_on_target,
               odds_home_win, odds_draw, odds_away_win,
               odds_over_25, odds_under_25,
               open_data.home_xg AS open_home_xg,
               open_data.away_xg AS open_away_xg,
               open_data.home_shots_inside_box AS open_home_shots_inside_box,
               open_data.away_shots_inside_box AS open_away_shots_inside_box,
               open_data.home_shots_outside_box AS open_home_shots_outside_box,
               open_data.away_shots_outside_box AS open_away_shots_outside_box,
               open_data.home_blocked_shots AS open_home_blocked_shots,
               open_data.away_blocked_shots AS open_away_blocked_shots,
               open_data.home_penalties AS open_home_penalties,
               open_data.away_penalties AS open_away_penalties,
               open_data.home_corners AS open_home_corners,
               open_data.away_corners AS open_away_corners,
               open_data.home_yellow_cards AS open_home_yellow_cards,
               open_data.away_yellow_cards AS open_away_yellow_cards,
               open_data.home_red_cards AS open_home_red_cards,
               open_data.away_red_cards AS open_away_red_cards,
               open_data.home_possession AS open_home_possession,
               open_data.away_possession AS open_away_possession,
               open_data.home_fouls AS open_home_fouls,
               open_data.away_fouls AS open_away_fouls,
               open_data.home_offsides AS open_home_offsides,
               open_data.away_offsides AS open_away_offsides,
               open_data.home_pass_accuracy AS open_home_pass_accuracy,
               open_data.away_pass_accuracy AS open_away_pass_accuracy,
               open_data.home_coach_name AS open_home_coach_name,
               open_data.away_coach_name AS open_away_coach_name,
               open_data.home_formation AS open_home_formation,
               open_data.away_formation AS open_away_formation,
               open_data.home_starter_ids AS open_home_starter_ids,
               open_data.away_starter_ids AS open_away_starter_ids
        FROM matches
        LEFT JOIN open_btts_match_data AS open_data
          ON open_data.match_id = matches.id
        WHERE matches.match_week IS NOT NULL
          AND matches.full_time_home_goals IS NOT NULL
          AND matches.full_time_away_goals IS NOT NULL
        ORDER BY competition, season, match_week, date, matches.id
        """
    )
    try:
        matches = pd.read_sql(query, session.bind)
    finally:
        session.close()
    return _build_weekly_features(matches, batch_by=batch_by)


def build_monthly_btts_dataset() -> pd.DataFrame:
    """Build features with all fixtures in a calendar month held together."""
    return build_weekly_btts_dataset(batch_by="month")


def _candidate_models() -> dict[str, Pipeline]:
    def logistic(c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, C=c_value),
                ),
            ]
        )

    return {
        "logistic_c005": logistic(0.05),
        "logistic_c02": logistic(0.2),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.035,
                        max_iter=140,
                        max_leaf_nodes=9,
                        min_samples_leaf=40,
                        l2_regularization=5.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        max_depth=9,
                        min_samples_leaf=12,
                        max_features=0.6,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _weekly_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target_accuracy: float = 0.60,
) -> dict:
    scored = frame[["season", "match_week", "target_btts"]].copy()
    scored["correct"] = predictions == scored["target_btts"].to_numpy()
    weekly = scored.groupby(["season", "match_week"])["correct"].mean()
    return {
        "weeks_at_target": int((weekly >= target_accuracy).sum()),
        "weeks_total": int(len(weekly)),
        "minimum_week_accuracy": float(weekly.min()),
        "mean_week_accuracy": float(weekly.mean()),
        "overall_accuracy": float(scored["correct"].mean()),
    }


def _weekly_objective(metrics: dict) -> tuple:
    return (
        metrics["minimum_week_accuracy"],
        metrics["weeks_at_target"],
        metrics["mean_week_accuracy"],
        metrics["overall_accuracy"],
    )


def _rank_selection_objective(metrics: dict) -> tuple:
    """Optimize the worst week first, then the requested weekly hit rate."""
    return (
        metrics["minimum_week_accuracy"],
        metrics["weeks_at_target"],
        metrics["mean_week_accuracy"],
        metrics["overall_accuracy"],
    )


def _target_threshold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    target_accuracy: float = 0.60,
) -> tuple[float, dict]:
    options = []
    for threshold in np.arange(0.35, 0.651, 0.01):
        metrics = _weekly_metrics(
            frame, probabilities >= threshold, target_accuracy=target_accuracy
        )
        options.append((metrics, float(threshold)))
    metrics, threshold = max(
        options,
        key=lambda item: (*_weekly_objective(item[0]), -abs(item[1] - 0.5)),
    )
    return threshold, metrics


def _week_threshold_schedule(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    target_accuracy: float = 0.60,
) -> tuple[dict[int, float], dict]:
    schedule = {}
    thresholds = np.zeros(len(frame))
    weeks = frame["match_week"].astype(int).to_numpy()
    for week in sorted(set(weeks)):
        mask = weeks == week
        threshold, _ = _target_threshold(
            frame.loc[mask],
            probabilities[mask],
            target_accuracy=target_accuracy,
        )
        schedule[int(week)] = threshold
        thresholds[mask] = threshold
    metrics = _weekly_metrics(
        frame, probabilities >= thresholds, target_accuracy=target_accuracy
    )
    return schedule, metrics


def _probability_options(
    probabilities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    options = dict(probabilities)
    names = list(probabilities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            options[f"{left}+{right}"] = (
                probabilities[left] + probabilities[right]
            ) / 2
    options["all_ensemble"] = np.mean(list(probabilities.values()), axis=0)
    return options


def _fit_components(
    component_names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Pipeline]:
    candidates = _candidate_models()
    fitted = {}
    for name in component_names:
        model = candidates[name]
        model.fit(X, y)
        fitted[name] = model
    return fitted


def _predict_components(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
) -> np.ndarray:
    return np.mean(
        [model.predict_proba(X)[:, 1] for model in models.values()],
        axis=0,
    )


def _adaptive_feature_sets(feature_columns: list[str]) -> dict[str, list[str]]:
    """Create compact, interpretable statistic combinations for weekly selection."""
    common_exact = {
        "week_number",
        "home_elo",
        "away_elo",
        "elo_gap",
        "home_promoted",
        "away_promoted",
        "home_prior_rank",
        "away_prior_rank",
        "home_prior_ppg",
        "away_prior_ppg",
        "season_matches_played",
    }
    goal_tokens = (
        "games_",
        "goals_for_",
        "goals_against_",
        "scored_",
        "conceded_",
        "btts_",
        "clean_sheet_",
        "failed_to_score_",
        "total_goals_",
        "over_25_",
    )
    chance_tokens = (
        "shots_for_",
        "shots_against_",
        "shots_on_target_",
        "xg_",
        "shots_inside_box_",
        "shots_outside_box_",
        "blocked_shots_",
        "penalties_",
        "corners_",
        "possession_",
        "fouls_",
        "offsides_",
        "pass_accuracy_",
    )
    common = {
        column
        for column in feature_columns
        if column in common_exact or column.startswith("competition_")
    }
    market = {
        column
        for column in feature_columns
        if column.startswith("market_") or column == "has_over_25_market"
    }
    goals = {
        column
        for column in feature_columns
        if any(token in column for token in goal_tokens)
    }
    chances = {
        column
        for column in feature_columns
        if any(token in column for token in chance_tokens)
    }
    compact_tokens = (
        "games_",
        "scored_",
        "conceded_",
        "btts_",
        "xg_",
        "expected_goals_",
        "poisson_btts_",
        "season_btts_rate_",
    )
    compact = {
        column
        for column in feature_columns
        if any(token in column for token in compact_tokens)
    }
    recent = {
        column
        for column in feature_columns
        if column.endswith("_5") or column.endswith("_10")
    }
    context = {
        column
        for column in feature_columns
        if any(
            token in column
            for token in ("coach_", "formation_", "lineup_", "returning_starters")
        )
    }
    head_to_head_goals = {
        column
        for column in feature_columns
        if column.startswith("head_to_head_")
        and any(token in column for token in goal_tokens)
    }

    def ordered(columns: set[str]) -> list[str]:
        return [column for column in feature_columns if column in columns]

    return {
        "market_goal_form": ordered(common | market | goals),
        "market_chance_form": ordered(common | market | chances),
        "market_chance_context": ordered(
            common | market | chances | context | head_to_head_goals
        ),
        "market_goal_context": ordered(common | market | goals | context),
        "goal_chance_form": ordered(common | goals | chances),
        "compact_btts_form": ordered(common | market | compact),
        "recent_form": ordered(common | market | recent),
        "all_stats": list(feature_columns),
    }


def _fast_weekly_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=300,
                    tol=1e-3,
                    alpha=0.0005,
                    average=True,
                    random_state=42,
                ),
            ),
        ]
    )


def _adaptive_score(
    completed_scores: list[float], validation_metrics: dict
) -> tuple:
    """Rank an expert only from weeks whose outcomes are already known."""
    if not completed_scores:
        return (
            validation_metrics["minimum_week_accuracy"],
            validation_metrics["weeks_at_target"],
            validation_metrics["mean_week_accuracy"],
            validation_metrics["overall_accuracy"],
        )
    recent = completed_scores[-6:]
    return (
        float(min(recent)),
        int(sum(score >= 0.50 for score in recent)),
        float(np.mean(recent)),
        validation_metrics["minimum_week_accuracy"],
    )


def _update_weekly_model(
    model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> None:
    """Learn a completed week without refitting the entire match history."""
    transformed = model.named_steps["imputer"].transform(X)
    transformed = model.named_steps["scaler"].transform(transformed)
    model.named_steps["model"].partial_fit(
        transformed, y, classes=np.array([0, 1])
    )


def _weekly_rank_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    fraction: float,
    by_competition: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest probabilities within each week without its labels."""
    ranked = frame.reset_index(drop=True)
    predictions = np.zeros(len(ranked), dtype=int)
    thresholds = np.ones(len(ranked), dtype=float)
    group_columns = ["season", "match_week"]
    if by_competition:
        group_columns.append("competition")
    for positions in ranked.groupby(group_columns, sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        take = max(1, int(round(len(positions) * fraction)))
        order = positions[np.argsort(probabilities[positions])]
        selected = order[-take:]
        predictions[selected] = 1
        cutoff = float(np.min(probabilities[selected]))
        thresholds[positions] = cutoff
    return predictions, thresholds


def train_weekly_btts_model() -> dict:
    """Select a statistic combination, predict, then learn after every week."""
    target_accuracy = 0.60
    df = build_weekly_btts_dataset()
    if df.empty:
        raise ValueError("No matches with match_week are available.")

    season_starts = df.groupby("season")["date"].min().sort_values()
    seasons = list(season_starts.index)
    if len(seasons) < 5:
        raise ValueError("At least five match-week seasons are required.")
    test_season = seasons[-1]
    calibration_seasons = seasons[-4:-1]
    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and df[column].notna().any()
    ]

    selection_train = df[df["season"] < calibration_seasons[0]]
    validation = df[df["season"].isin(calibration_seasons)].reset_index(drop=True)
    pretest = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    if selection_train.empty or validation.empty or test.empty:
        raise ValueError("Training, validation, and test seasons must be populated.")

    feature_sets = _adaptive_feature_sets(feature_columns)
    selection_results = []
    candidate_calibration = {}
    validation_probabilities = {}
    for name, columns in feature_sets.items():
        model = _fast_weekly_model()
        model.fit(selection_train[columns], selection_train["target_btts"])
        probabilities = model.predict_proba(validation[columns])[:, 1]
        validation_probabilities[name] = probabilities
        threshold, global_metrics = _target_threshold(
            validation, probabilities, target_accuracy=target_accuracy
        )
        threshold_schedule, schedule_metrics = _week_threshold_schedule(
            validation, probabilities, target_accuracy=target_accuracy
        )
        if _weekly_objective(schedule_metrics) > _weekly_objective(global_metrics):
            mode = "by_week"
            metrics = schedule_metrics
        else:
            mode = "global"
            metrics = global_metrics
        candidate_calibration[name] = {
            "threshold": threshold,
            "threshold_schedule": threshold_schedule,
            "mode": mode,
            "metrics": metrics,
        }
        selection_results.append(
            {
                "name": name,
                "mode": mode,
                "overall_accuracy": metrics["overall_accuracy"],
                "weeks_at_target": metrics["weeks_at_target"],
                "weeks_total": metrics["weeks_total"],
                "minimum_week_accuracy": metrics["minimum_week_accuracy"],
                "mean_week_accuracy": metrics["mean_week_accuracy"],
                "threshold": threshold,
                "feature_count": len(columns),
            }
        )

    validation_probabilities["mean_ensemble"] = np.mean(
        list(validation_probabilities.values()), axis=0
    )
    rank_results = []
    for name, probabilities in validation_probabilities.items():
        for fraction in np.arange(0.35, 0.751, 0.025):
            for by_competition in (False, True):
                rank_predictions, _ = _weekly_rank_predictions(
                    validation,
                    probabilities,
                    float(fraction),
                    by_competition=by_competition,
                )
                metrics = _weekly_metrics(
                    validation,
                    rank_predictions,
                    target_accuracy=target_accuracy,
                )
                rank_results.append(
                    {
                        "name": name,
                        "fraction": float(fraction),
                        "by_competition": by_competition,
                        "metrics": metrics,
                    }
                )
    best_rank = max(
        rank_results,
        key=lambda result: (
            *_rank_selection_objective(result["metrics"]),
            -abs(result["fraction"] - 0.55),
            not result["by_competition"],
        ),
    )
    selected_probability_source = best_rank["name"]
    selected_rank_fraction = best_rank["fraction"]
    selected_rank_by_competition = best_rank["by_competition"]
    best_metrics = best_rank["metrics"]
    probability_calibrator = PlattProbabilityCalibrator().fit(
        validation_probabilities[selected_probability_source],
        validation["target_btts"],
    )

    prediction_frames = []
    weekly_rows = []
    candidate_week_scores: dict[str, list[float]] = {
        name: [] for name in feature_sets
    }
    cumulative_targets: list[int] = []
    cumulative_predictions: list[int] = []
    online_models = {}
    for name, columns in feature_sets.items():
        model = _fast_weekly_model()
        model.fit(pretest[columns], pretest["target_btts"])
        online_models[name] = model
    for week in sorted(test["match_week"].astype(int).unique()):
        rank_rule = {
            "name": selected_probability_source,
            "fraction": selected_rank_fraction,
            "by_competition": selected_rank_by_competition,
        }
        selected_name = rank_rule["name"]
        print(
            f"Fitting week {week}; selected stats: {selected_name}...",
            flush=True,
        )
        week_test = test[test["match_week"] == week].copy()
        candidate_outputs = {}
        for name, columns in feature_sets.items():
            model = online_models[name]
            probabilities = model.predict_proba(week_test[columns])[:, 1]
            calibration = candidate_calibration[name]
            threshold = calibration["threshold"]
            if calibration["mode"] == "by_week":
                threshold = calibration["threshold_schedule"].get(
                    int(week), threshold
                )
            candidate_outputs[name] = {
                "probabilities": probabilities,
                "threshold": threshold,
                "predictions": (probabilities >= threshold).astype(int),
            }

        if selected_name == "mean_ensemble":
            probabilities = np.mean(
                [
                    output["probabilities"]
                    for output in candidate_outputs.values()
                ],
                axis=0,
            )
        else:
            probabilities = candidate_outputs[selected_name]["probabilities"]
        raw_probabilities = probabilities.copy()
        probabilities = probability_calibrator.transform(probabilities)
        predictions, thresholds = _weekly_rank_predictions(
            week_test,
            probabilities,
            rank_rule["fraction"],
            by_competition=rank_rule["by_competition"],
        )
        targets = week_test["target_btts"].astype(int).to_numpy()
        for name, output in candidate_outputs.items():
            candidate_week_scores[name].append(
                float(accuracy_score(targets, output["predictions"]))
            )
            week_test[f"probability_{name}"] = output["probabilities"]
            week_test[f"threshold_{name}"] = output["threshold"]
            _update_weekly_model(
                online_models[name],
                week_test[feature_sets[name]],
                week_test["target_btts"],
            )
        cumulative_targets.extend(targets.tolist())
        cumulative_predictions.extend(predictions.tolist())
        week_accuracy = accuracy_score(targets, predictions)
        cumulative_accuracy = accuracy_score(
            cumulative_targets, cumulative_predictions
        )

        week_test["raw_btts_probability"] = raw_probabilities
        week_test["btts_probability"] = probabilities
        week_test["decision_threshold"] = thresholds
        week_test["selected_combination"] = selected_name
        week_test["rank_fraction"] = rank_rule["fraction"]
        week_test["rank_by_competition"] = rank_rule["by_competition"]
        week_test["predicted_btts"] = predictions
        week_test["correct"] = (predictions == targets).astype(int)
        prediction_frames.append(week_test)
        weekly_rows.append(
            {
                "match_week": int(week),
                "matches": len(week_test),
                "correct": int((predictions == targets).sum()),
                "accuracy": float(week_accuracy),
                "target_met": bool(week_accuracy >= target_accuracy),
                "cumulative_accuracy": float(cumulative_accuracy),
                "training_rows": len(pretest)
                + int((test["match_week"] < week).sum()),
                "selected_combination": selected_name,
                "rank_fraction": rank_rule["fraction"],
                "rank_by_competition": rank_rule["by_competition"],
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    weekly_summary = pd.DataFrame(weekly_rows)
    y_test = predictions["target_btts"].astype(int)
    y_pred = predictions["predicted_btts"].astype(int)
    y_probability = predictions["btts_probability"].to_numpy()
    test_accuracy = accuracy_score(y_test, y_pred)
    baseline_class = int(pretest["target_btts"].mode().iloc[0])
    baseline_accuracy = accuracy_score(
        y_test, np.full(len(y_test), baseline_class)
    )
    test_loss = log_loss(y_test, y_probability, labels=[0, 1])
    test_brier = brier_score_loss(y_test, y_probability)
    test_calibration_error = expected_calibration_error(y_test, y_probability)
    weeks_at_target = int(weekly_summary["target_met"].sum())
    worst_week_accuracy = float(weekly_summary["accuracy"].min())
    worst_weeks = weekly_summary.loc[
        weekly_summary["accuracy"] == worst_week_accuracy,
        "match_week",
    ].astype(int).tolist()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    prediction_output_columns = [
        "match_id",
        "competition",
        "season",
        "match_week",
        "date",
        "home_team",
        "away_team",
        "raw_btts_probability",
        "btts_probability",
        "decision_threshold",
        "selected_combination",
        "rank_fraction",
        "rank_by_competition",
        "predicted_btts",
        "target_btts",
        "correct",
    ]
    for name in feature_sets:
        prediction_output_columns.extend(
            [f"probability_{name}", f"threshold_{name}"]
        )
    predictions[prediction_output_columns].to_csv(PREDICTION_PATH, index=False)
    weekly_summary.to_csv(SUMMARY_PATH, index=False)

    final_models = online_models
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": final_models,
        "probability_calibrator": probability_calibrator,
        "component_names": list(feature_sets),
        "feature_sets": feature_sets,
        "feature_columns": feature_columns,
        "candidate_calibration": candidate_calibration,
        "candidate_week_scores": candidate_week_scores,
        "rank_probability_source": selected_probability_source,
        "rank_fraction": selected_rank_fraction,
        "rank_by_competition": selected_rank_by_competition,
        "validation_expert_data": {
            "metadata": validation[
                ["season", "match_week", "competition", "target_btts"]
            ].reset_index(drop=True),
            "probabilities": validation_probabilities,
        },
        "calibration_seasons": calibration_seasons,
        "test_season": test_season,
    }
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0,
    )
    return {
        "rows_total": len(df),
        "calibration_seasons": calibration_seasons,
        "test_season": test_season,
        "validation_rows": len(validation),
        "test_rows": len(test),
        "selected_model": "calibrated_weekly_rank",
        "component_names": list(feature_sets),
        "selected_threshold": None,
        "threshold_mode": "weekly_probability_rank",
        "target_accuracy": target_accuracy,
        "rank_probability_source": selected_probability_source,
        "rank_fraction": selected_rank_fraction,
        "rank_by_competition": selected_rank_by_competition,
        "week_thresholds": {
            name: values["threshold_schedule"]
            for name, values in candidate_calibration.items()
        },
        "validation_accuracy": best_metrics["overall_accuracy"],
        "validation_weeks_at_target": best_metrics["weeks_at_target"],
        "validation_weeks_total": best_metrics["weeks_total"],
        "validation_minimum_week_accuracy": best_metrics[
            "minimum_week_accuracy"
        ],
        "selection_results": sorted(
            [
                {
                    "name": result["name"],
                    "fraction": result["fraction"],
                    "by_competition": result["by_competition"],
                    **result["metrics"],
                }
                for result in rank_results
            ],
            key=lambda item: _rank_selection_objective(item),
            reverse=True,
        ),
        "weekly_results": weekly_rows,
        "weeks_at_target": weeks_at_target,
        "weeks_total": len(weekly_rows),
        "worst_week_accuracy": worst_week_accuracy,
        "worst_weeks": worst_weeks,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "test_log_loss": test_loss,
        "test_brier_score": test_brier,
        "test_calibration_error": test_calibration_error,
        "predicted_no": int((y_pred == 0).sum()),
        "predicted_yes": int((y_pred == 1).sum()),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
        "predictions_path": str(PREDICTION_PATH),
        "summary_path": str(SUMMARY_PATH),
    }
