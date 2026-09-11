import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.config import BASE_DIR
from src.database.connection import get_session
from src.models.train_logistic import FEATURE_COLUMNS


MODEL_DIR = BASE_DIR / "artifacts" / "models"
PREDICTION_DIR = BASE_DIR / "artifacts" / "predictions"
HOME_SCORE_MODEL_PATH = MODEL_DIR / "logistic_home_score_model.pkl"
AWAY_SCORE_MODEL_PATH = MODEL_DIR / "logistic_away_score_model.pkl"
TEAM_SCORING_PREDICTION_PATH = (
    PREDICTION_DIR / "team_scoring_btts_predictions.csv"
)
SCORING_PROBABILITY_GAP_NO_THRESHOLD = 0.20
MINIMUM_TEAM_SCORE_PROBABILITY = 0.50
MINIMUM_OPTIONAL_RULE_ACCURACY_GAIN = 0.01


def load_team_scoring_dataset() -> pd.DataFrame:
    """Load pre-match features and targets indicating whether each team scored."""
    session = get_session()
    query = text("""
        SELECT
            features.*,
            CASE WHEN matches.full_time_home_goals > 0 THEN 1 ELSE 0 END
                AS target_home_scored,
            CASE WHEN matches.full_time_away_goals > 0 THEN 1 ELSE 0 END
                AS target_away_scored
        FROM match_features AS features
        JOIN matches ON matches.id = features.match_id
        WHERE matches.full_time_home_goals IS NOT NULL
          AND matches.full_time_away_goals IS NOT NULL
        ORDER BY features.date ASC, features.match_id ASC
    """)

    try:
        return pd.read_sql(query, session.bind)
    finally:
        session.close()


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def _positive_probabilities(model: Pipeline, X: pd.DataFrame):
    positive_class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_class_index]


def _bottom_attack_vs_top_defense_flags(df: pd.DataFrame) -> pd.Series:
    """Flag pre-match bottom-five attack versus top-five defense matchups."""
    season_teams: dict[str, set[str]] = {}
    for row in df.itertuples():
        teams = season_teams.setdefault(row.season, set())
        teams.add(row.home_team)
        teams.add(row.away_team)

    latest_form: dict[str, dict[str, float]] = {}
    flags: list[bool] = []

    for row in df.itertuples():
        latest_form[row.home_team] = {
            "scored": row.home_goals_scored_last_5,
            "conceded": row.home_goals_conceded_last_5,
        }
        latest_form[row.away_team] = {
            "scored": row.away_goals_scored_last_5,
            "conceded": row.away_goals_conceded_last_5,
        }

        ranked_teams = [
            team
            for team in season_teams[row.season]
            if team in latest_form
        ]
        bottom_five_scoring = set(
            sorted(
                ranked_teams,
                key=lambda team: (latest_form[team]["scored"], team),
            )[:5]
        )
        top_five_defending = set(
            sorted(
                ranked_teams,
                key=lambda team: (latest_form[team]["conceded"], team),
            )[:5]
        )

        flags.append(
            (
                row.home_team in bottom_five_scoring
                and row.away_team in top_five_defending
            )
            or (
                row.away_team in bottom_five_scoring
                and row.home_team in top_five_defending
            )
        )

    return pd.Series(flags, index=df.index)


def _btts_rule_predictions(
    home_score_probability,
    away_score_probability,
    *,
    gap_no_threshold: float = SCORING_PROBABILITY_GAP_NO_THRESHOLD,
    minimum_team_score_probability: float = MINIMUM_TEAM_SCORE_PROBABILITY,
    minimum_joint_probability: float = 0.0,
    upper_gap_no_threshold: float | None = None,
    ranking_no_flags=None,
):
    """Apply the mandatory gap rule plus validation-selected safeguards."""
    home_probability = pd.Series(home_score_probability).to_numpy(dtype=float)
    away_probability = pd.Series(away_score_probability).to_numpy(dtype=float)
    gap = abs(home_probability - away_probability)
    joint_probability = home_probability * away_probability

    # User-defined invariants: the gap must be at least 20 percentage points,
    # and each team's scoring chance must be strictly greater than 50%.
    no_mask = (
        (gap < gap_no_threshold)
        | (home_probability <= minimum_team_score_probability)
        | (away_probability <= minimum_team_score_probability)
    )
    if minimum_joint_probability > 0:
        no_mask |= joint_probability < minimum_joint_probability
    if upper_gap_no_threshold is not None:
        no_mask |= gap >= upper_gap_no_threshold
    if ranking_no_flags is not None:
        no_mask |= pd.Series(ranking_no_flags).to_numpy(dtype=bool)
    return (~no_mask).astype(int)


def _rule_metrics(actual_btts, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(actual_btts, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_btts, predictions)
        ),
    }


def _select_supplemental_rule(
    actual_btts,
    home_score_probability,
    away_score_probability,
    ranking_no_flags,
) -> dict:
    """Tune optional safeguards on validation data, never on final test data."""
    joint_thresholds = [0.0] + [
        float(value) for value in pd.RangeIndex(400, 801, 25) / 1000
    ]
    upper_gap_thresholds = [None] + [
        float(value) for value in pd.RangeIndex(225, 601, 25) / 1000
    ]
    candidates = []
    for joint_threshold in joint_thresholds:
        for upper_gap_threshold in upper_gap_thresholds:
            for use_ranking_rule in (False, True):
                predictions = _btts_rule_predictions(
                    home_score_probability,
                    away_score_probability,
                    minimum_joint_probability=joint_threshold,
                    upper_gap_no_threshold=upper_gap_threshold,
                    ranking_no_flags=(
                        ranking_no_flags if use_ranking_rule else None
                    ),
                )
                metrics = _rule_metrics(actual_btts, predictions)
                optional_rule_count = sum(
                    (
                        joint_threshold > 0,
                        upper_gap_threshold is not None,
                        use_ranking_rule,
                    )
                )
                candidates.append(
                    {
                        "minimum_joint_probability": joint_threshold,
                        "upper_gap_no_threshold": upper_gap_threshold,
                        "use_ranking_rule": use_ranking_rule,
                        "optional_rule_count": optional_rule_count,
                        **metrics,
                    }
                )

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            candidate["accuracy"],
            candidate["balanced_accuracy"],
            -candidate["optional_rule_count"],
            -candidate["minimum_joint_probability"],
            candidate["upper_gap_no_threshold"] is None,
            not candidate["use_ranking_rule"],
        ),
    )
    requested_rule_candidate = next(
        candidate
        for candidate in candidates
        if candidate["optional_rule_count"] == 0
    )
    if (
        best_candidate["accuracy"] - requested_rule_candidate["accuracy"]
        < MINIMUM_OPTIONAL_RULE_ACCURACY_GAIN
    ):
        return requested_rule_candidate
    return best_candidate


def train_team_scoring_models() -> dict:
    """Train scoring models and evaluate a validation-calibrated BTTS rule."""
    df = load_team_scoring_dataset()
    required_columns = FEATURE_COLUMNS + [
        "target_home_scored",
        "target_away_scored",
    ]
    df = df.dropna(subset=required_columns).copy()

    if len(df) < 2:
        raise ValueError("Not enough complete rows to train team-scoring models.")

    train_end = int(len(df) * 0.6)
    validation_end = int(len(df) * 0.8)
    if train_end == 0 or train_end == validation_end or validation_end == len(df):
        raise ValueError(
            "The team-scoring dataset is too small for a 60/20/20 split."
        )

    df["bottom_attack_vs_top_defense"] = _bottom_attack_vs_top_defense_flags(df)

    selection_train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    final_train_df = df.iloc[:validation_end]
    test_df = df.iloc[validation_end:].copy()

    X_selection_train = selection_train_df[FEATURE_COLUMNS]
    X_validation = validation_df[FEATURE_COLUMNS]
    X_final_train = final_train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_home_selection_train = selection_train_df[
        "target_home_scored"
    ].astype(int)
    y_away_selection_train = selection_train_df[
        "target_away_scored"
    ].astype(int)
    y_home_final_train = final_train_df["target_home_scored"].astype(int)
    y_away_final_train = final_train_df["target_away_scored"].astype(int)
    y_home_test = test_df["target_home_scored"].astype(int)
    y_away_test = test_df["target_away_scored"].astype(int)

    if (
        y_home_selection_train.nunique() < 2
        or y_away_selection_train.nunique() < 2
    ):
        raise ValueError("Both scoring models require scored and did-not-score rows.")

    selection_home_model = _build_model()
    selection_away_model = _build_model()
    selection_home_model.fit(X_selection_train, y_home_selection_train)
    selection_away_model.fit(X_selection_train, y_away_selection_train)
    validation_home_probability = _positive_probabilities(
        selection_home_model, X_validation
    )
    validation_away_probability = _positive_probabilities(
        selection_away_model, X_validation
    )
    validation_actual_btts = (
        validation_df["target_home_scored"].astype(int)
        & validation_df["target_away_scored"].astype(int)
    ).to_numpy()
    selected_rule = _select_supplemental_rule(
        validation_actual_btts,
        validation_home_probability,
        validation_away_probability,
        validation_df["bottom_attack_vs_top_defense"],
    )

    # Refit on all pre-test rows only after rule selection is complete.
    home_model = _build_model()
    away_model = _build_model()
    home_model.fit(X_final_train, y_home_final_train)
    away_model.fit(X_final_train, y_away_final_train)

    home_score_probability = _positive_probabilities(home_model, X_test)
    away_score_probability = _positive_probabilities(away_model, X_test)

    home_prediction = (home_score_probability >= 0.50).astype(int)
    away_prediction = (away_score_probability >= 0.50).astype(int)

    home_train_probability = _positive_probabilities(home_model, X_final_train)
    away_train_probability = _positive_probabilities(away_model, X_final_train)
    train_probability_gap = abs(
        home_train_probability - away_train_probability
    )
    average_gap = float(train_probability_gap.mean())
    uneven_gap_threshold = float(pd.Series(train_probability_gap).quantile(0.75))

    actual_btts = (y_home_test & y_away_test).astype(int).to_numpy()
    test_probability_gap = abs(home_score_probability - away_score_probability)
    most_uneven_mask = test_probability_gap >= uneven_gap_threshold
    below_average_gap_mask = test_probability_gap < average_gap
    ranking_no_mask = test_df[
        "bottom_attack_vs_top_defense"
    ].to_numpy(dtype=bool)

    # Retain the old rule only as an apples-to-apples benchmark.
    legacy_btts_no_mask = (
        most_uneven_mask
        | below_average_gap_mask
        | ranking_no_mask
    )
    legacy_btts_prediction = (~legacy_btts_no_mask).astype(int)
    requested_rule_prediction = _btts_rule_predictions(
        home_score_probability,
        away_score_probability,
    )
    rule_btts_prediction = _btts_rule_predictions(
        home_score_probability,
        away_score_probability,
        minimum_joint_probability=selected_rule["minimum_joint_probability"],
        upper_gap_no_threshold=selected_rule["upper_gap_no_threshold"],
        ranking_no_flags=(
            ranking_no_mask if selected_rule["use_ranking_rule"] else None
        ),
    )

    home_accuracy = accuracy_score(y_home_test, home_prediction)
    away_accuracy = accuracy_score(y_away_test, away_prediction)
    legacy_rule_accuracy = accuracy_score(
        actual_btts, legacy_btts_prediction
    )
    requested_rule_accuracy = accuracy_score(
        actual_btts, requested_rule_prediction
    )
    rule_accuracy = accuracy_score(actual_btts, rule_btts_prediction)

    home_loss = log_loss(
        y_home_test,
        home_model.predict_proba(X_test),
        labels=home_model.classes_,
    )
    away_loss = log_loss(
        y_away_test,
        away_model.predict_proba(X_test),
        labels=away_model.classes_,
    )
    rule_report = classification_report(
        actual_btts,
        rule_btts_prediction,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(HOME_SCORE_MODEL_PATH, "wb") as file:
        pickle.dump(home_model, file)
    with open(AWAY_SCORE_MODEL_PATH, "wb") as file:
        pickle.dump(away_model, file)

    test_df["home_score_probability"] = home_score_probability
    test_df["away_score_probability"] = away_score_probability
    test_df["scoring_probability_gap"] = test_probability_gap
    test_df["joint_scoring_probability"] = (
        home_score_probability * away_score_probability
    )
    test_df["target_btts"] = actual_btts
    test_df["legacy_predicted_btts"] = legacy_btts_prediction
    test_df["requested_rule_predicted_btts"] = requested_rule_prediction
    test_df["predicted_btts"] = rule_btts_prediction
    test_df["correct"] = (rule_btts_prediction == actual_btts).astype(int)
    prediction_columns = [
        "match_id",
        "competition",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_score_probability",
        "away_score_probability",
        "scoring_probability_gap",
        "joint_scoring_probability",
        "target_btts",
        "legacy_predicted_btts",
        "requested_rule_predicted_btts",
        "predicted_btts",
        "correct",
    ]
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    test_df[prediction_columns].to_csv(
        TEAM_SCORING_PREDICTION_PATH, index=False
    )

    latest_season = str(df.iloc[-1]["season"])
    latest_season_mask = test_df["season"].astype(str) == latest_season
    latest_actual = actual_btts[latest_season_mask.to_numpy()]
    latest_prediction = rule_btts_prediction[
        latest_season_mask.to_numpy()
    ]
    latest_legacy_prediction = legacy_btts_prediction[
        latest_season_mask.to_numpy()
    ]
    latest_correct = int((latest_actual == latest_prediction).sum())
    latest_legacy_correct = int(
        (latest_actual == latest_legacy_prediction).sum()
    )
    latest_season_accuracy = (
        float(accuracy_score(latest_actual, latest_prediction))
        if len(latest_actual)
        else None
    )
    latest_season_legacy_accuracy = (
        float(accuracy_score(latest_actual, latest_legacy_prediction))
        if len(latest_actual)
        else None
    )
    latest_season_majority_accuracy = (
        float(pd.Series(latest_actual).value_counts(normalize=True).max())
        if len(latest_actual)
        else None
    )

    return {
        "rows_total": len(df),
        "rows_train": len(selection_train_df),
        "rows_validation": len(validation_df),
        "rows_final_train": len(final_train_df),
        "rows_test": len(test_df),
        "validation_start_date": str(validation_df.iloc[0]["date"]),
        "validation_end_date": str(validation_df.iloc[-1]["date"]),
        "test_start_date": str(test_df.iloc[0]["date"]),
        "test_end_date": str(test_df.iloc[-1]["date"]),
        "home_score_accuracy": home_accuracy,
        "away_score_accuracy": away_accuracy,
        "home_score_log_loss": home_loss,
        "away_score_log_loss": away_loss,
        "legacy_rule_btts_accuracy": legacy_rule_accuracy,
        "requested_rule_btts_accuracy": requested_rule_accuracy,
        "rule_btts_accuracy": rule_accuracy,
        "gap_no_threshold": SCORING_PROBABILITY_GAP_NO_THRESHOLD,
        "minimum_team_score_probability": MINIMUM_TEAM_SCORE_PROBABILITY,
        "minimum_joint_probability": selected_rule[
            "minimum_joint_probability"
        ],
        "upper_gap_no_threshold": selected_rule[
            "upper_gap_no_threshold"
        ],
        "use_ranking_rule": selected_rule["use_ranking_rule"],
        "validation_rule_accuracy": selected_rule["accuracy"],
        "validation_rule_balanced_accuracy": selected_rule[
            "balanced_accuracy"
        ],
        "predicted_btts_no": int((rule_btts_prediction == 0).sum()),
        "predicted_btts_yes": int((rule_btts_prediction == 1).sum()),
        "average_gap": average_gap,
        "uneven_gap_threshold": uneven_gap_threshold,
        "most_uneven_count": int(most_uneven_mask.sum()),
        "below_average_gap_count": int(below_average_gap_mask.sum()),
        "ranking_no_count": int(ranking_no_mask.sum()),
        "home_probability_min": float(home_score_probability.min()),
        "home_probability_max": float(home_score_probability.max()),
        "away_probability_min": float(away_score_probability.min()),
        "away_probability_max": float(away_score_probability.max()),
        "classification_report": rule_report,
        "latest_season": latest_season,
        "latest_season_matches": int(latest_season_mask.sum()),
        "latest_season_correct": latest_correct,
        "latest_season_accuracy": latest_season_accuracy,
        "latest_season_legacy_correct": latest_legacy_correct,
        "latest_season_legacy_accuracy": latest_season_legacy_accuracy,
        "latest_season_majority_accuracy": latest_season_majority_accuracy,
        "home_model_path": str(HOME_SCORE_MODEL_PATH),
        "away_model_path": str(AWAY_SCORE_MODEL_PATH),
        "predictions_path": str(TEAM_SCORING_PREDICTION_PATH),
    }
