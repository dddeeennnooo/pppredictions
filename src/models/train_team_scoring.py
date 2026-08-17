import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.config import BASE_DIR
from src.database.connection import get_session
from src.models.train_logistic import FEATURE_COLUMNS


MODEL_DIR = BASE_DIR / "artifacts" / "models"
HOME_SCORE_MODEL_PATH = MODEL_DIR / "logistic_home_score_model.pkl"
AWAY_SCORE_MODEL_PATH = MODEL_DIR / "logistic_away_score_model.pkl"

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


def train_team_scoring_models() -> dict:
    """Train separate home/away scoring models and evaluate the BTTS band rule."""
    df = load_team_scoring_dataset()
    required_columns = FEATURE_COLUMNS + [
        "target_home_scored",
        "target_away_scored",
    ]
    df = df.dropna(subset=required_columns)

    if len(df) < 2:
        raise ValueError("Not enough complete rows to train team-scoring models.")

    split_index = int(len(df) * 0.8)
    if split_index == 0 or split_index == len(df):
        raise ValueError("The team-scoring dataset is too small for an 80/20 split.")

    df["bottom_attack_vs_top_defense"] = _bottom_attack_vs_top_defense_flags(df)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_home_train = train_df["target_home_scored"].astype(int)
    y_away_train = train_df["target_away_scored"].astype(int)
    y_home_test = test_df["target_home_scored"].astype(int)
    y_away_test = test_df["target_away_scored"].astype(int)

    if y_home_train.nunique() < 2 or y_away_train.nunique() < 2:
        raise ValueError("Both scoring models require scored and did-not-score rows.")

    home_model = _build_model()
    away_model = _build_model()
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    home_score_probability = _positive_probabilities(home_model, X_test)
    away_score_probability = _positive_probabilities(away_model, X_test)

    home_prediction = (home_score_probability >= 0.50).astype(int)
    away_prediction = (away_score_probability >= 0.50).astype(int)

    home_train_probability = _positive_probabilities(home_model, X_train)
    away_train_probability = _positive_probabilities(away_model, X_train)
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

    btts_no_mask = (
        most_uneven_mask
        | below_average_gap_mask
        | ranking_no_mask
    )
    rule_btts_prediction = (~btts_no_mask).astype(int)

    home_accuracy = accuracy_score(y_home_test, home_prediction)
    away_accuracy = accuracy_score(y_away_test, away_prediction)
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

    return {
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "test_start_date": str(test_df.iloc[0]["date"]),
        "test_end_date": str(test_df.iloc[-1]["date"]),
        "home_score_accuracy": home_accuracy,
        "away_score_accuracy": away_accuracy,
        "home_score_log_loss": home_loss,
        "away_score_log_loss": away_loss,
        "rule_btts_accuracy": rule_accuracy,
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
        "home_model_path": str(HOME_SCORE_MODEL_PATH),
        "away_model_path": str(AWAY_SCORE_MODEL_PATH),
    }
