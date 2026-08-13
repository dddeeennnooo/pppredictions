import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.config import BASE_DIR
from src.database.connection import get_session


MODEL_DIR = BASE_DIR / "artifacts" / "models"
MODEL_PATH = MODEL_DIR / "logistic_1x2_model.pkl"


FEATURE_COLUMNS = [
    "home_points_last_5",
    "away_points_last_5",
    "home_goals_scored_last_5",
    "away_goals_scored_last_5",
    "home_goals_conceded_last_5",
    "away_goals_conceded_last_5",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_draw_rate_last_5",
    "away_draw_rate_last_5",
    "home_loss_rate_last_5",
    "away_loss_rate_last_5",
    "implied_probability_home",
    "implied_probability_draw",
    "implied_probability_away",
    "bookmaker_margin",
]


def load_training_dataset() -> pd.DataFrame:
    session = get_session()

    query = text("""
        SELECT
            *
        FROM match_features
        WHERE target_result IN ('H', 'D', 'A')
    """)

    df = pd.read_sql(query, session.bind)
    session.close()

    return df


def train_logistic_model() -> dict:
    df = load_training_dataset()

    # odstránime riadky, kde nemáme kurzy alebo features
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_result"])

    X = df[FEATURE_COLUMNS]
    y = df["target_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,  # dôležité: pri časových dátach nechceme náhodne miešať
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba, labels=model.classes_)

    report = classification_report(y_test, y_pred, output_dict=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return {
        "rows_total": len(df),
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "accuracy": accuracy,
        "log_loss": loss,
        "classes": list(model.classes_),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
    }