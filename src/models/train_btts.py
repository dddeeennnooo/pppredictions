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
MODEL_PATH = MODEL_DIR / "logistic_btts_model.pkl"


def load_btts_training_dataset() -> pd.DataFrame:
    """Load pre-match features and derive the BTTS target from final scores."""
    session = get_session()

    query = text("""
        SELECT
            features.*,
            CASE
                WHEN matches.full_time_home_goals > 0
                 AND matches.full_time_away_goals > 0 THEN 1
                ELSE 0
            END AS target_btts
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


def train_btts_model() -> dict:
    """Train and evaluate a chronological both-teams-to-score baseline."""
    df = load_btts_training_dataset()
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_btts"])

    if len(df) < 2:
        raise ValueError("Not enough complete feature rows to train the BTTS model.")

    split_index = int(len(df) * 0.8)
    if split_index == 0 or split_index == len(df):
        raise ValueError("The BTTS dataset is too small for an 80/20 split.")

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["target_btts"].astype(int)
    y_test = test_df["target_btts"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("The BTTS training data must contain both Yes and No results.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    majority_class = int(y_train.mode().iloc[0])

    accuracy = accuracy_score(y_test, y_pred)
    baseline_accuracy = accuracy_score(
        y_test,
        [majority_class] * len(y_test),
    )
    loss = log_loss(y_test, y_proba, labels=model.classes_)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return {
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "test_start_date": str(test_df.iloc[0]["date"]),
        "test_end_date": str(test_df.iloc[-1]["date"]),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "log_loss": loss,
        "classification_report": report,
        "model_path": str(MODEL_PATH),
    }
