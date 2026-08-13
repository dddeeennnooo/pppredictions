import pickle

import pandas as pd
from sqlalchemy import text

from src.models.train_logistic import MODEL_PATH, FEATURE_COLUMNS
from src.database.connection import get_session


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train-logistic first."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def predict_match_by_teams(home_team: str, away_team: str) -> dict:
    """
    Zatiaľ zoberie posledný dostupný zápas týchto dvoch tímov ako feature template.
    Toto je iba testovací prediction command.

    Neskôr spravíme normálny upcoming match feature builder.
    """
    session = get_session()

    query = text("""
        SELECT *
        FROM match_features
        WHERE home_team = :home_team
          AND away_team = :away_team
        ORDER BY date DESC
        LIMIT 1
    """)

    df = pd.read_sql(
        query,
        session.bind,
        params={
            "home_team": home_team,
            "away_team": away_team,
        },
    )

    session.close()

    if df.empty:
        raise ValueError(
            f"No historical feature row found for {home_team} vs {away_team}."
        )

    model = load_model()

    X = df[FEATURE_COLUMNS]
    probabilities = model.predict_proba(X)[0]

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "source_match_date": str(df.iloc[0]["date"]),
        "probabilities": {},
    }

    for class_name, probability in zip(model.classes_, probabilities):
        result["probabilities"][class_name] = float(probability)

    return result