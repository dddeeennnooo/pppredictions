import pickle

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sqlalchemy import text

from src.database.connection import get_session
from src.models.train_logistic import MODEL_PATH, FEATURE_COLUMNS


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train-logistic first."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def _load_dataset() -> pd.DataFrame:
    session = get_session()

    query = text("""
        SELECT *
        FROM match_features
        WHERE target_result IN ('H', 'D', 'A')
    """)

    df = pd.read_sql(query, session.bind)
    session.close()

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_result"])
    return df


def _apply_draw_rule(
    row: pd.Series,
    max_home_away_threshold: float,
    home_away_diff_threshold: float,
) -> str:
    p_home = row["p_H"]
    p_draw = row["p_D"]
    p_away = row["p_A"]

    max_home_away = max(p_home, p_away)
    home_away_diff = abs(p_home - p_away)

    if (
        max_home_away < max_home_away_threshold
        and home_away_diff < home_away_diff_threshold
    ):
        return "D"

    probabilities = {
        "H": p_home,
        "D": p_draw,
        "A": p_away,
    }

    return max(probabilities, key=probabilities.get)


def evaluate_draw_rule(
    max_home_away_threshold: float = 0.47,
    home_away_diff_threshold: float = 0.10,
) -> dict:
    model = _load_model()
    df = _load_dataset()

    # rovnaký test split ako pri tréningu: posledných 20 %
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_result"]

    probabilities = model.predict_proba(X_test)

    for class_name_index, class_name in enumerate(model.classes_):
        test_df[f"p_{class_name}"] = probabilities[:, class_name_index]

    test_df["prediction_default"] = model.predict(X_test)

    test_df["prediction_draw_rule"] = test_df.apply(
        lambda row: _apply_draw_rule(
            row,
            max_home_away_threshold=max_home_away_threshold,
            home_away_diff_threshold=home_away_diff_threshold,
        ),
        axis=1,
    )

    default_accuracy = accuracy_score(y_test, test_df["prediction_default"])
    draw_rule_accuracy = accuracy_score(y_test, test_df["prediction_draw_rule"])

    default_report = classification_report(
        y_test,
        test_df["prediction_default"],
        output_dict=False,
    )

    draw_rule_report = classification_report(
        y_test,
        test_df["prediction_draw_rule"],
        output_dict=False,
    )

    draw_predictions_count = int((test_df["prediction_draw_rule"] == "D").sum())

    draw_bets = test_df[
        (test_df["prediction_draw_rule"] == "D")
        & (test_df["odds_draw"].notna())
    ].copy()

    draw_bets["profit"] = draw_bets.apply(
        lambda row: row["odds_draw"] - 1
        if row["target_result"] == "D"
        else -1,
        axis=1,
    )

    draw_bets_count = len(draw_bets)
    draw_bets_won = int((draw_bets["target_result"] == "D").sum())

    draw_bets_winrate = (
        draw_bets_won / draw_bets_count
        if draw_bets_count > 0
        else 0
    )

    draw_bets_avg_odds = (
        float(draw_bets["odds_draw"].mean())
        if draw_bets_count > 0
        else 0
    )

    draw_bets_profit = (
        float(draw_bets["profit"].sum())
        if draw_bets_count > 0
        else 0
    )

    draw_bets_roi = (
        draw_bets_profit / draw_bets_count
        if draw_bets_count > 0
        else 0
    )

    return {
        "max_home_away_threshold": max_home_away_threshold,
        "home_away_diff_threshold": home_away_diff_threshold,
        "rows_test": len(test_df),
        "default_accuracy": default_accuracy,
        "draw_rule_accuracy": draw_rule_accuracy,
        "draw_predictions_count": draw_predictions_count,
        "default_report": default_report,
        "draw_rule_report": draw_rule_report,
        "draw_bets_count": draw_bets_count,
        "draw_bets_won": draw_bets_won,
        "draw_bets_winrate": draw_bets_winrate,
        "draw_bets_avg_odds": draw_bets_avg_odds,
        "draw_bets_profit": draw_bets_profit,
        "draw_bets_roi": draw_bets_roi,
    }