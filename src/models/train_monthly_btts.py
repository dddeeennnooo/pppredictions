from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

from src.config import BASE_DIR
from src.models.train_weekly_btts import (
    METADATA_COLUMNS,
    _adaptive_feature_sets,
    _fast_weekly_model,
    _update_weekly_model,
    build_monthly_btts_dataset,
)


MODEL_PATH = BASE_DIR / "artifacts" / "models" / "monthly_btts_model.pkl"
PREDICTION_DIR = BASE_DIR / "artifacts" / "predictions"
PREDICTION_PATH = PREDICTION_DIR / "monthly_btts_predictions.csv"
SUMMARY_PATH = PREDICTION_DIR / "monthly_btts_summary.csv"


def _add_season_month_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    month_order = (
        frame[["season", "prediction_month"]]
        .drop_duplicates()
        .sort_values(["season", "prediction_month"])
    )
    month_order["season_month"] = (
        month_order.groupby("season").cumcount() + 1
    )
    return frame.merge(
        month_order,
        on=["season", "prediction_month"],
        how="left",
        validate="many_to_one",
    )


def _monthly_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target_accuracy: float = 0.60,
) -> dict:
    scored = frame[
        ["season", "prediction_month", "target_btts"]
    ].copy()
    scored["correct"] = (
        np.asarray(predictions) == scored["target_btts"].to_numpy()
    )
    monthly = scored.groupby(["season", "prediction_month"])["correct"].mean()
    return {
        "months_at_target": int((monthly >= target_accuracy).sum()),
        "months_total": int(len(monthly)),
        "minimum_month_accuracy": float(monthly.min()),
        "mean_month_accuracy": float(monthly.mean()),
        "overall_accuracy": float(scored["correct"].mean()),
    }


def _monthly_objective(metrics: dict) -> tuple:
    return (
        metrics["minimum_month_accuracy"],
        metrics["months_at_target"],
        metrics["mean_month_accuracy"],
        metrics["overall_accuracy"],
    )


def _monthly_rank_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    fraction: float,
    by_competition: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank probabilities inside each month without consulting its outcomes."""
    ranked = frame.reset_index(drop=True)
    predictions = np.zeros(len(ranked), dtype=int)
    thresholds = np.ones(len(ranked), dtype=float)
    group_columns = ["season", "prediction_month"]
    if by_competition:
        group_columns.append("competition")
    for positions in ranked.groupby(group_columns, sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        take = max(1, int(round(len(positions) * fraction)))
        order = positions[np.argsort(probabilities[positions])]
        selected = order[-take:]
        predictions[selected] = 1
        thresholds[positions] = float(np.min(probabilities[selected]))
    return predictions, thresholds


def _walk_forward_validation_probabilities(
    selection_train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    probabilities = {
        name: np.zeros(len(validation), dtype=float) for name in feature_sets
    }
    models = {}
    for name, columns in feature_sets.items():
        model = _fast_weekly_model()
        model.fit(selection_train[columns], selection_train["target_btts"])
        models[name] = model

    month_starts = validation.groupby("prediction_month")["date"].min().sort_values()
    for month in month_starts.index:
        positions = np.flatnonzero(
            validation["prediction_month"].to_numpy() == month
        )
        month_frame = validation.iloc[positions]
        for name, columns in feature_sets.items():
            probabilities[name][positions] = models[name].predict_proba(
                month_frame[columns]
            )[:, 1]
            _update_weekly_model(
                models[name],
                month_frame[columns],
                month_frame["target_btts"],
            )
    probabilities["mean_ensemble"] = np.mean(
        list(probabilities.values()), axis=0
    )
    return probabilities


def _calibrate_month_rule(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    target_accuracy: float,
) -> dict:
    results = []
    for name, values in probabilities.items():
        for fraction in np.arange(0.25, 0.801, 0.025):
            for by_competition in (False, True):
                predictions, _ = _monthly_rank_predictions(
                    frame,
                    values,
                    float(fraction),
                    by_competition=by_competition,
                )
                metrics = _monthly_metrics(
                    frame, predictions, target_accuracy=target_accuracy
                )
                results.append(
                    {
                        "name": name,
                        "fraction": float(fraction),
                        "by_competition": by_competition,
                        "metrics": metrics,
                    }
                )
    return max(
        results,
        key=lambda result: (
            *_monthly_objective(result["metrics"]),
            -abs(result["fraction"] - 0.55),
            not result["by_competition"],
        ),
    )


def train_monthly_btts_model() -> dict:
    """Predict a calendar month, then learn it before predicting the next one."""
    target_accuracy = 0.60
    df = _add_season_month_index(build_monthly_btts_dataset())
    if df.empty:
        raise ValueError("No completed matches with dates are available.")

    season_starts = df.groupby("season")["date"].min().sort_values()
    seasons = list(season_starts.index)
    if len(seasons) < 5:
        raise ValueError("At least five seasons are required.")
    test_season = seasons[-1]
    calibration_seasons = seasons[-4:-1]
    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS | {"season_month"}
        and df[column].notna().any()
    ]

    selection_train = df[df["season"] < calibration_seasons[0]]
    validation = df[df["season"].isin(calibration_seasons)].reset_index(drop=True)
    pretest = df[df["season"] < test_season]
    test = df[df["season"] == test_season].reset_index(drop=True)
    if selection_train.empty or validation.empty or test.empty:
        raise ValueError("Training, validation, and test seasons must be populated.")

    feature_sets = _adaptive_feature_sets(feature_columns)
    validation_probabilities = _walk_forward_validation_probabilities(
        selection_train, validation, feature_sets
    )
    global_rule = _calibrate_month_rule(
        validation, validation_probabilities, target_accuracy
    )
    month_schedule = {}
    for season_month in sorted(validation["season_month"].unique()):
        mask = validation["season_month"].to_numpy() == season_month
        month_schedule[int(season_month)] = _calibrate_month_rule(
            validation.loc[mask].reset_index(drop=True),
            {
                name: values[mask]
                for name, values in validation_probabilities.items()
            },
            target_accuracy,
        )

    online_models = {}
    for name, columns in feature_sets.items():
        model = _fast_weekly_model()
        model.fit(pretest[columns], pretest["target_btts"])
        online_models[name] = model

    prediction_frames = []
    monthly_rows = []
    cumulative_targets: list[int] = []
    cumulative_predictions: list[int] = []
    month_starts = test.groupby("prediction_month")["date"].min().sort_values()
    for month in month_starts.index:
        month_test = test[test["prediction_month"] == month].copy()
        season_month = int(month_test["season_month"].iloc[0])
        rule = month_schedule.get(season_month, global_rule)
        print(
            f"Predicting {month}; selected stats: {rule['name']}...",
            flush=True,
        )

        candidate_probabilities = {}
        for name, columns in feature_sets.items():
            candidate_probabilities[name] = online_models[name].predict_proba(
                month_test[columns]
            )[:, 1]
            month_test[f"probability_{name}"] = candidate_probabilities[name]
        candidate_probabilities["mean_ensemble"] = np.mean(
            list(candidate_probabilities.values()), axis=0
        )
        probabilities = candidate_probabilities[rule["name"]]
        predictions, thresholds = _monthly_rank_predictions(
            month_test,
            probabilities,
            rule["fraction"],
            by_competition=rule["by_competition"],
        )
        targets = month_test["target_btts"].astype(int).to_numpy()

        # The completed month is learned only after all its predictions exist.
        for name, columns in feature_sets.items():
            _update_weekly_model(
                online_models[name],
                month_test[columns],
                month_test["target_btts"],
            )

        cumulative_targets.extend(targets.tolist())
        cumulative_predictions.extend(predictions.tolist())
        month_accuracy = accuracy_score(targets, predictions)
        month_test["btts_probability"] = probabilities
        month_test["decision_threshold"] = thresholds
        month_test["selected_combination"] = rule["name"]
        month_test["rank_fraction"] = rule["fraction"]
        month_test["rank_by_competition"] = rule["by_competition"]
        month_test["predicted_btts"] = predictions
        month_test["correct"] = (predictions == targets).astype(int)
        prediction_frames.append(month_test)
        monthly_rows.append(
            {
                "prediction_month": month,
                "season_month": season_month,
                "matches": len(month_test),
                "correct": int((predictions == targets).sum()),
                "accuracy": float(month_accuracy),
                "target_met": bool(month_accuracy >= target_accuracy),
                "cumulative_accuracy": float(
                    accuracy_score(cumulative_targets, cumulative_predictions)
                ),
                "training_rows": len(pretest)
                + int((test["prediction_month"] < month).sum()),
                "selected_combination": rule["name"],
                "rank_fraction": rule["fraction"],
                "rank_by_competition": rule["by_competition"],
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    monthly_summary = pd.DataFrame(monthly_rows)
    y_test = predictions["target_btts"].astype(int)
    y_pred = predictions["predicted_btts"].astype(int)
    y_probability = predictions["btts_probability"].to_numpy()
    test_accuracy = accuracy_score(y_test, y_pred)
    baseline_class = int(pretest["target_btts"].mode().iloc[0])
    baseline_accuracy = accuracy_score(
        y_test, np.full(len(y_test), baseline_class)
    )
    test_loss = log_loss(y_test, y_probability, labels=[0, 1])
    months_at_target = int(monthly_summary["target_met"].sum())
    worst_month_accuracy = float(monthly_summary["accuracy"].min())
    worst_months = monthly_summary.loc[
        monthly_summary["accuracy"] == worst_month_accuracy,
        "prediction_month",
    ].tolist()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    prediction_output_columns = [
        "match_id",
        "competition",
        "season",
        "prediction_month",
        "season_month",
        "match_week",
        "date",
        "home_team",
        "away_team",
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
        prediction_output_columns.append(f"probability_{name}")
    predictions[prediction_output_columns].to_csv(PREDICTION_PATH, index=False)
    monthly_summary.to_csv(SUMMARY_PATH, index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": online_models,
        "component_names": list(feature_sets),
        "feature_sets": feature_sets,
        "feature_columns": feature_columns,
        "global_rule": global_rule,
        "month_schedule": month_schedule,
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
        "target_accuracy": target_accuracy,
        "validation_months_at_target": global_rule["metrics"]["months_at_target"],
        "validation_months_total": global_rule["metrics"]["months_total"],
        "monthly_results": monthly_rows,
        "months_at_target": months_at_target,
        "months_total": len(monthly_rows),
        "worst_month_accuracy": worst_month_accuracy,
        "worst_months": worst_months,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "test_log_loss": test_loss,
        "predicted_no": int((y_pred == 0).sum()),
        "predicted_yes": int((y_pred == 1).sum()),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
        "predictions_path": str(PREDICTION_PATH),
        "summary_path": str(SUMMARY_PATH),
    }
