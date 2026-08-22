from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.database.connection import get_session


FORM_WINDOW = 5
INITIAL_ELO = 1500.0
HOME_ELO_ADVANTAGE = 80.0
ELO_K_FACTOR = 20.0


FEATURE_COLUMNS = [
    "home_points_5",
    "away_points_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_draw_rate_5",
    "away_draw_rate_5",
    "home_btts_rate_5",
    "away_btts_rate_5",
    "home_over_25_rate_5",
    "away_over_25_rate_5",
    "home_clean_sheet_rate_5",
    "away_clean_sheet_rate_5",
    "home_failed_to_score_rate_5",
    "away_failed_to_score_rate_5",
    "home_venue_points_5",
    "away_venue_points_5",
    "home_venue_goals_for_5",
    "away_venue_goals_for_5",
    "home_venue_goals_against_5",
    "away_venue_goals_against_5",
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_rest_days",
    "away_rest_days",
    "league_home_goals",
    "league_away_goals",
    "league_btts_rate",
]


TARGET_COLUMNS = {
    "result": "target_result",
    "btts": "target_btts",
    "over_25": "target_over_25",
    "home_scores": "target_home_scores",
    "away_scores": "target_away_scores",
}


@dataclass
class TeamMatch:
    date: date
    venue: str
    points: int
    goals_for: int
    goals_against: int


@dataclass
class HistoryState:
    team_matches: dict[str, list[TeamMatch]] = field(
        default_factory=lambda: defaultdict(list)
    )
    elo: dict[str, float] = field(default_factory=dict)
    league_matches: int = 0
    league_home_goals: int = 0
    league_away_goals: int = 0
    league_btts: int = 0


def _mean(values: list[float], default: float) -> float:
    return float(sum(values) / len(values)) if values else default


def _team_form(
    state: HistoryState,
    team: str,
    venue: str | None = None,
) -> dict[str, float]:
    matches = state.team_matches.get(team, [])
    if venue is not None:
        matches = [match for match in matches if match.venue == venue]
    recent = matches[-FORM_WINDOW:]

    if not recent:
        return {
            "games": 0,
            "points": 1.0,
            "goals_for": 1.25,
            "goals_against": 1.25,
            "win_rate": 1 / 3,
            "draw_rate": 1 / 3,
            "btts_rate": 0.5,
            "over_25_rate": 0.5,
            "clean_sheet_rate": 0.25,
            "failed_to_score_rate": 0.25,
        }

    return {
        "games": len(recent),
        "points": _mean([match.points for match in recent], 1.0),
        "goals_for": _mean([match.goals_for for match in recent], 1.25),
        "goals_against": _mean(
            [match.goals_against for match in recent], 1.25
        ),
        "win_rate": _mean([match.points == 3 for match in recent], 1 / 3),
        "draw_rate": _mean([match.points == 1 for match in recent], 1 / 3),
        "btts_rate": _mean(
            [match.goals_for > 0 and match.goals_against > 0 for match in recent],
            0.5,
        ),
        "over_25_rate": _mean(
            [match.goals_for + match.goals_against > 2 for match in recent],
            0.5,
        ),
        "clean_sheet_rate": _mean(
            [match.goals_against == 0 for match in recent], 0.25
        ),
        "failed_to_score_rate": _mean(
            [match.goals_for == 0 for match in recent], 0.25
        ),
    }


def _rest_days(state: HistoryState, team: str, fixture_date: date) -> float:
    matches = state.team_matches.get(team, [])
    if not matches:
        return 14.0
    return float(min(max((fixture_date - matches[-1].date).days, 0), 30))


def build_upcoming_features(
    state: HistoryState,
    home_team: str,
    away_team: str,
    fixture_date: date,
) -> dict[str, float]:
    """Build a feature row using only results already present in ``state``."""
    home = _team_form(state, home_team)
    away = _team_form(state, away_team)
    home_venue = _team_form(state, home_team, "home")
    away_venue = _team_form(state, away_team, "away")
    home_elo = state.elo.get(home_team, INITIAL_ELO)
    away_elo = state.elo.get(away_team, INITIAL_ELO)
    league_games = state.league_matches

    return {
        "home_points_5": home["points"],
        "away_points_5": away["points"],
        "home_goals_for_5": home["goals_for"],
        "away_goals_for_5": away["goals_for"],
        "home_goals_against_5": home["goals_against"],
        "away_goals_against_5": away["goals_against"],
        "home_win_rate_5": home["win_rate"],
        "away_win_rate_5": away["win_rate"],
        "home_draw_rate_5": home["draw_rate"],
        "away_draw_rate_5": away["draw_rate"],
        "home_btts_rate_5": home["btts_rate"],
        "away_btts_rate_5": away["btts_rate"],
        "home_over_25_rate_5": home["over_25_rate"],
        "away_over_25_rate_5": away["over_25_rate"],
        "home_clean_sheet_rate_5": home["clean_sheet_rate"],
        "away_clean_sheet_rate_5": away["clean_sheet_rate"],
        "home_failed_to_score_rate_5": home["failed_to_score_rate"],
        "away_failed_to_score_rate_5": away["failed_to_score_rate"],
        "home_venue_points_5": home_venue["points"],
        "away_venue_points_5": away_venue["points"],
        "home_venue_goals_for_5": home_venue["goals_for"],
        "away_venue_goals_for_5": away_venue["goals_for"],
        "home_venue_goals_against_5": home_venue["goals_against"],
        "away_venue_goals_against_5": away_venue["goals_against"],
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo + HOME_ELO_ADVANTAGE - away_elo,
        "home_rest_days": _rest_days(state, home_team, fixture_date),
        "away_rest_days": _rest_days(state, away_team, fixture_date),
        "league_home_goals": (
            state.league_home_goals / league_games if league_games else 1.4
        ),
        "league_away_goals": (
            state.league_away_goals / league_games if league_games else 1.1
        ),
        "league_btts_rate": (
            state.league_btts / league_games if league_games else 0.5
        ),
    }


def _actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _update_state(state: HistoryState, row) -> None:
    home_goals = int(row.full_time_home_goals)
    away_goals = int(row.full_time_away_goals)
    result = _actual_result(home_goals, away_goals)
    home_points = 3 if result == "H" else 1 if result == "D" else 0
    away_points = 3 if result == "A" else 1 if result == "D" else 0

    state.team_matches[row.home_team].append(
        TeamMatch(row.date, "home", home_points, home_goals, away_goals)
    )
    state.team_matches[row.away_team].append(
        TeamMatch(row.date, "away", away_points, away_goals, home_goals)
    )

    home_elo = state.elo.get(row.home_team, INITIAL_ELO)
    away_elo = state.elo.get(row.away_team, INITIAL_ELO)
    expected_home = 1 / (
        1 + 10 ** ((away_elo - home_elo - HOME_ELO_ADVANTAGE) / 400)
    )
    actual_home = 1.0 if result == "H" else 0.5 if result == "D" else 0.0
    change = ELO_K_FACTOR * (actual_home - expected_home)
    state.elo[row.home_team] = home_elo + change
    state.elo[row.away_team] = away_elo - change

    state.league_matches += 1
    state.league_home_goals += home_goals
    state.league_away_goals += away_goals
    state.league_btts += int(home_goals > 0 and away_goals > 0)


def build_training_data(
    matches: pd.DataFrame,
    cutoff_date: date,
) -> tuple[pd.DataFrame, HistoryState]:
    """Create chronological training rows and state strictly before cutoff."""
    complete = matches[
        matches["full_time_home_goals"].notna()
        & matches["full_time_away_goals"].notna()
    ].copy()
    complete["date"] = pd.to_datetime(complete["date"]).dt.date
    complete = complete[complete["date"] < cutoff_date]
    complete = complete.sort_values(["date", "id"], kind="stable")

    state = HistoryState()
    training_rows: list[dict] = []
    for _, date_group in complete.groupby("date", sort=True):
        pending = []
        for row in date_group.itertuples(index=False):
            features = build_upcoming_features(
                state, row.home_team, row.away_team, row.date
            )
            home_goals = int(row.full_time_home_goals)
            away_goals = int(row.full_time_away_goals)
            training_rows.append(
                {
                    **features,
                    "match_date": row.date,
                    "target_result": _actual_result(home_goals, away_goals),
                    "target_btts": int(home_goals > 0 and away_goals > 0),
                    "target_over_25": int(home_goals + away_goals > 2),
                    "target_home_scores": int(home_goals > 0),
                    "target_away_scores": int(away_goals > 0),
                }
            )
            pending.append(row)

        # A date is one information batch: no match sees another result that day.
        for row in pending:
            _update_state(state, row)

    return pd.DataFrame(training_rows), state


def _load_matches(competition: str) -> pd.DataFrame:
    session = get_session()
    try:
        return pd.read_sql(
            text(
                """
                SELECT id, date, home_team, away_team,
                       full_time_home_goals, full_time_away_goals
                FROM matches
                WHERE competition = :competition
                ORDER BY date, id
                """
            ),
            session.bind,
            params={"competition": competition},
        )
    finally:
        session.close()


def _resolve_team(team: str, available: set[str]) -> str:
    exact = {name.casefold(): name for name in available}
    resolved = exact.get(team.casefold())
    if resolved:
        return resolved
    partial = sorted(name for name in available if team.casefold() in name.casefold())
    if len(partial) == 1:
        return partial[0]
    examples = ", ".join(sorted(available)[:12])
    raise ValueError(f"Unknown or ambiguous team '{team}'. Known examples: {examples}")


def _model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def _class_probabilities(model: Pipeline, fixture: pd.DataFrame) -> dict:
    values = model.predict_proba(fixture)[0]
    return {
        str(label): float(probability)
        for label, probability in zip(model.classes_, values)
    }


def predict_upcoming_match(
    home_team: str,
    away_team: str,
    fixture_date: date,
    competition: str = "I1",
) -> dict:
    matches = _load_matches(competition.upper())
    if matches.empty:
        raise ValueError(f"No historical matches found for competition {competition}.")

    dated = matches.copy()
    dated["date"] = pd.to_datetime(dated["date"]).dt.date
    past = dated[dated["date"] < fixture_date]
    available = set(past["home_team"]) | set(past["away_team"])
    home_team = _resolve_team(home_team, available)
    away_team = _resolve_team(away_team, available)
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    training, state = build_training_data(matches, fixture_date)
    if len(training) < 100:
        raise ValueError(
            f"Only {len(training)} completed matches are available before "
            f"{fixture_date}; at least 100 are required."
        )

    fixture_values = build_upcoming_features(
        state, home_team, away_team, fixture_date
    )
    fixture = pd.DataFrame([fixture_values], columns=FEATURE_COLUMNS)
    probabilities = {}
    validation_accuracy = {}
    split_index = int(len(training) * 0.8)
    validation_train = training.iloc[:split_index]
    validation_test = training.iloc[split_index:]
    for name, target in TARGET_COLUMNS.items():
        if training[target].nunique() < 2:
            raise ValueError(f"Training target {target} contains only one class.")
        validation_model = _model()
        validation_model.fit(
            validation_train[FEATURE_COLUMNS], validation_train[target]
        )
        validation_accuracy[name] = float(
            accuracy_score(
                validation_test[target],
                validation_model.predict(validation_test[FEATURE_COLUMNS]),
            )
        )
        model = _model()
        model.fit(training[FEATURE_COLUMNS], training[target])
        probabilities[name] = _class_probabilities(model, fixture)

    result_probabilities = probabilities["result"]
    home_win = result_probabilities.get("H", 0.0)
    draw = result_probabilities.get("D", 0.0)
    away_win = result_probabilities.get("A", 0.0)
    complete_dates = dated[
        dated["full_time_home_goals"].notna()
        & dated["full_time_away_goals"].notna()
        & (dated["date"] < fixture_date)
    ]["date"]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "fixture_date": fixture_date.isoformat(),
        "competition": competition.upper(),
        "training_matches": len(training),
        "data_through": max(complete_dates).isoformat(),
        "home_history_matches": len(state.team_matches.get(home_team, [])),
        "away_history_matches": len(state.team_matches.get(away_team, [])),
        "validation_accuracy": validation_accuracy,
        "form": {
            "home": _team_form(state, home_team),
            "away": _team_form(state, away_team),
        },
        "probabilities": {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "home_or_draw": home_win + draw,
            "draw_or_away": draw + away_win,
            "home_or_away": home_win + away_win,
            "btts_yes": probabilities["btts"].get("1", 0.0),
            "btts_no": probabilities["btts"].get("0", 0.0),
            "over_25": probabilities["over_25"].get("1", 0.0),
            "under_25": probabilities["over_25"].get("0", 0.0),
            "home_scores": probabilities["home_scores"].get("1", 0.0),
            "away_scores": probabilities["away_scores"].get("1", 0.0),
        },
    }
