import math
import pickle
import json
import re
import unicodedata
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, text

from src.config import BASE_DIR
from src.database.connection import Base, engine, get_session
from src.database.models import ProviderFixtureData, ProviderTeamEvent


MODEL_DIR = BASE_DIR / "artifacts" / "models"
MODEL_PATH = MODEL_DIR / "advanced_btts_model.pkl"
WINDOWS = (5, 10, 20)


def _mean(records: list[dict], key: str) -> float:
    values = [record[key] for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else np.nan


def _nan_aggregate(values: list[float], operation: str) -> float:
    available = [value for value in values if not pd.isna(value)]
    if not available:
        return np.nan
    if operation == "min":
        return float(min(available))
    return float(np.mean(available))


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
        ):
            features[f"{prefix}_{key}_{window}"] = _mean(recent, key)


def _normalized_market_probability(over_odds, under_odds) -> float:
    if pd.isna(over_odds) or pd.isna(under_odds) or over_odds <= 0 or under_odds <= 0:
        return np.nan
    raw_over = 1 / over_odds
    raw_under = 1 / under_odds
    return raw_over / (raw_over + raw_under)


def _normalized_1x2_probabilities(home_odds, draw_odds, away_odds) -> tuple:
    odds = (home_odds, draw_odds, away_odds)
    if any(pd.isna(value) or value <= 0 for value in odds):
        return np.nan, np.nan, np.nan
    raw = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    probabilities = raw / raw.sum()
    return tuple(float(value) for value in probabilities)


def _normalized_team_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    aliases = {
        "inter milan": "inter",
        "internazionale": "inter",
        "ac milan": "milan",
        "hellas verona": "verona",
    }
    return aliases.get(normalized, normalized)


def _provider_key(match_date, home_team: str, away_team: str) -> tuple:
    return (
        str(match_date),
        _normalized_team_name(home_team),
        _normalized_team_name(away_team),
    )


def _lineup_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    try:
        return {int(player_id) for player_id in json.loads(value)}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _season_context(matches: pd.DataFrame) -> dict:
    team_sets: dict[tuple, set[str]] = defaultdict(set)
    season_start: dict[tuple, pd.Timestamp] = {}
    stats: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {"games": 0, "points": 0, "goals_for": 0, "goals_against": 0}
    )

    for row in matches.itertuples(index=False):
        season_key = (row.competition, row.season)
        team_sets[season_key].update((row.home_team, row.away_team))
        match_date = pd.Timestamp(row.date)
        if season_key not in season_start or match_date < season_start[season_key]:
            season_start[season_key] = match_date

        home_key = (*season_key, row.home_team)
        away_key = (*season_key, row.away_team)
        home_goals = int(row.full_time_home_goals)
        away_goals = int(row.full_time_away_goals)
        stats[home_key]["games"] += 1
        stats[away_key]["games"] += 1
        stats[home_key]["goals_for"] += home_goals
        stats[home_key]["goals_against"] += away_goals
        stats[away_key]["goals_for"] += away_goals
        stats[away_key]["goals_against"] += home_goals
        if home_goals > away_goals:
            stats[home_key]["points"] += 3
        elif away_goals > home_goals:
            stats[away_key]["points"] += 3
        else:
            stats[home_key]["points"] += 1
            stats[away_key]["points"] += 1

    competition_seasons: dict[str, list[str]] = defaultdict(list)
    for competition, season in sorted(
        team_sets,
        key=lambda value: season_start[value],
    ):
        competition_seasons[competition].append(season)

    previous_season: dict[tuple, str] = {}
    prior_rank: dict[tuple, float] = {}
    prior_points_per_game: dict[tuple, float] = {}
    for competition, seasons in competition_seasons.items():
        for index, season in enumerate(seasons):
            if index == 0:
                continue
            previous = seasons[index - 1]
            previous_season[(competition, season)] = previous
            ranked = sorted(
                team_sets[(competition, previous)],
                key=lambda team: (
                    stats[(competition, previous, team)]["points"],
                    stats[(competition, previous, team)]["goals_for"]
                    - stats[(competition, previous, team)]["goals_against"],
                    stats[(competition, previous, team)]["goals_for"],
                ),
                reverse=True,
            )
            denominator = max(len(ranked) - 1, 1)
            for rank_index, team in enumerate(ranked):
                team_stats = stats[(competition, previous, team)]
                key = (competition, season, team)
                prior_rank[key] = rank_index / denominator
                prior_points_per_game[key] = (
                    team_stats["points"] / team_stats["games"]
                    if team_stats["games"]
                    else np.nan
                )

    return {
        "team_sets": team_sets,
        "previous_season": previous_season,
        "prior_rank": prior_rank,
        "prior_points_per_game": prior_points_per_game,
    }


def _recent_event_count(
    events: list[ProviderTeamEvent],
    match_date: pd.Timestamp,
    days: int,
    event_type: str,
) -> int:
    lower_bound = match_date.date() - timedelta(days=days)
    return sum(
        1
        for event in events
        if event.event_type == event_type
        and lower_bound <= event.event_date < match_date.date()
    )


def _match_record(row, team_is_home: bool) -> dict:
    if team_is_home:
        goals_for = int(row.full_time_home_goals)
        goals_against = int(row.full_time_away_goals)
        shots_for = row.home_shots
        shots_against = row.away_shots
        shots_on_target_for = row.home_shots_on_target
        shots_on_target_against = row.away_shots_on_target
    else:
        goals_for = int(row.full_time_away_goals)
        goals_against = int(row.full_time_home_goals)
        shots_for = row.away_shots
        shots_against = row.home_shots
        shots_on_target_for = row.away_shots_on_target
        shots_on_target_against = row.home_shots_on_target

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
    }


def build_advanced_btts_dataset() -> pd.DataFrame:
    """Build leakage-safe pre-match BTTS features directly from match history."""
    Base.metadata.create_all(bind=engine)
    session = get_session()
    query = text("""
        SELECT
            id, competition, season, date, home_team, away_team,
            full_time_home_goals, full_time_away_goals,
            home_shots, away_shots,
            home_shots_on_target, away_shots_on_target,
            odds_home_win, odds_draw, odds_away_win,
            odds_over_25, odds_under_25
        FROM matches
        WHERE full_time_home_goals IS NOT NULL
          AND full_time_away_goals IS NOT NULL
        ORDER BY date ASC, id ASC
    """)
    try:
        matches = pd.read_sql(query, session.bind)
        provider_rows = session.execute(
            select(ProviderFixtureData).where(
                ProviderFixtureData.provider == "sportmonks"
            )
        ).scalars().all()
        provider_events = session.execute(
            select(ProviderTeamEvent).where(
                ProviderTeamEvent.provider == "sportmonks"
            )
        ).scalars().all()
    finally:
        session.close()

    provider_data = {
        _provider_key(item.date, item.home_team, item.away_team): item
        for item in provider_rows
    }
    events_by_team: dict[str, list[ProviderTeamEvent]] = defaultdict(list)
    for event in provider_events:
        events_by_team[event.team_id].append(event)
    season_context = _season_context(matches)

    team_history: dict[tuple, list[dict]] = defaultdict(list)
    home_history: dict[tuple, list[dict]] = defaultdict(list)
    away_history: dict[tuple, list[dict]] = defaultdict(list)
    head_to_head: dict[tuple, list[dict]] = defaultdict(list)
    league_history: dict[str, list[dict]] = defaultdict(list)
    season_history: dict[tuple, list[dict]] = defaultdict(list)
    last_date: dict[tuple, pd.Timestamp] = {}
    xg_history: dict[tuple, list[float]] = defaultdict(list)
    previous_lineup: dict[tuple, set[int]] = defaultdict(set)
    player_starts: dict[tuple, int] = defaultdict(int)
    current_coach: dict[tuple, str] = {}
    coach_tenure: dict[tuple, int] = defaultdict(int)
    coach_change_dates: dict[tuple, list[pd.Timestamp]] = defaultdict(list)
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        match_date = pd.Timestamp(row.date)
        home_key = (row.competition, row.home_team)
        away_key = (row.competition, row.away_team)
        season_key = (row.competition, row.season)
        features = {
            "match_id": row.id,
            "competition": row.competition,
            "season": row.season,
            "date": match_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "target_btts": int(
                row.full_time_home_goals > 0 and row.full_time_away_goals > 0
            ),
            "target_home_scored": int(row.full_time_home_goals > 0),
            "target_away_scored": int(row.full_time_away_goals > 0),
            "target_home_goals": int(row.full_time_home_goals),
            "target_away_goals": int(row.full_time_away_goals),
        }

        previous = season_context["previous_season"].get(season_key)
        home_was_present = (
            previous is not None
            and row.home_team
            in season_context["team_sets"][(row.competition, previous)]
        )
        away_was_present = (
            previous is not None
            and row.away_team
            in season_context["team_sets"][(row.competition, previous)]
        )
        features.update(
            {
                "home_promoted": (
                    int(not home_was_present) if previous is not None else np.nan
                ),
                "away_promoted": (
                    int(not away_was_present) if previous is not None else np.nan
                ),
                "either_team_promoted": (
                    int(not home_was_present or not away_was_present)
                    if previous is not None
                    else np.nan
                ),
                "home_prior_season_rank_percentile": season_context[
                    "prior_rank"
                ].get((row.competition, row.season, row.home_team), 1.05),
                "away_prior_season_rank_percentile": season_context[
                    "prior_rank"
                ].get((row.competition, row.season, row.away_team), 1.05),
                "home_prior_season_points_per_game": season_context[
                    "prior_points_per_game"
                ].get((row.competition, row.season, row.home_team), 0.0),
                "away_prior_season_points_per_game": season_context[
                    "prior_points_per_game"
                ].get((row.competition, row.season, row.away_team), 0.0),
            }
        )

        _add_history_features(features, "home", team_history[home_key])
        _add_history_features(features, "away", team_history[away_key])
        _add_history_features(
            features,
            "home_venue",
            home_history[home_key],
            windows=(5, 10),
        )
        _add_history_features(
            features,
            "away_venue",
            away_history[away_key],
            windows=(5, 10),
        )

        pair_key = (row.competition, *sorted((row.home_team, row.away_team)))
        _add_history_features(
            features,
            "head_to_head",
            head_to_head[pair_key],
            windows=(5,),
        )
        _add_history_features(
            features,
            "league",
            league_history[row.competition],
            windows=(100, 380),
        )
        features["season_matches_played"] = len(season_history[season_key])
        for window in (20, 50, 100):
            recent_season = season_history[season_key][-window:]
            features[f"season_btts_rate_{window}"] = _mean(
                recent_season,
                "btts",
            )
            features[f"season_over_25_rate_{window}"] = _mean(
                recent_season,
                "over_25",
            )

        p_home, p_draw, p_away = _normalized_1x2_probabilities(
            row.odds_home_win,
            row.odds_draw,
            row.odds_away_win,
        )
        features.update(
            {
                "market_home_probability": p_home,
                "market_draw_probability": p_draw,
                "market_away_probability": p_away,
                "market_strength_gap": abs(p_home - p_away),
                "market_over_25_probability": _normalized_market_probability(
                    row.odds_over_25,
                    row.odds_under_25,
                ),
                "has_over_25_market": int(
                    not pd.isna(row.odds_over_25)
                    and not pd.isna(row.odds_under_25)
                ),
                "home_rest_days": (
                    (match_date - last_date[home_key]).days
                    if home_key in last_date
                    else np.nan
                ),
                "away_rest_days": (
                    (match_date - last_date[away_key]).days
                    if away_key in last_date
                    else np.nan
                ),
                "month_sin": math.sin(2 * math.pi * match_date.month / 12),
                "month_cos": math.cos(2 * math.pi * match_date.month / 12),
            }
        )

        enrichment = provider_data.get(
            _provider_key(row.date, row.home_team, row.away_team)
        )
        home_lineup = _lineup_ids(
            enrichment.home_lineup_player_ids if enrichment else None
        )
        away_lineup = _lineup_ids(
            enrichment.away_lineup_player_ids if enrichment else None
        )
        home_sidelined_players = _lineup_ids(
            enrichment.home_sidelined_player_ids if enrichment else None
        )
        away_sidelined_players = _lineup_ids(
            enrichment.away_sidelined_player_ids if enrichment else None
        )
        features["has_sportmonks_enrichment"] = int(enrichment is not None)
        features["sportmonks_home_xg_5"] = (
            float(np.mean(xg_history[home_key][-5:]))
            if xg_history[home_key]
            else np.nan
        )
        features["sportmonks_away_xg_5"] = (
            float(np.mean(xg_history[away_key][-5:]))
            if xg_history[away_key]
            else np.nan
        )
        features["sportmonks_home_xg_10"] = (
            float(np.mean(xg_history[home_key][-10:]))
            if xg_history[home_key]
            else np.nan
        )
        features["sportmonks_away_xg_10"] = (
            float(np.mean(xg_history[away_key][-10:]))
            if xg_history[away_key]
            else np.nan
        )
        features["home_sidelined"] = (
            enrichment.home_sidelined if enrichment else np.nan
        )
        features["away_sidelined"] = (
            enrichment.away_sidelined if enrichment else np.nan
        )
        features["home_sidelined_prior_starts"] = (
            float(
                np.mean(
                    [
                        player_starts[(home_key, player)]
                        for player in home_sidelined_players
                    ]
                )
            )
            if home_sidelined_players
            else np.nan
        )
        features["away_sidelined_prior_starts"] = (
            float(
                np.mean(
                    [
                        player_starts[(away_key, player)]
                        for player in away_sidelined_players
                    ]
                )
            )
            if away_sidelined_players
            else np.nan
        )
        features["home_confirmed_starters"] = (
            enrichment.home_starters if enrichment else np.nan
        )
        features["away_confirmed_starters"] = (
            enrichment.away_starters if enrichment else np.nan
        )
        features["home_returning_starter_rate"] = (
            len(home_lineup & previous_lineup[home_key]) / len(home_lineup)
            if home_lineup
            else np.nan
        )
        features["away_returning_starter_rate"] = (
            len(away_lineup & previous_lineup[away_key]) / len(away_lineup)
            if away_lineup
            else np.nan
        )
        features["home_lineup_prior_starts"] = (
            float(np.mean([player_starts[(home_key, player)] for player in home_lineup]))
            if home_lineup
            else np.nan
        )
        features["away_lineup_prior_starts"] = (
            float(np.mean([player_starts[(away_key, player)] for player in away_lineup]))
            if away_lineup
            else np.nan
        )
        features["market_btts_probability"] = (
            _normalized_market_probability(
                enrichment.btts_yes_odds,
                enrichment.btts_no_odds,
            )
            if enrichment
            else np.nan
        )
        for column_name in (
            "temperature_c",
            "feels_like_c",
            "humidity_percent",
            "wind_speed_kph",
            "precipitation_mm",
        ):
            features[column_name] = (
                getattr(enrichment, column_name) if enrichment else np.nan
            )
        features["bad_weather"] = (
            int(
                (enrichment.precipitation_mm or 0) >= 2
                or (enrichment.wind_speed_kph or 0) >= 30
                or (
                    enrichment.temperature_c is not None
                    and (
                        enrichment.temperature_c <= 2
                        or enrichment.temperature_c >= 30
                    )
                )
            )
            if enrichment
            else np.nan
        )

        home_provider_id = enrichment.home_team_id if enrichment else None
        away_provider_id = enrichment.away_team_id if enrichment else None
        for prefix, provider_id in (
            ("home", home_provider_id),
            ("away", away_provider_id),
        ):
            team_events = events_by_team.get(provider_id, []) if provider_id else []
            for days in (30, 90):
                features[f"{prefix}_transfers_in_{days}"] = (
                    _recent_event_count(team_events, match_date, days, "transfer_in")
                    if provider_id
                    else np.nan
                )
                features[f"{prefix}_transfers_out_{days}"] = (
                    _recent_event_count(team_events, match_date, days, "transfer_out")
                    if provider_id
                    else np.nan
                )

        for prefix, team_key, coach_id in (
            ("home", home_key, enrichment.home_coach_id if enrichment else None),
            ("away", away_key, enrichment.away_coach_id if enrichment else None),
        ):
            known_coach = current_coach.get(team_key)
            manager_changed = bool(
                coach_id and known_coach and str(coach_id) != str(known_coach)
            )
            features[f"{prefix}_manager_changed"] = (
                int(manager_changed) if coach_id else np.nan
            )
            features[f"{prefix}_manager_tenure_matches"] = (
                coach_tenure[team_key]
                if coach_id and str(coach_id) == str(known_coach)
                else (0 if coach_id else np.nan)
            )
            features[f"{prefix}_manager_changes_90"] = (
                sum(
                    1
                    for change_date in coach_change_dates[team_key]
                    if match_date - timedelta(days=90) <= change_date < match_date
                )
                if coach_id
                else np.nan
            )

        for window in WINDOWS:
            home_gf = features[f"home_goals_for_{window}"]
            home_ga = features[f"home_goals_against_{window}"]
            away_gf = features[f"away_goals_for_{window}"]
            away_ga = features[f"away_goals_against_{window}"]
            features[f"expected_home_goals_{window}"] = _nan_aggregate(
                [home_gf, away_ga],
                "mean",
            )
            features[f"expected_away_goals_{window}"] = _nan_aggregate(
                [away_gf, home_ga],
                "mean",
            )
            features[f"combined_btts_rate_{window}"] = _nan_aggregate(
                [
                    features[f"home_btts_{window}"],
                    features[f"away_btts_{window}"],
                ],
                "mean",
            )
            features[f"scoring_floor_{window}"] = _nan_aggregate(
                [
                    features[f"home_scored_{window}"],
                    features[f"away_scored_{window}"],
                ],
                "min",
            )

        rows.append(features)

        home_record = _match_record(row, team_is_home=True)
        away_record = _match_record(row, team_is_home=False)
        team_history[home_key].append(home_record)
        team_history[away_key].append(away_record)
        home_history[home_key].append(home_record)
        away_history[away_key].append(away_record)
        head_to_head[pair_key].append(home_record)
        league_history[row.competition].append(home_record)
        season_history[season_key].append(home_record)
        last_date[home_key] = match_date
        last_date[away_key] = match_date
        if enrichment and enrichment.home_xg is not None:
            xg_history[home_key].append(float(enrichment.home_xg))
        if enrichment and enrichment.away_xg is not None:
            xg_history[away_key].append(float(enrichment.away_xg))
        if home_lineup:
            previous_lineup[home_key] = home_lineup
            for player in home_lineup:
                player_starts[(home_key, player)] += 1
        if away_lineup:
            previous_lineup[away_key] = away_lineup
            for player in away_lineup:
                player_starts[(away_key, player)] += 1
        for team_key, coach_id in (
            (home_key, enrichment.home_coach_id if enrichment else None),
            (away_key, enrichment.away_coach_id if enrichment else None),
        ):
            if not coach_id:
                continue
            if team_key in current_coach and str(current_coach[team_key]) != str(coach_id):
                coach_change_dates[team_key].append(match_date)
                coach_tenure[team_key] = 1
            elif str(current_coach.get(team_key)) == str(coach_id):
                coach_tenure[team_key] += 1
            else:
                coach_tenure[team_key] = 1
            current_coach[team_key] = str(coach_id)

    return pd.DataFrame(rows)


def _candidate_models(feature_columns: list[str]) -> dict:
    market_columns = [
        "market_home_probability",
        "market_draw_probability",
        "market_away_probability",
        "market_strength_gap",
        "market_over_25_probability",
        "market_btts_probability",
        "has_sportmonks_enrichment",
        "league_btts_100",
        "league_btts_380",
        "league_over_25_100",
        "league_over_25_380",
    ]
    market_columns = [
        column for column in market_columns if column in feature_columns
    ]
    context_terms = (
        "promoted",
        "prior_season",
        "season_btts_rate",
        "season_over_25_rate",
        "season_matches_played",
        "temperature",
        "feels_like",
        "humidity",
        "wind_speed",
        "precipitation",
        "bad_weather",
        "manager_",
        "transfers_",
        "sidelined",
        "lineup",
        "returning_starter",
        "confirmed_starters",
        "sportmonks_",
    )
    context_columns = list(
        dict.fromkeys(
            market_columns
            + [
                column
                for column in feature_columns
                if any(term in column for term in context_terms)
            ]
        )
    )
    core_terms = (
        "goals_for",
        "goals_against",
        "scored",
        "conceded",
        "btts",
        "clean_sheet",
        "failed_to_score",
        "total_goals",
        "over_25",
        "expected_",
        "combined_btts",
        "scoring_floor",
        "xg",
        "sidelined",
        "lineup",
        "returning_starter",
        "confirmed_starters",
        "promoted",
        "prior_season",
        "manager_",
        "transfers_",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation",
        "bad_weather",
    )
    core_columns = [
        column
        for column in feature_columns
        if column in market_columns
        or any(term in column for term in core_terms)
        and not column.startswith("head_to_head")
    ]

    logistic = lambda c: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, C=c)),
        ]
    )
    hist = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=220,
                    max_leaf_nodes=9,
                    min_samples_leaf=40,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
        ]
    )
    extra_trees = lambda depth, leaf, max_features: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=max_features,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "market_logistic": {"model": logistic(0.2), "features": market_columns},
        "context_logistic": {"model": logistic(0.1), "features": context_columns},
        "core_logistic": {"model": logistic(0.1), "features": core_columns},
        "core_hist_gradient_boosting": {"model": hist, "features": core_columns},
        "core_extra_trees": {
            "model": extra_trees(8, 12, 0.7),
            "features": core_columns,
        },
        "all_extra_trees": {
            "model": extra_trees(10, 8, 0.7),
            "features": feature_columns,
        },
    }


def _best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple:
    candidates = np.arange(0.40, 0.611, 0.01)
    scored = [
        (accuracy_score(y_true, probabilities >= threshold), float(threshold))
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))


def _best_confidence_policy(
    y_true: pd.Series,
    probabilities: np.ndarray,
    target_accuracy: float = 0.60,
    minimum_coverage: float = 0.50,
) -> dict:
    options = []
    y_values = y_true.to_numpy()
    for no_threshold in np.arange(0.25, 0.501, 0.01):
        for yes_threshold in np.arange(0.50, 0.751, 0.01):
            if no_threshold >= yes_threshold:
                continue
            resolved = (
                (probabilities <= no_threshold)
                | (probabilities >= yes_threshold)
            )
            coverage = float(resolved.mean())
            if coverage < minimum_coverage:
                continue
            predictions = (probabilities[resolved] >= yes_threshold).astype(int)
            accuracy = accuracy_score(y_values[resolved], predictions)
            options.append(
                {
                    "no_threshold": float(no_threshold),
                    "yes_threshold": float(yes_threshold),
                    "accuracy": float(accuracy),
                    "coverage": coverage,
                }
            )

    feasible = [option for option in options if option["accuracy"] >= target_accuracy]
    if feasible:
        return max(feasible, key=lambda option: (option["coverage"], option["accuracy"]))
    return max(options, key=lambda option: (option["accuracy"], option["coverage"]))


def train_advanced_btts_model() -> dict:
    df = build_advanced_btts_dataset()
    # Over/under odds begin in 2019/20. Keeping one consistent data regime avoids
    # selecting a model on features that were completely absent during training.
    df = df[df["has_over_25_market"] == 1].reset_index(drop=True)
    metadata_columns = {
        "match_id",
        "competition",
        "season",
        "date",
        "home_team",
        "away_team",
        "target_btts",
        "target_home_scored",
        "target_away_scored",
        "target_home_goals",
        "target_away_goals",
    }
    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns and df[column].notna().any()
    ]

    seasons = list(dict.fromkeys(df["season"].tolist()))
    if len(seasons) < 3:
        raise ValueError("At least three seasons with over/under odds are required.")
    test_season = seasons[-1]
    final_train_df = df[df["season"] != test_season]
    current_season_df = df[df["season"] == test_season]
    current_season_midpoint = len(current_season_df) // 2
    validation_df = current_season_df.iloc[:current_season_midpoint]
    test_df = current_season_df.iloc[current_season_midpoint:]
    y_validation = validation_df["target_btts"]
    y_test = test_df["target_btts"]

    validation_results = []
    best_name = None
    best_feature_columns = None
    best_threshold = 0.5
    best_validation_accuracy = -1.0
    best_validation_probabilities = None
    best_validation_targets = None

    for name, candidate in _candidate_models(feature_columns).items():
        model = candidate["model"]
        candidate_features = candidate["features"]
        model.fit(
            final_train_df[candidate_features],
            final_train_df["target_btts"],
        )
        probabilities = model.predict_proba(
            validation_df[candidate_features]
        )[:, 1]
        target_series = y_validation.reset_index(drop=True)
        threshold_accuracy, threshold = _best_threshold(
            target_series,
            probabilities,
        )
        default_accuracy = accuracy_score(y_validation, probabilities >= 0.5)
        validation_results.append(
            {
                "name": name,
                "default_accuracy": default_accuracy,
                "selected_accuracy": threshold_accuracy,
                "threshold": threshold,
                "feature_count": len(candidate_features),
            }
        )
        if threshold_accuracy > best_validation_accuracy:
            best_name = name
            best_feature_columns = candidate_features
            best_threshold = threshold
            best_validation_accuracy = threshold_accuracy
            best_validation_probabilities = probabilities
            best_validation_targets = target_series

    selected_candidate = _candidate_models(feature_columns)[best_name]
    final_model = selected_candidate["model"]
    final_model.fit(
        final_train_df[best_feature_columns],
        final_train_df["target_btts"],
    )
    test_probabilities = final_model.predict_proba(
        test_df[best_feature_columns]
    )[:, 1]
    global_test_predictions = (test_probabilities >= best_threshold).astype(int)
    global_test_accuracy = accuracy_score(y_test, global_test_predictions)

    competition_thresholds = {}
    validation_competitions = validation_df["competition"].reset_index(drop=True)
    validation_targets = y_validation.reset_index(drop=True)
    for competition in sorted(validation_competitions.unique()):
        mask = validation_competitions == competition
        if int(mask.sum()) < 50 or validation_targets[mask].nunique() < 2:
            competition_thresholds[competition] = best_threshold
            continue
        _, competition_threshold = _best_threshold(
            validation_targets[mask],
            best_validation_probabilities[mask.to_numpy()],
        )
        competition_thresholds[competition] = competition_threshold

    test_thresholds = np.array(
        [
            competition_thresholds.get(competition, best_threshold)
            for competition in test_df["competition"]
        ]
    )
    competition_test_predictions = (test_probabilities >= test_thresholds).astype(int)
    competition_test_accuracy = accuracy_score(y_test, competition_test_predictions)

    validation_thresholds = np.array(
        [
            competition_thresholds.get(competition, best_threshold)
            for competition in validation_competitions
        ]
    )
    competition_validation_accuracy = accuracy_score(
        validation_targets,
        best_validation_probabilities >= validation_thresholds,
    )
    use_competition_thresholds = (
        competition_validation_accuracy > best_validation_accuracy
    )
    test_predictions = (
        competition_test_predictions
        if use_competition_thresholds
        else global_test_predictions
    )
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_loss = log_loss(y_test, test_probabilities, labels=[0, 1])
    majority = int(final_train_df["target_btts"].mode().iloc[0])
    baseline_accuracy = accuracy_score(y_test, [majority] * len(y_test))
    always_no_accuracy = accuracy_score(y_test, np.zeros(len(y_test), dtype=int))
    always_yes_accuracy = accuracy_score(y_test, np.ones(len(y_test), dtype=int))
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No", "Yes"],
        zero_division=0,
    )

    confidence_policy = _best_confidence_policy(
        best_validation_targets,
        best_validation_probabilities,
    )
    confident_test_mask = (
        (test_probabilities <= confidence_policy["no_threshold"])
        | (test_probabilities >= confidence_policy["yes_threshold"])
    )
    confident_test_predictions = (
        test_probabilities[confident_test_mask]
        >= confidence_policy["yes_threshold"]
    ).astype(int)
    confident_test_accuracy = (
        accuracy_score(
            y_test.to_numpy()[confident_test_mask],
            confident_test_predictions,
        )
        if confident_test_mask.any()
        else 0.0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_columns": best_feature_columns,
        "threshold": best_threshold,
        "competition_thresholds": competition_thresholds,
        "use_competition_thresholds": use_competition_thresholds,
        "model_name": best_name,
    }
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)

    return {
        "rows_total": len(df),
        "rows_train": len(final_train_df),
        "rows_validation": len(validation_df),
        "rows_test": len(test_df),
        "test_start_date": str(test_df.iloc[0]["date"].date()),
        "test_end_date": str(test_df.iloc[-1]["date"].date()),
        "feature_count": len(feature_columns),
        "selected_feature_count": len(best_feature_columns),
        "validation_results": validation_results,
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "global_test_accuracy": global_test_accuracy,
        "competition_test_accuracy": competition_test_accuracy,
        "competition_validation_accuracy": competition_validation_accuracy,
        "competition_thresholds": competition_thresholds,
        "use_competition_thresholds": use_competition_thresholds,
        "baseline_accuracy": baseline_accuracy,
        "always_no_accuracy": always_no_accuracy,
        "always_yes_accuracy": always_yes_accuracy,
        "test_btts_rate": float(y_test.mean()),
        "test_log_loss": test_loss,
        "predicted_no": int((test_predictions == 0).sum()),
        "predicted_yes": int((test_predictions == 1).sum()),
        "confidence_no_threshold": confidence_policy["no_threshold"],
        "confidence_yes_threshold": confidence_policy["yes_threshold"],
        "confidence_validation_accuracy": confidence_policy["accuracy"],
        "confidence_validation_coverage": confidence_policy["coverage"],
        "confidence_test_accuracy": confident_test_accuracy,
        "confidence_test_coverage": float(confident_test_mask.mean()),
        "confidence_test_resolved": int(confident_test_mask.sum()),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
    }
