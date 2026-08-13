from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from src.database.connection import get_session
from src.database.models import Match, MatchFeatures


def _points_for_team(match: Match, team: str) -> int:
    if match.full_time_result is None:
        return 0

    if match.home_team == team:
        if match.full_time_result == "H":
            return 3
        if match.full_time_result == "D":
            return 1
        return 0

    if match.away_team == team:
        if match.full_time_result == "A":
            return 3
        if match.full_time_result == "D":
            return 1
        return 0

    return 0


def _goals_scored(match: Match, team: str) -> int:
    if match.home_team == team:
        return match.full_time_home_goals or 0
    if match.away_team == team:
        return match.full_time_away_goals or 0
    return 0


def _goals_conceded(match: Match, team: str) -> int:
    if match.home_team == team:
        return match.full_time_away_goals or 0
    if match.away_team == team:
        return match.full_time_home_goals or 0
    return 0


def _result_for_team(match: Match, team: str) -> str:
    points = _points_for_team(match, team)

    if points == 3:
        return "W"
    if points == 1:
        return "D"
    return "L"


def _recent_matches_for_team(all_previous_matches: list[Match], team: str, limit: int = 5) -> list[Match]:
    team_matches = [
        match for match in all_previous_matches
        if match.home_team == team or match.away_team == team
    ]

    return team_matches[-limit:]


def _safe_avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _team_recent_features(previous_matches: list[Match], team: str, limit: int = 5) -> dict:
    recent = _recent_matches_for_team(previous_matches, team, limit)

    if not recent:
        return {
            "points": None,
            "goals_scored": None,
            "goals_conceded": None,
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
        }

    points = [_points_for_team(match, team) for match in recent]
    goals_scored = [_goals_scored(match, team) for match in recent]
    goals_conceded = [_goals_conceded(match, team) for match in recent]
    results = [_result_for_team(match, team) for match in recent]

    return {
        "points": _safe_avg(points),
        "goals_scored": _safe_avg(goals_scored),
        "goals_conceded": _safe_avg(goals_conceded),
        "win_rate": results.count("W") / len(results),
        "draw_rate": results.count("D") / len(results),
        "loss_rate": results.count("L") / len(results),
    }


def _implied_probabilities(odds_home: float | None, odds_draw: float | None, odds_away: float | None) -> dict:
    if not odds_home or not odds_draw or not odds_away:
        return {
            "home": None,
            "draw": None,
            "away": None,
            "margin": None,
        }

    raw_home = 1 / odds_home
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away

    margin = raw_home + raw_draw + raw_away

    if margin <= 0:
        return {
            "home": None,
            "draw": None,
            "away": None,
            "margin": None,
        }

    return {
        "home": raw_home / margin,
        "draw": raw_draw / margin,
        "away": raw_away / margin,
        "margin": margin - 1,
    }


def build_basic_features(clear_existing: bool = True) -> int:
    session = get_session()

    if clear_existing:
        session.execute(delete(MatchFeatures))
        session.commit()

    matches = session.execute(
        select(Match).order_by(Match.date.asc(), Match.id.asc())
    ).scalars().all()

    print(f"Loaded {len(matches)} matches from DB")

    previous_matches: list[Match] = []
    feature_rows: list[MatchFeatures] = []
    created_count = 0

    for index, match in enumerate(matches, start=1):
        if index % 500 == 0:
            print(f"Processed {index}/{len(matches)} matches...")

        if match.full_time_result not in ("H", "D", "A"):
            previous_matches.append(match)
            continue

        home_features = _team_recent_features(previous_matches, match.home_team, limit=5)
        away_features = _team_recent_features(previous_matches, match.away_team, limit=5)

        implied = _implied_probabilities(
            match.odds_home_win,
            match.odds_draw,
            match.odds_away_win,
        )

        feature_row = MatchFeatures(
            match_id=match.id,
            season=match.season,
            date=match.date,
            home_team=match.home_team,
            away_team=match.away_team,

            home_points_last_5=home_features["points"],
            away_points_last_5=away_features["points"],

            home_goals_scored_last_5=home_features["goals_scored"],
            away_goals_scored_last_5=away_features["goals_scored"],

            home_goals_conceded_last_5=home_features["goals_conceded"],
            away_goals_conceded_last_5=away_features["goals_conceded"],

            home_win_rate_last_5=home_features["win_rate"],
            away_win_rate_last_5=away_features["win_rate"],

            home_draw_rate_last_5=home_features["draw_rate"],
            away_draw_rate_last_5=away_features["draw_rate"],

            home_loss_rate_last_5=home_features["loss_rate"],
            away_loss_rate_last_5=away_features["loss_rate"],

            odds_home_win=match.odds_home_win,
            odds_draw=match.odds_draw,
            odds_away_win=match.odds_away_win,

            implied_probability_home=implied["home"],
            implied_probability_draw=implied["draw"],
            implied_probability_away=implied["away"],

            bookmaker_margin=implied["margin"],

            target_result=match.full_time_result,
        )

        feature_rows.append(feature_row)
        created_count += 1
        previous_matches.append(match)

    print(f"Saving {len(feature_rows)} feature rows to DB...")

    session.bulk_save_objects(feature_rows)
    session.commit()
    session.close()

    print(f"Created {created_count} feature rows")
    return created_count