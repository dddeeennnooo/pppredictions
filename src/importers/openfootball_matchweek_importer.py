from __future__ import annotations

from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
import re
import unicodedata

import requests
from sqlalchemy import select

from src.database.connection import get_session
from src.database.models import Match


OPENFOOTBALL_BASE_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
)

# football-data.co.uk competition code -> OpenFootball JSON filename.
OPENFOOTBALL_COMPETITIONS = {
    "I1": "it.1",
    "E0": "en.1",
    "D1": "de.1",
    "SP1": "es.1",
    "F1": "fr.1",
}

_CLUB_WORDS = {
    "ac",
    "afc",
    "as",
    "bc",
    "cf",
    "cfc",
    "fc",
    "sc",
    "ss",
    "ssc",
    "sv",
    "us",
}

_ALIASES = {
    "athletic club": "ath bilbao",
    "club atletico de madrid": "ath madrid",
    "internazionale milano": "inter",
    "internazionale": "inter",
    "manchester city": "man city",
    "manchester united": "man united",
    "nottingham forest": "nottm forest",
    "paris saint germain": "paris sg",
    "rcd espanyol de barcelona": "espanol",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
}


def _season_slug(season: str) -> str:
    start, end = season.split("/", maxsplit=1)
    return f"{start}-{end[-2:]}"


def _match_week(value: object) -> int | None:
    match = re.search(r"(\d+)$", str(value or "").strip())
    return int(match.group(1)) if match else None


def _normalise_team(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_name.casefold().replace("&", " and "))
    words = [word for word in words if word not in _CLUB_WORDS and not word.isdigit()]
    normalised = " ".join(words)
    return _ALIASES.get(normalised, normalised)


def _name_similarity(left: object, right: object) -> float:
    left_name = _normalise_team(left)
    right_name = _normalise_team(right)
    if left_name == right_name:
        return 1.0
    if left_name in right_name or right_name in left_name:
        return 0.9
    return SequenceMatcher(None, left_name, right_name).ratio()


def _fixture_similarity(payload: dict, match: Match) -> float:
    return (
        _name_similarity(payload.get("team1"), match.home_team)
        + _name_similarity(payload.get("team2"), match.away_team)
    ) / 2


def _score(payload: dict) -> tuple[int | None, int | None]:
    score = payload.get("score")
    full_time = score.get("ft") if isinstance(score, dict) else score
    if not isinstance(full_time, list) or len(full_time) != 2:
        return None, None
    return full_time[0], full_time[1]


def _fetch_season(competition: str, season: str) -> dict | None:
    filename = OPENFOOTBALL_COMPETITIONS[competition]
    url = f"{OPENFOOTBALL_BASE_URL}/{_season_slug(season)}/{filename}.json"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def import_openfootball_match_weeks(
    competitions: set[str] | None = None,
) -> dict[str, object]:
    """Fetch OpenFootball round labels and attach them to database matches."""
    selected = competitions or set(OPENFOOTBALL_COMPETITIONS)
    unknown = selected - set(OPENFOOTBALL_COMPETITIONS)
    if unknown:
        raise ValueError(f"Unsupported competitions: {', '.join(sorted(unknown))}")

    session = get_session()
    matched = 0
    unchanged = 0
    unmatched: list[dict[str, object]] = []
    unavailable: list[str] = []
    downloaded: list[str] = []

    try:
        season_rows = session.execute(
            select(Match.competition, Match.season)
            .where(Match.competition.in_(selected))
            .distinct()
            .order_by(Match.competition, Match.season)
        ).all()

        for competition, season in season_rows:
            payload = _fetch_season(competition, season)
            label = f"{competition} {season}"
            if payload is None:
                unavailable.append(label)
                continue
            downloaded.append(label)

            db_matches = session.execute(
                select(Match).where(
                    Match.competition == competition,
                    Match.season == season,
                )
            ).scalars().all()

            candidates: dict[tuple[date, int | None, int | None], list[Match]] = (
                defaultdict(list)
            )
            for db_match in db_matches:
                candidates[
                    (
                        db_match.date,
                        db_match.full_time_home_goals,
                        db_match.full_time_away_goals,
                    )
                ].append(db_match)

            fixtures = []
            for fixture in payload.get("matches", []):
                week = _match_week(fixture.get("round"))
                try:
                    fixture_date = date.fromisoformat(str(fixture.get("date")))
                except ValueError:
                    continue
                if week is None:
                    continue
                home_goals, away_goals = _score(fixture)
                possible = candidates.get(
                    (fixture_date, home_goals, away_goals), []
                )
                fixtures.append((fixture, week, possible))

            used_ids: set[int] = set()
            # Assign least-ambiguous fixtures first. Within a tied score/date
            # group, the best normalised home/away team-name pair wins.
            fixtures.sort(
                key=lambda item: (
                    len(item[2]) if item[2] else 999,
                    -max(
                        (_fixture_similarity(item[0], match) for match in item[2]),
                        default=0,
                    ),
                )
            )
            for fixture, week, possible in fixtures:
                possible = [match for match in possible if match.id not in used_ids]
                if not possible:
                    possible = [
                        match
                        for match in db_matches
                        if match.id not in used_ids
                        and _fixture_similarity(fixture, match) >= 0.9
                    ]
                if not possible:
                    home_goals, away_goals = _score(fixture)
                    unmatched.append(
                        {
                            "competition": competition,
                            "season": season,
                            "date": fixture.get("date"),
                            "home_team": fixture.get("team1"),
                            "away_team": fixture.get("team2"),
                            "score": f"{home_goals}-{away_goals}",
                            "match_week": week,
                        }
                    )
                    continue

                best = max(possible, key=lambda match: _fixture_similarity(fixture, match))
                used_ids.add(best.id)
                if best.match_week == week:
                    unchanged += 1
                else:
                    best.match_week = week
                    matched += 1

            session.commit()
    finally:
        session.close()

    return {
        "downloaded": downloaded,
        "unavailable": unavailable,
        "matched": matched,
        "unchanged": unchanged,
        "unmatched": unmatched,
    }
