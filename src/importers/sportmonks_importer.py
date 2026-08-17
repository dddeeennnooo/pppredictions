import json
import os
from datetime import date, datetime, timedelta, timezone

import requests
from sqlalchemy import select

from src.database.connection import Base, engine, get_session
from src.database.models import ProviderFixtureData, ProviderTeamEvent


BASE_URL = "https://api.sportmonks.com/v3/football"
BTTS_MARKET_ID = 14


def _token() -> str:
    token = os.getenv("SPORTMONKS_API_TOKEN")
    if not token:
        raise RuntimeError(
            "SPORTMONKS_API_TOKEN is not configured. Add your Sportmonks API "
            "token as an environment variable, then run this command again."
        )
    return token


def _request(path: str, params: dict | None = None) -> dict:
    request_params = dict(params or {})
    request_params["api_token"] = _token()
    response = requests.get(
        f"{BASE_URL}/{path.lstrip('/')}",
        params=request_params,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Sportmonks API error: {payload['error']}")
    return payload


def _pages(path: str, params: dict | None = None):
    page = 1
    while True:
        page_params = dict(params or {})
        page_params.update({"page": page, "per_page": 50})
        payload = _request(path, page_params)
        yield from payload.get("data", [])
        pagination = payload.get("pagination", {})
        if not pagination.get("has_more"):
            break
        page += 1


def _participant(fixture: dict, location: str) -> dict | None:
    for participant in fixture.get("participants", []):
        meta = participant.get("meta") or {}
        if meta.get("location") == location:
            return participant
    return None


def _xg(fixture: dict, location: str) -> float | None:
    for expected in fixture.get("expected", fixture.get("xGFixture", [])) or []:
        if expected.get("location") == location:
            value = (expected.get("data") or {}).get("value")
            return float(value) if value is not None else None
    return None


def _lineup_ids(fixture: dict, team_id: int | None) -> list[int]:
    if team_id is None:
        return []
    return sorted(
        int(item["player_id"])
        for item in fixture.get("lineups", []) or []
        if item.get("team_id") == team_id
        and item.get("type_id") == 11
        and item.get("player_id") is not None
    )


def _sidelined_count(fixture: dict, team_id: int | None) -> int | None:
    sidelined = fixture.get("sidelined")
    if sidelined is None or team_id is None:
        return None
    return sum(
        1
        for item in sidelined
        if item.get("participant_id", item.get("team_id")) == team_id
    )


def _sidelined_ids(fixture: dict, team_id: int | None) -> list[int]:
    if team_id is None:
        return []
    player_ids = []
    for item in fixture.get("sidelined", []) or []:
        if item.get("participant_id", item.get("team_id")) != team_id:
            continue
        player_id = item.get("player_id")
        if player_id is None and isinstance(item.get("sideline"), dict):
            player_id = item["sideline"].get("player_id")
        if player_id is not None:
            player_ids.append(int(player_id))
    return sorted(set(player_ids))


def _coach_id(fixture: dict, team_id: int | None) -> str | None:
    if team_id is None:
        return None
    for item in fixture.get("coaches", []) or []:
        participant_id = item.get(
            "participant_id",
            item.get("team_id"),
        )
        if participant_id != team_id:
            continue
        value = item.get("coach_id", item.get("id"))
        if value is not None:
            return str(value)
    return None


def _first_scalar(value):
    if isinstance(value, dict):
        for item in value.values():
            found = _first_scalar(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_scalar(item)
            if found is not None:
                return found
    elif value is not None:
        return value
    return None


def _find_weather_value(value, names: set[str]):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in names:
                return _first_scalar(item)
        for item in value.values():
            found = _find_weather_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_weather_value(item, names)
            if found is not None:
                return found
    return None


def _weather_number(weather, names: set[str]) -> float | None:
    value = _find_weather_value(weather, names)
    try:
        return float(str(value).replace("%", "").split()[0])
    except (TypeError, ValueError):
        return None


def _weather_values(fixture: dict) -> dict:
    weather = fixture.get("weatherreport", fixture.get("weatherReport")) or {}
    code = _find_weather_value(
        weather,
        {"code", "weather_code", "description", "weather_description"},
    )
    return {
        "temperature_c": _weather_number(
            weather,
            {"temperature", "temperature_c", "temp", "temp_c"},
        ),
        "feels_like_c": _weather_number(
            weather,
            {"feels_like", "feelslike", "feels_like_c"},
        ),
        "humidity_percent": _weather_number(
            weather,
            {"humidity", "humidity_percent"},
        ),
        "wind_speed_kph": _weather_number(
            weather,
            {"wind_speed", "wind_speed_kph", "windspeed"},
        ),
        "precipitation_mm": _weather_number(
            weather,
            {"precipitation", "precipitation_mm", "rain", "rain_mm"},
        ),
        "weather_code": str(code) if code is not None else None,
    }


def _average_btts_odds(fixture_id: int) -> tuple[float | None, float | None]:
    values: dict[str, list[float]] = {"yes": [], "no": []}
    path = f"odds/pre-match/fixtures/{fixture_id}/markets/{BTTS_MARKET_ID}"
    for odd in _pages(path):
        label = str(odd.get("label", odd.get("name", ""))).strip().lower()
        try:
            value = float(odd["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if label in values and value > 1:
            values[label].append(value)
    yes = sum(values["yes"]) / len(values["yes"]) if values["yes"] else None
    no = sum(values["no"]) / len(values["no"]) if values["no"] else None
    return yes, no


def _date_chunks(start_date: date, end_date: date):
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=99), end_date)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def import_sportmonks_enrichment(
    start_date: date,
    end_date: date,
    league_ids: set[int] | None = None,
    fetch_btts_odds: bool = True,
) -> dict:
    """Import optional xG, lineup, injuries and BTTS odds enrichment."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date.")

    Base.metadata.create_all(bind=engine)
    session = get_session()
    imported = 0
    updated = 0
    skipped = 0
    odds_calls = 0

    try:
        for chunk_start, chunk_end in _date_chunks(start_date, end_date):
            path = f"fixtures/between/{chunk_start.isoformat()}/{chunk_end.isoformat()}"
            params = {
                "include": (
                    "participants;xGFixture;lineups;sidelined;coaches;weatherReport"
                ),
            }
            for fixture in _pages(path, params):
                if league_ids and fixture.get("league_id") not in league_ids:
                    continue

                home = _participant(fixture, "home")
                away = _participant(fixture, "away")
                if not home or not away:
                    skipped += 1
                    continue

                fixture_id = str(fixture["id"])
                existing = session.execute(
                    select(ProviderFixtureData).where(
                        ProviderFixtureData.provider == "sportmonks",
                        ProviderFixtureData.provider_fixture_id == fixture_id,
                    )
                ).scalar_one_or_none()

                yes_odds = existing.btts_yes_odds if existing else None
                no_odds = existing.btts_no_odds if existing else None
                if fetch_btts_odds:
                    yes_odds, no_odds = _average_btts_odds(int(fixture_id))
                    odds_calls += 1

                home_id = home.get("id")
                away_id = away.get("id")
                home_lineup = _lineup_ids(fixture, home_id)
                away_lineup = _lineup_ids(fixture, away_id)
                home_sidelined = _sidelined_ids(fixture, home_id)
                away_sidelined = _sidelined_ids(fixture, away_id)
                starting_at = str(fixture["starting_at"])
                fixture_date = datetime.fromisoformat(
                    starting_at.replace("Z", "+00:00")
                ).date()

                values = {
                    "date": fixture_date,
                    "competition_id": str(fixture.get("league_id", "")) or None,
                    "season_id": str(fixture.get("season_id", "")) or None,
                    "home_team": home["name"],
                    "away_team": away["name"],
                    "home_team_id": str(home_id) if home_id is not None else None,
                    "away_team_id": str(away_id) if away_id is not None else None,
                    "home_xg": _xg(fixture, "home"),
                    "away_xg": _xg(fixture, "away"),
                    "btts_yes_odds": yes_odds,
                    "btts_no_odds": no_odds,
                    "home_starters": len(home_lineup) or None,
                    "away_starters": len(away_lineup) or None,
                    "home_sidelined": _sidelined_count(fixture, home_id),
                    "away_sidelined": _sidelined_count(fixture, away_id),
                    "home_sidelined_player_ids": (
                        json.dumps(home_sidelined) if home_sidelined else None
                    ),
                    "away_sidelined_player_ids": (
                        json.dumps(away_sidelined) if away_sidelined else None
                    ),
                    "home_lineup_player_ids": (
                        json.dumps(home_lineup) if home_lineup else None
                    ),
                    "away_lineup_player_ids": (
                        json.dumps(away_lineup) if away_lineup else None
                    ),
                    "home_coach_id": _coach_id(fixture, home_id),
                    "away_coach_id": _coach_id(fixture, away_id),
                    "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }
                values.update(_weather_values(fixture))

                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                    updated += 1
                else:
                    session.add(
                        ProviderFixtureData(
                            provider="sportmonks",
                            provider_fixture_id=fixture_id,
                            **values,
                        )
                    )
                    imported += 1

                if (imported + updated) % 100 == 0:
                    session.commit()
        session.commit()
    finally:
        session.close()

    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "odds_calls": odds_calls,
    }


def import_sportmonks_transfers(start_date: date, end_date: date) -> dict:
    """Import completed player transfers as dated team arrival/departure events."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date.")

    Base.metadata.create_all(bind=engine)
    session = get_session()
    imported = 0
    skipped = 0
    chunk_start = start_date

    try:
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=30), end_date)
            path = f"transfers/between/{chunk_start.isoformat()}/{chunk_end.isoformat()}"
            for transfer in _pages(path, {"order": "asc"}):
                if transfer.get("completed") is False:
                    skipped += 1
                    continue
                transfer_id = str(transfer["id"])
                event_date = date.fromisoformat(str(transfer["date"])[:10])
                player_id = transfer.get("player_id")
                from_team = transfer.get("from_team_id")
                to_team = transfer.get("to_team_id")
                event_specs = []
                if from_team is not None:
                    event_specs.append((from_team, "transfer_out", to_team))
                if to_team is not None:
                    event_specs.append((to_team, "transfer_in", from_team))

                for team_id, event_type, related_team_id in event_specs:
                    provider_event_id = transfer_id
                    existing = session.execute(
                        select(ProviderTeamEvent).where(
                            ProviderTeamEvent.provider == "sportmonks",
                            ProviderTeamEvent.provider_event_id == provider_event_id,
                            ProviderTeamEvent.team_id == str(team_id),
                            ProviderTeamEvent.event_type == event_type,
                        )
                    ).scalar_one_or_none()
                    if existing:
                        skipped += 1
                        continue
                    session.add(
                        ProviderTeamEvent(
                            provider="sportmonks",
                            provider_event_id=provider_event_id,
                            event_date=event_date,
                            team_id=str(team_id),
                            event_type=event_type,
                            player_id=(str(player_id) if player_id is not None else None),
                            related_team_id=(
                                str(related_team_id)
                                if related_team_id is not None
                                else None
                            ),
                            details=json.dumps(
                                {
                                    "type_id": transfer.get("type_id"),
                                    "position_id": transfer.get("position_id"),
                                    "amount": transfer.get("amount"),
                                }
                            ),
                            fetched_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        )
                    )
                    imported += 1
                if imported % 250 == 0:
                    session.commit()
            session.commit()
            chunk_start = chunk_end + timedelta(days=1)
    finally:
        session.close()

    return {"imported": imported, "skipped": skipped}
