from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re
import unicodedata

import pandas as pd
import pyarrow.parquet as pq
import requests
from sqlalchemy import select

from src.config import RAW_DATA_DIR
from src.database.connection import Base, engine, get_session
from src.database.models import Match, OpenBttsMatchData


DATASET_BASE_URL = (
    "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main"
)
DATASET_DIR = RAW_DATA_DIR / "open_btts"
DATASET_FILES = (
    "leagues.parquet",
    "teams.parquet",
    "fixtures.parquet",
    "match_stats.parquet",
    "fixture_lineups.parquet",
    "fixture_players.parquet",
)
COMPETITIONS = {"I1", "E0", "D1", "SP1", "F1"}

_ALIASES = {
    "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid",
    "internazionale": "inter",
    "internazionale milano": "inter",
    "manchester city": "man city",
    "manchester united": "man united",
    "nottingham forest": "nottm forest",
    "paris saint germain": "paris sg",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
}


def _normalise_team(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    normalised = " ".join(re.findall(r"[a-z0-9]+", ascii_name.casefold()))
    return _ALIASES.get(normalised, normalised)


def _download_file(filename: str, refresh: bool = False) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATASET_DIR / filename
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination

    response = requests.get(f"{DATASET_BASE_URL}/{filename}", timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _safe_number(value, converter):
    if pd.isna(value):
        return None
    return converter(value)


def _safe_text(value) -> str | None:
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _starter_ids(path: Path, fixture_ids: set[int]) -> dict[tuple[int, int], list[int]]:
    starters: dict[tuple[int, int], list[int]] = defaultdict(list)
    parquet = pq.ParquetFile(path)
    columns = ["fixture_id", "team_id", "player_id", "is_starter"]
    for batch in parquet.iter_batches(columns=columns, batch_size=250_000):
        players = batch.to_pandas()
        players = players[
            players["is_starter"] & players["fixture_id"].isin(fixture_ids)
        ]
        for row in players.itertuples(index=False):
            starters[(int(row.fixture_id), int(row.team_id))].append(
                int(row.player_id)
            )
    return {
        key: sorted(set(player_ids)) for key, player_ids in starters.items()
    }


def import_open_btts_data(refresh: bool = False) -> dict[str, object]:
    """Download and match open BTTS/xG records to local completed fixtures."""
    paths = {name: _download_file(name, refresh) for name in DATASET_FILES}
    leagues = pd.read_parquet(paths["leagues.parquet"])
    leagues = leagues[leagues["fd_code"].isin(COMPETITIONS)]
    league_codes = dict(zip(leagues["id"], leagues["fd_code"]))

    fixtures = pd.read_parquet(paths["fixtures.parquet"])
    fixtures = fixtures[
        fixtures["league_id"].isin(league_codes)
        & fixtures["is_played"]
        & fixtures["goals_home"].notna()
        & fixtures["goals_away"].notna()
    ].copy()
    fixtures["competition"] = fixtures["league_id"].map(league_codes)
    fixtures["date"] = pd.to_datetime(fixtures["date_utc"]).dt.date

    teams = pd.read_parquet(paths["teams.parquet"])[["id", "name", "fd_name"]]
    team_names = {
        int(row.id): row.fd_name if pd.notna(row.fd_name) else row.name
        for row in teams.itertuples(index=False)
    }
    fixtures["home_team"] = fixtures["home_team_id"].map(team_names)
    fixtures["away_team"] = fixtures["away_team_id"].map(team_names)

    stats_columns = [
        "fixture_id",
        "home_xg",
        "away_xg",
        "home_shots_inside_box",
        "away_shots_inside_box",
        "home_shots_outside_box",
        "away_shots_outside_box",
        "home_blocked_shots",
        "away_blocked_shots",
        "home_penalties",
        "away_penalties",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "home_possession",
        "away_possession",
        "home_fouls",
        "away_fouls",
        "home_offsides",
        "away_offsides",
        "home_pass_accuracy",
        "away_pass_accuracy",
    ]
    stats = pd.read_parquet(paths["match_stats.parquet"], columns=stats_columns)
    stats = stats[stats["fixture_id"].isin(fixtures["id"])]
    fixtures = fixtures.merge(
        stats,
        left_on="id",
        right_on="fixture_id",
        how="left",
        suffixes=("", "_stats"),
    )

    lineups = pd.read_parquet(paths["fixture_lineups.parquet"])
    lineups = lineups[lineups["fixture_id"].isin(fixtures["id"])].drop_duplicates(
        ["fixture_id", "team_id"], keep="last"
    )
    home_lineups = lineups.rename(
        columns={
            "fixture_id": "lineup_fixture_id",
            "team_id": "home_lineup_team_id",
            "coach_name": "home_coach_name",
            "formation": "home_formation",
        }
    )[
        [
            "lineup_fixture_id",
            "home_lineup_team_id",
            "home_coach_name",
            "home_formation",
        ]
    ]
    away_lineups = lineups.rename(
        columns={
            "fixture_id": "lineup_fixture_id",
            "team_id": "away_lineup_team_id",
            "coach_name": "away_coach_name",
            "formation": "away_formation",
        }
    )[
        [
            "lineup_fixture_id",
            "away_lineup_team_id",
            "away_coach_name",
            "away_formation",
        ]
    ]
    fixtures = fixtures.merge(
        home_lineups,
        left_on=["id", "home_team_id"],
        right_on=["lineup_fixture_id", "home_lineup_team_id"],
        how="left",
    ).drop(columns=["lineup_fixture_id", "home_lineup_team_id"])
    fixtures = fixtures.merge(
        away_lineups,
        left_on=["id", "away_team_id"],
        right_on=["lineup_fixture_id", "away_lineup_team_id"],
        how="left",
    ).drop(columns=["lineup_fixture_id", "away_lineup_team_id"])
    starter_ids = _starter_ids(
        paths["fixture_players.parquet"], set(fixtures["id"].astype(int))
    )

    Base.metadata.create_all(bind=engine)
    session = get_session()
    inserted = 0
    updated = 0
    target_disagreements = 0
    matched_with_xg = 0
    matched_with_shot_zones = 0
    matched_with_lineups = 0
    matched_with_starters = 0
    unmatched: list[dict[str, object]] = []
    try:
        local_matches = session.execute(
            select(Match).where(
                Match.competition.in_(COMPETITIONS),
                Match.full_time_home_goals.is_not(None),
                Match.full_time_away_goals.is_not(None),
            )
        ).scalars().all()
        by_key: dict[tuple, list[Match]] = defaultdict(list)
        by_pair: dict[tuple, list[Match]] = defaultdict(list)
        for match in local_matches:
            common = (
                match.competition,
                int(match.full_time_home_goals),
                int(match.full_time_away_goals),
            )
            by_key[(*common, match.date)].append(match)
            by_pair[
                (
                    *common,
                    _normalise_team(match.home_team),
                    _normalise_team(match.away_team),
                )
            ].append(match)

        existing = {
            row.match_id: row
            for row in session.execute(select(OpenBttsMatchData)).scalars().all()
        }
        used_match_ids: set[int] = set()
        for row in fixtures.itertuples(index=False):
            common = (
                row.competition,
                int(row.goals_home),
                int(row.goals_away),
            )
            candidates = [
                match
                for match in by_key.get((*common, row.date), [])
                if match.id not in used_match_ids
            ]
            if len(candidates) != 1:
                candidates = [
                    match
                    for match in by_pair.get(
                        (
                            *common,
                            _normalise_team(row.home_team),
                            _normalise_team(row.away_team),
                        ),
                        [],
                    )
                    if match.id not in used_match_ids
                    and abs((match.date - row.date).days) <= 1
                ]
            if len(candidates) != 1:
                unmatched.append(
                    {
                        "competition": row.competition,
                        "date": str(row.date),
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                    }
                )
                continue

            match = candidates[0]
            used_match_ids.add(match.id)
            source_btts = int(bool(row.btts))
            local_btts = int(
                match.full_time_home_goals > 0 and match.full_time_away_goals > 0
            )
            target_disagreements += int(source_btts != local_btts)
            matched_with_xg += int(
                pd.notna(row.home_xg) and pd.notna(row.away_xg)
            )
            matched_with_shot_zones += int(
                pd.notna(row.home_shots_inside_box)
                and pd.notna(row.away_shots_inside_box)
            )
            home_starters = starter_ids.get(
                (int(row.id), int(row.home_team_id)), []
            )
            away_starters = starter_ids.get(
                (int(row.id), int(row.away_team_id)), []
            )
            matched_with_lineups += int(
                pd.notna(row.home_formation) and pd.notna(row.away_formation)
            )
            matched_with_starters += int(
                bool(home_starters) and bool(away_starters)
            )
            values = {
                "source_fixture_id": int(row.id),
                "source_btts": source_btts,
                "home_xg": _safe_number(row.home_xg, float),
                "away_xg": _safe_number(row.away_xg, float),
                "home_shots_inside_box": _safe_number(
                    row.home_shots_inside_box, int
                ),
                "away_shots_inside_box": _safe_number(
                    row.away_shots_inside_box, int
                ),
                "home_shots_outside_box": _safe_number(
                    row.home_shots_outside_box, int
                ),
                "away_shots_outside_box": _safe_number(
                    row.away_shots_outside_box, int
                ),
                "home_blocked_shots": _safe_number(row.home_blocked_shots, int),
                "away_blocked_shots": _safe_number(row.away_blocked_shots, int),
                "home_penalties": _safe_number(row.home_penalties, int),
                "away_penalties": _safe_number(row.away_penalties, int),
                "home_corners": _safe_number(row.home_corners, int),
                "away_corners": _safe_number(row.away_corners, int),
                "home_yellow_cards": _safe_number(row.home_yellow_cards, int),
                "away_yellow_cards": _safe_number(row.away_yellow_cards, int),
                "home_red_cards": _safe_number(row.home_red_cards, int),
                "away_red_cards": _safe_number(row.away_red_cards, int),
                "home_possession": _safe_number(row.home_possession, float),
                "away_possession": _safe_number(row.away_possession, float),
                "home_fouls": _safe_number(row.home_fouls, int),
                "away_fouls": _safe_number(row.away_fouls, int),
                "home_offsides": _safe_number(row.home_offsides, int),
                "away_offsides": _safe_number(row.away_offsides, int),
                "home_pass_accuracy": _safe_number(
                    row.home_pass_accuracy, float
                ),
                "away_pass_accuracy": _safe_number(
                    row.away_pass_accuracy, float
                ),
                "home_coach_name": _safe_text(row.home_coach_name),
                "away_coach_name": _safe_text(row.away_coach_name),
                "home_formation": _safe_text(row.home_formation),
                "away_formation": _safe_text(row.away_formation),
                "home_starter_ids": (
                    json.dumps(home_starters) if home_starters else None
                ),
                "away_starter_ids": (
                    json.dumps(away_starters) if away_starters else None
                ),
            }
            enrichment = existing.get(match.id)
            if enrichment is None:
                session.add(OpenBttsMatchData(match_id=match.id, **values))
                inserted += 1
            else:
                for key, value in values.items():
                    setattr(enrichment, key, value)
                updated += 1

            if (inserted + updated) % 1000 == 0:
                session.commit()
        session.commit()
    finally:
        session.close()

    return {
        "downloaded_files": [str(path) for path in paths.values()],
        "source_fixtures": len(fixtures),
        "inserted": inserted,
        "updated": updated,
        "unmatched": unmatched,
        "target_disagreements": target_disagreements,
        "matched_with_xg": matched_with_xg,
        "matched_with_shot_zones": matched_with_shot_zones,
        "matched_with_lineups": matched_with_lineups,
        "matched_with_starters": matched_with_starters,
    }
