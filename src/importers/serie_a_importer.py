from pathlib import Path
import pandas as pd
from sqlalchemy.exc import IntegrityError

from src.config import SERIE_A_RAW_DIR, LEAGUES_RAW_DIR, PROCESSED_DATA_DIR
from src.database.connection import get_session
from src.database.models import Match


def _safe_int(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_date(value):
    if pd.isna(value):
        return None

    # football-data CSVs can use different date formats across old seasons
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.date()


def _pick_first_existing(row, columns):
    for col in columns:
        if col in row and not pd.isna(row[col]):
            return row[col]
    return None


def load_raw_serie_a_csv_files(write_processed: bool = True) -> pd.DataFrame:
    csv_files = sorted(SERIE_A_RAW_DIR.glob("serie_a_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {SERIE_A_RAW_DIR}. "
            f"Run download first."
        )

    frames = []

    for file_path in csv_files:
        file_name = file_path.stem
        # example: serie_a_2023_2024
        parts = file_name.split("_")
        season = f"{parts[-2]}/{parts[-1]}"

        df = pd.read_csv(file_path)
        df["season"] = season
        df["competition"] = "I1"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if write_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_DIR / "serie_a_combined.csv"
        combined.to_csv(output_path, index=False)

    return combined


def load_raw_league_csv_files() -> pd.DataFrame:
    """Load legacy Serie A plus any multi-league downloads without duplicates."""
    frames = [load_raw_serie_a_csv_files(write_processed=False)]
    for file_path in sorted(LEAGUES_RAW_DIR.glob("*/*.csv")):
        parts = file_path.stem.split("_")
        competition = parts[0].upper()
        start_year = parts[-2]
        end_year = parts[-1]
        # The legacy Serie A directory is authoritative for I1.
        if competition == "I1":
            continue
        frame = pd.read_csv(file_path)
        frame["season"] = f"{start_year}/{end_year}"
        frame["competition"] = competition
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def import_matches_to_db(include_all_leagues: bool = False) -> int:
    df = (
        load_raw_league_csv_files()
        if include_all_leagues
        else load_raw_serie_a_csv_files()
    )

    session = get_session()
    imported_count = 0

    for _, row in df.iterrows():
        match_date = _parse_date(row.get("Date"))

        if match_date is None:
            continue

        home_team = row.get("HomeTeam")
        away_team = row.get("AwayTeam")

        if pd.isna(home_team) or pd.isna(away_team):
            continue

        # Average market odds are usually named AvgH, AvgD, AvgA in newer files.
        # If not available, fallback to Bet365 columns B365H, B365D, B365A.
        odds_home = _pick_first_existing(row, ["AvgH", "B365H"])
        odds_draw = _pick_first_existing(row, ["AvgD", "B365D"])
        odds_away = _pick_first_existing(row, ["AvgA", "B365A"])

        odds_over_25 = _pick_first_existing(row, ["Avg>2.5", "B365>2.5"])
        odds_under_25 = _pick_first_existing(row, ["Avg<2.5", "B365<2.5"])

        match = Match(
            competition=str(row.get("competition", "I1")),
            season=row["season"],
            date=match_date,
            home_team=str(home_team),
            away_team=str(away_team),

            full_time_home_goals=_safe_int(row.get("FTHG")),
            full_time_away_goals=_safe_int(row.get("FTAG")),
            full_time_result=row.get("FTR") if not pd.isna(row.get("FTR")) else None,

            half_time_home_goals=_safe_int(row.get("HTHG")),
            half_time_away_goals=_safe_int(row.get("HTAG")),
            half_time_result=row.get("HTR") if not pd.isna(row.get("HTR")) else None,

            home_shots=_safe_int(row.get("HS")),
            away_shots=_safe_int(row.get("AS")),
            home_shots_on_target=_safe_int(row.get("HST")),
            away_shots_on_target=_safe_int(row.get("AST")),

            home_corners=_safe_int(row.get("HC")),
            away_corners=_safe_int(row.get("AC")),

            home_fouls=_safe_int(row.get("HF")),
            away_fouls=_safe_int(row.get("AF")),

            home_yellow_cards=_safe_int(row.get("HY")),
            away_yellow_cards=_safe_int(row.get("AY")),
            home_red_cards=_safe_int(row.get("HR")),
            away_red_cards=_safe_int(row.get("AR")),

            odds_home_win=_safe_float(odds_home),
            odds_draw=_safe_float(odds_draw),
            odds_away_win=_safe_float(odds_away),

            odds_over_25=_safe_float(odds_over_25),
            odds_under_25=_safe_float(odds_under_25),
        )

        try:
            session.add(match)
            session.commit()
            imported_count += 1
        except IntegrityError:
            session.rollback()

    session.close()
    return imported_count
