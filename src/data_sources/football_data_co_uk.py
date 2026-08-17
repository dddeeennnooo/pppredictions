from pathlib import Path
import requests

from src.config import SERIE_A_RAW_DIR, LEAGUES_RAW_DIR


DEFAULT_COMPETITIONS = {
    "I1": "Serie A",
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "F1": "Ligue 1",
}


def season_code(start_year: int) -> str:
    """
    2007 -> '0708'
    2023 -> '2324'
    """
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def download_serie_a_season(start_year: int) -> Path:
    """
    Downloads one Serie A season CSV from football-data.co.uk.

    Example:
    2023/2024 -> https://www.football-data.co.uk/mmz4281/2324/I1.csv
    """
    code = season_code(start_year)
    url = f"https://www.football-data.co.uk/mmz4281/{code}/I1.csv"

    SERIE_A_RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SERIE_A_RAW_DIR / f"serie_a_{start_year}_{start_year + 1}.csv"

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download season {start_year}/{start_year + 1}. "
            f"Status: {response.status_code}, URL: {url}"
        )

    output_path.write_bytes(response.content)
    return output_path


def download_serie_a_range(start_year: int, end_year: int) -> list[Path]:
    """
    Downloads seasons from start_year to end_year inclusive.

    Example:
    start_year=2007, end_year=2025
    downloads 2007/08 through 2025/26.
    """
    downloaded_files = []

    for year in range(start_year, end_year + 1):
        print(f"Downloading Serie A {year}/{year + 1}...")
        path = download_serie_a_season(year)
        downloaded_files.append(path)
        print(f"Saved: {path}")

    return downloaded_files


def download_competition_season(competition: str, start_year: int) -> Path:
    """Download one football-data.co.uk competition/season CSV."""
    competition = competition.upper()
    code = season_code(start_year)
    url = f"https://www.football-data.co.uk/mmz4281/{code}/{competition}.csv"
    output_dir = LEAGUES_RAW_DIR / competition
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{competition}_{start_year}_{start_year + 1}.csv"

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {competition} {start_year}/{start_year + 1}. "
            f"Status: {response.status_code}, URL: {url}"
        )
    output_path.write_bytes(response.content)
    return output_path


def download_league_range(
    start_year: int,
    end_year: int,
    competitions: list[str] | None = None,
) -> list[Path]:
    """Download multiple top-flight European leagues."""
    selected = competitions or list(DEFAULT_COMPETITIONS)
    downloaded_files = []
    for year in range(start_year, end_year + 1):
        for competition in selected:
            print(f"Downloading {competition} {year}/{year + 1}...")
            downloaded_files.append(
                download_competition_season(competition, year)
            )
    return downloaded_files
