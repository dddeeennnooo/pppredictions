from pathlib import Path
import requests

from src.config import SERIE_A_RAW_DIR


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