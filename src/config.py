from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SERIE_A_RAW_DIR = RAW_DATA_DIR / "serie_a"

DATABASE_URL = f"sqlite:///{BASE_DIR / 'football_predictor.db'}"