from __future__ import annotations

import os
import shutil
import signal
import threading
from pathlib import Path

from src.config import BASE_DIR, DATABASE_PATH
from src.webapp.server import create_server


SEED_DATABASE_PATH = BASE_DIR / "football_predictor.db"


def ensure_database() -> Path:
    """Create the production database file from the bundled seed if needed."""
    target = DATABASE_PATH
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if not SEED_DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {target}, and no seed database exists at "
            f"{SEED_DATABASE_PATH}. Commit football_predictor.db or restore it "
            "to the persistent disk before starting the app."
        )

    if SEED_DATABASE_PATH.resolve() != target.resolve():
        shutil.copy2(SEED_DATABASE_PATH, target)
    return target


def main() -> None:
    database_path = ensure_database()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))
    server = create_server(host, port)

    def request_shutdown(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; stopping Touchline.")
        # HTTPServer.shutdown() must run outside the serve_forever thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    print(f"Database ready at {database_path}")
    print(f"Touchline is ready at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
