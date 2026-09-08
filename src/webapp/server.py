from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.webapp.game import GameError, GameService


STATIC_DIR = Path(__file__).resolve().parent / "static"
ROUND_PATTERN = re.compile(r"^/api/rounds/([a-f0-9]{32})$")
PICK_PATTERN = re.compile(r"^/api/rounds/([a-f0-9]{32})/picks$")
COMPLETE_PATTERN = re.compile(r"^/api/rounds/([a-f0-9]{32})/complete$")


class GameRequestHandler(BaseHTTPRequestHandler):
    service: GameService
    server_version = "Touchline/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json({"status": "ok"})
                return
            if path == "/api/meta":
                self._json(self.service.metadata())
                return
            match = ROUND_PATTERN.fullmatch(path)
            if match:
                self._json(self.service.get_round(match.group(1)))
                return
            self._static(path)
        except GameError as exc:
            self._json({"error": str(exc)}, exc.status)
        except Exception:
            self._json({"error": "The server could not complete that request."}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/rounds":
                self._json(
                    self.service.create_round(
                        mode=payload.get("mode", "btts"),
                        season=payload.get("season"),
                        competition=payload.get("competition"),
                    ),
                    HTTPStatus.CREATED,
                )
                return
            match = PICK_PATTERN.fullmatch(path)
            if match:
                if "match_id" not in payload or "prediction" not in payload:
                    raise GameError("match_id and prediction are required.")
                self._json(
                    self.service.save_pick(
                        match.group(1), int(payload["match_id"]), payload["prediction"]
                    )
                )
                return
            match = COMPLETE_PATTERN.fullmatch(path)
            if match:
                self._json(self.service.complete_round(match.group(1)))
                return
            self._json({"error": "Endpoint not found."}, 404)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "Request body must be valid JSON."}, 400)
        except GameError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (TypeError, ValueError):
            self._json({"error": "The request contains an invalid value."}, 400)
        except Exception:
            self._json({"error": "The server could not complete that request."}, 500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64_000:
            raise GameError("Request body is too large.", 413)
        if not length:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise GameError("Request body must be a JSON object.")
        return payload

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.removeprefix("/")
        if relative.startswith("static/"):
            relative = relative.removeprefix("static/")
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            self._json({"error": "File not found."}, 404)
            return
        if not candidate.is_file():
            # Client-side routes all land on the application shell.
            candidate = STATIC_DIR / "index.html"
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message_format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} — {message_format % args}")


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    GameRequestHandler.service = GameService()
    return ThreadingHTTPServer((host, port), GameRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Human vs AI football game.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"Touchline is ready at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Touchline.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
