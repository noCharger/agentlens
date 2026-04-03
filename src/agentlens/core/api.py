"""Minimal HTTP API for querying and ingesting platform records."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentlens.core.service import CoreApiService
from agentlens.core.sqlite_repository import SQLiteCoreRepository

MAX_REQUEST_BODY_BYTES = 5 * 1024 * 1024


def create_handler(service: CoreApiService):
    class PlatformApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            body_payload = None
            if method == "POST":
                body_payload = self._read_json_body()
                if isinstance(body_payload, tuple):
                    status_code, error_payload = body_payload
                    self._write_response(status_code, error_payload)
                    return

            response = service.handle(
                method,
                self.path,
                body=body_payload,
                headers={k: v for k, v in self.headers.items()},
            )
            self._write_response(response.status_code, response.payload)

        def _read_json_body(self):
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return 411, {"error": "content_length_required"}
            try:
                content_length = int(raw_length)
            except ValueError:
                return 400, {"error": "invalid_content_length"}
            if content_length > MAX_REQUEST_BODY_BYTES:
                return 413, {"error": "payload_too_large"}

            raw_body = self.rfile.read(content_length)
            if not raw_body:
                return 400, {"error": "empty_body"}
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return 400, {"error": "invalid_json"}
            if not isinstance(payload, dict):
                return 400, {"error": "json_body_must_be_object"}
            return payload

        def _write_response(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return PlatformApiHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local AgentLens platform API")
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path(".agentlens-platform/platform.db"),
        help="Path to the platform SQLite database",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    repository = SQLiteCoreRepository(args.sqlite)
    service = CoreApiService(repository)
    handler = create_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        print(f"AgentLens platform API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
