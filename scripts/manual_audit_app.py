#!/usr/bin/env python3
"""Serve the local, write-enabled Research 2 manual-audit interface."""

from __future__ import annotations

import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review_app import ReviewStore  # noqa: E402


STATIC = ROOT / "review_app"
TIMELINES = ROOT / "reports/manual-audit"


def valid_basic_auth(header: str | None, username: str, password: str) -> bool:
    if not password:
        return True
    if not header or not header.startswith("Basic "):
        return False
    try:
        supplied = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(supplied, f"{username}:{password}")


class Handler(BaseHTTPRequestHandler):
    store = ReviewStore(ROOT)
    auth_username = "reviewer"
    auth_password = ""

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"review-app: {format % args}\n")

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def authenticate(self) -> bool:
        if valid_basic_auth(
            self.headers.get("Authorization"), self.auth_username, self.auth_password
        ):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Research 2 review"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self.authenticate():
            return
        path = urlparse(self.path).path
        if path == "/api/state":
            self.send_json(self.store.state())
            return
        if path.startswith("/timelines/"):
            name = Path(path).name
            if not name.endswith(".svg"):
                self.send_error(404)
                return
            self.send_file(TIMELINES / name)
            return
        target = "index.html" if path == "/" else path.lstrip("/")
        if target not in {"index.html", "styles.css", "app.js"}:
            self.send_error(404)
            return
        self.send_file(STATIC / target)

    def do_POST(self) -> None:  # noqa: N802
        if not self.authenticate():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 32_768:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            path = urlparse(self.path).path
            if path == "/api/review":
                result = self.store.record_review(
                    str(body.get("run_id", "")),
                    reviewer_name=str(body.get("reviewer_name", "")),
                    decision=str(body.get("decision", "")),
                    notes=str(body.get("notes", "")),
                    evidence_checked=body.get("evidence_checked") is True,
                )
            elif path == "/api/supervisor-review":
                result = self.store.record_supervisor_review(
                    reviewer_name=str(body.get("reviewer_name", "")),
                    rationale=str(body.get("rationale", "")),
                    role_confirmed=body.get("role_confirmed") is True,
                    protected_confirmed=body.get("protected_confirmed") is True,
                )
            else:
                self.send_error(404)
                return
            self.send_json(result)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, status=400)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    Handler.auth_username = os.environ.get("REVIEW_APP_USERNAME", "reviewer")
    Handler.auth_password = os.environ.get("REVIEW_APP_PASSWORD", "")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Manual audit review app: {url}")
    print("Password protection: enabled" if Handler.auth_password else "Password protection: disabled")
    print("Bound to localhost only. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
