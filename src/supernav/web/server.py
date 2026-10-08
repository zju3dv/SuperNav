"""Dependency-free local HTTP/SSE spectator server; never controls the simulator."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from supernav.web.store import SessionStore

STATIC = Path(__file__).with_name("static")


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, store):
        self.store = store
        self.stopping = threading.Event()
        super().__init__(address, ViewerHandler)

    def server_close(self):
        self.stopping.set()
        super().server_close()


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, mime="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, code=200):
        self._send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _get(self):
        url = urlsplit(self.path)
        if url.path == "/api/health":
            return self._json({"ok": True, "service": "supernav-web"})
        if url.path == "/api/sessions":
            return self._json({"sessions": self.server.store.sessions()})
        parts = url.path.strip("/").split("/")
        if len(parts) in (3, 4) and parts[:2] == ["api", "sessions"]:
            key = parts[2]
            snapshot = self.server.store.snapshot(key)
            if snapshot is None:
                return self._json({"error": "Session not found"}, 404)
            if len(parts) == 3:
                return self._json(snapshot)
            if parts[3] == "events":
                return self._events(key)
            if parts[3] == "trace":
                return self._json(self.server.store.trace(key))
            if parts[3] == "frame":
                query = parse_qs(url.query)
                try:
                    index = int(query.get("index", ["0"])[0])
                except ValueError:
                    return self._json({"error": "Invalid frame index"}, 400)
                path = self.server.store.frame(key, index, kind=query.get("kind", ["rgb"])[0], view=query.get("view", [""])[0])
                if path:
                    try:
                        return self._send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "image/jpeg")
                    except OSError:
                        pass
                return self._json({"error": "Frame unavailable"}, 404)
        asset = "index.html" if url.path == "/" else url.path.removeprefix("/")
        # Explicit allowlist: never expose run configs, raw credentials or arbitrary files.
        allowed = {"index.html", "app.css", "app.js", "favicon.svg", "fonts/space-grotesk.ttf", "fonts/ibm-plex-mono.ttf"}
        if asset in allowed and (STATIC / asset).is_file():
            return self._send(200, (STATIC / asset).read_bytes(),
                              mimetypes.guess_type(asset)[0] or "application/octet-stream")
        return self._json({"error": "Not found"}, 404)

    def _events(self, key):
        self.connection.settimeout(15)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        previous = None
        heartbeat = 0
        while not self.server.stopping.is_set():
            snapshot = self.server.store.snapshot(key)
            if snapshot is None:
                self.wfile.write(b'event: unavailable\ndata: {}\n\n')
                self.wfile.flush()
                break
            # Full snapshots make reconnect/resume idempotent, including stale producers.
            # Run metadata can arrive after the producer's final revision.
            data = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
            if data != previous:
                self.wfile.write(f"id: {snapshot.get('revision', 0)}\nevent: snapshot\ndata: {data}\n\n".encode())
                self.wfile.flush()
                previous = data
                heartbeat = time.monotonic()
            elif time.monotonic() - heartbeat >= 5:
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                heartbeat = time.monotonic()
            self.server.stopping.wait(0.2)


def main():
    parser = argparse.ArgumentParser(description="Watch SuperNav in your browser (read-only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live-dir", type=Path, default=Path(os.environ.get("SUPERNAV_LIVE_DIR", "data/nav_artifacts/live")))
    parser.add_argument("--visuals-root", action="append", type=Path, default=[], help="Existing trajectory sidecars and RGB; repeat for multiple roots.")
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"), help="Run metadata and emitted agent trace (default: data/runs).")
    parser.add_argument("--demo", action="store_true", help="Add an explicitly labelled procedural demonstration.")
    args = parser.parse_args()
    store = SessionStore(args.live_dir, visuals_roots=args.visuals_root, runs_root=args.runs_root, demo=args.demo)
    server = ViewerServer((args.host, args.port), store)
    print(f"SuperNav Live → http://{args.host}:{server.server_port}", flush=True)
    print(f"Watching {args.live_dir.resolve()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
