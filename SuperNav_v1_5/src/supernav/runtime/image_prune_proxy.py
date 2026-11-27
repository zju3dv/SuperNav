#!/usr/bin/env python3
"""Local relay proxy that prunes stale inline images from Responses API requests.

Codex resends the full conversation on every relay call, so tool-result
images accumulate and each call gets slower. This proxy sits between the
benchmark agent and the experiment relay and rewrites POST /responses bodies:
only the newest image-bearing input item keeps its images; older image parts
are replaced by a fixed text placeholder. The replacement is deterministic, so
the rewritten request prefix stays byte-identical across calls and upstream
prefix caching keeps working.

Usage:

    python -m supernav image-prune-proxy \
        --upstream https://api.example.com/v1 --listen-port 8993

Then point the harness at it:

    HAB_BENCH_RELAY_BASE_URL=http://127.0.0.1:8993/v1 \
    HAB_BENCH_RELAY_WIRE_API=responses ...

Use --dump-dir on the first real run to capture raw request bodies and confirm
the image-part detection matches the client's wire shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

import httpx

_IMAGE_TYPES = {"input_image", "image", "image_url"}
_DEFAULT_PLACEHOLDER = "[omitted stale surround image]"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
    "accept-encoding",
}


def _is_image_part(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = node.get("type")
    if isinstance(node_type, str) and node_type.lower() in _IMAGE_TYPES:
        return True
    # MCP-style inline image without a type tag: {data, mimeType: image/...}
    mime = node.get("mimeType") or node.get("mime_type")
    return (
        isinstance(mime, str)
        and mime.startswith("image/")
        and isinstance(node.get("data"), str)
    )


def _replace_image_parts(node: Any, placeholder: str, stats: Dict[str, int]) -> None:
    """Replace image parts in-place with a fixed text placeholder."""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if _is_image_part(value):
                node[key] = {"type": "input_text", "text": placeholder}
                stats["dropped"] += 1
            else:
                _replace_image_parts(value, placeholder, stats)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            if _is_image_part(value):
                node[idx] = {"type": "input_text", "text": placeholder}
                stats["dropped"] += 1
            else:
                _replace_image_parts(value, placeholder, stats)


def _count_image_parts(node: Any) -> int:
    if _is_image_part(node):
        return 1
    if isinstance(node, dict):
        return sum(_count_image_parts(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_image_parts(v) for v in node)
    return 0


def prune_stale_images(
    payload: Dict[str, Any],
    *,
    keep_latest_items: int = 1,
    placeholder: str = _DEFAULT_PLACEHOLDER,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Drop image parts from all but the newest image-bearing input items."""
    stats = {"kept": 0, "dropped": 0, "image_items": 0}
    items = payload.get("input")
    if not isinstance(items, list):
        return payload, stats
    image_item_indices = [
        idx for idx, item in enumerate(items) if _count_image_parts(item) > 0
    ]
    stats["image_items"] = len(image_item_indices)
    keep_from = max(0, len(image_item_indices) - max(1, keep_latest_items))
    keep_indices = set(image_item_indices[keep_from:])
    for idx in image_item_indices:
        if idx in keep_indices:
            stats["kept"] += _count_image_parts(items[idx])
        else:
            _replace_image_parts(items[idx], placeholder, stats)
    return payload, stats


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(1.0, min(60.0, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, 2.0 * attempt) + (attempt % 3) * 0.7


class _PruneProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited responses; simplest streaming
    server_version = "RelayImagePruneProxy/1.0"

    # -- plumbing ---------------------------------------------------------

    def _log(self, message: str) -> None:
        sys.stderr.write(f"[relay-prune] {message}\n")
        sys.stderr.flush()

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
        del fmt, args

    @property
    def _cfg(self) -> "_ProxyConfig":
        return self.server.cfg  # type: ignore[attr-defined]

    def _dump_request(self, body: bytes) -> None:
        dump_dir = self._cfg.dump_dir
        if dump_dir is None:
            return
        with self._cfg.lock:
            self._cfg.request_seq += 1
            seq = self._cfg.request_seq
        path = dump_dir / f"req_{seq:05d}.json"
        try:
            path.write_bytes(body)
        except OSError as exc:
            self._log(f"dump failed for {path}: {exc}")

    def _handle(self) -> None:
        cfg = self._cfg
        started = time.monotonic()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path
        if cfg.upstream_path and path.startswith(cfg.upstream_path + "/"):
            path = path[len(cfg.upstream_path):]
        target = cfg.upstream + path
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP
        }

        stats: Dict[str, int] = {}
        out_body = body
        if body and self.command == "POST":
            self._dump_request(body)
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("input"), list):
                payload, stats = prune_stale_images(
                    payload,
                    keep_latest_items=cfg.keep_latest_items,
                    placeholder=cfg.placeholder,
                )
                out_body = json.dumps(payload).encode("utf-8")

        sent = 0
        status = 0
        attempt = 0
        while True:
            attempt += 1
            try:
                with cfg.client.stream(
                    self.command,
                    target,
                    headers=headers,
                    content=out_body,
                ) as resp:
                    status = resp.status_code
                    if self._should_retry(status, attempt, started):
                        retry_after = resp.headers.get("retry-after")
                        resp.read()  # drain so the connection can be reused
                        delay = _retry_delay(attempt, retry_after)
                        self._log(
                            f"{self.command} {self.path} -> {status} "
                            f"attempt {attempt}/{cfg.retry_max}, retry in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    self.send_response(status)
                    for key, value in resp.headers.items():
                        if key.lower() in _HOP_BY_HOP:
                            continue
                        # We re-frame the body; upstream length/framing no longer apply.
                        if key.lower() in ("content-length", "content-encoding"):
                            continue
                        self.send_header(key, value)
                    self.end_headers()
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        sent += len(chunk)
            except httpx.HTTPError as exc:
                if self._should_retry(None, attempt, started):
                    delay = _retry_delay(attempt, None)
                    self._log(
                        f"upstream error {type(exc).__name__}: {exc} "
                        f"attempt {attempt}/{cfg.retry_max}, retry in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                self._log(f"upstream error for {self.path}: {type(exc).__name__}: {exc}")
                if not self.wfile.closed:
                    try:
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(
                            json.dumps(
                                {"error": f"relay prune proxy upstream failure: {exc}"}
                            ).encode("utf-8")
                        )
                    except (BrokenPipeError, ConnectionError):
                        pass
                return
            except (BrokenPipeError, ConnectionError):
                # Client went away mid-stream; nothing sensible left to do.
                return
            break
        elapsed = time.monotonic() - started
        self._log(
            f"{self.command} {self.path} -> {status} "
            f"req={len(out_body)}B resp={sent}B {elapsed:.1f}s "
            f"image_items={stats.get('image_items', 0)} "
            f"kept={stats.get('kept', 0)} dropped={stats.get('dropped', 0)}"
        )

    def _should_retry(
        self, status: int | None, attempt: int, started: float
    ) -> bool:
        cfg = self._cfg
        if attempt >= cfg.retry_max:
            return False
        if time.monotonic() - started >= cfg.retry_budget_s:
            return False
        if status is None:  # transport error
            return True
        return status in cfg.retry_statuses

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle


class _ProxyConfig:
    def __init__(self, args: argparse.Namespace) -> None:
        self.upstream = args.upstream.rstrip("/")
        # Clients address the proxy with the same base path they would use on
        # the relay (e.g. base_url ".../v1" -> POST /v1/responses). If the
        # configured upstream already ends with that path, drop the duplicate
        # prefix instead of forwarding /v1/v1/responses.
        self.upstream_path = urlsplit(self.upstream).path.rstrip("/")
        self.keep_latest_items = args.keep_latest_items
        self.placeholder = args.placeholder
        self.retry_max = args.retry_max
        self.retry_budget_s = args.retry_budget_s
        self.retry_statuses = {
            int(s) for s in args.retry_statuses.split(",") if s.strip()
        }
        self.dump_dir = Path(args.dump_dir) if args.dump_dir else None
        self.lock = threading.Lock()
        self.request_seq = 0
        # No read timeout: relay stalls (observed up to ~15 min) must pass
        # through to the client's own timeout handling.
        self.client = httpx.Client(
            timeout=httpx.Timeout(None, connect=30.0),
            follow_redirects=True,
            trust_env=True,
        )
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8993)
    parser.add_argument(
        "--upstream",
        default=os.environ.get("HAB_BENCH_RELAY_UPSTREAM", "").strip(),
        help="upstream base URL; required unless HAB_BENCH_RELAY_UPSTREAM is set",
    )
    parser.add_argument(
        "--keep-latest-items",
        type=int,
        default=1,
        help="how many of the newest image-bearing input items keep images",
    )
    parser.add_argument("--placeholder", default=_DEFAULT_PLACEHOLDER)
    parser.add_argument(
        "--retry-max",
        type=int,
        default=8,
        help="max upstream attempts per request on retryable statuses/errors",
    )
    parser.add_argument(
        "--retry-budget-s",
        type=float,
        default=240.0,
        help="total seconds a request may spend retrying before passing through",
    )
    parser.add_argument(
        "--retry-statuses",
        default="429,500,502,503,504",
        help="comma-separated upstream statuses retried internally",
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="optional directory for raw (pre-rewrite) request bodies",
    )
    args = parser.parse_args(argv)
    if not args.upstream or not args.upstream.strip():
        parser.error("set --upstream or HAB_BENCH_RELAY_UPSTREAM; no upstream is configured")

    cfg = _ProxyConfig(args)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), _PruneProxyHandler)
    server.cfg = cfg  # type: ignore[attr-defined]
    server.daemon_threads = True
    bind = server.server_address
    sys.stderr.write(
        f"[relay-prune] listening on http://{bind[0]}:{bind[1]} -> {cfg.upstream} "
        f"(keep_latest_items={cfg.keep_latest_items})\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cfg.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
