#!/usr/bin/env python
"""Persistent localnav policy service (point-conditioned diffusion policy).

Loaded ONCE (same pattern as grounding_server.py) so the bridge closed loop
never pays model cold-start inside a navigation request. Plain stdlib HTTP:

  POST /act      {"context": [b64 jpeg ×K], "goal_image": b64, "goal_point": [x, y],
                  "num_samples": 16, "denoise_steps": 10}
  -> {"ok": true, "trajectories": M×8×[dx, dy] (meters, robot frame),
      "scores": [M], "selected": i, "temporal_distance": d|null,
      "done_probability": p, "confidence": c, "latency_ms": t}
  GET /healthz   -> {"ok": true, "backend", "model", "ckpt", "device"}

Start:  python -m supernav.methods.localnav.server --port 18914 --backend debug
Env:    NAV_LOCALNAV_PORT / NAV_LOCALNAV_BACKEND / NAV_LOCALNAV_CKPT /
        NAV_LOCALNAV_DEVICE / NAV_LOCALNAV_MAX_REQUEST_BYTES /
        NAV_LOCALNAV_GUIDANCE (CFG scale; default auto from checkpoint)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image


from supernav.methods.localnav.contracts import LocalNavInputError, NormalizedPoint
from supernav.methods.localnav_policy.backends import PolicyBackend, build_backend

_DEFAULT_PORT = 18914
_DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_SAMPLES = 64
_MAX_DENOISE_STEPS = 50


def _decode_rgb(encoded: str) -> np.ndarray:
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image)


def _clamp_int(value, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet — telemetry lives bridge-side
        pass

    @property
    def _backend(self) -> PolicyBackend:
        return self.server.localnav_backend  # type: ignore[attr-defined]

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in ("", "/healthz"):
            self._send(404, {"ok": False, "code": "not_found",
                             "error": f"unknown path {self.path!r}"})
            return
        backend = self._backend
        self._send(
            200,
            {
                "ok": True,
                "backend": backend.name,
                "model": backend.model,
                "ckpt": backend.checkpoint,
                "device": backend.device,
                "context_mode": getattr(backend, "context_mode", "fixed"),
                "memory_budget": getattr(backend, "_memory_budget", None),
            },
        )

    def do_POST(self) -> None:
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length", 0))
            max_bytes = int(
                os.environ.get(
                    "NAV_LOCALNAV_MAX_REQUEST_BYTES", str(_DEFAULT_MAX_REQUEST_BYTES)
                )
            )
            if length > max_bytes:
                self._send(
                    413,
                    {
                        "ok": False,
                        "code": "request_too_large",
                        "error": f"request body exceeds {max_bytes} bytes",
                    },
                )
                return
            request = json.loads(self.rfile.read(length))

            context_raw = request.get("context")
            if not isinstance(context_raw, list) or not context_raw:
                raise LocalNavInputError(
                    "invalid_request", "context must be a non-empty list of b64 images"
                )
            goal_image_raw = request.get("goal_image")
            if not goal_image_raw:
                raise LocalNavInputError("invalid_request", "goal_image is required")
            goal_point = NormalizedPoint.parse(request.get("goal_point"))

            context = [_decode_rgb(item) for item in context_raw]
            goal_rgb = _decode_rgb(goal_image_raw)
            num_samples = _clamp_int(request.get("num_samples"), 1, _MAX_SAMPLES, 16)
            denoise_steps = _clamp_int(
                request.get("denoise_steps"), 1, _MAX_DENOISE_STEPS, 10
            )

            result = self._backend.act(
                context, goal_rgb, goal_point.as_tuple(), num_samples, denoise_steps
            )
            trajectories = np.round(
                np.asarray(result["trajectories"], dtype=np.float64), 4
            ).tolist()
            scores = np.round(np.asarray(result["scores"], dtype=np.float64), 4).tolist()
            temporal_distance = result.get("temporal_distance")
            self._send(
                200,
                {
                    "ok": True,
                    "trajectories": trajectories,
                    "scores": scores,
                    "selected": int(result["selected"]),
                    "temporal_distance": (
                        None if temporal_distance is None else round(float(temporal_distance), 3)
                    ),
                    "done_probability": round(float(result["done_probability"]), 4),
                    "confidence": round(float(result["confidence"]), 4),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
        except LocalNavInputError as exc:
            self._send(400, {"ok": False, "code": exc.code, "error": exc.message})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(
                400,
                {"ok": False, "code": "invalid_request", "error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception as exc:  # noqa: BLE001 — service must answer, not die
            self._send(
                500,
                {"ok": False, "code": "backend_error", "error": f"{type(exc).__name__}: {exc}"},
            )


def create_server(host: str, port: int, backend: PolicyBackend) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.localnav_backend = backend  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("NAV_LOCALNAV_PORT", _DEFAULT_PORT))
    )
    parser.add_argument(
        "--backend", default=os.environ.get("NAV_LOCALNAV_BACKEND", "debug")
    )
    parser.add_argument(
        "--checkpoint", default=os.environ.get("NAV_LOCALNAV_CKPT") or None
    )
    parser.add_argument("--device", default=os.environ.get("NAV_LOCALNAV_DEVICE") or None)
    parser.add_argument(
        "--guidance",
        type=float,
        default=(
            float(os.environ["NAV_LOCALNAV_GUIDANCE"])
            if os.environ.get("NAV_LOCALNAV_GUIDANCE")
            else None
        ),
        help="CFG guidance scale (default: auto from the checkpoint)",
    )
    parser.add_argument(
        "--selection",
        choices=("consensus", "bearing"),
        default=os.environ.get("NAV_LOCALNAV_SELECTION") or None,
    )
    args = parser.parse_args()

    backend = build_backend(
        args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        guidance_scale=args.guidance,
        selection=args.selection,
    )
    server = create_server(args.host, args.port, backend)
    print(
        f"[localnav] {backend.name} backend resident on {backend.device}; "
        f"serving {args.host}:{server.server_address[1]}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
