"""HTTP client for the localnav policy service + trajectory→command geometry.

Not part of the recovered contract set: this module owns the process-boundary
plumbing (image content upload, same pattern as the LocateAnything client) and
the pure-geometry conversion from waypoint deltas to ``VelocityAction`` chunks.
Torch-free by design; the bridge process imports it without pulling the GPU
stack.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from supernav.methods.localnav.contracts import LocalNavRuntimeError, VelocityAction

DEFAULT_POLICY_URL = "http://127.0.0.1:18914"
_DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def waypoint_deltas_to_actions(
    deltas: Sequence[Sequence[float]], control_dt_s: float
) -> "tuple[VelocityAction, ...]":
    """Per-step waypoint deltas (robot frame at prediction time, meters) →
    unicycle commands of fixed duration ``control_dt_s``.

    Each segment's heading is followed with a turn-then-drive command; a
    near-zero segment keeps the previous heading (no spurious rotation).
    """
    if control_dt_s <= 0.0:
        raise LocalNavRuntimeError("invalid_action", "control_dt_s must be positive")
    steps = np.asarray(deltas, dtype=np.float64).reshape(-1, 2)
    # Zero-displacement segments face the next significant segment when enabled.
    turn_to_next = float(os.environ.get("NAV_LOCALNAV_TURN_TO_NEXT", "1") or 0) > 0
    lookahead: List["float | None"] = [None] * len(steps)
    if turn_to_next:
        pending: "float | None" = None
        for i in range(len(steps) - 1, -1, -1):
            dx_i, dy_i = float(steps[i][0]), float(steps[i][1])
            if math.hypot(dx_i, dy_i) >= 0.06:
                pending = math.atan2(dy_i, dx_i)
            lookahead[i] = pending
    heading = 0.0
    actions: List[VelocityAction] = []
    for i, (dx, dy) in enumerate(steps):
        segment = math.hypot(float(dx), float(dy))
        # Sub-quantization segments carry NO usable heading: near the goal the
        # policy emits ~zero vectors whose direction is sampling noise, and
        # atan2 of noise executed as real 10°-quantized turns made the robot
        # pirouette on the spot (and the spun views are OOD for the stop
        # head). 0.06 m is far below the 0.25 m move quantum, so suppressing
        # the turn also yields zero primitives — which re-opens the
        # deliberate-stand-still stop path in the loop guards.
        if segment >= 0.06:
            new_heading = math.atan2(float(dy), float(dx))
        elif turn_to_next and lookahead[i] is not None:
            new_heading = lookahead[i]  # Turn in place toward the upcoming segment.
        else:
            new_heading = heading
        dyaw = _wrap_angle(new_heading - heading)
        actions.append(VelocityAction(segment / control_dt_s, dyaw / control_dt_s))
        heading = new_heading
    return tuple(actions)


def select_trajectory(
    trajectories: np.ndarray,
    scores: "Sequence[float] | None",
) -> int:
    """Choose the candidate with the highest server-side consensus score.

    Selection uses RGB-derived scores. Obstacle avoidance is handled by
    the policy using camera, proprioception and contact inputs.
    """
    trajs = np.asarray(trajectories, dtype=np.float64)
    count = trajs.shape[0]
    if scores is not None and len(scores) == count:
        base = np.asarray(scores, dtype=np.float64)
    else:
        base = np.zeros(count, dtype=np.float64)
    return int(np.argmax(base))


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class InProcessPolicyClient:
    """Wraps a PolicyBackend directly — no HTTP hop. Used by the evaluation
    harness (and tests) so the REAL bridge loop runs against an in-process
    backend with the exact same client contract as LocalNavPolicyClient."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    @property
    def context_mode(self) -> str:
        return getattr(self._backend, "context_mode", "fixed")

    def healthz(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "backend": getattr(self._backend, "name", "in-process"),
            "model": getattr(self._backend, "model", ""),
            "ckpt": getattr(self._backend, "checkpoint", None),
            "device": getattr(self._backend, "device", "cpu"),
        }

    def act(
        self,
        context_rgbs: Sequence[np.ndarray],
        goal_rgb: np.ndarray,
        goal_point: Sequence[float],
        num_samples: int = 16,
        denoise_steps: int = 10,
    ) -> Dict[str, Any]:
        try:
            result = self._backend.act(
                list(context_rgbs),
                np.asarray(goal_rgb),
                (float(goal_point[0]), float(goal_point[1])),
                int(num_samples),
                int(denoise_steps),
            )
        except Exception as exc:  # noqa: BLE001 — mirror the HTTP error contract
            raise LocalNavRuntimeError(
                "backend_error", f"{type(exc).__name__}: {exc}"
            ) from None
        return {"ok": True, **result}


class LocalNavPolicyClient:
    """Thin JSON/HTTP client for localnav_server (supports remote deployment
    via NAV_LOCALNAV_URL; uploads image content, never local paths)."""

    def __init__(
        self, base_url: "str | None" = None, timeout_s: "float | None" = None
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("NAV_LOCALNAV_URL") or DEFAULT_POLICY_URL
        ).rstrip("/")
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else os.environ.get("NAV_LOCALNAV_REQUEST_TIMEOUT", "30")
        )

    @property
    def context_mode(self) -> str:
        """Context protocol cached from the serving checkpoint through /healthz."""
        cached = getattr(self, "_context_mode_cache", None)
        if cached is None:
            try:
                cached = str(self.healthz().get("context_mode", "fixed"))
            except LocalNavError:
                cached = "fixed"
            self._context_mode_cache = cached
        return cached

    def healthz(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/healthz", timeout=self.timeout_s
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — network failures take many shapes
            raise LocalNavRuntimeError(
                "policy_service_unreachable",
                f"localnav policy service unreachable at {self.base_url}: {exc}",
            ) from None

    def act(
        self,
        context_rgbs: Sequence[np.ndarray],
        goal_rgb: np.ndarray,
        goal_point: Sequence[float],
        num_samples: int = 16,
        denoise_steps: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "context": [_encode_jpeg_b64(frame) for frame in context_rgbs],
            "goal_image": _encode_jpeg_b64(goal_rgb),
            "goal_point": [float(goal_point[0]), float(goal_point[1])],
            "num_samples": int(num_samples),
            "denoise_steps": int(denoise_steps),
        }
        body = json.dumps(payload).encode("utf-8")
        max_bytes = int(
            os.environ.get("NAV_LOCALNAV_MAX_REQUEST_BYTES", str(_DEFAULT_MAX_REQUEST_BYTES))
        )
        if len(body) > max_bytes:
            raise LocalNavRuntimeError(
                "request_too_large", f"policy request exceeds {max_bytes} bytes"
            )
        request = urllib.request.Request(
            f"{self.base_url}/act",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail: Optional[dict] = None
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except Exception:  # noqa: BLE001
                detail = None
            code = (detail or {}).get("code") or "policy_service_error"
            message = (detail or {}).get("error") or str(exc)
            raise LocalNavRuntimeError(code, message) from None
        except Exception as exc:  # noqa: BLE001
            raise LocalNavRuntimeError(
                "policy_service_unreachable",
                f"localnav policy service unreachable at {self.base_url}: {exc}",
            ) from None
        if not result.get("ok", False):
            raise LocalNavRuntimeError(
                str(result.get("code") or "policy_service_error"),
                str(result.get("error") or "policy service returned ok=false"),
            )
        return result
