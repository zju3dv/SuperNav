"""Policy backends behind the localnav service /act endpoint (numpy-only base)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

HFOV_DEG = 90.0
_STEP_LENGTH_M = 0.25
_FAN_SPREAD_RAD = math.radians(35.0)
_TURN_IN_STEPS = 4  # heading ramps to its target over the first N waypoints


def bearing_from_point(x: float, hfov_deg: float = HFOV_DEG) -> float:
    """Robot-frame bearing (rad, +left/CCW) of a normalized image column.

    Pinhole model: u = 2x−1 spans [−1, 1] across the image; a point right of
    center (u > 0) lies to the robot's right, i.e. a negative bearing.
    """
    u = 2.0 * float(x) - 1.0
    return -math.atan(u * math.tan(math.radians(hfov_deg) / 2.0))


def score_trajectories(
    trajectories: np.ndarray, goal_point: Sequence[float], hfov_deg: float = HFOV_DEG
) -> np.ndarray:
    """Score endpoint bearing and straightness in the hop-start frame.

    Rotation invalidates the fixed goal bearing; learned policies use consensus.
    """
    trajs = np.asarray(trajectories, dtype=np.float64)
    bearing = bearing_from_point(float(goal_point[0]), hfov_deg)
    endpoints = trajs.sum(axis=1)  # (M, 2)
    headings = np.arctan2(endpoints[:, 1], endpoints[:, 0])
    alignment = np.cos(headings - bearing)
    path_lengths = np.linalg.norm(trajs, axis=2).sum(axis=1)
    straightness = np.linalg.norm(endpoints, axis=1) / np.maximum(path_lengths, 1e-6)
    return 0.7 * alignment + 0.3 * straightness


def score_trajectories_consensus(
    trajectories: np.ndarray, straightness_weight: float = 0.3
) -> np.ndarray:
    """Mode-seeking score: endpoint alignment to the SAMPLE-MEAN displacement.

    The diffusion samples themselves carry the direction estimate (conditioned
    on the live context), so selection should amplify their consensus instead
    of trusting a stale geometric bearing. Near-zero consensus (terminal
    states emit zero-motion samples) degrades to straightness only.
    """
    trajs = np.asarray(trajectories, dtype=np.float64)
    endpoints = trajs.sum(axis=1)  # (M, 2)
    path_lengths = np.linalg.norm(trajs, axis=2).sum(axis=1)
    straightness = np.linalg.norm(endpoints, axis=1) / np.maximum(path_lengths, 1e-6)
    mean_endpoint = endpoints.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_endpoint))
    if mean_norm < 1e-6:
        return straightness_weight * straightness
    endpoint_norms = np.linalg.norm(endpoints, axis=1)
    alignment = (endpoints @ mean_endpoint) / (
        np.maximum(endpoint_norms, 1e-9) * mean_norm
    )
    return (1.0 - straightness_weight) * alignment + straightness_weight * straightness


class PolicyBackend:
    """Contract for /act backends. Subclasses return a dict with keys:
    trajectories (M×H×2 meters, robot frame, per-step deltas), scores (M),
    selected (int), temporal_distance (float|None), done_probability,
    confidence — the HTTP layer adds ok/latency_ms and serializes."""

    name = "base"
    model = ""
    checkpoint: "str | None" = None
    device = "cpu"

    def act(
        self,
        context_rgbs: List[np.ndarray],
        goal_rgb: np.ndarray,
        goal_point: Sequence[float],
        num_samples: int,
        denoise_steps: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DebugBackend(PolicyBackend):
    """Torch-free trajectory fan steered by the goal point's bearing.

    Mirrors DebugPointPolicy's role: exercises the full service + bridge-loop
    plumbing (shapes, scoring, selection) before a learned checkpoint exists.
    Not a navigation baseline — it never looks at the pixels.
    """

    name = "debug"
    model = "debug-point-fan"

    def act(
        self,
        context_rgbs: List[np.ndarray],
        goal_rgb: np.ndarray,
        goal_point: Sequence[float],
        num_samples: int,
        denoise_steps: int,
    ) -> Dict[str, Any]:
        del context_rgbs, goal_rgb, denoise_steps
        m = max(1, int(num_samples))
        bearing = bearing_from_point(float(goal_point[0]))
        horizon = 8
        trajectories = np.zeros((m, horizon, 2), dtype=np.float64)
        for i in range(m):
            fraction = 0.0 if m == 1 else (2.0 * i / (m - 1) - 1.0)
            total_turn = bearing + _FAN_SPREAD_RAD * fraction
            for k in range(horizon):
                heading = total_turn * min(1.0, (k + 1) / _TURN_IN_STEPS)
                trajectories[i, k, 0] = _STEP_LENGTH_M * math.cos(heading)
                trajectories[i, k, 1] = _STEP_LENGTH_M * math.sin(heading)
        scores = score_trajectories(trajectories, goal_point)
        return {
            "trajectories": trajectories,
            "scores": scores,
            "selected": int(np.argmax(scores)),
            "temporal_distance": None,
            "done_probability": 0.0,
            "confidence": 1.0,
        }


def build_backend(
    name: str,
    *,
    checkpoint: "str | None" = None,
    device: "str | None" = None,
    guidance_scale: "float | None" = None,
    selection: "str | None" = None,
) -> PolicyBackend:
    """Factory used by the service entrypoint; lazy-imports torch backends.

    guidance_scale None = auto: 2.0 for checkpoints trained with condition
    dropout, else 1.0. selection None = consensus; bearing uses the hop-start frame."""
    key = str(name or "debug").strip().lower()
    if key == "debug":
        return DebugBackend()
    if key == "nomad":
        from supernav.methods.localnav_policy.nomad_backend import NomadBackend

        return NomadBackend(
            checkpoint=checkpoint,
            device=device,
            guidance_scale=guidance_scale,
            selection=selection or "consensus",
        )
    raise ValueError(f"unknown localnav backend: {name!r} (expected debug|nomad)")
