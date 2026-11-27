"""Bridge-side localnav closed loop: learned point-goal hop execution.

This bridge-resident handler archives frames and executes movement through
``adapter.step_action``'s real discrete primitives (0.25 m / 10°) so
``collided`` and SPL telemetry stay honest — never ``set_agent_state``
teleports, never pathfinder/navmesh queries. RGB in, primitives out:
mapless-compliant by construction.

Registered as bridge actions 'navigate_with_localnav' / 'preload_localnav';
the LocalNavigateTool MCP wrapper is thin. Model inference lives in the
localnav_server process (see NAV_LOCALNAV_URL / NAV_LOCALNAV_AUTOSTART).
"""

from __future__ import annotations

from habitat_contract.navigation import NavigationBackend
from habitat_contract.navigation_state import HabitatAdapterError

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from collections import deque
from pathlib import Path
from supernav.paths import workspace_root
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from supernav.methods.localnav import (
    GoalSnapshot,
    HabitatActionAdapter,
    HabitatActionAdapterConfig,
    LocalNavError,
    snapshot_goal_from_registry,
)
from supernav.methods.localnav.policy_client import (
    LocalNavPolicyClient,
    select_trajectory,
    waypoint_deltas_to_actions,
)
from supernav.runtime.support.blocked_detector import BlockedDetector

_MAX_GOALS_PER_SESSION = 16
# A zero collision limit leaves termination to the blocked detector and step budget.
_DEFAULT_COLLISION_LIMIT = 0

# Per-session caches live for the lifetime of the bridge process.
_GOAL_CACHE: "Dict[str, OrderedDict[str, GoalSnapshot]]" = {}
_FRAME_COUNT: Dict[str, int] = {}
_LEG_COUNT: Dict[str, int] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _next_leg_id(session_id: str) -> int:
    leg = _LEG_COUNT.get(session_id, 0) + 1
    _LEG_COUNT[session_id] = leg
    return leg


def _save_frame(session_id: str, rgb: np.ndarray, *, leg_id: int, leg_step: int):
    """Persist per-replan first-person RGB (PIL, no cv2 dependency). Dedicated
    filename prefix so bridge visual captures can never clobber rollout frames."""
    if os.environ.get("NAV_LOCALNAV_SAVE_FRAMES", "1") != "1":
        return None, None
    from PIL import Image

    root = os.environ.get(
        "NAV_LOCALNAV_FRAME_ROOT",
        os.environ.get(
            "NAV_ARTIFACTS_DIR",
            str(workspace_root() / "data" / "runs" / "artifacts"),
        ),
    )
    directory = os.path.join(root, str(session_id))
    os.makedirs(directory, exist_ok=True)
    index = _FRAME_COUNT.get(session_id, 0) + 1
    _FRAME_COUNT[session_id] = index
    path = os.path.join(
        directory,
        f"localnav_{index:06d}_leg{int(leg_id):03d}_step{int(leg_step):03d}.png",
    )
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)
    return directory, path


def _save_goal_overlay(session_id: str, marked_rgb: np.ndarray, *, leg_id: int):
    """Persist the marked goal frame (hop-start view + the agent's clicked
    point) beside the rollout frames.

    Replay tooling needs the same artifact visual_point_navigate provides as
    `overlay_image`: without it a replay can show the robot walking but not
    WHERE the agent aimed, which is the one decision the agent actually made."""
    if os.environ.get("NAV_LOCALNAV_SAVE_FRAMES", "1") != "1":
        return None
    from PIL import Image

    root = os.environ.get(
        "NAV_LOCALNAV_FRAME_ROOT",
        os.environ.get(
            "NAV_ARTIFACTS_DIR",
            str(workspace_root() / "data" / "runs" / "artifacts"),
        ),
    )
    directory = os.path.join(root, str(session_id))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"localnav_goal_leg{int(leg_id):03d}.png")
    Image.fromarray(np.ascontiguousarray(marked_rgb)).save(path)
    return path


def _capture_observation(adapter: NavigationBackend, session) -> np.ndarray:
    """Render RGB at the current agent pose."""
    obs = adapter.sensor_observations(session)
    rgb = np.asarray(obs.get("color_sensor"))
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _agent_position(adapter: NavigationBackend, session) -> "Tuple[float, float, float]":
    position = adapter.agent_position(session)
    return (float(position[0]), float(position[1]), float(position[2]))


def _cache_goal(session_id: str, goal: GoalSnapshot) -> None:
    cache = _GOAL_CACHE.setdefault(session_id, OrderedDict())
    cache[goal.goal_id] = goal
    while len(cache) > _MAX_GOALS_PER_SESSION:
        cache.popitem(last=False)


def _split_into_primitives(payloads) -> list:
    """Expand quantized commands into single-primitive payloads.

    Mirrors the bridge's _resolve_n_steps decomposition (ceil(deg/10),
    ceil(dist/0.25)). Executing/filming per primitive keeps frame capture,
    physical-tempo playback, and collision granularity aligned — an
    aggregated 90° payload otherwise spins the camera between two frames.
    """
    primitives = []
    for payload in payloads:
        if payload.get("degrees") is not None:
            count = max(1, math.ceil(abs(float(payload["degrees"])) / 10.0))
            primitives.extend([{**payload, "degrees": 10.0}] * count)
        elif payload.get("distance") is not None:
            count = max(1, math.ceil(abs(float(payload["distance"])) / 0.25))
            primitives.extend([{**payload, "distance": 0.25}] * count)
        else:
            primitives.append(payload)
    return primitives


def _recover(
    adapter,
    session_id: str,
    rng: random.Random,
    frame_cb=None,
    streak: int = 1,
    direction: "str | None" = None,
    escalate: int = 0,
) -> int:
    """In-loop collision micro-recovery: back off, rotate away, resample.
    Semantically aligned with the MCP-level recovery gate, but runs inside the
    bridge loop without burning LLM rounds. Collisions during recovery are not
    double-counted. ``frame_cb(action)`` records per-primitive frames when
    real-motion video capture is enabled.

    ``escalate`` > 0 increases backoff and rotation over consecutive collisions
    while keeping the turn direction fixed.
    """
    steps = 0
    back = 0.25
    # Draw direction before angle to keep seeded recovery deterministic.
    if direction is None:
        direction = rng.choice(("turn_left", "turn_right"))
    turn = float(rng.choice((20.0, 30.0)))
    if escalate > 0 and streak > 1:
        k = min(int(streak), int(escalate))
        back = 0.25 * k
        turn = min(20.0 + 25.0 * (k - 1), 90.0)
    payloads = (
        {"action": "move_backward", "distance": back, "include_metrics": True},
        {
            "action": direction,
            "degrees": turn,
            "include_metrics": True,
        },
    )
    for payload in payloads:
        try:
            result = adapter.step_action(session_id, payload)
        except Exception:  # noqa: BLE001 — recovery is best-effort
            break
        steps += int(result.get("steps_taken") or 1)
        if frame_cb is not None:
            frame_cb(str(payload["action"]))
    return steps


# Panorama capture yaw offsets, mirroring the adapter's
# _PANORAMA_TURN_RIGHT_DEGREES: a panorama restores the pre-capture heading,
# so at hop start "front" is the robot's actual facing and the other rows are
# that many degrees to its right.
_PANO_TURN_RIGHT_DEGREES = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}


def _align_to_goal_direction(
    adapter, session_id: str, direction: "str | None", frame_cb=None
) -> int:
    """Face the heading the goal frame was captured at, before the hop starts.

    The policy is trained with the goal pair's second half being the hop-START
    frame at the robot's own heading: at t=0 the current view and the goal
    view are the same image. Marking a point on a side/back panorama row and
    then driving from an unrotated pose breaks that invariant — the policy
    gets a goal 90-270 degrees away from everything it sees, which is outside
    its training distribution entirely. Rotating first restores the invariant
    and costs only turn primitives (no translation, so SPL is unaffected).
    """
    offset = _PANO_TURN_RIGHT_DEGREES.get(str(direction or "front").lower())
    if not offset:
        return 0
    action = "turn_right" if offset <= 180.0 else "turn_left"
    degrees = offset if offset <= 180.0 else 360.0 - offset
    steps = 0
    for _ in range(max(1, int(round(degrees / 10.0)))):
        try:
            result = adapter.step_action(
                session_id,
                {"action": action, "degrees": 10.0, "include_metrics": True},
            )
        except Exception:  # noqa: BLE001 — alignment is best-effort
            break
        steps += int(result.get("steps_taken") or 1)
        if frame_cb is not None:
            frame_cb(action)
    return steps


def preload_localnav(adapter, session_id, payload) -> Dict[str, Any]:
    """Bridge action handler: ensure the policy service is reachable.

    Optionally autostarts a local server subprocess (NAV_LOCALNAV_AUTOSTART=1)
    so the launcher can warm everything before MCP accepts requests."""
    del adapter, session_id, payload
    client = LocalNavPolicyClient()
    try:
        health = client.healthz()
        return {"ok": True, "status": "ready", "autostarted": False, **_health_fields(health)}
    except LocalNavError:
        if not _env_flag("NAV_LOCALNAV_AUTOSTART", default=False):
            return {
                "ok": False,
                "code": "policy_service_unreachable",
                "error": (
                    f"localnav policy service unreachable at {client.base_url}; start "
                    "python -m supernav.methods.localnav.server, set NAV_LOCALNAV_URL, or "
                    "set NAV_LOCALNAV_AUTOSTART=1"
                ),
            }

    parsed = urlparse(client.base_url)
    process = subprocess.Popen(  # noqa: S603 — trusted local script
        [
            sys.executable,
            "-m",
            "supernav.methods.localnav.server",
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 18914),
        ],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    deadline = time.monotonic() + _env_float("NAV_LOCALNAV_PRELOAD_TIMEOUT", 300.0)
    while time.monotonic() < deadline:
        try:
            health = client.healthz()
            return {
                "ok": True,
                "status": "ready",
                "autostarted": True,
                "server_pid": process.pid,
                **_health_fields(health),
            }
        except LocalNavError:
            if process.poll() is not None:
                return {
                    "ok": False,
                    "code": "policy_service_start_failed",
                    "error": f"autostarted localnav server exited with {process.returncode}",
                }
            time.sleep(0.5)
    return {
        "ok": False,
        "code": "policy_service_timeout",
        "error": "localnav policy service did not become healthy before the preload timeout",
    }


def _health_fields(health: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "backend": health.get("backend"),
        "model": health.get("model"),
        "ckpt": health.get("ckpt"),
        "device": health.get("device"),
    }


def navigate_with_localnav(
    adapter: NavigationBackend, session_id, payload, *, policy_client=None
) -> Dict[str, Any]:
    """Bridge action handler: run one point-goal hop on ``session_id``'s sim.

    payload: {image_ref: str, point: [x, y]} to snapshot a fresh goal, or
             {goal_id: str} to reuse a cached GoalSnapshot (retry after
             needs_agent without re-facing the target); optional max_steps.
    """
    try:
        session = adapter.navigation_session(session_id)
    except HabitatAdapterError:
        return {"ok": False, "code": "no_session", "error": f"no session {session_id}"}

    goal_id = payload.get("goal_id")
    if goal_id:
        goal = _GOAL_CACHE.get(session_id, {}).get(str(goal_id))
        if goal is None:
            return {
                "ok": False,
                "code": "goal_not_found",
                "error": f"goal_id {goal_id!r} is not cached for this session",
            }
    else:
        image_ref = payload.get("image_ref")
        point = payload.get("point")
        if not image_ref or point is None:
            return {
                "ok": False,
                "code": "invalid_request",
                "error": "navigate_with_localnav requires image_ref + point (or goal_id)",
            }
        try:
            goal = snapshot_goal_from_registry(
                getattr(session, "last_visual_image_refs", {}) or {},
                str(image_ref),
                getattr(session, "latest_visual_capture_seq", None),
                point,
            )
        except LocalNavError as exc:
            return {"ok": False, "code": exc.code, "error": exc.message}
        _cache_goal(session_id, goal)

    client = policy_client or LocalNavPolicyClient()
    try:
        client.healthz()
    except LocalNavError as exc:
        return {"ok": False, "code": exc.code, "error": exc.message}

    max_steps = int(payload.get("max_steps", _env_int("NAV_LOCALNAV_MAX_STEPS", 100)))
    # End execution before the tool transport timeout to avoid blocking later requests.
    deadline_s = float(payload.get(
        "deadline_s", _env_float("NAV_LOCALNAV_DEADLINE_S", 270.0)
    ))
    num_samples = _env_int("NAV_LOCALNAV_SAMPLES", 16)
    denoise_steps = _env_int("NAV_LOCALNAV_DENOISE_STEPS", 10)
    execute_actions = max(1, _env_int("NAV_LOCALNAV_EXEC_ACTIONS", 3))
    control_dt_s = _env_float("NAV_LOCALNAV_CONTROL_DT_S", 0.5)
    done_threshold = _env_float("NAV_LOCALNAV_DONE_THRESHOLD", 0.5)
    # Confirm stopping across replans and corroborate with motion or holding still.
    done_confirmations = max(1, _env_int("NAV_LOCALNAV_DONE_CONFIRMATIONS", 2))
    min_steps_before_done = _env_int("NAV_LOCALNAV_MIN_STEPS_BEFORE_DONE", 4)
    # Separate confirmations by motion unless the policy is holding still.
    confirm_spacing_steps = _env_int("NAV_LOCALNAV_CONFIRM_SPACING_STEPS", 3)
    confidence_threshold = _env_float("NAV_LOCALNAV_CONF_THRESHOLD", 0.2)
    collision_limit = _env_int(
        "NAV_LOCALNAV_COLLISION_LIMIT", _DEFAULT_COLLISION_LIMIT
    )

    action_adapter = HabitatActionAdapter(
        HabitatActionAdapterConfig(control_dt_s=control_dt_s)
    )
    detector = BlockedDetector(
        window_size=_env_int("NAV_LOCALNAV_BLOCKED_WINDOW", 12),
        radius_m=_env_float("NAV_LOCALNAV_BLOCKED_RADIUS_M", 0.35),
    )
    rng = random.Random()
    # Fixed mode keeps eight frames; GCA keeps full history for the shared memory schedule.
    _ctx_mode = getattr(
        client, "context_mode",
        getattr(getattr(client, "backend", None), "context_mode", "fixed"),
    )
    history: deque = deque(maxlen=None if _ctx_mode == "gca" else 8)
    leg_id = _next_leg_id(session_id)
    goal_overlay_path = _save_goal_overlay(
        session_id, goal.marked_rgb, leg_id=leg_id
    )
    # Fresh side/back-row goals: face the capture heading first (see
    # _align_to_goal_direction). Skipped for goal_id retries — their stored
    # direction refers to a pose the robot has since left, and a mid-hop
    # current-vs-start difference is exactly what training covers.
    alignment_steps = 0

    # Shared progress and cancel files remain accessible while the single-threaded bridge is busy.
    _ctl_dir = Path(
        os.environ.get("NAV_LOCALNAV_FRAME_ROOT")
        or os.environ.get("NAV_ARTIFACTS_DIR")
        or str(workspace_root() / "data" / "runs" / "artifacts")
    ) / str(session_id)
    _ctl_dir.mkdir(parents=True, exist_ok=True)
    _progress_path = _ctl_dir / "localnav.progress.json"
    _cancel_path = _ctl_dir / "localnav.cancel"
    if _cancel_path.exists():
        _cancel_path.unlink()

    status = "timeout"
    needs_agent_reason: Optional[str] = None
    last_summary: Dict[str, Any] = {}
    done_probability = 0.0
    temporal_distance = None
    confidence = None
    primitive_steps = 0
    replans = 0
    collisions_total = 0
    consecutive_collisions = 0
    recovery_direction: Optional[str] = None
    # Zero uses fixed recovery; positive values cap escalation across consecutive collisions.
    recovery_escalate = _env_int("NAV_LOCALNAV_RECOVERY_ESCALATE", 0)
    consecutive_done = 0
    done_streak_start_steps: Optional[int] = None
    last_cycle_primitives: Optional[int] = None
    replan_records = []
    frames_dir = None
    frame_paths = []
    frame_events = []
    frames_enabled = os.environ.get("NAV_LOCALNAV_SAVE_FRAMES", "1") == "1"
    started = time.monotonic()
    # Traveled xz path length (SPL telemetry). Sampled after every primitive
    # (and after recovery), so each segment is a straight 0.25 m forward step
    # or a 0-length turn — exact for our discrete action space.
    path_length_m = 0.0
    collision_positions: list = []  # xz of every collided primitive (heatmap telemetry)
    last_track_position = _agent_position(adapter, session)

    def _track_motion() -> None:
        nonlocal path_length_m, last_track_position
        now = _agent_position(adapter, session)
        path_length_m += math.hypot(
            now[0] - last_track_position[0], now[2] - last_track_position[2]
        )
        last_track_position = now

    def _record_frame(rgb_frame, kind: str, action: "str | None" = None) -> None:
        nonlocal frames_dir
        frames_dir, path = _save_frame(
            session_id, rgb_frame, leg_id=leg_id, leg_step=len(frame_paths) + 1
        )
        if path:
            frame_paths.append(path)
            frame_events.append({"path": path, "kind": kind, "action": action})

    def _record_step_frame(action: str, kind: str = "step") -> None:
        # Per-primitive capture feeds BOTH the policy context history (so the
        # served 4-frame context has the same 0.25 m / 10° spacing the model
        # was trained on) and, when enabled, the real-motion video frames.
        step_rgb = _capture_observation(adapter, session)
        history.append(step_rgb)
        if frames_enabled:
            _record_frame(step_rgb, kind, action)

    # Fresh side/back-row goals: face the capture heading first (see
    # _align_to_goal_direction). Skipped for goal_id retries — their stored
    # direction refers to a pose the robot has since left, and a mid-hop
    # current-vs-start difference is exactly what training covers. Filmed
    # through the normal per-primitive path so the replay shows the turn
    # instead of an unexplained heading jump.
    if not goal_id:
        alignment_steps = _align_to_goal_direction(
            adapter,
            session_id,
            goal.direction,
            frame_cb=lambda action: _record_step_frame(action, kind="align"),
        )

    # Append history per primitive to match the training frame spacing.
    while primitive_steps < max_steps:
        rgb = _capture_observation(adapter, session)
        if not history:
            history.append(rgb)
        detector.record(_agent_position(adapter, session))
        _record_frame(rgb, "replan")
        if detector.is_blocked():
            status = "blocked"
            break

        # Send the CLEAN goal frame: the conditioning form (marker / crop /
        # coordinate embedding) is a backend-side choice rendered at policy
        # resolution from goal_point — a marker drawn at capture resolution
        # would shrink to ~2px after the 96×96 resize and wash out.
        # goal.marked_rgb remains for artifacts and human debugging.
        if time.monotonic() - started > deadline_s:
            status = "timeout"
            needs_agent_reason = "deadline_exceeded"
            break
        if _cancel_path.exists():
            status = "needs_agent"
            needs_agent_reason = "cancelled_by_agent"
            break
        try:
            _progress_path.write_text(json.dumps({
                "leg_id": leg_id, "goal_id": goal.goal_id,
                "steps": primitive_steps, "replans": replans,
                "collisions": collisions_total,
                "done_probability": round(done_probability, 4),
                "elapsed_s": round(time.monotonic() - started, 1),
                "status": "running",
            }))
        except OSError:
            pass
        try:
            response = client.act(
                list(history),
                goal.rgb,
                goal.point.as_tuple(),
                num_samples,
                denoise_steps,
            )
        except LocalNavError as exc:
            return {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "goal_id": goal.goal_id,
                "steps": primitive_steps,
                "replans": replans,
                "collisions": collisions_total,
            }
        replans += 1
        done_probability = float(response.get("done_probability") or 0.0)
        temporal_distance = response.get("temporal_distance")
        confidence = response.get("confidence")
        # Always-on stop-probability trace (independent of frame saving):
        # eval sweeps thresholds offline from these records.
        replan_records.append(
            {
                "replan": replans,
                "steps": primitive_steps,
                "done_probability": round(done_probability, 4),
                "confidence": None if confidence is None else round(float(confidence), 4),
            }
        )
        if frame_events and frame_events[-1]["kind"] == "replan":
            frame_events[-1]["done_probability"] = round(done_probability, 4)
        if done_probability >= done_threshold:
            consecutive_done += 1
            if consecutive_done == 1:
                done_streak_start_steps = primitive_steps
            holding_still = last_cycle_primitives == 0
            spaced = (
                primitive_steps - (done_streak_start_steps or 0)
                >= confirm_spacing_steps
            )
            if consecutive_done >= done_confirmations and (
                (primitive_steps >= min_steps_before_done and spaced)
                or holding_still
            ):
                status = "reached"
                break
        else:
            consecutive_done = 0
            done_streak_start_steps = None
        if confidence is not None and float(confidence) < confidence_threshold:
            status = "needs_agent"
            needs_agent_reason = "low_confidence"
            break

        trajectories = np.asarray(response.get("trajectories"), dtype=np.float64)
        if trajectories.ndim != 3 or trajectories.shape[0] < 1:
            return {
                "ok": False,
                "code": "invalid_policy_response",
                "error": f"policy returned trajectories with shape {trajectories.shape}",
                "goal_id": goal.goal_id,
            }
        selected = select_trajectory(trajectories, response.get("scores"))
        chunk = waypoint_deltas_to_actions(trajectories[selected], control_dt_s)

        cycle_start_steps = primitive_steps
        cycle_start_collisions = collisions_total
        interrupted = False
        for velocity_action in chunk[:execute_actions]:
            for step_payload in _split_into_primitives(
                action_adapter.to_payloads(velocity_action)
            ):
                result = adapter.step_action(session_id, step_payload)
                primitive_steps += int(result.get("steps_taken") or 1)
                _track_motion()
                last_summary = result.get("state_summary") or last_summary
                _record_step_frame(str(step_payload["action"]))
                if result.get("collided"):
                    collisions_total += 1
                    consecutive_collisions += 1
                    hit = _agent_position(adapter, session)
                    collision_positions.append(
                        [round(hit[0], 3), round(hit[2], 3)]
                    )
                    interrupted = True
                    if collision_limit > 0 and consecutive_collisions >= collision_limit:
                        status = "needs_agent"
                        needs_agent_reason = "repeated_collisions"
                    else:
                        if recovery_escalate > 0 and consecutive_collisions == 1:
                            # Keep one recovery direction across consecutive collisions.
                            recovery_direction = rng.choice(
                                ("turn_left", "turn_right")
                            )
                        primitive_steps += _recover(
                            adapter,
                            session_id,
                            rng,
                            frame_cb=lambda action: _record_step_frame(action, "recover"),
                            streak=consecutive_collisions,
                            direction=recovery_direction,
                            escalate=recovery_escalate,
                        )
                        _track_motion()
                    break
                if primitive_steps >= max_steps:
                    interrupted = True
                    break
            if interrupted:
                break
        last_cycle_primitives = primitive_steps - cycle_start_steps
        # Per-cycle outcome telemetry (collision-critic training labels):
        # whether THIS replan's executed chunk hit anything.
        if replan_records:
            replan_records[-1]["cycle_collided"] = bool(
                collisions_total > cycle_start_collisions
            )
            replan_records[-1]["cycle_steps"] = last_cycle_primitives
        if needs_agent_reason == "repeated_collisions":
            break
        if not interrupted:
            consecutive_collisions = 0

    result: Dict[str, Any] = {
        "ok": True,
        "status": status,
        "goal_id": goal.goal_id,
        "image_ref": goal.image_ref,
        "point": list(goal.point.as_tuple()),
        "steps": primitive_steps,
        "replans": replans,
        "collisions": collisions_total,
        "path_length_m": round(path_length_m, 3),
        # Mapless responses hide world positions; evaluator telemetry uses a separate protocol.
        "collision_positions": (
            collision_positions
            if not getattr(session, "mapless", False)
            else []
        ),
        "done_probability": round(done_probability, 4),
        "temporal_distance": temporal_distance,
        "confidence": confidence,
        "replan_records": replan_records,
        "duration_s": round(time.monotonic() - started, 3),
        "frames_dir": frames_dir,
        "goal_overlay_image": goal_overlay_path,
        "alignment_turn_steps": alignment_steps,
        "frame_paths": frame_paths,
        "frame_events": frame_events,
        "images": [{"path": p, "type": "image"} for p in frame_paths[-3:]],
        "state_summary": last_summary,
    }
    if needs_agent_reason:
        result["needs_agent_reason"] = needs_agent_reason
    try:
        _progress_path.write_text(json.dumps({
            "leg_id": leg_id, "goal_id": goal.goal_id,
            "steps": primitive_steps, "replans": replans,
            "collisions": collisions_total,
            "done_probability": round(done_probability, 4),
            "elapsed_s": round(time.monotonic() - started, 1),
            "status": status,
        }))
    except OSError:
        pass
    return result
