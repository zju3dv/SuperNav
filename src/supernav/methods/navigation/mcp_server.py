#!/usr/bin/env python3
"""Expose Habitat bridge actions as MCP tools in an independent process.

ToolRegistry supplies hab_ tools and their explicit metadata. Typed
wrappers create ToolContext, dispatch calls and serialize results.
NAV_BRIDGE_HOST and NAV_BRIDGE_PORT select the bridge; NAV_ARTIFACTS_DIR
selects artifact resources.

Usage: python -m supernav mcp --help
"""

from __future__ import annotations

import base64
from supernav.methods.navigation.result_projection import (
    _MODEL_HIDDEN_RESULT_KEYS, _VISIBLE_NAV_TARGET_FIELDS, _model_hidden_result_key, _normalize_visible_nav_targets, _redact_global_task_result, _strip_visual_ground_private_fields,
)
from supernav.methods.navigation.passability import (
    _mcp_float_or_none, _mcp_grid_cells,
    _mcp_depth_grid_judgment, _mcp_depth_analyze_judgment,
)
import inspect
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from supernav.paths import workspace_root
from typing import Any, Dict, List, Optional

from supernav.backends.habitat.trajectory import (
    McpTrajectoryLog,
    extract_pose,
    extract_step_count,
    extract_trajectory_points,
)
from supernav.methods.navigation.nav_goals_ledger import get_nav_goals_ledger

# Importing the navigation tools triggers ToolRegistry.register for
# every tool as a side effect, so the registry is populated before
# we try to enumerate it below.
import supernav.methods.navigation.tools  # noqa: F401
from supernav.evaluation.measurement.model_visible_gt_guard import (
    strip_model_visible_eval_fields,
)
from supernav.methods.navigation.tools._common import parse_bool_flag
from supernav.backends.habitat.bridge_client import BridgeClient
from supernav.runtime.support.config import (
    load_dotenv_from_project,
    restrict_bind_host_to_loopback,
)
from supernav.runtime.support.image_io import (
    is_depth_image_path,
    read_image_b64,
    select_llm_image_paths,
)
from supernav.methods.navigation.tools.base import RoundState, Tool, ToolContext, ToolRegistry
from supernav.methods.navigation.spatial_memory import (
    SpatialMemoryError,
    SpatialMemoryLedger,
)

try:
    from mcp.server.fastmcp import FastMCP, Image
except ImportError:
    sys.exit('ERROR: compatible MCP SDK unavailable. Run: pip install "mcp>=1.27,<2"')


# ---------------------------------------------------------------------------
# Bridge client + MCP state
# ---------------------------------------------------------------------------
# NOTE: Single-client only. All tools share one global BridgeClient.
# For multi-client transports (sse/streamable-http), per-connection
# isolation requires FastMCP to expose connection-scoped context,
# which is not yet available. Using concurrent MCP clients against
# the same server will cause session conflicts.
_bridge = BridgeClient()
_mcp_is_gaussian: bool = False  # mirrored from InitSceneTool after hab_init
_mcp_round_state = RoundState()
_mcp_latest_panorama_images: List[Dict[str, Any]] = []
_mcp_tool_sequence = 0


def _spatial_memory_enabled() -> bool:
    return parse_bool_flag(os.environ.get("HAB_MCP_SPATIAL_MEMORY"), default=False)


def _global_task_mode() -> bool:
    return parse_bool_flag(os.environ.get("HAB_MCP_GLOBAL_TASK"), default=False)


_spatial_memory = SpatialMemoryLedger(
    max_junctions=int(os.environ.get("HAB_MCP_SPATIAL_MEMORY_JUNCTIONS", "3")),
    long_move_m=float(os.environ.get("HAB_MCP_SPATIAL_MEMORY_LONG_MOVE_M", "3.0")),
)

# Multi-goal episode ledger; reloaded from HAB_MCP_NAV_GOALS_JSON on each
# init_scene so one MCP process can serve sequential benchmark episodes.
_nav_goals = get_nav_goals_ledger()


_MCP_PRIMARY_NAV_SKILL = "habitat-visible-target-nav"
_MCP_PASSABILITY_SKILL = "habitat-passability-check"
_MCP_OBSTACLE_SKILL = "habitat-obstacle-avoidance"
_MCP_GATE_ENV = "HAB_MCP_GATE_ENABLED"


def _new_mcp_passability_state() -> Dict[str, Any]:
    return {
        "action_seq": 0,
        "last_depth_grid": None,
        "last_depth_analyze": None,
        "last_side_depth_grid": None,
        "last_downward_look": None,
        "last_reminder": None,
        "post_collision_recovery_requires_check": False,
    }


_mcp_passability_state: Dict[str, Any] = _new_mcp_passability_state()


def _mcp_gate_enabled() -> bool:
    return parse_bool_flag(os.environ.get(_MCP_GATE_ENV), default=True)


def _mcp_capture_height_y() -> float:
    return float(os.environ.get("HAB_SENSOR_CAM_HEIGHT", "1.2"))


def _artifacts_dir() -> str:
    return os.environ.get(
        "NAV_ARTIFACTS_DIR",
        str(workspace_root() / "data" / "runs" / "artifacts"),
    )


_trajectory_log = McpTrajectoryLog(_artifacts_dir())
_private_trajectory_sessions: set[str] = set()


_AUDIT_TRAJECTORY_PAYLOAD = {"include_trajectory": True}
_AUDIT_POSE_FALLBACK_TOOLS = frozenset(
    {
        "backward",
        "forward",
        "turn",
        "navigate",
        "visual_local_navigate",
        "local_navigate",
        "visual_ground_preview",
        "visual_overlay_navigate",
        "visual_point_navigate",
        "oracle_local_navigate",
        "reset_agent_pose",
        "look",
        "look_vertical",
        "panorama",
    }
)


def _refresh_trajectory_log_root() -> McpTrajectoryLog:
    """Keep the sidecar root in sync with NAV_ARTIFACTS_DIR.

    Tests and users often set NAV_ARTIFACTS_DIR after module import.
    Rebuilding this lightweight helper per MCP call makes that env var
    effective without requiring a server restart.
    """
    global _trajectory_log
    current = Path(_artifacts_dir())
    if _trajectory_log.artifacts_dir != current:
        _trajectory_log = McpTrajectoryLog(current)
    return _trajectory_log


def _mark_private_trajectory_session(session_id: str) -> None:
    if session_id:
        _private_trajectory_sessions.add(session_id)


def _is_private_trajectory_session(session_id: str) -> bool:
    return bool(session_id and session_id in _private_trajectory_sessions)


def _response_redacts_absolute_pose(body: Dict[str, Any]) -> bool:
    """Return True when a response explicitly withholds absolute pose fields."""
    candidates: list[Dict[str, Any]] = []
    metrics = body.get("metrics")
    if isinstance(metrics, dict):
        for key in ("agent_state", "state_summary"):
            value = metrics.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    for key in ("agent_state", "state_summary"):
        value = body.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    return any("position" not in candidate for candidate in candidates)


def _capture_mcp_final_pose() -> tuple[
    str,
    Optional[Dict[str, Any]],
    str,
    list[list[float]],
    Optional[int],
]:
    """Best-effort final pose capture before close_session dispatch."""
    session_id = str(_bridge.session_id or "")
    if not session_id:
        return "", None, "close_session:no_active_session", [], None
    try:
        metrics = _bridge.call(
            "get_audit_metrics",
            _AUDIT_TRAJECTORY_PAYLOAD,
            timeout=3,
            audit_internal=True,
        )
        if isinstance(metrics, dict):
            pose = extract_pose(metrics)
            points = extract_trajectory_points(metrics)
            reason = "close_session"
            if pose is None:
                reason = "close_session:pose_unavailable"
            elif not isinstance(metrics.get("trajectory_points"), list):
                reason = "close_session:trajectory_unavailable"
            return (
                session_id,
                pose,
                reason,
                points,
                extract_step_count(metrics),
            )
    except Exception as exc:  # noqa: BLE001 - best-effort logging
        detail = " ".join(str(exc).split())[:240]
        return (
            session_id,
            None,
            f"close_session:get_metrics_failed:{type(exc).__name__}:{detail}",
            [],
            None,
        )
    return session_id, None, "close_session:pose_unavailable", [], None


def _capture_mcp_audit_state(
    *,
    session_id: str,
    reason: str,
) -> tuple[Optional[Dict[str, Any]], list[list[float]], Optional[int], str]:
    """Read the complete private trajectory prefix after one formal result."""
    if not session_id:
        return None, [], None, f"{reason}:no_active_session"
    try:
        metrics = _bridge.call(
            "get_audit_metrics",
            _AUDIT_TRAJECTORY_PAYLOAD,
            timeout=3,
            audit_internal=True,
        )
        if isinstance(metrics, dict):
            pose = extract_pose(metrics)
            points = extract_trajectory_points(metrics)
            if pose is None:
                diagnostic = f"{reason}:pose_unavailable"
            elif not isinstance(metrics.get("trajectory_points"), list):
                diagnostic = f"{reason}:trajectory_unavailable"
            else:
                diagnostic = reason
            return (
                pose,
                points,
                extract_step_count(metrics),
                diagnostic,
            )
    except Exception as exc:  # noqa: BLE001 - best-effort evaluator capture
        detail = " ".join(str(exc).split())[:240]
        return (
            None,
            [],
            None,
            f"{reason}:get_metrics_failed:{type(exc).__name__}:{detail}",
        )
    return None, [], None, f"{reason}:metrics_unavailable"


def _finish_mcp_trajectory(
    body: Dict[str, Any],
    *,
    session_id: str,
    pose: Optional[Dict[str, Any]],
    reason: str,
    trajectory_points: Optional[list[list[float]]] = None,
    tool_seq: Optional[int] = None,
    step_count: Optional[int] = None,
    diagnostic: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        log = _refresh_trajectory_log_root()
        path = log.finish(
            session_id=session_id,
            pose=pose,
            reason=reason,
            status="closed",
            trajectory_points=trajectory_points,
            tool_seq=tool_seq,
            step_count=step_count,
            diagnostic=diagnostic,
        )
    except Exception:  # noqa: BLE001 - sidecar logging is best-effort
        return body
    if path is not None and not _is_private_trajectory_session(session_id):
        body["trajectory_log"] = str(path)
    return body


def _record_mcp_trajectory_event(
    tool_name: str, body: Dict[str, Any], *, tool_seq: Optional[int] = None
) -> Dict[str, Any]:
    """Best-effort automatic trajectory logging for successful MCP calls."""
    try:
        log = _refresh_trajectory_log_root()
    except Exception:  # noqa: BLE001 - sidecar logging is best-effort
        return body
    session_id = ""
    if isinstance(body.get("session_id"), str):
        session_id = str(body["session_id"])
    elif _bridge.session_id:
        session_id = str(_bridge.session_id)

    pose = extract_pose(body)
    if tool_name == "init_scene":
        new_session_id = str(body.get("session_id") or "")
        if new_session_id:
            audit_pose, points, step_count, diagnostic = _capture_mcp_audit_state(
                session_id=new_session_id,
                reason=tool_name,
            )
            # The init result is the authoritative start sample when it carries
            # an absolute pose; the audit call primarily supplies the dense
            # prefix and fills pose only for redacted/minimal init responses.
            pose = pose or audit_pose
            try:
                log.start(
                    session_id=new_session_id,
                    scene=(
                        str(body.get("scene"))
                        if body.get("scene") is not None
                        else None
                    ),
                    pose=pose,
                    source_tool=tool_name,
                    source_response=body,
                    tool_seq=tool_seq,
                    step_count=step_count,
                    trajectory_points=points,
                    diagnostic=diagnostic,
                )
            except Exception:  # noqa: BLE001 - sidecar logging is best-effort
                pass
            # Do not return the init sidecar path. Task privacy is not known
            # and mapless callers must not be able to
            # read the server-side audit file by saving this path up front.
        return body

    if session_id:
        points: list[list[float]] = []
        diagnostic: Optional[str] = None
        step_count = extract_step_count(body)
        response_pose = pose
        if tool_name in _AUDIT_POSE_FALLBACK_TOOLS:
            audit_pose, points, audit_step_count, diagnostic = _capture_mcp_audit_state(
                session_id=session_id,
                reason=tool_name,
            )
            pose = audit_pose or pose
            if audit_step_count is not None:
                step_count = audit_step_count
            if (
                response_pose is None
                and pose is not None
                and _response_redacts_absolute_pose(body)
            ):
                # Redacted responses mean the absolute pose is audit-only.
                _mark_private_trajectory_session(session_id)
        try:
            log.sample(
                session_id=session_id,
                tool_name=tool_name,
                pose=pose,
                body=body,
                tool_seq=tool_seq,
                trajectory_points=points,
                step_count=step_count,
                diagnostic=diagnostic,
            )
        except Exception:  # noqa: BLE001 - sidecar logging is best-effort
            pass
        # Keep the path server-side until close_session. Exposing it before
        # returning it earlier could expose audit-only poses.
    return body


def _controller_measurement_dir() -> str:
    artifacts = Path(_artifacts_dir()).resolve()
    configured = os.environ.get("NAV_CONTROLLER_MEASUREMENT_DIR")
    if configured:
        candidate = Path(configured).resolve()
        if not _path_is_inside(candidate, artifacts):
            return str(candidate)
    return str(_default_controller_measurement_dir(artifacts))


def _default_controller_measurement_dir(artifacts: Path) -> Path:
    default_name = f"{artifacts.name or 'artifacts'}_controller"
    return (artifacts.parent / default_name / "measurement").resolve()


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


# Shape visual and session-init ToolResult bodies into the MCP top-level fields expected
# by clients. Other tool bodies pass through unchanged.


def _shape_init_scene_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Return scene_info, session identity and initial camera observations."""
    del args

    scene_info = _strip_agent_hidden_nav_target_fields(
        _strip_eval_only_fields(dict(body))
    )

    out = {
        "session_id": body.get("session_id"),
        "is_gaussian": body.get("is_gaussian", False),
        "scene_info": scene_info,
    }
    if "trajectory_log" in body:
        out["trajectory_log"] = body["trajectory_log"]

    if "panorama_error" in body:
        out["panorama_error"] = body["panorama_error"]
    rows = _normalize_panorama_images(body)
    if rows:
        paths = [row["path"] for row in rows]
        out["panorama_images"] = rows
        out["panorama_image_paths"] = paths
        out["images"] = paths
        if _visible_target_overlays_enabled() and isinstance(
            body.get("overlay_objlist"), dict
        ):
            out["overlay_objlist"] = body["overlay_objlist"]
        inline_paths = _llm_rgb_image_paths(paths, max_images=4)
    else:
        inline_paths = []
    return out, inline_paths


def _extract_visual_paths(body: Dict[str, Any]) -> List[str]:
    """Pull image file paths out of `visuals` (dict-of-sensor) and/or
    `images` / `panorama_images` list entries on a bridge response.
    Returns a flat ordered list of paths."""
    paths: List[str] = []
    visuals = body.get("visuals")
    if isinstance(visuals, dict):
        for sensor_data in visuals.values():
            if isinstance(sensor_data, dict):
                mp = sensor_data.get("path")
                if mp and isinstance(mp, str):
                    paths.append(mp)
    images = body.get("images")
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                mp = img.get("path")
                if mp and isinstance(mp, str):
                    paths.append(mp)
    panorama_images = body.get("panorama_images")
    if isinstance(panorama_images, list):
        for img in panorama_images:
            if isinstance(img, dict):
                mp = img.get("path")
                if mp and isinstance(mp, str):
                    paths.append(mp)
    return paths


def _strip_eval_only_fields(value: Any) -> Any:
    return strip_model_visible_eval_fields(value)


def _llm_rgb_image_paths(paths: List[str], *, max_images: int) -> List[str]:
    return _rgb_only_paths(select_llm_image_paths(paths))[:max_images]


def _visible_target_overlays_enabled() -> bool:
    return parse_bool_flag(
        os.environ.get("HAB_MCP_VISIBLE_TARGET_OVERLAYS"),
        default=False,
    )


_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS_ENV = "HAB_MCP_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS"
_DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS = 4


def _visible_target_overlay_max_objects() -> int:
    raw = os.environ.get(_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS


def _rgb_only_paths(paths: List[str]) -> List[str]:
    # Keep MCP visual parity RGB-only: depth PNGs remain text paths / numeric
    # analysis, never inline image blocks.
    return [path for path in paths if not _is_depth_capture_path(path)]


_PANORAMA_DIRECTIONS = ("front", "right", "back", "left")
_PANORAMA_TURN_RIGHT_DEGREES = {
    "front": 0,
    "right": 90,
    "back": 180,
    "left": 270,
}


def _normalize_panorama_images(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return direction-labelled panorama rows in canonical order.

    The bridge's `get_panorama` currently returns `images`; MCP movement
    responses expose the same observations as `panorama_images` so agents can
    distinguish final surround-camera views from rollout or image lists.
    """
    raw = body.get("panorama_images")
    if not isinstance(raw, list):
        raw = body.get("images")
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            direction = str(item.get("direction") or "").lower()
            if direction not in _PANORAMA_DIRECTIONS:
                continue
            row = dict(item)
            row["direction"] = direction
            row["view"] = direction
            row["turn_right_deg"] = _PANORAMA_TURN_RIGHT_DEGREES[direction]
            if _visible_target_overlays_enabled():
                overlay_path = row.get("overlay_path")
                if isinstance(overlay_path, str) and overlay_path:
                    row["path"] = overlay_path
            else:
                row.pop("overlay_path", None)
                row.pop("overlay_aliases", None)
            rows.append(row)
    order = {direction: idx for idx, direction in enumerate(_PANORAMA_DIRECTIONS)}
    rows.sort(key=lambda row: order.get(str(row.get("direction")), 999))
    return rows


def _front_only_observation() -> bool:
    return os.environ.get("HAB_MCP_OBSERVATION_MODE", "surround") == "front"


def _cache_mcp_panorama_images(rows: List[Dict[str, Any]]) -> None:
    global _mcp_latest_panorama_images
    if rows or _front_only_observation():
        _mcp_latest_panorama_images = [dict(row) for row in rows]


def _resolve_latest_view_image_ref(view: Any) -> str:
    view_name = str(view or "").strip().lower()
    if view_name not in _PANORAMA_DIRECTIONS:
        return ""
    for row in _mcp_latest_panorama_images:
        if str(row.get("direction") or row.get("view") or "").lower() != view_name:
            continue
        image_ref = row.get("image_ref")
        if isinstance(image_ref, str) and image_ref:
            return image_ref
    return ""


def _resolve_visual_view_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "visual_point_navigate" and _front_only_observation():
        latest = _resolve_latest_view_image_ref("front")
        if (
            args.get("view", "front") != "front"
            or (args.get("image_ref") and args["image_ref"] != latest)
            or not latest
        ):
            return {
                **args,
                "_view_resolution_error": "A point requires the latest front RGB image.",
            }
        return {**args, "image_ref": latest}
    if tool_name not in {
        "visual_ground_preview",
        "visual_point_navigate",
        "local_navigate",
    }:
        return args
    if args.get("image_ref"):
        return args
    view = args.get("view")
    if view is None:
        return args
    image_ref = _resolve_latest_view_image_ref(view)
    if not image_ref:
        view_name = str(view or "").strip().lower()
        if view_name not in _PANORAMA_DIRECTIONS:
            return {
                **args,
                "image_ref": "",
                "_view_resolution_error": (
                    "view must be one of front, right, back, or left"
                ),
            }
        return {
            **args,
            "image_ref": "",
            "_view_resolution_error": (
                "No latest panorama image_ref is cached for that view. Use a "
                "movement/perception tool that returns panorama_images, then "
                "call this tool with view='front', 'right', 'back', or 'left'."
            ),
        }
    resolved = dict(args)
    resolved["image_ref"] = image_ref
    return resolved


def _apply_benchmark_init_defaults(
    tool_name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Inject harness-owned spawn data without exposing it in the tool call."""
    if tool_name != "init_scene" or not _global_task_mode():
        return args
    raw = os.environ.get("HAB_MCP_INIT_DEFAULTS_JSON", "").strip()
    if not raw:
        return args
    try:
        defaults = json.loads(raw)
    except json.JSONDecodeError:
        return args
    if not isinstance(defaults, dict):
        return args
    merged = dict(args)
    # The benchmark owns episode selection and spawn.  Agent-supplied values
    # must not override the private row metadata injected by the harness.
    merged.update(defaults)
    return merged


def _next_mcp_tool_sequence(*, reset: bool = False) -> int:
    global _mcp_tool_sequence
    if reset:
        _mcp_tool_sequence = 0
    _mcp_tool_sequence += 1
    return _mcp_tool_sequence


def _benchmark_audit_path(session_id: str) -> Path:
    return Path(_artifacts_dir()) / f"{session_id}.benchmark_audit.jsonl"


def _write_benchmark_audit(
    *,
    session_id: str,
    tool_seq: int,
    tool_name: str,
    model_text: str,
    audit_body: Dict[str, Any],
) -> None:
    if not _global_task_mode() or not session_id:
        return
    record = {
        "schema_version": 1,
        "session_id": session_id,
        "tool_seq": tool_seq,
        "tool_name": f"hab_{tool_name}",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_content_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        "result": audit_body,
    }
    path = _benchmark_audit_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def _attach_spatial_memory(
    tool_name: str,
    args: Dict[str, Any],
    body: Dict[str, Any],
) -> Dict[str, Any]:
    if not _spatial_memory_enabled():
        return body
    memory = _spatial_memory.process_tool_result(tool_name, args, body)
    out = dict(body)
    out["spatial_memory"] = memory
    diagnostics = [
        event.get("diagnostic")
        for event in memory.get("new_events", [])
        if isinstance(event, dict) and isinstance(event.get("diagnostic"), dict)
    ]
    if diagnostics:
        out["spatial_diagnostics"] = diagnostics
    if tool_name == "close_session":
        blocked = next(
            (
                item
                for item in diagnostics
                if item.get("code") == "blocked_not_exhausted"
            ),
            None,
        )
        if blocked is not None:
            out["terminal_diagnostic"] = blocked
        audited = next(
            (
                item
                for item in diagnostics
                if item.get("code") == "blocked_frontier_audited"
            ),
            None,
        )
        if audited is not None:
            out["terminal_audit"] = audited
    return out


def _finalize_mcp_response(
    *,
    tool_name: str,
    tool_seq: int,
    body: Dict[str, Any],
    inline_paths: List[str],
    inline_images: bool,
) -> Any:
    audit_body = json.loads(json.dumps(body, default=str))
    model_body = _redact_global_task_result(body) if _global_task_mode() else body
    text = json.dumps(model_body, default=str)
    session_id = str(
        audit_body.get("session_id")
        or _spatial_memory.session_id
        or _bridge.session_id
        or ""
    )
    _write_benchmark_audit(
        session_id=session_id,
        tool_seq=tool_seq,
        tool_name=tool_name,
        model_text=text,
        audit_body=audit_body,
    )
    if inline_images:
        image_content = _inline_mcp_image_content(inline_paths, audit_body)
        if image_content:
            return [text, *image_content]
    return text


def _strip_agent_hidden_nav_target_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        k: v
        for k, v in body.items()
        if k
        not in (
            "visible_nav_targets",
            "visible_nav_targets_status",
            "matched_target_ref",
        )
    }
    if not _visible_target_overlays_enabled():
        out.pop("overlay_objlist", None)
    return out


def _panorama_image_paths(body: Dict[str, Any]) -> List[str]:
    return [
        str(row["path"])
        for row in _normalize_panorama_images(body)
        if isinstance(row.get("path"), str)
    ]


_MOTION_TOOLS_WITH_SURROUND_VIEW = frozenset(
    {
        "init_scene",
        "forward",
        "backward",
        "turn",
        "look_vertical",
        "navigate",
        "visual_local_navigate",
        "local_navigate",
        "visual_ground_preview",
        "visual_overlay_navigate",
        "visual_point_navigate",
        "oracle_local_navigate",
    }
)


def _capture_mcp_surround_view() -> Optional[Dict[str, Any]]:
    try:
        return _bridge.call(
            "get_panorama",
            {
                "include_depth_analysis": False,
                "front_only": _front_only_observation(),
                "output_dir": _artifacts_dir(),
                "agent_image_max_size": 256,
                "visible_target_overlay_max_objects": (
                    _visible_target_overlay_max_objects()
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001 - movement succeeded; report image failure
        return {"panorama_error": f"{type(exc).__name__}: {exc}"}


def _attach_mcp_surround_view(tool_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in _MOTION_TOOLS_WITH_SURROUND_VIEW:
        return body
    if _front_only_observation():
        # Discard pre-action frames and references even if fresh capture fails.
        _cache_mcp_panorama_images([])
        for key in (
            "panorama_images",
            "panorama_image_paths",
            "images",
            "visuals",
            "trace_frame_paths",
            "overlay_image",
            "overlay_objlist",
        ):
            body.pop(key, None)
        panorama = _capture_mcp_surround_view()
        rows = _normalize_panorama_images(panorama or {})
        if len(rows) != 1 or rows[0].get("direction") != "front":
            body["panorama_error"] = (panorama or {}).get(
                "panorama_error",
                "Front-only capture did not return exactly one front image.",
            )
            body["panorama_images"] = []
            return body
        _cache_mcp_panorama_images(rows)
        body["panorama_images"] = rows
        return body
    if (
        tool_name in {"visual_point_navigate", "visual_ground_preview"}
        and body.get("status") == "preview_ready"
    ):
        return body
    if "panorama_images" in body:
        rows = _normalize_panorama_images(body)
        _cache_mcp_panorama_images(rows)
        body["panorama_images"] = rows
        return body
    panorama = _capture_mcp_surround_view()
    if not isinstance(panorama, dict):
        return body
    if "panorama_error" in panorama:
        body["panorama_error"] = panorama["panorama_error"]
        return body
    rows = _normalize_panorama_images(panorama)
    _cache_mcp_panorama_images(rows)
    body["panorama_images"] = rows
    if _visible_target_overlays_enabled() and isinstance(
        panorama.get("overlay_objlist"), dict
    ):
        body["overlay_objlist"] = panorama["overlay_objlist"]
    return body


def _is_depth_capture_path(path: str) -> bool:
    return is_depth_image_path(Path(path).name)


def _shape_strip_visuals(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """For movement tools: strip sensor internals and expose surround views."""
    include_images = parse_bool_flag((args or {}).get("include_images"), default=True)
    out = {
        k: v
        for k, v in body.items()
        if k not in ("visuals", "images", "panorama_images", "trace_frame_paths")
    }
    out = _strip_eval_only_fields(out)
    out = _strip_agent_hidden_nav_target_fields(out)
    panorama_rows = _normalize_panorama_images(body)
    if panorama_rows:
        _cache_mcp_panorama_images(panorama_rows)
        paths = [row["path"] for row in panorama_rows]
        out["panorama_images"] = panorama_rows
        out["panorama_image_paths"] = paths
        out["images"] = paths
        inline_paths = _llm_rgb_image_paths(
            paths,
            max_images=4 if include_images else 0,
        )
        return out, inline_paths
    paths = _extract_visual_paths(body)
    out["images"] = paths
    inline_paths = _llm_rgb_image_paths(paths, max_images=1 if include_images else 0)
    return out, inline_paths


def _shape_visual_ground_preview_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """For visual_ground_preview: preview inlines the candidate overlay;
    confirm behaves like a movement and refreshes panorama context."""
    include_images = parse_bool_flag((args or {}).get("include_images"), default=True)
    if body.get("status") != "preview_ready":
        out, inline_paths = _shape_strip_visuals(body, args)
        out = _strip_visual_ground_private_fields(out)
        if body.get("ok") is not False:
            out.pop("image_ref", None)
            out.pop("direction", None)
            out["post_move_observation_reset"] = {
                "old_direction_expired": True,
                "instruction": (
                    "Previous image_ref/direction/confirm_token/candidate_id "
                    "were tied to the pre-move capture. Use only the latest "
                    "panorama_images for the next step."
                ),
            }
        return out, inline_paths

    out = {
        k: v
        for k, v in body.items()
        if k not in ("visuals", "images", "panorama_images", "trace_frame_paths")
    }
    out = _strip_eval_only_fields(out)
    out = _strip_visual_ground_private_fields(out)
    overlay = out.get("overlay_image")
    inline_paths = [overlay] if include_images and isinstance(overlay, str) else []
    return out, inline_paths


def _shape_visual_point_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    include_images = parse_bool_flag((args or {}).get("include_images"), default=True)
    if body.get("status") == "preview_ready":
        out = {
            k: v
            for k, v in body.items()
            if k not in ("visuals", "images", "panorama_images", "trace_frame_paths")
        }
        out = _strip_eval_only_fields(out)
        overlay = out.get("overlay_image")
        inline_paths = [overlay] if include_images and isinstance(overlay, str) else []
        return out, inline_paths

    out, inline_paths = _shape_strip_visuals(body, args)
    if body.get("ok") is not False:
        selected_anchor = out.get("selected_anchor")
        if not isinstance(selected_anchor, dict):
            selected_anchor = {}
        selected_anchor = dict(selected_anchor)
        if "image_ref" not in selected_anchor and body.get("image_ref") is not None:
            selected_anchor["image_ref"] = body.get("image_ref")
        if "direction" not in selected_anchor and body.get("direction") is not None:
            selected_anchor["direction"] = body.get("direction")
        if (
            "anchor_px" not in selected_anchor
            and body.get("selected_anchor_px") is not None
        ):
            selected_anchor["anchor_px"] = body.get("selected_anchor_px")
        if selected_anchor:
            out["selected_anchor"] = selected_anchor
        out.pop("image_ref", None)
        out.pop("direction", None)
        out["post_move_observation_reset"] = {
            "old_direction_expired": True,
            "instruction": (
                "Previous image_ref/direction/point were tied to the pre-move "
                "capture. Use only the latest panorama_images for the next step."
            ),
        }
    return out, inline_paths


def _shape_visual_overlay_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    out, inline_paths = _shape_strip_visuals(body, args)
    if body.get("ok") is not False:
        selected = out.get("selected_overlay_alias")
        if not isinstance(selected, dict):
            selected = {}
        selected = dict(selected)
        for key in ("image_ref", "target_alias", "direction"):
            if key not in selected and body.get(key) is not None:
                selected[key] = body.get(key)
        if selected:
            out["selected_overlay_alias"] = selected
        out.pop("image_ref", None)
        out.pop("target_alias", None)
        out.pop("direction", None)
        out["post_move_observation_reset"] = {
            "old_direction_expired": True,
            "instruction": (
                "Previous image_ref/target_alias/direction were tied to the "
                "pre-move capture. Use only the latest panorama_images for the "
                "next step."
            ),
        }
    return out, inline_paths


def _shape_topdown_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect visual and topdown image paths into the images field."""
    del args
    paths = _extract_visual_paths(body)
    td = body.get("topdown_map")
    if isinstance(td, dict):
        mp = td.get("path")
        if mp and isinstance(mp, str):
            paths.append(mp)
    out = {
        k: v for k, v in body.items() if k not in ("visuals", "images", "topdown_map")
    }
    out["images"] = paths
    return out


def _shape_oracle_local_map_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    include_image = parse_bool_flag((args or {}).get("include_image"), default=True)
    paths: List[str] = []
    map_image = body.get("map_image")
    if isinstance(map_image, dict):
        path = map_image.get("path")
        if isinstance(path, str) and path:
            paths.append(path)
    out = {k: v for k, v in body.items() if k not in ("map_image", "_debug")}
    out = _strip_eval_only_fields(out)
    out["images"] = paths
    return out, _llm_rgb_image_paths(paths, max_images=1 if include_image else 0)


def _shape_local_navigate_response(
    body: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Movement-tool shaping plus localnav rollout slimming.

    The surround view goes through the shared movement path so a finished hop
    inlines the four fresh direction images, exactly like every other movement
    tool. On top of that, the per-step rollout telemetry (frame paths, frame
    events, every replan record) is eval/audit material: it stays out of the
    model context, keeping only the last five replan records plus a total."""
    out, inline = _shape_strip_visuals(body, args)
    for key in ("frame_paths", "frame_events", "replan_records"):
        out.pop(key, None)
    records = body.get("replan_records") or []
    if records:
        out["replan_records_tail"] = records[-5:]
        out["replan_records_total"] = len(records)
    return out, inline


_MCP_RESPONSE_SHAPERS = {
    "init_scene": _shape_init_scene_response,
    "turn": _shape_strip_visuals,
    "look_vertical": _shape_strip_visuals,
    "forward": _shape_strip_visuals,
    "backward": _shape_strip_visuals,
    "navigate": _shape_strip_visuals,
    "visual_local_navigate": _shape_strip_visuals,
    "local_navigate": _shape_local_navigate_response,
    "visual_ground_preview": _shape_visual_ground_preview_response,
    "visual_overlay_navigate": _shape_visual_overlay_response,
    "visual_point_navigate": _shape_visual_point_response,
    "oracle_local_navigate": _shape_strip_visuals,
    "oracle_local_map": _shape_oracle_local_map_response,
    "topdown": _shape_topdown_response,
}


def _image_format_for_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    return None


def _inline_image_blocks(image_paths: List[str]) -> List[Image]:
    blocks: List[Image] = []
    artifacts_root = Path(_artifacts_dir()).resolve()
    for image_path in image_paths:
        if _is_depth_capture_path(image_path):
            continue
        image_format = _image_format_for_path(image_path)
        if image_format is None:
            continue
        path = Path(image_path).resolve()
        if not _path_is_inside(path, artifacts_root):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        blocks.append(Image(data=data, format=image_format))
    return blocks


def _inline_image_caption(path: str, body: Dict[str, Any]) -> str:
    for row in body.get("panorama_images") or []:
        if not isinstance(row, dict) or row.get("path") != path:
            continue
        direction = str(row.get("direction") or "").strip().lower()
        if direction == "front":
            return "Front view"
        if direction == "right":
            return "Right view"
        if direction == "back":
            return "Back view"
        if direction == "left":
            return "Left view"

    name = Path(path).name.lower()
    if "topdown" in name:
        return "Top-down map"
    if "third_person_color_sensor" in name:
        return "Third-person view"
    if "semantic_sensor" in name:
        return "Semantic view"
    if "color_sensor" in name:
        return "Current egocentric view"
    return f"Image: {Path(path).name}"


def _inline_mcp_image_content(
    image_paths: List[str],
    body: Dict[str, Any],
) -> List[Any]:
    content: List[Any] = []
    artifacts_root = Path(_artifacts_dir()).resolve()
    for image_path in image_paths:
        if _is_depth_capture_path(image_path):
            continue
        image_format = _image_format_for_path(image_path)
        if image_format is None:
            continue
        path = Path(image_path).resolve()
        if not _path_is_inside(path, artifacts_root):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        content.append(_inline_image_caption(image_path, body))
        content.append(Image(data=data, format=image_format))
    return content


def _build_mcp_context() -> ToolContext:
    """Build a ToolContext for one MCP tool call.

    MCP callers use `task_type="chat"` for session tools. `nav_mode`
    defaults to "navmesh" because the 4 navmesh-only tools (navigate,
    find_path, sample_point, topdown) should also be exposed via MCP
    — a mapless caller can still request them, and the bridge will
    return a real error if the scene has no navmesh.

    When the MCP gate is enabled, MCP keeps a shared `RoundState`
    across calls so dispatch-level safety gates (for example collision
    recovery) can enforce ordered multi-call protocols such as forward
    -> backward -> depth_analyze -> look_down. When disabled, each MCP
    call gets a fresh RoundState so those MCP-only gates cannot persist
    and block later calls.
    """
    return ToolContext(
        bridge=_bridge,
        session_id=_bridge.session_id or "",
        output_dir=_artifacts_dir(),
        nav_mode="navmesh",
        task_type="chat",
        is_gaussian=_mcp_is_gaussian,
        round_state=_mcp_round_state if _mcp_gate_enabled() else RoundState(),
        dispatch_source="mcp",
        controller_measurement_dir=_controller_measurement_dir(),
    )


def _reset_mcp_round_state() -> None:
    global _mcp_latest_panorama_images, _mcp_passability_state, _mcp_round_state
    _mcp_round_state = RoundState()
    _mcp_latest_panorama_images = []
    _mcp_passability_state = _new_mcp_passability_state()


def _mcp_skill_reminder_payload(
    *,
    status: str,
    risk_reason: Optional[List[str]] = None,
    suggested_next_tool: str = "hab_depth_grid",
) -> Dict[str, Any]:
    return {
        "status": status,
        "primary_skill": _MCP_PRIMARY_NAV_SKILL,
        "supporting_skill": _MCP_PASSABILITY_SKILL,
        "state_hint": "CHECK_CORRIDOR",
        "risk_reason": risk_reason or [],
        "suggested_next_tool": suggested_next_tool,
    }


def _mcp_collision_recovery_payload(error: str) -> Dict[str, Any]:
    return {
        "error": error,
        "required_skill": _MCP_OBSTACLE_SKILL,
        "supporting_skill": _MCP_PASSABILITY_SKILL,
        "primary_skill": _MCP_PRIMARY_NAV_SKILL,
        "recovery_sequence": [
            "hab_backward(distance_m=...)",
            "hab_depth_grid(rows=5, cols=5) or hab_depth_analyze()",
            "hab_look_vertical(direction='down', degrees>=60 cumulative)",
            "hab_passability_check(direction='forward', distance_m=...)",
        ],
    }


def _mcp_reminder_error(message: str, *, suggested_next_tool: str) -> Dict[str, Any]:
    payload = _mcp_skill_reminder_payload(
        status="requires_passability_check",
        risk_reason=["missing_or_stale_passability_reminder"],
        suggested_next_tool=suggested_next_tool,
    )
    payload["error"] = message
    payload["next_required_tool"] = suggested_next_tool
    return payload


def _mcp_invalidate_passability_reminder() -> None:
    _mcp_passability_state["last_reminder"] = None


def _mcp_record_pose_or_camera_action(tool_name: str) -> None:
    if tool_name in {
        "init_scene",
        "forward",
        "backward",
        "turn",
        "look_vertical",
        "visual_local_navigate",
        "visual_overlay_navigate",
        "visual_point_navigate",
        "oracle_local_navigate",
        "panorama",
        "close_session",
    }:
        _mcp_passability_state["action_seq"] += 1
        _mcp_invalidate_passability_reminder()


def _mcp_record_evidence(
    tool_name: str, args: Dict[str, Any], body: Dict[str, Any]
) -> None:
    seq = int(_mcp_passability_state.get("action_seq", 0))
    evidence = {
        "tool": tool_name,
        "action_seq": seq,
        "args": dict(args),
        "body": dict(body),
    }
    if tool_name == "depth_grid":
        _mcp_passability_state["last_depth_grid"] = evidence
    elif tool_name == "depth_analyze":
        _mcp_passability_state["last_depth_analyze"] = evidence
    elif tool_name == "side_depth_grid":
        _mcp_passability_state["last_side_depth_grid"] = evidence
    elif tool_name == "look_vertical" and args.get("direction") == "down":
        _mcp_passability_state["last_downward_look"] = evidence


def _mcp_compute_passability(
    *,
    direction: str,
    distance_m: float,
) -> Dict[str, Any]:
    if direction != "forward":
        return _mcp_skill_reminder_payload(
            status="uncertain",
            risk_reason=["unsupported_direction"],
            suggested_next_tool="hab_turn",
        )

    grid = _mcp_passability_state.get("last_depth_grid")
    current_seq = int(_mcp_passability_state.get("action_seq", 0))
    if isinstance(grid, dict) and grid.get("action_seq") == current_seq:
        judgment = _mcp_depth_grid_judgment(grid, distance_m=distance_m)
    else:
        depth_analyze = _mcp_passability_state.get("last_depth_analyze")
        if not (
            isinstance(depth_analyze, dict)
            and depth_analyze.get("action_seq") == current_seq
        ):
            depth_analyze = None
        judgment = _mcp_depth_analyze_judgment(depth_analyze)

    payload = _mcp_skill_reminder_payload(
        status=str(judgment.get("status", "uncertain")),
        risk_reason=list(judgment.get("risk_reason") or []),
        suggested_next_tool=str(
            judgment.get("suggested_next_tool") or "hab_depth_grid"
        ),
    )
    payload["direction"] = direction
    payload["distance_m"] = distance_m
    payload["evidence_summary"] = judgment.get("evidence_summary", {})
    if _mcp_passability_state.get("post_collision_recovery_requires_check"):
        payload["state_hint"] = "REACQUIRE or CHECK_CORRIDOR"
        payload["supporting_skill"] = _MCP_OBSTACLE_SKILL
        payload["secondary_supporting_skill"] = _MCP_PASSABILITY_SKILL
    if payload["status"] == "passable_hint":
        payload["valid_for"] = "one_forward"
        payload["expires_on"] = [
            "forward",
            "backward",
            "turn",
            "look_vertical",
            "panorama",
            "init",
            "close",
        ]
    return payload


def _mcp_store_passability_reminder(payload: Dict[str, Any]) -> None:
    _mcp_passability_state["last_reminder"] = {
        "action_seq": int(_mcp_passability_state.get("action_seq", 0)),
        "status": payload.get("status"),
        "direction": payload.get("direction"),
        "distance_m": payload.get("distance_m"),
        "used": False,
        "payload": dict(payload),
    }


def _mcp_forward_gate(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stage = _mcp_round_state.collision_recovery_stage
    if stage is not None:
        return _mcp_collision_recovery_payload(
            "collision recovery required before hab_forward"
        )
    reminder = _mcp_passability_state.get("last_reminder")
    if not isinstance(reminder, dict):
        return _mcp_reminder_error(
            "hab_forward requires hab_passability_check first",
            suggested_next_tool="hab_passability_check",
        )
    if reminder.get("used"):
        return _mcp_reminder_error(
            "hab_forward passability reminder was already consumed",
            suggested_next_tool="hab_passability_check",
        )
    if reminder.get("action_seq") != int(_mcp_passability_state.get("action_seq", 0)):
        return _mcp_reminder_error(
            "hab_forward passability reminder is stale after a pose or camera action",
            suggested_next_tool="hab_passability_check",
        )
    if reminder.get("status") != "passable_hint":
        payload = dict(reminder.get("payload") or {})
        payload["error"] = "hab_forward blocked by passability reminder"
        payload["next_required_tool"] = payload.get(
            "suggested_next_tool", "hab_depth_grid"
        )
        return payload
    expected_distance = _mcp_float_or_none(reminder.get("distance_m"))
    requested_distance = _mcp_float_or_none(args.get("distance_m", 0.5)) or 0.5
    if expected_distance is not None and requested_distance > expected_distance:
        return _mcp_reminder_error(
            "hab_forward distance exceeds the checked passability distance",
            suggested_next_tool="hab_passability_check",
        )
    reminder["used"] = True
    return None


def hab_passability_check(
    *,
    direction: str = "forward",
    distance_m: float = 0.5,
) -> str:
    """Return MCP-only skill reminders and conservative forward passability."""
    payload = _mcp_compute_passability(
        direction=direction,
        distance_m=float(distance_m),
    )
    _mcp_store_passability_reminder(payload)
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Dynamic wrapper factory
# ---------------------------------------------------------------------------

# JSON Schema types → Python types. FastMCP uses __annotations__ to
# build the MCP protocol schema, so the annotations must resolve to
# real Python types the caller can pass by kwarg.
_JSON_TYPE_TO_PYTHON = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _extract_params(tool: Tool) -> List[inspect.Parameter]:
    """Walk a Tool's parameters_schema and produce an ordered list of
    `inspect.Parameter` objects suitable for setting on a wrapper's
    `__signature__`.

    The ordering is: required params first, then optional params.
    This matters because FastMCP surfaces `required: [...]` in the MCP
    schema based on whether each parameter has a default.
    """
    schema = tool.metadata.parameters_schema or {}
    properties: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
    required: set = set(schema.get("required", []) or [])

    required_params: List[inspect.Parameter] = []
    optional_params: List[inspect.Parameter] = []

    for pname, pdef in properties.items():
        if (
            _front_only_observation()
            and tool.metadata.name == "visual_point_navigate"
            and pname in {"view", "image_ref"}
        ):
            continue
        if (
            os.environ.get("HAB_MCP_MINIMAL_TASK") == "1"
            and tool.metadata.name == "init_scene"
        ):
            continue
        if (
            os.environ.get("HAB_MCP_MINIMAL_TASK") == "1"
            and tool.metadata.name == "close_session"
            and pname not in {"outcome", "reason"}
        ):
            continue
        ptype_name = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
        ptype = _JSON_TYPE_TO_PYTHON.get(ptype_name, str)

        if pname in required:
            required_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=ptype,
                )
            )
        else:
            default = pdef.get("default", None) if isinstance(pdef, dict) else None
            optional_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=ptype,
                )
            )

    return required_params + optional_params


def _mcp_tool_description(tool: Tool) -> str:
    if os.environ.get("HAB_MCP_MINIMAL_TASK") == "1":
        observation = (
            "one current front RGB image"
            if _front_only_observation()
            else "front/right/back/left RGB images"
        )
        descriptions = {
            "init_scene": f"Initialize the configured episode. Call before other tools. Returns {observation}.",
            "turn": f"Rotate left or right by degrees. Returns {observation}.",
            "visual_point_navigate": (
                "Move using the navmesh follower toward a normalized image point [x,y], "
                "with x left-to-right and y top-to-bottom in [0,1]. "
                + (
                    "The point uses the latest front RGB image. "
                    if _front_only_observation()
                    else "Select the latest image using view=front/right/back/left or image_ref. "
                )
                + f"Executes immediately. Returns status and {observation}."
            ),
            "close_session": "End the episode and release the session. Returns closed=true on success.",
        }
        if tool.metadata.name in descriptions:
            return descriptions[tool.metadata.name]
    return tool.metadata.description


def _make_typed_wrapper(tool: Tool, mcp_name: str, *, inline_images: bool):
    """Build an MCP-ready wrapper function for `tool`.

    FastMCP uses `inspect.signature(fn)` + `fn.__annotations__` to
    derive the MCP protocol schema. A bare `**kwargs` wrapper
    collapses the schema to a single string param, so we set `__signature__` and `__annotations__`
    to the extracted parameter list. The resulting schema is
    indistinguishable from a hand-written `def hab_forward(...)`.
    """
    params = _extract_params(tool)
    tool_name = tool.metadata.name

    def _wrapper(**kwargs) -> Any:
        # Strip optional None arguments so omitted values retain the Tool parameter
        # schema semantics.
        cleaned_args = {k: v for k, v in kwargs.items() if v is not None}
        cleaned_args = _apply_benchmark_init_defaults(tool_name, cleaned_args)
        tool_seq = _next_mcp_tool_sequence(reset=tool_name == "init_scene")
        ctx = _build_mcp_context()
        if _spatial_memory_enabled():
            _spatial_memory.begin_tool(tool_name)
        if (
            tool_name == "close_session"
            and _global_task_mode()
            and _spatial_memory_enabled()
        ):
            blocked_close_result = _spatial_memory.evaluate_blocked_close(cleaned_args)
            if blocked_close_result is not None:
                return _finalize_mcp_response(
                    tool_name=tool_name,
                    tool_seq=tool_seq,
                    body=blocked_close_result,
                    inline_paths=[],
                    inline_images=inline_images,
                )
            achieved_close_result = _spatial_memory.evaluate_achieved_close(
                cleaned_args
            )
            if achieved_close_result is not None:
                return _finalize_mcp_response(
                    tool_name=tool_name,
                    tool_seq=tool_seq,
                    body=achieved_close_result,
                    inline_paths=[],
                    inline_images=inline_images,
                )
        close_session_id = ""
        close_pose: Optional[Dict[str, Any]] = None
        close_reason = ""
        close_trajectory_points: list[list[float]] = []
        close_step_count: Optional[int] = None
        if tool_name == "close_session":
            (
                close_session_id,
                close_pose,
                close_reason,
                close_trajectory_points,
                close_step_count,
            ) = _capture_mcp_final_pose()
        gate_enabled = _mcp_gate_enabled()
        if tool_name == "forward" and gate_enabled:
            gate_error = _mcp_forward_gate(cleaned_args)
            if gate_error is not None:
                return _finalize_mcp_response(
                    tool_name=tool_name,
                    tool_seq=tool_seq,
                    body=gate_error,
                    inline_paths=[],
                    inline_images=inline_images,
                )
        cleaned_args = _resolve_visual_view_args(tool_name, cleaned_args)
        view_error = cleaned_args.pop("_view_resolution_error", None)
        if view_error:
            return _finalize_mcp_response(
                tool_name=tool_name,
                tool_seq=tool_seq,
                body={"ok": False, "status": "invalid_view", "error": view_error},
                inline_paths=[],
                inline_images=inline_images,
            )
        if tool_name == "visual_ground_preview" and _spatial_memory_enabled():
            try:
                _spatial_memory.validate_grounding_arguments(cleaned_args)
            except SpatialMemoryError as exc:
                return _finalize_mcp_response(
                    tool_name=tool_name,
                    tool_seq=tool_seq,
                    body={
                        "ok": False,
                        "status": "invalid_branch_reference",
                        "error": str(exc),
                        "spatial_memory": _spatial_memory.result([]),
                    },
                    inline_paths=[],
                    inline_images=inline_images,
                )
        old_collision_stage = _mcp_round_state.collision_recovery_stage
        if _front_only_observation() and tool_name in _MOTION_TOOLS_WITH_SURROUND_VIEW:
            _cache_mcp_panorama_images([])
        result = ToolRegistry.dispatch(tool_name, cleaned_args, ctx)
        if not result.ok:
            if gate_enabled and _mcp_round_state.collision_recovery_stage is not None:
                error_body = _mcp_collision_recovery_payload(
                    result.error or "unknown error"
                )
            else:
                error_body = {"error": result.error or "unknown error"}
            return _finalize_mcp_response(
                tool_name=tool_name,
                tool_seq=tool_seq,
                body=error_body,
                inline_paths=[],
                inline_images=inline_images,
            )

        # InitSceneTool mutates ctx.is_gaussian; mirror to the
        # module-level state so subsequent MCP calls see the flag.
        if tool_name == "init_scene":
            global _mcp_is_gaussian
            _mcp_is_gaussian = bool(ctx.is_gaussian)
            _reset_mcp_round_state()
            _nav_goals.reset_from_env(ctx.bridge.session_id)
        elif tool_name == "close_session":
            _reset_mcp_round_state()

        # Apply per-tool MCP response shaping. The shapers exist to
        body = result.body
        body = dict(body)
        if tool_name == "close_session":
            outcome = str(cleaned_args.get("outcome") or "").strip().lower()
            if outcome in {"achieved", "blocked"}:
                body["terminal_claim"] = {
                    "outcome": outcome,
                    "reason": str(cleaned_args.get("reason") or ""),
                }
            metrics = body.get("metrics")
            agent_state = (
                metrics.get("agent_state") if isinstance(metrics, dict) else None
            )
            if (
                close_pose is not None
                and isinstance(agent_state, dict)
                and "position" not in agent_state
            ):
                _mark_private_trajectory_session(close_session_id)
            body = _finish_mcp_trajectory(
                body,
                session_id=close_session_id,
                pose=close_pose,
                reason=close_reason,
                trajectory_points=close_trajectory_points,
                tool_seq=tool_seq,
                step_count=close_step_count,
                diagnostic=close_reason,
            )
        if inline_images:
            body = _attach_mcp_surround_view(tool_name, body)
        if tool_name != "close_session":
            # Capture after all tool-owned visual work so the endpoint and
            # dense prefix represent the state visible at the formal result.
            body = _record_mcp_trajectory_event(tool_name, body, tool_seq=tool_seq)
        _mcp_record_evidence(tool_name, cleaned_args, body)
        _mcp_record_pose_or_camera_action(tool_name)
        if (
            gate_enabled
            and old_collision_stage is not None
            and _mcp_round_state.collision_recovery_stage is None
        ):
            _mcp_passability_state["post_collision_recovery_requires_check"] = True
            _mcp_invalidate_passability_reminder()
        if tool_name == "forward":
            _mcp_passability_state["post_collision_recovery_requires_check"] = False
        body = _attach_spatial_memory(tool_name, cleaned_args, body)
        shaper = _MCP_RESPONSE_SHAPERS.get(tool_name)
        inline_paths: List[str] = []
        if shaper is not None:
            shaped = shaper(body, cleaned_args)
            if isinstance(shaped, tuple):
                body, inline_paths = shaped
            else:
                body = shaped
        body = _strip_eval_only_fields(body)
        return _finalize_mcp_response(
            tool_name=tool_name,
            tool_seq=tool_seq,
            body=body,
            inline_paths=inline_paths,
            inline_images=inline_images,
        )

    # Critical: both __signature__ and __annotations__ must be set
    # so FastMCP generates a real schema instead of collapsing to
    # `**kwargs`.
    _wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=str)
    _wrapper.__annotations__ = {p.name: p.annotation for p in params}
    _wrapper.__annotations__["return"] = str
    _wrapper.__name__ = mcp_name
    _wrapper.__doc__ = _mcp_tool_description(tool)
    return _wrapper


# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "habitat-gs",
    instructions=(
        "Habitat-GS navigation simulator bridge. Use these tools to "
        "control a robot navigating indoor 3D scenes."
    ),
    stateless_http=True,
)


def _mcp_tool_allowed_by_whitelist(name: str) -> bool:
    if _front_only_observation() and name in {"hab_panorama", "hab_look_around"}:
        return False
    raw = os.environ.get("HAB_MCP_TOOL_WHITELIST", "").strip()
    if not raw:
        return True
    return name in {item.strip() for item in raw.split(",") if item.strip()}


if _mcp_tool_allowed_by_whitelist("hab_passability_check"):
    mcp.add_tool(
        fn=hab_passability_check,
        name="hab_passability_check",
        description=(
            "MCP-only conservative safety check before one "
            "bounded hab_forward call. Returns the Habitat navigation skills "
            "the caller should follow plus passable_hint/risky/blocked/uncertain."
        ),
    )


def hab_register_spatial_junction(*, branches: List[Dict[str, Any]]) -> str:
    """Register 2-4 agent-observed sibling openings from the latest panorama."""

    tool_seq = _next_mcp_tool_sequence()
    if not _spatial_memory_enabled():
        body: Dict[str, Any] = {
            "ok": False,
            "status": "spatial_memory_disabled",
            "error": "Spatial memory is not enabled for this benchmark session.",
        }
    elif not _bridge.session_id:
        body = {
            "ok": False,
            "status": "no_active_session",
            "error": "Call hab_init_scene before registering a junction.",
        }
    else:
        try:
            memory = _spatial_memory.register_junction(
                branches,
                _mcp_latest_panorama_images,
            )
            body = {
                "ok": True,
                "status": "junction_registered",
                "spatial_memory": memory,
            }
        except SpatialMemoryError as exc:
            body = {
                "ok": False,
                "status": "invalid_junction_registration",
                "error": str(exc),
                "spatial_memory": _spatial_memory.result([]),
            }
    return _finalize_mcp_response(
        tool_name="register_spatial_junction",
        tool_seq=tool_seq,
        body=body,
        inline_paths=[],
        inline_images=False,
    )


_spatial_memory_tool_registered = False
if _spatial_memory_enabled() and _mcp_tool_allowed_by_whitelist(
    "hab_register_spatial_junction"
):
    mcp.add_tool(
        fn=hab_register_spatial_junction,
        name="hab_register_spatial_junction",
        description=(
            "Register two to four sibling doorway/opening candidates observed "
            "in the latest panorama. Each branch requires view, phrase, and a "
            "normalized [x, y] point. Returns stable branch_id values."
        ),
        structured_output=False,
    )
    _spatial_memory_tool_registered = True


def _register_tools_from_registry() -> int:
    """Register canonical hab_ tool names and declared aliases.

    Store generated wrappers as module attributes for direct callers and
    return the count of MCP-visible names. Re-registration overwrites.
    """
    count = 0
    # Optional whitelist (comma-separated hab_* names) limits the registry
    # tools exposed by an experiment. Otherwise expose all MCP-visible tools.
    for tool in ToolRegistry.list_all():
        # Skip tools that opt out of MCP exposure.
        if not tool.metadata.mcp_visible:
            continue
        canonical_name = f"hab_{tool.metadata.name}"
        names_to_register = [canonical_name] + [
            f"hab_{legacy}" for legacy in sorted(tool.metadata.legacy_names)
        ]
        for mcp_name in names_to_register:
            if not _mcp_tool_allowed_by_whitelist(mcp_name):
                continue
            wrapper = _make_typed_wrapper(tool, mcp_name, inline_images=True)
            direct_wrapper = _make_typed_wrapper(tool, mcp_name, inline_images=False)
            mcp.add_tool(
                fn=wrapper,
                name=mcp_name,
                description=_mcp_tool_description(tool),
                structured_output=False,
            )
            # Stash on module globals for direct-call test access.
            globals()[mcp_name] = direct_wrapper
            count += 1
    return count


# Register tools eagerly at module import so `from mcp_server import
# hab_forward` works without having to call `main()` first.
_registered_count = _register_tools_from_registry() + int(
    _spatial_memory_tool_registered
)


# ---------------------------------------------------------------------------
# MCP Resources (stay manual — not tools)
# ---------------------------------------------------------------------------


def _is_private_sidecar_artifact(name: str) -> bool:
    return (
        name.endswith(".trajectory.json")
        or name.endswith(".trajectory.json.tmp")
        or name.endswith(".benchmark_audit.jsonl")
    )


def list_artifacts() -> str:
    """List all artifact files in the artifacts directory."""
    d = _artifacts_dir()
    if not os.path.isdir(d):
        return json.dumps({"files": [], "dir": d})
    files = sorted(
        name for name in os.listdir(d) if not _is_private_sidecar_artifact(name)
    )
    return json.dumps({"dir": d, "count": len(files), "files": files})


def read_artifact(filename: str) -> str:
    """Read an artifact file. Images return base64, JSON/text return content."""
    d = _artifacts_dir()
    safe_name = os.path.basename(filename)
    if _is_private_sidecar_artifact(safe_name):
        return json.dumps(
            {
                "error": (
                    "Private sidecars are server-side audit artifacts "
                    "and are not exposed through MCP resources."
                )
            }
        )
    path = os.path.join(d, safe_name)  # prevent traversal
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {filename}"})
    ext = os.path.splitext(path)[1].lower()
    _image_mime_by_ext = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    if ext in _image_mime_by_ext:
        b64 = read_image_b64(path)
        return json.dumps(
            {
                "type": "image",
                "filename": filename,
                "data": b64,
                "mimeType": _image_mime_by_ext[ext],
            }
        )
    elif ext == ".mp4":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return json.dumps(
            {
                "type": "video",
                "filename": filename,
                "data": b64,
                "mimeType": "video/mp4",
            }
        )
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content


# ---------------------------------------------------------------------------
if os.environ.get("HAB_MCP_MINIMAL_TASK") != "1":
    mcp.resource("habitat://artifacts/list")(list_artifacts)
    mcp.resource("habitat://artifacts/{filename}")(read_artifact)


# Vision: return the agent's current first-person view AS A VIEWABLE IMAGE so
# the LLM brain can actually perceive the room (the dynamic hab_* tools only
# return file paths, which a headless MCP client cannot see). jingyi MVP.
# ---------------------------------------------------------------------------
def hab_set_pose(session_id: str, x: float, z: float, yaw: float = 0.0) -> Any:
    """Place your camera at (x, z) facing heading `yaw` (radians), using the
    configured manual capture height. Call this ONCE right after init."""
    import math as _m

    _bridge.session_id = session_id
    rot = [
        0.0,
        _m.sin(float(yaw) / 2.0),
        0.0,
        _m.cos(float(yaw) / 2.0),
    ]  # xyzw, about +Y
    payload = {
        "position": [float(x), _mcp_capture_height_y(), float(z)],
        "rotation": rot,
        # Treat the configured capture height as a camera/eye target, not a
        # body-root target: snapping the agent state back onto the navmesh keeps
        # raw get_visuals/hab_see captures at one sensor-height above the floor.
        "snap_to_navmesh": True,
    }
    result = _bridge.call("set_agent_state", payload)
    pos = (
        (result.get("agent_state") or {}).get("position")
        if isinstance(result, dict)
        else None
    )
    body: Dict[str, Any] = {
        "ok": True,
        "x": float(x),
        "z": float(z),
        "yaw": float(yaw),
        "position": pos,
    }
    panorama = _capture_mcp_surround_view()
    if isinstance(panorama, dict) and "panorama_error" in panorama:
        body["panorama_error"] = panorama["panorama_error"]
    elif isinstance(panorama, dict):
        rows = _normalize_panorama_images(panorama)
        if rows:
            body["panorama_images"] = rows
            body["panorama_image_paths"] = [row["path"] for row in rows]
            body["images"] = [row["path"] for row in rows]

    text = json.dumps(_strip_eval_only_fields(body), default=str)
    inline_paths = _llm_rgb_image_paths(body.get("images", []), max_images=4)
    image_blocks = _inline_image_blocks(inline_paths)
    if image_blocks:
        return [text, *image_blocks]
    return text


if _mcp_tool_allowed_by_whitelist("hab_set_pose"):
    mcp.add_tool(hab_set_pose, name="hab_set_pose", structured_output=False)


def hab_see(session_id: str) -> Image:
    """Capture and RETURN the agent's CURRENT first-person camera view as an
    image you can actually look at. Use this to perceive the room (what objects
    and openings are around you) BEFORE deciding where to navigate. Call it
    again after moving to see the updated view."""
    import urllib.request as _U

    def _post(action, payload):
        env = {
            "request_id": "see",
            "action": action,
            "session_id": session_id,
            "payload": payload,
        }
        req = _U.Request(
            _bridge.base_url + "/v1/request",
            data=json.dumps(env).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _U.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return resp.get("result", resp) if isinstance(resp, dict) else {}

    body = _post("get_visuals", {"output_dir": _artifacts_dir()})
    cs = (body.get("visuals") or {}).get("color_sensor") or {}
    path = cs.get("path")
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"hab_see: no color_sensor image (keys={list(body.keys())})")
    return Image(path=path, format="png")


if os.environ.get(
    "HAB_ENABLE_LEGACY_SEE", "0"
) == "1" and _mcp_tool_allowed_by_whitelist("hab_see"):
    mcp.add_tool(
        hab_see,
        name="hab_see",
        description=(
            "Legacy single-view camera capture. Hidden by default because "
            "movement tools now return direction-labelled surround-camera "
            "observations."
        ),
    )


# ---------------------------------------------------------------------------
# "Look around to find things": one tool call that rotates 360° in place and
# returns N evenly-spaced first-person snapshots (labeled by heading) so the
# brain can survey the room in a SINGLE step instead of many hab_turn+observe
# round-trips. The tool only does the mechanical scan; deciding WHERE the target
# is stays with the brain (it judges from the returned views). Implemented purely
# in this MCP layer by composing existing bridge actions (get_visuals +
# step_and_capture) — no new bridge action / no bridge restart. jingyi MVP.
# ---------------------------------------------------------------------------
def hab_look_around(session_id: str, num_views: int = 6, target: str = ""):
    """Survey the whole room in ONE call: rotate a full 360 degrees in place and
    return `num_views` evenly-spaced first-person snapshots, each labeled by how
    far you would hab_turn RIGHT from your current heading to face it (0 deg =
    straight ahead). Use this instead of many separate hab_turn + observe calls
    when you want to find which direction holds what you are looking for.
    Optional `target`: a short object phrase you are seeking (e.g. "bed" or
    "armchair"); if a detector service is available, each snapshot is also
    annotated with where that object was detected and the EXACT degrees to turn
    right to face it. Trust your own eyes over the detector when they disagree.
    Afterwards you are back at your ORIGINAL heading; to head toward something you
    spotted, hab_turn right by that snapshot's degrees, then use an available
    navigation tool toward a target visible in that view."""
    import urllib.request as _U

    def _post(action, payload):
        env = {
            "request_id": "look",
            "action": action,
            "session_id": session_id,
            "payload": payload,
        }
        req = _U.Request(
            _bridge.base_url + "/v1/request",
            data=json.dumps(env).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _U.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return resp.get("result", resp) if isinstance(resp, dict) else {}

    n = max(2, min(int(num_views), 12))
    # The bridge turns in 10-degree atomic steps (ceil(deg/10)), so snap the
    # per-view spacing to the 10-degree grid to keep headings exact + the loop
    # closing on a full circle.
    step_deg = max(10, int(round(360.0 / n / 10.0)) * 10)

    # Manual MCP viewing uses a camera-level pose convention. Preserve x/z and
    # heading across the scan, but temporarily lift captures to the configured
    # visual height so the views match hab_set_pose / hab_see.
    x0 = z0 = None
    try:
        st = _post("get_metrics", {}).get("agent_state", {})
        p = st.get("position")
        if isinstance(p, (list, tuple)) and len(p) == 3:
            x0, z0 = float(p[0]), float(p[2])
    except Exception:
        pass

    def _raise_camera():
        if x0 is not None:
            _post("set_agent_state", {"position": [x0, _mcp_capture_height_y(), z0]})

    def _frame_path(body):
        cs = (body.get("visuals") or {}).get("color_sensor") or {}
        return cs.get("path")

    views = []  # (heading_deg_right_of_start, path)
    for i in range(n):
        if i > 0:
            _post("step_action", {"action": "turn_right", "degrees": step_deg})
        _raise_camera()
        body = _post(
            "get_visuals", {"include_metrics": True, "sensors": ["color_sensor"]}
        )
        views.append(((i * step_deg) % 360, _frame_path(body)))

    # Return to the original heading (we have rotated (n-1)*step_deg so far).
    # step_action (not step_and_capture): no frames needed while turning back.
    back = ((n - 1) * step_deg) % 360
    if back:
        _post("step_action", {"action": "turn_left", "degrees": back})
    _raise_camera()  # restore camera-level scan pose after return turn

    # Optional open-vocab grounding of `target` in each view, via the resident
    # detector service (NAV_GROUNDING_URL). Failures degrade to plain snapshots.
    ground = {}
    g_url = os.environ.get("NAV_GROUNDING_URL", "").strip()
    if g_url and target.strip():
        phrases = target.strip().lower().rstrip(".")
        phrases = (
            phrases if phrases.startswith(("a ", "an ", "the ")) else "a " + phrases
        ) + "."
        for heading, path in views:
            if not (path and os.path.isfile(path)):
                continue
            try:
                req = _U.Request(
                    g_url.rstrip("/") + "/ground",
                    data=json.dumps({"image_path": path, "phrases": phrases}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with _U.urlopen(req, timeout=30) as r:
                    body = json.loads(r.read())
                if body.get("ok") and body.get("detections"):
                    ground[heading] = body["detections"][0]  # best per view
            except Exception:
                pass

    out = [
        f"I rotated a full 360 degrees and took {n} snapshots, every {step_deg} "
        f"degrees apart. Each is labeled by how far to hab_turn RIGHT from my "
        f"current (restored) heading to face it (0 deg = straight ahead):"
    ]
    for heading, path in views:
        lab = f"--- snapshot at {heading} deg right of forward ---"
        d = ground.get(heading)
        if d and d["score"] >= 0.35:
            exact = int(round((heading + d["angle_off_deg"]) / 10.0)) * 10 % 360
            lab += (
                f" [detector: '{target}' here, confidence {d['score']:.2f}, "
                f"size {d['area_ratio']:.0%} of view -> hab_turn right "
                f"{exact} deg to face it]"
            )
        out.append(lab)
        if path and os.path.isfile(path):
            out.append(Image(path=path, format="png"))
        else:
            out.append("(no image captured at this heading)")
    if ground:
        best_h, best_d = max(ground.items(), key=lambda kv: kv[1]["score"])
        if best_d["score"] >= 0.35:
            exact = int(round((best_h + best_d["angle_off_deg"]) / 10.0)) * 10 % 360
            out.append(
                f"DETECTOR BEST MATCH for '{target}': the snapshot at "
                f"{best_h} deg (confidence {best_d['score']:.2f}). To face "
                f"it: hab_turn right {exact} deg. Verify with your own eyes "
                f"before navigating."
            )
    out.append(
        "I am now back at my original heading. To move toward something you "
        "see above, hab_turn right by that snapshot's degrees, then "
        "use an available navigation tool to approach a target visible in that view."
    )
    return out


# hab_look_around is the benchmark's arm-2 ("Tool") capability. Register it only
# when explicitly enabled, so the baseline / skill arms physically lack it and
# each arm gets a clean, deterministic tool set (independent of --allowedTools).
if os.environ.get(
    "HAB_ENABLE_LOOK_AROUND", "0"
) == "1" and _mcp_tool_allowed_by_whitelist("hab_look_around"):
    mcp.add_tool(hab_look_around, name="hab_look_around", structured_output=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_mcp_bind_host(host: str | None) -> str:
    return restrict_bind_host_to_loopback(host)


def main() -> None:
    import argparse

    load_dotenv_from_project()

    # Re-configure bridge after dotenv load
    host = os.environ.get("NAV_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("NAV_BRIDGE_PORT", "18911"))
    _bridge.configure(host=host, port=port)

    parser = argparse.ArgumentParser(description="habitat-gs MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18912,
        help="HTTP port for sse/streamable-http (default: 18912)",
    )
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--enable-mcp-gate",
        dest="mcp_gate_enabled",
        action="store_true",
        default=None,
        help=(
            "Enable MCP-only forward passability and collision recovery gates "
            f"(overrides {_MCP_GATE_ENV})."
        ),
    )
    gate_group.add_argument(
        "--disable-mcp-gate",
        dest="mcp_gate_enabled",
        action="store_false",
        help=(
            "Disable MCP-only forward passability and collision recovery gates "
            f"(overrides {_MCP_GATE_ENV})."
        ),
    )
    args = parser.parse_args()
    if args.mcp_gate_enabled is not None:
        os.environ[_MCP_GATE_ENV] = "1" if args.mcp_gate_enabled else "0"
    parser.add_argument(
        "--host",
        default=os.environ.get("NAV_MCP_HOST", "127.0.0.1"),
        help="HTTP host for sse/streamable-http (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    args.host = resolve_mcp_bind_host(args.host)

    print(
        f"habitat-gs MCP server starting (transport={args.transport}, "
        f"bridge={_bridge.base_url}, tools={_registered_count}, "
        f"mcp_gate={_mcp_gate_enabled()})",
        file=sys.stderr,
    )

    if args.transport in ("sse", "streamable-http"):
        # FastMCP reads host/port from self.settings, set them directly
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
