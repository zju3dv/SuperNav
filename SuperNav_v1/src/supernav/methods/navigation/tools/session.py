"""Scene initialization and session lifecycle tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from supernav.methods.navigation.tools._common import parse_bool_flag
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)
from supernav.backends.habitat.trajectory import McpTrajectoryLog, extract_pose
from supernav.methods.navigation.nav_goals_ledger import get_nav_goals_ledger

_SESSION_TASK_TYPES = {"chat"}
_COMPLETION_STATUSES = {"achieved", "blocked", "timeout", "error"}


# ---------------------------------------------------------------------------
# InitSceneTool
# ---------------------------------------------------------------------------


def _default_scene_dataset_config() -> str:
    """Return the operator's explicit dataset binding, if configured."""
    return os.environ.get("HAB_DEFAULT_DATASET_CONFIG", "")


class InitSceneTool:
    """Initialize a fresh simulator session. Mutates ctx.session_id and
    ctx.is_gaussian from the bridge's response."""

    metadata = ToolMetadata(
        name="init_scene",
        category=ToolCategory.SESSION,
        description=(
            "Initialize a simulation session for a given scene. Must be "
            "called first, before any movement or observation tools."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "description": "Scene ID (optional; uses bridge default when omitted)",
                },
                "scene_dataset_config_file": {
                    "type": "string",
                    "description": "Path to scene_dataset_config.json (optional)",
                },
                "depth": {
                    "type": "boolean",
                    "description": "Enable depth sensor (default true)",
                    "default": True,
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Enable semantic sensor (default false)",
                    "default": False,
                },
                "third_person": {
                    "type": "boolean",
                    "description": (
                        "Enable third-person over-the-shoulder RGB camera "
                        "(default false). When true, the bridge injects an "
                        "extra CameraSensorSpec at ~1.5m behind and 1.2m "
                        "above the agent and exposes it as `third_rgb_sensor` "
                        "in visuals/observations."
                    ),
                    "default": False,
                },
                "start_position": {
                    "type": "array",
                    "description": (
                        "Optional agent start position [x, y, z] in world "
                        "metres. When provided, the bridge applies this pose "
                        "during init_scene before recording the initial "
                        "trajectory point."
                    ),
                },
                "start_rotation": {
                    "type": "array",
                    "description": (
                        "Optional agent start rotation as an xyzw quaternion. "
                        "Used with start_position when initializing benchmark "
                        "episodes from task metadata."
                    ),
                },
                "allow_sliding": {
                    "type": "boolean",
                    "description": (
                        "Optional per-session motion physics override. When "
                        "false, a blocked step produces zero displacement "
                        "(habitat try_step_no_sliding). Omit to keep the "
                        "habitat-sim default (true)."
                    ),
                },
                "default_agent_navmesh": {
                    "type": "boolean",
                    "description": (
                        "Optional per-session navmesh source override. When "
                        "false, the bridge keeps the navmesh shipped by the "
                        "scene dataset (e.g. HM3D .basis.navmesh) instead of "
                        "recomputing one with default agent settings. Omit to "
                        "keep the harness default (true; env "
                        "HAB_DEFAULT_AGENT_NAVMESH overrides)."
                    ),
                },
                "sensor_height": {
                    "type": "number",
                    "description": (
                        "Optional camera height in metres for benchmark "
                        "episodes whose captured policy used a non-default "
                        "sensor height."
                    ),
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.MUTATING,
        requires_session=False,  # init_scene creates the session
        legacy_names={"init"},
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}

        scene = args.get("scene") or os.environ.get("HAB_DEFAULT_SCENE")
        if not scene:
            return ToolResult(ok=False, body={}, error="Provide scene or configure HAB_DEFAULT_SCENE")
        payload["scene"] = scene

        ds_config = (
            args.get("scene_dataset_config_file") or _default_scene_dataset_config()
        )
        if ds_config:
            payload["scene_dataset_config_file"] = ds_config

        _depth_default = os.environ.get("HAB_SENSOR_DEPTH", "1") == "1"
        depth_enabled = parse_bool_flag(
            args.get("depth", _depth_default), default=_depth_default
        )
        semantic_enabled = parse_bool_flag(args.get("semantic", False), default=False)
        third_person_enabled = parse_bool_flag(
            args.get("third_person", False), default=False
        )
        sensor_height = args.get(
            "sensor_height",
            os.environ.get("HAB_SENSOR_CAM_HEIGHT", "1.5"),
        )
        payload["sensor"] = {
            "width": int(os.environ.get("HAB_SENSOR_WIDTH", "512")),
            "height": int(os.environ.get("HAB_SENSOR_HEIGHT", "512")),
            "sensor_height": float(sensor_height),
            "hfov": int(os.environ.get("HAB_SENSOR_HFOV", "90")),
            "color_sensor": True,
            "depth_sensor": depth_enabled,
            "semantic_sensor": semantic_enabled,
            "third_person_color_sensor": third_person_enabled,
        }
        if args.get("start_position") is not None:
            payload["start_position"] = args["start_position"]
        if args.get("start_rotation") is not None:
            payload["start_rotation"] = args["start_rotation"]
        if args.get("allow_sliding") is not None:
            payload["allow_sliding"] = parse_bool_flag(args["allow_sliding"])
        # Datasets that ship an official navmesh (HM3D .basis.navmesh) must not
        # have it recomputed with default agent settings, or episode geodesic
        # distances diverge from the official ones. Only emitted when disabled
        # so existing sessions keep the harness default untouched.
        _navmesh_default = os.environ.get("HAB_DEFAULT_AGENT_NAVMESH", "1") == "1"
        default_agent_navmesh = parse_bool_flag(
            args.get("default_agent_navmesh", _navmesh_default),
            default=_navmesh_default,
        )
        if not default_agent_navmesh:
            payload["default_agent_navmesh"] = False

        # Clear any prior session_id so the bridge knows this is a new
        # session, not a re-init of the current one. Clear BOTH sides
        # (ctx.bridge.session_id AND ctx.session_id) so the failure
        # path leaves a coherent "no active session" state — a previous
        # version only cleared the bridge side, which left ctx.session_id
        # stale on a failed re-init.
        ctx.bridge.session_id = None
        ctx.session_id = ""

        try:
            result = ctx.bridge.call("init_scene", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        # Propagate bridge's new session_id and scene flavour into ctx
        # so subsequent tools (which key off ctx.session_id) see it.
        new_session_id = result.get("session_id") if isinstance(result, dict) else None
        if new_session_id:
            ctx.bridge.session_id = new_session_id
            ctx.session_id = new_session_id
        ctx.is_gaussian = (
            bool(result.get("is_gaussian", False))
            if isinstance(result, dict)
            else False
        )

        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


# ---------------------------------------------------------------------------
# CloseSessionTool
# ---------------------------------------------------------------------------


class CloseSessionTool:
    """Close the active simulator session and free its resources."""

    metadata = ToolMetadata(
        name="close_session",
        category=ToolCategory.SESSION,
        description="Close the current simulation session and free resources.",
        parameters_schema={
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["achieved", "blocked"],
                    "description": (
                        "Optional structured agent terminal claim for benchmark "
                        "outcome accounting. It never prevents the session from closing."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Optional short explanation for the terminal claim.",
                },
                "close_audit_token": {
                    "type": "string",
                    "description": (
                        "Token returned by a blocked_close_audit_required or "
                        "achieved_close_audit_required result. Use only for the "
                        "immediate audited close retry."
                    ),
                },
                "arrival_confirmation": {
                    "type": "object",
                    "description": (
                        "Required for the audited achieved-close retry. Fields: "
                        "final_approach_done (bool), target_base_cut (bool), "
                        "estimated_distance_m (number, must be <= 1.0), evidence "
                        "(short description of the visible proof)."
                    ),
                },
                "frontier_audit": {
                    "type": "array",
                    "description": (
                        "Exact audit of every active frontier branch for the second "
                        "blocked-close call. Each entry has branch_id, disposition, "
                        "and visual/action evidence."
                    ),
                    "items": {"type": "object"},
                },
                "post_entry_exception": {
                    "type": "object",
                    "description": (
                        "Required only when the close gate reports a pending "
                        "post-entry follow-up; disposition must be "
                        "no_safe_passable_interior with evidence."
                    ),
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.DESTRUCTIVE,
        legacy_names={"close"},
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("close_session", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        ctx.bridge.session_id = None
        ctx.session_id = ""
        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


# ---------------------------------------------------------------------------
# MarkCompletionTool
# ---------------------------------------------------------------------------


class MarkCompletionTool:
    """Record the agent's explicit task-completion decision."""

    metadata = ToolMetadata(
        name="mark_completion",
        category=ToolCategory.STATUS,
        description=(
            "Mark the current episode attempt as complete without closing the "
            "simulator session. Use status='achieved' only when you believe "
            "the instruction goal has been reached; otherwise use blocked, "
            "timeout, or error."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": sorted(_COMPLETION_STATUSES),
                    "description": "Agent-declared completion state.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation for the declaration.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Optional confidence from 0.0 to 1.0.",
                },
            },
            "required": ["status"],
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.bridge.session_id:
            return ToolResult(
                ok=False,
                body={},
                error="No active session. Call init_scene first.",
            )

        status = str(args.get("status", "")).strip().lower()
        if status not in _COMPLETION_STATUSES:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    "status must be one of: " + ", ".join(sorted(_COMPLETION_STATUSES))
                ),
            )

        reason = str(args.get("reason", "") or "")
        confidence = args.get("confidence")
        confidence_float = None
        if confidence is not None:
            try:
                confidence_float = float(confidence)
            except (TypeError, ValueError):
                return ToolResult(
                    ok=False,
                    body={},
                    error="confidence must be a number between 0.0 and 1.0.",
                )
            if confidence_float < 0.0 or confidence_float > 1.0:
                return ToolResult(
                    ok=False,
                    body={},
                    error="confidence must be a number between 0.0 and 1.0.",
                )

        metrics: Dict[str, Any] = {}
        try:
            response = ctx.bridge.call(
                "get_audit_metrics",
                {"include_trajectory": False},
                timeout=3,
                audit_internal=True,
            )
            if isinstance(response, dict):
                metrics = response
        except Exception as exc:
            metrics = {"audit_metrics_error": str(exc)}

        pose = extract_pose(metrics)
        artifacts_dir = ctx.output_dir or _artifacts_dir()
        McpTrajectoryLog(artifacts_dir).mark_completion(
            session_id=str(ctx.bridge.session_id),
            status=status,
            reason=reason,
            confidence=confidence_float,
            pose=pose,
            body={"metrics": metrics},
        )

        body: Dict[str, Any] = {
            "session_id": str(ctx.bridge.session_id),
            "completion_status": status,
            "completion_reason": reason,
            "completion_confidence": confidence_float,
        }
        return ToolResult(ok=True, body=body)


# ---------------------------------------------------------------------------
# NavGoalsTool — multi-goal episode ledger (status / mark)
# ---------------------------------------------------------------------------


class NavGoalsTool:
    """Query or mark per-goal arrival in a multi-goal navigation episode.

    The goal list is injected by the harness via environment (indexes and
    descriptions only); the tool never exposes ground-truth coordinates.
    A successful ``mark`` records the agent pose in the trajectory
    sidecar for offline per-goal scoring. Marks are one-shot and cannot
    be revoked.
    """

    metadata = ToolMetadata(
        name="nav_goals",
        category=ToolCategory.STATUS,
        description=(
            "Multi-goal episode ledger. action='status' lists every goal's "
            "index, description, and found/pending state. action='mark' with "
            "target_index declares that goal reached; call it only after "
            "arriving at the goal with final visual evidence, and check "
            "status first. Marks cannot be revoked."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "mark"],
                    "description": "Query the ledger or mark one goal as found.",
                },
                "target_index": {
                    "type": "integer",
                    "description": "1-based goal index; required for action='mark'.",
                },
            },
            "required": ["action"],
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.bridge.session_id:
            return ToolResult(
                ok=False,
                body={},
                error="No active session. Call init_scene first.",
            )

        action = str(args.get("action", "")).strip().lower()
        if action not in ("status", "mark"):
            return ToolResult(
                ok=False,
                body={},
                error="action must be one of: status, mark",
            )

        ledger = get_nav_goals_ledger()
        if not ledger.configured:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    "No navigation goals are configured for this session; "
                    "this is not a multi-goal episode."
                ),
            )

        if action == "status":
            body: Dict[str, Any] = {"session_id": str(ctx.bridge.session_id)}
            body.update(ledger.status())
            return ToolResult(ok=True, body=body)

        target_index_raw = args.get("target_index")
        if not isinstance(target_index_raw, (int, float)) or isinstance(
            target_index_raw, bool
        ):
            return ToolResult(
                ok=False,
                body={},
                error="target_index is required for action='mark'.",
            )
        if isinstance(target_index_raw, float) and not target_index_raw.is_integer():
            return ToolResult(
                ok=False,
                body={},
                error="target_index must be a whole-number goal index.",
            )
        target_index = int(target_index_raw)

        try:
            updated = ledger.mark(target_index)
        except ValueError as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        metrics: Dict[str, Any] = {}
        try:
            response = ctx.bridge.call(
                "get_audit_metrics",
                {"include_trajectory": False},
                timeout=3,
                audit_internal=True,
            )
            if isinstance(response, dict):
                metrics = response
        except Exception as exc:
            metrics = {"audit_metrics_error": str(exc)}

        pose = extract_pose(metrics)
        artifacts_dir = ctx.output_dir or _artifacts_dir()
        McpTrajectoryLog(artifacts_dir).mark_goal(
            session_id=str(ctx.bridge.session_id),
            target_index=target_index,
            pose=pose,
            body={"metrics": metrics},
        )

        body = {
            "session_id": str(ctx.bridge.session_id),
            "marked_index": target_index,
        }
        position = pose.get("position") if isinstance(pose, dict) else None
        if position is not None and parse_bool_flag(
            os.environ.get("HAB_MCP_GLOBAL_TASK"), default=False
        ):
            # Only attach coordinates under global-task mode, where the
            # redaction key list hides them from the agent while the audit
            # sidecar keeps them. The trajectory sidecar records the pose
            # regardless, so offline scoring never depends on this field.
            body["position"] = position
        body.update(updated)
        return ToolResult(ok=True, body=body)


# ---------------------------------------------------------------------------
class SceneGraphQueryTool:
    """Query the precomputed scene graph for rooms or objects by type/label.

    The scene graph (room_object_scene_graph.json) is generated offline using
    ``tools/scene_graph/generate_room_object_scene_graph.py`` and auto-loaded when a
    scene session is initialized. Each node includes a position and bounding box,
    allowing the agent to navigate to specific rooms or object instances.
    """

    metadata = ToolMetadata(
        name="scene_graph",
        category=ToolCategory.SESSION,
        description=(
            "Query the precomputed scene graph to get 3D positions and bounding boxes "
            "of objects in the scene. Use this before navigating to a target. "
            "IMPORTANT: Room nodes have NO semantic labels (only numeric IDs). "
            "To locate a functional area such as a kitchen or bedroom, query objects "
            "typically found there — e.g. query_type='object', object_label='sink' to "
            "find the kitchen area, then check the room_id field to identify the room. "
            "The room_type parameter does NOT filter rooms by name and should be omitted."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["all", "room", "object"],
                    "description": (
                        "'object' — find objects by label (RECOMMENDED for navigation). "
                        "'room' — list all rooms with centroids and areas (no label filtering). "
                        "'all' — objects and rooms mixed; apply object_label to filter objects."
                    ),
                },
                "room_type": {
                    "type": "string",
                    "description": (
                        "NOT SUPPORTED — room nodes carry no semantic labels in this scene graph. "
                        "Omit this parameter. To find a kitchen, use query_type='object' with "
                        "object_label='sink' or 'refrigerator' and follow the room_id field."
                    ),
                },
                "object_label": {
                    "type": "string",
                    "description": (
                        "Substring match on object label (case-insensitive). "
                        "Examples: 'chair', 'sofa', 'sink', 'table', 'door'. "
                        "Used with query_type='object' or query_type='all'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query_type"],
        },
        permission=PermissionLevel.READ_ONLY,
        mcp_visible=True,
        allowed_nav_modes={
            "navmesh"
        },  # SG gives absolute 3D positions — equivalent to a map
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("get_scene_graph", args)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        if not isinstance(result, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        return ToolResult(ok=True, body=result)


class CollisionProfileTool:
    """Return the agent's collision body dimensions for the active session."""

    metadata = ToolMetadata(
        name="collision_profile",
        category=ToolCategory.SESSION,
        description=(
            "Return the current agent collision body profile: body type, "
            "radius, diameter, height, sensor height, and movement collision "
            "model. This exposes only the agent's own dimensions, not scene "
            "geometry or map information."
        ),
        parameters_schema={
            "type": "object",
            "properties": {},
        },
        permission=PermissionLevel.READ_ONLY,
        mcp_visible=True,
        allowed_nav_modes={"navmesh", "mapless"},
        when_to_use=(
            "Check the agent's body radius/height before deciding whether "
            "a narrow visible gap is safe to approach."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        del args
        try:
            result = ctx.bridge.call("get_collision_profile", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        if not isinstance(result, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(InitSceneTool())
ToolRegistry.register(CloseSessionTool())
ToolRegistry.register(MarkCompletionTool())
ToolRegistry.register(NavGoalsTool())
ToolRegistry.register(SceneGraphQueryTool())
ToolRegistry.register(CollisionProfileTool())


__all__ = [
    "InitSceneTool",
    "CloseSessionTool",
    "MarkCompletionTool",
    "NavGoalsTool",
    "SceneGraphQueryTool",
    "CollisionProfileTool",
]
