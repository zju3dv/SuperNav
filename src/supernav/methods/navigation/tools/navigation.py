"""Navigation tools for primitive actions, pathfinding and point navigation.

Store per-round mutations on ToolContext.round_state. Navmesh-only
metadata gates exclude path-based tools from mapless schemas.
"""

from __future__ import annotations

from typing import Any, Dict

from supernav.methods.navigation.tools._common import collect_images, visual_payload
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)

_INCLUDE_IMAGES_PARAM = {
    "type": "boolean",
    "description": (
        "When true, MCP responses inline the four direction-labelled "
        "surround-camera RGB images captured after the action. Enabled by "
        "default; set false when only image paths are needed."
    ),
    "default": True,
}

# ---------------------------------------------------------------------------
# ForwardTool / BackwardTool
# ---------------------------------------------------------------------------


class ForwardTool:
    """Move the agent forward by distance_m metres.

    The bridge auto-decomposes `distance_m` into 0.25m atomic steps
    (the underlying unit of motion) and stops early on collision."""

    metadata = ToolMetadata(
        name="forward",
        category=ToolCategory.NAVIGATION,
        description=(
            "Move agent forward by `distance_m` metres. The bridge "
            "auto-decomposes the request into 0.25m atomic steps and "
            "stops early on collision. Returns metrics with position, "
            "heading, collision status, and euclidean_distance_to_goal."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": (
                        "Distance in metres. Must be a positive multiple "
                        "of the 0.25m atomic step. Stops early on collision."
                    ),
                    "default": 0.5,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
        },
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        dist = args.get("distance_m", 0.5)
        payload: Dict[str, Any] = {
            "action": "move_forward",
            "distance": dist,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        if ctx.round_state.last_collided:
            ctx.round_state.require_collision_backward()
        ctx.round_state.last_movement_action = "move_forward"
        col_tag = "!" if ctx.round_state.last_collided else ""
        ctx.round_state.round_actions.append(f"forward({dist}m){col_tag}")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


class BackwardTool:
    """Move the agent backward by distance_m metres.

    Intended primarily as a short collision recovery primitive. The
    bridge auto-decomposes `distance_m` into 0.25m atomic steps and
    stops early on collision.
    """

    metadata = ToolMetadata(
        name="backward",
        category=ToolCategory.NAVIGATION,
        description=(
            "Move agent backward by `distance_m` metres. The bridge "
            "auto-decomposes the request into 0.25m atomic steps and "
            "stops early on collision. Useful as the first short "
            "recovery action after a collided forward step."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": (
                        "Distance in metres. Must be positive; the "
                        "bridge decomposes it into 0.25m atomic steps "
                        "and stops early on collision."
                    ),
                    "default": 0.5,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
        },
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        dist = args.get("distance_m", 0.5)
        payload: Dict[str, Any] = {
            "action": "move_backward",
            "distance": dist,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.mark_collision_backward_done()
        ctx.round_state.last_movement_action = "move_backward"
        col_tag = "!" if ctx.round_state.last_collided else ""
        ctx.round_state.round_actions.append(f"backward({dist}m){col_tag}")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# TurnTool
# ---------------------------------------------------------------------------


class TurnTool:
    """Rotate the agent left or right by `degrees` degrees.

    The bridge auto-decomposes `degrees` into 10° atomic steps
    (the underlying unit of rotation)."""

    metadata = ToolMetadata(
        name="turn",
        category=ToolCategory.NAVIGATION,
        description=(
            "Rotate agent left or right by `degrees` degrees. The "
            "bridge auto-decomposes the request into 10° atomic steps. "
            "Returns metrics with updated heading and "
            "euclidean_distance_to_goal."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Turn direction",
                },
                "degrees": {
                    "type": "number",
                    "description": (
                        "Degrees to turn. Must be a positive multiple "
                        "of the 10° atomic step."
                    ),
                    "default": 10,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["direction"],
        },
        permission=PermissionLevel.MUTATING,
    )

    # Defensive enum check — the JSON schema declares enum=["left",
    # "right"] but neither OpenAI function calling nor FastMCP enforce
    # that constraint at the protocol layer, so the Tool itself has to
    # validate. Without this, an LLM hallucinated direction (e.g.
    # "up", "around") would form `action="turn_up"` and the bridge
    # would reject it with an opaque "Unknown action" error.
    _ALLOWED_DIRECTIONS = frozenset(("left", "right"))

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        direction = args.get("direction", "left")
        if direction not in self._ALLOWED_DIRECTIONS:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"turn direction must be 'left' or 'right', " f"got {direction!r}"
                ),
            )
        degrees = args.get("degrees", 10)
        action = f"turn_{direction}"
        payload: Dict[str, Any] = {
            "action": action,
            "degrees": degrees,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.last_movement_action = action
        ctx.round_state.round_actions.append(f"turn_{direction}({degrees}°)")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# LookVerticalTool
# ---------------------------------------------------------------------------


class LookVerticalTool:
    """Pitch the camera up or down without changing agent position/heading."""

    metadata = ToolMetadata(
        name="look_vertical",
        category=ToolCategory.NAVIGATION,
        description=(
            "Pitch the camera up or down by `degrees` without changing "
            "agent position or horizontal heading. The bridge "
            "auto-decomposes the request into 10° atomic steps and "
            "keeps pitch constrained to the simulator's look limit."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Vertical look direction",
                },
                "degrees": {
                    "type": "number",
                    "description": (
                        "Degrees to pitch. Must be a positive multiple "
                        "of the 10° atomic step."
                    ),
                    "default": 10,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["direction"],
        },
        permission=PermissionLevel.MUTATING,
    )

    _ALLOWED_DIRECTIONS = frozenset(("up", "down"))

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        direction = args.get("direction", "up")
        if direction not in self._ALLOWED_DIRECTIONS:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"look_vertical direction must be 'up' or 'down', "
                    f"got {direction!r}"
                ),
            )
        degrees = args.get("degrees", 10)
        action = f"look_{direction}"
        payload: Dict[str, Any] = {
            "action": action,
            "degrees": degrees,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.last_movement_action = action
        ctx.round_state.round_actions.append(f"look_{direction}({degrees}°)")
        ctx.round_state.mark_collision_look_vertical(direction, float(degrees))
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# NavigateTool — navmesh only
# ---------------------------------------------------------------------------


class NavigateTool:
    """Navmesh-based multi-step navigation to absolute coordinates."""

    metadata = ToolMetadata(
        name="navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Navigate toward absolute coordinates using navmesh greedy "
            "follower. Executes multiple steps."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate"},
                "y": {"type": "number", "description": "Target Y coordinate"},
                "z": {"type": "number", "description": "Target Z coordinate"},
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum steps (default 10)",
                    "default": 10,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["x", "y", "z"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            goal = [args["x"], args["y"], args["z"]]
        except KeyError as exc:
            return ToolResult(
                ok=False, body={}, error=f"navigate requires x, y, z (missing {exc})"
            )
        payload: Dict[str, Any] = {
            "goal": goal,
            "max_steps": args.get("max_steps", 10),
            "include_metrics": True,
            "include_visuals": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("navigate_step", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        collect_images(result, ctx)
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# FindPathTool — navmesh only
# ---------------------------------------------------------------------------


class FindPathTool:
    """Plan the shortest navmesh path to absolute coordinates."""

    metadata = ToolMetadata(
        name="find_path",
        category=ToolCategory.NAVIGATION,
        description=(
            "Plan shortest path to coordinates. Returns waypoints and "
            "geodesic distance."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y", "z"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            end = [args["x"], args["y"], args["z"]]
        except KeyError as exc:
            return ToolResult(
                ok=False, body={}, error=f"find_path requires x, y, z (missing {exc})"
            )
        try:
            result = ctx.bridge.call("find_shortest_path", {"end": end})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# SamplePointTool — navmesh only
# ---------------------------------------------------------------------------


class SamplePointTool:
    """Sample a random navigable point for exploration."""

    metadata = ToolMetadata(
        name="sample_point",
        category=ToolCategory.NAVIGATION,
        description="Sample a random navigable point for exploration.",
        parameters_schema={"type": "object", "properties": {}},
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("sample_navigable_point", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# ResetAgentPoseTool
# ---------------------------------------------------------------------------


class ResetAgentPoseTool:
    """Teleport to (x, y, z) with heading yaw in navmesh mode.

    This recovery primitive requires a loaded pathfinder for snapping.
    """

    metadata = ToolMetadata(
        name="reset_agent_pose",
        category=ToolCategory.NAVIGATION,
        description=(
            "Teleport the agent to an absolute (x, y, z) world pose "
            "with heading yaw (radians about Y). Snaps position to "
            "navmesh when one is loaded. Use this as a last-resort "
            "escape from geometry-deadlock where every motion tool "
            "returns immediate collision; not a routine navigation "
            "primitive."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "World X in metres"},
                "y": {"type": "number", "description": "World Y in metres"},
                "z": {"type": "number", "description": "World Z in metres"},
                "yaw": {
                    "type": "number",
                    "description": (
                        "Heading in radians about the habitat up "
                        "axis (Y). yaw=0 → facing -Z."
                    ),
                },
            },
            "required": ["x", "y", "z", "yaw"],
        },
        permission=PermissionLevel.MUTATING,
        allowed_nav_modes={"navmesh"},
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {
            "x": args.get("x"),
            "y": args.get("y"),
            "z": args.get("z"),
            "yaw": args.get("yaw"),
        }
        try:
            result = ctx.bridge.call("reset_agent_pose", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(ForwardTool())
ToolRegistry.register(BackwardTool())
ToolRegistry.register(TurnTool())
ToolRegistry.register(LookVerticalTool())
ToolRegistry.register(NavigateTool())
ToolRegistry.register(FindPathTool())
ToolRegistry.register(SamplePointTool())
ToolRegistry.register(ResetAgentPoseTool())


__all__ = [
    "ForwardTool",
    "BackwardTool",
    "TurnTool",
    "LookVerticalTool",
    "NavigateTool",
    "ResetAgentPoseTool",
    "FindPathTool",
    "SamplePointTool",
]
