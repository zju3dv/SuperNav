"""Read-only depth tools that query the bridge without image capture."""

from __future__ import annotations

from typing import Any, Dict


from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# DepthAnalyzeTool
# ---------------------------------------------------------------------------


class DepthAnalyzeTool:
    """Analyze depth sensor data in 3 front-facing regions."""

    metadata = ToolMetadata(
        name="depth_analyze",
        category=ToolCategory.PERCEPTION,
        description=(
            "Analyze depth sensor data in 3 regions (front_left, "
            "front_center, front_right). Returns min/mean distance per "
            "region to detect obstacles."
        ),
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("analyze_depth", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        ctx.round_state.mark_collision_depth_checked()
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# QueryDepthTool / DepthGridTool / SideDepthGridTool
# ---------------------------------------------------------------------------


class QueryDepthTool:
    """Query precise depth at points or a pixel bounding box."""

    metadata = ToolMetadata(
        name="query_depth",
        category=ToolCategory.PERCEPTION,
        description=(
            "Query depth sensor data at pixel points and/or a pixel bounding "
            "box. Coordinates default to the most recent visible RGB image. "
            "Returns metric depths without exposing raw depth arrays."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "description": "Optional list of [u, v] pixel coordinates.",
                },
                "bbox": {
                    "type": "array",
                    "description": "Optional [x1, y1, x2, y2] pixel region.",
                },
                "coordinate_space": {
                    "type": "string",
                    "enum": ["rgb", "depth"],
                    "description": (
                        "Coordinate space for points/bbox. Default 'rgb' means "
                        "the visible RGB image returned by the latest look or "
                        "panorama; 'depth' means raw depth sensor pixels."
                    ),
                    "default": "rgb",
                },
            },
        },
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}
        if args.get("points") is not None:
            payload["points"] = args["points"]
        if args.get("bbox") is not None:
            payload["bbox"] = args["bbox"]
        if args.get("coordinate_space") is not None:
            payload["coordinate_space"] = args["coordinate_space"]
        visual_metadata = ctx.round_state.last_visual_metadata
        if isinstance(visual_metadata, dict):
            for key in (
                "width",
                "height",
                "original_width",
                "original_height",
                "downsampled_for_agent",
                "path",
                "original_path",
            ):
                if visual_metadata.get(key) is not None:
                    payload[f"rgb_{key}"] = visual_metadata[key]
        try:
            result = ctx.bridge.call("query_depth", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


class DepthGridTool:
    """Analyze current forward depth as a configurable grid."""

    metadata = ToolMetadata(
        name="depth_grid",
        category=ToolCategory.PERCEPTION,
        description=(
            "Analyze the current forward depth frame as a grid. Defaults to "
            "5x5 cells with min/mean distance and clear flags per cell."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "rows": {
                    "type": "integer",
                    "description": "Grid rows, default 5.",
                    "default": 5,
                },
                "cols": {
                    "type": "integer",
                    "description": "Grid columns, default 5.",
                    "default": 5,
                },
                "clearance_threshold": {
                    "type": "number",
                    "description": "Meters required for a cell to be clear.",
                    "default": 0.5,
                },
            },
        },
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload = {
            "rows": args.get("rows", 5),
            "cols": args.get("cols", 5),
            "clearance_threshold": args.get("clearance_threshold", 0.5),
        }
        try:
            result = ctx.bridge.call("depth_grid", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        ctx.round_state.mark_collision_depth_checked()
        return ToolResult(ok=True, body=result)


class SideDepthGridTool:
    """Capture left/right 90-degree side depth grids and restore heading."""

    metadata = ToolMetadata(
        name="side_depth_grid",
        category=ToolCategory.PERCEPTION,
        description=(
            "Analyze left and right 90-degree side views as depth grids "
            "without changing the agent's final pose or heading."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "rows": {
                    "type": "integer",
                    "description": "Grid rows per side, default 5.",
                    "default": 5,
                },
                "cols": {
                    "type": "integer",
                    "description": "Grid columns per side, default 5.",
                    "default": 5,
                },
                "clearance_threshold": {
                    "type": "number",
                    "description": "Meters required for a cell to be clear.",
                    "default": 0.5,
                },
            },
        },
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload = {
            "rows": args.get("rows", 5),
            "cols": args.get("cols", 5),
            "clearance_threshold": args.get("clearance_threshold", 0.5),
        }
        try:
            result = ctx.bridge.call("side_depth_grid", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(DepthAnalyzeTool())
ToolRegistry.register(QueryDepthTool())
ToolRegistry.register(DepthGridTool())
ToolRegistry.register(SideDepthGridTool())


__all__ = [
    "DepthAnalyzeTool",
    "QueryDepthTool",
    "DepthGridTool",
    "SideDepthGridTool",
]
