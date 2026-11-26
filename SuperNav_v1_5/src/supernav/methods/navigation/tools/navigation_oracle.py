"""Visual and debug oracle local navigation tools for external MCP agents."""

from __future__ import annotations

import os
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

_INCLUDE_IMAGES_PARAM = {
    "type": "boolean",
    "description": (
        "When true, MCP responses inline the direction-labelled surround "
        "camera images captured after navigation."
    ),
    "default": True,
}
_DEFAULT_VISUAL_NAV_HORIZON_M = 40.0
# Geo-based executor mode (HAB_VISUAL_POINT_GEO_BASED_EXECUTOR): each call executes
# immediately (no preview/confirm round-trip), the follower step budget
# defaults to 200 without the 32-step cap, and the geodesic horizon is
# practically unbounded. The flag is forwarded to the bridge in the payload
# because arm-level environment reaches this MCP process, not the bridge.
_GEO_BASED_EXECUTOR_DEFAULT_MAX_STEPS = 200
_GEO_BASED_EXECUTOR_VISUAL_NAV_HORIZON_M = 10000.0


def _visual_point_geo_based_executor_mode() -> bool:
    return parse_bool_flag(
        os.environ.get("HAB_VISUAL_POINT_GEO_BASED_EXECUTOR"), default=False
    )


def _visual_point_point_only_mode() -> bool:
    return parse_bool_flag(os.environ.get("HAB_VISUAL_POINT_POINT_ONLY"), default=False)


def _visual_point_parameters_schema() -> Dict[str, Any]:
    geo_based_executor = _visual_point_geo_based_executor_mode()
    properties: Dict[str, Any] = {
        "view": {
            "type": "string",
            "description": (
                "Preferred MCP shortcut for the latest panorama view: front, "
                "right, back, or left. Use front when the target is visible in "
                "the front image, even if it is on that image's right side."
            ),
        },
        "image_ref": {
            "type": "string",
            "description": (
                "Compatibility field copied from the latest panorama_images row; "
                "prefer view when using MCP."
            ),
        },
        "point": {
            "type": "array",
            "description": "Normalized [x, y] visual anchor.",
        },
        "anchor": {
            "type": "string",
            "description": "auto, center, bottom_center, or lower_band.",
            "default": "auto",
        },
        "intent": {
            "type": "string",
            "description": "approach, pass_through, or inspect.",
            "default": "approach",
        },
        "search_radius": {
            "type": "number",
            "description": "Normalized local search radius around the anchor.",
            "default": 0.10,
        },
        "max_steps": {
            "type": "integer",
            "description": (
                "Maximum navmesh follower steps for this local hop."
                + (
                    " No hard step cap in this deployment."
                    if geo_based_executor
                    else ""
                )
            ),
            "default": _GEO_BASED_EXECUTOR_DEFAULT_MAX_STEPS if geo_based_executor else 20,
        },
        "horizon_m": {
            "type": "number",
            "description": (
                "Maximum geodesic distance for this local hop."
                + (
                    " Effectively unlimited in this deployment."
                    if geo_based_executor
                    else ""
                )
            ),
            "default": (
                _GEO_BASED_EXECUTOR_VISUAL_NAV_HORIZON_M
                if geo_based_executor
                else _DEFAULT_VISUAL_NAV_HORIZON_M
            ),
        },
        "goal_radius": {
            "type": "number",
            "description": "Stop when within this distance of the local target.",
            "default": 0.5,
        },
        "standoff_m": {
            "type": "number",
            "description": "Preferred distance to stop away from visual targets.",
            "default": 0.7,
        },
        "include_images": _INCLUDE_IMAGES_PARAM,
        "confirm_token": {
            "type": "string",
            "description": (
                "Unused in this deployment: every call executes immediately "
                "and no preview/confirm round-trip exists."
                if geo_based_executor
                else (
                    "Token returned by a preview_ready response. Omit for preview; "
                    "repeat the same image_ref and point with this token to execute."
                )
            ),
        },
    }
    required: list[str] = []
    if _visual_point_point_only_mode():
        required.append("point")
    else:
        properties["point"]["description"] = "Optional normalized [x, y] visual anchor."
        properties["bbox"] = {
            "type": "object",
            "description": (
                "Optional normalized {x,y,width,height} target region. "
                "Use this for visible objects when possible."
            ),
        }
    return {"type": "object", "properties": properties, "required": required}


def _visual_point_description() -> str:
    executor_note = (
        " In this deployment every call executes immediately: there is no "
        "preview/confirm round-trip, no confirm_token, no hard step cap, and "
        "no practical geodesic horizon limit."
        if _visual_point_geo_based_executor_mode()
        else ""
    )
    if _visual_point_point_only_mode():
        return (
            "Image-point local navigator. In MCP, provide view='front', 'right', "
            "'back', or 'left' plus a normalized point; image_ref remains "
            "available for compatibility. Use front when the target is visible "
            "in the front image, even if it is on that image's right side. The "
            "bridge uses depth and navmesh validation to move one bounded local "
            "hop toward a reachable point near that visual anchor. Does not "
            "accept bbox, world coordinates, or scene-graph refs." + executor_note
        )
    return (
        "Image-point local navigator. In MCP, provide view='front', 'right', "
        "'back', or 'left' plus a normalized point and/or bbox; image_ref remains "
        "available for compatibility. Use front when the target is visible in "
        "the front image, even if it is on that image's right side. The bridge "
        "uses depth and navmesh validation to move one bounded local hop toward "
        "a reachable point near that visual anchor. Does not accept world "
        "coordinates or scene-graph refs." + executor_note
    )


def _visual_point_when_to_use() -> str:
    if _visual_point_point_only_mode():
        return (
            "Use in navmesh mode when the target can be indicated directly "
            "on a latest panorama image. Use a normalized point on the visible "
            "target or reachable floor."
        )
    return (
        "Use in navmesh mode when the target can be indicated directly "
        "on a latest panorama image. Use bbox for visible objects and "
        "point for floor patches, doorways, or corridor openings."
    )


class OracleLocalNavigateTool:
    metadata = ToolMetadata(
        name="oracle_local_navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Temporary oracle local navigator for MCP agents. Choose an exact "
            "target_ref from the latest panorama visible_nav_targets. The "
            "bridge uses the session scene graph and navmesh to move a bounded "
            "local hop toward a reachable standoff point. Does not accept "
            "coordinates, free-text instructions, or label-only targets."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "target_ref": {
                    "type": "string",
                    "description": (
                        "Opaque target_ref copied exactly from the latest "
                        "visible_nav_targets row."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum navmesh follower steps for this local hop.",
                    "default": 40,
                },
                "horizon_m": {
                    "type": "number",
                    "description": "Maximum geodesic distance for this local hop.",
                    "default": 3.0,
                },
                "goal_radius": {
                    "type": "number",
                    "description": "Stop when within this distance of the local target.",
                    "default": 0.6,
                },
                "standoff_m": {
                    "type": "number",
                    "description": "Preferred distance to stop away from object centers.",
                    "default": 0.7,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["target_ref"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Approach a currently visible target_ref using "
            "privileged scene-graph/navmesh information."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        target_ref = str(args.get("target_ref") or "").strip()
        if not target_ref:
            return ToolResult(
                ok=False,
                body={},
                error="oracle_local_navigate requires 'target_ref'",
            )
        payload: Dict[str, Any] = {
            "target_ref": target_ref,
            "max_steps": int(args.get("max_steps", 40)),
            "horizon_m": float(args.get("horizon_m", 3.0)),
            "goal_radius": float(args.get("goal_radius", 0.6)),
            "standoff_m": float(args.get("standoff_m", 0.7)),
            "output_dir": ctx.output_dir,
        }
        try:
            result = ctx.bridge.call("navigate_oracle_local", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            return ToolResult(
                ok=False,
                body=body,
                error=str(body.get("error") or body.get("status")),
            )
        return ToolResult(ok=True, body=body)


class VisualLocalNavigateTool:
    metadata = ToolMetadata(
        name="visual_local_navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Vision-grounded local navigator. Provide a visual target phrase "
            "for the object or landmark currently sought in the FRONT field of "
            "view; the bridge uses Grounding DINO on the front-facing image, "
            "binds the detected box to an internal scene-graph object "
            "projection, and moves one bounded local hop toward it. Fails "
            "closed when the phrase cannot be visually grounded in the front "
            "view."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "Visual target phrase to ground in the latest "
                        "surround images, e.g. 'a red sofa' or 'the doorway'."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum navmesh follower steps for this local hop.",
                    "default": 40,
                },
                "horizon_m": {
                    "type": "number",
                    "description": "Maximum geodesic distance for this local hop.",
                    "default": 3.0,
                },
                "goal_radius": {
                    "type": "number",
                    "description": "Stop when within this distance of the local target.",
                    "default": 0.6,
                },
                "standoff_m": {
                    "type": "number",
                    "description": "Preferred distance to stop away from object centers.",
                    "default": 0.7,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["instruction"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Use in navmesh mode for one bounded local hop toward an object "
            "or landmark that is visible in the current FRONT-facing camera "
            "view. If it returns no_visual_grounding or not_in_front_view, "
            "turn toward the target with hab_turn and retry."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            return ToolResult(
                ok=False,
                body={},
                error="visual_local_navigate requires 'instruction'",
            )
        payload: Dict[str, Any] = {
            "instruction": instruction,
            "max_steps": int(args.get("max_steps", 40)),
            "horizon_m": float(args.get("horizon_m", 3.0)),
            "goal_radius": float(args.get("goal_radius", 0.6)),
            "standoff_m": float(args.get("standoff_m", 0.7)),
            "output_dir": ctx.output_dir,
        }
        try:
            result = ctx.bridge.call("navigate_visual_local", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            # Grounding failure is an expected closed outcome, not a transport
            # failure. Keep the structured status visible so the agent can
            # switch back to primitive visual exploration.
            return ToolResult(ok=True, body=body)
        return ToolResult(ok=True, body=body)


class VisualGroundPreviewTool:
    metadata = ToolMetadata(
        name="visual_ground_preview",
        category=ToolCategory.NAVIGATION,
        description=(
            "LocateAnything-based visual grounding. In MCP, provide view='front', "
            "'right', 'back', or 'left' plus a natural-language phrase; image_ref "
            "remains available for compatibility. Use front when the target is "
            "visible in the front image, even if it is on that image's right side. "
            "The bridge detects candidate targets, validates reachable navmesh "
            "points, and directly moves when there is exactly one reachable "
            "candidate. It returns preview_ready with a numbered overlay only "
            "when multiple candidates require agent selection; confirm by "
            "calling this same tool with confirm_token and candidate_id."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": (
                        "Preferred MCP shortcut for the latest panorama view: front, "
                        "right, back, or left. Use front when the target is visible "
                        "in the front image, even if it is on that image's right side."
                    ),
                },
                "image_ref": {
                    "type": "string",
                    "description": (
                        "Compatibility field copied from the latest panorama_images "
                        "row; prefer view when using MCP."
                    ),
                },
                "phrase": {
                    "type": "string",
                    "description": "Natural-language target description, e.g. 'light gray sofa'.",
                },
                "mode": {
                    "type": "string",
                    "description": "Detection mode: 'box' (default) or 'point'.",
                    "default": "box",
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return.",
                    "default": 4,
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum LocateAnything confidence score.",
                    "default": 0.30,
                },
                "horizon_m": {
                    "type": "number",
                    "description": "Maximum geodesic distance for a reachable candidate.",
                    "default": _DEFAULT_VISUAL_NAV_HORIZON_M,
                },
                "standoff_m": {
                    "type": "number",
                    "description": "Preferred stop distance from object centers.",
                    "default": 0.7,
                },
                "goal_radius": {
                    "type": "number",
                    "description": "Confirmed move stop radius.",
                    "default": 0.3,
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Optional navmesh follower step cap. Omit to navigate until "
                        "the visual target is reached or navigation is blocked."
                    ),
                },
                "confirm_token": {
                    "type": "string",
                    "description": (
                        "Token returned by a preview_ready response. Omit for "
                        "preview; include with candidate_id to execute."
                    ),
                },
                "candidate_id": {
                    "type": "integer",
                    "description": "Numbered candidate id from preview_ready candidates.",
                },
                "branch_id": {
                    "type": "string",
                    "description": (
                        "Optional branch id from hab_register_spatial_junction. "
                        "Repeat it on preview and confirmation so benchmark spatial "
                        "memory can bind formal movement to that registered opening."
                    ),
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": [],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Use in navmesh mode when you want to ground a phrase in a latest "
            "panorama view. Prefer view over image_ref in MCP. A single reachable "
            "candidate moves directly. If preview_ready is returned, inspect the "
            "numbered overlay_image and call this same tool with confirm_token "
            "and candidate_id to move."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_ref = str(args.get("image_ref") or "").strip()
        phrase = str(args.get("phrase") or "").strip()
        if not image_ref:
            return ToolResult(
                ok=False,
                body={},
                error="visual_ground_preview requires 'image_ref'",
            )
        confirm_token = str(args.get("confirm_token") or "").strip()
        if not phrase and not confirm_token:
            return ToolResult(
                ok=False,
                body={},
                error="visual_ground_preview requires 'phrase' or 'confirm_token'",
            )
        payload: Dict[str, Any] = {
            "image_ref": image_ref,
            "phrase": phrase,
            "mode": str(args.get("mode") or "box").strip().lower(),
            "max_candidates": int(args.get("max_candidates", 4)),
            "score_threshold": float(args.get("score_threshold", 0.30)),
            "horizon_m": float(
                args.get("horizon_m", _DEFAULT_VISUAL_NAV_HORIZON_M)
            ),
            "standoff_m": float(args.get("standoff_m", 0.7)),
            "goal_radius": float(args.get("goal_radius", 0.3)),
            "output_dir": ctx.output_dir,
        }
        if args.get("max_steps") is not None:
            payload["max_steps"] = int(args["max_steps"])
        if confirm_token:
            payload["confirm_token"] = confirm_token
        if args.get("candidate_id") is not None:
            payload["candidate_id"] = args["candidate_id"]
        try:
            result = ctx.bridge.call("navigate_visual_ground_preview", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            return ToolResult(ok=True, body=body)
        return ToolResult(ok=True, body=body)


class VisualPointNavigateTool:
    metadata = ToolMetadata(
        name="visual_point_navigate",
        category=ToolCategory.NAVIGATION,
        description=_visual_point_description(),
        parameters_schema=_visual_point_parameters_schema(),
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=_visual_point_when_to_use(),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_ref = str(args.get("image_ref") or "").strip()
        if not image_ref:
            return ToolResult(
                ok=False,
                body={},
                error="visual_point_navigate requires 'image_ref'",
            )
        if _visual_point_point_only_mode() and args.get("bbox") is not None:
            return ToolResult(
                ok=False,
                body={
                    "ok": False,
                    "status": "bbox_not_allowed",
                    "error": "visual_point point-only mode rejects bbox; use point=[x,y]",
                },
                error="visual_point point-only mode rejects bbox; use point=[x,y]",
            )
        if _visual_point_point_only_mode() and args.get("point") is None:
            return ToolResult(
                ok=False,
                body={
                    "ok": False,
                    "status": "point_required",
                    "error": "visual_point point-only mode requires point=[x,y]",
                },
                error="visual_point point-only mode requires point=[x,y]",
            )
        geo_based_executor = _visual_point_geo_based_executor_mode()
        payload: Dict[str, Any] = {
            "image_ref": image_ref,
            "anchor": args.get("anchor", "auto"),
            "intent": args.get("intent", "approach"),
            "search_radius": float(args.get("search_radius", 0.10)),
            "max_steps": int(
                args.get(
                    "max_steps",
                    _GEO_BASED_EXECUTOR_DEFAULT_MAX_STEPS if geo_based_executor else 20,
                )
            ),
            "horizon_m": float(
                args.get(
                    "horizon_m",
                    (
                        _GEO_BASED_EXECUTOR_VISUAL_NAV_HORIZON_M
                        if geo_based_executor
                        else _DEFAULT_VISUAL_NAV_HORIZON_M
                    ),
                )
            ),
            "goal_radius": float(args.get("goal_radius", 0.3)),
            "standoff_m": float(args.get("standoff_m", 0.7)),
            "output_dir": ctx.output_dir,
        }
        if geo_based_executor:
            payload["geo_based_executor"] = True
        if args.get("point") is not None:
            payload["point"] = args["point"]
        if args.get("bbox") is not None:
            payload["bbox"] = args["bbox"]
        if args.get("confirm_token") is not None:
            payload["confirm_token"] = args["confirm_token"]
        try:
            result = ctx.bridge.call("navigate_visual_point", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            return ToolResult(ok=True, body=body)
        return ToolResult(ok=True, body=body)


class VisualOverlayNavigateTool:
    metadata = ToolMetadata(
        name="visual_overlay_navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Overlay-alias local navigator. Provide image_ref from the latest "
            "overlay panorama_images row and target_alias exactly as printed "
            "on that image, e.g. obj_189. The bridge privately resolves the "
            "visible scene-graph object id to a current target and moves one "
            "bounded navmesh hop. Does not accept coordinates, labels, "
            "free-text target descriptions, or ids absent from the latest "
            "overlay image."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "image_ref copied from the latest panorama_images row.",
                },
                "target_alias": {
                    "type": "string",
                    "description": "Visible overlay alias copied from the image, e.g. obj_189.",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum navmesh follower steps for this local hop.",
                    "default": 40,
                },
                "horizon_m": {
                    "type": "number",
                    "description": "Maximum geodesic distance for this local overlay approach.",
                    "default": _DEFAULT_VISUAL_NAV_HORIZON_M,
                },
                "goal_radius": {
                    "type": "number",
                    "description": "Stop when within this distance of the local target.",
                    "default": 0.6,
                },
                "standoff_m": {
                    "type": "number",
                    "description": "Preferred distance to stop away from object centers.",
                    "default": 0.7,
                },
                "include_images": _INCLUDE_IMAGES_PARAM,
            },
            "required": ["image_ref", "target_alias"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Use in navmesh mode only when the latest panorama image visibly "
            "shows obj_<number> overlay labels. Copy the alias from the image; "
            "aliases are per-image and expire after the next capture."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_ref = str(args.get("image_ref") or "").strip()
        target_alias = str(args.get("target_alias") or "").strip()
        if not image_ref:
            return ToolResult(
                ok=False,
                body={},
                error="visual_overlay_navigate requires 'image_ref'",
            )
        if not target_alias:
            return ToolResult(
                ok=False,
                body={},
                error="visual_overlay_navigate requires 'target_alias'",
            )
        payload: Dict[str, Any] = {
            "image_ref": image_ref,
            "target_alias": target_alias,
            "max_steps": int(args.get("max_steps", 40)),
            "horizon_m": float(
                args.get("horizon_m", _DEFAULT_VISUAL_NAV_HORIZON_M)
            ),
            "goal_radius": float(args.get("goal_radius", 0.6)),
            "standoff_m": float(args.get("standoff_m", 0.7)),
            "output_dir": ctx.output_dir,
        }
        try:
            result = ctx.bridge.call("navigate_visual_overlay", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            return ToolResult(ok=True, body=body)
        return ToolResult(ok=True, body=body)


class OracleLocalMapTool:
    metadata = ToolMetadata(
        name="oracle_local_map",
        category=ToolCategory.MAPPING,
        description=(
            "Legacy oracle local map query, not registered for MCP agents. "
            "Returns a local "
            "topdown image and a label-indexed list of nearby scene-graph "
            "objects with distances, bearings, turn hints, and reachability. "
            "Does not expose coordinates."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "radius_m": {
                    "type": "number",
                    "description": "Local map radius around the agent in meters.",
                    "default": 5.0,
                },
                "max_objects": {
                    "type": "integer",
                    "description": "Maximum number of nearby objects to return.",
                    "default": 30,
                },
                "label_filter": {
                    "type": "string",
                    "description": (
                        "Optional object label filter, e.g. 'chair' or 'door'. "
                        "Leave empty to list all nearby labelled objects."
                    ),
                    "default": "",
                },
                "meters_per_pixel": {
                    "type": "number",
                    "description": "Topdown map resolution in meters per pixel.",
                    "default": 0.05,
                },
                "include_image": {
                    "type": "boolean",
                    "description": "When true, return and inline the local topdown map PNG.",
                    "default": True,
                },
            },
            "required": [],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
        requires_session=True,
        when_to_use=(
            "Bridge-internal diagnostic helper for inspecting nearby "
            "GT-labelled objects without seeing world coordinates."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {
            "radius_m": float(args.get("radius_m", 5.0)),
            "max_objects": int(args.get("max_objects", 30)),
            "label_filter": str(args.get("label_filter") or ""),
            "meters_per_pixel": float(args.get("meters_per_pixel", 0.05)),
            "include_image": parse_bool_flag(args.get("include_image"), default=True),
            "output_dir": ctx.output_dir,
        }
        try:
            result = ctx.bridge.call("get_oracle_local_map", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        body = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(body, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        if body.get("ok") is False:
            return ToolResult(
                ok=False,
                body=body,
                error=str(body.get("error") or body.get("status")),
            )
        return ToolResult(ok=True, body=body)


ToolRegistry.register(VisualLocalNavigateTool())
ToolRegistry.register(VisualGroundPreviewTool())
ToolRegistry.register(VisualOverlayNavigateTool())
ToolRegistry.register(VisualPointNavigateTool())

if os.environ.get("HAB_ENABLE_ORACLE_LOCAL_NAV", "0") == "1":
    ToolRegistry.register(OracleLocalNavigateTool())


__all__ = [
    "OracleLocalNavigateTool",
    "VisualGroundPreviewTool",
    "VisualLocalNavigateTool",
    "VisualOverlayNavigateTool",
    "VisualPointNavigateTool",
]
