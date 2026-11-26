"""LocalNavigateTool — learned point-goal local navigation (localnav).

The big-brain agent points at something it can currently see —
`hab_local_navigate(image_ref="pano:…:front", point=[0.62, 0.55])` — and the
bridge runs the localnav closed loop (diffusion-policy trajectory sampling →
scored selection → real discrete primitives) until the point is reached or
the loop escalates. The executor consumes RGB(-D) at runtime, without
navmesh planning or teleport execution.

Thin wrapper around the 'navigate_with_localnav' bridge action (localnav_nav.py);
model inference is served by python -m supernav.methods.localnav.server.
"""

from __future__ import annotations

from supernav.paths import workspace_root

import os
from typing import Any, Dict

from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


class LocalNavigateTool:
    metadata = ToolMetadata(
        name="local_navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Learned closed-loop navigation to a POINT you can currently see. "
            "Pick a pixel on an image from the LATEST visual capture (panorama or "
            "view) and pass its image_ref plus the normalized point [x, y] "
            "((0,0)=top-left, (1,1)=bottom-right); the local policy walks there "
            "with real forward/turn primitives (collision-honest, works in "
            "mapless mode). Returns status reached|needs_agent|timeout|blocked "
            "together with a FRESH four-view surround: every hop must be marked "
            "on the latest returned surround — never reuse a previous call's "
            "point or goal, and never aim from memory. The referenced image "
            "must be from the latest capture (stale refs are rejected); inspect "
            "the new view yourself before declaring the task done."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
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
                    "description": "image_ref of a frame from the LATEST visual capture",
                },
                "point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "normalized [x, y] target point on that image, each in [0, 1]",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "max discrete primitives for this hop (default 100)",
                },
                "include_images": {
                    "type": "boolean",
                    "description": "inline the last rollout frames in the MCP response",
                    "default": True,
                },
            },
            "required": [],
        },
        allowed_nav_modes={"navmesh", "mapless"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Walk to something visible in the latest capture (doorway, object, "
            "corridor end) — the learned, mapless-capable local hop."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal_id = args.get("goal_id")
        image_ref = args.get("image_ref")
        point = args.get("point")
        if goal_id:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    "goal_id retries are removed: every hop must pass image_ref "
                    "+ point marked on the LATEST surround returned by the "
                    "previous call. Inspect that surround and mark a fresh point."
                ),
            )
        if not image_ref or point is None:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    "hab_local_navigate requires image_ref + point (normalized "
                    "[x, y] on that image) marked on the latest surround"
                ),
            )
        payload: Dict[str, Any] = {}
        payload["image_ref"] = image_ref
        payload["point"] = point
        if args.get("max_steps") is not None:
            payload["max_steps"] = int(args["max_steps"])
        call_timeout = float(os.environ.get("NAV_LOCALNAV_CALL_TIMEOUT", "300"))
        # The bridge deadline precedes the transport timeout to avoid queued requests.
        payload["deadline_s"] = call_timeout * 0.9
        try:
            res = ctx.bridge.call(
                "navigate_with_localnav",
                payload,
                timeout=call_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — surface bridge transport errors
            return ToolResult(ok=False, body={}, error=str(exc))
        body = res.get("result", res) if isinstance(res, dict) else res
        if isinstance(body, dict) and body.get("ok") is False:
            return ToolResult(ok=False, body=body, error=body.get("error"))
        return ToolResult(ok=True, body=body)


def _localnav_ctl_dir(session_id: str) -> str:
    import tempfile

    root = (
        os.environ.get("NAV_LOCALNAV_FRAME_ROOT")
        or os.environ.get("NAV_ARTIFACTS_DIR")
        or str(workspace_root() / "data" / "runs" / "artifacts")
    )
    return os.path.join(root, str(session_id))


class LocalNavStatusTool:
    """Progress of the current/last localnav hop — reads the progress file the
    bridge loop writes each replan, so it works even WHILE the single-threaded
    bridge is busy executing the hop (the whole point: `/healthz` and every
    bridge action would queue behind it)."""

    metadata = ToolMetadata(
        name="localnav_status",
        category=ToolCategory.STATUS,
        description=(
            "Progress of the running (or most recent) hab_local_navigate hop: "
            "steps walked, replans, collisions, stop-head probability, elapsed "
            "seconds and status (running|reached|needs_agent|timeout|blocked). "
            "Works while the hop is still executing."
        ),
        parameters_schema={"type": "object", "properties": {}, "required": []},
        allowed_nav_modes={"navmesh", "mapless"},
        permission=PermissionLevel.READ_ONLY,
        requires_session=True,
        when_to_use="Poll a long local_navigate hop instead of waiting blind.",
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        import json as _json

        path = os.path.join(
            _localnav_ctl_dir(ctx.session_id), "localnav.progress.json"
        )
        if not os.path.exists(path):
            return ToolResult(
                ok=True, body={"status": "no_hop_recorded"},
            )
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return ToolResult(ok=True, body=_json.load(fh))
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, body={}, error=str(exc))


class LocalNavStopTool:
    """Cooperative cancel for a running hop — writes the cancel flag the loop
    checks every replan. File-based so it bypasses the blocked bridge."""

    metadata = ToolMetadata(
        name="localnav_stop",
        category=ToolCategory.NAVIGATION,
        description=(
            "Request cancellation of the running hab_local_navigate hop. The "
            "loop stops at the next replan boundary and returns status "
            "needs_agent with reason cancelled_by_agent."
        ),
        parameters_schema={"type": "object", "properties": {}, "required": []},
        allowed_nav_modes={"navmesh", "mapless"},
        permission=PermissionLevel.MUTATING,
        requires_session=True,
        when_to_use=(
            "Abort a hop that is clearly heading the wrong way instead of "
            "waiting for its timeout."
        ),
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctl = _localnav_ctl_dir(ctx.session_id)
        try:
            os.makedirs(ctl, exist_ok=True)
            with open(os.path.join(ctl, "localnav.cancel"), "w") as fh:
                fh.write("cancel")
        except OSError as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body={"status": "cancel_requested"})


ToolRegistry.register(LocalNavigateTool())
ToolRegistry.register(LocalNavStatusTool())
ToolRegistry.register(LocalNavStopTool())
