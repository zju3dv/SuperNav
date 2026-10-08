"""Video-export tool."""

from __future__ import annotations

from typing import Any, Dict

from supernav.methods.navigation.tools._common import visual_payload
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)

class ExportVideoTool:
    """Export the accumulated frame trace as an mp4 video file."""

    metadata = ToolMetadata(
        name="export_video",
        category=ToolCategory.STATUS,
        description="Export video trace of the navigation session.",
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("export_video_trace", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


ToolRegistry.register(ExportVideoTool())


__all__ = ["ExportVideoTool"]
