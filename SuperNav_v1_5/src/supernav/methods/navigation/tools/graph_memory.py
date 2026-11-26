"""Agent-authored memory nodes tools backed by the public v7 schema."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from supernav.methods.navigation.graph_memory import GraphMemoryError, GraphMemoryStore
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


def _store(ctx: ToolContext) -> GraphMemoryStore:
    configured = (
        os.environ.get("SUPERNAV_MEMORY_NODES_PATH", "").strip()
        or os.environ.get("SUPERNAV_GRAPH_MEMORY_PATH", "").strip()
    )
    if configured:
        return GraphMemoryStore(Path(configured))
    session_id = str(ctx.bridge.session_id or ctx.session_id or "").strip()
    if not session_id:
        raise GraphMemoryError("Call hab_init_scene before using memory nodes.")
    path = Path(ctx.output_dir) / session_id / "memory_nodes.json"
    return GraphMemoryStore(path)


def _current_slam_pose(ctx: ToolContext) -> dict[str, Any] | None:
    try:
        result = ctx.bridge.call("get_memory_slam_status", {})
    except Exception:
        return None
    pose = result.get("slam_pose") if isinstance(result, dict) else None
    return pose if isinstance(pose, dict) else None


def _refresh_memory_nodes(ctx: ToolContext, store: GraphMemoryStore) -> None:
    try:
        ctx.bridge.call("refresh_memory_nodes", {"nodes": store.map_nodes()})
    except Exception:
        pass


def _ok(body: Any) -> ToolResult:
    return ToolResult(ok=True, body=body if isinstance(body, dict) else {"result": body})


class ListNodesTool:
    metadata = ToolMetadata(
        name="list_nodes", category=ToolCategory.MEMORY,
        description="List compact memory-node area and door records.",
        permission=PermissionLevel.READ_ONLY,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        del args
        return _ok({"nodes": _store(ctx).list_nodes()})


class GrepNodeTool:
    metadata = ToolMetadata(
        name="grep_node", category=ToolCategory.MEMORY,
        description="Search memory-node names, descriptions, and records.",
        parameters_schema={"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
        permission=PermissionLevel.READ_ONLY,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return _ok({"nodes": _store(ctx).grep_node(args.get("term"))})


class InspectNodeTool:
    metadata = ToolMetadata(
        name="inspect_node", category=ToolCategory.MEMORY,
        description="Inspect one memory node, optionally including panorama paths.",
        parameters_schema={"type": "object", "properties": {"id": {"type": "string"}, "return_images": {"type": "boolean", "default": False}}, "required": ["id"]},
        permission=PermissionLevel.READ_ONLY,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return _ok({"node": _store(ctx).inspect_node(args.get("id"), return_images=bool(args.get("return_images", False)))})


class QueryNodeContextTool:
    metadata = ToolMetadata(
        name="query_node_context", category=ToolCategory.MEMORY,
        description="Search and inspect memory nodes from one consistent snapshot.",
        parameters_schema={"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string"}}, "inspect_ids": {"type": "array", "items": {"type": "string"}}, "return_images": {"type": "boolean", "default": False}}, "required": ["terms", "inspect_ids"]},
        permission=PermissionLevel.READ_ONLY,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return _ok(_store(ctx).query_node_context(terms=args.get("terms"), inspect_ids=args.get("inspect_ids"), return_images=bool(args.get("return_images", False))))


class AddNodeTool:
    metadata = ToolMetadata(
        name="add_node", category=ToolCategory.MEMORY,
        description="Create an area or door node at the current SLAM pose.",
        parameters_schema={"type": "object", "properties": {"node": {"type": "object"}}, "required": ["node"]},
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        store.add_node(args.get("node"), slam_pose=_current_slam_pose(ctx))
        _refresh_memory_nodes(ctx, store)
        return _ok({"status": "node_added"})


class UpsertNodeTool:
    metadata = ToolMetadata(
        name="upsert_node", category=ToolCategory.MEMORY,
        description="Create or update an area or door node at the current SLAM pose.",
        parameters_schema={"type": "object", "properties": {"node": {"type": "object"}}, "required": ["node"]},
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        status = store.upsert_node(args.get("node"), slam_pose=_current_slam_pose(ctx))
        _refresh_memory_nodes(ctx, store)
        return _ok({"status": status})


class UpdateNodeTool:
    metadata = ToolMetadata(
        name="update_node", category=ToolCategory.MEMORY,
        description="Update mutable fields on an existing memory node.",
        parameters_schema={"type": "object", "properties": {"id": {"type": "string"}, "node": {"type": "object"}}, "required": ["id", "node"]},
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        store.update_node(args.get("id"), args.get("node"), slam_pose=_current_slam_pose(ctx))
        _refresh_memory_nodes(ctx, store)
        return _ok({"status": "node_updated"})


class ApplyMemoryNodeMutationsTool:
    metadata = ToolMetadata(
        name="apply_memory_node_mutations", category=ToolCategory.MEMORY,
        description="Atomically apply memory-node create, upsert, and update operations.",
        parameters_schema={"type": "object", "properties": {"mutations": {"type": "array", "items": {"type": "object"}}}, "required": ["mutations"]},
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        mutations = store.apply_mutations(args.get("mutations"), slam_pose=_current_slam_pose(ctx))
        _refresh_memory_nodes(ctx, store)
        return _ok({"mutations": mutations})


class DeleteNodeTool:
    metadata = ToolMetadata(
        name="delete_node", category=ToolCategory.MEMORY,
        description="Delete one memory node.",
        parameters_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        permission=PermissionLevel.DESTRUCTIVE,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        store.delete_node(args.get("id"))
        _refresh_memory_nodes(ctx, store)
        return _ok({"status": "node_deleted"})


class ReportMemoryPosTool:
    metadata = ToolMetadata(
        name="report_memory_pos", category=ToolCategory.MEMORY,
        description="Report which memory node the agent says it currently occupies.",
        parameters_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        permission=PermissionLevel.READ_ONLY,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        body = store.report_memory_pos(args.get("id"))
        _refresh_memory_nodes(ctx, store)
        try:
            body["relocalization_prior"] = ctx.bridge.call(
                "check_relocalization_prior",
                {
                    "node_id": args.get("id"),
                    "tool_seq": ctx.tool_seq,
                    "node": body.get("current_node"),
                },
            )
        except Exception as exc:
            body["relocalization_prior"] = {"accepted": False, "error": str(exc)}
        try:
            memory_slam = ctx.bridge.call("get_memory_slam_status", {})
        except Exception as exc:
            memory_slam = {"enabled": False, "error": str(exc)}
        body["spatial_memory"] = memory_slam
        images: list[str] = []
        if isinstance(memory_slam, dict):
            image_path = memory_slam.get("planning_occupancy_image") or memory_slam.get("occupancy_image")
            if isinstance(image_path, str) and Path(image_path).is_file():
                images.append(image_path)
        return ToolResult(ok=True, body=body, captured_images=images)


class NavigateMemoryNodeOccTool:
    metadata = ToolMetadata(
        name="navigate_to_memory_node_occ", category=ToolCategory.NAVIGATION,
        description="Navigate to a memory node using only its SLAM pose and the occupancy map.",
        parameters_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        node = _store(ctx).inspect_node(args.get("id"), return_images=False)
        pose = node.get("slam_pose")
        if not isinstance(pose, dict):
            raise GraphMemoryError(f"node has no SLAM pose: {args.get('id')}")
        return _ok(
            ctx.bridge.call(
                "navigate_slam_occupancy",
                {
                    "goal": pose,
                    "node_id": args.get("id"),
                    "output_dir": ctx.output_dir,
                },
            )
        )


for _tool in (
    ListNodesTool(), GrepNodeTool(), InspectNodeTool(), QueryNodeContextTool(),
    AddNodeTool(), UpsertNodeTool(), UpdateNodeTool(), ApplyMemoryNodeMutationsTool(),
    DeleteNodeTool(), ReportMemoryPosTool(), NavigateMemoryNodeOccTool(),
):
    ToolRegistry.register(_tool)


__all__ = [name for name in globals() if name.endswith("Tool")]
