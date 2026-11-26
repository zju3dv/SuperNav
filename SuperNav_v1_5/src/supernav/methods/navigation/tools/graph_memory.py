"""Agent-authored memory nodes tools backed by the public v7 schema."""

from __future__ import annotations

import hashlib
import json
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


_memory_node_occ_nav_previews: dict[str, dict[str, Any]] = {}


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


def _slam_area_node(ctx: ToolContext, node_id: str) -> dict[str, Any] | None:
    """Resolve an ephemeral A<N> target from the latest rendered SLAM map."""
    try:
        result = ctx.bridge.call("get_memory_slam_status", {})
    except Exception:
        return None
    nodes = result.get("slam_areas") if isinstance(result, dict) else None
    if not isinstance(nodes, list):
        return None
    return next(
        (
            dict(node)
            for node in nodes
            if isinstance(node, dict) and node.get("id") == node_id
        ),
        None,
    )


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
        parameters_schema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "confirm_token": {"type": "string", "default": ""},
                "max_steps": {"type": "integer", "default": 80},
                "goal_radius_m": {"type": "number", "default": 0.45},
                "replan_attempts": {"type": "integer", "default": 2},
                "include_images": {"type": "boolean", "default": True},
            },
            "required": ["node_id"],
        },
        permission=PermissionLevel.MUTATING,
    )
    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        node_id = str(args.get("node_id") or "").strip()
        if not node_id:
            raise GraphMemoryError("navigate_to_memory_node_occ requires node_id")
        try:
            node = _store(ctx).inspect_node(node_id, return_images=False)
        except GraphMemoryError:
            node = _slam_area_node(ctx, node_id)
            if node is None:
                raise
        pose = node.get("slam_pose")
        if not isinstance(pose, dict):
            raise GraphMemoryError(f"node has no SLAM pose: {node_id}")
        confirm_token = str(args.get("confirm_token") or "").strip()
        if not confirm_token:
            plan = ctx.bridge.call(
                "navigate_slam_occupancy",
                {
                    "goal_slam_pose": pose,
                    "preview_only": True,
                    "output_dir": ctx.output_dir,
                },
            )
            if plan.get("ok") is False:
                return _ok(
                    {
                        "ok": False,
                        "status": str(plan.get("status") or "unreachable"),
                        "node_id": node_id,
                        "node": {
                            key: node[key]
                            for key in ("id", "kind", "name", "description")
                            if key in node
                        },
                        "plan": plan.get("plan", {}),
                        "2d_slam_map": plan.get("2d_slam_map"),
                        "error": plan.get("error"),
                    }
                )
            material = json.dumps(
                {
                    "session_id": str(ctx.bridge.session_id or ctx.session_id or ""),
                    "node_id": node_id,
                    "slam_pose": pose,
                    "tool_seq": ctx.tool_seq,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            token = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
            _memory_node_occ_nav_previews[token] = {
                "session_id": str(ctx.bridge.session_id or ctx.session_id or ""),
                "node_id": node_id,
                "slam_pose": dict(pose),
                "tool_seq": ctx.tool_seq,
            }
            return _ok(
                {
                    "ok": True,
                    "status": "preview_ready",
                    "node_id": node_id,
                    "node": {
                        key: node[key]
                        for key in ("id", "kind", "name", "description")
                        if key in node
                    },
                    "confirm_token": token,
                    "plan": plan.get("plan", {}),
                    "2d_slam_map": plan.get("2d_slam_map"),
                }
            )
        preview = _memory_node_occ_nav_previews.get(confirm_token)
        session_id = str(ctx.bridge.session_id or ctx.session_id or "")
        if not isinstance(preview, dict) or any(
            (
                preview.get("session_id") != session_id,
                preview.get("node_id") != node_id,
                preview.get("slam_pose") != dict(pose),
            )
        ):
            return _ok(
                {
                    "ok": False,
                    "status": "invalid_confirm_token",
                    "node_id": node_id,
                    "error": "confirm_token does not match the latest occupancy preview for this memory node.",
                }
            )
        _memory_node_occ_nav_previews.pop(confirm_token, None)
        max_steps = max(1, int(args.get("max_steps", 80)))
        timeout_s = max(
            60.0,
            float(
                os.environ.get(
                    "HAB_MCP_MEMORY_NODE_OCC_NAV_TIMEOUT_S",
                    str(30.0 + 2.0 * max_steps),
                )
            ),
        )
        result = ctx.bridge.call(
            "navigate_slam_occupancy",
            {
                "goal_slam_pose": pose,
                "preview_only": False,
                "max_steps": max_steps,
                "goal_radius_m": float(args.get("goal_radius_m", 0.45)),
                "replan_attempts": max(0, int(args.get("replan_attempts", 2))),
                "include_cartographer_map": True,
                "output_dir": ctx.output_dir,
            },
            timeout=timeout_s,
        )
        result["node_id"] = node_id
        result["node"] = {
            key: node[key]
            for key in ("id", "kind", "name", "description")
            if key in node
        }
        return _ok(result)


for _tool in (
    ListNodesTool(), GrepNodeTool(), InspectNodeTool(), QueryNodeContextTool(),
    AddNodeTool(), UpsertNodeTool(), UpdateNodeTool(), ApplyMemoryNodeMutationsTool(),
    DeleteNodeTool(), ReportMemoryPosTool(), NavigateMemoryNodeOccTool(),
):
    ToolRegistry.register(_tool)


__all__ = [name for name in globals() if name.endswith("Tool")]
