# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run-local Go2 graph-memory store for long-instruction external harness runs."""

from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import re
import uuid
from typing import Any


GRAPH_MEMORY_TOOL_NAMES = (
    "go2_list_graph_nodes",
    "go2_grep_graph_node",
    "go2_inspect_graph_node",
    "go2_query_route_context",
    "go2_report_graph_position",
    "go2_graph_map_nodes",
    "go2_apply_graph_mutations",
)
_FORBIDDEN_DOOR_NOTE_WORDS = frozenset({"left", "right", "front", "back", "far", "near"})


def graph_memory_tool_specs() -> list[dict[str, Any]]:
    """Return MCP tool declarations for the proxy-local graph-memory tools."""

    return [
        {
            "name": "go2_list_graph_nodes",
            "description": "List concise summaries of run-local Go2 graph-memory nodes.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "go2_grep_graph_node",
            "description": (
                "Search graph-memory node names, descriptions, records, waypoints, "
                "and directly observed transition notes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
                "additionalProperties": False,
            },
        },
        {
            "name": "go2_inspect_graph_node",
            "description": "Inspect one graph-memory node, optionally including stored image refs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "return_images": {"type": "boolean", "default": True},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "go2_query_route_context",
            "description": (
                "Read term matches and inspected nodes from one graph snapshot. "
                "The response also includes the latest SLAM-map metadata and image when "
                "available. Routes are chosen from current RGB/SLAM evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "current_node_id": {"type": ["string", "null"]},
                    "terms": {"type": "array", "items": {"type": "string"}},
                    "inspect_ids": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "integer", "minimum": 0, "default": 5},
                    "return_images": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "go2_report_graph_position",
            "description": (
                "Report the graph node believed to match the current physical position, "
                "wait up to 10 seconds for prior-triggered relocalization, and return its "
                "status plus the latest SLAM map."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "go2_graph_map_nodes",
            "description": (
                "Read graph-memory node badges for SLAM map overlays. This is internal "
                "navigation-map context and does not expose raw coordinates to policy."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "go2_apply_graph_mutations",
            "description": (
                "Atomically create/update node-only graph memory and area waypoints. Nodes "
                "with pano_images are internally anchored to the latest Go2 SLAM pose "
                "when one is available."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"mutations": {"type": "array", "items": {"type": "object"}}},
                "required": ["mutations"],
                "additionalProperties": False,
            },
        },
    ]


class Go2GraphMemoryStore:
    """Small JSON-backed graph store with fail-closed mutation validation."""

    # Matches the latest Habitat-GS graph-memory document version.  DimOS keeps
    # the Go2-prefixed MCP surface, but writes the same private anchoring fields.
    SCHEMA_VERSION = 6

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.panorama_root = self.path.parent / "panoramas"

    def save_panorama(self, images: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Persist labeled image blocks and return run-relative references."""
        capture_dir = self.panorama_root / f"capture-{uuid.uuid4().hex[:12]}"
        refs: dict[str, str] = {}
        for direction, block in images.items():
            data = block.get("data")
            if not isinstance(data, str):
                continue
            try:
                raw = base64.b64decode(data, validate=True)
            except (ValueError, TypeError):
                continue
            mime = str(block.get("mimeType") or "image/jpeg").lower()
            suffix = ".png" if mime == "image/png" else ".jpg"
            path = capture_dir / f"{direction}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            refs[direction] = str(path.relative_to(self.path.parent.parent))
        if not refs:
            return {}
        return refs

    def load_panorama(self, reference: str) -> dict[str, Any] | None:
        """Load one run-relative panorama reference as an MCP image block."""
        candidate = (self.path.parent.parent / reference).resolve()
        root = self.panorama_root.resolve()
        if root not in candidate.parents or not candidate.is_file():
            return None
        try:
            data = base64.b64encode(candidate.read_bytes()).decode("ascii")
        except OSError:
            return None
        mime = "image/png" if candidate.suffix.lower() == ".png" else "image/jpeg"
        return {"type": "image", "data": data, "mimeType": mime}

    def image_blocks_for_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Hydrate stored panorama refs for a graph inspection response."""
        blocks: list[dict[str, Any]] = []

        def visit(node: Any) -> None:
            if not isinstance(node, dict):
                return
            refs = node.get("pano_images")
            if isinstance(refs, dict):
                for direction, reference in refs.items():
                    if not isinstance(reference, str):
                        continue
                    block = self.load_panorama(reference)
                    if block is not None:
                        blocks.extend(
                            [{"type": "text", "text": str(direction)}, block]
                        )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)

        visit(payload)
        return blocks

    def handle_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        slam_pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args = arguments or {}
        try:
            if name == "go2_list_graph_nodes":
                return _ok({"nodes": [self._summary(node) for node in self._nodes().values()]})
            if name == "go2_grep_graph_node":
                return _ok({"matches": self.grep(str(args.get("term") or ""))})
            if name == "go2_inspect_graph_node":
                return _ok(
                    {
                        "node": self.inspect(
                            str(args.get("id") or ""),
                            return_images=_bool_arg(args, "return_images", True),
                        )
                    }
                )
            if name == "go2_query_route_context":
                return _ok(self.query_route_context(args))
            if name == "go2_report_graph_position":
                return _ok(self.report_position(str(args.get("id") or "")))
            if name == "go2_graph_map_nodes":
                return _ok(
                    {
                        "graph_nodes": [
                            _map_node_for_response(node) for node in self.map_nodes()
                        ],
                        "anchored_node_count": len(
                            [
                                node
                                for node in self.map_nodes()
                                if node.get("pano_images") and node.get("slam_pose")
                            ]
                        ),
                    }
                )
            if name == "go2_apply_graph_mutations":
                return _ok(self.apply_mutations(args.get("mutations"), slam_pose=slam_pose))
        except ValueError as exc:
            return {"success": False, "error": str(exc), "store_path": str(self.path)}
        return {
            "success": False,
            "error": f"unknown graph-memory tool: {name}",
            "store_path": str(self.path),
        }

    def grep(self, term: str) -> list[dict[str, Any]]:
        needle = term.strip().lower()
        if not needle:
            raise ValueError("term must not be empty")
        matches = []
        for node in self._nodes().values():
            haystack = json.dumps(node, ensure_ascii=False).lower()
            if needle in haystack:
                matches.append(self._summary(node))
        return matches

    def inspect(self, node_id: str, *, return_images: bool = True) -> dict[str, Any]:
        nodes = self._nodes()
        node = nodes.get(_require_id(node_id, "id"))
        if node is None:
            raise ValueError(f"unknown graph node id: {node_id}")
        return _node_for_response(node, return_images=return_images)

    def navigation_target(self, node_id: str) -> dict[str, Any] | None:
        """Return private target data for the trusted navigation handoff."""
        node = self._nodes().get(_require_id(node_id, "id"))
        if node is None:
            return None
        pose = node.get("slam_pose") or node.get("navigation_goal")
        return dict(pose) if isinstance(pose, dict) else None

    def report_position(self, node_id: str) -> dict[str, Any]:
        node = self.inspect(node_id, return_images=True)
        return {"current_node": node}

    def query_route_context(self, args: dict[str, Any]) -> dict[str, Any]:
        nodes = self._nodes()
        return_images = _bool_arg(args, "return_images", True)
        current_node_id = _optional_id(args.get("current_node_id"))
        inspect_ids = [
            _require_id(item, "inspect_ids item") for item in args.get("inspect_ids") or []
        ]
        terms = [str(item) for item in args.get("terms") or [] if str(item).strip()]
        # ``depth`` remains accepted for client compatibility, but topology is
        # intentionally not persisted or traversed by the node-only schema.
        _depth(args.get("depth", 5))

        current_node = None
        adjacent_context: list[dict[str, Any]] = []
        paths: list[list[dict[str, str]]] = []
        if current_node_id:
            if current_node_id not in nodes:
                raise ValueError(f"unknown current_node_id: {current_node_id}")
            current_node = _node_for_response(nodes[current_node_id], return_images=return_images)

        inspected = {}
        for node_id in inspect_ids:
            if node_id not in nodes:
                raise ValueError(f"unknown inspect id: {node_id}")
            inspected[node_id] = _node_for_response(nodes[node_id], return_images=return_images)

        matches = {term: self.grep(term) for term in terms}
        return {
            "current_node": current_node,
            "adjacent_context": [],
            "matches": matches,
            "inspected_nodes": inspected,
            # Kept as an empty compatibility field; routes are selected from
            # current RGB/SLAM evidence, never from stored topology.
            "paths": [],
            "store_path": str(self.path),
        }

    def map_nodes(self) -> list[dict[str, Any]]:
        """Return managed fields required to annotate model-visible SLAM maps."""

        return [
            {
                "id": node["id"],
                "order_id": node["order_id"],
                "name": node["name"],
                "kind": node["kind"],
                "has_pano_images": bool(node.get("pano_images")),
                "slam_pose": deepcopy(node.get("slam_pose")),
            }
            for node in sorted(
                self._nodes().values(),
                key=lambda item: int(item["order_id"]),
            )
        ]

    def apply_mutations(
        self,
        mutations: Any,
        *,
        slam_pose: dict[str, Any] | None = None,
        navigation_goal: list[float] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("mutations must be a non-empty array")
        current = self._load()
        nodes = deepcopy(current.get("nodes", {}))
        next_order_id = _next_order_id(current)
        affected: set[str] = set()
        for mutation in mutations:
            if not isinstance(mutation, dict):
                raise ValueError("each mutation must be an object")
            op = str(mutation.get("op") or "")
            if op == "create":
                node = _validated_node(mutation.get("node"), require_complete=True)
                node_id = str(node["id"])
                if node_id in nodes:
                    raise ValueError(f"node already exists: {node_id}")
                nodes[node_id] = _with_system_fields(
                    node,
                    order_id=next_order_id,
                    slam_pose=slam_pose if node.get("pano_images") else None,
                    navigation_goal=navigation_goal,
                )
                next_order_id += 1
                affected.add(node_id)
            elif op == "upsert":
                node = mutation.get("node")
                if not isinstance(node, dict):
                    raise ValueError("upsert mutation requires node object")
                node_id = _require_id(node.get("id"), "node.id")
                if node_id in nodes:
                    nodes[node_id] = _updated_existing_node(
                        nodes[node_id],
                        node,
                        slam_pose=slam_pose,
                        navigation_goal=navigation_goal,
                    )
                else:
                    created = _validated_node(node, require_complete=True)
                    nodes[node_id] = _with_system_fields(
                        created,
                        order_id=next_order_id,
                        slam_pose=slam_pose if created.get("pano_images") else None,
                        navigation_goal=navigation_goal,
                    )
                    next_order_id += 1
                affected.add(node_id)
            elif op == "update":
                node_id = _require_id(mutation.get("id"), "id")
                if node_id not in nodes:
                    raise ValueError(f"cannot update unknown node: {node_id}")
                patch = mutation.get("node")
                if not isinstance(patch, dict):
                    raise ValueError("update mutation requires node object")
                nodes[node_id] = _updated_existing_node(
                    nodes[node_id],
                    patch,
                    slam_pose=slam_pose,
                    navigation_goal=navigation_goal,
                )
                affected.add(node_id)
            elif op == "add_waypoint":
                node_id = _require_id(mutation.get("id"), "id")
                if node_id not in nodes:
                    raise ValueError(f"cannot add waypoint to unknown node: {node_id}")
                node = deepcopy(nodes[node_id])
                if node.get("kind") != "area":
                    raise ValueError("waypoints can only be added to area nodes")
                name = _require_id(mutation.get("name"), "waypoint name")
                description = _require_text(mutation.get("description"), "waypoint description")
                waypoints = dict(node.get("waypoints") or {})
                waypoints[name] = description
                node["waypoints"] = waypoints
                nodes[node_id] = _updated_existing_node(
                    nodes[node_id],
                    {"waypoints": waypoints},
                    slam_pose=None,
                )
                affected.add(node_id)
            else:
                raise ValueError(f"unsupported graph mutation op: {op}")

        self._validate_graph(nodes)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "version": self.SCHEMA_VERSION,
            "next_order_id": next_order_id,
            "nodes": nodes,
        }
        self._save(payload)
        return {
            "affected_nodes": {
                node_id: _node_for_response(nodes[node_id], return_images=False)
                for node_id in sorted(affected)
            },
            "node_count": len(nodes),
            "store_path": str(self.path),
        }

    def _paths_from(self, start_id: str, *, depth: int) -> list[list[dict[str, str]]]:
        nodes = self._nodes()
        out: list[list[dict[str, str]]] = []
        queue: deque[list[str]] = deque([[start_id]])
        while queue:
            path = queue.popleft()
            if len(path) > 1:
                out.append([self._path_step(nodes[node_id]) for node_id in path])
            if len(path) - 1 >= depth:
                continue
            tail = nodes[path[-1]]
            for adjacent in tail.get("adjacents", []):
                next_id = str(adjacent.get("id") or "")
                if next_id in nodes and next_id not in path:
                    queue.append([*path, next_id])
        return out

    def _summary(self, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": node["id"],
            "order_id": node.get("order_id"),
            "kind": node["kind"],
            "name": node["name"],
            "description": node["description"],
            "record_count": len(node.get("records", [])),
            "waypoint_names": sorted((node.get("waypoints") or {}).keys()),
        }

    def _path_step(self, node: dict[str, Any]) -> dict[str, str]:
        return {"id": str(node["id"]), "kind": str(node["kind"]), "name": str(node["name"])}

    def _nodes(self) -> dict[str, dict[str, Any]]:
        return self._load().get("nodes", {})

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "next_order_id": 1, "nodes": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), dict):
            raise ValueError(f"invalid graph-memory store: {self.path}")
        return _migrate_payload(data)

    def _save(self, payload: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _validate_graph(self, nodes: dict[str, dict[str, Any]]) -> None:
        for node_id, node in nodes.items():
            if node_id != node.get("id"):
                raise ValueError(f"node key/id mismatch: {node_id}")
            _validated_node(node, require_complete=True)


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, **payload}


def _validated_node(value: Any, *, require_complete: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("node must be an object")
    node = deepcopy(value)
    node_id = _require_id(node.get("id"), "node.id")
    kind = str(node.get("kind") or "")
    if kind not in {"area", "door"}:
        raise ValueError(f"node {node_id} has invalid kind: {kind!r}")
    if require_complete:
        _require_text(node.get("name"), "node.name")
        _require_text(node.get("description"), "node.description")
    node["id"] = node_id
    node["kind"] = kind
    node["name"] = _require_text(node.get("name"), "node.name")
    node["description"] = _require_text(node.get("description"), "node.description")
    if "adjacents" in node or "paths" in node:
        raise ValueError("node-only graph memory does not accept adjacency or path fields")
    node["records"] = _string_list(node.get("records", []), "records")
    node["pano_images"] = _string_mapping(node.get("pano_images", {}), "pano_images")
    node["slam_pose"] = _validated_slam_pose(node.get("slam_pose"))
    node["navigation_goal"] = _validated_navigation_goal(node.get("navigation_goal"))
    if kind == "area":
        node["waypoints"] = _string_mapping(node.get("waypoints", {}), "waypoints")
    elif "waypoints" in node:
        raise ValueError("door nodes must not have waypoints")
    return node


def _with_system_fields(
    node: dict[str, Any],
    *,
    order_id: int,
    slam_pose: dict[str, Any] | None,
    navigation_goal: list[float] | None = None,
) -> dict[str, Any]:
    return {
        **node,
        "order_id": _require_order_id(order_id),
        "slam_pose": _validated_slam_pose(slam_pose),
        "navigation_goal": _validated_navigation_goal(navigation_goal),
    }


def _updated_existing_node(
    existing: dict[str, Any],
    patch: dict[str, Any],
    *,
    slam_pose: dict[str, Any] | None,
    navigation_goal: list[float] | None = None,
) -> dict[str, Any]:
    system = {
        "order_id": _require_order_id(existing.get("order_id")),
        "slam_pose": _validated_slam_pose(existing.get("slam_pose")),
        "navigation_goal": _validated_navigation_goal(existing.get("navigation_goal")),
    }
    merged = {**deepcopy(existing), **deepcopy(patch), "id": existing["id"]}
    validated = _validated_node(merged, require_complete=True)
    if "pano_images" in patch and slam_pose is not None:
        system["slam_pose"] = _validated_slam_pose(slam_pose)
    if "pano_images" in patch and navigation_goal is not None:
        system["navigation_goal"] = _validated_navigation_goal(navigation_goal)
    return {**validated, **system}


def _validated_adjacents(value: Any, *, node_id: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"node {node_id} adjacents must be an array")
    out = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name", "id", "note"}:
            raise ValueError("each adjacency must contain exactly name, id, and note")
        out.append(
            {
                "name": _require_text(row.get("name"), "adjacency.name"),
                "id": _require_id(row.get("id"), "adjacency.id"),
                "note": _require_text(row.get("note"), "adjacency.note"),
            }
        )
    return out


def _node_for_response(node: dict[str, Any], *, return_images: bool) -> dict[str, Any]:
    payload = deepcopy(node)
    payload.pop("slam_pose", None)
    payload.pop("navigation_goal", None)
    if not return_images:
        payload.pop("pano_images", None)
    return payload


def _map_node_for_response(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node["id"],
        "order_id": node["order_id"],
        "name": node["name"],
        "kind": node["kind"],
        "has_pano_images": bool(node.get("pano_images")),
        "anchored": bool(node.get("pano_images") and node.get("slam_pose")),
    }


def _migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("schema_version", payload.get("version"))
    if version is None:
        raise ValueError(
            "graph-memory store has no schema version; archive/reset it explicitly before use"
        )
    if version is not None and version != Go2GraphMemoryStore.SCHEMA_VERSION:
        raise ValueError(
            f"incompatible graph-memory schema {version}; expected {Go2GraphMemoryStore.SCHEMA_VERSION}; reset/archive the store explicitly"
        )
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, dict):
        raise ValueError("graph-memory payload nodes must be an object")
    nodes: dict[str, dict[str, Any]] = {}
    next_order_id = 1
    for node_id, raw_node in raw_nodes.items():
        if not isinstance(raw_node, dict):
            raise ValueError("graph-memory node entries must be objects")
        node = _validated_node(raw_node, require_complete=True)
        if node_id != node["id"]:
            raise ValueError(f"node key/id mismatch: {node_id}")
        order_id = raw_node.get("order_id")
        if order_id is None:
            order_id = next_order_id
        order_id = _require_order_id(order_id)
        next_order_id = max(next_order_id, order_id + 1)
        nodes[node["id"]] = _with_system_fields(
            node,
            order_id=order_id,
            slam_pose=_validated_slam_pose(raw_node.get("slam_pose")),
            navigation_goal=_validated_navigation_goal(raw_node.get("navigation_goal")),
        )
    configured_next = payload.get("next_order_id")
    if configured_next is not None:
        next_order_id = max(next_order_id, _require_order_id(configured_next))
    return {
        "schema_version": Go2GraphMemoryStore.SCHEMA_VERSION,
        "version": Go2GraphMemoryStore.SCHEMA_VERSION,
        "next_order_id": next_order_id,
        "nodes": nodes,
    }


def _next_order_id(payload: dict[str, Any]) -> int:
    return _require_order_id(payload.get("next_order_id", 1))


def _require_order_id(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("order_id must be a positive integer")
    return value


def _validated_slam_pose(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("slam_pose must be an object or null")
    expected = {"sequence", "x", "y", "yaw_rad"}
    if not expected <= set(value) or set(value) - expected - {"frame_id"}:
        raise ValueError("slam_pose must contain sequence, x, y, yaw_rad, and optional frame_id")
    sequence = value["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("slam_pose.sequence must be a non-negative integer")
    result: dict[str, Any] = {"sequence": sequence}
    for field in ("x", "y", "yaw_rad"):
        raw = value[field]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"slam_pose.{field} must be numeric")
        result[field] = float(raw)
    if "frame_id" in value:
        if not isinstance(value["frame_id"], str) or not value["frame_id"].strip():
            raise ValueError("slam_pose.frame_id must be a nonempty string")
        result["frame_id"] = value["frame_id"]
    return result


def _validated_navigation_goal(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("navigation_goal must be an array of three numbers or null")
    result: list[float] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"navigation_goal[{index}] must be numeric")
        result.append(float(raw))
    return result


def _validate_door_adjacency_note(note: str, *, source_id: str, target_id: str) -> None:
    lowered = note.lower()
    if "leads to" not in lowered:
        raise ValueError(
            f"adjacency from {source_id} to door {target_id} must say what it leads to"
        )
    if "explored" not in lowered and "unexplored" not in lowered:
        raise ValueError(
            f"adjacency from {source_id} to door {target_id} must include explored/unexplored state"
        )
    words = set(re.findall(r"[a-z]+", lowered))
    forbidden = sorted(words & _FORBIDDEN_DOOR_NOTE_WORDS)
    if forbidden:
        raise ValueError(
            f"adjacency from {source_id} to door {target_id} uses relative-only "
            f"door note words {forbidden}; use stable landmarks instead"
        )


def _bool_arg(args: dict[str, Any], name: str, default: bool) -> bool:
    value = args.get(name, default)
    return value if isinstance(value, bool) else default


def _depth(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 5
    return max(parsed, 0)


def _optional_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_id(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [str(item) for item in value]


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {str(key): str(item) for key, item in value.items()}
