"""Persistent, agent-authored memory nodes for MCP sessions."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class GraphMemoryError(ValueError):
    """A user-facing memory-node validation or persistence error."""


class GraphMemoryStore:
    """Validate and persist node-only semantic memory records."""

    VERSION = 7
    _LEGACY_VERSIONS = (1, 2, 3, 4, 5, 6)
    _MUTABLE_FIELDS = (
        "name",
        "description",
        "records",
        "pano_images",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list_nodes(self) -> list[dict[str, str]]:
        nodes = self._load_nodes()
        return [
            {
                "id": node_id,
                "kind": node["kind"],
                "name": node["name"],
                "description": node["description"],
            }
            for node_id, node in sorted(nodes.items())
        ]

    def map_nodes(self) -> list[dict[str, Any]]:
        """Return the managed fields required to annotate a SLAM map."""
        nodes = self._load_nodes()
        return [
            {
                "id": node_id,
                "order_id": node["order_id"],
                "kind": node["kind"],
                "name": node["name"],
                "slam_pose": deepcopy(node["slam_pose"]),
                "has_pano_images": bool(node["pano_images"]),
            }
            for node_id, node in sorted(nodes.items(), key=lambda item: item[1]["order_id"])
        ]

    def grep_node(self, term: str) -> list[dict[str, str]]:
        query = self._require_string(term, "term").casefold()
        nodes = self._load_nodes()
        matches: list[dict[str, str]] = []
        for node_id, node in sorted(nodes.items()):
            searchable = (
                node["name"],
                node["description"],
                *node["records"],
            )
            if any(query in value.casefold() for value in searchable):
                matches.append(
                    {
                        "id": node_id,
                        "kind": node["kind"],
                        "name": node["name"],
                        "description": node["description"],
                    }
                )
        return matches

    def inspect_node(self, node_id: str, *, return_images: bool) -> dict[str, Any]:
        node = self._get_node(node_id)
        result = self._public_node(node)
        if not return_images:
            result.pop("pano_images", None)
        return result

    def report_memory_pos(self, node_id: str) -> dict[str, Any]:
        node_id = self._require_string(node_id, "id")
        nodes = self._load_nodes()
        if node_id not in nodes:
            raise GraphMemoryError(f"unknown node: {node_id}")
        return {"current_node": self._public_node(nodes[node_id])}

    def query_node_context(
        self,
        *,
        terms: Any,
        inspect_ids: Any,
        return_images: bool,
    ) -> dict[str, Any]:
        """Read search results and node details from one consistent snapshot."""
        if not isinstance(terms, list) or any(
            not isinstance(term, str) or not term.strip() for term in terms
        ):
            raise GraphMemoryError("terms must be an array of non-empty strings")
        if not isinstance(inspect_ids, list) or any(
            not isinstance(node_id, str) or not node_id.strip() for node_id in inspect_ids
        ):
            raise GraphMemoryError("inspect_ids must be an array of non-empty strings")
        if not isinstance(return_images, bool):
            raise GraphMemoryError("return_images must be a boolean")

        nodes = self._load_nodes()
        matches = {
            term: self._grep_nodes_in(nodes, term)
            for term in dict.fromkeys(terms)
        }
        inspected: dict[str, dict[str, Any]] = {}
        for node_id in dict.fromkeys(inspect_ids):
            if node_id not in nodes:
                raise GraphMemoryError(f"unknown node: {node_id}")
            node = self._public_node(nodes[node_id])
            if not return_images:
                node.pop("pano_images", None)
            inspected[node_id] = node

        return {
            "matches": matches,
            "inspected_nodes": inspected,
        }

    def add_node(
        self,
        raw_node: Mapping[str, Any],
        *,
        slam_pose: Mapping[str, Any] | None = None,
    ) -> None:
        node = self._validate_node(raw_node, require_id=True)
        nodes, next_order_id = self._load_document()
        node_id = node["id"]
        if node_id in nodes:
            raise GraphMemoryError(f"node already exists: {node_id}")
        nodes[node_id] = self._with_system_fields(
            node, next_order_id, slam_pose
        )
        self._save_nodes(nodes, next_order_id + 1)

    def update_node(
        self,
        node_id: str,
        raw_update: Mapping[str, Any],
        *,
        slam_pose: Mapping[str, Any] | None = None,
    ) -> None:
        node_id = self._require_string(node_id, "id")
        nodes, next_order_id = self._load_document()
        if node_id not in nodes:
            raise GraphMemoryError(f"unknown node: {node_id}")
        update = self._validate_node_patch(raw_update)
        nodes[node_id] = self._apply_update(
            nodes[node_id], update, slam_pose
        )
        self._save_nodes(nodes, next_order_id)

    def upsert_node(
        self,
        raw_node: Mapping[str, Any],
        *,
        slam_pose: Mapping[str, Any] | None = None,
    ) -> str:
        nodes, next_order_id = self._load_document()
        status, next_order_id = self._upsert_node_in(
            nodes, raw_node, next_order_id, slam_pose
        )
        self._save_nodes(nodes, next_order_id)
        return status

    def apply_mutations(
        self,
        raw_mutations: Any,
        *,
        slam_pose: Mapping[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Apply node creates, upserts, and updates atomically."""
        if not isinstance(raw_mutations, list) or not raw_mutations:
            raise GraphMemoryError("mutations must be a non-empty array")
        nodes, next_order_id = self._load_document()
        results: list[dict[str, str]] = []
        for index, raw_mutation in enumerate(raw_mutations):
            if not isinstance(raw_mutation, Mapping):
                raise GraphMemoryError(f"mutations[{index}] must be an object")
            operation = self._require_string(raw_mutation.get("op"), f"mutations[{index}].op")
            if operation == "create":
                if set(raw_mutation) != {"op", "node"}:
                    raise GraphMemoryError(
                        f"mutations[{index}] create requires only op and node"
                    )
                node = self._validate_node(raw_mutation["node"], require_id=True)
                node_id = node["id"]
                if node_id in nodes:
                    raise GraphMemoryError(f"node already exists: {node_id}")
                nodes[node_id] = self._with_system_fields(
                    node, next_order_id, slam_pose
                )
                next_order_id += 1
                status = "node_added"
            elif operation == "upsert":
                if set(raw_mutation) != {"op", "node"}:
                    raise GraphMemoryError(
                        f"mutations[{index}] upsert requires only op and node"
                    )
                status, next_order_id = self._upsert_node_in(
                    nodes,
                    raw_mutation["node"],
                    next_order_id,
                    slam_pose,
                )
                node_id = self._require_string(raw_mutation["node"].get("id"), "id")
            elif operation == "update":
                if set(raw_mutation) != {"op", "id", "node"}:
                    raise GraphMemoryError(
                        f"mutations[{index}] update requires only op, id, and node"
                    )
                node_id = self._require_string(raw_mutation["id"], "id")
                if node_id not in nodes:
                    raise GraphMemoryError(f"unknown node: {node_id}")
                update = self._validate_node_patch(raw_mutation["node"])
                nodes[node_id] = self._apply_update(
                    nodes[node_id], update, slam_pose
                )
                status = "node_updated"
            else:
                raise GraphMemoryError(
                    f"mutations[{index}].op must be create, upsert, or update"
                )
            results.append({"index": str(index), "status": status, "id": node_id})

        self._save_nodes(nodes, next_order_id)
        return results

    def delete_node(self, node_id: str) -> None:
        node_id = self._require_string(node_id, "id")
        nodes, next_order_id = self._load_document()
        if node_id not in nodes:
            raise GraphMemoryError(f"unknown node: {node_id}")
        del nodes[node_id]
        self._save_nodes(nodes, next_order_id)

    def _upsert_node_in(
        self,
        nodes: dict[str, dict[str, Any]],
        raw_node: Mapping[str, Any],
        next_order_id: int,
        slam_pose: Mapping[str, Any] | None,
    ) -> tuple[str, int]:
        if not isinstance(raw_node, Mapping):
            raise GraphMemoryError("node must be an object")
        node_id = self._require_string(raw_node.get("id"), "id")
        if node_id in nodes:
            patch = self._validate_node_patch(
                {key: value for key, value in raw_node.items() if key != "id"},
                allow_empty=True,
            )
            nodes[node_id] = self._apply_update(
                nodes[node_id], patch, slam_pose
            )
            return "node_updated", next_order_id
        if "name" not in raw_node or "description" not in raw_node:
            raise GraphMemoryError("new nodes require name and description")
        candidate = {
            "id": node_id,
            "name": raw_node.get("name"),
            "description": raw_node.get("description"),
            "records": raw_node.get("records", []),
            "pano_images": raw_node.get("pano_images", {}),
        }
        if "kind" in raw_node:
            candidate["kind"] = raw_node["kind"]
        unexpected = set(raw_node) - set(candidate)
        if unexpected:
            raise GraphMemoryError(
                "invalid node fields (unexpected fields: "
                + ", ".join(sorted(unexpected))
                + ")"
            )
        nodes[node_id] = self._with_system_fields(
            self._validate_node(candidate, require_id=True),
            next_order_id,
            slam_pose,
        )
        return "node_added", next_order_id + 1

    @staticmethod
    def _public_node(node: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(node))
        result.pop("navigation_goal", None)  # Defensive for in-memory legacy records.
        return result

    def _grep_nodes_in(
        self, nodes: Mapping[str, Mapping[str, Any]], term: str
    ) -> list[dict[str, str]]:
        query = self._require_string(term, "term").casefold()
        matches: list[dict[str, str]] = []
        for node_id, node in sorted(nodes.items()):
            searchable = (
                node["name"],
                node["description"],
                *node["records"],
            )
            if any(query in value.casefold() for value in searchable):
                matches.append(
                    {
                        "id": node_id,
                        "kind": node["kind"],
                        "name": node["name"],
                        "description": node["description"],
                    }
                )
        return matches

    def _get_node(self, node_id: str) -> dict[str, Any]:
        node_id = self._require_string(node_id, "id")
        nodes = self._load_nodes()
        if node_id not in nodes:
            raise GraphMemoryError(f"unknown node: {node_id}")
        return nodes[node_id]

    def _load_nodes(self) -> dict[str, dict[str, Any]]:
        return self._load_document()[0]

    def _load_document(self) -> tuple[dict[str, dict[str, Any]], int]:
        if not self.path.exists():
            return {}, 1
        try:
            with self.path.open(encoding="utf-8") as handle:
                document = json.load(handle)
        except OSError as exc:
            raise GraphMemoryError(f"could not read memory nodes: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GraphMemoryError(f"memory-node store is not valid JSON: {exc.msg}") from exc
        if not isinstance(document, dict):
            raise GraphMemoryError("memory-node store has an unsupported document version")
        version = document.get("version")
        if version in self._LEGACY_VERSIONS:
            document = self._migrate_legacy_document(document)
        elif version != self.VERSION:
            raise GraphMemoryError("memory-node store has an unsupported document version")
        raw_nodes = document.get("nodes")
        if not isinstance(raw_nodes, dict):
            raise GraphMemoryError("memory-node document must contain a nodes object")
        nodes: dict[str, dict[str, Any]] = {}
        for node_id, raw_node in raw_nodes.items():
            if not isinstance(node_id, str) or not isinstance(raw_node, dict):
                raise GraphMemoryError("memory-node store contains an invalid node entry")
            node = self._validate_node(raw_node, require_id=True, stored=True)
            if node["id"] != node_id:
                raise GraphMemoryError(f"memory-node key does not match id: {node_id}")
            nodes[node_id] = node
        next_order_id = document.get("next_order_id")
        if (
            not isinstance(next_order_id, int)
            or isinstance(next_order_id, bool)
            or next_order_id < 1
        ):
            raise GraphMemoryError("memory-node document has an invalid next_order_id")
        return nodes, next_order_id

    def _save_nodes(self, nodes: Mapping[str, Mapping[str, Any]], next_order_id: int) -> None:
        document = {"version": self.VERSION, "next_order_id": next_order_id, "nodes": nodes}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(document, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.path)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        except OSError as exc:
            raise GraphMemoryError(f"could not save memory nodes: {exc}") from exc

    def _validate_node(
        self, raw_node: Mapping[str, Any], *, require_id: bool, stored: bool = False
    ) -> dict[str, Any]:
        if not isinstance(raw_node, Mapping):
            raise GraphMemoryError("node must be an object")
        kind = self._require_kind(raw_node.get("kind"))
        expected = set(self._MUTABLE_FIELDS)
        expected.add("kind")
        if require_id:
            expected.add("id")
        if stored:
            expected.update(("order_id", "slam_pose"))
        actual = set(raw_node)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise GraphMemoryError("invalid node fields (" + "; ".join(details) + ")")
        node: dict[str, Any] = {
            "kind": kind,
            "name": self._require_string(raw_node["name"], "name"),
            "description": self._require_string(raw_node["description"], "description"),
            "records": self._require_string_list(raw_node["records"], "records"),
            "pano_images": self._require_image_object(raw_node["pano_images"]),
        }
        if require_id:
            node = {"id": self._require_string(raw_node["id"], "id"), **node}
        if stored:
            node = {
                "order_id": self._require_order_id(raw_node["order_id"]),
                "slam_pose": self._require_slam_pose(raw_node["slam_pose"]),
                **node,
            }
        return node

    def _with_system_fields(
        self,
        node: Mapping[str, Any],
        order_id: int,
        slam_pose: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "order_id": order_id,
            "slam_pose": self._require_slam_pose(slam_pose),
            **node,
        }

    def _apply_update(
        self,
        node: Mapping[str, Any],
        update: Mapping[str, Any],
        slam_pose: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        result = {**node, **update, "id": node["id"]}
        if "pano_images" in update and slam_pose is not None:
            result["slam_pose"] = self._require_slam_pose(slam_pose)
        return result

    def _validate_node_patch(
        self, raw_update: Mapping[str, Any], *, allow_empty: bool = False
    ) -> dict[str, Any]:
        if not isinstance(raw_update, Mapping):
            raise GraphMemoryError("node update must be an object")
        expected = set(self._MUTABLE_FIELDS)
        actual = set(raw_update)
        unexpected = sorted(actual - expected)
        if unexpected:
            raise GraphMemoryError(
                "invalid node fields (unexpected fields: " + ", ".join(unexpected) + ")"
            )
        if not actual and not allow_empty:
            raise GraphMemoryError("node update must include at least one mutable field")
        update: dict[str, Any] = {}
        if "name" in raw_update:
            update["name"] = self._require_string(raw_update["name"], "name")
        if "description" in raw_update:
            update["description"] = self._require_string(
                raw_update["description"], "description"
            )
        if "records" in raw_update:
            update["records"] = self._require_string_list(raw_update["records"], "records")
        if "pano_images" in raw_update:
            update["pano_images"] = self._require_image_object(raw_update["pano_images"])
        return update

    @staticmethod
    def _require_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise GraphMemoryError(f"{field} must be a non-empty string")
        return value

    @staticmethod
    def _require_string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise GraphMemoryError(f"{field} must be an array of strings")
        return list(value)

    @staticmethod
    def _require_kind(value: Any) -> str:
        if value not in {"area", "door"}:
            raise GraphMemoryError("kind must be 'area' or 'door'")
        return value

    @staticmethod
    def _require_image_object(value: Any) -> dict[str, str]:
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and key.strip() and isinstance(item, str)
            for key, item in value.items()
        ):
            raise GraphMemoryError("pano_images must be an object of direction strings")
        return dict(value)

    @staticmethod
    def _require_order_id(value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise GraphMemoryError("order_id must be a positive integer")
        return value

    @staticmethod
    def _require_slam_pose(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GraphMemoryError("slam_pose must be an object or null")
        expected = {"sequence", "x", "y", "yaw_rad"}
        if set(value) != expected:
            raise GraphMemoryError("slam_pose must contain sequence, x, y, and yaw_rad")
        sequence = value["sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise GraphMemoryError("slam_pose.sequence must be a non-negative integer")
        result: dict[str, Any] = {"sequence": sequence}
        for field in ("x", "y", "yaw_rad"):
            raw = value[field]
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise GraphMemoryError(f"slam_pose.{field} must be numeric")
            result[field] = float(raw)
        return result

    def _migrate_legacy_document(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Upgrade legacy nodes and drop obsolete adjacency/path fields."""
        raw_nodes = document.get("nodes")
        if not isinstance(raw_nodes, dict):
            raise GraphMemoryError("memory-node document must contain a nodes object")
        migrated: dict[str, dict[str, Any]] = {}
        for node_id, raw_node in raw_nodes.items():
            if not isinstance(node_id, str) or not isinstance(raw_node, dict):
                raise GraphMemoryError("memory-node store contains an invalid node entry")
            kind = raw_node.get("kind")
            if kind not in {"area", "door"}:
                description = raw_node.get("description")
                if not isinstance(description, str):
                    raise GraphMemoryError(
                        f"cannot migrate legacy node without description: {node_id}"
                    )
                if description.startswith("Area node."):
                    kind = "area"
                elif description.startswith("Door node."):
                    kind = "door"
                else:
                    raise GraphMemoryError(
                        f"cannot migrate legacy node with ambiguous kind: {node_id}"
                    )
            migrated[node_id] = {
                key: value
                for key, value in raw_node.items()
                if key not in {"waypoints", "adjacents", "navigation_goal"}
            }
            migrated[node_id]["kind"] = kind
        next_order_id = 1
        for node in migrated.values():
            order_id = node.get("order_id")
            if not isinstance(order_id, int) or isinstance(order_id, bool) or order_id < 1:
                order_id = next_order_id
            node["order_id"] = order_id
            if "slam_pose" not in node:
                node["slam_pose"] = None
            next_order_id = max(next_order_id, order_id + 1)
        saved_next_order_id = document.get("next_order_id")
        if isinstance(saved_next_order_id, int) and not isinstance(saved_next_order_id, bool):
            next_order_id = max(next_order_id, saved_next_order_id)
        return {"version": self.VERSION, "next_order_id": next_order_id, "nodes": migrated}
