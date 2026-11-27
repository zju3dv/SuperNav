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

from __future__ import annotations

import asyncio
import base64
import json
from json import dumps as json_dumps
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image as PILImage
import requests
from starlette.responses import StreamingResponse

from dimos.agents.mcp.external_harness_authority_proxy import (
    ExternalHarnessAuthorityProxy,
    MovementAuthorityEnvelope,
    _handle_proxy_request,
    _overlay_graph_badges_on_image_block,
    _slam_pose_from_map_metadata,
    _sanitize_agent_navigation_result,
    create_authority_proxy_app,
)


class FakeMcpAdapter:
    url = "memory://fake-dimos"

    def __init__(
        self,
        *,
        battery_percent: float | None = None,
        include_required_tools: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
        self.tools = [
            {"name": "observe", "inputSchema": {"type": "object"}},
            {
                "name": "relative_move",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "rotate_in_place",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "full_lie_down",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "lie_down",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "stand_up",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "balance_stand",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/uses": ["movement"], "dimos/lifecycle": "instant"},
            },
            {
                "name": "stop_movement",
                "inputSchema": {"type": "object"},
                "_meta": {"dimos/lifecycle": "instant"},
            },
            {"name": "reset_pixel_goal_map", "inputSchema": {"type": "object"}},
        ]
        if battery_percent is not None or include_required_tools:
            self.tools.extend(
                [
                    {"name": "go2_robot_status", "inputSchema": {"type": "object"}},
                    {"name": "motion_safety_status", "inputSchema": {"type": "object"}},
                    {"name": "go2_situation", "inputSchema": {"type": "object"}},
                    {"name": "last_motion_diagnostics", "inputSchema": {"type": "object"}},
                    {
                        "name": "go2_save_relocalization_bundle",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "go2_start_navigation_episode",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "go2_slam_map_snapshot",
                        "inputSchema": {"type": "object"},
                    },
                ]
            )
        self.battery_percent = battery_percent
        self.observe_result = {
            "content": [
                {
                    "type": "image",
                    "mimeType": "image/jpeg",
                    "data": "ZmFrZS1qcGVn",
                }
            ]
        }

    def initialize(self) -> dict[str, Any]:
        return {"serverInfo": {"name": "fake-dimos"}}

    def list_tools(self) -> list[dict[str, Any]]:
        return self.tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((name, arguments or {}, meta))
        if name == "observe":
            return self.observe_result
        if name == "go2_robot_status":
            payload = {
                "success": True,
                "metadata": {
                    "robot_status": {
                        "battery": {
                            "state": "normal",
                            "soc_percent": self.battery_percent,
                            "source": "unitree_lowstate",
                        },
                        "motors": {
                            "state": "normal",
                            "lost_count": 0,
                            "max_temperature_c": 39.0,
                            "source": "unitree_lowstate",
                        },
                    }
                },
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                    }
                ]
            }
        if name == "go2_situation":
            payload = {
                "success": True,
                "metadata": {
                    "situation": {
                        "battery": {"state": "unknown", "soc_percent": None},
                        "motor": {"state": "unknown", "lost_count": None},
                        "authority": {"tool_profile": {"name": "vision_odom_local_motion"}},
                    }
                },
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                    }
                ]
            }
        if name == "relative_move":
            payload = {
                "success": True,
                "metadata": {
                    "2d_slam_map": {
                        "status": "current",
                        "source": "global_costmap",
                        "map_frame": "world",
                        "resolution_m_per_cell": 1.0,
                        "origin_xy": [0.0, 0.0],
                        "grid_size_cells": [20, 20],
                        "crop_grid_bounds": {
                            "x_min": 0,
                            "y_min": 0,
                            "x_max": 20,
                            "y_max": 20,
                        },
                        "width_px": 80,
                        "height_px": 80,
                        "pose": {
                            "frame_id": "world",
                            "position_m": {"x": 10.0, "y": 10.0, "z": 0.0},
                            "yaw_rad": 0.0,
                        },
                        "pose_sequence": 7,
                    }
                },
            }
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    _fake_png_block(),
                ]
            }
        if name == "go2_save_relocalization_bundle":
            payload = {
                "success": True,
                "metadata": {
                    "relocalization_bundle": {
                        "status": "current",
                        "bundle_path": "/tmp/go2/slam/relocalization",
                        "manifest_path": "/tmp/go2/slam/relocalization/manifest.json",
                        "global_map_file": "/tmp/go2/slam/relocalization/global_map.pc2.lcm",
                        "map_available": True,
                        "point_count": 456,
                        "frame_id": "world",
                        "saved_at_epoch_s": 1234.0,
                        "pose": {"x": 1.0, "y": 2.0, "yaw_rad": 3.0},
                    }
                },
            }
            return {"content": [{"type": "text", "text": json.dumps(payload)}]}
        if name == "go2_report_relocalization_prior":
            payload = {
                "success": True,
                "metadata": {
                    "relocalization_prior": {
                        "status": "reported",
                        "node_id": (arguments or {}).get("node_id"),
                        "slam_pose": (arguments or {}).get("slam_pose"),
                    },
                    "relocalization_wait": {
                        "status": "localized",
                        "localized": True,
                        "wait_window_s": 10.0,
                    },
                    "2d_slam_map": {
                        "status": "current",
                        "source": "native_global_map_area_segmented",
                        "map_frame": "world",
                        "resolution_m_per_cell": 1.0,
                        "origin_xy": [0.0, 0.0],
                        "grid_size_cells": [20, 20],
                        "crop_grid_bounds": {
                            "x_min": 0,
                            "y_min": 0,
                            "x_max": 20,
                            "y_max": 20,
                        },
                        "width_px": 80,
                        "height_px": 80,
                        "pose": {
                            "frame_id": "world",
                            "position_m": {"x": 10.0, "y": 10.0, "z": 0.0},
                            "yaw_rad": 0.0,
                        },
                        "pose_sequence": 7,
                    },
                },
            }
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    _fake_png_block(),
                ]
            }
        if name == "go2_slam_map_snapshot":
            payload = {
                "success": True,
                "metadata": {
                    "relocalization_wait": {
                        "status": "localized",
                        "localized": True,
                    },
                    "2d_slam_map": {
                        "status": "current",
                        "source": "native_global_map_area_segmented",
                        "map_frame": "world",
                        "resolution_m_per_cell": 1.0,
                        "origin_xy": [0.0, 0.0],
                        "grid_size_cells": [20, 20],
                        "crop_grid_bounds": {
                            "x_min": 0,
                            "y_min": 0,
                            "x_max": 20,
                            "y_max": 20,
                        },
                        "width_px": 80,
                        "height_px": 80,
                        "pose": {
                            "frame_id": "world",
                            "position_m": {"x": 10.0, "y": 10.0, "z": 0.0},
                            "yaw_rad": 0.0,
                        },
                        "pose_sequence": 7,
                    },
                },
            }
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    _fake_png_block(),
                ]
            }
        if name == "go2_start_navigation_episode":
            payload = {
                "success": True,
                "metadata": {
                    "navigation_episode": {
                        "status": "started",
                        "clear_slam_map": bool(arguments.get("clear_slam_map", True))
                        if arguments
                        else True,
                        "reset_signal_path": "/tmp/dimos/slam/voxel_map_reset_signal.json",
                    }
                },
            }
            return {"content": [{"type": "text", "text": json.dumps(payload)}]}
        return {"content": [{"type": "text", "text": f"called {name}"}]}

    def call_tool_text(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> str:
        return self.call_tool(name, arguments, meta=meta)["content"][0]["text"]


class TimeoutMovementMcpAdapter(FakeMcpAdapter):
    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((name, arguments or {}, meta))
        if name == "relative_move":
            raise requests.ReadTimeout("planner completion timed out")
        return super().call_tool(name, arguments, meta=meta)


class PixelGoalMcpAdapter(FakeMcpAdapter):
    def __init__(self, *, terminal_images: bool = True, terminal: bool = True) -> None:
        super().__init__()
        self.terminal_images = terminal_images
        self.terminal = terminal
        self.tools.extend(
            [
                {
                    "name": "pixelgoal_navigate",
                    "description": "Start a non-blocking hop.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "point": {"type": "array", "items": {"type": "number"}},
                            "direction": {"type": "string"},
                            "goal_id": {"type": "string"},
                            "frame_ts": {"type": "number"},
                            "frame_jpeg_b64": {"type": "string"},
                        },
                    },
                    "_meta": {"dimos/uses": ["movement"]},
                },
                {"name": "pixelgoal_status", "inputSchema": {"type": "object"}},
                {"name": "pixelgoal_stop", "inputSchema": {"type": "object"}},
                {
                    "name": "observe_surround",
                    "description": "Call before every navigation action.",
                    "inputSchema": {"type": "object"},
                },
                {"name": "server_status", "inputSchema": {"type": "object"}},
            ]
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((name, arguments or {}, meta))
        if name == "pixelgoal_navigate":
            return _text_content({"ok": True, "status": "started", "goal_id": "goal-1"})
        if name == "pixelgoal_status":
            payload = {
                "ok": True,
                "hop_running": not self.terminal,
                "status": {"goal_id": "goal-1", "status": "reached"},
                "last_result": {"goal_id": "goal-1", "status": "reached"},
                "events": [
                    {
                        "ts": 1.0,
                        "type": "hop_start",
                        "data": {"goal_id": "old-goal"},
                    },
                    {
                        "ts": 2.0,
                        "type": "hop_end",
                        "data": {"goal_id": "old-goal"},
                    },
                    {
                        "ts": 3.0,
                        "type": "hop_start",
                        "data": {"goal_id": "goal-1"},
                    },
                    {"ts": 4.0, "type": "ray_projected", "data": {}},
                    {
                        "ts": 5.0,
                        "type": "hop_end",
                        "data": {"goal_id": "goal-1"},
                    },
                ],
            }
            result = _text_content(payload)
            if arguments and arguments.get("include_images") and self.terminal_images:
                result["content"].extend(_four_view_content())
            return result
        if name == "observe_surround":
            return {
                "content": [
                    {"type": "text", "text": json.dumps({"ok": True})},
                    *_four_view_content(),
                ]
            }
        if name == "pixelgoal_stop":
            return _text_content({"ok": True, "status": "cancel_requested"})
        if name == "server_status":
            return _text_content(
                {
                    "agent_runtime_policy": {
                        "allowed_movement_tools": ["drive", "pixelgoal_navigate"]
                    },
                    "go2_harness_profile": {
                        "name": "vision_odom_local_motion",
                        "description": "No maps.",
                        "allowed_strategy": "local motion",
                        "disallowed_strategy": "map-backed navigation",
                    }
                }
            )
        return super().call_tool(name, arguments, meta=meta)


def _text_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _four_view_content() -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for direction in ("front", "right", "back", "left"):
        content.extend([{"type": "text", "text": direction}, _fake_png_block()])
    return content


def _fake_png_block() -> dict[str, Any]:
    image = PILImage.new("RGB", (80, 80), (255, 255, 255))
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {
        "type": "image",
        "mimeType": "image/png",
        "width_px": 80,
        "height_px": 80,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def test_graph_overlay_suppresses_area_label_inside_region() -> None:
    image_block = _fake_png_block()
    slam_map = {
        "origin_xy": [0.0, 0.0],
        "resolution_m_per_cell": 1.0,
        "grid_size_cells": [20, 20],
        "width_px": 80,
        "height_px": 80,
        "slam_area_nodes": [
            {
                "label": "A1",
                "target_slam_pose": {"x": 4.0, "y": 4.0},
                "bbox_grid": [2, 2, 8, 8],
            },
            {
                "label": "A2",
                "target_slam_pose": {"x": 12.0, "y": 12.0},
                "footprint_radius_m": 0.2,
            },
        ],
    }
    displayed = _overlay_graph_badges_on_image_block(
        image_block,
        slam_map,
        [
            {
                "id": "office",
                "order_id": 1,
                "kind": "area",
                "name": "office",
                "has_pano_images": True,
                "slam_pose": {"x": 5.0, "y": 4.0},
            }
        ],
    )

    assert displayed == [{"order_id": 1, "id": "office", "name": "office", "kind": "area"}]
    assert image_block["metadata"]["suppressed_slam_area_labels"] == ["A1"]


def test_agent_navigation_result_hides_internal_slam_geometry_without_changing_image() -> None:
    image_block = _fake_png_block()
    image_data = image_block["data"]
    result = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "success": True,
                        "metadata": {
                            "2d_slam_map": {
                                "status": "current",
                                "source": "global_costmap",
                                "map_frame": "world",
                                "grid_ts": 123.0,
                                "slam_area_memory_names": [{"A1": "hall"}],
                                "slam_area_nodes": [
                                    {
                                        "id": "slam_area_1",
                                        "label": "A1",
                                        "area_id": 1,
                                        "pixel_count": 456,
                                        "centroid_grid": [64.0, 113.5],
                                        "target_grid": [65.0, 113.0],
                                        "bbox_grid": [19, 55, 110, 173],
                                        "target_slam_pose": {
                                            "x": 1.0,
                                            "y": 2.0,
                                            "yaw_rad": 0.0,
                                        },
                                        "map_frame": "world",
                                        "grid_ts": 123.0,
                                    }
                                ],
                            }
                        },
                    }
                ),
            },
            image_block,
        ]
    }

    proxy = ExternalHarnessAuthorityProxy(FakeMcpAdapter())
    proxy._cache_navigation_context_from_result(result)
    sanitized = _sanitize_agent_navigation_result(result)
    payload = json.loads(sanitized["content"][0]["text"])
    slam_map = payload["metadata"]["2d_slam_map"]

    assert proxy._latest_slam_map_metadata is not None
    assert proxy._latest_slam_map_metadata["slam_area_nodes"][0][
        "target_slam_pose"
    ] == {"x": 1.0, "y": 2.0, "yaw_rad": 0.0}
    assert slam_map == {
        "status": "current",
        "source": "global_costmap",
        "slam_area_nodes": [
            {
                "id": "slam_area_1",
                "label": "A1",
                "area_id": 1,
                "pixel_count": 456,
            }
        ],
    }
    assert sanitized["content"][1] is image_block
    assert sanitized["content"][1]["data"] == image_data


def test_graph_overlay_uses_grid_row_without_vertical_mirroring() -> None:
    image_block = _fake_png_block()
    slam_map = {
        "origin_xy": [0.0, 0.0],
        "resolution_m_per_cell": 1.0,
        "grid_size_cells": [20, 20],
        "width_px": 80,
        "height_px": 80,
        "slam_area_nodes": [
            {
                "label": "A7",
                "target_slam_pose": {"x": 4.0, "y": 15.0},
                "bbox_grid": [12, 2, 18, 8],
            },
        ],
    }
    _overlay_graph_badges_on_image_block(
        image_block,
        slam_map,
        [
            {
                "id": "upper-area",
                "order_id": 2,
                "kind": "area",
                "name": "upper area",
                "has_pano_images": True,
                "slam_pose": {"x": 4.0, "y": 15.0},
            }
        ],
    )

    assert image_block["metadata"]["suppressed_slam_area_labels"] == ["A7"]


def test_proxy_forwards_observe_image_blocks_without_shape_change() -> None:
    upstream = FakeMcpAdapter()
    proxy = ExternalHarnessAuthorityProxy(upstream)

    result = proxy.call_tool("observe", {"save": False}, meta={"progressToken": "read-1"})

    assert result == upstream.observe_result
    assert result["content"][0] == {
        "type": "image",
        "mimeType": "image/jpeg",
        "data": "ZmFrZS1qcGVn",
    }
    assert upstream.calls == [("observe", {"save": False}, {"progressToken": "read-1"})]


def test_proxy_refuses_movement_without_operator_envelope() -> None:
    upstream = FakeMcpAdapter()
    proxy = ExternalHarnessAuthorityProxy(upstream)

    result = proxy.call_tool("relative_move", {"forward": 0.5})

    assert "no active external-harness authority envelope" in result["content"][0]["text"]
    assert upstream.calls == []


def test_proxy_hides_movement_tools_without_operator_envelope() -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    proxy = ExternalHarnessAuthorityProxy(upstream)

    tool_names = {tool["name"] for tool in proxy.list_tools()}

    assert "observe" in tool_names
    assert "go2_situation" in tool_names
    assert "stop_movement" not in tool_names
    assert "relative_move" not in tool_names
    assert "rotate_in_place" not in tool_names
    assert "stand_up" not in tool_names
    assert "reset_pixel_goal_map" not in tool_names

    result = proxy.call_tool("reset_pixel_goal_map")
    assert "operator-only" in result["content"][0]["text"]
    assert upstream.calls == []


def test_proxy_exposes_and_persists_run_local_graph_memory_tools(tmp_path: Path) -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    store_path = tmp_path / "go2-graph-memory.json"
    proxy = ExternalHarnessAuthorityProxy(upstream, graph_memory_path=store_path)

    tool_names = {tool["name"] for tool in proxy.list_tools()}

    assert "go2_apply_graph_mutations" in tool_names
    result = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "hall-area",
                        "kind": "area",
                        "name": "hall area",
                        "description": "White hallway with a cabinet.",
                        "adjacents": [
                            {
                                "name": "kitchen doorway",
                                "id": "kitchen-door",
                                "note": (
                                    "Doorway beside the cabinet leads to the kitchen; "
                                    "unexplored."
                                ),
                            }
                        ],
                        "records": ["No kitchen table visible yet."],
                        "pano_images": {"front": "artifact:observe-1:front"},
                        "waypoints": {"cabinet": "Viewpoint beside the cabinet."},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "kitchen-door",
                        "kind": "door",
                        "name": "kitchen doorway",
                        "description": "Doorway between the hall and kitchen.",
                        "adjacents": [
                            {
                                "name": "hall area",
                                "id": "hall-area",
                                "note": "Hall side has a white cabinet and bright floor; explored.",
                            }
                        ],
                        "records": [],
                        "pano_images": {},
                    },
                },
            ]
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is True
    assert store_path.exists()
    assert set(payload["affected_nodes"]) == {"hall-area", "kitchen-door"}

    inspect_result = proxy.call_tool(
        "go2_inspect_graph_node",
        {"id": "hall-area", "return_images": True},
    )
    inspected = json.loads(inspect_result["content"][0]["text"])
    assert inspected["node"]["pano_images"]["front"] == "artifact:observe-1:front"
    assert inspected["node"]["order_id"] == 1
    assert "slam_pose" not in inspected["node"]


def test_route_query_keeps_fresh_map_node_navigable_without_relocalization(tmp_path: Path) -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    proxy = ExternalHarnessAuthorityProxy(upstream, graph_memory_path=tmp_path / "graph.json")
    proxy._cache_navigation_context_from_result(upstream.call_tool("relative_move"))
    created = proxy.call_tool("go2_apply_graph_mutations", {"mutations": [{
        "op": "create", "node": {
            "id": "hall", "kind": "area", "name": "hall",
            "description": "Hallway with a cabinet.",
            "records": [], "pano_images": {"front": "artifact:front"}, "waypoints": {},
        },
    }]})
    assert json.loads(created["content"][0]["text"])["success"] is True
    assert proxy.graph_memory is not None
    assert proxy.graph_memory.navigation_target("hall")["frame_id"] == "world"

    queried = proxy.call_tool("go2_query_route_context", {"return_images": False})
    assert json.loads(queried["content"][0]["text"])["2d_slam_map"]["status"] == "current"
    target, error = proxy._resolve_memory_navigation_target("hall")
    assert error is None
    assert target is not None
    assert target["map_frame"] == target["target_slam_pose"]["frame_id"] == "world"

    proxy._latest_slam_map_metadata = {"status": "static", "map_frame": "world"}
    target, error = proxy._resolve_memory_navigation_target("hall")
    assert target is None
    assert "current SLAM map unavailable" in error


def test_pixelgoal_terminal_map_pose_anchors_new_memory_node(tmp_path: Path) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True), graph_memory_path=tmp_path / "graph.json"
    )
    proxy._cache_navigation_context_from_result({"content": [{"type": "text", "text": json.dumps({
        "2d_slam_map": {
            "status": "current", "source": "native_global_map_area_segmented",
            "map_frame": "world",
            "pose": {
                "frame_id": "world", "position_m": {"x": 2.0, "y": 3.0, "z": 0.3},
                "yaw_rad": 0.5,
            },
        },
    })}]})
    result = proxy.call_tool("go2_apply_graph_mutations", {"mutations": [{
        "op": "create", "node": {
            "id": "hall", "kind": "area", "name": "hall",
            "description": "Hallway with cabinet.",
            "records": [], "pano_images": {"front": "artifact:front"}, "waypoints": {},
        },
    }]})

    assert json.loads(result["content"][0]["text"])["success"] is True
    assert proxy.graph_memory is not None
    assert proxy.graph_memory.navigation_target("hall") == {
        "sequence": 0, "x": 2.0, "y": 3.0, "yaw_rad": 0.5, "frame_id": "world",
    }


def test_memory_navigation_rejects_legacy_anchor_without_frame(tmp_path: Path) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True), graph_memory_path=tmp_path / "graph.json"
    )
    proxy._latest_slam_map_metadata = {
        "status": "current", "map_frame": "world", "pose": {"frame_id": "world"},
    }
    assert proxy.graph_memory is not None
    proxy.graph_memory.navigation_target = lambda _node_id: {
        "sequence": 1, "x": 1.0, "y": 2.0, "yaw_rad": 0.0,
    }
    target, error = proxy._resolve_memory_navigation_target("hall")
    assert target is None
    assert "legacy nodes without a frame" in error


def test_map_crop_cannot_be_used_as_robot_pose_anchor() -> None:
    assert _slam_pose_from_map_metadata({
        "status": "current", "map_frame": "world", "origin_xy": [0, 0],
        "crop_grid_bounds": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10},
        "resolution_m_per_cell": 0.1,
    }) is None
    assert _slam_pose_from_map_metadata({
        "status": "current", "map_frame": "world",
        "pose": {"frame_id": "sport_mode_odom", "position_m": {"x": 1, "y": 2}, "yaw_rad": 0},
    }) is None

def test_proxy_graph_memory_anchors_pano_nodes_from_latest_slam_pose(
    tmp_path: Path,
) -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the kitchen",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )
    store_path = tmp_path / "go2-graph-memory.json"
    proxy = ExternalHarnessAuthorityProxy(
        upstream,
        envelope=envelope,
        graph_memory_path=store_path,
    )

    move_result = proxy.call_tool("relative_move", {"forward": 0.4})
    move_payload = json.loads(move_result["content"][0]["text"])
    assert move_payload["metadata"]["2d_slam_map"]["status"] == "current"

    result = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "hall",
                        "kind": "area",
                        "name": "hall",
                        "description": "Hall area.",
                        "adjacents": [],
                        "records": [],
                        "pano_images": {"front": "artifact:observe-1:front"},
                        "waypoints": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "door",
                        "kind": "door",
                        "name": "door",
                        "description": "Door without panorama.",
                        "adjacents": [],
                        "records": [],
                        "pano_images": {},
                    },
                },
            ]
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is True
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == proxy.graph_memory.SCHEMA_VERSION
    assert stored["nodes"]["hall"]["order_id"] == 1
    assert stored["nodes"]["hall"]["slam_pose"] == {
        "sequence": 7,
        "x": 10.0,
        "y": 10.0,
        "yaw_rad": 0.0,
        "frame_id": "world",
    }
    assert stored["nodes"]["door"]["order_id"] == 2
    assert stored["nodes"]["door"]["slam_pose"] is None

    map_nodes = json.loads(proxy.call_tool("go2_graph_map_nodes")["content"][0]["text"])
    assert map_nodes["graph_nodes"] == [
        {
            "id": "hall",
            "order_id": 1,
            "name": "hall",
            "kind": "area",
            "has_pano_images": True,
            "anchored": True,
        },
        {
            "id": "door",
            "order_id": 2,
            "name": "door",
            "kind": "door",
            "has_pano_images": False,
            "anchored": False,
        },
    ]

    context = json.loads(
        proxy.call_tool("go2_navigation_map_context")["content"][0]["text"]
    )
    assert context["pose_available"] is True
    assert context["graph_overlay"] is True
    assert context["graph_nodes"][0]["order_id"] == 1

    second_move = proxy.call_tool("relative_move", {"forward": 0.2})
    second_payload = json.loads(second_move["content"][0]["text"])
    slam_map = second_payload["metadata"]["2d_slam_map"]
    assert slam_map["graph_overlay"] is True
    assert slam_map["graph_nodes"] == [
        {"order_id": 1, "id": "hall", "name": "hall", "kind": "area"}
    ]


def test_proxy_saves_slam_status_from_movement_map(
    tmp_path: Path,
) -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the kitchen",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )
    status_path = tmp_path / "go2-slam-status.json"
    proxy = ExternalHarnessAuthorityProxy(
        upstream,
        envelope=envelope,
        graph_memory_path=tmp_path / "go2-graph-memory.json",
        slam_status_path=status_path,
    )

    proxy.call_tool("relative_move", {"forward": 0.4})

    assert status_path.exists()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source"] == "authority_proxy_cached_movement_map_metadata"
    assert payload["latest_slam_map_metadata"]["status"] == "current"
    assert payload["latest_slam_pose"] == {
        "sequence": 7,
        "x": 10.0,
        "y": 10.0,
        "yaw_rad": 0.0,
        "frame_id": "world",
    }
    assert payload["pose_available"] is True


def test_proxy_forwards_graph_position_as_relocalization_prior(tmp_path: Path) -> None:
    graph_path = tmp_path / "go2-graph-memory.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "next_order_id": 2,
                "nodes": {
                    "A1": {
                        "id": "A1",
                        "kind": "area",
                        "name": "Kitchen",
                        "description": "Kitchen area",
                        "records": [],
                        "pano_images": {},
                        "slam_pose": {"x": 1.0, "y": 2.0, "yaw_rad": 0.3, "sequence": 4},
                        "navigation_goal": None,
                        "waypoints": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    upstream = FakeMcpAdapter(include_required_tools=True)
    upstream.tools.append(
        {
            "name": "go2_report_relocalization_prior",
            "inputSchema": {"type": "object"},
        }
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, graph_memory_path=graph_path)

    result = proxy.call_tool("go2_report_graph_position", {"id": "A1"})

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is True
    assert payload["relocalization_prior"]["success"] is True
    assert payload["relocalization_wait"]["status"] == "localized"
    assert payload["2d_slam_map"]["status"] == "current"
    assert payload["2d_slam_map"]["source"] == "native_global_map_area_segmented"
    assert any(block.get("mimeType") == "image/png" for block in result["content"])
    assert (
        "go2_report_relocalization_prior",
        {"node_id": "A1", "slam_pose": {"x": 1.0, "y": 2.0, "yaw_rad": 0.3, "sequence": 4}},
        None,
    ) in upstream.calls
    visible = {tool["name"] for tool in proxy.list_tools()}
    assert "go2_report_relocalization_prior" not in visible
    assert "go2_slam_map_snapshot" not in visible


def test_proxy_returns_slam_map_with_graph_route_context(tmp_path: Path) -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    proxy = ExternalHarnessAuthorityProxy(
        upstream,
        graph_memory_path=tmp_path / "go2-graph-memory.json",
    )

    result = proxy.call_tool(
        "go2_query_route_context",
        {"terms": ["kitchen"], "inspect_ids": [], "return_images": False},
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is True
    assert payload["2d_slam_map"]["status"] == "current"
    assert payload["2d_slam_map"]["source"] == "native_global_map_area_segmented"
    assert any(block.get("mimeType") == "image/png" for block in result["content"])
    assert (
        "go2_slam_map_snapshot",
        {},
        None,
    ) in upstream.calls


def test_proxy_loads_initial_slam_status_before_first_movement(
    tmp_path: Path,
) -> None:
    initial_path = tmp_path / "initial-go2-slam-status.json"
    initial_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "latest_slam_map_metadata": {
                    "status": "static",
                    "source": "known_map",
                    "graph_overlay": True,
                    "pose": {
                        "frame_id": "world",
                        "position_m": {"x": 3.0, "y": 4.0, "z": 0.0},
                        "yaw_rad": 1.0,
                    },
                    "pose_sequence": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True),
        slam_status_path=tmp_path / "next-go2-slam-status.json",
        initial_slam_status_path=initial_path,
    )
    saved_path = tmp_path / "next-go2-slam-status.json"

    tool_names = {tool["name"] for tool in proxy.list_tools()}
    assert "go2_navigation_map_context" in tool_names
    assert "go2_apply_graph_mutations" not in tool_names

    context = json.loads(
        proxy.call_tool("go2_navigation_map_context")["content"][0]["text"]
    )
    assert context["status"] == "static"
    assert context["pose_available"] is False
    assert context["loaded_initial_status"] is True
    assert context["latest_2d_slam_map"]["source"] == "known_map"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["loaded_initial_status"] is True
    assert saved["latest_slam_map_metadata"]["status"] == "static"


def test_proxy_reports_initial_relocalization_bundle_context(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    map_path = bundle_dir / "global_map.pc2.lcm"
    map_path.write_bytes(b"fake-map-bytes")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "current",
                "global_map_file": str(map_path),
                "point_count": 123,
                "frame_id": "map",
                "saved_at_epoch_s": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    initial_status = tmp_path / "initial-status.json"
    initial_status.write_text(
        json.dumps({"latest_slam_map_metadata": {"status": "static"}}),
        encoding="utf-8",
    )
    status_path = tmp_path / "next-status.json"
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True),
        slam_status_path=status_path,
        initial_slam_status_path=initial_status,
        initial_relocalization_bundle=bundle_dir,
    )

    payload = json.loads(proxy.call_tool("go2_relocalization_status")["content"][0]["text"])
    assert payload["relocalization_bundle"]["map_available"] is True
    assert payload["relocalization_bundle"]["global_map_file"] == str(map_path)
    assert "fake-map-bytes" not in json.dumps(payload)

    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["relocalization_bundle_path"] == str(bundle_dir)
    assert saved["relocalization_map_file"] == str(map_path)
    assert saved["relocalization_map_available"] is True
    assert saved["relocalization_map_point_count"] == 123
    assert "fake-map-bytes" not in json.dumps(saved)


def test_proxy_save_relocalization_bundle_updates_slam_status(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "go2-slam-status.json"
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True),
        envelope=MovementAuthorityEnvelope.create(
            task_goal="save map",
            allowed_movement_tools=("relative_move",),
        ),
        slam_status_path=status_path,
    )

    proxy.call_tool("relative_move", {"forward": 0.5})
    result = proxy.call_tool("go2_save_relocalization_bundle", {})

    payload = json.loads(result["content"][0]["text"])
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert "yaw_rad" not in json.dumps(payload["metadata"]["relocalization_bundle"])
    assert saved["relocalization_map_available"] is True
    assert saved["relocalization_map_file"] == (
        "/tmp/go2/slam/relocalization/global_map.pc2.lcm"
    )
    assert saved["relocalization_map_point_count"] == 456
    assert saved["relocalization_map_frame_id"] == "world"
    relocalization_fields = {
        key: value for key, value in saved.items() if key.startswith("relocalization_")
    }
    assert "pose" not in json.dumps(relocalization_fields)
    assert "yaw_rad" not in json.dumps(relocalization_fields)


def test_proxy_start_navigation_episode_clears_cached_slam_status(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "go2-slam-status.json"
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(include_required_tools=True),
        envelope=MovementAuthorityEnvelope.create(
            task_goal="reset map",
            allowed_movement_tools=("relative_move",),
        ),
        slam_status_path=status_path,
    )

    proxy.call_tool("relative_move", {"forward": 0.5})
    before = json.loads(status_path.read_text(encoding="utf-8"))
    assert before["latest_slam_map"]["status"] == "current"
    assert before["pose_available"] is True
    assert proxy._latest_slam_map_image_block is not None

    result = proxy.call_tool("go2_start_navigation_episode", {"clear_slam_map": True})

    payload = json.loads(result["content"][0]["text"])
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert saved["latest_slam_map"]["status"] == "reset"
    assert saved["latest_slam_map"]["source"] == "episode_start_reset"
    assert saved["pose_available"] is False
    assert saved["latest_slam_pose"] is None
    assert proxy._latest_slam_map_image_block is None


def test_proxy_graph_memory_rejects_invalid_batch_without_partial_write(
    tmp_path: Path,
) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(),
        graph_memory_path=tmp_path / "graph.json",
    )

    result = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "area-a",
                        "kind": "area",
                        "name": "area a",
                        "description": "Area A.",
                        "adjacents": [
                            {"name": "area b", "id": "area-b", "note": "Invalid direct area link."}
                        ],
                        "records": [],
                        "pano_images": {},
                        "waypoints": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "area-b",
                        "kind": "area",
                        "name": "area b",
                        "description": "Area B.",
                        "adjacents": [],
                        "records": [],
                        "pano_images": {},
                        "waypoints": {},
                    },
                },
            ]
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is False
    assert "links must alternate" in payload["error"]
    assert not (tmp_path / "graph.json").exists()


def test_proxy_graph_memory_rejects_hierarchy_door_note_rules(
    tmp_path: Path,
) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(),
        graph_memory_path=tmp_path / "graph.json",
    )

    result = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "area-a",
                        "kind": "area",
                        "name": "area a",
                        "description": "Area A.",
                        "adjacents": [
                            {
                                "name": "door a",
                                "id": "door-a",
                                "note": "Door on the left leads to office; unexplored.",
                            }
                        ],
                        "records": [],
                        "pano_images": {},
                        "waypoints": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "door-a",
                        "kind": "door",
                        "name": "door a",
                        "description": "Door A.",
                        "adjacents": [],
                        "records": [],
                        "pano_images": {},
                    },
                },
            ]
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is False
    assert "relative-only" in payload["error"]
    assert not (tmp_path / "graph.json").exists()


def test_proxy_graph_memory_rejects_adjacency_name_mismatch(
    tmp_path: Path,
) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(),
        graph_memory_path=tmp_path / "graph.json",
    )

    result = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "area-a",
                        "kind": "area",
                        "name": "area a",
                        "description": "Area A.",
                        "adjacents": [
                            {
                                "name": "wrong door name",
                                "id": "door-a",
                                "note": "Door beside the cabinet leads to office; unexplored.",
                            }
                        ],
                        "records": [],
                        "pano_images": {},
                        "waypoints": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "door-a",
                        "kind": "door",
                        "name": "door a",
                        "description": "Door A.",
                        "adjacents": [],
                        "records": [],
                        "pano_images": {},
                    },
                },
            ]
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is False
    assert "expected 'door a'" in payload["error"]
    assert not (tmp_path / "graph.json").exists()


def test_proxy_graph_memory_query_route_context_returns_matches_and_paths(
    tmp_path: Path,
) -> None:
    proxy = ExternalHarnessAuthorityProxy(
        FakeMcpAdapter(),
        graph_memory_path=tmp_path / "graph.json",
    )
    proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "hall",
                        "kind": "area",
                        "name": "hall",
                        "description": "Hall with white cabinet.",
                        "adjacents": [
                            {
                                "name": "door",
                                "id": "door",
                                "note": (
                                    "Door beside the white cabinet leads to the kitchen; "
                                    "unexplored."
                                ),
                            }
                        ],
                        "records": [],
                        "pano_images": {},
                        "waypoints": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "door",
                        "kind": "door",
                        "name": "door",
                        "description": "Door to kitchen.",
                        "adjacents": [
                            {
                                "name": "hall",
                                "id": "hall",
                                "note": "Hall side has cabinet.",
                            },
                            {"name": "kitchen", "id": "kitchen", "note": "Kitchen side has table."},
                        ],
                        "records": [],
                        "pano_images": {},
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": "kitchen",
                        "kind": "area",
                        "name": "kitchen",
                        "description": "Kitchen with table.",
                        "adjacents": [
                            {
                                "name": "door",
                                "id": "door",
                                "note": "Door beside the table.",
                            }
                        ],
                        "records": ["Target mug seen on table."],
                        "pano_images": {"front": "artifact:kitchen-front"},
                        "waypoints": {},
                    },
                },
            ]
        },
    )

    result = proxy.call_tool(
        "go2_query_route_context",
        {
            "current_node_id": "hall",
            "terms": ["mug"],
            "inspect_ids": ["kitchen"],
            "depth": 2,
            "return_images": False,
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["success"] is True
    assert payload["current_node"]["id"] == "hall"
    assert payload["matches"]["mug"][0]["id"] == "kitchen"
    assert payload["inspected_nodes"]["kitchen"]["records"] == ["Target mug seen on table."]
    assert "pano_images" not in payload["inspected_nodes"]["kitchen"]
    assert ["hall", "door", "kitchen"] in [
        [step["id"] for step in path] for path in payload["paths"]
    ]


def test_proxy_signs_movement_once_per_call_inside_envelope() -> None:
    upstream = FakeMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the doorway",
        scope="turn and move within the operator-cleared office",
        area="office test area",
        duration_s=60.0,
        allowed_movement_tools=("relative_move",),
        max_movement_calls=2,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    first = proxy.call_tool("relative_move", {"forward": 0.5})
    second = proxy.call_tool("relative_move", {"forward": 0.25})
    refused = proxy.call_tool("relative_move", {"forward": 0.1})

    assert first["content"][0]["text"] == "called relative_move"
    assert second["content"][0]["text"] == "called relative_move"
    assert "call budget exhausted" in refused["content"][0]["text"]
    assert len(upstream.calls) == 2
    first_meta = upstream.calls[0][2] or {}
    second_meta = upstream.calls[1][2] or {}
    assert first_meta["dimos/external_harness"] is True
    assert first_meta["dimos/movement_confirmed"] is True
    assert first_meta["dimos/task_goal"] == "find the doorway"
    assert first_meta["dimos/authority_envelope"]["scope"] == (
        "turn and move within the operator-cleared office"
    )
    assert (
        first_meta["dimos/movement_authority_token"]
        != second_meta["dimos/movement_authority_token"]
    )


def test_proxy_waits_for_pixelgoal_and_returns_terminal_four_views() -> None:
    upstream = PixelGoalMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    tools = {tool["name"]: tool for tool in proxy.list_tools()}
    tool = tools["pixelgoal_navigate"]
    result = proxy.call_tool(
        "pixelgoal_navigate", {"point": [0.5, 0.8], "direction": "front"}
    )

    assert "waits for the hop to finish" in tool["description"]
    assert "clearly visible surface" in tool["description"]
    assert "farthest clearly visible safe option" in tool["description"]
    assert "floor-wall boundary" in tool["description"]
    assert set(tool["inputSchema"]["properties"]) == {"point", "direction"}
    assert tool["inputSchema"]["required"] == ["point", "direction"]
    assert tool["inputSchema"]["additionalProperties"] is False
    assert "already attaches the initial views" in tools["observe_surround"]["description"]
    assert [block["text"] for block in result["content"] if block["type"] == "text"][-4:] == [
        "front",
        "right",
        "back",
        "left",
    ]
    payload = json.loads(result["content"][0]["text"])
    assert [event["type"] for event in payload["events"]] == [
        "hop_start",
        "ray_projected",
        "hop_end",
    ]
    assert [call[0] for call in upstream.calls] == [
        "pixelgoal_navigate",
        "pixelgoal_status",
        "pixelgoal_status",
    ]
    assert envelope.movement_calls_signed == 1


def test_proxy_persists_movement_panoramas_and_hydrates_graph_images(tmp_path: Path) -> None:
    upstream = PixelGoalMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(
        upstream,
        envelope=envelope,
        graph_memory_path=tmp_path / "logs" / "go2-graph-memory.json",
    )

    movement = proxy.call_tool(
        "pixelgoal_navigate", {"point": [0.5, 0.8], "direction": "front"}
    )
    movement_payload = json.loads(movement["content"][0]["text"])
    refs = movement_payload["pano_images"]
    assert set(refs) == {"front", "right", "back", "left"}
    assert all(ref.startswith("logs/panoramas/") for ref in refs.values())
    assert all((tmp_path / ref).is_file() for ref in refs.values())

    created = proxy.call_tool(
        "go2_apply_graph_mutations",
        {
            "mutations": [
                {
                    "op": "create",
                    "node": {
                        "id": "entry-area",
                        "kind": "area",
                        "name": "Entry",
                        "description": "Entry area",
                        "pano_images": refs,
                    },
                }
            ]
        },
    )
    assert json.loads(created["content"][0]["text"])["success"] is True

    inspected = proxy.call_tool(
        "go2_inspect_graph_node", {"id": "entry-area", "return_images": True}
    )
    assert sum(block["type"] == "image" for block in inspected["content"]) == 4


def test_proxy_rejects_internal_pixelgoal_arguments_without_signing() -> None:
    upstream = PixelGoalMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool(
        "pixelgoal_navigate",
        {"point": [0.5, 0.8], "goal_id": "codex-thread-id"},
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["code"] == "agent_arguments_not_allowed"
    missing = proxy.call_tool("pixelgoal_navigate", {"point": [0.5, 0.8]})
    assert "requires: direction" in missing["content"][0]["text"]
    assert upstream.calls == []
    assert envelope.movement_calls_signed == 0


def test_proxy_rewrites_pixelgoal_server_status_context() -> None:
    upstream = PixelGoalMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool("server_status", {})

    payload = json.loads(result["content"][0]["text"])
    profile = payload["go2_harness_profile"]
    assert "agent selects PixelGoal points" in profile["description"]
    assert "internal live LiDAR costmap" in profile["allowed_strategy"]
    assert "prebuilt or agent-visible map planning" in profile["disallowed_strategy"]
    assert payload["external_authority"]["agent_map_access"] == "none"
    assert payload["external_authority"]["allowed_movement_tools"] == [
        "pixelgoal_navigate"
    ]
    assert payload["agent_runtime_policy"]["allowed_movement_tools"] == [
        "pixelgoal_navigate"
    ]


def test_proxy_falls_back_to_observe_when_terminal_pixelgoal_images_are_missing() -> None:
    upstream = PixelGoalMcpAdapter(terminal_images=False)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool(
        "pixelgoal_navigate", {"point": [0.5, 0.8], "direction": "front"}
    )

    assert [call[0] for call in upstream.calls][-1] == "observe_surround"
    assert sum(block["type"] == "image" for block in result["content"]) == 4


def test_proxy_stops_pixelgoal_when_agent_wait_times_out(monkeypatch) -> None:
    monkeypatch.setattr(
        "dimos.agents.mcp.external_harness_authority_proxy.PIXELGOAL_COMPLETION_TIMEOUT_S",
        0.0,
    )
    upstream = PixelGoalMcpAdapter(terminal=False)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="go to the doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool(
        "pixelgoal_navigate", {"point": [0.5, 0.8], "direction": "front"}
    )
    payload = json.loads(result["content"][0]["text"])

    assert payload["code"] == "agent_wait_timeout"
    assert [call[0] for call in upstream.calls][-1] == "pixelgoal_stop"


def test_proxy_returns_agent_facing_result_when_movement_upstream_times_out() -> None:
    upstream = TimeoutMovementMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="move near the doorway",
        tool_profile="vision_odom_local_motion",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool("relative_move", {"forward": 0.8, "left": -0.4})
    payload = json.loads(result["content"][0]["text"])

    assert payload["success"] is False
    assert payload["error_code"] == "MOVEMENT_COMPLETION_TIMEOUT_AFTER_DISPATCH"
    assert payload["metadata"]["command_may_be_active"] is True
    assert payload["metadata"]["recommended_next_action"] == (
        "check_situation_observe_then_report_operator"
    )
    assert "local planning or Go2 obstacle protection" in payload["message"]
    assert upstream.calls[0][0] == "relative_move"
    assert (upstream.calls[0][2] or {})["dimos/external_harness"] is True


def test_proxy_refuses_movement_outside_envelope_tool_scope() -> None:
    upstream = FakeMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="translate only",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool("rotate_in_place", {"degrees": 90})

    assert "outside the approved envelope scope" in result["content"][0]["text"]
    assert upstream.calls == []


def test_proxy_posture_only_envelope_allows_lie_down_but_refuses_locomotion() -> None:
    upstream = FakeMcpAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="safe shutdown posture",
        allowed_movement_tools=("full_lie_down", "lie_down"),
        duration_s=60.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    posture = proxy.call_tool("full_lie_down")
    locomotion = proxy.call_tool("relative_move", {"forward": 0.25})
    rotation = proxy.call_tool("rotate_in_place", {"degrees": 20})

    assert posture["content"][0]["text"] == "called full_lie_down"
    assert "outside the approved envelope scope" in locomotion["content"][0]["text"]
    assert "outside the approved envelope scope" in rotation["content"][0]["text"]
    assert len(upstream.calls) == 1
    assert upstream.calls[0][0] == "full_lie_down"
    assert (upstream.calls[0][2] or {})["dimos/external_harness"] is True


def test_proxy_posture_startup_tools_list_is_limited_to_startup_posture() -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="stand up before the live task",
        tool_profile="posture_startup",
        allowed_movement_tools=("stand_up", "balance_stand"),
        duration_s=60.0,
        max_movement_calls=2,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    tools = proxy.list_tools()
    tool_names = {tool["name"] for tool in tools}

    assert {"observe", "go2_situation", "stand_up", "balance_stand"}.issubset(
        tool_names
    )
    assert "stop_movement" not in tool_names
    assert "relative_move" not in tool_names
    assert "rotate_in_place" not in tool_names
    assert "full_lie_down" not in tool_names
    stand_meta = next(tool for tool in tools if tool["name"] == "stand_up")["_meta"]
    assert stand_meta["dimos/authority_proxy"]["tool_profile"] == "posture_startup"
    assert stand_meta["dimos/authority_proxy"]["movement_calls_remaining"] == 2


def test_proxy_refuses_agent_stop_tools_without_forwarding() -> None:
    upstream = FakeMcpAdapter()
    proxy = ExternalHarnessAuthorityProxy(upstream)

    movement = proxy.call_tool("stop_movement")
    pixelgoal = proxy.call_tool("pixelgoal_stop")

    assert "disabled for agent use" in movement["content"][0]["text"]
    assert "disabled for agent use" in pixelgoal["content"][0]["text"]
    assert upstream.calls == []


def test_proxy_battery_gate_refuses_new_movement_below_start_threshold() -> None:
    upstream = FakeMcpAdapter(battery_percent=34.0)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the doorway",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool("relative_move", {"forward": 0.5})

    assert "below start threshold" in result["content"][0]["text"]
    assert upstream.calls == [("go2_robot_status", {}, None)]


def test_proxy_overlays_external_authority_and_robot_status_on_go2_situation() -> None:
    upstream = FakeMcpAdapter(battery_percent=91.0, include_required_tools=True)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the doorway",
        tool_profile="vision_odom_local_motion",
        allowed_movement_tools=("relative_move", "rotate_in_place"),
        duration_s=60.0,
        max_movement_calls=4,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)

    result = proxy.call_tool("go2_situation")
    payload = json.loads(result["content"][0]["text"])
    situation = payload["metadata"]["situation"]

    assert situation["external_authority"]["source"] == "authority_proxy"
    assert situation["external_authority"]["tool_profile"] == "vision_odom_local_motion"
    assert situation["external_authority"]["allowed_movement_tools"] == [
        "relative_move",
        "rotate_in_place",
    ]
    assert situation["external_authority"]["movement_calls_remaining"] == 4
    assert situation["battery"]["soc_percent"] == 91.0
    assert situation["battery"]["source"] == "unitree_lowstate"
    assert situation["motor"]["lost_count"] == 0
    assert upstream.calls == [("go2_situation", {}, None), ("go2_robot_status", {}, None)]


def test_proxy_jsonrpc_handler_preserves_image_content_result_shape() -> None:
    upstream = FakeMcpAdapter()
    proxy = ExternalHarnessAuthorityProxy(upstream)

    response = _handle_proxy_request(
        proxy,
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "observe", "arguments": {}},
        },
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": upstream.observe_result,
    }


def test_proxy_app_exposes_streamable_http_tools_list_for_codex_client() -> None:
    upstream = FakeMcpAdapter(include_required_tools=True)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="find the doorway",
        allowed_movement_tools=("relative_move", "rotate_in_place"),
        duration_s=60.0,
    )
    app = create_authority_proxy_app(
        upstream_url="memory://fake-dimos",
        envelope=envelope,
        upstream=upstream,
    )

    client = TestClient(app)
    get_route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/mcp" and "GET" in getattr(route, "methods", set())
    )
    sse_response = asyncio.run(get_route.endpoint())
    assert isinstance(sse_response, StreamingResponse)
    assert sse_response.media_type == "text/event-stream"
    app.state.sse_hub.close_all()

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init-1", "method": "initialize"},
    )
    tools = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list"},
    )

    assert initialize.status_code == 200
    assert tools.status_code == 200
    assert initialize.json()["result"] == {"serverInfo": {"name": "fake-dimos"}}
    tool_names = {tool["name"] for tool in tools.json()["result"]["tools"]}
    assert {
        "observe",
        "go2_robot_status",
        "motion_safety_status",
        "go2_situation",
        "rotate_in_place",
        "relative_move",
        "last_motion_diagnostics",
        "stop_movement",
    }.issubset(tool_names)


def test_proxy_app_returns_structured_timeout_result_for_movement_timeout(
    monkeypatch,
) -> None:
    envelope = MovementAuthorityEnvelope.create(
        task_goal="move near the doorway",
        tool_profile="vision_odom_local_motion",
        allowed_movement_tools=("relative_move",),
        duration_s=60.0,
    )

    class FakeResponse:
        def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self) -> dict[str, Any]:
            return self._payload

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        del url, timeout
        method = json.get("method")
        params = json.get("params") if isinstance(json.get("params"), dict) else {}
        if method == "tools/list":
            return FakeResponse(
                {
                    "result": {
                        "tools": [
                            {
                                "name": "go2_robot_status",
                                "inputSchema": {"type": "object"},
                            },
                            {
                                "name": "relative_move",
                                "inputSchema": {"type": "object"},
                                "_meta": {"dimos/uses": ["movement"]},
                            },
                        ]
                    }
                }
            )
        if method == "tools/call" and params.get("name") == "go2_robot_status":
            return FakeResponse(
                {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json_dumps(
                                    {
                                        "success": True,
                                        "metadata": {
                                            "robot_status": {
                                                "battery": {
                                                    "state": "normal",
                                                    "soc_percent": 88.0,
                                                    "source": "unitree_lowstate",
                                                }
                                            }
                                        },
                                    }
                                ),
                            }
                        ]
                    }
                }
            )
        if method == "tools/call" and params.get("name") == "relative_move":
            raise requests.ReadTimeout("planner completion timed out")
        raise AssertionError(f"unexpected MCP call: {json}")

    monkeypatch.setattr(requests, "post", fake_post)
    app = create_authority_proxy_app(
        upstream_url="http://fake-upstream/mcp",
        envelope=envelope,
    )

    client = TestClient(app)
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "move-1",
            "method": "tools/call",
            "params": {"name": "relative_move", "arguments": {"forward": 0.8, "left": -0.4}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "error" not in payload
    result = payload["result"]
    tool_payload = json.loads(result["content"][0]["text"])
    assert tool_payload["success"] is False
    assert tool_payload["error_code"] == "MOVEMENT_COMPLETION_TIMEOUT_AFTER_DISPATCH"
    assert tool_payload["metadata"]["command_may_be_active"] is True
    assert tool_payload["metadata"]["recommended_next_action"] == (
        "check_situation_observe_then_stop_or_retry_shorter"
    )
