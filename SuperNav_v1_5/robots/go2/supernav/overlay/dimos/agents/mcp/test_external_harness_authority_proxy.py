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
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image as PILImage
import requests
from starlette.responses import StreamingResponse

from dimos.agents.mcp.external_harness_authority_proxy import (
    ExternalHarnessAuthorityProxy,
    MovementAuthorityEnvelope,
    _handle_proxy_request,
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



def _four_view_content() -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for direction in ("front", "right", "back", "left"):
        content.extend([{"type": "text", "text": direction}, _fake_png_block()])
    return content


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
    assert "stop_movement" in tool_names
    assert "relative_move" not in tool_names
    assert "rotate_in_place" not in tool_names
    assert "stand_up" not in tool_names
    assert "reset_pixel_goal_map" not in tool_names

    result = proxy.call_tool("reset_pixel_goal_map")
    assert "operator-only" in result["content"][0]["text"]
    assert upstream.calls == []


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
        "check_situation_observe_then_stop_or_retry_shorter"
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

    assert {"observe", "go2_situation", "stop_movement", "stand_up", "balance_stand"}.issubset(
        tool_names
    )
    assert "relative_move" not in tool_names
    assert "rotate_in_place" not in tool_names
    assert "full_lie_down" not in tool_names
    stand_meta = next(tool for tool in tools if tool["name"] == "stand_up")["_meta"]
    assert stand_meta["dimos/authority_proxy"]["tool_profile"] == "posture_startup"
    assert stand_meta["dimos/authority_proxy"]["movement_calls_remaining"] == 2


def test_proxy_forwards_stop_tool_without_movement_authority() -> None:
    upstream = FakeMcpAdapter()
    proxy = ExternalHarnessAuthorityProxy(upstream)

    result = proxy.call_tool("stop_movement")

    assert result["content"][0]["text"] == "called stop_movement"
    assert upstream.calls == [("stop_movement", {}, None)]


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
