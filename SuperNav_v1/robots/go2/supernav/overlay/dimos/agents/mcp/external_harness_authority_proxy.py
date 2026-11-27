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

"""Bounded MCP proxy for external Codex harness movement authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import requests
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from dimos.agents.mcp.external_harness_authority import ExternalHarnessMovementAuthority
from dimos.agents.mcp.external_harness_types import ExternalHarnessMcpAdapter
from dimos.agents.mcp.mcp_adapter import McpAdapter
from dimos.agents.mcp.streamable_http import StreamableHttpSseHub
from dimos.agents.runtime_policy import is_movement_tool, is_stop_tool

DEFAULT_AUTHORITY_PROXY_CLIENT_ID = "codex-external-harness"
DEFAULT_AUTHORITY_PROXY_DURATION_S = 3600.0
DEFAULT_AUTHORITY_PROXY_SCOPE = "operator-approved live Go2 task"
DEFAULT_AUTHORITY_PROXY_AREA = "operator-approved local test area"
DEFAULT_MIN_START_BATTERY_PERCENT = 35.0
DEFAULT_CRITICAL_BATTERY_PERCENT = 30.0
BATTERY_SAFE_TOOL_NAMES = frozenset(
    {
        "full_lie_down",
        "go2_robot_status",
        "lie_down",
        "motion_safety_status",
        "stop_movement",
        "stop_navigation",
    }
)
DISABLED_MEMORY_TOOL_NAMES = frozenset(
    {
        "go2_list_graph_nodes",
        "go2_grep_graph_node",
        "go2_inspect_graph_node",
        "go2_query_route_context",
        "go2_report_graph_position",
        "go2_graph_map_nodes",
        "go2_apply_graph_mutations",
        "go2_relocalization_status",
        "go2_known_slam_map",
        "go2_navigation_map_context",
        "go2_start_navigation_episode",
        "go2_save_relocalization_bundle",
    }
)
OPERATOR_ONLY_TOOL_NAMES = frozenset({"reset_pixel_goal_map"})
PIXELGOAL_COMPLETION_TIMEOUT_S = 300.0
PIXELGOAL_POLL_INTERVAL_S = 1.0
SURROUND_DIRECTIONS = frozenset({"front", "right", "back", "left"})


@dataclass
class MovementAuthorityEnvelope:
    """Operator-approved boundary for signing external movement MCP calls."""

    task_goal: str
    scope: str
    area: str
    client_id: str
    approved_at_epoch_s: float
    expires_at_epoch_s: float
    approval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approved_by: str = "operator"
    tool_profile: str | None = None
    allowed_movement_tools: tuple[str, ...] = ()
    max_movement_calls: int | None = None
    movement_calls_signed: int = 0

    @classmethod
    def create(
        cls,
        *,
        task_goal: str,
        scope: str = DEFAULT_AUTHORITY_PROXY_SCOPE,
        area: str = DEFAULT_AUTHORITY_PROXY_AREA,
        duration_s: float = DEFAULT_AUTHORITY_PROXY_DURATION_S,
        client_id: str = DEFAULT_AUTHORITY_PROXY_CLIENT_ID,
        approved_by: str = "operator",
        tool_profile: str | None = None,
        allowed_movement_tools: tuple[str, ...] | list[str] = (),
        max_movement_calls: int | None = None,
        now: float | None = None,
    ) -> MovementAuthorityEnvelope:
        """Create one bounded movement-approval envelope for a proxy process."""

        now = time.time() if now is None else float(now)
        duration_s = float(duration_s)
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if max_movement_calls is not None and max_movement_calls <= 0:
            raise ValueError("max_movement_calls must be positive when set")
        task_goal = task_goal.strip()
        if not task_goal:
            raise ValueError("task_goal must not be empty")
        scope = scope.strip()
        if not scope:
            raise ValueError("scope must not be empty")
        area = area.strip()
        if not area:
            raise ValueError("area must not be empty")

        allowed = tuple(sorted({tool for tool in allowed_movement_tools if tool}))
        profile = tool_profile.strip() if isinstance(tool_profile, str) else None
        return cls(
            task_goal=task_goal,
            scope=scope,
            area=area,
            client_id=client_id.strip() or DEFAULT_AUTHORITY_PROXY_CLIENT_ID,
            approved_by=approved_by.strip() or "operator",
            tool_profile=profile or None,
            approved_at_epoch_s=now,
            expires_at_epoch_s=now + duration_s,
            allowed_movement_tools=allowed,
            max_movement_calls=max_movement_calls,
        )

    def refusal_for_tool(self, tool_name: str, *, now: float | None = None) -> str | None:
        """Return a refusal reason when ``tool_name`` is outside the envelope."""

        now = time.time() if now is None else float(now)
        if now >= self.expires_at_epoch_s:
            return "movement authority envelope expired"
        if self.allowed_movement_tools and tool_name not in self.allowed_movement_tools:
            return f"movement tool '{tool_name}' is outside the approved envelope scope"
        if (
            self.max_movement_calls is not None
            and self.movement_calls_signed >= self.max_movement_calls
        ):
            return "movement authority envelope call budget exhausted"
        return None

    def sign_call(self, tool_name: str, *, now: float | None = None) -> dict[str, Any] | str:
        """Return fresh one-shot authority metadata or a refusal string."""

        refusal = self.refusal_for_tool(tool_name, now=now)
        if refusal is not None:
            return refusal

        self.movement_calls_signed += 1
        authority = ExternalHarnessMovementAuthority.create(
            task_goal=self.task_goal,
            client_id=self.client_id,
        )
        meta = authority.meta()
        meta["dimos/authority_envelope"] = self.model_dump()
        return meta

    def movement_calls_remaining(self) -> int | None:
        """Return remaining signed movement calls, or ``None`` when unbounded."""

        if self.max_movement_calls is None:
            return None
        return max(self.max_movement_calls - self.movement_calls_signed, 0)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatteryGatePolicy:
    """Battery thresholds for refusing new live movement through the proxy."""

    min_start_percent: float = DEFAULT_MIN_START_BATTERY_PERCENT
    critical_percent: float = DEFAULT_CRITICAL_BATTERY_PERCENT

    def refusal_for_tool(
        self,
        tool_name: str,
        *,
        battery_percent: float | None,
    ) -> str | None:
        if battery_percent is None or tool_name in BATTERY_SAFE_TOOL_NAMES:
            return None
        if battery_percent < self.critical_percent:
            return (
                f"battery {battery_percent:.1f}% is below critical threshold "
                f"{self.critical_percent:.1f}%; only stop/lie_down/status tools are allowed"
            )
        if battery_percent < self.min_start_percent:
            return (
                f"battery {battery_percent:.1f}% is below start threshold "
                f"{self.min_start_percent:.1f}%; refusing new live movement"
            )
        return None


class ExternalHarnessAuthorityProxy:
    """MCP adapter facade that signs movement calls only inside an envelope."""

    def __init__(
        self,
        upstream: ExternalHarnessMcpAdapter,
        *,
        envelope: MovementAuthorityEnvelope | None = None,
        battery_gate: BatteryGatePolicy | None = None,
        url: str = "memory://external-harness-authority-proxy",
    ) -> None:
        self.upstream = upstream
        self.envelope = envelope
        self.battery_gate = battery_gate or BatteryGatePolicy()
        self.url = url
        self._tools_by_name: dict[str, dict[str, Any]] = {}

    def initialize(self) -> dict[str, Any]:
        return self.upstream.initialize()

    def list_tools(self) -> list[dict[str, Any]]:
        tools = self.upstream.list_tools()
        self._tools_by_name = {str(tool.get("name")): tool for tool in tools if tool.get("name")}
        visible_tools = [
            self._agent_facing_tool(tool) for tool in tools if self._tool_visible(tool)
        ]
        return visible_tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward read-only calls unchanged and sign movement calls per envelope."""

        arguments = arguments or {}
        meta = dict(meta or {})
        if name in DISABLED_MEMORY_TOOL_NAMES:
            return _text_result(f"Tool '{name}' refused: memory and agent-facing SLAM tools are disabled.")
        if name in OPERATOR_ONLY_TOOL_NAMES:
            return _text_result(f"Tool '{name}' refused: operator-only tool.")

        uses = self._uses_for_tool(name)
        if not is_movement_tool(name, uses) or is_stop_tool(name):
            result = self.upstream.call_tool(name, arguments, meta=meta or None)
            return self._agent_facing_result(name, result)

        if self.envelope is None:
            return _text_result(
                f"Movement tool '{name}' refused: no active external-harness authority envelope."
            )

        if name == "pixelgoal_navigate":
            unexpected = sorted(set(arguments) - {"point", "direction"})
            missing = sorted({"point", "direction"} - set(arguments))
            if unexpected or missing:
                details = []
                if unexpected:
                    details.append(f"does not accept: {', '.join(unexpected)}")
                if missing:
                    details.append(f"requires: {', '.join(missing)}")
                return _text_result(
                    json.dumps(
                        {
                            "ok": False,
                            "code": "agent_arguments_not_allowed",
                            "error": f"agent-facing pixelgoal_navigate {'; '.join(details)}",
                        }
                    )
                )

        battery_refusal = self._battery_refusal_for_tool(name)
        if battery_refusal is not None:
            return _text_result(f"Movement tool '{name}' refused: {battery_refusal}.")

        signed = self.envelope.sign_call(name)
        if isinstance(signed, str):
            return _text_result(f"Movement tool '{name}' refused: {signed}.")

        forwarded_meta = {**meta, **signed}
        try:
            result = self.upstream.call_tool(name, arguments, meta=forwarded_meta)
            if name == "pixelgoal_navigate":
                return self._wait_for_pixelgoal(result)
            return result
        except requests.Timeout as exc:
            if name == "pixelgoal_navigate":
                try:
                    self.upstream.call_tool("pixelgoal_stop", {})
                except Exception:
                    pass
            return _movement_timeout_result(
                name,
                arguments,
                envelope=self.envelope,
                exc=exc,
            )

    def _tool_visible(self, tool: dict[str, Any]) -> bool:
        name = str(tool.get("name") or "")
        if not name or name in OPERATOR_ONLY_TOOL_NAMES or name in DISABLED_MEMORY_TOOL_NAMES:
            return False
        uses = self._tool_uses(tool)
        if is_stop_tool(name) or not is_movement_tool(name, uses):
            return True
        if self.envelope is None:
            return False
        return name in self.envelope.allowed_movement_tools

    def _agent_facing_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        name = str(tool.get("name") or "")
        payload = dict(tool)
        uses = self._tool_uses(tool)
        pixelgoal_enabled = bool(
            self.envelope and "pixelgoal_navigate" in self.envelope.allowed_movement_tools
        )
        if pixelgoal_enabled and name == "observe_surround":
            payload["description"] = (
                "Return four-way camera health and attach only fresh front/right/back/left "
                "frames. The harness already attaches the initial views, and each normal "
                "pixelgoal_navigate result supplies the next views. Call this only to recover "
                "from missing/stale images or diagnose an abnormal result."
            )
        if self.envelope is not None and is_movement_tool(name, uses) and not is_stop_tool(name):
            meta = dict(payload.get("_meta") or {})
            meta["dimos/authority_proxy"] = {
                "status": "allowed_by_active_envelope",
                "tool_profile": self.envelope.tool_profile,
                "allowed_movement_tools": list(self.envelope.allowed_movement_tools),
                "movement_calls_remaining": self.envelope.movement_calls_remaining(),
                "expires_at_epoch_s": self.envelope.expires_at_epoch_s,
            }
            payload["_meta"] = meta
            if name == "pixelgoal_navigate":
                schema = payload.get("inputSchema")
                properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
                payload["inputSchema"] = {
                    "type": "object",
                    "properties": {
                        "point": properties.get(
                            "point", {"type": "array", "items": {"type": "number"}}
                        ),
                        "direction": properties.get(
                            "direction",
                            {
                                "type": "string",
                                "enum": ["front", "right", "back", "left"],
                            },
                        ),
                    },
                    "required": ["point", "direction"],
                    "additionalProperties": False,
                }
                payload["description"] = (
                    "Navigate to a normalized point in a front/right/back/left camera view. "
                    "The point must be on a clearly visible surface: traversable floor, or "
                    "a large stable vertical surface (wall/cabinet/solid door/furniture "
                    "face) as a progress proxy — the robot stops about 0.35 m in front of "
                    "the marked surface. When direction is clear, choose the farthest "
                    "clearly visible safe option in that direction. Glass/reflections, "
                    "thin structures (legs, cables), and the floor-wall boundary are "
                    "invalid; never mark the search target itself. "
                    "This agent-facing call waits for the hop to finish and returns the "
                    "terminal status plus camera freshness metadata and only fresh four-way "
                    "images; do not poll pixelgoal_status during normal operation. "
                    "With the map backend, an unresolved surface allows successive "
                    "exploration legs of up to 4 m, rechecking the target after each "
                    "successful leg, within a default 60 s hop and 10 m exploration budget. "
                    "Budget/free-space exhaustion returns needs_agent / exploration_complete; "
                    "the hop deadline returns timeout / deadline_exceeded. "
                    "Use the returned views to choose the next point. A no_progress "
                    "result means the hop stopped for lack of net approach; reassess "
                    "the obstacle or doorway before issuing another movement."
                )
        return payload

    def _wait_for_pixelgoal(self, started_result: dict[str, Any]) -> dict[str, Any]:
        started = _parse_first_text_json(started_result)
        if not started or started.get("ok") is not True or started.get("status") != "started":
            return started_result
        goal_id = str(started.get("goal_id") or "")
        deadline = time.monotonic() + PIXELGOAL_COMPLETION_TIMEOUT_S
        while time.monotonic() < deadline:
            status_result = self.upstream.call_tool("pixelgoal_status", {"include_images": False})
            status = _parse_first_text_json(status_result)
            if (
                status
                and status.get("hop_running") is False
                and _pixelgoal_status_goal_id(status) == goal_id
            ):
                final = self.upstream.call_tool("pixelgoal_status", {"include_images": True})
                final_status = _parse_first_text_json(final)
                if (
                    not final_status
                    or final_status.get("hop_running") is not False
                    or _pixelgoal_status_goal_id(final_status) != goal_id
                ):
                    # Status can change between the metadata poll and image fetch.
                    # Never return another hop's frames as this call's outcome.
                    time.sleep(PIXELGOAL_POLL_INTERVAL_S)
                    continue
                _compact_pixelgoal_events(final, goal_id)
                if _surround_image_directions(final) == SURROUND_DIRECTIONS:
                    return final
                observed = self.upstream.call_tool("observe_surround", {})
                content = final.get("content")
                observed_content = observed.get("content")
                if isinstance(content, list) and content and isinstance(observed_content, list):
                    merged = dict(final)
                    merged["content"] = [content[0], *observed_content]
                    health = _parse_first_text_json(observed)
                    terminal = _parse_first_text_json(merged)
                    if terminal is not None and health is not None and "stale" in health:
                        terminal["camera_health"] = health
                        _rewrite_first_text_json(merged, terminal)
                    return merged
                return final
            time.sleep(PIXELGOAL_POLL_INTERVAL_S)

        stop_result = self.upstream.call_tool("pixelgoal_stop", {})
        return _text_result(
            json.dumps(
                {
                    "ok": False,
                    "status": "timeout",
                    "code": "agent_wait_timeout",
                    "goal_id": goal_id,
                    "error": (
                        "PixelGoal did not reach a matching terminal state within "
                        f"{PIXELGOAL_COMPLETION_TIMEOUT_S:g}s; pixelgoal_stop was requested."
                    ),
                    "stop_result": _parse_first_text_json(stop_result),
                }
            )
        )

    def _agent_facing_result(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        if self.envelope is None:
            return result
        if name == "go2_situation":
            return _overlay_go2_situation_proxy_context(
                result,
                envelope=self.envelope,
                robot_status_result=self._safe_robot_status_result(),
            )
        if name == "server_status" and "pixelgoal_navigate" in self.envelope.allowed_movement_tools:
            return _overlay_pixelgoal_server_status(result, envelope=self.envelope)
        return result

    def _safe_robot_status_result(self) -> dict[str, Any] | None:
        if "go2_robot_status" not in self._tools_by_name:
            self.list_tools()
        if "go2_robot_status" not in self._tools_by_name:
            return None
        try:
            return self.upstream.call_tool("go2_robot_status", {})
        except Exception:
            return None


    def call_tool_text(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> str:
        result = self.call_tool(name, arguments, meta=meta)
        content = result.get("content", [])
        if not content:
            return ""
        first = content[0]
        return str(first.get("text", first))

    def _uses_for_tool(self, name: str) -> list[str]:
        if name not in self._tools_by_name:
            self.list_tools()
        tool = self._tools_by_name.get(name) or {}
        return self._tool_uses(tool)

    def _tool_uses(self, tool: dict[str, Any]) -> list[str]:
        tool_meta = tool.get("_meta") if isinstance(tool, dict) else {}
        uses = tool_meta.get("dimos/uses") if isinstance(tool_meta, dict) else []
        return [str(use) for use in uses] if isinstance(uses, list) else []

    def _battery_refusal_for_tool(self, name: str) -> str | None:
        if "go2_robot_status" not in self._tools_by_name:
            self.list_tools()
        if "go2_robot_status" not in self._tools_by_name:
            return None
        try:
            status_result = self.upstream.call_tool("go2_robot_status", {})
        except Exception as exc:
            return f"battery status unavailable ({exc.__class__.__name__}: {exc})"
        return self.battery_gate.refusal_for_tool(
            name,
            battery_percent=_extract_battery_percent(status_result),
        )


def create_authority_proxy_app(
    *,
    upstream_url: str,
    envelope: MovementAuthorityEnvelope | None,
    proxy_url: str = "http://127.0.0.1:9992/mcp",
    upstream: ExternalHarnessMcpAdapter | None = None,
) -> FastAPI:
    """Create a Streamable-HTTP MCP proxy app for Codex external harness runs."""

    proxy = ExternalHarnessAuthorityProxy(
        upstream or McpAdapter(upstream_url),
        envelope=envelope,
        url=proxy_url,
    )
    app = FastAPI()
    app.state.sse_hub = StreamableHttpSseHub()

    @app.get("/mcp")
    async def mcp_sse_endpoint() -> StreamingResponse:
        return await app.state.sse_hub.response()

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        body = await request.json()
        response = _handle_proxy_request(proxy, body)
        if response is None:
            return Response(status_code=204)
        return JSONResponse(response)

    app.state.proxy = proxy
    return app


def _handle_proxy_request(
    proxy: ExternalHarnessAuthorityProxy,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    req_id = request.get("id")
    if "id" not in request:
        return None

    method = str(request.get("method") or "")
    params = request.get("params") or {}
    if method == "initialize":
        return _jsonrpc_result(req_id, proxy.initialize())
    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": proxy.list_tools()})
    if method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") if isinstance(params, dict) else {}
        meta = params.get("_meta") if isinstance(params, dict) else None
        return _jsonrpc_result(req_id, proxy.call_tool(name, args, meta=meta))
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown: {method}"},
    }


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _movement_timeout_result(
    name: str,
    arguments: dict[str, Any],
    *,
    envelope: MovementAuthorityEnvelope,
    exc: requests.Timeout,
) -> dict[str, Any]:
    payload = {
        "success": False,
        "message": (
            f"Movement tool '{name}' timed out waiting for completion after the command was "
            "sent. The robot may already be moving or may have made partial progress; this can "
            "happen when local planning or Go2 obstacle protection slows, clips, or blocks the "
            "requested motion. Check go2_situation/observe and call stop_movement before retrying "
            "a shorter corrective move."
        ),
        "error_code": "MOVEMENT_COMPLETION_TIMEOUT_AFTER_DISPATCH",
        "metadata": {
            "tool_name": name,
            "requested_arguments": arguments,
            "command_may_be_active": True,
            "upstream_exception": exc.__class__.__name__,
            "recommended_next_action": "check_situation_observe_then_stop_or_retry_shorter",
            "external_authority": {
                "tool_profile": envelope.tool_profile,
                "movement_calls_signed": envelope.movement_calls_signed,
                "movement_calls_remaining": envelope.movement_calls_remaining(),
            },
        },
    }
    return _text_result(json.dumps(payload, ensure_ascii=False))


def _overlay_go2_situation_proxy_context(
    result: dict[str, Any],
    *,
    envelope: MovementAuthorityEnvelope,
    robot_status_result: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = _parse_first_text_json(result)
    if payload is None:
        return result
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return result
    situation = metadata.setdefault("situation", {})
    if not isinstance(situation, dict):
        return result

    situation["external_authority"] = {
        "available": True,
        "source": "authority_proxy",
        "tool_profile": envelope.tool_profile,
        "scope": envelope.scope,
        "area": envelope.area,
        "allowed_movement_tools": list(envelope.allowed_movement_tools),
        "movement_calls_signed": envelope.movement_calls_signed,
        "movement_calls_remaining": envelope.movement_calls_remaining(),
        "expires_at_epoch_s": envelope.expires_at_epoch_s,
    }
    robot_status = _extract_robot_status(robot_status_result)
    if robot_status is not None:
        _merge_robot_status_into_situation(situation, robot_status)
    _rewrite_first_text_json(result, payload)
    return result


def _overlay_pixelgoal_server_status(
    result: dict[str, Any], *, envelope: MovementAuthorityEnvelope
) -> dict[str, Any]:
    payload = _parse_first_text_json(result)
    if payload is None:
        return result
    profile = payload.get("go2_harness_profile")
    if isinstance(profile, dict):
        profile.update(
            {
                "description": (
                    "The agent selects PixelGoal points from current RGB evidence and has no "
                    "map access; the PixelGoal executor may use a live LiDAR costmap internally."
                ),
                "allowed_strategy": (
                    "current RGB observation and pixelgoal_navigate; internal live LiDAR "
                    "costmap planning inside PixelGoal"
                ),
                "disallowed_strategy": (
                    "prebuilt or agent-visible map planning, semantic memory, tags, frontier, "
                    "patrol, and speech"
                ),
            }
        )
    runtime_policy = payload.get("agent_runtime_policy")
    if isinstance(runtime_policy, dict):
        runtime_policy["allowed_movement_tools"] = list(envelope.allowed_movement_tools)
    payload["external_authority"] = {
        "source": "authority_proxy",
        "allowed_movement_tools": list(envelope.allowed_movement_tools),
        "agent_map_access": "none",
        "pixelgoal_internal_live_lidar_costmap": "allowed",
    }
    _rewrite_first_text_json(result, payload)
    return result


def _parse_first_text_json(result: dict[str, Any]) -> dict[str, Any] | None:
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, dict):
        return None
    text = first.get("text")
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _pixelgoal_status_goal_id(payload: dict[str, Any]) -> str:
    last_result = payload.get("last_result")
    if isinstance(last_result, dict) and last_result.get("goal_id"):
        return str(last_result["goal_id"])
    status = payload.get("status")
    return str(status.get("goal_id") or "") if isinstance(status, dict) else ""


def _compact_pixelgoal_events(result: dict[str, Any], goal_id: str) -> None:
    payload = _parse_first_text_json(result)
    if payload is None:
        return
    events = payload.get("events")
    if not isinstance(events, list):
        return
    start = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if isinstance(events[index], dict)
            and events[index].get("type") == "hop_start"
            and isinstance(events[index].get("data"), dict)
            and str(events[index]["data"].get("goal_id") or "") == goal_id
        ),
        None,
    )
    if start is None:
        payload["events"] = [
            event
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("data"), dict)
            and str(event["data"].get("goal_id") or "") == goal_id
        ]
    else:
        end = next(
            (
                index + 1
                for index in range(start, len(events))
                if isinstance(events[index], dict)
                and events[index].get("type") == "hop_end"
                and isinstance(events[index].get("data"), dict)
                and str(events[index]["data"].get("goal_id") or "") == goal_id
            ),
            len(events),
        )
        payload["events"] = events[start:end]
    _rewrite_first_text_json(result, payload)


def _surround_image_directions(result: dict[str, Any]) -> frozenset[str]:
    directions: set[str] = set()
    label = ""
    for block in result.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = str(block.get("text") or "").strip().lower()
            label = text if text in SURROUND_DIRECTIONS else ""
        elif block.get("type") == "image" and label:
            directions.add(label)
            label = ""
    return frozenset(directions)


def _rewrite_first_text_json(result: dict[str, Any], payload: dict[str, Any]) -> None:
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content[0]["text"] = json.dumps(payload, ensure_ascii=False)


def _extract_robot_status(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = _parse_first_text_json(result)
    if payload is None:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    robot_status = metadata.get("robot_status")
    return robot_status if isinstance(robot_status, dict) else None


def _merge_robot_status_into_situation(
    situation: dict[str, Any],
    robot_status: dict[str, Any],
) -> None:
    battery = robot_status.get("battery")
    if isinstance(battery, dict):
        situation_battery = situation.get("battery")
        if not isinstance(situation_battery, dict) or situation_battery.get("soc_percent") is None:
            situation["battery"] = {
                key: battery.get(key)
                for key in ("state", "soc_percent", "severity", "source")
                if key in battery
            }
    motors = robot_status.get("motors")
    if isinstance(motors, dict):
        situation_motor = situation.get("motor")
        if not isinstance(situation_motor, dict) or situation_motor.get("lost_count") is None:
            situation["motor"] = {
                key: motors.get(key)
                for key in ("state", "lost_count", "max_temperature_c", "severity", "source")
                if key in motors
            }


def _extract_battery_percent(result: dict[str, Any]) -> float | None:
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, dict):
        return None
    text = first.get("text")
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return None
    robot_status = metadata.get("robot_status")
    if not isinstance(robot_status, dict):
        return None
    battery = robot_status.get("battery")
    if not isinstance(battery, dict):
        return None
    soc = battery.get("soc_percent")
    try:
        return float(soc)
    except (TypeError, ValueError):
        return None
