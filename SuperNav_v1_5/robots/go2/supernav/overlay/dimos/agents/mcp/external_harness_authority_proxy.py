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

import base64
import copy
from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Any
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage, ImageDraw, ImageFont
import requests
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from dimos.agents.harness.go2_graph_memory import (
    GRAPH_MEMORY_TOOL_NAMES,
    Go2GraphMemoryStore,
    graph_memory_tool_specs,
)
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
GO2_NAVIGATION_MAP_TOOL_NAMES = frozenset(
    {
        "go2_relocalization_status",
        "go2_known_slam_map",
        "go2_navigation_map_context",
    }
)
GO2_MEMORY_NAV_TOOL_NAME = "go2_navigate_to_memory_node"
GO2_RELOCALIZATION_SAVE_TOOL_NAME = "go2_save_relocalization_bundle"
GO2_RELOCALIZATION_PRIOR_TOOL_NAME = "go2_report_relocalization_prior"
GO2_SLAM_MAP_SNAPSHOT_TOOL_NAME = "go2_slam_map_snapshot"
GO2_NAVIGATION_EPISODE_START_TOOL_NAME = "go2_start_navigation_episode"
OPERATOR_ONLY_TOOL_NAMES = frozenset(
    {
        "reset_pixel_goal_map",
        GO2_RELOCALIZATION_PRIOR_TOOL_NAME,
        GO2_SLAM_MAP_SNAPSHOT_TOOL_NAME,
    }
)
AGENT_DISABLED_TOOL_NAMES = frozenset(
    {
        "pixelgoal_stop",
        "stop_movement",
    }
)
PIXELGOAL_COMPLETION_TIMEOUT_S = 300.0
PIXELGOAL_POLL_INTERVAL_S = 1.0
SURROUND_DIRECTIONS = frozenset({"front", "right", "back", "left"})
AGENT_HIDDEN_SLAM_MAP_FIELDS = frozenset(
    {
        "grid_ts",
        "map_frame",
        "slam_area_memory_names",
    }
)
AGENT_HIDDEN_SLAM_AREA_FIELDS = frozenset(
    {
        "bbox_grid",
        "centroid_grid",
        "grid_ts",
        "map_frame",
        "target_grid",
        "target_slam_pose",
    }
)


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
        graph_memory_path: Path | None = None,
        slam_status_path: Path | None = None,
        initial_slam_status_path: Path | None = None,
        initial_relocalization_bundle: Path | None = None,
        url: str = "memory://external-harness-authority-proxy",
    ) -> None:
        self.upstream = upstream
        self.envelope = envelope
        self.battery_gate = battery_gate or BatteryGatePolicy()
        self.graph_memory = (
            Go2GraphMemoryStore(graph_memory_path) if graph_memory_path is not None else None
        )
        self.url = url
        self._tools_by_name: dict[str, dict[str, Any]] = {}
        self._latest_slam_pose: dict[str, Any] | None = None
        self._latest_slam_map_metadata: dict[str, Any] | None = None
        self._latest_slam_map_image_block: dict[str, Any] | None = None
        self._slam_status_path = slam_status_path
        self._initial_slam_status_path = initial_slam_status_path
        self._initial_relocalization_bundle = initial_relocalization_bundle
        self._relocalization_bundle_context = _load_relocalization_bundle_context(
            initial_relocalization_bundle
        )
        self._loaded_slam_status = self._load_initial_slam_status(initial_slam_status_path)
        self._save_slam_status()

    def initialize(self) -> dict[str, Any]:
        return self.upstream.initialize()

    def list_tools(self) -> list[dict[str, Any]]:
        tools = self.upstream.list_tools()
        self._tools_by_name = {str(tool.get("name")): tool for tool in tools if tool.get("name")}
        visible_tools = [
            self._agent_facing_tool(tool) for tool in tools if self._tool_visible(tool)
        ]
        if self.graph_memory is not None:
            visible_tools.extend(graph_memory_tool_specs())
        if (
            self.graph_memory is not None
            or self._slam_status_path is not None
            or self._latest_slam_map_metadata is not None
        ):
            visible_tools.extend(_navigation_map_tool_specs())
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
        if name in AGENT_DISABLED_TOOL_NAMES:
            return _text_result(f"Tool '{name}' refused: disabled for agent use.")
        if name == GO2_MEMORY_NAV_TOOL_NAME:
            target, error = self._resolve_memory_navigation_target(arguments.get("node_id"))
            if error is not None:
                return _text_result(json.dumps({
                    "success": False,
                    "error": error,
                }))
            arguments = {**arguments, **target}
        if name in OPERATOR_ONLY_TOOL_NAMES:
            return _text_result(f"Tool '{name}' refused: operator-only tool.")
        if name in GRAPH_MEMORY_TOOL_NAMES:
            if self.graph_memory is None:
                return _text_result(
                    json.dumps(
                        {
                            "success": False,
                            "error": "go2 graph memory is not enabled for this run",
                        },
                        ensure_ascii=False,
                    )
                )
            slam_pose = self._latest_slam_pose if name == "go2_apply_graph_mutations" else None
            if name == "go2_query_route_context":
                self._refresh_slam_map_snapshot()
            payload = self.graph_memory.handle_tool(name, arguments, slam_pose=slam_pose)
            if (
                name == "go2_apply_graph_mutations"
                and isinstance(payload, dict)
                and payload.get("success") is True
            ):
                self._save_slam_status()
            if name == "go2_report_graph_position" and isinstance(payload, dict):
                prior_returned_map = False
                prior_result = self._forward_graph_position_prior(
                    arguments.get("id", arguments.get("node_id"))
                )
                if prior_result is not None:
                    payload["relocalization_prior"] = prior_result
                    metadata = prior_result.get("metadata")
                    prior_returned_map = isinstance(metadata, dict) and isinstance(
                        metadata.get("2d_slam_map"), dict
                    )
                    if isinstance(metadata, dict) and isinstance(
                        metadata.get("relocalization_wait"), dict
                    ):
                        payload["relocalization_wait"] = metadata["relocalization_wait"]
                if not prior_returned_map:
                    self._refresh_slam_map_snapshot()
            response = _text_result(json.dumps(payload, ensure_ascii=False))
            return_images = name == "go2_report_graph_position" or bool(
                arguments.get("return_images", True)
            )
            if return_images and name in {
                "go2_inspect_graph_node",
                "go2_query_route_context",
                "go2_report_graph_position",
            }:
                response["content"].extend(self.graph_memory.image_blocks_for_payload(payload))
            if name in {"go2_query_route_context", "go2_report_graph_position"}:
                self._append_cached_slam_map(response, payload)
            return response
        if name in GO2_NAVIGATION_MAP_TOOL_NAMES:
            return _text_result(
                json.dumps(
                    self._navigation_map_tool_payload(name, arguments),
                    ensure_ascii=False,
                )
            )

        uses = self._uses_for_tool(name)
        if not is_movement_tool(name, uses) or is_stop_tool(name):
            result = self.upstream.call_tool(name, arguments, meta=meta or None)
            self._capture_panoramas(result)
            self._cache_navigation_context_from_result(result)
            if name == GO2_NAVIGATION_EPISODE_START_TOOL_NAME:
                self._cache_navigation_episode_reset_from_result(result)
            if name == GO2_RELOCALIZATION_SAVE_TOOL_NAME:
                self._cache_relocalization_bundle_from_result(result)
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
            self._cache_navigation_context_from_result(result)
            if name == "pixelgoal_navigate":
                result = self._wait_for_pixelgoal(result)
            if name == GO2_MEMORY_NAV_TOOL_NAME:
                result = self._append_surround_result(result)
            return self._agent_facing_result(name, result)
        except requests.Timeout as exc:
            if name == "pixelgoal_navigate":
                try:
                    self.upstream.call_tool("pixelgoal_stop", {})
                except Exception:  # noqa: BLE001,S110 — preserve the original timeout result
                    pass
            timeout_result = _movement_timeout_result(
                name,
                arguments,
                envelope=self.envelope,
                exc=exc,
            )
            if name == GO2_MEMORY_NAV_TOOL_NAME:
                timeout_result = self._append_surround_result(timeout_result)
            return self._agent_facing_result(name, timeout_result)

    def _append_surround_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add fresh four-way evidence without changing the movement outcome."""
        try:
            observed = self.upstream.call_tool("observe_surround", {})
        except Exception:  # noqa: BLE001 — navigation result remains authoritative
            return result
        content = result.get("content")
        observed_content = observed.get("content") if isinstance(observed, dict) else None
        if not isinstance(content, list) or not isinstance(observed_content, list):
            self._capture_panoramas(result)
            return result
        merged = dict(result)
        # Keep the navigation result's status JSON and SLAM-map image first;
        # append only the labelled surround images from the fresh observation.
        merged["content"] = [*content, *observed_content[1:]]
        self._capture_panoramas(merged)
        self._cache_navigation_context_from_result(merged)
        return merged

    def _forward_graph_position_prior(self, node_id: Any) -> dict[str, Any] | None:
        clean_id = str(node_id or "").strip()
        if not clean_id or self.graph_memory is None:
            return None
        pose = self.graph_memory.navigation_target(clean_id)
        if not isinstance(pose, dict):
            return {"reported": False, "reason": "graph node has no stored slam_pose"}
        if GO2_RELOCALIZATION_PRIOR_TOOL_NAME not in self._tools_by_name:
            self.list_tools()
        if GO2_RELOCALIZATION_PRIOR_TOOL_NAME not in self._tools_by_name:
            return {"reported": False, "reason": "upstream relocalization-prior tool unavailable"}
        try:
            result = self.upstream.call_tool(
                GO2_RELOCALIZATION_PRIOR_TOOL_NAME,
                {"node_id": clean_id, "slam_pose": pose},
            )
        except Exception as exc:  # noqa: BLE001 - keep graph report useful even if prior fails.
            return {"reported": False, "reason": f"upstream prior failed: {exc}"}
        self._cache_navigation_context_from_result(result)
        parsed = _parse_first_text_json(result)
        if isinstance(parsed, dict):
            return parsed
        return {"reported": True, "result": result}

    def _refresh_slam_map_snapshot(self) -> dict[str, Any] | None:
        if GO2_SLAM_MAP_SNAPSHOT_TOOL_NAME not in self._tools_by_name:
            self.list_tools()
        if GO2_SLAM_MAP_SNAPSHOT_TOOL_NAME not in self._tools_by_name:
            return None
        try:
            result = self.upstream.call_tool(
                GO2_SLAM_MAP_SNAPSHOT_TOOL_NAME,
                {},
            )
        except Exception:  # noqa: BLE001 - graph memory remains useful without a map.
            return None
        self._cache_navigation_context_from_result(result)
        return _parse_first_text_json(result)

    def _append_cached_slam_map(
        self,
        response: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        if self._latest_slam_map_metadata is None:
            return
        payload["2d_slam_map"] = _sanitize_slam_map_metadata_for_agent(
            self._latest_slam_map_metadata
        )
        _rewrite_first_text_json(response, payload)
        if self._latest_slam_map_image_block is not None:
            response.setdefault("content", []).append(
                copy.deepcopy(self._latest_slam_map_image_block)
            )

    def _resolve_memory_navigation_target(
        self, node_id: Any
    ) -> tuple[dict[str, Any] | None, str | None]:
        clean_id = str(node_id or "").strip()
        if not clean_id:
            return None, "memory node ID is required"
        latest = self._latest_slam_map_metadata or {}
        if latest.get("status") != "current":
            status = latest.get("status") or "unavailable"
            return None, f"current SLAM map unavailable (status: {status})"
        frame = latest.get("map_frame")
        robot_pose = latest.get("pose")
        if (
            not isinstance(frame, str)
            or not isinstance(robot_pose, dict)
            or robot_pose.get("frame_id") != frame
        ):
            return None, "current robot pose is not in the SLAM map frame"
        if clean_id.startswith("slam_area_"):
            for row in latest.get("slam_area_nodes", []) or []:
                if isinstance(row, dict) and row.get("id") == clean_id:
                    pose = row.get("target_slam_pose")
                    if isinstance(pose, dict):
                        return {
                            "target_slam_pose": {**pose, "frame_id": frame},
                            "map_frame": frame,
                            "map_sequence": pose.get("sequence"),
                        }, None
            return None, "unknown SLAM area node target"
        if self.graph_memory is None:
            return None, "graph memory is not enabled"
        pose = self.graph_memory.navigation_target(clean_id)
        if not isinstance(pose, dict):
            return None, "unknown or pose-less memory node target"
        if pose.get("frame_id") != frame:
            return None, (
                "memory node pose frame differs from current map "
                "(legacy nodes without a frame must be re-anchored)"
            )
        return {
            "target_slam_pose": pose,
            "map_frame": frame,
            "map_sequence": pose.get("sequence"),
        }, None

    def _tool_visible(self, tool: dict[str, Any]) -> bool:
        name = str(tool.get("name") or "")
        if (
            not name
            or name in AGENT_DISABLED_TOOL_NAMES
            or name in OPERATOR_ONLY_TOOL_NAMES
        ):
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
            self.envelope
            and "pixelgoal_navigate" in self.envelope.allowed_movement_tools
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
                    "images; do not poll pixelgoal_status during normal operation."
                )
        return payload

    def _wait_for_pixelgoal(self, started_result: dict[str, Any]) -> dict[str, Any]:
        started = _parse_first_text_json(started_result)
        if not started or started.get("ok") is not True or started.get("status") != "started":
            return started_result
        goal_id = str(started.get("goal_id") or "")
        deadline = time.monotonic() + PIXELGOAL_COMPLETION_TIMEOUT_S
        while time.monotonic() < deadline:
            status_result = self.upstream.call_tool(
                "pixelgoal_status", {"include_images": False}
            )
            status = _parse_first_text_json(status_result)
            if (
                status
                and status.get("hop_running") is False
                and _pixelgoal_status_goal_id(status) == goal_id
            ):
                final = self.upstream.call_tool(
                    "pixelgoal_status", {"include_images": True}
                )
                _compact_pixelgoal_events(final, goal_id)
                self._capture_panoramas(final)
                self._cache_navigation_context_from_result(final)
                if _surround_image_directions(final) == SURROUND_DIRECTIONS:
                    return final
                observed = self.upstream.call_tool("observe_surround", {})
                content = final.get("content")
                observed_content = observed.get("content")
                if isinstance(content, list) and content and isinstance(observed_content, list):
                    merged = dict(final)
                    merged["content"] = [content[0], *observed_content]
                    self._capture_panoramas(merged)
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

    def _capture_panoramas(self, result: dict[str, Any]) -> None:
        """Persist labeled movement images and expose their run-relative refs."""
        if self.graph_memory is None:
            return
        content = result.get("content")
        if not isinstance(content, list):
            return
        images: dict[str, dict[str, Any]] = {}
        label = ""
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = str(block.get("text") or "").strip().lower()
                label = text if text in SURROUND_DIRECTIONS else ""
            elif block.get("type") == "image" and label:
                images[label] = block
                label = ""
        refs = self.graph_memory.save_panorama(images)
        if not refs:
            return
        payload = _parse_first_text_json(result)
        if payload is None:
            return
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata["pano_images"] = refs
        else:
            payload["pano_images"] = refs
        _rewrite_first_text_json(result, payload)

    def _agent_facing_result(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        if self.envelope is not None and name == "go2_situation":
            result = _overlay_go2_situation_proxy_context(
                result,
                envelope=self.envelope,
                robot_status_result=self._safe_robot_status_result(),
            )
        elif (
            self.envelope is not None
            and name == "server_status"
            and "pixelgoal_navigate" in self.envelope.allowed_movement_tools
        ):
            result = _overlay_pixelgoal_server_status(result, envelope=self.envelope)
        return _sanitize_agent_navigation_result(result)

    def _safe_robot_status_result(self) -> dict[str, Any] | None:
        if "go2_robot_status" not in self._tools_by_name:
            self.list_tools()
        if "go2_robot_status" not in self._tools_by_name:
            return None
        try:
            return self.upstream.call_tool("go2_robot_status", {})
        except Exception:
            return None

    def _navigation_map_tool_payload(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del arguments
        latest_map = _sanitize_slam_map_metadata_for_agent(
            self._latest_slam_map_metadata or {}
        )
        graph_nodes = (
            self.graph_memory.handle_tool("go2_graph_map_nodes", {})
            if self.graph_memory is not None
            else {"success": False, "graph_nodes": [], "anchored_node_count": 0}
        )
        graph_payload = graph_nodes if isinstance(graph_nodes, dict) else {}
        if name == "go2_relocalization_status":
            status = str(latest_map.get("localization_status") or latest_map.get("status") or "")
            if not latest_map:
                status = "unavailable"
            elif status in {"current", "localized", "relocalized"}:
                status = "localized"
            elif status in {"static", "collecting", "matching", "anchoring"}:
                status = status
            else:
                status = "unavailable" if status in {"", "stale"} else status
            return {
                "success": True,
                "status": status,
                "source": "authority_proxy_cached_movement_map_metadata",
                "map_status": latest_map.get("status"),
                "pose_available": self._latest_slam_pose is not None,
                "graph_overlay": bool(latest_map.get("graph_overlay")),
                "loaded_initial_status": self._loaded_slam_status,
                "status_saved": self._slam_status_path is not None,
                "relocalization_bundle": self._relocalization_bundle_context,
            }
        if name == "go2_known_slam_map":
            return {
                "success": True,
                "status": "static" if latest_map else "unavailable",
                "source": "authority_proxy_cached_movement_map_metadata",
                "message": (
                    "Static known-map image is available only when the upstream Go2 "
                    "tool attached one to the latest movement/map result."
                    if latest_map
                    else "No cached known-map image is available yet."
                ),
                "graph_nodes": graph_payload.get("graph_nodes", []),
                "graph_overlay": bool(graph_payload.get("anchored_node_count", 0)),
                "loaded_initial_status": self._loaded_slam_status,
                "status_saved": self._slam_status_path is not None,
                "relocalization_bundle": self._relocalization_bundle_context,
            }
        return {
            "success": True,
            "status": latest_map.get("status") or "unavailable",
            "source": "authority_proxy_cached_movement_map_metadata",
            "latest_2d_slam_map": latest_map,
            "pose_available": self._latest_slam_pose is not None,
            "graph_nodes": graph_payload.get("graph_nodes", []),
            "graph_overlay": bool(
                latest_map.get("graph_overlay") or graph_payload.get("anchored_node_count", 0)
            ),
            "loaded_initial_status": self._loaded_slam_status,
            "status_saved": self._slam_status_path is not None,
            "relocalization_bundle": self._relocalization_bundle_context,
        }

    def _cache_navigation_context_from_result(self, result: dict[str, Any]) -> None:
        payload = _parse_first_text_json(result)
        if payload is None:
            return
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            slam_map = metadata.get("2d_slam_map")
        else:
            slam_map = payload.get("2d_slam_map")
        if not isinstance(slam_map, dict):
            return
        self._latest_slam_map_metadata = dict(slam_map)
        pose = (
            _slam_pose_from_map_metadata(slam_map)
            if slam_map.get("status") == "current"
            else None
        )
        self._latest_slam_pose = pose
        self._overlay_graph_nodes_on_result(result)
        image_block = _find_2d_slam_image_block(result)
        self._latest_slam_map_image_block = (
            copy.deepcopy(image_block) if image_block is not None else None
        )
        self._save_slam_status()

    def _cache_relocalization_bundle_from_result(self, result: dict[str, Any]) -> None:
        payload = _parse_first_text_json(result)
        if payload is None:
            return
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return
        bundle = metadata.get("relocalization_bundle")
        if not isinstance(bundle, dict):
            return
        sanitized = _sanitize_relocalization_bundle_context(bundle)
        metadata["relocalization_bundle"] = sanitized
        _rewrite_first_text_json(result, payload)
        self._relocalization_bundle_context = sanitized
        self._save_slam_status()

    def _cache_navigation_episode_reset_from_result(self, result: dict[str, Any]) -> None:
        payload = _parse_first_text_json(result)
        success = bool(payload.get("success")) if isinstance(payload, dict) else False
        if not success:
            return
        self._latest_slam_map_metadata = {
            "status": "reset",
            "source": "episode_start_reset",
            "reset_at_epoch_s": time.time(),
            "graph_nodes": [],
            "graph_overlay": False,
        }
        self._latest_slam_pose = None
        self._latest_slam_map_image_block = None
        self._save_slam_status()

    def _overlay_graph_nodes_on_result(self, result: dict[str, Any]) -> None:
        if self.graph_memory is None:
            return
        payload = _parse_first_text_json(result)
        if payload is None:
            return
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            slam_map = metadata.get("2d_slam_map")
        else:
            metadata = payload
            slam_map = payload.get("2d_slam_map")
        if not isinstance(slam_map, dict):
            return
        graph_nodes = self.graph_memory.map_nodes()
        image_block = _find_2d_slam_image_block(result)
        displayed = _overlay_graph_badges_on_image_block(
            image_block,
            slam_map,
            graph_nodes,
        )
        if displayed is None:
            displayed = _displayable_graph_nodes(graph_nodes)
        suppressed = {
            str(label)
            for label in (
                (image_block or {}).get("metadata", {}).get(
                    "suppressed_slam_area_labels", []
                )
                if isinstance((image_block or {}).get("metadata", {}), dict)
                else []
            )
        }
        memory_names = _memory_area_name_map(slam_map, graph_nodes)
        suppressed.update(str(next(iter(mapping))) for mapping in memory_names if mapping)
        if suppressed:
            slam_map["slam_area_nodes"] = [
                area
                for area in slam_map.get("slam_area_nodes", []) or []
                if not isinstance(area, dict) or str(area.get("label")) not in suppressed
            ]
        slam_map["graph_nodes"] = displayed
        slam_map["graph_overlay"] = bool(displayed)
        self._latest_slam_map_metadata = dict(slam_map)
        _rewrite_first_text_json(result, payload)

    def _load_initial_slam_status(self, path: Path | None) -> bool:
        if path is None:
            return False
        try:
            payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        slam_map = payload.get("latest_slam_map_metadata")
        if not isinstance(slam_map, dict):
            slam_map = payload.get("latest_2d_slam_map")
        if not isinstance(slam_map, dict):
            slam_map = payload.get("2d_slam_map")
        if not isinstance(slam_map, dict):
            return False
        self._latest_slam_map_metadata = dict(slam_map)
        pose = (
            _slam_pose_from_map_metadata(self._latest_slam_map_metadata)
            if self._latest_slam_map_metadata.get("status") == "current"
            else None
        )
        self._latest_slam_pose = dict(pose) if isinstance(pose, dict) else None
        return True

    def _save_slam_status(self) -> None:
        if self._slam_status_path is None or self._latest_slam_map_metadata is None:
            return
        graph_payload = (
            self.graph_memory.handle_tool("go2_graph_map_nodes", {})
            if self.graph_memory is not None
            else {"success": False, "graph_nodes": [], "anchored_node_count": 0}
        )
        graph_nodes = graph_payload if isinstance(graph_payload, dict) else {}
        payload = {
            "schema_version": 1,
            "saved_at_epoch_s": time.time(),
            "source": "authority_proxy_cached_movement_map_metadata",
            "loaded_initial_status": self._loaded_slam_status,
            "latest_slam_map_metadata": self._latest_slam_map_metadata,
            "latest_slam_pose": self._latest_slam_pose,
            "pose_available": self._latest_slam_pose is not None,
            "graph_nodes": graph_nodes.get("graph_nodes", []),
            "graph_overlay": bool(
                self._latest_slam_map_metadata.get("graph_overlay")
                or graph_nodes.get("anchored_node_count", 0)
            ),
            "relocalization_bundle_path": self._relocalization_bundle_context.get("bundle_path"),
            "relocalization_map_file": self._relocalization_bundle_context.get(
                "global_map_file"
            ),
            "relocalization_map_available": bool(
                self._relocalization_bundle_context.get("map_available")
            ),
            "relocalization_map_point_count": self._relocalization_bundle_context.get(
                "point_count"
            ),
            "relocalization_map_frame_id": self._relocalization_bundle_context.get("frame_id"),
            "relocalization_bundle_saved_at_epoch_s": self._relocalization_bundle_context.get(
                "saved_at_epoch_s"
            ),
        }
        self._slam_status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._slam_status_path.with_suffix(self._slam_status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self._slam_status_path)

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
    graph_memory_path: Path | None = None,
    slam_status_path: Path | None = None,
    initial_slam_status_path: Path | None = None,
    initial_relocalization_bundle: Path | None = None,
) -> FastAPI:
    """Create a Streamable-HTTP MCP proxy app for Codex external harness runs."""

    _publish_caption_route(graph_memory_path)
    proxy = ExternalHarnessAuthorityProxy(
        upstream or McpAdapter(upstream_url),
        envelope=envelope,
        graph_memory_path=graph_memory_path,
        slam_status_path=slam_status_path,
        initial_slam_status_path=initial_slam_status_path,
        initial_relocalization_bundle=initial_relocalization_bundle,
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


def _publish_caption_route(graph_memory_path: Path | None) -> None:
    """Tell the long-lived Go2 worker which operator run owns captions."""
    if graph_memory_path is None:
        return
    logs_dir = graph_memory_path.expanduser().resolve().parent
    run_dir = logs_dir.parent
    operator_root = run_dir.parent
    if logs_dir.name != "logs" or operator_root.name != "operator-console":
        return
    route_file = operator_root / ".active-caption-dir"
    temporary = route_file.with_name(f".{route_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(str(logs_dir / "captions"), encoding="utf-8")
        temporary.replace(route_file)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


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


def _navigation_map_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "go2_relocalization_status",
            "description": "Read cached Go2 known-map relocalization status without raw transforms.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "go2_known_slam_map",
            "description": "Read cached static known-map availability and graph badge context.",
            "inputSchema": {
                "type": "object",
                "properties": {"return_image": {"type": "boolean", "default": True}},
                "additionalProperties": False,
            },
        },
        {
            "name": "go2_navigation_map_context",
            "description": "Read compact current/static navigation-map context and graph badges.",
            "inputSchema": {
                "type": "object",
                "properties": {"return_image": {"type": "boolean", "default": True}},
                "additionalProperties": False,
            },
        },
    ]


def _load_relocalization_bundle_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "map_available": False,
            "status": "not_provided",
        }
    bundle_path = path.expanduser()
    manifest_path = bundle_path / "manifest.json" if bundle_path.is_dir() else bundle_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "provided": True,
            "map_available": False,
            "status": "missing_manifest",
            "bundle_path": str(bundle_path),
        }
    except (OSError, json.JSONDecodeError):
        return {
            "provided": True,
            "map_available": False,
            "status": "invalid_manifest",
            "bundle_path": str(bundle_path),
        }
    if not isinstance(manifest, dict):
        return {
            "provided": True,
            "map_available": False,
            "status": "invalid_manifest",
            "bundle_path": str(bundle_path),
        }
    global_map_file = manifest.get("global_map_file") or manifest.get("global_map_path")
    map_path = Path(str(global_map_file)).expanduser() if global_map_file else None
    if map_path is not None and not map_path.is_absolute():
        map_path = manifest_path.parent / map_path
    map_available = bool(
        manifest.get("status") == "current"
        and map_path is not None
        and map_path.exists()
    )
    return {
        "provided": True,
        "map_available": map_available,
        "status": str(manifest.get("status") or "unavailable"),
        "bundle_path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "global_map_file": str(map_path) if map_path is not None else None,
        "point_count": manifest.get("point_count"),
        "frame_id": manifest.get("frame_id"),
        "saved_at_epoch_s": manifest.get("saved_at_epoch_s"),
        "backend_note": (
            "Use global_map.pc2.lcm as RelocalizationModule.map_file input; "
            "live point-cloud matching still controls acceptance."
        ),
    }


def _sanitize_relocalization_bundle_context(bundle: dict[str, Any]) -> dict[str, Any]:
    """Normalize save-tool metadata while keeping raw poses/transforms out."""

    global_map_file = bundle.get("global_map_file") or bundle.get("global_map_path")
    bundle_path = bundle.get("bundle_path")
    manifest_path = bundle.get("manifest_path")
    return {
        "provided": True,
        "map_available": bool(bundle.get("map_available")),
        "status": str(bundle.get("status") or "unavailable"),
        "reason": bundle.get("reason"),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "global_map_file": str(global_map_file) if global_map_file is not None else None,
        "point_count": bundle.get("point_count"),
        "frame_id": bundle.get("frame_id"),
        "saved_at_epoch_s": bundle.get("saved_at_epoch_s"),
        "costmap_saved": bundle.get("costmap_saved"),
        "global_map_age_s": bundle.get("global_map_age_s"),
        "max_age_s": bundle.get("max_age_s"),
        "backend_note": (
            "Use global_map.pc2.lcm as RelocalizationModule.map_file input; "
            "live point-cloud matching still controls acceptance."
        ),
    }


def _slam_pose_from_map_metadata(slam_map: dict[str, Any]) -> dict[str, Any] | None:
    pose = slam_map.get("pose")
    if not isinstance(pose, dict):
        pose = slam_map.get("robot_pose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position_m") if isinstance(pose.get("position_m"), dict) else pose
    if not isinstance(position, dict):
        return None
    frame_id = pose.get("frame_id")
    if not isinstance(frame_id, str) or frame_id != slam_map.get("map_frame"):
        return None
    try:
        return {
            "sequence": int(slam_map.get("pose_sequence") or slam_map.get("step_count") or 0),
            "x": float(position.get("x")),
            "y": float(position.get("y")),
            "yaw_rad": float(pose.get("yaw_rad")),
            "frame_id": frame_id,
        }
    except (TypeError, ValueError):
        return None


def _find_2d_slam_image_block(result: dict[str, Any]) -> dict[str, Any] | None:
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for block in reversed(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image" and block.get("mimeType") == "image/png":
            return block
    return None


def _overlay_graph_badges_on_image_block(
    image_block: dict[str, Any] | None,
    slam_map: dict[str, Any],
    graph_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if image_block is None:
        return None
    data = image_block.get("data")
    if not isinstance(data, str):
        return None
    crop = slam_map.get("crop_grid_bounds")
    origin = slam_map.get("origin_xy")
    grid_size = slam_map.get("grid_size_cells")
    resolution = slam_map.get("resolution_m_per_cell")
    # PixelGoal's segmented renderer emits the complete occupancy grid rather
    # than the cropped view used by the regular agent-map renderer.
    if not isinstance(crop, dict) and isinstance(grid_size, list) and len(grid_size) >= 2:
        crop = {
            "x_min": 0,
            "y_min": 0,
            "x_max": grid_size[0],
            "y_max": grid_size[1],
        }
    width_px = image_block.get("width_px") or slam_map.get("width_px")
    height_px = image_block.get("height_px") or slam_map.get("height_px")
    if (
        not isinstance(crop, dict)
        or not isinstance(origin, list)
        or len(origin) < 2
        or not isinstance(grid_size, list)
        or len(grid_size) < 2
    ):
        return None
    try:
        image = PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")
        width = int(width_px or image.width)
        height = int(height_px or image.height)
        if image.size != (width, height):
            width, height = image.size
        x0 = float(crop["x_min"])
        y1 = float(crop["y_max"])
        scale_x = width / max(1.0, float(crop["x_max"]) - x0)
        scale_y = height / max(1.0, y1 - float(crop["y_min"]))
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        res = float(resolution)
    except Exception:
        return None
    draw = ImageDraw.Draw(image)
    displayed: list[dict[str, Any]] = []
    occupied: list[tuple[float, float]] = []
    graph_world_positions: list[tuple[float, float]] = []
    radius = max(11, width // 42)
    for node in graph_nodes:
        slam_pose = node.get("slam_pose")
        if not node.get("has_pano_images") or not isinstance(slam_pose, dict):
            continue
        try:
            order_id = int(node["order_id"])
            gx = (float(slam_pose["x"]) - origin_x) / res
            gy = (float(slam_pose["y"]) - origin_y) / res
            px = (gx - x0 + 0.5) * scale_x
            py = (y1 - gy - 0.5) * scale_y
            graph_world_positions.append((float(slam_pose["x"]), float(slam_pose["y"])))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not (0 <= px < width and 0 <= py < height):
            continue
        # Persistent graph markers must stay on their measured SLAM pose. Do
        # not displace them to avoid overlap: the connector line makes a
        # marker look like it belongs to a different area on dense maps.
        marker_px = (px, py)
        occupied.append(marker_px)
        _draw_badge(draw, marker_px, radius, str(order_id))
        displayed.append(
            {
                "order_id": order_id,
                "id": str(node.get("id") or ""),
                "name": str(node.get("name") or ""),
                "kind": str(node.get("kind") or ""),
            }
        )
    area_labels_drawn = False
    suppressed_area_labels: list[str] = []
    area_font = ImageFont.load_default(size=max(18, min(width, height) // 24))
    for area in slam_map.get("slam_area_nodes", []) or []:
        if not isinstance(area, dict) or not area.get("label"):
            continue
        pose = area.get("target_slam_pose")
        if not isinstance(pose, dict):
            continue
        try:
            area_x, area_y = float(pose["x"]), float(pose["y"])
            graph_in_area = False
            bbox = area.get("bbox_grid")
            if isinstance(bbox, list) and len(bbox) == 4:
                row_min, col_min, row_max, col_max = (float(value) for value in bbox)
                for graph_x, graph_y in graph_world_positions:
                    graph_col = (graph_x - origin_x) / res
                    # ``region_labels`` uses the same bottom-origin grid
                    # coordinates as ``OccupancyGrid``. Only the final image
                    # renderer flips Y for display; do not flip it here.
                    graph_row = (graph_y - origin_y) / res
                    if col_min <= graph_col < col_max and row_min <= graph_row < row_max:
                        graph_in_area = True
                        break
            else:
                # Compatibility with older map metadata without a bbox.
                graph_in_area = any(
                    math.hypot(area_x - graph_x, area_y - graph_y) <= 0.75
                    for graph_x, graph_y in graph_world_positions
                )
            if graph_in_area:
                suppressed_area_labels.append(str(area["label"]))
                continue
            gx = (area_x - origin_x) / res
            gy = (area_y - origin_y) / res
            px = (gx - x0 + 0.5) * scale_x
            py = (y1 - gy - 0.5) * scale_y
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not (0 <= px < width and 0 <= py < height):
            continue
        label_px = _displaced_badge_position((px, py), occupied, radius, width, height)
        occupied.append(label_px)
        draw.text(
            label_px,
            str(area["label"]),
            font=area_font,
            fill=(255, 255, 255),
            stroke_width=max(1, radius // 8),
            stroke_fill=(0, 0, 0),
            anchor="mm",
        )
        area_labels_drawn = True
    if not displayed and not area_labels_drawn:
        return []
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_block["data"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_block["width_px"] = image.width
    image_block["height_px"] = image.height
    image_block.setdefault("metadata", {})
    if suppressed_area_labels:
        image_block["metadata"]["suppressed_slam_area_labels"] = suppressed_area_labels
    return displayed


def _displayable_graph_nodes(graph_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    displayed = []
    for node in graph_nodes:
        if not node.get("has_pano_images") or not node.get("slam_pose"):
            continue
        displayed.append(
            {
                "order_id": node.get("order_id"),
                "id": node.get("id"),
                "name": node.get("name"),
                "kind": node.get("kind"),
            }
        )
    return displayed


def _memory_area_name_map(
    slam_map: dict[str, Any],
    graph_nodes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Map suppressed live area labels to the marked memory node names."""
    origin = slam_map.get("origin_xy")
    resolution = slam_map.get("resolution_m_per_cell")
    if not isinstance(origin, list) or len(origin) < 2:
        return []
    try:
        origin_x, origin_y, res = float(origin[0]), float(origin[1]), float(resolution)
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    result: list[dict[str, str]] = []
    for area in slam_map.get("slam_area_nodes", []) or []:
        if not isinstance(area, dict) or not area.get("label"):
            continue
        bbox = area.get("bbox_grid")
        area_pose = area.get("target_slam_pose")
        if not isinstance(area_pose, dict):
            continue
        try:
            area_x, area_y = float(area_pose["x"]), float(area_pose["y"])
        except (KeyError, TypeError, ValueError):
            continue
        match: dict[str, Any] | None = None
        for node in graph_nodes:
            if not node.get("has_pano_images") or not isinstance(node.get("slam_pose"), dict):
                continue
            pose = node["slam_pose"]
            try:
                graph_col = (float(pose["x"]) - origin_x) / res
                graph_row = (float(pose["y"]) - origin_y) / res
                inside = False
                if isinstance(bbox, list) and len(bbox) == 4:
                    row_min, col_min, row_max, col_max = (float(value) for value in bbox)
                    inside = col_min <= graph_col < col_max and row_min <= graph_row < row_max
                else:
                    inside = math.hypot(area_x - float(pose["x"]), area_y - float(pose["y"])) <= 0.75
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if inside:
                match = node
                break
        if match is not None and str(match.get("name") or ""):
            result.append({str(area["label"]): str(match["name"])})
    return result


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: int,
    text: str,
) -> None:
    cx, cy = center
    font = ImageFont.load_default(size=max(18, radius * 2))
    draw.text(
        (cx, cy),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=max(1, radius // 8),
        stroke_fill=(0, 0, 0),
        anchor="mm",
    )


def _displaced_badge_position(
    base: tuple[float, float],
    occupied: list[tuple[float, float]],
    radius: int,
    width: int,
    height: int,
) -> tuple[float, float]:
    min_distance = radius * 2.4
    candidates = [base]
    for ring in range(1, 4):
        distance = ring * radius * 2.6
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            angle = math.radians(angle_deg)
            candidates.append(
                (
                    base[0] + math.cos(angle) * distance,
                    base[1] + math.sin(angle) * distance,
                )
            )
    for candidate in candidates:
        cx = min(max(radius + 2, candidate[0]), width - radius - 2)
        cy = min(max(radius + 2, candidate[1]), height - radius - 2)
        if all(math.hypot(cx - ox, cy - oy) >= min_distance for ox, oy in occupied):
            return (cx, cy)
    return (
        min(max(radius + 2, base[0]), width - radius - 2),
        min(max(radius + 2, base[1]), height - radius - 2),
    )


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
            "requested motion. Check go2_situation/observe, do not retry while motion may still "
            "be active, and report the condition to the operator."
        ),
        "error_code": "MOVEMENT_COMPLETION_TIMEOUT_AFTER_DISPATCH",
        "metadata": {
            "tool_name": name,
            "requested_arguments": arguments,
            "command_may_be_active": True,
            "upstream_exception": exc.__class__.__name__,
            "recommended_next_action": "check_situation_observe_then_report_operator",
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


def _sanitize_slam_map_metadata_for_agent(slam_map: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in slam_map.items()
        if key not in AGENT_HIDDEN_SLAM_MAP_FIELDS
    }
    areas = slam_map.get("slam_area_nodes")
    if isinstance(areas, list):
        sanitized["slam_area_nodes"] = [
            {
                key: value
                for key, value in area.items()
                if key not in AGENT_HIDDEN_SLAM_AREA_FIELDS
            }
            if isinstance(area, dict)
            else area
            for area in areas
        ]
    return sanitized


def _sanitize_agent_navigation_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_first_text_json(result)
    if payload is None:
        return result
    metadata = payload.get("metadata")
    container = metadata if isinstance(metadata, dict) else payload
    slam_map = container.get("2d_slam_map")
    if not isinstance(slam_map, dict):
        return result
    container["2d_slam_map"] = _sanitize_slam_map_metadata_for_agent(slam_map)
    _rewrite_first_text_json(result, payload)
    return result


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
