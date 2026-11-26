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

"""Shared context carried with asynchronous tool-status updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.agents.harness.movement_lease import MovementLease, MovementLeaseTicket
from dimos.agents.runtime_policy import AgentRuntimePolicy
from dimos.navigation.motion_diagnostics import build_motion_diagnostics

TOOL_STATUS_CONTEXT_META_KEY = "dimos/status"


def build_tool_status_context(
    *,
    policy: AgentRuntimePolicy,
    client_id: str,
    request_meta: Mapping[str, Any] | None = None,
    runtime_config: Any | None = None,
    movement_ticket: MovementLeaseTicket | None = None,
    movement_lease: MovementLease | None = None,
) -> dict[str, Any]:
    """Build a compact status snapshot for UI panels and traces.

    The snapshot intentionally mirrors MCP runtime status at a smaller scale:
    policy/tool scope, active movement authority, protection diagnostics, and
    task identity. Producers may merge tool-specific fields such as a navigation
    request as the background task progresses.
    """

    request_meta = request_meta or {}
    diagnostics = _protection_diagnostics(runtime_config)
    return {
        "policy": {
            "profile": policy.profile,
            "movement_mode": policy.movement_mode,
            "tool_scope": policy.tool_scope,
            "admin_override": policy.tool_policy_admin_override,
        },
        "movement_authority": {
            "client_id": client_id,
            "authorization": _optional_str(request_meta.get("dimos/movement_authorization")),
            "confirmed": bool(request_meta.get("dimos/movement_confirmed")),
            "task_id": _optional_str(request_meta.get("dimos/task_id")),
            "task_goal": _optional_str(request_meta.get("dimos/task_goal")),
            "lease_id": movement_ticket.lease_id if movement_ticket is not None else None,
            "epoch": movement_ticket.epoch if movement_ticket is not None else None,
            "remaining": movement_lease.remaining() if movement_lease is not None else None,
        },
        "protection": {
            "blocker": diagnostics.get("blocker"),
            "recommended_next_action": diagnostics.get("recommended_next_action"),
            "collision_obstacle_protection": diagnostics.get("collision_obstacle_protection"),
            "unitree_protection": diagnostics.get("unitree_protection"),
            "battery_status": diagnostics.get("battery_status"),
        },
        "navigation_task": None,
    }


def merge_tool_status_context(
    base: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge base call context with one tool-update metadata payload."""

    if not base and not update:
        return None
    merged = _dict_copy(base)
    update_payload = _dict_copy(update)
    embedded = update_payload.pop(TOOL_STATUS_CONTEXT_META_KEY, None)
    if isinstance(embedded, Mapping):
        _merge_mapping(merged, embedded)
    _merge_mapping(merged, update_payload)
    return merged


def tool_status_context_from_meta(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract a status context from JSON-RPC ``params._meta``."""

    if not isinstance(meta, Mapping):
        return None
    context = meta.get(TOOL_STATUS_CONTEXT_META_KEY)
    return dict(context) if isinstance(context, Mapping) else None


def _protection_diagnostics(runtime_config: Any | None) -> dict[str, Any]:
    obstacle_avoidance = (
        getattr(runtime_config, "obstacle_avoidance", None) if runtime_config is not None else None
    )
    return build_motion_diagnostics(obstacle_avoidance_config=obstacle_avoidance)


def _merge_mapping(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        key_str = str(key)
        current = target.get(key_str)
        if isinstance(value, Mapping) and isinstance(current, dict) and key != "navigation_task":
            _merge_mapping(current, value)
        else:
            target[key_str] = value


def _dict_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
