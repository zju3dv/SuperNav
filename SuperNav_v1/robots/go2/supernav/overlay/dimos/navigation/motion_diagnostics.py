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

"""Motion safety and blocker diagnostics shared by navigation-facing tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

MotionBlocker = Literal[
    "runtime_policy",
    "tool_scope",
    "capability_busy",
    "planner_no_path",
    "local_obstacle_stop",
    "unitree_protection",
    "posture_not_ready",
    "no_odom",
    "comms_error",
    "battery_safety",
    "unknown",
    "none",
]

_RECOMMENDED_ACTIONS: dict[str, str] = {
    "none": "continue",
    "runtime_policy": "ask human to change movement policy or run a read-only inspection",
    "tool_scope": "choose an allowed local/inspection tool or restart with a broader tool scope",
    "capability_busy": "call the relevant stop tool, wait, then retry",
    "planner_no_path": "rotate/observe, choose a closer target, or ask the human for a new target",
    "local_obstacle_stop": "stop, rotate/observe, then retry a shorter or lower-speed local move",
    "unitree_protection": "ask the human to inspect robot protection state before retrying",
    "posture_not_ready": "recover posture before moving",
    "no_odom": "wait for odometry or restart the localization/navigation stack",
    "comms_error": "check robot/network connection before retrying",
    "battery_safety": "stop motion and charge or power down the robot",
    "unknown": "stop, inspect status, and ask the human if motion remains blocked",
}


def classify_motion_blocker(
    *,
    last_failure: str | None = None,
    last_stop_reason: str | None = None,
    unitree_protection: str | Mapping[str, Any] | None = None,
    battery_status: str | Mapping[str, Any] | None = None,
    runtime_allowed: bool | None = None,
    scope_allowed: bool | None = None,
    capability_busy: bool = False,
) -> MotionBlocker:
    """Classify the strongest known reason a motion command cannot continue."""

    if runtime_allowed is False:
        return "runtime_policy"
    if scope_allowed is False:
        return "tool_scope"
    if capability_busy:
        return "capability_busy"
    if _unitree_protection_triggered(unitree_protection):
        return "unitree_protection"
    if _battery_safety_triggered(battery_status):
        return "battery_safety"
    if last_failure == "NO_ODOM":
        return "no_odom"
    if last_failure in {"NO_PATH", "NO_SAFE_GOAL"}:
        return "planner_no_path"
    if last_failure == "OBSTACLE" or last_stop_reason == "obstacle_found":
        return "local_obstacle_stop"
    if last_failure == "EXECUTION_FAILED":
        return "comms_error"
    if last_failure == "RETRY_LIMIT_EXCEEDED":
        if last_stop_reason == "obstacle_found":
            return "local_obstacle_stop"
        return "unknown"
    if last_failure in {"CANCELLED", "TIMEOUT"}:
        return "unknown"
    if last_failure:
        return "unknown"
    return "none"


def recommended_next_action(blocker: str) -> str:
    """Return the recommended next action for a blocker classification."""

    return _RECOMMENDED_ACTIONS.get(blocker, _RECOMMENDED_ACTIONS["unknown"])


def build_motion_diagnostics(
    *,
    navigation_state: str | None = None,
    last_failure: str | None = None,
    last_stop_reason: str | None = None,
    last_stop_time: float | None = None,
    planner_goal_result: dict[str, Any] | None = None,
    obstacle_avoidance_config: bool | None = None,
    runtime_allowed: bool | None = None,
    scope_allowed: bool | None = None,
    low_level_command_sent: bool | None = None,
    robot_acknowledged: bool | None = None,
    measured_result: dict[str, Any] | None = None,
    posture_mode: str = "unknown",
    unitree_protection: str | Mapping[str, Any] = "unknown",
    battery_status: str | Mapping[str, Any] = "unknown",
    motor_status: str | Mapping[str, Any] = "unknown",
    robot_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable motion diagnostics payload."""

    robot_status_payload = dict(robot_status) if isinstance(robot_status, Mapping) else None
    if robot_status_payload is not None:
        if battery_status == "unknown" and isinstance(robot_status_payload.get("battery"), Mapping):
            battery_status = robot_status_payload["battery"]
        if motor_status == "unknown" and isinstance(robot_status_payload.get("motors"), Mapping):
            motor_status = robot_status_payload["motors"]
        if unitree_protection == "unknown" and isinstance(
            robot_status_payload.get("unitree_protection"), Mapping
        ):
            unitree_protection = robot_status_payload["unitree_protection"]

    unitree_protection_payload = _unitree_protection_payload(unitree_protection)
    battery_status_payload = _battery_status_payload(battery_status)
    motor_status_payload = _motor_status_payload(motor_status)
    blocker = classify_motion_blocker(
        last_failure=last_failure,
        last_stop_reason=last_stop_reason,
        unitree_protection=unitree_protection_payload,
        battery_status=battery_status_payload,
        runtime_allowed=runtime_allowed,
        scope_allowed=scope_allowed,
    )
    local_obstacle_state = "triggered" if blocker == "local_obstacle_stop" else "unknown"
    return {
        "posture_mode": posture_mode,
        "collision_obstacle_protection": {
            "enabled": None,
            "state": local_obstacle_state,
            "last_trigger_time": last_stop_time if blocker == "local_obstacle_stop" else None,
            "source": "dimos_local_planner" if blocker == "local_obstacle_stop" else "unknown",
            "obstacle_avoidance_config": obstacle_avoidance_config,
        },
        "unitree_protection": unitree_protection_payload,
        "battery_status": battery_status_payload,
        "motor_status": motor_status_payload,
        "robot_status": robot_status_payload,
        "command_acceptance": {
            "runtime_allowed": runtime_allowed,
            "scope_allowed": scope_allowed,
            "planner_accepted": (
                planner_goal_result.get("accepted")
                if isinstance(planner_goal_result, dict)
                else None
            ),
            "low_level_command_sent": low_level_command_sent,
            "robot_acknowledged": robot_acknowledged,
        },
        "measured_result": measured_result
        or {
            "odom_delta": None,
            "yaw_delta": None,
            "elapsed_s": None,
            "physically_satisfied": None,
        },
        "navigation_state": navigation_state,
        "last_failure": last_failure,
        "last_stop_reason": last_stop_reason,
        "planner_goal_result": planner_goal_result,
        "blocker": blocker,
        "recommended_next_action": recommended_next_action(blocker),
    }


def _unitree_protection_payload(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional Unitree firmware protection data without inventing live state."""

    if isinstance(value, Mapping):
        payload = dict(value)
        state = str(payload.get("state") or "unknown")
    else:
        payload = {}
        state = str(value or "unknown")

    payload["state"] = state
    payload.setdefault("last_trigger_time", None)
    payload.setdefault("source", "unitree_firmware")
    if state == "unknown":
        payload.setdefault("available", False)
        payload.setdefault("needs_live_sdk_probe", True)
        payload.setdefault("reason", "unitree_firmware_status_unavailable")
    else:
        payload.setdefault("available", True)
        payload.setdefault("needs_live_sdk_probe", False)
    return payload


def _unitree_protection_triggered(value: str | Mapping[str, Any] | None) -> bool:
    payload = _unitree_protection_payload(value)
    state = payload.get("state")
    return state in {"triggered", "active", "blocked", "protective_stop"}


def _battery_status_payload(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        state = str(payload.get("state") or "unknown")
    else:
        payload = {}
        state = str(value or "unknown")
    payload["state"] = state
    payload.setdefault("source", "unknown")
    payload.setdefault("soc_percent", None)
    return payload


def _battery_safety_triggered(value: str | Mapping[str, Any] | None) -> bool:
    payload = _battery_status_payload(value)
    state = payload.get("state")
    return state in {"critical", "safety_stop", "shutdown"}


def _motor_status_payload(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        state = str(payload.get("state") or "unknown")
    else:
        payload = {}
        state = str(value or "unknown")

    lost_count = _int_or_none(payload.get("lost_count"))
    max_temperature_c = _float_or_none(payload.get("max_temperature_c"))
    warning = bool((lost_count or 0) > 0)
    severity = "warning" if warning else "normal"
    if state == "unknown" and (lost_count is not None or max_temperature_c is not None):
        state = "warning" if warning else "normal"

    payload["state"] = state
    payload["lost_count"] = lost_count
    payload["max_temperature_c"] = max_temperature_c
    payload["warning"] = warning
    payload["severity"] = severity
    payload.setdefault("source", "unitree_lowstate" if state != "unknown" else "unknown")
    return payload


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
