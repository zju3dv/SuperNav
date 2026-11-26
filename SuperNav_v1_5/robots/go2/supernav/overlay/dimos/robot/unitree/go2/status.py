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

"""Go2 low-level status summaries used by agent-facing diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_BATTERY_LOW_SOC = 25.0
_BATTERY_CRITICAL_SOC = 15.0


def unavailable_go2_status(reason: str) -> dict[str, Any]:
    """Return a stable status payload when no Unitree low-state sample is available."""

    return {
        "available": False,
        "source": "unitree_lowstate",
        "reason": reason,
        "battery": {"state": "unknown", "soc_percent": None},
        "motors": {"count": 0, "lost_count": None, "max_temperature_c": None},
        "unitree_protection": {
            "state": "unknown",
            "available": False,
            "needs_live_sdk_probe": True,
            "reason": "unitree_firmware_status_unavailable",
            "source": "unitree_lowstate",
        },
    }


def summarize_go2_low_state(message: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize a Unitree low-state message without exposing raw bulky fields."""

    if not isinstance(message, Mapping):
        return unavailable_go2_status("lowstate_not_received")

    data = _mapping(message.get("data"))
    if not data:
        return unavailable_go2_status("lowstate_missing_data")

    bms = _mapping(data.get("bms_state"))
    motors = _sequence(data.get("motor_state"))
    motor_temperatures = [
        temperature
        for motor in motors
        if isinstance(motor, Mapping)
        for temperature in [_number(motor.get("temperature"))]
        if temperature is not None
    ]
    lost_count = sum(
        1
        for motor in motors
        if isinstance(motor, Mapping) and _number(motor.get("lost"), default=0.0) > 0
    )
    foot_force = [_number(force) for force in _sequence(data.get("foot_force"))]
    soc = _number(bms.get("soc"))

    return {
        "available": True,
        "source": "unitree_lowstate",
        "topic": message.get("topic"),
        "battery": {
            "state": _battery_state(soc),
            "soc_percent": soc,
            "current_ma": _number(bms.get("current")),
            "cycle_count": _number(bms.get("cycle")),
            "voltage_v": _number(data.get("power_v")),
            "bq_ntc_c": [_number(value) for value in _sequence(bms.get("bq_ntc"))],
            "mcu_ntc_c": [_number(value) for value in _sequence(bms.get("mcu_ntc"))],
        },
        "motors": {
            "count": len(motors),
            "lost_count": lost_count,
            "max_temperature_c": max(motor_temperatures) if motor_temperatures else None,
        },
        "contact": {
            "foot_force": foot_force,
        },
        "imu": {
            "rpy": [
                _number(value) for value in _sequence(_mapping(data.get("imu_state")).get("rpy"))
            ],
        },
        "unitree_protection": {
            "state": "unknown",
            "available": False,
            "needs_live_sdk_probe": True,
            "reason": "unitree_firmware_status_unavailable",
            "source": "unitree_lowstate",
        },
    }


def _battery_state(soc: float | None) -> str:
    if soc is None:
        return "unknown"
    if soc <= _BATTERY_CRITICAL_SOC:
        return "critical"
    if soc <= _BATTERY_LOW_SOC:
        return "low"
    return "normal"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return []


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
