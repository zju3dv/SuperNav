"""Adapters from continuous LocalNav commands to existing runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from supernav.methods.localnav.contracts import LocalNavRuntimeError, VelocityAction


@dataclass(frozen=True)
class HabitatActionAdapterConfig:
    """Habitat's current discrete action resolution and LocalNav cadence."""

    control_dt_s: float = 0.5
    forward_step_m: float = 0.25
    turn_step_degrees: float = 10.0
    zero_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        for name, value in (
            ("control_dt_s", self.control_dt_s),
            ("forward_step_m", self.forward_step_m),
            ("turn_step_degrees", self.turn_step_degrees),
            ("zero_epsilon", self.zero_epsilon),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite")


class HabitatActionAdapter:
    """Quantize one continuous ``[v, omega]`` command for ``step_and_capture``.

    Positive angular velocity means left turn; negative means right turn.
    Habitat currently exposes fixed 0.25 m and 10 degree primitives.  A
    command containing both rotation and translation is approximated as a
    turn followed by a move, so the policy contract can remain continuous
    while the existing simulator bridge stays unchanged.
    """

    def __init__(self, config: "HabitatActionAdapterConfig | None" = None) -> None:
        self._config = config or HabitatActionAdapterConfig()

    @property
    def config(self) -> HabitatActionAdapterConfig:
        return self._config

    def to_payloads(self, action: VelocityAction) -> tuple[dict[str, Any], ...]:
        if not isinstance(action, VelocityAction):
            raise LocalNavRuntimeError(
                "invalid_action", "Habitat adapter requires a VelocityAction"
            )
        payloads: list[dict[str, Any]] = []

        yaw_degrees = math.degrees(action.angular_rps * self._config.control_dt_s)
        if abs(action.angular_rps) > self._config.zero_epsilon:
            quantized_degrees = self._quantize(yaw_degrees, self._config.turn_step_degrees)
            if quantized_degrees > 0.0:
                payloads.append(
                    {
                        "action": "turn_left" if action.angular_rps > 0 else "turn_right",
                        "degrees": quantized_degrees,
                        "include_metrics": True,
                    }
                )

        distance_m = action.linear_mps * self._config.control_dt_s
        if abs(action.linear_mps) > self._config.zero_epsilon:
            quantized_distance = self._quantize(distance_m, self._config.forward_step_m)
            if quantized_distance > 0.0:
                payloads.append(
                    {
                        "action": "move_forward" if action.linear_mps > 0 else "move_backward",
                        "distance": quantized_distance,
                        "include_metrics": True,
                    }
                )
        return tuple(payloads)

    @staticmethod
    def _quantize(value: float, step: float) -> float:
        counts = int(round(abs(value) / step))
        return counts * step
