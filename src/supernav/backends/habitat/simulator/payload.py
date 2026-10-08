"""Shared simulator payload coercion; independent of navigation loop state."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np

from habitat_contract.navigation_state import HabitatAdapterError


class SimulatorPayloadMixin:
    @staticmethod
    def _maybe_update_int(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = SimulatorPayloadMixin._coerce_int(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _maybe_update_float(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = SimulatorPayloadMixin._coerce_float(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _maybe_update_bool(
        target: MutableMapping[str, Any],
        key: str,
        source: Mapping[str, Any],
    ) -> None:
        if key in source:
            target[key] = SimulatorPayloadMixin._coerce_bool(
                source[key], field_name=f"payload.{key}"
            )

    @staticmethod
    def _coerce_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise HabitatAdapterError(f'Field "{field_name}" must be int, got bool')
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HabitatAdapterError(f'Field "{field_name}" must be int') from exc

    @staticmethod
    def _coerce_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise HabitatAdapterError(
                f'Field "{field_name}" must be float, got bool'
            )
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise HabitatAdapterError(f'Field "{field_name}" must be float') from exc

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, np.integer)):
            return bool(value)
        raise HabitatAdapterError(f'Field "{field_name}" must be bool')

    @staticmethod
    def _coerce_float_list(value: Any, expected_len: int, field_name: str) -> list[float]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != expected_len:
            raise HabitatAdapterError(
                f'Field "{field_name}" must be list[float] length {expected_len}'
            )
        return [
            SimulatorPayloadMixin._coerce_float(item, field_name=field_name)
            for item in value
        ]
