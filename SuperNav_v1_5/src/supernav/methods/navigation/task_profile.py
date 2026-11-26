"""The experiment's navigation contract, shared by prompts and platform binding.

``benchmark_profile`` is the authoritative public declaration. Backend process
flags are compiled from it and must not define a second, independent contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class NavigationTaskProfile:
    name: str

    @property
    def is_global(self) -> bool:
        return self.name == "global_task"

    @property
    def metrics_profile(self) -> str:
        return "global_task" if self.is_global else ""


def resolve_task_profile(config: Mapping[str, Any]) -> NavigationTaskProfile:
    value = config.get("benchmark_profile")
    if value is None or value == "":
        value = "standard"
    if value not in ("standard", "global_task"):
        raise ValueError(
            f"Unsupported benchmark_profile={value!r}; expected 'standard' or 'global_task'"
        )
    return NavigationTaskProfile(str(value))
