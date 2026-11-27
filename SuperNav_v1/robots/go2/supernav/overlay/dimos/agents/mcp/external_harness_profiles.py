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

"""Named validation profiles for external MCP harness bring-up."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalHarnessValidationProfile:
    """A named bundle of ``dimos mcp validate-harness`` options."""

    name: str
    description: str
    options: Mapping[str, Any]

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable profile summary."""

        return {
            "name": self.name,
            "description": self.description,
            "options": dict(self.options),
        }


EXTERNAL_HARNESS_VALIDATION_PROFILES: dict[str, ExternalHarnessValidationProfile] = {
    "readonly": ExternalHarnessValidationProfile(
        name="readonly",
        description=(
            "Read-only external harness surface: inspection/diagnostics visible, "
            "runtime idle, movement authority denied, and hidden movement dispatch refused."
        ),
        options={
            "expected_tool_scope": "readonly",
            "require_visual_readback": False,
            "require_hidden_movement_refusal": True,
            "require_idle_runtime": True,
        },
    ),
    "readonly-visual": ExternalHarnessValidationProfile(
        name="readonly-visual",
        description=(
            "Read-only external harness surface plus read_image artifact readback. "
            "Use with --artifact-id or --latest-image-artifact."
        ),
        options={
            "expected_tool_scope": "readonly",
            "require_visual_readback": True,
            "require_hidden_movement_refusal": True,
            "require_idle_runtime": True,
        },
    ),
    "vision-lidar-supervised": ExternalHarnessValidationProfile(
        name="vision-lidar-supervised",
        description=(
            "Supervised low-speed bring-up surface: task-authority local motion "
            "and diagnostics visible, runtime idle, prebuilt map/frontier/semantic "
            "tools hidden."
        ),
        options={
            "expected_tool_scope": "vision_lidar_only",
            "require_visual_readback": False,
            "require_movement_authority_refusal": True,
            "require_hidden_movement_refusal": False,
            "require_idle_runtime": True,
        },
    ),
}


def available_external_harness_profiles() -> tuple[ExternalHarnessValidationProfile, ...]:
    """Return profiles in CLI display order."""

    return tuple(
        EXTERNAL_HARNESS_VALIDATION_PROFILES[name]
        for name in sorted(EXTERNAL_HARNESS_VALIDATION_PROFILES)
    )


def external_harness_profile(name: str) -> ExternalHarnessValidationProfile:
    """Resolve a named external harness validation profile."""

    try:
        return EXTERNAL_HARNESS_VALIDATION_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXTERNAL_HARNESS_VALIDATION_PROFILES))
        raise ValueError(
            f"Unknown external harness validation profile '{name}'. Available: {available}"
        ) from exc


def merge_external_harness_validation_options(
    profile_name: str,
    explicit_options: Mapping[str, Any],
) -> tuple[dict[str, Any], ExternalHarnessValidationProfile | None]:
    """Merge a named profile with explicit CLI options."""

    profile = external_harness_profile(profile_name) if profile_name else None
    merged: dict[str, Any] = {
        "expected_tool_scope": "readonly",
        "require_visual_readback": False,
        "require_movement_authority_refusal": False,
        "require_hidden_movement_refusal": True,
        "require_idle_runtime": True,
    }
    if profile is not None:
        merged.update(profile.options)
    expected_tool_scope = explicit_options.get("expected_tool_scope")
    if expected_tool_scope is not None:
        merged["expected_tool_scope"] = expected_tool_scope
        if expected_tool_scope != "readonly":
            merged["require_hidden_movement_refusal"] = False
    if explicit_options.get("require_visual_readback"):
        merged["require_visual_readback"] = True
    if explicit_options.get("require_movement_authority_refusal"):
        merged["require_movement_authority_refusal"] = True
    if explicit_options.get("require_hidden_movement_refusal"):
        merged["require_hidden_movement_refusal"] = True
    if explicit_options.get("require_idle_runtime"):
        merged["require_idle_runtime"] = True
    return merged, profile
