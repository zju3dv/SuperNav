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

"""Repeatable Go2 harness gate checklists.

These profiles connect the independently validated MCP, visual, and trace-audit
gates into a small operator-facing sequence. They intentionally render commands
instead of starting robot motion; real Go2 movement remains a human-controlled
validation step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateStepKind = Literal[
    "start_runtime",
    "mcp_validation",
    "visual_preflight",
    "provider_validation",
    "trace_audit",
    "manual_hardware",
]


@dataclass(frozen=True)
class Go2HarnessGateStep:
    """One step in a Go2 harness gate profile."""

    name: str
    kind: GateStepKind
    description: str
    commands: tuple[str, ...] = ()
    requires_robot: bool = False
    requires_running_dimos: bool = False
    requires_provider: bool = False

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Go2HarnessGateProfile:
    """A named gate sequence for a Go2 harness stage."""

    name: str
    description: str
    steps: tuple[Go2HarnessGateStep, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.model_dump() for step in self.steps],
        }

    def render_text(self) -> str:
        lines = [f"{self.name}: {self.description}"]
        for index, step in enumerate(self.steps, start=1):
            flags: list[str] = []
            if step.requires_robot:
                flags.append("robot")
            if step.requires_running_dimos:
                flags.append("dimos-running")
            if step.requires_provider:
                flags.append("provider")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            lines.append(f"{index}. {step.name}{suffix}: {step.description}")
            for command in step.commands:
                lines.append(f"   $ {command}")
        return "\n".join(lines)


_READONLY_START = (
    "uv run dimos --robot-ip <GO2_IP> --agent-profile supervised_live "
    "--agent-tool-scope readonly --agent-movement-mode deny "
    "run unitree-go2-agentic --daemon"
)
_SUPERVISED_START = (
    "uv run dimos --robot-ip <GO2_IP> --agent-profile supervised_live "
    "--agent-tool-scope vision_lidar_only --agent-movement-mode task_authority "
    "--agent-dangerous-skills deny run unitree-go2-agentic --daemon"
)
_PROVIDER_STATUS = "uv run dimos agent-provider-status --require-native"


GO2_HARNESS_GATE_PROFILES: dict[str, Go2HarnessGateProfile] = {
    "readonly-prelive": Go2HarnessGateProfile(
        name="readonly-prelive",
        description=(
            "Read-only MCP/visual bring-up before allowing any external harness "
            "or native agent movement authority."
        ),
        steps=(
            Go2HarnessGateStep(
                name="check-agent-provider-readiness",
                kind="provider_validation",
                description=(
                    "Print display-safe native-agent model/provider readiness "
                    "before starting the robot runtime."
                ),
                commands=(_PROVIDER_STATUS,),
            ),
            Go2HarnessGateStep(
                name="start-readonly-runtime",
                kind="start_runtime",
                description="Start Go2 agentic runtime with inspection-only tools.",
                commands=(_READONLY_START,),
                requires_robot=True,
            ),
            Go2HarnessGateStep(
                name="validate-readonly-mcp",
                kind="mcp_validation",
                description="Verify external clients can inspect but not move.",
                commands=("uv run dimos mcp validate-harness --profile readonly",),
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="validate-visual-readback",
                kind="mcp_validation",
                description="After observe creates an artifact, verify MCP read_image payloads.",
                commands=(
                    'uv run dimos mcp call observe --json-args "{}"',
                    "uv run dimos mcp validate-harness "
                    "--profile readonly-visual --latest-image-artifact",
                    "uv run dimos agent-vision-roundtrip --preflight --latest-image-artifact",
                ),
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="validate-provider-visual-roundtrip",
                kind="provider_validation",
                description=(
                    "Call the configured vision-capable provider with the latest "
                    "DimOS image artifact; this uses API credentials but no robot motion."
                ),
                commands=("uv run dimos agent-vision-roundtrip --latest-image-artifact",),
                requires_running_dimos=True,
                requires_provider=True,
            ),
        ),
    ),
    "supervised-prelive": Go2HarnessGateProfile(
        name="supervised-prelive",
        description=(
            "Low-speed supervised readiness gate: local motion tools visible, "
            "strong map/frontier tools hidden, tokenless external authority refused."
        ),
        steps=(
            Go2HarnessGateStep(
                name="check-agent-provider-readiness",
                kind="provider_validation",
                description=(
                    "Print display-safe native-agent model/provider readiness "
                    "before enabling supervised task authority."
                ),
                commands=(_PROVIDER_STATUS,),
            ),
            Go2HarnessGateStep(
                name="start-supervised-runtime",
                kind="start_runtime",
                description="Start Go2 runtime with vision/lidar-only task authority.",
                commands=(_SUPERVISED_START,),
                requires_robot=True,
            ),
            Go2HarnessGateStep(
                name="validate-supervised-mcp",
                kind="mcp_validation",
                description=(
                    "Verify scope, diagnostics, local motion tools, and tokenless "
                    "movement-authority refusal before dispatch."
                ),
                commands=("uv run dimos mcp validate-harness --profile vision-lidar-supervised",),
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="audit-native-harness-trace",
                kind="trace_audit",
                description="Audit the latest native harness session for prelive invariants.",
                commands=("uv run dimos trace-audit --profile go2-prelive --all",),
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="confirm-physical-stop-path",
                kind="manual_hardware",
                description=(
                    "Before any live movement, confirm a human-held controller or "
                    "physical e-stop/kill path is ready; software stop is best-effort."
                ),
                requires_robot=True,
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="stop-before-motion-drill",
                kind="manual_hardware",
                description=(
                    "Stop here for user approval before any real movement drill; "
                    "do not treat this profile as live motion acceptance."
                ),
                requires_robot=True,
                requires_running_dimos=True,
            ),
        ),
    ),
    "live-motion-audit": Go2HarnessGateProfile(
        name="live-motion-audit",
        description=(
            "Post-run audit for the real Go2 movement drill after the user performs "
            "the live session."
        ),
        steps=(
            Go2HarnessGateStep(
                name="audit-live-motion-trace",
                kind="trace_audit",
                description=(
                    "Require relative-move, navigation, rotation, stop, and preemption evidence."
                ),
                commands=("uv run dimos trace-audit --profile go2-live-motion --all",),
                requires_robot=True,
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="review-motion-diagnostics",
                kind="manual_hardware",
                description=(
                    "If motion failed or stalled, inspect motion_safety_status and "
                    "last_motion_diagnostics before retrying."
                ),
                commands=(
                    "uv run dimos mcp call motion_safety_status",
                    "uv run dimos mcp call last_motion_diagnostics",
                ),
                requires_robot=True,
                requires_running_dimos=True,
            ),
        ),
    ),
    "navigation-live-audit": Go2HarnessGateProfile(
        name="navigation-live-audit",
        description=(
            "Post-run audit for Go2 live navigation drills covering path-found, "
            "local-obstacle, and no-path/retry-limit cases."
        ),
        steps=(
            Go2HarnessGateStep(
                name="audit-navigation-success-trace",
                kind="trace_audit",
                description=(
                    "Audit path-found or local-obstacle navigation runs; retry-limit "
                    "escalation must be absent."
                ),
                commands=("uv run dimos trace-audit --profile go2-navigation-success --all",),
                requires_robot=True,
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="audit-navigation-escalation-trace",
                kind="trace_audit",
                description=(
                    "Audit no-path/retry-limit navigation runs; structured failure "
                    "escalation must be present."
                ),
                commands=("uv run dimos trace-audit --profile go2-navigation-escalation --all",),
                requires_robot=True,
                requires_running_dimos=True,
            ),
            Go2HarnessGateStep(
                name="review-navigation-diagnostics",
                kind="manual_hardware",
                description=("Inspect diagnostics before retrying a failed navigation drill."),
                commands=(
                    "uv run dimos mcp call navigation_status",
                    "uv run dimos mcp call motion_safety_status",
                    "uv run dimos mcp call last_motion_diagnostics",
                ),
                requires_robot=True,
                requires_running_dimos=True,
            ),
        ),
    ),
}


def available_go2_harness_gate_profiles() -> tuple[Go2HarnessGateProfile, ...]:
    """Return profiles in CLI display order."""

    return tuple(GO2_HARNESS_GATE_PROFILES[name] for name in sorted(GO2_HARNESS_GATE_PROFILES))


def go2_harness_gate_profile(name: str) -> Go2HarnessGateProfile:
    """Resolve a named Go2 harness gate profile."""

    try:
        return GO2_HARNESS_GATE_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(GO2_HARNESS_GATE_PROFILES))
        raise ValueError(
            f"Unknown Go2 harness gate profile '{name}'. Available: {available}"
        ) from exc
