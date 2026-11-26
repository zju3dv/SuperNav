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

"""Configuration for the first-class operator console."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from dimos.agents.harness.go2_harness_profiles import (
    Go2ProfileAxes,
    resolve_go2_profile_axes,
)

ConsolePromptMode = Literal["default"]


class StartupPolicy(str, Enum):
    """How ``dimos operate`` should obtain a DimOS MCP connection."""

    ATTACH = "attach"
    START = "start"


class AgentMode(str, Enum):
    """Agent harness mode requested by the operator."""

    AUTO = "auto"
    READONLY = "readonly"
    AUTHORITY_PROXY = "authority-proxy"
    ADMIN_OVERRIDE_DEBUG = "admin-override-debug"


@dataclass(frozen=True)
class ConsoleConfig:
    """Operator console settings parsed by the thin Typer command."""

    blueprint: str = "unitree-go2-edge-mcp"
    startup_policy: StartupPolicy = StartupPolicy.ATTACH
    no_launch: bool = False
    mcp_url: str | None = None
    mcp_ready_timeout_s: float = 10.0
    stack: str = "full"
    control: str = "local-planner"
    mode: AgentMode = AgentMode.AUTO
    model: str = "gpt-5.4"
    reasoning: str = "medium"
    run_root: Path | None = None
    viewer: str = "none"
    authority_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "startup_policy", StartupPolicy(self.startup_policy))
        object.__setattr__(self, "mode", AgentMode(self.mode))
        if self.mcp_ready_timeout_s < 0:
            raise ValueError("mcp_ready_timeout_s must be non-negative")
        axes = resolve_go2_profile_axes(self.stack, self.control)
        object.__setattr__(self, "stack", axes.stack)
        object.__setattr__(self, "control", axes.control)
        object.__setattr__(
            self,
            "authority_tools",
            tuple(dict.fromkeys(tool.strip() for tool in self.authority_tools if tool.strip())),
        )

    @property
    def resolved_axes(self) -> Go2ProfileAxes:
        """Return the validated stack/control projection onto legacy mechanisms."""

        return resolve_go2_profile_axes(self.stack, self.control)

    @property
    def resolved_edge_profile_env(self) -> str | None:
        """Return the legacy edge profile derived from ``stack``."""

        return self.resolved_axes.edge_profile

    @property
    def tool_profile(self) -> str:
        """Return the legacy harness profile derived from ``control``."""

        return self.resolved_axes.tool_profile

    @property
    def resolved_tool_scope(self) -> str:
        """Resolve the selected harness profile to its runtime tool scope."""

        return self.resolved_axes.tool_scope

    @property
    def resolved_agent_mode(self) -> AgentMode:
        """Resolve ``auto`` without introducing a second permission model."""

        if self.mode is not AgentMode.AUTO:
            return self.mode
        if self.resolved_tool_scope == "readonly":
            return AgentMode.READONLY
        return AgentMode.AUTHORITY_PROXY

    @property
    def resolved_codex_harness_mode(self) -> AgentMode:
        """Return the existing Codex harness permission mode used under the UI mode."""

        resolved = self.resolved_agent_mode
        return resolved

    @property
    def prompt_mode(self) -> ConsolePromptMode:
        """Return the prompt policy variant for the prepared Codex run."""

        return "default"
