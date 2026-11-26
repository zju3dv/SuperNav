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

"""Attach-first startup orchestration for ``dimos operate``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Protocol

from dimos.agents.console.config import ConsoleConfig, StartupPolicy
from dimos.agents.console.mcp_client import ConsoleMcpClient, McpAdapterProtocol, McpStatusSnapshot
from dimos.agents.console.state import ToolInfo
from dimos.agents.mcp.mcp_adapter import DEFAULT_TIMEOUT, McpAdapter
from dimos.core.run_registry import get_most_recent


class LauncherProtocol(Protocol):
    """Starts DimOS through an existing CLI-equivalent path."""

    def launch(self, config: ConsoleConfig) -> subprocess.Popen[str]: ...


@dataclass(frozen=True)
class StartupResult:
    """Outcome of attach/start readiness work."""

    ready: bool
    mcp_url: str
    tools: tuple[ToolInfo, ...] = ()
    server_info: dict[str, Any] | None = None
    run_id: str | None = None
    run_log_dir: str | None = None
    messages: tuple[str, ...] = ()
    error: str | None = None
    pending_confirmation: bool = False
    launch_attempted: bool = False
    launch_pid: int | None = None


class SubprocessDimOSLauncher:
    """Launch DimOS by reusing the public ``dimos run`` command path."""

    def launch(self, config: ConsoleConfig) -> subprocess.Popen[str]:
        cmd = [
            sys.executable,
            "-m",
            "dimos.robot.cli.dimos",
            "--viewer",
            config.viewer,
            "run",
            config.blueprint,
            "--daemon",
            "--stack",
            config.stack,
            "--control",
            config.control,
        ]
        return subprocess.Popen(cmd, text=True)


class StartupService:
    """Attach to MCP first; start DimOS only with explicit operator confirmation."""

    def __init__(
        self,
        *,
        adapter_factory: Callable[[str | None, Any | None], McpAdapterProtocol] | None = None,
        client_factory: Callable[[McpAdapterProtocol], ConsoleMcpClient] = ConsoleMcpClient,
        run_entry_getter: Callable[[], Any | None] = lambda: get_most_recent(alive_only=True),
        launcher: LauncherProtocol | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self._client_factory = client_factory
        self._run_entry_getter = run_entry_getter
        self._launcher = launcher or SubprocessDimOSLauncher()

    def attach_or_start(
        self,
        config: ConsoleConfig,
        *,
        confirm_start: Callable[[], bool] | None = None,
    ) -> StartupResult:
        """Attach to an active MCP server or prepare an explicitly confirmed start."""

        entry = None if config.mcp_url else self._run_entry_getter()
        first = self._probe(config, entry)
        messages = ["attach-first: probing MCP readiness"]
        if first.ready:
            messages.append("MCP initialize succeeded")
            return _result_from_snapshot(first, entry, messages=tuple(messages))

        messages.append(first.error or "MCP server is not reachable")
        if config.no_launch or config.startup_policy is StartupPolicy.ATTACH:
            messages.append("DimOS start was not attempted")
            return _result_from_snapshot(first, entry, messages=tuple(messages))

        if confirm_start is None or not confirm_start():
            messages.append("DimOS start requires explicit operator confirmation")
            return StartupResult(
                ready=False,
                mcp_url=first.url,
                run_id=_entry_attr(entry, "run_id"),
                run_log_dir=_entry_attr(entry, "log_dir"),
                messages=tuple(messages),
                error=first.error,
                pending_confirmation=True,
            )

        process = self._launcher.launch(config)
        messages.append(f"DimOS start command launched pid={process.pid}")
        launched_entry = entry if config.mcp_url else self._run_entry_getter()
        second = self._probe(config, launched_entry)
        result = _result_from_snapshot(second, launched_entry, messages=tuple(messages))
        return StartupResult(
            ready=result.ready,
            mcp_url=result.mcp_url,
            tools=result.tools,
            server_info=result.server_info,
            run_id=result.run_id,
            run_log_dir=result.run_log_dir,
            messages=result.messages,
            error=result.error,
            launch_attempted=True,
            launch_pid=process.pid,
        )

    def _probe(self, config: ConsoleConfig, entry: Any | None) -> McpStatusSnapshot:
        adapter = self._adapter_factory(config.mcp_url, entry)
        client = self._client_factory(adapter)
        return client.readiness_snapshot(config.mcp_ready_timeout_s)


def _default_adapter_factory(mcp_url: str | None, entry: Any | None) -> McpAdapter:
    if mcp_url:
        return McpAdapter(url=mcp_url, timeout=DEFAULT_TIMEOUT)
    return McpAdapter.from_run_entry(entry, timeout=DEFAULT_TIMEOUT)


def _result_from_snapshot(
    snapshot: McpStatusSnapshot,
    entry: Any | None,
    *,
    messages: tuple[str, ...],
) -> StartupResult:
    return StartupResult(
        ready=snapshot.ready,
        mcp_url=snapshot.url,
        tools=snapshot.tools,
        server_info=snapshot.server_info,
        run_id=_entry_attr(entry, "run_id"),
        run_log_dir=_entry_attr(entry, "log_dir"),
        messages=messages,
        error=snapshot.error,
    )


def _entry_attr(entry: Any | None, name: str) -> str | None:
    value = getattr(entry, name, None)
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return str(value)
