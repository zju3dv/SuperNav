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

"""Run-local Codex harness preparation for DimOS MCP experiments.

This module intentionally stops at launch preparation. It creates an isolated
``CODEX_HOME`` and workspace, writes only the DimOS MCP entry plus minimal model
provider settings, and records an auditable command. Benchmark policy and live
robot execution stay outside this helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Literal
from urllib.parse import urlparse

import requests

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from dimos.agents.harness.go2_harness_profiles import (
    Go2HarnessToolProfile,
    go2_harness_profile_prompt_lines,
    go2_harness_tool_profile,
)
from dimos.agents.mcp.external_harness_authority_proxy import (
    DEFAULT_AUTHORITY_PROXY_AREA,
    DEFAULT_AUTHORITY_PROXY_DURATION_S,
    DEFAULT_AUTHORITY_PROXY_SCOPE,
    DEFAULT_CRITICAL_BATTERY_PERCENT,
    DEFAULT_MIN_START_BATTERY_PERCENT,
)
from dimos.agents.mcp.mcp_adapter import McpAdapter
from dimos.agents.mcp.skill_assets import copy_skill, read_references, read_skill

CodexHarnessMode = Literal["readonly", "authority-proxy", "admin-override-debug"]
CodexHarnessPromptMode = Literal["default"]

DEFAULT_AUTHORITY_PROXY_URL = "http://127.0.0.1:9992/mcp"
DEFAULT_AUTHORITY_PROXY_MOVEMENT_TOOLS = (
    "relative_move",
    "rotate_in_place",
    "full_lie_down",
    "lie_down",
)
DEFAULT_MCP_SMOKE_REQUIRED_TOOLS = (
    "observe",
    "go2_robot_status",
    "motion_safety_status",
    "rotate_in_place",
    "relative_move",
    "last_motion_diagnostics",
    "stop_movement",
)
DEFAULT_MCP_SMOKE_REQUIRED_CALL_TOOLS = (
    "observe",
    "go2_robot_status",
    "motion_safety_status",
    "last_motion_diagnostics",
)
DEFAULT_MCP_SMOKE_TIMEOUT_S = 300.0
DEFAULT_MCP_SMOKE_ATTEMPTS = 2
DEFAULT_MCP_SMOKE_RETRY_DELAY_S = 2.0
DEFAULT_MCP_SMOKE_REASONING_EFFORT = "low"
DEFAULT_MCP_SMOKE_TRANSPORT = "direct_mcp"

DRIVE_GO2_SKILL_TEXT = read_skill("drive-go2")

GO2_PIXELGOAL_SKILL_TEXT = read_skill("go2-pixelgoal")

GO2_GLOBAL_NAVIGATION_PIXELGOAL_SKILL_TEXT = read_skill("global-navigation-pixelgoal")

GO2_GLOBAL_NAVIGATION_PIXELGOAL_REFERENCES = read_references(
    "global-navigation-pixelgoal",
    ("entrance-search.md", "exploration-and-return.md", "recovery-and-stop.md"),
)


MINIMAL_CODEX_ROOT_KEYS = (
    "model_provider",
    "model",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
    "model_context_window",
    "model_auto_compact_token_limit",
    "service_tier",
    "web_search",
)

MINIMAL_CODEX_TABLE_KEYS = ("sandbox_workspace_write",)

# These can discover account-level or plugin-provided MCP servers even with a
# fresh CODEX_HOME. Keep the robot run limited to its explicit DimOS endpoint.
DISABLED_CODEX_FEATURES = ("hooks", "memories", "apps", "plugins", "remote_plugin")

_SAFE_PROVIDER_KEYS = {
    "name",
    "base_url",
    "wire_api",
    "requires_openai_auth",
    "env_key",
    "bearer_token_env_var",
    "query_params",
}

_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class CodexMcpSmokeGateResult:
    """Result of running the MCP-only Codex visibility smoke gate."""

    ok: bool
    returncode: int
    reason: str
    log_path: Path
    stderr_path: Path
    result_path: Path
    required_mcp_tool_names: tuple[str, ...]
    missing_mcp_tool_names: tuple[str, ...] = ()
    called_mcp_tool_names: tuple[str, ...] = ()
    attempts: int = 1
    transport: str = DEFAULT_MCP_SMOKE_TRANSPORT

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "reason": self.reason,
            "log_path": str(self.log_path),
            "stderr_path": str(self.stderr_path),
            "result_path": str(self.result_path),
            "required_mcp_tool_names": list(self.required_mcp_tool_names),
            "missing_mcp_tool_names": list(self.missing_mcp_tool_names),
            "called_mcp_tool_names": list(self.called_mcp_tool_names),
            "attempts": self.attempts,
            "transport": self.transport,
        }


@dataclass(frozen=True)
class CodexHarnessRun:
    """Prepared run-local Codex invocation details."""

    run_root: Path
    workspace: Path
    codex_home: Path
    config_path: Path
    prompt_path: Path
    mcp_smoke_prompt_path: Path
    mcp_smoke_log_path: Path
    mcp_smoke_stderr_path: Path
    mcp_smoke_result_path: Path
    audit_path: Path
    shareable_summary_path: Path
    shareable_summary_markdown_path: Path
    trace_root: Path
    command: list[str]
    mcp_smoke_command: list[str]
    mcp_smoke_transport: str
    env: dict[str, str]
    shell_command: str
    mcp_smoke_shell_command: str
    mcp_url: str
    upstream_mcp_url: str
    server_name: str
    task: str
    mode: CodexHarnessMode
    prompt_mode: CodexHarnessPromptMode
    codex_mcp_bypass_enabled: bool
    mcp_smoke_required: bool
    model: str | None = None
    model_reasoning_effort: str | None = None
    service_tier: str | None = None
    codex_exec_timeout_s: float | None = None
    mcp_smoke_timeout_s: float | None = None
    mcp_smoke_required_tool_names: tuple[str, ...] = DEFAULT_MCP_SMOKE_REQUIRED_TOOLS
    tool_profile: dict[str, Any] = field(default_factory=dict)
    tool_profile_summary: tuple[str, ...] = ()
    proxy_url: str | None = None
    proxy_command: list[str] | None = None
    proxy_shell_command: str | None = None
    drive_go2_skill_path: Path | None = None
    drive_go2_workspace_skill_path: Path | None = None
    go2_pixelgoal_workspace_skill_path: Path | None = None
    global_navigation_pixelgoal_workspace_skill_path: Path | None = None
    authority_envelope: dict[str, Any] | None = None
    battery_gate: dict[str, Any] = field(default_factory=dict)
    copied_auth_files: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_root": str(self.run_root),
            "workspace": str(self.workspace),
            "codex_home": str(self.codex_home),
            "config_path": str(self.config_path),
            "prompt_path": str(self.prompt_path),
            "mcp_smoke_prompt_path": str(self.mcp_smoke_prompt_path),
            "mcp_smoke_log_path": str(self.mcp_smoke_log_path),
            "mcp_smoke_stderr_path": str(self.mcp_smoke_stderr_path),
            "mcp_smoke_result_path": str(self.mcp_smoke_result_path),
            "audit_path": str(self.audit_path),
            "shareable_summary_path": str(self.shareable_summary_path),
            "shareable_summary_markdown_path": str(self.shareable_summary_markdown_path),
            "trace_root": str(self.trace_root),
            "command": self.command,
            "mcp_smoke_command": self.mcp_smoke_command,
            "mcp_smoke_transport": self.mcp_smoke_transport,
            "env": self.env,
            "shell_command": self.shell_command,
            "mcp_smoke_shell_command": self.mcp_smoke_shell_command,
            "mcp_url": self.mcp_url,
            "upstream_mcp_url": self.upstream_mcp_url,
            "server_name": self.server_name,
            "task": self.task,
            "mode": self.mode,
            "prompt_mode": self.prompt_mode,
            "model": self.model,
            "model_reasoning_effort": self.model_reasoning_effort,
            "service_tier": self.service_tier,
            "codex_exec_timeout_s": self.codex_exec_timeout_s,
            "mcp_smoke_timeout_s": self.mcp_smoke_timeout_s,
            "codex_mcp_bypass_enabled": self.codex_mcp_bypass_enabled,
            "mcp_smoke_required": self.mcp_smoke_required,
            "mcp_smoke_required_tool_names": list(self.mcp_smoke_required_tool_names),
            "tool_profile": self.tool_profile,
            "tool_profile_summary": list(self.tool_profile_summary),
            "proxy_url": self.proxy_url,
            "proxy_command": self.proxy_command,
            "proxy_shell_command": self.proxy_shell_command,
            "drive_go2_skill_path": str(self.drive_go2_skill_path)
            if self.drive_go2_skill_path is not None
            else None,
            "drive_go2_workspace_skill_path": str(self.drive_go2_workspace_skill_path)
            if self.drive_go2_workspace_skill_path is not None
            else None,
            "go2_pixelgoal_workspace_skill_path": str(self.go2_pixelgoal_workspace_skill_path)
            if self.go2_pixelgoal_workspace_skill_path is not None
            else None,
            "global_navigation_pixelgoal_workspace_skill_path": str(
                self.global_navigation_pixelgoal_workspace_skill_path
            )
            if self.global_navigation_pixelgoal_workspace_skill_path is not None
            else None,
            "authority_envelope": self.authority_envelope,
            "battery_gate": self.battery_gate,
            "copied_auth_files": list(self.copied_auth_files),
        }
        return payload


def prepare_codex_harness_run(
    *,
    run_root: Path,
    task: str,
    mcp_url: str,
    source_codex_home: Path | None = None,
    server_name: str = "dimos",
    model: str | None = None,
    model_reasoning_effort: str | None = "medium",
    service_tier: str | None = None,
    codex_exec_timeout_s: float | None = 1800.0,
    mcp_smoke_timeout_s: float | None = DEFAULT_MCP_SMOKE_TIMEOUT_S,
    sandbox: str = "workspace-write",
    copy_auth: bool = True,
    use_goal_tools: bool = True,
    dangerously_bypass_sandbox: bool = False,
    mode: CodexHarnessMode = "readonly",
    prompt_mode: CodexHarnessPromptMode = "default",
    tool_profile: str = "",
    proxy_url: str | None = None,
    authority_scope: str = DEFAULT_AUTHORITY_PROXY_SCOPE,
    authority_area: str = DEFAULT_AUTHORITY_PROXY_AREA,
    authority_duration_s: float = DEFAULT_AUTHORITY_PROXY_DURATION_S,
    authority_allowed_movement_tools: tuple[str, ...] | list[str] = (),
    max_movement_calls: int | None = None,
    mcp_smoke_required: bool | None = None,
) -> CodexHarnessRun:
    """Prepare a run-local ``codex exec`` invocation for DimOS MCP.

    The generated config deliberately omits global MCP servers, hooks, project
    trust entries, memories, and personal skills. Auth files are copied only into
    the run-local home so the execution can use the user's existing Codex login
    without writing sessions into ``~/.codex``.
    """

    if not task.strip():
        raise ValueError("task must not be empty")
    if not mcp_url.strip():
        raise ValueError("mcp_url must not be empty")
    if mode not in {"readonly", "authority-proxy", "admin-override-debug"}:
        raise ValueError("mode must be readonly, authority-proxy, or admin-override-debug")
    if prompt_mode != "default":
        raise ValueError("this release supports only the memory-free default prompt mode")
    if codex_exec_timeout_s is not None and codex_exec_timeout_s <= 0.0:
        raise ValueError("codex_exec_timeout_s must be positive when set")
    if mcp_smoke_timeout_s is not None and mcp_smoke_timeout_s <= 0.0:
        raise ValueError("mcp_smoke_timeout_s must be positive when set")
    resolved_tool_profile = _resolve_tool_profile(mode=mode, tool_profile=tool_profile)
    allowed_movement_tools = (
        _resolve_authority_movement_tools(
            explicit_tools=authority_allowed_movement_tools,
            tool_profile=resolved_tool_profile,
        )
        if mode == "authority-proxy"
        else ()
    )
    pixelgoal_enabled = "pixelgoal_navigate" in allowed_movement_tools
    if pixelgoal_enabled:
        prompt_mode = "default"

    run_root = run_root.expanduser().resolve()
    workspace = run_root / "workspace"
    codex_home = run_root / "codex_home"
    trace_root = run_root / "trace"
    logs_dir = run_root / "logs"
    for path in (workspace, codex_home, trace_root, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_home = (source_codex_home or Path.home() / ".codex").expanduser()
    copied_auth_files = _copy_run_local_codex_auth(source_home, codex_home) if copy_auth else []
    upstream_mcp_url = mcp_url
    codex_mcp_url = _codex_facing_mcp_url(
        mode=mode, upstream_mcp_url=upstream_mcp_url, proxy_url=proxy_url
    )
    config_path = write_run_local_codex_config(
        codex_home=codex_home,
        source_codex_home=source_home,
        mcp_url=codex_mcp_url,
        server_name=server_name,
        model=model,
        model_reasoning_effort=model_reasoning_effort,
        service_tier=service_tier,
        mcp_tool_timeout_sec=(320 if "pixelgoal_navigate" in allowed_movement_tools else None),
    )
    effective_codex_config = _load_codex_config(config_path)
    drive_go2_skill_path = None if pixelgoal_enabled else _write_drive_go2_skill(codex_home)
    drive_go2_workspace_skill_path = (
        None if pixelgoal_enabled else _write_workspace_drive_go2_skill(workspace)
    )
    go2_pixelgoal_workspace_skill_path = (
        _write_workspace_go2_pixelgoal_skill(workspace) if pixelgoal_enabled else None
    )
    global_navigation_pixelgoal_workspace_skill_path = (
        _write_workspace_global_navigation_pixelgoal_skill(workspace) if pixelgoal_enabled else None
    )
    prompt = codex_goal_prompt(
        task=task,
        server_name=server_name,
        use_goal_tools=use_goal_tools,
        mode=mode,
        tool_profile=resolved_tool_profile,
        prompt_mode=prompt_mode,
        pixelgoal_enabled=pixelgoal_enabled,
    )
    prompt_path = run_root / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    effective_bypass = dangerously_bypass_sandbox or mode in {
        "authority-proxy",
        "admin-override-debug",
    }
    command = _codex_command(
        workspace=workspace,
        sandbox=sandbox,
        dangerously_bypass_sandbox=effective_bypass,
    )
    mcp_smoke_command = _codex_command(
        workspace=workspace,
        sandbox=sandbox,
        dangerously_bypass_sandbox=effective_bypass,
        config_overrides={"model_reasoning_effort": DEFAULT_MCP_SMOKE_REASONING_EFFORT},
    )
    smoke_required_tools = _mcp_smoke_required_tools(
        mode=mode,
        allowed_movement_tools=allowed_movement_tools,
        prompt_mode=prompt_mode,
    )
    effective_mcp_smoke_required = (
        mode == "authority-proxy" if mcp_smoke_required is None else mcp_smoke_required
    )
    mcp_smoke_prompt = codex_mcp_smoke_prompt(
        server_name=server_name,
        required_tools=smoke_required_tools,
        mode=mode,
    )
    mcp_smoke_prompt_path = run_root / "mcp_smoke_prompt.md"
    mcp_smoke_prompt_path.write_text(mcp_smoke_prompt, encoding="utf-8")
    mcp_smoke_log_path = logs_dir / "mcp-smoke.jsonl"
    mcp_smoke_stderr_path = logs_dir / "mcp-smoke.stderr"
    mcp_smoke_result_path = logs_dir / "mcp-smoke-result.json"
    shareable_summary_path = logs_dir / "shareable-summary.json"
    shareable_summary_markdown_path = logs_dir / "shareable-summary.md"
    env = {
        "CODEX_HOME": str(codex_home),
        "CODEX_ROLLOUT_TRACE_ROOT": str(trace_root),
    }
    shell_command = _shell_command(
        command,
        env=env,
        stdin_path=prompt_path,
        timeout_s=codex_exec_timeout_s,
    )
    mcp_smoke_shell_command = _shell_command(
        mcp_smoke_command,
        env=env,
        stdin_path=mcp_smoke_prompt_path,
        stdout_path=mcp_smoke_log_path,
        stderr_path=mcp_smoke_stderr_path,
        timeout_s=mcp_smoke_timeout_s,
    )
    audit_path = logs_dir / "launch.json"
    authority_envelope = _authority_envelope_audit(
        task=task,
        mode=mode,
        scope=authority_scope,
        area=authority_area,
        duration_s=authority_duration_s,
        allowed_movement_tools=allowed_movement_tools,
        max_movement_calls=max_movement_calls,
        tool_profile_name=resolved_tool_profile.name,
    )
    proxy_command = (
        _authority_proxy_command(
            upstream_mcp_url=upstream_mcp_url,
            proxy_url=codex_mcp_url,
            task=task,
            scope=authority_scope,
            area=authority_area,
            duration_s=authority_duration_s,
            allowed_movement_tools=allowed_movement_tools,
            max_movement_calls=max_movement_calls,
            tool_profile_name=resolved_tool_profile.name,
        )
        if mode == "authority-proxy"
        else None
    )
    prepared = CodexHarnessRun(
        run_root=run_root,
        workspace=workspace,
        codex_home=codex_home,
        config_path=config_path,
        prompt_path=prompt_path,
        mcp_smoke_prompt_path=mcp_smoke_prompt_path,
        mcp_smoke_log_path=mcp_smoke_log_path,
        mcp_smoke_stderr_path=mcp_smoke_stderr_path,
        mcp_smoke_result_path=mcp_smoke_result_path,
        audit_path=audit_path,
        shareable_summary_path=shareable_summary_path,
        shareable_summary_markdown_path=shareable_summary_markdown_path,
        trace_root=trace_root,
        command=command,
        mcp_smoke_command=mcp_smoke_command,
        mcp_smoke_transport=DEFAULT_MCP_SMOKE_TRANSPORT,
        env=env,
        shell_command=shell_command,
        mcp_smoke_shell_command=mcp_smoke_shell_command,
        mcp_url=codex_mcp_url,
        upstream_mcp_url=upstream_mcp_url,
        server_name=server_name,
        task=task,
        mode=mode,
        prompt_mode=prompt_mode,
        model=_optional_str(effective_codex_config.get("model")),
        model_reasoning_effort=_optional_str(effective_codex_config.get("model_reasoning_effort")),
        service_tier=_optional_str(effective_codex_config.get("service_tier")),
        codex_exec_timeout_s=codex_exec_timeout_s,
        mcp_smoke_timeout_s=mcp_smoke_timeout_s,
        codex_mcp_bypass_enabled=effective_bypass,
        mcp_smoke_required=effective_mcp_smoke_required,
        mcp_smoke_required_tool_names=smoke_required_tools,
        tool_profile=resolved_tool_profile.model_dump(),
        tool_profile_summary=tuple(go2_harness_profile_prompt_lines(resolved_tool_profile)),
        proxy_url=codex_mcp_url if mode == "authority-proxy" else None,
        proxy_command=proxy_command,
        proxy_shell_command=shlex.join(proxy_command) if proxy_command is not None else None,
        drive_go2_skill_path=drive_go2_skill_path,
        drive_go2_workspace_skill_path=drive_go2_workspace_skill_path,
        go2_pixelgoal_workspace_skill_path=go2_pixelgoal_workspace_skill_path,
        global_navigation_pixelgoal_workspace_skill_path=(
            global_navigation_pixelgoal_workspace_skill_path
        ),
        authority_envelope=authority_envelope,
        battery_gate=_battery_gate_audit(),
        copied_auth_files=copied_auth_files,
    )
    _write_launch_audit(prepared)
    write_shareable_run_summary(prepared)
    return prepared


def codex_mcp_smoke_prompt(
    *,
    server_name: str = "dimos",
    required_tools: tuple[str, ...] | list[str] = DEFAULT_MCP_SMOKE_REQUIRED_TOOLS,
    mode: CodexHarnessMode = "readonly",
) -> str:
    """Build the MCP-only visibility gate prompt for nested ``codex exec``."""

    codex_tool_names = [
        _codex_mcp_tool_name(tool, server_name=server_name) for tool in required_tools
    ]
    tool_prefix = f"mcp__{server_name}__"
    lines = [
        "You are running the DimOS MCP-only preflight smoke gate.",
        "Your job is to prove that the configured MCP server is visible to this isolated Codex process.",
        "Do not use shell, Python, local files, browser tools, or `dimos mcp` for DimOS control.",
        "Use only the built-in goal tools and the DimOS MCP tools exposed directly to this Codex run.",
        f"When comparing tool names, strip the exact `{tool_prefix}` prefix from both required and available names; bare and prefixed forms are equivalent.",
        "If any required DimOS MCP tool is not visible/callable, stop immediately and answer:",
        "DIMOS_MCP_SMOKE: FAIL missing=<comma-separated missing mcp__dimos__ tool names>",
        "If all required tools are visible, call only these read-only/status tools before passing: go2_robot_status, motion_safety_status, observe, and last_motion_diagnostics.",
        "Do not call movement or posture tools during this visibility gate; those are checked for visibility only before hardware movement.",
        "After the read-only checks, answer exactly one final line:",
        "DIMOS_MCP_SMOKE: PASS tools=<comma-separated required mcp tool names>",
        f"Run mode: {mode}",
        "Required MCP tools:",
    ]
    lines.extend(f"- {name}" for name in codex_tool_names)
    return "\n".join(lines) + "\n"


def _mcp_smoke_required_tools(
    *,
    mode: CodexHarnessMode,
    allowed_movement_tools: tuple[str, ...] | list[str],
    prompt_mode: CodexHarnessPromptMode = "default",
) -> tuple[str, ...]:
    read_tools = (
        "observe",
        "go2_robot_status",
        "motion_safety_status",
        "last_motion_diagnostics",
        "stop_movement",
    )
    if mode != "authority-proxy":
        return DEFAULT_MCP_SMOKE_REQUIRED_TOOLS
    pixelgoal_tools = (
        ("observe_surround", "pixelgoal_status", "pixelgoal_stop")
        if "pixelgoal_navigate" in allowed_movement_tools
        else ()
    )
    return tuple(
        dict.fromkeys(
            (*read_tools, *allowed_movement_tools, *pixelgoal_tools)
        )
    )


def run_codex_mcp_smoke_gate(
    run: CodexHarnessRun,
    *,
    timeout_s: float | None = None,
    attempts: int = DEFAULT_MCP_SMOKE_ATTEMPTS,
    retry_delay_s: float = DEFAULT_MCP_SMOKE_RETRY_DELAY_S,
) -> CodexMcpSmokeGateResult:
    """Execute the generated MCP-only Codex smoke gate and persist its result."""

    if run.mcp_smoke_transport == DEFAULT_MCP_SMOKE_TRANSPORT:
        result = run_direct_mcp_smoke_gate(
            mcp_url=run.mcp_url,
            server_name=run.server_name,
            required_tools=run.mcp_smoke_required_tool_names,
            log_path=run.mcp_smoke_log_path,
            stderr_path=run.mcp_smoke_stderr_path,
            result_path=run.mcp_smoke_result_path,
            timeout_s=timeout_s or run.mcp_smoke_timeout_s or DEFAULT_MCP_SMOKE_TIMEOUT_S,
        )
        _update_launch_smoke_audit(run, result)
        write_shareable_run_summary(run, smoke_result=result)
        return result

    effective_timeout_s = timeout_s or run.mcp_smoke_timeout_s or DEFAULT_MCP_SMOKE_TIMEOUT_S
    max_attempts = max(1, int(attempts))
    attempt_index = 0
    while attempt_index < max_attempts:
        attempt_index += 1
        stdout_path = (
            run.mcp_smoke_log_path
            if max_attempts == 1
            else _mcp_smoke_attempt_path(run.mcp_smoke_log_path, attempt_index)
        )
        stderr_path = (
            run.mcp_smoke_stderr_path
            if max_attempts == 1
            else _mcp_smoke_attempt_path(run.mcp_smoke_stderr_path, attempt_index)
        )
        try:
            with run.mcp_smoke_prompt_path.open("rb") as stdin:
                with stdout_path.open("wb") as stdout:
                    with stderr_path.open("wb") as stderr:
                        completed = subprocess.run(
                            run.mcp_smoke_command,
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            env={**os_environ_safe(), **run.env},
                            timeout=effective_timeout_s,
                            check=False,
                        )
            log_text = stdout_path.read_text(encoding="utf-8", errors="replace")
            ok, reason, missing, called = codex_mcp_smoke_log_result(
                log_text,
                server_name=run.server_name,
                required_tools=run.mcp_smoke_required_tool_names,
                returncode=completed.returncode,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            log_text = (
                stdout_path.read_text(encoding="utf-8", errors="replace")
                if stdout_path.exists()
                else ""
            )
            ok, parsed_reason, missing, called = codex_mcp_smoke_log_result(
                log_text,
                server_name=run.server_name,
                required_tools=run.mcp_smoke_required_tool_names,
                returncode=0,
            )
            timeout_message = f"codex exec smoke timed out after {effective_timeout_s:.1f}s"
            if ok:
                reason = f"{timeout_message} after PASS marker: {parsed_reason}"
            else:
                reason = f"{timeout_message} before a valid PASS marker: {parsed_reason}"
            returncode = -1
            previous_stderr = (
                stderr_path.read_text(encoding="utf-8", errors="replace")
                if stderr_path.exists()
                else ""
            )
            stderr_path.write_text(
                (previous_stderr.rstrip() + "\n" if previous_stderr else "") + reason + "\n",
                encoding="utf-8",
            )
        if stdout_path != run.mcp_smoke_log_path:
            shutil.copyfile(stdout_path, run.mcp_smoke_log_path)
        if stderr_path != run.mcp_smoke_stderr_path:
            shutil.copyfile(stderr_path, run.mcp_smoke_stderr_path)
        if ok or not _retryable_codex_smoke_disconnect(
            reason=reason,
            returncode=returncode,
            called_mcp_tool_names=called,
        ):
            break
        if attempt_index < max_attempts and retry_delay_s > 0:
            time.sleep(retry_delay_s)
    if attempt_index > 1:
        reason = f"{reason} (after {attempt_index} smoke attempts)"
    result = CodexMcpSmokeGateResult(
        ok=ok,
        returncode=returncode,
        reason=reason,
        log_path=run.mcp_smoke_log_path,
        stderr_path=run.mcp_smoke_stderr_path,
        result_path=run.mcp_smoke_result_path,
        required_mcp_tool_names=tuple(
            f"mcp__{run.server_name}__{tool}" for tool in run.mcp_smoke_required_tool_names
        ),
        missing_mcp_tool_names=tuple(missing),
        called_mcp_tool_names=tuple(called),
        attempts=attempt_index,
    )
    run.mcp_smoke_result_path.write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _update_launch_smoke_audit(run, result)
    write_shareable_run_summary(run, smoke_result=result)
    return result


def run_direct_mcp_smoke_gate(
    *,
    mcp_url: str,
    server_name: str,
    required_tools: tuple[str, ...] | list[str],
    log_path: Path,
    stderr_path: Path,
    result_path: Path,
    timeout_s: float,
) -> CodexMcpSmokeGateResult:
    """Run the live preflight directly against MCP without an LLM inference."""

    required_bare = tuple(
        dict.fromkeys(_bare_mcp_tool_name(tool, server_name=server_name) for tool in required_tools)
    )
    required_mcp = tuple(
        _codex_mcp_tool_name(tool, server_name=server_name) for tool in required_bare
    )
    called: list[str] = []
    missing: list[str] = []
    log_lines: list[str] = [
        json.dumps(
            {
                "type": "mcp_smoke.started",
                "transport": DEFAULT_MCP_SMOKE_TRANSPORT,
                "server": server_name,
                "url": mcp_url,
                "required_mcp_tool_names": list(required_mcp),
            },
            sort_keys=True,
        )
    ]
    stderr_text = ""
    ok = False
    returncode = 1
    reason = "direct MCP smoke did not complete"
    try:
        adapter = McpAdapter(mcp_url, timeout=max(1, int(timeout_s)))
        adapter.initialize()
        tools = adapter.list_tools()
        visible = {str(tool.get("name")) for tool in tools if isinstance(tool, dict)}
        missing = [
            _codex_mcp_tool_name(tool, server_name=server_name)
            for tool in required_bare
            if tool not in visible
        ]
        log_lines.append(
            json.dumps(
                {
                    "type": "mcp_smoke.tools_listed",
                    "visible_tool_names": sorted(visible),
                    "missing_mcp_tool_names": missing,
                },
                sort_keys=True,
            )
        )
        if missing:
            reason = "direct MCP smoke missing required tools"
        else:
            required_call_tools = [
                tool for tool in DEFAULT_MCP_SMOKE_REQUIRED_CALL_TOOLS if tool in required_bare
            ]
            for tool in required_call_tools:
                adapter.call_tool(tool, {})
                codex_name = _codex_mcp_tool_name(tool, server_name=server_name)
                called.append(codex_name)
                log_lines.append(_mcp_smoke_tool_completed_jsonl(server_name, tool))
            pass_line = "DIMOS_MCP_SMOKE: PASS tools=" + ",".join(required_mcp)
            log_lines.append(pass_line)
            ok = True
            returncode = 0
            reason = "direct MCP smoke listed required tools and read-only calls ran"
    except (requests.RequestException, OSError, ValueError) as exc:
        reason = f"direct MCP smoke failed: {exc}"
        stderr_text = reason + "\n"
        missing = list(required_mcp)
    except Exception as exc:
        reason = f"direct MCP smoke failed: {type(exc).__name__}: {exc}"
        stderr_text = reason + "\n"
        missing = list(required_mcp)

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    result = CodexMcpSmokeGateResult(
        ok=ok,
        returncode=returncode,
        reason=reason,
        log_path=log_path,
        stderr_path=stderr_path,
        result_path=result_path,
        required_mcp_tool_names=required_mcp,
        missing_mcp_tool_names=tuple(missing),
        called_mcp_tool_names=tuple(called),
        attempts=1,
        transport=DEFAULT_MCP_SMOKE_TRANSPORT,
    )
    result_path.write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _mcp_smoke_tool_completed_jsonl(server_name: str, tool: str) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": server_name,
                "tool": tool,
                "status": "completed",
                "error": None,
            },
        },
        sort_keys=True,
    )


def _mcp_smoke_attempt_path(path: Path, attempt: int) -> Path:
    return path.with_name(f"{path.stem}.attempt-{attempt}{path.suffix}")


def _retryable_codex_smoke_disconnect(
    *,
    reason: str,
    returncode: int,
    called_mcp_tool_names: list[str],
) -> bool:
    if returncode == 0 or called_mcp_tool_names:
        return False
    lowered = reason.lower()
    return (
        "stream disconnected before completion" in lowered
        or "error sending request for url" in lowered
    )


def codex_mcp_smoke_log_result(
    log_text: str,
    *,
    server_name: str,
    required_tools: tuple[str, ...] | list[str],
    returncode: int = 0,
) -> tuple[bool, str, list[str], list[str]]:
    """Parse a nested Codex smoke JSONL/text log for the fail-closed PASS marker."""

    required_bare_names = [
        _bare_mcp_tool_name(tool, server_name=server_name) for tool in required_tools
    ]
    required_mcp_names = [
        _codex_mcp_tool_name(tool, server_name=server_name) for tool in required_bare_names
    ]
    called_mcp_names = _called_mcp_tool_names(log_text, server_name=server_name)
    if returncode != 0:
        codex_error = _codex_terminal_error_message(log_text)
        detail = f": {codex_error}" if codex_error else ""
        return (
            False,
            f"codex exec exited with status {returncode}{detail}",
            required_mcp_names,
            called_mcp_names,
        )
    if _codex_log_contains_forbidden_cli_fallback(log_text):
        return (
            False,
            "smoke log used forbidden shell fallback `dimos mcp`",
            required_mcp_names,
            called_mcp_names,
        )
    pass_lines = [line for line in log_text.splitlines() if "DIMOS_MCP_SMOKE: PASS" in line]
    if not pass_lines:
        fail_missing = _codex_smoke_fail_missing_names(log_text)
        if fail_missing:
            return (
                False,
                "smoke reported missing required MCP tools",
                [_codex_mcp_tool_name(name, server_name=server_name) for name in fail_missing],
                called_mcp_names,
            )
        return (
            False,
            "smoke log did not contain DIMOS_MCP_SMOKE: PASS",
            required_mcp_names,
            called_mcp_names,
        )
    reported_bare_names = {
        _bare_mcp_tool_name(name, server_name=server_name)
        for name in _codex_smoke_pass_tool_names(pass_lines)
    }
    missing = [
        canonical
        for bare, canonical in zip(required_bare_names, required_mcp_names, strict=True)
        if bare not in reported_bare_names
    ]
    if missing:
        return False, "smoke PASS marker omitted required MCP tool names", missing, called_mcp_names
    required_tool_set = set(required_bare_names)
    required_call_names = {
        f"mcp__{server_name}__{tool}"
        for tool in DEFAULT_MCP_SMOKE_REQUIRED_CALL_TOOLS
        if tool in required_tool_set
    }
    missing_calls = sorted(required_call_names - set(called_mcp_names))
    if missing_calls:
        return (
            False,
            "smoke PASS marker was not backed by required read-only MCP tool calls",
            missing_calls,
            called_mcp_names,
        )
    return (
        True,
        "all required MCP tool names were reported visible and read-only calls ran",
        [],
        called_mcp_names,
    )


def _called_mcp_tool_names(log_text: str, *, server_name: str) -> list[str]:
    called: set[str] = set()
    for line in log_text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = payload.get("item") if isinstance(payload, dict) else None
        if not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_tool_call" or item.get("server") != server_name:
            continue
        if item.get("status") != "completed" or item.get("error") is not None:
            continue
        tool = item.get("tool")
        if isinstance(tool, str) and tool:
            called.add(_codex_mcp_tool_name(tool, server_name=server_name))
    return sorted(called)


def _codex_smoke_pass_tool_names(pass_lines: list[str]) -> list[str]:
    names: list[str] = []
    for line in pass_lines:
        if "tools=" not in line:
            continue
        names_text = line.split("tools=", 1)[1].split('"', 1)[0]
        names.extend(item.strip() for item in names_text.split(",") if item.strip())
    return names


def _bare_mcp_tool_name(name: str, *, server_name: str) -> str:
    return name.removeprefix(f"mcp__{server_name}__")


def _codex_mcp_tool_name(name: str, *, server_name: str) -> str:
    return f"mcp__{server_name}__{_bare_mcp_tool_name(name, server_name=server_name)}"


def _codex_smoke_fail_missing_names(log_text: str) -> list[str]:
    for line in log_text.splitlines():
        if "DIMOS_MCP_SMOKE: FAIL" not in line or "missing=" not in line:
            continue
        missing_text = line.split("missing=", 1)[1]
        # The marker can be inside JSON text, so stop at a quote when present.
        missing_text = missing_text.split('"', 1)[0]
        names = [item.strip() for item in missing_text.split(",") if item.strip()]
        if names:
            return names
    return []


def _codex_terminal_error_message(log_text: str) -> str | None:
    """Return the final Codex error message from a JSONL run log, if present."""

    latest_error: str | None = None
    for line in log_text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "turn.failed":
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                latest_error = error["message"]
                continue
        if payload.get("type") == "error" and isinstance(payload.get("message"), str):
            latest_error = payload["message"]
    return latest_error


def _codex_log_contains_forbidden_cli_fallback(log_text: str) -> bool:
    for line in log_text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _json_command_field_contains(payload, "dimos mcp"):
            return True
    return False


def _json_command_field_contains(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"cmd", "command", "shell_command"}:
                haystack = (
                    " ".join(str(part) for part in item) if isinstance(item, list) else str(item)
                )
                if needle in haystack:
                    return True
            if _json_command_field_contains(item, needle):
                return True
    if isinstance(value, list):
        return any(_json_command_field_contains(item, needle) for item in value)
    return False


def os_environ_safe() -> dict[str, str]:
    """Return the ambient environment for a child process.

    Kept behind a helper so tests can monkeypatch the subprocess environment seam
    without copying the whole runner implementation.
    """

    import os

    return dict(os.environ)


def codex_goal_prompt(
    *,
    task: str,
    server_name: str = "dimos",
    use_goal_tools: bool = True,
    mode: CodexHarnessMode = "readonly",
    tool_profile: Go2HarnessToolProfile | None = None,
    prompt_mode: CodexHarnessPromptMode = "default",
    pixelgoal_enabled: bool = False,
) -> str:
    """Build the prompt used by the isolated Codex operator run."""

    goal_text = task.strip()
    lines = [
        f"Your only goal is: {goal_text}",
        "",
        "You are an autonomous Codex harness operator for a DimOS Go2 experiment.",
        f"Use the `{server_name}` MCP server for robot status, visual observation, and motion tools.",
        "Do not rely on moment-by-moment human tactical hints. Ask only for physical safety facts that are not available through tools.",
        "Treat DimOS movement and safety tool results as evidence; after movement, inspect measured results before declaring success.",
        "For visual claims, use direct MCP observe/image tool outputs that are visible to the model, not a human description.",
    ]
    if pixelgoal_enabled:
        lines.extend(
            [
                (
                    "Use the run-local `global-navigation-pixelgoal` skill for search "
                    "and `go2-pixelgoal` for safe point selection. Read "
                    "`.agents/skills/global-navigation-pixelgoal/SKILL.md` and "
                    "`.agents/skills/go2-pixelgoal/SKILL.md` before moving."
                ),
                (
                    "For map-backend results, exploration_complete means inspect the "
                    "returned views and choose again. Multi-leg exploration uses up to "
                    "4 m per leg, a default 60 s hop deadline, and a 10 m exploration budget. "
                    "deadline_exceeded also requires fresh views before the next point. "
                    "no_progress or exploration_failed "
                    "means reassess the obstruction before moving; never blindly retry "
                    "the same point or doorway. Operator/safety stops override exploration."
                ),
                (
                    "Call `pixelgoal_navigate` directly on the `dimos` MCP server. "
                    "Use the shell only to read skill files."
                ),
                (
                    "Stop discipline: for find/search tasks, target merely appearing "
                    "is not success. If it is distant or small, approach via safe "
                    "visible floor proxies until it is close enough for "
                    "human-verifiable identity; never click the target itself."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Follow the run-local drive-go2 operating skill when using Go2 tools.",
                "The skill file is written at `.agents/skills/drive-go2/SKILL.md`; the same operating notes are embedded below so MCP-only runs do not need shell/file access to read them.",
                "",
                "Embedded drive-go2 operating skill:",
                DRIVE_GO2_SKILL_TEXT.strip(),
                "",
            ]
        )
    if tool_profile is not None and not pixelgoal_enabled:
        lines.extend(go2_harness_profile_prompt_lines(tool_profile))
        lines.append("")
    if mode == "readonly":
        lines.append(
            "This run is READONLY: use observe/status tools only. Do not call movement tools."
        )
    elif mode == "authority-proxy":
        lines.append(
            "This run uses an authority-proxy MCP endpoint: movement tools are allowed only inside the operator-approved envelope; refusals are safety evidence."
        )
    elif mode == "admin-override-debug":
        lines.append(
            "This run is ADMIN-OVERRIDE DEBUG only: never treat movement success as benchmark evidence."
        )
    if use_goal_tools:
        lines.extend(
            [
                "Before doing substantive work, call the built-in create_goal tool with the task objective above.",
                'Do not call get_goal immediately after create_goal; use get_goal/update_goal only after context loss or when progress needs re-anchoring. Call update_goal(status="complete") only when the task is actually complete, or status="blocked" only at a true impasse.',
            ]
        )
    return "\n".join(lines) + "\n"


def write_run_local_codex_config(
    *,
    codex_home: Path,
    source_codex_home: Path,
    mcp_url: str,
    server_name: str = "dimos",
    model: str | None = None,
    model_reasoning_effort: str | None = "medium",
    model_reasoning_summary: str | None = "detailed",
    service_tier: str | None = None,
    mcp_tool_timeout_sec: int | None = None,
) -> Path:
    """Write a minimal run-local Codex config with a DimOS MCP entry."""

    source_config = _load_codex_config(source_codex_home / "config.toml")
    root_config = {
        key: source_config[key]
        for key in MINIMAL_CODEX_ROOT_KEYS
        if key in source_config and _is_toml_scalarish(source_config[key])
    }
    if model:
        root_config["model"] = model
    # Do not silently inherit a global high/xhigh reasoning profile into live
    # robot experiments. The caller must choose the live-run effort explicitly,
    # or leave it unset to let Codex use its own default.
    root_config.pop("model_reasoning_effort", None)
    if model_reasoning_effort:
        root_config["model_reasoning_effort"] = model_reasoning_effort
    # Surface the model's reasoning summary so the operator console can show the
    # agent's thinking. Set explicitly (not inherited from the global profile)
    # and on by default; pass model_reasoning_summary=None to disable.
    root_config.pop("model_reasoning_summary", None)
    if model_reasoning_summary:
        root_config["model_reasoning_summary"] = model_reasoning_summary
    if service_tier:
        root_config["service_tier"] = service_tier

    lines = [
        "# Auto-generated by DimOS external harness preparation.",
        "# Contains only model/provider settings plus the DimOS MCP server entry.",
        "# Global MCP servers, hooks, project trust, memories, and personal skills are omitted.",
        "",
    ]
    for key in MINIMAL_CODEX_ROOT_KEYS:
        if key in root_config:
            lines.append(f"{key} = {_toml_value(root_config[key])}")
    if root_config:
        lines.append("")

    provider_name = root_config.get("model_provider")
    provider_tables = source_config.get("model_providers")
    if isinstance(provider_name, str) and isinstance(provider_tables, dict):
        provider_config = provider_tables.get(provider_name)
        if isinstance(provider_config, dict):
            safe_provider_config = _safe_provider_config(provider_config)
            _write_toml_table(lines, f"model_providers.{provider_name}", safe_provider_config)

    for table in MINIMAL_CODEX_TABLE_KEYS:
        value = source_config.get(table)
        if isinstance(value, dict):
            _write_toml_table(lines, table, _toml_safe_mapping(value))

    mcp_config: dict[str, Any] = {"url": mcp_url}
    if mcp_tool_timeout_sec is not None:
        mcp_config["tool_timeout_sec"] = mcp_tool_timeout_sec
    _write_toml_table(lines, f"mcp_servers.{server_name}", mcp_config)
    _write_toml_table(lines, "features", dict.fromkeys(DISABLED_CODEX_FEATURES, False))

    config_path = codex_home / "config.toml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _codex_facing_mcp_url(
    *,
    mode: CodexHarnessMode,
    upstream_mcp_url: str,
    proxy_url: str | None,
) -> str:
    if mode == "authority-proxy":
        resolved_proxy_url = (proxy_url or DEFAULT_AUTHORITY_PROXY_URL).strip()
        if not resolved_proxy_url:
            raise ValueError("authority-proxy mode requires a non-empty proxy_url")
        return resolved_proxy_url
    if proxy_url:
        raise ValueError("proxy_url is only valid in authority-proxy mode")
    return upstream_mcp_url


def _resolve_tool_profile(
    *,
    mode: CodexHarnessMode,
    tool_profile: str,
) -> Go2HarnessToolProfile:
    if mode == "readonly" and not tool_profile:
        return go2_harness_tool_profile("readonly_inspection")
    return go2_harness_tool_profile(tool_profile or "vision_odom_local_motion")


def _resolve_authority_movement_tools(
    *,
    explicit_tools: tuple[str, ...] | list[str],
    tool_profile: Go2HarnessToolProfile,
) -> tuple[str, ...]:
    tools = tuple(explicit_tools) or tool_profile.default_authority_movement_tools
    if not tools:
        raise ValueError(
            f"Go2 harness tool profile '{tool_profile.name}' grants no movement tools; "
            "use readonly mode or pass explicit --authority-tool values."
        )
    return tuple(sorted({tool for tool in tools if tool}))


def _authority_envelope_audit(
    *,
    task: str,
    mode: CodexHarnessMode,
    scope: str,
    area: str,
    duration_s: float,
    allowed_movement_tools: tuple[str, ...] | list[str],
    max_movement_calls: int | None,
    tool_profile_name: str = "",
) -> dict[str, Any] | None:
    if mode != "authority-proxy":
        return None
    tools = tuple(sorted({tool for tool in allowed_movement_tools if tool}))
    if not tools:
        raise ValueError("authority-proxy mode requires at least one allowed movement tool")
    if duration_s <= 0.0:
        raise ValueError("authority_duration_s must be positive")
    if max_movement_calls is not None and max_movement_calls <= 0:
        raise ValueError("max_movement_calls must be positive when set")
    return {
        "task_goal": task.strip(),
        "scope": scope.strip(),
        "area": area.strip(),
        "duration_s": float(duration_s),
        "allowed_movement_tools": list(tools),
        "max_movement_calls": max_movement_calls,
        "tool_profile": tool_profile_name or None,
        "approval_required": True,
    }


def _battery_gate_audit() -> dict[str, Any]:
    return {
        "min_start_percent": DEFAULT_MIN_START_BATTERY_PERCENT,
        "critical_percent": DEFAULT_CRITICAL_BATTERY_PERCENT,
        "critical_allows_only": [
            "full_lie_down",
            "go2_robot_status",
            "lie_down",
            "motion_safety_status",
            "stop_movement",
            "stop_navigation",
        ],
    }


def _authority_proxy_command(
    *,
    upstream_mcp_url: str,
    proxy_url: str,
    task: str,
    scope: str,
    area: str,
    duration_s: float,
    allowed_movement_tools: tuple[str, ...] | list[str],
    max_movement_calls: int | None,
    tool_profile_name: str,
) -> list[str]:
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError("proxy_url must include scheme, host, and port")
    command = [
        *_dimos_cli_command(),
        "mcp",
        "authority-proxy",
        "--upstream-mcp-url",
        upstream_mcp_url,
        "--proxy-host",
        parsed.hostname,
        "--proxy-port",
        str(parsed.port),
        "--task",
        task,
        "--scope",
        scope,
        "--area",
        area,
        "--duration-s",
        str(float(duration_s)),
    ]
    for tool in sorted({tool for tool in allowed_movement_tools if tool}):
        command.extend(["--authority-tool", tool])
    if tool_profile_name:
        command.extend(["--tool-profile", tool_profile_name])
    if max_movement_calls is not None:
        command.extend(["--max-movement-calls", str(max_movement_calls)])
    return command


def _dimos_cli_command() -> list[str]:
    """Run the proxy with the same interpreter as the operator."""

    return [sys.executable, "-m", "dimos.robot.cli.dimos"]


def _write_drive_go2_skill(codex_home: Path) -> Path:
    return copy_skill("drive-go2", codex_home / "skills")


def _write_workspace_drive_go2_skill(workspace: Path) -> Path:
    return copy_skill("drive-go2", workspace / ".agents" / "skills")


def _write_workspace_go2_pixelgoal_skill(workspace: Path) -> Path:
    return copy_skill("go2-pixelgoal", workspace / ".agents" / "skills")


def _write_workspace_global_navigation_pixelgoal_skill(workspace: Path) -> Path:
    return copy_skill("global-navigation-pixelgoal", workspace / ".agents" / "skills")


def _copy_run_local_codex_auth(source_home: Path, codex_home: Path) -> list[str]:
    copied: list[str] = []
    for name in ("installation_id", "auth.json"):
        source = source_home / name
        destination = codex_home / name
        if not source.exists() or destination.exists():
            continue
        shutil.copy2(source, destination)
        copied.append(name)
    return copied


def _load_codex_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def _safe_provider_config(config: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in _SAFE_PROVIDER_KEYS:
        if key not in config:
            continue
        value = config[key]
        if key in {"env_key", "bearer_token_env_var"}:
            if isinstance(value, str) and _ENV_VAR_RE.match(value):
                safe[key] = value
            continue
        if key == "query_params" and isinstance(value, dict):
            safe_query = _toml_safe_mapping(value)
            if safe_query:
                safe[key] = safe_query
            continue
        if _is_toml_scalarish(value):
            safe[key] = value
    return safe


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _toml_safe_mapping(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if _secretish_key(str(key)):
            continue
        if _is_toml_scalarish(item):
            safe[str(key)] = item
        elif isinstance(item, dict):
            nested = _toml_safe_mapping(item)
            if nested:
                safe[str(key)] = nested
    return safe


def _secretish_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("token", "secret", "password", "api_key"))


def _is_toml_scalarish(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_toml_scalarish(item) and not isinstance(item, dict) for item in value)
    return False


def _write_toml_table(lines: list[str], table: str, values: dict[str, Any]) -> None:
    if not values:
        return
    scalar_items = {key: value for key, value in values.items() if not isinstance(value, dict)}
    nested_items = {key: value for key, value in values.items() if isinstance(value, dict)}
    if scalar_items:
        lines.append(f"[{table}]")
        for key in sorted(scalar_items):
            lines.append(f"{key} = {_toml_value(scalar_items[key])}")
        lines.append("")
    for key in sorted(nested_items):
        _write_toml_table(lines, f"{table}.{key}", nested_items[key])


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def _codex_command(
    *,
    workspace: Path,
    sandbox: str,
    dangerously_bypass_sandbox: bool,
    config_overrides: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        "codex",
        "exec",
        "--json",
        "--ignore-rules",
        "--skip-git-repo-check",
    ]
    for feature in DISABLED_CODEX_FEATURES:
        command.extend(["--disable", feature])
    for key, value in (config_overrides or {}).items():
        command.extend(["-c", f"{key}={_toml_value(value)}"])
    command.extend(["-C", str(workspace)])
    if dangerously_bypass_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--sandbox", sandbox])
    command.append("-")
    return command


def _shell_command(
    command: list[str],
    *,
    env: dict[str, str],
    stdin_path: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    timeout_s: float | None = None,
) -> str:
    env_parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    command_text = shlex.join(command)
    if timeout_s is not None:
        command_text = f"timeout --preserve-status {float(timeout_s):g}s {command_text}"
    parts = [*env_parts, command_text, "<", shlex.quote(str(stdin_path))]
    if stdout_path is not None:
        parts.extend([">", shlex.quote(str(stdout_path))])
    if stderr_path is not None:
        parts.extend(["2>", shlex.quote(str(stderr_path))])
    return " ".join(parts)


def _write_launch_audit(run: CodexHarnessRun) -> None:
    payload = run.model_dump()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["runner"] = "codex_exec"
    payload["image_delivery_gate"] = "requires_direct_mcp_image_content"
    payload["model_visible_image_content"] = True
    payload["time_budgets"] = {
        "codex_exec_timeout_s": run.codex_exec_timeout_s,
        "mcp_smoke_timeout_s": run.mcp_smoke_timeout_s,
        "authority_duration_s": (run.authority_envelope or {}).get("duration_s")
        if run.authority_envelope is not None
        else None,
        "max_movement_calls": (run.authority_envelope or {}).get("max_movement_calls")
        if run.authority_envelope is not None
        else None,
    }
    payload["codex_runtime_config"] = {
        "model": run.model,
        "model_reasoning_effort": run.model_reasoning_effort,
        "service_tier": run.service_tier,
        "reasoning_effort_inherited_from_source": False,
    }
    payload["mcp_smoke_gate"] = {
        "required": run.mcp_smoke_required,
        "status": "prepared_not_run",
        "transport": run.mcp_smoke_transport,
        "model_reasoning_effort": DEFAULT_MCP_SMOKE_REASONING_EFFORT,
        "prompt_path": str(run.mcp_smoke_prompt_path),
        "log_path": str(run.mcp_smoke_log_path),
        "stderr_path": str(run.mcp_smoke_stderr_path),
        "result_path": str(run.mcp_smoke_result_path),
        "required_mcp_tool_names": [
            f"mcp__{run.server_name}__{tool}" for tool in run.mcp_smoke_required_tool_names
        ],
        "shell_fallback_forbidden": True,
        "forbidden_shell_fallback": "dimos mcp",
    }
    run.audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_shareable_run_summary(
    run: CodexHarnessRun,
    *,
    smoke_result: CodexMcpSmokeGateResult | None = None,
) -> Path:
    """Write a credential-safe run summary for review/benchmark sharing.

    The shareable summary deliberately omits the run-local ``codex_home`` path,
    copied auth filenames, child-process env, and any token-bearing config
    contents. ``launch.json`` remains the local audit; this file is the safe
    handoff surface.
    """

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": "codex_exec",
        "task": run.task,
        "mode": run.mode,
        "prompt_mode": run.prompt_mode,
        "tool_profile": run.tool_profile,
        "tool_profile_summary": list(run.tool_profile_summary),
        "mcp": {
            "server_name": run.server_name,
            "mcp_url": run.mcp_url,
            "upstream_mcp_url": run.upstream_mcp_url,
            "proxy_url": run.proxy_url,
        },
        "paths": {
            "run_root": str(run.run_root),
            "workspace": str(run.workspace),
            "prompt": str(run.prompt_path),
            "launch_audit": str(run.audit_path),
            "trace_root": str(run.trace_root),
            "mcp_smoke_log": str(run.mcp_smoke_log_path),
            "mcp_smoke_result": str(run.mcp_smoke_result_path),
        },
        "codex_runtime_config": {
            "model": run.model,
            "model_reasoning_effort": run.model_reasoning_effort,
            "service_tier": run.service_tier,
            "codex_exec_timeout_s": run.codex_exec_timeout_s,
            "mcp_smoke_timeout_s": run.mcp_smoke_timeout_s,
        },
        "authority_envelope": run.authority_envelope,
        "battery_gate": run.battery_gate,
        "mcp_smoke_gate": {
            "required": run.mcp_smoke_required,
            "result": smoke_result.model_dump() if smoke_result is not None else None,
        },
        "credential_hygiene": {
            "run_local_agent_home_excluded": True,
            "agent_auth_material_excluded": True,
            "process_environment_excluded": True,
            "excluded_categories": [
                "run-local agent home",
                "agent auth files",
                "installation identifiers",
                "API credentials",
                "secret values",
                "password values",
            ],
        },
    }
    _assert_shareable_summary_is_credential_safe(payload)
    run.shareable_summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run.shareable_summary_markdown_path.write_text(
        _shareable_summary_markdown(payload),
        encoding="utf-8",
    )
    return run.shareable_summary_path


def _assert_shareable_summary_is_credential_safe(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True)
    forbidden = ("auth.json", "installation_id", "CODEX_HOME", "api_key", "secret-token")
    leaks = [item for item in forbidden if item in text]
    allowed_hygiene_markers: set[str] = set()
    unexpected = [item for item in leaks if item not in allowed_hygiene_markers]
    if unexpected:
        raise ValueError(f"shareable run summary contains credential-bearing fields: {unexpected}")


def _shareable_summary_markdown(payload: dict[str, Any]) -> str:
    mcp = payload.get("mcp") if isinstance(payload.get("mcp"), dict) else {}
    runtime = (
        payload.get("codex_runtime_config")
        if isinstance(payload.get("codex_runtime_config"), dict)
        else {}
    )
    profile = payload.get("tool_profile") if isinstance(payload.get("tool_profile"), dict) else {}
    smoke = payload.get("mcp_smoke_gate") if isinstance(payload.get("mcp_smoke_gate"), dict) else {}
    smoke_result = smoke.get("result") if isinstance(smoke.get("result"), dict) else None
    smoke_status = (
        "pass"
        if isinstance(smoke_result, dict) and smoke_result.get("ok") is True
        else "prepared_not_run"
        if smoke_result is None
        else "fail"
    )
    return "\n".join(
        [
            "# DimOS Codex Harness Run Summary",
            "",
            f"- Task: {payload.get('task')}",
            f"- Mode: {payload.get('mode')}",
            f"- Tool profile: {profile.get('name') or '(none)'}",
            f"- MCP URL: {mcp.get('mcp_url')}",
            f"- Upstream MCP URL: {mcp.get('upstream_mcp_url')}",
            f"- Model: {runtime.get('model') or '(config/default)'}",
            f"- Reasoning effort: {runtime.get('model_reasoning_effort') or '(default)'}",
            f"- Codex timeout: {runtime.get('codex_exec_timeout_s')}s",
            f"- MCP smoke: {smoke_status}",
            "",
            "Credential hygiene: run-local codex_home/auth/env are excluded from this summary.",
            "",
        ]
    )


def _update_launch_smoke_audit(
    run: CodexHarnessRun,
    result: CodexMcpSmokeGateResult,
) -> None:
    if not run.audit_path.exists():
        return
    try:
        payload = json.loads(run.audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    gate = payload.get("mcp_smoke_gate")
    if not isinstance(gate, dict):
        gate = {}
        payload["mcp_smoke_gate"] = gate
    gate["status"] = "pass" if result.ok else "fail"
    gate["result"] = result.model_dump()
    run.audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
