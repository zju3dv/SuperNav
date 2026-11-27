from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

from supernav.runtime.config import write_json
from supernav.runtime.mcp import resolve_mcp_spec


@dataclass
class AgentResult:
    returncode: int
    duration_s: float
    command: List[str]
    raw_path: Path
    stderr_path: Path
    session_id: Optional[str] = None
    export_path: Optional[Path] = None
    delete_returncode: Optional[int] = None
    timed_out: bool = False
    error: Optional[str] = None


class HarnessAgent(Protocol):
    name: str

    def prepare_project(
        self,
        *,
        run_dir: Path,
        workspace_root: Path,
        config: Mapping[str, Any],
        arm_cfg: Mapping[str, Any],
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
    ) -> Path: ...

    def run(
        self,
        *,
        prompt: str,
        run_dir: Path,
        project_dir: Path,
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        timeout_s: int,
    ) -> AgentResult: ...

    def parse(
        self,
        raw_path: str | Path,
        out_path: str | Path,
    ) -> List[Dict[str, Any]]: ...


def run_agent(
    agent: HarnessAgent,
    *,
    prompt: str,
    run_dir: Path,
    project_dir: Path,
    agent_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    timeout_s: int,
) -> AgentResult:
    """Keep partial CLI evidence available to collection after a timeout.

    The supported Codex, CodexProfile, OpenCode and Kimi clients all stream
    directly to ``raw.jsonl`` and ``stderr.log`` in ``run_dir``. Their own
    subprocess context managers kill/reap the CLI; Kimi also tears down its
    MCP server in ``finally`` before the exception reaches this boundary.
    """
    started = time.monotonic()
    try:
        return agent.run(
            prompt=prompt, run_dir=run_dir, project_dir=project_dir,
            agent_cfg=agent_cfg, model_cfg=model_cfg, timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        command = [exc.cmd] if isinstance(exc.cmd, str) else list(exc.cmd)
        # Never rewrite or synthesize client stdout/stderr: these files contain
        # the native stream, including any incomplete final line.
        return AgentResult(
            returncode=124,
            duration_s=time.monotonic() - started,
            command=command,
            raw_path=run_dir / "raw.jsonl",
            stderr_path=run_dir / "stderr.log",
            timed_out=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def selected_agent_config(config: Mapping[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    agent_name = str(config.get("agent") or "opencode").strip()
    agents = config.get("agents", {}) if isinstance(config.get("agents"), dict) else {}
    if agent_name not in agents:
        raise ValueError(f"config agent={agent_name!r} missing from agents dict")
    raw_agent_cfg = agents.get(agent_name, {})
    if not isinstance(raw_agent_cfg, dict):
        raise ValueError(f"agents.{agent_name} must be an object")
    raw_model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    return agent_name, dict(raw_agent_cfg), dict(raw_model_cfg)


def get_agent_backend(agent_name: str) -> HarnessAgent:
    name = agent_name.strip().lower()
    if name == "opencode":
        from supernav.runtime.opencode_agent import OpenCodeAgent

        return OpenCodeAgent()
    if name == "codex":
        from supernav.runtime.codex_agent import CodexAgent

        return CodexAgent()
    if name == "codex_profile":
        from supernav.runtime.codex_agent import CodexAgentProfile

        return CodexAgentProfile()
    if name == "kimi":
        from supernav.runtime.kimi_agent import KimiAgent

        return KimiAgent()

    raise ValueError(f"unknown agent={name!r}; expected one of: codex, codex_profile, kimi, opencode")


def model_name(model_cfg: Mapping[str, Any]) -> Optional[str]:
    for key in ("name", "id", "model"):
        value = model_cfg.get(key)
        if value:
            return str(value)
    return None


def mcp_env(
    *,
    workspace_root: Path,
    config: Mapping[str, Any],
    arm_cfg: Mapping[str, Any],
) -> tuple[Path, str, Dict[str, str]]:
    """Return the MCP server path, command and environment for benchmark warm-up."""
    spec = resolve_mcp_spec(
        workspace_root=workspace_root, config=config, arm_cfg=arm_cfg
    )
    return Path(spec.args[0]) if spec.args else Path(spec.command), spec.command, dict(spec.environment)


def save_command(run_dir: Path, result: AgentResult) -> None:
    write_json(
        run_dir / "command.json",
        {
            "returncode": result.returncode,
            "duration_s": result.duration_s,
            "command": result.command,
            "raw_path": str(result.raw_path),
            "stderr_path": str(result.stderr_path),
            "session_id": result.session_id,
            "export_path": str(result.export_path) if result.export_path else None,
            "delete_returncode": result.delete_returncode,
            **({"timed_out": True, "error": result.error} if result.timed_out else {}),
        },
    )


def session_id_from_raw(raw_path: Path) -> Optional[str]:
    for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = data.get("sessionID")
        if isinstance(session_id, str) and session_id:
            return session_id
        part = data.get("part") if isinstance(data.get("part"), dict) else {}
        session_id = part.get("sessionID")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None
