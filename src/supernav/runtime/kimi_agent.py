from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from supernav.runtime.agents import AgentResult, model_name
from supernav.runtime.mcp import resolve_mcp_spec
from supernav.runtime.instructions import agent_instructions as default_agent_instructions
from supernav.runtime.config import write_json
from supernav.runtime.skill_runtime import copy_snapshot_skills
from supernav.runtime.streams import _clean_tool_name, _text_from_content, iter_json_lines
from supernav.runtime.providers import configured_provider, kimi_provider_config, kimi_provider_environment, validate_provider_environment

PROXY_ENV_KEYS = {
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
}

KIMI_AUTH_RELATIVE_PATHS = (
    "auth.json",
    "credentials.json",
    ".auth.json",
    "config.toml",
    "tui.toml",
    "device_id",
    "oauth/kimi-code",
    "credentials/kimi-code.json",
    "mcp.json",
)

KIMI_SESSION_RE = re.compile(
    r"session_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def canonicalize_kimi(
    raw_path: str | Path, out_path: str | Path
) -> List[Dict[str, Any]]:
    raw = Path(raw_path)
    source = _preferred_parse_source(raw)
    events: List[Dict[str, Any]] = []
    tool_names_by_id: Dict[str, str] = {}
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    for idx, data in enumerate(iter_json_lines(source)):
        timestamp = data.get("timestamp") or data.get("time")
        row_type = str(data.get("type") or data.get("event") or data.get("kind") or "")
        role = str(data.get("role") or "")
        if role == "assistant":
            text = _kimi_text(data)
            if text:
                events.append(
                    {
                        "type": "assistant_text",
                        "text": text,
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            tool_calls = data.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, Mapping):
                        _append_openai_tool_call(
                            tool_call,
                            events=events,
                            raw_index=idx,
                            timestamp=timestamp,
                            tool_names_by_id=tool_names_by_id,
                            seen_tool_calls=seen_tool_calls,
                        )
            continue
        if role == "tool":
            _append_tool_result(
                data.get("content"),
                payload=data,
                events=events,
                raw_index=idx,
                timestamp=timestamp,
                tool_names_by_id=tool_names_by_id,
                seen_tool_results=seen_tool_results,
            )
            continue
        if row_type == "context.append_loop_event":
            loop_event = (
                data.get("event") if isinstance(data.get("event"), dict) else {}
            )
            _consume_kimi_event(
                loop_event,
                events=events,
                raw_index=idx,
                timestamp=timestamp or loop_event.get("time"),
                tool_names_by_id=tool_names_by_id,
                seen_tool_calls=seen_tool_calls,
                seen_tool_results=seen_tool_results,
            )
            continue
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
        typ = str(payload.get("type") or row_type)
        if typ in {"tool.call", "tool_call", "function_call"}:
            _append_tool_call(
                payload,
                events=events,
                raw_index=idx,
                timestamp=timestamp,
                tool_names_by_id=tool_names_by_id,
                seen_tool_calls=seen_tool_calls,
            )
            continue
        if typ == "function":
            _append_openai_tool_call(
                payload,
                events=events,
                raw_index=idx,
                timestamp=timestamp,
                tool_names_by_id=tool_names_by_id,
                seen_tool_calls=seen_tool_calls,
            )
            continue
        if typ in {
            "tool.result",
            "tool_result",
            "tool_call_result",
            "function_call_output",
        }:
            _append_tool_result(
                payload.get("result") if "result" in payload else payload,
                payload=payload,
                events=events,
                raw_index=idx,
                timestamp=timestamp,
                tool_names_by_id=tool_names_by_id,
                seen_tool_results=seen_tool_results,
            )
            continue
        if typ in {"content.part", "message", "assistant", "assistant_message"}:
            text = _kimi_text(payload)
            if text:
                events.append(
                    {
                        "type": "assistant_text",
                        "text": text,
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            continue
        if typ in {
            "step.end",
            "turn_end",
            "assistant_turn_end",
            "done",
            "complete",
            "completed",
        }:
            events.append(
                {
                    "type": "run_result",
                    "timestamp": timestamp,
                    "raw": data,
                    "raw_index": idx,
                }
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return events


def _preferred_parse_source(raw_path: Path) -> Path:
    wire = raw_path.parent / "kimi_wire.jsonl"
    if wire.is_file():
        return wire
    export_path = raw_path.parent / "kimi_session_export.zip"
    if export_path.is_file():
        extracted = _extract_kimi_wire(export_path, raw_path.parent / "kimi_wire.jsonl")
        if extracted.is_file():
            return extracted
    return raw_path


def _consume_kimi_event(
    event: Mapping[str, Any],
    *,
    events: List[Dict[str, Any]],
    raw_index: int,
    timestamp: Any,
    tool_names_by_id: Dict[str, str],
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
) -> None:
    event_type = str(event.get("type") or "")
    if event_type == "tool.call":
        _append_tool_call(
            event,
            events=events,
            raw_index=raw_index,
            timestamp=timestamp,
            tool_names_by_id=tool_names_by_id,
            seen_tool_calls=seen_tool_calls,
        )
        return
    if event_type == "tool.result":
        _append_tool_result(
            event.get("result") if "result" in event else event,
            payload=event,
            events=events,
            raw_index=raw_index,
            timestamp=timestamp,
            tool_names_by_id=tool_names_by_id,
            seen_tool_results=seen_tool_results,
        )
        return
    if event_type == "content.part":
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        text = _text_from_content(part.get("text") or part.get("content"))
        thinking = _text_from_content(part.get("think") or part.get("thinking"))
        if thinking:
            events.append(
                {
                    "type": "assistant_reasoning",
                    "text": thinking,
                    "timestamp": timestamp,
                    "raw_index": raw_index,
                }
            )
        if text:
            events.append(
                {
                    "type": "assistant_text",
                    "text": text,
                    "timestamp": timestamp,
                    "raw_index": raw_index,
                }
            )
        return
    if event_type == "step.end":
        events.append(
            {
                "type": "run_result",
                "timestamp": timestamp,
                "raw": dict(event),
                "raw_index": raw_index,
            }
        )


def _append_tool_call(
    payload: Mapping[str, Any],
    *,
    events: List[Dict[str, Any]],
    raw_index: int,
    timestamp: Any,
    tool_names_by_id: Dict[str, str],
    seen_tool_calls: set[str],
) -> None:
    name = payload.get("name") or payload.get("tool")
    if not name:
        return
    call_id = (
        payload.get("toolCallId")
        or payload.get("uuid")
        or payload.get("id")
        or payload.get("call_id")
    )
    if isinstance(call_id, str):
        if call_id in seen_tool_calls:
            return
        seen_tool_calls.add(call_id)
        tool_names_by_id[call_id] = str(name)
    raw_input = (
        payload.get("args") or payload.get("arguments") or payload.get("input") or {}
    )
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError:
            raw_input = {"raw": raw_input}
    events.append(
        {
            "type": "tool_call",
            "name": _clean_tool_name(str(name)),
            "input": raw_input if isinstance(raw_input, dict) else {"value": raw_input},
            "timestamp": timestamp,
            "raw_index": raw_index,
        }
    )


def _append_openai_tool_call(
    tool_call: Mapping[str, Any],
    *,
    events: List[Dict[str, Any]],
    raw_index: int,
    timestamp: Any,
    tool_names_by_id: Dict[str, str],
    seen_tool_calls: set[str],
) -> None:
    function = (
        tool_call.get("function")
        if isinstance(tool_call.get("function"), Mapping)
        else {}
    )
    payload: Dict[str, Any] = dict(tool_call)
    if function:
        payload["name"] = function.get("name")
        payload["arguments"] = function.get("arguments")
    _append_tool_call(
        payload,
        events=events,
        raw_index=raw_index,
        timestamp=timestamp,
        tool_names_by_id=tool_names_by_id,
        seen_tool_calls=seen_tool_calls,
    )


def _append_tool_result(
    result: Any,
    *,
    payload: Mapping[str, Any],
    events: List[Dict[str, Any]],
    raw_index: int,
    timestamp: Any,
    tool_names_by_id: Mapping[str, str],
    seen_tool_results: set[str],
) -> None:
    call_id = (
        payload.get("toolCallId")
        or payload.get("parentUuid")
        or payload.get("tool_call_id")
        or payload.get("call_id")
        or payload.get("id")
    )
    if isinstance(call_id, str):
        if call_id in seen_tool_results:
            return
        seen_tool_results.add(call_id)
    name = payload.get("name") or payload.get("tool")
    if not name and isinstance(call_id, str):
        name = tool_names_by_id.get(call_id)
    if not name:
        name = "unknown"
    events.append(
        {
            "type": "tool_result",
            "name": _clean_tool_name(str(name)),
            "content": _kimi_result_text(result),
            "timestamp": timestamp,
            "raw_index": raw_index,
        }
    )


def _kimi_text(payload: Mapping[str, Any]) -> str:
    for key in ("text", "content", "message"):
        text = _text_from_content(payload.get(key))
        if text:
            return text
    return ""


def _kimi_result_text(result: Any) -> str:
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return result
            text = _text_from_content(parsed)
            if text:
                return text
            if isinstance(parsed, Mapping):
                return _kimi_result_text(parsed)
            return result
    if isinstance(result, Mapping):
        output = result.get("output")
        if isinstance(output, str):
            return output
        text = _text_from_content(output)
        if text:
            return text
        content = result.get("content")
        text = _text_from_content(content)
        if text:
            return text
        return json.dumps(result, ensure_ascii=False)
    text = _text_from_content(result)
    return text or str(result or "")


def _extract_kimi_wire(export_path: Path, out_path: Path) -> Path:
    with zipfile.ZipFile(export_path) as zf:
        names = zf.namelist()
        wire_name = "agents/main/wire.jsonl"
        if wire_name not in names:
            candidates = sorted(name for name in names if name.endswith("/wire.jsonl"))
            if not candidates:
                return out_path
            wire_name = candidates[0]
        out_path.write_bytes(zf.read(wire_name))
    return out_path


def _kimi_session_id_from_raw(raw_path: Path) -> str:
    for line in reversed(
        raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if not line.startswith("{"):
            match = KIMI_SESSION_RE.search(line)
            if match:
                return match.group(0)
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            match = KIMI_SESSION_RE.search(line)
            if match:
                return match.group(0)
            continue
        session_id = data.get("session_id")
        if isinstance(session_id, str) and KIMI_SESSION_RE.fullmatch(session_id):
            return session_id
        command = data.get("command")
        if isinstance(command, str):
            match = KIMI_SESSION_RE.search(command)
            if match:
                return match.group(0)
        text = json.dumps(data, ensure_ascii=False, default=str)
        match = KIMI_SESSION_RE.search(text)
        if match:
            return match.group(0)
    return ""


class KimiAgent:
    name = "kimi"

    def validate_environment(self, *, agent_cfg: Mapping[str, Any]) -> None:
        validate_provider_environment(agent_cfg)

    def prepare_project(
        self,
        *,
        run_dir: Path,
        workspace_root: Path,
        config: Mapping[str, Any],
        arm_cfg: Mapping[str, Any],
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
    ) -> Path:
        project_dir = run_dir / "kimi_project"
        kimi_home = project_dir / "kimi_home"
        app_home = kimi_home / ".kimi-code"
        project_dir.mkdir(parents=True, exist_ok=True)
        app_home.mkdir(parents=True, exist_ok=True)
        snapshot_root = arm_cfg.get("_skill_snapshot_root")
        if str(arm_cfg.get("skill_mode") or "").lower() == "native":
            if not snapshot_root:
                raise ValueError("native skill arm requires a verified snapshot")
            copy_snapshot_skills(
                Path(str(snapshot_root)), project_dir / ".kimi-code" / "skills"
            )
        source_home = Path(
            str(agent_cfg.get("source_home") or Path.home() / ".kimi-code")
        ).expanduser()
        provider = configured_provider(agent_cfg)
        if provider:
            selected = model_name(model_cfg)
            if not selected:
                raise ValueError("explicit Kimi API provider requires a model")
            for config_root in (kimi_home, app_home):
                (config_root / "config.toml").write_text(kimi_provider_config(provider, selected), encoding="utf-8")
        else:
            _copy_kimi_auth(source_home=source_home, kimi_home=kimi_home)
        spec = resolve_mcp_spec(workspace_root=workspace_root, config=config, arm_cfg=arm_cfg)
        _write_kimi_mcp_configs(
            server_name=spec.name,
            kimi_home=kimi_home,
            project_dir=project_dir,
            mcp_url="http://127.0.0.1:0/mcp",
        )
        agent_instructions = default_agent_instructions(config, "kimi")
        if str(arm_cfg.get("skill_mode") or "").lower() == "native":
            skill_name = str(arm_cfg.get("skill_name") or "")
            agent_instructions += (
                f"Before any scene action, invoke `/skill:{skill_name}` and follow the "
                "loaded instructions.\n"
            )
        (project_dir / "AGENTS.md").write_text(agent_instructions, encoding="utf-8")
        write_json(
            project_dir / "kimi_project.json",
            {
                "kimi_home": str(kimi_home),
                "app_home": str(app_home),
                "source_home": str(source_home),
                "workspace_root": str(workspace_root),
                "config": dict(config),
                "arm_cfg": dict(arm_cfg),
                "agent": dict(agent_cfg),
                "model": dict(model_cfg),
            },
        )
        return project_dir

    def run(
        self,
        *,
        prompt: str,
        run_dir: Path,
        project_dir: Path,
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        timeout_s: int,
    ) -> AgentResult:
        raw_path = run_dir / "raw.jsonl"
        validate_provider_environment(agent_cfg)
        stderr_path = run_dir / "stderr.log"
        export_path = run_dir / "kimi_session_export.zip"
        wire_path = run_dir / "kimi_wire.jsonl"
        for stale_path in (export_path, wire_path):
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass
        kimi_home = project_dir / "kimi_home"
        mcp_proc: subprocess.Popen[str] | None = None
        mcp_port = _free_local_port()
        mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
        meta = json.loads((project_dir / "kimi_project.json").read_text(encoding="utf-8"))
        spec = resolve_mcp_spec(workspace_root=Path(meta["workspace_root"]), config=meta.get("config", {}), arm_cfg=meta.get("arm_cfg", {}))
        _write_kimi_mcp_configs(
            kimi_home=kimi_home, project_dir=project_dir, mcp_url=mcp_url, server_name=spec.name
        )
        mcp_proc = _start_mcp_server(
            project_dir=project_dir, stderr_path=stderr_path, port=mcp_port
        )

        command = [str(agent_cfg.get("command") or "kimi")]
        selected_model = model_name(model_cfg)
        if selected_model and not configured_provider(agent_cfg):
            command.extend(["--model", selected_model])
        if bool(agent_cfg.get("yolo", False)):
            command.append("-y")
        extra_args = agent_cfg.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(_without_skills_dir([str(x) for x in extra_args]))
        skills_root = project_dir / ".kimi-code" / "skills"
        if skills_root.is_dir():
            command.extend(["--skills-dir", str(skills_root)])
        command.extend(
            [
                "-p",
                prompt,
                "--output-format",
                str(agent_cfg.get("output_format") or "stream-json"),
            ]
        )

        env = dict(os.environ)
        env["HOME"] = str(kimi_home)
        env["KIMI_HOME"] = str(kimi_home / ".kimi-code")
        env["KIMI_CODE_HOME"] = str(kimi_home / ".kimi-code")
        env.update(kimi_provider_environment(agent_cfg, str(selected_model or "")))

        start = time.monotonic()
        try:
            with (
                raw_path.open("w", encoding="utf-8") as out,
                stderr_path.open("a", encoding="utf-8") as err,
            ):
                proc = subprocess.run(
                    command,
                    stdout=out,
                    stderr=err,
                    text=True,
                    timeout=timeout_s,
                    cwd=str(project_dir),
                    env=env,
                )
        finally:
            if mcp_proc is not None:
                _terminate_process(mcp_proc, stderr_path=stderr_path)
        duration_s = time.monotonic() - start

        export_rc: Optional[int] = None
        kimi_session_id = _kimi_session_id_from_raw(raw_path)
        if proc.returncode == 0:
            export_rc = _export_kimi_session(
                command=str(agent_cfg.get("command") or "kimi"),
                session_id=kimi_session_id,
                kimi_home=kimi_home,
                project_dir=project_dir,
                export_path=export_path,
                stderr_path=stderr_path,
                timeout_s=int(agent_cfg.get("export_timeout_s", 60)),
                env=env,
            )
            if export_path.is_file():
                try:
                    _extract_kimi_wire(export_path, wire_path)
                except (OSError, zipfile.BadZipFile):
                    pass
        return AgentResult(
            proc.returncode,
            duration_s,
            command,
            raw_path,
            stderr_path,
            session_id=None,
            export_path=export_path if export_path.is_file() else None,
            delete_returncode=export_rc,
        )

    def parse(self, raw_path: str | Path, out_path: str | Path) -> List[Dict[str, Any]]:
        return canonicalize_kimi(raw_path, out_path)


def _without_skills_dir(argv: List[str]) -> List[str]:
    """Drop user-provided skill roots so only the benchmark snapshot is visible."""
    cleaned: List[str] = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg == "--skills-dir":
            skip_value = True
            continue
        if arg.startswith("--skills-dir="):
            continue
        cleaned.append(arg)
    return cleaned


def _copy_kimi_auth(*, source_home: Path, kimi_home: Path) -> None:
    app_home = kimi_home / ".kimi-code"
    for relative in KIMI_AUTH_RELATIVE_PATHS:
        src = source_home / relative
        if src.is_file():
            for target_root in (kimi_home, app_home):
                dst = target_root / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def _write_kimi_mcp_configs(
    *, kimi_home: Path, project_dir: Path, mcp_url: str, server_name: str = "habitat-gs"
) -> None:
    payload = {"mcpServers": {server_name: {"url": mcp_url}}}
    for path in (
        kimi_home / "mcp.json",
        kimi_home / ".kimi-code" / "mcp.json",
        project_dir / ".kimi-code" / "mcp.json",
    ):
        write_json(path, payload)
    _ensure_kimi_permission_rule(kimi_home / "config.toml", server_name)
    _ensure_kimi_permission_rule(kimi_home / ".kimi-code" / "config.toml", server_name)


def _ensure_kimi_permission_rule(config_path: Path, server_name: str = "habitat-gs") -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        config_path.read_text(encoding="utf-8", errors="replace")
        if config_path.is_file()
        else ""
    )
    if f'pattern = "mcp__{server_name}__*"' in text:
        return
    block = "\n".join(
        [
            "",
            "[[permission.rules]]",
            'decision = "allow"',
            f'pattern = "mcp__{server_name}__*"',
            "",
        ]
    )
    config_path.write_text(text.rstrip() + block, encoding="utf-8")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_mcp_server(
    *, project_dir: Path, stderr_path: Path, port: int
) -> subprocess.Popen[str]:
    meta = json.loads((project_dir / "kimi_project.json").read_text(encoding="utf-8"))
    workspace_root = Path(meta["workspace_root"])
    config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    arm_cfg = meta.get("arm_cfg", {}) if isinstance(meta.get("arm_cfg"), dict) else {}
    if not config:
        raise ValueError("kimi_project.json must contain the resolved experiment config")
    spec = resolve_mcp_spec(
        workspace_root=workspace_root, config=config, arm_cfg=arm_cfg
    )
    env = dict(os.environ)
    env.update(spec.environment)
    # The benchmark owns episode selection and spawn; inject the private
    # per-run defaults (HAB_MCP_INIT_DEFAULTS_JSON) like the codex backend.
    benchmark_env = arm_cfg.get("_benchmark_process_environment")
    if isinstance(benchmark_env, dict):
        env.update({str(k): str(v) for k, v in benchmark_env.items()})
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = env.get("NO_PROXY") or "127.0.0.1,localhost,::1"
    env["no_proxy"] = env.get("no_proxy") or "127.0.0.1,localhost,::1"
    if spec.http_args is None:
        raise ValueError("Kimi requires mcp.http_args for a streamable-http server")
    cmd = [spec.command, *spec.http_args, "--port", str(port)]
    with stderr_path.open("a", encoding="utf-8") as err:
        err.write(f"[kimi-agent] starting MCP: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=err,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(workspace_root),
            env=env,
            close_fds=True,
        )
    _wait_for_port("127.0.0.1", port, proc=proc, timeout_s=30)
    return proc


def _wait_for_port(
    host: str, port: int, *, proc: subprocess.Popen[str], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Kimi MCP server exited early with returncode {proc.returncode}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for Kimi MCP server at {host}:{port}")


def _terminate_process(proc: subprocess.Popen[str], *, stderr_path: Path) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    with stderr_path.open("a", encoding="utf-8") as err:
        err.write(f"[kimi-agent] MCP exited with returncode {proc.returncode}\n")


def _export_kimi_session(
    *,
    command: str,
    session_id: str,
    kimi_home: Path,
    project_dir: Path,
    export_path: Path,
    stderr_path: Path,
    timeout_s: int,
    env: Mapping[str, str],
) -> int:
    export_path = export_path.resolve()
    export_cmd = [command, "export"]
    if session_id:
        export_cmd.append(session_id)
    export_cmd.extend(
        ["--output", str(export_path), "--yes", "--no-include-global-log"]
    )
    with stderr_path.open("a", encoding="utf-8") as err:
        if session_id:
            err.write(f"[kimi-agent] exporting Kimi session {session_id}\n")
        else:
            err.write(
                "[kimi-agent] exporting most recent Kimi session; no session id found in raw output\n"
            )
        proc = subprocess.run(
            export_cmd,
            stdout=err,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            cwd=str(project_dir),
            env=dict(env),
            check=False,
        )
    return int(proc.returncode)
