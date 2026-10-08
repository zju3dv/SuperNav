from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from supernav.runtime.agents import AgentResult, model_name
from supernav.runtime.mcp import resolve_mcp_spec
from supernav.runtime.instructions import agent_instructions as default_agent_instructions, agent_policy
from supernav.runtime.config import write_json
from supernav.runtime.providers import configured_provider
from supernav.runtime.streams import (
    _clean_tool_name,
    _reasoning_summary_text,
    _text_from_content,
    iter_json_lines,
)
from supernav.runtime.subagents import ensure_backend_subagent

NATIVE_SKILL_FULL_ACCESS_MARKER = ".native_skill_full_access"
CODEX_PROVIDER_EXPERIMENT = "experiment"
CODEX_PROVIDER_USER = "user"


def _effective_sandbox(project_dir: Path, agent_cfg: Mapping[str, Any]) -> Any:
    if (Path(project_dir) / NATIVE_SKILL_FULL_ACCESS_MARKER).is_file():
        return "danger-full-access"
    return agent_cfg.get("sandbox")


from supernav.runtime.skill_runtime import copy_snapshot_skills


def canonicalize_codex(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    timing_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    call_names: Dict[str, str] = {}
    completed_call_ids: set[str] = set()
    seen_agent_messages: set[str] = set()
    for idx, data in enumerate(iter_json_lines(raw_path)):
        typ = str(data.get("type") or data.get("event") or data.get("kind") or "")
        timestamp = (
            data.get("timestamp")
            or data.get("time")
            or data.get("created_at")
            or data.get("created")
        )
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        payload_type = str(payload.get("type") or "")
        if typ == "response_item" and payload_type == "function_call":
            name = payload.get("name") or payload.get("tool") or "unknown"
            namespace = str(payload.get("namespace") or "")
            if not namespace.startswith("mcp__"):
                continue
            call_id = payload.get("call_id")
            clean_name = _clean_tool_name(str(name))
            if isinstance(call_id, str):
                call_names[call_id] = clean_name
            arguments = payload.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            events.append(
                {
                    "type": "tool_call",
                    "name": clean_name,
                    "input": (
                        arguments
                        if isinstance(arguments, dict)
                        else {"value": arguments}
                    ),
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue
        if typ == "event_msg" and payload_type == "mcp_tool_call_end":
            invocation = (
                payload.get("invocation")
                if isinstance(payload.get("invocation"), dict)
                else {}
            )
            name = invocation.get("tool") or call_names.get(
                str(payload.get("call_id")), "unknown"
            )
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                completed_call_ids.add(call_id)
            result = (
                payload.get("result") if isinstance(payload.get("result"), dict) else {}
            )
            ok = result.get("Ok") if isinstance(result.get("Ok"), dict) else None
            err = result.get("Err") or result.get("Error")
            content: Any = ok.get("content") if ok else err
            events.append(
                {
                    "type": "tool_result",
                    "name": _clean_tool_name(str(name or "unknown")),
                    "content": _text_from_content(content),
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue
        if typ == "response_item" and payload_type == "function_call_output":
            if payload.get("call_id") in completed_call_ids:
                continue
            name = call_names.get(str(payload.get("call_id")), "unknown")
            if name == "unknown":
                continue
            events.append(
                {
                    "type": "tool_result",
                    "name": name,
                    "content": _text_from_content(payload.get("output")),
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue
        if typ == "event_msg" and payload_type == "agent_message":
            text = payload.get("message")
            if isinstance(text, str) and text.strip():
                seen_agent_messages.add(text)
                events.append(
                    {
                        "type": "assistant_text",
                        "text": text,
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            continue
        if typ == "response_item" and payload_type == "message":
            if payload.get("role") == "assistant":
                text = _text_from_content(payload.get("content"))
                if text in seen_agent_messages:
                    continue
                if text.strip():
                    events.append(
                        {
                            "type": "assistant_text",
                            "text": text,
                            "timestamp": timestamp,
                            "raw_index": idx,
                        }
                    )
            continue
        if typ == "response_item" and payload_type == "reasoning":
            text = _reasoning_summary_text(payload.get("summary"))
            if not text and payload.get("encrypted_content"):
                text = "reasoning (encrypted)"
            if text:
                events.append(
                    {
                        "type": "assistant_reasoning",
                        "text": text,
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            continue
        if typ == "event_msg" and payload_type == "task_complete":
            events.append(
                {
                    "type": "run_result",
                    "timestamp": timestamp,
                    "raw": data,
                    "raw_index": idx,
                }
            )
            continue
        item = data.get("item") if isinstance(data.get("item"), dict) else {}
        msg = data.get("message") if isinstance(data.get("message"), dict) else {}
        call = data.get("call") if isinstance(data.get("call"), dict) else {}
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        error = data.get("error") or item.get("error") or payload.get("error")

        combined_type = " ".join(
            str(x or "")
            for x in (
                typ,
                item.get("type"),
                msg.get("type"),
                payload.get("type"),
                call.get("type"),
            )
        ).lower()
        role = item.get("role") or msg.get("role") or data.get("role")

        tool_name = (
            data.get("name")
            or data.get("tool")
            or data.get("tool_name")
            or item.get("name")
            or item.get("tool")
            or item.get("tool_name")
            or call.get("name")
            or call.get("tool")
            or payload.get("name")
            or payload.get("tool")
        )
        tool_input = (
            data.get("input")
            or data.get("arguments")
            or item.get("input")
            or item.get("arguments")
            or call.get("input")
            or call.get("arguments")
            or payload.get("input")
            or payload.get("arguments")
            or {}
        )
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {"raw": tool_input}

        if tool_name and (
            "tool" in combined_type
            or "function_call" in combined_type
            or "mcp" in combined_type
        ):
            clean_name = _clean_tool_name(str(tool_name))
            if any(
                marker in combined_type
                for marker in ("result", "output", "complete", "completed")
            ):
                content = (
                    data.get("content")
                    or data.get("result")
                    or item.get("result")
                    or output
                    or item.get("output")
                    or payload.get("output")
                )
                if not content and error:
                    content = error.get("message") if isinstance(error, dict) else error
                if isinstance(content, dict) and isinstance(
                    content.get("content"), list
                ):
                    content = content["content"]
                events.append(
                    {
                        "type": "tool_result",
                        "name": clean_name,
                        "content": _text_from_content(content),
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            else:
                events.append(
                    {
                        "type": "tool_call",
                        "name": clean_name,
                        "input": (
                            tool_input
                            if isinstance(tool_input, dict)
                            else {"value": tool_input}
                        ),
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            continue

        text = (
            data.get("text")
            or data.get("delta")
            or data.get("content")
            or item.get("text")
            or item.get("content")
            or msg.get("content")
            or payload.get("text")
            or payload.get("content")
        )
        text_value = _text_from_content(text) if not isinstance(text, str) else text
        if text_value.strip():
            is_reasoning = any(
                marker in combined_type
                for marker in ("reasoning", "thinking", "thought")
            )
            event_type = "assistant_reasoning" if is_reasoning else "assistant_text"
            if role and str(role) not in ("assistant", "model") and not is_reasoning:
                event_type = "text"
            events.append(
                {
                    "type": event_type,
                    "text": text_value,
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue

        if any(marker in combined_type for marker in ("done", "complete", "completed")):
            events.append(
                {
                    "type": "run_result",
                    "timestamp": timestamp,
                    "raw": data,
                    "raw_index": idx,
                }
            )

    if timing_path is not None:
        _merge_codex_session_timeline(events, timing_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return events


def _merge_codex_session_timeline(
    events: List[Dict[str, Any]], timing_path: str | Path
) -> None:
    """Merge the timestamped session timeline into authoritative exec events.

    Newer Codex stdout provides the authoritative MCP start/result structure but
    omits timestamps. The session dump retains timestamps and emits one
    ``mcp_tool_call_end`` per completed MCP call. Depending on the model/runtime,
    the corresponding start is either a namespaced MCP ``function_call`` or a
    pending ``custom_tool_call``. Ordinary shell calls have already produced
    ``custom_tool_call_output`` before a later MCP completion.

    Agent messages also exist in both sources, while reasoning is session-only.
    Match stdout messages by content, then append the timestamped reasoning. This
    keeps internal ``exec`` calls out of canonical tool events without losing the
    model trace needed by a real-time replay.
    """
    path = Path(timing_path)
    if not path.is_file():
        return
    pending_starts: List[tuple[int, str | None]] = []
    timed_calls: List[tuple[str, str | None, str | None]] = []
    timed_messages: List[tuple[str, str]] = []
    timed_reasoning: List[Dict[str, Any]] = []
    task_complete_timestamps: List[str] = []
    seen_messages: set[str] = set()
    for row in iter_json_lines(path):
        timestamp = row.get("timestamp")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        typ = str(row.get("type") or "")
        payload_type = str(payload.get("type") or "")
        timestamp_text = str(timestamp) if timestamp else ""
        if typ == "event_msg" and payload_type == "agent_message":
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                seen_messages.add(message)
                if timestamp_text:
                    timed_messages.append((message, timestamp_text))
            continue
        if typ == "response_item" and payload_type == "message":
            if payload.get("role") == "assistant":
                message = _text_from_content(payload.get("content"))
                if message.strip() and message not in seen_messages and timestamp_text:
                    timed_messages.append((message, timestamp_text))
            continue
        if typ == "response_item" and payload_type == "reasoning":
            text = _reasoning_summary_text(payload.get("summary"))
            if not text and payload.get("encrypted_content"):
                text = "reasoning (encrypted)"
            if text and timestamp_text:
                timed_reasoning.append(
                    {
                        "type": "assistant_reasoning",
                        "text": text,
                        "timestamp": timestamp_text,
                        "timing_source": "codex_session",
                    }
                )
            continue
        if typ == "event_msg" and payload_type == "task_complete":
            if timestamp_text:
                task_complete_timestamps.append(timestamp_text)
            continue
        if typ == "response_item" and payload_type == "custom_tool_call":
            pending_starts.append(
                (len(pending_starts), str(timestamp) if timestamp else None)
            )
            continue
        if typ == "response_item" and payload_type == "function_call":
            namespace = str(payload.get("namespace") or "")
            if namespace.startswith("mcp__"):
                pending_starts.append(
                    (len(pending_starts), str(timestamp) if timestamp else None)
                )
            continue
        if typ == "response_item" and payload_type == "custom_tool_call_output":
            if pending_starts:
                pending_starts.pop()
            continue
        if typ != "event_msg" or payload_type != "mcp_tool_call_end":
            continue
        invocation = (
            payload.get("invocation")
            if isinstance(payload.get("invocation"), dict)
            else {}
        )
        name = _clean_tool_name(str(invocation.get("tool") or "unknown"))
        start_timestamp = pending_starts.pop()[1] if pending_starts else None
        timed_calls.append(
            (name, start_timestamp, str(timestamp) if timestamp else None)
        )

    call_events = [event for event in events if event.get("type") == "tool_call"]
    result_events = [event for event in events if event.get("type") == "tool_result"]
    call_index = 0
    result_index = 0
    for name, start_timestamp, end_timestamp in timed_calls:
        while (
            call_index < len(call_events)
            and call_events[call_index].get("name") != name
        ):
            call_index += 1
        while (
            result_index < len(result_events)
            and result_events[result_index].get("name") != name
        ):
            result_index += 1
        if call_index < len(call_events):
            if start_timestamp:
                call_events[call_index]["timestamp"] = start_timestamp
            call_index += 1
        if result_index < len(result_events):
            if end_timestamp:
                result_events[result_index]["timestamp"] = end_timestamp
            result_index += 1

    # Content matching is stable across the two Codex streams and avoids
    # assigning a nearby internal exec timestamp to an agent message.
    message_events = [
        event for event in events if event.get("type") == "assistant_text"
    ]
    message_index = 0
    for text, timestamp in timed_messages:
        while message_index < len(message_events):
            event = message_events[message_index]
            message_index += 1
            if str(event.get("text") or "") != text:
                continue
            if not event.get("timestamp"):
                event["timestamp"] = timestamp
                event["timing_source"] = "codex_session"
            break

    run_results = [event for event in events if event.get("type") == "run_result"]
    for event, timestamp in zip(
        reversed(run_results), reversed(task_complete_timestamps)
    ):
        if not event.get("timestamp"):
            event["timestamp"] = timestamp
            event["timing_source"] = "codex_session"

    # Stdout does not expose reasoning summaries in current Codex versions.
    # Avoid duplicating them if a future stdout format starts doing so.
    existing_reasoning = {
        (str(event.get("text") or ""), str(event.get("timestamp") or ""))
        for event in events
        if event.get("type") in ("assistant_reasoning", "assistant_thinking")
    }
    for event in timed_reasoning:
        key = (str(event["text"]), str(event["timestamp"]))
        if key not in existing_reasoning:
            events.append(event)
            existing_reasoning.add(key)


class CodexAgent:
    name = "codex"

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
        project_dir = run_dir / "codex_project"
        codex_home = project_dir / ".codex_home"
        project_dir.mkdir(parents=True, exist_ok=True)
        codex_home.mkdir(parents=True, exist_ok=True)
        snapshot_root = arm_cfg.get("_skill_snapshot_root")
        if str(arm_cfg.get("skill_mode") or "").lower() == "native":
            if not snapshot_root:
                raise ValueError("native skill arm requires a verified snapshot")
            copy_snapshot_skills(
                Path(str(snapshot_root)),
                codex_home / "skills",
                names=arm_cfg.get("skill_names"),
            )
            (project_dir / NATIVE_SKILL_FULL_ACCESS_MARKER).write_text(
                "Native Codex skills require shell-based read access on this host.\n",
                encoding="utf-8",
            )

        spec = resolve_mcp_spec(
            workspace_root=workspace_root,
            config=config,
            arm_cfg=arm_cfg,
        )
        private_process_env = arm_cfg.get("_benchmark_process_environment")
        self._benchmark_process_environment = (
            {str(key): str(value) for key, value in private_process_env.items()}
            if isinstance(private_process_env, Mapping)
            else {}
        )
        env = dict(spec.environment)

        source_home = Path(
            os.environ.get("CODEX_HOME", Path.home() / ".codex")
        ).expanduser()
        source_config = source_home / "config.toml"
        source_runtime_config = ""
        native_skill = str(arm_cfg.get("skill_mode") or "").lower() == "native"
        provider_mode = _codex_provider_mode(agent_cfg)
        self._provider = _codex_provider_metadata(provider_mode, agent_cfg)
        if source_config.is_file():
            source_text = source_config.read_text(encoding="utf-8", errors="replace")
            if provider_mode == CODEX_PROVIDER_EXPERIMENT:
                source_runtime_config = _native_codex_runtime_config(
                    source_text,
                    include_model=False,
                    include_provider=False,
                )
            else:
                source_runtime_config = (
                    _native_codex_runtime_config(source_text)
                    if native_skill or arm_cfg.get("prompt_mode") == "minimal"
                    else _drop_codex_mcp_block(source_text, server_name=spec.name)
                )
        base_config = source_runtime_config
        if provider_mode == CODEX_PROVIDER_EXPERIMENT:
            experiment_config = _experiment_provider_config(self._provider)
            base_config = (
                f"{source_runtime_config}\n\n{experiment_config}"
                if source_runtime_config
                else experiment_config
            )

        lines = [
            "# Generated by SuperNav for this benchmark run.",
            f"[mcp_servers.{_toml_value(spec.name)}]",
            f"command = {_toml_value(spec.command)}",
            f"args = {_toml_value(list(spec.args))}",
            "enabled = true",
            "required = true",
            *(
                [
                    "env_vars = "
                    + _toml_value(sorted(self._benchmark_process_environment))
                ]
                if self._benchmark_process_environment
                else []
            ),
            f"default_tools_approval_mode = {_toml_value(str(agent_cfg.get('mcp_tools_approval', 'approve')))}",
            f"startup_timeout_sec = {_toml_value(int(agent_cfg.get('mcp_startup_timeout_s', 30)))}",
            f"tool_timeout_sec = {_toml_value(int(agent_cfg.get('mcp_tool_timeout_s', 300)))}",
            "",
            f"[mcp_servers.{_toml_value(spec.name)}.env]",
        ]
        for key in sorted(env):
            lines.append(f"{key} = {_toml_value(env[key])}")
        generated_config = "\n".join(lines)
        if arm_cfg.get("prompt_mode") == "minimal":
            # Prevent repository/user instructions and unrelated capabilities from
            # adding a second navigation policy to the controlled task prompt.
            isolation = (
                "[features]\napps = false\nplugins = false\nmulti_agent = false\n"
                "memories = false\nskill_search = false\nview_image = false\n"
                "browser_use = false\ncomputer_use = false\nimage_generation = false\n"
                f"shell_tool = {str(native_skill).lower()}\n"
            )
            # Runtime config can contain provider tables, so global keys must lead.
            base_config = (
                'project_doc_max_bytes = 0\nweb_search = "disabled"\n' + base_config
            )
            generated_config = isolation + "\n" + generated_config
        config_text = (
            f"{base_config}\n\n{generated_config}\n"
            if base_config
            else generated_config + "\n"
        )
        (codex_home / "config.toml").write_text(config_text, encoding="utf-8")
        ensure_backend_subagent(
            backend="codex",
            project_dir=project_dir,
            workspace_root=workspace_root,
            config=config,
            agent_cfg=agent_cfg,
            model_cfg=model_cfg,
        )

        source_auth = source_home / "auth.json"
        target_auth = codex_home / "auth.json"
        if provider_mode == CODEX_PROVIDER_EXPERIMENT:
            if target_auth.is_symlink() or target_auth.is_file():
                target_auth.unlink()
            elif target_auth.exists():
                raise ValueError(
                    f"run-local Codex auth path is not a file: {target_auth}"
                )
        if (
            provider_mode == CODEX_PROVIDER_USER
            and source_auth.is_file()
            and not target_auth.exists()
        ):
            try:
                target_auth.symlink_to(source_auth)
            except OSError:
                shutil.copy2(source_auth, target_auth)

        agent_instructions = default_agent_instructions(config, "codex")
        if arm_cfg.get("prompt_mode") == "minimal":
            agent_instructions = ""
        if str(arm_cfg.get("skill_mode") or "").lower() == "native":
            skill_name = str(arm_cfg.get("skill_name") or "")
            skill_path = codex_home / "skills" / skill_name / "SKILL.md"
            agent_instructions += (
                "Before any scene action, use Codex's native skill workflow: read the "
                f"complete `{skill_path}` file. Read-only file access for this exact skill "
                "bundle is allowed; do not use shell or files for scene perception or action.\n"
            )
            native_skill_path = skill_path
        else:
            native_skill_path = None
        agent_instructions += agent_policy(
            config,
            native_skill_path=native_skill_path,
            arm_cfg=arm_cfg,
        )
        (project_dir / "AGENTS.md").write_text(agent_instructions, encoding="utf-8")
        write_json(
            project_dir / "codex_project.json",
            {
                "codex_home": str(codex_home),
                "config": str(codex_home / "config.toml"),
                "source_config": (
                    str(source_config) if source_config.is_file() else None
                ),
                "source_config_inherited": bool(source_runtime_config),
                "provider": self._provider,
                "auth_linked": target_auth.exists(),
                "workspace_root": str(workspace_root),
                "agent": dict(agent_cfg),
                "model": dict(model_cfg),
            },
        )
        return project_dir

    def validate_environment(self, *, agent_cfg: Mapping[str, Any]) -> None:
        environment = dict(os.environ)
        environment.update(getattr(self, "_benchmark_process_environment", {}))
        _validate_provider_environment(
            getattr(
                self,
                "_provider",
                _codex_provider_metadata(_codex_provider_mode(agent_cfg), agent_cfg),
            ),
            environment,
        )

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
        session_path = run_dir / "codex_session.jsonl"
        stderr_path = run_dir / "stderr.log"
        codex_home = project_dir / ".codex_home"
        command = [str(agent_cfg.get("command") or "codex")]
        approval = agent_cfg.get("approval")
        if approval:
            command.extend(["--ask-for-approval", str(approval)])
        command.extend(
            [
                "exec",
                "--json",
                "-C",
                str(project_dir),
            ]
        )
        selected_model = model_name(model_cfg)
        if selected_model:
            command.extend(["--model", selected_model])
        sandbox = _effective_sandbox(project_dir, agent_cfg)
        if sandbox:
            command.extend(["--sandbox", str(sandbox)])
        if bool(agent_cfg.get("skip_git_repo_check", True)):
            command.append("--skip-git-repo-check")
        extra_args = agent_cfg.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(str(x) for x in extra_args)
        command.extend(
            _provider_cli_overrides(
                getattr(
                    self,
                    "_provider",
                    _codex_provider_metadata(_codex_provider_mode(agent_cfg), agent_cfg),
                )
            )
        )
        command.append(prompt)

        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)
        env.update(getattr(self, "_benchmark_process_environment", {}))
        _validate_provider_environment(
            getattr(
                self,
                "_provider",
                _codex_provider_metadata(_codex_provider_mode(agent_cfg), agent_cfg),
            ),
            env,
        )
        start = time.monotonic()
        with (
            raw_path.open("w", encoding="utf-8") as out,
            stderr_path.open(
                "w",
                encoding="utf-8",
            ) as err,
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
        duration_s = time.monotonic() - start
        latest_session = _latest_session_dump(codex_home)
        if latest_session is not None:
            shutil.copy2(latest_session, session_path)
        return AgentResult(
            proc.returncode,
            duration_s,
            command,
            raw_path,
            stderr_path,
            session_id=_thread_id_from_raw(raw_path),
            export_path=raw_path,
        )

    def parse(self, raw_path: str | Path, out_path: str | Path) -> List[Dict[str, Any]]:
        raw = Path(raw_path)
        session_path = raw.with_name("codex_session.jsonl")
        return canonicalize_codex(
            raw,
            out_path,
            timing_path=session_path if session_path.is_file() else None,
        )


class CodexAgentProfile(CodexAgent):
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
        session_path = run_dir / "codex_session.jsonl"
        stderr_path = run_dir / "stderr.log"
        trace_path = run_dir / "trace"
        codex_home = project_dir / ".codex_home"
        command = [str(agent_cfg.get("command") or "codex")]
        approval = agent_cfg.get("approval")
        if approval:
            command.extend(["--ask-for-approval", str(approval)])
        command.extend(
            [
                "-c features.runtime_metrics=true",
                "exec",
                "--json",
                "-C",
                str(project_dir),
            ]
        )
        selected_model = model_name(model_cfg)
        if selected_model:
            command.extend(["--model", selected_model])
        sandbox = _effective_sandbox(project_dir, agent_cfg)
        if sandbox:
            command.extend(["--sandbox", str(sandbox)])
        if bool(agent_cfg.get("skip_git_repo_check", True)):
            command.append("--skip-git-repo-check")
        extra_args = agent_cfg.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(str(x) for x in extra_args)
        command.extend(
            _provider_cli_overrides(
                getattr(
                    self,
                    "_provider",
                    _codex_provider_metadata(_codex_provider_mode(agent_cfg), agent_cfg),
                )
            )
        )
        command.append(prompt)

        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)
        env["CODEX_ROLLOUT_TRACE_ROOT"] = str(trace_path)
        env["CODEX_ROLLOUT_TRACE_STREAM_EVENTS"] = "1"
        env.update(getattr(self, "_benchmark_process_environment", {}))
        _validate_provider_environment(
            getattr(
                self,
                "_provider",
                _codex_provider_metadata(_codex_provider_mode(agent_cfg), agent_cfg),
            ),
            env,
        )
        start = time.monotonic()
        with (
            raw_path.open("w", encoding="utf-8") as out,
            stderr_path.open(
                "w",
                encoding="utf-8",
            ) as err,
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
        duration_s = time.monotonic() - start
        latest_session = _latest_session_dump(codex_home)
        if latest_session is not None:
            shutil.copy2(latest_session, session_path)
        return AgentResult(
            proc.returncode,
            duration_s,
            command,
            raw_path,
            stderr_path,
            session_id=_thread_id_from_raw(raw_path),
            export_path=raw_path,
        )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


_NATIVE_CODEX_RUNTIME_KEYS = (
    "model_provider",
    "model",
    "review_model",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
    "disable_response_storage",
    "network_access",
)


def _native_codex_runtime_config(
    text: str,
    *,
    include_model: bool = True,
    include_provider: bool = True,
) -> str:
    """Keep selected runtime fields while excluding user extension config."""
    try:
        parsed = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, TypeError):
        return ""

    lines: List[str] = []
    for key in _NATIVE_CODEX_RUNTIME_KEYS:
        if key in {"model", "review_model"} and not include_model:
            continue
        if key == "model_provider" and not include_provider:
            continue
        value = parsed.get(key)
        if _toml_scalar_or_list(value):
            lines.append(f"{key} = {_toml_value(value)}")

    providers = parsed.get("model_providers") if include_provider else None
    if isinstance(providers, Mapping):
        for provider_name in sorted(providers, key=str):
            provider = providers[provider_name]
            if not isinstance(provider, Mapping):
                continue
            if lines:
                lines.append("")
            lines.append(f"[model_providers.{_toml_value(provider_name)}]")
            for key in sorted(provider, key=str):
                value = provider[key]
                if _toml_scalar_or_list(value):
                    lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).rstrip()


def _codex_provider_mode(agent_cfg: Mapping[str, Any]) -> str:
    mode = (
        str(agent_cfg.get("provider_mode") or CODEX_PROVIDER_USER).strip().lower()
    )
    if mode not in {CODEX_PROVIDER_EXPERIMENT, CODEX_PROVIDER_USER}:
        raise ValueError(
            f"Codex provider_mode must be 'experiment' or 'user', got {mode!r}"
        )
    return mode


def _codex_provider_metadata(mode: str, agent_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if mode == CODEX_PROVIDER_EXPERIMENT:
        return _experiment_provider(agent_cfg)
    return {"mode": CODEX_PROVIDER_USER}


def _experiment_provider(agent_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Read explicit experiment routing without a built-in provider endpoint."""
    configured = configured_provider(agent_cfg) or {}
    base_url = os.environ.get("HAB_BENCH_RELAY_BASE_URL", "").strip() or str(
        configured.get("base_url") or ""
    ).strip()
    if not base_url:
        raise ValueError(
            "Codex experiment mode requires agents.codex.provider.base_url "
            "in your local config or HAB_BENCH_RELAY_BASE_URL"
        )
    provider = {
        "mode": CODEX_PROVIDER_EXPERIMENT,
        "id": str(configured.get("id") or "configured"),
        "name": str(configured.get("name") or "Configured Provider"),
        "base_url": base_url,
        "wire_api": str(configured.get("wire_api") or "responses"),
        "env_key": str(configured.get("env_key") or "OPENAI_API_KEY"),
        "requires_openai_auth": False,
    }
    wire_api = os.environ.get("HAB_BENCH_RELAY_WIRE_API", "").strip()
    if wire_api:
        provider["wire_api"] = wire_api
    for env_name, key in (
        ("HAB_BENCH_RELAY_REQUEST_MAX_RETRIES", "request_max_retries"),
        ("HAB_BENCH_RELAY_STREAM_MAX_RETRIES", "stream_max_retries"),
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            provider[key] = int(raw)
    return provider


def _experiment_provider_config(provider: Mapping[str, Any]) -> str:
    lines = [
        f"model_provider = {_toml_value(provider['id'])}",
        "",
        f"[model_providers.{_toml_value(provider['id'])}]",
        f"name = {_toml_value(provider['name'])}",
        f"base_url = {_toml_value(provider['base_url'])}",
        f"wire_api = {_toml_value(provider['wire_api'])}",
        f"env_key = {_toml_value(provider['env_key'])}",
        "requires_openai_auth = false",
    ]
    for key in ("request_max_retries", "stream_max_retries"):
        if key in provider:
            lines.append(f"{key} = {_toml_value(provider[key])}")
    return "\n".join(lines)


def _provider_cli_overrides(provider: Mapping[str, Any]) -> List[str]:
    if provider.get("mode") != CODEX_PROVIDER_EXPERIMENT:
        return []
    provider_id = str(provider["id"])
    optional = "".join(
        f",{key}={_toml_value(provider[key])}"
        for key in ("request_max_retries", "stream_max_retries")
        if key in provider
    )
    provider_value = "".join(
        [
            "{",
            f"name={_toml_value(provider['name'])},",
            f"base_url={_toml_value(provider['base_url'])},",
            f"wire_api={_toml_value(provider['wire_api'])},",
            f"env_key={_toml_value(provider['env_key'])},",
            "requires_openai_auth=false",
            optional,
            "}",
        ]
    )
    return [
        "-c",
        f"model_provider={_toml_value(provider_id)}",
        "-c",
        f"model_providers.{provider_id}={provider_value}",
    ]


def _validate_provider_environment(
    provider: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    if provider.get("mode") != CODEX_PROVIDER_EXPERIMENT:
        return
    env_key = str(provider["env_key"])
    if not str(environment.get(env_key) or "").strip():
        raise ValueError(
            f"Codex bench provider {provider['id']!r} requires non-empty "
            f"environment variable {env_key}; refusing to fall back to user auth"
        )


def _toml_scalar_or_list(value: Any) -> bool:
    if isinstance(value, (str, bool, int, float)):
        return True
    return isinstance(value, list) and all(
        isinstance(item, (str, bool, int, float)) for item in value
    )


def _drop_codex_mcp_block(text: str, server_name: str = "habitat-gs") -> str:
    prefixes = (
        f'[mcp_servers."{server_name}"]',
        f'[mcp_servers."{server_name}".',
        f"[mcp_servers.{server_name}]",
        f"[mcp_servers.{server_name}.",
    )
    kept: List[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = any(stripped.startswith(prefix) for prefix in prefixes)
        if not skipping:
            kept.append(line)
    return "\n".join(kept).rstrip()


def _latest_session_dump(codex_home: Path) -> Path | None:
    sessions = [
        path for path in (codex_home / "sessions").rglob("*.jsonl") if path.is_file()
    ]
    if not sessions:
        return None
    return max(sessions, key=lambda path: path.stat().st_mtime)


def _thread_id_from_raw(raw_path: Path) -> str | None:
    try:
        first_line = raw_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[0]
    except IndexError:
        return None
    try:
        data = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    thread_id = data.get("thread_id")
    return thread_id if isinstance(thread_id, str) and thread_id else None
