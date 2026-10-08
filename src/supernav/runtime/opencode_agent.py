from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from supernav.runtime.agents import AgentResult, model_name, session_id_from_raw
from supernav.runtime.mcp import resolve_mcp_spec
from supernav.runtime.instructions import agent_instructions as default_agent_instructions, agent_policy
from supernav.runtime.config import write_json
from supernav.runtime.streams import _clean_tool_name, _text_from_content, _time_fields, iter_json_lines
from supernav.runtime.subagents import ensure_backend_subagent
from supernav.runtime.skill_runtime import copy_snapshot_skills
from supernav.runtime.providers import configured_provider, client_model, validate_provider_environment


def canonicalize_opencode(
    raw_path: str | Path, out_path: str | Path
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx, data in enumerate(iter_json_lines(raw_path)):
        typ = str(data.get("type") or data.get("event") or data.get("kind") or "")
        timestamp = data.get("timestamp") or data.get("time") or data.get("created")
        part = data.get("part") if isinstance(data.get("part"), dict) else {}
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        if typ == "tool_use" or part.get("type") == "tool":
            name = (
                part.get("tool")
                or state.get("tool")
                or data.get("name")
                or data.get("tool")
            )
            clean_name = _clean_tool_name(str(name or "unknown"))
            timing = _time_fields(state.get("time"))
            events.append(
                {
                    "type": "tool_call",
                    "name": clean_name,
                    "input": state.get("input") or data.get("input") or {},
                    "timestamp": timestamp,
                    "raw_index": idx,
                    **timing,
                }
            )
            output = state.get("output")
            attachments = state.get("attachments")
            if output is not None or attachments:
                events.append(
                    {
                        "type": "tool_result",
                        "name": clean_name,
                        "content": _text_from_content(output),
                        "attachments": attachments or [],
                        "timestamp": timestamp,
                        "raw_index": idx,
                        **timing,
                    }
                )
            continue
        part_type = str(part.get("type") or "")
        if typ in ("text", "thinking", "reasoning") or part_type in (
            "text",
            "thinking",
            "reasoning",
        ):
            text = (
                part.get("text")
                or part.get("thinking")
                or part.get("reasoning")
                or data.get("text")
                or data.get("thinking")
                or data.get("reasoning")
                or ""
            )
            if isinstance(text, str) and text.strip():
                is_reasoning = typ in ("thinking", "reasoning") or part_type in (
                    "thinking",
                    "reasoning",
                )
                timing = _time_fields(part.get("time"))
                events.append(
                    {
                        "type": (
                            "assistant_reasoning" if is_reasoning else "assistant_text"
                        ),
                        "text": text,
                        "timestamp": timestamp,
                        "raw_index": idx,
                        **timing,
                    }
                )
            continue
        if typ == "step_finish" or part_type == "step-finish":
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            reasoning_tokens = tokens.get("reasoning")
            if isinstance(reasoning_tokens, int) and reasoning_tokens > 0:
                events.append(
                    {
                        "type": "assistant_reasoning",
                        "text": f"reasoning tokens: {reasoning_tokens}",
                        "timestamp": timestamp,
                        "raw_index": idx,
                    }
                )
            continue
        name = data.get("name") or data.get("tool") or data.get("toolName")
        if "tool" in typ and (
            "call" in typ or "use" in typ or data.get("input") is not None
        ):
            events.append(
                {
                    "type": "tool_call",
                    "name": _clean_tool_name(str(name or "unknown")),
                    "input": data.get("input") or data.get("arguments") or {},
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue
        if "tool" in typ and ("result" in typ or "output" in typ):
            events.append(
                {
                    "type": "tool_result",
                    "name": _clean_tool_name(str(name or "unknown")),
                    "content": _text_from_content(
                        data.get("content") or data.get("output") or data.get("result")
                    ),
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
            continue
        message = data.get("message") if isinstance(data.get("message"), dict) else data
        role = message.get("role")
        content = message.get("content")
        text = _text_from_content(content)
        if text:
            events.append(
                {
                    "type": (
                        "assistant_text"
                        if role == "assistant" or typ == "assistant"
                        else "text"
                    ),
                    "text": text,
                    "timestamp": timestamp,
                    "raw_index": idx,
                }
            )
        if typ in ("result", "done", "complete", "completed"):
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


class OpenCodeAgent:
    name = "opencode"

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
        project_dir = run_dir / "opencode_project"
        project_dir.mkdir(parents=True, exist_ok=True)
        skill_name = str(arm_cfg.get("skill_name") or "").strip()
        snapshot_root = arm_cfg.get("_skill_snapshot_root")
        native_skill = str(arm_cfg.get("skill_mode") or "").lower() == "native"
        if native_skill:
            if not snapshot_root or not skill_name:
                raise ValueError(
                    "native skill arm requires skill_name and verified snapshot"
                )
            copy_snapshot_skills(
                Path(str(snapshot_root)), project_dir / ".opencode" / "skills"
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
        opencode_config = {
            "$schema": "https://opencode.ai/config.json",
            "instructions": ["AGENTS.md"],
            "mcp": {
                spec.name: {
                    "type": "local",
                    "enabled": True,
                    "command": [spec.command, *spec.args],
                    "environment": dict(spec.environment),
                }
            },
        }
        if native_skill:
            opencode_config["permission"] = {
                "skill": {"*": "deny", skill_name: "allow"}
            }
        provider = configured_provider(agent_cfg)
        if provider:
            selected = model_name(model_cfg)
            if not selected:
                raise ValueError("explicit OpenCode API provider requires a model")
            opencode_config["provider"] = {
                provider["id"]: {
                    "npm": provider.get("npm", "@ai-sdk/openai"),
                    "options": {"baseURL": provider["base_url"], "apiKey": "{env:" + provider["env_key"] + "}"},
                    "models": {selected: {"name": selected, "attachment": True, "tool_call": True}},
                }
            }
            opencode_config["enabled_providers"] = [provider["id"]]
            opencode_config["model"] = client_model(agent_cfg, selected)
        write_json(project_dir / "opencode.json", opencode_config)
        ensure_backend_subagent(
            backend="opencode",
            project_dir=project_dir,
            workspace_root=workspace_root,
            config=config,
            agent_cfg=agent_cfg,
            model_cfg=model_cfg,
        )
        agent_instructions = default_agent_instructions(config, "opencode")
        if native_skill:
            agent_instructions += (
                f'Before any scene action, call the native skill tool with name "{skill_name}" '
                "and follow the returned instructions.\n"
            )
        (project_dir / "AGENTS.md").write_text(agent_instructions, encoding="utf-8")
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
        export_path = run_dir / "opencode_session.json"
        command = [
            str(agent_cfg.get("command") or "opencode"),
            "run",
            "--format",
            "json",
            "--thinking",
            "--dir",
            str(project_dir),
        ]
        command.extend(["--title", f"bench-temp-{run_dir.name}"])
        selected_model = client_model(agent_cfg, model_name(model_cfg))
        if selected_model:
            command.extend(["--model", selected_model])
        agent = agent_cfg.get("agent")
        if agent:
            command.extend(["--agent", str(agent)])
        variant = agent_cfg.get("variant")
        if variant:
            command.extend(["--variant", str(variant)])
        extra_args = agent_cfg.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(str(x) for x in extra_args)
        command.append(prompt)

        start = time.monotonic()
        isolated_home = project_dir / "opencode_home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        # Private episode initialization belongs only in the child environment,
        # never in the agent-readable opencode.json configuration.
        env.update(getattr(self, "_benchmark_process_environment", {}))
        env.update(
            {
                "HOME": str(isolated_home),
                "XDG_CONFIG_HOME": str(isolated_home / ".config"),
                "XDG_DATA_HOME": str(isolated_home / ".local" / "share"),
                "XDG_CACHE_HOME": str(isolated_home / ".cache"),
            }
        )
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
        session_id = session_id_from_raw(raw_path)
        delete_returncode = None
        if session_id:
            opencode_bin = str(agent_cfg.get("command") or "opencode")
            with (
                export_path.open("w", encoding="utf-8") as export_out,
                stderr_path.open(
                    "a",
                    encoding="utf-8",
                ) as err,
            ):
                subprocess.run(
                    [opencode_bin, "export", session_id],
                    stdout=export_out,
                    stderr=err,
                    text=True,
                    timeout=60,
                    cwd=str(project_dir),
                    env=env,
                    check=False,
                )
                delete_proc = subprocess.run(
                    [opencode_bin, "session", "delete", session_id],
                    stdout=err,
                    stderr=err,
                    text=True,
                    timeout=60,
                    cwd=str(project_dir),
                    env=env,
                    check=False,
                )
                delete_returncode = delete_proc.returncode
        return AgentResult(
            proc.returncode,
            duration_s,
            command,
            raw_path,
            stderr_path,
            session_id,
            export_path if session_id else None,
            delete_returncode,
        )

    def parse(self, raw_path: str | Path, out_path: str | Path) -> List[Dict[str, Any]]:
        return canonicalize_opencode(raw_path, out_path)
