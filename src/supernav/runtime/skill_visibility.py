"""Backend-native visibility checks for benchmark skill snapshots."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping
from supernav.runtime.providers import configured_provider, kimi_provider_environment


def _run(
    command: list[str], *, cwd: Path, env: Mapping[str, str], timeout_s: int
) -> tuple[str, int | None, str, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return "missing_cli", None, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return "timeout", None, str(exc.stdout or ""), str(exc.stderr or "")
    return (
        ("completed" if proc.returncode == 0 else "failed"),
        proc.returncode,
        proc.stdout,
        proc.stderr,
    )


def probe_skill_visibility(
    *,
    backend: str,
    project_dir: Path,
    skill_name: str,
    description: str,
    agent_cfg: Mapping[str, Any],
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Use the client's native discovery surface and require the run-local path."""
    backend = backend.lower()
    project_dir = Path(project_dir).resolve()
    env = dict(os.environ)
    if backend in {"codex", "codex_profile"}:
        home = project_dir / ".codex_home"
        expected = home / "skills" / skill_name / "SKILL.md"
        env["CODEX_HOME"] = str(home)
        command = [
            str(agent_cfg.get("command") or "codex"),
            "debug",
            "prompt-input",
            f"Use /{skill_name}.",
        ]
    elif backend == "kimi":
        skills = project_dir / ".kimi-code" / "skills"
        expected = skills / skill_name / "SKILL.md"
        home = project_dir / "kimi_home"
        env.update(
            {
                "HOME": str(home),
                "KIMI_HOME": str(home / ".kimi-code"),
                "KIMI_CODE_HOME": str(home / ".kimi-code"),
            }
        )
        if configured_provider(agent_cfg):
            meta = json.loads((project_dir / "kimi_project.json").read_text())
            env.update(kimi_provider_environment(agent_cfg, str(meta["model"]["name"])))
        command = [
            str(agent_cfg.get("command") or "kimi"),
            "--skills-dir",
            str(skills),
            "-p",
            (
                f"Load /skill:{skill_name} and print its name, description, and source path. "
                "Copy the description exactly as shown in the loaded skill, character for "
                "character; do not paraphrase or translate."
            ),
            "--output-format",
            "text",
        ]
    elif backend == "opencode":
        expected = project_dir / ".opencode" / "skills" / skill_name / "SKILL.md"
        home = project_dir / "opencode_home"
        env.update(
            {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_CACHE_HOME": str(home / ".cache"),
            }
        )
        command = [
            str(agent_cfg.get("command") or "opencode"),
            "run",
            "--format",
            "json",
            "--dir",
            str(project_dir),
            (
                f'Call the native skill tool with name "{skill_name}". '
                "Print its name, description, and source path; do not call MCP tools."
            ),
        ]
    else:
        raise ValueError(f"unsupported skill visibility backend: {backend}")

    phase, returncode, stdout, stderr = _run(
        command, cwd=project_dir, env=env, timeout_s=timeout_s
    )
    expected_text = str(expected.resolve())
    combined = stdout + "\n" + stderr
    matched_name = skill_name in combined
    # Clients may add inline code markup to an otherwise verbatim description.
    # Do not accept paraphrases or a name-only match.
    matched_description = description.replace("`", "") in combined.replace("`", "")
    matched_path = expected_text in combined
    if not matched_path and backend in {"codex", "codex_profile"}:
        # Codex can render skill paths relative to its explicit roots table.
        # Resolve those aliases before comparing with the run-local snapshot.
        try:
            messages = json.loads(stdout)
            rendered = "\n".join(
                block.get("text", "")
                for message in messages
                for block in message.get("content", [])
                if isinstance(block, dict)
            )
        except (ValueError, TypeError, AttributeError):
            rendered = stdout
        roots = dict(re.findall(r"- `(r\d+)` = `([^`]+)`", rendered))
        for alias, relative in re.findall(r"\(file: (r\d+)/([^\n)]+)\)", rendered):
            if (
                alias in roots
                and str((Path(roots[alias]) / relative).resolve()) == expected_text
            ):
                matched_path = True
                break
    visible = (
        phase == "completed" and matched_name and matched_description and matched_path
    )
    return {
        "status": (
            "visible" if visible else phase if phase != "completed" else "not_visible"
        ),
        "backend": backend,
        "skill_name": skill_name,
        "expected_path": expected_text,
        "command": command,
        "returncode": returncode,
        "matched_name": matched_name,
        "matched_description": matched_description,
        "matched_path": matched_path,
        "stdout": stdout,
        "stderr": stderr,
    }


def write_visibility_report(path: Path, result: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
