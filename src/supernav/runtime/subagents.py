from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from supernav.runtime.agents import model_name
from supernav.paths import resolve_asset_path


_IDENT_RE = re.compile(r"[^a-z0-9_-]+")


def ensure_backend_subagent(
    *,
    backend: str,
    project_dir: Path,
    workspace_root: Path,
    config: Mapping[str, Any],
    agent_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> Path | None:
    selected = str(agent_cfg.get("subagent") or "").strip()
    if not selected:
        return None
    subagents = config.get("subagents") if isinstance(config.get("subagents"), dict) else {}
    spec = subagents.get(selected)
    if not isinstance(spec, dict):
        raise ValueError(
            f"agents.{backend}.subagent={selected!r} missing from config.subagents"
        )
    source_path = resolve_asset_path(spec.get("source_file"), base=workspace_root)
    if source_path is None or not source_path.is_file():
        raise ValueError(
            f"config.subagents.{selected}.source_file must resolve to an existing file"
        )

    description, body = _read_subagent_source(source_path, description=spec.get("description"))
    backend_cfg = spec.get(backend) if isinstance(spec.get(backend), dict) else {}
    agent_name = _slugify(
        str(backend_cfg.get("name") or spec.get("name") or selected)
    )
    if not agent_name:
        raise ValueError(f"config.subagents.{selected} must resolve to a non-empty agent name")

    if backend == "opencode":
        target = project_dir / ".opencode" / "agents" / f"{agent_name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        temperature = backend_cfg.get("temperature", 0.1)
        target.write_text(
            _render_opencode_subagent(
                description=description,
                body=body,
                temperature=temperature,
                model=str(backend_cfg.get("model") or model_name(model_cfg) or ""),
                reasoning_effort=backend_cfg.get("reasoningEffort"),
            ),
            encoding="utf-8",
        )
        return target

    if backend == "codex":
        target = project_dir / ".codex" / "agents" / f"{agent_name}.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _render_codex_subagent(
                name=agent_name,
                description=description,
                model=str(
                    backend_cfg.get("model")
                    or model_name(model_cfg)
                    or "gpt-5.3-codex-spark"
                ),
                reasoning_effort=str(
                    backend_cfg.get("model_reasoning_effort") or "medium"
                ),
                developer_instructions=str(
                    backend_cfg.get("developer_instructions") or _default_codex_developer_instructions(body)
                ),
            ),
            encoding="utf-8",
        )
        return target

    raise ValueError(f"unsupported backend for subagent generation: {backend!r}")


def _read_subagent_source(source_path: Path, *, description: Any) -> tuple[str, str]:
    raw = source_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"subagent source is empty: {source_path}")
    lines = raw.splitlines()
    inline_description = ""
    if lines and lines[0].lower().startswith("description:"):
        inline_description = lines[0].split(":", 1)[1].strip()
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    final_description = str(description or inline_description or "Navigation helper subagent.").strip()
    body = "\n".join(lines).strip()
    if not body:
        raise ValueError(f"subagent body is empty after parsing: {source_path}")
    return final_description, body + "\n"


def _render_opencode_subagent(
    *,
    description: str,
    body: str,
    temperature: Any,
    model: str | None,
    reasoning_effort: Any,
) -> str:
    frontmatter = [
        "---",
        f"description: {description}",
        "mode: subagent",
    ]
    if model:
        frontmatter.append(f"model: {model}")
    if reasoning_effort:
        frontmatter.append(f"reasoningEffort: {reasoning_effort}")
    frontmatter.extend(
        [
            "tools:",
            "  write: true",
            "  edit: true",
            "  bash: true",
            f"temperature: {temperature}",
            "---",
            "",
        ]
    )
    return "\n".join(frontmatter) + body


def _render_codex_subagent(
    *,
    name: str,
    description: str,
    model: str,
    reasoning_effort: str,
    developer_instructions: str,
) -> str:
    header = [
        f'name = "{_escape_toml(name)}"',
        f'description = "{_escape_toml(description)}"',
        f'model = "{_escape_toml(model)}"',
        f'model_reasoning_effort = "{_escape_toml(reasoning_effort)}"',
        'sandbox_mode = "workspace-write"',
        'developer_instructions = """',
        developer_instructions.rstrip(),
        '"""',
        "",
    ]
    return "\n".join(header)


def _default_codex_developer_instructions(body: str) -> str:
    return (
        # "Stay in exploration mode.\n"
        "Use the subagent workflow to inspect surroundings, turn toward the chosen target, and verify the new view.\n"
        # "Write any small helper files or notes you need inside the workspace.\n\n"
        + body.rstrip()
    )


def _slugify(value: str) -> str:
    normalized = _IDENT_RE.sub("-", value.strip().lower())
    return normalized.strip("-")


def _escape_toml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = ["ensure_backend_subagent"]
