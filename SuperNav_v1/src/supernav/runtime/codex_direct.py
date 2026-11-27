from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping


def direct_mcp_overrides(
    command: str,
    project_dir: Path,
    environment: Mapping[str, str],
    model: str | None,
    provider_overrides: list[str],
) -> list[str]:
    if not model:
        raise ValueError("Direct MCP experiments require an explicit model")
    result = subprocess.run(
        [command, "debug", "models", *provider_overrides],
        cwd=project_dir,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    catalog = json.loads(result.stdout)
    models = catalog["models"]
    if not any(entry["slug"] == model for entry in models):
        raise ValueError(f"Requested model is absent from Codex catalog: {model}")
    # 使用真实模型目录，将 MCP 工具直接暴露给模型。
    for entry in models:
        entry["tool_mode"] = "direct"
        entry["supports_search_tool"] = False
    catalog_path = project_dir.resolve() / "direct_mcp_models.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    return [
        "-c", f"model_catalog_json={json.dumps(str(catalog_path))}",
        "-c", "features.code_mode=false",
        "-c", "features.code_mode_only=false",
    ]


def validate_direct_mcp_session(session_path: Path) -> None:
    with session_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            event = json.loads(line)
            if event.get("type") != "response_item":
                continue
            payload = event["payload"]
            if payload.get("type") in ("custom_tool_call", "function_call"):
                if payload.get("name") in ("exec", "js", "js_repl"):
                    raise RuntimeError(
                        f"Code Mode call violates direct MCP policy: {session_path}:{line_number}"
                    )
