"""Verify the simulator boundary with real agent preparation and subprocess output."""

from __future__ import annotations

import importlib.abc
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from supernav.runtime.agents import get_agent_backend, save_command
from supernav.runtime.mcp import resolve_mcp_spec
from supernav.paths import asset_root, python_paths


class _NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "habitat_sim" or fullname.startswith("habitat_sim."):
            raise AssertionError(f"Simulator imported by the agent runtime: {fullname}")
        if fullname.startswith("habitat_agent.evolution"):
            raise AssertionError(f"Forbidden import: {fullname}")
        return None


@pytest.mark.parametrize("name", ["codex", "codex_profile", "kimi", "opencode"])
def test_agent_clients_prepare_generic_mcp_without_simulator(name, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "meta_path", [_NoSimulator(), *sys.meta_path])
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty_codex_home"))
    config = {
        "mcp": {
            "name": "test-world",
            "command": sys.executable,
            "args": ["mock_world.py", "--transport", "stdio"],
            "http_args": ["mock_world.py", "--transport", "streamable-http"],
            "environment": {"WORLD_SETTING": "base"},
        },
        "agent_instructions": "Complete the supplied task using test-world tools.\n",
        "agent_policy": "",
    }
    arm = {"environment": {"WORLD_SETTING": "arm"}}
    spec = resolve_mcp_spec(workspace_root=tmp_path, config=config, arm_cfg=arm)
    assert dict(spec.environment) == {"WORLD_SETTING": "arm"}
    backend = get_agent_backend(name)
    project = backend.prepare_project(
        run_dir=tmp_path / "run", workspace_root=tmp_path, config=config,
        arm_cfg=arm, agent_cfg={"source_home": str(tmp_path / "empty_kimi_home")},
        model_cfg={},
    )
    assert (project / "AGENTS.md").read_text() == config["agent_instructions"]
    if name.startswith("codex"):
        generated = tomllib.loads((project / ".codex_home/config.toml").read_text())
        server = generated["mcp_servers"]["test-world"]
        assert server["command"] == spec.command
        assert server["args"] == list(spec.args)
        assert server["env"] == dict(spec.environment)
    elif name == "opencode":
        generated = json.loads((project / "opencode.json").read_text())
        assert generated["mcp"]["test-world"]["command"] == [spec.command, *spec.args]
    else:
        generated = json.loads((project / "kimi_home/mcp.json").read_text())
        assert set(generated["mcpServers"]) == {"test-world"}


def test_cli_result_and_original_error_survive_event_conversion(tmp_path):
    fake = tmp_path / "fake-opencode"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print(json.dumps({'type':'tool_use','part':{'type':'tool','tool':'world_action','state':{'input':{},'status':'error','output':'world refused action'}}}))\n"
        "sys.exit(7)\n"
    )
    fake.chmod(0o755)
    run_dir = tmp_path / "run"
    backend = get_agent_backend("opencode")
    project = backend.prepare_project(
        run_dir=run_dir, workspace_root=tmp_path,
        config={"mcp": {"name": "test-world", "command": "unused-mcp"}, "agent_instructions": "test"},
        arm_cfg={}, agent_cfg={}, model_cfg={},
    )
    result = backend.run(
        prompt="Attempt the task", run_dir=run_dir, project_dir=project,
        agent_cfg={"command": str(fake)}, model_cfg={}, timeout_s=10,
    )
    save_command(run_dir, result)
    events = backend.parse(result.raw_path, run_dir / "canonical.jsonl")
    assert result.returncode == 7
    assert json.loads((run_dir / "command.json").read_text())["returncode"] == 7
    assert "world refused action" in result.raw_path.read_text()
    assert any("world refused action" in json.dumps(event) for event in events)


@pytest.mark.parametrize("package", ["supernav", "habitat_contract"])
def test_canonical_package_dependency_boundaries(package):
    root = Path(__file__).resolve().parents[1] / "src" / package
    forbidden = {"harness", "habitat_agent", "pluggable_harness", "analytics"}
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in forbidden:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    assert not violations, "Forbidden imports in canonical code:\n" + "\n".join(violations)


def test_runtime_has_no_simulator_sdk_imports_even_inside_lazy_functions():
    root = Path(__file__).resolve().parents[1] / "src" / "supernav" / "runtime"
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in {"habitat_sim", "ai2thor"}:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    assert not violations, "Simulator SDK imports belong in backends:\n" + "\n".join(violations)
