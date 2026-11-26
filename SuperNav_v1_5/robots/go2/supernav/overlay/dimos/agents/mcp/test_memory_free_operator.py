"""Memory-free operator preparation and MCP exposure; no robot or model calls."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tomllib

import pytest

from dimos.agents.mcp.external_harness_authority_proxy import (
    DISABLED_MEMORY_TOOL_NAMES,
    ExternalHarnessAuthorityProxy,
    MovementAuthorityEnvelope,
)
from dimos.agents.mcp.external_harness_codex import prepare_codex_harness_run
from dimos.agents.mcp.test_external_harness_authority_proxy import FakeMcpAdapter


@pytest.mark.parametrize("old_venv", [None, "/old/development-venv"])
def test_pixelgoal_preparation_only_installs_navigation_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_venv: str | None
) -> None:
    if old_venv is None:
        monkeypatch.delenv("DIMOS_VENV", raising=False)
    else:
        monkeypatch.setenv("DIMOS_VENV", old_venv)
    run = prepare_codex_harness_run(
        run_root=tmp_path / "run",
        source_codex_home=tmp_path / "empty-home",
        task="Find the exit from current camera observations",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        authority_allowed_movement_tools=("pixelgoal_navigate",),
        copy_auth=False,
    )
    skills = run.workspace / ".agents/skills"
    assert {p.name for p in skills.iterdir()} == {
        "go2-pixelgoal", "global-navigation-pixelgoal"
    }
    assert len(list(skills.rglob("*.md"))) == 5
    assert run.prompt_mode == "default"
    assert run.proxy_command
    assert run.proxy_command[:5] == [
        sys.executable, "-m", "dimos.robot.cli.dimos", "mcp", "authority-proxy"
    ]
    assert "live_environment_preflight" not in run.model_dump()
    assert "--authority-tool" in run.proxy_command
    assert "pixelgoal_navigate" in run.proxy_command
    assert not any("slam" in arg or "memory" in arg or "relocalization" in arg for arg in run.proxy_command)
    assert not any("graph_memory" in key or "slam" in key or "relocalization" in key for key in run.model_dump())
    assert not (run.run_root / "logs/go2-graph-memory.json").exists()
    assert not (run.run_root / "logs/go2-slam-status.json").exists()


def test_removed_prompt_mode_is_rejected_before_creating_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="memory-free"):
        prepare_codex_harness_run(
            run_root=tmp_path / "run",
            task="Find the exit",
            mcp_url="http://127.0.0.1:9990/mcp",
            mode="authority-proxy",
            prompt_mode="long_instruction",
            copy_auth=False,
        )
    assert not (tmp_path / "run").exists()


def test_run_disables_account_apps_and_plugins_without_losing_navigation_skills(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.toml").write_text('''
model = "gpt-6-sol"
preferred_auth_method = "chatgpt"
disable_response_storage = true
[features]
apps = true
plugins = true
remote_plugin = true
[mcp_servers.unrelated]
url = "https://example.invalid/mcp"
''')
    run = prepare_codex_harness_run(
        run_root=tmp_path / "run", source_codex_home=source, task="Approach the chair",
        mcp_url="http://127.0.0.1:9990/mcp", mode="authority-proxy",
        authority_allowed_movement_tools=("pixelgoal_navigate",), copy_auth=False,
    )
    config = tomllib.loads((run.codex_home / "config.toml").read_text())
    assert set(config["mcp_servers"]) == {"dimos"}
    assert not {"preferred_auth_method", "disable_response_storage"} & config.keys()
    for feature in ("apps", "plugins", "remote_plugin", "hooks", "memories"):
        assert config["features"][feature] is False
        for command in (run.command, run.mcp_smoke_command):
            assert any(command[i:i+2] == ["--disable", feature] for i in range(len(command)-1))
    assert (run.workspace / ".agents/skills/go2-pixelgoal/SKILL.md").exists()
    assert (run.workspace / ".agents/skills/global-navigation-pixelgoal/SKILL.md").exists()

    # Native CLI config inspection is offline: no model call or robot access.
    if shutil.which("codex"):
        env = {**os.environ, "CODEX_HOME": str(run.codex_home)}
        features = subprocess.run(["codex", "features", "list"], cwd=run.workspace,
            env=env, capture_output=True, text=True, check=True, timeout=20)
        states = {row.split()[0]: row.split()[-1] for row in features.stdout.splitlines()}
        # Codex 0.160 reports hooks=true even when explicitly disabled; the
        # config and launch flags above still verify our hooks setting.
        for feature in ("apps", "plugins", "remote_plugin", "memories"):
            assert states[feature] == "false"
        servers = subprocess.run(["codex", "mcp", "list", "--json"], cwd=run.workspace,
            env=env, capture_output=True, text=True, check=True, timeout=20)
        assert [server["name"] for server in json.loads(servers.stdout)] == ["dimos"]


def test_memory_tools_hidden_and_refused_even_if_upstream_advertises_them() -> None:
    assert {"go2_apply_graph_mutations", "go2_query_route_context", "go2_known_slam_map",
            "go2_save_relocalization_bundle"} <= DISABLED_MEMORY_TOOL_NAMES
    upstream = FakeMcpAdapter()
    upstream.tools.extend(
        {"name": name, "inputSchema": {"type": "object"}}
        for name in DISABLED_MEMORY_TOOL_NAMES
    )
    proxy = ExternalHarnessAuthorityProxy(
        upstream,
        envelope=MovementAuthorityEnvelope.create(
            task_goal="Find the exit", allowed_movement_tools=("pixelgoal_navigate",)
        ),
    )
    visible = {tool["name"] for tool in proxy.list_tools()}
    assert not visible & DISABLED_MEMORY_TOOL_NAMES
    assert "observe" in visible
    assert "stop_movement" in visible
    for name in DISABLED_MEMORY_TOOL_NAMES:
        result = proxy.call_tool(name, {})
        assert "disabled" in result["content"][0]["text"]
    assert upstream.calls == []
