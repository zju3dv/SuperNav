"""OpenCode receives private Habitat task bindings without persisting them."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from supernav.backends.habitat.experiment import HabitatBackend
from supernav.runtime.opencode_agent import OpenCodeAgent


PRIVATE_KEYS = ("HAB_MCP_INIT_DEFAULTS_JSON", "HAB_MCP_NAV_GOALS_JSON")


def _prepare(agent, root, *, profile="global_task", scene="private-scene", multi_goal=True):
    config = {
        "environment": {"backend": "habitat"},
        "benchmark_profile": profile,
        "scene": scene,
        "scene_dataset_config_file": str(root / "private-dataset.json"),
        "spawn": {"start_position": [13.5, 0.25, -27.0], "sensor_height": 1.2},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
    }
    arm = {"movement": "visual_point", "skill_mode": "none"}
    metadata = {
        "ordered": True,
        "targets": [{"index": 1, "description": "chair"}, {"index": 2, "description": "table"}],
    } if multi_goal else {}
    HabitatBackend().prepare_task(
        config, SimpleNamespace(instruction="Find the furniture.", metadata=metadata), arm, root,
    )
    project = agent.prepare_project(
        run_dir=root, workspace_root=root, config=config, arm_cfg=arm, agent_cfg={}, model_cfg={},
    )
    return project, arm.get("_benchmark_process_environment", {})


def _run(agent, root, project):
    return agent.run(
        prompt="Find the furniture.", run_dir=root, project_dir=project,
        agent_cfg={}, model_cfg={}, timeout_s=10,
    )


@pytest.fixture
def captured_children(monkeypatch):
    for key in PRIVATE_KEYS:
        monkeypatch.delenv(key, raising=False)
    launches = []

    def capture(command, **kwargs):
        launches.append({"command": command, "environment": dict(kwargs["env"])})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", capture)
    return launches


def test_global_task_private_bindings_reach_opencode_child_only(tmp_path, monkeypatch, captured_children):
    # An explicit episode must win over stale values in the parent shell.
    for key in PRIVATE_KEYS:
        monkeypatch.setenv(key, "stale-shell-value")
    agent = OpenCodeAgent()
    project, private = _prepare(agent, tmp_path / "global")
    expected = dict(private)
    defaults = json.loads(expected[PRIVATE_KEYS[0]])
    assert defaults["scene"] == "private-scene"
    assert defaults["start_position"] == [13.5, 0.25, -27.0]
    assert json.loads(expected[PRIVATE_KEYS[1]])["targets"][1]["description"] == "table"

    # Prepared state is a snapshot, independent of later caller mutation.
    private[PRIVATE_KEYS[0]] = "changed-after-preparation"
    result = _run(agent, tmp_path / "global", project)
    assert result.returncode == 0
    assert len(captured_children) == 1
    child = captured_children[0]
    assert {key: child["environment"][key] for key in PRIVATE_KEYS} == expected
    assert "private-scene" not in " ".join(child["command"])
    assert all(os.environ[key] == "stale-shell-value" for key in PRIVATE_KEYS)

    config = json.loads((project / "opencode.json").read_text())
    assert config["mcp"]["habitat-gs"]["environment"]["HAB_MCP_GLOBAL_TASK"] == "1"
    for path in project.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert all(key not in text for key in PRIVATE_KEYS)
            assert "private-scene" not in text
            assert "start_position" not in text


@pytest.mark.parametrize("next_profile", ["standard", "global_task"])
def test_reusing_opencode_agent_replaces_private_episode_bindings(tmp_path, captured_children, next_profile):
    agent = OpenCodeAgent()
    first, _ = _prepare(agent, tmp_path / "first")
    _run(agent, tmp_path / "first", first)
    assert all(key in captured_children[-1]["environment"] for key in PRIVATE_KEYS)

    second, private = _prepare(
        agent, tmp_path / "second", profile=next_profile, scene="next-private-scene", multi_goal=False,
    )
    _run(agent, tmp_path / "second", second)
    child_env = captured_children[-1]["environment"]
    assert PRIVATE_KEYS[1] not in child_env
    if next_profile == "standard":
        assert PRIVATE_KEYS[0] not in child_env
    else:
        assert child_env[PRIVATE_KEYS[0]] == private[PRIVATE_KEYS[0]]
        assert json.loads(child_env[PRIVATE_KEYS[0]])["scene"] == "next-private-scene"
