"""Public command discovery, argument forwarding, and canonical experiments."""

import json
import os
import subprocess
import sys

import pytest

from supernav import cli
from supernav.paths import asset_root


def invoke(*args, cwd, env=None):
    return subprocess.run(
        [sys.executable, "-m", "supernav", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("args", [[], ["--help"]])
def test_root_help_lists_primary_workflows(args, tmp_path):
    result = invoke(*args, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "supernav config list" in result.stdout
    for name in ("run", "run-one", "config", "web", "mcp", "score-objectnav"):
        assert name in result.stdout


@pytest.mark.parametrize(
    "command,option",
    [
        ("run", "--experiment"),
        ("run-one", "--instruction"),
        ("config", "show"),
        ("web", "--runs-root"),
        ("mcp", "--transport"),
        ("habitat-bridge", "--port"),
        ("prompt", "--arm"),
        ("score-objectnav", "--criterion"),
        ("score-multi-objectnav", "--runs-root"),
    ],
)
def test_subcommand_help_reaches_its_parser(command, option, tmp_path):
    result = invoke(command, "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert option in result.stdout


@pytest.mark.parametrize("failure", [None, SystemExit(7), RuntimeError("command failed")])
def test_dispatch_preserves_child_arguments_and_restores_argv(failure, monkeypatch):
    original = ["caller", "original-argument"]
    monkeypatch.setattr(sys, "argv", original)
    received = []

    def handler():
        received.extend(sys.argv)
        sys.argv = ["child-replaced-argv"]
        if failure is not None:
            raise failure

    monkeypatch.setattr(cli, "run", handler)
    args = ["run", "--experiment", "habitat-geo-based-executor", "--task-ids", "one,two", "--dry-run"]
    if failure is None:
        assert cli.main(args) == 0
    else:
        with pytest.raises(type(failure)) as exc:
            cli.main(args)
        assert exc.value is failure
    assert received == ["supernav run", *args[1:]]
    assert sys.argv is original


def test_config_list_and_show_use_catalog(tmp_path):
    result = invoke("config", "list", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    names = {line.split("\t")[0] for line in result.stdout.splitlines()}
    assert names == {"ai2thor-primitive", "habitat-geo-based-executor", "habitat-learned-executor"}
    result = invoke("config", "show", "--experiment", "ai2thor-primitive", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["environment"]["backend"] == "ai2thor"
    assert config["agent"] == "codex"
    assert config["arms"]["default"]["skill_mode"] == "none"
    assert "extends" not in config


@pytest.mark.parametrize("name", ["missing-example", "unsupported-example"])
def test_unknown_experiment_names_are_rejected(name, tmp_path):
    result = invoke("config", "show", "--experiment", name, cwd=tmp_path)
    assert result.returncode == 2
    assert "Unknown experiment" in result.stderr


@pytest.mark.parametrize("path", ["bench/config/ai2thor/config.json", "config/context_budget.yaml"])
def test_unavailable_asset_paths_are_not_resolved_by_cli(path, tmp_path):
    result = invoke("config", "show", "--config", path, cwd=tmp_path)
    assert result.returncode == 2
    assert "Cannot load experiment config" in result.stderr


def test_bridge_startup_failure_reaches_the_cli_exit_code(monkeypatch):
    from supernav.backends.habitat import http_server

    monkeypatch.setattr(http_server, "main", lambda: 1)
    with pytest.raises(SystemExit) as exc:
        cli.main(["habitat-bridge"])
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "args",
    [
        ["missing-command"],
        ["tui"],
        ["habitat-launch"],
        ["run", "--experiment", "missing-experiment", "--dry-run"],
        ["run", "--experiment", "ai2thor-primitive", "--config", "other.json", "--dry-run"],
        ["config", "show", "--experiment", "missing-experiment"],
        ["config", "show", "--experiment", "ai2thor-primitive", "--config", "other.json"],
    ],
)
def test_invalid_selection_is_a_cli_usage_error(args, tmp_path):
    result = invoke(*args, cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("command", ["run", "run-one", "prompt"])
def test_experiment_commands_require_explicit_recipe_selection(command, tmp_path):
    result = invoke(command, cwd=tmp_path)
    assert result.returncode == 2
    assert "--experiment" in result.stderr and "--config" in result.stderr
    assert not (tmp_path / "bench").exists()


@pytest.mark.parametrize("command", ["habitat-bridge"])
def test_process_help_needs_no_agent_or_simulator_dependencies(command, tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('habitat_sim', 'ai2thor', 'numpy', 'openai', 'mcp'):
            raise AssertionError('Unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from supernav.cli import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, command, "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_named_geometry_recipe_dry_run_uses_canonical_runtime(tmp_path):
    import tomllib
    root = asset_root()
    output = tmp_path / "runs"
    instructions = tmp_path / "synthetic-tasks.json"
    instructions.write_text(json.dumps({"instructions": [{
        "task_id": "synthetic-red-cube", "slug": "synthetic-red-cube",
        "text": "Find the red test cube.", "scene": "synthetic_scene",
        "spawn": {"start_position": [0.0, 0.0, 0.0]},
    }]}))
    env = dict(os.environ, CODEX_HOME=str(tmp_path / "empty-codex-home"))
    result = invoke(
        "run", "--experiment", "habitat-geo-based-executor",
        "--instructions", str(instructions), "--task-ids", "synthetic-red-cube", "--arms", "default",
        "--output-dir", str(output), "--dry-run", cwd=root, env=env,
    )
    assert result.returncode == 0, result.stderr
    prompts = list(output.rglob("prompt.txt"))
    assert len(prompts) == 1
    run = prompts[0].parent
    recorded = json.loads((run / "run.json").read_text())
    assert recorded["backend"] == "habitat"
    assert recorded["task_id"] == "synthetic-red-cube"
    assert recorded["arm"] == "default"
    assert recorded["skill"]["snapshot_sha256"]
    project = tomllib.loads((run / "codex_project/.codex_home/config.toml").read_text())
    server = project["mcp_servers"]["habitat-gs"]
    assert server["args"][:3] == ["-m", "supernav", "mcp"]
    assert server["env"]["HAB_MCP_GLOBAL_TASK"] == "1"
