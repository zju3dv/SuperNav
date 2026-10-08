"""Task composition, machine bindings and effective experiment evidence."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from supernav.experiments.deployment import deploy_task
from supernav.runtime.config import instruction_rows, load_instruction_manifest, raw_instruction_rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manifest_includes_are_relative_and_defaults_only_fill_missing_fields(tmp_path, monkeypatch):
    source = write_json(tmp_path / "sources" / "one.json", {
        "instructions": [
            {"task_id": "raw", "ground_truth": {"distance": 1.0}, "scene": None},
            {"task_id": "agent", "slug": "  agent  ", "text": "  Original text  "},
            "retained raw entry",
        ],
    })
    middle = write_json(tmp_path / "groups" / "middle.json", {
        "includes": [{"path": "../sources/one.json", "defaults": {"scene": "child"}}],
    })
    root = write_json(tmp_path / "recipes" / "tasks.json", {
        "description": "composed task set",
        "includes": [{"path": "../groups/middle.json", "defaults": {
            "scene": "parent", "ground_truth": {"distance": 9.0, "extra": 2},
        }}],
        "instructions": [{"task_id": "last"}],
    })
    before = {path: path.read_bytes() for path in (source, middle, root)}
    monkeypatch.chdir(tmp_path / "sources")
    manifest = load_instruction_manifest(root)
    assert "includes" not in manifest
    assert manifest["description"] == "composed task set"
    assert manifest["instructions"][0] == {
        "task_id": "raw", "ground_truth": {"distance": 1.0}, "scene": None,
    }
    assert manifest["instructions"][1]["scene"] == "child"
    assert manifest["instructions"][-2:] == ["retained raw entry", {"task_id": "last"}]
    assert [row["task_id"] for row in raw_instruction_rows(root)] == ["raw", "agent", "last"]
    agent_rows = instruction_rows(root)
    assert len(agent_rows) == 1
    assert agent_rows[0]["text"] == "Original text"
    assert agent_rows[0]["slug"] == "agent"
    assert before == {path: path.read_bytes() for path in before}


def test_manifest_include_cycle_and_shared_child(tmp_path):
    first = write_json(tmp_path / "first.json", {"includes": ["second.json"]})
    second = write_json(tmp_path / "second.json", {"includes": ["first.json"]})
    with pytest.raises(ValueError, match=r"include cycle:.*first.json.*second.json.*first.json"):
        load_instruction_manifest(first)
    write_json(second, {"instructions": [{"task_id": "shared"}]})
    write_json(first, {"includes": ["second.json", "second.json"]})
    assert raw_instruction_rows(first) == [{"task_id": "shared"}, {"task_id": "shared"}]


@pytest.mark.parametrize("entry", [None, {}, {"path": ""}, {"path": "one.json", "defaults": []}])
def test_invalid_manifest_include_is_rejected(tmp_path, entry):
    path = write_json(tmp_path / "invalid.json", {"includes": [entry]})
    with pytest.raises(ValueError, match="manifest include"):
        load_instruction_manifest(path)


def test_deployment_changes_only_bound_paths_and_preserves_source_row(monkeypatch):
    monkeypatch.setenv("TEST_TASK_ROOT", "/mounted/data")
    config = {"deployment": {"path_prefixes": {
        "/original": "${TEST_TASK_ROOT}",
        "/original/special": "/specialized",
    }}}
    row = {
        "source_episode": "/original/special/scene.json.gz",
        "scene_dataset_config_file": "/original/scenes.json",
        "text": "Look in /original/special and preserve ${TEST_TASK_ROOT}.",
        "ground_truth": {"source_episode": "/original-other/untouched.json.gz"},
        "targets": [{"source_episode": "/original/second.json.gz", "position": [1, 2, 3]}],
    }
    before = copy.deepcopy((config, row))
    actual = deploy_task(config, row)
    assert actual["source_episode"] == "/specialized/scene.json.gz"
    assert actual["scene_dataset_config_file"] == "/mounted/data/scenes.json"
    assert actual["text"] == row["text"]
    assert actual["ground_truth"] == row["ground_truth"]
    assert actual["targets"][0]["source_episode"] == "/mounted/data/second.json.gz"
    assert (config, row) == before
    actual["targets"][0]["position"].append(4)
    assert row["targets"][0]["position"] == [1, 2, 3]


@pytest.mark.parametrize("deployment", [[], False, {"path_prefixes": []},
    {"path_prefixes": {"": "/x"}}, {"path_prefixes": {"/x": None}}])
def test_deployment_rejects_malformed_bindings(deployment):
    with pytest.raises(ValueError, match="deployment"):
        deploy_task({"deployment": deployment}, {})


def test_episode_evidence_matches_effective_agent_settings(tmp_path, monkeypatch):
    from supernav.backends.base import EpisodeTask
    from supernav.experiments import episode

    prepared = {}

    class Backend:
        name = "habitat"
        default_port = 0

        def prepare_task(self, config, args, arm, root):
            return EpisodeTask("first", args.instruction, "scene")

        def configure_agent(self, config, arm, task, root, run_dir):
            config["_mcp_launch"] = {"name": "world", "port": config["bridge"]["port"]}
            arm["environment"] = {"WORLD_PORT": str(config["bridge"]["port"])}

        @contextmanager
        def episode(self, **kwargs):
            yield None

        def prompt(self, **kwargs):
            return kwargs["task"].instruction

    class Agent:
        name = "codex"

        def prepare_project(self, **kwargs):
            prepared.update(copy.deepcopy(kwargs))
            project = kwargs["run_dir"] / "project"
            project.mkdir()
            return project

    monkeypatch.setattr(episode, "get_backend", lambda config: Backend())
    monkeypatch.setattr(episode, "get_agent_backend", lambda name: Agent())
    config = write_json(tmp_path / "experiment.json", {"prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(tmp_path), "output_dir": "default-output",
        "agent": "codex", "agents": {"codex": {"provider_mode": "experiment", "timeout_s": 500}},
        "model": {"name": "old"}, "arms": {"plain": {"skill_mode": "none"}},
    })
    original = config.read_bytes()
    output = tmp_path / "actual-output"
    monkeypatch.setattr(sys, "argv", [
        "run-one", "--config", str(config), "--arm", "plain", "--slug", "first",
        "--instruction", "Original instruction", "--model", "selected-model",
        "--codex-provider", "user", "--timeout-s", "7", "--bridge-port", "32123",
        "--output-dir", str(output), "--dry-run",
    ])
    assert episode.main() == 0
    run = next(path.parent for path in output.rglob("execution.resolved.json"))
    frozen = json.loads((run / "execution.resolved.json").read_text())
    assert frozen["agent_config"] == prepared["agent_cfg"]
    assert frozen["agent_config"]["provider_mode"] == "user"
    assert frozen["model"] == prepared["model_cfg"] == {"name": "selected-model"}
    assert frozen["arm_config"] == prepared["arm_cfg"]
    assert frozen["timeout_s"] == 7
    assert frozen["arm_config"]["environment"]["WORLD_PORT"] == "32123"
    composed = json.loads((run / "experiment.resolved.json").read_text())
    assert composed == prepared["config"]
    assert composed["_mcp_launch"]["port"] == 32123
    assert (run / "prompt.txt").read_text() == "Original instruction"
    assert config.read_bytes() == original


def test_sweep_freezes_selected_deployed_rows_and_cli_overrides(tmp_path, monkeypatch):
    from supernav.experiments import sweep

    rows = [
        {"task_id": "one", "slug": "one", "text": "First", "source_episode": "/old/one.json"},
        {"task_id": "two", "slug": "two", "text": "Second", "source_episode": "/old/two.json"},
    ]
    backend = SimpleNamespace(name="habitat", tasks=lambda *args: rows)
    monkeypatch.setattr(sweep, "get_backend", lambda config: backend)
    calls = []
    monkeypatch.setattr(sweep, "run_one", lambda args: calls.append(args) or {"dry_run": True})
    config = write_json(tmp_path / "experiment.json", {"prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(tmp_path), "output_dir": "default-output",
        "arms": {"plain": {"skill_mode": "none"}},
        "deployment": {"path_prefixes": {"/old": "/new"}},
    })
    output = tmp_path / "chosen-output"
    monkeypatch.setattr(sys, "argv", [
        "run", "--config", str(config), "--task-ids", "two", "--sweep-id", "test",
        "--model", "selected", "--codex-provider", "user", "--bridge-port", "32123",
        "--reps", "2", "--rep-start", "3", "--output-dir", str(output), "--dry-run",
    ])
    assert sweep.main() == 0
    frozen = json.loads((output / "_sweeps/test/instructions.resolved.json").read_text())
    assert frozen["instructions"] == [{**rows[1], "source_episode": "/new/two.json"}]
    assert len(calls) == 2
    assert [call.rep for call in calls] == ["3", "4"]
    assert all(call.model == "selected" and call.codex_provider == "user" for call in calls)
    assert all(call.bridge_port == 32123 for call in calls)
    assert all(json.loads(call.metadata) == frozen["instructions"][0] for call in calls)
    assert rows[1]["source_episode"] == "/old/two.json"
