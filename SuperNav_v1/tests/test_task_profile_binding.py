"""One task profile binds prompts, private initialization, MCP and metrics."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from supernav.backends.habitat.config import build_mcp_spec
from supernav.backends.habitat.experiment import HabitatBackend
from supernav.experiments.config import load_experiment_config
from supernav.methods.navigation.prompts import global_task_agent_policy
from supernav.methods.navigation.task_profile import resolve_task_profile
from supernav.runtime.config import merged_arm
from supernav.runtime.mcp import resolve_mcp_spec

ROOT = Path(__file__).resolve().parents[1]
FLAG = "HAB_MCP_GLOBAL_TASK"


def _args(**updates):
    values = {
        "instruction": "Find the chair.", "task_id": "test-chair", "slug": "chair",
        "metadata": {}, "rep": None, "run_id": None, "overwrite": False, "dry_run": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _config(tmp_path, profile="global_task"):
    return {
        "benchmark_profile": profile,
        "workspace_root": str(ROOT),
        "output_dir": str(tmp_path / "runs"),
        "scene": "task-scene",
        "scene_dataset_config_file": str(tmp_path / "scenes.json"),
        "spawn": {"start_position": [1.25, 0.2, -2.5], "start_rotation": [0.0, 0.0, 0.0, 1.0], "sensor_height": 1.25},
        "bridge": {"host": "127.0.0.1", "per_episode": True},
        "mcp": {},
        "prompts_dir": str(ROOT / "configs/benchmarks/global_task/prompts"),
        "environment": {"backend": "habitat"},
        "arms": {"visual_point_skill": {"movement": "visual_point", "skill_mode": "none"}},
    }


@pytest.mark.parametrize("configured,name,global_mode,metrics", [
    (None, "standard", False, ""),
    ("", "standard", False, ""),
    ("standard", "standard", False, ""),
    ("global_task", "global_task", True, "global_task"),
])
def test_task_profile_has_one_interpretation(configured, name, global_mode, metrics):
    profile = resolve_task_profile({"benchmark_profile": configured})
    assert (profile.name, profile.is_global, profile.metrics_profile) == (name, global_mode, metrics)


@pytest.mark.parametrize("value", ["global", "global-task", "unknown", 1, False, {}, []])
def test_unknown_task_profiles_are_rejected(value):
    with pytest.raises(ValueError, match="benchmark_profile"):
        resolve_task_profile({"benchmark_profile": value})


def test_canonical_geometry_recipe_binds_task_profile_without_manual_flag(tmp_path):
    config = load_experiment_config(ROOT / "configs/experiments/habitat-geo-based-executor.json")
    config["scene_dataset_config_file"] = str(tmp_path / "private-scenes.json")
    arm = merged_arm(config, "default")
    assert FLAG not in config["mcp"].get("environment", {})
    assert FLAG not in arm.get("environment", {})
    row = {
        "scene": "manifest-scene", "scene_dataset_config_file": str(tmp_path / "row-scenes.json"),
        "spawn": {"start_position": [3.0, 0.25, -8.0], "start_rotation": [0.0, 1.0, 0.0, 0.0], "sensor_height": 1.3},
        "ground_truth": {"final_position": [50.0, 0.25, 60.0]},
    }
    task = HabitatBackend().prepare_task(config, _args(metadata=row), arm, ROOT)
    spec = build_mcp_spec(workspace_root=ROOT, config=config, arm_cfg=arm)
    assert spec.environment[FLAG] == "1"
    defaults = json.loads(arm["_benchmark_process_environment"]["HAB_MCP_INIT_DEFAULTS_JSON"])
    assert defaults == {
        "scene": "manifest-scene", "scene_dataset_config_file": str(tmp_path / "row-scenes.json"),
        "start_position": [3.0, 0.25, -8.0], "start_rotation": [0.0, 1.0, 0.0, 0.0], "sensor_height": 1.3, "depth": True,
    }
    assert "ground_truth" not in defaults
    assert "HAB_MCP_INIT_DEFAULTS_JSON" not in spec.environment
    assert task.ground_truth == row["ground_truth"]
    assert "Call hab_init_scene exactly once" in global_task_agent_policy(config, arm_cfg=arm)


@pytest.mark.parametrize("profile", [None, "standard"])
def test_standard_profile_overrides_stale_shell_global_flag_without_private_defaults(profile, tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    config = _config(tmp_path, profile)
    arm = merged_arm(config, "visual_point_skill")
    HabitatBackend().prepare_task(config, _args(), arm, ROOT)
    spec = build_mcp_spec(workspace_root=ROOT, config=config, arm_cfg=arm)
    assert spec.environment[FLAG] == "0"
    assert {**os.environ, **spec.environment}[FLAG] == "0"
    assert "_benchmark_process_environment" not in arm
    assert global_task_agent_policy(config, arm_cfg=arm) == ""


@pytest.mark.parametrize("profile,value,expected", [
    ("global_task", "1", "1"), ("global_task", True, "1"),
    ("global_task", "yes", "1"), ("standard", "0", "0"),
    ("standard", False, "0"), ("standard", "off", "0"),
])
def test_public_task_flags_are_rejected_even_when_they_match_profile(profile, value, expected, tmp_path):
    config = _config(tmp_path, profile)
    config["mcp"]["environment"] = {FLAG: value}
    arm = {"environment": {FLAG: value}}
    with pytest.raises(ValueError, match=FLAG):
        build_mcp_spec(workspace_root=ROOT, config=config, arm_cfg=arm)


@pytest.mark.parametrize("source", ["mcp", "arm"])
@pytest.mark.parametrize("profile,value", [("global_task", "0"), ("standard", "1"), ("global_task", "maybe")])
def test_conflicting_or_invalid_flags_fail_before_bridge_or_agent_start(source, profile, value, tmp_path, monkeypatch):
    import supernav.experiments.episode as episode
    config = _config(tmp_path, profile)
    if source == "mcp":
        config["mcp"]["environment"] = {FLAG: value}
    else:
        config["arms"]["visual_point_skill"]["environment"] = {FLAG: value}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    started = []

    def unexpected_start(*args, **kwargs):
        started.append(True)
        raise AssertionError("validation happened after an external component was selected")

    monkeypatch.setattr(HabitatBackend, "episode", unexpected_start)
    monkeypatch.setattr(episode, "get_agent_backend", unexpected_start)
    with pytest.raises(ValueError, match=FLAG):
        episode.run_one(_args(config=str(path), arm="visual_point_skill"))
    assert started == []
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("profile,expected", [("global_task", "1"), ("standard", "0")])
def test_explicit_mcp_command_also_receives_the_backend_task_profile(profile, expected, tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "0" if expected == "1" else "1")
    config = _config(tmp_path, profile)
    config["mcp"] = {"command": sys.executable, "args": ["-m", "supernav", "mcp"]}
    arm = merged_arm(config, "visual_point_skill")
    backend = HabitatBackend()
    task = backend.prepare_task(config, _args(), arm, ROOT)
    backend.configure_agent(config, arm, task, ROOT, tmp_path / "run")
    spec = resolve_mcp_spec(workspace_root=ROOT, config=config, arm_cfg=arm)
    assert spec.environment[FLAG] == expected


@pytest.mark.parametrize("profile,expected_success,semantics", [
    ("global_task", False, "closed_and_structured_achieved"),
    ("standard", True, "legacy_close_call"),
])
def test_metrics_consume_the_same_profile_without_turning_a_claim_into_ground_truth(profile, expected_success, semantics, tmp_path):
    config = _config(tmp_path, profile)
    arm = merged_arm(config, "visual_point_skill")
    backend = HabitatBackend()
    args = _args(arm="visual_point_skill")
    task = backend.prepare_task(config, args, arm, ROOT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {"type": "tool_call", "name": "hab_close_session", "arguments": {"outcome": "achieved"}},
        {"type": "tool_result", "name": "hab_close_session", "content": json.dumps({"closed": False, "terminal_claim": {"outcome": "achieved"}})},
    ]
    metrics = backend.collect(config=config, root=ROOT, run_dir=run_dir, task=task, args=args, result=None, events=events, session=None)
    assert metrics["success"] is expected_success
    assert metrics["success_semantics"] == semantics
    assert metrics["agent_terminal_claim"] is None
    assert "spl" not in metrics


def _mcp_contract_snapshot(environment, private_environment, tmp_path):
    code = r'''
import asyncio, json
from supernav.methods.navigation import mcp_server as module
body = {'session_id':'binding-test','status':'ok','scene_dataset_config_file':'/private/scenes.json',
        'start_position':[1,2,3],'metrics':{'distance_to_goal':0.25},
        'visible_nav_targets':[{'target_ref':'visible-chair','label':'chair','position':[9,8,7]}]}
module._write_benchmark_audit = lambda **kwargs: None
visible = module._finalize_mcp_response(tool_name='init_scene',tool_seq=1,body=body,inline_paths=[],inline_images=False)
print(json.dumps({'tools':[tool.model_dump(mode='json') for tool in asyncio.run(module.mcp.list_tools())],
                  'init':module._apply_benchmark_init_defaults('init_scene',{'scene':'model-selected'}),
                  'visible':json.loads(visible),'original':body},sort_keys=True))
'''
    env = {
        "PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "src"),
        "NAV_ARTIFACTS_DIR": str(tmp_path), **environment, **private_environment,
    }
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_derived_global_binding_preserves_mcp_schema_and_hidden_result_contract(tmp_path):
    config = _config(tmp_path)
    arm = merged_arm(config, "visual_point_skill")
    HabitatBackend().prepare_task(config, _args(), arm, ROOT)
    derived = build_mcp_spec(workspace_root=ROOT, config=config, arm_cfg=arm)
    private = arm["_benchmark_process_environment"]
    current = _mcp_contract_snapshot(dict(derived.environment), private, tmp_path / "derived")
    # Compare with the original low-level protocol binding; public recipes no
    # longer accept this flag, while the backend still owns this same schema.
    original_environment = dict(derived.environment)
    original_environment[FLAG] = "1"
    previous = _mcp_contract_snapshot(original_environment, private, tmp_path / "original")
    assert current == previous
    assert current["tools"]
    assert current["init"]["scene"] == "task-scene"
    assert current["init"]["start_position"] == [1.25, 0.2, -2.5]
    assert current["visible"] == {
        "session_id": "binding-test", "status": "ok",
        "visible_nav_targets": [{"target_ref": "visible-chair", "label": "chair"}],
    }
    assert current["original"]["metrics"] == {"distance_to_goal": 0.25}
    assert current["original"]["start_position"] == [1, 2, 3]


def test_sweep_validates_a_later_conflicting_arm_before_tasks_snapshot_or_episode(tmp_path, monkeypatch):
    import supernav.experiments.sweep as sweep
    config = _config(tmp_path)
    config["skill_runtime"] = {"root": "skills", "skill_set_id": "test"}
    config["arms"] = {
        "valid": {"skill_mode": "native", "skill_name": "global-navigation"},
        "conflict": {"skill_mode": "native", "skill_name": "global-navigation", "environment": {FLAG: "0"}},
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(config))
    touched = []

    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError("sweep performed work before validating all selected arms")

    monkeypatch.setattr(HabitatBackend, "tasks", forbidden)
    monkeypatch.setattr(HabitatBackend, "episode", forbidden)
    monkeypatch.setattr(sweep, "create_skill_snapshot", forbidden)
    monkeypatch.setattr(sweep, "load_skill_snapshot", forbidden)
    monkeypatch.setattr(sweep, "run_one", forbidden)
    monkeypatch.setattr(sys, "argv", ["supernav", "--config", str(path), "--arms", "valid,conflict"])
    with pytest.raises(ValueError, match=FLAG):
        sweep.main()
    assert touched == []
    assert not (tmp_path / "runs").exists()


def test_sweep_rejects_unsupported_fields_even_in_an_unselected_arm(tmp_path, monkeypatch):
    import supernav.experiments.sweep as sweep
    config = _config(tmp_path)
    config["arms"] = {
        "valid": {"skill_mode": "none"},
        "unselected_conflict": {"environment": {FLAG: "0"}},
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(HabitatBackend, "tasks", lambda *args: [
        {"task_id": "test-chair", "slug": "chair", "text": "Find the chair."},
    ])
    selected = []

    def run_one(args):
        selected.append(args.arm)
        return {"metrics": {"returncode": 0}}

    monkeypatch.setattr(sweep, "run_one", run_one)
    monkeypatch.setattr(sys, "argv", ["supernav", "--config", str(path), "--arms", "valid"])
    with pytest.raises(ValueError, match=FLAG):
        sweep.main()
    assert selected == []
    assert not (tmp_path / "runs").exists()


def test_standard_binding_disables_inherited_private_initialization_and_global_projection(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    spec = build_mcp_spec(workspace_root=ROOT, config=_config(tmp_path, "standard"), arm_cfg={})
    snapshot = _mcp_contract_snapshot(dict(spec.environment), {
        "HAB_MCP_INIT_DEFAULTS_JSON": json.dumps({"scene": "stale-private-scene", "start_position": [7, 8, 9]}),
    }, tmp_path / "standard")
    assert snapshot["init"] == {"scene": "model-selected"}
    assert snapshot["visible"]["start_position"] == [1, 2, 3]
    assert snapshot["visible"]["metrics"] == {"distance_to_goal": 0.25}
