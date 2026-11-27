"""Public GS adapter, trusted motion evidence, and ordered scoring contracts."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat import primitive_trace
from supernav.evaluation.objectnav.habitatgs_data import SUCCESS, load_release, sha256
from supernav.evaluation.objectnav.export_habitatgs import export_runs, trace_events, validate_session_binding
from supernav.evaluation.objectnav.score_habitatgs import score_submissions, stage_success_rates
from supernav.experiments.preparation.habitatgs import instruction_row, prepare
from supernav.runtime.config import instruction_rows

ROOT = Path(__file__).resolve().parents[1]


def episode(ident="synthetic", count=2):
    pose = lambda x: {"position": [x, 0, 0], "rotation": [0, 0, 0, 1]}
    targets = [{"index": i, "id": f"object_{i}", "description": f"Find synthetic object {i}",
                "instances": {"ids": [str(i)], "labels": ["synthetic"]},
                "bbox": {"center": [i + 1, 0, 0], "size": [1, 1, 1], "coordinate_system": "habitat_world"},
                "viewpoints": [{"id": f"view_{i}", "status": "approved", "pose": pose(i + 1), "visibility": {}}]}
               for i in range(count)]
    route = {"selection_policy": "synthetic", "targets": [
        {"target_index": i, "viewpoint_id": f"view_{i}", "geodesic_distance": 1,
         "path": [[i, 0, 0], [i + 1, 0, 0]]} for i in range(count)],
        "route": [[i, 0, 0] for i in range(count + 1)], "total_geodesic_distance": count}
    return {"schema": "habitat_gs/objectnav/v1", "episode_id": ident,
            "scene": {"id": "synthetic_scene", "asset": "synthetic_scene/test.gs.ply"},
            "start": pose(0), "instruction": "Find the synthetic objects in order.",
            "ordered": count > 1, "targets": targets, "evaluation": {"success": dict(SUCCESS)},
            "ground_truth": {"status": "complete", "variants": {"local_shortest": route, "global_shortest": route}},
            "provenance": {"synthetic": True}}


def submit(e, xs, terminal="stop"):
    events = [{"action": "start", "position": e["start"]["position"]}]
    events.extend({"action": "move_forward", "position": [x, 0, 0]} for x in xs)
    if terminal:
        row = {"action": terminal}
        if terminal == "stop":
            row["position"] = events[-1]["position"]
        events.append(row)
    return {"episode_id": e["episode_id"], "events": events}


def geometry(e):
    return lambda p, gs: min(math.dist(p, g) for g in gs), lambda p: True


def score(es, ss, geo=geometry):
    return score_submissions(es, {"episodes": ss}, geo)


def test_ordered_pass_through_stop_and_mean_spl():
    a, b = episode("a"), episode("b", 3)
    result = score([a, b], [submit(a, [1, 2, 4]), submit(b, [2, 1, 3])])
    assert result["episodes"][0]["success"] is True
    assert result["episodes"][0]["spl"] == .5
    assert result["episodes"][1]["visited_targets"] == 1
    assert result["episodes"][1]["success"] is False
    assert result["episodes"][1]["spl"] == 0
    assert result["summary"]["spl"] == .25
    assert [r["success_rate"] for r in result["stage_sr"]] == [1, .5, 0]
    assert [r["eligible_total"] for r in result["stage_sr"]] == [2, 2, 1]


def test_stage_completion_does_not_require_stop_and_first_terminal_wins():
    e = episode()
    sub = submit(e, [1, 2], "budget_exhausted")
    sub["events"].append({"action": "stop", "position": [2, 0, 0]})
    result = score([e], [sub])
    assert result["summary"]["success_rate"] == 0
    assert [r["success_rate"] for r in result["stage_sr"]] == [1, 1]
    early = submit(e, [1])
    early["events"].extend(sub["events"][2:])
    assert score([e], [early])["episodes"][0]["visited_targets"] == 1


def test_overlap_geodesic_callback_and_strict_boundary():
    e = episode()
    e["targets"][1]["viewpoints"] = copy.deepcopy(e["targets"][0]["viewpoints"])
    assert score([e], [submit(e, [1])])["episodes"][0]["visited_targets"] == 2
    for distance, expected in [(.1, False), (.099, True), (math.inf, False)]:
        out = score([e], [submit(e, [1])], lambda _: (lambda p, gs: distance, lambda p: True))
        assert out["episodes"][0]["success"] is expected


def test_single_requires_terminal_hit_and_stop():
    e = episode(count=1)
    assert not score([e], [submit(e, [1, 3])])["episodes"][0]["success"]
    assert not score([e], [submit(e, [1], None)])["episodes"][0]["success"]
    out = score([e], [submit(e, [1])])
    assert out["episodes"][0]["success"]
    assert "stage_sr" not in out


def test_missing_and_runtime_error_denominators():
    es = [episode("a"), episode("b", 3), episode("c", 3)]
    out = score(es, [submit(es[0], [1, 2]), submit(es[1], [1, 2], "runtime_error")])
    assert out["summary"] == {"total": 3, "evaluated": 2, "runtime_errors": 1,
                              "successes": 1, "success_rate": .5, "spl": .5, "missing_submissions": 1}
    assert out["stage_sr"][2] == {"stage": 3, "eligible_total": 2, "evaluated": 1,
                                   "runtime_errors": 1, "successes": 0, "success_rate": 0}
    assert stage_success_rates([{**out["episodes"][1], "visited_targets": 3}])[0]["success_rate"] is None


def test_unsubmitted_episodes_need_no_mesh_but_remain_in_denominator():
    es = [episode("submitted"), episode("missing")]

    def selected_geometry(e):
        assert e["episode_id"] == "submitted"
        return geometry(e)

    out = score(es, [submit(es[0], [1, 2])], selected_geometry)
    assert out["summary"]["total"] == 2
    assert out["summary"]["missing_submissions"] == 1
    assert out["summary"]["success_rate"] == .5


@pytest.mark.parametrize("case", ["unknown", "duplicate", "wrong_start", "moving_stop", "nan", "offmesh"])
def test_bad_submissions_rejected(case):
    e = episode()
    sub = submit(e, [1, 2])
    ss, geo = [sub], geometry
    if case == "unknown": sub["episode_id"] = "unknown"
    if case == "duplicate": ss.append(copy.deepcopy(sub))
    if case == "wrong_start": sub["events"][0]["position"] = [9, 0, 0]
    if case == "moving_stop": sub["events"][-1]["position"] = [3, 0, 0]
    if case == "nan": sub["events"][1]["position"] = [math.nan, 0, 0]
    if case == "offmesh": geo = lambda _: (lambda p, gs: 0, lambda p: False)
    with pytest.raises(ValueError): score([e], ss, geo)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "dataset"
    scenes = tmp_path / "assets"
    e = episode()
    files = {}
    def put(name, data):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    assets = {}
    for name in ("synthetic_scene/test.gs.ply", "synthetic_scene/test.navmesh"):
        p = scenes / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic runtime asset for checksum tests")
        assets[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    put("scene-assets/v1/manifest.json", {"files": assets})
    for task, row in (("objectnav", episode(count=1)), ("multi-objectnav", e)):
        put(f"{task}/v1/episodes.json", {"episodes": [row]})
        put(f"{task}/v1/episode_v1.schema.json", {
            "type": "object", "required": ["instruction"], "properties": {"instruction": {"type": "string"}}})
    (root / "manifest.json").write_text(json.dumps({"schema": "habitat_gs_public_release/v1", "files": files}))
    return root, scenes


@pytest.mark.parametrize("task", ["objectnav", "multi-objectnav"])
@pytest.mark.parametrize("executor,local_skill,movement_tool", [
    ("geo-based-executor", "localnav-pointnav-geo-based-executor", "hab_visual_point_navigate"),
    ("learned-executor", "localnav-pointnav", "hab_local_navigate"),
])
def test_prepared_runtime_tools_skills_and_private_prompt(
    release, tmp_path, monkeypatch, task, executor, local_skill, movement_tool,
):
    from supernav.experiments import sweep
    from supernav.runtime.codex_agent import CodexAgent

    private_bindings = []
    prepare_project = CodexAgent.prepare_project

    def capture_project(agent, **kwargs):
        project = prepare_project(agent, **kwargs)
        private_bindings.append(dict(agent._benchmark_process_environment))
        return project

    monkeypatch.setattr(CodexAgent, "prepare_project", capture_project)
    dataset, scenes = release
    out = tmp_path / "prepared"
    prepare(dataset_root=dataset, task=task, scene_root=scenes, output_dir=out,
            template_config=ROOT / f"configs/experiments/habitat-{executor}.json",
            sensor_height=1.25, dataset_revision="a" * 40)
    monkeypatch.setattr(sys, "argv", [
        "supernav run", "--config", str(out / "run.json"),
        "--dry-run", "--output-dir", str(out / "dry-run"),
    ])
    assert sweep.main() == 0
    assert len(private_bindings) == 1
    runs = list((out / "dry-run").glob("*/run.json"))
    assert len(runs) == 1
    run = runs[0].parent
    prompt = (run / "prompt.txt").read_text()
    project = run / "codex_project" / ".codex_home"
    client = tomllib.loads((project / "config.toml").read_text())
    server = client["mcp_servers"]["habitat-gs"]
    env = server["env"]
    private = private_bindings[0]
    # Private values live only in the launcher environment; TOML forwards names.
    assert set(server["env_vars"]) == set(private)
    assert "HAB_MCP_INIT_DEFAULTS_JSON" not in env
    assert "HAB_MCP_NAV_GOALS_JSON" not in env
    defaults = json.loads(private["HAB_MCP_INIT_DEFAULTS_JSON"])
    assert defaults["start_position"] == episode()["start"]["position"]
    tools = env["HAB_MCP_TOOL_WHITELIST"].split(",")
    assert movement_tool in tools and "hab_close_session" in tools
    assert "start_position" not in prompt and "start_rotation" not in prompt
    assert str(out / "scenes.scene_dataset_config.json") not in prompt
    assert "Execute this initialization call exactly once" in prompt
    if task == "multi-objectnav":
        assert tools.count("hab_nav_goals") == 1
        goals = json.loads(private["HAB_MCP_NAV_GOALS_JSON"])
        assert [g["index"] for g in goals["targets"]] == [1, 2]
        assert "ORDERED GOALS (2 total)" in prompt
        expected_skill = f"multi-global-navigation-{executor}"
        assert expected_skill in prompt
        required = [expected_skill, f"global-navigation-{executor}", local_skill]
    else:
        assert "hab_nav_goals" not in tools
        assert "HAB_MCP_NAV_GOALS_JSON" not in private
        required = [f"global-navigation-{executor}", local_skill]
    for name in required:
        assert (project / "skills" / name / "SKILL.md").is_file(), name


@pytest.mark.parametrize("override", [
    {"movement": "primitive"}, {"skill_mode": "none"},
    {"skill_name": "unrelated-skill"}, {"tool_whitelist": None},
])
def test_multi_prepare_rejects_unsupported_arm_before_writing(release, tmp_path, override):
    from supernav.experiments.config import load_experiment_config

    dataset, scenes = release
    config = load_experiment_config(ROOT / "configs/experiments/habitat-geo-based-executor.json")
    config["arms"]["default"].update(override)
    template = tmp_path / "unsupported.json"
    template.write_text(json.dumps(config))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="arm"):
        prepare(dataset_root=dataset, task="multi-objectnav", scene_root=scenes, output_dir=out,
                template_config=template, sensor_height=1.25, dataset_revision="a" * 40)
    assert not out.exists()


def test_prepare_preserves_identity_order_pose_and_excludes_gt(release, tmp_path):
    root, scenes = release
    out = tmp_path / "prepared"
    args = dict(dataset_root=root, task="multi-objectnav", scene_root=scenes, output_dir=out,
                template_config=ROOT / "configs/experiments/habitat-geo-based-executor.json",
                sensor_height=1.25, dataset_revision="a" * 40)
    receipt = prepare(**args)
    rows = instruction_rows(out / "instructions.json")
    e = episode()
    row = rows[0]
    assert row["task_id"] == e["episode_id"]
    assert row["text"] == e["instruction"]
    assert row["spawn"]["start_position"] == e["start"]["position"]
    assert row["spawn"]["start_rotation"] == e["start"]["rotation"]
    assert [t["index"] for t in row["targets"]] == [1, 2]
    assert [t["instruction"] for t in row["targets"]] == [t["description"] for t in e["targets"]]
    for hidden in ("viewpoints", "bbox", "ground_truth", "instances", "total_geodesic_distance"):
        assert hidden not in json.dumps(rows)
    assert receipt["assets_verified"]
    cfg = json.loads((out / "run.json").read_text())
    assert cfg["environment"]["primitive_trace_dir"] == str(out / "visuals")
    assert cfg["mcp"]["environment"]["HAB_DEFAULT_AGENT_NAVMESH"] == "0"
    with pytest.raises(ValueError, match="exists"): prepare(**args)
    (scenes / "synthetic_scene/test.navmesh").write_text("changed mesh")
    args["output_dir"] = tmp_path / "rejected"
    with pytest.raises(ValueError, match="checksum"): prepare(**args)
    assert not args["output_dir"].exists()


def test_release_rejects_changed_annotation_hash(release):
    root, _ = release
    (root / "multi-objectnav/v1/episodes.json").write_text('{"episodes": []}')
    with pytest.raises(ValueError, match="checksum"): load_release(root, "multi-objectnav")


def test_trace_keeps_stationary_and_tiny_primitive_moves(tmp_path, monkeypatch):
    from supernav.backends.habitat.simulator.motion import HabitatAdapterNavigationMixin
    from supernav.backends.habitat.simulator.types import _Session
    session = _Session("trace", None, "synthetic_scene", {}, trajectory=[[0, 0, 0]], audit_trajectory=[[0, 0, 0]])
    monkeypatch.setenv("SUPERNAV_PRIMITIVE_TRACE_DIR", str(tmp_path))
    primitive_trace.start_trace(session, [0, 0, 0])
    adapter = SimpleNamespace(_current_position=lambda _: np.array(pos))
    for i, (action, pos) in enumerate([("turn_left", [0, 0, 0]), ("move_forward", [0, 0, 0]),
                                        ("move_forward", [1e-8, 0, 0])], 1):
        session.step_count, session.last_action = i, action
        HabitatAdapterNavigationMixin._record_pose(adapter, session)
    primitive_trace.stop_trace(session, pos)
    events = trace_events(Path(session.primitive_trace_path), episode())
    assert [r["action"] for r in events] == ["start", "turn_left", "move_forward", "move_forward", "stop"]
    assert events[-2]["position"] == [1e-8, 0, 0]
    assert len(session.audit_trajectory) == 1


def test_trace_rejects_action_gap_and_records_reset_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERNAV_PRIMITIVE_TRACE_DIR", str(tmp_path))
    s = SimpleNamespace(session_id="s", scene="synthetic_scene", step_count=0)
    primitive_trace.start_trace(s, [0, 0, 0])
    s.step_count, s.last_action = 2, "move_forward"
    primitive_trace.record_step(s, [1, 0, 0])
    with pytest.raises(ValueError, match="missing/repeated"): trace_events(Path(s.primitive_trace_path), episode())
    s = SimpleNamespace(session_id="reset", scene="synthetic_scene", step_count=0)
    primitive_trace.start_trace(s, [0, 0, 0])
    primitive_trace.invalidate_trace(s, "agent_pose_reset")
    primitive_trace.stop_trace(s, [0, 0, 0])
    assert trace_events(Path(s.primitive_trace_path), episode())[-1] == {"action": "runtime_error"}


def test_export_binds_dataset_and_handles_timeout_and_duplicates(release, tmp_path, monkeypatch):
    root, _ = release
    es, _, provenance = load_release(root, "multi-objectnav")
    runs, visuals = tmp_path / "runs", tmp_path / "visuals"
    d = runs / "first"
    d.mkdir(parents=True)
    monkeypatch.setenv("SUPERNAV_PRIMITIVE_TRACE_DIR", str(visuals))
    s = SimpleNamespace(session_id="s", scene="synthetic_scene", step_count=0)
    primitive_trace.start_trace(s, [0, 0, 0])
    for name, value in {
        "run.json": {"task_id": "synthetic", "terminal_status": "timeout"},
        "task.resolved.json": instruction_row(es[0], 1.25, provenance),
        "metrics.json": {"task_id": "synthetic", "session_id": "s", "timed_out": True, "returncode": 124},
        "execution.resolved.json": {"dry_run": False},
        "canonical.jsonl": {"type": "tool_result", "name": "hab_init_scene", "content": '{"session_id": "s"}'},
    }.items(): (d / name).write_text(json.dumps(value))
    args = dict(episodes=es, provenance=provenance, runs_root=runs, visuals_root=visuals)
    exported = export_runs(**args)
    assert exported["episodes"][0]["events"][-1] == {"action": "budget_exhausted"}
    assert score(es, exported["episodes"])["summary"]["evaluated"] == 1
    (d / "execution.resolved.json").write_text('{"dry_run": true}')
    with pytest.raises(ValueError, match="dry-run"): export_runs(**args)
    (d / "execution.resolved.json").write_text('{"dry_run": false}')
    other = runs / "retry"
    other.mkdir()
    (other / "run.json").write_text('{"task_id": "synthetic"}')
    with pytest.raises(ValueError, match="duplicate"): export_runs(**args)


def test_export_rejects_in_run_session_restarts(tmp_path):
    path = tmp_path / "canonical.jsonl"
    event = lambda s: {"type": "tool_result", "name": "hab_init_scene", "content": json.dumps({"session_id": s})}
    path.write_text("\n".join(json.dumps(event(s)) for s in ("first", "retry")))
    with pytest.raises(ValueError, match="exactly one"): validate_session_binding(path, "retry")
