"""Benchmark tools operate from installed package resources."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = {
    "build_prompt": "supernav.experiments.prompt",
    "build_global_task_configs": "supernav.experiments.preparation.global_tasks",
    "build_multi_global_task_configs": "supernav.experiments.preparation.multi_global_tasks",
    "build_objectnav_hm3d_instructions": "supernav.experiments.preparation.objectnav_hm3d",
    "build_ovon_mtu3d_instructions": "supernav.experiments.preparation.ovon_mtu3d",
    "score_objectnav_hm3d": "supernav.evaluation.objectnav.score_hm3d",
    "score_multi_objectnav": "supernav.evaluation.objectnav.score_multi",
    "collect_objectnav_scores": "supernav.evaluation.objectnav.collection",
    "aggregate": "supernav.evaluation.aggregate",
    "make_video": "supernav.evaluation.video.render",
    "make_all_videos": "supernav.evaluation.video.batch",
    "render_objectnav_gt_gallery": "supernav.evaluation.video.gallery",
    "metrics": "supernav.evaluation.metrics_cli",
    "timing_summary": "supernav.runtime.timing_summary",
}


def test_portable_tool_import_boundaries(tmp_path):
    code = """
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'harness', 'habitat_agent', 'pluggable_harness', 'habitat_sim', 'ai2thor',
        }:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
for name in sys.argv[1:]:
    importlib.import_module(name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *TOOLS.values()],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", [
    "supernav.experiments.prompt", "supernav.evaluation.objectnav.score_hm3d",
    "supernav.evaluation.objectnav.score_multi",
])
def test_canonical_module_runs_outside_checkout(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_batch_video_default_spawns_installed_renderer(tmp_path, monkeypatch):
    from supernav.evaluation.video import batch

    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    result = batch.run_make_video(
        None, "episode", tmp_path / "runs", tmp_path / "visuals", tmp_path,
        False, accelerated=True, max_wait_s=1.5,
    )
    assert result == 0
    command, options = commands[0]
    assert command[:3] == (sys.executable, "-m", "supernav.evaluation.video.render")
    assert str(tmp_path / "runs" / "episode" / "replay_accelerated.mp4") in command
    assert command[-3:] == ("--accelerated", "--max-wait-s", "1.5")
    assert options["cwd"] == str(tmp_path)


def test_shared_success_preserves_bbox_boundary_and_viewpoint_fallback(monkeypatch):
    from supernav.evaluation import success

    task = {
        "task_id": "box", "goal_radius": 1.0,
        "target_object": {"bbox_center": [0, 0, 0], "bbox_size": [2, 2, 2]},
    }
    assert success.goal_success([2, 99, 0], task) == (True, 1.0, success.CRITERION_BBOX)
    assert success.goal_success([2.001, 0, 0], task)[0] is False
    task.pop("target_object")
    task["goal_position"] = [0, 0, 0]
    assert success.goal_success([0, 1, 0], task) == (True, 1.0, success.CRITERION_VIEWPOINT)


def test_collection_freezes_composed_manifest_and_preserves_source(tmp_path):
    from supernav.evaluation.objectnav import collection
    from supernav.runtime.config import load_instruction_manifest

    source = tmp_path / "inputs"
    source.mkdir()
    (source / "rows.json").write_text(json.dumps({
        "instructions": [{"task_id": "first", "scene": "scene"}],
    }))
    recipe = source / "manifest.json"
    recipe.write_text(json.dumps({
        "includes": [{"path": "rows.json", "defaults": {"object_category": "chair"}}],
    }))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    result = collection.score_snapshot(
        {"instructions": str(recipe), "scenes_root": str(tmp_path / "scenes")},
        [], snapshot,
    )
    assert result["0.2m"]["episodes"] == 0
    assert (snapshot / "instructions.source.json").read_bytes() == recipe.read_bytes()
    # The snapshot must be loadable after the original include files disappear.
    (source / "rows.json").unlink()
    frozen = load_instruction_manifest(snapshot / "instructions.json")
    assert frozen["instructions"] == [{
        "task_id": "first", "scene": "scene", "object_category": "chair",
    }]


def test_collection_preserves_plain_manifest_bytes(tmp_path):
    from supernav.evaluation.objectnav import collection

    manifest = tmp_path / "manifest.json"
    original = b'{"description":"synthetic", "instructions": []}\n'
    manifest.write_bytes(original)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    collection.score_snapshot(
        {"instructions": str(manifest), "scenes_root": str(tmp_path / "scenes")},
        [], snapshot,
    )
    assert (snapshot / "instructions.json").read_bytes() == original
    assert not (snapshot / "instructions.source.json").exists()


def test_scorer_keeps_composed_evaluation_rows_without_prompt_fields(tmp_path, monkeypatch):
    from supernav.evaluation.objectnav import score_hm3d

    (tmp_path / "rows.json").write_text(json.dumps({
        "instructions": [{
            "task_id": "first", "scene": "scene", "object_category": "chair",
            "source_episode": "episode.json.gz", "goal_key": "chair",
            "ground_truth": {"initial_geodesic_distance_m": 5.0, "hm3d_split": "val"},
        }],
    }))
    recipe = tmp_path / "manifest.json"
    recipe.write_text(json.dumps({"includes": ["rows.json"]}))
    runs = tmp_path / "runs"
    episode = runs / "episode"
    episode.mkdir(parents=True)
    (episode / "run.json").write_text('{"task_id":"first"}')
    (episode / "metrics.json").write_text('{"success":true,"close_called":false}')
    monkeypatch.setattr(score_hm3d, "NavmeshGeodesic", lambda path: None)
    monkeypatch.setattr(score_hm3d, "load_goal_geometry", lambda *args: ([], [[0, 0, 0]]))
    monkeypatch.setattr(sys, "argv", [
        "score", "--runs-root", str(runs), "--instructions", str(recipe),
        "--scenes-root", str(tmp_path / "scenes"),
    ])
    assert score_hm3d.main() == 0
    report = json.loads((runs / "official_score.json").read_text())
    assert report["summary"]["episodes"] == 1
    assert report["summary"]["sr"] == 0.0
    assert report["episodes"][0]["claim_success"] is True
    assert report["episodes"][0]["reason"] == "no_stop_called"
