"""Tests for the HM3D ObjectNav bench tooling.

Covers ``supernav.experiments.preparation.objectnav_hm3d`` (episode ->
instructions/config generation) and ``supernav.evaluation.objectnav.score_hm3d``
(official-criterion SR/SPL rescore). All fixtures are synthetic; the
geodesic provider is stubbed so no navmesh or native habitat_sim import is
required.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


import supernav.experiments.preparation.objectnav_hm3d as gen  # noqa: E402
import supernav.evaluation.objectnav.score_hm3d as score  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic episode dataset
# ---------------------------------------------------------------------------

SCENE_A = "hm3d_v0.2/val/00877-aaaSceneA/aaaSceneA.basis.glb"
SCENE_B = "hm3d_v0.2/val/00878-bbbSceneB/bbbSceneB.basis.glb"


def _episode(ep_id, scene_id, category, geod, start=(1.0, 0.0, 2.0)):
    return {
        "episode_id": ep_id,
        "scene_id": scene_id,
        "scene_dataset_config": "./data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        "object_category": category,
        "start_position": list(start),
        "start_rotation": [0.0, 0.7071, 0.0, 0.7071],
        "goals": [],
        "info": {"geodesic_distance": geod, "euclidean_distance": geod * 0.8,
                 "closest_goal_object_id": 7},
    }


def _goal(position, n_view_points=2):
    return {
        "object_category": "synthetic_cube",
        "object_id": 7,
        "object_name": "synthetic_cube_7",
        "position": list(position),
        "view_points": [
            {"agent_state": {"position": [position[0] + 0.5, position[1], position[2] + i * 0.3],
                             "rotation": [0, 1, 0, 0]},
             "iou": 1.0}
            for i in range(n_view_points)
        ],
    }


def _write_scene_gz(path: Path, episodes, goals_by_category):
    payload = {"episodes": episodes, "goals_by_category": goals_by_category}
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


@pytest.fixture()
def episodes_root(tmp_path):
    """Two scenes: A has synthetic_cube+synthetic_sphere episodes, B has one synthetic_cube episode."""
    content = tmp_path / "val" / "content"
    content.mkdir(parents=True)
    eps_a = [
        _episode(0, SCENE_A, "synthetic_cube", 5.0),
        _episode(1, SCENE_A, "synthetic_cube", 3.0),
        _episode(2, SCENE_A, "synthetic_sphere", 7.5),
    ]
    goals_a = {
        "aaaSceneA.basis.glb_synthetic_cube": [_goal([10.0, 0.0, 10.0])],
        "aaaSceneA.basis.glb_synthetic_sphere": [_goal([20.0, 0.0, 20.0])],
    }
    _write_scene_gz(content / "aaaSceneA.json.gz", eps_a, goals_a)
    eps_b = [_episode(0, SCENE_B, "synthetic_cube", 4.0)]
    goals_b = {"bbbSceneB.basis.glb_synthetic_cube": [_goal([30.0, 0.0, 30.0])]}
    _write_scene_gz(content / "bbbSceneB.json.gz", eps_b, goals_b)
    return tmp_path


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def test_scene_hash_from_episode():
    assert gen.scene_hash_from_episode(SCENE_A) == "aaaSceneA"
    assert gen.scene_hash_from_episode("hm3d_v0.2/val/00877-syntheticSceneC/syntheticSceneC.basis.glb") == "syntheticSceneC"


def test_split_from_episode():
    assert gen.split_from_episode(SCENE_A, "val") == "val"
    assert gen.split_from_episode("weird", "train") == "train"


def test_build_rows_fields_and_grouping(episodes_root):
    rows = gen.build_rows(episodes_root, "val", "/scenes/cfg.json", 1.25)
    assert len(rows) == 4
    # grouped by scene (A's three episodes first, then B)
    assert [r["scene"] for r in rows] == ["aaaSceneA"] * 3 + ["bbbSceneB"]
    r0 = rows[0]
    # task ids/slugs are pre-normalized (harness slugify lowercases them);
    # the scene hash itself keeps its case for the case-sensitive handle lookup
    assert r0["task_id"] == "objectnav_synthetic_cube_aaascenea_0"
    assert r0["slug"] == "synthetic_cube-aaascenea-0"
    assert r0["text"] == "Find and approach a synthetic cube."
    assert r0["scene"] == "aaaSceneA"  # bare hash, not the episode scene_id
    assert r0["scene_dataset_config_file"] == "/scenes/cfg.json"
    assert r0["spawn"]["start_position"] == [1.0, 0.0, 2.0]
    assert r0["spawn"]["start_rotation"] == [0.0, 0.7071, 0.0, 0.7071]
    assert r0["spawn"]["sensor_height"] == 1.25
    assert r0["goal_key"] == "aaaSceneA.basis.glb_synthetic_cube"
    gt = r0["ground_truth"]
    assert gt["task_type"] == "objectnav"
    assert gt["initial_geodesic_distance_m"] == 5.0
    assert gt["goal_instance_count"] == 1
    assert gt["goal_viewpoint_count"] == 2
    assert gt["hm3d_scene_id"] == SCENE_A
    assert gt["hm3d_split"] == "val"
    assert gt["closest_goal_object_id"] == 7


def test_build_rows_scene_filter_and_max(episodes_root):
    rows = gen.build_rows(episodes_root, "val", "/c.json", 1.25, scenes={"bbbSceneB"})
    assert [r["scene"] for r in rows] == ["bbbSceneB"]
    rows = gen.build_rows(episodes_root, "val", "/c.json", 1.25, max_episodes=2)
    assert len(rows) == 2


def test_write_config_injects_navmesh_env(tmp_path):
    template = {
        "prompts_dir": "configs/benchmarks/objectnav_hm3d_ovon/prompts",
        "scene": "old_scene",
        "scene_dataset_config_file": "/old.json",
        "output_dir": "data/runs/old",
        "mcp": {"environment": {"EXISTING": "1"}},
        "arms": {"localnav_native_skill": {"movement": "localnav"}},
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template))
    out_path = tmp_path / "out.config.json"
    gen.write_config(template_path, out_path, Path("/tmp/instr.json"), "/scenes/cfg.json", "data/runs/new")
    cfg = json.loads(out_path.read_text())
    assert cfg["scene_dataset_config_file"] == "/scenes/cfg.json"
    assert cfg["output_dir"] == "data/runs/new"
    assert cfg["instructions_file"] == "/tmp/instr.json"
    assert cfg["mcp"]["environment"]["HAB_DEFAULT_AGENT_NAVMESH"] == "0"
    assert cfg["mcp"]["environment"]["EXISTING"] == "1"
    assert cfg["arms"] == template["arms"]  # arm set cloned untouched


# ---------------------------------------------------------------------------
# Rescore helpers
# ---------------------------------------------------------------------------


def test_load_goal_geometry(episodes_root):
    src = str(episodes_root / "val" / "content" / "aaaSceneA.json.gz")
    positions, view_points = score.load_goal_geometry(src, "aaaSceneA.basis.glb_synthetic_cube")
    assert positions == [[10.0, 0.0, 10.0]]
    assert len(view_points) == 2
    assert view_points[0] == [10.5, 0.0, 10.0]


def test_min_euclid_distance_to_points():
    points = [[10.5, 0.0, 10.0], [4.0, 0.0, 0.0]]
    assert score.min_euclid_distance([10.0, 0.0, 9.0], points) == pytest.approx(
        (0.5**2 + 1.0**2) ** 0.5
    )
    assert score.min_euclid_distance([0.0, 0.0, 0.0], []) == float("inf")


def test_path_length():
    pts = [[0, 0, 0], [3, 0, 0], [3, 4, 0]]
    assert score.path_length(pts) == pytest.approx(7.0)


def test_load_trajectory_dense_preferred_and_sparse_fallback(tmp_path):
    sidecar = tmp_path / "s.trajectory.json"
    sidecar.write_text(json.dumps({
        "dense_trajectory": {"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        "trajectory": [{"pose": {"position": [9, 9, 9]}}],
    }))
    assert score.load_trajectory(sidecar) == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    sidecar.write_text(json.dumps({
        "trajectory": [{"pose": {"position": [9, 9, 9]}}, {"pose": {"position": [8, 8, 8]}}],
    }))
    assert score.load_trajectory(sidecar) == [[9, 9, 9], [8, 8, 8]]


# ---------------------------------------------------------------------------
# score_run
# ---------------------------------------------------------------------------


class _StubGeodesic:
    def __init__(self, dist):
        self.dist = dist
        self.calls = []

    def geodesic_to_any(self, split, scene_hash, start, ends):
        self.calls.append((split, scene_hash, start, ends))
        return self.dist


def _row(tmp_path, geod=5.0):
    return {
        "task_id": "objectnav_synthetic_cube_aaaSceneA_0",
        "scene": "aaaSceneA",
        "object_category": "synthetic_cube",
        "source_episode": "unused",
        "goal_key": "aaaSceneA.basis.glb_synthetic_cube",
        "ground_truth": {
            "initial_geodesic_distance_m": geod,
            "hm3d_split": "val",
        },
    }


def _write_run(tmp_path, name, close_called=True, session="sess-1"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"task_id": "objectnav_synthetic_cube_aaaSceneA_0"}))
    (run_dir / "metrics.json").write_text(json.dumps({
        "session_id": session,
        "close_called": close_called,
        "success": close_called,
    }))
    return run_dir


def _write_sidecar(visuals_root, session, points):
    visuals_root.mkdir(parents=True, exist_ok=True)
    (visuals_root / f"{session}.trajectory.json").write_text(json.dumps({
        "dense_trajectory": {"points": points},
    }))


VPS = [[10.5, 0.0, 10.0]]


def test_score_run_euclidean_viewpoint_success(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    # stop 0.71 m from the stored view point (10.5, 0, 10); p > l so SPL = l/p
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], [0, 0, 4], [10.0, 0.0, 9.5]])
    rec = score.score_run(run_dir, _row(tmp_path), VPS, None, visuals)
    assert rec["stopped"] is True
    assert rec["dist_view_point_euclid_m"] == pytest.approx((0.5**2 + 0.5**2) ** 0.5, abs=1e-3)
    assert rec["success"] is True
    assert rec["path_length_m"] == pytest.approx(4.0 + 11.4126, abs=0.01)
    assert rec["spl"] == pytest.approx(5.0 / rec["path_length_m"], abs=0.01)
    assert rec["dtg_geodesic_m"] is None  # no provider supplied


def test_score_run_euclidean_stopped_too_far(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], [10.0, 0.0, 8.4]])  # 1.68 m from VP
    rec = score.score_run(run_dir, _row(tmp_path), VPS, None, visuals)
    assert rec["success"] is False
    assert rec["reason"] == "stopped_too_far"
    assert rec["spl"] == 0.0


def test_score_run_no_stop_is_failure(tmp_path):
    run_dir = _write_run(tmp_path, "run1", close_called=False)
    visuals = tmp_path / "visuals"
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], [10.5, 0.0, 10.0]])  # on the view point
    rec = score.score_run(run_dir, _row(tmp_path), VPS, None, visuals)
    assert rec["success"] is False
    assert rec["reason"] == "no_stop_called"
    assert rec["spl"] == 0.0


def test_score_run_missing_sidecar_is_failure(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    rec = score.score_run(run_dir, _row(tmp_path), VPS, None, tmp_path / "nope")
    assert rec["success"] is False
    assert rec["reason"] == "trajectory_sidecar_missing"


def test_score_run_dtg_recorded_under_euclidean_criterion(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    stop = [10.0, 0.0, 9.5]
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], stop])
    geo = _StubGeodesic(1.83)
    rec = score.score_run(run_dir, _row(tmp_path), VPS, geo, visuals)
    assert rec["success"] is True  # euclidean decides, geodesic only reports DTG
    assert rec["dtg_geodesic_m"] == pytest.approx(1.83)
    assert geo.calls[0][2] == stop


def test_score_run_viewpoint_geodesic_criterion(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    stop = [10.0, 0.0, 9.5]  # 0.71 euclid from VP: euclidean_viewpoint would pass
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], stop])
    geo = _StubGeodesic(0.42)  # geodesic to nearest view point > 0.1
    rec = score.score_run(
        run_dir, _row(tmp_path), VPS, geo, visuals,
        criterion=score.CRITERION_VIEWPOINT_GEODESIC,
    )
    assert rec["success"] is False
    assert rec["success_distance_m"] == 0.1
    assert geo.calls[0][0] == "val"
    assert geo.calls[0][1] == "aaaSceneA"
    assert geo.calls[0][2] == stop

    geo_close = _StubGeodesic(0.05)
    rec2 = score.score_run(
        run_dir, _row(tmp_path), VPS, geo_close, visuals,
        criterion=score.CRITERION_VIEWPOINT_GEODESIC,
    )
    assert rec2["success"] is True


def test_score_run_success_distance_override(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], [10.0, 0.0, 8.4]])  # 1.68 m from VP
    rec = score.score_run(
        run_dir, _row(tmp_path), VPS, None, visuals, success_distance=2.0
    )
    assert rec["success"] is True


def test_aggregate_math():
    records = [
        {"success": True, "spl": 0.5, "stopped": True, "object_category": "synthetic_cube",
         "dtg_geodesic_m": 0.4},
        {"success": False, "spl": 0.0, "stopped": True, "reason": "stopped_too_far",
         "object_category": "synthetic_cube", "dtg_geodesic_m": 2.2},
        {"success": True, "spl": 1.0, "stopped": True, "object_category": "synthetic_sphere",
         "dtg_geodesic_m": None},
    ]
    summary = score.aggregate(records)
    assert summary["episodes"] == 3
    assert summary["sr"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["spl"] == pytest.approx(0.5, abs=1e-3)
    assert summary["dtg_geodesic_mean_m"] == pytest.approx(1.3, abs=1e-3)
    assert summary["per_category"]["synthetic_cube"]["sr"] == 0.5
    assert summary["per_category"]["synthetic_sphere"]["spl"] == 1.0
    assert summary["failure_reasons"] == {"stopped_too_far": 1}


def test_latest_by_task_dedupes(tmp_path):
    import time

    older = _write_run(tmp_path, "run_old")
    newer = _write_run(tmp_path, "run_new")
    old_metrics = older / "metrics.json"
    new_metrics = newer / "metrics.json"
    past = time.time() - 100
    os.utime(old_metrics, (past, past))
    chosen = score._latest_by_task([older, newer])
    assert list(chosen.values()) == [newer]


class _StartAwareGeodesic:
    """Returns distance keyed on the queried start position."""

    def __init__(self, by_start):
        self.by_start = {tuple(s): d for s, d in by_start.items()}

    def geodesic_to_any(self, split, scene_hash, start, ends):
        return self.by_start[tuple(start)]


def _row_with_spawn(tmp_path, geod=5.0):
    row = _row(tmp_path, geod=geod)
    row["spawn"] = {"start_position": [0.0, 0.0, 0.0]}
    return row


def test_score_run_l_source_viewpoint_geodesic_recomputes_l(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    stop = [10.0, 0.0, 9.5]
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], stop])
    # start->VP geodesic 4.0 (vs episode's 5.0), stop->VP geodesic 0.5
    geo = _StartAwareGeodesic({(0.0, 0.0, 0.0): 4.0, tuple(stop): 0.5})
    rec = score.score_run(
        run_dir, _row_with_spawn(tmp_path), VPS, geo, visuals,
        l_source=score.L_SOURCE_VIEWPOINT_GEODESIC,
    )
    assert rec["success"] is True  # euclidean 0.71 <= 1.0
    assert rec["l_star_m"] == 4.0
    # SPL uses the recomputed l: 4.0 / path(15.41)
    assert rec["spl"] == pytest.approx(4.0 / rec["path_length_m"], abs=0.01)
    # episode source keeps the stored value
    rec2 = score.score_run(
        run_dir, _row_with_spawn(tmp_path), VPS, geo, visuals,
        l_source=score.L_SOURCE_EPISODE,
    )
    assert rec2["l_star_m"] == 5.0


def test_score_run_l_source_falls_back_when_unreachable(tmp_path):
    run_dir = _write_run(tmp_path, "run1")
    visuals = tmp_path / "visuals"
    stop = [10.0, 0.0, 9.5]
    _write_sidecar(visuals, "sess-1", [[0, 0, 0], stop])
    geo = _StartAwareGeodesic({(0.0, 0.0, 0.0): float("inf"), tuple(stop): 0.5})
    rec = score.score_run(
        run_dir, _row_with_spawn(tmp_path), VPS, geo, visuals,
        l_source=score.L_SOURCE_VIEWPOINT_GEODESIC,
    )
    assert rec["l_star_m"] == 5.0  # episode fallback


def test_find_navmesh_falls_back_across_splits(tmp_path):
    """val_mini episodes name minival/<id>/...; dataset copies that only
    keep val/ must still resolve the same building's navmesh."""
    nav_dir = tmp_path / "val" / "00800-TEEsavR23oF"
    nav_dir.mkdir(parents=True)
    nav = nav_dir / "TEEsavR23oF.basis.navmesh"
    nav.write_bytes(b"fake")
    assert score.find_navmesh(tmp_path, "minival", "TEEsavR23oF") == nav
    assert score.find_navmesh(tmp_path, "val", "TEEsavR23oF") == nav
    assert score.find_navmesh(tmp_path, "val", "missing") is None


def test_discover_runs_includes_published_links_but_not_reservations(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    ordinary = root / 'ordinary'
    ordinary.mkdir()
    (ordinary / 'run.json').write_text('{}')
    published = tmp_path / 'lane_result'
    published.mkdir()
    (published / 'run.json').write_text('{}')
    (root / 'published').symlink_to(published, target_is_directory=True)
    reserved = tmp_path / 'reservation'
    reserved.mkdir()
    (reserved / 'reservation.json').write_text('{}')
    (root / 'pending').symlink_to(reserved, target_is_directory=True)
    (root / 'missing').symlink_to(tmp_path / 'absent', target_is_directory=True)
    assert score.discover_run_dirs(root) == [ordinary, root / 'published']
