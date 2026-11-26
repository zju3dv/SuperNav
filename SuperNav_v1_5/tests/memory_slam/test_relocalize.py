from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat.memory_slam import relocalize as _MODULE


def _grid_fixture() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 5.0, 101)
    occupied = np.column_stack((x, np.zeros_like(x)))
    free = np.column_stack((x, np.full_like(x, 0.5)))
    bins = np.arange(len(x), dtype=np.int64) % 4
    return (
        {
            "occupied_points": occupied,
            "occupied_bins": bins,
            "free_points": free,
            "free_bins": bins,
        },
        occupied,
        free,
    )


def test_geometric_metrics_reward_comparable_agreement() -> None:
    source_grid, target_occupied, target_free = _grid_fixture()
    context = _MODULE._prepare_geometry_context(
        source_grid, target_occupied, target_free
    )

    metrics = _MODULE._geometric_metrics(context, np.eye(4))

    assert metrics["occupied_match_ratio"] == pytest.approx(1.0)
    assert metrics["occupied_vs_free_conflict_ratio"] == pytest.approx(0.0)
    assert metrics["free_vs_occupied_conflict_ratio"] == pytest.approx(0.0)
    assert metrics["unique_target_support_count"] == len(target_occupied)
    assert metrics["support_trajectory_bins"]["supported_bin_count"] == 4


def test_geometric_conflict_gate_rejects_occupied_on_free() -> None:
    source_grid, target_occupied, target_free = _grid_fixture()
    context = _MODULE._prepare_geometry_context(
        source_grid, target_occupied, target_free
    )
    transform = np.eye(4)
    transform[1, 3] = 0.5

    metrics = _MODULE._geometric_metrics(context, transform)
    hard, _, _ = _MODULE._candidate_gate_failures(
        metrics,
        sparse_node_prior=True,
        fitness=1.0,
        rmse=0.0,
        inliers=1000,
        region_ok=True,
    )

    assert metrics["occupied_vs_free_conflict_ratio"] == pytest.approx(1.0)
    assert "occupied_vs_free_conflict" in hard


def test_geometric_conflict_gate_rejects_free_on_occupied() -> None:
    source_grid, target_occupied, target_free = _grid_fixture()
    context = _MODULE._prepare_geometry_context(
        source_grid, target_occupied, target_free
    )
    transform = np.eye(4)
    transform[1, 3] = -0.5

    metrics = _MODULE._geometric_metrics(context, transform)
    hard, _, _ = _MODULE._candidate_gate_failures(
        metrics,
        sparse_node_prior=True,
        fitness=1.0,
        rmse=0.0,
        inliers=1000,
        region_ok=True,
    )

    assert metrics["free_vs_occupied_conflict_ratio"] == pytest.approx(1.0)
    assert "free_vs_occupied_conflict" in hard


def test_positive_evidence_gate_is_independent_of_high_fitness() -> None:
    source_grid, target_occupied, target_free = _grid_fixture()
    context = _MODULE._prepare_geometry_context(
        source_grid, target_occupied, target_free
    )
    metrics = _MODULE._geometric_metrics(context, np.eye(4))
    metrics["occupied_match_ratio"] = 0.60

    _, positive, diagnostics = _MODULE._candidate_gate_failures(
        metrics,
        sparse_node_prior=True,
        fitness=0.99,
        rmse=0.01,
        inliers=1000,
        region_ok=True,
    )

    assert "low_occupied_match_ratio" in positive
    assert diagnostics == []


def test_candidate_preselection_keeps_geometric_and_fitness_hypotheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_grid, target_occupied, target_free = _grid_fixture()
    context = _MODULE._prepare_geometry_context(
        source_grid, target_occupied, target_free
    )
    correct = np.eye(4)
    wrong = np.eye(4)
    wrong[1, 3] = 0.5

    monkeypatch.setattr(
        _MODULE,
        "_score",
        lambda source, target, transform: (
            (0.95, 0.01, 500, 500)
            if transform[1, 3] > 0.1
            else (0.60, 0.03, 300, 300)
        ),
    )

    selected = _MODULE._candidate_preselection(
        [wrong, correct], None, None, context, 2
    )

    assert np.allclose(selected[0][1], correct)
    assert any(np.allclose(transform, wrong) for _, transform in selected)


def test_sparse_node_prior_crops_only_target_points_before_ransac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_inside = np.column_stack(
        (
            np.linspace(-0.8, 0.8, 25),
            np.linspace(-0.7, 0.7, 25),
        )
    )
    target_inside = np.column_stack(
        (
            np.linspace(9.2, 10.8, 25),
            np.linspace(9.3, 10.7, 25),
        )
    )
    source_points = np.vstack((source_inside, [[25.0, 25.0], [26.0, 24.5]])).astype(
        np.float64
    )
    target_points = np.vstack((target_inside, [[40.0, -12.0], [39.5, -11.5]])).astype(
        np.float64
    )
    prior = {
        "region_bbox": [9.0, 9.0, 11.0, 11.0],
        "agent_local_pose": {"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
        "agent_local_pose_sequence": 7,
    }

    monkeypatch.setattr(_MODULE, "SCALE_PLAN", [(1.0, 1)])
    monkeypatch.setattr(_MODULE, "SPARSE_MIN_SOURCE_POINTS", 1)
    monkeypatch.setattr(_MODULE, "SPARSE_MIN_TARGET_POINTS", 1)
    monkeypatch.setattr(
        _MODULE, "_cloud", lambda points: SimpleNamespace(points=points)
    )
    monkeypatch.setattr(
        _MODULE,
        "_preprocess",
        lambda cloud, voxel_size: (cloud, None),
    )
    seen: list[tuple[int, int]] = []

    def fake_ransac(source, target, source_fpfh, target_fpfh, voxel_size):
        seen.append((len(source.points), len(target.points)))
        return np.eye(4)

    monkeypatch.setattr(_MODULE, "_ransac", fake_ransac)
    monkeypatch.setattr(_MODULE, "_score", lambda *args, **kwargs: (1.0, 0.0, 5, 5))
    monkeypatch.setattr(
        _MODULE,
        "_REG",
        SimpleNamespace(
            evaluate_registration=lambda *args, **kwargs: SimpleNamespace(
                fitness=1.0,
                inlier_rmse=0.0,
                correspondence_set=np.array([[0, 0]], dtype=np.int32),
            ),
            registration_icp=lambda *args, **kwargs: SimpleNamespace(
                transformation=np.eye(4)
            ),
            TransformationEstimationPointToPoint=lambda *args, **kwargs: object(),
            ICPConvergenceCriteria=lambda *args, **kwargs: object(),
        ),
    )

    result = _MODULE.relocalize(
        source_points,
        target_points,
        profile="sparse_node_prior",
        prior=prior,
        source_grid={
            "occupied_points": source_points,
            "occupied_bins": np.arange(len(source_points)) % 4,
            "free_points": source_points + np.array([0.0, 0.3]),
            "free_bins": np.arange(len(source_points)) % 4,
        },
        target_free_points=target_points + np.array([0.0, 0.3]),
    )

    assert seen == [(27, 25)]
    assert result["source_point_count"] == 27
    assert result["target_point_count"] == 25
    assert result["original_source_point_count"] == 27
    assert result["original_target_point_count"] == 27
    assert result["source_region_bbox"] is None
    assert result["target_region_bbox"] == pytest.approx([9.0, 9.0, 11.0, 11.0])


def test_insufficient_points_preserves_source_representation() -> None:
    result = _MODULE.relocalize(
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0]], dtype=np.float64),
        source_representation="active_trajectory_occupancy",
    )

    assert result["accepted"] is False
    assert result["reason"] == "insufficient_points"
    assert result["source_representation"] == "active_trajectory_occupancy"


def test_main_records_recent_trajectory_source_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.xy"
    target = tmp_path / "target.xy"
    output = tmp_path / "result.json"
    source.write_text("0 0\n", encoding="utf-8")
    target.write_text("0 0\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relocalize.py",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(output),
            "--source-representation",
            "recent_trajectory_occupancy",
            "--source-window-length-m",
            "8",
            "--source-window-actual-length-m",
            "7.75",
            "--source-window-first-sequence",
            "12",
            "--source-window-last-sequence",
            "45",
            "--source-window-scan-count",
            "34",
            "--source-window-return-point-count",
            "2048",
            "--source-window-miss-point-count",
            "96",
            "--source-window-observed-cell-count",
            "4096",
            "--source-window-occupied-cell-count",
            "321",
            "--source-window-free-cell-count",
            "777",
            "--source-window-occupancy-probability-threshold",
            "0.65",
            "--source-window-free-probability-threshold",
            "0.46",
            "--source-window-occupancy-build-duration-ms",
            "12.5",
        ],
    )

    assert _MODULE.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_representation"] == "recent_trajectory_occupancy"
    assert result["source_window"] == {
        "requested_length_m": 8.0,
        "actual_length_m": 7.75,
        "first_sequence": 12,
        "last_sequence": 45,
        "scan_count": 34,
        "return_point_count": 2048,
        "miss_point_count": 96,
        "observed_cell_count": 4096,
        "occupied_cell_count": 321,
        "free_cell_count": 777,
        "occupancy_probability_threshold": 0.65,
        "free_probability_threshold": 0.46,
        "occupancy_build_duration_ms": 12.5,
    }
