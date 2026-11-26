from __future__ import annotations

import json
from pathlib import Path

import pytest

from dimos.agents.harness.go2_prior_run import (
    resolve_go2_prior_run,
    seed_go2_prior_graph_memory,
)


def _write_prior_root(root: Path) -> None:
    (root / "logs" / "panoramas").mkdir(parents=True)
    (root / "logs" / "go2-graph-memory.json").write_text(
        json.dumps({"nodes": [{"id": "A1"}]}), encoding="utf-8"
    )
    (root / "logs" / "go2-slam-status.json").write_text(
        json.dumps({"latest_slam_map_metadata": {"status": "static"}}), encoding="utf-8"
    )
    (root / "logs" / "panoramas" / "front.jpg").write_bytes(b"jpeg")
    relocalization = root / "slam" / "relocalization"
    relocalization.mkdir(parents=True)
    (relocalization / "global_map.pc2.lcm").write_bytes(b"map")
    (relocalization / "global_costmap.occ.lcm").write_bytes(b"occ")
    (relocalization / "manifest.json").write_text(
        json.dumps({"global_map_file": "global_map.pc2.lcm"}), encoding="utf-8"
    )


def test_resolve_go2_prior_run_discovers_complete_artifacts(tmp_path: Path) -> None:
    _write_prior_root(tmp_path)

    artifacts = resolve_go2_prior_run(tmp_path)

    assert artifacts is not None
    assert artifacts.graph_memory_path == tmp_path / "logs" / "go2-graph-memory.json"
    assert artifacts.slam_status_path == tmp_path / "logs" / "go2-slam-status.json"
    assert artifacts.panoramas_path == tmp_path / "logs" / "panoramas"
    assert artifacts.relocalization_bundle_path == tmp_path / "slam" / "relocalization"
    assert artifacts.relocalization_map_file == tmp_path / "slam" / "relocalization" / "global_map.pc2.lcm"
    assert artifacts.relocalization_costmap_file == tmp_path / "slam" / "relocalization" / "global_costmap.occ.lcm"
    assert artifacts.map_available is True
    assert artifacts.missing == ()


def test_resolve_go2_prior_run_allows_partial_root(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()

    artifacts = resolve_go2_prior_run(tmp_path)

    assert artifacts is not None
    assert artifacts.map_available is False
    assert "logs/go2-graph-memory.json" in artifacts.missing


def test_resolve_go2_prior_run_accepts_native_dimos_log_root(tmp_path: Path) -> None:
    graph_memory = tmp_path / "go2-graph-memory.json"
    graph_memory.write_text('{"schema_version": 6, "nodes": {}}', encoding="utf-8")
    panoramas = tmp_path / "panoramas"
    panoramas.mkdir()

    artifacts = resolve_go2_prior_run(tmp_path)

    assert artifacts is not None
    assert artifacts.graph_memory_path == graph_memory
    assert artifacts.panoramas_path == panoramas
    assert "slam/relocalization/global_map.pc2.lcm" in artifacts.missing

    destination = tmp_path / "operator-console" / "task" / "logs" / "go2-graph-memory.json"
    seed = seed_go2_prior_graph_memory(artifacts, graph_memory_path=destination)

    assert seed is not None
    assert seed["graph_memory_seeded"] is True
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 6


def test_resolve_go2_prior_run_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_go2_prior_run(tmp_path / "missing")


def test_seed_go2_prior_graph_memory_copies_memory_and_panoramas(tmp_path: Path) -> None:
    prior = tmp_path / "prior"
    _write_prior_root(prior)
    artifacts = resolve_go2_prior_run(prior)
    destination = tmp_path / "new" / "logs" / "go2-graph-memory.json"

    seed = seed_go2_prior_graph_memory(artifacts, graph_memory_path=destination)

    assert seed is not None
    assert seed["graph_memory_seeded"] is True
    assert seed["panoramas_seeded"] is True
    assert json.loads(destination.read_text(encoding="utf-8"))["nodes"][0]["id"] == "A1"
    assert (destination.parent / "panoramas" / "front.jpg").read_bytes() == b"jpeg"
