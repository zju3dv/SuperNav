from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from supernav.experiments import paired


def _args(tmp_path: Path, **updates) -> argparse.Namespace:
    values = {
        "config": "unused.json",
        "retrieve_spawn": None,
        "run_tag": None,
        "model": None,
        "codex_provider": None,
        "bridge_port": None,
        "overwrite": False,
        "resume": False,
        "dry_run": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _config() -> dict:
    return {
        "mcp": {
            "environment": {
                "HAB_CARTOGRAPHER_ENABLED": "1",
                "HAB_STRUCTURED_SPATIAL_MEMORY_ENABLED": "1",
            }
        },
        "arms": {
            "default": {
                "tool_whitelist": [
                    "hab_list_nodes",
                    "hab_report_memory_pos",
                    "hab_close_session",
                ]
            }
        },
    }


def _memory_document(name: str = "explored") -> dict:
    return {
        "version": 7,
        "next_order_id": 2,
        "nodes": {
            "area-1": {
                "id": "area-1",
                "order_id": 1,
                "kind": "area",
                "name": name,
                "description": "source memory",
                "records": [],
                "pano_images": {},
                "slam_pose": {
                    "sequence": 1,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                },
            }
        },
    }


def test_pair_snapshots_dual_slam_and_uses_writable_retrieval_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run_one(namespace, *, runtime_overrides):
        calls.append((namespace, runtime_overrides))
        environment = runtime_overrides.process_environment
        memory = Path(environment["SUPERNAV_MEMORY_NODES_PATH"])
        memory.parent.mkdir(parents=True, exist_ok=True)
        if namespace.run_id.endswith("__explore"):
            memory.write_text(json.dumps(_memory_document()) + "\n")
            session_id = "exploration-session"
            root = Path(environment["HAB_CARTOGRAPHER_OUTPUT_DIR"]) / session_id
            for stream in ("visual", "planning"):
                stream_dir = root / stream
                stream_dir.mkdir(parents=True, exist_ok=True)
                (stream_dir / "occupancy.png.pbstream").write_bytes(
                    f"{stream}-state".encode()
                )
            visual = root / "visual"
            (visual / "occupancy_step1.png").write_bytes(b"rendered-map")
            (visual / "occupancy_step1.amcl.pgm").write_bytes(b"P5\n1 1\n255\n\xff")
            (visual / "occupancy_step1.png.nodes.json").write_text(
                '{"sequence":1,"nodes":[]}'
            )
            (visual / "occupancy_step1.png.render.json").write_text(
                '{"resolution":0.05}'
            )
            (visual / "slam_area_keyframes.jsonl").write_text(
                '{"keyframe_id":"k1"}\n'
            )
            (visual / "slam_area_caption_results.jsonl").write_text(
                '{"status":"ready","keyframe_id":"k1"}\n'
            )
            (visual / "occupancy.png.poses.jsonl").write_text(
                '{"sequence":1}\n'
            )
        else:
            session_id = "retrieval-session"
            assert environment["HAB_CARTOGRAPHER_LOCALIZATION_MODE"] == (
                "delayed_fpfh_relocalization"
            )
            assert environment["HAB_CARTOGRAPHER_RELOCALIZER_COMMAND"].endswith(
                "/memory_slam/relocalize.py"
            )
            assert Path(environment["HAB_CARTOGRAPHER_LOAD_STATE"]).is_file()
            assert Path(
                environment["HAB_CARTOGRAPHER_PLANNING_LOAD_STATE"]
            ).is_file()
            prior_static = Path(
                environment["HAB_CARTOGRAPHER_PRIOR_VISUAL_ARTIFACT_DIR"]
            )
            assert (prior_static / "occupancy_step0.png").read_bytes() == b"rendered-map"
            assert (prior_static / "occupancy_step0.amcl.pgm").is_file()
            assert (prior_static / "occupancy_step0.png.nodes.json").is_file()
            assert (prior_static / "occupancy_step0.png.render.json").is_file()
            assert runtime_overrides.init_defaults[
                "_slam_area_caption_sources"
            ][0]["evidence_origin"] == "exploration"
            document = json.loads(memory.read_text())
            document["nodes"]["area-1"]["name"] = "retrieval-mutated"
            memory.write_text(json.dumps(document) + "\n")
        return {
            "run_id": namespace.run_id,
            "run_dir": str(tmp_path / namespace.run_id),
            "metrics": {
                "returncode": 0,
                "process_completed": True,
                "session_closed": True,
                "session_id": session_id,
            },
        }

    monkeypatch.setattr(paired, "run_one", fake_run_one)
    row = {
        "task_id": "task-1",
        "slug": "task-1",
        "text": "find it",
        "spawn": {"x": 0},
        "retrieval_spawn": {"x": 3},
    }
    result = paired._run_pair(
        _args(tmp_path),
        config=_config(),
        backend=SimpleNamespace(name="habitat"),
        row=row,
        arm="default",
        rep=None,
        out_root=tmp_path,
        snapshot_root=None,
        visibility_cache_dir=None,
        sweep_id="test-sweep",
        row_count=1,
    )

    assert result["status"] == "completed"
    assert len(calls) == 2
    assert calls[0][0].spawn == {"x": 0}
    assert calls[1][0].spawn == {"x": 3}
    pair_dir = tmp_path / "_pairs" / "default_task-1"
    snapshot = pair_dir / "snapshot" / "memory_nodes.json"
    retrieval = pair_dir / "retrieval" / "memory_nodes.json"
    assert snapshot != retrieval
    assert json.loads(snapshot.read_text())["nodes"]["area-1"]["name"] == "explored"
    assert json.loads(retrieval.read_text())["nodes"]["area-1"]["name"] == (
        "retrieval-mutated"
    )
    assert paired._verify_snapshot(pair_dir)[0]["schema"] == (
        "supernav/paired-session-snapshot/v1"
    )


def test_pair_requires_both_cartographer_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_one(namespace, *, runtime_overrides):
        memory = Path(
            runtime_overrides.process_environment["SUPERNAV_MEMORY_NODES_PATH"]
        )
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text(json.dumps(_memory_document()) + "\n")
        root = (
            Path(runtime_overrides.process_environment["HAB_CARTOGRAPHER_OUTPUT_DIR"])
            / "session"
            / "visual"
        )
        root.mkdir(parents=True, exist_ok=True)
        (root / "occupancy.png.pbstream").write_bytes(b"visual")
        return {
            "run_id": namespace.run_id,
            "metrics": {
                "returncode": 0,
                "process_completed": True,
                "session_closed": True,
                "session_id": "session",
            },
        }

    monkeypatch.setattr(paired, "run_one", fake_run_one)
    with pytest.raises(RuntimeError, match="planning Cartographer state"):
        paired._run_pair(
            _args(tmp_path),
            config=_config(),
            backend=SimpleNamespace(name="habitat"),
            row={
                "task_id": "task-1",
                "slug": "task-1",
                "text": "find it",
                "spawn": {},
            },
            arm="default",
            rep=None,
            out_root=tmp_path,
            snapshot_root=None,
            visibility_cache_dir=None,
            sweep_id="test-sweep",
            row_count=1,
        )


def test_snapshot_hash_mismatch_fails_resume(tmp_path: Path) -> None:
    pair_dir = tmp_path / "pair"
    snapshot = pair_dir / "snapshot"
    snapshot.mkdir(parents=True)
    memory = snapshot / "memory_nodes.json"
    memory.write_text(json.dumps(_memory_document()) + "\n")
    manifest = {
        "schema": "supernav/paired-session-snapshot/v1",
        "source_session_id": "session",
        "caption_inheritance": {"status": "no_keyframes"},
        "artifacts": [
            {
                "role": "memory_nodes",
                "path": "memory_nodes.json",
                "bytes": memory.stat().st_size,
                "sha256": "wrong",
            }
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="hash check failed"):
        paired._verify_snapshot(pair_dir)
