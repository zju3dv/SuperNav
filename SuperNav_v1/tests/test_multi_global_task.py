"""Tests for the multi-object-nav bench pipeline.

Covers the multi goal env/instruction composition in run_one, the
multi_global_task config builder, and its prompts and native skill.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.experiments.preparation.multi_global_tasks import build_configs
from supernav.runtime.skill_runtime import lint_skill_root  # noqa: E402
from supernav.backends.habitat.experiment import _compose_multi_goal_instruction, _multi_goal_env_payload

# All scenes, object names, coordinates and trajectories below are invented.
_ROW = {
    "ordered": True,
    "targets": [
        {
            "index": 1,
            "instruction": "find the red test cube",
            "position": [3.0, 0.0, 0.0],
            "radius_m": 1.0,
            "target_object": {
                "bbox_center": [3.0, 0.0, 0.0],
                "bbox_size": [1, 1, 2],
            },
        },
        {
            "index": 2,
            "instruction": "find the blue test sphere",
            "position": [6.0, 0.0, 0.0],
            "radius_m": 1.0,
            "target_object": {"bbox_center": [7.0, 0.0, 0.0], "bbox_size": [1, 1, 1]},
        },
    ],
}


def test_multi_goal_env_payload_carries_no_coordinates() -> None:
    payload = _multi_goal_env_payload(_ROW)
    assert payload is not None
    assert payload["ordered"] is True
    assert payload["targets"] == [
        {"index": 1, "description": "find the red test cube"},
        {"index": 2, "description": "find the blue test sphere"},
    ]
    assert "position" not in json.dumps(payload)
    assert "bbox" not in json.dumps(payload)


def test_multi_goal_env_payload_none_for_single_goal_rows() -> None:
    assert _multi_goal_env_payload({"goal": {"position": [0, 0, 0]}}) is None
    assert _multi_goal_env_payload({"targets": []}) is None
    assert _multi_goal_env_payload({"targets": [{"instruction": "x"}]}) is None


def test_compose_multi_goal_instruction_appends_numbered_list() -> None:
    text = _compose_multi_goal_instruction(
        "First, find the red test cube. Then, find the blue test sphere.", _ROW
    )
    assert text.startswith("First, find the red test cube.")
    assert "ORDERED GOALS (2 total)" in text
    assert "[1] find the red test cube" in text
    assert "[2] find the blue test sphere" in text
    assert 'hab_nav_goals(action="mark", target_index=k)' in text
    assert 'hab_nav_goals(action="status")' in text
    assert "cannot be revoked" in text


def test_compose_multi_goal_instruction_unordered_wording() -> None:
    row = dict(_ROW, ordered=False)
    text = _compose_multi_goal_instruction("Find both objects.", row)
    assert "GOALS (2 total). Find all of them, in any order." in text
    assert "ORDERED" not in text


def test_compose_instruction_passthrough_for_single_goal() -> None:
    assert (
        _compose_multi_goal_instruction("find the sofa", {"goal": {}})
        == "find the sofa"
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _episode_file(tmp_path: Path) -> Path:
    episode = {
        "episode_id": "synthetic_multi_001",
        "scene_id": "/datasets/scenes/0000_000001/0000_000001.gs.ply",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "task": {
            "instruction": "First, find the red test cube. Then, find the blue test sphere.",
            "ordered": True,
            "target_count": 2,
            "targets": [
                {
                    "index": 0,
                    "instruction": "find the red test cube",
                    "target_id": "synthetic_red_cube",
                    "object": {
                        "primary_bbox": {
                            "center": [3.0, 0.0, 0.0],
                            "size": [1.0, 1.0, 1.0],
                        }
                    },
                },
                {
                    "index": 1,
                    "instruction": "find the blue test sphere",
                    "target_id": "synthetic_blue_sphere",
                    "object": {
                        "primary_bbox": {
                            "center": [7.0, 0.0, 0.0],
                            "size": [0.5, 0.5, 0.5],
                        }
                    },
                },
            ],
        },
        "evaluation": {
            "success": {
                "ordered": True,
                "targets": [
                    {
                        "target_index": 0,
                        "target_id": "synthetic_red_cube",
                        "position": [2.0, 0.0, 0.0],
                        "radius": 1.0,
                    },
                    {
                        "target_index": 1,
                        "target_id": "synthetic_blue_sphere",
                        "position": [6.0, 0.0, 0.0],
                        "radius": 1.0,
                    },
                ],
            }
        },
        "ground_truth": {
            "geodesic_distance": 6.0,
            "path": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "segments": [
                {
                    "target_index": 0,
                    "endpoint_position": [2.0, 0.0, 0.0],
                    "object_position": [3.0, 0.0, 0.0],
                    "geodesic_distance": 2.0,
                },
                {
                    "target_index": 1,
                    "endpoint_position": [6.0, 0.0, 0.0],
                    "object_position": [7.0, 0.0, 0.0],
                    "geodesic_distance": 4.0,
                },
            ],
        },
        "metadata": {
            "assets": {"navmesh": "/other/people/0000_000001.navmesh"},
        },
    }
    path = tmp_path / "interior_0000_000001.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"episodes": [episode], "metadata": {}}, fh)
    return path


def test_build_multi_configs(tmp_path: Path) -> None:
    episodes_dir = tmp_path / "content"
    episodes_dir.mkdir()
    _episode_file(episodes_dir)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"navmesh_instances": {"interior_0000_000001": {}}}))
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "model": {"name": "custom-navigation-model"},
                "agents": {
                    "codex": {"extra_args": ["-c", 'model_reasoning_effort="low"']}
                },
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/multi_global_task/prompts"),
                "arms": {"default": {
                    "movement": "localnav", "environment": {"NAV_LOCALNAV_EXEC_ACTIONS": "2"},
                    "skill_mode": "native", "skill_name": "global-navigation-learned-executor",
                    "skill_names": ["global-navigation-learned-executor", "localnav-pointnav"],
                    "tool_whitelist": ["hab_init_scene", "hab_local_navigate", "hab_nav_goals", "hab_close_session"],
                }},
            }
        )
    )
    out_dir = tmp_path / "out"

    written = build_configs(
        episodes_dir=episodes_dir,
        dataset_config=dataset,
        base_config_path=base_config,
        output_dir=out_dir,
        instructions_dir=tmp_path / "instructions",
    )
    assert len(written) == 2

    manifest = json.loads(
        (tmp_path / "instructions/interior_0000_000001.instructions.json").read_text()
    )
    row = manifest["instructions"][0]
    assert row["task_id"] == "synthetic_multi_001"
    assert row["ordered"] is True
    assert row["spawn"]["start_rotation"] == [0.0, 0.0, 0.0, 1.0]
    assert [t["index"] for t in row["targets"]] == [1, 2]
    first = row["targets"][0]
    assert first["source_target_index"] == 0
    assert first["radius_m"] == 1.0
    assert first["position"] == [2.0, 0.0, 0.0]
    assert first["target_object"]["bbox_center"] == [3.0, 0.0, 0.0]
    assert row["ground_truth"]["segments"][1]["target_index"] == 2
    # External absolute paths from the episode metadata must not leak through.
    assert "/other/people" not in json.dumps(manifest)

    config = json.loads((out_dir / "multi-global-task-interior_0000_000001.json").read_text())
    assert config["model"] == {"name": "custom-navigation-model"}
    assert config["agents"]["codex"]["extra_args"] == [
        "-c", 'model_reasoning_effort="low"'
    ]
    assert set(config["arms"]) == {"default"}
    arm = config["arms"]["default"]
    assert arm["skill_name"] == "multi-global-navigation-learned-executor"
    assert arm["skill_mode"] == "native"
    assert arm["skill_names"] == ["multi-global-navigation-learned-executor", "localnav-pointnav"]
    assert arm["movement"] == "localnav"
    assert arm["environment"] == {"NAV_LOCALNAV_EXEC_ACTIONS": "2"}
    assert arm["tool_whitelist"].count("hab_nav_goals") == 1
    assert "hab_local_navigate" in arm["tool_whitelist"]
    assert config["benchmark_profile"] == "global_task"
    assert config["instructions_file"] == str(tmp_path / "instructions/interior_0000_000001.instructions.json")
    assert not list(out_dir.glob("*.instructions.json"))


def test_build_multi_configs_rejects_unknown_scene(tmp_path: Path) -> None:
    episodes_dir = tmp_path / "content"
    episodes_dir.mkdir()
    _episode_file(episodes_dir)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"navmesh_instances": {}}))
    base_config = tmp_path / "base.json"
    base_config.write_text(json.dumps({"prompts_dir": str(REPO_ROOT / "configs/benchmarks/multi_global_task/prompts"), "arms": {"default": {"movement": "localnav"}}}))
    try:
        build_configs(
            episodes_dir=episodes_dir,
            dataset_config=dataset,
            base_config_path=base_config,
            output_dir=tmp_path / "out",
            instructions_dir=tmp_path / "instructions",
        )
    except ValueError as exc:
        assert "absent from" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown scene")


@pytest.mark.parametrize("movement,skill_mode,skill_name", [
    ("visual_point", "native", "global-navigation-geo-based-executor"),
    ("primitive", "none", None),
])
def test_build_multi_configs_rejects_incompatible_executor(tmp_path, movement, skill_mode, skill_name):
    episodes = tmp_path / "content"
    episodes.mkdir()
    _episode_file(episodes)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"navmesh_instances": {"interior_0000_000001": {}}}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "prompts_dir": str(REPO_ROOT / "configs/benchmarks/multi_global_task/prompts"),
        "arms": {"default": {
            "movement": movement, "skill_mode": skill_mode, "skill_name": skill_name,
        }},
    }))
    with pytest.raises(ValueError, match="habitat-learned-executor"):
        build_configs(
            episodes_dir=episodes, dataset_config=dataset, base_config_path=base,
            output_dir=tmp_path / "out", instructions_dir=tmp_path / "instructions",
        )
    assert not list(tmp_path.glob("out/*.json"))
    assert not list(tmp_path.glob("instructions/*.json"))


# ---------------------------------------------------------------------------
# Arm assets
# ---------------------------------------------------------------------------


def test_multi_skill_lints_and_is_present() -> None:
    report = lint_skill_root(REPO_ROOT / "skills")
    assert report.ok, report.errors
    names = [skill.name for skill in report.skills]
    assert "multi-global-navigation-learned-executor" in names


def test_multi_prompts_exist_and_describe_mark_protocol() -> None:
    prompts = REPO_ROOT / "configs" / "benchmarks" / "multi_global_task" / "prompts"
    common = (prompts / "common_localnav.md").read_text(encoding="utf-8")
    end = (prompts / "end.md").read_text(encoding="utf-8")
    assert "{instruction}" in common
    assert "{scene}" in common
    assert "hab_nav_goals" in common
    assert "found/pending" in common
    assert "hab_nav_goals" in end
    assert "closed=true" in end


# ---------------------------------------------------------------------------
# Offline scorer
# ---------------------------------------------------------------------------


def _write_score_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    visuals = tmp_path / "visuals"
    visuals.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "task_id": "synthetic_multi_001",
                "scene": "interior_0000_000001",
                "ordered": True,
                "targets": [
                    {
                        "index": 1,
                        "target_id": "synthetic_red_cube",
                        "instruction": "find the red test cube",
                        "position": [2.0, 0.0, 0.0],
                        "radius_m": 1.0,
                        "target_object": {
                            "bbox_center": [3.0, 0.0, 0.0],
                            "bbox_size": [1.0, 1.0, 1.0],
                        },
                    },
                    {
                        "index": 2,
                        "target_id": "synthetic_blue_sphere",
                        "instruction": "find the blue test sphere",
                        "position": [6.0, 0.0, 0.0],
                        "radius_m": 1.0,
                        "target_object": {
                            "bbox_center": [7.0, 0.0, 0.0],
                            "bbox_size": [0.5, 0.5, 0.5],
                        },
                    },
                ],
                "ground_truth": {"geodesic_distance_m": 6.0, "segments": []},
            }
        )
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"session_id": "s1", "success": True})
    )
    sidecar = {
        "goal_marks": [
            {
                "target_index": 1,
                "ts": "2026-08-24T01:00:00Z",
                "position": [2.5, 0.0, 0.0],
            },
            {
                "target_index": 2,
                "ts": "2026-08-24T01:05:00Z",
                "position": [6.5, 0.0, 0.0],
            },
        ],
        "trajectory": [
            {"pose": {"position": [0.0, 0.0, 0.0]}},
            {"pose": {"position": [2.5, 0.0, 0.0]}},
        ],
        "dense_trajectory": {
            "points": [[0.0, 0.0, 0.0]] + [[0.1 * i, 0.0, 0.0] for i in range(100)]
        },
        "events": [],
    }
    (visuals / "s1.trajectory.json").write_text(json.dumps(sidecar))
    return run_dir, visuals


def test_score_run_hits_and_dense_path_length(tmp_path: Path) -> None:
    from supernav.evaluation.objectnav.score_multi import score_run

    run_dir, visuals = _write_score_run(tmp_path)
    record = score_run(run_dir, visuals)
    assert record is not None
    assert "error" not in record
    assert record["found"] == 2
    assert record["hits"] == 2
    assert record["order_ok"] is True
    assert record["success"] is True
    assert record["claim_success"] is True
    # Path length comes from the 101-point dense trajectory, not the
    # 2-point sparse fallback (2.5 m sparse vs 9.9 m dense here).
    assert record["path_length_m"] > 9.0
    assert record["spl"] is not None and record["spl"] <= 1.0


def test_score_run_missing_sidecar_is_error_record(tmp_path: Path) -> None:
    from supernav.evaluation.objectnav.score_multi import score_run

    run_dir, _ = _write_score_run(tmp_path)
    record = score_run(run_dir, tmp_path / "empty_visuals")
    assert record is not None
    assert record["error"] == "trajectory sidecar not found"
