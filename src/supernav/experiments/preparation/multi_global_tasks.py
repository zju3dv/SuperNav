#!/usr/bin/env python3
"""Build multi-object-nav benchmark configs and instruction manifests.

Reads the multi_objectnav_global_task episode dataset (one gzip'd JSON per
scene, schema ``multi_objectnav_global_task/v2``) and writes one config +
one instruction manifest per scene under ``configs/benchmarks/multi_global_task/``.

Goal indexes are 1-based throughout the manifest (matching the prompt's
numbered list and the ``hab_nav_goals`` tool's ``target_index``); the
source dataset's 0-based ``target_index`` is preserved as
``source_target_index``. External absolute paths from the episode
metadata (navmesh/labels/assets) are never propagated.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

from supernav.experiments.config import load_experiment_config
from supernav.runtime.config import resolve_prompts_dir
from supernav.paths import workspace_root

DEFAULT_EPISODES_DIR = os.environ.get("SUPERNAV_MULTI_OBJECTNAV_EPISODES_DIR")
DEFAULT_DATASET_CONFIG = os.environ.get("SUPERNAV_SCENE_DATASET_CONFIG")
_SCENE_ID_PATTERN = re.compile(r"(\d{4}_\d{6})")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"episode file root must be an object: {path}")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"no episodes found in {path}")
    return [ep for ep in episodes if isinstance(ep, dict)]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _scene_id_from_episode(episode: dict[str, Any], source: Path) -> str:
    match = _SCENE_ID_PATTERN.search(str(episode.get("scene_id") or ""))
    if not match:
        raise ValueError(
            f"cannot parse scene_id from {source}: {episode.get('scene_id')!r}"
        )
    return f"interior_{match.group(1)}"


def _float_vec(value: Any, length: int, what: str, episode_id: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"invalid {what} for {episode_id}: {value!r}")
    return [float(v) for v in value]


def _target_rows(episode: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    task = episode.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"episode {episode_id} has no task object")
    task_targets = task.get("targets")
    if not isinstance(task_targets, list) or not task_targets:
        raise ValueError(f"episode {episode_id} has no task.targets")
    eval_targets = (
        episode.get("evaluation", {}).get("success", {}).get("targets")
        if isinstance(episode.get("evaluation"), dict)
        else None
    )
    if not isinstance(eval_targets, list) or len(eval_targets) != len(task_targets):
        raise ValueError(
            f"episode {episode_id}: evaluation.success.targets must align with task.targets"
        )
    eval_by_index = {
        int(entry.get("target_index")): entry
        for entry in eval_targets
        if isinstance(entry, dict)
    }
    rows: list[dict[str, Any]] = []
    for position_0based, target in enumerate(task_targets):
        if not isinstance(target, dict):
            raise ValueError(f"episode {episode_id}: malformed target entry")
        source_index = int(target.get("index", position_0based))
        eval_entry = eval_by_index.get(source_index)
        if eval_entry is None:
            raise ValueError(
                f"episode {episode_id}: no evaluation target for index {source_index}"
            )
        instruction = str(target.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(
                f"episode {episode_id}: target {source_index} has no instruction"
            )
        primary_bbox = (target.get("object") or {}).get("primary_bbox") or {}
        bbox_center = primary_bbox.get("center")
        bbox_size = primary_bbox.get("size")
        target_object = None
        if bbox_center is not None and bbox_size is not None:
            target_object = {
                "bbox_center": _float_vec(bbox_center, 3, "bbox_center", episode_id),
                "bbox_size": _float_vec(bbox_size, 3, "bbox_size", episode_id),
            }
        rows.append(
            {
                "index": source_index + 1,
                "source_target_index": source_index,
                "target_id": target.get("target_id"),
                "instruction": instruction,
                "position": _float_vec(
                    eval_entry.get("position"), 3, "target viewpoint", episode_id
                ),
                "radius_m": float(eval_entry.get("radius", 1.0)),
                "target_object": target_object,
            }
        )
    rows.sort(key=lambda row: row["index"])
    return rows


def _instruction_row(
    episode: dict[str, Any], scene_id: str, source: Path
) -> dict[str, Any]:
    episode_id = str(episode.get("episode_id") or "").strip()
    task = episode.get("task") if isinstance(episode.get("task"), dict) else {}
    instruction = str(task.get("instruction") or "").strip()
    if not episode_id or not instruction:
        raise ValueError(f"episode_id and task.instruction are required in {source}")
    ground_truth = (
        episode.get("ground_truth")
        if isinstance(episode.get("ground_truth"), dict)
        else {}
    )
    segments = [
        {
            "target_index": int(segment.get("target_index")) + 1,
            "endpoint_position": segment.get("endpoint_position"),
            "object_position": segment.get("object_position"),
            "geodesic_distance_m": segment.get("geodesic_distance"),
        }
        for segment in ground_truth.get("segments", [])
        if isinstance(segment, dict)
    ]
    return {
        "task_id": episode_id,
        "slug": episode_id,
        "text": instruction,
        "ordered": bool(task.get("ordered", True)),
        "spawn": {
            "start_position": _float_vec(
                episode.get("start_position"), 3, "start_position", episode_id
            ),
            "start_rotation": _float_vec(
                episode.get("start_rotation"), 4, "start_rotation", episode_id
            ),
            # Native camera height for the learned navigation policy.
            "sensor_height": 1.25,
        },
        "targets": _target_rows(episode, episode_id),
        "ground_truth": {
            "geodesic_distance_m": ground_truth.get("geodesic_distance"),
            "path_points": copy.deepcopy(ground_truth.get("path", [])),
            "segments": segments,
            "source_episodes_file": source.name,
            "source_episode_id": episode_id,
        },
    }


def build_configs(
    *,
    episodes_dir: Path,
    dataset_config: Path,
    base_config_path: Path,
    output_dir: Path,
    instructions_dir: Path,
) -> list[Path]:
    base_config = load_experiment_config(base_config_path)
    resolve_prompts_dir(base_config, workspace_root=workspace_root())
    dataset = _load_json(dataset_config)
    known_scenes = dataset.get("navmesh_instances", {})
    written: list[Path] = []
    for episodes_path in sorted(episodes_dir.glob("*.json.gz")):
        episodes = _load_episodes(episodes_path)
        scene_ids = {_scene_id_from_episode(ep, episodes_path) for ep in episodes}
        if len(scene_ids) != 1:
            raise ValueError(f"mixed scenes in {episodes_path}: {sorted(scene_ids)}")
        scene_id = scene_ids.pop()
        if scene_id not in known_scenes:
            raise ValueError(f"scene {scene_id!r} is absent from {dataset_config}")

        manifest_name = f"{scene_id}.instructions.json"
        config_name = f"multi-global-task-{scene_id}.json"
        manifest_path = instructions_dir / manifest_name
        config_path = output_dir / config_name
        manifest = {
            "source": str(episodes_path),
            "instructions": [
                _instruction_row(episode, scene_id, episodes_path)
                for episode in episodes
            ],
        }
        config = copy.deepcopy(base_config)
        arms = config.get("arms")
        if not isinstance(arms, dict):
            raise ValueError(f"base config has no arms object: {base_config_path}")
        default_arm = arms.get("default")
        if not isinstance(default_arm, dict):
            raise ValueError("base config has no default arm")
        default_arm = copy.deepcopy(default_arm)
        skill_name = "global-navigation-learned-executor"
        multi_skill_name = "multi-global-navigation-learned-executor"
        if (
            default_arm.get("movement") != "localnav"
            or default_arm.get("skill_mode") != "native"
            or default_arm.get("skill_name") != skill_name
        ):
            raise ValueError(
                "multi-goal generation requires the learned executor; "
                "select habitat-learned-executor as the base config"
            )
        default_arm["skill_name"] = multi_skill_name
        skill_names = default_arm.get("skill_names", [skill_name])
        if not isinstance(skill_names, list):
            raise ValueError("default.skill_names must be a list")
        default_arm["skill_names"] = [
            multi_skill_name if name == skill_name else name
            for name in skill_names
        ]
        for required_skill in (multi_skill_name, "localnav-pointnav"):
            if required_skill not in default_arm["skill_names"]:
                default_arm["skill_names"].append(required_skill)
        tool_whitelist = default_arm.get("tool_whitelist", [])
        if not isinstance(tool_whitelist, list):
            raise ValueError("default.tool_whitelist must be a list")
        default_arm["tool_whitelist"] = list(tool_whitelist)
        if "hab_nav_goals" not in default_arm["tool_whitelist"]:
            default_arm["tool_whitelist"].append("hab_nav_goals")
        config["arms"] = {"default": default_arm}
        first_spawn = manifest["instructions"][0]["spawn"]
        config.update(
            {
                "benchmark_profile": "global_task",
                "output_dir": f"data/runs/multi_global_task/{scene_id}",
                "instructions_file": str(manifest_path.resolve()),
                "scene": scene_id,
                "scene_dataset_config_file": str(dataset_config),
                "spawn": copy.deepcopy(first_spawn),
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(config_path, config)
        written.extend((config_path, manifest_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build one benchmark config and instruction manifest per "
            "multi-object-nav scene."
        )
    )
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR, required=not DEFAULT_EPISODES_DIR)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG, required=not DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Explicit canonical experiment recipe to extend.",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=workspace_root() / "configs" / "experiments"
    )
    parser.add_argument(
        "--instructions-dir", type=Path,
        default=workspace_root() / "configs" / "benchmarks" / "multi_global_task",
    )
    args = parser.parse_args()
    written = build_configs(
        episodes_dir=args.episodes_dir.resolve(),
        dataset_config=args.dataset_config.resolve(),
        base_config_path=args.base_config.resolve(),
        output_dir=args.config_dir,
        instructions_dir=args.instructions_dir,
    )
    print(f"wrote {len(written) // 2} scene configs and manifests to {args.config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
