#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from supernav.experiments.config import load_experiment_config
from supernav.runtime.config import resolve_prompts_dir
from supernav.paths import workspace_root

DEFAULT_DATA_ROOT = os.environ.get("SUPERNAV_GLOBAL_TASK_ROOT")
DEFAULT_DATASET_CONFIG = os.environ.get("SUPERNAV_SCENE_DATASET_CONFIG")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolved_start(task: dict[str, Any], scene_data: dict[str, Any]) -> list[float]:
    start = task.get("start_position")
    if start == "default":
        start = scene_data.get("default_start_position")
    if not isinstance(start, list) or len(start) != 3:
        raise ValueError(f"invalid start_position for {task.get('task_id')}: {start!r}")
    return [float(value) for value in start]


def _instruction_row(
    task: dict[str, Any], scene_data: dict[str, Any], source: Path
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "").strip()
    instruction = str(task.get("instruction") or "").strip()
    if not task_id or not instruction:
        raise ValueError(f"task_id and instruction are required in {source}")
    goal = task.get("goal_position")
    if not isinstance(goal, list) or len(goal) != 3:
        raise ValueError(f"invalid goal_position for {task_id}: {goal!r}")
    start_yaw = float(task.get("start_yaw", scene_data.get("default_start_yaw", 0.0)))
    goal_radius = float(task.get("goal_radius", 0.25))
    return {
        "task_id": task_id,
        "slug": task_id,
        "text": instruction,
        "spawn": {
            "start_position": _resolved_start(task, scene_data),
            "yaw": start_yaw,
            "sensor_height": 1.2,
        },
        "goal": {
            "instruction": instruction,
            "position": [float(value) for value in goal],
            "radius_m": goal_radius,
            "provenance": copy.deepcopy(task.get("goal_provenance", {})),
        },
        "ground_truth": {
            "final_position": [float(value) for value in goal],
            "distance_threshold_m": goal_radius,
            # bbox-surface XZ criterion reference (common/success.goal_success);
            # distance_threshold_m doubles as the bbox-surface threshold
            "success_criterion": task.get(
                "success_criterion",
                "dist_xz_to_target_bbox_surface <= goal_radius",
            ),
            "target_object": copy.deepcopy(task.get("target_object")),
            "geodesic_distance_m": task.get("geodesic_distance_gt"),
            "path_points": copy.deepcopy(task.get("ground_truth_path", [])),
            "source_tasks_json": str(source),
            "source_task_id": task_id,
        },
    }


def build_configs(
    *,
    data_root: Path,
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
    for tasks_path in sorted(data_root.glob("*/tasks.json")):
        scene_data = _load_json(tasks_path)
        scene_id = str(scene_data.get("scene_id") or tasks_path.parent.name)
        if scene_id not in known_scenes:
            raise ValueError(f"scene {scene_id!r} is absent from {dataset_config}")
        tasks = scene_data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"no tasks found in {tasks_path}")

        manifest_name = f"{scene_id}.instructions.json"
        config_name = f"global-task-{scene_id}.json"
        manifest_path = instructions_dir / manifest_name
        config_path = output_dir / config_name
        manifest = {
            "source": str(tasks_path),
            "instructions": [
                _instruction_row(task, scene_data, tasks_path)
                for task in tasks
                if isinstance(task, dict)
            ],
        }
        config = copy.deepcopy(base_config)
        arms = config.get("arms")
        if not isinstance(arms, dict):
            raise ValueError(f"base config has no arms object: {base_config_path}")
        default_arm = arms.get("default")
        if not isinstance(default_arm, dict):
            raise ValueError("base config has no default arm")
        config["arms"] = {"default": copy.deepcopy(default_arm)}
        config.update(
            {
                "benchmark_profile": "global_task",
                "output_dir": f"data/runs/global_task/{scene_id}",
                "instructions_file": str(manifest_path.resolve()),
                "scene": scene_id,
                "scene_dataset_config_file": str(dataset_config),
                "spawn": {
                    "start_position": scene_data["default_start_position"],
                    "yaw": float(scene_data.get("default_start_yaw", 0.0)),
                    "sensor_height": 1.2,
                },
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(config_path, config)
        written.extend((config_path, manifest_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one benchmark config and instruction manifest per global-task scene."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, required=not DEFAULT_DATA_ROOT)
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
        default=workspace_root() / "configs" / "benchmarks" / "global_task",
    )
    args = parser.parse_args()
    written = build_configs(
        data_root=args.data_root.resolve(),
        dataset_config=args.dataset_config.resolve(),
        base_config_path=args.base_config.resolve(),
        output_dir=args.config_dir,
        instructions_dir=args.instructions_dir,
    )
    print(f"wrote {len(written) // 2} scene configs and manifests to {args.config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
