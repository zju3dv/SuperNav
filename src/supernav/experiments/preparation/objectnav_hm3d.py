#!/usr/bin/env python3
"""Build harness instructions + config for HM3D ObjectNav (v0.2) episodes.

Reads the official ``objectnav_hm3d_v2`` episode dataset (per-scene
``content/*.json.gz`` files with top-level ``goals_by_category``) and emits:

1. An HQ100-style instructions file (one row per episode) that
   ``supernav run-one`` / ``supernav run`` consume directly.
2. A harness config cloning the proven localnav arm set, with the HM3D
   scene dataset config and ``HAB_DEFAULT_AGENT_NAVMESH=0`` so the bridge
   keeps the official ``.basis.navmesh`` (episode geodesic distances were
   computed against it; a recompute with default agent settings would
   override it via the Simulator's navmesh_settings path).

Scene naming: rows carry the bare scene hash (e.g. ``4ok3usBNeis``), not the
episode's verbatim ``scene_id`` (``hm3d_v0.2/val/00877-4ok3usBNeis/...``).
The bridge resolves a bare name through habitat-sim's scene-instance
substring lookup, which lands on ``val/<id>/<name>.basis.scene_instance.json``
with its navmesh and semantic handles; the verbatim episode scene_id matches
no registered handle and would fall back to a bare-stage load without the
official navmesh.

Usage:

    python -m supernav prepare objectnav \
      --episodes-root /path/to/objectnav_hm3d_v2 \
      --split val \
      --scene-dataset-config /path/to/hm3d_annotated_basis.scene_dataset_config.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

from supernav.paths import asset_root, workspace_root
from supernav.experiments.config import load_experiment_config
from supernav.runtime.config import resolve_prompts_dir

REPO = asset_root()

DEFAULT_EPISODES_ROOT = os.environ.get("SUPERNAV_HM3D_EPISODES_ROOT")
DEFAULT_SCENE_DATASET_CONFIG = os.environ.get("SUPERNAV_SCENE_DATASET_CONFIG")
DEFAULT_OUT_DIR = workspace_root() / "configs" / "benchmarks" / "objectnav_hm3d"

# localnav policy checkpoint native camera height; the official HM3D v1
# protocol used 0.88 m, so report comparisons must disclose this deviation.
DEFAULT_SENSOR_HEIGHT = 1.25


def scene_hash_from_episode(scene_id: str) -> str:
    """``hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb`` -> ``4ok3usBNeis``."""
    base = scene_id.rstrip("/").rsplit("/", 1)[-1]
    for suffix in (".basis.glb", ".glb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def split_from_episode(scene_id: str, fallback: str) -> str:
    parts = scene_id.split("/")
    return parts[1] if len(parts) >= 3 else fallback


def load_scene_episodes(content_path: Path) -> dict:
    with gzip.open(content_path, "rt") as f:
        return json.load(f)


def build_rows(
    episodes_root: Path,
    split: str,
    scene_dataset_config: str,
    sensor_height: float,
    scenes: set[str] | None = None,
    max_episodes: int | None = None,
) -> list[dict]:
    content_dir = episodes_root / split / "content"
    if not content_dir.is_dir():
        raise SystemExit(f"episode content dir not found: {content_dir}")

    rows: list[dict] = []
    for content_path in sorted(content_dir.glob("*.json.gz")):
        data = load_scene_episodes(content_path)
        goals_by_category = data.get("goals_by_category") or {}
        for ep in data.get("episodes", []):
            scene_id = ep["scene_id"]
            scene_hash = scene_hash_from_episode(scene_id)
            if scenes and scene_hash not in scenes:
                continue
            category = ep["object_category"]
            goal_key = f"{scene_hash}.basis.glb_{category}"
            goals = goals_by_category.get(goal_key) or []
            ep_split = split_from_episode(scene_id, split)
            ep_id = str(ep["episode_id"])
            # Harness row_task_id slugifies (lowercases) task ids for selection
            # and run naming; emit them pre-normalized so manifest ids match
            # run artifacts verbatim. The scene hash itself stays mixed-case —
            # scene-instance handle lookup is case-sensitive.
            rows.append(
                {
                    "task_id": f"objectnav_{category}_{scene_hash}_{ep_id}".lower(),
                    "slug": f"{category}-{scene_hash}-{ep_id}".lower(),
                    "text": f"Find and approach a {category.replace('_', ' ')}.",
                    "scene": scene_hash,
                    "scene_dataset_config_file": scene_dataset_config,
                    "spawn": {
                        "start_position": [float(v) for v in ep["start_position"]],
                        "start_rotation": [float(v) for v in ep["start_rotation"]],
                        "sensor_height": sensor_height,
                    },
                    "object_category": category,
                    "source_episode": str(content_path),
                    "source_episode_id": ep_id,
                    "goal_key": goal_key,
                    "ground_truth": {
                        "task_type": "objectnav",
                        "source_episode": str(content_path),
                        "source_episode_id": ep_id,
                        "goal_key": goal_key,
                        "object_category": category,
                        "initial_geodesic_distance_m": float(
                            ep["info"]["geodesic_distance"]
                        ),
                        "goal_instance_count": len(goals),
                        "goal_viewpoint_count": sum(
                            len(g.get("view_points") or []) for g in goals
                        ),
                        "hm3d_scene_id": scene_id,
                        "hm3d_split": ep_split,
                        "closest_goal_object_id": (ep.get("info") or {}).get(
                            "closest_goal_object_id"
                        ),
                    },
                }
            )

    # Group consecutive episodes of the same scene to avoid redundant scene
    # reloads on the bridge (same ordering convention as common/hq100.py).
    rows.sort(key=lambda r: (r["scene"], r["task_id"]))
    if max_episodes is not None:
        rows = rows[:max_episodes]
    return rows


def write_instructions(rows: list[dict], out_path: Path, episodes_root: Path, split: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(episodes_root / split / "content"),
        "selection": f"all official objectnav_hm3d_v2 {split} episodes",
        "instructions": rows,
    }
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def write_config(
    template_path: Path,
    out_path: Path,
    instructions_path: Path,
    scene_dataset_config: str,
    output_dir: str,
) -> None:
    config = load_experiment_config(template_path)
    resolve_prompts_dir(config, workspace_root=workspace_root())
    config["scene"] = "4ok3usBNeis"  # placeholder; manifest rows override per task
    config["scene_dataset_config_file"] = scene_dataset_config
    config["output_dir"] = output_dir
    config["instructions_file"] = str(instructions_path)
    # Multi-hop searches can exceed the 1800 s template default; a timeout kill
    # loses all episode artifacts, so give benchmark episodes generous headroom.
    # The 30 s MCP startup budget is also too tight on cold NFS python imports —
    # episodes die in the codex session handshake before turn 1, so widen it.
    for agent_cfg in (config.get("agents") or {}).values():
        if isinstance(agent_cfg, dict):
            if "timeout_s" in agent_cfg:
                agent_cfg["timeout_s"] = 3600
            if "mcp_startup_timeout_s" in agent_cfg:
                agent_cfg["mcp_startup_timeout_s"] = 180
    # Keep the dataset's official navmesh: default recompute would override it.
    mcp_env = config.setdefault("mcp", {}).setdefault("environment", {})
    mcp_env["HAB_DEFAULT_AGENT_NAVMESH"] = "0"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT, required=not DEFAULT_EPISODES_ROOT)
    parser.add_argument("--split", default="val", help="val | val_mini | train")
    parser.add_argument(
        "--scene-dataset-config", default=DEFAULT_SCENE_DATASET_CONFIG,
        required=not DEFAULT_SCENE_DATASET_CONFIG
    )
    parser.add_argument("--sensor-height", type=float, default=DEFAULT_SENSOR_HEIGHT)
    parser.add_argument("--template-config", type=Path, help="Canonical experiment recipe; required unless --no-config.")
    parser.add_argument("--config-dir", type=Path, default=workspace_root() / "configs" / "experiments")
    parser.add_argument("--instructions-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scenes", default=None, help="comma-separated scene hash subset"
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--no-config", action="store_true", help="only write instructions"
    )
    args = parser.parse_args()
    if not args.no_config and args.template_config is None:
        parser.error("--template-config is required unless --no-config is selected")

    scenes = {s for s in (args.scenes or "").split(",") if s} or None
    rows = build_rows(
        args.episodes_root,
        args.split,
        args.scene_dataset_config,
        args.sensor_height,
        scenes=scenes,
        max_episodes=args.max_episodes,
    )
    if not rows:
        raise SystemExit("no episodes matched")

    instructions_path = args.instructions_dir / f"hm3d_v2_{args.split}.instructions.json"
    write_instructions(rows, instructions_path, args.episodes_root, args.split)

    config_path = None
    if not args.no_config:
        config_path = args.config_dir / f"hm3d-{args.split}-localnav.json"
        write_config(
            args.template_config,
            config_path,
            instructions_path,
            args.scene_dataset_config,
            output_dir=f"data/runs/objectnav_hm3d_v2/{args.split}",
        )

    cats: dict[str, int] = {}
    scene_set = set()
    for r in rows:
        cats[r["object_category"]] = cats.get(r["object_category"], 0) + 1
        scene_set.add(r["scene"])
    print(f"episodes: {len(rows)}  scenes: {len(scene_set)}  categories: {cats}")
    print(f"instructions: {instructions_path}")
    if config_path:
        print(f"config:       {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
