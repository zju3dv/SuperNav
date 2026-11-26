#!/usr/bin/env python3
"""Build harness instructions + config for the OVON val_unseen MTU3D-120 set.

Companion to ``build_objectnav_hm3d_instructions.py``: same row schema, but the
episode selection comes from MTU3D's published 120-episode manifest
(``our-set/ovon_full_set.json`` → ``val_unseen``) instead of the full official
split, so results are directly comparable to MTU3D's reported table. Each
manifest row is ``(scan_id_suffix, episode_index, object_category)`` where
``episode_index`` is the array position inside the official OVON per-scene
``content/<scene>.json.gz`` ``episodes`` list; every selected row is asserted
to match the manifest's category (the manifest was verified 120/120 against
the official files).

Run once per machine with that machine's paths, e.g. locally:

    python -m supernav prepare ovon \
      --episodes-root /path/to/hm3d_ovon/hm3d \
      --manifest /path/to/ovon_full_set.json \
      --scene-dataset-config /path/to/hm3d_annotated_basis.scene_dataset_config.json \
      --output-dir data/runs/objectnav_hm3d_ovon/val_unseen_120

"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from supernav.paths import asset_root, workspace_root

REPO = asset_root()
from supernav.experiments.preparation.objectnav_hm3d import (  # noqa: E402
    DEFAULT_SENSOR_HEIGHT,
    scene_hash_from_episode,
    split_from_episode,
    write_config,
)

DEFAULT_OUT_DIR = workspace_root() / "configs" / "benchmarks" / "objectnav_hm3d_ovon"


def load_manifest_rows(manifest_path: Path, split: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get(split)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"manifest has no rows for split {split!r}: {manifest_path}")
    return rows


def build_rows(
    episodes_root: Path,
    split: str,
    manifest_rows: list[dict],
    scene_dataset_config: str,
    sensor_height: float,
) -> list[dict]:
    content_dir = episodes_root / split / "content"
    if not content_dir.is_dir():
        raise SystemExit(f"episode content dir not found: {content_dir}")

    by_scene: dict[str, list[dict]] = {}
    for entry in manifest_rows:
        by_scene.setdefault(str(entry["scan_id_suffix"]), []).append(entry)

    rows: list[dict] = []
    mismatches: list[str] = []
    for scene_hash, entries in sorted(by_scene.items()):
        content_path = content_dir / f"{scene_hash}.json.gz"
        if not content_path.is_file():
            raise SystemExit(f"manifest scene missing from dataset: {content_path}")
        with gzip.open(content_path, "rt") as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        goals_by_category = data.get("goals_by_category") or {}
        for entry in entries:
            idx = int(entry["episode_index"])
            ep = episodes[idx]
            category = str(ep["object_category"])
            if category != str(entry["object_category"]):
                mismatches.append(
                    f"{scene_hash}[{idx}]: manifest {entry['object_category']!r} "
                    f"!= episode {category!r}"
                )
                continue
            scene_id = ep["scene_id"]
            if scene_hash_from_episode(scene_id) != scene_hash:
                raise SystemExit(f"scene hash mismatch for {scene_id}")
            goal_key = f"{scene_hash}.basis.glb_{category}"
            goals = goals_by_category.get(goal_key) or []
            ep_split = split_from_episode(scene_id, split)
            ep_id = str(ep["episode_id"])
            rows.append(
                {
                    "task_id": f"ovon_{category}_{scene_hash}_{ep_id}".lower().replace(
                        " ", "_"
                    ),
                    "slug": f"{category}-{scene_hash}-{ep_id}".lower().replace(" ", "_"),
                    "text": f"Find and approach a {category}.",
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
                        "mtu3d_episode_index": idx,
                    },
                }
            )
    if mismatches:
        raise SystemExit("manifest/category mismatches:\n" + "\n".join(mismatches))

    # Same ordering convention as the v2 builder: group by scene so a lane
    # keeps one loaded scene across consecutive episodes.
    rows.sort(key=lambda r: (r["scene"], r["task_id"]))
    return rows


def write_instructions(
    rows: list[dict], out_path: Path, episodes_root: Path, split: str, manifest: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(episodes_root / split / "content"),
        "selection": f"MTU3D our-set {split} 120 episodes ({manifest})",
        "instructions": rows,
    }
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sensor-height", type=float, default=DEFAULT_SENSOR_HEIGHT)
    parser.add_argument("--template-config", type=Path, required=True, help="Canonical experiment recipe.")
    parser.add_argument("--config-dir", type=Path, default=workspace_root() / "configs" / "experiments")
    parser.add_argument("--instructions-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(args.manifest, args.split)
    rows = build_rows(
        args.episodes_root,
        args.split,
        manifest_rows,
        args.scene_dataset_config,
        args.sensor_height,
    )

    instructions_path = args.instructions_dir / f"ovon_{args.split}_mtu3d120.instructions.json"
    write_instructions(rows, instructions_path, args.episodes_root, args.split, args.manifest)

    config_path = args.config_dir / f"ovon-{args.split}-mtu3d120-localnav.json"
    write_config(
        args.template_config,
        config_path,
        instructions_path,
        args.scene_dataset_config,
        output_dir=args.output_dir,
    )

    cats: dict[str, int] = {}
    scene_set = set()
    for r in rows:
        cats[r["object_category"]] = cats.get(r["object_category"], 0) + 1
        scene_set.add(r["scene"])
    print(f"episodes: {len(rows)}  scenes: {len(scene_set)}  categories: {len(cats)}")
    print(f"instructions: {instructions_path}")
    print(f"config:       {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
