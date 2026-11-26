"""Read the public HabitatGS release without putting evaluator GT in run inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DATASET_ID = "yyh929/HabitatGS-Benchmark"
TESTED_REVISION = "4a8fe5cf654c1d19f0c3c992720d3c4f57844f12"
SUCCESS = {"metric": "distance_to_approved_viewpoint_set", "threshold_m": 0.1, "comparator": "<"}
TASKS = ("objectnav", "multi-objectnav")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def asset_path(root: Path, relative: str) -> Path:
    # Allow locally prepared symlinks to existing assets; validate the *logical*
    # relative path, then verify file contents against the release hashes.
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"asset must be a relative path without '..': {relative}")
    return root / path


def verify_file(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"release checksum mismatch: {path}")


def navmesh_name(episode: dict) -> str:
    return str(Path(episode["scene"]["asset"]).with_suffix("").with_suffix(".navmesh"))


def load_release(root: Path, task: str) -> tuple[list[dict], dict, dict]:
    """Verify the selected annotations/schema/asset manifest, then validate episodes."""
    from jsonschema import Draft202012Validator

    if task not in TASKS:
        raise ValueError(f"unknown HabitatGS task: {task}")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "habitat_gs_public_release/v1":
        raise ValueError("expected a public HabitatGS release root with manifest.json")
    names = [f"{task}/v1/episodes.json", f"{task}/v1/episode_v1.schema.json", "scene-assets/v1/manifest.json"]
    for name in names:
        verify_file(root / name, manifest["files"][name])
    schema = read_json(root / names[1])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    episodes = read_json(root / names[0])["episodes"]
    if not episodes or len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("empty dataset or duplicate episode IDs")
    assets = read_json(root / names[2])["files"]
    scenes = {}
    for episode in episodes:
        validator.validate(episode)
        if episode["schema"] != "habitat_gs/objectnav/v1" or episode["evaluation"]["success"] != SUCCESS:
            raise ValueError("unsupported HabitatGS evaluation contract")
        if episode["ordered"] != (task == "multi-objectnav") or (task == "objectnav" and len(episode["targets"]) != 1):
            raise ValueError("episode task type does not match selected task")
        if [t["index"] for t in episode["targets"]] != list(range(len(episode["targets"]))):
            raise ValueError("target indexes must be consecutive and zero-based")
        if any(not t["viewpoints"] or any(v["status"] != "approved" for v in t["viewpoints"])
               for t in episode["targets"]):
            raise ValueError("release contains missing or unapproved viewpoints")
        if episode["ground_truth"]["status"] != "complete":
            raise ValueError("release ground truth is not complete")
        for variant in ("local_shortest", "global_shortest"):
            distance = episode["ground_truth"]["variants"][variant]["total_geodesic_distance"]
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("invalid reference distance")
        scene = episode["scene"]
        if scene["id"] in scenes and scenes[scene["id"]] != scene["asset"]:
            raise ValueError("one scene ID refers to multiple assets")
        scenes[scene["id"]] = scene["asset"]
        for name in (scene["asset"], navmesh_name(episode)):
            asset_path(Path("."), name)
            if name not in assets:
                raise ValueError(f"asset is absent from scene manifest: {name}")
    provenance = {
        "dataset": DATASET_ID, "task": task, "episode_count": len(episodes),
        "manifest_sha256": sha256(root / "manifest.json"),
        "episodes_sha256": manifest["files"][names[0]]["sha256"],
        "scene_manifest_sha256": manifest["files"][names[2]]["sha256"],
    }
    return episodes, assets, provenance
