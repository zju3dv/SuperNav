"""Validate and materialize a frozen NeedNav archive without exposing GT to policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import zipfile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def object_ids(objects) -> set[str]:
    result = set()
    for obj in objects:
        result.add(obj["id"])
        result.update(object_ids(obj.get("children", [])))
    return result


def policy_task(episode: dict) -> dict:
    """Only these fields may cross the evaluator/policy boundary."""
    return {"episode_id": episode["episode_id"], "instruction": episode["instruction"]}


def dataset_path(dataset: Path, relative: str) -> Path:
    """Resolve only files inside the selected dataset, never producer paths."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Unsafe dataset path")
    resolved = (dataset / relative).resolve()
    if not resolved.is_relative_to(dataset.resolve()):
        raise ValueError("Dataset path escapes its root")
    return resolved


def validate_dataset(dataset: Path) -> dict:
    """Recheck a materialized release without writing to the source dataset."""
    audit = json.loads((dataset / "audit.json").read_text())
    rows = audit["episodes"]
    if not rows or len(rows) != audit["episode_count"]:
        raise ValueError("Dataset episode count mismatch")
    identities, scenes = set(), set()
    for row in rows:
        identity = row["episode_id"]
        if identity in identities:
            raise ValueError("Duplicate episode ID")
        identities.add(identity)
        episode, house = load_episode(dataset, identity)
        if episode["scene_id"] in scenes:
            raise ValueError("Duplicate scene ID")
        scenes.add(episode["scene_id"])
        candidates = {c["object_id"] for stage in episode["stage_plan"]
                      for c in stage["allowed_target_candidates"]}
        if candidates - object_ids(house["objects"]):
            raise ValueError("GT objects missing from house")
    if len(scenes) != audit["scene_count"]:
        raise ValueError("Dataset scene count mismatch")
    return dict(valid=True, episode_count=len(identities), scene_count=len(scenes),
                archive_sha256=audit["archive_sha256"],
                model_conditioned_selection=audit["model_conditioned_selection"],
                success_scoring="withheld")


def audit_archive(path: Path, expected_count: int = 200) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Duplicate archive member")
        if sum(item.file_size for item in members) > 500_000_000:
            raise ValueError("Unexpected archive size")
        for item in members:
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename:
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(mode):
                raise ValueError("Archive symlinks are not allowed")
        names = sorted(n.filename for n in members if n.filename.startswith("test/episodes/") and n.filename.endswith(".json"))
        if len(names) != expected_count:
            raise ValueError(f"Expected {expected_count} episodes, found {len(names)}")
        episodes = []
        seen_ids, seen_scenes = set(), set()
        for name in names:
            episode_bytes = archive.read(name)
            ep = json.loads(episode_bytes)
            identity, scene = ep["episode_id"], ep["scene_id"]
            if identity in seen_ids or scene in seen_scenes:
                raise ValueError("Duplicate episode or scene")
            seen_ids.add(identity)
            seen_scenes.add(scene)
            for value in (*ep["start_position"].values(), ep["start_rotation_y"], ep["start_horizon"]):
                if not math.isfinite(float(value)):
                    raise ValueError("Non-finite start pose")
            house_path = f"test/reproducibility/{identity}/house_data.json"
            house_bytes = archive.read(house_path)
            house_sha = hashlib.sha256(house_bytes).hexdigest()
            if house_sha != ep["reproducibility"]["house_data_sha256"]:
                raise ValueError(f"House hash mismatch: {identity}")
            ids = object_ids(json.loads(house_bytes)["objects"])
            candidates = {c["object_id"] for stage in ep["stage_plan"] for c in stage["allowed_target_candidates"]}
            if candidates - ids:
                raise ValueError(f"GT objects missing from house: {identity}")
            episodes.append(dict(episode_id=identity, scene_id=scene, episode_path=name,
                                 episode_sha256=hashlib.sha256(episode_bytes).hexdigest(),
                                 house_path=house_path, house_sha256=house_sha,
                                 instruction=ep["instruction"], stage_count=len(ep["stage_plan"])))
        selection = json.loads(archive.read("test/result_replacement_manifest.json"))
        return dict(schema_version=1, archive_sha256=digest(path), episode_count=len(episodes),
                    scene_count=len(seen_scenes), episodes=episodes,
                    model_conditioned_selection=bool(selection.get("model_conditioned")),
                    replacement_count=selection.get("replacement_count"),
                    success_scoring="withheld", evaluation_status="engineering_only_pending_baseline_protocol")


def materialize(archive_path: Path, destination: Path) -> dict:
    audit = audit_archive(archive_path)
    receipt = destination / "audit.json"
    if destination.exists():
        if not receipt.is_file() or json.loads(receipt.read_text())["archive_sha256"] != audit["archive_sha256"]:
            raise ValueError("Refusing to overwrite an existing or incomplete dataset directory")
        for row in audit["episodes"]:
            if digest(destination / row["house_path"]) != row["house_sha256"]:
                raise ValueError("Materialized house changed")
            if digest(destination / row["episode_path"]) != row["episode_sha256"]:
                raise ValueError("Materialized instruction/start/GT episode changed")
        write_json(receipt, audit)
        return audit
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    write_json(receipt, audit)
    return audit


def load_episode(dataset: Path, episode_id: str) -> tuple[dict, dict]:
    audit = json.loads((dataset / "audit.json").read_text())
    if not episode_id or episode_id in (".", "..") or any(c in episode_id for c in "/\\"):
        raise ValueError("Unsafe episode ID")
    row = next((row for row in audit["episodes"] if row["episode_id"] == episode_id), None)
    if row is None:
        raise ValueError(f"Unknown episode ID: {episode_id}")
    house_path = dataset_path(dataset, row["house_path"])
    episode_path = dataset_path(dataset, row["episode_path"])
    if digest(episode_path) != row["episode_sha256"]:
        raise ValueError("Instruction/start/GT fingerprint changed")
    if digest(house_path) != row["house_sha256"]:
        raise ValueError("House fingerprint changed")
    episode = json.loads(episode_path.read_text())
    if episode["episode_id"] != episode_id or episode["scene_id"] != row["scene_id"]:
        raise ValueError("Episode identity differs from audit")
    if episode["reproducibility"]["house_data_sha256"] != row["house_sha256"]:
        raise ValueError("Episode house fingerprint differs from audit")
    values = [episode["start_position"][k] for k in "xyz"]
    values.extend((episode["start_rotation_y"], episode["start_horizon"]))
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Non-finite start pose")
    return episode, json.loads(house_path.read_text())
