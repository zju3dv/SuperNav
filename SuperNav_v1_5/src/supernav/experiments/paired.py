#!/usr/bin/env python3
"""Run paired exploration and retrieval episodes from an immutable snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from supernav.backends import BACKENDS, get_backend
from supernav.experiments.arguments import add_config_arguments, select_config
from supernav.experiments.config import load_experiment_config
from supernav.experiments.deployment import deploy_task
from supernav.experiments.episode import (
    EpisodeRuntimeOverrides,
    make_run_id,
    run_one,
)
from supernav.methods.navigation.graph_memory import GraphMemoryStore
from supernav.paths import resolve_asset_path
from supernav.runtime.config import (
    csv_list,
    merged_arm,
    repo_root_from_here,
    resolve_path,
    row_task_id,
    slugify,
)
from supernav.runtime.skill_runtime import create_skill_snapshot, load_skill_snapshot


_CAPTION_EVIDENCE_FILES = (
    "slam_area_keyframes.jsonl",
    "slam_area_caption_results.jsonl",
    "occupancy.png.poses.jsonl",
)


def _relocalizer_command() -> str:
    """Bind the source-faithful matcher to the Habitat/Open3D interpreter."""

    python = os.environ.get("SUPERNAV_HABITAT_PYTHON", "").strip() or sys.executable
    script = (
        Path(__file__).resolve().parents[1]
        / "backends"
        / "habitat"
        / "memory_slam"
        / "relocalize.py"
    )
    return f"{shlex.quote(python)} {shlex.quote(str(script))}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stash(path: Path) -> Path:
    stale = path.with_name(
        f"{path.name}.stale-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    )
    path.rename(stale)
    return stale


def _artifact(path: Path, *, role: str, root: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _jsonl_count(path: Path, *, ready_only: bool = False) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if ready_only and row.get("status") != "ready":
            continue
        count += 1
    return count


def _require_completed(result: Mapping[str, Any], *, role: str) -> None:
    metrics = result.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    if metrics.get("returncode") != 0:
        raise RuntimeError(
            f"{role} agent failed: returncode={metrics.get('returncode')}"
        )
    if metrics.get("process_completed") is False:
        raise RuntimeError(f"{role} agent process did not complete")
    if not metrics.get("session_closed"):
        raise RuntimeError(f"{role} did not close its simulator session")


def _session_id(result: Mapping[str, Any]) -> str:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return ""
    return str(metrics.get("session_id") or "")


def _episode_namespace(
    args: argparse.Namespace,
    row: Mapping[str, Any],
    *,
    backend: str,
    arm: str,
    run_id: str,
    rep: str | None,
    spawn: Mapping[str, Any] | None,
    output_dir: Path,
    snapshot_root: str | None,
    visibility_cache_dir: Path | None,
    sweep_id: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        backend=backend,
        arm=arm,
        slug=row["slug"],
        instruction=row["text"],
        rep=rep,
        run_id=run_id,
        task_id=row_task_id(row),
        scene=row.get("scene"),
        scene_dataset_config_file=row.get("scene_dataset_config_file"),
        spawn=dict(spawn) if isinstance(spawn, Mapping) else None,
        metadata=json.dumps(row, ensure_ascii=False),
        ground_truth=row.get("ground_truth"),
        timeout_s=None,
        model=args.model,
        codex_provider=args.codex_provider,
        output_dir=str(output_dir),
        bridge_port=args.bridge_port,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        skill_snapshot_root=snapshot_root,
        skill_visibility_cache_dir=(
            str(visibility_cache_dir) if visibility_cache_dir is not None else None
        ),
        sweep_id=sweep_id,
    )


def _retrieval_spawn(
    args: argparse.Namespace, row: Mapping[str, Any], *, row_count: int
) -> tuple[dict[str, Any] | None, str]:
    if args.retrieve_spawn:
        if row_count != 1:
            raise ValueError("--retrieve-spawn requires exactly one selected task")
        parsed = json.loads(args.retrieve_spawn)
        if not isinstance(parsed, dict):
            raise ValueError("--retrieve-spawn JSON must be an object")
        return parsed, "cli"
    value = row.get("retrieval_spawn")
    if isinstance(value, Mapping):
        return dict(value), "manifest"
    value = row.get("spawn")
    return (dict(value), "task") if isinstance(value, Mapping) else (None, "task")


def _snapshot_caption_evidence(
    *, source_dir: Path, snapshot_dir: Path, source_session_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    caption_dir = snapshot_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    artifacts: list[dict[str, Any]] = []
    for name in _CAPTION_EVIDENCE_FILES:
        source = source_dir / name
        destination = caption_dir / name
        if source.is_file():
            _atomic_copy(source, destination)
            copied.append(name)
            artifacts.append(
                _artifact(destination, role=f"caption:{name}", root=snapshot_dir)
            )
    keyframes = _jsonl_count(caption_dir / "slam_area_keyframes.jsonl")
    results = _jsonl_count(caption_dir / "slam_area_caption_results.jsonl")
    ready = _jsonl_count(
        caption_dir / "slam_area_caption_results.jsonl", ready_only=True
    )
    poses = _jsonl_count(caption_dir / "occupancy.png.poses.jsonl")
    if keyframes <= 0:
        status = "no_keyframes"
    elif ready <= 0:
        status = "no_ready_results"
    elif poses <= 0:
        status = "no_poses"
    else:
        status = "available"
    manifest = {
        "mode": "paired_exploration",
        "status": status,
        "source_session_id": source_session_id,
        "copied_files": copied,
        "keyframe_count": keyframes,
        "result_count": results,
        "ready_result_count": ready,
        "pose_count": poses,
    }
    _atomic_json(caption_dir / "manifest.json", manifest)
    sources = []
    if status == "available":
        sources.append(
            {
                "artifact_dir": str(caption_dir),
                "evidence_origin": "exploration",
                "source_session_id": source_session_id,
            }
        )
    return sources, manifest, artifacts


def _create_snapshot(
    *,
    pair_dir: Path,
    memory_path: Path,
    cartographer_root: Path,
    exploration: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    GraphMemoryStore(memory_path).list_nodes()
    session_id = _session_id(exploration)
    if not session_id:
        raise RuntimeError("exploration result has no simulator session_id")
    source_root = cartographer_root / session_id
    visual = source_root / "visual" / "occupancy.png.pbstream"
    planning = source_root / "planning" / "occupancy.png.pbstream"
    for name, path in (("visual", visual), ("planning", planning)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                f"exploration did not produce the final {name} Cartographer state: {path}"
            )

    snapshot_dir = pair_dir / "snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True)
    destinations = {
        "memory_nodes": snapshot_dir / "memory_nodes.json",
        "cartographer_visual": snapshot_dir / "cartographer_visual.pbstream",
        "cartographer_planning": snapshot_dir / "cartographer_planning.pbstream",
    }
    for source, destination in (
        (memory_path, destinations["memory_nodes"]),
        (visual, destinations["cartographer_visual"]),
        (planning, destinations["cartographer_planning"]),
    ):
        _atomic_copy(source, destination)
    GraphMemoryStore(destinations["memory_nodes"]).list_nodes()
    artifacts = [
        _artifact(path, role=role, root=snapshot_dir)
        for role, path in destinations.items()
    ]
    # A pbstream is sufficient for Cartographer to localize eventually, but it
    # is not an agent-visible map.  Preserve the last immutable visual render
    # and all coordinate sidecars so retrieval can show the exploration map
    # before its first localized pose is available.
    visual_dir = visual.parent
    rendered = sorted(
        visual_dir.glob("occupancy_step*.png"),
        key=lambda path: int(path.stem.removeprefix("occupancy_step")),
    )
    if not rendered:
        raise RuntimeError(
            f"exploration did not produce a rendered visual occupancy map: {visual_dir}"
        )
    latest = rendered[-1]
    static_dir = snapshot_dir / "visual_static"
    static_dir.mkdir()
    static_sources = {
        latest: static_dir / "occupancy_step0.png",
        latest.with_suffix(".amcl.pgm"): static_dir / "occupancy_step0.amcl.pgm",
        visual_dir / f"{latest.name}.nodes.json": static_dir / "occupancy_step0.png.nodes.json",
        visual_dir / f"{latest.name}.render.json": static_dir / "occupancy_step0.png.render.json",
    }
    for optional_name in ("slam_area_nodes.json", "metadata.json"):
        optional = visual_dir / optional_name
        if optional.is_file():
            static_sources[optional] = static_dir / optional_name
    for source, destination in static_sources.items():
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(
                f"exploration visual map bundle is incomplete: {source}"
            )
        _atomic_copy(source, destination)
        artifacts.append(
            _artifact(
                destination,
                role=f"cartographer_visual_static:{destination.name}",
                root=snapshot_dir,
            )
        )
    caption_sources, caption_manifest, caption_artifacts = _snapshot_caption_evidence(
        source_dir=visual.parent,
        snapshot_dir=snapshot_dir,
        source_session_id=session_id,
    )
    artifacts.extend(caption_artifacts)
    manifest = {
        "schema": "supernav/paired-session-snapshot/v1",
        "created_at": _now(),
        "source_session_id": session_id,
        "artifacts": artifacts,
        "caption_inheritance": caption_manifest,
    }
    _atomic_json(snapshot_dir / "manifest.json", manifest)
    return manifest, caption_sources


def _verify_snapshot(pair_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot_dir = pair_dir / "snapshot"
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("paired resume requires snapshot/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.get("artifacts", []):
        path = snapshot_dir / str(artifact.get("path") or "")
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise RuntimeError(f"paired snapshot hash check failed: {path}")
    GraphMemoryStore(snapshot_dir / "memory_nodes.json").list_nodes()
    caption = manifest.get("caption_inheritance")
    sources: list[dict[str, Any]] = []
    if isinstance(caption, Mapping) and caption.get("status") == "available":
        sources.append(
            {
                "artifact_dir": str(snapshot_dir / "captions"),
                "evidence_origin": "exploration",
                "source_session_id": str(manifest.get("source_session_id") or ""),
            }
        )
    return manifest, sources


def _validate_pair_recipe(config: Mapping[str, Any], arm: str) -> None:
    environment = (config.get("mcp") or {}).get("environment") or {}
    if str(environment.get("HAB_CARTOGRAPHER_ENABLED", "0")) != "1":
        raise ValueError("paired sessions require a Cartographer-enabled experiment")
    if str(environment.get("HAB_STRUCTURED_SPATIAL_MEMORY_ENABLED", "0")) != "1":
        raise ValueError("paired sessions require structured spatial memory")
    tools = merged_arm(config, arm).get("tool_whitelist") or []
    required = {"hab_list_nodes", "hab_report_memory_pos", "hab_close_session"}
    missing = sorted(required - set(tools))
    if missing:
        raise ValueError(f"paired arm is missing required tools: {missing}")


def _run_pair(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    backend: Any,
    row: Mapping[str, Any],
    arm: str,
    rep: str | None,
    out_root: Path,
    snapshot_root: str | None,
    visibility_cache_dir: Path | None,
    sweep_id: str,
    row_count: int,
) -> dict[str, Any]:
    _validate_pair_recipe(config, arm)
    task_id = row_task_id(row)
    base_id = make_run_id(arm, row["slug"], rep, task_id)
    if args.run_tag:
        base_id += f"_{slugify(args.run_tag, default='experiment')}"
    pair_id = slugify(base_id, default="paired-session")
    pair_dir = out_root / "_pairs" / pair_id
    pair_path = pair_dir / "pair.json"
    exploration_id = f"{base_id}__explore"
    retrieval_id = f"{base_id}__retrieve"
    exploration_memory = pair_dir / "exploration" / "memory_nodes.json"
    retrieval_memory = pair_dir / "retrieval" / "memory_nodes.json"
    exploration_cartographer = pair_dir / "exploration" / "cartographer"
    retrieval_cartographer = pair_dir / "retrieval" / "cartographer"
    episode_output = pair_dir / "episodes"
    retrieval_spawn, retrieval_spawn_source = _retrieval_spawn(
        args, row, row_count=row_count
    )
    common: dict[str, Any] = {
        "schema": "supernav/paired-session/v1",
        "pair_id": pair_id,
        "task_id": task_id,
        "arm": arm,
        "rep": rep,
        "exploration_run_id": exploration_id,
        "retrieval_run_id": retrieval_id,
        "exploration_spawn": row.get("spawn"),
        "retrieval_spawn": retrieval_spawn,
        "retrieval_spawn_source": retrieval_spawn_source,
        "updated_at": _now(),
    }
    if pair_path.is_file():
        previous = json.loads(pair_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "completed"
            and not args.resume
            and not args.overwrite
        ):
            return previous
        if args.resume:
            snapshot_manifest, caption_sources = _verify_snapshot(pair_dir)
            exploration = previous.get("exploration")
            if not isinstance(exploration, Mapping):
                raise RuntimeError("paired resume has no recorded exploration result")
            _require_completed(exploration, role="exploration")
        elif not args.overwrite:
            raise FileExistsError(
                f"pair already exists: {pair_dir} (use --resume or --overwrite)"
            )
        else:
            _stash(pair_dir)
            pair_dir.mkdir(parents=True)
            snapshot_manifest = None
            caption_sources = []
            exploration = None
    else:
        pair_dir.mkdir(parents=True, exist_ok=True)
        snapshot_manifest = None
        caption_sources = []
        exploration = None

    planned: dict[str, Any] = {**common, "status": "planned"}
    # Keep the validated exploration record durable while a resumed retrieval
    # is being prepared.  Otherwise an unrelated stashing/startup failure can
    # overwrite pair.json with a state that can no longer be resumed even
    # though the immutable snapshot is still valid.
    if isinstance(exploration, Mapping):
        planned["exploration"] = dict(exploration)
    if isinstance(snapshot_manifest, Mapping):
        planned["snapshot"] = dict(snapshot_manifest)
    _atomic_json(pair_path, planned)
    relationship_common = {
        "type": "paired_session_snapshot",
        "pair_id": pair_id,
        "task_id": task_id,
    }
    if exploration is None:
        exploration_ns = _episode_namespace(
            args,
            row,
            backend=backend.name,
            arm=arm,
            run_id=exploration_id,
            rep=rep,
            spawn=row.get("spawn"),
            output_dir=episode_output,
            snapshot_root=snapshot_root,
            visibility_cache_dir=visibility_cache_dir,
            sweep_id=sweep_id,
        )
        _atomic_json(pair_path, {**common, "status": "exploration_running"})
        try:
            exploration = run_one(
                exploration_ns,
                runtime_overrides=EpisodeRuntimeOverrides(
                    process_environment={
                        "SUPERNAV_MEMORY_NODES_PATH": str(exploration_memory),
                        "HAB_CARTOGRAPHER_OUTPUT_DIR": str(exploration_cartographer),
                    },
                    relationship={
                        **relationship_common,
                        "role": "exploration",
                        "paired_run_id": retrieval_id,
                    },
                ),
            )
            if not args.dry_run:
                _require_completed(exploration, role="exploration")
        except Exception as exc:
            _atomic_json(
                pair_path,
                {
                    **common,
                    "status": "exploration_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        if args.dry_run:
            retrieval_ns = _episode_namespace(
                args,
                row,
                backend=backend.name,
                arm=arm,
                run_id=retrieval_id,
                rep=rep,
                spawn=retrieval_spawn,
                output_dir=episode_output,
                snapshot_root=snapshot_root,
                visibility_cache_dir=visibility_cache_dir,
                sweep_id=sweep_id,
            )
            retrieval = run_one(
                retrieval_ns,
                runtime_overrides=EpisodeRuntimeOverrides(
                    relationship={
                        **relationship_common,
                        "role": "retrieval",
                        "paired_run_id": exploration_id,
                    }
                ),
            )
            result = {
                **common,
                "status": "dry_run",
                "exploration": exploration,
                "retrieval": retrieval,
            }
            _atomic_json(pair_path, result)
            return result
        try:
            snapshot_manifest, caption_sources = _create_snapshot(
                pair_dir=pair_dir,
                memory_path=exploration_memory,
                cartographer_root=exploration_cartographer,
                exploration=exploration,
            )
        except Exception as exc:
            _atomic_json(
                pair_path,
                {
                    **common,
                    "status": "snapshot_failed",
                    "exploration": exploration,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    assert snapshot_manifest is not None
    snapshot_dir = pair_dir / "snapshot"
    _atomic_copy(snapshot_dir / "memory_nodes.json", retrieval_memory)
    retrieval_run_dir = episode_output / retrieval_id
    if args.resume and retrieval_run_dir.exists():
        _stash(retrieval_run_dir)
    retrieval_ns = _episode_namespace(
        args,
        row,
        backend=backend.name,
        arm=arm,
        run_id=retrieval_id,
        rep=rep,
        spawn=retrieval_spawn,
        output_dir=episode_output,
        snapshot_root=snapshot_root,
        visibility_cache_dir=visibility_cache_dir,
        sweep_id=sweep_id,
    )
    _atomic_json(
        pair_path,
        {
            **common,
            "status": "retrieval_running",
            "exploration": exploration,
            "snapshot": snapshot_manifest,
        },
    )
    try:
        retrieval = run_one(
            retrieval_ns,
            runtime_overrides=EpisodeRuntimeOverrides(
                process_environment={
                    "SUPERNAV_MEMORY_NODES_PATH": str(retrieval_memory),
                    "HAB_CARTOGRAPHER_OUTPUT_DIR": str(retrieval_cartographer),
                    "HAB_CARTOGRAPHER_LOAD_STATE": str(
                        snapshot_dir / "cartographer_visual.pbstream"
                    ),
                    "HAB_CARTOGRAPHER_PLANNING_LOAD_STATE": str(
                        snapshot_dir / "cartographer_planning.pbstream"
                    ),
                    "HAB_CARTOGRAPHER_PRIOR_VISUAL_ARTIFACT_DIR": str(
                        snapshot_dir / "visual_static"
                    ),
                    "HAB_CARTOGRAPHER_LOCALIZATION_MODE": (
                        "delayed_fpfh_relocalization"
                    ),
                    "HAB_CARTOGRAPHER_RELOCALIZATION_ENABLED": "1",
                    "HAB_CARTOGRAPHER_AGENT_GUIDED_RELOCALIZATION_ENABLED": "1",
                    "HAB_CARTOGRAPHER_RELOCALIZER_COMMAND": _relocalizer_command(),
                },
                init_defaults={"_slam_area_caption_sources": caption_sources},
                relationship={
                    **relationship_common,
                    "role": "retrieval",
                    "paired_run_id": exploration_id,
                    "snapshot_manifest": str(snapshot_dir / "manifest.json"),
                },
            ),
        )
        _require_completed(retrieval, role="retrieval")
    except Exception as exc:
        _atomic_json(
            pair_path,
            {
                **common,
                "status": "retrieval_failed",
                "exploration": exploration,
                "snapshot": snapshot_manifest,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    # The retrieval process must never mutate the immutable parent snapshot.
    _verify_snapshot(pair_dir)
    result = {
        **common,
        "status": "completed",
        "completed_at": _now(),
        "exploration": exploration,
        "snapshot": snapshot_manifest,
        "retrieval": retrieval,
    }
    _atomic_json(pair_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run exploration/retrieval pairs from immutable spatial-memory snapshots."
    )
    add_config_arguments(parser)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--retrieve-spawn", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bridge-port", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--codex-provider", choices=("experiment", "user"), default=None
    )
    parser.add_argument("--skill-snapshot-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    select_config(args, parser)
    if args.reps < 1 or args.rep_start < 1:
        raise ValueError("--reps and --rep-start must be positive")

    config = load_experiment_config(args.config)
    if args.backend:
        config["environment"] = dict(config.get("environment") or {})
        config["environment"]["backend"] = args.backend
    backend = get_backend(config)
    arms = csv_list(args.arms, (config.get("arms") or {}).keys())
    root = (
        resolve_path(config.get("workspace_root", "."), base=repo_root_from_here())
        or repo_root_from_here()
    )
    instructions = args.instructions or config.get("instructions_file")
    instructions_path = (
        resolve_asset_path(str(instructions), base=root) if instructions else None
    )
    rows = [
        deploy_task(config, row)
        for row in backend.tasks(config, str(instructions_path) if instructions_path else None, root)
        if args.include_hard or not row.get("hard")
    ]
    selected_ids = csv_list(args.task_ids, [])
    if selected_ids:
        available = {row_task_id(row) for row in rows}
        unknown = sorted(set(selected_ids) - available)
        if unknown:
            raise ValueError(f"unknown task IDs {unknown}; available={sorted(available)}")
        rows = [row for row in rows if row_task_id(row) in set(selected_ids)]
    if not rows or not arms:
        raise ValueError("No tasks or arms selected")

    out_root = (
        resolve_path(args.output_dir or config.get("output_dir"), base=root)
        or root / "data" / "runs" / "habitat-spatial-memory"
    )
    sweep_id = args.sweep_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    )
    sweep_root = out_root / "_paired_sweeps" / sweep_id
    snapshot_root: str | None = None
    visibility_cache_dir: Path | None = None
    if any(
        str(merged_arm(config, arm).get("skill_mode") or "").lower() == "native"
        for arm in arms
    ):
        if args.skill_snapshot_root:
            supplied = resolve_path(args.skill_snapshot_root, base=root)
            if supplied is None:
                raise ValueError("--skill-snapshot-root must resolve to a path")
            snapshot = load_skill_snapshot(supplied)
        else:
            runtime = config.get("skill_runtime") or {}
            source = resolve_asset_path(runtime.get("root"), base=root)
            if source is None:
                raise ValueError("skill_runtime.root is required for native arms")
            snapshot = create_skill_snapshot(
                source,
                run_root=sweep_root,
                skill_set_id=str(runtime.get("skill_set_id") or "bench-native"),
            )
        snapshot_root = str(snapshot.snapshot_root)
        visibility_cache_dir = sweep_root / "visibility"

    selection = {
        "schema": "supernav/paired-session-selection/v1",
        "config_source": str(args.config),
        "instructions_source": str(instructions_path or ""),
        "arms": arms,
        "task_ids": selected_ids,
        "reps": args.reps,
        "rep_start": args.rep_start,
        "created_at": _now(),
    }
    _atomic_json(sweep_root / "selection.json", selection)
    failures = 0
    results_path = sweep_root / "paired-results.jsonl"
    for rep in range(args.rep_start, args.rep_start + args.reps):
        rep_s = str(rep) if args.reps > 1 or args.rep_start != 1 else None
        for row in rows:
            for arm in arms:
                arm_root = out_root / arm if config.get("separate_arm_outputs") else out_root
                try:
                    result = _run_pair(
                        args,
                        config=config,
                        backend=backend,
                        row=row,
                        arm=arm,
                        rep=rep_s,
                        out_root=arm_root,
                        snapshot_root=snapshot_root,
                        visibility_cache_dir=visibility_cache_dir,
                        sweep_id=sweep_id,
                        row_count=len(rows),
                    )
                except Exception as exc:
                    failures += 1
                    result = {
                        "status": "failed",
                        "task_id": row_task_id(row),
                        "arm": arm,
                        "rep": rep_s,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[paired:error] {result['task_id']}: {exc}")
                with results_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
