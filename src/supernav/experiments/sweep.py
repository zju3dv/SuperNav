#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from supernav.runtime.config import (
    csv_list,
    instruction_rows,
    load_json,
    merged_arm,
    repo_root_from_here,
    resolve_path,
    row_task_id,
    slugify,
    write_json,
)
from supernav.experiments.deployment import deploy_task
from supernav.experiments.arguments import add_config_arguments, select_config
from supernav.experiments.config import load_experiment_config
from supernav.runtime.skill_runtime import create_skill_snapshot, load_skill_snapshot
from supernav.backends import BACKENDS, get_backend
from supernav.backends.base import BridgeStartupError
from supernav.experiments.episode import make_run_id, run_one
from supernav.paths import asset_root, resolve_asset_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a portable benchmark sweep.")
    add_config_arguments(parser)
    parser.add_argument(
        "--instructions",
        default=None,
        help=(
            "External instruction manifest. Defaults to config.instructions_file; "
            "required for Habitat experiments."
        ),
    )
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="Simulator backend; overrides environment.backend in the config.")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--rep-start",
        type=int,
        default=1,
        help="First one-based repetition index; useful for append-only reruns.",
    )
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.model.name for every run in this sweep.",
    )
    parser.add_argument(
        "--codex-provider",
        choices=("experiment", "user"),
        default=None,
        help=(
            "Codex API provider mode. Defaults to the isolated experiment relay; "
            "use 'user' to inherit the normal Codex provider and auth."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Append a stable experiment tag to each run id without changing task slugs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config.output_dir for this sweep.",
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=None,
        help="Override config.bridge.port for every run in this sweep.",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs to run; all selected IDs must exist.",
    )
    parser.add_argument(
        "--skill-snapshot-root",
        default=None,
        help=(
            "Reuse an existing validated immutable skill snapshot for native arms "
            "instead of creating one from config.skill_runtime.root."
        ),
    )
    args = parser.parse_args()
    select_config(args, parser)
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.rep_start < 1:
        raise ValueError("--rep-start must be at least 1")
    cfg = load_experiment_config(args.config)
    cfg["environment"] = dict(cfg.get("environment") or {})
    if args.backend:
        cfg["environment"]["backend"] = args.backend
    backend = get_backend(cfg)
    arms = csv_list(args.arms, (cfg.get("arms") or {}).keys())
    # Backends may validate their protocol bindings without SDK imports or I/O.
    # Validate the whole selection before the first episode can spend compute.
    validate_config = getattr(backend, "validate_config", None)
    if validate_config is not None:
        for arm_name in arms:
            validate_config(cfg, merged_arm(cfg, arm_name))
    root = (
        resolve_path(cfg.get("workspace_root", "."), base=repo_root_from_here())
        or repo_root_from_here()
    )
    configured_instructions = cfg.get("instructions_file")
    instructions_path = args.instructions
    if not instructions_path and configured_instructions:
        resolved_instructions = resolve_asset_path(str(configured_instructions), base=root)
        instructions_path = (
            str(resolved_instructions) if resolved_instructions else None
        )
    out_root = resolve_path(
        args.output_dir or cfg.get("output_dir", "data/runs/main"), base=root
    ) or (root / "data" / "runs" / "main")
    native_arms = [
        name
        for name in arms
        if str(merged_arm(cfg, name).get("skill_mode") or "").lower() == "native"
    ]
    snapshot_root = None
    visibility_cache_dir = None
    sweep_id = args.sweep_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    sweep_root = out_root / "_sweeps" / sweep_id
    if native_arms:
        runtime_cfg = cfg.get("skill_runtime")
        if not isinstance(runtime_cfg, dict):
            raise ValueError("native skill arms require config.skill_runtime")
        sweep_id = (
            sweep_id
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        )
        sweep_root = out_root / "_sweeps" / sweep_id
        if args.skill_snapshot_root:
            supplied_snapshot = resolve_path(args.skill_snapshot_root, base=root)
            if supplied_snapshot is None:
                raise ValueError("--skill-snapshot-root must resolve to a path")
            snapshot = load_skill_snapshot(supplied_snapshot)
        else:
            source_root = resolve_asset_path(runtime_cfg.get("root"), base=root)
            if source_root is None:
                raise ValueError("skill_runtime.root is required for native skill arms")
            snapshot = create_skill_snapshot(
                source_root,
                run_root=sweep_root,
                skill_set_id=str(runtime_cfg.get("skill_set_id") or "bench-native"),
            )
        snapshot_root = str(snapshot.snapshot_root)
        visibility_cache_dir = str(sweep_root / "visibility")
    rows = [
        deploy_task(cfg, r)
        for r in backend.tasks(cfg, instructions_path, root)
        if args.include_hard or not r.get("hard")
    ]
    selected_task_ids = csv_list(args.task_ids, [])
    if selected_task_ids:
        selected = set(selected_task_ids)
        available = {task_id for row in rows if (task_id := row_task_id(row))}
        unknown = sorted(selected - available)
        if unknown:
            raise ValueError(
                f"unknown task IDs {unknown}; available={sorted(available)}"
            )
        rows = [row for row in rows if row_task_id(row) in selected]
    if not rows or not arms:
        raise ValueError("No tasks or arms selected")
    write_json(sweep_root / "experiment.resolved.json", cfg)
    write_json(sweep_root / "instructions.resolved.json", {"instructions": rows})
    write_json(sweep_root / "selection.json", {
        "config_source": str(args.config), "instructions_source": instructions_path,
        "arms": arms, "task_ids": selected_task_ids, "reps": args.reps,
        "rep_start": args.rep_start, "model_override": args.model,
        "codex_provider_override": args.codex_provider, "dry_run": args.dry_run,
    })
    total = len(rows) * len(arms) * args.reps
    print(
        f"[harness-dev] {len(rows)} instructions x {len(arms)} arms x {args.reps} reps = {total}"
    )
    idx = failures = 0
    for rep in range(args.rep_start, args.rep_start + args.reps):
        rep_s = str(rep) if args.reps > 1 or args.rep_start != 1 else None
        for row in rows:
            for arm in arms:
                idx += 1
                task_id = row_task_id(row)
                run_id = make_run_id(arm, row["slug"], rep_s, task_id)
                if args.run_tag:
                    run_id += f"_{slugify(args.run_tag, default='experiment')}"
                arm_root = (
                    out_root / arm if cfg.get("separate_arm_outputs") else out_root
                )
                if (arm_root / run_id).exists() and not args.overwrite:
                    print(f"[skip] {run_id}: already exists")
                    continue
                print(f"\n========== [{idx}/{total}] {run_id} ==========")
                ns = argparse.Namespace(
                    config=args.config,
                    backend=backend.name,
                    arm=arm,
                    slug=row["slug"],
                    instruction=row["text"],
                    rep=rep_s,
                    run_id=run_id,
                    task_id=task_id,
                    scene=row.get("scene"),
                    scene_dataset_config_file=row.get("scene_dataset_config_file"),
                    spawn=row.get("spawn"),
                    metadata=json.dumps(row, ensure_ascii=False),
                    ground_truth=row.get("ground_truth"),
                    timeout_s=None,
                    model=args.model,
                    codex_provider=args.codex_provider,
                    output_dir=str(arm_root),
                    bridge_port=args.bridge_port,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    skill_snapshot_root=snapshot_root,
                    skill_visibility_cache_dir=visibility_cache_dir,
                    sweep_id=sweep_id,
                )
                try:
                    result = run_one(ns)
                    if not args.dry_run and result.get("metrics", {}).get("returncode", 0) != 0:
                        failures += 1
                except BridgeStartupError:
                    raise
                except SystemExit as exc:
                    failures += 1
                    print(f"[skip/fail] {run_id}: {exc}")
                except Exception:
                    failures += 1
                    # An episode that raises (e.g. codex exec hitting the agent
                    # timeout) must not kill the lane: record and move on. The
                    # partial run dir is skipped on rerun until cleaned up.
                    print(f"[error] {run_id}: episode raised; continuing sweep")
                    traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
