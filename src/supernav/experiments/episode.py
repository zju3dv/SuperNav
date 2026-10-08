#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from supernav.runtime.agents import (
    get_agent_backend,
    mcp_env,
    run_agent,
    save_command,
    selected_agent_config,
)
from supernav.runtime.config import (
    load_json,
    merged_arm,
    resolve_prompts_dir,
    repo_root_from_here,
    resolve_path,
    write_json,
)
from supernav.experiments.arguments import add_config_arguments, select_config
from supernav.experiments.config import load_experiment_config
from supernav.experiments.deployment import deploy_task
from supernav.evaluation.metrics import append_jsonl
from supernav.runtime.skill_runtime import create_skill_snapshot, load_skill_snapshot
from supernav.runtime.skill_visibility import probe_skill_visibility, write_visibility_report
from supernav.runtime.timing_summary import generate_timing_summary_artifacts
from supernav.paths import resolve_asset_path
from supernav.backends import BACKENDS, get_backend
from supernav.paths import asset_root


def perform_grounding_warmup(**kwargs):
    from supernav.backends.habitat.grounding_warmup import perform_grounding_warmup as warmup
    return warmup(**kwargs)


def make_run_id(
    arm: str,
    slug: str,
    rep: Optional[str],
    task_id: Optional[str] = None,
) -> str:
    task_part = f"{task_id}_" if task_id and task_id != slug else ""
    return f"{arm}_{task_part}{slug}{('__r' + rep) if rep else ''}"


def _arg_value(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _arg_mapping(args: argparse.Namespace, name: str) -> Dict[str, Any]:
    value = getattr(args, name, None)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} JSON must be an object")
        return dict(parsed)
    return {}


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_experiment_config(args.config)
    args = copy.copy(args)
    metadata = deploy_task(cfg, _arg_mapping(args, "metadata"))
    args.metadata = json.dumps(metadata, ensure_ascii=False)
    if not _arg_value(args, "scene_dataset_config_file") and metadata.get("scene_dataset_config_file"):
        args.scene_dataset_config_file = metadata["scene_dataset_config_file"]
    environment = dict(cfg.get("environment") or {})
    if _arg_value(args, "backend"):
        environment["backend"] = args.backend
    cfg["environment"] = environment
    backend = get_backend(cfg)
    bridge = dict(cfg.get("bridge") or {})
    bridge["port"] = _arg_value(args, "bridge_port") if _arg_value(args, "bridge_port") is not None else bridge.get("port", backend.default_port)
    cfg["bridge"] = bridge
    root = resolve_path(cfg.get("workspace_root", "."), base=repo_root_from_here()) or repo_root_from_here()
    out_root = resolve_path(_arg_value(args, "output_dir") or cfg.get("output_dir", "data/runs/main"), base=root)
    arm_cfg = merged_arm(cfg, args.arm)
    task = backend.prepare_task(cfg, args, arm_cfg, root)
    run_id = args.run_id or make_run_id(args.arm, args.slug, args.rep, task.task_id)
    run_dir = out_root / run_id
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"run already exists: {run_dir} (use --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "experiment.resolved.json", cfg)
    write_json(run_dir / "task.resolved.json", {
        **metadata, "task_id": task.task_id, "slug": args.slug,
        "text": task.instruction, "scene": task.scene,
        "scene_dataset_config_file": str(task.scene_config or ""),
        "spawn": task.spawn, "ground_truth": task.ground_truth,
    })
    try:
        backend.configure_agent(cfg, arm_cfg, task, root, run_dir)
        # Backends may allocate a port or prepare MCP startup settings here.
        write_json(run_dir / "experiment.resolved.json", cfg)
        with backend.episode(root=root, config=cfg, task=task, run_dir=run_dir, dry_run=args.dry_run,
                             live_context={"run_id": run_id, "task_id": task.task_id,
                                           "instruction": task.instruction, "agent": cfg.get("agent", "")}) as session:
            return _execute(args, cfg, backend, task, arm_cfg, root, out_root, run_dir, run_id, session)
    except BaseException as exc:
        path = run_dir / "run.json"
        payload = load_json(path) if path.is_file() else {"run_id": run_id, "backend": backend.name, "task_id": task.task_id}
        payload.setdefault("terminal_status", "interrupted" if isinstance(exc, KeyboardInterrupt) else "execution_error")
        payload["error"] = f"{type(exc).__name__}: {exc}"
        write_json(path, payload)
        raise


def _execute(args, cfg, backend, task, arm_cfg, root, out_root, run_dir, run_id, session):
    agent_name, agent_cfg, model_cfg = selected_agent_config(cfg)
    model_override = _arg_value(args, "model", None)
    if model_override:
        model_cfg = dict(model_cfg)
        model_cfg["name"] = str(model_override)
    provider_override = _arg_value(args, "codex_provider", None)
    if provider_override:
        if agent_name not in {"codex", "codex_profile"}:
            raise ValueError("--codex-provider requires a Codex agent backend")
        agent_cfg = dict(agent_cfg)
        agent_cfg["provider_mode"] = str(provider_override)
    agent = get_agent_backend(agent_name)
    arm_cfg["_agent_backend"] = agent_name
    task_id, scene, scene_cfg = task.task_id, task.scene, task.scene_config
    scene_cfg_value = scene_cfg
    spawn, ground_truth, goal = task.spawn, task.ground_truth, task.goal
    row_metadata = _arg_mapping(args, "metadata")
    multi_goal_warning = task.warning
    skill_summary: Dict[str, Any] = {"mode": "none"}
    skill_mode = str(arm_cfg.get("skill_mode") or "").strip().lower()
    if not skill_mode and arm_cfg.get("skill_file"):
        skill_mode = "prompt"
    if skill_mode == "native":
        runtime_cfg = cfg.get("skill_runtime")
        if not isinstance(runtime_cfg, dict):
            raise ValueError("native skill arm requires config.skill_runtime")
        supplied_snapshot = _arg_value(args, "skill_snapshot_root")
        if supplied_snapshot:
            snapshot = load_skill_snapshot(Path(str(supplied_snapshot)))
        else:
            source_root = resolve_asset_path(runtime_cfg.get("root"), base=root)
            if source_root is None:
                raise ValueError("skill_runtime.root is required for native skill arms")
            snapshot = create_skill_snapshot(
                source_root,
                run_root=run_dir,
                skill_set_id=str(runtime_cfg.get("skill_set_id") or "bench-native"),
            )
        skill_name = str(arm_cfg.get("skill_name") or "").strip()
        skill_rows = snapshot.manifest.get("skills", [])
        selected = next(
            (
                row
                for row in skill_rows
                if isinstance(row, dict) and row.get("name") == skill_name
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"native skill {skill_name!r} is not present in snapshot")
        arm_cfg["_skill_snapshot_root"] = str(snapshot.snapshot_root)
        skill_summary = {
            "mode": "native",
            "name": skill_name,
            "description": str(selected.get("description") or ""),
            "skill_set_id": snapshot.manifest.get("skill_set_id"),
            "snapshot_sha256": snapshot.snapshot_sha256,
            "snapshot_root": str(snapshot.snapshot_root),
            "manifest": dict(snapshot.manifest),
            "visibility": None,
            "sandbox_override": (
                "danger-full-access"
                if agent_name in {"codex", "codex_profile"}
                else None
            ),
        }
    elif skill_mode == "prompt":
        skill_summary = {
            "mode": "prompt",
            "skill_file": str(arm_cfg.get("skill_file") or ""),
        }
    elif skill_mode not in {"", "none"}:
        raise ValueError(f"unsupported skill_mode={skill_mode!r}")
    prompts_dir = resolve_prompts_dir(cfg, workspace_root=root)
    prompt = backend.prompt(task=task, arm_name=args.arm, arm_cfg=arm_cfg,
                            workspace_root=root, prompts_dir=prompts_dir)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    run_payload = {
        "run_id": run_id,
        "backend": backend.name,
        "environment": cfg["environment"],
        "arm": args.arm,
        "slug": args.slug,
        "rep": args.rep,
        "task_id": task_id,
        "scene": scene,
        "scene_dataset_config_file": str(scene_cfg or scene_cfg_value or ""),
        "spawn": spawn,
        "sensor_height": spawn.get("sensor_height"),
        "goal": goal,
        "targets": row_metadata.get("targets"),
        "ordered": row_metadata.get("ordered"),
        "ground_truth": ground_truth,
        "instruction": args.instruction,
        "agent": agent.name,
        "model": model_cfg,
        "sweep_id": _arg_value(args, "sweep_id"),
        "skill": skill_summary,
    }
    if multi_goal_warning:
        run_payload["multi_goal_warning"] = multi_goal_warning
    if bool(arm_cfg.get("grounding_warmup")):
        _, _, warmup_env = mcp_env(
            workspace_root=root,
            config=cfg,
            arm_cfg=arm_cfg,
        )
        try:
            warmup = perform_grounding_warmup(
                config=cfg,
                environment=warmup_env,
                workspace_root=root,
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:
            warmup = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        run_payload["grounding_warmup"] = warmup
        write_json(run_dir / "locate_anything_warmup.json", warmup)
        if not args.dry_run and warmup.get("status") != "passed":
            run_payload["terminal_status"] = "locate_anything_warmup_blocked"
            write_json(run_dir / "run.json", run_payload)
            raise RuntimeError(
                "LocateAnything warm-up blocked the run: "
                f"{warmup.get('error') or warmup.get('status')}"
            )
    timeout_s = int(args.timeout_s or agent_cfg.get("timeout_s", 1800))
    write_json(run_dir / "execution.resolved.json", {
        "agent": agent_name, "agent_config": agent_cfg, "model": model_cfg,
        "arm": args.arm, "arm_config": arm_cfg, "dry_run": bool(args.dry_run),
        "timeout_s": timeout_s,
    })
    project_dir = agent.prepare_project(
        run_dir=run_dir,
        workspace_root=root,
        config=cfg,
        arm_cfg=arm_cfg,
        agent_cfg=agent_cfg,
        model_cfg=model_cfg,
    )
    from supernav.runtime.providers import configured_provider
    explicit_provider = configured_provider(agent_cfg)
    if explicit_provider:
        run_payload["model_provider"] = {
            key: explicit_provider[key]
            for key in ("id", "base_url", "env_key", "type", "npm")
            if key in explicit_provider
        }
        run_payload["model_provider"]["mode"] = "explicit_api"
    project_metadata_path = project_dir / "codex_project.json"
    if project_metadata_path.is_file():
        project_metadata = json.loads(project_metadata_path.read_text(encoding="utf-8"))
        provider_metadata = project_metadata.get("provider")
        if isinstance(provider_metadata, dict):
            run_payload["model_provider"] = dict(provider_metadata)
    validate_environment = getattr(agent, "validate_environment", None)
    if not args.dry_run and callable(validate_environment):
        try:
            validate_environment(agent_cfg=agent_cfg)
        except ValueError as exc:
            prefix = "codex" if agent.name.startswith("codex") else agent.name
            run_payload["terminal_status"] = f"{prefix}_provider_environment_blocked"
            run_payload["provider_error"] = str(exc)
            write_json(run_dir / "run.json", run_payload)
            raise
    if skill_mode == "native":
        runtime_cfg = cfg.get("skill_runtime", {})
        gate = str(runtime_cfg.get("visibility_gate") or "required").lower()
        if gate not in {"required", "off"}:
            raise ValueError(
                "skill_runtime.visibility_gate must be 'required' or 'off'"
            )
        cache_dir_value = _arg_value(args, "skill_visibility_cache_dir")
        cache_dir = Path(str(cache_dir_value)) if cache_dir_value else run_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (
            f"{agent_name}-{skill_summary['name']}-{skill_summary['snapshot_sha256']}.json"
        )
        if args.dry_run:
            visibility = {
                "status": "not_executed_dry_run",
                "backend": agent_name,
                "skill_name": skill_summary["name"],
            }
        elif gate == "off":
            visibility = {
                "status": "disabled",
                "backend": agent_name,
                "skill_name": skill_summary["name"],
            }
        elif cache_path.is_file():
            visibility = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            visibility = probe_skill_visibility(
                backend=agent_name,
                project_dir=project_dir,
                skill_name=str(skill_summary["name"]),
                description=str(skill_summary["description"]),
                agent_cfg=agent_cfg,
                timeout_s=int(runtime_cfg.get("visibility_timeout_s") or 30),
            )
            write_visibility_report(cache_path, visibility)
        skill_summary["visibility"] = visibility
        write_visibility_report(run_dir / "skill_visibility_report.json", visibility)
        if (
            gate == "required"
            and not args.dry_run
            and visibility.get("status") != "visible"
        ):
            run_payload["terminal_status"] = "skill_visibility_blocked"
            write_json(run_dir / "run.json", run_payload)
            raise RuntimeError(
                "native skill visibility blocked: "
                f"backend={agent_name} skill={skill_summary['name']} "
                f"status={visibility.get('status')}"
            )
    write_json(run_dir / "run.json", run_payload)
    if args.dry_run:
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "project_dir": str(project_dir),
            "dry_run": True,
        }
    result = run_agent(
        agent,
        prompt=prompt,
        run_dir=run_dir,
        project_dir=project_dir,
        agent_cfg=agent_cfg,
        model_cfg=model_cfg,
        timeout_s=timeout_s,
    )
    if result.timed_out:
        run_payload.update(terminal_status="timeout", timed_out=True,
                           returncode=result.returncode, error=result.error)
        write_json(run_dir / "run.json", run_payload)
    save_command(run_dir, result)
    canonical_path = run_dir / "canonical.jsonl"
    events = agent.parse(result.raw_path, canonical_path)
    metrics = backend.collect(config=cfg, root=root, run_dir=run_dir, task=task, args=args,
                              result=result, events=events, session=session)
    session_id = metrics.get("session_id")
    metrics.update(
        {
            "run_id": run_id,
            "backend": backend.name,
            "task_id": task_id,
            "scene": scene,
            "scene_dataset_config_file": str(scene_cfg or scene_cfg_value or ""),
            "spawn": spawn,
            "sensor_height": spawn.get("sensor_height"),
            "goal": goal,
            "targets": row_metadata.get("targets"),
            "ordered": row_metadata.get("ordered"),
            "ground_truth": ground_truth,
            "agent": agent.name,
            "model": model_cfg,
            "model_provider": run_payload.get("model_provider"),
            "returncode": result.returncode,
            "process_completed": result.returncode == 0,
            "duration_s": round(result.duration_s, 3),
            "session_id": session_id,
        }
    )
    if str(cfg.get("benchmark_profile") or "") == "global_task":
        metrics["success"] = bool(
            metrics.get("success") and metrics["process_completed"]
        )
        metrics["success_semantics"] = (
            "returncode_zero_and_session_closed_and_structured_achieved"
        )
    if result.timed_out:
        metrics.update(terminal_status="timeout", timed_out=True, error=result.error)
        # Unscored backends retain success=None. A completed process is still
        # required for a positive engineering completion claim elsewhere.
        if metrics.get("success") is not None:
            metrics["success"] = False
    result.session_id = session_id
    save_command(run_dir, result)
    write_json(run_dir / "metrics.json", metrics)
    metrics.update(backend.replay(run_dir, cfg, root))
    write_json(run_dir / "metrics.json", metrics)
    timing_artifacts = generate_timing_summary_artifacts(run_dir)
    append_jsonl(out_root / "results.jsonl", metrics)
    result_payload = {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}
    manifest_path = run_dir / "manifest.json"
    result_payload["manifest"] = str(manifest_path) if manifest_path.is_file() else None
    if timing_artifacts is not None:
        result_payload["timing_summary"] = {
            "json_path": timing_artifacts["json_path"],
            "image_path": timing_artifacts["image_path"],
        }
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one portable benchmark episode.")
    add_config_arguments(parser)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--rep", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--scene-dataset-config-file", default=None)
    parser.add_argument(
        "--spawn", default=None, help="JSON object overriding config.spawn"
    )
    parser.add_argument(
        "--metadata", default=None, help="JSON object with task metadata"
    )
    parser.add_argument(
        "--ground-truth", default=None, help="JSON object copied to artifacts"
    )
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=None,
        help="Override config.bridge.port for this run without modifying the config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.model.name for this run without modifying the config.",
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
    parser.add_argument("--skill-snapshot-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skill-visibility-cache-dir", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--sweep-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    select_config(args, parser)
    result = run_one(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if (result.get("metrics") or {}).get("returncode", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
