"""Habitat experiment bindings."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
from supernav.backends.base import EpisodeTask
from supernav.runtime.config import instruction_rows, row_task_id, row_spawn, row_ground_truth, resolve_path
from supernav.paths import asset_root
from supernav.backends.habitat.config import scene_config_path, task_profile_environment, build_mcp_spec
from supernav.methods.navigation.task_profile import resolve_task_profile
from supernav.backends.habitat.bridge_lifecycle import episode_bridge
from supernav.methods.navigation.prompts import build_prompt
from supernav.evaluation.metrics import compute_metrics
from supernav.evaluation.replay_manifest import build_manifest, default_visuals_root
from supernav.runtime.streams import augment_canonical_events, session_id_from_events, write_canonical_events


def _arg_value(args, name, default=None):
    return getattr(args, name, default)


def _arg_mapping(args, name):
    value = getattr(args, name, None)
    if isinstance(value, dict): return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if not isinstance(parsed, dict): raise ValueError(f"{name} JSON must be an object")
        return parsed
    return {}


def _visuals_root(cfg: Dict[str, Any], *, workspace_root: Path) -> Path:
    configured = cfg.get("visuals_root")
    if configured:
        resolved = resolve_path(configured, base=workspace_root)
        if resolved is not None:
            return resolved
    env_root = default_visuals_root()
    if str(env_root) != "/tmp/habitat_gs_visuals":
        return env_root
    return workspace_root / "data" / "nav_artifacts"


def _global_task_init_defaults(
    *,
    scene: str,
    scene_dataset_config_file: str,
    spawn: Dict[str, Any],
) -> Dict[str, Any]:
    position = spawn.get("start_position") or spawn.get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        position = [
            float(spawn.get("x", 0.0)),
            float(spawn.get("y", 0.2)),
            float(spawn.get("z", 0.0)),
        ]
    else:
        position = [float(value) for value in position]
    rotation = spawn.get("start_rotation") or spawn.get("rotation")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
        yaw = float(spawn.get("yaw", 0.0))
        rotation = [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]
    else:
        rotation = [float(value) for value in rotation]
    defaults: Dict[str, Any] = {
        "scene": scene,
        "scene_dataset_config_file": scene_dataset_config_file,
        "depth": True,
        "start_position": position,
        "start_rotation": rotation,
    }
    if spawn.get("sensor_height") is not None:
        defaults["sensor_height"] = float(spawn["sensor_height"])
    return defaults


def _multi_goal_env_payload(row_metadata: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build the HAB_MCP_NAV_GOALS_JSON payload from a manifest row.

    Carries 1-based goal indexes and descriptions only — never ground-truth
    coordinates. Returns None for single-goal rows.
    """
    targets = row_metadata.get("targets")
    if not isinstance(targets, list) or not targets:
        return None
    entries: list[Dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            return None
        try:
            index = int(target.get("index"))
        except (TypeError, ValueError):
            return None
        description = str(target.get("instruction") or target.get("description") or "")
        entries.append({"index": index, "description": description})
    return {
        "ordered": bool(row_metadata.get("ordered", True)),
        "targets": entries,
    }


def _compose_multi_goal_instruction(
    instruction: str, row_metadata: Dict[str, Any]
) -> str:
    """Append the numbered goal list and mark protocol to the instruction."""
    payload = _multi_goal_env_payload(row_metadata)
    if payload is None:
        return instruction
    targets = payload["targets"]
    if payload["ordered"]:
        header = (
            f"ORDERED GOALS ({len(targets)} total). Complete them in the "
            "listed order."
        )
    else:
        header = f"GOALS ({len(targets)} total). Find all of them, in any order."
    lines = [
        instruction.rstrip(),
        "",
        header,
        "After arriving at a goal with final visual evidence, mark it with "
        'hab_nav_goals(action="mark", target_index=k). Check '
        'hab_nav_goals(action="status") before marking and whenever unsure '
        "of your progress. Marks cannot be revoked. Close the session with "
        "outcome='achieved' only after every goal is found.",
    ]
    lines.extend(f"[{target['index']}] {target['description']}" for target in targets)
    return "\n".join(lines)


class HabitatBackend:
    name = "habitat"
    default_port = 18911

    def tasks(self, config, instructions, root):
        if not instructions:
            raise ValueError("Habitat experiments require instructions_file or --instructions")
        return instruction_rows(instructions)

    def validate_config(self, config, arm):
        from supernav.runtime.mcp import validate_mcp_config

        validate_mcp_config(config.get("mcp") or {})
        task_profile_environment(config, arm)

    def prepare_task(self, config, args, arm, root):
        # Validate before creating an episode or launching any external process.
        self.validate_config(config, arm)
        profile = resolve_task_profile(config)
        task_id = _arg_value(args, "task_id", None)
        row_metadata = _arg_mapping(args, "metadata")
        row_for_defaults = dict(row_metadata)
        if task_id:
            row_for_defaults["task_id"] = task_id
        scene = str(
            _arg_value(args, "scene", None)
            or row_metadata.get("scene")
            or config.get("scene", "")
        )
        scene_config_value = (
            _arg_value(args, "scene_dataset_config_file", None)
            or row_metadata.get("scene_dataset_config_file")
            or config.get("scene_dataset_config_file")
        )
        scene_config = scene_config_path(scene_config_value, base=root, config=config)
        spawn = row_spawn(
            row_metadata, config.get("spawn", {}) if isinstance(config.get("spawn"), dict) else {}
        )
        explicit_spawn = _arg_mapping(args, "spawn")
        if explicit_spawn:
            spawn = explicit_spawn
        ground_truth = _arg_mapping(args, "ground_truth") or row_ground_truth(row_metadata)
        goal = (
            row_metadata.get("goal") if isinstance(row_metadata.get("goal"), dict) else None
        )
        task_id = task_id or row_task_id(row_for_defaults)
        multi_goal_warning: Optional[str] = None
        if profile.is_global:
            init_defaults = _global_task_init_defaults(
                scene=scene,
                scene_dataset_config_file=str(scene_config or scene_config_value or ""),
                spawn=spawn,
            )
            benchmark_env = {
                "HAB_MCP_INIT_DEFAULTS_JSON": json.dumps(
                    init_defaults,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            }
            nav_goals_payload = _multi_goal_env_payload(row_metadata)
            if nav_goals_payload is not None:
                benchmark_env["HAB_MCP_NAV_GOALS_JSON"] = json.dumps(
                    nav_goals_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            elif isinstance(row_metadata.get("targets"), list) and row_metadata["targets"]:
                multi_goal_warning = (
                    "row carries targets but no valid multi-goal payload; "
                    "episode runs without a nav_goals ledger"
                )
            arm["_benchmark_process_environment"] = benchmark_env
        instruction = str(args.instruction)
        if profile.is_global:
            instruction = _compose_multi_goal_instruction(instruction, row_metadata)
        return EpisodeTask(task_id, instruction, scene, str(scene_config or scene_config_value or ""),
                           spawn, ground_truth, goal, multi_goal_warning)

    def configure_agent(self, config, arm, task, root, run_dir):
        # Validate and compile without injecting derived protocol flags into the
        # public experiment configuration. The runtime binds Habitat directly.
        build_mcp_spec(workspace_root=root, config=config, arm_cfg=arm)

    def prompt(self, *, task, **kwargs):
        return build_prompt(instruction=task.instruction, scene=task.scene,
                            scene_dataset_config_file=task.scene_config, spawn=task.spawn, **kwargs)

    def episode(self, *, root, config, task, run_dir, dry_run, live_context):
        return episode_bridge(root=root, config=config, port=config.get("bridge", {}).get("port", self.default_port),
                              log_path=run_dir.parent/"_bridge_logs"/(run_dir.name+".log"),
                              dry_run=dry_run, live_context=live_context)

    def collect(self, *, config, root, run_dir, task, args, result, events, session):
        session_id = session_id_from_events(events)
        visuals = _visuals_root(config, workspace_root=root)
        audit_path = visuals/f"{session_id}.benchmark_audit.jsonl" if session_id else None
        events = augment_canonical_events(events, audit_path=audit_path)
        canonical = run_dir/"canonical.jsonl"
        write_canonical_events(canonical, events)
        metrics = compute_metrics(canonical, arm=args.arm, slug=args.slug, rep=args.rep,
                                  benchmark_profile=resolve_task_profile(config).metrics_profile)
        metrics["session_id"] = session_id
        return metrics

    def replay(self, run_dir, config, root):
        try:
            build_manifest(run_dir, _visuals_root(config, workspace_root=root))
            return {"replay_manifest_status": "written"}
        except RuntimeError as exc:
            return {"replay_manifest_status": "unavailable", "replay_manifest_error": str(exc)}
