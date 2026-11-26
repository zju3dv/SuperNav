from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

from supernav.paths import resolve_asset_path, asset_root
from supernav.methods.navigation.task_profile import resolve_task_profile

VISUAL_NAV_MOVEMENTS = {
    "oracle",
    "visual_overlay",
    "visual_point",
    "visual_combined",
    "visual_ground_preview",
    "visual_ground_preview_locate",
    "localnav",
}


def global_task_agent_policy(
    config: Mapping[str, Any],
    *,
    native_skill_path: Path | None = None,
    arm_cfg: Mapping[str, Any] | None = None,
) -> str:
    """Return run-local hard constraints for global-task episodes."""
    if arm_cfg and arm_cfg.get("prompt_mode") == "minimal":
        return ""
    if not resolve_task_profile(config).is_global:
        return ""
    policy_options = config.get("global_task_policy")
    spatial_memory_enabled = not (
        isinstance(policy_options, Mapping)
        and policy_options.get("spatial_memory") is False
    )
    whitelist = arm_cfg.get("tool_whitelist") if isinstance(arm_cfg, Mapping) else None

    def _has_tool(tool: str) -> bool:
        # Arms without an explicit whitelist may call everything.
        return whitelist is None or tool in whitelist

    skill_rule = ""
    if native_skill_path is not None and _has_tool("hab_visual_ground_preview"):
        locate_path = native_skill_path.parent.parent / "locate-anything" / "SKILL.md"
        skill_rule = (
            f"Before the first grounding call, read `{locate_path}` completely. Do not "
            "substitute user-level or global skills.\n"
        )
    memory_rule = ""
    if spatial_memory_enabled and _has_tool("hab_register_spatial_junction"):
        memory_rule = (
            "At a local junction with two or more plausible doorway/opening choices, call "
            "hab_register_spatial_junction before choosing one, then pass its branch_id to "
            "hab_visual_ground_preview for that branch. Treat memory warnings as conservative "
            "visual facts, never as inferred room identity. "
        )
    blocked_close_rule = ""
    if memory_rule:
        blocked_close_rule = (
            "A blocked close may first return blocked_close_audit_required without closing; "
            "in that case follow the native skill and immediately retry with its token and "
            "complete audit. "
        )
    return (
        "\nGLOBAL-TASK BENCHMARK RULES:\n"
        "Use only the whitelisted Habitat-GS MCP tools for scene perception and action. "
        "Never inspect the scene through shell commands, Python, files, image directories, "
        "or run artifacts. Never read goal coordinates, ground-truth paths, trajectories, "
        "or evaluator metadata. Remain mapless; do not query maps or navmeshes and do not "
        "plan from world coordinates.\n"
        "This is a global whole-house exploration task, not a single-room task. Treat the "
        "initial room as only the starting area: the target may be beyond multiple doorways "
        "or rooms. If the current room lacks the complete target evidence, continue bounded "
        "exploration through visible transitions and semantic proxies; do not stop merely "
        "because the target is absent from the current room.\n"
        + skill_rule
        + "Call hab_init_scene exactly once. After a successful init, never call it again: "
        "it is not an observation refresh tool, and movement results provide the latest "
        "surround. If init times out or fails, do not blindly retry it; report the failure. "
        + memory_rule
        + "Complete exactly one successful hab_close_session with outcome='achieved' when "
        "the requested target is verified or outcome='blocked' when giving up. "
        + blocked_close_rule
        + "Only produce the final answer after a result with closed=true. Do not bypass the "
        "tool whitelist or blindly repeat a failed mutating call.\n"
    )


def _read_template(
    prompts_dir: Path,
    name: str,
    *,
    fallback_dir: Path | None = None,
) -> str:
    path = prompts_dir / name
    if not path.is_file() and fallback_dir is not None:
        fallback_path = fallback_dir / name
        if fallback_path.is_file():
            path = fallback_path
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _skill_text(arm_cfg: Mapping[str, Any], *, workspace_root: Path) -> str:
    mode = str(arm_cfg.get("skill_mode") or "").strip().lower()
    if mode == "native":
        skill_name = str(arm_cfg.get("skill_name") or "").strip()
        if not skill_name:
            raise ValueError("native skill arm requires skill_name")
        backend = str(arm_cfg.get("_agent_backend") or "").strip().lower()
        if backend in {"codex", "codex_profile"}:
            load_action = f"explicitly use the native skill `/{skill_name}`"
        elif backend == "kimi":
            load_action = f"invoke the native skill `/skill:{skill_name}`"
        elif backend == "opencode":
            load_action = f'call the native skill tool with name "{skill_name}"'
        else:
            load_action = f"load the native skill `{skill_name}`"
        return (
            "\nNATIVE SKILL (required):\n"
            f"Before taking any navigation action, {load_action}, read its complete "
            "instructions, and follow them for the rest of the episode. If it cannot "
            "be loaded, stop and report that failure. Do not substitute user-level or "
            "global skills.\n"
        )
    if mode not in {"", "none", "prompt"}:
        raise ValueError(f"unsupported skill_mode={mode!r}")
    if mode == "none":
        return ""
    skill_path = resolve_asset_path(arm_cfg.get("skill_file"), base=workspace_root)
    if not skill_path or not skill_path.is_file():
        return ""
    text = skill_path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"\nNAVIGATION SKILL (follow this):\n{text}\n"


def _float_arg(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _fmt_float_list(values: list[float]) -> str:
    return "[" + ", ".join(_fmt_float(value) for value in values) + "]"


def _float_list_arg(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def build_prompt(
    *,
    arm_name: str,
    arm_cfg: Mapping[str, Any],
    instruction: str,
    scene: str,
    scene_dataset_config_file: str,
    spawn: Mapping[str, Any],
    workspace_root: Path,
    prompts_dir: Optional[Path] = None,
) -> str:
    if arm_cfg.get("prompt_mode") == "minimal":
        return (
            f"{instruction.strip()}\nCall hab_close_session when finished.\n"
            + _skill_text(arm_cfg, workspace_root=workspace_root)
        )
    explicit_position = _float_list_arg(
        spawn.get("start_position") or spawn.get("position"),
        length=3,
    )
    explicit_rotation = _float_list_arg(
        spawn.get("start_rotation") or spawn.get("rotation"),
        length=4,
    )
    if explicit_position is not None:
        start_position = _fmt_float_list(explicit_position)
    else:
        x = _float_arg(spawn.get("x"), default=0.0)
        y = _float_arg(spawn.get("y"), default=0.2)
        z = _float_arg(spawn.get("z"), default=0.0)
        start_position = _fmt_float_list([x, y, z])
    if explicit_rotation is not None:
        start_rotation = _fmt_float_list(explicit_rotation)
    else:
        yaw = _float_arg(spawn.get("yaw"), default=0.0)
        half_yaw = yaw / 2.0
        start_rotation = _fmt_float_list(
            [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)]
        )
    sensor_height = spawn.get("sensor_height")
    sensor_height_arg = ""
    if sensor_height is not None:
        sensor_height_arg = (
            f", sensor_height={_fmt_float(_float_arg(sensor_height, default=1.5))}"
        )
    movement = str(arm_cfg.get("movement", "primitive"))

    prompts_dir = resolve_asset_path(
        prompts_dir or "configs/benchmarks/main/prompts", base=workspace_root
    )
    fallback_prompts_dir = asset_root() / "configs" / "benchmarks" / "main" / "prompts"

    task_context = _read_template(
        prompts_dir,
        (
            "visual_nav_context.md"
            if movement in VISUAL_NAV_MOVEMENTS
            else "default_context.md"
        ),
        fallback_dir=fallback_prompts_dir,
    ).strip()
    tool_safety = _read_template(
        prompts_dir,
        "safety_tools.md" if movement == "primitive" else "depth_tools.md",
        fallback_dir=fallback_prompts_dir,
    ).strip()
    # A config-local `common_<movement>.md` overrides the shared common.md, so an
    # arm family can swap the episode framing without touching other arms.
    common_name = f"common_{movement}.md"
    if not (prompts_dir / common_name).is_file():
        common_name = "common.md"
    common = (
        _read_template(
            prompts_dir,
            common_name,
            fallback_dir=fallback_prompts_dir,
        )
        .format(
            instruction=instruction,
            task_context=task_context,
            scene=scene,
            scene_dataset_config_file=scene_dataset_config_file,
            start_position=start_position,
            start_rotation=start_rotation,
            sensor_height_arg=sensor_height_arg,
            tool_safety=tool_safety,
        )
        .rstrip()
    )
    move = _read_template(
        prompts_dir,
        f"movement_{movement}.md",
        fallback_dir=fallback_prompts_dir,
    )
    end = _read_template(prompts_dir, "end.md", fallback_dir=fallback_prompts_dir)
    return (
        common
        + "\n"
        + move
        + _skill_text(arm_cfg, workspace_root=workspace_root)
        + "\n"
        + end
    )
