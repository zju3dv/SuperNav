"""Convert a public HabitatGS-Benchmark v1 release into GT-free SuperNav tasks."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from supernav.evaluation.objectnav.habitatgs_data import (
    TASKS, asset_path, load_release, navmesh_name, verify_file, write_new,
)
from supernav.experiments.config import load_experiment_config
from supernav.paths import resolve_asset_path, workspace_root
from supernav.runtime.config import slugify


def configure_multi_goal_arms(config: dict) -> None:
    """Bind each supported executor to its multi-goal policy and ledger tool."""
    families = {
        "visual_point": ("geo-based-executor", "localnav-pointnav-geo-based-executor"),
        "localnav": ("learned-executor", "localnav-pointnav"),
    }
    arms = config.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("multi-objectnav template must define native executor arms")
    for name, arm in arms.items():
        if not isinstance(arm, dict) or arm.get("movement") not in families or arm.get("skill_mode") != "native":
            raise ValueError(f"multi-objectnav arm {name!r} must use a native Geo or learned executor")
        family, local_skill = families[arm["movement"]]
        base_skill = f"global-navigation-{family}"
        multi_skill = f"multi-global-navigation-{family}"
        if arm.get("skill_name") not in (base_skill, multi_skill):
            raise ValueError(f"multi-objectnav arm {name!r} must use {base_skill} or {multi_skill}")
        skills = arm.get("skill_names", [base_skill, local_skill])
        whitelist = arm.get("tool_whitelist")
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise ValueError(f"arm {name!r} skill_names must be a list of names")
        if not isinstance(whitelist, list) or not all(isinstance(t, str) for t in whitelist):
            raise ValueError(f"arm {name!r} must declare a tool_whitelist")
        arm["skill_name"] = multi_skill
        # The wrapper explicitly reads the base global skill and its references.
        # Include those dependencies in the selected run-local skill bundle.
        arm["skill_names"] = list(dict.fromkeys([multi_skill, base_skill, local_skill, *skills]))
        arm["tool_whitelist"] = list(dict.fromkeys([*whitelist, "hab_nav_goals"]))


def instruction_row(episode: dict, sensor_height: float, provenance: dict) -> dict:
    ident = episode["episode_id"]
    # run-one normalizes task IDs. Fail instead of silently losing case/identity.
    if slugify(ident) != ident:
        raise ValueError(f"episode ID cannot be represented losslessly by SuperNav: {ident}")
    row = {
        "task_id": ident, "slug": ident, "text": episode["instruction"],
        "scene": episode["scene"]["id"],
        "spawn": {"start_position": episode["start"]["position"],
                  "start_rotation": episode["start"]["rotation"], "sensor_height": sensor_height},
        "habitatgs": {"episode_id": ident, **provenance},
    }
    if episode["ordered"]:
        row.update(ordered=True, targets=[
            {"index": t["index"] + 1, "instruction": t["description"]}
            for t in episode["targets"]
        ])
    return row


def prepare(*, dataset_root: Path, task: str, scene_root: Path, output_dir: Path,
            template_config: Path, sensor_height: float, dataset_revision: str,
            task_ids: list[str] | None = None) -> dict:
    if not math.isfinite(sensor_height) or sensor_height <= 0:
        raise ValueError("sensor height must be finite and positive")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset_revision):
        raise ValueError("dataset revision must be a full Hugging Face commit SHA")
    if output_dir.exists():
        raise ValueError("output directory exists; use a new preparation directory")
    episodes, assets, provenance = load_release(dataset_root, task)
    provenance["dataset_revision"] = dataset_revision
    if task_ids:
        selected = set(task_ids)
        if not selected <= {e["episode_id"] for e in episodes}:
            raise ValueError("unknown selected episode ID")
        episodes = [e for e in episodes if e["episode_id"] in selected]
    config = load_experiment_config(template_config)
    if config.get("environment", {}).get("backend") != "habitat":
        raise ValueError("template must use the Habitat backend")
    if task == "multi-objectnav":
        configure_multi_goal_arms(config)
    # Complete validation before creating any output.
    rows = [instruction_row(e, sensor_height, provenance) for e in episodes]
    scenes = {e["scene"]["id"]: e for e in episodes}
    for name, e in scenes.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe scene ID")
        for relative in (e["scene"]["asset"], navmesh_name(e)):
            verify_file(asset_path(scene_root, relative), assets[relative])
    output_dir = output_dir.resolve()
    scene_config = output_dir / "scenes.scene_dataset_config.json"
    navmeshes = {}
    for name, e in scenes.items():
        stage_path = output_dir / "stages" / f"{name}.stage_config.json"
        write_new(stage_path, {
            "render_asset": str(asset_path(scene_root, e["scene"]["asset"]).absolute()),
            "render_asset_type": "gaussian_splatting", "units_to_meters": 1.0,
            "orient_up": [0, 1, 0], "orient_front": [0, 0, -1], "origin": [0, 0, 0],
            "frustum_culling": False, "light_setup": "default",
        })
        write_new(output_dir / "scenes" / f"{name}.scene_instance.json", {
            "stage_instance": {"template_name": str(stage_path)}, "navmesh_instance": name,
        })
        navmeshes[name] = str(asset_path(scene_root, navmesh_name(e)).absolute())
    write_new(scene_config, {
        "stages": {"paths": {".json": ["stages"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}}, "navmesh_instances": navmeshes,
    })
    instructions = output_dir / "instructions.json"
    write_new(instructions, {"habitatgs": provenance, "instructions": rows})
    prompts = "multi_global_task" if task == "multi-objectnav" else "global_task"
    config.update(
        workspace_root=str(workspace_root()), benchmark_profile="global_task",
        instructions_file=str(instructions), scene_dataset_config_file=str(scene_config),
        prompts_dir=str(resolve_asset_path(f"configs/benchmarks/{prompts}/prompts", base=workspace_root())),
        output_dir=str(output_dir / "runs"), visuals_root=str(output_dir / "visuals"),
    )
    # Bridge owns primitive-action evidence; a waypoint or completion claim is
    # never sufficient to construct the public benchmark submission.
    config.setdefault("environment", {})["primitive_trace_dir"] = str(output_dir / "visuals")
    config.setdefault("mcp", {}).setdefault("environment", {})["HAB_DEFAULT_AGENT_NAVMESH"] = "0"
    write_new(output_dir / "run.json", config)
    receipt = {**provenance, "selected_episode_ids": [e["episode_id"] for e in episodes],
               "sensor_height": sensor_height, "scene_root": str(scene_root.absolute()),
               "assets_verified": True, "policy_inputs_contain_gt": False}
    write_new(output_dir / "preparation.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--sensor-height", type=float, required=True)
    parser.add_argument("--task-ids", help="Optional comma-separated subset; default is the full release")
    args = vars(parser.parse_args())
    args["task_ids"] = args["task_ids"].split(",") if args["task_ids"] else None
    receipt = prepare(**args)
    print(f"Prepared {len(receipt['selected_episode_ids'])} episodes in {args['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
