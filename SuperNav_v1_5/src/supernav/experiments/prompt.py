#!/usr/bin/env python3
from __future__ import annotations

import argparse

from supernav.runtime.config import (
    merged_arm,
    resolve_prompts_dir,
    repo_root_from_here,
    resolve_path,
)
from supernav.runtime.agents import selected_agent_config
from supernav.methods.navigation.prompts import build_prompt
from supernav.backends.habitat.config import scene_config_path
from supernav.experiments.arguments import add_config_arguments, select_config
from supernav.experiments.config import load_experiment_config
from supernav.paths import resolve_asset_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one portable harness prompt.")
    add_config_arguments(parser)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    select_config(args, parser)
    cfg = load_experiment_config(args.config)
    root = (
        resolve_path(cfg.get("workspace_root", "."), base=repo_root_from_here())
        or repo_root_from_here()
    )
    arm_cfg = merged_arm(cfg, args.arm)
    agent_name, _agent_cfg, _model_cfg = selected_agent_config(cfg)
    arm_cfg["_agent_backend"] = agent_name
    scene_cfg = scene_config_path(cfg.get("scene_dataset_config_file"), base=root, config=cfg)
    prompts_dir = resolve_prompts_dir(cfg, workspace_root=root)
    print(
        build_prompt(
            arm_name=args.arm,
            arm_cfg=arm_cfg,
            instruction=args.instruction,
            scene=str(cfg.get("scene", "")),
            scene_dataset_config_file=str(
                scene_cfg or cfg.get("scene_dataset_config_file", "")
            ),
            spawn=cfg.get("spawn", {}),
            workspace_root=root,
            prompts_dir=prompts_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
