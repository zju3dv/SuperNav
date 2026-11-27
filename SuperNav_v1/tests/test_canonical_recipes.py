"""Public recipes compose synthetic tasks, prompts, skills and execution settings."""

from __future__ import annotations

import copy
import hashlib
import json
import ipaddress
import re
from pathlib import Path

import pytest

from supernav.experiments.config import load_experiment_config
from supernav.methods.navigation.prompts import build_prompt, global_task_agent_policy
from supernav.runtime.config import instruction_rows, merged_arm
from supernav.runtime.skill_runtime import create_skill_snapshot


ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAMES = ["habitat-geo-based-executor", "habitat-learned-executor"]


def _recipe(name: str) -> dict:
    return load_experiment_config(ROOT / "configs/experiments" / f"{name}.json")


@pytest.fixture
def synthetic_manifests(tmp_path: Path) -> dict[str, Path]:
    """Create two-level indexes with invented tasks and row/default overrides."""
    result = {}
    for name in RECIPE_NAMES:
        directory = tmp_path / name
        directory.mkdir()
        rows = [
            {"task_id": "synthetic-red", "slug": "synthetic-red", "text": "Find the red test cube."},
            {"task_id": "synthetic-blue", "slug": "synthetic-blue", "text": "Find the blue test sphere.",
             "scene": "synthetic_override", "spawn": {"start_position": [2.0, 0.0, 0.0]}},
        ]
        (directory / "leaf.json").write_text(json.dumps({"instructions": rows}))
        (directory / "middle.json").write_text(json.dumps({"includes": [{
            "path": "leaf.json", "defaults": {"scene": "synthetic_scene"},
        }]}))
        index = directory / "index.json"
        index.write_text(json.dumps({"includes": [{
            "path": "middle.json", "defaults": {
                "scene": "outer_default", "spawn": {"start_position": [0.0, 0.0, 0.0]},
            },
        }]}))
        result[name] = index
    return result


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_habitat_recipes_declare_the_task_profile_without_an_independent_flag(name: str) -> None:
    canonical = _recipe(name)
    assert canonical["benchmark_profile"] == "global_task"
    assert canonical["environment"]["backend"] == "habitat"
    assert canonical["scene_dataset_config_file"] == "${SUPERNAV_SCENE_DATASET_CONFIG}"
    assert canonical["agents"]["codex"]["provider_mode"] == "user"
    assert canonical["bridge"]["per_episode"] is True
    assert canonical["bridge"]["startup_timeout_s"] == 120
    assert not {"server", "python"} & canonical["mcp"].keys()
    assert not {"NAV_LOCATE_ANYTHING_URL", "NO_PROXY", "no_proxy"} & canonical["mcp"]["environment"].keys()
    assert "HAB_MCP_GLOBAL_TASK" not in json.dumps(canonical)


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_public_recipes_require_an_explicit_external_task_manifest(name: str) -> None:
    assert _recipe(name).get("instructions_file") is None


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_task_indexes_preserve_leaf_rows_order_and_scene_defaults(name, synthetic_manifests) -> None:
    index_path = synthetic_manifests[name]
    original_files = {path: path.read_bytes() for path in index_path.parent.glob("*.json")}
    rows = instruction_rows(index_path)
    assert [row["task_id"] for row in rows] == ["synthetic-red", "synthetic-blue"]
    assert rows[0]["scene"] == "synthetic_scene"
    assert rows[0]["spawn"] == {"start_position": [0.0, 0.0, 0.0]}
    assert rows[1]["scene"] == "synthetic_override"
    assert rows[1]["spawn"] == {"start_position": [2.0, 0.0, 0.0]}
    assert all(path.read_bytes() == original for path, original in original_files.items())


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_synthetic_tasks_and_all_arms_have_instruction_preserving_prompts(name, synthetic_manifests) -> None:
    config = _recipe(name)
    original = copy.deepcopy(config)
    prompts = ROOT / config["prompts_dir"]
    assert prompts.is_dir()
    for arm_name in config["arms"]:
        arm = merged_arm(config, arm_name)
        assert "extends" not in arm
        policy = global_task_agent_policy(config, arm_cfg=arm)
        if arm.get("prompt_mode") == "minimal":
            assert policy == ""
        else:
            assert "Call hab_init_scene exactly once" in policy
        for row in instruction_rows(synthetic_manifests[name]):
            prompt = build_prompt(
                arm_name=arm_name, arm_cfg=arm, instruction=row["text"],
                scene=row["scene"], scene_dataset_config_file=config["scene_dataset_config_file"],
                spawn=row["spawn"], workspace_root=ROOT, prompts_dir=prompts,
            )
            assert row["text"].strip() in prompt, (name, row["task_id"], arm_name)
            assert "hab_close_session" in prompt
            if arm.get("skill_mode") == "native":
                assert arm["skill_name"] in prompt
    assert config == original


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_native_skill_snapshots_include_selected_skills_and_preserve_source_bytes(name: str, tmp_path: Path) -> None:
    config = _recipe(name)
    runtime = config["skill_runtime"]
    source = ROOT / runtime["root"]
    snapshot = create_skill_snapshot(source, run_root=tmp_path, skill_set_id=runtime["skill_set_id"])
    assert snapshot.manifest["skill_set_id"] == runtime["skill_set_id"]
    assert snapshot.manifest["files"]
    for record in snapshot.manifest["files"]:
        expected = (source / record["path"]).read_bytes()
        assert (snapshot.skills_root / record["path"]).read_bytes() == expected
        assert record["sha256"] == hashlib.sha256(expected).hexdigest()
    available = {row["name"] for row in snapshot.manifest["skills"]}
    for arm_name in config["arms"]:
        arm = merged_arm(config, arm_name)
        if arm.get("skill_mode") == "native":
            assert arm["skill_name"] in available
            assert set(arm.get("skill_names", [])) <= available


def test_arm_inheritance_preserves_parent_fields_and_replaces_child_nested_values() -> None:
    config = {"arms": {
        "parent": {"movement": "visual_point", "skill_file": None,
                   "environment": {"PARENT_ONLY": "1"}, "tool_whitelist": ["hab_init_scene"]},
        "child": {"extends": "parent", "environment": {"CHILD_ONLY": "1"},
                  "tool_whitelist": ["hab_close_session"]},
    }}
    original = copy.deepcopy(config)
    assert merged_arm(config, "child") == {
        "movement": "visual_point", "skill_file": None,
        "environment": {"CHILD_ONLY": "1"}, "tool_whitelist": ["hab_close_session"],
    }
    assert config == original


@pytest.mark.parametrize("name", RECIPE_NAMES + ["ai2thor-primitive"])
def test_new_shared_recipes_have_no_machine_specific_paths(name: str) -> None:
    serialized = json.dumps(_recipe(name))
    for forbidden in ("/home/", "/mnt/", "/nas1/", "http://10.", "https://10."):
        assert forbidden not in serialized
    for address in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", serialized):
        ip = ipaddress.ip_address(address)
        assert ip.is_loopback or ip.is_global


@pytest.mark.parametrize("name", RECIPE_NAMES + ["ai2thor-primitive"])
def test_shared_recipes_write_evidence_under_data_runs(name: str) -> None:
    recipe = _recipe(name)
    assert recipe["output_dir"].startswith("data/runs/")
    if recipe.get("visuals_root") is not None:
        assert recipe["visuals_root"].startswith("data/runs/")


def test_ai2thor_recipe_selects_primitive_movement_with_explicit_user_provider() -> None:
    config = _recipe("ai2thor-primitive")
    assert config["environment"] == {"backend": "ai2thor", "views": "four", "gpu": 0}
    assert config["agents"]["codex"]["provider_mode"] == "user"
    assert config["arms"] == {"default": {"movement": "primitive", "skill_mode": "none"}}
    assert config["output_dir"] == "data/runs/ai2thor-primitive"
    assert (ROOT / config["prompts_dir"]).is_dir()
