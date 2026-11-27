from __future__ import annotations

import json

import pytest
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.experiments.preparation.global_tasks import build_configs
from supernav.methods.navigation.prompts import build_prompt, global_task_agent_policy  # noqa: E402
from supernav.runtime.skill_runtime import lint_skill_root  # noqa: E402


def test_global_task_policy_is_profile_scoped_and_enforces_lifecycle() -> None:
    skill = Path("/run/.codex_home/skills/global-navigation/SKILL.md")
    assert global_task_agent_policy({}, native_skill_path=skill) == ""

    policy = global_task_agent_policy(
        {"benchmark_profile": "global_task"}, native_skill_path=skill
    )
    assert "Call hab_init_scene exactly once" in policy
    assert "exactly one successful hab_close_session" in policy
    assert "blocked_close_audit_required" in policy
    assert "closed=true" in policy
    assert "not an observation refresh tool" in policy
    assert "goal coordinates" in policy
    assert "Remain mapless" in policy
    assert str(skill) not in policy
    assert "/run/.codex_home/skills/locate-anything/SKILL.md" in policy
    assert "global whole-house exploration task" in policy
    assert "initial room as only the starting area" in policy
    assert "beyond multiple doorways or rooms" in policy


def test_global_task_policy_can_disable_spatial_memory_rules() -> None:
    policy = global_task_agent_policy(
        {
            "benchmark_profile": "global_task",
            "global_task_policy": {"spatial_memory": False},
        }
    )

    assert "global whole-house exploration task" in policy
    assert "hab_register_spatial_junction" not in policy
    assert "branch_id" not in policy
    assert "blocked_close_audit_required" not in policy
    assert "closed=true" in policy


def test_global_task_policy_drops_grounding_rules_for_localnav_arm() -> None:
    skill = Path("/run/.codex_home/skills/global-navigation-learned-executor/SKILL.md")
    localnav_arm = {
        "movement": "localnav",
        "skill_mode": "native",
        "skill_name": "global-navigation-learned-executor",
        "tool_whitelist": [
            "hab_init_scene",
            "hab_turn",
            "hab_local_navigate",
            "hab_localnav_status",
            "hab_localnav_stop",
            "hab_close_session",
        ],
    }
    policy = global_task_agent_policy(
        {"benchmark_profile": "global_task"},
        native_skill_path=skill,
        arm_cfg=localnav_arm,
    )

    # The arm cannot ground or register junctions, so rules referencing those
    # tools (including the locate-anything reading requirement) must not fire.
    assert "locate-anything" not in policy
    assert "hab_register_spatial_junction" not in policy
    assert "hab_visual_ground_preview" not in policy
    assert "blocked_close_audit_required" not in policy
    # Lifecycle and mapless constraints still apply.
    assert "Call hab_init_scene exactly once" in policy
    assert "exactly one successful hab_close_session" in policy
    assert "Remain mapless" in policy
    assert "global whole-house exploration task" in policy


def test_global_task_prompt_is_an_instance_contract_without_hidden_gt() -> None:
    prompt = build_prompt(
        arm_name="default",
        arm_cfg={
            "movement": "visual_ground_preview_locate",
            "skill_mode": "native",
            "skill_name": "global-navigation",
            "_agent_backend": "codex",
        },
        instruction="Find the red test cube beside the blue test sphere.",
        scene="synthetic_scene_a",
        scene_dataset_config_file="/data/global.scene_dataset_config.json",
        spawn={
            "start_position": [1.0, 0.2, 3.0],
            "start_rotation": [0.0, 1.0, 0.0, 0.0],
            "sensor_height": 1.2,
        },
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "global_task" / "prompts",
    )
    assert "Execute this initialization call exactly once" in prompt
    assert prompt.count("hab_init_scene(scene=") == 1
    assert "Detector" in prompt and "success" in prompt
    assert "complete instruction" in prompt
    assert "report blocked rather than looping" in prompt
    assert "ground_truth" not in prompt
    assert "goal_position" not in prompt
    assert "scene_dataset_config_file" not in prompt
    assert "start_position" not in prompt
    assert "start_rotation" not in prompt
    assert "/data/global.scene_dataset_config.json" not in prompt
    assert "[1, 0.2, 3]" not in prompt
    assert "hab_register_spatial_junction" in prompt
    assert 'outcome="achieved"' in prompt
    assert "NAVIGATION SKILL (follow this)" not in prompt
    assert "native skill `/global-navigation`" in prompt


def test_localnav_prompt_uses_movement_common_override() -> None:
    prompt = build_prompt(
        arm_name="localnav_native_skill",
        arm_cfg={
            "movement": "localnav",
            "skill_mode": "native",
            "skill_name": "localnav-pointnav",
            "_agent_backend": "codex",
        },
        instruction="Find the blue test sphere in the synthetic side room.",
        scene="synthetic_scene_b",
        scene_dataset_config_file="/data/global.scene_dataset_config.json",
        spawn={
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "sensor_height": 0.4,
        },
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "global_task" / "prompts",
    )
    # The localnav common override keeps the global-task episode contract but
    # swaps the LocateAnything/spatial-junction paragraphs for the hop toolset.
    assert "Execute this initialization call exactly once" in prompt
    assert prompt.count("hab_init_scene(scene=") == 1
    assert "point-marked local hops" in prompt
    assert "LocateAnything" not in prompt
    assert "hab_register_spatial_junction" not in prompt
    assert "visual_nav_context" not in prompt
    assert "You do not start with a global map" in prompt
    assert "ONLY translation tool" in prompt
    assert "report blocked rather than looping" in prompt
    assert 'outcome="achieved"' in prompt
    assert "start_position" not in prompt
    assert "native skill `/localnav-pointnav`" in prompt


def test_canonical_prompt_paths_preserve_the_benchmark_specific_contract() -> None:
    arguments = {
        "arm_name": "localnav",
        "arm_cfg": {"movement": "localnav", "skill_mode": "none"},
        "instruction": "Find the toilet.",
        "scene": "test_scene",
        "scene_dataset_config_file": "/data/scene_dataset.json",
        "spawn": {},
        "workspace_root": REPO_ROOT,
    }
    expected = build_prompt(
        **arguments,
        prompts_dir=REPO_ROOT / "configs/benchmarks/global_task/prompts",
    )

    for unavailable in (
        Path("bench/config/global_task/prompts"),
        REPO_ROOT / "bench/config/global_task/prompts",
    ):
        with pytest.raises(FileNotFoundError, match="missing prompt template"):
            build_prompt(**arguments, prompts_dir=unavailable)
    assert "point-marked local hops" in expected
    assert "report blocked rather than looping" in expected


def test_global_navigation_skill_bundle_is_valid() -> None:
    report = lint_skill_root(REPO_ROOT / "skills")
    assert report.ok, report.errors
    names = {entry.name for entry in report.skills}
    assert {"global-navigation", "locate-anything"} <= names


def test_global_navigation_skill_relevance_policy_contract() -> None:
    root = REPO_ROOT / "skills" / "global-navigation"
    main = (root / "SKILL.md").read_text(encoding="utf-8")
    reference_paths = [
        root / "references" / "spatial-memory.md",
        root / "references" / "exploration-and-return.md",
        root / "references" / "entrance-search.md",
        root / "references" / "recovery-and-stop.md",
    ]
    assert len(main.encode("utf-8")) <= 6600
    for path in reference_paths:
        assert path.is_file()
        assert str(path.relative_to(root)) in main
    corpus = "\n".join([main, *(path.read_text() for path in reference_paths)])
    normalized = " ".join(corpus.split())
    exploration = " ".join(reference_paths[1].read_text().split())
    spatial = " ".join(reference_paths[0].read_text().split())

    rule_ids = re.findall(
        r"\*\*(GN-(?:LIFE|OBS|MEM|MOVE|ENTRY|TERM)-\d{3})\s+—", corpus
    )
    assert rule_ids
    assert len(rule_ids) == len(set(rule_ids))
    for required in (
        "GN-LIFE-001",
        "GN-OBS-002",
        "GN-MEM-001",
        "GN-MOVE-003",
        "GN-TERM-001",
        "GN-TERM-002",
    ):
        assert required in main

    assert "Current pixels outrank bounded memory" in normalized
    assert "Do not register only the intended choice" in normalized
    assert "Junction eviction or disappearance" in normalized
    assert "A threshold touch is not an exhaustion test" in normalized
    assert "target-relevant current route" in normalized
    assert "defer it in working state, but keep it in the frontier" in normalized
    assert (
        "do not approach an irrelevant interior object merely to clear follow-up"
        in normalized
    )
    assert "treat the branch operationally as approached" in normalized
    assert (
        "compact working frontier until pixels prove a real crossing or meaningful inspection"
        in normalized
    )
    assert "greatest useful progress toward the selected visible opening" in normalized
    assert (
        "Do not choose a shorter lower-edge candidate merely to preserve siblings"
        in normalized
    )
    assert (
        "do not repeat the same phrase, candidate, or point without new visual information"
        in normalized
    )
    assert "target itself or a target-relevant current route" in normalized
    assert "GN-ENTRY-010" in normalized
    assert "a more relevant local sibling" in exploration
    assert (
        "audit only when independent current evidence supports an accepted disposition"
        in exploration
    )
    assert "After any action that yields no new visual information" in exploration
    assert (
        "A guessed or visually mismatched room is not an accepted disposition"
        in spatial
    )
    assert "familiar public or starting-area context" in normalized
    assert "An aimless return is forbidden" in normalized
    assert "frontier_reminder" in normalized
    assert "long_move_with_local_frontier" in normalized
    assert (
        "the harness cannot warn about an opening that was never registered"
        in normalized
    )
    assert "post_entry_followup_required=true" in normalized
    assert "blocked_close_audit_required" in normalized
    assert "frontier_audit" in normalized
    assert "no_safe_passable_interior" in normalized
    assert "Any intervening non-close tool invalidates it" in normalized
    assert "closed=true" in normalized


def test_global_task_generator_sets_profile_and_arm_preserving_model(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "global_task"
    scene_dir = data_root / "scene_a"
    scene_dir.mkdir(parents=True)
    (scene_dir / "tasks.json").write_text(
        json.dumps(
            {
                "scene_id": "scene_a",
                "default_start_position": [1, 0.2, 2],
                "default_start_yaw": 0.5,
                "tasks": [
                    {
                        "task_id": "task_001",
                        "instruction": "Find the target.",
                        "start_position": "default",
                        "goal_position": [3, 0.2, 4],
                        "goal_radius": 0.4,
                        "geodesic_distance_gt": 5.0,
                        "ground_truth_path": [[1, 0.2, 2], [3, 0.2, 4]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps({"navmesh_instances": {"scene_a": {}}}), encoding="utf-8"
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "model": {"name": "custom-navigation-model"},
                "agents": {
                    "codex": {"extra_args": ["-c", 'model_reasoning_effort="low"']}
                },
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/global_task/prompts"),
                "arms": {
                    "default": {
                        "movement": "visual_point",
                        "environment": {"HAB_VISUAL_POINT_POINT_ONLY": "1"},
                        "tool_whitelist": [
                            "hab_init_scene",
                            "hab_visual_ground_preview",
                            "hab_close_session",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    build_configs(
        data_root=data_root,
        dataset_config=dataset,
        base_config_path=base,
        output_dir=output,
        instructions_dir=tmp_path / "instructions",
    )

    config = json.loads((output / "global-task-scene_a.json").read_text())
    manifest = json.loads((tmp_path / "instructions/scene_a.instructions.json").read_text())
    assert config["benchmark_profile"] == "global_task"
    assert config["model"] == {"name": "custom-navigation-model"}
    assert config["agents"]["codex"]["extra_args"] == [
        "-c", 'model_reasoning_effort="low"'
    ]
    # Converting task data preserves the selected method, including its tools.
    assert config["arms"] == json.loads(base.read_text())["arms"]
    assert config["instructions_file"] == str(tmp_path / "instructions/scene_a.instructions.json")
    assert not list(output.glob("*.instructions.json"))
    row = manifest["instructions"][0]
    assert row["spawn"]["start_position"] == [1.0, 0.2, 2.0]
    assert row["ground_truth"]["path_points"] == [[1, 0.2, 2], [3, 0.2, 4]]


def test_minimal_task_prompt_has_no_navigation_guidance():
    arm = {"prompt_mode": "minimal", "skill_mode": "none", "movement": "visual_point"}
    text = build_prompt(
        arm_name="no_skill",
        arm_cfg=arm,
        instruction="Find the synthetic yellow test prism.",
        scene="hidden",
        scene_dataset_config_file="hidden.json",
        spawn={"start_position": [1, 2, 3]},
        workspace_root=REPO_ROOT,
    )
    assert (
        text
        == "Find the synthetic yellow test prism.\nCall hab_close_session when finished.\n"
    )
    assert (
        global_task_agent_policy({"benchmark_profile": "global_task"}, arm_cfg=arm)
        == ""
    )
