from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.runtime.agents import get_agent_backend, mcp_env, selected_agent_config
from supernav.runtime.codex_agent import CodexAgent, CodexAgentProfile
from supernav.runtime.opencode_agent import OpenCodeAgent
from supernav.runtime.codex_agent import canonicalize_codex
from supernav.runtime.streams import session_id_from_events
from supernav.runtime.kimi_agent import KimiAgent, _export_kimi_session, _kimi_session_id_from_raw, canonicalize_kimi
from supernav.runtime.opencode_agent import canonicalize_opencode
from supernav.runtime.config import load_json, merged_arm
from supernav.evaluation.metrics import compute_metrics
from supernav.methods.navigation.prompts import build_prompt
from supernav.runtime.subagents import ensure_backend_subagent
from supernav.runtime.timing_summary import generate_timing_summary_artifacts
from supernav.evaluation.video.render import _collect_replay_frames, compressed_interval_s, synthetic_trace_events, trace_events
from supernav.evaluation.video.batch import run_make_video
from supernav.experiments.episode import make_run_id, run_one


def _test_provider():
    return {"id": "test_provider", "name": "Test Provider",
            "base_url": "https://provider.example/v1", "wire_api": "responses",
            "env_key": "AUTH_TOKEN", "requires_openai_auth": False}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def experiment_provider_env(monkeypatch):
    monkeypatch.delenv("HAB_BENCH_RELAY_BASE_URL", raising=False)
    monkeypatch.setenv("HAB_BENCH_RELAY_WIRE_API", "responses")
    monkeypatch.delenv("HAB_BENCH_RELAY_REQUEST_MAX_RETRIES", raising=False)
    monkeypatch.delenv("HAB_BENCH_RELAY_STREAM_MAX_RETRIES", raising=False)


@pytest.fixture
def subagent_source(tmp_path: Path) -> Path:
    source = tmp_path / "synthetic-subagent.md"
    source.write_text(
        "description: Synthetic navigation helper.\n\n"
        "Inspect the surroundings and turn to the designated intended target.\n",
        encoding="utf-8",
    )
    return source


def test_get_agent_backend_selects_supported_backends() -> None:
    assert isinstance(get_agent_backend("opencode"), OpenCodeAgent)
    assert isinstance(get_agent_backend("codex"), CodexAgent)
    assert isinstance(get_agent_backend("kimi"), KimiAgent)


def test_get_agent_backend_rejects_unknown_backend() -> None:
    try:
        get_agent_backend("missing")
    except ValueError as exc:
        assert "unknown agent" in str(exc)
    else:
        raise AssertionError("unknown backend should raise ValueError")


def test_selected_agent_config_uses_agents_dict_and_model_config() -> None:
    name, agent_cfg, model_cfg = selected_agent_config(
        {
            "agent": "codex",
            "model": {"name": "test-provider/test-model"},
            "agents": {"codex": {"command": "codex"}},
        }
    )

    assert name == "codex"
    assert agent_cfg == {"command": "codex"}
    assert model_cfg == {"name": "test-provider/test-model"}


def test_selected_agent_config_requires_named_agent_config() -> None:
    try:
        selected_agent_config({"agent": "codex", "agents": {"opencode": {}}})
    except ValueError as exc:
        assert "agents dict" in str(exc)
    else:
        raise AssertionError("selected agent must exist in agents dict")


def test_oracle_prompt_uses_visual_grounding_and_intermediate_targets() -> None:
    prompt = build_prompt(
        arm_name="oracle",
        arm_cfg={"movement": "oracle", "skill_file": None},
        instruction="Go to the sofa.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "main" / "prompts",
    )

    assert "hab_visual_local_navigate" in prompt
    assert "Grounding DINO" in prompt
    assert "visible in the latest front image" in prompt
    assert "visual inspection of those images is the primary success signal" in prompt
    assert (
        "close the session even if visual local navigation did not ground that object"
        in prompt
    )
    assert "short hab_forward steps" in prompt
    assert "not proof that the object is absent from the images" in prompt
    assert "Base this decision on the images first" in prompt
    assert "intermediate" in prompt
    assert "no_visual_grounding" in prompt
    assert "Do not use coordinates" in prompt
    assert "Do not call hab_navigate_wam" in prompt
    assert "hab_oracle_local_navigate" not in prompt
    assert "visible_nav_targets" not in prompt
    assert "target_ref" not in prompt
    assert "hab_oracle_local_map" not in prompt
    assert "target_label" not in prompt
    assert 'instruction="go to' not in prompt
    assert "instruction fallback" not in prompt
    assert "hab_navigate_wam(instruction=" not in prompt
    assert "There is no map, no list of objects" not in prompt


def test_visual_point_prompt_uses_image_refs_and_normalized_anchors() -> None:
    prompt = build_prompt(
        arm_name="visual_point",
        arm_cfg={"movement": "visual_point", "skill_file": None},
        instruction="Go to the sofa.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "main" / "prompts",
    )

    assert "hab_visual_point_navigate" in prompt
    assert "view" in prompt
    assert "Prefer view over image_ref" in prompt
    assert 'use view="front" even when it appears on the right or left side' in prompt
    assert "normalized image coordinates" in prompt
    assert 'hab_visual_point_navigate(view="front", point=[0.5,0.8])' in prompt
    assert "Use point only. Do not pass bbox in this arm." in prompt
    assert "For visible objects, click one precise point" in prompt
    assert "Use bbox for visible objects" not in prompt
    assert "Use point for floor patches" not in prompt
    assert "Do not call hab_visual_local_navigate" in prompt
    assert "hab_navigate_wam" in prompt
    assert "visual inspection of those images is the primary success signal" in prompt


def test_visual_ground_preview_prompt_requires_visual_first_semantic_proxy() -> None:
    prompt = build_prompt(
        arm_name="visual_ground_preview",
        arm_cfg={"movement": "visual_ground_preview", "skill_file": None},
        instruction="Go to the sofa.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "main" / "prompts",
    )

    assert "hab_visual_ground_preview" in prompt
    assert 'hab_visual_ground_preview(view="front", phrase="...")' in prompt
    assert "Prefer `view` over `image_ref`" in prompt
    assert 'use `view="front"` even when it appears on the right or left side' in prompt
    assert "Do not confuse an object on the right side of the front image" in prompt
    assert "visual-first semantic grounding" in prompt
    assert "Your own inspection of the latest `panorama_images`" in prompt
    assert "semantic proxy" in prompt
    assert (
        "If the final target is not clearly visible, use a visible proxy anchor first"
        in prompt
    )
    assert "the rug leading toward the seating area" in prompt
    assert "the open doorway on the right" in prompt
    assert "for a pillow goal, ground a visible bed" in prompt
    assert "Prefer phrases with at least one visible qualifier" in prompt
    assert "Use a bare noun such as `sofa` or `rug` only when" in prompt
    assert "blindly pass the user's final target phrase" in prompt
    assert "not as a blind search engine for the final instruction" in prompt
    assert (
        "default to a visible proxy anchor instead of asking LocateAnything for the final target"
        in prompt
    )
    assert (
        "Use LocateAnything's language strength by grounding a referring expression"
        in prompt
    )
    assert "retry `hab_visual_ground_preview` with a more specific phrase" in prompt
    assert (
        "retry with a more specific phrase before using `confirm_token` and `candidate_id`"
        in prompt
    )
    assert (
        "Do not repeatedly ask LocateAnything for the same final target phrase"
        in prompt
    )
    assert "confirm_token" in prompt
    assert "candidate_id" in prompt
    assert "finds exactly one reachable candidate" in prompt
    assert "finds multiple candidate objects" in prompt
    assert "If you abandon that preview for another action" in prompt
    assert "Prefer `hab_visual_ground_preview` as the primary navigation tool" in prompt
    assert (
        "Do not use repeated point hops as the primary way to explore rooms" in prompt
    )
    assert "micro-adjusting the viewpoint" in prompt
    assert "crossing a grounded doorway threshold" in prompt
    assert "no prior grounding failure is required" not in prompt
    assert "Treat them as peer tools" not in prompt
    assert "use its `point` with `hab_visual_point_navigate`" not in prompt
    assert "visual inspection of those images is the primary success signal" in prompt
    assert "Use this FIRST whenever you have a natural-language target" not in prompt
    assert "If you can describe the target with a concise visual phrase" not in prompt
    assert "There is no map, no list of objects" not in prompt


def test_make_run_id_includes_task_id_only_when_present() -> None:
    assert (
        make_run_id("visual_ground_preview", "sofa", None)
        == "visual_ground_preview_sofa"
    )
    assert (
        make_run_id(
            "visual_ground_preview",
            "synthetic-task-0003",
            None,
            "synthetic-task-0003",
        )
        == "visual_ground_preview_synthetic-task-0003"
    )
    assert (
        make_run_id("visual_ground_preview", "sofa", "2", "synthetic-task-0003")
        == "visual_ground_preview_synthetic-task-0003_sofa__r2"
    )


def test_run_one_dry_run_uses_task_scene_spawn_and_records_hidden_gt(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    out_dir = tmp_path / "runs"
    config_path.write_text(
        json.dumps(
            {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
                "workspace_root": str(REPO_ROOT),
                "output_dir": str(out_dir),
                "scene": "fallback_scene",
                "scene_dataset_config_file": "/tmp/fallback.scene_dataset_config.json",
                "spawn": {"x": 9.0, "z": 9.0, "yaw": 0.0},
                "bridge": {"host": "127.0.0.1", "port": 18911},
                "mcp": {
                    "transport": "stdio",
                    "sensor_depth": True,
                    "environment": {
                        "HAB_MCP_GATE_ENABLED": "0",
                        "NAV_LOCATE_ANYTHING_URL": "http://127.0.0.1:8915/ground",
                    },
                },
                "grounding_warmup": {
                    "image": "data/test_assets/hbao_tests/van-gogh-room.color.png",
                    "phrase": "bed",
                    "mode": "box",
                    "timeout_s": 120,
                },
                "agent": "opencode",
                "model": {"name": None},
                "agents": {"opencode": {"command": "opencode", "timeout_s": 1800}},
                "arms": {
                    "visual_ground_preview": {
                        "movement": "visual_ground_preview",
                        "grounding_warmup": True,
                        "skill_file": None,
                        "tool_whitelist": [
                            "hab_init_scene",
                            "hab_visual_ground_preview",
                            "hab_visual_point_navigate",
                            "hab_close_session",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    row = {
        "task_id": "synthetic-door-001",
        "slug": "door",
        "text": "go to the door",
        "scene": "0004_840011",
        "scene_dataset_config_file": "/data/scenes.scene_dataset_config.json",
        "spawn": {
            "start_position": [-1.0, 0.2, 6.0],
            "start_rotation": [0.0, -0.05, 0.0, 0.99],
            "sensor_height": 1.2,
        },
        "goal": {"instruction": "go to the door", "target": {"obj_id": 19}},
        "ground_truth": {"final_position": [3.0, 0.2, 1.5]},
    }

    result = run_one(
        SimpleNamespace(
            config=str(config_path),
            arm="visual_ground_preview",
            slug=row["slug"],
            instruction=row["text"],
            rep=None,
            run_id=None,
            task_id=row["task_id"],
            scene=row["scene"],
            scene_dataset_config_file=row["scene_dataset_config_file"],
            spawn=row["spawn"],
            metadata=json.dumps(row),
            ground_truth=row["ground_truth"],
            timeout_s=None,
            model="gpt-5.4",
            overwrite=False,
            dry_run=True,
        )
    )

    run_dir = Path(result["run_dir"])
    prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    warmup = json.loads(
        (run_dir / "locate_anything_warmup.json").read_text(encoding="utf-8")
    )

    assert (
        result["run_id"] == "visual_ground_preview_synthetic-door-001_door"
    )
    assert 'scene="0004_840011"' in prompt
    assert "fallback_scene" not in prompt
    assert "start_position=[-1, 0.2, 6]" in prompt
    assert "start_rotation=[0, -0.05, 0, 0.99]" in prompt
    assert "sensor_height=1.2" in prompt
    assert "final_position" not in prompt
    assert "obj_id" not in prompt
    assert run_json["task_id"] == row["task_id"]
    assert run_json["scene"] == row["scene"]
    assert run_json["spawn"] == row["spawn"]
    assert run_json["ground_truth"] == row["ground_truth"]
    assert run_json["model"] == {"name": "gpt-5.4"}
    assert run_json["grounding_warmup"]["status"] == "not_executed_dry_run"
    assert warmup["status"] == "not_executed_dry_run"


def test_aggregate_keeps_same_slug_separate_by_task_id(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_jsonl(
        runs_dir / "results.jsonl",
        [
            {
                "task_id": "task-a",
                "slug": "door",
                "arm": "visual_ground_preview",
                "success": True,
                "llm_turns": 2,
            },
            {
                "task_id": "task-b",
                "slug": "door",
                "arm": "visual_ground_preview",
                "success": False,
                "llm_turns": 4,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m", "supernav.evaluation.aggregate",
            "--runs-dir",
            str(runs_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads((runs_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["arms"]["visual_ground_preview"]["n"] == 2
    assert summary["arms"]["visual_ground_preview"]["agent_completion_rate"] == 0.5
    assert summary["cells"]["task-a/door/visual_ground_preview"]["agent_completion_rate"] == 1
    assert summary["cells"]["task-b/door/visual_ground_preview"]["agent_completion_rate"] == 0


def test_metrics_counts_visual_point_navigate_as_nav_leg(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "type": "tool_call",
                "name": "hab_visual_point_navigate",
                "input": {"image_ref": "pano:s:1:front", "point": [0.5, 0.8]},
            },
            {
                "type": "tool_result",
                "name": "hab_visual_point_navigate",
                "content": '{"status": "preview_ready", "confirm_token": "tok"}',
            },
            {
                "type": "tool_call",
                "name": "hab_visual_point_navigate",
                "input": {
                    "image_ref": "pano:s:1:front",
                    "point": [0.5, 0.8],
                    "confirm_token": "tok",
                },
            },
            {
                "type": "tool_result",
                "name": "hab_visual_point_navigate",
                "content": '{"status": "reached_visual_point", "steps_executed": 3}',
            },
            {"type": "tool_call", "name": "hab_close_session", "input": {}},
        ],
    )

    metrics = compute_metrics(canonical, arm="visual_point", slug="sofa")

    assert metrics["nav_legs"] == 1
    assert metrics["nav_steps_total"] == 3
    assert metrics["movement_tool_calls"] == 1
    assert metrics["tool_calls"]["hab_visual_point_navigate"] == 2


def test_metrics_counts_visual_ground_preview_confirm_as_nav_leg(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "type": "tool_call",
                "name": "hab_visual_ground_preview",
                "input": {"image_ref": "pano:s:1:front", "phrase": "sofa"},
            },
            {
                "type": "tool_result",
                "name": "hab_visual_ground_preview",
                "content": '{"status": "preview_ready", "confirm_token": "tok"}',
            },
            {
                "type": "tool_call",
                "name": "hab_visual_ground_preview",
                "input": {
                    "image_ref": "pano:s:1:front",
                    "confirm_token": "tok",
                    "candidate_id": 0,
                },
            },
            {
                "type": "tool_result",
                "name": "hab_visual_ground_preview",
                "content": '{"status": "reached_visual_ground_candidate", "steps_executed": 5}',
            },
            {"type": "tool_call", "name": "hab_close_session", "input": {}},
        ],
    )

    metrics = compute_metrics(canonical, arm="visual_ground_preview", slug="sofa")

    assert metrics["nav_legs"] == 1
    assert metrics["nav_steps_total"] == 5
    assert metrics["movement_tool_calls"] == 1
    assert metrics["tool_calls"]["hab_visual_ground_preview"] == 2


def test_visual_overlay_prompt_uses_obj_aliases_not_point_or_phrase_nav() -> None:
    prompt = build_prompt(
        arm_name="visual_overlay",
        arm_cfg={"movement": "visual_overlay", "skill_file": None},
        instruction="Go to the sofa.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "main" / "prompts",
    )

    assert "hab_visual_overlay_navigate" in prompt
    assert "start_position" in prompt
    assert "start_rotation" in prompt
    assert "hab_set_pose" not in prompt
    assert "target_alias" in prompt
    assert "overlay_objlist" in prompt
    assert "obj_<number>" in prompt
    assert "Never reuse an alias" in prompt
    assert "Prefer overlay navigation whenever a clear alias marks the target" in prompt
    assert "no suitable overlay alias is visible" in prompt
    assert "Do not call hab_visual_local_navigate" in prompt
    assert "hab_visual_point_navigate" in prompt
    assert "visual inspection of those images is the primary success signal" in prompt


def test_metrics_counts_visual_overlay_navigate_as_nav_leg(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "type": "tool_call",
                "name": "hab_visual_overlay_navigate",
                "input": {"image_ref": "pano:s:1:front", "target_alias": "obj_189"},
            },
            {
                "type": "tool_result",
                "name": "hab_visual_overlay_navigate",
                "content": '{"steps_executed": 4}',
            },
            {"type": "tool_call", "name": "hab_close_session", "input": {}},
        ],
    )

    metrics = compute_metrics(canonical, arm="visual_overlay", slug="sofa")

    assert metrics["nav_legs"] == 1
    assert metrics["nav_steps_total"] == 4
    assert metrics["movement_tool_calls"] == 1


def test_mcp_env_merges_arm_environment_for_overlay() -> None:
    _server, _python, env = mcp_env(
        workspace_root=REPO_ROOT,
        config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
            "mcp": {
                "environment": {"HAB_MCP_GATE_ENABLED": "0"},
            },
            "bridge": {"host": "127.0.0.1", "port": 18911},
        },
        arm_cfg={
            "tool_whitelist": ["hab_visual_overlay_navigate"],
            "environment": {"HAB_MCP_VISIBLE_TARGET_OVERLAYS": "1"},
        },
    )

    assert env["HAB_MCP_GATE_ENABLED"] == "0"
    assert env["HAB_MCP_VISIBLE_TARGET_OVERLAYS"] == "1"
    assert env["HAB_MCP_TOOL_WHITELIST"] == "hab_visual_overlay_navigate"


def test_visual_point_arm_enables_point_only_mode() -> None:
    cfg = {
        "environment": {"backend": "habitat"},
        "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "arms": {
            "base": {
                "movement": "visual_point",
                "environment": {"HAB_VISUAL_POINT_POINT_ONLY": "1"},
                "tool_whitelist": ["hab_init_scene", "hab_visual_point_navigate", "hab_close_session"],
            },
            "default": {"extends": "base", "skill_file": "skills/global-navigation-geo-based-executor/SKILL.md"},
        },
    }
    visual_point = merged_arm(cfg, "base")
    visual_point_skill = merged_arm(cfg, "default")

    assert visual_point["environment"]["HAB_VISUAL_POINT_POINT_ONLY"] == "1"
    assert visual_point_skill["environment"]["HAB_VISUAL_POINT_POINT_ONLY"] == "1"

    _server, _python, env = mcp_env(
        workspace_root=REPO_ROOT,
        config=cfg,
        arm_cfg=visual_point,
    )

    assert env["HAB_VISUAL_POINT_POINT_ONLY"] == "1"
    assert "hab_visual_point_navigate" in env["HAB_MCP_TOOL_WHITELIST"]


def test_oracle_visual_ground_preview_locate_skill_arm() -> None:
    cfg = {
        "environment": {"backend": "habitat"},
        "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "arms": {"default": {
            "movement": "visual_ground_preview_locate",
            "environment": {"HAB_VISUAL_POINT_POINT_ONLY": "1"},
            "tool_whitelist": ["hab_init_scene", "hab_visual_ground_preview", "hab_visual_point_navigate", "hab_close_session"],
            "skill_file": "skills/locate-anything/SKILL.md",
        }},
    }
    arm = merged_arm(cfg, "default")

    assert arm["environment"]["HAB_VISUAL_POINT_POINT_ONLY"] == "1"
    assert arm["skill_file"].endswith("skills/locate-anything/SKILL.md")

    _server, _python, env = mcp_env(
        workspace_root=REPO_ROOT,
        config=cfg,
        arm_cfg=arm,
    )

    assert env["HAB_VISUAL_POINT_POINT_ONLY"] == "1"
    assert "hab_visual_ground_preview" in env["HAB_MCP_TOOL_WHITELIST"]
    assert "hab_visual_point_navigate" in env["HAB_MCP_TOOL_WHITELIST"]

    prompt = build_prompt(
        arm_name="default",
        arm_cfg=arm,
        instruction="Go to the bed.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=REPO_ROOT / "configs" / "benchmarks" / "main" / "prompts",
    )

    assert "LocateAnything Phrase Grounding" in prompt
    assert 'view="front", phrase="the bed on the right side of the image"' in prompt
    assert "LocateAnything phrase grounding as the primary navigation tool" in prompt
    assert "bounded local adjustment" in prompt
    assert "Do not chain point hops as the primary exploration method" in prompt
    assert "two peer visual navigation tools" not in prompt
    assert "It does not require a prior LocateAnything failure" not in prompt
    assert "These tools are peers" not in prompt
    assert "Follow the LocateAnything skill below for phrase selection" in prompt
    assert "If multiple candidates appear, inspect the numbered overlay" in prompt
    assert "make sure the target is reasonably close and clear" in prompt
    assert "visible only in a right/left/back panorama image" in prompt
    assert "Short direct `hab_forward` steps are acceptable" not in prompt
    assert "retry with a more specific phrase" not in prompt
    assert "# Habitat-GS" not in prompt


def test_prompt_loader_falls_back_per_missing_template(tmp_path: Path) -> None:
    empty_prompts_dir = tmp_path / "prompts"
    empty_prompts_dir.mkdir()

    prompt = build_prompt(
        arm_name="oracle",
        arm_cfg={"movement": "oracle", "skill_file": None},
        instruction="Go to the sofa.",
        scene="gs_scene",
        scene_dataset_config_file="/tmp/interior.scene_dataset_config.json",
        spawn={"x": 1.0, "z": 2.0, "yaw": 0.5},
        workspace_root=REPO_ROOT,
        prompts_dir=empty_prompts_dir,
    )

    assert "hab_visual_local_navigate" in prompt
    assert "Grounding DINO" in prompt


def test_generate_codex_project_writes_isolated_config(
    tmp_path: Path, experiment_provider_env, subagent_source: Path,
) -> None:
    fake_home = tmp_path / "source_codex_home"
    fake_home.mkdir(parents=True)
    (fake_home / "config.toml").write_text(
        'model = "test-provider/test-model"\n\n'
        '[mcp_servers."habitat-gs"]\n'
        'command = "old"\n\n'
        '[mcp_servers."other"]\n'
        'command = "other"\n',
        encoding="utf-8",
    )
    old_codex_home = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(fake_home)
    codex_agent = CodexAgent()
    try:
        project_dir = codex_agent.prepare_project(
            run_dir=tmp_path,
            workspace_root=REPO_ROOT,
            config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
                "mcp": {
                    "transport": "stdio",
                },
                "bridge": {"host": "127.0.0.1", "port": 18911},
                "subagents": {
                    "see_turn": {
                        "source_file": str(subagent_source),
                        "codex": {"name": "see_turn", "model": "gpt-5.3-codex-spark"},
                    }
                },
            },
            arm_cfg={
                "tool_whitelist": ["hab_see", "hab_close_session"],
                "_benchmark_process_environment": {
                    "HAB_MCP_INIT_DEFAULTS_JSON": '{"start_position":[1,2,3]}'
                },
            },
            agent_cfg={
                "command": "codex", "provider_mode": "experiment", "provider": _test_provider(),
                "mcp_tools_approval": "approve",
                "subagent": "see_turn",
            },
            model_cfg={"name": "test-provider/test-model"},
        )
    finally:
        if old_codex_home is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = old_codex_home

    config_text = (project_dir / ".codex_home" / "config.toml").read_text(
        encoding="utf-8"
    )
    assert 'model = "test-provider/test-model"' not in config_text
    assert 'model_provider = "test_provider"' in config_text
    assert '[mcp_servers."other"]' not in config_text
    assert 'command = "old"' not in config_text
    assert '[mcp_servers."habitat-gs"]' in config_text
    assert f'command = {json.dumps(sys.executable)}' in config_text
    assert 'default_tools_approval_mode = "approve"' in config_text
    assert "required = true" in config_text
    assert 'args = ["-m", "supernav", "mcp", "--transport", "stdio"]' in config_text
    assert 'HAB_MCP_TOOL_WHITELIST = "hab_see,hab_close_session"' in config_text
    assert 'env_vars = ["HAB_MCP_INIT_DEFAULTS_JSON"]' in config_text
    assert "start_position" not in config_text
    assert codex_agent._benchmark_process_environment == {
        "HAB_MCP_INIT_DEFAULTS_JSON": '{"start_position":[1,2,3]}'
    }
    assert (project_dir / "AGENTS.md").is_file()
    subagent_text = (project_dir / ".codex" / "agents" / "see_turn.toml").read_text(
        encoding="utf-8"
    )
    assert 'name = "see_turn"' in subagent_text
    assert 'model = "gpt-5.3-codex-spark"' in subagent_text
    assert 'sandbox_mode = "workspace-write"' in subagent_text
    assert 'developer_instructions = """' in subagent_text
    assert (
        "Inspect the surroundings and turn to the designated intended target."
        in subagent_text
    )


def test_codex_user_provider_preserves_user_config_and_auth(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        'model_provider = "test_user_provider"\n\n'
        "[model_providers.test_user_provider]\n"
        'name = "Test User Provider"\n'
        'base_url = "https://test-user-provider.invalid"\n'
        'wire_api = "responses"\n\n'
        '[mcp_servers."other"]\n'
        'command = "other"\n',
        encoding="utf-8",
    )
    (source_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-user-provider-secret"}),
        encoding="utf-8",
    )
    old_codex_home = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(source_home)
    try:
        project_dir = CodexAgent().prepare_project(
            run_dir=tmp_path / "run",
            workspace_root=REPO_ROOT,
            config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
                "mcp": {
                    "transport": "stdio",
                },
                "bridge": {"host": "127.0.0.1", "port": 18911},
            },
            arm_cfg={"tool_whitelist": []},
            agent_cfg={"provider_mode": "user"},
            model_cfg={"name": "gpt-5.4"},
        )
    finally:
        if old_codex_home is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = old_codex_home

    codex_home = project_dir / ".codex_home"
    config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'model_provider = "test_user_provider"' in config_text
    assert 'base_url = "https://test-user-provider.invalid"' in config_text
    assert '[mcp_servers."other"]' in config_text
    assert json.loads((codex_home / "auth.json").read_text()) == {
        "OPENAI_API_KEY": "test-user-provider-secret"
    }
    metadata = json.loads((project_dir / "codex_project.json").read_text())
    assert metadata["provider"] == {"mode": "user"}
    assert metadata["auth_linked"] is True


def test_codex_explicit_experiment_provider_without_user_auth(
    tmp_path: Path,
    experiment_provider_env,
) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        'model = "test-user-provider-model"\n'
        'model_provider = "test_user_provider"\n'
        'model_reasoning_effort = "high"\n\n'
        "[model_providers.test_user_provider]\n"
        'name = "Test User Provider"\n'
        'base_url = "https://test-user-provider.invalid"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = true\n",
        encoding="utf-8",
    )
    (source_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-user-provider-secret"}),
        encoding="utf-8",
    )
    old_codex_home = os.environ.get("CODEX_HOME")
    old_auth_token = os.environ.get("AUTH_TOKEN")
    os.environ["CODEX_HOME"] = str(source_home)
    os.environ["AUTH_TOKEN"] = "experiment-secret-must-not-be-persisted"
    try:
        project_dir = CodexAgent().prepare_project(
            run_dir=tmp_path / "run",
            workspace_root=REPO_ROOT,
            config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
                "mcp": {
                    "transport": "stdio",
                },
                "bridge": {"host": "127.0.0.1", "port": 18911},
            },
            arm_cfg={"tool_whitelist": []},
            agent_cfg={"provider_mode": "experiment", "provider": _test_provider()},
            model_cfg={"name": "gpt-5.4"},
        )
    finally:
        if old_codex_home is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = old_codex_home
        if old_auth_token is None:
            os.environ.pop("AUTH_TOKEN", None)
        else:
            os.environ["AUTH_TOKEN"] = old_auth_token

    codex_home = project_dir / ".codex_home"
    config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'model_provider = "test_provider"' in config_text
    assert '[model_providers."test_provider"]' in config_text
    assert 'base_url = "https://provider.example/v1"' in config_text
    assert 'wire_api = "responses"' in config_text
    assert 'env_key = "AUTH_TOKEN"' in config_text
    assert "requires_openai_auth = false" in config_text
    assert 'model_reasoning_effort = "high"' in config_text
    assert "test_user_provider" not in config_text
    assert "test-user-provider-model" not in config_text
    assert "test-user-provider.invalid" not in config_text
    assert not (codex_home / "auth.json").exists()

    metadata = json.loads((project_dir / "codex_project.json").read_text())
    assert metadata["provider"] == {
        "mode": "experiment",
        "id": "test_provider",
        "name": "Test Provider",
        "base_url": "https://provider.example/v1",
        "wire_api": "responses",
        "env_key": "AUTH_TOKEN",
        "requires_openai_auth": False,
    }
    assert metadata["auth_linked"] is False
    for path in project_dir.rglob("*"):
        if path.is_file():
            assert b"experiment-secret-must-not-be-persisted" not in path.read_bytes()


@pytest.mark.parametrize("base_url", [None, " \t "])
@pytest.mark.parametrize("agent_type", [CodexAgent, CodexAgentProfile])
def test_codex_experiment_provider_requires_explicit_endpoint(
    tmp_path: Path, monkeypatch, base_url, agent_type,
) -> None:
    if base_url is None:
        monkeypatch.delenv("HAB_BENCH_RELAY_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("HAB_BENCH_RELAY_BASE_URL", base_url)
    monkeypatch.setenv("AUTH_TOKEN", "test-experiment-token")
    agent = agent_type()

    with pytest.raises(ValueError, match="experiment.*HAB_BENCH_RELAY_BASE_URL"):
        agent.validate_environment(agent_cfg={"provider_mode": "experiment"})

    def unexpected_launch(*_args, **_kwargs):
        pytest.fail("Codex must validate the relay endpoint before launching")

    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    with pytest.raises(ValueError, match="experiment.*HAB_BENCH_RELAY_BASE_URL"):
        agent.run(
            prompt="Reply with OK.",
            run_dir=tmp_path,
            project_dir=tmp_path / "codex_project",
            agent_cfg={"provider_mode": "experiment"},
            model_cfg={"name": "gpt-6-astra"},
            timeout_s=30,
        )
    assert not (tmp_path / "raw.jsonl").exists()
    assert not (tmp_path / "stderr.log").exists()


@pytest.mark.parametrize("agent_cfg", [{}, {"provider_mode": "user"}])
def test_codex_official_user_provider_requires_no_relay_configuration(
    tmp_path: Path, monkeypatch, agent_cfg,
) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model_provider = "openai"\n')
    auth = {"OPENAI_API_KEY": "test-openai-key"}
    (source_home / "auth.json").write_text(json.dumps(auth))
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.delenv("HAB_BENCH_RELAY_BASE_URL", raising=False)
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    agent = CodexAgent()
    project_dir = agent.prepare_project(
        run_dir=tmp_path / "run",
        workspace_root=REPO_ROOT,
        config={
            "environment": {"backend": "habitat"},
            "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
            "mcp": {"transport": "stdio"},
            "bridge": {"host": "127.0.0.1", "port": 18911},
        },
        arm_cfg={"tool_whitelist": []},
        agent_cfg=agent_cfg,
        model_cfg={"name": "gpt-6-astra"},
    )
    agent.validate_environment(agent_cfg=agent_cfg)
    config_text = (project_dir / ".codex_home/config.toml").read_text()
    assert 'model_provider = "openai"' in config_text
    assert "model_providers" not in config_text
    assert json.loads((project_dir / ".codex_home/auth.json").read_text()) == auth
    metadata = json.loads((project_dir / "codex_project.json").read_text())
    assert metadata["provider"] == {"mode": "user"}


def test_codex_experiment_provider_requires_auth_token_before_launch(
    tmp_path: Path,
    monkeypatch,
    experiment_provider_env,
) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    agent = CodexAgent()
    run_dir = tmp_path / "run"
    project_dir = agent.prepare_project(
        run_dir=run_dir,
        workspace_root=REPO_ROOT,
        config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
            "mcp": {
                "transport": "stdio",
            },
            "bridge": {"host": "127.0.0.1", "port": 18911},
        },
        arm_cfg={"tool_whitelist": []},
        agent_cfg={"provider_mode": "experiment", "provider": _test_provider()},
        model_cfg={"name": "gpt-5.4"},
    )
    launched = False

    def unexpected_launch(*_args, **_kwargs):
        nonlocal launched
        launched = True
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    try:
        agent.run(
            prompt="Reply with OK.",
            run_dir=run_dir,
            project_dir=project_dir,
            agent_cfg={"provider_mode": "experiment", "provider": _test_provider()},
            model_cfg={"name": "gpt-5.4"},
            timeout_s=30,
        )
    except ValueError as exc:
        assert "AUTH_TOKEN" in str(exc)
        assert "test_provider" in str(exc)
    else:
        raise AssertionError("missing AUTH_TOKEN should block Codex before launch")

    assert launched is False
    assert not (run_dir / "raw.jsonl").exists()
    assert not (run_dir / "stderr.log").exists()


def test_codex_experiment_provider_removes_stale_run_local_user_auth(
    tmp_path: Path,
    experiment_provider_env,
) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        'model_provider = "test_user_provider"\n', encoding="utf-8"
    )
    (source_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-user-provider-secret"}),
        encoding="utf-8",
    )
    old_codex_home = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(source_home)
    agent = CodexAgent()
    run_dir = tmp_path / "run"
    kwargs = {
        "run_dir": run_dir,
        "workspace_root": REPO_ROOT,
        "config": {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
            "mcp": {
                "transport": "stdio",
            },
            "bridge": {"host": "127.0.0.1", "port": 18911},
        },
        "arm_cfg": {"tool_whitelist": []},
        "model_cfg": {"name": "gpt-5.4"},
    }
    try:
        project_dir = agent.prepare_project(
            **kwargs,
            agent_cfg={"provider_mode": "user"},
        )
        auth_path = project_dir / ".codex_home/auth.json"
        assert auth_path.exists()

        project_dir = agent.prepare_project(**kwargs, agent_cfg={"provider_mode": "experiment", "provider": _test_provider()})
    finally:
        if old_codex_home is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = old_codex_home

    assert not auth_path.exists()
    metadata = json.loads((project_dir / "codex_project.json").read_text())
    assert metadata["provider"]["mode"] == "experiment"
    assert metadata["auth_linked"] is False


def test_codex_experiment_provider_is_authoritative_after_extra_args(
    tmp_path: Path,
    monkeypatch,
    experiment_provider_env,
) -> None:
    source_home = tmp_path / "source_codex_home"
    source_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("AUTH_TOKEN", "experiment-secret")
    captured_commands = []

    def capture_launch(command, **_kwargs):
        captured_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", capture_launch)
    for agent_type in (CodexAgent, CodexAgentProfile):
        agent = agent_type()
        run_dir = tmp_path / agent_type.__name__
        project_dir = agent.prepare_project(
            run_dir=run_dir,
            workspace_root=REPO_ROOT,
            config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
                "mcp": {
                    "transport": "stdio",
                },
                "bridge": {"host": "127.0.0.1", "port": 18911},
            },
            arm_cfg={"tool_whitelist": []},
            agent_cfg={"provider_mode": "experiment", "provider": _test_provider()},
            model_cfg={"name": "gpt-5.4"},
        )
        agent.run(
            prompt="Reply with OK.",
            run_dir=run_dir,
            project_dir=project_dir,
            agent_cfg={
                "command": "codex", "provider_mode": "experiment", "provider": _test_provider(),
                "extra_args": ["-c", 'model_provider="test_user_provider"'],
            },
            model_cfg={"name": "gpt-5.4"},
            timeout_s=30,
        )

    assert len(captured_commands) == 2
    for command in captured_commands:
        assert command[-5:] == [
            "-c",
            'model_provider="test_provider"',
            "-c",
            (
                'model_providers.test_provider={name="Test Provider",'
                'base_url="https://provider.example/v1",'
                'wire_api="responses",env_key="AUTH_TOKEN",'
                "requires_openai_auth=false}"
            ),
            "Reply with OK.",
        ]


def test_generate_opencode_project_writes_subagent_file(
    tmp_path: Path, subagent_source: Path,
) -> None:
    project_dir = OpenCodeAgent().prepare_project(
        run_dir=tmp_path,
        workspace_root=REPO_ROOT,
        config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
            "mcp": {
                "transport": "stdio",
            },
            "bridge": {"host": "127.0.0.1", "port": 18911},
            "subagents": {
                "see_turn": {
                    "source_file": str(subagent_source),
                    "opencode": {
                        "name": "see-turn",
                        "model": "test-provider/test-model",
                        "temperature": 0.1,
                        "reasoningEffort": "medium",
                    },
                }
            },
        },
        arm_cfg={"tool_whitelist": ["hab_see", "hab_close_session"]},
        agent_cfg={"command": "opencode", "subagent": "see_turn"},
        model_cfg={"name": "test-provider/test-model"},
    )

    subagent_text = (project_dir / ".opencode" / "agents" / "see-turn.md").read_text(
        encoding="utf-8"
    )
    assert "mode: subagent" in subagent_text
    assert "model: test-provider/test-model" in subagent_text
    assert "reasoningEffort: medium" in subagent_text
    assert "  write: true" in subagent_text
    assert "  edit: true" in subagent_text
    assert (
        "Inspect the surroundings and turn to the designated intended target."
        in subagent_text
    )


def test_generate_kimi_project_writes_run_local_home_and_mcp_config(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_kimi"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        'default_model = "kimi-code/kimi-for-coding"\n',
        encoding="utf-8",
    )
    (source_home / "mcp.json").write_text(
        json.dumps({"mcpServers": {"old": {"url": "http://127.0.0.1:1/mcp"}}}),
        encoding="utf-8",
    )

    project_dir = KimiAgent().prepare_project(
        run_dir=tmp_path / "run",
        workspace_root=REPO_ROOT,
        config={
                "environment": {"backend": "habitat"},
                "prompts_dir": str(REPO_ROOT / "configs/benchmarks/main/prompts"),
            "mcp": {
                "transport": "streamable-http",
            },
            "bridge": {"host": "127.0.0.1", "port": 8911},
        },
        arm_cfg={"tool_whitelist": ["hab_init_scene", "hab_close_session"]},
        agent_cfg={"command": "kimi", "source_home": str(source_home)},
        model_cfg={"name": "kimi-code/kimi-for-coding"},
    )

    assert project_dir == tmp_path / "run" / "kimi_project"
    assert (project_dir / "kimi_home" / ".kimi-code" / "config.toml").is_file()
    assert (project_dir / "AGENTS.md").is_file()
    config_text = (project_dir / "kimi_home" / ".kimi-code" / "config.toml").read_text(
        encoding="utf-8"
    )
    assert 'pattern = "mcp__habitat-gs__*"' in config_text
    mcp = json.loads(
        (project_dir / "kimi_home" / ".kimi-code" / "mcp.json").read_text(
            encoding="utf-8"
        )
    )
    assert mcp["mcpServers"]["habitat-gs"]["url"] == "http://127.0.0.1:0/mcp"
    meta = json.loads((project_dir / "kimi_project.json").read_text(encoding="utf-8"))
    assert meta["workspace_root"] == str(REPO_ROOT)
    assert meta["arm_cfg"]["tool_whitelist"] == ["hab_init_scene", "hab_close_session"]


def test_canonicalize_kimi_parses_wire_tool_calls_and_results(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    wire = tmp_path / "kimi_wire.jsonl"
    out = tmp_path / "canonical.jsonl"
    raw.write_text('{"type":"metadata"}\n', encoding="utf-8")
    _write_jsonl(
        wire,
        [
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "tool.call",
                    "toolCallId": "tool_1",
                    "name": "mcp__habitat-gs__hab_init_scene",
                    "args": {"scene": "0006_840125"},
                },
            },
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "tool.result",
                    "toolCallId": "tool_1",
                    "result": {
                        "output": '{"session_id": "75cc3ac5-70c9-46fa-b5f6-063ee332a095", "images": ["/tmp/frame.png"]}'
                    },
                },
            },
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "tool.call",
                    "toolCallId": "tool_2",
                    "name": "mcp__habitat-gs__hab_close_session",
                    "args": {},
                },
            },
        ],
    )

    events = canonicalize_kimi(raw, out)

    assert events[0] == {
        "type": "tool_call",
        "name": "hab_init_scene",
        "input": {"scene": "0006_840125"},
        "timestamp": None,
        "raw_index": 0,
    }
    assert events[1]["type"] == "tool_result"
    assert events[1]["name"] == "hab_init_scene"
    assert "75cc3ac5-70c9-46fa-b5f6-063ee332a095" in events[1]["content"]
    assert events[2]["name"] == "hab_close_session"
    assert out.is_file()


def test_kimi_session_id_is_extracted_from_resume_hint(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(
        raw,
        [
            {"role": "assistant", "content": "done"},
            {
                "role": "meta",
                "type": "session.resume_hint",
                "session_id": "session_04b106ec-b955-43a5-851e-0a181c284267",
                "command": "kimi -r session_04b106ec-b955-43a5-851e-0a181c284267",
            },
        ],
    )

    assert (
        _kimi_session_id_from_raw(raw) == "session_04b106ec-b955-43a5-851e-0a181c284267"
    )


def test_export_kimi_session_passes_explicit_session_id(tmp_path: Path) -> None:
    fake_kimi = tmp_path / "fake_kimi.py"
    argv_path = tmp_path / "argv.json"
    fake_kimi.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f"printf '%s\\n' \"$@\" > {str(argv_path)!r}",
                'while [ "$1" != "" ]; do',
                '  if [ "$1" = "--output" ]; then',
                "    shift",
                "    printf 'PK\\005\\006\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000\\000' > \"$1\"",
                "    exit 0",
                "  fi",
                "  shift",
                "done",
                "exit 1",
            ]
        ),
        encoding="utf-8",
    )
    fake_kimi.chmod(0o755)

    rc = _export_kimi_session(
        command=str(fake_kimi),
        session_id="session_04b106ec-b955-43a5-851e-0a181c284267",
        kimi_home=tmp_path,
        project_dir=tmp_path,
        export_path=tmp_path / "export.zip",
        stderr_path=tmp_path / "stderr.log",
        timeout_s=10,
        env={},
    )

    assert rc == 0
    argv = argv_path.read_text(encoding="utf-8").splitlines()
    assert argv[:3] == [
        "export",
        "session_04b106ec-b955-43a5-851e-0a181c284267",
        "--output",
    ]
    assert "--yes" in argv
    assert "--no-include-global-log" in argv


def test_canonicalize_kimi_parses_wire_reasoning_with_timestamps(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    wire = tmp_path / "kimi_wire.jsonl"
    out = tmp_path / "canonical.jsonl"
    raw.write_text('{"type":"metadata"}\n', encoding="utf-8")
    _write_jsonl(
        wire,
        [
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "content.part",
                    "part": {
                        "type": "think",
                        "think": "I should inspect the panorama first.",
                    },
                },
                "time": 1783606910685,
            },
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "content.part",
                    "part": {
                        "type": "text",
                        "text": "I will use the hallway as a proxy.",
                    },
                },
                "time": 1783606911685,
            },
        ],
    )

    events = canonicalize_kimi(raw, out)

    assert events == [
        {
            "type": "assistant_reasoning",
            "text": "I should inspect the panorama first.",
            "timestamp": 1783606910685,
            "raw_index": 0,
        },
        {
            "type": "assistant_text",
            "text": "I will use the hallway as a proxy.",
            "timestamp": 1783606911685,
            "raw_index": 1,
        },
    ]


def test_canonicalize_kimi_parses_stream_json_tool_calls_and_results(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "canonical.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "role": "assistant",
                "content": "I will initialize.",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "tool_1",
                        "function": {
                            "name": "mcp__habitat-gs__hab_init_scene",
                            "arguments": '{"scene":"0006_840125"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool_1",
                "content": json.dumps(
                    [
                        {
                            "type": "text",
                            "text": (
                                '{"session_id": "44580152-a906-45dc-a59e-c9ef1cec1204", '
                                '"panorama_image_paths": ["/tmp/front.png"]}'
                            ),
                        },
                        {"type": "text", "text": "Front view"},
                    ]
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "tool_2",
                        "function": {
                            "name": "mcp__habitat-gs__hab_close_session",
                            "arguments": "{}",
                        },
                    }
                ],
            },
        ],
    )

    events = canonicalize_kimi(raw, out)
    metrics = compute_metrics(out, arm="visual_ground_preview", slug="kimi")

    assert [event["type"] for event in events] == [
        "assistant_text",
        "tool_call",
        "tool_result",
        "tool_call",
    ]
    assert events[1]["name"] == "hab_init_scene"
    assert events[1]["input"] == {"scene": "0006_840125"}
    assert events[2]["name"] == "hab_init_scene"
    assert events[2]["content"].splitlines()[0].startswith('{"session_id"')
    assert events[3]["name"] == "hab_close_session"
    assert metrics["success"] is True


def test_kimi_replay_synthetic_trace_and_frames_use_manifest_order() -> None:
    events = [
        {
            "type": "tool_call",
            "name": "hab_init_scene",
            "input": {"scene": "0006_840125"},
            "timestamp": None,
        },
        {
            "type": "tool_result",
            "name": "hab_init_scene",
            "content": '{"session_id": "44580152-a906-45dc-a59e-c9ef1cec1204"}',
            "timestamp": None,
        },
        {
            "type": "tool_call",
            "name": "hab_visual_ground_preview",
            "input": {"view": "front", "phrase": "the hallway"},
            "timestamp": None,
        },
        {
            "type": "tool_result",
            "name": "hab_visual_ground_preview",
            "content": (
                '{"status": "reached_visual_ground_candidate", '
                '"steps_executed": 14, "movement_frame_count": 14, '
                '"candidate_count": 1, "selected_candidate_id": 0}'
            ),
            "timestamp": None,
        },
        {
            "type": "assistant_text",
            "text": "Done. I reached the door.",
            "timestamp": None,
        },
    ]
    manifest = {
        "entries": [
            {
                "tool": "hab_init_scene",
                "frames": [
                    "/tmp/pano_front_step000001_color_sensor.png",
                    "/tmp/pano_right_step000001_color_sensor.png",
                    "/tmp/pano_back_step000001_color_sensor.png",
                    "/tmp/pano_left_step000001_color_sensor.png",
                ],
            },
            {
                "tool": "hab_visual_ground_preview",
                "frames": [
                    "/tmp/step000004_color_sensor.png",
                    "/tmp/step000005_color_sensor.png",
                ],
            },
        ]
    }

    trace = synthetic_trace_events(events, start_ts=0.0, end_ts=12.0)
    frames = _collect_replay_frames(manifest, synthetic_time=True)

    assert trace[0][0] == 0.0
    assert trace[0][2].startswith("> hab_init_scene")
    assert trace[1][0] == 0.8
    assert trace[1][2] == "< hab_init_scene: done"
    assert trace[2][0] == 4.0
    assert "the hallway" in trace[2][2]
    assert trace[3][0] == 4.8
    assert "steps=14" in trace[3][2]
    assert any(
        row[1] == "text" and "Done. I reached the door." in row[2] for row in trace
    )
    assert frames[0][0] == 0.7
    assert frames[0][1] == "hab_init_scene"
    assert frames[0][2].startswith("pano_concat::")
    assert frames[1][0] == 4.7
    assert frames[1][1] == "hab_visual_ground_preview"
    assert frames[2][0] == 7.7
    assert frames[2][1] == "hab_visual_ground_preview"


def test_timing_summary_falls_back_to_canonical_jsonl(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "canonical.jsonl",
        [
            {
                "type": "assistant_text",
                "text": "I will initialize.",
                "timestamp": "2026-07-09T00:00:00Z",
            },
            {
                "type": "tool_call",
                "name": "hab_init_scene",
                "input": {"scene": "0006_840125"},
                "timestamp": "2026-07-09T00:00:01Z",
            },
        ],
    )

    result = generate_timing_summary_artifacts(tmp_path)

    assert result is not None
    assert (tmp_path / "timing_summary.json").is_file()
    assert (tmp_path / "timing_summary.svg").is_file()
    summary = json.loads((tmp_path / "timing_summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["source_kind"] == "canonical"
    assert summary["summary"]["counts"]["model_output"] == 1
    assert summary["summary"]["counts"]["tool_call"] == 1


def test_ensure_backend_subagent_uses_inline_description_when_source_starts_with_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "helper.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "description: Inspect and rotate.\n\nStep 1: Look around.\n",
        encoding="utf-8",
    )

    target = ensure_backend_subagent(
        backend="codex",
        project_dir=tmp_path / "project",
        workspace_root=tmp_path,
        config={"subagents": {"inspect": {"source_file": str(source)}}},
        agent_cfg={"subagent": "inspect"},
        model_cfg={},
    )

    assert target is not None
    text = target.read_text(encoding="utf-8")
    assert 'description = "Inspect and rotate."' in text
    assert 'developer_instructions = """' in text
    assert "Step 1: Look around." in text


def test_canonicalize_opencode_preserves_existing_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "canonical.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "tool_use",
                "timestamp": 1,
                "part": {
                    "type": "tool",
                    "tool": "mcp__habitat-gs__hab_see",
                    "state": {"input": {"session_id": "s"}, "output": "ok"},
                },
            },
            {"type": "text", "timestamp": 2, "part": {"type": "text", "text": "done"}},
        ],
    )

    events = canonicalize_opencode(raw, out)

    assert events[0]["type"] == "tool_call"
    assert events[0]["name"] == "hab_see"
    assert events[1]["type"] == "tool_result"
    assert events[2]["type"] == "assistant_text"
    assert out.is_file()


def test_canonicalize_codex_maps_tool_and_text_events(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "canonical.jsonl"
    timing = tmp_path / "codex_session.jsonl"
    _write_jsonl(
        raw,
        [
            {"type": "response.reasoning.delta", "timestamp": 1, "delta": "think"},
            {
                "timestamp": "2026-06-24T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoned plan"}],
                },
            },
            {
                "timestamp": "2026-06-24T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "abc",
                },
            },
            {
                "type": "mcp_tool_call",
                "timestamp": 2,
                "name": "habitat-gs.hab_see",
                "arguments": {"session_id": "s"},
            },
            {
                "type": "mcp_tool_result",
                "timestamp": 3,
                "name": "habitat-gs.hab_see",
                "content": "image saved",
            },
            {
                "type": "item.started",
                "item": {
                    "id": "item_1",
                    "type": "mcp_tool_call",
                    "server": "habitat-gs",
                    "tool": "hab_visual_ground_preview",
                    "arguments": {"view": "front", "phrase": "the sofa"},
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "mcp_tool_call",
                    "server": "habitat-gs",
                    "tool": "hab_visual_ground_preview",
                    "arguments": {"view": "front", "phrase": "the sofa"},
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"status":"reached_visual_ground_candidate","overlay_image":"/tmp/overlay.png"}',
                            }
                        ]
                    },
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "habitat-gs.hab_init_scene",
                    "error": {"message": "user cancelled MCP tool call"},
                },
            },
            {
                "type": "message",
                "timestamp": 4,
                "role": "assistant",
                "content": "finished",
            },
        ],
    )
    _write_jsonl(
        timing,
        [
            {
                "timestamp": "2026-06-24T00:00:04Z",
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "exec"},
            },
            {
                "timestamp": "2026-06-24T00:00:05Z",
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "invocation": {"tool": "hab_visual_ground_preview"},
                },
            },
            {
                "timestamp": "2026-06-24T00:00:05Z",
                "type": "response_item",
                "payload": {"type": "custom_tool_call_output"},
            },
        ],
    )

    events = canonicalize_codex(raw, out, timing_path=timing)

    assert events[0]["type"] == "assistant_reasoning"
    assert events[1]["type"] == "assistant_reasoning"
    assert events[1]["text"] == "reasoned plan"
    assert events[2]["type"] == "assistant_reasoning"
    assert events[2]["text"] == "reasoning (encrypted)"
    assert events[3]["type"] == "tool_call"
    assert events[3]["name"] == "hab_see"
    assert events[3]["input"] == {"session_id": "s"}
    assert events[4]["type"] == "tool_result"
    assert events[4]["content"] == "image saved"
    assert events[5]["type"] == "tool_call"
    assert events[5]["name"] == "hab_visual_ground_preview"
    assert events[5]["input"] == {"view": "front", "phrase": "the sofa"}
    assert events[5]["timestamp"] == "2026-06-24T00:00:04Z"
    assert events[6]["type"] == "tool_result"
    assert "overlay.png" in events[6]["content"]
    assert events[6]["timestamp"] == "2026-06-24T00:00:05Z"
    assert events[7]["type"] == "tool_result"
    assert events[7]["content"] == "user cancelled MCP tool call"
    assert events[8]["type"] == "assistant_text"


def test_canonicalize_codex_merges_agent_timeline_for_replay(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "canonical.jsonl"
    timing = tmp_path / "codex_session.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "I found the target."},
            },
            {
                "type": "item.started",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "hab_close_session",
                    "arguments": {"session_id": "s"},
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "hab_close_session",
                    "result": {
                        "content": [{"type": "text", "text": '{"closed":true}'}]
                    },
                },
            },
        ],
    )
    _write_jsonl(
        timing,
        [
            {
                "timestamp": "2026-07-12T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Confirm visually."}],
                },
            },
            {
                "timestamp": "2026-07-12T00:00:02Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "I found the target."},
            },
            {
                "timestamp": "2026-07-12T00:00:03Z",
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "exec"},
            },
            {
                "timestamp": "2026-07-12T00:00:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "invocation": {"tool": "hab_close_session"},
                },
            },
        ],
    )

    events = canonicalize_codex(raw, out, timing_path=timing)

    message = next(event for event in events if event.get("type") == "assistant_text")
    assert message["timestamp"] == "2026-07-12T00:00:02Z"
    reasoning = next(
        event for event in events if event.get("type") == "assistant_reasoning"
    )
    assert reasoning["text"] == "Confirm visually."
    assert reasoning["timestamp"] == "2026-07-12T00:00:01Z"
    trace = trace_events(events)
    assert any(
        kind == "text" and "I found the target." in text for _, kind, text, _ in trace
    )
    assert any(
        kind == "reasoning" and "Confirm visually." in text
        for _, kind, text, _ in trace
    )


def test_canonicalize_codex_uses_namespaced_function_call_start_time(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "canonical.jsonl"
    timing = tmp_path / "codex_session.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "item.started",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "hab_init_scene",
                    "arguments": {"scene": "scene"},
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "hab_init_scene",
                    "result": {"content": [{"type": "text", "text": "{}"}]},
                },
            },
        ],
    )
    _write_jsonl(
        timing,
        [
            {
                "timestamp": "2026-07-12T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "hab_init_scene",
                    "namespace": "mcp__habitat_gs",
                    "call_id": "call-1",
                },
            },
            {
                "timestamp": "2026-07-12T00:00:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "call_id": "call-1",
                    "invocation": {"tool": "hab_init_scene"},
                },
            },
        ],
    )

    events = canonicalize_codex(raw, out, timing_path=timing)

    assert events[0]["timestamp"] == "2026-07-12T00:00:03Z"
    assert events[1]["timestamp"] == "2026-07-12T00:00:04Z"


def test_session_id_prefers_closed_session_after_reinitialization() -> None:
    events = [
        {
            "type": "tool_result",
            "name": "hab_init_scene",
            "content": '{"session_id":"11111111-1111-1111-1111-111111111111"}',
        },
        {
            "type": "tool_result",
            "name": "hab_init_scene",
            "content": '{"session_id":"22222222-2222-2222-2222-222222222222"}',
        },
        {
            "type": "tool_result",
            "name": "hab_close_session",
            "content": '{"session_id":"22222222-2222-2222-2222-222222222222"}',
        },
    ]
    assert session_id_from_events(events) == "22222222-2222-2222-2222-222222222222"


def test_accelerated_replay_caps_only_long_intervals() -> None:
    assert compressed_interval_s(0.25, accelerated=True, max_wait_s=1.0) == 0.25
    assert compressed_interval_s(8.0, accelerated=True, max_wait_s=1.0) == 1.0
    assert compressed_interval_s(8.0, accelerated=False, max_wait_s=1.0) == 8.0


def test_batch_video_forwards_accelerated_options(tmp_path: Path, capsys) -> None:
    rc = run_make_video(
        tmp_path / "supernav.evaluation.video.render.sh",
        "run",
        tmp_path / "runs",
        None,
        tmp_path,
        True,
        accelerated=True,
        max_wait_s=1.5,
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert str(tmp_path / "data" / "nav_artifacts") in output
    assert "--accelerated --max-wait-s 1.5" in output


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_get_agent_backend_selects_supported_backends()
        test_get_agent_backend_rejects_unknown_backend()
        test_selected_agent_config_uses_agents_dict_and_model_config()
        test_selected_agent_config_requires_named_agent_config()
        test_generate_codex_project_writes_isolated_config(Path(tmp) / "project")
        test_generate_opencode_project_writes_subagent_file(
            Path(tmp) / "opencode_project"
        )
        test_ensure_backend_subagent_uses_inline_description_when_source_starts_with_prefix(
            Path(tmp) / "inline_subagent"
        )
        test_canonicalize_opencode_preserves_existing_contract(Path(tmp) / "opencode")
        test_canonicalize_codex_maps_tool_and_text_events(Path(tmp) / "codex")


def test_minimal_codex_project_excludes_inherited_instructions_and_tools(tmp_path, monkeypatch):
    import tomllib
    home = tmp_path / "source"
    home.mkdir()
    (home / "config.toml").write_text('developer_instructions = "EXPLORE EVERYTHING"\n[mcp_servers.other]\ncommand = "other"\n')
    monkeypatch.setenv("CODEX_HOME", str(home))
    project = CodexAgent().prepare_project(
        run_dir=tmp_path / "run", workspace_root=REPO_ROOT,
        config={"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"), "benchmark_profile": "global_task", "mcp": {}},
        arm_cfg={"skill_mode": "none", "prompt_mode": "minimal", "tool_whitelist": ["hab_turn"]},
        agent_cfg={"provider_mode": "user"}, model_cfg={"name": "gpt-5.6-terra"},
    )
    cfg = tomllib.loads((project / ".codex_home/config.toml").read_text())
    assert cfg["project_doc_max_bytes"] == 0
    assert "developer_instructions" not in cfg
    assert set(cfg["mcp_servers"]) == {"habitat-gs"}
    assert cfg["features"]["shell_tool"] is False
    assert cfg["features"]["plugins"] is False
    assert (project / "AGENTS.md").read_text() == ""
    assert not (project / ".codex_home/skills").exists()
