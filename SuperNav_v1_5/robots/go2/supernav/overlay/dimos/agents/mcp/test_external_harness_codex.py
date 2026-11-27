# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any

from dimos.agents.mcp.external_harness_codex import (
    DEFAULT_MCP_SMOKE_REQUIRED_TOOLS,
    codex_goal_prompt,
    codex_mcp_smoke_log_result,
    codex_mcp_smoke_prompt,
    prepare_codex_harness_run,
    run_codex_mcp_smoke_gate,
    write_run_local_codex_config,
    write_shareable_run_summary,
)


def test_write_run_local_codex_config_keeps_only_dimos_mcp(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    codex_home = tmp_path / "run" / "codex_home"
    source_home.mkdir()
    codex_home.mkdir(parents=True)
    (source_home / "config.toml").write_text(
        "\n".join(
            [
                'model_provider = "local_openai"',
                'model = "gpt-test"',
                'model_reasoning_effort = "xhigh"',
                'preferred_auth_method = "apikey"',
                "",
                "[model_providers.local_openai]",
                'name = "local_openai"',
                'base_url = "https://example.invalid/v1"',
                'wire_api = "responses"',
                'env_key = "sk-do-not-copy"',
                'bearer_token_env_var = "OPENAI_API_KEY"',
                "",
                "[mcp_servers.personal]",
                'url = "http://127.0.0.1:1/mcp"',
                "",
                "[features]",
                "hooks = true",
                "memories = true",
            ]
        ),
        encoding="utf-8",
    )

    config_path = write_run_local_codex_config(
        codex_home=codex_home,
        source_codex_home=source_home,
        mcp_url="http://127.0.0.1:9990/mcp",
        model_reasoning_effort="low",
    )
    text = config_path.read_text(encoding="utf-8")

    assert "[mcp_servers.dimos]" in text
    assert 'url = "http://127.0.0.1:9990/mcp"' in text
    assert "[mcp_servers.personal]" not in text
    assert "sk-do-not-copy" not in text
    assert 'bearer_token_env_var = "OPENAI_API_KEY"' in text
    assert 'model_reasoning_effort = "low"' in text
    assert 'model_reasoning_effort = "xhigh"' not in text
    assert "hooks = false" in text
    assert "memories = false" in text


def test_prepare_codex_harness_run_writes_prompt_audit_and_auth(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "auth.json").write_text("{}", encoding="utf-8")
    (source_home / "installation_id").write_text("install-test", encoding="utf-8")
    (source_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="找到房门口并运动到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
    )

    assert run.workspace.is_dir()
    assert run.codex_home.is_dir()
    assert (run.codex_home / "auth.json").exists()
    assert (run.codex_home / "installation_id").exists()
    prompt = run.prompt_path.read_text(encoding="utf-8")
    assert "create_goal" in prompt
    assert prompt.startswith("Your only goal is: 找到房门口并运动到门口\n")
    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert audit["env"]["CODEX_HOME"] == str(run.codex_home)
    assert audit["mcp_url"] == "http://127.0.0.1:9990/mcp"
    assert audit["mode"] == "readonly"
    assert audit["tool_profile"]["name"] == "readonly_inspection"
    assert audit["tool_profile"]["tool_scope"] == "readonly"
    assert audit["codex_mcp_bypass_enabled"] is False
    assert audit["image_delivery_gate"] == "requires_direct_mcp_image_content"
    assert audit["model_visible_image_content"] is True
    assert audit["drive_go2_skill_path"] == str(run.drive_go2_skill_path)
    assert audit["drive_go2_workspace_skill_path"] == str(run.drive_go2_workspace_skill_path)
    assert audit["codex_runtime_config"]["model_reasoning_effort"] == "medium"
    assert audit["codex_runtime_config"]["reasoning_effort_inherited_from_source"] is False
    assert audit["time_budgets"]["codex_exec_timeout_s"] == 1800.0
    assert run.drive_go2_skill_path is not None
    assert run.drive_go2_workspace_skill_path is not None
    assert (
        "relative_move` is a DimOS skill backed by local planning"
        in run.drive_go2_skill_path.read_text(encoding="utf-8")
    )
    assert (run.workspace / ".agents" / "skills" / "drive-go2" / "SKILL.md").exists()
    assert "codex exec" in audit["shell_command"]
    assert f"< {run.prompt_path}" in audit["shell_command"]
    summary = json.loads(run.shareable_summary_path.read_text(encoding="utf-8"))
    assert summary["tool_profile"]["name"] == "readonly_inspection"
    assert "codex_home" not in json.dumps(summary["paths"])
    assert "auth.json" not in json.dumps(summary)
    assert "install-test" not in run.shareable_summary_path.read_text(encoding="utf-8")
    assert run.shareable_summary_markdown_path.exists()


def test_prepare_codex_harness_run_authority_proxy_mode_points_codex_at_proxy(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="找到房门口并运动到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        proxy_url="http://127.0.0.1:9992/mcp",
        authority_scope="doorway search in cleared office",
        authority_area="office",
        authority_duration_s=300.0,
        model_reasoning_effort="low",
        service_tier="priority",
        codex_exec_timeout_s=2400.0,
        mcp_smoke_timeout_s=90.0,
        authority_allowed_movement_tools=("relative_move", "rotate_in_place"),
        max_movement_calls=8,
    )

    config = run.config_path.read_text(encoding="utf-8")
    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert 'url = "http://127.0.0.1:9992/mcp"' in config
    assert 'service_tier = "priority"' in config
    assert run.mcp_url == "http://127.0.0.1:9992/mcp"
    assert run.upstream_mcp_url == "http://127.0.0.1:9990/mcp"
    assert run.proxy_command is not None
    assert "authority-proxy" in run.proxy_command
    assert "--dangerously-bypass-approvals-and-sandbox" in run.command
    assert audit["mode"] == "authority-proxy"
    assert audit["tool_profile"]["name"] == "vision_odom_local_motion"
    assert "Go2 harness tool profile" in run.prompt_path.read_text(encoding="utf-8")
    assert audit["mcp_smoke_gate"]["required"] is True
    assert audit["mcp_smoke_gate"]["shell_fallback_forbidden"] is True
    assert audit["mcp_smoke_gate"]["model_reasoning_effort"] == "low"
    assert 'model_reasoning_effort="low"' in audit["mcp_smoke_command"]
    assert "mcp-smoke.jsonl" in audit["mcp_smoke_shell_command"]
    smoke_prompt = run.mcp_smoke_prompt_path.read_text(encoding="utf-8")
    assert "Do not use shell" in smoke_prompt
    assert "dimos mcp" in smoke_prompt
    assert "mcp__dimos__observe" in smoke_prompt
    assert "mcp__dimos__rotate_in_place" in smoke_prompt
    assert "mcp__dimos__go2_situation" not in smoke_prompt
    assert "go2_apply_graph_mutations" not in run.mcp_smoke_required_tool_names
    assert "mcp__dimos__go2_apply_graph_mutations" not in smoke_prompt
    assert audit["authority_envelope"]["scope"] == "doorway search in cleared office"
    assert audit["authority_envelope"]["allowed_movement_tools"] == [
        "relative_move",
        "rotate_in_place",
    ]
    assert audit["authority_envelope"]["max_movement_calls"] == 8
    assert audit["authority_envelope"]["tool_profile"] == "vision_odom_local_motion"
    assert audit["codex_runtime_config"]["model_reasoning_effort"] == "low"
    assert audit["codex_runtime_config"]["service_tier"] == "priority"
    assert audit["time_budgets"]["codex_exec_timeout_s"] == 2400.0
    assert audit["time_budgets"]["mcp_smoke_timeout_s"] == 90.0
    assert audit["time_budgets"]["authority_duration_s"] == 300.0
    assert audit["battery_gate"]["min_start_percent"] == 35.0
    assert "timeout --preserve-status 2400s codex exec" in audit["shell_command"]


def test_prepare_pixelgoal_agent_sets_blocking_tool_timeout_and_smoke_tools(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "pixelgoal",
        source_codex_home=source_home,
        task="走到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        prompt_mode="long_instruction",
        authority_allowed_movement_tools=("pixelgoal_navigate",),
    )

    config = run.config_path.read_text(encoding="utf-8")
    prompt = run.prompt_path.read_text(encoding="utf-8")
    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert "tool_timeout_sec = 320" in config
    assert {"observe_surround", "pixelgoal_status"}.issubset(
        run.mcp_smoke_required_tool_names
    )
    assert "pixelgoal_stop" not in run.mcp_smoke_required_tool_names
    assert "stop_movement" not in run.mcp_smoke_required_tool_names
    assert run.prompt_mode == "long_instruction"
    assert "go2_list_graph_nodes" not in run.mcp_smoke_required_tool_names
    assert prompt.startswith("Your only goal is: 走到门口\n")
    assert "Use the run-local `global-navigation-pixelgoal` skill" in prompt
    assert ".agents/skills/go2-pixelgoal/SKILL.md" in prompt
    assert ".agents/skills/global-navigation-pixelgoal/SKILL.md" in prompt
    assert "task objective above" in prompt
    assert "Do not call get_goal immediately after create_goal" in prompt
    assert "# Go2 PixelGoal Operating Notes" not in prompt
    assert "# Drive Go2 Operating Notes" not in prompt
    assert "Go2 harness tool profile" not in prompt
    assert run.drive_go2_skill_path is None
    assert run.drive_go2_workspace_skill_path is None
    assert run.go2_graph_memory_skill_path is not None
    assert run.go2_2d_slam_skill_path is None
    assert run.go2_graph_slam_navigation_skill_path is None
    assert run.go2_pixelgoal_workspace_skill_path is not None
    assert run.global_navigation_pixelgoal_workspace_skill_path is not None
    skill = run.go2_pixelgoal_workspace_skill_path.read_text(encoding="utf-8")
    assert skill.startswith("---\nname: go2-pixelgoal\n")
    assert "large stable vertical surface" in skill
    assert "farthest clearly visible safe" in skill
    assert "floor-wall boundary" in skill
    assert "Never select glass" in skill
    assert "exactly one `pixelgoal_navigate" in skill
    assert "missing or stale view" in skill
    assert "For a find/search task" in skill
    assert "do not call `server_status`" in skill
    assert (run.workspace / ".agents" / "skills" / "go2-pixelgoal" / "SKILL.md").exists()
    global_skill = run.global_navigation_pixelgoal_workspace_skill_path.read_text(encoding="utf-8")
    assert global_skill.startswith("---\nname: global-navigation-pixelgoal\n")
    assert "`go2-pixelgoal/SKILL.md`" in global_skill
    assert "`pixelgoal_navigate" in global_skill
    assert "hab_" not in global_skill
    references = run.global_navigation_pixelgoal_workspace_skill_path.parent / "references"
    assert {path.name for path in references.iterdir()} == {
        "entrance-search.md",
        "exploration-and-return.md",
        "recovery-and-stop.md",
    }
    assert "hab_" not in "".join(path.read_text(encoding="utf-8") for path in references.iterdir())
    assert not (run.workspace / ".agents" / "skills" / "drive-go2").exists()
    assert audit["drive_go2_skill_path"] is None
    assert audit["go2_pixelgoal_workspace_skill_path"] == str(
        run.go2_pixelgoal_workspace_skill_path
    )
    assert audit["global_navigation_pixelgoal_workspace_skill_path"] == str(
        run.global_navigation_pixelgoal_workspace_skill_path
    )


def test_prepare_codex_harness_run_defaults_mcp_smoke_timeout_for_slow_relays(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="go to the window",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert run.mcp_smoke_timeout_s == 300.0
    assert audit["time_budgets"]["mcp_smoke_timeout_s"] == 300.0
    assert "timeout --preserve-status 300s codex exec" in audit["mcp_smoke_shell_command"]


def test_prepare_codex_harness_run_preserves_provider_base_url(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text(
        "\n".join(
            [
                'model_provider = "crs"',
                'model = "gpt-test"',
                "",
                "[model_providers.crs]",
                'name = "crs"',
                'base_url = "https://api.yangyifan18.com/openai"',
                'wire_api = "responses"',
                "requires_openai_auth = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="go to the window",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
    )

    config = run.config_path.read_text(encoding="utf-8")
    assert 'base_url = "https://api.yangyifan18.com/openai"' in config
    assert 'wire_api = "responses"' in config


def test_prepare_codex_harness_run_authority_proxy_uses_promoted_venv_dimos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DIMOS_VENV", str(tmp_path / "promoted-venv"))
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="找到房门口并运动到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert run.proxy_command is not None
    assert run.proxy_command[:3] == [
        str(tmp_path / "promoted-venv" / "bin" / "dimos"),
        "mcp",
        "authority-proxy",
    ]
    assert run.proxy_command[:3] == audit["proxy_command"][:3]
    assert "uv run dimos mcp authority-proxy" not in audit["proxy_shell_command"]


def test_prepare_codex_harness_run_direct_motion_profile_omits_relative_move(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="找到房门口并运动到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        tool_profile="vision_odom_direct_motion",
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert audit["tool_profile"]["name"] == "vision_odom_direct_motion"
    assert audit["tool_profile"]["control"] == "direct"
    assert audit["authority_envelope"]["allowed_movement_tools"] == [
        "drive",
        "full_lie_down",
        "lie_down",
        "rotate_in_place",
    ]
    assert "relative_move" not in audit["authority_envelope"]["allowed_movement_tools"]
    assert "relative_move" not in run.mcp_smoke_required_tool_names
    assert "mcp__dimos__relative_move" not in run.mcp_smoke_prompt_path.read_text(encoding="utf-8")
    assert "--authority-tool relative_move" not in audit["proxy_shell_command"]


def test_go2_navigation_profiles_smoke_only_requires_edge_tools(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    for profile in ("mapping_allowed", "full_navigation_allowed"):
        run = prepare_codex_harness_run(
            run_root=tmp_path / "runs" / profile,
            source_codex_home=source_home,
            task="navigate with Go2 edge tools",
            mcp_url="http://127.0.0.1:9990/mcp",
            mode="authority-proxy",
            tool_profile=profile,
        )
        required = run.mcp_smoke_required_tool_names
        assert "navigate_with_text" not in required
        assert "return_to_task_start" not in required

        log_text = "\n".join(
            [
                _mcp_tool_completed("observe"),
                _mcp_tool_completed("go2_robot_status"),
                _mcp_tool_completed("motion_safety_status"),
                _mcp_tool_completed("last_motion_diagnostics"),
                "DIMOS_MCP_SMOKE: PASS tools="
                + ",".join(f"mcp__dimos__{tool}" for tool in required),
            ]
        )
        ok, _, missing, _ = codex_mcp_smoke_log_result(
            log_text,
            server_name="dimos",
            required_tools=required,
        )
        assert ok is True
        assert missing == []


def test_prepare_codex_harness_run_posture_profile_defaults_to_posture_only_tools(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "posture",
        source_codex_home=source_home,
        task="让机器狗完全趴下",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        tool_profile="posture_shutdown",
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert audit["tool_profile"]["name"] == "posture_shutdown"
    assert audit["authority_envelope"]["allowed_movement_tools"] == [
        "full_lie_down",
        "lie_down",
    ]
    assert audit["authority_envelope"]["tool_profile"] == "posture_shutdown"
    assert "--authority-tool full_lie_down" in audit["proxy_shell_command"]
    assert "--authority-tool relative_move" not in audit["proxy_shell_command"]
    smoke_prompt = run.mcp_smoke_prompt_path.read_text(encoding="utf-8")
    assert "mcp__dimos__full_lie_down" in smoke_prompt
    assert "mcp__dimos__lie_down" in smoke_prompt
    assert "mcp__dimos__relative_move" not in smoke_prompt


def test_prepare_codex_harness_run_can_mark_mcp_smoke_not_required(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="找到房门口并运动到门口",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        mcp_smoke_required=False,
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert run.mcp_smoke_required is False
    assert audit["mcp_smoke_required"] is False
    assert audit["mcp_smoke_gate"]["required"] is False
    assert "drive" in audit["authority_envelope"]["allowed_movement_tools"]
    assert "--authority-tool drive" in audit["proxy_shell_command"]


def test_prepare_codex_harness_run_posture_startup_profile_uses_narrow_start_tools(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "startup",
        source_codex_home=source_home,
        task="让机器狗站起来准备测试",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        tool_profile="posture_startup",
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert audit["tool_profile"]["name"] == "posture_startup"
    assert audit["authority_envelope"]["allowed_movement_tools"] == [
        "balance_stand",
        "stand_up",
    ]
    assert audit["authority_envelope"]["tool_profile"] == "posture_startup"
    assert "--tool-profile posture_startup" in audit["proxy_shell_command"]
    assert "--authority-tool balance_stand" in audit["proxy_shell_command"]
    assert "--authority-tool stand_up" in audit["proxy_shell_command"]
    assert "--authority-tool relative_move" not in audit["proxy_shell_command"]
    smoke_prompt = run.mcp_smoke_prompt_path.read_text(encoding="utf-8")
    assert "mcp__dimos__balance_stand" in smoke_prompt
    assert "mcp__dimos__stand_up" in smoke_prompt
    assert "mcp__dimos__relative_move" not in smoke_prompt
    assert "Do not call movement or posture tools" in smoke_prompt


def test_prepare_codex_harness_run_rejects_authority_profile_with_no_movement_tools(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    try:
        prepare_codex_harness_run(
            run_root=tmp_path / "runs" / "readonly-authority",
            source_codex_home=source_home,
            task="read status",
            mcp_url="http://127.0.0.1:9990/mcp",
            mode="authority-proxy",
            tool_profile="readonly_inspection",
        )
    except ValueError as exc:
        assert "grants no movement tools" in str(exc)
    else:
        raise AssertionError("readonly profile unexpectedly prepared movement authority")


def test_prepare_codex_harness_run_rejects_proxy_url_outside_proxy_mode(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()

    try:
        prepare_codex_harness_run(
            run_root=tmp_path / "runs" / "door",
            source_codex_home=source_home,
            task="read status",
            mcp_url="http://127.0.0.1:9990/mcp",
            proxy_url="http://127.0.0.1:9992/mcp",
        )
    except ValueError as exc:
        assert "proxy_url is only valid" in str(exc)
    else:
        raise AssertionError("prepare_codex_harness_run accepted proxy_url in readonly mode")


def test_codex_goal_prompt_puts_only_goal_first() -> None:
    prompt = codex_goal_prompt(task="巡航到门口")

    assert prompt.startswith("Your only goal is: 巡航到门口\n")
    assert "call the built-in create_goal tool" in prompt
    assert "Task command:" not in prompt
    assert "Task objective for create_goal:" not in prompt


def test_codex_goal_prompt_embeds_drive_go2_skill_for_mcp_only_runs() -> None:
    prompt = codex_goal_prompt(task="寻找房门口", mode="authority-proxy")

    assert "Embedded drive-go2 operating skill" in prompt
    assert ".agents/skills/drive-go2/SKILL.md" in prompt
    assert "MCP-only runs do not need shell/file access" in prompt
    assert "`relative_move` is a DimOS skill backed by local planning" in prompt
    assert "undershoot, drift in yaw, or be clipped by Go2" in prompt
    assert "`stand_up` then `balance_stand`" in prompt
    assert "Do not request broad" in prompt
    assert "Go2 已开启碰撞保护与障碍规避" in prompt
    assert "停止工具不向 agent 开放" in prompt
    assert "Use compact situation first" in prompt


def test_codex_goal_prompt_composes_pixelgoal_and_long_instruction_when_enabled() -> None:
    prompt = codex_goal_prompt(
        task="寻找房门口",
        mode="authority-proxy",
        prompt_mode="long_instruction",
        pixelgoal_enabled=True,
    )

    assert prompt.startswith("Your only goal is: 寻找房门口\n")
    assert "Use the run-local `global-navigation-pixelgoal` skill" in prompt
    assert ".agents/skills/go2-pixelgoal/SKILL.md" in prompt
    assert ".agents/skills/global-navigation-pixelgoal/SKILL.md" in prompt
    assert "hallway beside the wooden cabinets" in prompt
    assert "target merely appearing is not success" in prompt
    assert "do not move closer unless asked" not in prompt
    assert "moment-by-moment human tactical hints" in prompt
    assert "Embedded go2-graph-memory-hierarchy operating skill" in prompt
    assert ".agents/skills/go2-graph-memory-hierarchy/SKILL.md" in prompt
    assert "Embedded go2-2d-slam operating skill" in prompt
    assert "Embedded go2-graph-slam-navigation operating skill" in prompt
    assert "GO2 LONG-INSTRUCTION MODE" in prompt
    assert "go2_apply_graph_mutations" in prompt
    assert "drive-go2" not in prompt
    assert "# Drive Go2 Operating Notes" not in prompt
    assert "# Go2 PixelGoal Operating Notes" not in prompt
    assert "farthest clearly visible patch" not in prompt


def test_codex_goal_prompt_long_instruction_embeds_graph_memory_policy() -> None:
    prompt = codex_goal_prompt(
        task="find the kitchen",
        mode="authority-proxy",
        prompt_mode="long_instruction",
    )

    assert "Embedded drive-go2 operating skill" in prompt
    assert "Embedded go2-graph-memory-hierarchy operating skill" in prompt
    assert ".agents/skills/go2-graph-memory-hierarchy/SKILL.md" in prompt
    assert "Embedded go2-2d-slam operating skill" in prompt
    assert ".agents/skills/go2-2d-slam/SKILL.md" in prompt
    assert "Embedded go2-graph-slam-navigation operating skill" in prompt
    assert ".agents/skills/go2-graph-slam-navigation/SKILL.md" in prompt
    assert "Embedded go2-graph-memory-hierarchy references" in prompt
    assert "references/route-fusion.md" in prompt
    assert "go2_apply_graph_mutations" in prompt
    assert "2d_slam_map.status" in prompt
    assert "2d_slam_map.graph_nodes" in prompt
    assert "GO2 LONG-INSTRUCTION MODE" in prompt
    assert "Long-instruction graph-memory tools are explicitly allowed" in prompt
    assert "Before the first movement action" in prompt
    assert "After every successful movement or turn" in prompt
    assert "go2_start_navigation_episode" in prompt
    assert "go2_save_relocalization_bundle" in prompt
    assert "hidden map context" in prompt


def test_prepare_codex_harness_run_long_instruction_writes_graph_memory_artifacts(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    initial_slam_status_path = tmp_path / "initial-go2-slam-status.json"
    initial_slam_status_path.write_text('{"schema_version": 1}', encoding="utf-8")
    initial_relocalization_bundle = tmp_path / "initial-relocalization"
    initial_relocalization_bundle.mkdir()

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "long",
        source_codex_home=source_home,
        task="find the kitchen",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        prompt_mode="long_instruction",
        initial_slam_status_path=initial_slam_status_path,
        initial_relocalization_bundle=initial_relocalization_bundle,
    )

    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert run.prompt_mode == "long_instruction"
    assert audit["prompt_mode"] == "long_instruction"
    assert audit["graph_memory_path"] == str(run.graph_memory_path)
    assert audit["slam_status_path"] == str(run.slam_status_path)
    assert audit["initial_slam_status_path"] == str(initial_slam_status_path.resolve())
    assert audit["initial_relocalization_bundle"] == str(
        initial_relocalization_bundle.resolve()
    )
    assert audit["go2_graph_memory_skill_path"] == str(run.go2_graph_memory_skill_path)
    assert audit["go2_2d_slam_skill_path"] == str(run.go2_2d_slam_skill_path)
    assert audit["go2_graph_slam_navigation_skill_path"] == str(
        run.go2_graph_slam_navigation_skill_path
    )
    assert run.graph_memory_path == run.run_root / "logs" / "go2-graph-memory.json"
    assert run.slam_status_path == run.run_root / "logs" / "go2-slam-status.json"
    assert "go2_list_graph_nodes" in run.mcp_smoke_required_tool_names
    assert "go2_apply_graph_mutations" in run.mcp_smoke_required_tool_names
    assert "go2_relocalization_status" in run.mcp_smoke_required_tool_names
    assert "go2_navigation_map_context" in run.mcp_smoke_required_tool_names
    assert "go2_start_navigation_episode" in run.mcp_smoke_required_tool_names
    assert "go2_save_relocalization_bundle" in run.mcp_smoke_required_tool_names
    smoke_prompt = run.mcp_smoke_prompt_path.read_text(encoding="utf-8")
    assert "mcp__dimos__go2_list_graph_nodes" in smoke_prompt
    assert "mcp__dimos__go2_apply_graph_mutations" in smoke_prompt
    assert "mcp__dimos__go2_start_navigation_episode" in smoke_prompt
    assert "mcp__dimos__go2_save_relocalization_bundle" in smoke_prompt
    assert "go2_list_graph_nodes" in smoke_prompt
    assert "go2_relocalization_status" in smoke_prompt
    assert run.initial_slam_status_path == initial_slam_status_path.resolve()
    assert run.initial_relocalization_bundle == initial_relocalization_bundle.resolve()
    assert run.go2_graph_memory_skill_path is not None
    assert run.go2_graph_memory_workspace_skill_path is not None
    assert run.go2_2d_slam_skill_path is not None
    assert run.go2_2d_slam_workspace_skill_path is not None
    assert run.go2_graph_slam_navigation_skill_path is not None
    assert run.go2_graph_slam_navigation_workspace_skill_path is not None
    assert run.go2_graph_memory_skill_path.exists()
    assert run.go2_graph_memory_workspace_skill_path.exists()
    assert run.go2_2d_slam_skill_path.exists()
    assert run.go2_2d_slam_workspace_skill_path.exists()
    assert run.go2_graph_slam_navigation_skill_path.exists()
    assert run.go2_graph_slam_navigation_workspace_skill_path.exists()
    assert (run.go2_graph_memory_skill_path.parent / "references" / "node-schema.md").exists()
    assert (
        run.go2_graph_slam_navigation_skill_path.parent / "references" / "route-fusion.md"
    ).exists()
    assert run.go2_graph_memory_skill_path.name == "SKILL.md"
    assert run.go2_graph_memory_skill_path.parent.name == "go2-graph-memory-hierarchy"
    assert run.go2_2d_slam_skill_path.parent.name == "go2-2d-slam"
    assert run.go2_graph_slam_navigation_skill_path.parent.name == (
        "go2-graph-slam-navigation"
    )
    assert "--graph-memory-path" in audit["proxy_shell_command"]
    assert "--slam-status-path" in audit["proxy_shell_command"]
    assert "--initial-slam-status-path" in audit["proxy_shell_command"]
    assert "--initial-relocalization-bundle" in audit["proxy_shell_command"]
    summary = json.loads(run.shareable_summary_path.read_text(encoding="utf-8"))
    assert summary["paths"]["slam_status"] == str(run.slam_status_path)
    assert summary["paths"]["initial_slam_status"] == str(initial_slam_status_path.resolve())
    assert summary["paths"]["initial_relocalization_bundle"] == str(
        initial_relocalization_bundle.resolve()
    )
    prompt = run.prompt_path.read_text(encoding="utf-8")
    assert "go2-graph-memory-hierarchy" in prompt
    assert "go2-2d-slam" in prompt
    assert "go2-graph-slam-navigation" in prompt



def test_prepare_codex_harness_run_seeds_prior_run_graph_memory_and_slam(
    tmp_path: Path,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    prior_root = tmp_path / "prior-run"
    (prior_root / "logs" / "panoramas").mkdir(parents=True)
    prior_graph = prior_root / "logs" / "go2-graph-memory.json"
    prior_graph.write_text(json.dumps({"nodes": [{"id": "A1"}]}), encoding="utf-8")
    prior_status = prior_root / "logs" / "go2-slam-status.json"
    prior_status.write_text(json.dumps({"latest_slam_map_metadata": {"status": "static"}}), encoding="utf-8")
    (prior_root / "logs" / "panoramas" / "front.jpg").write_bytes(b"jpeg")
    prior_bundle = prior_root / "slam" / "relocalization"
    prior_bundle.mkdir(parents=True)
    prior_map = prior_bundle / "global_map.pc2.lcm"
    prior_map.write_bytes(b"map")
    (prior_bundle / "manifest.json").write_text(
        json.dumps({"global_map_file": str(prior_map)}), encoding="utf-8"
    )

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "prior",
        source_codex_home=source_home,
        task="find the kitchen",
        mcp_url="http://127.0.0.1:9990/mcp",
        mode="authority-proxy",
        prompt_mode="long_instruction",
        prior_run_root=prior_root,
    )

    assert run.prior_run_root == prior_root.resolve()
    assert run.initial_slam_status_path == prior_status.resolve()
    assert run.initial_relocalization_bundle == prior_bundle.resolve()
    assert run.graph_memory_path is not None
    assert json.loads(run.graph_memory_path.read_text(encoding="utf-8"))["nodes"][0]["id"] == "A1"
    assert (run.graph_memory_path.parent / "panoramas" / "front.jpg").read_bytes() == b"jpeg"
    audit = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert audit["prior_run_root"] == str(prior_root.resolve())
    assert audit["prior_run_artifacts"]["relocalization_map_file"] == str(prior_map.resolve())
    assert audit["prior_graph_seed"]["graph_memory_seeded"] is True
    summary = json.loads(run.shareable_summary_path.read_text(encoding="utf-8"))
    assert summary["paths"]["prior_run_root"] == str(prior_root.resolve())
    assert summary["paths"]["prior_graph_seed"]["panoramas_seeded"] is True

def test_write_shareable_run_summary_excludes_run_local_credentials(tmp_path: Path) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token":"secret-token"}', encoding="utf-8")
    (source_home / "installation_id").write_text("install-secret", encoding="utf-8")
    (source_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")

    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="read status",
        mcp_url="http://127.0.0.1:9990/mcp",
    )
    summary_path = write_shareable_run_summary(run)
    summary_text = summary_path.read_text(encoding="utf-8")

    assert "secret-token" not in summary_text
    assert "install-secret" not in summary_text
    assert str(run.codex_home) not in summary_text
    assert "CODEX_HOME" not in summary_text
    assert "copied_auth_files" not in summary_text


def test_codex_mcp_smoke_prompt_requires_mcp_tools_and_forbids_shell_fallback() -> None:
    prompt = codex_mcp_smoke_prompt(required_tools=DEFAULT_MCP_SMOKE_REQUIRED_TOOLS)

    assert "DIMOS_MCP_SMOKE: PASS" in prompt
    assert "DIMOS_MCP_SMOKE: FAIL" in prompt
    assert "Do not use shell" in prompt
    assert "`dimos mcp`" in prompt
    assert "mcp__dimos__observe" in prompt
    assert "mcp__dimos__go2_situation" not in prompt
    assert "mcp__dimos__stop_movement" not in prompt
    assert "strip the exact `mcp__dimos__` prefix" in prompt
    assert "Do not call movement or posture tools" in prompt


def test_codex_mcp_smoke_log_result_fails_closed_without_required_tools() -> None:
    log_text = "\n".join(
        [
            "Required MCP tools: mcp__dimos__observe mcp__dimos__rotate_in_place",
            "DIMOS_MCP_SMOKE: PASS tools=mcp__dimos__observe",
        ]
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("observe", "rotate_in_place"),
    )

    assert ok is False
    assert "omitted required" in reason
    assert missing == ["mcp__dimos__rotate_in_place"]
    assert called == []


def test_codex_mcp_smoke_log_result_includes_codex_terminal_error() -> None:
    log_text = "\n".join(
        [
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "Reconnecting... 5/5 (stream disconnected before completion: "
                        "stream closed before response.completed)"
                    ),
                }
            ),
            json.dumps(
                {
                    "type": "turn.failed",
                    "error": {
                        "message": (
                            "stream disconnected before completion: "
                            "stream closed before response.completed"
                        )
                    },
                }
            ),
        ]
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("observe",),
        returncode=1,
    )

    assert ok is False
    assert "codex exec exited with status 1" in reason
    assert "stream closed before response.completed" in reason
    assert missing == ["mcp__dimos__observe"]
    assert called == []


def test_codex_mcp_smoke_log_result_rejects_shell_dimos_mcp_command() -> None:
    log_text = "\n".join(
        [
            json.dumps({"msg": "prompt says do not use `dimos mcp`"}),
            json.dumps({"cmd": ["uv", "run", "dimos", "mcp", "list-tools"]}),
            "DIMOS_MCP_SMOKE: PASS tools=mcp__dimos__observe",
        ]
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("observe",),
    )

    assert ok is False
    assert "forbidden shell fallback" in reason
    assert missing == ["mcp__dimos__observe"]
    assert called == []


def test_codex_mcp_smoke_log_result_requires_real_readonly_mcp_calls() -> None:
    log_text = "DIMOS_MCP_SMOKE: PASS tools=" + ",".join(
        f"mcp__dimos__{tool}" for tool in DEFAULT_MCP_SMOKE_REQUIRED_TOOLS
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=DEFAULT_MCP_SMOKE_REQUIRED_TOOLS,
    )

    assert ok is False
    assert "not backed" in reason
    assert "mcp__dimos__observe" in missing
    assert called == []


def test_codex_mcp_smoke_log_result_uses_fail_marker_missing_names() -> None:
    log_text = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "DIMOS_MCP_SMOKE: FAIL missing=mcp__dimos__relative_move",
            },
        }
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=DEFAULT_MCP_SMOKE_REQUIRED_TOOLS,
    )

    assert ok is False
    assert "reported missing" in reason
    assert missing == ["mcp__dimos__relative_move"]
    assert called == []


def test_codex_mcp_smoke_log_result_passes_when_tools_are_reported_and_called() -> None:
    required = DEFAULT_MCP_SMOKE_REQUIRED_TOOLS
    call_lines = [
        _mcp_tool_completed("observe"),
        _mcp_tool_completed("go2_robot_status"),
        _mcp_tool_completed("motion_safety_status"),
        _mcp_tool_completed("last_motion_diagnostics"),
    ]
    log_text = "\n".join(
        [
            *call_lines,
            "DIMOS_MCP_SMOKE: PASS tools=" + ",".join(f"mcp__dimos__{tool}" for tool in required),
        ]
    )

    ok, reason, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=required,
    )

    assert ok is True
    assert "all required" in reason
    assert missing == []
    assert "mcp__dimos__observe" in called
    assert "mcp__dimos__rotate_in_place" not in called


def test_codex_mcp_smoke_log_result_matches_bare_tools_to_prefixed_requirements() -> None:
    log_text = "\n".join(
        [
            _mcp_tool_completed("observe"),
            "DIMOS_MCP_SMOKE: PASS tools=observe,drive",
        ]
    )

    ok, _, missing, _ = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("mcp__dimos__observe", "mcp__dimos__drive"),
    )

    assert ok is True
    assert missing == []


def test_codex_mcp_smoke_log_result_reports_only_truly_missing_bare_tool() -> None:
    log_text = "\n".join(
        [
            _mcp_tool_completed("observe"),
            "DIMOS_MCP_SMOKE: PASS tools=observe",
        ]
    )

    ok, _, missing, _ = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("mcp__dimos__observe", "mcp__dimos__drive"),
    )

    assert ok is False
    assert missing == ["mcp__dimos__drive"]


def test_codex_mcp_smoke_log_result_accepts_mixed_tool_name_prefixes() -> None:
    log_text = "\n".join(
        [
            _mcp_tool_completed("mcp__dimos__observe"),
            "DIMOS_MCP_SMOKE: PASS tools=mcp__dimos__observe,drive",
        ]
    )

    ok, _, missing, called = codex_mcp_smoke_log_result(
        log_text,
        server_name="dimos",
        required_tools=("observe", "mcp__dimos__drive"),
    )

    assert ok is True
    assert missing == []
    assert called == ["mcp__dimos__observe"]


def test_run_codex_mcp_smoke_gate_writes_fail_closed_result(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="test task",
        mcp_url="http://127.0.0.1:9992/mcp",
        mode="authority-proxy",
    )
    run = replace(run, mcp_smoke_transport="codex_exec")

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        stdout = kwargs["stdout"]
        stdout.write(b"DIMOS_MCP_SMOKE: PASS tools=mcp__dimos__observe\n")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("dimos.agents.mcp.external_harness_codex.subprocess.run", fake_run)

    result = run_codex_mcp_smoke_gate(run, timeout_s=1.0)

    assert result.ok is False
    assert "omitted required" in result.reason
    assert "mcp__dimos__rotate_in_place" in result.missing_mcp_tool_names
    written = json.loads(run.mcp_smoke_result_path.read_text(encoding="utf-8"))
    assert written["ok"] is False
    launch = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert launch["mcp_smoke_gate"]["status"] == "fail"
    assert launch["mcp_smoke_gate"]["result"]["ok"] is False


def test_run_codex_mcp_smoke_gate_retries_transient_stream_disconnect(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="test task",
        mcp_url="http://127.0.0.1:9992/mcp",
        mode="authority-proxy",
    )
    run = replace(run, mcp_smoke_transport="codex_exec")
    calls = 0

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        stdout = kwargs["stdout"]
        if calls == 1:
            stdout.write(
                b'{"type":"error","message":"stream disconnected before completion: '
                b'error sending request for url (https://api.example.invalid/responses)"}\n'
            )
            return subprocess.CompletedProcess(args=args[0], returncode=1)
        required = run.mcp_smoke_required_tool_names
        log_text = "\n".join(
            [
                _mcp_tool_completed("observe"),
                _mcp_tool_completed("go2_robot_status"),
                _mcp_tool_completed("motion_safety_status"),
                _mcp_tool_completed("last_motion_diagnostics"),
                "DIMOS_MCP_SMOKE: PASS tools="
                + ",".join(f"mcp__dimos__{tool}" for tool in required),
            ]
        )
        stdout.write(log_text.encode())
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("dimos.agents.mcp.external_harness_codex.subprocess.run", fake_run)

    result = run_codex_mcp_smoke_gate(run, timeout_s=1.0, retry_delay_s=0.0)

    assert result.ok is True
    assert result.attempts == 2
    assert calls == 2
    assert "after 2 smoke attempts" in result.reason
    assert (run.mcp_smoke_log_path.with_name("mcp-smoke.attempt-1.jsonl")).exists()
    assert "DIMOS_MCP_SMOKE: PASS" in run.mcp_smoke_log_path.read_text(encoding="utf-8")
    written = json.loads(run.mcp_smoke_result_path.read_text(encoding="utf-8"))
    assert written["attempts"] == 2
    launch = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert launch["mcp_smoke_gate"]["status"] == "pass"


def test_run_codex_mcp_smoke_gate_preserves_pass_evidence_on_timeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="test task",
        mcp_url="http://127.0.0.1:9992/mcp",
        mode="authority-proxy",
    )
    run = replace(run, mcp_smoke_transport="codex_exec")

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        required = run.mcp_smoke_required_tool_names
        log_text = "\n".join(
            [
                _mcp_tool_completed("observe"),
                _mcp_tool_completed("go2_robot_status"),
                _mcp_tool_completed("motion_safety_status"),
                _mcp_tool_completed("last_motion_diagnostics"),
                "DIMOS_MCP_SMOKE: PASS tools="
                + ",".join(f"mcp__dimos__{tool}" for tool in required),
            ]
        )
        stdout = kwargs["stdout"]
        stdout.write(log_text.encode())
        stdout.flush()
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("dimos.agents.mcp.external_harness_codex.subprocess.run", fake_run)

    result = run_codex_mcp_smoke_gate(run, timeout_s=1.0)

    assert result.ok is True
    assert "timed out" in result.reason
    assert "after PASS marker" in result.reason
    assert result.missing_mcp_tool_names == ()
    assert "mcp__dimos__observe" in result.called_mcp_tool_names
    written = json.loads(run.mcp_smoke_result_path.read_text(encoding="utf-8"))
    assert written["ok"] is True
    launch = json.loads(run.audit_path.read_text(encoding="utf-8"))
    assert launch["mcp_smoke_gate"]["status"] == "pass"


def test_run_codex_mcp_smoke_gate_uses_direct_mcp_by_default(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir()
    run = prepare_codex_harness_run(
        run_root=tmp_path / "runs" / "door",
        source_codex_home=source_home,
        task="test task",
        mcp_url="http://127.0.0.1:9992/mcp",
        mode="authority-proxy",
    )
    called: list[str] = []

    class FakeAdapter:
        def __init__(self, url: str, timeout: int) -> None:
            assert url == "http://127.0.0.1:9992/mcp"
            assert timeout == 1

        def initialize(self) -> dict[str, Any]:
            return {"serverInfo": {"name": "fake"}}

        def list_tools(self) -> list[dict[str, Any]]:
            return [{"name": name} for name in run.mcp_smoke_required_tool_names]

        def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
            *,
            meta: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del arguments, meta
            called.append(name)
            return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr("dimos.agents.mcp.external_harness_codex.McpAdapter", FakeAdapter)

    result = run_codex_mcp_smoke_gate(run, timeout_s=1.0)

    assert result.ok is True
    assert result.transport == "direct_mcp"
    assert called == [
        "observe",
        "go2_robot_status",
        "motion_safety_status",
        "last_motion_diagnostics",
    ]
    assert "rotate_in_place" not in called
    assert "relative_move" not in called
    written = json.loads(run.mcp_smoke_result_path.read_text(encoding="utf-8"))
    assert written["transport"] == "direct_mcp"


def _mcp_tool_completed(tool: str) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "dimos",
                "tool": tool,
                "status": "completed",
                "error": None,
            },
        }
    )
