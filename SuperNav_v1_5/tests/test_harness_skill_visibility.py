from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.runtime.skill_visibility import probe_skill_visibility  # noqa: E402
from supernav.runtime.skill_runtime import create_skill_snapshot  # noqa: E402
from supernav.runtime.codex_agent import CodexAgent, _effective_sandbox  # noqa: E402
from supernav.runtime.kimi_agent import KimiAgent, _without_skills_dir  # noqa: E402
from supernav.runtime.opencode_agent import OpenCodeAgent  # noqa: E402
import supernav.experiments.episode as run_one_module  # noqa: E402


def _completed(command, **kwargs):
    cwd = Path(kwargs["cwd"])
    backend = command[0]
    if backend == "codex":
        path = cwd / ".codex_home" / "skills" / "demo-skill" / "SKILL.md"
    elif backend == "kimi":
        path = cwd / ".kimi-code" / "skills" / "demo-skill" / "SKILL.md"
    else:
        path = cwd / ".opencode" / "skills" / "demo-skill" / "SKILL.md"
    return subprocess.CompletedProcess(
        command, 0, f"demo-skill Demo visibility skill {path.resolve()}", ""
    )


def _snapshot(tmp_path: Path):
    root = tmp_path / "source"
    skill = root / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo visibility skill\n---\nBody\n",
        encoding="utf-8",
    )
    return create_skill_snapshot(
        root, run_root=tmp_path / "snapshot-run", skill_set_id="test"
    )


def _config():
    return {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "mcp": {
            "transport": "stdio",
        },
        "bridge": {"host": "127.0.0.1", "port": 18911},
    }


def test_backends_install_snapshot_in_native_discovery_locations(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    arm = {
        "skill_mode": "native",
        "skill_name": "demo-skill",
        "_skill_snapshot_root": str(snapshot.snapshot_root),
        "tool_whitelist": [],
    }
    source_codex_home = tmp_path / "source-codex-home"
    source_codex_home.mkdir()
    (source_codex_home / "config.toml").write_text(
        'model_provider = "OpenAI"\n'
        "disable_response_storage = true\n\n"
        "[model_providers.OpenAI]\n"
        'name = "OpenAI"\n'
        'base_url = "http://api-gateway.test:28888"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = true\n\n"
        '[mcp_servers."user-server"]\n'
        'command = "user-command"\n\n'
        '[plugins."uncontrolled"]\n'
        "enabled = true\n",
        encoding="utf-8",
    )
    (source_codex_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-api-key"}), encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(source_codex_home))
    codex = CodexAgent().prepare_project(
        run_dir=tmp_path / "codex-run",
        workspace_root=REPO_ROOT,
        config=_config(),
        arm_cfg=arm,
        agent_cfg={"provider_mode": "user"},
        model_cfg={},
    )
    opencode = OpenCodeAgent().prepare_project(
        run_dir=tmp_path / "opencode-run",
        workspace_root=REPO_ROOT,
        config=_config(),
        arm_cfg=arm,
        agent_cfg={},
        model_cfg={},
    )
    kimi = KimiAgent().prepare_project(
        run_dir=tmp_path / "kimi-run",
        workspace_root=REPO_ROOT,
        config=_config(),
        arm_cfg=arm,
        agent_cfg={"source_home": str(tmp_path / "empty-kimi-home")},
        model_cfg={},
    )

    assert (codex / ".codex_home/skills/demo-skill/SKILL.md").is_file()
    assert _effective_sandbox(codex, {"sandbox": "workspace-write"}) == (
        "danger-full-access"
    )
    codex_config = (codex / ".codex_home/config.toml").read_text(encoding="utf-8")
    assert 'model_provider = "OpenAI"' in codex_config
    assert 'base_url = "http://api-gateway.test:28888"' in codex_config
    assert 'wire_api = "responses"' in codex_config
    assert "requires_openai_auth = true" in codex_config
    assert "disable_response_storage = true" in codex_config
    assert "user-server" not in codex_config
    assert "uncontrolled" not in codex_config
    assert json.loads(
        (codex / ".codex_home/auth.json").read_text(encoding="utf-8")
    ) == {"OPENAI_API_KEY": "test-api-key"}
    assert (opencode / ".opencode/skills/demo-skill/SKILL.md").is_file()
    assert (kimi / ".kimi-code/skills/demo-skill/SKILL.md").is_file()


def test_codex_non_native_project_keeps_configured_sandbox(tmp_path):
    assert _effective_sandbox(tmp_path, {"sandbox": "workspace-write"}) == (
        "workspace-write"
    )


def test_kimi_removes_all_uncontrolled_skills_dirs():
    assert _without_skills_dir(
        ["--skills-dir", "/global", "--model", "x", "--skills-dir=/other"]
    ) == ["--model", "x"]


def test_sweep_creates_one_snapshot_shared_by_all_dry_runs(tmp_path):
    source = tmp_path / "skills" / "demo-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo visibility skill\n---\nBody\n",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(output),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "skill_runtime": {
            "root": str(tmp_path / "skills"),
            "skill_set_id": "test-set",
            "visibility_gate": "required",
        },
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {},
        "agents": {"codex": {"command": "codex"}},
        "arms": {
            "native": {
                "movement": "primitive",
                "skill_mode": "native",
                "skill_name": "demo-skill",
                "tool_whitelist": [],
            }
        },
    }
    instructions = {
        "instructions": [
            {"slug": "one", "text": "First"},
            {"slug": "two", "text": "Second"},
        ]
    }
    config_path = tmp_path / "config.json"
    instructions_path = tmp_path / "instructions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--arms",
            "native",
            "--sweep-id",
            "shared-test",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    snapshot = output / "_sweeps/shared-test/skill_snapshot"
    assert snapshot.is_dir()
    hashes = {
        json.loads((output / f"native_{slug}/run.json").read_text(encoding="utf-8"))[
            "skill"
        ]["snapshot_sha256"]
        for slug in ("one", "two")
    }
    assert len(hashes) == 1


def test_sweep_output_and_task_filters_are_cli_overridable(tmp_path):
    skills = tmp_path / "skills" / "demo-skill"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    configured_output = tmp_path / "configured-output"
    override_output = tmp_path / "override-output"
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(configured_output),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "skill_runtime": {"root": str(tmp_path / "skills")},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {},
        "agents": {"codex": {"command": "codex"}},
        "arms": {
            "native": {
                "movement": "primitive",
                "skill_mode": "native",
                "skill_name": "demo-skill",
                "tool_whitelist": [],
            }
        },
    }
    instructions = {
        "instructions": [
            {"task_id": "task_001", "slug": "one", "text": "First"},
            {"task_id": "task_002", "slug": "two", "text": "Second"},
        ]
    }
    config_path = tmp_path / "config.json"
    instructions_path = tmp_path / "instructions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--arms",
            "native",
            "--output-dir",
            str(override_output),
            "--task-ids",
            "task_002",
            "--sweep-id",
            "filtered-test",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert not configured_output.exists()
    assert not (override_output / "native_task_001_one").exists()
    assert (override_output / "native_task_002_two" / "run.json").is_file()


def test_sweep_can_reuse_an_existing_immutable_snapshot(tmp_path):
    snapshot = _snapshot(tmp_path)
    source_skill = tmp_path / "source" / "demo-skill" / "SKILL.md"
    source_skill.write_text(
        "---\nname: demo-skill\ndescription: Mutated source.\n---\nChanged\n",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(output),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "skill_runtime": {"root": str(tmp_path / "source")},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {},
        "agents": {"codex": {"command": "codex"}},
        "arms": {
            "native": {
                "movement": "primitive",
                "skill_mode": "native",
                "skill_name": "demo-skill",
                "tool_whitelist": [],
            }
        },
    }
    instructions = {"instructions": [{"slug": "one", "text": "First"}]}
    config_path = tmp_path / "config.json"
    instructions_path = tmp_path / "instructions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--arms",
            "native",
            "--skill-snapshot-root",
            str(snapshot.snapshot_root),
            "--sweep-id",
            "reuse-test",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    run = json.loads((output / "native_one/run.json").read_text(encoding="utf-8"))
    assert run["skill"]["snapshot_sha256"] == snapshot.snapshot_sha256
    assert not (output / "_sweeps/reuse-test/skill_snapshot").exists()


def test_sweep_rep_start_appends_only_requested_rep_index(tmp_path):
    snapshot = _snapshot(tmp_path)
    output = tmp_path / "runs"
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(output),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "skill_runtime": {"root": str(tmp_path / "source")},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {},
        "agents": {"codex": {"command": "codex"}},
        "arms": {
            "native": {
                "movement": "primitive",
                "skill_mode": "native",
                "skill_name": "demo-skill",
                "tool_whitelist": [],
            }
        },
    }
    instructions = {"instructions": [{"slug": "one", "text": "First"}]}
    config_path = tmp_path / "config.json"
    instructions_path = tmp_path / "instructions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--arms",
            "native",
            "--skill-snapshot-root",
            str(snapshot.snapshot_root),
            "--rep-start",
            "3",
            "--reps",
            "1",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output / "native_one__r3/run.json").is_file()
    assert not (output / "native_one/run.json").exists()
    assert not (output / "native_one__r1/run.json").exists()


def test_sweep_codex_provider_defaults_to_user_and_allows_experiment_override(
    tmp_path,
):
    source_codex_home = tmp_path / "source-codex-home"
    source_codex_home.mkdir()
    (source_codex_home / "config.toml").write_text(
        'model_provider = "test_user_provider"\n\n'
        "[model_providers.test_user_provider]\n"
        'name = "Test User Provider"\n'
        'base_url = "https://test-user-provider.invalid"\n'
        'wire_api = "responses"\n',
        encoding="utf-8",
    )
    (source_codex_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-user-provider-secret"}),
        encoding="utf-8",
    )
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(tmp_path / "unused"),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {"name": "gpt-5.4"},
        "agents": {"codex": {
            "command": "codex",
            "provider": {
                "id": "test_provider",
                "base_url": "https://provider.example/v1",
                "env_key": "TEST_PROVIDER_KEY",
            },
        }},
        "arms": {
            "plain": {
                "movement": "primitive",
                "skill_mode": "none",
                "tool_whitelist": [],
            }
        },
    }
    instructions = {"instructions": [{"slug": "one", "text": "First"}]}
    config_path = tmp_path / "config.json"
    instructions_path = tmp_path / "instructions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")
    environment = dict(os.environ)
    environment["CODEX_HOME"] = str(source_codex_home)
    environment.pop("HAB_BENCH_RELAY_BASE_URL", None)
    environment["SUPERNAV_WORKSPACE_ROOT"] = str(tmp_path)
    assert not (tmp_path / "configs/local").exists()

    experiment_output = tmp_path / "experiment"
    experiment = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--output-dir",
            str(experiment_output),
            "--codex-provider", "experiment",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert experiment.returncode == 0, experiment.stderr
    experiment_run = json.loads(
        (experiment_output / "plain_one/run.json").read_text(encoding="utf-8")
    )
    assert experiment_run["model_provider"]["mode"] == "experiment"
    assert experiment_run["model_provider"]["base_url"] == "https://provider.example/v1"
    assert not (
        experiment_output / "plain_one/codex_project/.codex_home/auth.json"
    ).exists()

    user_output = tmp_path / "user"
    environment.pop("HAB_BENCH_RELAY_BASE_URL", None)
    user = subprocess.run(
        [
            sys.executable,
            "-m", "supernav", "run",
            "--config",
            str(config_path),
            "--instructions",
            str(instructions_path),
            "--output-dir",
            str(user_output),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert user.returncode == 0, user.stderr
    user_run = json.loads(
        (user_output / "plain_one/run.json").read_text(encoding="utf-8")
    )
    assert user_run["model_provider"] == {"mode": "user"}
    assert (user_output / "plain_one/codex_project/.codex_home/auth.json").exists()


def test_native_run_validates_experiment_token_before_visibility_probe(
    tmp_path,
    monkeypatch,
):
    snapshot = _snapshot(tmp_path)
    source_codex_home = tmp_path / "source-codex-home"
    source_codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source_codex_home))
    monkeypatch.setenv("HAB_BENCH_RELAY_BASE_URL", "https://relay.example.invalid/openai")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    output = tmp_path / "runs"
    config = {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(REPO_ROOT),
        "output_dir": str(output),
        "scene": "dummy",
        "scene_dataset_config_file": "/tmp/dummy.scene_dataset_config.json",
        "spawn": {"x": 0, "z": 0, "yaw": 0},
        "skill_runtime": {
            "root": str(tmp_path / "source"),
            "visibility_gate": "required",
        },
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
        "agent": "codex",
        "model": {"name": "gpt-5.4"},
        "agents": {"codex": {"command": "codex", "provider_mode": "experiment", "provider": {"id": "test_provider", "base_url": "https://provider.example/v1", "env_key": "AUTH_TOKEN"}}},
        "arms": {
            "native": {
                "movement": "primitive",
                "skill_mode": "native",
                "skill_name": "demo-skill",
                "tool_whitelist": [],
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    visibility_probe_called = False

    def unexpected_probe(**_kwargs):
        nonlocal visibility_probe_called
        visibility_probe_called = True
        return {"status": "visible"}

    monkeypatch.setattr(run_one_module, "probe_skill_visibility", unexpected_probe)
    args = Namespace(
        config=str(config_path),
        arm="native",
        slug="one",
        instruction="First",
        rep=None,
        run_id=None,
        overwrite=False,
        dry_run=False,
        timeout_s=None,
        skill_snapshot_root=str(snapshot.snapshot_root),
    )
    try:
        run_one_module.run_one(args)
    except ValueError as exc:
        assert "AUTH_TOKEN" in str(exc)
    else:
        raise AssertionError("missing AUTH_TOKEN should fail before visibility probe")

    assert visibility_probe_called is False
    run_dir = output / "native_one"
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["terminal_status"] == "codex_provider_environment_blocked"
    assert run_payload["model_provider"]["mode"] == "experiment"
    assert "AUTH_TOKEN" in run_payload["provider_error"]
    assert not (run_dir / "skill_visibility_report.json").exists()
    assert not (run_dir / "raw.jsonl").exists()


def test_visibility_probe_uses_each_backend_native_surface(tmp_path, monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return _completed(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    for backend in ("codex", "codex_profile", "kimi", "opencode"):
        result = probe_skill_visibility(
            backend=backend,
            project_dir=tmp_path,
            skill_name="demo-skill",
            description="Demo visibility skill",
            agent_cfg={"command": "codex" if backend.startswith("codex") else backend},
        )
        assert result["status"] == "visible"

    assert seen[0][0][1:3] == ["debug", "prompt-input"]
    assert "--skills-dir" in seen[2][0]
    assert seen[3][0][1] == "run"
    assert seen[3][1]["env"]["HOME"].endswith("opencode_home")


def test_visibility_probe_wrong_path_is_not_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            "demo-skill Demo visibility skill /home/user/global/SKILL.md",
            "",
        ),
    )
    result = probe_skill_visibility(
        backend="codex",
        project_dir=tmp_path,
        skill_name="demo-skill",
        description="Demo visibility skill",
        agent_cfg={"command": "codex"},
    )
    assert result["status"] == "not_visible"
    assert result["matched_path"] is False


def test_visibility_probe_missing_cli_is_fail_closed(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", missing)
    result = probe_skill_visibility(
        backend="kimi",
        project_dir=tmp_path,
        skill_name="demo-skill",
        description="Demo visibility skill",
        agent_cfg={"command": "kimi"},
    )
    assert result["status"] == "missing_cli"
    assert result["returncode"] is None


def test_codex_visibility_resolves_only_run_local_skill_alias(tmp_path, monkeypatch):
    from supernav.runtime import skill_visibility

    project = tmp_path / "project"
    local_root = project / ".codex_home/skills"

    def probe(root):
        text = f"### Skill roots\n- `r0` = `{root}`\n### Available skills\n- demo-skill: Demo visibility skill (file: r0/demo-skill/SKILL.md)"
        output = json.dumps([{"content": [{"type": "input_text", "text": text}]}])
        monkeypatch.setattr(
            skill_visibility, "_run", lambda *a, **k: ("completed", 0, output, "")
        )
        return probe_skill_visibility(
            backend="codex",
            project_dir=project,
            skill_name="demo-skill",
            description="Demo visibility skill",
            agent_cfg={},
        )

    assert probe(local_root)["status"] == "visible"
    assert probe(tmp_path / "global-skills")["status"] == "not_visible"
