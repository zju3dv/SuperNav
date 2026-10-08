"""Timeouts keep native client bytes and pass through normal evidence collection."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from supernav.runtime.agents import AgentResult, get_agent_backend, run_agent


EVENT = {
    "type": "item.completed",
    "item": {
        "id": "call-1", "type": "mcp_tool_call", "server": "navigation",
        "tool": "ddn_step", "arguments": {"action": "MoveAhead"},
        "result": {"content": [{"type": "text", "text": '{"action_index": 1}'}]},
        "status": "completed",
    },
}
START_EVENT = {
    "type": "item.started",
    "item": {key: value for key, value in EVENT["item"].items() if key != "result"},
}
START_EVENT["item"]["status"] = "in_progress"
RAW_TEXT = json.dumps(START_EVENT) + "\n" + json.dumps(EVENT) + "\n"


def sleeping_cli(tmp_path: Path) -> Path:
    script = tmp_path / "sleeping-client"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, time\n"
        "pathlib.Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        f"print({RAW_TEXT!r}, end='', flush=True)\n"
        "sys.stderr.write('native client stderr before timeout\\n'); sys.stderr.flush()\n"
        "time.sleep(30)\n"
    )
    script.chmod(0o700)
    return script


@pytest.mark.parametrize("name", ["codex", "codex_profile", "opencode", "kimi"])
def test_each_client_preserves_actual_subprocess_output_on_timeout(tmp_path, monkeypatch, name):
    command = sleeping_cli(tmp_path)
    run = tmp_path / "run"
    project = run / "project"
    project.mkdir(parents=True)
    cleanup = []
    if name == "kimi":
        from supernav.runtime import kimi_agent
        (project / "kimi_project.json").write_text(json.dumps({
            "workspace_root": str(tmp_path),
            "config": {"prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"), "mcp": {"command": sys.executable, "args": []}},
            "arm_cfg": {},
        }))
        monkeypatch.setattr(kimi_agent, "_write_kimi_mcp_configs", lambda **kwargs: None)
        monkeypatch.setattr(kimi_agent, "_start_mcp_server", lambda **kwargs: "owned-mcp")
        monkeypatch.setattr(kimi_agent, "_terminate_process", lambda process, **kwargs: cleanup.append(process))
    result = run_agent(
        get_agent_backend(name), prompt="Original task", run_dir=run,
        project_dir=project, agent_cfg={"command": str(command), "provider_mode": "user"},
        model_cfg={}, timeout_s=0.3,
    )

    assert result.returncode == 124 and result.timed_out
    assert result.error.startswith("TimeoutExpired:")
    assert result.command[0] == str(command)
    assert result.raw_path.read_text() == RAW_TEXT
    assert result.stderr_path.read_text() == "native client stderr before timeout\n"
    with pytest.raises(ProcessLookupError):
        os.kill(int(command.with_suffix(".pid").read_text()), 0)
    if name == "kimi":
        assert cleanup == ["owned-mcp"]


def test_completed_agent_result_is_returned_without_changes(tmp_path):
    expected = AgentResult(0, 1.0, ["client"], tmp_path / "raw", tmp_path / "stderr")
    agent = SimpleNamespace(run=lambda **kwargs: expected)
    assert run_agent(agent, prompt="task", run_dir=tmp_path, project_dir=tmp_path,
                     agent_cfg={}, model_cfg={}, timeout_s=1) is expected


def test_unrelated_client_errors_remain_exceptions(tmp_path):
    def fail(**kwargs):
        raise FileNotFoundError("missing executable")
    with pytest.raises(FileNotFoundError, match="missing executable"):
        run_agent(SimpleNamespace(run=fail), prompt="task", run_dir=tmp_path,
                  project_dir=tmp_path, agent_cfg={}, model_cfg={}, timeout_s=1)


def test_timeout_cli_collects_raw_canonical_metrics_and_closes_backend(tmp_path, monkeypatch):
    from supernav.backends.ai2thor.experiment import Ai2ThorBackend
    from supernav.backends.base import EpisodeTask
    from supernav.experiments import episode
    from supernav.runtime.codex_agent import CodexAgent

    command = sleeping_cli(tmp_path)
    lifecycle = []

    class Session:
        path = tmp_path / "native"

        def finish(self, reason="agent_returned_without_stop"):
            lifecycle.append(reason)
            return {"state": "closed_unscored", "action_count": 1,
                    "stop_called": False, "reason": reason, "success_scoring": "withheld"}

    class Backend(Ai2ThorBackend):
        def prepare_task(self, config, args, arm, root):
            return EpisodeTask("one", args.instruction, "scene")

        def configure_agent(self, *args):
            pass

        @contextmanager
        def episode(self, **kwargs):
            lifecycle.append("opened")
            try:
                yield Session()
            finally:
                lifecycle.append("cleaned_up")

    class Agent(CodexAgent):
        def prepare_project(self, *, run_dir, **kwargs):
            project = run_dir / "project"
            project.mkdir()
            return project

    config = tmp_path / "recipe.json"
    config.write_text(json.dumps({"prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
        "workspace_root": str(tmp_path), "output_dir": "runs",
        "agent": "codex", "agents": {"codex": {
            "command": str(command), "provider_mode": "user", "timeout_s": 1,
        }},
        "environment": {"backend": "ai2thor", "views": "four"},
        "arms": {"primitive": {"movement": "primitive", "skill_mode": "none"}},
    }))
    monkeypatch.setattr(episode, "get_backend", lambda config: Backend())
    monkeypatch.setattr(episode, "get_agent_backend", lambda name: Agent())
    monkeypatch.setattr(episode, "generate_timing_summary_artifacts", lambda path: None)
    monkeypatch.setattr(sys, "argv", [
        "run-one", "--config", str(config), "--arm", "primitive", "--slug", "one",
        "--instruction", "Original task",
    ])

    assert episode.main() == 1
    run = tmp_path / "runs/primitive_one"
    assert (run / "raw.jsonl").read_text() == RAW_TEXT
    canonical = [json.loads(line) for line in (run / "canonical.jsonl").read_text().splitlines()]
    assert any(row.get("name") == "ddn_step" for row in canonical)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["terminal_status"] == "timeout"
    assert metrics["timed_out"] and not metrics["process_completed"]
    assert metrics["returncode"] == 124
    assert metrics["success"] is None and metrics["spl"] is None
    assert metrics["success_scoring"] == "withheld"
    assert metrics["simulator_status"]["reason"] == "wall_timeout"
    assert metrics["tool_calls"]["ddn_step"] == 1
    assert json.loads((run / "command.json").read_text())["timed_out"]
    assert json.loads((run / "run.json").read_text())["terminal_status"] == "timeout"
    assert json.loads((run / "run.json").read_text())["error"].startswith("TimeoutExpired:")
    assert lifecycle == ["opened", "wall_timeout", "cleaned_up"]
    assert json.loads((tmp_path / "runs/results.jsonl").read_text())["timed_out"]
