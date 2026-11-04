"""Timeouts keep native client bytes and pass through normal evidence collection."""

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


@pytest.mark.parametrize("name", ["opencode", "kimi"])
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
