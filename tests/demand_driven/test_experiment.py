"""Unified runner lifecycle, protocol isolation, and artifact semantics."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from test_dataset import archive
from supernav.backends import get_backend
from supernav.evaluation.demand_driven.dataset import audit_archive, digest, write_json
from supernav.experiments.episode import run_one
from supernav.paths import python_paths

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def experiment(archive, tmp_path):
    dataset = tmp_path/"dataset"
    with zipfile.ZipFile(archive) as source:
        source.extractall(dataset)
    audit = audit_archive(archive, expected_count=1)
    row = audit["episodes"][0]
    house_path, episode_path = dataset/row["house_path"], dataset/row["episode_path"]
    house = json.loads(house_path.read_text())
    house["rooms"] = [{"floorPolygon": [{"y": 0}]}]
    write_json(house_path, house)
    episode = json.loads(episode_path.read_text())
    episode["reproducibility"]["house_data_sha256"] = digest(house_path)
    write_json(episode_path, episode)
    row.update(house_sha256=digest(house_path), episode_sha256=digest(episode_path))
    write_json(dataset/"audit.json", audit)
    # Run the real bridge/HTTP/MCP pipeline, replacing only the Unity controller.
    python = tmp_path/"simulator-python"
    python.write_text(f'''#!{sys.executable}
import sys, runpy
sys.path.insert(0, {str(ROOT/'tests/demand_driven')!r})
from test_fourview import Controller
from supernav.backends.ai2thor import native
class FakeController(Controller):
    def stop(self): pass
native.make_controller = lambda *args, **kwargs: FakeController()
sys.argv = ["bridge", *sys.argv[3:]]
runpy.run_module("supernav.backends.ai2thor.cli", run_name="__main__")
''')
    python.chmod(0o755)
    probe = tmp_path/"probe-agent"
    probe.write_text(f'#!{sys.executable}\nimport runpy\nrunpy.run_path({str(ROOT/"tests/fixtures/mcp_probe_agent.py")!r}, run_name="__main__")\n')
    probe.chmod(0o755)
    cfg = dict(prompts_dir=str(Path(__file__).resolve().parents[2] / "configs/benchmarks/main/prompts"), workspace_root=str(tmp_path), output_dir=str(tmp_path/"runs"),
               environment=dict(backend="ai2thor", dataset=str(dataset), releases=str(tmp_path/"releases"),
                                python=str(python), views="four"),
               bridge=dict(port=0, startup_timeout_s=10), agent="opencode", model={},
               agents=dict(opencode=dict(command=str(probe), timeout_s=30)),
               arms=dict(primitive=dict(movement="primitive")))
    path = tmp_path/"config.json"
    write_json(path, cfg)
    return path, cfg


def arguments(path, **overrides):
    return SimpleNamespace(config=str(path), arm="primitive", slug="e", task_id="e",
        instruction="Prepare a drink", rep=None, run_id=None, timeout_s=None,
        overwrite=False, dry_run=overrides.pop("dry_run", False), **overrides)


def test_backend_defaults_and_rejection():
    assert get_backend({}).name == "habitat"
    with pytest.raises(ValueError, match="Unknown environment"):
        get_backend({"environment": {"backend": "missing"}})


def test_ai2thor_dry_run_has_public_demand_and_no_habitat_instructions(experiment):
    path, cfg = experiment
    cfg["environment"]["backend"] = "habitat"
    write_json(path, cfg)
    result = run_one(arguments(path, dry_run=True, backend="ai2thor"))
    run = Path(result["run_dir"])
    assert (run/"prompt.txt").read_text() == "Prepare a drink"
    text = (Path(result["project_dir"])/"AGENTS.md").read_text()
    assert "ddn_step" in text and "hab_init_scene" not in text and "Mug|1" not in text
    assert not list(run.glob("simulator-*/status.json"))
    prepared = json.loads((run/"backend.json").read_text())
    assert prepared["environment"]["views"] == "four"
    assert "--views" in prepared["mcp"]["args"]


@pytest.mark.parametrize("flags,stop,exit_code", [([], True, 0), (["--no-stop"], False, 0), (["--fail-run"], False, 7)])
def test_common_runner_preserves_tools_stop_and_unscored_results(experiment, flags, stop, exit_code):
    path, cfg = experiment
    cfg["agents"]["opencode"]["extra_args"] = flags
    write_json(path, cfg)
    result = run_one(arguments(path))
    run, metrics = Path(result["run_dir"]), result["metrics"]
    assert metrics["backend"] == "ai2thor" and metrics["success"] is None and metrics["spl"] is None
    assert metrics["stop_called"] is stop and metrics["action_count"] == (2 if stop else 1)
    assert metrics["returncode"] == exit_code and metrics["process_completed"] is (exit_code == 0)
    assert metrics["observation_views"] == ["front", "right", "back", "left"]
    process = json.loads((run/"simulator-process.json").read_text())
    assert process["returncode"] == 0 and process["ready"]
    events = [json.loads(line) for line in (run/"canonical.jsonl").read_text().splitlines()]
    observed = next(e for e in events if e["type"] == "tool_result" and e["name"] == "ddn_observe")
    assert sum(c["type"] == "image" for c in json.loads(observed["content"])["content"]) == 4
    assert (run.parent/"results.jsonl").is_file()


def test_sweep_returns_nonzero_when_agent_fails(experiment):
    path, cfg = experiment
    cfg["agents"]["opencode"]["extra_args"] = ["--fail-run"]
    write_json(path, cfg)
    result = subprocess.run([sys.executable, "-m", "supernav.experiments.sweep", "--backend", "ai2thor",
                             "--config", str(path), "--task-ids", "e"],
                            capture_output=True, text=True, timeout=45,
                            env=dict(os.environ, PYTHONPATH=os.pathsep.join(python_paths())))
    assert result.returncode == 1, result.stdout+result.stderr
    metrics = json.loads((Path(cfg["output_dir"])/"primitive_e/metrics.json").read_text())
    assert metrics["returncode"] == 7 and metrics["success"] is None


def test_habitat_arms_cannot_start_ai2thor(experiment):
    path, cfg = experiment
    cfg["arms"]["primitive"]["tool_whitelist"] = ["hab_init_scene"]
    write_json(path, cfg)
    with pytest.raises(ValueError, match="Habitat tools"):
        run_one(arguments(path))
    assert not Path(cfg["output_dir"]).exists()


def test_cleanup_on_agent_exception(experiment, monkeypatch):
    from supernav.experiments import episode
    path, cfg = experiment
    class Agent:
        name = "opencode"
        def prepare_project(self, **kwargs):
            raise RuntimeError("agent preparation failed")
    monkeypatch.setattr(episode, "get_agent_backend", lambda name: Agent())
    with pytest.raises(RuntimeError, match="agent preparation failed"):
        run_one(arguments(path))
    run = Path(cfg["output_dir"])/"primitive_e"
    status = json.loads((run/"simulator-1/e/status.json").read_text())
    assert not status["stop_called"] and status["reason"] == "operator_interrupt"
    assert json.loads((run/"simulator-process.json").read_text())["returncode"] == 130


def test_kimi_http_transport_uses_its_own_port_and_same_four_views(experiment):
    import asyncio
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from supernav.runtime.kimi_agent import _free_local_port, _start_mcp_server, _terminate_process
    path, config = experiment
    backend = get_backend(config)
    root = Path(config["workspace_root"])
    run = root/"http-check"
    run.mkdir()
    arm = config["arms"]["primitive"]
    task = backend.prepare_task(config, arguments(path), arm, root)
    backend.configure_agent(config, arm, task, root, run)
    write_json(run/"kimi_project.json", dict(workspace_root=str(root), config=config, arm_cfg=arm))
    port = _free_local_port()

    async def check():
        async with httpx.AsyncClient(trust_env=False) as http:
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http) as streams:
                async with ClientSession(*streams[:2]) as client:
                    await client.initialize()
                    schemas = await client.list_tools()
                    assert len(schemas.tools) == 5
                    obs = await client.call_tool("ddn_observe", {})
                    assert not obs.isError and sum(c.type == "image" for c in obs.content) == 4
                    stop = await client.call_tool("ddn_stop", {})
                    assert not stop.isError
    with backend.episode(root=root, config=config, task=task, run_dir=run, dry_run=False, live_context={}):
        process = _start_mcp_server(project_dir=run, stderr_path=run/"mcp.log", port=port)
        try:
            asyncio.run(check())
        finally:
            _terminate_process(process, stderr_path=run/"mcp.log")


def test_experiment_imports_without_simulator_sdks(tmp_path):
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('habitat_sim', 'ai2thor'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from supernav.experiments import sweep as experiment, episode
from supernav.backends import get_backend
assert get_backend({}).name == 'habitat'
assert get_backend({'environment':{'backend':'ai2thor'}}).name == 'ai2thor'
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(python_paths())), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
