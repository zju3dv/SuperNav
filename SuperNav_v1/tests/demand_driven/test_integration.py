"""Portable launch, read-only build and process boundary regressions."""

import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from supernav.backends.ai2thor.native import make_controller
from supernav.backends.ai2thor.protocol import BUILD_COMMIT, NativeConfig
from supernav.evaluation.demand_driven.dataset import dataset_path, load_episode
from supernav.runtime.mcp import resolve_mcp_spec


def test_external_build_is_never_downloaded_or_locked(monkeypatch, tmp_path):
    calls = []
    ai2thor = ModuleType("ai2thor")
    ai2thor.__version__ = "5.0.0"
    build_module = ModuleType("ai2thor.build")
    controller_module = ModuleType("ai2thor.controller")
    platform_module = ModuleType("ai2thor.platform")
    platform_module.CloudRendering = object()
    executable = tmp_path / "release" / "unity"
    executable.parent.mkdir()
    executable.write_text("external build")
    metadata = executable.with_name("metadata.json")
    metadata.write_text("{}")

    class Build:
        def __init__(self, platform, commit, private, releases):
            assert commit == BUILD_COMMIT
            self.executable_path = str(executable)
            self.metadata_path = str(metadata)

        def download(self):
            pytest.fail("Attempted to download or modify a read-only external build")

        def lock_sh(self):
            pytest.fail("Attempted to write a lock beside the external build")

    class Controller:
        def __init__(self, **kwargs):
            build = self.find_build()
            build.download()
            build.lock_sh()
            build.unlock()
            calls.append(kwargs)

        def stop(self):
            calls.append("stop")

    build_module.Build = Build
    controller_module.Controller = Controller
    for name, module in [("ai2thor", ai2thor), ("ai2thor.build", build_module),
                         ("ai2thor.controller", controller_module), ("ai2thor.platform", platform_module)]:
        monkeypatch.setitem(sys.modules, name, module)
    make_controller({}, NativeConfig(), tmp_path / "runtime", executable.parent, 0)
    assert calls[0]["scene"] == {}
    executable.unlink()
    with pytest.raises(RuntimeError, match="Pinned CloudRendering build missing"):
        make_controller({}, NativeConfig(), tmp_path / "runtime", executable.parent, 0)
    assert calls[-1] == "stop"


def test_dataset_symlinks_cannot_escape(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text("{}")
    (dataset / "link.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        dataset_path(dataset, "link.json")
    with pytest.raises(ValueError, match="Unsafe"):
        dataset_path(dataset, "../private.json")


def test_unknown_episode_is_an_explicit_error(tmp_path):
    (tmp_path / "audit.json").write_text('{"episodes": []}')
    with pytest.raises(ValueError, match="Unknown episode"):
        load_episode(tmp_path, "missing")


def test_runtime_and_public_tools_import_without_either_simulator(tmp_path):
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('habitat_sim', 'ai2thor'):
            raise AssertionError('Unexpected simulator SDK import: ' + fullname)
sys.meta_path.insert(0, Block())
from supernav.runtime.agents import get_agent_backend
from supernav.methods.demand_driven import mcp, check_mcp, fourview_mcp
from supernav.backends.ai2thor import cli, native, fourview
from supernav.evaluation.demand_driven.dataset import validate_dataset
print('portable_imports_ok')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "portable_imports_ok" in result.stdout
    spec = resolve_mcp_spec(workspace_root=tmp_path, config={"mcp": {
        "name": "demand-driven", "command": sys.executable,
        "args": ["-m", "supernav.methods.demand_driven.mcp", "--storage-root", str(tmp_path)],
    }}, arm_cfg={})
    assert spec.name == "demand-driven"
    assert "supernav.methods.demand_driven.mcp" in spec.args


def test_public_mcp_lifecycle_without_simulator(tmp_path, monkeypatch):
    import multiprocessing
    import socket
    import time
    import urllib.request
    from types import SimpleNamespace

    import numpy as np
    from supernav.backends.ai2thor.cli import serve
    from supernav.backends.ai2thor.native import NativeSession
    from supernav.methods.demand_driven.check_mcp import check
    from supernav.paths import python_paths

    # Exercise the real HTTP and stdio MCP transports with only Unity replaced.
    class Controller:
        last_event = SimpleNamespace(
            frame=np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8),
            metadata=dict(agent=dict(position=dict(x=0, y=.95, z=0),
                                     rotation=dict(x=0, y=0, z=0), cameraHorizon=0),
                          fov=120, cameraPosition=dict(x=0, y=1.625, z=0),
                          lastActionSuccess=True, collided=False, objects=[]))

        def step(self, **kwargs):
            return self.last_event

    episode = dict(episode_id="e", scene_id="train.jsonl_1", instruction="Find a drink",
                   start_position=dict(x=0, y=.95, z=0), start_rotation_y=0, start_horizon=0,
                   reproducibility={"house_data_sha256": "hash"}, stage_plan=[])
    session = NativeSession(Controller(), episode, NativeConfig(), tmp_path / "episode")
    session.initialize()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = multiprocessing.get_context("fork").Process(target=serve, args=(session, port, None))
    process.start()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(python_paths()))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    assert json.load(response)["ok"]
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        # Run check() in a subprocess: its cache redirection is process-local.
        result = subprocess.run([sys.executable, "-m", "supernav.methods.demand_driven.check_mcp",
                                 "--storage-root", str(tmp_path), "--python", sys.executable,
                                 "--port", str(port), "--output", str(tmp_path / "mcp.json")],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert json.loads((tmp_path / "mcp.json").read_text())["valid"]
        assert len((tmp_path / "mcp.tools.jsonl").read_text().splitlines()) == 5
        status = json.loads((tmp_path / "episode/status.json").read_text())
        assert status["stop_called"] and status["action_count"] == 2
        assert status["success_scoring"] == "withheld"
    finally:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
