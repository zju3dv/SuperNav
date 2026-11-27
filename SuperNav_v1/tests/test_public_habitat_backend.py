"""SuperNav's protocol works without private Habitat-GS application modules."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_owned_adapter_protocol_and_state_without_any_simulator_sdk(tmp_path):
    code = r'''
import importlib.abc
import sys

class BlockSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"habitat_sim", "habitat_agent", "harness", "tools"}:
            raise AssertionError("Unexpected external dependency: " + fullname)

sys.meta_path.insert(0, BlockSimulator())
from supernav.backends.habitat.adapter import HabitatAdapter, HabitatAdapterError
adapter = HabitatAdapter()
description = adapter.handle_request({"action": "describe_api", "request_id": "public-sdk"})
assert description["ok"] is True
assert description["request_id"] == "public-sdk"
assert description["result"]["api_version"] == "habitat-gs/v1"
assert {"init_scene", "step_action", "get_panorama", "navigate_visual_point", "reset_agent_pose"} <= set(description["result"]["actions"])
reset = adapter.handle_request({"action": "reset_agent_pose", "session_id": "missing", "payload": {"x": 0, "y": 0, "z": 0, "yaw": 0}})
assert reset["ok"] is False
assert reset["error"]["type"] == "HabitatAdapterError"
assert "Unknown session_id" in reset["error"]["message"]
assert adapter.get_runtime_status()["active_sessions"] == 0
assert all(base.__module__.startswith("supernav.") for base in type(adapter).__mro__[:-1])

invalid = adapter.handle_request({"action": "missing_action", "request_id": 9})
assert invalid == {"ok": False, "action": "missing_action", "request_id": 9,
                   "session_id": None, "error": {"type": "ValueError", "message": "Unsupported action: missing_action"}}
private = adapter.handle_request({"action": "get_audit_metrics"})
assert private["error"]["type"] == "PermissionError"
adapter.close_all()
'''
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("SUPERNAV_", "NAV_", "HAB_", "HABITAT_"))
    }
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def public_settings_sdk(monkeypatch):
    """Model public make_cfg's three-action space, without a native extension."""
    monkeypatch.delenv("SUPERNAV_HABITAT_ROOT", raising=False)
    habitat_sim = types.ModuleType("habitat_sim")
    habitat_sim.__path__ = []
    agent = types.ModuleType("habitat_sim.agent")
    agent.ActuationSpec = lambda **kwargs: types.SimpleNamespace(**kwargs)
    agent.ActionSpec = lambda name, actuation: types.SimpleNamespace(name=name, actuation=actuation)
    habitat_sim.agent = agent
    utils = types.ModuleType("habitat_sim.utils")
    utils.__path__ = []
    settings_module = types.ModuleType("habitat_sim.utils.settings")
    settings_module.default_sim_settings = {
        "scene": "NONE", "default_agent": 0,
        "physics_config_file": "data/default.physics_config.json",
    }
    calls = []

    def make_cfg(settings):
        calls.append(dict(settings))
        actions = {
            name: agent.ActionSpec(name, agent.ActuationSpec(amount=amount))
            for name, amount in (("move_forward", 0.25), ("turn_left", 10.0), ("turn_right", 10.0))
        }
        return types.SimpleNamespace(
            sim_cfg=types.SimpleNamespace(gpu_device_id=0, allow_sliding=True),
            agents=[types.SimpleNamespace(action_space=actions)],
        )

    settings_module.make_cfg = make_cfg
    utils.settings = settings_module
    habitat_sim.utils = utils
    for module in (habitat_sim, agent, utils, settings_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return settings_module, calls


@pytest.mark.parametrize("explicit,environment,expected", [
    (None, "1", True), (None, "0", False), (True, "0", True), (False, "1", False),
])
def test_public_sdk_configuration_preserves_action_and_motion_contract(
    monkeypatch, public_settings_sdk, explicit, environment, expected,
):
    from supernav.backends.habitat.simulator.configuration import default_settings, make_configuration

    upstream, calls = public_settings_sdk
    monkeypatch.setenv("HAB_GPU_DEVICE_ID", "2")
    monkeypatch.setenv("HAB_ALLOW_SLIDING", environment)
    settings = default_settings()
    assert "physics_config_file" not in settings
    assert upstream.default_sim_settings["physics_config_file"] == "data/default.physics_config.json"
    if explicit is not None:
        settings["allow_sliding"] = explicit
    configuration = make_configuration(settings)

    assert calls == [settings]
    assert configuration.sim_cfg.gpu_device_id == 2
    assert configuration.sim_cfg.allow_sliding is expected
    actions = configuration.agents[0].action_space
    assert set(actions) == {"move_forward", "move_backward", "turn_left", "turn_right", "look_up", "look_down"}
    assert actions["move_forward"].actuation.amount == actions["move_backward"].actuation.amount == 0.25
    for name in ("turn_left", "turn_right", "look_up", "look_down"):
        assert actions[name].actuation.amount == 10.0
    assert actions["look_up"].actuation.constraint == actions["look_down"].actuation.constraint == 60.0


def test_public_sdk_configuration_rejects_unconfigured_agent(public_settings_sdk):
    from supernav.backends.habitat.simulator.configuration import make_configuration

    with pytest.raises(ValueError, match="default_agent 1"):
        make_configuration({"scene": "NONE", "default_agent": 1})
