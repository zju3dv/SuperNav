"""Build SuperNav's simulator configuration using the public Habitat SDK.

The upstream settings helper supplies sensor/navmesh defaults. Action-space and
deployment settings belong here, so a stock Habitat-GS installation needs no
SuperNav-specific edits to ``habitat_sim.utils.settings``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Mapping


def default_settings() -> dict[str, Any]:
    from supernav.backends.habitat.simulator.sdk import load

    load()
    from habitat_sim.utils.settings import default_sim_settings

    settings = default_sim_settings.copy()
    # Habitat's native default physics settings do not depend on the launch cwd.
    # A repository-relative physics_config_file from the example helper does.
    settings.pop("physics_config_file", None)
    return settings


def make_configuration(settings: Mapping[str, Any]):
    from supernav.backends.habitat.simulator.sdk import load

    habitat_sim = load()
    from habitat_sim.utils.settings import make_cfg

    config = make_cfg(dict(settings))
    config.sim_cfg.gpu_device_id = int(os.environ.get("HAB_GPU_DEVICE_ID", "0"))
    if "allow_sliding" in settings:
        config.sim_cfg.allow_sliding = bool(settings["allow_sliding"])
    elif os.environ.get("HAB_ALLOW_SLIDING", "1") == "0":
        config.sim_cfg.allow_sliding = False
    agent_id = int(settings.get("default_agent", 0))
    if agent_id < 0 or agent_id >= len(config.agents):
        raise ValueError(f"default_agent {agent_id} is outside the configured agent list")
    config.agents[agent_id].action_space.update({
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=60.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=60.0)
        ),
    })
    return config


def make_simulator(config):
    """Retain physics-step timing without reading private SDK bookkeeping."""
    from supernav.backends.habitat.simulator.sdk import load

    habitat_sim = load()

    class TimedSimulator(habitat_sim.Simulator):
        last_physics_step_time_s = None

        def step_world(self, dt):
            started = time.time()
            result = super().step_world(dt)
            self.last_physics_step_time_s = time.time() - started
            return result

    return TimedSimulator(config)
