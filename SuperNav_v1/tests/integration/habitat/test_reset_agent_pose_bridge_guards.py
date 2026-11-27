"""Pose recovery preserves navigation and scoring behavior without a native SDK."""
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat.simulator import motion
from supernav.backends.habitat.simulator.payload import SimulatorPayloadMixin


@pytest.mark.parametrize("loaded", [True, False])
def test_reset_snaps_only_loaded_navmesh_and_rebases_trajectory(monkeypatch, loaded):
    class Agent:
        state = SimpleNamespace(position=np.array([0.0, 0.0, 0.0]))

        def set_state(self, state, *, infer_sensor_states):
            assert infer_sensor_states
            self.state = state

    agent = Agent()
    snapped = []

    def snap(point):
        snapped.append(point.tolist())
        return [1.0, 0.0, 2.0]

    pathfinder = SimpleNamespace(is_loaded=loaded, snap_point=snap)
    session = SimpleNamespace(
        session_id="s", agent_id=0,
        simulator=SimpleNamespace(get_agent=lambda _: agent),
        trajectory=[[90.0, 0.0, 90.0]], audit_trajectory=[[90.0, 0.0, 90.0]],
        camera_pitch_deg=15, cumulative_path_length=4.0,
    )

    class Adapter(motion.HabitatAdapterNavigationMixin, SimulatorPayloadMixin):
        def _require_session(self, session_id):
            assert session_id == "s"
            return session

        def _get_pathfinder(self, _):
            return pathfinder

        def _current_position(self, _):
            return agent.state.position

        def _capture_sensor_observations(self, _):
            return {"color_sensor": "fresh"}

        def _build_metrics(self, _):
            return {"agent_state": {"position": agent.state.position.tolist()}}

    monkeypatch.setattr(motion, "habitat_sim", SimpleNamespace(AgentState=SimpleNamespace))
    result = Adapter()._reset_agent_pose("s", {"x": 1.1, "y": 0.3, "z": 2.1, "yaw": 0.0})
    expected = [1.0, 0.0, 2.0] if loaded else [1.1, 0.3, 2.1]
    assert bool(snapped) is loaded
    assert result["agent_state"]["position"] == pytest.approx(expected)
    assert agent.state.rotation == [0.0, 0.0, 0.0, 1.0]
    assert session.trajectory[0] == pytest.approx(expected)
    assert session.audit_trajectory == session.trajectory
    assert session.cumulative_path_length == 4.0
    assert session.last_sensor_obs == {"color_sensor": "fresh"}
    assert session.camera_pitch_deg == 0.0
