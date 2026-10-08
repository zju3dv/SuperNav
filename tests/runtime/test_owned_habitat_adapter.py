from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat.adapter import HabitatAdapter
from supernav.backends.habitat.simulator import navigation_access, session_scene
from supernav.backends.habitat.simulator.types import _Session


class FakeAgent:
    def __init__(self):
        self.state = SimpleNamespace(position=np.zeros(3), rotation=[1.0, 0.0, 0.0, 0.0])
        self.agent_config = SimpleNamespace(action_space={
            "move_forward": SimpleNamespace(name="move_forward")
        })

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state, **kwargs):
        self.state = copy.deepcopy(state)


class FakeSimulator:
    def __init__(self):
        self.agent = FakeAgent()
        self.calls = 0
        self.pathfinder = None

    def get_agent(self, agent_id):
        assert agent_id == 0
        return self.agent

    def get_sensor_observations(self, *, agent_ids):
        return {"color_sensor": np.zeros((2, 3, 3), dtype=np.uint8)}

    def reset(self):
        return self.get_sensor_observations(agent_ids=0)

    def step(self, action, *, dt):
        assert action == "move_forward"
        self.calls += 1
        collided = self.calls == 2
        if not collided:
            self.agent.state.position[2] -= 0.25
        return {**self.get_sensor_observations(agent_ids=0), "collided": collided}


def make_adapter():
    adapter = HabitatAdapter()
    sim = FakeSimulator()
    session = _Session("session", sim, "room", {}, mapless=True)
    session.trajectory = [[0.0, 0.0, 0.0]]
    session.audit_trajectory = [[0.0, 0.0, 0.0]]
    adapter._sessions[session.session_id] = session
    return adapter, session


def test_owned_motion_preserves_collision_path_and_audit_separation():
    adapter, session = make_adapter()
    response = adapter.handle_request({
        "action": "step_action", "session_id": "session",
        "payload": {"action": "move_forward", "distance": 1.25},
    })
    assert response["ok"] is True
    result = response["result"]
    assert result["steps_taken"] == 2
    assert result["collided"] is True
    assert session.step_count == 2
    assert session.cumulative_path_length == pytest.approx(0.25)
    assert session.audit_trajectory == [[0.0, 0.0, 0.0], [0.0, 0.0, -0.25]]
    assert len(session.collision_points) == 1
    assert "position" not in result["state_summary"]
    denied = adapter.handle_request({"action": "get_audit_metrics", "session_id": "session"})
    assert denied["error"]["type"] == "PermissionError"
    audit = adapter.handle_request({
        "action": "get_audit_metrics", "session_id": "session",
        "_audit_internal_authorized": True,
    })
    assert audit["result"]["agent_state"]["position"] == [0.0, 0.0, -0.25]


def test_navigation_observation_probe_restores_pose_on_failure(monkeypatch):
    adapter, session = make_adapter()
    monkeypatch.setattr(navigation_access, "habitat_sim", SimpleNamespace(AgentState=SimpleNamespace))

    def fail_capture(**kwargs):
        assert session.simulator.agent.state.position.tolist() == [2.0, 0.0, 1.0]
        raise RuntimeError("render failed")

    monkeypatch.setattr(session.simulator, "get_sensor_observations", fail_capture)
    with pytest.raises(RuntimeError, match="render failed"):
        adapter.observations_at(session, [2, 0, 1], [0, 0, 0, 1])
    assert session.simulator.agent.state.position.tolist() == [0.0, 0.0, 0.0]
    assert session.step_count == 0
    assert session.audit_trajectory == [[0.0, 0.0, 0.0]]




def test_navigation_path_translates_sdk_objects(monkeypatch):
    monkeypatch.setattr(navigation_access, "habitat_sim", SimpleNamespace(ShortestPath=SimpleNamespace))

    class Pathfinder:
        is_loaded = True

        def find_path(self, path):
            assert path.requested_start.dtype == np.float32
            path.geodesic_distance = 2.5
            path.points = [path.requested_start, path.requested_end]
            return True

    path = navigation_access.HabitatPathfinder(Pathfinder()).shortest_path([0, 0, 0], [2, 0, 0])
    assert path.geodesic_distance == 2.5
    assert path.points == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def public_dataset(tmp_path):
    stages = tmp_path / "configs" / "stages"
    scenes = tmp_path / "configs" / "scenes"
    stages.mkdir(parents=True)
    scenes.mkdir(parents=True)
    dataset = tmp_path / "mixed.scene_dataset_config.json"
    dataset.write_text(json.dumps({
        "stages": {"paths": {".json": ["configs/stages"]}},
        "scene_instances": {"paths": {".json": ["configs/scenes"]}},
        "navmesh_instances": {"playroom_navmesh": "playroom.navmesh"},
    }))
    (tmp_path / "playroom.navmesh").write_bytes(b"navmesh")
    for name, asset in (("playroom", "../../stages/playroom.gs.ply"), ("mesh", "../../stages/mesh.glb")):
        (stages / f"{name}_stage.stage_config.json").write_text(json.dumps({"render_asset": asset}))
        (scenes / f"{name}.scene_instance.json").write_text(json.dumps({
            "stage_instance": {"template_name": f"{name}_stage"},
            "navmesh_instance": "playroom_navmesh",
        }))
    return dataset


def test_public_scene_layout_classifies_selected_stage_and_navmesh(tmp_path):
    dataset = public_dataset(tmp_path)
    adapter = HabitatAdapter()
    assert adapter._is_gaussian_scene("playroom", str(dataset))
    assert not adapter._is_gaussian_scene("mesh", str(dataset))
    assert adapter._resolve_dataset_navmesh_path(str(dataset), "playroom") == str(tmp_path / "playroom.navmesh")
    assert adapter._is_gaussian_scene(str(tmp_path / "room.gs.ply"), None)
    assert not adapter._is_gaussian_scene(str(tmp_path / "room.glb"), None)


@pytest.mark.parametrize("scene,is_gaussian", [("playroom", True), ("mesh", False)])
def test_public_gaussian_stage_controls_native_initialization(monkeypatch, tmp_path, scene, is_gaussian):
    dataset = public_dataset(tmp_path)
    monkeypatch.setattr(session_scene, "default_settings", lambda: {
        "default_agent": 0, "default_agent_navmesh": True, "enable_physics": True,
    })
    configured = []

    def factory(settings):
        configured.append(dict(settings))
        return FakeSimulator()

    adapter = HabitatAdapter(simulator_factory=factory)
    response = adapter.handle_request({"action": "init_scene", "payload": {
        "scene": scene, "scene_dataset_config_file": str(dataset),
    }})
    assert response["ok"] is True, response
    assert response["result"]["is_gaussian"] is is_gaussian
    assert configured[0]["default_agent_navmesh"] is (not is_gaussian)
    assert configured[0]["enable_physics"] is (not is_gaussian)
