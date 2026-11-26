"""Native-free regression tests for evaluator isolation and episode lifecycle."""

import json
from types import SimpleNamespace

import numpy as np
import pytest



from supernav.evaluation.demand_driven.dataset import policy_task
from supernav.backends.ai2thor.native import NativeSession
from supernav.backends.ai2thor.protocol import NativeConfig, horizontal_fov, vertical_fov


class FakeController:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.last_event = SimpleNamespace(
            frame=np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
            metadata=dict(agent=dict(position=dict(x=0, y=.95, z=0), rotation=dict(x=0,y=0,z=0), cameraHorizon=0),
                          fov=120, cameraPosition=dict(x=0, y=1.625, z=0),
                          lastActionSuccess=True, collided=False, objects=[dict(objectId="Mug|1")]))

    def step(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("simulator timeout")
        return self.last_event


@pytest.fixture
def session(tmp_path):
    ep = dict(episode_id="e", scene_id="train.jsonl_1", instruction="Prepare a drink", start_position=dict(x=0,y=.95,z=0),
              start_rotation_y=0, start_horizon=0, reproducibility={"house_data_sha256":"123"},
              stage_plan=[dict(allowed_target_candidates=[dict(object_id="Mug|1")])])
    s = NativeSession(FakeController(), ep, NativeConfig(), tmp_path / "episode")
    s.initialize()
    return s


def test_fov_is_converted_not_copied():
    assert vertical_fov(90,640,480) == pytest.approx(73.739795)
    assert horizontal_fov(vertical_fov(79,640,480),640,480) == pytest.approx(79)
    assert horizontal_fov(120,640,480) > 130


def test_formal_evaluation_is_not_silently_enabled():
    with pytest.raises(ValueError):
        NativeConfig(success_scoring="enabled")
    with pytest.raises(ValueError):
        NativeConfig(profile="formal")
    assert not NativeConfig().receipt()["baseline_alignment_confirmed"]


def test_only_instruction_and_rgb_are_exposed(session):
    obs = session.observe()
    assert set(policy_task(session.episode)) == {"episode_id", "instruction"}
    for key in ("stage_plan", "objects", "pose", "scene_id", "reference_path", "metrics"):
        assert key not in obs
    assert "Mug|1" not in json.dumps(obs)
    assert obs["success_scoring"] == "withheld"


@pytest.mark.parametrize("action", ["TeleportFull", "GetShortestPath", "OpenObject", "GetReachablePositions"])
def test_policy_cannot_teleport_or_query_map(session, action):
    before = session.actions
    with pytest.raises(ValueError):
        session.step(action)
    assert session.actions == before


def test_failed_collision_attempt_consumes_budget(session):
    session.controller.last_event.metadata.update(lastActionSuccess=False, collided=True)
    result = session.step("forward")
    assert result["collided"] and not result["action_success"]
    assert session.actions == 1
    assert not session.infra_failed


def test_unity_failure_is_infrastructure_not_policy_failure(session):
    session.controller.fail = True
    with pytest.raises(TimeoutError):
        session.step("forward")
    result = session.finish("infrastructure_error")
    assert result["state"] == "infra_failed"
    assert "success" not in result
    assert not result["stop_called"]


def test_claims_are_not_auto_accepted_and_cannot_use_stale_images(session):
    old = session.observation_id
    assert session.claim(old, "Mug", [.5,.5]) == dict(recorded=True, success_scoring="withheld")
    session.step("left")
    with pytest.raises(ValueError):
        session.claim(old, "Mug", [.5,.5])
    assert session.actions == 1


def test_stop_is_explicit_and_terminal(session):
    result = session.finish()
    assert result["stop_called"] and result["action_count"] == 1
    with pytest.raises(ValueError):
        session.step("forward")
    with pytest.raises(ValueError):
        session.finish()


def test_exhaustion_does_not_synthesize_stop(session):
    session.actions = session.config.max_actions
    with pytest.raises(ValueError):
        session.finish()
    result = session.finish("engineering_action_cap")
    assert not result["stop_called"]


def test_stop_guard_requires_reconfirmation_without_gt(session):
    calls = []
    class Policy:
        def act(self, history, goal, pixel):
            calls.append((len(history), pixel))
            return dict(trajectories=[[[0,0]] * 8], selected=0, scores=[1], done_probability=.9, confidence=1)
    result = session.local_navigate(session.observation_id, [.5,.75], Policy(), 3)
    assert len(calls) == 2
    assert session.actions == 0
    assert result["reason"] == "local_policy_stop_not_task_success"
    assert not session.terminal


@pytest.mark.parametrize("lateral,expected", [(.2,"RotateLeft"),(-.2,"RotateRight")])
def test_nomad_robot_lateral_sign_maps_to_native_turns(session, lateral, expected):
    class Policy:
        def act(self, history, goal, pixel):
            return dict(trajectories=[[[.2,lateral]] * 8], selected=0, scores=[1], done_probability=0, confidence=1)
    session.local_navigate(session.observation_id, [.5,.75], Policy(), 1)
    assert session.controller.calls[1]["action"] == expected
    assert all(call["action"] != "TeleportFull" for call in session.controller.calls[1:])


def test_existing_output_is_not_overwritten(session):
    with pytest.raises(FileExistsError):
        NativeSession(FakeController(), session.episode, NativeConfig(), session.output)


def test_vertical_settling_is_explicit_not_claimed_exact(session):
    session.controller.last_event.metadata["agent"]["position"]["y"] = .901
    session.initialize()
    receipt = json.loads((session.private / "scene_receipt.json").read_text())
    assert receipt["valid"] and not receipt["strict_start_pose_match"]
    assert receipt["vertical_settling_m"] == pytest.approx(-.049)
    assert not receipt["baseline_alignment_confirmed"]


def test_horizontal_spawn_drift_is_rejected(session):
    session.controller.last_event.metadata["agent"]["position"]["x"] = .1
    with pytest.raises(RuntimeError):
        session.initialize()


@pytest.mark.parametrize("bad_field", [None, "rotation", "fieldOfView", "position"])
def test_height_controlled_camera_is_front_only_and_pose_verified(session, tmp_path, bad_field):
    class CameraController(FakeController):
        def step(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["action"] in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
                camera = {k: kwargs[k].copy() if isinstance(kwargs[k], dict) else kwargs[k]
                          for k in ("position", "rotation", "fieldOfView")}
                if bad_field == "fieldOfView":
                    camera[bad_field] = 90
                elif bad_field:
                    camera[bad_field]["y"] += 10
                self.last_event.metadata["thirdPartyCameras"] = [camera]
                # A distinct RGBA image proves the native main view is not exposed.
                rgba = np.zeros((480, 640, 4), dtype=np.uint8)
                rgba[:, :320, 0] = 255
                rgba[:, :, 3] = 255
                self.last_event.third_party_camera_frames = [rgba]
            return self.last_event

    controller = CameraController()
    config = NativeConfig(profile="neednav_h125_unscored", camera_height_m=1.25)
    s = NativeSession(controller, session.episode, config, tmp_path / "height_test",
                      house={"rooms": [{"floorPolygon": [{"y": 0}]}]})
    if bad_field:
        with pytest.raises(RuntimeError):
            s.initialize()
        return
    s.initialize()
    assert s.actions == 0 and s.last_rgb.shape == (480, 640, 3)
    assert s.last_rgb[:, :320, 0].min() == 255
    assert s.last_rgb[:, 320:, :].max() == 0
    assert controller.calls[-1]["position"]["y"] == 1.25
    s.step("forward")
    assert controller.calls[-1]["action"] == "UpdateThirdPartyCamera"
    assert s.actions == 1


def test_last_physical_action_closes_without_fabricated_stop(session):
    session.actions = session.config.max_actions - 1
    result = session.step("forward")
    assert result["terminal"] and session.terminal
    status = json.loads((session.output / "status.json").read_text())
    assert status["reason"] == "action_budget_exhausted"
    assert not status["stop_called"]
