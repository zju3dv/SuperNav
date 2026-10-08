"""Navigation methods run against plain capabilities without simulator SDKs."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from habitat_contract.navigation import NavigationPath
from habitat_contract.navigation_state import HabitatAdapterError
from supernav.methods.navigation.oracle_local_nav.visual_point import (
    _candidate_can_see_hint,
    navigate_visual_point,
)


class FlatPathfinder:
    is_loaded = True

    def snap_point(self, point):
        result = np.asarray(point, dtype=np.float32).copy()
        result[1] = 0.0
        return result

    def shortest_path(self, start, end):
        return NavigationPath(float(np.linalg.norm(np.asarray(end) - start)), [list(start), list(end)])


class NavigationFake:
    """No simulator, private adapter operations or SDK-shaped value objects."""

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.pathfinder = FlatPathfinder()
        self.moves = []
        self.session = SimpleNamespace(
            session_id="nav-session", latest_visual_capture_seq=1,
            last_visual_image_refs={
                "front:1": {
                    "capture_seq": 1, "direction": "front", "width": 64,
                    "height": 64, "hfov": 90.0,
                    "depth": np.full((64, 64), 4.0, dtype=np.float32),
                    "camera_position": [0.0, 1.5, 0.0],
                    "camera_forward": [0.0, 0.0, -1.0],
                    "camera_right": [1.0, 0.0, 0.0],
                    "camera_up": [0.0, 1.0, 0.0],
                }
            },
        )

    def navigation_session(self, session_id):
        if session_id != self.session.session_id:
            raise HabitatAdapterError("unknown session")
        return self.session

    def assert_map_planning_allowed(self, session, **kwargs):
        assert session is self.session

    def navigation_pathfinder(self, session):
        return self.pathfinder

    def agent_position(self, session):
        return self.position.copy()

    def agent_forward(self, session):
        return [0.0, 0.0, -1.0]

    def navigate_step(self, session_id, payload):
        self.moves.append(payload)
        self.position = np.asarray(payload["goal"], dtype=np.float32)
        return {
            "nav_status": "reached", "steps_executed": 8,
            "trace_frame_paths": ["original-move.png"],
            "metrics": {"success": False, "spl": 0.0},
            "state_summary": {"position": self.position.tolist()},
        }


def test_point_navigation_preserves_preview_confirmation_frames_and_scoring():
    adapter = NavigationFake()
    payload = {"image_ref": "front:1", "point": [0.5, 0.85]}
    preview = navigate_visual_point(adapter, "nav-session", payload)
    assert preview["status"] == "preview_ready"
    assert adapter.moves == []
    result = navigate_visual_point(adapter, "nav-session", {
        **payload, "confirm_token": preview["confirm_token"],
    })
    assert result["status"] == "reached_visual_point"
    assert result["steps_executed"] == 8
    assert result["movement_frames"] == ["original-move.png"]
    assert result["metrics"] == {"success": False, "spl": 0.0}
    assert adapter.session.pending_visual_point_preview is None


def test_new_capture_invalidates_preview_without_movement():
    adapter = NavigationFake()
    payload = {"image_ref": "front:1", "point": [0.5, 0.85]}
    preview = navigate_visual_point(adapter, "nav-session", payload)
    adapter.session.latest_visual_capture_seq = 2
    result = navigate_visual_point(adapter, "nav-session", {
        **payload, "confirm_token": preview["confirm_token"],
    })
    assert result["status"] == "stale_confirm_token"
    assert adapter.moves == []


@pytest.mark.parametrize(("hit", "visible"), [(1.0, False), (5.0, True)])
def test_candidate_visibility_consumes_distance_hits(hit, visible):
    calls = []

    def cast_ray(origin, direction, *, max_distance):
        calls.append((origin, direction, max_distance))
        return [hit]

    assert _candidate_can_see_hint(
        simulator=SimpleNamespace(cast_ray=cast_ray),
        snapped=np.zeros(3), hint=np.array([0.0, 1.5, -4.0]), eye_height_m=1.5,
    ) is visible
    assert len(calls) == 1
    assert calls[0][2] == pytest.approx(4.25)


def test_localnav_missing_session_preserves_native_error_result():
    from supernav.methods.localnav.navigation import navigate_with_localnav

    result = navigate_with_localnav(NavigationFake(), "missing", {})
    assert result["ok"] is False
    assert result["error"] == "no session missing"
    assert result["code"] == "no_session"


def test_localnav_rgb_capture_reads_current_observation_each_time():
    from supernav.methods.localnav.navigation import _capture_observation

    frames = iter([np.zeros((3, 3, 4), dtype=np.uint8), np.ones((3, 3, 4), dtype=np.uint8)])
    adapter = SimpleNamespace(sensor_observations=lambda session: {"color_sensor": next(frames)})
    first = _capture_observation(adapter, object())
    second = _capture_observation(adapter, object())
    assert first.shape == second.shape == (3, 3, 3)
    assert not first.any()
    assert second.all()


def test_methods_never_import_simulator_sdks_or_read_native_session_objects():
    root = Path(__file__).resolve().parents[2] / "src" / "supernav" / "methods"
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                imports = []
            if any(name.split(".")[0] in {"habitat_sim", "habitat_agent", "localnav_data"} for name in imports):
                violations.append(f"{path.name}:{node.lineno}: import")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "session" and node.attr == "simulator":
                    violations.append(f"{path.name}:{node.lineno}: simulator")
                if node.value.id == "adapter" and node.attr.startswith("_"):
                    violations.append(f"{path.name}:{node.lineno}: private adapter")
    assert violations == []


def test_jsonl_samples_round_trip_without_external_data_package(tmp_path):
    from supernav.methods.localnav.contracts import NormalizedPoint, VelocityAction
    from supernav.methods.localnav.dataset import LocalNavTrainingSample, load_samples_jsonl

    sample = LocalNavTrainingSample(
        "sample-1", "supervised", "episode-1", "goal.png", 0,
        ("current.png",), 1, 2, NormalizedPoint(0.5, 0.8),
        (VelocityAction(0.25, 0.0),), 1.0,
        metadata={"waypoint_deltas": [[0.25, 0.0]]},
    )
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(sample.as_dict()) + "\n\n")
    assert load_samples_jsonl(path) == [sample]


@pytest.mark.parametrize(
    ("rotation", "expected_yaw"),
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),
        ([2 ** -0.5, 0.0, 2 ** -0.5, 0.0], np.pi / 2),
        ([2 ** -0.5, 0.0, -(2 ** -0.5), 0.0], -np.pi / 2),
    ],
)
def test_training_pose_keeps_habitat_yaw_convention(rotation, expected_yaw):
    from supernav.methods.localnav_policy.rollout_geometry import _agent_pose

    adapter = NavigationFake()
    adapter.position = np.array([2.0, 0.0, -4.0])
    adapter.agent_rotation_wxyz = lambda session: rotation
    position, yaw = _agent_pose(adapter, adapter.session)
    np.testing.assert_array_equal(position, adapter.position)
    assert yaw == pytest.approx(expected_yaw)


def test_training_geometry_rejects_occluded_targets_and_resamples_paths():
    from supernav.methods.localnav_policy.rollout_geometry import (
        CameraIntrinsics, resample_polyline, select_hop_target,
    )

    points = resample_polyline([[0, 0, 0], [0, 0, 0], [0, 0, -4]], spacing=1.0)
    assert points[:, 2].tolist() == [0.0, -1.0, -2.0, -3.0, -4.0]
    camera = CameraIntrinsics(64, 64, 90.0)
    selected = select_hop_target(points, np.zeros(5), camera, 1.5, np.full((64, 64), 4.0))
    assert selected[0] == 4
    assert selected[1][0] == pytest.approx(0.5)
    assert select_hop_target(points, np.zeros(5), camera, 1.5, np.ones((64, 64))) is None


def test_training_hop_sampling_seeds_native_rng_deterministically():
    from supernav.methods.localnav_policy.rollout_geometry import _sample_hop_endpoints

    class SeededPathfinder(FlatPathfinder):
        def seed(self, value):
            self.rng = random.Random(value)

        def get_random_navigable_point(self):
            return np.array([self.rng.uniform(-3, 3), 0.0, self.rng.uniform(-3, 3)])

    first = _sample_hop_endpoints(SeededPathfinder(), random.Random(42), 1.0, 5.0)
    second = _sample_hop_endpoints(SeededPathfinder(), random.Random(42), 1.0, 5.0)
    assert first[2] == second[2]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert 1.0 <= first[2] <= 5.0


def test_training_geometry_rejects_invalid_camera_and_path_inputs():
    from supernav.methods.localnav_policy.rollout_geometry import (
        CameraIntrinsics, resample_polyline, select_hop_target,
    )

    with pytest.raises(ValueError, match="dimensions"):
        CameraIntrinsics(0, 64, 90)
    with pytest.raises(ValueError, match="field of view"):
        CameraIntrinsics(64, 64, 180)
    with pytest.raises(ValueError, match="3D"):
        resample_polyline([[0, 0, 0], [float("nan"), 0, 1]], spacing=0.25)
    with pytest.raises(ValueError, match="one yaw"):
        select_hop_target([[0, 0, 0], [0, 0, -2]], [], CameraIntrinsics(64, 64, 90), 1.5, np.ones((64, 64)))
