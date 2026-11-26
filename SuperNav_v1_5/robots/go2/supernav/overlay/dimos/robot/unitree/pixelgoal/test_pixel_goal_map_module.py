# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PixelGoalMapNavModule tests: skills, hop slot, session state machine, exclusion."""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
import numpy as np
import pytest

from dimos.agents.skill_result import SkillResult
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.go2.surround_camera import CAMERA_DIRECTIONS
from dimos.robot.unitree.pixelgoal.pixel_goal_map import PixelGoalMapConfig
from dimos.robot.unitree.pixelgoal.pixel_goal_map_module import (
    PixelGoalMapNavModule,
    _HopMotionRecorder,
)

FRAME_W, FRAME_H = 1600, 1200


class FakeMotion:
    """UnitreeSkillContainer stand-in returning real SkillResult objects."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def drive(self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0) -> Any:
        self.calls.append(("drive", {"forward": forward, "left": left, "degrees": degrees}))
        return SkillResult.ok("moved")

    def rotate_in_place(
        self, degrees: float, max_duration_s: float = 5.0, angular_speed_limit: float = 0.4
    ) -> Any:
        self.calls.append(("rotate", {"degrees": degrees}))
        return SkillResult.ok("rotated")

    def stop_movement(self) -> Any:
        self.calls.append(("stop", {}))
        return SkillResult.ok("stopped")


class FakeNomad:
    """NomadStatusSpec stand-in for cross-engine exclusion tests."""

    def __init__(self, hop_running: bool = False, raises: bool = False) -> None:
        self._hop_running = hop_running
        self._raises = raises

    def nomad_status(self, include_images: bool = True) -> Any:
        if self._raises:
            raise RuntimeError("nomad module unreachable")
        return json.dumps(
            {"ok": True, "hop_running": self._hop_running, "status": {}, "events": []}
        )


class FakePlanner:
    """ReplanningAStarPlannerSpec stand-in recording cancel/complete/reset."""

    def __init__(self) -> None:
        self.cancels = 0
        self.completes = 0
        self.resets = 0

    def cancel_goal(self) -> bool:
        self.cancels += 1
        return True

    def complete_goal(self) -> bool:
        self.completes += 1
        return True

    def reset_navigation_map(self) -> bool:
        self.resets += 1
        return True


class FakeMapper:
    def __init__(self) -> None:
        self.resets = 0

    def reset_voxel_map(self, timeout_s: float = 5.0) -> bool:
        self.resets += 1
        return True


class NullTf:
    """TF stand-in with no buffered transforms (forces the odom fallback)."""

    def get(self, *_args: object, **_kwargs: object) -> None:
        return None

    def stop(self) -> None:
        pass


def _frame(ts: float = 1000.0) -> Image:
    rgb = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    return Image.from_numpy(rgb, format=ImageFormat.RGB, ts=ts)


def _wall_map(distance_m: float = 2.0, frame_id: str = "world") -> PointCloud2:
    """A small vertical wall straight ahead at camera height, plus floor points."""
    wall = np.array(
        [
            [distance_m - 0.02, 0.0, 0.6],
            [distance_m, 0.0, 0.6],
            [distance_m + 0.02, 0.0, 0.6],
            [distance_m, 0.02, 0.65],
            [distance_m, -0.02, 0.55],
        ]
    )
    floor = np.column_stack((np.linspace(-1.0, 3.0, 40), np.full(40, 1.5), np.zeros(40)))
    return PointCloud2.from_numpy(np.vstack((wall, floor)), frame_id=frame_id, timestamp=1000.0)


def _free_costmap(frame_id: str = "world") -> OccupancyGrid:
    return OccupancyGrid(
        grid=np.full((100, 100), CostValues.FREE, dtype=np.int8),
        resolution=0.1,
        origin=Pose(position=Vector3(-5.0, -5.0, 0.0)),
        frame_id=frame_id,
    )


@pytest.fixture()
def module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S", "9999999999")
    mod = PixelGoalMapNavModule(live_dir=str(tmp_path / "live"))
    mod._config = PixelGoalMapConfig(map_timeout_s=0.5, deadline_s=30.0)
    mod._motion = FakeMotion()
    mod._planner_spec = FakePlanner()
    mod._tf = NullTf()
    yield mod
    mod.stop()  # joins the hop thread


def _wait_hop_done(module: PixelGoalMapNavModule, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = json.loads(module.pixelgoal_status(include_images=False))
        if not status["hop_running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("pixel-goal map hop did not finish in time")


def _wait_phase(module: PixelGoalMapNavModule, phase: str, timeout_s: float = 5.0) -> None:
    """Wait until the session state machine reaches a phase (goal published)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if module._phase == phase:
            return
        time.sleep(0.02)
    raise AssertionError(f"session never reached phase {phase!r} (phase={module._phase!r})")


def _prime_mapping(module: PixelGoalMapNavModule, world_map: PointCloud2) -> None:
    module._on_global_map(world_map)
    module._on_costmap(_free_costmap())
    module._on_odom(PoseStamped(position=[0.0, 0.0, 0.3], frame_id="world"))
    module.global_map.get_next = lambda timeout: world_map  # type: ignore[method-assign]


# ---------------------------------------------------------------------- validation
def test_navigate_validates_point_and_direction(module: PixelGoalMapNavModule) -> None:
    module._on_surround_image("front", _frame())
    assert json.loads(module.pixelgoal_navigate(point=[1.5, 0.5]))["code"] == "invalid_point"
    assert json.loads(module.pixelgoal_navigate(point=[]))["code"] == "invalid_point"
    assert (
        json.loads(module.pixelgoal_navigate(point=[0.5, 0.5], direction="up"))["code"]
        == "invalid_direction"
    )
    assert json.loads(module.pixelgoal_navigate(point=[], goal_id="missing"))["code"] == (
        "goal_not_found"
    )


def test_navigate_requires_frame(module: PixelGoalMapNavModule) -> None:
    payload = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert payload["ok"] is False
    assert payload["code"] == "no_frame"


def test_navigate_rejects_stale_camera_frame(
    module: PixelGoalMapNavModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S", "3")
    module._on_surround_image("left", _frame(ts=time.time() - 10.0))

    payload = json.loads(
        module.pixelgoal_navigate(point=[0.5, 0.8], direction="left")
    )

    assert payload["code"] == "stale_frame"
    assert payload["camera_health"]["stale"] == ["left"]


def test_rerun_click_maps_camera_entity_and_half_resolution_pixel(
    module: PixelGoalMapNavModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    module._on_surround_image("right", _frame())
    calls: list[dict[str, Any]] = []

    def navigate(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(module, "pixelgoal_navigate", navigate)
    module._on_pixel_click(PointStamped(x=400.0, y=300.0, frame_id="world/surround_right_image"))
    module._on_pixel_click(PointStamped(x=801.0, y=300.0, frame_id="world/surround_right_image"))

    assert calls == [{"point": [0.5, 0.5], "direction": "right"}]


def test_navigate_uses_exact_dashboard_frame(module: PixelGoalMapNavModule) -> None:
    world_map = _wall_map()
    _prime_mapping(module, world_map)
    started = json.loads(
        module.pixelgoal_navigate(
            point=[0.5, 0.5],
            frame_ts=1000.0,
            frame_jpeg_b64=_frame().to_base64(),
        )
    )
    assert started["ok"] is True
    deadline = time.monotonic() + 5.0
    while module._snapshot_image is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert module._snapshot_image is not None
    assert module._snapshot_image.ts == 1000.0
    module.pixelgoal_stop()
    _wait_hop_done(module)


def test_navigate_fisheye_model_rejects_side_direction(tmp_path: Path) -> None:
    mod = PixelGoalMapNavModule(live_dir=str(tmp_path / "live"), camera_model="fisheye")
    mod._motion = FakeMotion()
    mod._planner_spec = FakePlanner()
    try:
        payload = json.loads(mod.pixelgoal_navigate(point=[0.5, 0.5], direction="right"))
        assert payload["code"] == "invalid_direction"
        assert "fisheye" in payload["error"]
    finally:
        mod.stop()


def test_navigate_refuses_while_nomad_hop_running(module: PixelGoalMapNavModule) -> None:
    module._on_surround_image("front", _frame())
    module._nomad = FakeNomad(hop_running=True)
    payload = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert payload["code"] == "nomad_hop_running"


def test_dead_nomad_probe_never_blocks(module: PixelGoalMapNavModule) -> None:
    module._on_surround_image("front", _frame())
    module._nomad = FakeNomad(raises=True)
    # no mapping data: the hop starts and then ends needs_agent/no_map_data
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert started["ok"] is True
    final = _wait_hop_done(module)
    assert final["last_result"]["code"] == "no_map_data"


def test_hop_without_map_data_fails_cleanly(module: PixelGoalMapNavModule) -> None:
    module._on_surround_image("front", _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert started["ok"] is True and started["status"] == "started"
    final = _wait_hop_done(module)
    assert final["status"]["status"] == "needs_agent"
    assert final["last_result"]["code"] == "no_map_data"
    assert final["last_result"]["backend"] == "map"
    types = [event["type"] for event in final["events"]]
    assert "hop_start" in types and "hop_end" in types


# ---------------------------------------------------------------------- resolved surface hop
def test_front_hop_publishes_contact_goal_and_reaches(module: PixelGoalMapNavModule) -> None:
    world_map = _wall_map(2.0)
    _prime_mapping(module, world_map)
    module._on_surround_image("front", _frame())
    goals: list[PoseStamped] = []
    unsubscribe = module.goal_request.subscribe(goals.append)
    try:
        started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5], direction="front"))
        assert started["ok"] is True
        deadline = time.monotonic() + 5.0
        while not goals and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(goals) == 1
        # front-edge contact: 0.35 m short of the 2 m wall, yaw straight ahead
        assert (goals[0].x, goals[0].y, goals[0].z) == pytest.approx((1.65, 0.0, 0.0), abs=1e-6)
        assert goals[0].yaw == pytest.approx(0.0, abs=1e-3)
        assert goals[0].frame_id == "world"
        assert module._phase == "final"
        module._on_goal_reached(Bool(True))
        final = _wait_hop_done(module)
    finally:
        unsubscribe()
    assert final["status"]["status"] == "reached"
    assert final["last_result"]["ok"] is True
    assert final["last_result"]["target_kind"] == "elevated_surface"
    assert final["last_result"]["stop_point"] == pytest.approx([1.65, 0.0, 0.0])
    assert final["last_result"]["goals_published"] == 1
    types = [event["type"] for event in final["events"]]
    assert "ray_projected" in types and "surface_resolved" in types


def test_side_click_projects_ray_without_pre_turn(module: PixelGoalMapNavModule) -> None:
    """A right-camera click yields a map-frame goal to the right, no rotation call."""
    world_map = _wall_map(2.0)  # wall at +x, seen by the FRONT camera in world terms
    _prime_mapping(module, world_map)
    module._on_surround_image("right", _frame())
    goals: list[PoseStamped] = []
    unsubscribe = module.goal_request.subscribe(goals.append)
    try:
        started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5], direction="right"))
        assert started["ok"] is True and started["direction"] == "right"
        deadline = time.monotonic() + 5.0
        while not goals and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        unsubscribe()
        module.pixelgoal_stop()
        _wait_hop_done(module)
    # right-camera center ray points along -y; the wall at +x never intersects
    # the ROI, so the hop went exploring along -y — with zero rotate calls.
    assert goals, "expected at least an exploration goal"
    assert goals[0].y < 0.0
    motion = module._motion
    assert [call for call in motion.calls if call[0] == "rotate"] == []


def test_hop_slot_and_cooperative_stop(module: PixelGoalMapNavModule) -> None:
    world_map = _wall_map(2.0)
    _prime_mapping(module, world_map)
    module._on_surround_image("front", _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert started["ok"] is True
    # no goal_reached fed: the hop stays active and owns the slot
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if json.loads(module.pixelgoal_status(include_images=False))["status"].get("phase"):
            break
        time.sleep(0.02)
    second = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert second["code"] == "hop_already_running"
    stopped = json.loads(module.pixelgoal_stop())
    assert stopped["status"] == "cancel_requested"
    assert stopped["robot_stop"] == "sent"
    final = _wait_hop_done(module)
    assert final["last_result"]["code"] == "cancelled_by_agent"
    planner = module._planner_spec
    assert planner.cancels >= 1
    assert ("stop", {}) in module._motion.calls


def test_resume_by_goal_id_reprojects_on_latest_frame(module: PixelGoalMapNavModule) -> None:
    world_map = _wall_map(2.0)
    _prime_mapping(module, world_map)
    module._on_surround_image("front", _frame())
    first = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert first["ok"] is True
    module.pixelgoal_stop()
    final = _wait_hop_done(module)
    assert final["last_result"]["code"] == "cancelled_by_agent"

    resumed = json.loads(module.pixelgoal_navigate(point=[], goal_id=first["goal_id"]))
    assert resumed["ok"] is True and resumed["goal_id"] == first["goal_id"]
    _wait_phase(module, "final")
    module._on_goal_reached(Bool(True))
    final = _wait_hop_done(module)
    assert final["status"]["status"] == "reached"


# ---------------------------------------------------------------------- motion recorder
def test_motion_recorder_rate_limits_and_writes_meta(tmp_path: Path) -> None:
    recorder = _HopMotionRecorder(tmp_path, "localgoal-deadbeefcafe1234", hz=2.0)
    recorder.start()
    for k in range(10):  # 10 Hz offered, 2 Hz kept
        recorder.offer("front", _frame(ts=2000.0 + k * 0.1))
    recorder.finish({"status": "reached", "record_hz": 2.0})
    assert recorder.dir.name.endswith("_cafe1234")
    jpgs = list((recorder.dir / "front").glob("*.jpg"))
    assert 2 <= len(jpgs) <= 3
    meta = json.loads((recorder.dir / "meta.json").read_text())
    assert meta["goal_id"] == "localgoal-deadbeefcafe1234"
    assert meta["status"] == "reached"
    assert meta["frames_saved"] == len(jpgs)


def test_hop_motion_recorder_captures_surround_frames(
    module: PixelGoalMapNavModule, tmp_path: Path
) -> None:
    _prime_mapping(module, _wall_map(2.0))
    module._on_surround_image("front", _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5], direction="front"))
    assert started["ok"] is True
    _wait_phase(module, "final")
    for k in range(3):  # fresh frames while the hop runs, 1 s apart at 2 Hz pacing
        module._on_surround_image("front", _frame(ts=time.time() + 1.0 + k))
    module._on_goal_reached(Bool(True))
    final = _wait_hop_done(module)
    assert final["status"]["status"] == "reached"
    tail = "".join(c for c in started["goal_id"] if c.isalnum())[-8:]
    dirs = [p for p in (tmp_path / "motion").glob("hop_*_*") if p.is_dir()]
    assert len(dirs) == 1 and dirs[0].name.endswith(tail)
    jpgs = list((dirs[0] / "front").glob("*.jpg"))
    assert len(jpgs) == 3
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["goal_id"] == started["goal_id"]
    assert meta["status"] == "reached"
    assert meta["direction"] == "front"
    assert meta["frames_saved"] == 3
    assert module._motion_rec is None


def test_hop_motion_recorder_disabled_at_zero_hz(
    module: PixelGoalMapNavModule, tmp_path: Path
) -> None:
    module._motion_rec_hz = 0.0
    _prime_mapping(module, _wall_map(2.0))
    module._on_surround_image("front", _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5], direction="front"))
    assert started["ok"] is True
    _wait_phase(module, "final")
    module._on_goal_reached(Bool(True))
    _wait_hop_done(module)
    assert module._motion_rec is None
    assert not (tmp_path / "motion").exists()


def test_terminal_status_attaches_four_images_once(module: PixelGoalMapNavModule) -> None:
    world_map = _wall_map(2.0)
    _prime_mapping(module, world_map)
    for direction in ("front", "right", "back", "left"):
        module._on_surround_image(direction, _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    _wait_phase(module, "final")
    module._on_goal_reached(Bool(True))
    final = _wait_hop_done(module)
    assert final["last_result"]["goal_id"] == started["goal_id"]

    content = module.pixelgoal_status(include_images=True)
    assert isinstance(content, list)
    payload = json.loads(content[0]["text"])
    assert payload["camera_health"]["ok"] is True
    assert payload["2d_slam_map"]["status"] == "current"
    assert payload["2d_slam_map"]["source"] == "native_global_map_area_segmented"
    assert payload["2d_slam_map"]["map_frame"] == "world"
    assert payload["2d_slam_map"]["pose"]["frame_id"] == "world"
    assert payload["2d_slam_map"]["pose"]["position_m"]["z"] == 0.3
    labels = [item["text"] for item in content if item["type"] == "text"]
    for direction in ("front", "right", "back", "left"):
        assert direction in labels
    again = module.pixelgoal_status(include_images=True)
    assert isinstance(again, str)
    assert json.loads(again)["last_result"]["goal_id"] == started["goal_id"]


def test_terminal_slam_map_rejects_odom_in_different_frame(
    module: PixelGoalMapNavModule,
) -> None:
    module._on_costmap(_free_costmap(frame_id="world"))
    module._on_odom(PoseStamped(position=[1.0, 2.0, 0.0], frame_id="sport_mode_odom"))
    payload: dict[str, Any] = {}

    image_block = module._terminal_2d_slam_map_content(payload)

    assert image_block is not None
    slam_map = payload["2d_slam_map"]
    assert slam_map["map_frame"] == "world"
    assert slam_map["status"] == "pose_unavailable"
    assert "pose" not in slam_map


def test_terminal_slam_map_requires_robot_pose(module: PixelGoalMapNavModule) -> None:
    module._on_costmap(_free_costmap())
    payload: dict[str, Any] = {}

    assert module._terminal_2d_slam_map_content(payload) is not None
    assert payload["2d_slam_map"]["status"] == "pose_unavailable"
    assert "pose" not in payload["2d_slam_map"]


def test_terminal_status_reports_and_omits_stale_view(
    module: PixelGoalMapNavModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S", "3")
    now = time.time()
    module._last_result = {"goal_id": "goal-stale", "status": "reached"}
    for direction in CAMERA_DIRECTIONS:
        module._on_surround_image(
            direction,
            _frame(ts=now - 10.0 if direction == "left" else now),
        )

    content = module.pixelgoal_status(include_images=True)

    assert isinstance(content, list)
    payload = json.loads(content[0]["text"])
    assert payload["camera_health"]["stale"] == ["left"]
    assert "left" not in [item.get("text") for item in content]
    assert module._terminal_images_goal_id is None


# ---------------------------------------------------------------------- exploration state machine
def test_exploration_waypoint_when_no_surface_in_roi(module: PixelGoalMapNavModule) -> None:
    """ROI without LiDAR returns: bounded exploration along the frozen ray."""
    off_axis_map = PointCloud2.from_numpy(
        np.column_stack((np.full(10, -3.0), np.linspace(3.0, 4.0, 10), np.full(10, 0.5))),
        frame_id="world",
        timestamp=1000.0,
    )
    _prime_mapping(module, off_axis_map)
    module._on_surround_image("front", _frame())
    goals: list[PoseStamped] = []
    unsubscribe = module.goal_request.subscribe(goals.append)
    try:
        started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
        assert started["ok"] is True
        deadline = time.monotonic() + 5.0
        while not goals and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(goals) == 1
        # exploration waypoint: forward along +x, within the step budget
        assert 0.4 <= goals[0].x <= 1.6
        assert abs(goals[0].y) <= 2.0
        assert module._phase == "explore"
        # a failed explore leg re-projects and publishes a different waypoint
        module._on_goal_reached(Bool(False))
        deadline = time.monotonic() + 5.0
        while len(goals) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(goals) == 2
        assert module._exploration_failures == 1
        module.pixelgoal_stop()
        final = _wait_hop_done(module)
    finally:
        unsubscribe()
    assert final["last_result"]["code"] == "cancelled_by_agent"


def test_recovery_exhaustion_ends_needs_agent(module: PixelGoalMapNavModule) -> None:
    module._latest_costmap = OccupancyGrid(
        grid=np.full((100, 100), CostValues.OCCUPIED, dtype=np.int8),
        resolution=0.1,
        origin=Pose(position=Vector3(-5.0, -5.0, 0.0)),
        frame_id="world",
    )
    module._latest_odom = PoseStamped(position=[0.0, 0.0, 0.0], frame_id="world")
    module._ray_origin = np.array([0.0, 0.0, 0.6])
    module._ray_direction = np.array([1.0, 0.0, 0.0])
    module._snapshot_frame_id = "world"
    module._goal_published = True
    module._phase = "explore"
    module._resolve_pixel_target = lambda _world_map: None  # type: ignore[method-assign]
    module._latest_map = PointCloud2.from_numpy(np.ones((3, 3)), frame_id="world")
    published: list[PoseStamped] = []
    unsubscribe = module.goal_request.subscribe(published.append)
    generation = module._session_generation
    try:
        module._on_goal_reached(Bool(False))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if module._session_done.is_set():
                break
            time.sleep(0.02)
    finally:
        unsubscribe()
    assert module._session_done.is_set()
    assert module._session_end_reason == "recovery_exhausted"
    assert module._session_generation > generation
    assert published == []


def test_resolved_goal_rejects_occupied_stop_and_occluding_wall(
    module: PixelGoalMapNavModule,
) -> None:
    grid = _free_costmap()
    module._on_costmap(grid)
    module._ray_origin = np.array([0.0, 0.0, 0.6])
    target = np.array([2.0, 0.0, 0.5])
    assert not module._resolved_goal_blocked(target)
    grid.grid[50, 60] = CostValues.OCCUPIED
    assert module._resolved_goal_blocked(target)
    grid.grid[50, 60] = CostValues.FREE
    grid.grid[50, 66] = CostValues.OCCUPIED
    assert module._resolved_goal_blocked(target)
    grid.grid[50, 66] = CostValues.FREE
    grid.grid[50, 70] = CostValues.OCCUPIED  # target surface itself is permitted
    assert not module._resolved_goal_blocked(target)


def test_final_goal_stall_cancels_and_reports_needs_agent(
    module: PixelGoalMapNavModule,
) -> None:
    _prime_mapping(module, _wall_map(2.0))
    module._on_surround_image("front", _frame())
    started = json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))
    assert started["ok"]
    _wait_phase(module, "final")
    with module._state_lock:
        module._final_progress_at = time.monotonic() - 11.0
    result = _wait_hop_done(module)
    assert result["last_result"]["code"] == "final_goal_stalled"
    assert result["status"]["status"] == "needs_agent"
    assert module._planner_spec.cancels >= 1


def test_final_progress_resets_stall_window(module: PixelGoalMapNavModule) -> None:
    module._goal_published = True
    module._phase = "final"
    module._final_target = np.array([2.0, 0.0, 0.0])
    module._final_stop = np.array([1.65, 0.0, 0.0])
    module._final_progress_distance = 1.65
    module._final_progress_at = time.monotonic() - 9.0
    module._on_odom(PoseStamped(position=[0.25, 0.0, 0.0], frame_id="world"))
    assert module._final_progress_distance == pytest.approx(1.4)
    assert time.monotonic() - module._final_progress_at < 1.0


def test_unchanged_unknown_frontier_does_not_repeat(module: PixelGoalMapNavModule) -> None:
    grid = np.full((100, 100), CostValues.UNKNOWN, dtype=np.int8)
    grid[48:53, 30:36] = CostValues.FREE
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    module._on_costmap(costmap)
    module._on_odom(PoseStamped(position=[0.0, 0.0, 0.0], frame_id="world"))
    module._ray_origin = np.array([0.0, 0.0, 0.6])
    module._ray_direction = np.array([1.0, 0.0, 0.0])
    module._snapshot_frame_id = "world"
    module._goal_published = True
    assert module._publish_exploration_goal()
    assert module._unknown_frontier_goal is not None
    assert not module._publish_exploration_goal()
    refreshed = OccupancyGrid(
        grid=grid.copy(), resolution=costmap.resolution,
        origin=costmap.origin, frame_id="world",
    )
    refreshed.grid[50, 58:61] = CostValues.FREE
    module._on_costmap(refreshed)
    module._on_odom(PoseStamped(position=[0.5, 0.0, 0.0], frame_id="world"))
    assert module._publish_exploration_goal()


def test_blocked_final_target_reports_needs_agent_without_planner_goal(
    module: PixelGoalMapNavModule,
) -> None:
    _prime_mapping(module, _wall_map(2.0))
    costmap = _free_costmap()
    costmap.grid[50, 60] = CostValues.OCCUPIED
    module._on_costmap(costmap)
    module._resolve_pixel_target = lambda _: (np.array([2.0, 0.0, 0.5]), "elevated_surface")  # type: ignore[method-assign]
    goals: list[PoseStamped] = []
    unsubscribe = module.goal_request.subscribe(goals.append)
    try:
        module._on_surround_image("front", _frame())
        assert json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))["ok"]
        result = _wait_hop_done(module)
    finally:
        unsubscribe()
    assert result["last_result"]["code"] == "blocked_pixel_target"
    assert result["status"]["status"] == "needs_agent"
    assert goals == []


def test_planner_final_failure_ends_without_waiting_for_stall(
    module: PixelGoalMapNavModule,
) -> None:
    _prime_mapping(module, _wall_map(2.0))
    module._on_surround_image("front", _frame())
    assert json.loads(module.pixelgoal_navigate(point=[0.5, 0.5]))["ok"]
    _wait_phase(module, "final")
    module._on_goal_reached(Bool(False))
    result = _wait_hop_done(module)
    assert result["last_result"]["code"] == "final_goal_failed"


# ---------------------------------------------------------------------- front-edge contact (ported)
def test_final_goal_completes_when_robot_front_reaches_target(
    module: PixelGoalMapNavModule,
) -> None:
    module._goal_published = True
    module._phase = "final"
    module._final_target = np.array([2.0, 0.0, 0.0])
    module._on_odom(PoseStamped(position=[1.5, 0.0, 0.0], frame_id="world"))
    deadline = time.monotonic() + 2.0
    planner = module._planner_spec
    while planner.completes == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert planner.completes == 1
    assert module._contact_completion_requested


def test_final_goal_does_not_complete_when_laterally_far(module: PixelGoalMapNavModule) -> None:
    module._goal_published = True
    module._phase = "final"
    module._final_target = np.array([2.0, 0.0, 0.0])
    module._on_odom(PoseStamped(position=[1.5, 0.3, 0.0], frame_id="world"))
    assert module._planner_spec.completes == 0
    assert not module._contact_completion_requested


def test_final_goal_completes_at_center_stop_without_final_rotation(
    module: PixelGoalMapNavModule,
) -> None:
    module._goal_published = True
    module._phase = "final"
    module._final_target = np.array([2.0, 0.0, 0.0])
    module._final_stop = np.array([1.65, 0.0, 0.0])
    odom = PoseStamped(
        position=[1.65, 0.0, 0.0],
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, math.pi / 2.0)),
        frame_id="world",
    )
    module._on_odom(odom)
    deadline = time.monotonic() + 2.0
    planner = module._planner_spec
    while planner.completes == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert planner.completes == 1


# ---------------------------------------------------------------------- map reset skill
def test_reset_pixel_goal_map_resets_planner_and_mapper(module: PixelGoalMapNavModule) -> None:
    module._mapper_spec = FakeMapper()
    module._goal_published = True
    module._phase = "explore"
    result = module.reset_pixel_goal_map()
    assert result.startswith("Pixel goal map reset")
    assert module._planner_spec.resets == 1
    assert module._mapper_spec.resets == 1
    assert module._latest_map is None
    assert module._latest_map_frame_id == ""


def test_reset_pixel_goal_map_reports_missing_mapper(module: PixelGoalMapNavModule) -> None:
    module._mapper_spec = None
    result = module.reset_pixel_goal_map()
    assert "incomplete" in result
    assert module._planner_spec.resets == 1


def test_external_stop_movement_ends_active_session(module: PixelGoalMapNavModule) -> None:
    module._goal_published = True
    module._phase = "explore"
    module._on_stop_movement(Bool(True))
    assert module._session_done.is_set()
    assert module._session_end_reason == "movement_stopped"
    assert module._phase == "idle"
    # a stray external stop with no active session is ignored
    module._session_done = threading.Event()
    module._on_stop_movement(Bool(True))
    assert not module._session_done.is_set()
