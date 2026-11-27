# Copyright 2025-2026 Dimensional Inc.
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

import base64
from io import BytesIO
import difflib
import json
import math
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage
import numpy as np
from PIL import Image as PILImage

from dimos.agents.skill_result import SkillResult
from dimos.core.coordination.blueprints import DisabledModuleProxy
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.base import NavigationState
import dimos.robot.unitree.unitree_skill_container as unitree_skill_module
from dimos.robot.unitree.unitree_skill_container import (
    _UNITREE_COMMANDS,
    DEFAULT_DRIVE_TIMEOUT_S,
    DEFAULT_FLOOR_PLAN_BOUNDARY_WALL_WIDTH_M,
    DEFAULT_RELATIVE_MOVE_TIMEOUT_S,
    DRIVE_TIMEOUT_ENV,
    RELATIVE_MOVE_TIMEOUT_ENV,
    UnitreeSkillContainer,
    _env_float,
)


class StubNavigation(Module):
    @rpc
    def set_goal(self, goal: PoseStamped) -> bool:
        return True

    @rpc
    def get_state(self) -> NavigationState:
        return NavigationState.IDLE

    @rpc
    def is_goal_reached(self) -> bool:
        return False

    @rpc
    def cancel_goal(self) -> bool:
        return True


class StubGO2Connection(Module):
    @rpc
    def publish_request(
        self, topic: str, data: dict[str, Any], timeout_s: float | None = None
    ) -> dict[Any, Any]:
        return {}


class MockedUnitreeSkill(UnitreeSkillContainer):
    pass


class DirectNavigation:
    def __init__(
        self,
        *,
        motion_diagnostics: dict[str, Any] | None = None,
        state: NavigationState = NavigationState.IDLE,
        last_failure: str | None = None,
    ) -> None:
        self.motion_diagnostics = motion_diagnostics or {}
        self.state = state
        self.last_failure = last_failure
        self.recorded_motion_results: list[dict[str, Any]] = []

    def set_goal(self, goal: PoseStamped) -> bool:
        return True

    def get_state(self) -> NavigationState:
        return self.state

    def is_goal_reached(self) -> bool:
        return False

    def cancel_goal(self) -> bool:
        return True

    def get_last_failure(self) -> str | None:
        return self.last_failure

    def get_motion_diagnostics(self) -> dict[str, Any]:
        return dict(self.motion_diagnostics)

    def record_motion_result(self, result: dict[str, Any]) -> None:
        self.recorded_motion_results.append(result)
        self.motion_diagnostics["measured_result"] = result


class DirectUnitreeSkill(UnitreeSkillContainer):
    def __init__(self) -> None:
        # Keep direct unit tests independent from Module/RPC/LCM setup.
        pass


class NoDiagnosticsNavigation:
    def get_state(self) -> NavigationState:
        return NavigationState.IDLE

    def is_goal_reached(self) -> bool:
        return False

    def cancel_goal(self) -> bool:
        return True


class TimeoutOnceGO2Connection:
    def __init__(self) -> None:
        self.calls = 0

    def publish_request(
        self, topic: str, data: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "ok": False,
            "error": "publish_request_timeout",
            "timed_out": True,
            "topic": topic,
            "request": data,
            "timeout_s": 0.05,
            "status": {"code": -408},
            "command_may_have_been_accepted": True,
        }


class FailedPublishGO2Connection:
    def __init__(self) -> None:
        self.calls = 0

    def publish_request(
        self, topic: str, data: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "ok": False,
            "error": "publish_request_failed",
            "timed_out": False,
            "topic": topic,
            "request": data,
            "exception": "Data channel is not open",
            "command_may_have_been_accepted": False,
        }


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> PoseStamped:
    return PoseStamped(
        frame_id="world",
        position=Vector3(x, y, z),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, math.radians(yaw_deg))),
    )


def _pose_at_yaw(yaw_deg: float) -> PoseStamped:
    return _pose(yaw_deg=yaw_deg)


def _costmap() -> OccupancyGrid:
    grid = np.zeros((24, 24), dtype=np.int8)
    grid[:2, :] = 100
    grid[:, :2] = 100
    return OccupancyGrid(grid=grid, resolution=0.5, frame_id="world")


def _global_map() -> PointCloud2:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    return PointCloud2.from_numpy(points, frame_id="world", timestamp=123.5)


def _camera_image() -> Image:
    data = np.zeros((24, 32, 3), dtype=np.uint8)
    data[:, :, 0] = 255
    return Image(data=data, format=ImageFormat.RGB, frame_id="camera_optical")


def _image_from_block(block: dict[str, object]) -> PILImage.Image:
    data = str(block["data"])
    return PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")


class FakeTransform:
    def __init__(self, poses: list[PoseStamped]) -> None:
        self.poses = poses

    def to_pose(self) -> PoseStamped:
        return self.poses[-1]


class FakeTf:
    def __init__(self, poses: list[PoseStamped]) -> None:
        self.poses = poses

    def get(self, source: str, target: str) -> FakeTransform:
        assert (source, target) == ("world", "base_link")
        return FakeTransform(self.poses)


class OdomUnitreeSkill(DirectUnitreeSkill):
    def __init__(
        self,
        sport_poses: list[PoseStamped],
        *,
        lidar_poses: list[PoseStamped] | None = None,
    ) -> None:
        self.sport_poses = sport_poses
        self.lidar_poses = lidar_poses if lidar_poses is not None else sport_poses

    def _current_base_pose(self) -> PoseStamped | None:
        return self.lidar_poses[-1] if self.lidar_poses else None

    def _current_direct_motion_pose(self) -> PoseStamped | None:
        return self.sport_poses[-1] if self.sport_poses else None


class RecordingTeleCmdVel:
    def __init__(self, poses: list[PoseStamped], *, yaw_step_deg: float) -> None:
        self.poses = poses
        self.yaw_step_deg = yaw_step_deg
        self.commands: list[Any] = []

    def publish(self, twist: Any) -> None:
        self.commands.append(twist)
        if abs(twist.angular.z) > 0.0 and self.yaw_step_deg:
            current_yaw = math.degrees(self.poses[-1].yaw)
            direction = 1.0 if twist.angular.z > 0 else -1.0
            self.poses.append(_pose_at_yaw(current_yaw + direction * self.yaw_step_deg))


class RecordingDriveTeleCmdVel:
    def __init__(self, poses: list[PoseStamped], *, dt_s: float) -> None:
        self.poses = poses
        self.dt_s = dt_s
        self.commands: list[Any] = []

    def publish(self, twist: Any) -> None:
        self.commands.append(twist)
        if (
            abs(twist.linear.x) <= 0.0
            and abs(twist.linear.y) <= 0.0
            and abs(twist.angular.z) <= 0.0
        ):
            return
        pose = self.poses[-1]
        yaw = pose.yaw
        dx = math.cos(yaw) * twist.linear.x - math.sin(yaw) * twist.linear.y
        dy = math.sin(yaw) * twist.linear.x + math.cos(yaw) * twist.linear.y
        self.poses.append(
            _pose(
                x=pose.x + dx * self.dt_s,
                y=pose.y + dy * self.dt_s,
                yaw_deg=math.degrees(yaw + twist.angular.z * self.dt_s),
            )
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


class DelayedOdomDriveTeleCmdVel:
    def __init__(
        self,
        poses: list[PoseStamped],
        clock: FakeClock,
        *,
        startup_delay_s: float,
        progress_steps_m: list[float],
    ) -> None:
        self.poses = poses
        self.clock = clock
        self.startup_delay_s = startup_delay_s
        self.progress_steps_m = progress_steps_m
        self.commands: list[Any] = []

    def publish(self, twist: Any) -> None:
        self.commands.append(twist)
        if (
            abs(twist.linear.x) <= 0.0
            and abs(twist.linear.y) <= 0.0
            and abs(twist.angular.z) <= 0.0
        ):
            return
        if self.clock.now < self.startup_delay_s or not self.progress_steps_m:
            return
        pose = self.poses[-1]
        self.poses.append(_pose(x=pose.x + self.progress_steps_m.pop(0)))


class VerticalSportDriveTeleCmdVel:
    def __init__(self, sport_poses: list[PoseStamped]) -> None:
        self.sport_poses = sport_poses
        self.commands: list[Any] = []

    def publish(self, twist: Any) -> None:
        self.commands.append(twist)
        if abs(twist.linear.x) <= 0.0 and abs(twist.linear.y) <= 0.0:
            return
        pose = self.sport_poses[-1]
        self.sport_poses.append(_pose(x=pose.x, y=pose.y, z=pose.z + 0.02))


class RaisingTeleCmdVel:
    def publish(self, twist: Any) -> None:
        del twist
        raise RuntimeError("cmd_vel transport missing")


class CompletingNavigation(DirectNavigation):
    def __init__(self, poses: list[PoseStamped], end_pose: PoseStamped) -> None:
        super().__init__(state=NavigationState.IDLE)
        self.poses = poses
        self.end_pose = end_pose
        self.set_goal_calls = 0

    def set_goal(self, goal: PoseStamped) -> bool:
        del goal
        self.set_goal_calls += 1
        self.poses.append(self.end_pose)
        return True

    def is_goal_reached(self) -> bool:
        return True


class TimeoutRpc:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def __call__(self) -> NavigationState:
        raise AssertionError("relative_move should use bounded RPC calls")

    def call_with_timeout(self, rpc_timeout: float) -> NavigationState:
        self.timeouts.append(rpc_timeout)
        raise TimeoutError(f"simulated get_state timeout after {rpc_timeout}")


class BoundedCancelRpc:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def __call__(self) -> bool:
        raise AssertionError("rotate_in_place should bound navigation cancel_goal")

    def call_with_timeout(self, rpc_timeout: float) -> bool:
        self.timeouts.append(rpc_timeout)
        return True


class ZeroFirstCancelRpc:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.timeouts: list[float] = []

    def __call__(self) -> bool:
        raise AssertionError("stop_movement should bound navigation cancel_goal")

    def call_with_timeout(self, rpc_timeout: float) -> bool:
        self.timeouts.append(rpc_timeout)
        self.events.append("cancel_goal")
        assert self.events[0] == "zero_velocity"
        return True


class TimeoutStateNavigation(DirectNavigation):
    def __init__(self) -> None:
        super().__init__(state=NavigationState.FOLLOWING_PATH)
        self.get_state = TimeoutRpc()  # type: ignore[method-assign]
        self.cancel_calls = 0

    def cancel_goal(self) -> bool:
        self.cancel_calls += 1
        return True


class BoundedCancelNavigation(DirectNavigation):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_goal = BoundedCancelRpc()  # type: ignore[method-assign]


class ZeroFirstCancelNavigation(DirectNavigation):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.cancel_goal = ZeroFirstCancelRpc(events)  # type: ignore[method-assign]


class OrderedTeleCmdVel:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.commands: list[Any] = []

    def publish(self, twist: Any) -> None:
        self.commands.append(twist)
        if (
            abs(twist.linear.x) <= 0.0
            and abs(twist.linear.y) <= 0.0
            and abs(twist.angular.z) <= 0.0
        ):
            self.events.append("zero_velocity")
        else:
            self.events.append("velocity")


def test_pounce(agent_setup) -> None:
    history = agent_setup(
        blueprints=[
            MockedUnitreeSkill.blueprint(),
            StubNavigation.blueprint(),
            StubGO2Connection.blueprint(),
        ],
        messages=[HumanMessage("Pounce! Use the execute_sport_command tool.")],
    )

    response = history[-1].content.lower()
    assert "pounce" in response


def test_stand_up_reports_publish_request_timeout_diagnostic() -> None:
    skill = DirectUnitreeSkill()
    connection = TimeoutOnceGO2Connection()
    skill._connection = connection
    skill._navigation = DirectNavigation()

    result = skill.stand_up()

    assert connection.calls == 1
    assert result.success is False
    assert result.error_code == "TIMEOUT"
    assert "may or may not have been accepted" in result.message
    response = result.metadata["command_response"]
    assert response["error"] == "publish_request_timeout"
    assert response["timed_out"] is True


def test_stand_up_reports_publish_request_failure_diagnostic() -> None:
    skill = DirectUnitreeSkill()
    connection = FailedPublishGO2Connection()
    skill._connection = connection
    skill._navigation = DirectNavigation()

    result = skill.stand_up()

    assert connection.calls == 1
    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"
    assert "publish_request_failed" in result.message
    assert "Data channel is not open" in result.message
    assert result.metadata["command_response"]["ok"] is False


def test_stop_movement_returns_structured_error_when_cmd_vel_publish_raises() -> None:
    skill = DirectUnitreeSkill()
    skill.tele_cmd_vel = RaisingTeleCmdVel()
    skill._navigation = DirectNavigation()

    result = skill.stop_movement()

    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"
    assert result.metadata["navigation_cancelled"] is True
    assert result.metadata["zero_velocity_sent"] is False
    response = result.metadata["zero_velocity_response"]
    assert response["error"] == "cmd_vel_publish_failed"
    assert response["exception_type"] == "RuntimeError"
    assert response["exception"] == "cmd_vel transport missing"


def test_stop_movement_sends_zero_velocity_before_bounded_navigation_cancel() -> None:
    events: list[str] = []
    skill = DirectUnitreeSkill()
    skill.tele_cmd_vel = OrderedTeleCmdVel(events)
    navigation = ZeroFirstCancelNavigation(events)
    skill._navigation = navigation

    result = skill.stop_movement()

    assert result.success is True
    assert result.metadata["zero_velocity_sent"] is True
    assert result.metadata["navigation_cancelled"] is True
    assert events == ["zero_velocity", "cancel_goal"]
    assert navigation.cancel_goal.timeouts == [1.0]
    assert skill.tele_cmd_vel.commands[-1].linear.x == 0.0
    assert skill.tele_cmd_vel.commands[-1].angular.z == 0.0


def test_relative_move_timeout_default_can_come_from_env(monkeypatch) -> None:
    monkeypatch.setenv(RELATIVE_MOVE_TIMEOUT_ENV, "12.5")

    assert _env_float(RELATIVE_MOVE_TIMEOUT_ENV, DEFAULT_RELATIVE_MOVE_TIMEOUT_S) == 12.5


def test_drive_timeout_default_can_come_from_env(monkeypatch) -> None:
    monkeypatch.setenv(DRIVE_TIMEOUT_ENV, "4.25")

    assert _env_float(DRIVE_TIMEOUT_ENV, DEFAULT_DRIVE_TIMEOUT_S) == 4.25


def test_relative_move_normal_completion_still_reports_arrived() -> None:
    poses = [_pose()]
    skill = DirectUnitreeSkill()
    skill._tf = FakeTf(poses)
    skill._navigation = CompletingNavigation(poses, _pose(x=1.0))
    skill._relative_move_settle_s = 0.0

    result = skill.relative_move(forward=1.0)

    assert result.success is True
    assert result.metadata["status"] == "ARRIVED"
    assert result.metadata["set_goal_accepted"] is True
    assert result.metadata["goal_reached"] is True
    assert math.isclose(result.metadata["measured_delta"]["linear_m"], 1.0, abs_tol=1e-6)


def test_drive_uses_sport_mode_pose_when_lidar_pose_stalls() -> None:
    sport_poses = [_pose()]
    lidar_poses = [_pose()]
    tele_cmd_vel = RecordingDriveTeleCmdVel(sport_poses, dt_s=0.2)
    skill = OdomUnitreeSkill(sport_poses, lidar_poses=lidar_poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.0
    skill._drive_timeout_s = 1.0
    skill._drive_linear_gain = 2.0
    skill._drive_max_linear_speed_mps = 1.0

    result = skill.drive(forward=0.5)

    assert result.success is True
    assert result.metadata["status"] == "ARRIVED"
    assert result.metadata["zero_velocity_sent"] is True
    assert result.metadata["arrival_check"]["verified"] is True
    assert (
        result.metadata["residual_distance_m"] <= result.metadata["arrival_tolerance"]["linear_m"]
    )
    assert math.isclose(
        result.metadata["measured_delta"]["linear_m"],
        sport_poses[-1].x,
        abs_tol=1e-6,
    )
    assert lidar_poses[-1].x == 0.0
    assert tele_cmd_vel.commands[-1].linear.x == 0.0
    assert tele_cmd_vel.commands[-1].angular.z == 0.0


def test_drive_records_latest_motion_diagnostics_when_navigation_backend_exists() -> None:
    poses = [_pose()]
    tele_cmd_vel = RecordingDriveTeleCmdVel(poses, dt_s=0.2)
    navigation = DirectNavigation()
    skill = OdomUnitreeSkill(poses)
    skill._navigation = navigation
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.0
    skill._drive_timeout_s = 1.0
    skill._drive_linear_gain = 2.0
    skill._drive_max_linear_speed_mps = 1.0

    result = skill.drive(forward=0.5)

    assert result.success is True
    assert navigation.recorded_motion_results
    recorded = navigation.motion_diagnostics["measured_result"]
    assert recorded["tool_name"] == "drive"
    assert recorded["success"] is True
    assert recorded["status"] == "ARRIVED"
    assert recorded["precision_class"] in {"exact", "within_tolerance"}
    assert recorded["measured_delta"] == result.metadata["measured_delta"]
    assert recorded["physically_satisfied"] is True


def test_drive_records_heading_error_as_motion_diagnostics_heading_drift() -> None:
    skill = DirectUnitreeSkill()
    navigation = DirectNavigation()
    skill._navigation = navigation
    metadata = skill._drive_measurement_metadata(
        _pose(),
        _pose(x=0.25, yaw_deg=5.0),
        _pose(x=0.5, yaw_deg=20.0),
    )

    result = SkillResult.ok("drive complete", status="ARRIVED", **metadata)
    skill._trace_direct_movement_skill_result("drive", result, movement_kind="drive")

    recorded = navigation.motion_diagnostics["measured_result"]
    assert math.isclose(
        recorded["heading_drift_deg"],
        result.metadata["heading_error_deg"],
        abs_tol=1e-6,
    )


def test_drive_attaches_current_global_costmap_image(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    skill._2d_slam_map_view_size_m = 8.0

    result = skill.drive()

    assert result.success is True
    assert result.metadata["floor_plan"]["status"] == "current"
    assert result.metadata["floor_plan"]["source"] == "native_global_map_cleaned"
    assert result.metadata["floor_plan"]["pose_overlay"]["status"] == "current"
    assert result.metadata["floor_plan"]["cleaning"]["boundary_wall_width_m"] == 0.05
    assert result.metadata["area_segmentation"]["status"] == "current"
    assert result.metadata["area_segmentation"]["source"] == "native_global_map_area_segmented"
    assert result.metadata["area_segmentation"]["wall_inflation_m"] == 0.06
    assert result.metadata["2d_slam_map"]["status"] == "current"
    assert result.metadata["2d_slam_map"]["source"] == "global_costmap"
    assert len(result.content_blocks) == 3
    assert result.content_blocks[0]["type"] == "image"
    assert result.content_blocks[0]["mimeType"] == "image/png"
    assert result.content_blocks[1]["mimeType"] == "image/png"
    assert result.content_blocks[2]["mimeType"] == "image/png"
    floor_plan_image = _image_from_block(result.content_blocks[0])
    assert any(
        floor_plan_image.getpixel((x, y)) == (230, 0, 0)
        for y in range(floor_plan_image.height)
        for x in range(floor_plan_image.width)
    )
    encoded = result.agent_encode()
    assert encoded[0]["type"] == "text"
    assert encoded[1] == result.content_blocks[0]


def test_memory_navigation_attaches_segmented_slam_map(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    skill = OdomUnitreeSkill([_pose(x=5.0, y=5.0)])
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    result = SkillResult.ok("memory navigation complete", status="ARRIVED")

    skill._attach_2d_slam_map("go2_navigate_to_memory_node", result)

    slam_map = result.metadata["2d_slam_map"]
    assert slam_map["status"] == "current"
    assert slam_map["source"] == "native_global_map_area_segmented"
    assert slam_map["wall_inflation_m"] == 0.06
    assert slam_map["room_count"] >= 1
    assert slam_map["slam_area_nodes"]
    assert len(result.content_blocks) == 1
    assert result.content_blocks[0]["type"] == "image"
    assert result.content_blocks[0]["mimeType"] == "image/png"


def test_static_slam_snapshot_attaches_segmented_map_without_pose_marker(
    monkeypatch,
) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    skill = OdomUnitreeSkill([_pose(x=5.0, y=5.0)])
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    result = SkillResult.ok("static SLAM snapshot")

    skill._attach_2d_slam_map(
        "go2_slam_map_snapshot",
        result,
        include_pose=False,
    )

    slam_map = result.metadata["2d_slam_map"]
    assert slam_map["status"] == "static"
    assert slam_map["source"] == "native_global_map_area_segmented"
    assert slam_map["pose_overlay"]["status"] == "unavailable"
    assert "pose" not in slam_map
    image = _image_from_block(result.content_blocks[0])
    assert all(
        image.getpixel((x, y)) != (230, 0, 0)
        for y in range(image.height)
        for x in range(image.width)
    )


def test_fresh_map_snapshot_has_world_pose_without_relocalization(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    skill = OdomUnitreeSkill([PoseStamped(frame_id="sport_mode_odom", position=[2, 3, 0])],
                            lidar_poses=[_pose(x=5.0, y=5.0)])
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()

    class NoRelocalizationTf:
        def get(self, source: str, target: str) -> None:
            assert (source, target) == ("world", "map")
            return None

    skill._tf = NoRelocalizationTf()
    result = skill.go2_slam_map_snapshot()
    slam_map = result.metadata["2d_slam_map"]
    assert result.metadata["relocalization_status"]["localized"] is False
    assert slam_map["status"] == "current"
    assert slam_map["pose"]["frame_id"] == slam_map["map_frame"] == "world"
    assert slam_map["pose"]["position_m"]["x"] == 5.0


def test_map_attachment_rejects_sport_pose_without_matching_map_pose() -> None:
    skill = OdomUnitreeSkill([PoseStamped(frame_id="sport_mode_odom", position=[2, 3, 0])],
                            lidar_poses=[])
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    result = SkillResult.ok("map")
    skill._attach_2d_slam_map("go2_slam_map_snapshot", result)
    assert result.metadata["2d_slam_map"]["status"] == "pose_unavailable"


def test_default_floor_plan_boundary_wall_is_disabled() -> None:
    assert DEFAULT_FLOOR_PLAN_BOUNDARY_WALL_WIDTH_M == 0.05


def test_drive_attaches_latest_camera_image_after_global_costmap(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_camera_image = _camera_image()
    skill._latest_camera_image_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    skill._2d_slam_map_view_size_m = 8.0

    result = skill.drive()

    assert result.metadata["floor_plan"]["status"] == "current"
    assert result.metadata["2d_slam_map"]["status"] == "current"
    assert result.metadata["camera_image"]["status"] == "current"
    assert result.metadata["camera_image"]["source"] == "color_image"
    assert result.metadata["camera_image"]["frame_id"] == "camera_optical"
    assert len(result.content_blocks) == 4
    assert result.content_blocks[0]["mimeType"] == "image/png"
    assert result.content_blocks[1]["mimeType"] == "image/png"
    assert result.content_blocks[2]["mimeType"] == "image/png"
    assert result.content_blocks[3]["type"] == "image_url"
    assert result.content_blocks[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_drive_exports_per_task_rerun_recording(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: tmp_path)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_sport_odom = poses[-1]
    skill._2d_slam_map_size_px = 96
    skill._2d_slam_map_view_size_m = 8.0
    captured: dict[str, Any] = {}

    def fake_write(self: UnitreeSkillContainer, rrd_path, **kwargs: Any) -> dict[str, Any]:
        del self
        captured["path"] = str(rrd_path)
        captured["kwargs"] = kwargs
        return {
            "global_costmap": kwargs["global_costmap"] is not None,
            "global_map": kwargs["global_map"] is not None,
            "lidar": kwargs["lidar"] is not None,
            "sport_odom": kwargs["sport_odom"] is not None,
            "robot_pose": kwargs["pose"] is not None,
            "trace_points": len(kwargs["trace"]),
        }

    monkeypatch.setattr(UnitreeSkillContainer, "_write_task_rerun_recording", fake_write)

    result = skill.drive(
        _mcp_context={
            "task": {
                "task_id": "task_000123",
                "task_goal": "inspect doorway",
                "task_origin": "goal_command",
            }
        }
    )

    recording = result.metadata["rerun_recording"]
    assert recording["status"] == "current"
    assert recording["task_id"] == "task_000123"
    assert recording["task_goal"] == "inspect doorway"
    assert recording["path"] == str(tmp_path / "rerun" / "task_000123.rrd")
    assert recording["global_costmap"] is True
    assert recording["sport_odom"] is True
    assert recording["robot_pose"] is True
    assert recording["trace_points"] == 1
    assert captured["kwargs"]["task_id"] == "task_000123"


def test_rerun_recording_trace_resets_on_new_task(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: tmp_path)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    skill._2d_slam_map_view_size_m = 8.0
    trace_lengths: list[int] = []

    def fake_write(self: UnitreeSkillContainer, rrd_path, **kwargs: Any) -> dict[str, Any]:
        del self, rrd_path
        trace_lengths.append(len(kwargs["trace"]))
        return {
            "global_costmap": True,
            "global_map": False,
            "lidar": False,
            "sport_odom": False,
            "robot_pose": True,
            "trace_points": len(kwargs["trace"]),
        }

    monkeypatch.setattr(UnitreeSkillContainer, "_write_task_rerun_recording", fake_write)

    skill.drive(_mcp_context={"task": {"task_id": "task_000001"}})
    skill.drive(_mcp_context={"task": {"task_id": "task_000001"}})
    skill.drive(_mcp_context={"task": {"task_id": "task_000002"}})

    assert trace_lengths == [1, 2, 1]


def test_stale_camera_image_reports_metadata_without_image(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_camera_image = _camera_image()
    skill._latest_camera_image_received_at_s = unitree_skill_module.time.monotonic() - 10.0
    skill._camera_image_max_age_s = 1.0

    result = skill.drive()

    assert result.success is True
    assert result.metadata["2d_slam_map"]["status"] == "unavailable"
    assert result.metadata["camera_image"]["status"] == "stale"
    assert result.content_blocks == []


def test_rotate_attaches_map_but_emergency_stop_stays_lightweight(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = RecordingTeleCmdVel(poses, yaw_step_deg=0.0)
    skill._navigation = DirectNavigation()
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._2d_slam_map_size_px = 96
    skill._2d_slam_map_view_size_m = 8.0

    rotate_result = skill.rotate_in_place(degrees=0.0)
    stop_result = skill.stop_movement()

    assert rotate_result.metadata["floor_plan"]["status"] == "current"
    assert rotate_result.metadata["area_segmentation"]["status"] == "current"
    assert rotate_result.metadata["2d_slam_map"]["status"] == "current"
    assert rotate_result.content_blocks[0]["mimeType"] == "image/png"
    assert stop_result.metadata["floor_plan"]["status"] == "current"
    assert stop_result.metadata["area_segmentation"]["status"] == "current"
    assert stop_result.metadata["2d_slam_map"]["status"] == "current"
    assert "camera_image" not in stop_result.metadata
    assert len(stop_result.content_blocks) == 3


def test_stale_global_costmap_reports_metadata_without_image(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    poses = [_pose(x=5.0, y=5.0)]
    skill = OdomUnitreeSkill(poses)
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic() - 10.0
    skill._2d_slam_map_max_age_s = 1.0

    result = skill.drive()

    assert result.success is True
    assert result.metadata["floor_plan"]["status"] == "stale"
    assert result.metadata["2d_slam_map"]["status"] == "stale"
    assert result.content_blocks == []


def test_final_floor_plan_saves_on_shutdown_hook(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    skill = DirectUnitreeSkill()
    skill._latest_global_costmap = _costmap()
    skill._floor_plan_size_px = 96
    skill._floor_plan_occupied_threshold = 50
    skill._floor_plan_closing_radius_m = 0.15
    skill._floor_plan_opening_radius_m = 0.0
    skill._floor_plan_min_component_area_m2 = 0.35

    skill._save_final_floor_plan()

    png_path = tmp_path / "slam" / "floor_plan.png"
    metadata_path = tmp_path / "slam" / "floor_plan.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert png_path.exists()
    assert metadata["status"] == "finalized"
    assert metadata["source"] == "native_global_map_cleaned"


def test_relocalization_bundle_saves_final_global_map(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    skill = DirectUnitreeSkill()
    skill._latest_global_map = _global_map()
    skill._latest_global_map_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_global_costmap = _costmap()
    skill._relocalization_map_max_age_s = 30.0

    skill._save_relocalization_bundle()

    bundle_dir = tmp_path / "slam" / "relocalization"
    map_path = bundle_dir / "global_map.pc2.lcm"
    costmap_path = bundle_dir / "global_costmap.occ.lcm"
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    decoded = PointCloud2.lcm_decode(map_path.read_bytes())
    assert map_path.exists()
    assert costmap_path.exists()
    assert len(decoded) == 4
    assert decoded.frame_id == "world"
    assert manifest["status"] == "current"
    assert manifest["global_map_file"] == str(map_path)
    assert manifest["point_count"] == 4
    assert manifest["costmap_saved"] is True
    assert "live point-cloud matching still controls acceptance" in manifest["backend_note"]


def test_prepare_stop_synchronously_saves_relocalization_bundle(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    skill = DirectUnitreeSkill()
    skill._latest_global_map = _global_map()
    skill._latest_global_map_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_global_costmap = _costmap()
    skill._relocalization_map_max_age_s = 30.0

    metadata = skill.prepare_stop()

    map_path = tmp_path / "slam" / "relocalization" / "global_map.pc2.lcm"
    assert map_path.exists()
    assert len(PointCloud2.lcm_decode(map_path.read_bytes())) == 4
    assert metadata["status"] == "current"
    assert metadata["point_count"] == 4


def test_go2_save_relocalization_bundle_skill_returns_sanitized_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    skill = DirectUnitreeSkill()
    skill._latest_global_map = _global_map()
    skill._latest_global_map_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_global_costmap = _costmap()
    skill._relocalization_map_max_age_s = 30.0

    result = skill.go2_save_relocalization_bundle()

    bundle = result.metadata["relocalization_bundle"]
    assert result.success is True
    assert bundle["status"] == "current"
    assert bundle["map_available"] is True
    assert bundle["global_map_file"].endswith("global_map.pc2.lcm")
    assert bundle["point_count"] == 4
    assert "pose" not in json.dumps(bundle)
    assert (tmp_path / "slam" / "relocalization" / "global_map.pc2.lcm").exists()


def test_go2_start_navigation_episode_resets_cached_slam_state(
    monkeypatch,
    tmp_path,
) -> None:
    reset_path = tmp_path / "voxel_map_reset_signal.json"
    monkeypatch.setattr(
        unitree_skill_module,
        "write_voxel_map_reset_signal",
        lambda **_: reset_path,
    )
    skill = DirectUnitreeSkill()
    skill._latest_global_map = _global_map()
    skill._latest_global_map_received_at_s = unitree_skill_module.time.monotonic()
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    skill._slam_pose_trace = [_pose()]
    skill._active_navigation_task_id = "old-task"
    skill._task_rerun_export_count = 3

    result = skill.go2_start_navigation_episode(
        clear_slam_map=True,
        reason="test episode",
    )

    assert result.success is True
    episode = result.metadata["navigation_episode"]
    assert episode["status"] == "started"
    assert episode["clear_slam_map"] is True
    assert episode["reset_signal_path"] == str(reset_path)
    assert skill._latest_global_map is None
    assert skill._latest_global_map_received_at_s is None
    assert skill._latest_global_costmap is None
    assert skill._latest_global_costmap_received_at_s is None
    assert skill._slam_pose_trace == []
    assert skill._active_navigation_task_id is None
    assert skill._task_rerun_export_count == 0


def test_report_relocalization_prior_waits_for_changed_transform(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    old_transform = Transform(
        translation=Vector3(1.0, 2.0, 0.0),
        frame_id="world",
        child_frame_id="map",
    )
    new_transform = Transform(
        translation=Vector3(3.0, 4.0, 0.0),
        frame_id="world",
        child_frame_id="map",
    )

    class RelocalizationTf:
        transform = old_transform

        def get(self, source: str, target: str) -> Transform:
            assert (source, target) == ("world", "map")
            return self.transform

    tf = RelocalizationTf()
    skill = DirectUnitreeSkill()
    skill._tf = tf
    skill.relocalization_request = SimpleNamespace(
        publish=lambda _message: setattr(tf, "transform", new_transform)
    )
    monkeypatch.setattr(skill, "_safe_current_relocalization_pose", lambda: None)
    attached: dict[str, Any] = {}
    monkeypatch.setattr(
        skill,
        "_attach_2d_slam_map",
        lambda _name, _result, **kwargs: attached.update(kwargs),
    )

    result = skill.go2_report_relocalization_prior(
        "A1",
        {"x": 1.0, "y": 2.0, "yaw_rad": 0.3, "sequence": 4},
    )

    assert result.success is True
    assert result.metadata["relocalization_wait"]["status"] == "localized"
    assert result.metadata["relocalization_wait"]["localized"] is True
    assert result.metadata["relocalization_wait"]["wait_window_s"] == 10.0
    assert attached["include_pose"] is True


def test_report_relocalization_prior_reports_matching_for_unchanged_transform(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    monkeypatch.setattr(unitree_skill_module, "GO2_RELOCALIZATION_REPORT_WAIT_S", 0.0)
    transform = Transform(
        translation=Vector3(1.0, 2.0, 0.0),
        frame_id="world",
        child_frame_id="map",
    )

    class StaticTf:
        def get(self, source: str, target: str) -> Transform:
            assert (source, target) == ("world", "map")
            return transform

    skill = DirectUnitreeSkill()
    skill._tf = StaticTf()
    skill.relocalization_request = SimpleNamespace(publish=lambda _message: None)
    monkeypatch.setattr(skill, "_safe_current_relocalization_pose", lambda: None)
    attached: dict[str, Any] = {}
    monkeypatch.setattr(
        skill,
        "_attach_2d_slam_map",
        lambda _name, _result, **kwargs: attached.update(kwargs),
    )

    result = skill.go2_report_relocalization_prior(
        "A1",
        {"x": 1.0, "y": 2.0, "yaw_rad": 0.3, "sequence": 4},
    )

    assert result.success is True
    assert result.metadata["relocalization_wait"]["status"] == "matching"
    assert result.metadata["relocalization_wait"]["localized"] is False
    assert attached["include_pose"] is False


def test_memory_node_navigation_records_surround_motion_frames(
    monkeypatch,
    tmp_path,
) -> None:
    skill = DirectUnitreeSkill()
    skill._memory_motion_rec_hz = 15.0
    skill._memory_motion_rec_root = tmp_path / "motion"
    skill._memory_motion_recorder = None
    skill._tf = FakeTf([_pose()])
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    goals: list[PoseStamped] = []

    class RecordingNavigation(DirectNavigation):
        def set_goal(self, goal: PoseStamped) -> bool:
            goals.append(goal)
            for direction in ("front", "right", "back", "left"):
                skill._on_surround_image(direction, _camera_image())
            return True

        def is_goal_reached(self) -> bool:
            return True

    skill._navigation = RecordingNavigation(state=NavigationState.IDLE)
    monkeypatch.setattr(
        skill,
        "_trace_movement_skill_result",
        lambda _tool_name, result, **_kwargs: result,
    )

    result = skill.go2_navigate_to_memory_node(
        "hall",
        target_slam_pose={"x": 1.0, "y": 2.0, "yaw_rad": 0.3, "frame_id": "world"},
        map_frame="world",
    )

    assert result.success is True
    assert goals[0].frame_id == "world"
    assert skill._memory_motion_recorder is None
    hop_dirs = list((tmp_path / "motion").glob("hop_*"))
    assert len(hop_dirs) == 1
    hop_dir = hop_dirs[0]
    assert sum(len(list((hop_dir / direction).glob("*.jpg"))) for direction in (
        "front",
        "right",
        "back",
        "left",
    )) == 4
    metadata = json.loads((hop_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["tool_name"] == "go2_navigate_to_memory_node"
    assert metadata["node_id"] == "hall"
    assert metadata["status"] == "ARRIVED"
    assert metadata["record_hz"] == 15.0


def test_memory_navigation_rejects_target_in_different_frame(monkeypatch) -> None:
    skill = DirectUnitreeSkill()
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()
    monkeypatch.setattr(
        skill, "_trace_movement_skill_result", lambda _name, result, **_kwargs: result
    )
    result = skill.go2_navigate_to_memory_node(
        "hall",
        target_slam_pose={"x": 1.0, "y": 2.0, "yaw_rad": 0.0, "frame_id": "sport_mode_odom"},
        map_frame="world",
    )
    assert result.success is False
    assert result.error_code == "FRAME_MISMATCH"


def test_relocalization_bundle_writes_unavailable_manifest_without_global_map(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: artifact_dir)
    skill = DirectUnitreeSkill()

    skill._save_relocalization_bundle()

    bundle_dir = tmp_path / "slam" / "relocalization"
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "unavailable"
    assert manifest["reason"] == "global_map_unavailable"
    assert not (bundle_dir / "global_map.pc2.lcm").exists()


def test_sport_posture_command_does_not_attach_global_costmap(monkeypatch) -> None:
    monkeypatch.setattr(unitree_skill_module, "resolve_artifact_dir", lambda **_: None)
    skill = DirectUnitreeSkill()
    skill._connection = StubGO2Connection()
    skill._navigation = DirectNavigation()
    skill._latest_global_costmap = _costmap()
    skill._latest_global_costmap_received_at_s = unitree_skill_module.time.monotonic()

    result = skill.stand_up()

    assert result.success is True
    assert "floor_plan" not in result.metadata
    assert "2d_slam_map" not in result.metadata
    assert result.content_blocks == []


def test_drive_pure_yaw_delegates_to_rotate_in_place() -> None:
    class DriveSkill(DirectUnitreeSkill):
        def __init__(self) -> None:
            self.rotate_calls: list[float] = []

        def rotate_in_place(
            self,
            degrees: float,
            max_duration_s: float = 5.0,
            angular_speed_limit: float = 0.4,
        ) -> SkillResult:
            del max_duration_s, angular_speed_limit
            self.rotate_calls.append(degrees)
            return SkillResult.ok("delegated", delegated_to="rotate_in_place")

    skill = DriveSkill()

    result = skill.drive(degrees=30.0)

    assert result.success is True
    assert skill.rotate_calls == [30.0]
    assert result.metadata["delegated_to"] == "rotate_in_place"


def test_drive_timeout_returns_structured_result_and_stops() -> None:
    poses = [_pose()]
    tele_cmd_vel = RecordingDriveTeleCmdVel(poses, dt_s=0.0)
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.001
    skill._drive_timeout_s = 0.01

    result = skill.drive(forward=0.5)

    assert result.success is False
    assert result.error_code == "TIMEOUT"
    assert result.duration_ms < 1000.0
    assert result.metadata["status"] == "TIMEOUT"
    assert result.metadata["zero_velocity_sent"] is True
    assert result.metadata["measured_delta"]["linear_m"] == 0.0
    assert tele_cmd_vel.commands[-1].linear.x == 0.0
    assert tele_cmd_vel.commands[-1].angular.z == 0.0


def test_drive_keeps_missing_sport_odom_as_no_odom() -> None:
    result = OdomUnitreeSkill([]).drive(forward=0.2)

    assert result.success is False
    assert result.error_code == "NO_ODOM"
    assert result.metadata["status"] == "NO_ODOM"
    assert result.metadata["feedback_watchdog_triggered"] is False


def test_drive_allows_delayed_first_odom_progress_within_startup_grace(monkeypatch) -> None:
    poses = [_pose()]
    clock = FakeClock()
    tele_cmd_vel = DelayedOdomDriveTeleCmdVel(
        poses,
        clock,
        startup_delay_s=0.9,
        progress_steps_m=[0.1, 0.1, 0.1, 0.1],
    )
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.1
    skill._drive_timeout_s = 3.0
    skill._direct_motion_initial_feedback_timeout_s = 1.5
    skill._direct_motion_feedback_timeout_s = 0.5
    monkeypatch.setattr(unitree_skill_module, "time", clock)

    result = skill.drive(forward=0.5)

    assert result.success is True
    assert result.metadata["status"] == "ARRIVED"
    assert result.metadata["feedback_watchdog_triggered"] is False
    assert clock.now >= 0.9


def test_drive_stops_when_startup_grace_has_no_odom_progress(monkeypatch) -> None:
    sport_poses = [_pose()]
    lidar_poses = [_pose()]
    clock = FakeClock()
    tele_cmd_vel = DelayedOdomDriveTeleCmdVel(
        lidar_poses,
        clock,
        startup_delay_s=0.0,
        progress_steps_m=[0.1] * 10,
    )
    skill = OdomUnitreeSkill(sport_poses, lidar_poses=lidar_poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.05
    skill._drive_timeout_s = 2.0
    skill._direct_motion_initial_feedback_timeout_s = 0.4
    skill._direct_motion_feedback_timeout_s = 0.1
    monkeypatch.setattr(unitree_skill_module, "time", clock)

    result = skill.drive(forward=0.5)

    assert result.success is False
    assert result.error_code == "NO_PROGRESS"
    assert result.metadata["status"] == "NO_PROGRESS"
    assert result.metadata["feedback_watchdog_triggered"] is True
    assert "fresh body odometry" in result.message
    assert result.metadata["zero_velocity_sent"] is True
    assert 0.4 <= clock.now <= 0.45
    assert lidar_poses[-1].x > 0.0
    assert tele_cmd_vel.commands[-1].linear.x == 0.0


def test_drive_does_not_count_sport_mode_body_height_as_progress(monkeypatch) -> None:
    sport_poses = [_pose(z=0.3)]
    clock = FakeClock()
    tele_cmd_vel = VerticalSportDriveTeleCmdVel(sport_poses)
    skill = OdomUnitreeSkill(sport_poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.05
    skill._drive_timeout_s = 2.0
    skill._direct_motion_initial_feedback_timeout_s = 0.2
    monkeypatch.setattr(unitree_skill_module, "time", clock)

    result = skill.drive(forward=0.5)

    assert result.error_code == "NO_PROGRESS"
    assert result.metadata["status"] == "NO_PROGRESS"
    assert result.metadata["feedback_watchdog_triggered"] is True
    assert result.metadata["measured_delta"]["linear_m"] == 0.0
    assert sport_poses[-1].z > sport_poses[0].z


def test_direct_motion_pose_rejects_stale_sport_mode_sample(monkeypatch) -> None:
    clock = FakeClock()
    skill = DirectUnitreeSkill()
    skill._direct_motion_feedback_timeout_s = 0.5
    monkeypatch.setattr(unitree_skill_module, "time", clock)

    pose = _pose()
    skill._on_sport_odom(pose)
    assert skill._current_direct_motion_pose() is pose

    clock.sleep(0.51)
    assert skill._current_direct_motion_pose() is None


def test_drive_stops_quickly_when_odom_stalls_after_first_progress(monkeypatch) -> None:
    poses = [_pose()]
    clock = FakeClock()
    tele_cmd_vel = DelayedOdomDriveTeleCmdVel(
        poses,
        clock,
        startup_delay_s=0.3,
        progress_steps_m=[0.02],
    )
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._drive_poll_s = 0.05
    skill._drive_timeout_s = 2.0
    skill._direct_motion_initial_feedback_timeout_s = 1.5
    skill._direct_motion_feedback_timeout_s = 0.2
    monkeypatch.setattr(unitree_skill_module, "time", clock)

    result = skill.drive(forward=0.5)

    assert result.success is False
    assert result.error_code == "NO_PROGRESS"
    assert result.metadata["status"] == "NO_PROGRESS"
    assert result.metadata["feedback_watchdog_triggered"] is True
    assert 0.5 <= clock.now < 1.0
    assert tele_cmd_vel.commands[-1].linear.x == 0.0


def test_drive_cmd_vel_publish_failure_returns_structured_error() -> None:
    poses = [_pose()]
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = RaisingTeleCmdVel()
    skill._drive_timeout_s = 1.0

    result = skill.drive(forward=0.5)

    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"
    assert result.metadata["status"] == "EXECUTION_FAILED"
    assert result.metadata["zero_velocity_sent"] is False
    response = result.metadata["connection_move_response"]
    assert response["error"] == "cmd_vel_publish_failed"
    assert response["exception_type"] == "RuntimeError"
    assert response["exception"] == "cmd_vel transport missing"
    assert result.metadata["zero_velocity_response"]["error"] == "cmd_vel_publish_failed"


def test_relative_move_bounded_get_state_timeout_returns_structured_result() -> None:
    poses = [_pose()]
    navigation = TimeoutStateNavigation()
    skill = DirectUnitreeSkill()
    skill._tf = FakeTf(poses)
    skill._navigation = navigation
    skill._relative_move_settle_s = 0.0
    skill._relative_move_timeout_s = 0.05

    result = skill.relative_move(forward=0.5)

    assert result.success is False
    assert result.error_code == "TIMEOUT"
    assert result.duration_ms < 1000.0
    assert result.metadata["status"] == "TIMEOUT"
    assert result.metadata["command_may_be_active"] is True
    assert result.metadata["measured_delta"] is not None
    assert result.metadata["navigation_state_error"].startswith("TimeoutError")
    assert navigation.cancel_calls == 1
    assert navigation.get_state.timeouts


def test_relative_move_returns_not_configured_when_navigation_is_disabled() -> None:
    skill = DirectUnitreeSkill()
    skill._navigation = DisabledModuleProxy("NavigationInterfaceSpec")

    result = skill.relative_move(forward=0.5)

    assert result.success is False
    assert result.error_code == "NOT_CONFIGURED"
    assert result.metadata["status"] == "NOT_CONFIGURED"
    assert result.metadata["navigation_configured"] is False
    assert result.metadata["command_may_be_active"] is False


def test_rotate_in_place_uses_tele_cmd_vel_stream_and_odom_success() -> None:
    sport_poses = [_pose_at_yaw(0.0)]
    lidar_poses = [_pose_at_yaw(0.0)]
    tele_cmd_vel = RecordingTeleCmdVel(sport_poses, yaw_step_deg=10.0)
    skill = OdomUnitreeSkill(sport_poses, lidar_poses=lidar_poses)
    skill.tele_cmd_vel = tele_cmd_vel
    navigation = BoundedCancelNavigation()
    skill._navigation = navigation
    skill._rotate_in_place_poll_s = 0.0

    result = skill.rotate_in_place(
        degrees=30.0,
        max_duration_s=5.0,
        angular_speed_limit=0.4,
    )

    assert result.success is True
    assert result.metadata["status"] == "ARRIVED"
    assert result.metadata["cancel_goal_sent"] is True
    assert result.metadata["zero_velocity_sent"] is True
    assert navigation.cancel_goal.timeouts == [1.0]
    assert tele_cmd_vel.commands[-1].linear.x == 0.0
    assert tele_cmd_vel.commands[-1].angular.z == 0.0
    assert tele_cmd_vel.commands[0].angular.z == 0.4
    assert math.isclose(result.metadata["measured_delta"]["yaw_deg"], 30.0, abs_tol=1e-6)
    assert lidar_poses[-1].yaw == 0.0


def test_rotate_in_place_cmd_vel_stream_reports_odom_timeout_and_stops() -> None:
    poses = [_pose_at_yaw(0.0)]
    tele_cmd_vel = RecordingTeleCmdVel(poses, yaw_step_deg=0.0)
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._navigation = DirectNavigation()
    skill._rotate_in_place_poll_s = 0.001

    result = skill.rotate_in_place(
        degrees=30.0,
        max_duration_s=0.02,
        angular_speed_limit=0.4,
    )

    assert result.success is False
    assert result.error_code == "TIMEOUT"
    assert result.duration_ms < 1000.0
    assert result.metadata["zero_velocity_sent"] is True
    assert tele_cmd_vel.commands[-1].linear.x == 0.0
    assert tele_cmd_vel.commands[-1].angular.z == 0.0


def test_rotate_in_place_stops_when_odometry_shows_no_progress() -> None:
    poses = [_pose_at_yaw(0.0)]
    tele_cmd_vel = RecordingTeleCmdVel(poses, yaw_step_deg=0.0)
    skill = OdomUnitreeSkill(poses)
    skill.tele_cmd_vel = tele_cmd_vel
    skill._navigation = DirectNavigation()
    skill._rotate_in_place_poll_s = 0.001
    skill._direct_motion_initial_feedback_timeout_s = 0.01
    skill._direct_motion_feedback_timeout_s = 0.01

    result = skill.rotate_in_place(
        degrees=90.0,
        max_duration_s=1.0,
        angular_speed_limit=0.4,
    )

    assert result.success is False
    assert result.error_code == "NO_PROGRESS"
    assert result.duration_ms < 500.0
    assert result.metadata["status"] == "NO_PROGRESS"
    assert result.metadata["feedback_watchdog_triggered"] is True
    assert "fresh body odometry" in result.message
    assert result.metadata["zero_velocity_sent"] is True
    assert tele_cmd_vel.commands[-1].angular.z == 0.0


def test_did_you_mean() -> None:
    suggestions = difflib.get_close_matches("Pounce", _UNITREE_COMMANDS.keys(), n=3, cutoff=0.6)
    assert "FrontPounce" in suggestions
    assert "Pose" in suggestions


def test_motion_safety_status_reports_low_progress_breaker_as_local_obstacle_stop() -> None:
    skill = DirectUnitreeSkill()
    skill._navigation = DirectNavigation(
        motion_diagnostics={
            "last_failure": "RETRY_LIMIT_EXCEEDED",
            "last_stop_reason": "obstacle_found",
            "last_stop_time": 123.0,
        }
    )

    status = skill.motion_safety_status()

    assert status.success is True
    diagnostics = status.metadata["diagnostics"]
    assert diagnostics["blocker"] == "local_obstacle_stop"
    assert diagnostics["recommended_next_action"] == (
        "stop, rotate/observe, then retry a shorter or lower-speed local move"
    )
    assert "blocker=local_obstacle_stop" in status.message


def test_motion_safety_status_respects_runtime_policy_blockers() -> None:
    skill = DirectUnitreeSkill()
    skill.config = SimpleNamespace(
        g=SimpleNamespace(
            agent_profile="supervised_live",
            agent_movement_mode="deny",
            agent_tool_scope="readonly",
        )
    )
    skill._navigation = DirectNavigation()

    status = skill.motion_safety_status()

    assert status.success is True
    diagnostics = status.metadata["diagnostics"]
    assert diagnostics["blocker"] == "runtime_policy"


def test_motion_safety_status_fails_closed_without_backend_diagnostics() -> None:
    skill = DirectUnitreeSkill()
    skill._navigation = NoDiagnosticsNavigation()

    status = skill.motion_safety_status()

    assert status.success is False
    assert status.error_code == "NOT_CONFIGURED"
    assert status.metadata["diagnostics_available"] is False


def test_last_motion_diagnostics_carries_measured_result_and_robot_status() -> None:
    measured_result = {
        "tool_name": "relative_move",
        "precision_class": "short_stop",
        "measured_delta": {"linear_m": 0.12, "yaw_deg": 4.0},
        "physically_satisfied": False,
    }
    skill = DirectUnitreeSkill()
    skill._navigation = DirectNavigation(
        motion_diagnostics={
            "measured_result": measured_result,
            "robot_status": {
                "battery": {
                    "state": "normal",
                    "soc_percent": 88.0,
                    "source": "unitree_lowstate",
                },
                "motors": {
                    "state": "warning",
                    "lost_count": 1,
                    "max_temperature_c": 51.0,
                    "source": "unitree_lowstate",
                },
            },
        }
    )

    status = skill.last_motion_diagnostics()

    assert status.success is True
    diagnostics = status.metadata["diagnostics"]
    assert diagnostics["measured_result"] == measured_result
    assert diagnostics["battery_status"]["soc_percent"] == 88.0
    assert diagnostics["motor_status"]["lost_count"] == 1


def test_go2_situation_returns_compact_agent_facing_summary() -> None:
    skill = DirectUnitreeSkill()
    skill._navigation = DirectNavigation(
        motion_diagnostics={
            "last_failure": "RETRY_LIMIT_EXCEEDED",
            "last_stop_reason": "obstacle_found",
            "measured_result": {
                "tool_name": "relative_move",
                "precision_class": "short_stop",
                "status": "TIMEOUT",
            },
            "robot_status": {
                "battery": {
                    "state": "normal",
                    "soc_percent": 91.0,
                    "source": "unitree_lowstate",
                },
                "motors": {
                    "state": "normal",
                    "lost_count": 0,
                    "max_temperature_c": 39.0,
                    "source": "unitree_lowstate",
                },
            },
        }
    )

    status = skill.go2_situation()

    assert status.success is True
    situation = status.metadata["situation"]
    assert situation["blocker"] == "local_obstacle_stop"
    assert situation["battery"]["soc_percent"] == 91.0
    assert situation["motor"]["lost_count"] == 0
    assert situation["last_motion"]["precision_class"] == "short_stop"
    assert "Go2 situation: blocker=local_obstacle_stop" in status.message


def test_robot_protection_status_summarizes_battery_motor_and_obstacle_states() -> None:
    skill = DirectUnitreeSkill()
    skill._navigation = DirectNavigation(
        motion_diagnostics={
            "last_stop_reason": "obstacle_found",
            "robot_status": {
                "battery": {"state": "critical", "soc_percent": 8.0},
                "motors": {"state": "warning", "lost_count": 1},
            },
        }
    )

    status = skill.robot_protection_status()

    assert status.success is True
    assert "Robot protection status: blocker=battery_safety" in status.message
    assert "local_obstacle=unknown" in status.message
    assert "battery=critical" in status.message
    assert "motor=warning" in status.message
