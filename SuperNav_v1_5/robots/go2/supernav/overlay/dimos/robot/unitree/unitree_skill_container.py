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

from __future__ import annotations

import base64
import datetime
import difflib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Literal
import uuid

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from reactivex.disposable import Disposable
from unitree_webrtc_connect.constants import RTC_TOPIC

from dimos.agents.annotation import current_skill_context, skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.harness.artifact_ledger import ArtifactLedger, resolve_artifact_dir
from dimos.agents.harness.skill_timing import record_skill_duration
from dimos.agents.harness.skill_trace import emit_skill_result_trace
from dimos.agents.runtime_policy import resolve_agent_runtime_policy
from dimos.agents.skill_result import SkillResult
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.mapping.occupancy.agent_map import render_agent_occupancy_map
from dimos.mapping.occupancy.floor_plan import FloorPlanConfig, render_clean_floor_plan
from dimos.mapping.occupancy.room_segmentation import (
    RoomSegmentationConfig,
    render_room_segmentation,
)
from dimos.mapping.occupancy.habitat_area_segmentation import build_slam_area_nodes
from dimos.mapping.voxels import write_voxel_map_reset_signal
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.base import NavigationState
from dimos.navigation.motion_diagnostics import build_motion_diagnostics
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.navigation.skill_errors import NavigationError, NavigationSkillError
from dimos.navigation.skill_metadata import odom_delta_metadata, pose_metadata
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.robot.unitree.go2.motion_recorder import (
    HopMotionRecorder,
    MOTION_REC_DIR_ENV,
    MOTION_REC_HZ_ENV,
)
from dimos.robot.unitree.go2.surround_camera import CAMERA_DIRECTIONS
from dimos.utils.logging_config import setup_logger
from dimos.utils.trigonometry import angle_diff

logger = setup_logger()

DEFAULT_ROTATE_IN_PLACE_MIN_EFFECTIVE_YAW_SPEED_RAD_S = 0.3
DEFAULT_DRIVE_TIMEOUT_S = 12.0
DRIVE_TIMEOUT_ENV = "DIMOS_GO2_DRIVE_TIMEOUT_S"
DEFAULT_RELATIVE_MOVE_TIMEOUT_S = 100.0
RELATIVE_MOVE_TIMEOUT_ENV = "DIMOS_GO2_RELATIVE_MOVE_TIMEOUT_S"
DEFAULT_2D_SLAM_MAP_MAX_AGE_S = 3.0
DEFAULT_2D_SLAM_MAP_SIZE_PX = 768
DEFAULT_2D_SLAM_TRACE_MAX_POINTS = 200
DEFAULT_2D_SLAM_MAP_VIEW_SIZE_M = 12.0
DEFAULT_FLOOR_PLAN_MAX_AGE_S = 5.0
DEFAULT_FLOOR_PLAN_SIZE_PX = 768
DEFAULT_FLOOR_PLAN_MIN_COMPONENT_AREA_M2 = 0.35
DEFAULT_FLOOR_PLAN_CLOSING_RADIUS_M = 0.05
DEFAULT_FLOOR_PLAN_OPENING_RADIUS_M = 0.0
DEFAULT_FLOOR_PLAN_OCCUPIED_THRESHOLD = 50
DEFAULT_FLOOR_PLAN_FREE_SPACE_CLOSING_RADIUS_M = 0.25
DEFAULT_FLOOR_PLAN_MAX_FILL_HOLE_AREA_M2 = 2.5
DEFAULT_FLOOR_PLAN_BOUNDARY_WALL_WIDTH_M = 0.05
DEFAULT_FLOOR_PLAN_STRUCTURAL_WALL_MIN_LENGTH_M = 4.0
DEFAULT_FLOOR_PLAN_STRUCTURAL_WALL_WIDTH_M = 0.05
DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M = 0.06
DEFAULT_CAMERA_IMAGE_MAX_AGE_S = 3.0
DEFAULT_TASK_RERUN_EXPORT_ENABLED = True
DEFAULT_TASK_RERUN_POINT_VOXEL_SIZE_M = 0.05
DEFAULT_RELOCALIZATION_MAP_MAX_AGE_S = 30.0
_MOVEMENT_IMAGE_ATTACHMENT_TOOLS = {
    "drive",
    "go2_navigate_to_memory_node",
    "relative_move",
    "rotate_in_place",
    "stop_movement",
}
_SLAM_MAP_ATTACHMENT_TOOLS = {
    *_MOVEMENT_IMAGE_ATTACHMENT_TOOLS,
    "go2_report_relocalization_prior",
    "go2_slam_map_snapshot",
}
GO2_RELOCALIZATION_POLL_INTERVAL_S = 0.1
GO2_RELOCALIZATION_REPORT_WAIT_S = 10.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _safe_task_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return token[:80] or "task"


def _task_rerun_summary(task_id: str, task_goal: str | None) -> str:
    lines = ["# DimOS Go2 Navigation Task", f"- task_id: {task_id}"]
    if task_goal:
        lines.append(f"- goal: {task_goal}")
    lines.append("- recording: offline per-task Rerun export")
    return "\n".join(lines)


UNITREE_WEBRTC_CONTROLS: list[tuple[str, int, str]] = [
    ("Damp", 1001, "Lowers the robot to the ground fully."),
    (
        "BalanceStand",
        1002,
        "Activates a mode that maintains the robot in a balanced standing position.",
    ),
    (
        "StandUp",
        1004,
        "Commands the robot to transition from a sitting or prone position to a standing posture.",
    ),
    (
        "StandDown",
        1005,
        "Instructs the robot to move from a standing position to a sitting or prone posture.",
    ),
    (
        "RecoveryStand",
        1006,
        "Recovers the robot to a state from which it can take more commands. Useful to run after multiple dynamic commands like front flips, Must run after skills like sit and jump and standup.",
    ),
    ("Sit", 1009, "Commands the robot to sit down from a standing or moving stance."),
    (
        "RiseSit",
        1010,
        "Commands the robot to rise back to a standing position from a sitting posture.",
    ),
    (
        "SwitchGait",
        1011,
        "Switches the robot's walking pattern or style dynamically, suitable for different terrains or speeds.",
    ),
    ("Trigger", 1012, "Triggers a specific action or custom routine programmed into the robot."),
    (
        "BodyHeight",
        1013,
        "Adjusts the height of the robot's body from the ground, useful for navigating various obstacles.",
    ),
    (
        "FootRaiseHeight",
        1014,
        "Controls how high the robot lifts its feet during movement, which can be adjusted for different surfaces.",
    ),
    (
        "SpeedLevel",
        1015,
        "Sets or adjusts the speed at which the robot moves, with various levels available for different operational needs.",
    ),
    (
        "Hello",
        1016,
        "Performs a greeting action, which could involve a wave or other friendly gesture.",
    ),
    ("Stretch", 1017, "Engages the robot in a stretching routine."),
    (
        "TrajectoryFollow",
        1018,
        "Directs the robot to follow a predefined trajectory, which could involve complex paths or maneuvers.",
    ),
    (
        "ContinuousGait",
        1019,
        "Enables a mode for continuous walking or running, ideal for long-distance travel.",
    ),
    ("Content", 1020, "To display or trigger when the robot is happy."),
    ("Wallow", 1021, "The robot falls onto its back and rolls around."),
    (
        "Dance1",
        1022,
        "Performs a predefined dance routine 1, programmed for entertainment or demonstration.",
    ),
    ("Dance2", 1023, "Performs another variant of a predefined dance routine 2."),
    ("GetBodyHeight", 1024, "Retrieves the current height of the robot's body from the ground."),
    (
        "GetFootRaiseHeight",
        1025,
        "Retrieves the current height at which the robot's feet are being raised during movement.",
    ),
    (
        "GetSpeedLevel",
        1026,
        "Retrieves the current speed level setting of the robot.",
    ),
    (
        "SwitchJoystick",
        1027,
        "Switches the robot's control mode to respond to joystick input for manual operation.",
    ),
    (
        "Pose",
        1028,
        "Commands the robot to assume a specific pose or posture as predefined in its programming.",
    ),
    ("Scrape", 1029, "The robot performs a scraping motion."),
    (
        "FrontFlip",
        1030,
        "Commands the robot to perform a front flip, showcasing its agility and dynamic movement capabilities.",
    ),
    (
        "FrontJump",
        1031,
        "Instructs the robot to jump forward, demonstrating its explosive movement capabilities.",
    ),
    (
        "FrontPounce",
        1032,
        "Commands the robot to perform a pouncing motion forward.",
    ),
    (
        "WiggleHips",
        1033,
        "The robot performs a hip wiggling motion, often used for entertainment or demonstration purposes.",
    ),
    (
        "GetState",
        1034,
        "Retrieves the current operational state of the robot, including its mode, position, and status.",
    ),
    (
        "EconomicGait",
        1035,
        "Engages a more energy-efficient walking or running mode to conserve battery life.",
    ),
    ("FingerHeart", 1036, "Performs a finger heart gesture while on its hind legs."),
    (
        "Handstand",
        1301,
        "Commands the robot to perform a handstand, demonstrating balance and control.",
    ),
    (
        "CrossStep",
        1302,
        "Commands the robot to perform cross-step movements.",
    ),
    (
        "OnesidedStep",
        1303,
        "Commands the robot to perform one-sided step movements.",
    ),
    ("Bound", 1304, "Commands the robot to perform bounding movements."),
    ("MoonWalk", 1305, "Commands the robot to perform a moonwalk motion."),
    ("LeftFlip", 1042, "Executes a flip towards the left side."),
    ("RightFlip", 1043, "Performs a flip towards the right side."),
    ("Backflip", 1044, "Executes a backflip, a complex and dynamic maneuver."),
]


_UNITREE_COMMANDS = {
    name: (id_, description)
    for name, id_, description in UNITREE_WEBRTC_CONTROLS
    if name not in ["Reverse", "Spin"]
}


def _unitree_status_code(response: Any) -> int | None:
    """Extract Unitree ``status.code`` from WebRTC responses when present."""

    if not isinstance(response, dict):
        return None

    status = response.get("status")
    if isinstance(status, dict) and "code" in status:
        return _int_or_none(status.get("code"))

    for key in ("code", "status_code", "statusCode"):
        if key in response:
            return _int_or_none(response.get(key))
    return None


def _unitree_publish_request_timed_out(response: Any) -> bool:
    return isinstance(response, dict) and response.get("error") == "publish_request_timeout"


def _unitree_publish_request_failed(response: Any) -> bool:
    return isinstance(response, dict) and response.get("ok") is False


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _diagnostic_state(value: Any, *, default: str) -> str:
    if isinstance(value, dict):
        return str(value.get("state") or default)
    if value is None:
        return default
    return str(value)


def _compact_state_payload(value: Any, *, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"state": str(value or "unknown")}
    return {key: value.get(key) for key in keys if key in value}


def _compact_authority_payload(runtime_policy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime_policy, dict):
        return {"available": False}
    return {
        "available": True,
        "profile": runtime_policy.get("profile"),
        "movement_mode": runtime_policy.get("movement_mode"),
        "tool_scope": runtime_policy.get("tool_scope"),
        "admin_override": runtime_policy.get("tool_policy_admin_override"),
    }


def _relocalization_bundle_agent_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return agent/proxy-facing bundle metadata without raw pose coordinates."""

    global_map_file = manifest.get("global_map_file") or manifest.get("global_map_path")
    return {
        "status": manifest.get("status"),
        "reason": manifest.get("reason"),
        "bundle_path": manifest.get("bundle_path"),
        "manifest_path": (
            str(Path(str(manifest.get("bundle_path"))) / "manifest.json")
            if manifest.get("bundle_path")
            else None
        ),
        "global_map_file": global_map_file,
        "map_available": bool(manifest.get("status") == "current" and global_map_file),
        "point_count": manifest.get("point_count"),
        "frame_id": manifest.get("frame_id"),
        "saved_at_epoch_s": manifest.get("saved_at_epoch_s"),
        "costmap_saved": manifest.get("costmap_saved"),
        "global_map_age_s": manifest.get("global_map_age_s"),
        "max_age_s": manifest.get("max_age_s"),
        "backend_note": manifest.get("backend_note"),
    }


class UnitreeSkillContainer(Module):
    """Container for Unitree Go2 robot skills using the new framework."""

    _navigation: NavigationInterfaceSpec
    _connection: GO2ConnectionSpec
    sport_odom: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    global_map: In[PointCloud2]
    lidar: In[PointCloud2]
    color_image: In[Image]
    surround_front_image: In[Image]
    surround_right_image: In[Image]
    surround_back_image: In[Image]
    surround_left_image: In[Image]
    tele_cmd_vel: Out[Twist]
    relocalization_request: Out[Bool]
    _latest_sport_odom: PoseStamped | None = None
    _latest_sport_odom_received_at_s: float | None = None
    _latest_global_costmap: OccupancyGrid | None = None
    _latest_global_costmap_received_at_s: float | None = None
    _latest_global_map: PointCloud2 | None = None
    _latest_global_map_received_at_s: float | None = None
    _latest_lidar: PointCloud2 | None = None
    _latest_lidar_received_at_s: float | None = None
    _latest_camera_image: Image | None = None
    _latest_camera_image_received_at_s: float | None = None
    _memory_motion_recorder: HopMotionRecorder | None = None
    _slam_pose_trace: list[PoseStamped]
    _active_navigation_task_id: str | None = None
    _task_rerun_export_count: int = 0
    _relative_move_timeout_s: float = _env_float(
        RELATIVE_MOVE_TIMEOUT_ENV,
        DEFAULT_RELATIVE_MOVE_TIMEOUT_S,
    )
    _relative_move_settle_s: float = 1.0
    _relative_move_poll_s: float = 0.1
    _relative_move_linear_tolerance_m: float = 0.25
    _relative_move_yaw_tolerance_deg: float = 15.0
    _pure_rotation_linear_epsilon_m: float = 1e-3
    _rotate_in_place_tolerance_deg: float = 5.0
    _rotate_in_place_poll_s: float = 0.05
    _rotate_in_place_min_effective_yaw_speed_rad_s: float = (
        DEFAULT_ROTATE_IN_PLACE_MIN_EFFECTIVE_YAW_SPEED_RAD_S
    )
    _drive_timeout_s: float = _env_float(DRIVE_TIMEOUT_ENV, DEFAULT_DRIVE_TIMEOUT_S)
    _drive_poll_s: float = 0.05
    _drive_linear_gain: float = 0.9
    _drive_yaw_gain: float = 1.2
    _drive_max_linear_speed_mps: float = 0.45
    _drive_max_yaw_speed_rad_s: float = 0.4
    _direct_motion_initial_feedback_timeout_s: float = 1.75
    _direct_motion_feedback_timeout_s: float = 0.5
    _drive_min_feedback_progress_m: float = 0.01
    _rotate_min_feedback_progress_deg: float = 2.0
    _2d_slam_map_max_age_s: float = _env_float(
        "DIMOS_GO2_2D_SLAM_MAP_MAX_AGE_S",
        DEFAULT_2D_SLAM_MAP_MAX_AGE_S,
    )
    _2d_slam_map_size_px: int = _env_int(
        "DIMOS_GO2_2D_SLAM_MAP_SIZE_PX",
        DEFAULT_2D_SLAM_MAP_SIZE_PX,
    )
    _2d_slam_map_view_size_m: float = _env_float(
        "DIMOS_GO2_2D_SLAM_MAP_VIEW_SIZE_M",
        DEFAULT_2D_SLAM_MAP_VIEW_SIZE_M,
    )
    _2d_slam_trace_max_points: int = _env_int(
        "DIMOS_GO2_2D_SLAM_TRACE_MAX_POINTS",
        DEFAULT_2D_SLAM_TRACE_MAX_POINTS,
    )
    _floor_plan_max_age_s: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_MAX_AGE_S",
        DEFAULT_FLOOR_PLAN_MAX_AGE_S,
    )
    _floor_plan_size_px: int = _env_int(
        "DIMOS_GO2_FLOOR_PLAN_SIZE_PX",
        DEFAULT_FLOOR_PLAN_SIZE_PX,
    )
    _floor_plan_min_component_area_m2: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_MIN_COMPONENT_AREA_M2",
        DEFAULT_FLOOR_PLAN_MIN_COMPONENT_AREA_M2,
    )
    _floor_plan_closing_radius_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_CLOSING_RADIUS_M",
        DEFAULT_FLOOR_PLAN_CLOSING_RADIUS_M,
    )
    _floor_plan_opening_radius_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_OPENING_RADIUS_M",
        DEFAULT_FLOOR_PLAN_OPENING_RADIUS_M,
    )
    _floor_plan_occupied_threshold: int = _env_int(
        "DIMOS_GO2_FLOOR_PLAN_OCCUPIED_THRESHOLD",
        DEFAULT_FLOOR_PLAN_OCCUPIED_THRESHOLD,
    )
    _floor_plan_free_space_closing_radius_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_FREE_SPACE_CLOSING_RADIUS_M",
        DEFAULT_FLOOR_PLAN_FREE_SPACE_CLOSING_RADIUS_M,
    )
    _floor_plan_max_fill_hole_area_m2: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_MAX_FILL_HOLE_AREA_M2",
        DEFAULT_FLOOR_PLAN_MAX_FILL_HOLE_AREA_M2,
    )
    _floor_plan_boundary_wall_width_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_BOUNDARY_WALL_WIDTH_M",
        DEFAULT_FLOOR_PLAN_BOUNDARY_WALL_WIDTH_M,
    )
    _floor_plan_structural_wall_min_length_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_STRUCTURAL_WALL_MIN_LENGTH_M",
        DEFAULT_FLOOR_PLAN_STRUCTURAL_WALL_MIN_LENGTH_M,
    )
    _floor_plan_structural_wall_width_m: float = _env_float(
        "DIMOS_GO2_FLOOR_PLAN_STRUCTURAL_WALL_WIDTH_M",
        DEFAULT_FLOOR_PLAN_STRUCTURAL_WALL_WIDTH_M,
    )
    _camera_image_max_age_s: float = _env_float(
        "DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S",
        DEFAULT_CAMERA_IMAGE_MAX_AGE_S,
    )
    _task_rerun_export_enabled: bool = _env_bool(
        "DIMOS_GO2_TASK_RERUN_EXPORT",
        DEFAULT_TASK_RERUN_EXPORT_ENABLED,
    )
    _task_rerun_point_voxel_size_m: float = _env_float(
        "DIMOS_GO2_TASK_RERUN_POINT_VOXEL_SIZE_M",
        DEFAULT_TASK_RERUN_POINT_VOXEL_SIZE_M,
    )
    _relocalization_map_max_age_s: float = _env_float(
        "DIMOS_GO2_RELOCALIZATION_MAP_MAX_AGE_S",
        DEFAULT_RELOCALIZATION_MAP_MAX_AGE_S,
    )
    _memory_motion_rec_hz: float = _env_float(MOTION_REC_HZ_ENV, 15.0)
    _memory_motion_rec_root: Path = Path(
        os.environ.get(MOTION_REC_DIR_ENV) or Path.home() / "nomad-live" / "motion"
    )

    @rpc
    def start(self) -> None:
        super().start()
        # Initialize TF early so it can start receiving transforms.
        _ = self.tf
        self.register_disposable(Disposable(self.sport_odom.subscribe(self._on_sport_odom)))
        self.register_disposable(
            Disposable(self.global_costmap.subscribe(self._on_global_costmap))
        )
        self.register_disposable(Disposable(self.global_map.subscribe(self._on_global_map)))
        self.register_disposable(Disposable(self.lidar.subscribe(self._on_lidar)))
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self._memory_motion_recorder = None
        for direction in CAMERA_DIRECTIONS:
            stream = getattr(self, f"surround_{direction}_image")
            if stream.transport is not None:
                self.register_disposable(
                    Disposable(
                        stream.subscribe(
                            lambda image, direction=direction: self._on_surround_image(
                                direction, image
                            )
                        )
                    )
                )

    @rpc
    def stop(self) -> None:
        # Save first: stop() is dispatched without waiting for its response and
        # worker teardown has a finite grace period. Recorder/floor-plan cleanup
        # can be slow enough to otherwise prevent the map write from running.
        self._save_relocalization_bundle()
        recorder = self._memory_motion_recorder
        self._finish_memory_motion_recorder(
            recorder,
            {
                "tool_name": "go2_navigate_to_memory_node",
                "status": "module_stopped",
                "direction": "memory_node",
                "record_hz": self._memory_motion_rec_hz,
            },
        )
        self._save_final_floor_plan()
        super().stop()

    @rpc
    def prepare_stop(self) -> dict[str, Any]:
        """Synchronously persist the latest relocalization map before teardown."""

        manifest = self._save_relocalization_bundle()
        if manifest.get("status") != "current":
            reason = manifest.get("reason") or "unknown"
            raise RuntimeError(f"Could not persist final relocalization map: {reason}")
        logger.info(
            "Prepared final Go2 relocalization bundle for shutdown",
            path=manifest.get("global_map_path"),
            point_count=manifest.get("point_count"),
            pointcloud_ts=manifest.get("pointcloud_ts"),
        )
        return _relocalization_bundle_agent_metadata(manifest)

    @skill
    def go2_save_relocalization_bundle(self) -> SkillResult[NavigationSkillError]:
        """Save the current Go2 global map as a relocalization input bundle."""

        manifest = self._save_relocalization_bundle()
        metadata = _relocalization_bundle_agent_metadata(manifest)
        if manifest.get("status") == "current":
            return SkillResult.ok(
                "Saved Go2 relocalization bundle.",
                relocalization_bundle=metadata,
            )
        return SkillResult.fail(
            "EXECUTION_FAILED",
            "Go2 relocalization bundle was not available to save.",
            relocalization_bundle=metadata,
        )

    @skill
    def go2_report_relocalization_prior(
        self,
        node_id: str,
        slam_pose: dict[str, Any],
        region_radius_m: float = 4.0,
    ) -> SkillResult[NavigationSkillError]:
        """Record a trusted graph-node SLAM prior for the next relocalization attempt."""

        clean_id = str(node_id or "").strip()
        if not clean_id:
            return SkillResult.fail("INVALID_TARGET", "A graph node id is required.")
        if not isinstance(slam_pose, dict):
            return SkillResult.fail("INVALID_TARGET", "A graph node SLAM pose is required.")
        try:
            x = float(slam_pose["x"])
            y = float(slam_pose["y"])
            yaw = float(slam_pose.get("yaw_rad", 0.0))
            radius = max(float(region_radius_m), 0.5)
        except (KeyError, TypeError, ValueError) as exc:
            return SkillResult.fail("INVALID_TARGET", f"Invalid graph-node SLAM pose: {exc}")

        artifact_dir = Path(resolve_artifact_dir(runtime_config=self._runtime_config()))
        run_root = artifact_dir.parent
        prior_dir = run_root / "slam" / "relocalization"
        prior_path = prior_dir / "reported_memory_node.json"
        payload = {
            "schema_version": 1,
            "source": "go2_report_graph_position",
            "status": "reported",
            "node_id": clean_id,
            "slam_pose": {**slam_pose, "x": x, "y": y, "yaw_rad": yaw},
            "region_radius_m": radius,
            "region_bbox": {
                "min_x": x - radius,
                "max_x": x + radius,
                "min_y": y - radius,
                "max_y": y + radius,
            },
            "reported_at_epoch_s": time.time(),
        }
        current_pose = self._safe_current_relocalization_pose()
        if current_pose is not None:
            payload["current_pose"] = pose_metadata(current_pose)
        prior_dir.mkdir(parents=True, exist_ok=True)
        prior_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        baseline_transform = self.tf.get("world", "map")
        started_at_s = time.monotonic()
        self.relocalization_request.publish(Bool(data=True))
        deadline_s = started_at_s + GO2_RELOCALIZATION_REPORT_WAIT_S
        transform = self.tf.get("world", "map")
        while (
            (transform is None or transform == baseline_transform)
            and time.monotonic() < deadline_s
        ):
            time.sleep(
                min(GO2_RELOCALIZATION_POLL_INTERVAL_S, max(deadline_s - time.monotonic(), 0.0))
            )
            transform = self.tf.get("world", "map")
        localized = transform is not None and transform != baseline_transform
        result = SkillResult.ok(
            "Recorded Go2 relocalization prior and requested an immediate attempt.",
            relocalization_prior={**payload, "path": str(prior_path)},
            relocalization_requested=True,
            relocalization_wait={
                "status": "localized" if localized else "matching",
                "localized": localized,
                "wait_window_s": GO2_RELOCALIZATION_REPORT_WAIT_S,
                "waited_s": time.monotonic() - started_at_s,
                "transform_epoch_s": float(transform.ts) if transform is not None else None,
            },
        )
        self._attach_2d_slam_map(
            "go2_report_relocalization_prior",
            result,
            include_pose=localized,
        )
        return result

    @skill
    def go2_slam_map_snapshot(self) -> SkillResult[NavigationSkillError]:
        """Return the latest SLAM map and current relocalization transform availability."""

        transform = self.tf.get("world", "map")
        result = SkillResult.ok(
            "Returned the latest Go2 SLAM map snapshot.",
            relocalization_status={
                "status": "localized" if transform is not None else "unavailable",
                "localized": transform is not None,
                "transform_epoch_s": float(transform.ts) if transform is not None else None,
            },
        )
        # A fresh mapping episode has a live world pose without a world->map
        # relocalization transform. Keep that diagnostic separate from map pose.
        self._attach_2d_slam_map("go2_slam_map_snapshot", result)
        return result

    @skill
    def go2_start_navigation_episode(
        self,
        *,
        clear_slam_map: bool = True,
        reason: str | None = None,
    ) -> SkillResult[NavigationSkillError]:
        """Start a navigation episode and optionally reset accumulated SLAM state."""

        reset_path: Path | None = None
        if clear_slam_map:
            reset_path = write_voxel_map_reset_signal(
                source="go2_start_navigation_episode",
                reason=reason,
            )
            self._latest_global_map = None
            self._latest_global_map_received_at_s = None
            self._latest_global_costmap = None
            self._latest_global_costmap_received_at_s = None
            self._slam_pose_trace = []
        self._active_navigation_task_id = None
        self._task_rerun_export_count = 0
        return SkillResult.ok(
            "Started Go2 navigation episode.",
            navigation_episode={
                "status": "started",
                "clear_slam_map": bool(clear_slam_map),
                "reset_signal_path": str(reset_path) if reset_path else None,
                "reason": reason,
            },
        )

    def _on_sport_odom(self, pose: PoseStamped) -> None:
        self._latest_sport_odom = pose
        self._latest_sport_odom_received_at_s = time.monotonic()

    def _on_global_costmap(self, occupancy_grid: OccupancyGrid) -> None:
        self._latest_global_costmap = occupancy_grid
        self._latest_global_costmap_received_at_s = time.monotonic()

    def _on_global_map(self, pointcloud: PointCloud2) -> None:
        self._latest_global_map = pointcloud
        self._latest_global_map_received_at_s = time.monotonic()

    def _on_lidar(self, pointcloud: PointCloud2) -> None:
        self._latest_lidar = pointcloud
        self._latest_lidar_received_at_s = time.monotonic()

    def _on_color_image(self, image: Image) -> None:
        self._latest_camera_image = image
        self._latest_camera_image_received_at_s = time.monotonic()

    def _on_surround_image(self, direction: str, image: Image) -> None:
        recorder = self._memory_motion_recorder
        if recorder is not None:
            recorder.offer(direction, image)

    def _start_memory_motion_recorder(self, node_id: str) -> HopMotionRecorder | None:
        if self._memory_motion_rec_hz <= 0.0 or self._memory_motion_recorder is not None:
            return None
        recorder = HopMotionRecorder(
            self._memory_motion_rec_root,
            f"memorynode-{uuid.uuid4().hex}",
            self._memory_motion_rec_hz,
        )
        try:
            recorder.start()
        except Exception as exc:  # noqa: BLE001 - recording must not break navigation
            logger.warning("Unable to start memory-node motion recorder for %s: %s", node_id, exc)
            return None
        self._memory_motion_recorder = recorder
        return recorder

    def _finish_memory_motion_recorder(
        self,
        recorder: HopMotionRecorder | None,
        metadata: dict[str, Any],
    ) -> None:
        if recorder is None or self._memory_motion_recorder is not recorder:
            return
        self._memory_motion_recorder = None
        try:
            recorder.finish(metadata)
        except Exception as exc:  # noqa: BLE001 - recording must not break navigation
            logger.warning("Unable to finish memory-node motion recorder: %s", exc)

    def _publish_tele_cmd_vel(self, twist: Twist) -> tuple[bool, dict[str, Any]]:
        try:
            self.tele_cmd_vel.publish(twist)
            return True, {"ok": True, "ack_waited": False, "transport": "tele_cmd_vel"}
        except Exception as exc:
            return False, {
                "ok": False,
                "error": "cmd_vel_publish_failed",
                "timed_out": False,
                "phase": "cmd_vel_publish",
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "command_may_have_been_accepted": False,
            }

    @skill
    def stop_movement(self) -> SkillResult[NavigationSkillError]:
        """Stop active Go2 motion and cancel any active navigation goal.

        This is the first recovery tool for stalls, protection blockers, or
        uncertain motion. It sends zero velocity and returns whether navigation
        cancel and zero-velocity publication were accepted; it is not a posture
        command and does not make the robot sit or lie down.
        """

        started_at_s = time.monotonic()
        command_sent, move_response = self._publish_tele_cmd_vel(Twist())
        navigation_cancelled = self._navigation_cancel_goal(timeout_s=1.0)
        metadata = {
            "navigation_cancelled": navigation_cancelled,
            "zero_velocity_sent": command_sent,
        }
        if not isinstance(move_response, bool):
            metadata["zero_velocity_response"] = move_response
        if not command_sent:
            result = SkillResult.fail(
                "EXECUTION_FAILED",
                "Failed to send zero-velocity stop command to the robot connection.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                "stop_movement",
                result,
                movement_kind="stop",
                started_at_s=started_at_s,
            )
        result = SkillResult.ok("Stopped robot movement.", **metadata)
        return self._trace_movement_skill_result(
            "stop_movement",
            result,
            movement_kind="stop",
            started_at_s=started_at_s,
        )

    @skill(uses=[CAP_MOVEMENT], requires_modules=["CostMapper", "ReplanningAStarPlanner"])
    def go2_navigate_to_memory_node(
        self,
        node_id: str,
        target_slam_pose: dict[str, Any] | None = None,
        map_frame: str | None = None,
        map_sequence: int | None = None,
        timeout_s: float = DEFAULT_RELATIVE_MOVE_TIMEOUT_S,
    ) -> SkillResult[NavigationSkillError]:
        """Navigate to a current graph/A<N> SLAM target using native DimOS planning.

        The authority proxy resolves the public node ID to ``target_slam_pose``;
        keeping resolution outside this skill prevents stale labels from being
        silently reused.
        """
        started_at_s = time.monotonic()
        if not str(node_id).strip() or not isinstance(target_slam_pose, dict):
            result = SkillResult.fail(
                "INVALID_TARGET", "A current SLAM target pose is required.", node_id=node_id
            )
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        occupancy_grid = getattr(self, "_latest_global_costmap", None)
        grid_frame = getattr(occupancy_grid, "frame_id", None)
        grid_received_at = getattr(self, "_latest_global_costmap_received_at_s", None)
        if grid_received_at is None or time.monotonic() - grid_received_at > self._2d_slam_map_max_age_s:
            result = SkillResult.fail("STALE_MAP", "Current navigation costmap is unavailable or stale.")
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        if (not map_frame or not grid_frame or map_frame != grid_frame
                or target_slam_pose.get("frame_id") != grid_frame):
            result = SkillResult.fail(
                "FRAME_MISMATCH", "Memory target frame must match the current costmap.",
                node_id=node_id, map_frame=map_frame, grid_frame=grid_frame,
                target_frame=target_slam_pose.get("frame_id"),
            )
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        try:
            x = float(target_slam_pose["x"])
            y = float(target_slam_pose["y"])
            yaw = float(target_slam_pose.get("yaw_rad", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            result = SkillResult.fail("INVALID_TARGET", f"Invalid SLAM target pose: {exc}")
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        if self._navigation_not_configured():
            result = SkillResult.fail("NOT_CONFIGURED", "Native navigation is not configured.")
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        tf = self.tf.get("world", "base_link")
        if tf is None or tf.to_pose().frame_id != grid_frame:
            result = SkillResult.fail("NO_ODOM", "Current SLAM pose is unavailable.")
            return self._trace_movement_skill_result(
                "go2_navigate_to_memory_node", result, movement_kind="memory_node", started_at_s=started_at_s
            )
        start_pose = tf.to_pose()
        goal_pose = PoseStamped(
            position=Vector3(x, y, start_pose.position.z),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
            frame_id=grid_frame,
        )
        metadata: dict[str, Any] = {
            "node_id": node_id,
            "target_slam_pose": {"x": x, "y": y, "yaw_rad": yaw},
            "map_sequence": map_sequence,
            "goal": pose_metadata(goal_pose),
        }
        timeout = max(float(timeout_s), 1.0)
        recorder = self._start_memory_motion_recorder(node_id)
        try:
            try:
                accepted = self._navigation_call("set_goal", goal_pose, timeout_s=timeout)
            except TimeoutError:
                metadata.update({"status": "TIMEOUT", "command_may_be_active": True})
                metadata["cancel_goal_sent"] = self._navigation_cancel_goal(timeout_s=1.0)
                result = SkillResult.fail(
                    "TIMEOUT",
                    "Timed out dispatching memory-node navigation.",
                    **metadata,
                )
            else:
                metadata["set_goal_accepted"] = bool(accepted)
                if not accepted:
                    metadata["status"] = self._navigation_failure_or("NO_PATH")
                    result = SkillResult.fail(
                        metadata["status"],
                        "Native planner rejected the memory-node target.",
                        **metadata,
                    )
                else:
                    status, state, navigation_started = (
                        self._wait_for_memory_navigation_terminal(timeout_s=timeout)
                    )
                    metadata["navigation_started"] = navigation_started
                    if state is not None:
                        metadata["navigation_state"] = state.value
                    if status == "TIMEOUT":
                        self._navigation_cancel_goal(timeout_s=1.0)
                        metadata.update(
                            {"status": "TIMEOUT", "command_may_be_active": True}
                        )
                        result = SkillResult.fail(
                            "TIMEOUT",
                            "Memory-node navigation timed out.",
                            **metadata,
                        )
                    elif status == "ARRIVED":
                        metadata["goal_reached"] = True
                        metadata["status"] = "ARRIVED"
                        result = SkillResult.ok(
                            "Reached memory-node target.", **metadata
                        )
                    else:
                        metadata["goal_reached"] = False
                        metadata["status"] = status
                        result = SkillResult.fail(
                            status,
                            "Memory-node navigation did not reach its target.",
                            **metadata,
                        )
        finally:
            self._finish_memory_motion_recorder(
                recorder,
                {
                    "tool_name": "go2_navigate_to_memory_node",
                    "node_id": node_id,
                    "target_slam_pose": {"x": x, "y": y, "yaw_rad": yaw},
                    "status": metadata.get("status", "ERROR"),
                    "direction": "memory_node",
                    "duration_s": time.monotonic() - started_at_s,
                    "record_hz": self._memory_motion_rec_hz,
                },
            )
        return self._trace_movement_skill_result(
            "go2_navigate_to_memory_node",
            result,
            movement_kind="memory_node",
            started_at_s=started_at_s,
        )

    @skill(
        uses=[CAP_MOVEMENT],
        requires_modules=["CostMapper", "ReplanningAStarPlanner"],
    )
    def relative_move(
        self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0
    ) -> SkillResult[NavigationSkillError]:
        """Move the Go2 relative to its current pose using DimOS local planning.

        This is a skill-level local move, not raw velocity. The request may be
        clipped by local planning, collision/protection layers, odometry, or
        timeout. A useful partial move is normal physical feedback, not
        automatically a code failure. The result reports requested/measured
        displacement, residual distance, heading drift, planner arrival,
        odometry confirmation, `precision_class`, `agent_interpretation`, and
        `recommended_next_action`. Treat `within_tolerance` as close enough,
        `partial` as useful progress that may need observe/shorter follow-up,
        `short_stop` or `blocked_or_protected` as a reason to stop/re-observe or
        change angle, and `unknown` as requiring diagnostics. Pure yaw requests
        are routed to `rotate_in_place`.

        The `degrees` argument is the desired final relative heading.

        Example calls:

            # Move to a point that's 2 meters forward and 1 to the right.
            relative_move(forward=2, left=-1, degrees=0)

            # Move back 1 meter, while still facing the same direction.
            relative_move(forward=-1, left=0, degrees=0)

            # Rotate 90 degrees to the right (in place)
            relative_move(forward=0, left=0, degrees=-90)

            # Move 3 meters left, and face that direction
            relative_move(forward=0, left=3, degrees=90)
        """
        started_at_s = time.monotonic()
        forward, left, degrees = float(forward), float(left), float(degrees)
        if (
            abs(forward) < self._pure_rotation_linear_epsilon_m
            and abs(left) < self._pure_rotation_linear_epsilon_m
            and abs(degrees) > 0.0
        ):
            return self.rotate_in_place(degrees=degrees)

        requested = {
            "forward_m": forward,
            "left_m": left,
            "yaw_deg": degrees,
        }

        if self._navigation_not_configured():
            metadata = {
                "requested": requested,
                "status": "NOT_CONFIGURED",
                "navigation_configured": False,
                "command_may_be_active": False,
            }
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                "NOT_CONFIGURED",
                self._relative_move_result_message(
                    metadata,
                    default="Relative move requires the navigation planner, but it is disabled.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        tf = self.tf.get("world", "base_link")
        if tf is None:
            metadata = {"requested": requested, "status": "NO_ODOM"}
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                "NO_ODOM",
                self._relative_move_result_message(
                    metadata,
                    default="Failed to get the current robot pose.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        start_pose = tf.to_pose()
        goal_pose = self._generate_new_goal(start_pose, forward, left, degrees)
        metadata = {
            "requested": requested,
            "start": pose_metadata(start_pose),
            "goal": pose_metadata(goal_pose),
            "arrival_tolerance": {
                "linear_m": self._relative_move_linear_tolerance_m,
                "yaw_deg": self._relative_move_yaw_tolerance_deg,
            },
        }

        timeout = self._relative_move_timeout_s
        try:
            accepted = self._navigation_call("set_goal", goal_pose, timeout_s=timeout)
        except TimeoutError as exc:
            metadata.update(
                self._measured_delta_metadata(
                    start_pose,
                    navigation_state_error=f"{exc.__class__.__name__}: {exc}",
                    goal_reached=False,
                )
            )
            metadata["status"] = "TIMEOUT"
            metadata["timeout_s"] = timeout
            metadata["timeout_phase"] = "set_goal"
            metadata["command_may_be_active"] = True
            metadata["cancel_goal_sent"] = self._navigation_cancel_goal(timeout_s=1.0)
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                "TIMEOUT",
                self._relative_move_result_message(
                    metadata,
                    default="Navigation timed out while dispatching the relative movement goal.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )
        metadata["set_goal_accepted"] = accepted
        if not accepted:
            reason = self._navigation_failure_or("NO_PATH")
            metadata["status"] = reason
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
                precision_class="blocked_or_protected",
            )
            result = SkillResult.fail(
                reason,
                self._relative_move_result_message(
                    metadata,
                    default="Navigation planner did not accept the relative movement goal.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        # TODO: Replace this polling with navigation completion events.
        time.sleep(self._relative_move_settle_s)

        start_time = time.monotonic()
        state: NavigationState | None = None
        state_error: str | None = None
        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0.0:
                state_error = None
                break
            try:
                state = self._navigation_get_state(timeout_s=remaining)
            except TimeoutError as exc:
                state_error = f"{exc.__class__.__name__}: {exc}"
                break
            if state != NavigationState.FOLLOWING_PATH:
                break
            time.sleep(min(self._relative_move_poll_s, max(remaining, 0.0)))

        if state is None or state == NavigationState.FOLLOWING_PATH:
            self._navigation_cancel_goal(timeout_s=1.0)
            metadata.update(
                self._measured_delta_metadata(
                    start_pose,
                    navigation_state=state,
                    navigation_state_error=state_error,
                    goal_reached=False,
                )
            )
            metadata["status"] = "TIMEOUT"
            metadata["timeout_s"] = timeout
            metadata["command_may_be_active"] = True
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                "TIMEOUT",
                self._relative_move_result_message(
                    metadata,
                    default="Navigation timed out before the relative movement completed.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        time.sleep(self._relative_move_settle_s)

        try:
            goal_check_timeout = max(timeout - (time.monotonic() - start_time), 0.001)
            goal_reached = self._navigation_is_goal_reached(timeout_s=goal_check_timeout)
        except TimeoutError as exc:
            metadata.update(
                self._measured_delta_metadata(
                    start_pose,
                    navigation_state=state,
                    navigation_state_error=f"{exc.__class__.__name__}: {exc}",
                    goal_reached=False,
                )
            )
            metadata["status"] = "TIMEOUT"
            metadata["timeout_s"] = timeout
            metadata["command_may_be_active"] = False
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                "TIMEOUT",
                self._relative_move_result_message(
                    metadata,
                    default="Navigation timed out while checking the final goal state.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        metadata.update(
            self._measured_delta_metadata(
                start_pose,
                navigation_state=state,
                goal_reached=goal_reached,
            )
        )
        if not goal_reached:
            reason = self._navigation_failure_or("CANCELLED")
            metadata["status"] = reason
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=False,
                odometry_confirmed=False,
            )
            result = SkillResult.fail(
                reason,
                self._relative_move_result_message(
                    metadata,
                    default="Navigation was cancelled or failed before reaching the relative movement goal.",
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        arrival_check = self._relative_move_arrival_check(requested, metadata.get("measured_delta"))
        metadata["arrival_check"] = arrival_check
        if not arrival_check["verified"]:
            reason = (
                "NO_ODOM"
                if arrival_check["reason"] == "missing_measured_delta"
                else "ARRIVAL_NOT_CONFIRMED"
            )
            metadata["status"] = reason
            self._annotate_relative_move_contract(
                metadata,
                requested=requested,
                planner_arrived=True,
                odometry_confirmed=False,
                precision_class=str(arrival_check.get("precision_class") or "unknown"),
            )
            result = SkillResult.fail(
                reason,
                self._relative_move_result_message(
                    metadata,
                    default=(
                        "Navigation reported arrival, but measured odometry only "
                        f"classified the movement as {metadata['precision_class']}."
                    ),
                ),
                **metadata,
            )
            return self._trace_movement_skill_result(
                "relative_move",
                result,
                movement_kind="relative_move",
                started_at_s=started_at_s,
            )

        metadata["status"] = "ARRIVED"
        self._annotate_relative_move_contract(
            metadata,
            requested=requested,
            planner_arrived=True,
            odometry_confirmed=True,
            precision_class=str(arrival_check.get("precision_class") or "unknown"),
        )
        result = SkillResult.ok(
            self._relative_move_result_message(
                metadata,
                default="Navigation goal reached.",
            ),
            **metadata,
        )
        return self._trace_movement_skill_result(
            "relative_move",
            result,
            movement_kind="relative_move",
            started_at_s=started_at_s,
        )

    @skill(uses=[CAP_MOVEMENT])
    def drive(
        self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0
    ) -> SkillResult[NavigationSkillError]:
        """Drive the Go2 with direct odometry-closed velocity, bypassing planning.

        This is the lightweight VLN execution primitive for short bounded
        translations. It publishes `tele_cmd_vel` fire-and-forget commands and
        verifies progress with firmware sport-mode odometry; it does not call
        the navigation planner or use lidar SLAM pose for closure.
        Pure yaw requests are delegated to `rotate_in_place`.

        Args:
            forward: Requested forward displacement in meters.
            left: Requested left displacement in meters.
            degrees: Requested final relative yaw in degrees.
        """
        started_at_s = time.monotonic()
        forward, left, degrees = float(forward), float(left), float(degrees)
        if (
            abs(forward) < self._pure_rotation_linear_epsilon_m
            and abs(left) < self._pure_rotation_linear_epsilon_m
            and abs(degrees) > 0.0
        ):
            return self.rotate_in_place(degrees=degrees)

        requested = {
            "forward_m": forward,
            "left_m": left,
            "yaw_deg": degrees,
        }
        requested_linear_m = math.hypot(forward, left)
        linear_tolerance_m = self._relative_move_effective_linear_tolerance_m(requested_linear_m)
        yaw_tolerance_deg = self._relative_move_yaw_tolerance_deg
        timeout_s = max(self._drive_timeout_s, 0.001)
        metadata: dict[str, Any] = {
            "requested": requested,
            "arrival_tolerance": {
                "linear_m": linear_tolerance_m,
                "yaw_deg": yaw_tolerance_deg,
            },
            "timeout_s": timeout_s,
            "controller": {
                "transport": "tele_cmd_vel",
                "ack_waited": False,
                "linear_gain": self._drive_linear_gain,
                "yaw_gain": self._drive_yaw_gain,
                "max_linear_speed_mps": self._drive_max_linear_speed_mps,
                "max_yaw_speed_rad_s": self._drive_max_yaw_speed_rad_s,
                "poll_s": self._drive_poll_s,
                "feedback_source": "rt/lf/sportmodestate",
                "max_feedback_age_s": self._direct_motion_feedback_timeout_s,
                "initial_feedback_timeout_s": self._direct_motion_initial_feedback_timeout_s,
                "feedback_timeout_s": self._direct_motion_feedback_timeout_s,
                "min_feedback_progress_m": self._drive_min_feedback_progress_m,
            },
            "feedback_watchdog_triggered": False,
        }

        start_pose = self._current_direct_motion_pose()
        if start_pose is None:
            metadata.update(
                {
                    "start": None,
                    "end": None,
                    "measured_delta": None,
                    "status": "NO_ODOM",
                    "zero_velocity_sent": False,
                }
            )
            self._annotate_drive_guidance(metadata)
            result = SkillResult.fail(
                "NO_ODOM",
                metadata["agent_interpretation"],
                **metadata,
            )
            return self._trace_direct_movement_skill_result(
                "drive",
                result,
                movement_kind="drive",
                started_at_s=started_at_s,
            )

        goal_pose = self._generate_new_goal(start_pose, forward, left, degrees)
        metadata["start"] = pose_metadata(start_pose)
        metadata["goal"] = pose_metadata(goal_pose)

        if self._drive_goal_reached(
            start_pose,
            goal_pose,
            linear_tolerance_m=linear_tolerance_m,
            yaw_tolerance_deg=yaw_tolerance_deg,
        ):
            metadata.update(self._drive_measurement_metadata(start_pose, start_pose, goal_pose))
            metadata["status"] = "ARRIVED"
            metadata["zero_velocity_sent"] = False
            self._annotate_drive_guidance(metadata)
            result = SkillResult.ok(metadata["agent_interpretation"], **metadata)
            return self._trace_direct_movement_skill_result(
                "drive",
                result,
                movement_kind="drive",
                started_at_s=started_at_s,
            )

        status = "TIMEOUT"
        last_pose = start_pose
        deadline = time.monotonic() + timeout_s
        # Direct velocity must fail closed instead of running open-loop on frozen odometry.
        feedback_anchor = start_pose
        feedback_deadline = time.monotonic() + self._direct_motion_initial_feedback_timeout_s
        try:
            while time.monotonic() < deadline:
                current_pose = self._current_direct_motion_pose()
                if current_pose is None:
                    status = "NO_ODOM"
                    break
                last_pose = current_pose
                if self._drive_goal_reached(
                    current_pose,
                    goal_pose,
                    linear_tolerance_m=linear_tolerance_m,
                    yaw_tolerance_deg=yaw_tolerance_deg,
                ):
                    status = "ARRIVED"
                    break

                if (
                    self._drive_residual_distance_m(current_pose, feedback_anchor)
                    >= self._drive_min_feedback_progress_m
                ):
                    feedback_anchor = current_pose
                    feedback_deadline = time.monotonic() + self._direct_motion_feedback_timeout_s
                elif time.monotonic() >= feedback_deadline:
                    status = "NO_PROGRESS"
                    metadata["feedback_watchdog_triggered"] = True
                    break

                command = self._drive_command_toward_goal(current_pose, goal_pose)
                command_sent, move_response = self._publish_tele_cmd_vel(command)
                if not command_sent:
                    status = "EXECUTION_FAILED"
                    metadata["connection_move_response"] = move_response
                    break
                time.sleep(max(self._drive_poll_s, 0.0))
        finally:
            stop_sent, stop_response = self._publish_tele_cmd_vel(Twist())
            metadata["zero_velocity_sent"] = stop_sent
            if not stop_sent:
                metadata["zero_velocity_response"] = stop_response

        end_pose = self._current_direct_motion_pose() or last_pose
        metadata.update(self._drive_measurement_metadata(start_pose, end_pose, goal_pose))
        metadata["status"] = status
        self._annotate_drive_guidance(metadata)

        if status == "ARRIVED":
            result = SkillResult.ok(metadata["agent_interpretation"], **metadata)
        elif status == "NO_PROGRESS":
            result = SkillResult.fail("NO_PROGRESS", metadata["agent_interpretation"], **metadata)
        elif status == "NO_ODOM":
            result = SkillResult.fail("NO_ODOM", metadata["agent_interpretation"], **metadata)
        elif status == "EXECUTION_FAILED":
            result = SkillResult.fail(
                "EXECUTION_FAILED",
                metadata["agent_interpretation"],
                **metadata,
            )
        else:
            result = SkillResult.fail("TIMEOUT", metadata["agent_interpretation"], **metadata)
        return self._trace_direct_movement_skill_result(
            "drive",
            result,
            movement_kind="drive",
            started_at_s=started_at_s,
        )

    @skill(uses=[CAP_MOVEMENT])
    def rotate_in_place(
        self,
        degrees: float,
        max_duration_s: float = 5.0,
        angular_speed_limit: float = 0.4,
    ) -> SkillResult[NavigationSkillError]:
        """Rotate the robot in place using direct yaw velocity, not global navigation.

        Rotation is calibrated for the Go2 yaw deadband, but the robot may still
        drift in translation or stop short when protection/odometry intervenes.
        Check the returned measured yaw and translation drift before using the
        result for return-to-start math. For large rotations, re-observe or read
        odometry before assuming the robot pivoted perfectly.

        Args:
            degrees: Requested yaw rotation in degrees. Positive turns left, negative turns right.
            max_duration_s: Safety timeout for the rotation command.
            angular_speed_limit: Maximum absolute yaw speed in radians per second.
        """
        started_at_s = time.monotonic()
        degrees = float(degrees)
        max_duration_s = float(max_duration_s)
        angular_speed_limit = abs(float(angular_speed_limit))
        requested = {
            "yaw_deg": degrees,
            "max_duration_s": max_duration_s,
            "angular_speed_limit_rad_s": angular_speed_limit,
            "min_effective_yaw_speed_rad_s": self._rotate_in_place_min_effective_yaw_speed_rad_s,
            "tolerance_deg": self._rotate_in_place_tolerance_deg,
        }

        if max_duration_s <= 0.0 or angular_speed_limit <= 0.0:
            result = SkillResult.fail(
                "INVALID_INPUT",
                "max_duration_s and angular_speed_limit must be positive.",
                requested=requested,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )

        start_pose = self._current_direct_motion_pose()
        if start_pose is None:
            result = SkillResult.fail(
                "NO_ODOM",
                "Failed to get the current robot pose before rotating.",
                requested=requested,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )

        target_rad = math.radians(degrees)
        target_abs = abs(target_rad)
        tolerance_rad = math.radians(self._rotate_in_place_tolerance_deg)
        metadata = {
            "requested": requested,
            "start": pose_metadata(start_pose),
            "target_yaw_rad": target_rad,
            "tolerance_rad": tolerance_rad,
            "feedback_watchdog": {
                "source": "rt/lf/sportmodestate",
                "max_age_s": self._direct_motion_feedback_timeout_s,
                "initial_timeout_s": self._direct_motion_initial_feedback_timeout_s,
                "timeout_s": self._direct_motion_feedback_timeout_s,
                "min_yaw_progress_deg": self._rotate_min_feedback_progress_deg,
            },
            "feedback_watchdog_triggered": False,
        }

        if target_abs <= tolerance_rad:
            metadata.update(self._rotation_measurement_metadata(start_pose, start_pose, 0.0))
            metadata["status"] = "ARRIVED"
            result = SkillResult.ok("Requested rotation is already within tolerance.", **metadata)
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )

        direction = 1.0 if target_rad > 0.0 else -1.0
        min_effective_yaw_speed = self._rotate_in_place_min_effective_yaw_speed_rad_s
        if angular_speed_limit < min_effective_yaw_speed:
            metadata["status"] = "YAW_SPEED_BELOW_EFFECTIVE_MIN"
            metadata["min_effective_yaw_speed_rad_s"] = min_effective_yaw_speed
            result = SkillResult.fail(
                "YAW_SPEED_BELOW_EFFECTIVE_MIN",
                "angular_speed_limit is below the calibrated minimum effective Go2 yaw speed.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )

        # Direct teleop-style rotation should not fight an existing navigation goal,
        # but this preemption must stay bounded on slow Jetson RPC links.
        metadata["cancel_goal_sent"] = self._navigation_cancel_goal(timeout_s=1.0)

        # ``max_duration_s`` is a safety deadline, not a requested motion
        # duration.  Keep the commanded rate independent so callers can leave
        # enough real-hardware margin without unintentionally slowing the turn.
        yaw_speed = angular_speed_limit
        command = Twist(
            linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, direction * yaw_speed)
        )

        cumulative_yaw_rad = 0.0
        feedback_anchor_yaw_rad = 0.0
        last_pose = start_pose
        deadline = time.monotonic() + max_duration_s
        feedback_deadline = time.monotonic() + self._direct_motion_initial_feedback_timeout_s
        status: NavigationError | Literal["ARRIVED"] = "TIMEOUT"
        move_failed = False
        try:
            while time.monotonic() < deadline:
                command_sent, move_response = self._publish_tele_cmd_vel(command)
                if not command_sent:
                    move_failed = True
                    status = "EXECUTION_FAILED"
                    metadata["connection_move_response"] = move_response
                    break
                time.sleep(self._rotate_in_place_poll_s)
                current_pose = self._current_direct_motion_pose()
                if current_pose is None:
                    status = "NO_ODOM"
                    break
                cumulative_yaw_rad += angle_diff(current_pose.yaw, last_pose.yaw)
                last_pose = current_pose
                if direction * cumulative_yaw_rad >= target_abs - tolerance_rad:
                    status = "ARRIVED"
                    break
                if abs(cumulative_yaw_rad - feedback_anchor_yaw_rad) >= math.radians(
                    self._rotate_min_feedback_progress_deg
                ):
                    feedback_anchor_yaw_rad = cumulative_yaw_rad
                    feedback_deadline = time.monotonic() + self._direct_motion_feedback_timeout_s
                elif time.monotonic() >= feedback_deadline:
                    status = "NO_PROGRESS"
                    metadata["feedback_watchdog_triggered"] = True
                    break
        finally:
            stop_sent, stop_response = self._publish_tele_cmd_vel(Twist())
            metadata["zero_velocity_sent"] = stop_sent
            if not stop_sent:
                metadata["zero_velocity_response"] = stop_response

        end_pose = self._current_direct_motion_pose() or last_pose
        metadata.update(
            self._rotation_measurement_metadata(start_pose, end_pose, cumulative_yaw_rad)
        )
        metadata["commanded_yaw_speed_rad_s"] = direction * yaw_speed
        metadata["status"] = status

        if status == "ARRIVED":
            result = SkillResult.ok("Rotation completed.", **metadata)
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )
        if status == "NO_ODOM":
            result = SkillResult.fail(
                "NO_ODOM",
                "Lost robot odometry while rotating.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )
        if status == "NO_PROGRESS":
            result = SkillResult.fail(
                "NO_PROGRESS",
                "Rotation received fresh body odometry but made no measurable yaw progress.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )
        if move_failed:
            result = SkillResult.fail(
                "EXECUTION_FAILED",
                "The robot connection rejected the rotation velocity command.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                "rotate_in_place",
                result,
                movement_kind="rotation",
                started_at_s=started_at_s,
            )
        result = SkillResult.fail(
            "TIMEOUT",
            "Rotation timed out before measured yaw reached the requested angle.",
            **metadata,
        )
        return self._trace_movement_skill_result(
            "rotate_in_place",
            result,
            movement_kind="rotation",
            started_at_s=started_at_s,
        )

    def _trace_movement_skill_result(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
        *,
        started_at_s: float | None = None,
        **context: object,
    ) -> SkillResult[NavigationSkillError]:
        if started_at_s is not None:
            record_skill_duration(result, started_at_s)
        self._update_active_navigation_task(tool_name)
        # Memory navigation has one canonical map contract. The generic
        # movement path also emits cleaned and segmented variants, which would
        # give the agent three nearly identical maps on a failed attempt.
        if tool_name != "go2_navigate_to_memory_node":
            self._attach_floor_plan(tool_name, result)
            self._attach_area_segmentation(tool_name, result)
        self._attach_2d_slam_map(tool_name, result)
        self._attach_latest_camera_image(tool_name, result)
        self._export_task_rerun_recording(tool_name, result)
        self._record_latest_motion_result(tool_name, result, context)
        return emit_skill_result_trace(
            logger,
            "movement_skill_result",
            tool_name,
            result,
            **context,
        )

    def _trace_direct_movement_skill_result(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
        *,
        started_at_s: float | None = None,
        **context: object,
    ) -> SkillResult[NavigationSkillError]:
        if started_at_s is not None:
            record_skill_duration(result, started_at_s)
        self._update_active_navigation_task(tool_name)
        self._attach_floor_plan(tool_name, result)
        self._attach_area_segmentation(tool_name, result)
        self._attach_2d_slam_map(tool_name, result)
        self._attach_latest_camera_image(tool_name, result)
        self._export_task_rerun_recording(tool_name, result)
        self._record_latest_motion_result(tool_name, result, context)
        return emit_skill_result_trace(
            logger,
            "movement_skill_result",
            tool_name,
            result,
            **context,
        )

    def _floor_plan_config(self) -> FloorPlanConfig:
        return FloorPlanConfig(
            occupied_threshold=self._floor_plan_occupied_threshold,
            closing_radius_m=self._floor_plan_closing_radius_m,
            opening_radius_m=self._floor_plan_opening_radius_m,
            min_component_area_m2=self._floor_plan_min_component_area_m2,
            free_space_closing_radius_m=self._floor_plan_free_space_closing_radius_m,
            max_fill_hole_area_m2=self._floor_plan_max_fill_hole_area_m2,
            boundary_wall_width_m=self._floor_plan_boundary_wall_width_m,
            structural_wall_min_length_m=self._floor_plan_structural_wall_min_length_m,
            structural_wall_width_m=self._floor_plan_structural_wall_width_m,
            output_max_size_px=self._floor_plan_size_px,
        )

    def _attach_floor_plan(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
    ) -> None:
        if tool_name not in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS:
            return

        metadata: dict[str, Any] = {
            "status": "unavailable",
            "source": "native_global_map_cleaned",
        }
        occupancy_grid = getattr(self, "_latest_global_costmap", None)
        received_at_s = getattr(self, "_latest_global_costmap_received_at_s", None)
        if occupancy_grid is None:
            result.metadata["floor_plan"] = metadata
            return

        now_s = time.monotonic()
        map_age_s = None if received_at_s is None else now_s - received_at_s
        metadata.update(
            {
                "map_frame": occupancy_grid.frame_id,
                "grid_ts": occupancy_grid.ts,
                "grid_age_s": map_age_s,
                "resolution_m_per_cell": occupancy_grid.resolution,
            }
        )
        if map_age_s is None or map_age_s > self._floor_plan_max_age_s:
            metadata["status"] = "stale"
            metadata["max_age_s"] = self._floor_plan_max_age_s
            result.metadata["floor_plan"] = metadata
            return

        pose = self._current_2d_slam_map_pose(tool_name)
        try:
            rendered = render_clean_floor_plan(
                occupancy_grid,
                config=self._floor_plan_config(),
                pose=pose,
            )
        except Exception as exc:
            metadata.update(
                {
                    "status": "render_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            result.metadata["floor_plan"] = metadata
            logger.exception("Failed to render clean floor plan for %s.", tool_name)
            return

        image_block = rendered.image_block
        metadata.update(rendered.metadata)
        metadata["status"] = "current"
        metadata["grid_age_s"] = map_age_s
        if pose is not None:
            metadata["pose"] = pose_metadata(pose)
        self._record_floor_plan_artifact(tool_name, image_block, metadata)
        result.metadata["floor_plan"] = metadata
        result.content_blocks.append(image_block)

    def _attach_area_segmentation(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
    ) -> None:
        if tool_name not in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS:
            return

        metadata: dict[str, Any] = {
            "status": "unavailable",
            "source": "native_global_map_area_segmented",
            "wall_inflation_m": DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M,
        }
        occupancy_grid = getattr(self, "_latest_global_costmap", None)
        received_at_s = getattr(self, "_latest_global_costmap_received_at_s", None)
        if occupancy_grid is None:
            result.metadata["area_segmentation"] = metadata
            return

        now_s = time.monotonic()
        map_age_s = None if received_at_s is None else now_s - received_at_s
        metadata.update(
            {
                "map_frame": occupancy_grid.frame_id,
                "grid_ts": occupancy_grid.ts,
                "grid_age_s": map_age_s,
                "resolution_m_per_cell": occupancy_grid.resolution,
            }
        )
        if map_age_s is None or map_age_s > self._floor_plan_max_age_s:
            metadata["status"] = "stale"
            metadata["max_age_s"] = self._floor_plan_max_age_s
            result.metadata["area_segmentation"] = metadata
            return

        pose = self._current_2d_slam_map_pose(tool_name)
        try:
            rendered = render_room_segmentation(
                occupancy_grid,
                floor_plan_config=self._floor_plan_config(),
                config=RoomSegmentationConfig(
                    wall_inflation_m=DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M,
                    output_max_size_px=self._floor_plan_size_px,
                ),
                pose=pose,
            )
        except Exception as exc:
            metadata.update(
                {
                    "status": "render_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            result.metadata["area_segmentation"] = metadata
            logger.exception("Failed to render area segmentation for %s.", tool_name)
            return

        image_block = rendered.image_block
        metadata.update(rendered.metadata)
        metadata["slam_area_nodes"] = build_slam_area_nodes(
            occupancy_grid, rendered.region_labels
        )
        metadata["status"] = "current"
        metadata["grid_age_s"] = map_age_s
        metadata["wall_inflation_m"] = DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M
        if pose is not None:
            metadata["pose"] = pose_metadata(pose)
        self._record_area_segmentation_artifact(tool_name, image_block, metadata)
        result.metadata["area_segmentation"] = metadata
        result.content_blocks.append(image_block)

    def _attach_2d_slam_map(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
        *,
        include_pose: bool = True,
    ) -> None:
        if tool_name not in _SLAM_MAP_ATTACHMENT_TOOLS:
            return

        metadata: dict[str, Any] = {
            "status": "unavailable",
            "source": "global_costmap",
        }
        occupancy_grid = getattr(self, "_latest_global_costmap", None)
        received_at_s = getattr(self, "_latest_global_costmap_received_at_s", None)
        if occupancy_grid is None:
            result.metadata["2d_slam_map"] = metadata
            return

        now_s = time.monotonic()
        map_age_s = None if received_at_s is None else now_s - received_at_s
        metadata.update(
            {
                "map_frame": occupancy_grid.frame_id,
                "grid_ts": occupancy_grid.ts,
                "grid_age_s": map_age_s,
                "resolution_m_per_cell": occupancy_grid.resolution,
            }
        )
        if map_age_s is None or map_age_s > self._2d_slam_map_max_age_s:
            metadata["status"] = "stale"
            metadata["max_age_s"] = self._2d_slam_map_max_age_s
            result.metadata["2d_slam_map"] = metadata
            return

        pose = self._map_frame_pose(occupancy_grid.frame_id) if include_pose else None
        if include_pose and pose is None:
            metadata["status"] = "pose_unavailable"
            result.metadata["2d_slam_map"] = metadata
            return

        if tool_name in {
            "go2_navigate_to_memory_node",
            "go2_report_relocalization_prior",
            "go2_slam_map_snapshot",
        }:
            self._attach_segmented_2d_slam_map(
                tool_name,
                result,
                occupancy_grid=occupancy_grid,
                pose=pose,
                map_age_s=map_age_s,
                metadata=metadata,
            )
            return

        trace = self._append_slam_trace_pose(pose)
        slam_area_nodes: list[dict[str, Any]] = []
        try:
            segmented = render_room_segmentation(
                occupancy_grid,
                floor_plan_config=self._floor_plan_config(),
                config=RoomSegmentationConfig(
                    wall_inflation_m=DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M,
                    output_max_size_px=self._floor_plan_size_px,
                ),
                pose=pose,
            )
            slam_area_nodes = build_slam_area_nodes(occupancy_grid, segmented.region_labels)
        except Exception:
            logger.debug("Unable to derive ephemeral SLAM area targets", exc_info=True)
        try:
            rendered = render_agent_occupancy_map(
                occupancy_grid,
                pose,
                trace,
                slam_area_nodes=slam_area_nodes,
                output_size_px=self._2d_slam_map_size_px,
                view_size_m=self._2d_slam_map_view_size_m,
            )
        except Exception as exc:
            metadata.update(
                {
                    "status": "render_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "pose_frame": pose.frame_id,
                }
            )
            result.metadata["2d_slam_map"] = metadata
            logger.exception("Failed to render 2D SLAM occupancy map for %s.", tool_name)
            return

        image_block = rendered.image_block
        metadata.update(rendered.metadata)
        # The segmentation pass is the source of ephemeral A<N> targets.
        metadata["slam_area_nodes"] = slam_area_nodes
        metadata["status"] = "current"
        metadata["grid_age_s"] = map_age_s
        metadata["pose"] = pose_metadata(pose)
        metadata["pose_sequence"] = len(getattr(self, "_slam_pose_trace", []) or [])
        self._record_2d_slam_map_artifact(tool_name, image_block, metadata)
        result.metadata["2d_slam_map"] = metadata
        result.content_blocks.append(image_block)

    def _attach_segmented_2d_slam_map(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
        *,
        occupancy_grid: OccupancyGrid,
        pose: PoseStamped | None,
        map_age_s: float | None,
        metadata: dict[str, Any],
    ) -> None:
        try:
            rendered = render_room_segmentation(
                occupancy_grid,
                floor_plan_config=self._floor_plan_config(),
                config=RoomSegmentationConfig(
                    wall_inflation_m=DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M,
                    output_max_size_px=self._2d_slam_map_size_px,
                ),
                pose=pose,
            )
            slam_area_nodes = build_slam_area_nodes(
                rendered.occupancy_grid,
                rendered.region_labels,
            )
        except Exception as exc:
            metadata.update(
                {
                    "status": "render_failed",
                    "source": "native_global_map_area_segmented",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            if pose is not None:
                metadata["pose_frame"] = pose.frame_id
            result.metadata["2d_slam_map"] = metadata
            logger.exception("Failed to render segmented 2D SLAM map for %s.", tool_name)
            return

        image_block = rendered.image_block
        metadata.update(rendered.metadata)
        metadata["slam_area_nodes"] = slam_area_nodes
        metadata["status"] = "current" if pose is not None else "static"
        metadata["grid_age_s"] = map_age_s
        metadata["wall_inflation_m"] = DEFAULT_AREA_SEGMENTATION_WALL_INFLATION_M
        metadata["image_coordinate_convention"] = "world_x_right_world_y_up"
        if pose is not None:
            metadata["pose"] = pose_metadata(pose)
            metadata["pose_sequence"] = len(getattr(self, "_slam_pose_trace", []) or [])
        self._record_2d_slam_map_artifact(tool_name, image_block, metadata)
        result.metadata["2d_slam_map"] = metadata
        result.content_blocks.append(image_block)

    def _update_active_navigation_task(self, tool_name: str) -> None:
        if tool_name not in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS:
            return
        task = self._current_skill_task_context()
        task_id = task.get("task_id")
        if not task_id:
            return
        if getattr(self, "_active_navigation_task_id", None) == task_id:
            return
        self._active_navigation_task_id = task_id
        self._task_rerun_export_count = 0
        self._slam_pose_trace = []

    def _export_task_rerun_recording(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
    ) -> None:
        if tool_name not in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS:
            return
        if not getattr(self, "_task_rerun_export_enabled", DEFAULT_TASK_RERUN_EXPORT_ENABLED):
            return

        task = self._current_skill_task_context()
        task_id = task.get("task_id")
        if not task_id:
            return

        metadata: dict[str, Any] = {
            "status": "unavailable",
            "task_id": task_id,
            "task_goal": task.get("task_goal"),
            "source": "offline_rerun_export",
        }
        try:
            artifact_dir = resolve_artifact_dir(runtime_config=self._runtime_config())
        except Exception:
            logger.exception("Failed to resolve artifact directory for task Rerun export.")
            metadata["status"] = "artifact_dir_error"
            result.metadata["rerun_recording"] = metadata
            return
        if artifact_dir is None:
            metadata["status"] = "artifact_dir_unavailable"
            result.metadata["rerun_recording"] = metadata
            return

        pose = self._current_2d_slam_map_pose(tool_name)
        if pose is None:
            metadata["status"] = "pose_unavailable"
            result.metadata["rerun_recording"] = metadata
            return

        trace = getattr(self, "_slam_pose_trace", None)
        if not isinstance(trace, list) or not trace:
            trace = self._append_slam_trace_pose(pose)

        rrd_path = Path(artifact_dir) / "rerun" / f"{_safe_task_token(task_id)}.rrd"
        try:
            export_metadata = self._write_task_rerun_recording(
                rrd_path,
                task_id=task_id,
                task_goal=task.get("task_goal"),
                pose=pose,
                trace=trace,
                global_costmap=getattr(self, "_latest_global_costmap", None),
                global_map=getattr(self, "_latest_global_map", None),
                lidar=getattr(self, "_latest_lidar", None),
                sport_odom=getattr(self, "_latest_sport_odom", None),
            )
        except Exception as exc:
            logger.exception("Failed to export task Rerun recording.", task_id=task_id)
            metadata.update(
                {
                    "status": "export_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "path": str(rrd_path),
                }
            )
            result.metadata["rerun_recording"] = metadata
            return

        self._task_rerun_export_count = getattr(self, "_task_rerun_export_count", 0) + 1
        metadata.update(export_metadata)
        metadata.update(
            {
                "status": "current",
                "path": str(rrd_path),
                "export_count": self._task_rerun_export_count,
            }
        )
        result.metadata["rerun_recording"] = metadata

    def _write_task_rerun_recording(
        self,
        rrd_path: Path,
        *,
        task_id: str,
        task_goal: str | None,
        pose: PoseStamped,
        trace: list[PoseStamped],
        global_costmap: OccupancyGrid | None,
        global_map: PointCloud2 | None,
        lidar: PointCloud2 | None,
        sport_odom: PoseStamped | None,
    ) -> dict[str, Any]:
        import rerun as rr

        from dimos.visualization.rerun.init import rerun_init

        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rerun_init(f"dimos-go2-task-{task_id}")
        rr.save(str(rrd_path))
        try:
            rr.set_time("movement_export", sequence=getattr(self, "_task_rerun_export_count", 0))

            rr.log("task", rr.TextDocument(_task_rerun_summary(task_id, task_goal)), static=True)
            if global_costmap is not None:
                rr.log("world/global_costmap", global_costmap.to_rerun())
            if global_map is not None:
                global_map_rr, global_map_points = self._task_rerun_pointcloud_to_rerun(
                    global_map,
                    colors=[80, 180, 255],
                )
                rr.log(
                    "world/global_map",
                    global_map_rr,
                )
            else:
                global_map_points = 0
            if lidar is not None:
                lidar_rr, lidar_points = self._task_rerun_pointcloud_to_rerun(
                    lidar,
                    colors=[255, 255, 255],
                )
                rr.log(
                    "world/lidar",
                    lidar_rr,
                )
            else:
                lidar_points = 0
            if sport_odom is not None:
                rr.log("world/odom/sport", sport_odom.to_rerun())
                rr.log("world/odom/sport_facing", sport_odom.to_rerun_arrow())
            rr.log("world/robot/base_link", pose.to_rerun())
            rr.log("world/robot/facing", pose.to_rerun_arrow())
            trace_points = [[p.x, p.y, p.z] for p in trace]
            if len(trace_points) >= 2:
                rr.log(
                    "world/robot/trace",
                    rr.LineStrips3D([trace_points], colors=[0, 210, 230]),
                )
            elif trace_points:
                rr.log(
                    "world/robot/trace_points",
                    rr.Points3D(trace_points, colors=[0, 210, 230]),
                )
        finally:
            rr.rerun_shutdown()
        return {
            "global_costmap": global_costmap is not None,
            "global_map": global_map is not None,
            "lidar": lidar is not None,
            "global_map_points": global_map_points,
            "lidar_points": lidar_points,
            "sport_odom": sport_odom is not None,
            "robot_pose": True,
            "trace_points": len(trace_points),
        }

    def _task_rerun_pointcloud_to_rerun(
        self,
        pointcloud: PointCloud2,
        *,
        colors: list[int],
    ) -> tuple[Any, int]:
        voxel_size = max(0.0, self._task_rerun_point_voxel_size_m)
        export_cloud = pointcloud.voxel_downsample(voxel_size) if voxel_size > 0 else pointcloud
        return (
            export_cloud.to_rerun(
                voxel_size=voxel_size,
                colors=colors,
                mode="points",
            ),
            len(export_cloud),
        )

    def _current_skill_task_context(self) -> dict[str, str]:
        context = current_skill_context()
        if not isinstance(context, dict):
            return {}
        task = context.get("task")
        if isinstance(task, dict) and task.get("task_id"):
            return {
                key: str(value)
                for key, value in task.items()
                if key in {"task_id", "task_goal", "task_origin"} and value is not None
            }
        status_context = context.get("tool_status_context")
        if not isinstance(status_context, dict):
            return {}
        movement_authority = status_context.get("movement_authority")
        if not isinstance(movement_authority, dict) or not movement_authority.get("task_id"):
            return {}
        task_context = {"task_id": str(movement_authority["task_id"])}
        task_goal = movement_authority.get("task_goal")
        if task_goal is not None:
            task_context["task_goal"] = str(task_goal)
        return task_context

    def _append_slam_trace_pose(self, pose: PoseStamped) -> list[PoseStamped]:
        trace = getattr(self, "_slam_pose_trace", None)
        if not isinstance(trace, list):
            trace = []
            self._slam_pose_trace = trace
        trace.append(pose)
        max_points = max(1, int(getattr(self, "_2d_slam_trace_max_points", 1)))
        if len(trace) > max_points:
            del trace[: len(trace) - max_points]
        return list(trace)

    def _record_2d_slam_map_artifact(
        self,
        tool_name: str,
        image_block: dict[str, object],
        metadata: dict[str, Any],
    ) -> None:
        self._record_movement_image_artifact(
            tool_name,
            image_block,
            metadata,
            kind="2d_slam_map",
        )

    def _record_floor_plan_artifact(
        self,
        tool_name: str,
        image_block: dict[str, object],
        metadata: dict[str, Any],
    ) -> None:
        self._record_movement_image_artifact(
            tool_name,
            image_block,
            metadata,
            kind="floor_plan",
        )

    def _record_area_segmentation_artifact(
        self,
        tool_name: str,
        image_block: dict[str, object],
        metadata: dict[str, Any],
    ) -> None:
        self._record_movement_image_artifact(
            tool_name,
            image_block,
            metadata,
            kind="area_segmentation",
        )

    def _save_final_floor_plan(self) -> None:
        occupancy_grid = getattr(self, "_latest_global_costmap", None)
        if occupancy_grid is None:
            return
        try:
            rendered = render_clean_floor_plan(
                occupancy_grid,
                config=self._floor_plan_config(),
                finalized=True,
            )
            artifact_dir = resolve_artifact_dir(runtime_config=self._runtime_config())
            if artifact_dir is None:
                return
            slam_dir = Path(artifact_dir).parent / "slam"
            slam_dir.mkdir(parents=True, exist_ok=True)
            png_path = slam_dir / "floor_plan.png"
            metadata_path = slam_dir / "floor_plan.json"
            image_bytes = rendered.image_block.get("data")
            if not isinstance(image_bytes, str):
                raise ValueError("floor plan image did not encode to base64")

            png_path.write_bytes(base64.b64decode(image_bytes))
            metadata_path.write_text(json.dumps(rendered.metadata, indent=2), encoding="utf-8")
            logger.info(
                "Saved final clean floor plan",
                path=str(png_path),
                metadata_path=str(metadata_path),
            )
        except Exception:
            logger.exception("Failed to save final clean floor plan.")

    def _save_relocalization_bundle(self) -> dict[str, Any]:
        manifest = self._relocalization_bundle_manifest_base()
        try:
            artifact_dir = resolve_artifact_dir(runtime_config=self._runtime_config())
        except Exception:
            logger.exception("Failed to resolve artifact directory for relocalization bundle.")
            manifest["reason"] = "artifact_dir_error"
            return manifest
        if artifact_dir is None:
            manifest["reason"] = "artifact_dir_unavailable"
            return manifest

        run_root = Path(artifact_dir).parent
        bundle_dir = run_root / "slam" / "relocalization"
        manifest_path = bundle_dir / "manifest.json"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        global_map = getattr(self, "_latest_global_map", None)
        received_at_s = getattr(self, "_latest_global_map_received_at_s", None)
        now_s = time.monotonic()
        map_age_s = None if received_at_s is None else now_s - received_at_s
        manifest["global_map_age_s"] = map_age_s
        manifest["max_age_s"] = self._relocalization_map_max_age_s

        if global_map is None:
            manifest["status"] = "unavailable"
            manifest["reason"] = "global_map_unavailable"
            self._write_relocalization_manifest(manifest_path, manifest)
            return manifest
        if map_age_s is None or map_age_s > self._relocalization_map_max_age_s:
            manifest.update(
                {
                    "status": "unavailable",
                    "reason": "global_map_stale",
                    "frame_id": global_map.frame_id,
                    "point_count": len(global_map),
                }
            )
            self._write_relocalization_manifest(manifest_path, manifest)
            return manifest

        global_map_path = bundle_dir / "global_map.pc2.lcm"
        costmap_path = bundle_dir / "global_costmap.occ.lcm"
        try:
            global_map_path.write_bytes(global_map.lcm_encode())
            costmap = getattr(self, "_latest_global_costmap", None)
            costmap_saved = False
            if costmap is not None and hasattr(costmap, "lcm_encode"):
                costmap_path.write_bytes(costmap.lcm_encode())
                costmap_saved = True
            pose = self._safe_current_relocalization_pose()
            slam_status_path = run_root / "logs" / "go2-slam-status.json"
            graph_memory_path = run_root / "logs" / "go2-graph-memory.json"
            manifest.update(
                {
                    "status": "current",
                    "bundle_path": str(bundle_dir),
                    "global_map_path": str(global_map_path),
                    "global_map_file": str(global_map_path),
                    "frame_id": global_map.frame_id,
                    "point_count": len(global_map),
                    "pointcloud_ts": global_map.ts,
                    "saved_at_epoch_s": time.time(),
                    "costmap_path": str(costmap_path) if costmap_saved else None,
                    "costmap_saved": costmap_saved,
                    "pose": pose_metadata(pose) if pose is not None else None,
                    "slam_status_path": str(slam_status_path)
                    if slam_status_path.exists()
                    else None,
                    "graph_memory_path": str(graph_memory_path)
                    if graph_memory_path.exists()
                    else None,
                }
            )
            self._write_relocalization_manifest(manifest_path, manifest)
            logger.info(
                "Saved Go2 relocalization bundle",
                path=str(bundle_dir),
                global_map_path=str(global_map_path),
                point_count=len(global_map),
            )
            return manifest
        except Exception:
            logger.exception("Failed to save Go2 relocalization bundle.")
            manifest["status"] = "save_failed"
            self._write_relocalization_manifest(manifest_path, manifest)
            return manifest

    def _relocalization_bundle_manifest_base(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": "unitree_skill_container_final_global_map",
            "status": "unavailable",
            "relocalization_input": "global_map_pointcloud2_lcm",
            "backend_note": (
                "Use global_map.pc2.lcm as RelocalizationModule.map_file input; "
                "live point-cloud matching still controls acceptance."
            ),
        }

    def _write_relocalization_manifest(self, path: Path, manifest: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _safe_current_relocalization_pose(self) -> PoseStamped | None:
        try:
            return self._current_direct_motion_pose()
        except Exception:
            pass
        try:
            return self._current_base_pose()
        except Exception:
            return None

    def _attach_latest_camera_image(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
    ) -> None:
        if tool_name not in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS:
            return

        metadata: dict[str, Any] = {
            "status": "unavailable",
            "source": "color_image",
        }
        image = getattr(self, "_latest_camera_image", None)
        received_at_s = getattr(self, "_latest_camera_image_received_at_s", None)
        if image is None:
            result.metadata["camera_image"] = metadata
            return

        now_s = time.monotonic()
        image_age_s = None if received_at_s is None else now_s - received_at_s
        metadata.update(
            {
                "frame_id": image.frame_id,
                "image_ts": image.ts,
                "image_age_s": image_age_s,
                "width_px": image.width,
                "height_px": image.height,
                "format": image.format.value,
            }
        )
        if image_age_s is None or image_age_s > self._camera_image_max_age_s:
            metadata["status"] = "stale"
            metadata["max_age_s"] = self._camera_image_max_age_s
            result.metadata["camera_image"] = metadata
            return

        try:
            encoded = image.agent_encode()
            if not encoded or not isinstance(encoded[0], dict):
                raise ValueError("camera image did not encode to a content block")
            image_block = encoded[0]
        except Exception as exc:
            metadata.update(
                {
                    "status": "encode_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            result.metadata["camera_image"] = metadata
            logger.exception("Failed to encode latest camera image for %s.", tool_name)
            return

        metadata["status"] = "current"
        self._record_movement_image_artifact(
            tool_name,
            image_block,
            metadata,
            kind="camera_image",
        )
        result.metadata["camera_image"] = metadata
        result.content_blocks.append(image_block)

    def _record_movement_image_artifact(
        self,
        tool_name: str,
        image_block: dict[str, Any],
        metadata: dict[str, Any],
        *,
        kind: str,
    ) -> None:
        try:
            artifact_dir = resolve_artifact_dir(runtime_config=self._runtime_config())
        except Exception:
            logger.exception("Failed to resolve artifact directory for movement image.")
            return
        if artifact_dir is None:
            return
        try:
            ref = ArtifactLedger(artifact_dir=artifact_dir).record_block(
                image_block,
                source=f"movement:{tool_name}:{kind}",
                message_type="tool",
            )
        except Exception:
            logger.exception("Failed to persist movement image artifact.", kind=kind)
            return
        metadata["artifact_id"] = ref.artifact_id
        metadata["artifact_path"] = ref.path

    def _current_2d_slam_map_pose(self, tool_name: str) -> PoseStamped | None:
        try:
            pose = self._current_direct_motion_pose()
        except Exception:
            logger.exception("Failed to read direct pose for 2D SLAM map.", tool_name=tool_name)
            pose = None
        if pose is not None:
            return pose
        try:
            return self._current_base_pose()
        except Exception:
            logger.exception("Failed to read base pose for 2D SLAM map.", tool_name=tool_name)
            return None

    def _map_frame_pose(self, frame_id: str) -> PoseStamped | None:
        """Use only a live pose expressed in the occupancy grid's frame."""
        try:
            base_pose = self._current_base_pose()
            if base_pose is not None and base_pose.frame_id == frame_id:
                return base_pose
            direct_pose = self._current_direct_motion_pose()
            if direct_pose is not None and direct_pose.frame_id == frame_id:
                return direct_pose
        except Exception:
            logger.exception("Failed to read pose for 2D SLAM map.")
        return None

    def _drive_command_toward_goal(
        self, current_pose: PoseStamped, goal_pose: PoseStamped
    ) -> Twist:
        world_error = goal_pose.position - current_pose.position
        world_error.z = 0.0
        local_error = current_pose.orientation.conjugate().rotate_vector(world_error)
        vx = local_error.x * self._drive_linear_gain
        vy = local_error.y * self._drive_linear_gain
        speed = math.hypot(vx, vy)
        max_speed = max(self._drive_max_linear_speed_mps, 0.0)
        if max_speed <= 0.0:
            vx = 0.0
            vy = 0.0
        elif speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        yaw_error = angle_diff(goal_pose.yaw, current_pose.yaw)
        wz = self._clamp(
            yaw_error * self._drive_yaw_gain,
            -abs(self._drive_max_yaw_speed_rad_s),
            abs(self._drive_max_yaw_speed_rad_s),
        )
        return Twist(linear=Vector3(vx, vy, 0.0), angular=Vector3(0.0, 0.0, wz))

    def _drive_goal_reached(
        self,
        current_pose: PoseStamped,
        goal_pose: PoseStamped,
        *,
        linear_tolerance_m: float,
        yaw_tolerance_deg: float,
    ) -> bool:
        residual_m = self._drive_residual_distance_m(current_pose, goal_pose)
        yaw_error_deg = abs(math.degrees(angle_diff(goal_pose.yaw, current_pose.yaw)))
        return residual_m <= linear_tolerance_m and yaw_error_deg <= yaw_tolerance_deg

    def _drive_measurement_metadata(
        self,
        start_pose: PoseStamped,
        end_pose: PoseStamped,
        goal_pose: PoseStamped,
    ) -> dict[str, Any]:
        heading_error_deg = math.degrees(angle_diff(goal_pose.yaw, end_pose.yaw))
        return {
            "end": pose_metadata(end_pose),
            "measured_delta": self._direct_motion_delta_metadata(start_pose, end_pose),
            "residual_distance_m": self._drive_residual_distance_m(end_pose, goal_pose),
            "heading_error_deg": heading_error_deg,
            "heading_drift_deg": heading_error_deg,
        }

    def _direct_motion_delta_metadata(
        self, start_pose: PoseStamped, end_pose: PoseStamped
    ) -> dict[str, Any]:
        delta = odom_delta_metadata(start_pose, end_pose)
        assert delta is not None
        delta["linear_m"] = self._drive_residual_distance_m(start_pose, end_pose)
        return delta

    def _drive_residual_distance_m(
        self, current_pose: PoseStamped, goal_pose: PoseStamped
    ) -> float:
        dx = goal_pose.position.x - current_pose.position.x
        dy = goal_pose.position.y - current_pose.position.y
        return math.hypot(dx, dy)

    def _annotate_drive_guidance(self, metadata: dict[str, Any]) -> None:
        requested = metadata.get("requested")
        measured_delta = metadata.get("measured_delta")
        has_measurement = isinstance(measured_delta, dict) and metadata.get("end") is not None
        requested_displacement_m = (
            math.hypot(
                self._float_metadata(requested.get("forward_m")),
                self._float_metadata(requested.get("left_m")),
            )
            if isinstance(requested, dict)
            else 0.0
        )
        measured_displacement_m = (
            abs(self._float_metadata(measured_delta.get("linear_m"))) if has_measurement else None
        )
        residual_distance_m = self._float_metadata(metadata.get("residual_distance_m"))
        heading_error_deg = abs(self._float_metadata(metadata.get("heading_error_deg")))
        linear_tolerance_m = self._float_metadata(
            metadata.get("arrival_tolerance", {}).get("linear_m")
            if isinstance(metadata.get("arrival_tolerance"), dict)
            else None
        )
        yaw_tolerance_deg = self._float_metadata(
            metadata.get("arrival_tolerance", {}).get("yaw_deg")
            if isinstance(metadata.get("arrival_tolerance"), dict)
            else None
        )
        translation_ok = has_measurement and residual_distance_m <= linear_tolerance_m
        yaw_ok = has_measurement and heading_error_deg <= yaw_tolerance_deg
        progress_ratio = (
            min((measured_displacement_m or 0.0) / requested_displacement_m, 1.0)
            if has_measurement and requested_displacement_m > 0.0
            else None
        )
        if not has_measurement:
            reason = "missing_measured_delta"
        elif translation_ok and yaw_ok:
            reason = "within_tolerance"
        elif translation_ok:
            reason = "yaw_out_of_tolerance"
        elif yaw_ok:
            reason = "translation_out_of_tolerance"
        else:
            reason = "translation_and_yaw_out_of_tolerance"

        if not has_measurement:
            precision_class = "unknown"
        elif requested_displacement_m <= 0.0:
            precision_class = "within_tolerance" if yaw_ok else "partial"
        elif translation_ok and yaw_ok:
            precision_class = "exact" if residual_distance_m <= 0.05 else "within_tolerance"
        elif progress_ratio is None or progress_ratio < 0.25:
            precision_class = "short_stop"
        else:
            precision_class = "partial"

        metadata.update(
            {
                "requested_displacement_m": requested_displacement_m,
                "measured_displacement_m": measured_displacement_m,
                "precision_class": precision_class,
                "arrival_check": {
                    "verified": translation_ok and yaw_ok,
                    "reason": reason,
                    "translation_ok": translation_ok,
                    "yaw_ok": yaw_ok,
                    "requested_linear_m": requested_displacement_m,
                    "measured_linear_m": measured_displacement_m,
                    "linear_progress_ratio": progress_ratio,
                    "linear_tolerance_m": linear_tolerance_m,
                    "yaw_error_deg": heading_error_deg,
                    "yaw_tolerance_deg": yaw_tolerance_deg,
                },
            }
        )
        interpretation, next_action = self._drive_agent_guidance(metadata)
        metadata["agent_interpretation"] = interpretation
        metadata["recommended_next_action"] = next_action

    def _drive_agent_guidance(self, metadata: dict[str, Any]) -> tuple[str, str]:
        status = str(metadata.get("status") or "")
        precision_class = str(metadata.get("precision_class") or "unknown")
        if status == "ARRIVED":
            return (
                "Drive reached the requested odometry target within tolerance.",
                "Continue the VLN task; use observe only if the next decision needs fresh scene evidence.",
            )
        if status == "TIMEOUT" and precision_class == "partial":
            return (
                "Drive made useful odometry progress but timed out before confirmed arrival.",
                "Use the measured_delta/residual_distance fields before deciding to retry a shorter drive or observe.",
            )
        if status == "TIMEOUT":
            return (
                "Drive timed out before useful confirmed completion.",
                "Do not repeat blindly; check pose/situation, then choose a shorter bounded drive or stop.",
            )
        if status == "NO_PROGRESS":
            return (
                "Drive received fresh body odometry but made no measurable progress.",
                "Do not repeat blindly; check for protection, traction, or an obstacle before trying a safer move.",
            )
        if status == "NO_ODOM":
            return (
                "Drive cannot be verified because odometry is unavailable.",
                "Do not issue blind translation; restore odometry or ask the operator for a safety check.",
            )
        if status == "EXECUTION_FAILED":
            return (
                "Drive could not publish a velocity command to the robot connection.",
                "Stop movement if needed and inspect cmd_vel transport before retrying.",
            )
        return (
            "Drive result is uncertain.",
            "Inspect measured_delta and current situation before issuing another movement.",
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    def _record_latest_motion_result(
        self,
        tool_name: str,
        result: SkillResult[NavigationSkillError],
        context: dict[str, object],
    ) -> None:
        navigation = getattr(self, "_navigation", None)
        record_motion_result = getattr(navigation, "record_motion_result", None)
        if not callable(record_motion_result):
            return
        heading_drift_deg = result.metadata.get("heading_drift_deg")
        if heading_drift_deg is None:
            heading_drift_deg = result.metadata.get("heading_error_deg")
        try:
            record_motion_result(
                {
                    "tool_name": tool_name,
                    "movement_kind": context.get("movement_kind"),
                    "success": result.success,
                    "error_code": result.error_code,
                    "message": result.message,
                    "duration_ms": round(result.duration_ms, 1),
                    "status": result.metadata.get("status"),
                    "requested": result.metadata.get("requested"),
                    "measured_delta": result.metadata.get("measured_delta"),
                    "arrival_check": result.metadata.get("arrival_check"),
                    "precision_class": result.metadata.get("precision_class"),
                    "residual_distance_m": result.metadata.get("residual_distance_m"),
                    "heading_drift_deg": heading_drift_deg,
                    "odometry_confirmed": result.metadata.get("odometry_confirmed"),
                    "planner_arrived": result.metadata.get("planner_arrived"),
                    "agent_interpretation": result.metadata.get("agent_interpretation"),
                    "recommended_next_action": result.metadata.get("recommended_next_action"),
                    "goal_reached": result.metadata.get("goal_reached"),
                    "navigation_state": result.metadata.get("navigation_state"),
                    "physically_satisfied": self._motion_physically_satisfied(result),
                }
            )
        except Exception:
            logger.exception("Failed to record latest motion result.", tool_name=tool_name)

    def _motion_physically_satisfied(
        self, result: SkillResult[NavigationSkillError]
    ) -> bool | None:
        arrival_check = result.metadata.get("arrival_check")
        if isinstance(arrival_check, dict):
            return bool(arrival_check.get("verified"))
        if result.metadata.get("status") == "ARRIVED":
            return True
        if result.metadata.get("measured_delta") is not None:
            return False if not result.success else None
        return None

    def _generate_new_goal(
        self, current_pose: PoseStamped, forward: float, left: float, degrees: float
    ) -> PoseStamped:
        local_offset = Vector3(forward, left, 0)
        global_offset = current_pose.orientation.rotate_vector(local_offset)
        goal_position = current_pose.position + global_offset

        current_euler = current_pose.orientation.to_euler()
        goal_yaw = current_euler.yaw + math.radians(degrees)
        goal_euler = Vector3(current_euler.roll, current_euler.pitch, goal_yaw)
        goal_orientation = Quaternion.from_euler(goal_euler)

        return PoseStamped(
            position=goal_position,
            orientation=goal_orientation,
            frame_id=current_pose.frame_id,
        )

    def _navigation_not_configured(self) -> bool:
        from dimos.core.coordination.blueprints import DisabledModuleProxy

        return isinstance(getattr(self, "_navigation", None), DisabledModuleProxy)

    def _navigation_call(
        self, name: str, *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Any:
        method = getattr(self._navigation, name)
        call_with_timeout = getattr(method, "call_with_timeout", None)
        if timeout_s is not None and callable(call_with_timeout):
            return call_with_timeout(max(timeout_s, 0.001), *args, **kwargs)
        return method(*args, **kwargs)

    def _navigation_get_state(self, *, timeout_s: float | None = None) -> NavigationState:
        return self._navigation_call("get_state", timeout_s=timeout_s)

    def _navigation_is_goal_reached(self, *, timeout_s: float | None = None) -> bool:
        return bool(self._navigation_call("is_goal_reached", timeout_s=timeout_s))

    def _navigation_cancel_goal(self, *, timeout_s: float | None = None) -> bool:
        try:
            return bool(self._navigation_call("cancel_goal", timeout_s=timeout_s))
        except TimeoutError:
            return False

    def _wait_for_memory_navigation_terminal(
        self, *, timeout_s: float
    ) -> tuple[NavigationError | Literal["ARRIVED"], NavigationState | None, bool]:
        """Wait through planner startup and replans until a confirmed terminal result."""
        deadline = time.monotonic() + timeout_s
        state: NavigationState | None = None
        navigation_started = False
        while time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.001)
            try:
                if self._navigation_is_goal_reached(timeout_s=remaining):
                    return "ARRIVED", state, navigation_started

                failure = self._navigation_call(
                    "get_last_failure",
                    timeout_s=max(deadline - time.monotonic(), 0.001),
                )
                if not failure:
                    state = self._navigation_get_state(
                        timeout_s=max(deadline - time.monotonic(), 0.001)
                    )
            except TimeoutError:
                return "TIMEOUT", state, navigation_started
            if failure:
                return (
                    self._navigation_failure_code(failure, "CANCELLED"),
                    state,
                    navigation_started,
                )
            if state in {NavigationState.FOLLOWING_PATH, NavigationState.RECOVERY}:
                navigation_started = True

            # IDLE without arrival/failure is transient during initial planning
            # and replanning. Only explicit terminal signals end this wait.
            time.sleep(
                min(
                    self._relative_move_poll_s,
                    max(deadline - time.monotonic(), 0.0),
                )
            )
        return "TIMEOUT", state, navigation_started

    def _measured_delta_metadata(
        self,
        start_pose: PoseStamped,
        *,
        navigation_state: NavigationState | None = None,
        navigation_state_error: str | None = None,
        goal_reached: bool | None = None,
    ) -> dict[str, object]:
        end_pose = self._current_base_pose()
        metadata: dict[str, object] = {
            "end": pose_metadata(end_pose),
            "measured_delta": odom_delta_metadata(start_pose, end_pose),
        }
        if navigation_state is not None:
            metadata["navigation_state"] = navigation_state.value
        if navigation_state_error is not None:
            metadata["navigation_state_error"] = navigation_state_error
        if goal_reached is not None:
            metadata["goal_reached"] = goal_reached
        if "navigation_state" not in metadata and navigation_state_error is None:
            metadata["navigation_state"] = self._navigation.get_state().value
        if "goal_reached" not in metadata:
            metadata["goal_reached"] = self._navigation.is_goal_reached()
        return metadata

    def _relative_move_arrival_check(
        self,
        requested: dict[str, float],
        measured_delta: Any,
    ) -> dict[str, Any]:
        if not isinstance(measured_delta, dict):
            return {
                "verified": False,
                "reason": "missing_measured_delta",
                "precision_class": "unknown",
            }

        requested_linear_m = math.hypot(requested["forward_m"], requested["left_m"])
        requested_yaw_signed_deg = self._normalize_degrees(requested["yaw_deg"])
        requested_yaw_deg = abs(requested_yaw_signed_deg)
        measured_linear_m = abs(self._float_metadata(measured_delta.get("linear_m")))
        measured_yaw_signed_deg = self._normalize_degrees(
            self._float_metadata(measured_delta.get("yaw_deg"))
        )
        measured_yaw_deg = abs(measured_yaw_signed_deg)
        linear_error_m = max(0.0, requested_linear_m - measured_linear_m)
        signed_yaw_error_deg = self._normalize_degrees(
            measured_yaw_signed_deg - requested_yaw_signed_deg
        )
        yaw_error_deg = abs(signed_yaw_error_deg) if requested_yaw_deg > 0.0 else 0.0
        linear_tolerance_m = self._relative_move_effective_linear_tolerance_m(requested_linear_m)
        linear_ok = linear_error_m <= linear_tolerance_m
        yaw_ok = yaw_error_deg <= self._relative_move_yaw_tolerance_deg
        yaw_wrong_direction = (
            requested_yaw_deg > self._relative_move_yaw_tolerance_deg
            and measured_yaw_deg > self._relative_move_yaw_tolerance_deg
            and math.copysign(1.0, requested_yaw_signed_deg)
            != math.copysign(1.0, measured_yaw_signed_deg)
        )
        yaw_overshoot = (
            requested_yaw_deg > 0.0
            and not yaw_wrong_direction
            and measured_yaw_deg > requested_yaw_deg + self._relative_move_yaw_tolerance_deg
        )
        progress_ratio = (
            min(measured_linear_m / requested_linear_m, 1.0) if requested_linear_m > 0.0 else None
        )
        precision_class = self._relative_move_precision_class(
            requested_linear_m=requested_linear_m,
            linear_ok=linear_ok,
            yaw_ok=yaw_ok,
            linear_error_m=linear_error_m,
            progress_ratio=progress_ratio,
        )
        reason = self._relative_move_arrival_reason(
            linear_ok=linear_ok,
            yaw_ok=yaw_ok,
            yaw_wrong_direction=yaw_wrong_direction,
            yaw_overshoot=yaw_overshoot,
        )
        return {
            "verified": linear_ok and yaw_ok,
            "reason": reason,
            "precision_class": precision_class,
            "translation_ok": linear_ok,
            "yaw_ok": yaw_ok,
            "requested_linear_m": requested_linear_m,
            "measured_linear_m": measured_linear_m,
            "linear_error_m": linear_error_m,
            "linear_tolerance_m": linear_tolerance_m,
            "configured_linear_tolerance_m": self._relative_move_linear_tolerance_m,
            "linear_progress_ratio": progress_ratio,
            "requested_yaw_deg": requested_yaw_deg,
            "requested_yaw_signed_deg": requested_yaw_signed_deg,
            "measured_yaw_deg": measured_yaw_deg,
            "measured_yaw_signed_deg": measured_yaw_signed_deg,
            "signed_yaw_error_deg": signed_yaw_error_deg,
            "yaw_error_deg": yaw_error_deg,
            "heading_drift_deg": measured_yaw_signed_deg if requested_yaw_deg == 0.0 else 0.0,
            "yaw_tolerance_deg": self._relative_move_yaw_tolerance_deg,
        }

    def _relative_move_arrival_reason(
        self,
        *,
        linear_ok: bool,
        yaw_ok: bool,
        yaw_wrong_direction: bool,
        yaw_overshoot: bool,
    ) -> str:
        if linear_ok and yaw_ok:
            return "within_tolerance"
        if linear_ok and yaw_wrong_direction:
            return "yaw_wrong_direction"
        if linear_ok and yaw_overshoot:
            return "yaw_overshoot"
        if linear_ok and not yaw_ok:
            return "yaw_out_of_tolerance"
        if yaw_ok:
            return "measured_delta_out_of_tolerance"
        if yaw_wrong_direction:
            return "translation_and_yaw_wrong_direction"
        if yaw_overshoot:
            return "translation_and_yaw_overshoot"
        return "translation_and_yaw_out_of_tolerance"

    def _relative_move_effective_linear_tolerance_m(self, requested_linear_m: float) -> float:
        if requested_linear_m <= 0.0:
            return self._relative_move_linear_tolerance_m
        proportional = max(0.05, requested_linear_m * 0.30)
        return min(self._relative_move_linear_tolerance_m, proportional)

    def _relative_move_precision_class(
        self,
        *,
        requested_linear_m: float,
        linear_ok: bool,
        yaw_ok: bool,
        linear_error_m: float,
        progress_ratio: float | None,
    ) -> str:
        if requested_linear_m <= 0.0:
            return "within_tolerance" if yaw_ok else "partial"
        if linear_ok and yaw_ok:
            return "exact" if linear_error_m <= 0.05 else "within_tolerance"
        if progress_ratio is None:
            return "unknown"
        if progress_ratio < 0.25:
            return "short_stop"
        return "partial"

    def _annotate_relative_move_contract(
        self,
        metadata: dict[str, Any],
        *,
        requested: dict[str, float],
        planner_arrived: bool,
        odometry_confirmed: bool,
        precision_class: str | None = None,
    ) -> None:
        measured_delta = metadata.get("measured_delta")
        requested_displacement_m = math.hypot(requested["forward_m"], requested["left_m"])
        measured_displacement_m = (
            abs(self._float_metadata(measured_delta.get("linear_m")))
            if isinstance(measured_delta, dict)
            else None
        )
        heading_drift_deg = (
            self._float_metadata(measured_delta.get("yaw_deg"))
            if isinstance(measured_delta, dict)
            else None
        )
        residual_distance_m = (
            max(0.0, requested_displacement_m - measured_displacement_m)
            if measured_displacement_m is not None
            else None
        )
        inferred_precision = precision_class
        if inferred_precision is None:
            if measured_delta is None:
                inferred_precision = "unknown"
            elif metadata.get("status") in {"NO_PATH", "NO_SAFE_GOAL", "OBSTACLE", "LEASE_DENIED"}:
                inferred_precision = "blocked_or_protected"
            elif metadata.get("status") in {"CANCELLED", "TIMEOUT"}:
                inferred_precision = (
                    "partial" if (measured_displacement_m or 0.0) > 0.0 else "short_stop"
                )
            else:
                inferred_precision = "unknown"
        metadata.update(
            {
                "requested_displacement_m": requested_displacement_m,
                "measured_displacement_m": measured_displacement_m,
                "residual_distance_m": residual_distance_m,
                "heading_drift_deg": heading_drift_deg,
                "planner_arrived": planner_arrived,
                "odometry_confirmed": odometry_confirmed,
                "precision_class": inferred_precision,
            }
        )
        self._annotate_relative_move_agent_guidance(metadata)

    def _annotate_relative_move_agent_guidance(self, metadata: dict[str, Any]) -> None:
        interpretation, next_action = self._relative_move_agent_guidance(metadata)
        metadata["agent_interpretation"] = interpretation
        metadata["recommended_next_action"] = next_action

    def _relative_move_result_message(self, metadata: dict[str, Any], *, default: str) -> str:
        interpretation = metadata.get("agent_interpretation")
        if isinstance(interpretation, str) and interpretation:
            return interpretation
        return default

    def _relative_move_agent_guidance(self, metadata: dict[str, Any]) -> tuple[str, str]:
        status = str(metadata.get("status") or "")
        precision_class = str(metadata.get("precision_class") or "unknown")
        arrival_check = metadata.get("arrival_check")
        reason = (
            str(arrival_check.get("reason") or "")
            if isinstance(arrival_check, dict)
            else status.lower()
        )

        if status == "ARRIVED" and precision_class == "exact":
            return (
                "Relative move reached the requested target within odometry tolerance.",
                "Continue the task; do not re-diagnose this move unless the scene changed.",
            )
        if status == "ARRIVED" and precision_class == "within_tolerance":
            return (
                "Relative move is close enough for the requested task; minor physical error is expected on Go2.",
                "Continue the task or observe only if the next decision depends on precise placement.",
            )
        if precision_class == "partial":
            if reason.startswith("yaw_") or "_yaw_" in reason:
                return (
                    "Relative move made useful translation progress, but the final heading was not exact; this is expected physical feedback, not a system failure.",
                    "If heading matters, correct with rotate_in_place or observe once before the next bounded move; otherwise continue.",
                )
            return (
                "Relative move made useful partial progress but did not exactly satisfy the request; this can happen with Go2 local planning or protection.",
                "If closer approach still matters, observe once or try a shorter bounded move; otherwise accept the progress and continue.",
            )
        if precision_class == "short_stop":
            return (
                "Relative move barely progressed, which usually means protection, local planning, or an obstacle limited the command.",
                "Do not repeat blindly; observe, rotate for a clearer angle, or choose a shorter safer move.",
            )
        if precision_class == "blocked_or_protected" or status in {
            "NO_PATH",
            "NO_SAFE_GOAL",
            "OBSTACLE",
            "LEASE_DENIED",
        }:
            return (
                "Relative move was blocked or protection-limited before useful progress.",
                "Stop if still moving, observe the scene, then choose a safer shorter motion or different angle.",
            )
        if status == "TIMEOUT":
            return (
                "Relative move timed out before confirmed completion; this is a runtime budget issue, not proof that the task is impossible.",
                "Check current pose/situation before deciding whether to continue, retry shorter, or stop.",
            )
        if status == "NO_ODOM":
            return (
                "Relative move cannot be verified because odometry is unavailable.",
                "Do not issue more blind movement; restore pose tracking or ask the operator for a safety check.",
            )
        if status == "NOT_CONFIGURED":
            return (
                "Relative move is unavailable because the navigation planner is disabled or not configured.",
                "Use rotate_in_place plus observe/odometry, or restart with navigation enabled for planned relative motion.",
            )
        if status in {"CANCELLED", "ARRIVAL_NOT_CONFIRMED"}:
            return (
                "Relative move did not reach a confirmed final state.",
                "Inspect compact situation or observe before retrying with a smaller or safer command.",
            )
        return (
            "Relative move result is uncertain.",
            "Use compact situation or raw diagnostics before issuing another movement.",
        )

    def _float_metadata(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _normalize_degrees(self, value: float) -> float:
        normalized = (float(value) + 180.0) % 360.0 - 180.0
        if normalized == -180.0:
            return 180.0
        return normalized

    def _navigation_failure_or(self, fallback: NavigationError) -> NavigationError:
        get_last_failure = getattr(self._navigation, "get_last_failure", None)
        reason = get_last_failure() if callable(get_last_failure) else None
        return self._navigation_failure_code(reason, fallback)

    def _navigation_failure_code(
        self, reason: Any, fallback: NavigationError
    ) -> NavigationError:
        if reason in (
            "NO_ODOM",
            "NO_SAFE_GOAL",
            "NO_PATH",
            "TARGET_NOT_FOUND",
            "VISION_UNAVAILABLE",
            "ARRIVAL_NOT_CONFIRMED",
            "CANCELLED",
            "OBSTACLE",
            "TIMEOUT",
            "LEASE_DENIED",
            "RETRY_LIMIT_EXCEEDED",
        ):
            return reason  # type: ignore[return-value]
        return fallback

    def _current_base_pose(self) -> PoseStamped | None:
        tf = self.tf.get("world", "base_link")
        return tf.to_pose() if tf is not None else None

    def _current_direct_motion_pose(self) -> PoseStamped | None:
        received_at_s = self._latest_sport_odom_received_at_s
        if received_at_s is not None and (
            time.monotonic() - received_at_s <= self._direct_motion_feedback_timeout_s
        ):
            return self._latest_sport_odom
        # Sport-mode odometry (rt/lf/sportmodestate) is sparse/throttled on
        # compute-limited edge hosts (Jetson: ~3Hz with >0.5s gaps), so its strict
        # freshness window is often empty and drive would hard-fail NO_ODOM. Fall
        # back to the TF base pose (world->base_link) used before the sport-odom
        # closure change (97f4799) so drive stays closed-loop and usable.
        return self._current_base_pose()

    def _rotation_measurement_metadata(
        self,
        start_pose: PoseStamped,
        end_pose: PoseStamped,
        cumulative_yaw_rad: float,
    ) -> dict[str, object]:
        return {
            "end": pose_metadata(end_pose),
            "measured_delta": self._direct_motion_delta_metadata(start_pose, end_pose),
            "measured_cumulative_yaw_rad": cumulative_yaw_rad,
            "measured_cumulative_yaw_deg": math.degrees(cumulative_yaw_rad),
        }

    @skill
    def wait(self, seconds: float) -> str:
        """Wait for a specified amount of time.

        Args:
            seconds: Seconds to wait
        """
        time.sleep(seconds)
        return f"Wait completed with length={seconds}s"

    @skill
    def current_time(self) -> str:
        """Provides current time."""
        return str(datetime.datetime.now())

    @skill
    def go2_robot_status(self) -> SkillResult[NavigationSkillError]:
        """Get the latest read-only Unitree Go2 status known to DimOS.

        The payload includes low-state battery/motor summaries when available.
        Unitree firmware protection fields remain explicit as unknown until a
        live SDK signal is connected.
        """

        get_robot_status = getattr(self._connection, "get_robot_status", None)
        if not callable(get_robot_status):
            return SkillResult.fail(
                "NOT_CONFIGURED",
                "The Go2 connection does not expose read-only robot status.",
                robot_status_available=False,
            )
        try:
            status = get_robot_status()
        except Exception as e:
            return SkillResult.fail(
                "EXECUTION_FAILED",
                "Failed to read Go2 robot status from the connection.",
                error=f"{e.__class__.__name__}: {e}",
            )
        self._record_robot_status(status)
        return SkillResult.ok(
            "Go2 robot status is available.",
            robot_status=status if isinstance(status, dict) else {"value": status},
        )

    def _record_robot_status(self, status: Any) -> None:
        record_robot_status = getattr(self._navigation, "record_robot_status", None)
        if not callable(record_robot_status) or not isinstance(status, dict):
            return
        try:
            record_robot_status(status)
        except Exception:
            logger.exception("Failed to record Go2 robot status for diagnostics.")

    @skill
    def motion_safety_status(self) -> SkillResult[NavigationSkillError]:
        """Get current motion safety status and blocker classification."""

        diagnostics, failure = self._motion_diagnostics_payload()
        if failure is not None:
            return failure
        assert diagnostics is not None
        return SkillResult.ok(
            self._motion_diagnostics_message("Motion safety status", diagnostics),
            diagnostics=diagnostics,
        )

    @skill
    def go2_situation(self) -> SkillResult[NavigationSkillError]:
        """Get a compact Go2 motion situation summary for action selection."""

        diagnostics, failure = self._motion_diagnostics_payload()
        if failure is not None:
            return failure
        assert diagnostics is not None
        situation = self._compact_motion_situation(diagnostics)
        return SkillResult.ok(
            self._go2_situation_message(situation),
            situation=situation,
        )

    @skill
    def last_motion_diagnostics(self) -> SkillResult[NavigationSkillError]:
        """Get the latest motion command diagnostics and recommended next action."""

        diagnostics, failure = self._motion_diagnostics_payload()
        if failure is not None:
            return failure
        assert diagnostics is not None
        return SkillResult.ok(
            self._motion_diagnostics_message("Latest motion diagnostics", diagnostics),
            diagnostics=diagnostics,
        )

    @skill
    def robot_protection_status(self) -> SkillResult[NavigationSkillError]:
        """Get robot protection status known to DimOS."""

        diagnostics, failure = self._motion_diagnostics_payload()
        if failure is not None:
            return failure
        assert diagnostics is not None
        return SkillResult.ok(
            self._robot_protection_message(diagnostics),
            robot_protection_status={
                "posture_mode": diagnostics.get("posture_mode"),
                "collision_obstacle_protection": diagnostics.get("collision_obstacle_protection"),
                "unitree_protection": diagnostics.get("unitree_protection"),
                "battery_status": diagnostics.get("battery_status"),
                "motor_status": diagnostics.get("motor_status"),
                "blocker": diagnostics.get("blocker"),
                "recommended_next_action": diagnostics.get("recommended_next_action"),
            },
            diagnostics=diagnostics,
        )

    def _motion_diagnostics_payload(
        self,
    ) -> tuple[dict[str, Any] | None, SkillResult[NavigationSkillError] | None]:
        backend, failure = self._backend_motion_diagnostics()
        if failure is not None:
            return None, failure
        assert backend is not None
        runtime_policy = self._runtime_policy_payload()
        runtime_allowed = None
        scope_allowed = None
        if runtime_policy is not None:
            runtime_allowed = runtime_policy.get("movement_mode") != "deny"
            scope_allowed = runtime_policy.get("tool_scope") != "readonly"
        robot_status = backend.get("robot_status")
        diagnostics = build_motion_diagnostics(
            navigation_state=backend.get("navigation_state") or self._safe_navigation_state(),
            last_failure=backend.get("last_failure") or self._safe_last_navigation_failure(),
            last_stop_reason=backend.get("last_stop_reason"),
            last_stop_time=backend.get("last_stop_time"),
            obstacle_avoidance_config=self._obstacle_avoidance_config(),
            measured_result=backend.get("measured_result"),
            battery_status=self._battery_status_from_backend(backend),
            motor_status=self._motor_status_from_backend(backend),
            runtime_allowed=runtime_allowed,
            scope_allowed=scope_allowed,
            robot_status=robot_status if isinstance(robot_status, dict) else None,
        )
        diagnostics["runtime_policy"] = runtime_policy
        diagnostics["backend"] = backend
        return diagnostics, None

    def _backend_motion_diagnostics(
        self,
    ) -> tuple[dict[str, Any] | None, SkillResult[NavigationSkillError] | None]:
        get_motion_diagnostics = getattr(self._navigation, "get_motion_diagnostics", None)
        if not callable(get_motion_diagnostics):
            return None, SkillResult.fail(
                "NOT_CONFIGURED",
                "The navigation backend does not expose motion diagnostics.",
                diagnostics_available=False,
            )
        try:
            payload = get_motion_diagnostics()
        except Exception as e:
            logger.exception("Failed to read navigation motion diagnostics.")
            return None, SkillResult.fail(
                "EXECUTION_FAILED",
                "Failed to read motion diagnostics from the navigation backend.",
                error=f"{e.__class__.__name__}: {e}",
            )
        if not isinstance(payload, dict):
            return None, SkillResult.fail(
                "EXECUTION_FAILED",
                "Navigation backend returned an invalid motion diagnostics payload.",
                diagnostics_type=type(payload).__name__,
            )
        return payload, None

    def _runtime_policy_payload(self) -> dict[str, Any] | None:
        try:
            config = self._runtime_config()
            return resolve_agent_runtime_policy(config).model_dump() if config is not None else None
        except Exception:
            logger.exception("Failed to resolve runtime policy for Go2 status tools.")
            return None

    def _runtime_config(self) -> Any | None:
        config = getattr(self, "config", None)
        return getattr(config, "g", None)

    def _obstacle_avoidance_config(self) -> bool | None:
        config = self._runtime_config()
        return getattr(config, "obstacle_avoidance", None) if config is not None else None

    def _battery_status_from_backend(self, backend: dict[str, Any]) -> Any:
        robot_status = backend.get("robot_status")
        if isinstance(robot_status, dict) and "battery" in robot_status:
            return robot_status.get("battery")
        return "unknown"

    def _motor_status_from_backend(self, backend: dict[str, Any]) -> Any:
        robot_status = backend.get("robot_status")
        if isinstance(robot_status, dict) and "motors" in robot_status:
            return robot_status.get("motors")
        return "unknown"

    def _safe_navigation_state(self) -> str | None:
        try:
            return self._navigation.get_state().value
        except Exception:
            return None

    def _safe_last_navigation_failure(self) -> str | None:
        get_last_failure = getattr(self._navigation, "get_last_failure", None)
        if not callable(get_last_failure):
            return None
        try:
            failure = get_last_failure()
        except Exception:
            logger.exception("Failed to read last navigation failure.")
            return None
        return str(failure) if failure else None

    def _motion_diagnostics_message(self, prefix: str, diagnostics: dict[str, Any]) -> str:
        blocker = str(diagnostics.get("blocker") or "unknown")
        action = str(diagnostics.get("recommended_next_action") or "").strip()
        message = f"{prefix}: blocker={blocker}."
        if action:
            message += f" Recommended next action: {action}."
        return message

    def _robot_protection_message(self, diagnostics: dict[str, Any]) -> str:
        collision = _diagnostic_state(
            diagnostics.get("collision_obstacle_protection"),
            default="unknown",
        )
        unitree = _diagnostic_state(diagnostics.get("unitree_protection"), default="unknown")
        battery = _diagnostic_state(diagnostics.get("battery_status"), default="unknown")
        motor = _diagnostic_state(diagnostics.get("motor_status"), default="unknown")
        return (
            self._motion_diagnostics_message("Robot protection status", diagnostics)
            + f" Protection states: local_obstacle={collision}, unitree={unitree}, "
            + f"battery={battery}, motor={motor}."
        )

    def _compact_motion_situation(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        measured = diagnostics.get("measured_result")
        measured_payload = measured if isinstance(measured, dict) else {}
        runtime_policy = diagnostics.get("runtime_policy")
        return {
            "blocker": diagnostics.get("blocker") or "unknown",
            "recommended_next_action": diagnostics.get("recommended_next_action") or "inspect",
            "navigation_state": diagnostics.get("navigation_state"),
            "battery": _compact_state_payload(
                diagnostics.get("battery_status"),
                keys=("state", "soc_percent", "severity", "source"),
            ),
            "unitree_protection": _compact_state_payload(
                diagnostics.get("unitree_protection"),
                keys=("state", "available", "needs_live_sdk_probe"),
            ),
            "collision_obstacle_protection": _compact_state_payload(
                diagnostics.get("collision_obstacle_protection"),
                keys=("state", "enabled", "source", "last_trigger_time"),
            ),
            "motor": _compact_state_payload(
                diagnostics.get("motor_status"),
                keys=("state", "lost_count", "max_temperature_c", "severity", "source"),
            ),
            "authority": _compact_authority_payload(
                runtime_policy if isinstance(runtime_policy, dict) else None
            ),
            "last_motion": {
                "tool_name": measured_payload.get("tool_name"),
                "status": measured_payload.get("status"),
                "success": measured_payload.get("success"),
                "precision_class": measured_payload.get("precision_class"),
                "physically_satisfied": measured_payload.get("physically_satisfied"),
                "residual_distance_m": measured_payload.get("residual_distance_m"),
                "heading_drift_deg": measured_payload.get("heading_drift_deg"),
                "measured_delta": measured_payload.get("measured_delta"),
            },
        }

    def _go2_situation_message(self, situation: dict[str, Any]) -> str:
        blocker = str(situation.get("blocker") or "unknown")
        action = str(situation.get("recommended_next_action") or "inspect")
        battery = situation.get("battery") if isinstance(situation.get("battery"), dict) else {}
        soc = battery.get("soc_percent") if isinstance(battery, dict) else None
        last_motion = (
            situation.get("last_motion") if isinstance(situation.get("last_motion"), dict) else {}
        )
        precision = last_motion.get("precision_class") if isinstance(last_motion, dict) else None
        parts = [f"Go2 situation: blocker={blocker}"]
        if soc is not None:
            parts.append(f"battery={soc}%")
        if precision:
            parts.append(f"last_motion_precision={precision}")
        parts.append(f"next={action}")
        return "; ".join(parts) + "."

    @skill(uses=[CAP_MOVEMENT])
    def execute_sport_command(self, command_name: str) -> SkillResult[NavigationSkillError]:
        started_at_s = time.monotonic()
        if command_name not in _UNITREE_COMMANDS:
            suggestions = difflib.get_close_matches(
                command_name, _UNITREE_COMMANDS.keys(), n=3, cutoff=0.6
            )
            result = SkillResult.fail(
                "INVALID_INPUT",
                f"Unknown Unitree sport command: '{command_name}'.",
                command_name=command_name,
                suggestions=suggestions,
            )
            return self._trace_movement_skill_result(
                "execute_sport_command",
                result,
                movement_kind="sport_command",
                started_at_s=started_at_s,
            )

        id_, _ = _UNITREE_COMMANDS[command_name]
        return self._execute_known_sport_command(
            command_name,
            id_,
            tool_name="execute_sport_command",
            started_at_s=started_at_s,
        )

    @skill(uses=[CAP_MOVEMENT])
    def full_lie_down(self) -> SkillResult[NavigationSkillError]:
        """Lower the Go2 fully to the ground using the Unitree Damp command.

        Use this when the operator asks the robot to completely lie down or power
        down posture. This is different from `Sit` and from `StandDown`, which
        may fail or stop at a higher posture on Go2 Air.
        """

        started_at_s = time.monotonic()
        id_, _ = _UNITREE_COMMANDS["Damp"]
        return self._execute_known_sport_command(
            "Damp",
            id_,
            tool_name="full_lie_down",
            started_at_s=started_at_s,
            success_message="Go2 full lie-down command accepted.",
        )

    @skill(uses=[CAP_MOVEMENT])
    def lie_down(self) -> SkillResult[NavigationSkillError]:
        """Alias for `full_lie_down`; lower the Go2 fully to the ground."""

        started_at_s = time.monotonic()
        id_, _ = _UNITREE_COMMANDS["Damp"]
        return self._execute_known_sport_command(
            "Damp",
            id_,
            tool_name="lie_down",
            started_at_s=started_at_s,
            success_message="Go2 lie-down command accepted.",
        )

    @skill(uses=[CAP_MOVEMENT])
    def stand_up(self) -> SkillResult[NavigationSkillError]:
        """Raise the Go2 from prone/sitting into a standing posture.

        This narrow startup posture tool wraps the Unitree `StandUp` command so
        agents do not need broad `execute_sport_command` access just to begin a
        live test. Follow with `balance_stand` before local motion when the
        robot has just powered on or recovered from lying down.
        """

        started_at_s = time.monotonic()
        id_, _ = _UNITREE_COMMANDS["StandUp"]
        return self._execute_known_sport_command(
            "StandUp",
            id_,
            tool_name="stand_up",
            started_at_s=started_at_s,
            success_message="Go2 stand-up command accepted.",
        )

    @skill(uses=[CAP_MOVEMENT])
    def balance_stand(self) -> SkillResult[NavigationSkillError]:
        """Put the Go2 into balanced standing mode before local motion.

        Use this after `stand_up` and before navigation/local movement. It wraps
        only the Unitree `BalanceStand` command and is intentionally narrower
        than broad sport-command access.
        """

        started_at_s = time.monotonic()
        id_, _ = _UNITREE_COMMANDS["BalanceStand"]
        return self._execute_known_sport_command(
            "BalanceStand",
            id_,
            tool_name="balance_stand",
            started_at_s=started_at_s,
            success_message="Go2 balance-stand command accepted.",
        )

    def _execute_known_sport_command(
        self,
        command_name: str,
        api_id: int,
        *,
        tool_name: str,
        started_at_s: float,
        success_message: str | None = None,
    ) -> SkillResult[NavigationSkillError]:
        try:
            response = self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": api_id})
            metadata = {
                "command_name": command_name,
                "api_id": api_id,
                "command_response": response,
            }
            if _unitree_publish_request_timed_out(response):
                result = SkillResult.fail(
                    "TIMEOUT",
                    (
                        f"Unitree sport command '{command_name}' did not return before the "
                        "WebRTC request timeout. The command may or may not have been accepted; "
                        "check observe/status before retrying."
                    ),
                    **metadata,
                )
                return self._trace_movement_skill_result(
                    tool_name,
                    result,
                    movement_kind="sport_command",
                    started_at_s=started_at_s,
                )
            if _unitree_publish_request_failed(response):
                reason = str(response.get("error") or "publish_request_failed")
                exception = response.get("exception")
                detail = f": {exception}" if exception else ""
                result = SkillResult.fail(
                    "EXECUTION_FAILED",
                    f"Unitree sport command '{command_name}' did not complete ({reason}{detail}).",
                    **metadata,
                )
                return self._trace_movement_skill_result(
                    tool_name,
                    result,
                    movement_kind="sport_command",
                    started_at_s=started_at_s,
                )
            status_code = _unitree_status_code(response)
            if status_code is not None:
                metadata["unitree_status_code"] = status_code
            if status_code not in (None, 0):
                result = SkillResult.fail(
                    "EXECUTION_FAILED",
                    (
                        f"Unitree rejected sport command '{command_name}' "
                        f"with status.code={status_code}."
                    ),
                    **metadata,
                )
                return self._trace_movement_skill_result(
                    tool_name,
                    result,
                    movement_kind="sport_command",
                    started_at_s=started_at_s,
                )
            result = SkillResult.ok(
                success_message or f"Executed Unitree sport command '{command_name}'.",
                **metadata,
            )
            return self._trace_movement_skill_result(
                tool_name,
                result,
                movement_kind="sport_command",
                started_at_s=started_at_s,
            )
        except Exception as e:
            logger.error(f"Failed to execute {command_name}: {e}")
            result = SkillResult.fail(
                "EXECUTION_FAILED",
                f"Failed to execute Unitree sport command '{command_name}'.",
                command_name=command_name,
                api_id=api_id,
                error=f"{e.__class__.__name__}: {e}",
            )
            return self._trace_movement_skill_result(
                tool_name,
                result,
                movement_kind="sport_command",
                started_at_s=started_at_s,
            )


_commands = "\n".join(
    [f'- "{name}": {description}' for name, (_, description) in _UNITREE_COMMANDS.items()]
)

UnitreeSkillContainer.execute_sport_command.__doc__ = f"""Execute a Unitree sport command.

Example usage:

    execute_sport_command("FrontPounce")

Here are all the command names and what they do.

{_commands}
"""
