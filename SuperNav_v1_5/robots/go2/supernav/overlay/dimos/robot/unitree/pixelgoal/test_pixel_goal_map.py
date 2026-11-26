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

"""pixel_goal_map pure-logic tests: pinhole/mount geometry, rays, ROI, surfaces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.pixelgoal.pixel_goal_map import (
    DEFAULT_MOUNT_YAW_DEG,
    DEFAULT_USB_HFOV_DEG,
    MountExtrinsics,
    PinholeModel,
    PixelGoalMapConfig,
    SurroundCameraRig,
    contact_goal,
    directed_waypoint,
    estimate_floor_height,
    floor_intersection_limit,
    front_surface_point,
    pinhole_camera_info,
    pixel_ray,
    projection_bbox,
    projection_diagnostics,
    projection_search_radii,
    rectify_image,
    select_surface_candidate,
    world_to_optical_from_odom,
)


def _camera_info() -> CameraInfo:
    return CameraInfo.from_intrinsics(100.0, 100.0, 50.0, 50.0, 100, 100, "camera_optical")


# ---------------------------------------------------------------------- pinhole model
def test_pinhole_from_hfov_centers_principal_point() -> None:
    model = PinholeModel.from_hfov(1600, 1200, DEFAULT_USB_HFOV_DEG)
    assert model.fx == pytest.approx(
        (1600 - 1) / 2 / math.tan(math.radians(DEFAULT_USB_HFOV_DEG) / 2)
    )
    assert model.fy == model.fx
    assert (model.cx, model.cy) == pytest.approx(((1600 - 1) / 2, (1200 - 1) / 2))
    info = model.to_camera_info("go2/front/camera_optical")
    assert (info.width, info.height) == (1600, 1200)
    assert info.frame_id == "go2/front/camera_optical"
    # center pixel back-projects onto the optical axis, edge pixel at half-HFOV
    origin, center_ray = pixel_ray(799, 599, info, Transform.identity())
    assert origin == pytest.approx((0.0, 0.0, 0.0))
    assert center_ray == pytest.approx((0.0, 0.0, 1.0), abs=1e-3)
    _, edge_ray = pixel_ray(0, 599, info, Transform.identity())
    assert math.degrees(math.atan2(-edge_ray[0], edge_ray[2])) == pytest.approx(
        DEFAULT_USB_HFOV_DEG / 2, abs=0.05
    )


# ---------------------------------------------------------------------- mount extrinsics
@pytest.mark.parametrize(
    ("yaw_deg", "forward", "right"),
    [
        (0.0, (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        (90.0, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        (-90.0, (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
        (180.0, (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ],
)
def test_mount_rotation_axes(yaw_deg: float, forward: tuple, right: tuple) -> None:
    matrix = MountExtrinsics(yaw_deg=yaw_deg).rotation_matrix()
    assert matrix[:, 2] == pytest.approx(forward)
    assert matrix[:, 0] == pytest.approx(right)
    assert matrix[:, 1] == pytest.approx((0.0, 0.0, -1.0))  # image down = base down
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_mount_pitch_tilts_forward_axis_down() -> None:
    matrix = MountExtrinsics(yaw_deg=0.0, pitch_deg=30.0).rotation_matrix()
    assert matrix[:, 2] == pytest.approx((math.cos(math.radians(30)), 0.0, -0.5))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_default_mount_yaws_match_direction_yaw_convention() -> None:
    assert DEFAULT_MOUNT_YAW_DEG == {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0}
    rig = SurroundCameraRig()
    assert rig.mount("right").yaw_deg == -90.0
    assert rig.mount("unknown-direction").yaw_deg == 0.0  # safe default


def test_world_to_optical_composes_odom_and_mount() -> None:
    odom = PoseStamped(position=[0.0, 0.0, 0.3], frame_id="world")
    world_to_optical = world_to_optical_from_odom(
        odom, MountExtrinsics(yaw_deg=0.0, height_m=0.6), "go2/front/camera_optical"
    )
    assert world_to_optical.frame_id == "go2/front/camera_optical"
    assert world_to_optical.child_frame_id == "world"
    camera_origin = world_to_optical.inverse().translation.to_numpy()
    assert camera_origin == pytest.approx((0.0, 0.0, 0.6))
    # a point 1 m ahead at camera height lands on the optical axis, z=1
    point = world_to_optical.to_matrix() @ np.array([1.0, 0.0, 0.6, 1.0])
    assert point[:3] == pytest.approx((0.0, 0.0, 1.0))


def test_world_to_optical_for_side_camera_and_moved_base() -> None:
    odom = PoseStamped(position=[1.0, 2.0, 0.3], frame_id="world")
    left = world_to_optical_from_odom(
        odom, MountExtrinsics(yaw_deg=90.0, height_m=0.6), "go2/left/camera_optical"
    )
    # camera at (1, 2, 0.6) looking along +y: a point 1 m left of the base is centered
    point = left.to_matrix() @ np.array([1.0, 3.0, 0.6, 1.0])
    assert point[:3] == pytest.approx((0.0, 0.0, 1.0))
    right = world_to_optical_from_odom(
        odom, MountExtrinsics(yaw_deg=-90.0, height_m=0.6), "go2/right/camera_optical"
    )
    point = right.to_matrix() @ np.array([1.0, 1.0, 0.6, 1.0])
    assert point[:3] == pytest.approx((0.0, 0.0, 1.0))


def test_rig_from_env_reads_per_direction_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIMOS_PIXELGOAL_MAP_HFOV_DEG", "90.0")
    monkeypatch.setenv("DIMOS_PIXELGOAL_MAP_MOUNT_RIGHT_YAW_DEG", "-87.5")
    monkeypatch.setenv("DIMOS_PIXELGOAL_MAP_MOUNT_FRONT_HEIGHT_M", "0.55")
    monkeypatch.setenv("DIMOS_PIXELGOAL_MAP_BASE_HEIGHT_M", "0.28")
    monkeypatch.setenv("DIMOS_PIXELGOAL_MAP_MOUNT_BACK_PITCH_DEG", "10.0")
    rig = SurroundCameraRig.from_env()
    assert rig.hfov_deg == 90.0
    assert rig.mount("right").yaw_deg == -87.5
    assert rig.mount("front").height_m == 0.55
    assert rig.base_height_m == 0.28
    assert rig.mount("back").pitch_deg == 10.0
    assert rig.mount("left").yaw_deg == 90.0  # untouched default


def test_rig_from_env_scales_calibrated_intrinsics(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "FX": "611",
        "FY": "614",
        "CX": "640",
        "CY": "360",
        "CALIBRATION_WIDTH": "1280",
        "CALIBRATION_HEIGHT": "720",
    }.items():
        monkeypatch.setenv(f"DIMOS_PIXELGOAL_MAP_{name}", value)

    model = SurroundCameraRig.from_env().pinhole(640, 360)
    assert (model.fx, model.fy, model.cx, model.cy) == pytest.approx((305.5, 307, 320, 180))
    assert model.to_camera_info().D == [0.0] * 5


def test_config_from_env_defaults() -> None:
    config = PixelGoalMapConfig.from_env()
    assert config.radius_px == 8
    assert config.robot_front_offset_m == pytest.approx(0.35)
    assert config.rig.hfov_deg == pytest.approx(76.3)
    assert config.rig.base_height_m == pytest.approx(0.3)


# ---------------------------------------------------------------------- rays / ROI
def test_projection_bbox_and_front_surface() -> None:
    info = _camera_info()
    assert projection_bbox(50, 50, 4, info, pinhole_camera_info(info)) == pytest.approx(
        (46, 46, 54, 54)
    )
    points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [-0.1, 0.0, 2.0], [0.0, 0.0, 5.0]])
    target = front_surface_point(points, np.zeros(3))
    assert target == pytest.approx((0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="fewer than three"):
        front_surface_point(points[:2], np.zeros(3))


def test_projection_radii_preserve_stable_branch_angular_tolerance() -> None:
    stable = CameraInfo.from_intrinsics(548.4, 548.4, 640, 360, 1280, 720)
    usb = PinholeModel.from_hfov(1600, 1200, DEFAULT_USB_HFOV_DEG).to_camera_info()
    assert projection_search_radii(stable, 8) == (4, 8, 12, 16)
    assert projection_search_radii(usb, 8) == (7, 15, 22, 30)


def test_pixel_ray_identity_and_usb_mount() -> None:
    info = _camera_info()
    ray_origin, ray = pixel_ray(50, 50, info, Transform.identity())
    assert ray_origin == pytest.approx((0.0, 0.0, 0.0))
    assert ray == pytest.approx((0.0, 0.0, 1.0))

    # USB front camera on identity odom: center ray is horizontal, straight ahead
    rig = SurroundCameraRig()
    model_info = rig.pinhole(1600, 1200).to_camera_info("go2/front/camera_optical")
    odom = PoseStamped(position=[0.0, 0.0, 0.3], frame_id="world")
    world_to_optical = world_to_optical_from_odom(
        odom, rig.mount("front"), "go2/front/camera_optical"
    )
    ray_origin, ray = pixel_ray(799, 599, model_info, world_to_optical)
    assert ray_origin == pytest.approx((0.0, 0.0, 0.6))
    assert ray == pytest.approx((1.0, 0.0, 0.0), abs=1e-3)
    # a click left of center bears toward base +y (left), with HFOV-scale angle
    _, left_ray = pixel_ray(0, 599, model_info, world_to_optical)
    assert left_ray[1] > 0.5
    # the right camera's center ray points along base -y; no pre-turn needed
    right_info = rig.pinhole(1600, 1200).to_camera_info("go2/right/camera_optical")
    right_transform = world_to_optical_from_odom(
        odom, rig.mount("right"), "go2/right/camera_optical"
    )
    _, right_ray = pixel_ray(799, 599, right_info, right_transform)
    assert right_ray == pytest.approx((0.0, -1.0, 0.0), abs=1e-3)


def test_floor_helpers() -> None:
    floor = np.column_stack((np.linspace(-2.0, 2.0, 80), np.zeros(80), np.full(80, -0.075)))
    low_outliers = np.column_stack((np.linspace(-1.0, 1.0, 20), np.zeros(20), np.full(20, -0.225)))
    floor_z = estimate_floor_height(np.vstack((floor, low_outliers)), np.array([0.0, 0.0, 0.3]))
    assert floor_z == pytest.approx(-0.075)
    # empty/degenerate clouds fall back to camera height minus a body offset
    assert estimate_floor_height(np.empty((0, 3)), np.array([0.0, 0.0, 0.6])) == pytest.approx(0.3)

    assert floor_intersection_limit(
        np.array([0.0, 0.0, 0.5]),
        np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5)]),
        floor_z=0.0,
    ) == pytest.approx(0.5)
    assert floor_intersection_limit(np.zeros(3), np.array([1.0, 0.0, 0.0]), floor_z=-0.5) is None


def test_contact_goal_and_projection_diagnostics() -> None:
    assert contact_goal(np.array([2.0, 0.0, 0.4]), np.zeros(3), offset_m=0.3) == pytest.approx(
        (1.7, 0.0, 0.0)
    )
    # closer than the offset: the stop point collapses onto the ray origin side
    near = contact_goal(np.array([0.2, 0.0, 0.0]), np.zeros(3), offset_m=0.35)
    assert near == pytest.approx((0.0, 0.0, 0.0))
    diagnostics = projection_diagnostics(
        np.array([[2.0, 0.0, 0.0], [3.0, 0.1, 0.0], [4.0, -0.1, 0.0]]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([3.0, 0.1, 0.0]),
    )
    assert diagnostics["target_ray_depth_m"] == pytest.approx(3.0)
    assert diagnostics["target_ray_miss_m"] == pytest.approx(0.1)


def test_select_surface_candidate() -> None:
    background = {"radius_px": 4, "ray_depth_m": 3.8, "ray_miss_m": 0.02}
    foreground = {"radius_px": 8, "ray_depth_m": 2.1, "ray_miss_m": 0.04}
    assert select_surface_candidate([background, foreground]) is foreground
    tight = {"radius_px": 4, "ray_depth_m": 3.8, "ray_miss_m": 0.02}
    wider = {"radius_px": 8, "ray_depth_m": 3.7, "ray_miss_m": 0.04}
    assert select_surface_candidate([tight, wider]) is tight
    with pytest.raises(ValueError, match="no pixel surface candidates"):
        select_surface_candidate([])


# ---------------------------------------------------------------------- fisheye path
def test_fisheye_rectify_roundtrip() -> None:
    from dimos.robot.unitree.go2.connection import GO2Connection

    fisheye = GO2Connection.camera_info_static
    pinhole = pinhole_camera_info(fisheye)
    assert (pinhole.width, pinhole.height) == (fisheye.width, fisheye.height)
    bbox = projection_bbox(100, 100, 24, fisheye, pinhole)
    assert bbox == pytest.approx((76, 76, 124, 124))
    image = Image.from_numpy(
        np.random.default_rng(0).integers(
            0, 255, (fisheye.height, fisheye.width, 3), dtype=np.uint8
        ),
        ts=10.0,
    )
    rectified = rectify_image(image, fisheye, pinhole)
    assert rectified.shape == image.shape
    assert rectified.ts == image.ts
    assert not np.array_equal(rectified.data, image.data)


def test_rectify_image_passes_through_zero_distortion() -> None:
    info = _camera_info()  # plumb_bob with zero D
    image = Image.from_numpy(np.full((100, 100, 3), 77, dtype=np.uint8), ts=3.0)
    rectified = rectify_image(image, info, info)
    assert rectified.ts == image.ts
    assert np.array_equal(rectified.data, image.data)


# ---------------------------------------------------------------------- exploration waypoints
def test_directed_waypoint_stays_in_connected_known_free_space() -> None:
    grid = np.full((80, 80), CostValues.UNKNOWN, dtype=np.int8)
    grid[30:51, 20:61] = CostValues.FREE
    grid[30:51, 43] = CostValues.OCCUPIED
    costmap = OccupancyGrid(
        grid=grid,
        resolution=0.1,
        origin=Pose(position=Vector3(-2.0, -4.0, 0.0)),
        frame_id="world",
    )

    waypoint = directed_waypoint(
        costmap,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        max_step_m=3.0,
    )

    assert waypoint is not None
    assert 1.5 <= waypoint[0] < 2.3
    assert abs(waypoint[1]) <= 0.4
    assert costmap.cell_value(Vector3(*waypoint)) == CostValues.FREE


def test_directed_waypoint_does_not_detour_sideways_around_inflated_obstacle() -> None:
    grid = np.full((120, 120), CostValues.FREE, dtype=np.int8)
    grid[55:66, 65:70] = CostValues.OCCUPIED
    costmap = OccupancyGrid(
        grid=grid,
        resolution=0.1,
        origin=Pose(position=Vector3(-6.0, -6.0, 0.0)),
        frame_id="world",
    )

    waypoint = directed_waypoint(
        costmap,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        max_step_m=1.5,
        footprint_diameter_m=0.33,
    )

    assert waypoint is None


def test_directed_waypoint_excludes_previously_unreachable_goal() -> None:
    costmap = OccupancyGrid(
        grid=np.full((100, 100), CostValues.FREE, dtype=np.int8),
        resolution=0.1,
        origin=Pose(position=Vector3(-5.0, -5.0, 0.0)),
        frame_id="world",
    )
    failed = np.array([1.5, 0.0, 0.0])

    waypoint = directed_waypoint(
        costmap,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        excluded_waypoints=[failed],
    )

    assert waypoint is not None
    assert np.linalg.norm(waypoint[:2] - failed[:2]) > 0.75


def test_directed_waypoint_rejects_empty_and_vertical() -> None:
    assert (
        directed_waypoint(
            OccupancyGrid(grid=np.zeros((0, 0), dtype=np.int8), resolution=0.1),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
        )
        is None
    )
    costmap = OccupancyGrid(
        grid=np.full((10, 10), CostValues.FREE, dtype=np.int8),
        resolution=0.1,
        origin=Pose(position=Vector3(-0.5, -0.5, 0.0)),
        frame_id="world",
    )
    assert directed_waypoint(costmap, np.zeros(3), np.array([0.0, 0.0, -1.0])) is None


def test_directed_waypoint_extends_half_meter_into_unknown_on_ray() -> None:
    grid = np.full((100, 100), CostValues.UNKNOWN, dtype=np.int8)
    grid[48:53, 30:36] = CostValues.FREE
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    waypoint = directed_waypoint(costmap, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert waypoint is not None
    assert costmap.cell_value(Vector3(*waypoint)) == CostValues.UNKNOWN
    assert 0.5 < waypoint[0] <= 1.0 + 1e-9
    assert abs(waypoint[1]) <= 0.1 + 1e-9


def test_directed_waypoint_rejects_off_ray_free_strip() -> None:
    grid = np.full((100, 100), CostValues.UNKNOWN, dtype=np.int8)
    grid[48:53, 30:34] = CostValues.FREE
    grid[53:67, 30:47] = CostValues.FREE
    grid[48:53, 34] = CostValues.OCCUPIED
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    assert directed_waypoint(costmap, np.zeros(3), np.array([1.0, 0.0, 0.0])) is None


def test_unknown_extension_cannot_jump_thin_obstacle() -> None:
    grid = np.full((100, 100), CostValues.UNKNOWN, dtype=np.int8)
    grid[48:53, 30:36] = CostValues.FREE
    grid[:, 37] = CostValues.OCCUPIED
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    waypoint = directed_waypoint(
        costmap, np.zeros(3), np.array([1.0, 0.0, 0.0]), footprint_diameter_m=0.0
    )
    assert waypoint is not None
    assert waypoint[0] < 0.7


def test_on_ray_goal_beyond_wall_cannot_use_sideways_free_detour() -> None:
    grid = np.full((100, 100), CostValues.FREE, dtype=np.int8)
    grid[48:53, 34] = CostValues.OCCUPIED
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    assert directed_waypoint(costmap, np.zeros(3), np.array([1.0, 0.0, 0.0])) is None


def test_unknown_extension_stops_before_disconnected_free_island() -> None:
    grid = np.full((100, 100), CostValues.UNKNOWN, dtype=np.int8)
    grid[48:53, 30:36] = CostValues.FREE
    grid[48:53, 38:42] = CostValues.FREE
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1,
        origin=Pose(position=Vector3(-3.0, -5.0, 0.0)), frame_id="world",
    )
    waypoint = directed_waypoint(costmap, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert waypoint is not None
    assert costmap.cell_value(Vector3(*waypoint)) == CostValues.UNKNOWN
    assert waypoint[0] < 0.8


def test_reachable_free_cells_cannot_cut_diagonal_obstacle_corner() -> None:
    grid = np.full((10, 10), CostValues.OCCUPIED, dtype=np.int8)
    np.fill_diagonal(grid, CostValues.FREE)
    costmap = OccupancyGrid(
        grid=grid, resolution=0.1, origin=Pose(position=Vector3(0.0, 0.0, 0.0)),
        frame_id="world",
    )
    assert directed_waypoint(
        costmap, np.zeros(3), np.array([1.0, 1.0, 0.0]), footprint_diameter_m=0.0
    ) is None
