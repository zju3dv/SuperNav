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

"""LiDAR-map pixel-goal geometry: ray back-projection, surface heuristics, A* goals.

Pure functions and frozen data — no streams, no RPC, no threads — so the whole
geometry stack can be tested without a robot. The Go2 PixelGoal stack uses:

* the four surround cameras are modeled as pinhole with calibrated intrinsics
  or a fallback HFOV, plus per-direction mount extrinsics (yaw/height/pitch);
  ``world_to_optical_from_odom`` composes the odom/TF snapshot pose with the
  mount, so no USB camera frame needs to exist in TF;
* the original fisheye + ``CameraInfo`` + TF path (Go2 built-in front camera,
  frame ``camera_optical``) is kept verbatim as the ``fisheye`` camera model;
* everything else (ROI escalation, nearest-quartile front-surface heuristic,
  floor estimation, front-edge contact goal, costmap-directed exploration
  waypoints) uses the same projection and exploration rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any

import cv2
import numpy as np

from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image

MIN_RADIUS_PX = 4
MAX_RADIUS_PX = 100
MIN_TARGET_DISTANCE_M = 0.5
MAX_TARGET_DISTANCE_M = 10.0
ELEVATED_SURFACE_HEIGHT_M = 0.12
ROBOT_FRONT_OFFSET_M = 0.35
EXPLORATION_STEP_M = 1.5
MAX_EXPLORATION_DISTANCE_M = MAX_TARGET_DISTANCE_M
EXPLORATION_FOOTPRINT_SCALE = 1.1
EXPLORATION_MAX_LATERAL_M = 2.0
EXPLORATION_RAY_LATERAL_M = 0.1
EXPLORATION_UNKNOWN_EXTENSION_M = 0.5
EXPLORATION_FAILED_GOAL_CLEARANCE_M = 0.75
MAX_CONSECUTIVE_EXPLORATION_FAILURES = 8
PROJECTION_SEARCH_RADII_PX = (4, 8, 12, 16)
# Focal length of the 1280x720 rectified Go2 camera. Its
# pixel radii are scaled by fx so USB cameras retain the same angular ROI.
PROJECTION_REFERENCE_FOCAL_PX = 548.4
SURFACE_DEPTH_HYSTERESIS_M = 0.15
FLOOR_HEIGHT_BIN_M = 0.05

# IMX335 full active width: 2592 px * 2.0 um with the confirmed 3.3 mm lens.
DEFAULT_USB_HFOV_DEG = 76.3
DEFAULT_USB_WIDTH = 1600
DEFAULT_USB_HEIGHT = 1200
DEFAULT_MOUNT_HEIGHT_M = 0.60
DEFAULT_BASE_HEIGHT_M = 0.30
DEFAULT_MOUNT_PITCH_DEG = 0.0
# Ideal 90°-apart USB mount headings (positive = CCW/left).
DEFAULT_MOUNT_YAW_DEG = {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0}

BASE_FRAME = "base_link"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# ---------------------------------------------------------------------- camera models
@dataclass(frozen=True)
class PinholeModel:
    """Pinhole intrinsics for one frame size (square pixels, principal point centered)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> PinholeModel:
        """Approximate an uncalibrated USB frame as pinhole from its HFOV.

        Pixel centers span [0, width - 1] (same convention as
        ``NormalizedPoint.to_pixel``), so the principal point sits at
        ((width - 1) / 2, (height - 1) / 2) and fx maps the half-FOV tangent
        onto the half pixel span. Distortion is ignored.
        """
        fx = ((int(width) - 1) / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
        return cls(
            fx=fx,
            fy=fx,
            cx=(int(width) - 1) / 2.0,
            cy=(int(height) - 1) / 2.0,
            width=int(width),
            height=int(height),
        )

    def to_camera_info(self, frame_id: str = "") -> CameraInfo:
        return CameraInfo.from_intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            frame_id=frame_id,
        )


@dataclass(frozen=True)
class MountExtrinsics:
    """USB camera yaw, absolute height above ground, and downward pitch."""

    yaw_deg: float = 0.0  # 0=front, +90=left, -90=right, 180=back
    height_m: float = DEFAULT_MOUNT_HEIGHT_M
    pitch_deg: float = DEFAULT_MOUNT_PITCH_DEG  # positive = looking down

    def rotation_matrix(self) -> np.ndarray:
        """Base-frame rotation of the optical frame (columns: right, down, forward).

        Base frame: x forward, y left, z up. Optical frame: x image-right,
        y image-down, z into the scene. No roll: image-right stays horizontal
        (``forward x up``), image-down closes the right-handed frame.
        """
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        forward = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                -math.sin(pitch),
            ]
        )
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        down = np.cross(forward, right)
        return np.column_stack([right, down, forward])

    def optical_to_base(
        self, child_frame_id: str, base_height_m: float = DEFAULT_BASE_HEIGHT_M
    ) -> Transform:
        """Pose of the optical frame in the base frame (p_base = M @ p_optical)."""
        return Transform(
            translation=Vector3(0.0, 0.0, self.height_m - base_height_m),
            rotation=Quaternion.from_rotation_matrix(self.rotation_matrix()),
            frame_id=BASE_FRAME,
            child_frame_id=child_frame_id,
        )


def world_to_optical_from_odom(
    odom: PoseStamped,
    mount: MountExtrinsics,
    optical_frame: str,
    base_height_m: float = DEFAULT_BASE_HEIGHT_M,
) -> Transform:
    """Snapshot-time map→optical transform: odom base pose ∘ mount extrinsics.

    ``odom`` is the base pose in the global map frame (its ``frame_id``); the
    returned transform has ``frame_id=optical_frame`` and
    ``child_frame_id=odom.frame_id`` so ``to_matrix()`` maps world points into
    the camera, exactly like the TF lookup the fisheye path uses.
    """
    base_in_world = Transform.from_pose(BASE_FRAME, odom)  # parent=map, child=base
    optical_in_base = mount.optical_to_base(
        optical_frame, base_height_m
    )  # parent=base, child=optical
    return (base_in_world + optical_in_base).inverse()


@dataclass(frozen=True)
class SurroundCameraRig:
    """Four surround cameras: shared intrinsics + per-direction mount extrinsics."""

    hfov_deg: float = DEFAULT_USB_HFOV_DEG
    base_height_m: float = DEFAULT_BASE_HEIGHT_M
    mounts: dict[str, MountExtrinsics] = field(
        default_factory=lambda: {
            direction: MountExtrinsics(yaw_deg=yaw)
            for direction, yaw in DEFAULT_MOUNT_YAW_DEG.items()
        }
    )
    calibration: PinholeModel | None = None

    @classmethod
    def from_env(cls, prefix: str = "DIMOS_PIXELGOAL_MAP_") -> SurroundCameraRig:
        hfov = _env_float(f"{prefix}HFOV_DEG", DEFAULT_USB_HFOV_DEG)
        base_height = _env_float(f"{prefix}BASE_HEIGHT_M", DEFAULT_BASE_HEIGHT_M)
        intrinsic_names = (
            "FX",
            "FY",
            "CX",
            "CY",
            "CALIBRATION_WIDTH",
            "CALIBRATION_HEIGHT",
        )
        intrinsic_values = [os.environ.get(f"{prefix}{name}") for name in intrinsic_names]
        if any(value is not None for value in intrinsic_values):
            if not all(value is not None for value in intrinsic_values):
                missing = [
                    f"{prefix}{name}"
                    for name, value in zip(intrinsic_names, intrinsic_values, strict=True)
                    if value is None
                ]
                raise ValueError(
                    f"calibrated intrinsics must be set together; missing {', '.join(missing)}"
                )
            fx, fy, cx, cy = (float(value) for value in intrinsic_values[:4])
            width, height = (int(value) for value in intrinsic_values[4:])
            if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
                raise ValueError("calibrated focal lengths and image dimensions must be positive")
            calibration = PinholeModel(fx, fy, cx, cy, width, height)
        else:
            calibration = None
        mounts = {}
        for direction, default_yaw in DEFAULT_MOUNT_YAW_DEG.items():
            key = f"{prefix}MOUNT_{direction.upper()}"
            mounts[direction] = MountExtrinsics(
                yaw_deg=_env_float(f"{key}_YAW_DEG", default_yaw),
                height_m=_env_float(f"{key}_HEIGHT_M", DEFAULT_MOUNT_HEIGHT_M),
                pitch_deg=_env_float(f"{key}_PITCH_DEG", DEFAULT_MOUNT_PITCH_DEG),
            )
        return cls(
            hfov_deg=hfov,
            base_height_m=base_height,
            mounts=mounts,
            calibration=calibration,
        )

    def mount(self, direction: str) -> MountExtrinsics:
        return self.mounts.get(direction) or MountExtrinsics()

    def pinhole(self, width: int, height: int) -> PinholeModel:
        if self.calibration is not None:
            scale_x = int(width) / self.calibration.width
            scale_y = int(height) / self.calibration.height
            return PinholeModel(
                fx=self.calibration.fx * scale_x,
                fy=self.calibration.fy * scale_y,
                cx=self.calibration.cx * scale_x,
                cy=self.calibration.cy * scale_y,
                width=int(width),
                height=int(height),
            )
        return PinholeModel.from_hfov(width, height, self.hfov_deg)

    def optical_frame(self, direction: str) -> str:
        return f"go2/{direction}/camera_optical"


# ---------------------------------------------------------------------- rays / ROI
def pinhole_camera_info(camera_info: CameraInfo) -> CameraInfo:
    """Rectified pinhole equivalent of a (possibly fisheye) calibration."""
    model = (camera_info.distortion_model or "").lower()
    matrix = np.asarray(camera_info.K, dtype=np.float64).reshape(3, 3)
    if model == "equidistant":
        distortion = np.asarray(camera_info.D, dtype=np.float64)
        if distortion.size != 4:
            raise ValueError("equidistant camera calibration requires four coefficients")
        # Preserve the widest useful part of the Go2 fisheye frame.
        matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix,
            distortion,
            (camera_info.width, camera_info.height),
            np.asarray(camera_info.R, dtype=np.float64).reshape(3, 3),
            balance=1.0,
            new_size=(camera_info.width, camera_info.height),
        )
    elif model not in ("", "plumb_bob"):
        raise ValueError(f"unsupported distortion model: {camera_info.distortion_model}")

    return CameraInfo.from_intrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=camera_info.width,
        height=camera_info.height,
        frame_id=camera_info.frame_id,
    )


def projection_bbox(
    x: int,
    y: int,
    radius_px: int,
    camera_info: CameraInfo,
    pinhole_info: CameraInfo,
) -> tuple[float, float, float, float]:
    """Create an ROI in the rectified image used for both display and projection."""
    if (camera_info.width, camera_info.height) != (pinhole_info.width, pinhole_info.height):
        raise ValueError("raw and rectified image sizes differ")
    x1 = max(0.0, float(x - radius_px))
    y1 = max(0.0, float(y - radius_px))
    x2 = min(float(pinhole_info.width), float(x + radius_px))
    y2 = min(float(pinhole_info.height), float(y + radius_px))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("pixel ROI is outside the calibrated projection")
    return float(x1), float(y1), float(x2), float(y2)


def projection_search_radii(camera_info: CameraInfo, requested_radius_px: int) -> tuple[int, ...]:
    """Stable-branch search radii scaled to preserve their angular tolerance."""
    focal = (float(camera_info.K[0]) + float(camera_info.K[4])) / 2.0
    scale = focal / PROJECTION_REFERENCE_FOCAL_PX
    limit = max(int(requested_radius_px), PROJECTION_SEARCH_RADII_PX[-1])
    reference_radii = {
        *(radius for radius in PROJECTION_SEARCH_RADII_PX if radius <= limit),
        int(requested_radius_px),
    }
    return tuple(sorted({max(MIN_RADIUS_PX, round(radius * scale)) for radius in reference_radii}))


def rectify_image(image: Image, camera_info: CameraInfo, pinhole_info: CameraInfo) -> Image:
    """Rectify the frozen frame so clicks and projection share one camera model."""
    model = (camera_info.distortion_model or "").lower()
    if model == "" or not camera_info.D or not np.any(camera_info.D):
        return image.copy()

    matrix = np.asarray(camera_info.K, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info.D, dtype=np.float64)
    rectification = np.asarray(camera_info.R, dtype=np.float64).reshape(3, 3)
    projection = np.asarray(pinhole_info.K, dtype=np.float64).reshape(3, 3)
    size = (camera_info.width, camera_info.height)
    if model == "equidistant":
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, distortion, rectification, projection, size, cv2.CV_32FC1
        )
        data = cv2.remap(image.data, map_x, map_y, cv2.INTER_LINEAR)
    elif model == "plumb_bob":
        data = cv2.undistort(image.data, matrix, distortion, None, projection)
    else:
        raise ValueError(f"unsupported distortion model: {camera_info.distortion_model}")
    return Image(data=data, format=image.format, frame_id=image.frame_id, ts=image.ts)


def pixel_ray(
    x: int, y: int, camera_info: CameraInfo, world_to_optical: Transform
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera origin and a unit world-frame ray for a rectified pixel."""
    fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]
    optical_ray = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
    optical_to_world = world_to_optical.inverse().to_matrix()
    direction = optical_to_world[:3, :3] @ optical_ray
    direction /= np.linalg.norm(direction)
    return optical_to_world[:3, 3].copy(), direction


# ---------------------------------------------------------------------- surfaces
def estimate_floor_height(points: np.ndarray, camera_position: np.ndarray) -> float:
    valid = points[np.isfinite(points).all(axis=1)]
    if len(valid) == 0:
        return float(camera_position[2] - 0.3)
    below_camera = valid[valid[:, 2] <= camera_position[2] + 0.2]
    if len(below_camera) == 0:
        return float(camera_position[2] - 0.3)
    nearby = below_camera[np.linalg.norm(below_camera[:, :2] - camera_position[:2], axis=1) <= 4.0]
    sample = nearby if len(nearby) >= 20 else below_camera
    # Voxel maps quantize a broad floor plane into one or two dominant height
    # bins. A low quantile is biased by sparse points below the physical floor.
    height_bins = np.rint(sample[:, 2] / FLOOR_HEIGHT_BIN_M).astype(np.int64)
    bins, counts = np.unique(height_bins, return_counts=True)
    floor_bin = int(bins[int(np.argmax(counts))])
    floor_points = sample[height_bins == floor_bin, 2]
    return float(np.median(floor_points))


def floor_intersection_limit(
    origin: np.ndarray,
    direction: np.ndarray,
    floor_z: float,
) -> float | None:
    """Return a conservative horizontal exploration limit for a downward ray."""
    if direction[2] >= -1e-3:
        return None
    scale = float((floor_z - origin[2]) / direction[2])
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    floor_point = origin + scale * direction
    distance = float(np.linalg.norm(floor_point[:2] - origin[:2]))
    if not math.isfinite(distance):
        return None
    return distance


def front_surface_point(points: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
    valid = points[np.isfinite(points).all(axis=1)]
    if len(valid) < 3:
        raise ValueError("fewer than three projected map points")

    distances = np.linalg.norm(valid - camera_position, axis=1)
    # ponytail: nearest-quartile surface heuristic; replace with depth clustering only
    # if recorded Go2 scenes show foreground/background mixing failures.
    front = valid[distances <= np.quantile(distances, 0.25) + 0.15]
    return np.median(front, axis=0)


def select_surface_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the nearest credible surface, preferring tighter ROIs for depth ties."""
    if not candidates:
        raise ValueError("no pixel surface candidates")
    nearest_depth = min(float(candidate["ray_depth_m"]) for candidate in candidates)
    foreground = [
        candidate
        for candidate in candidates
        if float(candidate["ray_depth_m"]) <= nearest_depth + SURFACE_DEPTH_HYSTERESIS_M
    ]
    return min(
        foreground,
        key=lambda candidate: (
            int(candidate["radius_px"]),
            float(candidate["ray_miss_m"]),
        ),
    )


def contact_goal(
    target: np.ndarray,
    ray_origin: np.ndarray,
    offset_m: float = ROBOT_FRONT_OFFSET_M,
) -> np.ndarray:
    """Place the robot center so its front edge reaches the clicked XY point."""
    delta = np.asarray(target[:2] - ray_origin[:2], dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    goal = np.asarray(target, dtype=np.float64).copy()
    if distance > 1e-6:
        goal[:2] -= delta * (min(offset_m, distance) / distance)
    goal[2] = 0.0
    return goal


def projection_diagnostics(
    points: np.ndarray,
    camera_position: np.ndarray,
    ray_direction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | list[float]]:
    """Summarize whether ROI points lie on the clicked world ray."""
    ray = np.asarray(ray_direction, dtype=np.float64)
    ray /= np.linalg.norm(ray)
    deltas = np.asarray(points, dtype=np.float64) - camera_position
    along = deltas @ ray
    perpendicular = np.linalg.norm(deltas - along[:, None] * ray, axis=1)
    target_delta = np.asarray(target, dtype=np.float64) - camera_position
    target_along = float(target_delta @ ray)
    target_miss = float(np.linalg.norm(target_delta - target_along * ray))
    return {
        "ray_depth_quantiles_m": np.quantile(along, [0.1, 0.5, 0.9]).round(3).tolist(),
        "ray_miss_quantiles_m": np.quantile(perpendicular, [0.1, 0.5, 0.9]).round(3).tolist(),
        "target_ray_depth_m": round(target_along, 3),
        "target_ray_miss_m": round(target_miss, 3),
    }


def segment_crosses_obstacle(
    costmap: OccupancyGrid, start: np.ndarray, end: np.ndarray
) -> bool:
    """Check a segment against known obstacles on an already inflated map."""
    distance = float(np.linalg.norm(end[:2] - start[:2]))
    samples = max(2, math.ceil(distance / (costmap.resolution / 2)) + 1)
    for fraction in np.linspace(0.0, 1.0, samples):
        xy = start[:2] + (end[:2] - start[:2]) * fraction
        cell = costmap.world_to_grid(Vector3(float(xy[0]), float(xy[1]), 0.0))
        x, y = math.floor(cell.x), math.floor(cell.y)
        if 0 <= x < costmap.width and 0 <= y < costmap.height:
            if costmap.grid[y, x] >= 50:
                return True
    return False


def directed_waypoint(
    costmap: OccupancyGrid,
    origin: np.ndarray,
    direction: np.ndarray,
    max_step_m: float = EXPLORATION_STEP_M,
    footprint_diameter_m: float = 0.33,
    excluded_waypoints: list[np.ndarray] | None = None,
) -> np.ndarray | None:
    """Choose a ray-aligned free or short unknown-frontier exploration goal."""
    horizontal = np.asarray(direction[:2], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6 or costmap.grid.size == 0:
        return None
    horizontal /= norm

    planning_map = simple_inflate(costmap, footprint_diameter_m / 2.0)
    start = planning_map.world_to_grid(Vector3(float(origin[0]), float(origin[1]), 0.0))
    sx, sy = int(start.x), int(start.y)
    if not (0 <= sx < planning_map.width and 0 <= sy < planning_map.height):
        return None

    traversable = (planning_map.grid >= CostValues.FREE) & (planning_map.grid < 50)
    if not traversable[sy, sx]:
        free_y, free_x = np.nonzero(traversable)
        if len(free_x) == 0:
            return None
        distances = (free_x - sx) ** 2 + (free_y - sy) ** 2
        nearest = int(np.argmin(distances))
        if distances[nearest] > max(1, math.ceil(0.5 / planning_map.resolution)) ** 2:
            return None
        sx, sy = int(free_x[nearest]), int(free_y[nearest])

    search_radius_m = max_step_m + EXPLORATION_MAX_LATERAL_M
    max_cells = max(1, math.ceil(search_radius_m / planning_map.resolution))
    min_forward = min(0.5, max_step_m * 0.4)
    queue = [(sx, sy)]
    visited = {(sx, sy)}
    best: tuple[float, int, int] | None = None
    # Unknown space is not flood-filled: it is only entered from an actually
    # reachable free cell, in short ray-aligned increments.
    def consider(gx: int, gy: int) -> None:
        nonlocal best
        point = planning_map.grid_to_world((gx, gy))
        delta = np.array([point.x - origin[0], point.y - origin[1]], dtype=np.float64)
        forward = float(delta @ horizontal)
        lateral = abs(float(delta[0] * horizontal[1] - delta[1] * horizontal[0]))
        if not (
            min_forward <= forward <= max_step_m
            and lateral <= EXPLORATION_RAY_LATERAL_M + 1e-9
            and np.linalg.norm(delta) <= max_step_m + 1e-9
        ):
            return
        # A free detour found by the flood fill must not turn a click beyond a
        # known wall into a sideways exploration route.
        if segment_crosses_obstacle(planning_map, origin, np.array([point.x, point.y, 0.0])):
            return
        if any(
            np.linalg.norm(np.array([point.x, point.y]) - np.asarray(candidate)[:2])
            <= EXPLORATION_FAILED_GOAL_CLEARANCE_M
            for candidate in (excluded_waypoints or [])
        ):
            return
        score = forward - 0.25 * lateral
        if best is None or score > best[0]:
            best = (score, gx, gy)

    for gx, gy in queue:
        delta = np.array([gx - sx, gy - sy], dtype=np.float64) * planning_map.resolution
        distance = float(np.linalg.norm(delta))
        if distance <= search_radius_m:
            consider(gx, gy)
            free_point = planning_map.grid_to_world((gx, gy))
            # Sample at sub-cell spacing to avoid jumping over a thin occupied
            # strip; stop at the first occupied cell or free-space gap.
            free_delta = np.array([free_point.x, free_point.y]) - origin[:2]
            free_lateral = abs(float(free_delta @ np.array([-horizontal[1], horizontal[0]])))
            if free_lateral <= EXPLORATION_RAY_LATERAL_M:
                entered_unknown = False
                seen: set[tuple[int, int]] = set()
                for step in np.arange(
                    planning_map.resolution / 2,
                    EXPLORATION_UNKNOWN_EXTENSION_M + planning_map.resolution / 4,
                    planning_map.resolution / 2,
                ):
                    world = np.array([free_point.x, free_point.y]) + horizontal * step
                    cell = planning_map.world_to_grid(
                        Vector3(float(world[0]), float(world[1]), 0.0)
                    )
                    ux, uy = int(cell.x), int(cell.y)
                    if not (0 <= ux < planning_map.width and 0 <= uy < planning_map.height):
                        break
                    if (ux, uy) in seen:
                        continue
                    seen.add((ux, uy))
                    value = planning_map.grid[uy, ux]
                    if value == CostValues.UNKNOWN:
                        entered_unknown = True
                        consider(ux, uy)
                    elif entered_unknown or value >= 50:
                        break
        if max(abs(gx - sx), abs(gy - sy)) >= max_cells:
            continue
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            nx, ny = gx + dx, gy + dy
            if (
                0 <= nx < planning_map.width
                and 0 <= ny < planning_map.height
                and traversable[ny, nx]
                and (
                    dx == 0 or dy == 0 or (traversable[gy, nx] and traversable[ny, gx])
                )
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))

    if best is None:
        return None
    point = planning_map.grid_to_world((best[1], best[2]))
    return np.array([point.x, point.y, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------- config
DEFAULT_MAP_TIMEOUT_S = 4.0
DEFAULT_DEADLINE_S = 60.0
DEFAULT_RADIUS_PX = 8


@dataclass(frozen=True)
class PixelGoalMapConfig:
    """Map-backend knobs; ``from_env`` applies DIMOS_PIXELGOAL_MAP_* overrides."""

    rig: SurroundCameraRig = field(default_factory=SurroundCameraRig)
    radius_px: int = DEFAULT_RADIUS_PX
    min_target_distance_m: float = MIN_TARGET_DISTANCE_M
    max_target_distance_m: float = MAX_TARGET_DISTANCE_M
    robot_front_offset_m: float = ROBOT_FRONT_OFFSET_M
    exploration_step_m: float = EXPLORATION_STEP_M
    max_exploration_distance_m: float = MAX_EXPLORATION_DISTANCE_M
    map_timeout_s: float = DEFAULT_MAP_TIMEOUT_S
    deadline_s: float = DEFAULT_DEADLINE_S
    no_progress_timeout_s: float = 15.0
    progress_epsilon_m: float = 0.1

    def __post_init__(self) -> None:
        for name in ("no_progress_timeout_s", "progress_epsilon_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_env(cls) -> PixelGoalMapConfig:
        prefix = "DIMOS_PIXELGOAL_MAP_"
        return cls(
            rig=SurroundCameraRig.from_env(prefix),
            radius_px=_env_int(f"{prefix}RADIUS_PX", DEFAULT_RADIUS_PX),
            min_target_distance_m=_env_float(
                f"{prefix}MIN_TARGET_DISTANCE_M", MIN_TARGET_DISTANCE_M
            ),
            max_target_distance_m=_env_float(
                f"{prefix}MAX_TARGET_DISTANCE_M", MAX_TARGET_DISTANCE_M
            ),
            robot_front_offset_m=_env_float(f"{prefix}ROBOT_FRONT_OFFSET_M", ROBOT_FRONT_OFFSET_M),
            exploration_step_m=_env_float(f"{prefix}EXPLORATION_STEP_M", EXPLORATION_STEP_M),
            max_exploration_distance_m=_env_float(
                f"{prefix}MAX_EXPLORATION_DISTANCE_M", MAX_EXPLORATION_DISTANCE_M
            ),
            map_timeout_s=_env_float(f"{prefix}MAP_TIMEOUT_S", DEFAULT_MAP_TIMEOUT_S),
            deadline_s=_env_float(f"{prefix}DEADLINE_S", DEFAULT_DEADLINE_S),
            no_progress_timeout_s=_env_float(f"{prefix}NO_PROGRESS_TIMEOUT_S", 15.0),
            progress_epsilon_m=_env_float(f"{prefix}PROGRESS_EPSILON_M", 0.1),
        )


__all__ = [
    "BASE_FRAME",
    "DEFAULT_BASE_HEIGHT_M",
    "DEFAULT_MOUNT_YAW_DEG",
    "DEFAULT_USB_HFOV_DEG",
    "MountExtrinsics",
    "PinholeModel",
    "PixelGoalMapConfig",
    "SurroundCameraRig",
    "contact_goal",
    "directed_waypoint",
    "estimate_floor_height",
    "floor_intersection_limit",
    "front_surface_point",
    "pinhole_camera_info",
    "pixel_ray",
    "projection_bbox",
    "projection_diagnostics",
    "projection_search_radii",
    "rectify_image",
    "select_surface_candidate",
    "world_to_optical_from_odom",
]
