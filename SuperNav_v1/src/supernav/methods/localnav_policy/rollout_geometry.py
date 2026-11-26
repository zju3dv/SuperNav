"""Owned, simulator-independent geometry for LocalNav training rollouts.

Depth is used only to choose supervised rollout targets. The deployed LocalNav
policy continues to consume RGB and discrete action feedback exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from habitat_contract.navigation import NavigationBackend, NavigationPathfinder
from supernav.methods.localnav.contracts import NormalizedPoint


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    hfov_deg: float

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Camera dimensions must be at least two pixels")
        if not (0.0 < self.hfov_deg < 180.0):
            raise ValueError("Camera horizontal field of view must be between 0 and 180 degrees")

    @property
    def focal_length(self) -> float:
        return float(self.width) / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))


def _quat_wxyz_to_yaw(q_wxyz: list[float]) -> float:
    """Extract yaw around +Y; identity faces -Z in the rollout frame."""
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(siny_cosp, cosy_cosp)


def _agent_pose(adapter: NavigationBackend, session: Any) -> tuple[np.ndarray, float]:
    return (
        np.asarray(adapter.agent_position(session), dtype=np.float32),
        _quat_wxyz_to_yaw(adapter.agent_rotation_wxyz(session)),
    )


def _capture(adapter: NavigationBackend, session: Any) -> tuple[np.ndarray, np.ndarray]:
    observations = adapter.sensor_observations(session)
    rgb = np.asarray(observations["color_sensor"])[..., :3]
    return np.ascontiguousarray(rgb, dtype=np.uint8), np.asarray(observations["depth_sensor"])


def _shortest_path_points(pathfinder: NavigationPathfinder, start: Any, goal: Any):
    path = pathfinder.shortest_path(start, goal)
    if path is None:
        return None
    return np.asarray(path.points, dtype=np.float32), float(path.geodesic_distance)


def _geodesic(pathfinder: NavigationPathfinder, start: Any, goal: Any) -> float:
    path = pathfinder.shortest_path(start, goal)
    return float(path.geodesic_distance) if path is not None else float("inf")


def _quat_xyzw_from_yaw(yaw: float) -> list[float]:
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _sample_hop_endpoints(pathfinder: NavigationPathfinder, rng: Any, min_geo: float, max_geo: float):
    """Bound rejection sampling to reachable point pairs in the requested bin."""
    if not (0.0 <= min_geo < max_geo):
        raise ValueError("Expected 0 <= min_geo < max_geo")
    # Habitat uses a native RNG for navmesh sampling; Python's rollout seed
    # must seed that generator too, otherwise repeated runs choose new hops.
    pathfinder.seed(rng.randrange(2**31))
    for _ in range(256):
        start = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        goal = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if rng.random() < 0.5:
            start, goal = goal, start
        distance = _geodesic(pathfinder, start, goal)
        if math.isfinite(distance) and min_geo <= distance <= max_geo:
            return start, goal, distance
    return None


def resample_polyline(points: Any, spacing: float) -> np.ndarray:
    """Sample fixed arc-length increments and retain the final endpoint."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty sequence of 3D points")
    if spacing <= 0 or not math.isfinite(spacing):
        raise ValueError("spacing must be finite and positive")
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    # Duplicate vertices have no arc length and must not affect interpolation.
    unique = np.concatenate(([True], np.diff(distances) > 1e-8))
    points, distances = points[unique], distances[unique]
    if len(points) == 1:
        return points.copy()
    samples = np.arange(0.0, distances[-1], spacing)
    samples = np.append(samples, distances[-1])
    return np.stack([np.interp(samples, distances, points[:, i]) for i in range(3)], axis=1).astype(np.float32)


def select_hop_target(positions: Any, yaws: Any, camera: CameraIntrinsics, sensor_height: float, depth: Any):
    """Choose the furthest future path point visible in the initial RGB frame.

    Return a normalized image point only when the mounted camera projects it
    inside the image and the depth sensor does not place an obstacle in front.
    """
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(yaws) != len(positions):
        raise ValueError("Expected 3D path points and one yaw per point")
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        raise ValueError("Path coordinates and headings must be finite")
    if len(positions) < 2:
        return None
    yaw = float(yaws[0])
    eye = positions[0].copy()
    eye[1] += float(sensor_height)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    forward = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    depth = np.asarray(depth).squeeze()
    if depth.ndim != 2:
        raise ValueError("Expected a 2D depth observation")
    focal = camera.focal_length
    for index in range(len(positions) - 1, 0, -1):
        delta = positions[index] - eye
        distance = float(np.dot(delta, forward))
        if distance <= 0.05:
            continue
        px = (camera.width - 1) / 2.0 + focal * float(np.dot(delta, right)) / distance
        py = (camera.height - 1) / 2.0 - focal * float(delta[1]) / distance
        if not (0 <= px <= camera.width - 1 and 0 <= py <= camera.height - 1):
            continue
        x = min(depth.shape[1] - 1, round(px * (depth.shape[1] - 1) / max(1, camera.width - 1)))
        y = min(depth.shape[0] - 1, round(py * (depth.shape[0] - 1) / max(1, camera.height - 1)))
        measured = float(depth[y, x])
        if not math.isfinite(measured) or measured <= 0 or measured + 0.25 < distance:
            continue
        return index, NormalizedPoint(
            px / max(1, camera.width - 1), py / max(1, camera.height - 1)
        ).as_tuple()
    return None
