#!/usr/bin/env python3
"""Planar Open3D relocalization sidecar for Cartographer retrieval sessions."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
try:
    from scipy import ndimage  # type: ignore[import-untyped]
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - runtime env may omit sparse-only CSM deps
    ndimage = None
    cKDTree = None

try:
    import open3d as o3d  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised in runtime env checks
    o3d = None


_REG = o3d.pipelines.registration if o3d is not None else None
SCALE_PLAN: list[tuple[float, int]] = [(0.2, 8), (0.3, 8), (0.8, 1)]
RANSAC_ITERS = 500_000
FINE_VOXEL = 0.1
RERANK_DIST = 0.15
SPARSE_MIN_SOURCE_POINTS = 120
SPARSE_MIN_TARGET_POINTS = 120
QUADRANT_YAW_DEGREES: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
SPARSE_YAW_REFINEMENT_DEGREES: tuple[float, ...] = (
    -16.0,
    -12.0,
    -10.0,
    -8.0,
    -6.0,
    -4.0,
    -2.0,
    0.0,
    2.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    16.0,
)
CSM_GRID_RESOLUTION_M = 0.02
CSM_COARSE_XY_WINDOW_M = 1.5
CSM_COARSE_XY_STEP_M = 0.10
CSM_COARSE_YAW_WINDOW_DEG = 15.0
CSM_COARSE_YAW_STEP_DEG = 2.0
CSM_FINE_XY_WINDOW_M = 0.10
CSM_FINE_XY_STEP_M = 0.02
CSM_FINE_YAW_WINDOW_DEG = 2.0
CSM_FINE_YAW_STEP_DEG = 0.5
CSM_COARSE_KEEP_PER_SEED = 5
CSM_FINE_CENTERS_PER_SEED = 3
CSM_FINE_KEEP_PER_CENTER = 3
CSM_MAX_CANDIDATES = 30
CSM_SEED_LIMIT = 15
CSM_DISTINCT_TRANSLATION_M = 0.10
CSM_DISTINCT_YAW_DEG = 1.0

OCCUPIED_MATCH_TOLERANCE_M = 0.15
TARGET_CELL_TOLERANCE_M = 0.04
FREE_OCCUPIED_CONFLICT_TOLERANCE_M = 0.06
SOURCE_FREE_CLEARANCE_M = 0.15
MAX_OCCUPIED_VS_FREE_CONFLICT_RATIO = 0.15
MAX_FREE_VS_OCCUPIED_CONFLICT_RATIO = 0.10
MIN_OCCUPIED_MATCH_RATIO = 0.70
MIN_SUPPORTED_TRAJECTORY_BINS = 3
MIN_GEOMETRIC_SCORE_MARGIN = 0.03
ACCUMULATED_MIN_COMPARABLE_OCCUPIED = 250
ACCUMULATED_MIN_COMPARABLE_FREE = 250
ACCUMULATED_MIN_UNIQUE_TARGET_SUPPORT = 100
ACCUMULATED_DIAGNOSTIC_MIN_INLIERS = 100
SPARSE_MIN_COMPARABLE_OCCUPIED = 80
SPARSE_MIN_COMPARABLE_FREE = 120
SPARSE_MIN_UNIQUE_TARGET_SUPPORT = 50
SPARSE_DIAGNOSTIC_MIN_INLIERS = 50
DIAGNOSTIC_MIN_FITNESS = 0.10
ACCUMULATED_DIAGNOSTIC_MAX_RMSE_M = 0.15
SPARSE_DIAGNOSTIC_MAX_RMSE_M = 0.20
ACCUMULATED_ICP_CANDIDATE_LIMIT = 40
SPARSE_ICP_CANDIDATE_LIMIT = 60


def _read_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"expected XY point cloud in {path}")
    points = points[:, :2]
    return points[np.isfinite(points).all(axis=1)]


def _read_source_grid(path: Path) -> dict[str, np.ndarray]:
    cells = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if cells.ndim != 2 or cells.shape[1] < 5:
        raise ValueError(
            f"expected x, y, state, probability, trajectory_bin in {path}"
        )
    cells = cells[np.isfinite(cells[:, :5]).all(axis=1)]
    occupied = cells[:, 2] >= 0.5
    return {
        "occupied_points": cells[occupied, :2],
        "occupied_bins": cells[occupied, 4].astype(np.int64),
        "free_points": cells[~occupied, :2],
        "free_bins": cells[~occupied, 4].astype(np.int64),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:2, :2].T + transform[:2, 3]


def _query_nearest(tree: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    distances, indices = tree.query(points, k=1)
    return np.asarray(distances), np.asarray(indices, dtype=np.int64)


def _support_extent(points: np.ndarray) -> dict[str, float | int]:
    if len(points) == 0:
        return {"point_count": 0, "span_x": 0.0, "span_y": 0.0, "extent": 0.0}
    lower, upper = np.percentile(points, [5.0, 95.0], axis=0)
    span = upper - lower
    return {
        "point_count": int(len(points)),
        "span_x": float(span[0]),
        "span_y": float(span[1]),
        "extent": float(np.linalg.norm(span)),
    }


def _prepare_geometry_context(
    source_grid: dict[str, np.ndarray],
    target_occupied: np.ndarray,
    target_free: np.ndarray,
) -> dict[str, Any]:
    if cKDTree is None:
        raise RuntimeError("SciPy cKDTree is required for occupancy validation")
    source_occupied = source_grid["occupied_points"]
    source_free = source_grid["free_points"]
    if len(source_occupied) == 0 or len(source_free) == 0:
        raise ValueError("source occupancy grid lacks occupied or confident-free cells")
    if len(target_occupied) == 0 or len(target_free) == 0:
        raise ValueError("target occupancy grid lacks occupied or free cells")
    source_occupied_tree = cKDTree(source_occupied)
    free_clearance, _ = _query_nearest(source_occupied_tree, source_free)
    retained_free = free_clearance >= SOURCE_FREE_CLEARANCE_M
    return {
        "source_occupied": source_occupied,
        "source_occupied_bins": source_grid["occupied_bins"],
        "source_free": source_free[retained_free],
        "source_free_bins": source_grid["free_bins"][retained_free],
        "source_free_uneroded_count": int(len(source_free)),
        "target_occupied_tree": cKDTree(target_occupied),
        "target_free_tree": cKDTree(target_free),
    }


def _sample_geometry_context(
    context: dict[str, Any], max_points_per_state: int = 750
) -> dict[str, Any]:
    occupied_step = max(
        1, math.ceil(len(context["source_occupied"]) / max_points_per_state)
    )
    free_step = max(
        1, math.ceil(len(context["source_free"]) / max_points_per_state)
    )
    return {
        **context,
        "source_occupied": context["source_occupied"][::occupied_step],
        "source_occupied_bins": context["source_occupied_bins"][::occupied_step],
        "source_free": context["source_free"][::free_step],
        "source_free_bins": context["source_free_bins"][::free_step],
    }


def _geometric_metrics(
    context: dict[str, Any], transform: np.ndarray
) -> dict[str, Any]:
    source_occupied = context["source_occupied"]
    transformed_occupied = _transform_points(source_occupied, transform)
    occupied_distances, occupied_indices = _query_nearest(
        context["target_occupied_tree"], transformed_occupied
    )
    occupied_free_distances, _ = _query_nearest(
        context["target_free_tree"], transformed_occupied
    )
    occupied_matches = occupied_distances <= OCCUPIED_MATCH_TOLERANCE_M
    occupied_free_conflicts = (
        (occupied_free_distances <= TARGET_CELL_TOLERANCE_M)
        & (occupied_distances > OCCUPIED_MATCH_TOLERANCE_M)
    )
    occupied_comparable = occupied_matches | occupied_free_conflicts

    transformed_free = _transform_points(context["source_free"], transform)
    free_occupied_distances, _ = _query_nearest(
        context["target_occupied_tree"], transformed_free
    )
    free_free_distances, _ = _query_nearest(
        context["target_free_tree"], transformed_free
    )
    free_occupied_conflicts = (
        free_occupied_distances <= FREE_OCCUPIED_CONFLICT_TOLERANCE_M
    )
    free_matches = (
        (free_free_distances <= TARGET_CELL_TOLERANCE_M)
        & ~free_occupied_conflicts
    )
    free_comparable = free_matches | free_occupied_conflicts

    occupied_match_count = int(np.count_nonzero(occupied_matches))
    occupied_conflict_count = int(np.count_nonzero(occupied_free_conflicts))
    occupied_comparable_count = int(np.count_nonzero(occupied_comparable))
    free_match_count = int(np.count_nonzero(free_matches))
    free_conflict_count = int(np.count_nonzero(free_occupied_conflicts))
    free_comparable_count = int(np.count_nonzero(free_comparable))
    supported_target_indices = np.unique(occupied_indices[occupied_matches])
    occupied_bins = context["source_occupied_bins"]
    bin_rows = []
    for bin_id in sorted(int(value) for value in np.unique(occupied_bins)):
        bin_mask = occupied_bins == bin_id
        bin_rows.append(
            {
                "bin": bin_id,
                "source_occupied_count": int(np.count_nonzero(bin_mask)),
                "comparable_count": int(
                    np.count_nonzero(bin_mask & occupied_comparable)
                ),
                "matched_count": int(np.count_nonzero(bin_mask & occupied_matches)),
            }
        )
    supported_bin_count = sum(row["matched_count"] >= 10 for row in bin_rows)
    known_source_count = len(source_occupied) + len(context["source_free"])
    comparable_count = occupied_comparable_count + free_comparable_count
    return {
        "comparable_coverage": _ratio(comparable_count, known_source_count),
        "occupied_comparable_coverage": _ratio(
            occupied_comparable_count, len(source_occupied)
        ),
        "free_comparable_coverage": _ratio(
            free_comparable_count, len(context["source_free"])
        ),
        "occupied_match_ratio": _ratio(
            occupied_match_count, occupied_comparable_count
        ),
        "occupied_vs_free_conflict_ratio": _ratio(
            occupied_conflict_count, occupied_comparable_count
        ),
        "free_match_ratio": _ratio(free_match_count, free_comparable_count),
        "free_vs_occupied_conflict_ratio": _ratio(
            free_conflict_count, free_comparable_count
        ),
        "unique_target_support_ratio": _ratio(
            len(supported_target_indices), occupied_match_count
        ),
        "unique_target_support_count": int(len(supported_target_indices)),
        "support_extent_m": _support_extent(
            transformed_occupied[occupied_matches]
        ),
        "support_trajectory_bins": {
            "method": "nearest_scan_origin_quartile",
            "supported_bin_count": supported_bin_count,
            "minimum_matches_per_supported_bin": 10,
            "bins": bin_rows,
        },
        "counts": {
            "source_occupied": int(len(source_occupied)),
            "source_free_uneroded": context["source_free_uneroded_count"],
            "source_free_eroded": int(len(context["source_free"])),
            "occupied_comparable": occupied_comparable_count,
            "occupied_matches": occupied_match_count,
            "occupied_vs_free_conflicts": occupied_conflict_count,
            "free_comparable": free_comparable_count,
            "free_matches": free_match_count,
            "free_vs_occupied_conflicts": free_conflict_count,
        },
    }


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    if o3d is None:
        raise RuntimeError("Open3D is required for relocalization")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.column_stack((points, np.zeros(len(points), dtype=np.float64)))
    )
    return cloud


def _preprocess(
    cloud: o3d.geometry.PointCloud, voxel_size: float
) -> tuple[o3d.geometry.PointCloud, Any]:
    if o3d is None:
        raise RuntimeError("Open3D is required for relocalization")
    down = cloud.voxel_down_sample(voxel_size)
    # Planar scans have a single normal direction. FPFH remains a coarse
    # proposal generator only; SE(2) scoring below is authoritative.
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return down, _REG.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )


def _se2_transform(transform: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    result = np.eye(4)
    result[0, 0] = math.cos(yaw)
    result[0, 1] = -math.sin(yaw)
    result[1, 0] = math.sin(yaw)
    result[1, 1] = math.cos(yaw)
    result[:2, 3] = transform[:2, 3]
    return result


def _yaw_rotation_transform(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    result = np.eye(4)
    result[0, 0] = math.cos(yaw)
    result[0, 1] = -math.sin(yaw)
    result[1, 0] = math.sin(yaw)
    result[1, 1] = math.cos(yaw)
    return result


def _source_pivot_rotation_transform(
    pivot_xy: np.ndarray,
    yaw_degrees: float,
) -> np.ndarray:
    result = np.eye(4)
    result[:2, :2] = _yaw_rotation_transform(yaw_degrees)[:2, :2]
    result[:2, 3] = pivot_xy - result[:2, :2] @ pivot_xy
    return result


def _rotate_candidate_about_source_pivot(
    candidate: np.ndarray,
    pivot_xy: np.ndarray,
    yaw_degrees: float,
) -> np.ndarray:
    return _se2_transform(
        candidate @ _source_pivot_rotation_transform(pivot_xy, yaw_degrees)
    )


def _ransac(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh: Any,
    target_fpfh: Any,
    voxel_size: float,
) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("Open3D is required for relocalization")
    distance = voxel_size * 1.5
    result = _REG.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance,
        estimation_method=_REG.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            _REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            _REG.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        criteria=_REG.RANSACConvergenceCriteria(RANSAC_ITERS, 0.995),
    )
    return _se2_transform(np.asarray(result.transformation))


def _score(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
) -> tuple[float, float, int, int]:
    evaluated = _REG.evaluate_registration(source, target, RERANK_DIST, transform)
    correspondences = np.asarray(evaluated.correspondence_set)
    correspondence_count = len(correspondences)
    unique_target_count = (
        len(np.unique(correspondences[:, 1])) if correspondence_count > 0 else 0
    )
    return (
        float(evaluated.fitness),
        float(evaluated.inlier_rmse),
        correspondence_count,
        unique_target_count,
    )


def _distinct(left: np.ndarray, right: np.ndarray) -> bool:
    translation = float(np.linalg.norm(left[:2, 3] - right[:2, 3]))
    yaw_left = math.atan2(float(left[1, 0]), float(left[0, 0]))
    yaw_right = math.atan2(float(right[1, 0]), float(right[0, 0]))
    yaw_delta = abs((yaw_left - yaw_right + math.pi) % (2 * math.pi) - math.pi)
    return translation >= 1.0 or yaw_delta >= math.radians(30.0)


def _region_bbox_from_prior(prior: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(prior, dict):
        return None
    bbox = prior.get("region_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            values = []
        if len(values) == 4 and all(math.isfinite(value) for value in values):
            x0, y0, x1, y1 = values
            if x0 < x1 and y0 < y1:
                return [x0, y0, x1, y1]
    pose = prior.get("slam_pose") if isinstance(prior.get("slam_pose"), dict) else None
    if isinstance(pose, dict):
        try:
            x = float(pose["x"])
            y = float(pose["y"])
            radius = float(prior.get("region_radius_m", 4.0))
        except (KeyError, TypeError, ValueError):
            return None
        if (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(radius)
            and radius > 0
        ):
            return [x - radius, y - radius, x + radius, y + radius]
    return None


def _filter_points_to_bbox(points: np.ndarray, bbox: list[float] | None) -> np.ndarray:
    if bbox is None:
        return points
    x0, y0, x1, y1 = bbox
    mask = (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
    )
    return points[mask]


def _crop_sparse_node_prior_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    bbox: list[float] | None,
    agent_local_pose: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del agent_local_pose
    if bbox is None:
        return source_points, target_points, {
            "source_region_bbox": None,
            "target_region_bbox": None,
        }
    target_cropped = _filter_points_to_bbox(target_points, bbox)
    return (
        source_points,
        target_cropped,
        {
            "source_region_bbox": None,
            "target_region_bbox": bbox,
        },
    )


def _transformed_centroid_inside_bbox(
    source_points: np.ndarray, transform: np.ndarray, bbox: list[float] | None
) -> bool:
    if bbox is None or len(source_points) == 0:
        return True
    center = source_points.mean(axis=0)
    transformed = transform[:2, :2] @ center + transform[:2, 3]
    x0, y0, x1, y1 = bbox
    return bool(x0 <= transformed[0] <= x1 and y0 <= transformed[1] <= y1)


def _agent_local_pose_from_prior(
    prior: dict[str, Any] | None,
) -> tuple[float, float, float] | None:
    if not isinstance(prior, dict):
        return None
    pose = prior.get("agent_local_pose")
    if not isinstance(pose, dict):
        return None
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose.get("yaw_rad", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if all(math.isfinite(value) for value in (x, y, yaw)):
        return (x, y, yaw)
    return None


def _transformed_agent_pose(
    transform: np.ndarray, agent_local_pose: tuple[float, float, float] | None
) -> tuple[float, float, float] | None:
    if agent_local_pose is None:
        return None
    local_x, local_y, local_yaw = agent_local_pose
    transformed_xy = (
        transform[:2, :2] @ np.array([local_x, local_y]) + transform[:2, 3]
    )
    transform_yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    yaw = (transform_yaw + local_yaw + math.pi) % (2 * math.pi) - math.pi
    return (float(transformed_xy[0]), float(transformed_xy[1]), yaw)


def _transformed_agent_pose_inside_bbox(
    transform: np.ndarray,
    agent_local_pose: tuple[float, float, float] | None,
    bbox: list[float] | None,
) -> bool:
    if bbox is None:
        return True
    transformed = _transformed_agent_pose(transform, agent_local_pose)
    if transformed is None:
        return False
    x0, y0, x1, y1 = bbox
    return bool(x0 <= transformed[0] <= x1 and y0 <= transformed[1] <= y1)


def _yaw_degrees(transform: np.ndarray) -> float:
    return math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))


def _se2_from_xy_yaw(x: float, y: float, yaw_degrees: float) -> np.ndarray:
    transform = _yaw_rotation_transform(yaw_degrees)
    transform[0, 3] = x
    transform[1, 3] = y
    return transform


def _yaw_delta_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _csm_region_mask(
    translations: np.ndarray,
    yaw_degrees: float,
    source_points: np.ndarray,
    agent_local_pose: tuple[float, float, float] | None,
    bbox: list[float] | None,
) -> np.ndarray:
    if bbox is None:
        return np.ones(len(translations), dtype=bool)
    x0, y0, x1, y1 = bbox
    if agent_local_pose is not None:
        pivot = np.array([agent_local_pose[0], agent_local_pose[1]], dtype=np.float64)
    else:
        pivot = source_points.mean(axis=0)
    yaw = math.radians(yaw_degrees)
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    transformed = pivot @ rotation.T + translations
    return (
        (transformed[:, 0] >= x0)
        & (transformed[:, 0] <= x1)
        & (transformed[:, 1] >= y0)
        & (transformed[:, 1] <= y1)
    )


def _build_csm_distance_field(
    target_points: np.ndarray, resolution_m: float
) -> tuple[np.ndarray, np.ndarray, float]:
    if ndimage is None:
        raise RuntimeError("SciPy is required for sparse CSM relocalization")
    padding = CSM_COARSE_XY_WINDOW_M + RERANK_DIST + resolution_m
    origin = target_points.min(axis=0) - padding
    upper = target_points.max(axis=0) + padding
    shape_xy = np.ceil((upper - origin) / resolution_m).astype(np.int64) + 1
    occupied = np.zeros((int(shape_xy[1]), int(shape_xy[0])), dtype=bool)
    indices = np.floor((target_points - origin) / resolution_m).astype(np.int64)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < shape_xy[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < shape_xy[1])
    )
    occupied[indices[valid, 1], indices[valid, 0]] = True
    distance_field = ndimage.distance_transform_edt(~occupied) * resolution_m
    return distance_field.astype(np.float32), origin, resolution_m


def _csm_search_stage(
    source_points: np.ndarray,
    distance_field: np.ndarray,
    origin: np.ndarray,
    resolution_m: float,
    seed: np.ndarray,
    *,
    xy_window_m: float,
    xy_step_m: float,
    yaw_window_degrees: float,
    yaw_step_degrees: float,
    keep: int,
    agent_local_pose: tuple[float, float, float] | None,
    bbox: list[float] | None,
) -> list[tuple[tuple[int, float, float], np.ndarray, int, float]]:
    seed_x = float(seed[0, 3])
    seed_y = float(seed[1, 3])
    seed_yaw = _yaw_degrees(seed)
    offsets = np.arange(-xy_window_m, xy_window_m + 1e-9, xy_step_m)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="xy")
    translations = np.column_stack(
        (
            seed_x + offset_x.ravel(),
            seed_y + offset_y.ravel(),
        )
    )
    yaws = np.arange(
        seed_yaw - yaw_window_degrees,
        seed_yaw + yaw_window_degrees + 1e-9,
        yaw_step_degrees,
    )
    rows: list[tuple[tuple[int, float, float], np.ndarray, int, float]] = []
    clip = float(RERANK_DIST)
    for yaw_degrees in yaws:
        region_mask = _csm_region_mask(
            translations, float(yaw_degrees), source_points, agent_local_pose, bbox
        )
        if not np.any(region_mask):
            continue
        active_translations = translations[region_mask]
        yaw = math.radians(float(yaw_degrees))
        rotation_t = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        rotated_source = source_points @ rotation_t
        transformed = rotated_source[None, :, :] + active_translations[:, None, :]
        grid_indices = np.floor((transformed - origin) / resolution_m).astype(np.int64)
        valid = (
            (grid_indices[:, :, 0] >= 0)
            & (grid_indices[:, :, 0] < distance_field.shape[1])
            & (grid_indices[:, :, 1] >= 0)
            & (grid_indices[:, :, 1] < distance_field.shape[0])
        )
        distances = np.full(grid_indices.shape[:2], clip, dtype=np.float32)
        transform_indices, point_indices = np.where(valid)
        distances[transform_indices, point_indices] = np.minimum(
            distance_field[
                grid_indices[transform_indices, point_indices, 1],
                grid_indices[transform_indices, point_indices, 0],
            ],
            clip,
        )
        hit_counts = np.count_nonzero(distances < clip, axis=1)
        mean_distances = distances.mean(axis=1)
        local_keep = min(keep, len(hit_counts))
        if local_keep == 0:
            continue
        top_indices = np.argpartition(-hit_counts, local_keep - 1)[:local_keep]
        for index in top_indices:
            translation = active_translations[index]
            transform = _se2_from_xy_yaw(
                float(translation[0]), float(translation[1]), float(yaw_degrees)
            )
            closeness = (
                abs(float(translation[0]) - seed_x)
                + abs(float(translation[1]) - seed_y)
                + _yaw_delta_degrees(float(yaw_degrees), seed_yaw) * 0.01
            )
            key = (
                int(hit_counts[index]),
                -float(mean_distances[index]),
                -float(closeness),
            )
            row = (
                key,
                transform,
                int(hit_counts[index]),
                float(mean_distances[index]),
            )
            if len(rows) < keep:
                rows.append(row)
                rows.sort(key=lambda item: item[0])
            elif key > rows[0][0]:
                rows[0] = row
                rows.sort(key=lambda item: item[0])
    return sorted(rows, reverse=True, key=lambda item: item[0])


def _append_distinct_csm_candidate(
    rows: list[tuple[tuple[int, float, float], np.ndarray, int, float]],
    row: tuple[tuple[int, float, float], np.ndarray, int, float],
) -> None:
    _, transform, _, _ = row
    yaw = _yaw_degrees(transform)
    for _, existing, _, _ in rows:
        if (
            float(np.linalg.norm(transform[:2, 3] - existing[:2, 3]))
            < CSM_DISTINCT_TRANSLATION_M
            and _yaw_delta_degrees(yaw, _yaw_degrees(existing)) < CSM_DISTINCT_YAW_DEG
        ):
            return
    rows.append(row)


def _transforms_are_csm_distinct(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        float(np.linalg.norm(left[:2, 3] - right[:2, 3])) >= CSM_DISTINCT_TRANSLATION_M
        or _yaw_delta_degrees(_yaw_degrees(left), _yaw_degrees(right))
        >= CSM_DISTINCT_YAW_DEG
    )


def _append_distinct_csm_seed(seeds: list[np.ndarray], seed: np.ndarray) -> None:
    if all(_transforms_are_csm_distinct(seed, existing) for existing in seeds):
        seeds.append(seed)


def _expand_sparse_csm_candidates(
    ransac_candidates: list[np.ndarray],
    source_points: np.ndarray,
    target_points: np.ndarray,
    agent_local_pose: tuple[float, float, float] | None,
    bbox: list[float] | None,
) -> list[np.ndarray]:
    if ndimage is None or not ransac_candidates:
        return []
    distance_field, origin, resolution_m = _build_csm_distance_field(
        target_points, CSM_GRID_RESOLUTION_M
    )
    rows: list[tuple[tuple[int, float, float], np.ndarray, int, float]] = []
    for seed in ransac_candidates:
        coarse_rows = _csm_search_stage(
            source_points,
            distance_field,
            origin,
            resolution_m,
            seed,
            xy_window_m=CSM_COARSE_XY_WINDOW_M,
            xy_step_m=CSM_COARSE_XY_STEP_M,
            yaw_window_degrees=CSM_COARSE_YAW_WINDOW_DEG,
            yaw_step_degrees=CSM_COARSE_YAW_STEP_DEG,
            keep=CSM_COARSE_KEEP_PER_SEED,
            agent_local_pose=agent_local_pose,
            bbox=bbox,
        )
        for _, coarse_transform, _, _ in coarse_rows[:CSM_FINE_CENTERS_PER_SEED]:
            fine_rows = _csm_search_stage(
                source_points,
                distance_field,
                origin,
                resolution_m,
                coarse_transform,
                xy_window_m=CSM_FINE_XY_WINDOW_M,
                xy_step_m=CSM_FINE_XY_STEP_M,
                yaw_window_degrees=CSM_FINE_YAW_WINDOW_DEG,
                yaw_step_degrees=CSM_FINE_YAW_STEP_DEG,
                keep=CSM_FINE_KEEP_PER_CENTER,
                agent_local_pose=agent_local_pose,
                bbox=bbox,
            )
            for row in fine_rows:
                _append_distinct_csm_candidate(rows, row)
    rows = sorted(rows, reverse=True, key=lambda item: item[0])[:CSM_MAX_CANDIDATES]
    return [transform for _, transform, _, _ in rows]


def _expand_quadrant_candidates(
    candidates: list[np.ndarray], source_points: np.ndarray
) -> list[np.ndarray]:
    center = source_points.mean(axis=0)
    return [
        _rotate_candidate_about_source_pivot(candidate, center, yaw_degrees)
        for candidate in candidates
        for yaw_degrees in QUADRANT_YAW_DEGREES
    ]


def _expand_sparse_yaw_refinement_candidates(
    candidates: list[np.ndarray],
    source_points: np.ndarray,
    agent_local_pose: tuple[float, float, float] | None,
) -> list[np.ndarray]:
    pivot = (
        np.array([agent_local_pose[0], agent_local_pose[1]], dtype=np.float64)
        if agent_local_pose is not None
        else source_points.mean(axis=0)
    )
    return [
        _rotate_candidate_about_source_pivot(candidate, pivot, yaw_degrees)
        for candidate in candidates
        for yaw_degrees in SPARSE_YAW_REFINEMENT_DEGREES
    ]


def _geometric_score(metrics: dict[str, Any]) -> float:
    occupied_match = metrics["occupied_match_ratio"]
    occupied_conflict = metrics["occupied_vs_free_conflict_ratio"]
    free_conflict = metrics["free_vs_occupied_conflict_ratio"]
    occupied_match = 0.0 if occupied_match is None else occupied_match
    occupied_conflict = 1.0 if occupied_conflict is None else occupied_conflict
    free_conflict = 1.0 if free_conflict is None else free_conflict
    unique_support = min(metrics["unique_target_support_count"] / 100.0, 1.0)
    supported_bins = metrics["support_trajectory_bins"]["supported_bin_count"]
    return float(
        occupied_match
        - 1.5 * occupied_conflict
        - 2.0 * free_conflict
        + 0.10 * unique_support
        + 0.05 * min(supported_bins / 4.0, 1.0)
    )


def _candidate_gate_failures(
    metrics: dict[str, Any],
    *,
    sparse_node_prior: bool,
    fitness: float,
    rmse: float,
    inliers: int,
    region_ok: bool,
) -> tuple[list[str], list[str], list[str]]:
    hard_rejections: list[str] = []
    positive_evidence_failures: list[str] = []
    diagnostic_failures: list[str] = []
    occupied_conflict = metrics["occupied_vs_free_conflict_ratio"]
    free_conflict = metrics["free_vs_occupied_conflict_ratio"]
    if (
        occupied_conflict is not None
        and occupied_conflict > MAX_OCCUPIED_VS_FREE_CONFLICT_RATIO
    ):
        hard_rejections.append("occupied_vs_free_conflict")
    if (
        free_conflict is not None
        and free_conflict > MAX_FREE_VS_OCCUPIED_CONFLICT_RATIO
    ):
        hard_rejections.append("free_vs_occupied_conflict")

    if sparse_node_prior:
        min_occupied = SPARSE_MIN_COMPARABLE_OCCUPIED
        min_free = SPARSE_MIN_COMPARABLE_FREE
        min_unique = SPARSE_MIN_UNIQUE_TARGET_SUPPORT
        min_inliers = SPARSE_DIAGNOSTIC_MIN_INLIERS
        max_rmse = SPARSE_DIAGNOSTIC_MAX_RMSE_M
    else:
        min_occupied = ACCUMULATED_MIN_COMPARABLE_OCCUPIED
        min_free = ACCUMULATED_MIN_COMPARABLE_FREE
        min_unique = ACCUMULATED_MIN_UNIQUE_TARGET_SUPPORT
        min_inliers = ACCUMULATED_DIAGNOSTIC_MIN_INLIERS
        max_rmse = ACCUMULATED_DIAGNOSTIC_MAX_RMSE_M

    counts = metrics["counts"]
    if counts["occupied_comparable"] < min_occupied:
        positive_evidence_failures.append("insufficient_comparable_occupied")
    if counts["free_comparable"] < min_free:
        positive_evidence_failures.append("insufficient_comparable_free")
    occupied_match = metrics["occupied_match_ratio"]
    if occupied_match is None or occupied_match < MIN_OCCUPIED_MATCH_RATIO:
        positive_evidence_failures.append("low_occupied_match_ratio")
    if metrics["unique_target_support_count"] < min_unique:
        positive_evidence_failures.append("insufficient_unique_target_support")
    if (
        metrics["support_trajectory_bins"]["supported_bin_count"]
        < MIN_SUPPORTED_TRAJECTORY_BINS
    ):
        positive_evidence_failures.append("insufficient_trajectory_bin_support")

    if fitness < DIAGNOSTIC_MIN_FITNESS:
        diagnostic_failures.append("low_fitness")
    if not math.isfinite(rmse) or rmse > max_rmse:
        diagnostic_failures.append("high_rmse")
    if inliers < min_inliers:
        diagnostic_failures.append("insufficient_raw_inliers")
    if not region_ok:
        diagnostic_failures.append("outside_prior_region")
    return hard_rejections, positive_evidence_failures, diagnostic_failures


def _candidate_preselection(
    candidates: list[np.ndarray],
    source_fine: Any,
    target_fine: Any,
    geometry_context: dict[str, Any],
    limit: int,
) -> list[tuple[float, np.ndarray]]:
    rows = []
    for candidate in candidates:
        fitness = _score(source_fine, target_fine, candidate)[0]
        geometry_score = _geometric_score(
            _geometric_metrics(geometry_context, candidate)
        )
        rows.append((fitness, geometry_score, candidate))
    by_fitness = sorted(rows, key=lambda row: row[0], reverse=True)
    by_geometry = sorted(rows, key=lambda row: row[1], reverse=True)
    selected: list[tuple[float, np.ndarray]] = []
    interleaved = []
    for index in range(max(len(by_geometry), len(by_fitness))):
        if index < len(by_geometry):
            interleaved.append(by_geometry[index])
        if index < len(by_fitness):
            interleaved.append(by_fitness[index])
    for fitness, _, candidate in interleaved:
        if any(
            not _transforms_are_csm_distinct(candidate, existing)
            for _, existing in selected
        ):
            continue
        selected.append((fitness, candidate))
        if len(selected) >= limit:
            break
    return selected


def relocalize(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    profile: str = "accumulated_points",
    prior: dict[str, Any] | None = None,
    source_representation: str = "raw_endpoints",
    source_grid: dict[str, np.ndarray] | None = None,
    target_free_points: np.ndarray | None = None,
) -> dict[str, Any]:
    sparse_node_prior = profile == "sparse_node_prior"
    bbox = _region_bbox_from_prior(prior) if sparse_node_prior else None
    agent_local_pose = (
        _agent_local_pose_from_prior(prior) if sparse_node_prior else None
    )
    original_source_count = len(source_points)
    original_target_count = len(target_points)
    crop_meta: dict[str, Any] = {
        "source_region_bbox": None,
        "target_region_bbox": bbox,
    }
    if sparse_node_prior:
        source_points, target_points, crop_meta = _crop_sparse_node_prior_points(
            source_points,
            target_points,
            bbox,
            agent_local_pose,
        )
        if target_free_points is not None:
            target_free_points = _filter_points_to_bbox(target_free_points, bbox)
    min_source_points = SPARSE_MIN_SOURCE_POINTS if sparse_node_prior else 600
    min_target_points = SPARSE_MIN_TARGET_POINTS if sparse_node_prior else 600
    if len(source_points) < min_source_points or len(target_points) < min_target_points:
        return {
            "accepted": False,
            "reason": "insufficient_points",
            "profile": profile,
            "source_representation": source_representation,
            "source_point_count": len(source_points),
            "target_point_count": len(target_points),
            "original_source_point_count": original_source_count,
            "original_target_point_count": original_target_count,
            "region_bbox": bbox,
            **crop_meta,
        }
    if source_grid is None or target_free_points is None:
        return {
            "accepted": False,
            "reason": "geometric_metrics_unavailable",
            "profile": profile,
            "source_representation": source_representation,
            "source_point_count": len(source_points),
            "target_point_count": len(target_points),
            "original_source_point_count": original_source_count,
            "original_target_point_count": original_target_count,
            "region_bbox": bbox,
            **crop_meta,
        }
    geometry_context = _prepare_geometry_context(
        source_grid, target_points, target_free_points
    )
    preselection_geometry_context = _sample_geometry_context(geometry_context)
    source = _cloud(source_points)
    target = _cloud(target_points)
    source_fine, _ = _preprocess(source, FINE_VOXEL)
    target_fine, _ = _preprocess(target, FINE_VOXEL)
    candidates: list[np.ndarray] = []
    for voxel_size, runs in SCALE_PLAN:
        source_down, source_fpfh = _preprocess(source, voxel_size)
        target_down, target_fpfh = _preprocess(target, voxel_size)
        if len(source_down.points) < 20 or len(target_down.points) < 20:
            continue
        for _ in range(runs):
            candidates.append(
                _ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            )
    if not candidates:
        return {
            "accepted": False,
            "reason": "no_ransac_candidates",
            "profile": profile,
            "source_representation": source_representation,
        }
    ransac_candidate_count = len(candidates)
    candidates = _expand_quadrant_candidates(candidates, source_points)
    quadrant_candidate_count = len(candidates)
    yaw_refined_candidate_count = 0
    if sparse_node_prior:
        candidates = _expand_sparse_yaw_refinement_candidates(
            candidates,
            source_points,
            agent_local_pose,
        )
        yaw_refined_candidate_count = len(candidates)
    icp_rerank_limit = (
        SPARSE_ICP_CANDIDATE_LIMIT
        if sparse_node_prior
        else ACCUMULATED_ICP_CANDIDATE_LIMIT
    )
    ranked = _candidate_preselection(
        candidates,
        source_fine,
        target_fine,
        preselection_geometry_context,
        icp_rerank_limit,
    )
    if sparse_node_prior:
        csm_seed_candidates: list[np.ndarray] = []
        for _, candidate in ranked[:CSM_SEED_LIMIT]:
            _append_distinct_csm_seed(csm_seed_candidates, candidate)
        candidates.extend(
            _expand_sparse_csm_candidates(
                csm_seed_candidates,
                source_points,
                target_points,
                agent_local_pose,
                bbox,
            )
        )
        ranked = _candidate_preselection(
            candidates,
            source_fine,
            target_fine,
            preselection_geometry_context,
            icp_rerank_limit,
        )
    polished: list[dict[str, Any]] = []
    estimation = _REG.TransformationEstimationPointToPoint(False)
    for _, initial in ranked:
        result = _REG.registration_icp(
            source_fine,
            target_fine,
            RERANK_DIST,
            initial,
            estimation,
            _REG.ICPConvergenceCriteria(max_iteration=70),
        )
        transform = _se2_transform(np.asarray(result.transformation))
        fitness, rmse, inliers, unique_target_inliers = _score(
            source_fine, target_fine, transform
        )
        region_ok = (
            _transformed_agent_pose_inside_bbox(transform, agent_local_pose, bbox)
            if sparse_node_prior and agent_local_pose is not None
            else _transformed_centroid_inside_bbox(source_points, transform, bbox)
        )
        geometric_metrics = _geometric_metrics(geometry_context, transform)
        geometric_score = _geometric_score(geometric_metrics)
        hard_rejections, positive_failures, diagnostic_failures = (
            _candidate_gate_failures(
                geometric_metrics,
                sparse_node_prior=sparse_node_prior,
                fitness=fitness,
                rmse=rmse,
                inliers=inliers,
                region_ok=region_ok,
            )
        )
        polished.append(
            {
                "fitness": fitness,
                "rmse_m": rmse,
                "inlier_count": inliers,
                "unique_target_inlier_count": unique_target_inliers,
                "transform": transform,
                "candidate_region_ok": region_ok,
                "geometric_metrics": geometric_metrics,
                "geometric_score": geometric_score,
                "hard_rejections": hard_rejections,
                "positive_evidence_failures": positive_failures,
                "diagnostic_failures": diagnostic_failures,
            }
        )
    polished.sort(
        key=lambda item: (
            not item["hard_rejections"]
            and not item["positive_evidence_failures"]
            and not item["diagnostic_failures"],
            not item["hard_rejections"],
            item["geometric_score"],
        ),
        reverse=True,
    )
    best_row = polished[0]
    best = best_row["transform"]
    best_fitness = best_row["fitness"]
    best_rmse = best_row["rmse_m"]
    best_inliers = best_row["inlier_count"]
    best_unique_target_inliers = best_row["unique_target_inlier_count"]
    runner_up = next(
        (
            item
            for item in polished[1:]
            if _distinct(best, item["transform"])
            and not item["hard_rejections"]
            and not item["positive_evidence_failures"]
            and not item["diagnostic_failures"]
        ),
        None,
    )
    runner_up_fitness = runner_up["fitness"] if runner_up else None
    runner_up_geometric_score = (
        runner_up["geometric_score"] if runner_up else None
    )
    geometric_score_margin = (
        best_row["geometric_score"] - runner_up_geometric_score
        if runner_up_geometric_score is not None
        else None
    )
    ambiguity_failure = (
        geometric_score_margin is not None
        and geometric_score_margin < MIN_GEOMETRIC_SCORE_MARGIN
    )
    region_ok = best_row["candidate_region_ok"]
    transformed_agent_pose = _transformed_agent_pose(best, agent_local_pose)
    accepted = (
        not best_row["hard_rejections"]
        and not best_row["positive_evidence_failures"]
        and not best_row["diagnostic_failures"]
        and not ambiguity_failure
    )
    if accepted:
        reason = "accepted"
    elif best_row["hard_rejections"]:
        reason = "geometric_conflict"
    elif best_row["positive_evidence_failures"]:
        reason = "insufficient_positive_evidence"
    elif best_row["diagnostic_failures"]:
        reason = "failed_refinement_diagnostics"
    else:
        reason = "geometrically_ambiguous"

    candidate_summaries = []
    for item in polished:
        candidate_summaries.append(
            {
                "geometric_score": item["geometric_score"],
                "fitness": item["fitness"],
                "rmse_m": item["rmse_m"],
                "inlier_count": item["inlier_count"],
                "candidate_region_ok": item["candidate_region_ok"],
                "hard_rejections": item["hard_rejections"],
                "positive_evidence_failures": item[
                    "positive_evidence_failures"
                ],
                "diagnostic_failures": item["diagnostic_failures"],
                "geometric_metrics": item["geometric_metrics"],
                "transform_old_from_local": item["transform"].tolist(),
            }
        )
    return {
        "accepted": accepted,
        "reason": reason,
        "profile": profile,
        "source_representation": source_representation,
        "threshold_profile": (
            "sparse_node_prior" if sparse_node_prior else "accumulated_points"
        ),
        "fitness": best_fitness,
        "rmse_m": best_rmse,
        "inlier_count": best_inliers,
        "unique_target_inlier_count": best_unique_target_inliers,
        "runner_up_fitness": runner_up_fitness,
        "geometric_score": best_row["geometric_score"],
        "runner_up_geometric_score": runner_up_geometric_score,
        "geometric_score_margin": geometric_score_margin,
        "geometric_metrics": best_row["geometric_metrics"],
        "hard_rejections": best_row["hard_rejections"],
        "positive_evidence_failures": best_row["positive_evidence_failures"],
        "diagnostic_failures": best_row["diagnostic_failures"],
        "candidate_evaluations": candidate_summaries,
        "ransac_candidate_count": ransac_candidate_count,
        "quadrant_candidate_count": quadrant_candidate_count,
        "yaw_refined_candidate_count": yaw_refined_candidate_count,
        "candidate_count": len(candidates),
        "icp_rerank_count": len(ranked),
        "region_bbox": bbox,
        "original_source_point_count": original_source_count,
        "original_target_point_count": original_target_count,
        **crop_meta,
        "candidate_region_ok": region_ok,
        "agent_local_pose": (
            {
                "x": agent_local_pose[0],
                "y": agent_local_pose[1],
                "yaw_rad": agent_local_pose[2],
                "sequence": prior.get("agent_local_pose_sequence")
                if isinstance(prior, dict)
                else None,
            }
            if agent_local_pose is not None
            else None
        ),
        "candidate_agent_pose_old": (
            {
                "x": transformed_agent_pose[0],
                "y": transformed_agent_pose[1],
                "yaw_rad": transformed_agent_pose[2],
            }
            if transformed_agent_pose is not None
            else None
        ),
        "source_point_count": len(source_points),
        "target_point_count": len(target_points),
        "acceptance_thresholds": {
            "max_occupied_vs_free_conflict_ratio": (
                MAX_OCCUPIED_VS_FREE_CONFLICT_RATIO
            ),
            "max_free_vs_occupied_conflict_ratio": (
                MAX_FREE_VS_OCCUPIED_CONFLICT_RATIO
            ),
            "min_occupied_match_ratio": MIN_OCCUPIED_MATCH_RATIO,
            "min_supported_trajectory_bins": MIN_SUPPORTED_TRAJECTORY_BINS,
            "min_geometric_score_margin": MIN_GEOMETRIC_SCORE_MARGIN,
            "min_comparable_occupied": (
                SPARSE_MIN_COMPARABLE_OCCUPIED
                if sparse_node_prior
                else ACCUMULATED_MIN_COMPARABLE_OCCUPIED
            ),
            "min_comparable_free": (
                SPARSE_MIN_COMPARABLE_FREE
                if sparse_node_prior
                else ACCUMULATED_MIN_COMPARABLE_FREE
            ),
            "min_unique_target_support": (
                SPARSE_MIN_UNIQUE_TARGET_SUPPORT
                if sparse_node_prior
                else ACCUMULATED_MIN_UNIQUE_TARGET_SUPPORT
            ),
            "diagnostic_min_fitness": DIAGNOSTIC_MIN_FITNESS,
            "diagnostic_max_rmse_m": (
                SPARSE_DIAGNOSTIC_MAX_RMSE_M
                if sparse_node_prior
                else ACCUMULATED_DIAGNOSTIC_MAX_RMSE_M
            ),
            "diagnostic_min_inliers": (
                SPARSE_DIAGNOSTIC_MIN_INLIERS
                if sparse_node_prior
                else ACCUMULATED_DIAGNOSTIC_MIN_INLIERS
            ),
            "occupied_match_tolerance_m": OCCUPIED_MATCH_TOLERANCE_M,
            "target_cell_tolerance_m": TARGET_CELL_TOLERANCE_M,
            "free_occupied_conflict_tolerance_m": (
                FREE_OCCUPIED_CONFLICT_TOLERANCE_M
            ),
            "source_free_clearance_m": SOURCE_FREE_CLEARANCE_M,
        },
        "transform_old_from_local": best.tolist(),
        **(
            {"yaw_refinement_degrees": list(SPARSE_YAW_REFINEMENT_DEGREES)}
            if sparse_node_prior
            else {}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-open3d", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--source-grid", type=Path)
    parser.add_argument("--target-free", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        choices=("accumulated_points", "sparse_node_prior"),
        default="accumulated_points",
    )
    parser.add_argument(
        "--source-representation",
        choices=(
            "raw_endpoints",
            "active_trajectory_occupancy",
            "recent_trajectory_raw_endpoints",
            "recent_trajectory_occupancy",
        ),
        default="raw_endpoints",
    )
    parser.add_argument("--source-window-length-m", type=float)
    parser.add_argument("--source-window-actual-length-m", type=float)
    parser.add_argument("--source-window-first-sequence", type=int)
    parser.add_argument("--source-window-last-sequence", type=int)
    parser.add_argument("--source-window-scan-count", type=int)
    parser.add_argument("--source-window-return-point-count", type=int)
    parser.add_argument("--source-window-miss-point-count", type=int)
    parser.add_argument("--source-window-observed-cell-count", type=int)
    parser.add_argument("--source-window-occupied-cell-count", type=int)
    parser.add_argument("--source-window-free-cell-count", type=int)
    parser.add_argument(
        "--source-window-occupancy-probability-threshold", type=float
    )
    parser.add_argument("--source-window-free-probability-threshold", type=float)
    parser.add_argument("--source-window-occupancy-build-duration-ms", type=float)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--agent-local-pose", nargs=3, type=float)
    parser.add_argument("--agent-local-sequence", type=int)
    args = parser.parse_args()
    if args.check_open3d:
        return 0 if o3d is not None else 1
    if args.source is None or args.target is None or args.output is None:
        parser.error("--source, --target, and --output are required")
    try:
        prior = None
        if args.prior is not None:
            prior = json.loads(args.prior.read_text(encoding="utf-8"))
            if not isinstance(prior, dict):
                prior = None
        if prior is not None and args.agent_local_pose is not None:
            prior["agent_local_pose"] = {
                "x": args.agent_local_pose[0],
                "y": args.agent_local_pose[1],
                "yaw_rad": args.agent_local_pose[2],
            }
            if args.agent_local_sequence is not None:
                prior["agent_local_pose_sequence"] = args.agent_local_sequence
        result = relocalize(
            _read_points(args.source),
            _read_points(args.target),
            profile=args.profile,
            prior=prior,
            source_representation=args.source_representation,
            source_grid=(
                _read_source_grid(args.source_grid)
                if args.source_grid is not None
                else None
            ),
            target_free_points=(
                _read_points(args.target_free)
                if args.target_free is not None
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "accepted": False,
            "reason": "sidecar_error",
            "error": str(exc),
            "profile": args.profile,
            "source_representation": args.source_representation,
        }
    if args.source_window_length_m is not None:
        result["source_window"] = {
            "requested_length_m": args.source_window_length_m,
            "actual_length_m": args.source_window_actual_length_m,
            "first_sequence": args.source_window_first_sequence,
            "last_sequence": args.source_window_last_sequence,
            "scan_count": args.source_window_scan_count,
            "return_point_count": args.source_window_return_point_count,
            "miss_point_count": args.source_window_miss_point_count,
            "observed_cell_count": args.source_window_observed_cell_count,
            "occupied_cell_count": args.source_window_occupied_cell_count,
            "free_cell_count": args.source_window_free_cell_count,
            "occupancy_probability_threshold": (
                args.source_window_occupancy_probability_threshold
            ),
            "free_probability_threshold": (
                args.source_window_free_probability_threshold
            ),
            "occupancy_build_duration_ms": (
                args.source_window_occupancy_build_duration_ms
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as temporary:
        json.dump(result, temporary, separators=(",", ":"))
        temporary.write("\n")
    os.replace(temporary.name, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
