from __future__ import annotations

import heapq
import json
import math
import re
import importlib.util
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SlamOccupancyMap:
    image_path: Path
    render_metadata_path: Path
    grid: np.ndarray
    render_metadata: Mapping[str, Any]
    sequence: int | None

    @property
    def resolution(self) -> float:
        return float(self.render_metadata.get("resolution", 0.05))

    @property
    def origin_x(self) -> float:
        return float(self.render_metadata["mosaic_origin_x"])

    @property
    def origin_y(self) -> float:
        return float(self.render_metadata["mosaic_origin_y"])

    def map_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(round(float(x) / self.resolution + self.origin_x))
        row = int(round(self.origin_y - float(y) / self.resolution))
        return row, col

    def grid_to_map(self, row: int, col: int) -> tuple[float, float]:
        x = (float(col) - self.origin_x) * self.resolution
        y = (self.origin_y - float(row)) * self.resolution
        return x, y


@dataclass(frozen=True)
class SlamPlan:
    reachable: bool
    status: str
    start_cell: tuple[int, int] | None = None
    goal_cell: tuple[int, int] | None = None
    snapped_start_cell: tuple[int, int] | None = None
    snapped_goal_cell: tuple[int, int] | None = None
    path_cells: list[tuple[int, int]] | None = None
    path_slam_points: list[tuple[float, float]] | None = None
    distance_m: float | None = None
    map_sequence: int | None = None
    start_clear_radius_m: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "status": self.status,
            "map_sequence": self.map_sequence,
            "path_cell_count": len(self.path_cells or []),
            "distance_m": None if self.distance_m is None else round(self.distance_m, 3),
            "start_clear_radius_m": (
                None
                if self.start_clear_radius_m is None
                else round(float(self.start_clear_radius_m), 3)
            ),
        }


@dataclass(frozen=True)
class SlamExecutorTurn:
    action: str
    degrees: float
    yaw_delta_deg: float
    remaining_yaw_error_deg: float
    arbitrary_turn: bool


@dataclass(frozen=True)
class CompiledSlamAction:
    action: str
    degrees: float | None = None
    distance: float | None = None
    target_yaw_rad: float | None = None
    yaw_delta_deg: float | None = None
    waypoint: tuple[float, float] | None = None

    def payload(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {"action": self.action}
        if self.degrees is not None:
            payload["degrees"] = self.degrees
        if self.distance is not None:
            payload["distance"] = self.distance
        return payload

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"action": self.action}
        if self.degrees is not None:
            result["degrees"] = round(float(self.degrees), 6)
        if self.distance is not None:
            result["distance"] = round(float(self.distance), 6)
        if self.target_yaw_rad is not None:
            result["target_yaw_rad"] = float(self.target_yaw_rad)
        if self.yaw_delta_deg is not None:
            result["yaw_delta_deg"] = round(float(self.yaw_delta_deg), 6)
        if self.waypoint is not None:
            result["waypoint"] = [float(self.waypoint[0]), float(self.waypoint[1])]
        return result


@dataclass(frozen=True)
class CompiledSlamRoute:
    status: str
    actions: list[CompiledSlamAction]
    simulated_poses: list[dict[str, float]]
    waypoints: list[tuple[float, float]]
    failure_reason: str | None = None
    failed_waypoint: tuple[float, float] | None = None

    @property
    def executable(self) -> bool:
        return self.status in {"compiled", "already_reached"}

    def summary(self) -> dict[str, Any]:
        forward_count = sum(1 for action in self.actions if action.action == "move_forward")
        turn_count = len(self.actions) - forward_count
        distance_m = sum(
            float(action.distance or 0.0)
            for action in self.actions
            if action.action == "move_forward"
        )
        final_pose = self.simulated_poses[-1] if self.simulated_poses else None
        result: dict[str, Any] = {
            "compile_status": self.status,
            "compile_failure_reason": self.failure_reason,
            "compiled_action_count": len(self.actions),
            "compiled_turn_count": turn_count,
            "compiled_forward_count": forward_count,
            "compiled_distance_m": round(distance_m, 3),
            "waypoint_count": len(self.waypoints),
            "final_simulated_pose": final_pose,
        }
        if self.failed_waypoint is not None:
            result["failed_waypoint"] = [
                float(self.failed_waypoint[0]),
                float(self.failed_waypoint[1]),
            ]
        if self.waypoints:
            result["active_waypoint"] = [
                float(self.waypoints[0][0]),
                float(self.waypoints[0][1]),
            ]
        return result


def select_heading_turn(
    yaw_delta_rad: float,
    *,
    turn_quantum_deg: float = 10.0,
    heading_tolerance_deg: float = 2.0,
    arbitrary_turn_enabled: bool = False,
) -> SlamExecutorTurn | None:
    """Choose a heading correction before moving along a SLAM occupancy path."""
    yaw_delta_deg = math.degrees(float(yaw_delta_rad))
    abs_delta = abs(yaw_delta_deg)
    tolerance = max(0.0, float(heading_tolerance_deg))
    if abs_delta <= tolerance:
        return None

    action = "turn_left" if yaw_delta_deg > 0.0 else "turn_right"
    if arbitrary_turn_enabled:
        degrees = abs_delta
        remaining = 0.0
    else:
        quantum = max(1e-6, float(turn_quantum_deg))
        steps = max(1, int(round(abs_delta / quantum)))
        degrees = steps * quantum
        remaining = yaw_delta_deg - math.copysign(degrees, yaw_delta_deg)
        if abs(remaining) >= abs_delta:
            return None
    return SlamExecutorTurn(
        action=action,
        degrees=degrees,
        yaw_delta_deg=yaw_delta_deg,
        remaining_yaw_error_deg=remaining,
        arbitrary_turn=bool(arbitrary_turn_enabled),
    )


def compile_slam_occupancy_route(
    slam_map: SlamOccupancyMap,
    plan: SlamPlan,
    costmap: np.ndarray,
    *,
    start_pose: Mapping[str, Any],
    goal: Mapping[str, Any] | tuple[float, float],
    goal_radius_m: float = 0.45,
    turn_quantum_deg: float = 10.0,
    forward_step_m: float = 0.25,
    heading_tolerance_deg: float = 2.0,
    max_actions: int = 4096,
) -> CompiledSlamRoute:
    """Compile a SLAM grid route into atomic simulator actions."""
    try:
        pose = {
            "x": float(start_pose["x"]),
            "y": float(start_pose["y"]),
            "yaw_rad": float(start_pose["yaw_rad"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return CompiledSlamRoute(
            status="unexecutable",
            actions=[],
            simulated_poses=[],
            waypoints=[],
            failure_reason=f"invalid_start_pose:{type(exc).__name__}",
        )
    if not plan.reachable or not plan.path_cells or not plan.path_slam_points:
        return CompiledSlamRoute(
            status="unreachable",
            actions=[],
            simulated_poses=[dict(pose)],
            waypoints=[],
            failure_reason=plan.status,
        )

    goal_xy = _xy(goal)
    if math.hypot(pose["x"] - goal_xy[0], pose["y"] - goal_xy[1]) <= goal_radius_m:
        return CompiledSlamRoute(
            status="already_reached",
            actions=[],
            simulated_poses=[dict(pose)],
            waypoints=[],
        )

    waypoints = _simplify_path_cells_by_los(slam_map, costmap, plan.path_cells)
    if not waypoints:
        return CompiledSlamRoute(
            status="unexecutable",
            actions=[],
            simulated_poses=[dict(pose)],
            waypoints=[],
            failure_reason="empty_waypoint_list",
        )
    if len(waypoints) > 1:
        waypoints = waypoints[1:]

    actions: list[CompiledSlamAction] = []
    simulated_poses: list[dict[str, float]] = [dict(pose)]
    waypoint_index = 0
    quantum = max(1e-6, float(turn_quantum_deg))
    forward_step = max(1e-6, float(forward_step_m))
    waypoint_radius = forward_step * 0.75
    tolerance = max(0.0, float(heading_tolerance_deg))
    max_action_count = max(1, int(max_actions))
    resimplify_seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while waypoint_index < len(waypoints):
        if len(actions) >= max_action_count:
            return CompiledSlamRoute(
                status="unexecutable",
                actions=actions,
                simulated_poses=simulated_poses,
                waypoints=waypoints,
                failure_reason="max_compiled_actions_exceeded",
                failed_waypoint=waypoints[waypoint_index],
            )
        if math.hypot(pose["x"] - goal_xy[0], pose["y"] - goal_xy[1]) <= goal_radius_m:
            return CompiledSlamRoute(
                status="compiled",
                actions=actions,
                simulated_poses=simulated_poses,
                waypoints=waypoints,
            )

        waypoint = waypoints[waypoint_index]
        distance_to_waypoint = math.hypot(
            waypoint[0] - pose["x"], waypoint[1] - pose["y"]
        )
        if distance_to_waypoint <= waypoint_radius:
            waypoint_index += 1
            continue

        target_yaw = math.atan2(waypoint[1] - pose["y"], waypoint[0] - pose["x"])
        yaw_delta = math.atan2(
            math.sin(target_yaw - pose["yaw_rad"]),
            math.cos(target_yaw - pose["yaw_rad"]),
        )
        yaw_delta_deg = math.degrees(yaw_delta)
        before_distance = distance_to_waypoint
        selected = _select_traversable_quantized_forward(
            slam_map,
            costmap,
            pose,
            waypoint=waypoint,
            target_yaw=target_yaw,
            yaw_delta_deg=yaw_delta_deg,
            quantum=quantum,
            forward_step=forward_step,
            tolerance=tolerance,
            before_distance=before_distance,
        )
        if selected is None:
            current_cell = slam_map.map_to_grid(float(pose["x"]), float(pose["y"]))
            failed_cell = slam_map.map_to_grid(float(waypoint[0]), float(waypoint[1]))
            resimplify_key = (current_cell, failed_cell)
            if resimplify_key in resimplify_seen:
                return CompiledSlamRoute(
                    status="unexecutable",
                    actions=actions,
                    simulated_poses=simulated_poses,
                    waypoints=waypoints,
                    failure_reason="forward_sweep_blocked",
                    failed_waypoint=waypoint,
                )
            resimplify_seen.add(resimplify_key)
            rebuilt_waypoints = _resimplify_waypoints_from_pose(
                slam_map,
                costmap,
                plan.path_cells,
                pose,
            )
            if rebuilt_waypoints and rebuilt_waypoints != waypoints[waypoint_index:]:
                waypoints = rebuilt_waypoints
                waypoint_index = 0
                continue
            return CompiledSlamRoute(
                status="unexecutable",
                actions=actions,
                simulated_poses=simulated_poses,
                waypoints=waypoints,
                failure_reason="forward_sweep_blocked",
                failed_waypoint=waypoint,
            )

        signed_turn_steps, pose_after_turns, next_pose, remaining_yaw = selected
        if len(actions) + abs(signed_turn_steps) + 1 > max_action_count:
            return CompiledSlamRoute(
                status="unexecutable",
                actions=actions,
                simulated_poses=simulated_poses,
                waypoints=waypoints,
                failure_reason="max_compiled_actions_exceeded",
                failed_waypoint=waypoint,
            )
        turn_action = "turn_left" if signed_turn_steps > 0 else "turn_right"
        for _ in range(abs(signed_turn_steps)):
            action = CompiledSlamAction(
                action=turn_action,
                degrees=quantum,
                target_yaw_rad=target_yaw,
                yaw_delta_deg=yaw_delta_deg,
                waypoint=waypoint,
            )
            actions.append(action)
            pose = advance_slam_pose_by_action(
                pose,
                action=turn_action,
                degrees=quantum,
                distance_m=forward_step,
            )
            simulated_poses.append(dict(pose))
        pose = pose_after_turns
        actions.append(
            CompiledSlamAction(
                action="move_forward",
                distance=forward_step,
                target_yaw_rad=target_yaw,
                yaw_delta_deg=remaining_yaw,
                waypoint=waypoint,
            )
        )
        pose = {
            "x": float(next_pose["x"]),
            "y": float(next_pose["y"]),
            "yaw_rad": float(next_pose["yaw_rad"]),
        }
        simulated_poses.append(dict(pose))

    status = (
        "compiled"
        if math.hypot(pose["x"] - goal_xy[0], pose["y"] - goal_xy[1]) <= goal_radius_m
        else "partial"
    )
    return CompiledSlamRoute(
        status=status,
        actions=actions,
        simulated_poses=simulated_poses,
        waypoints=waypoints,
        failure_reason=None if status == "compiled" else "waypoints_exhausted_before_goal",
    )


def _select_traversable_quantized_forward(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    pose: Mapping[str, float],
    *,
    waypoint: tuple[float, float],
    target_yaw: float,
    yaw_delta_deg: float,
    quantum: float,
    forward_step: float,
    tolerance: float,
    before_distance: float,
) -> tuple[int, dict[str, float], dict[str, float], float] | None:
    for signed_steps in _nearby_quantized_turn_candidates(yaw_delta_deg, quantum):
        pose_after_turns = _advance_pose_by_signed_turn_steps(
            pose,
            signed_steps=signed_steps,
            quantum=quantum,
            forward_step=forward_step,
        )
        remaining_yaw = math.degrees(
            math.atan2(
                math.sin(target_yaw - pose_after_turns["yaw_rad"]),
                math.cos(target_yaw - pose_after_turns["yaw_rad"]),
            )
        )
        improving_turn_steps = int(round(abs(remaining_yaw) / quantum))
        if abs(remaining_yaw) > max(45.0, tolerance) and improving_turn_steps > 0:
            continue
        next_pose = advance_slam_pose_by_action(
            pose_after_turns,
            action="move_forward",
            distance_m=forward_step,
            degrees=quantum,
        )
        if not _swept_segment_is_traversable(
            slam_map,
            costmap,
            (pose_after_turns["x"], pose_after_turns["y"]),
            (next_pose["x"], next_pose["y"]),
        ):
            continue
        after_distance = math.hypot(
            waypoint[0] - next_pose["x"], waypoint[1] - next_pose["y"]
        )
        if after_distance > before_distance + slam_map.resolution:
            continue
        return signed_steps, pose_after_turns, next_pose, remaining_yaw
    return None


def _nearby_quantized_turn_candidates(
    yaw_delta_deg: float,
    quantum: float,
) -> list[int]:
    base_steps = int(round(float(yaw_delta_deg) / max(1e-6, float(quantum))))
    candidates = {0}
    for offset in range(-6, 7):
        candidates.add(base_steps + offset)
    return sorted(
        candidates,
        key=lambda steps: (
            abs(float(yaw_delta_deg) - float(steps) * float(quantum)),
            abs(steps),
            steps,
        ),
    )


def _advance_pose_by_signed_turn_steps(
    pose: Mapping[str, float],
    *,
    signed_steps: int,
    quantum: float,
    forward_step: float,
) -> dict[str, float]:
    result = {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "yaw_rad": float(pose["yaw_rad"]),
    }
    if signed_steps == 0:
        return result
    action = "turn_left" if signed_steps > 0 else "turn_right"
    for _ in range(abs(signed_steps)):
        result = advance_slam_pose_by_action(
            result,
            action=action,
            degrees=quantum,
            distance_m=forward_step,
        )
    return result


def _resimplify_waypoints_from_pose(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    path_cells: list[tuple[int, int]] | None,
    pose: Mapping[str, float],
) -> list[tuple[float, float]]:
    if not path_cells:
        return []
    current_cell = slam_map.map_to_grid(float(pose["x"]), float(pose["y"]))
    height, width = costmap.shape
    if not _in_bounds(current_cell, height, width):
        return []
    if not math.isfinite(float(costmap[current_cell])):
        return []
    nearest_index = min(
        range(len(path_cells)),
        key=lambda index: (
            (path_cells[index][0] - current_cell[0]) ** 2
            + (path_cells[index][1] - current_cell[1]) ** 2
        ),
    )
    rebuilt_cells = [current_cell]
    for cell in path_cells[nearest_index:]:
        if cell != rebuilt_cells[-1]:
            rebuilt_cells.append(cell)
    return _simplify_path_cells_by_los(slam_map, costmap, rebuilt_cells)


def derive_slam_region_bbox(
    artifact_dir_or_image: str | Path,
    *,
    slam_pose: Mapping[str, Any],
    padding_m: float = 0.5,
    merge_radius_m: float = 0.5,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
) -> dict[str, Any] | None:
    """Derive a region bbox from the segmented free-space area around ``slam_pose``."""
    try:
        slam_map = load_slam_occupancy_map(artifact_dir_or_image)
        try:
            from .area_segmentation import analyze_slam_occupancy
        except Exception:
            module_name = "_supernav_slam_area_segmentation_helper"
            module = sys.modules.get(module_name)
            if module is None:
                module_path = Path(__file__).with_name("area_segmentation.py")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            analyze_slam_occupancy = getattr(module, "analyze_slam_occupancy")
    except Exception:
        return None

    try:
        pose_x = float(slam_pose["x"])
        pose_y = float(slam_pose["y"])
    except (KeyError, TypeError, ValueError):
        return None

    try:
        region_data, _, area_grid = analyze_slam_occupancy(
            slam_map.image_path,
            min_area_px=min_area_px,
            min_frontier_px=min_frontier_px,
        )
    except Exception:
        return None

    areas = region_data.get("areas")
    if not isinstance(areas, list) or not areas:
        return None

    area_by_id: dict[int, dict[str, Any]] = {}
    for area in areas:
        if not isinstance(area, dict):
            continue
        try:
            area_id = int(area["id"])
            bbox_px = area["bbox"]
            if not isinstance(bbox_px, list) or len(bbox_px) != 4:
                continue
            area_by_id[area_id] = area
        except (KeyError, TypeError, ValueError):
            continue
    if not area_by_id:
        return None

    node_row, node_col = slam_map.map_to_grid(pose_x, pose_y)
    primary_area_id: int | None = None
    if (
        0 <= node_row < area_grid.shape[0]
        and 0 <= node_col < area_grid.shape[1]
        and int(area_grid[node_row, node_col]) > 0
    ):
        primary_area_id = int(area_grid[node_row, node_col])
    else:
        primary_area_id = _nearest_area_id_to_point(
            area_by_id,
            slam_map,
            pose_x,
            pose_y,
        )
    if primary_area_id is None:
        return None

    included_area_ids = _merge_nearby_area_ids(
        primary_area_id,
        area_by_id,
        slam_map,
        node_x=pose_x,
        node_y=pose_y,
        merge_radius_m=max(0.0, float(merge_radius_m)),
    )
    if not included_area_ids:
        return None

    bbox_px = _union_area_pixel_bbox(area_by_id[area_id]["bbox"] for area_id in included_area_ids)
    bbox_m = _pixel_bbox_to_map_bbox(
        bbox_px,
        resolution=slam_map.resolution,
        origin_x=slam_map.origin_x,
        origin_y=slam_map.origin_y,
        padding_m=max(0.0, float(padding_m)),
    )
    if bbox_m is None:
        return None

    x0, y0, x1, y1 = bbox_m
    region_radius_m = max((x1 - x0) / 2.0, (y1 - y0) / 2.0)
    primary_area = area_by_id[primary_area_id]
    return {
        "region_bbox": [x0, y0, x1, y1],
        "region_radius_m": region_radius_m,
        "region_geometry": {
            "source": "slam_area_segmentation",
            "primary_area_ids": [primary_area_id],
            "merged_area_ids": included_area_ids,
            "padding_m": max(0.0, float(padding_m)),
            "merge_radius_m": max(0.0, float(merge_radius_m)),
            "node_grid_rc": [node_row, node_col],
            "map_resolution_m": slam_map.resolution,
            "mosaic_origin_x": slam_map.origin_x,
            "mosaic_origin_y": slam_map.origin_y,
            "primary_area_bbox_px": primary_area.get("bbox"),
            "region_bbox_px": [bbox_px[0], bbox_px[1], bbox_px[2], bbox_px[3]],
        },
    }


def _nearest_area_id_to_point(
    area_by_id: Mapping[int, Mapping[str, Any]],
    slam_map: SlamOccupancyMap,
    x: float,
    y: float,
) -> int | None:
    best_area_id: int | None = None
    best_distance = float("inf")
    for area_id, area in area_by_id.items():
        bbox = area.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        distance = _point_to_bbox_distance_m(
            x,
            y,
            bbox,
            resolution=slam_map.resolution,
            origin_x=slam_map.origin_x,
            origin_y=slam_map.origin_y,
        )
        if distance < best_distance:
            best_distance = distance
            best_area_id = area_id
    return best_area_id


def _merge_nearby_area_ids(
    primary_area_id: int,
    area_by_id: Mapping[int, Mapping[str, Any]],
    slam_map: SlamOccupancyMap,
    *,
    node_x: float,
    node_y: float,
    merge_radius_m: float,
) -> list[int]:
    merged = {primary_area_id}
    if primary_area_id not in area_by_id:
        return []

    for area_id, area in area_by_id.items():
        if area_id == primary_area_id:
            continue
        bbox = area.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        distance = _point_to_bbox_distance_m(
            node_x,
            node_y,
            bbox,
            resolution=slam_map.resolution,
            origin_x=slam_map.origin_x,
            origin_y=slam_map.origin_y,
        )
        if distance <= merge_radius_m:
            merged.add(area_id)
    return sorted(merged)


def _union_area_pixel_bbox(
    bboxes: Any,
) -> tuple[int, int, int, int] | None:
    row0: int | None = None
    col0: int | None = None
    row1: int | None = None
    col1: int | None = None
    for bbox in bboxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox_row0 = int(bbox[0])
            bbox_col0 = int(bbox[1])
            bbox_row1 = int(bbox[2])
            bbox_col1 = int(bbox[3])
        except (TypeError, ValueError):
            continue
        row0 = bbox_row0 if row0 is None else min(row0, bbox_row0)
        col0 = bbox_col0 if col0 is None else min(col0, bbox_col0)
        row1 = bbox_row1 if row1 is None else max(row1, bbox_row1)
        col1 = bbox_col1 if col1 is None else max(col1, bbox_col1)
    if None in (row0, col0, row1, col1):
        return None
    return (row0, col0, row1, col1)


def _area_bbox_to_map_bbox(
    bbox: Any,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        row0 = int(bbox[0])
        col0 = int(bbox[1])
        row1 = int(bbox[2])
        col1 = int(bbox[3])
    except (TypeError, ValueError):
        return None
    x0 = (float(col0) - origin_x - 0.5) * resolution
    x1 = (float(col1) - origin_x + 0.5) * resolution
    y0 = (origin_y - float(row1) - 0.5) * resolution
    y1 = (origin_y - float(row0) + 0.5) * resolution
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _pixel_bbox_to_map_bbox(
    bbox: tuple[int, int, int, int] | None,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
    padding_m: float,
) -> list[float] | None:
    if bbox is None:
        return None
    row0, col0, row1, col1 = bbox
    x0 = (float(col0) - origin_x - 0.5) * resolution - padding_m
    x1 = (float(col1) - origin_x + 0.5) * resolution + padding_m
    y0 = (origin_y - float(row1) - 0.5) * resolution - padding_m
    y1 = (origin_y - float(row0) + 0.5) * resolution + padding_m
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _point_to_bbox_distance_m(
    x: float,
    y: float,
    bbox: list[int],
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> float:
    map_bbox = _area_bbox_to_map_bbox(
        bbox,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if map_bbox is None:
        return float("inf")
    x0, y0, x1, y1 = map_bbox
    dx = max(0.0, max(x0 - x, x - x1))
    dy = max(0.0, max(y0 - y, y - y1))
    return float(math.hypot(dx, dy))


def latest_slam_occupancy_paths(artifact_dir: str | Path) -> tuple[Path, Path, int | None]:
    root = Path(artifact_dir)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("occupancy_step*.amcl.pgm"):
        match = re.fullmatch(r"occupancy_step(\d+)\.amcl", path.stem)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        sequence, image_path = max(candidates, key=lambda item: item[0])
        return image_path, root / f"occupancy_step{sequence}.png.render.json", sequence
    image_path = root / "occupancy.png.amcl.pgm"
    return image_path, root / "occupancy.png.render.json", None


def load_slam_occupancy_map(
    artifact_dir_or_image: str | Path,
    *,
    render_metadata_path: str | Path | None = None,
) -> SlamOccupancyMap:
    path = Path(artifact_dir_or_image)
    sequence: int | None = None
    if path.is_dir():
        image_path, metadata_path, sequence = latest_slam_occupancy_paths(path)
    else:
        image_path = path
        if render_metadata_path is not None:
            metadata_path = Path(render_metadata_path)
        elif image_path.name == "occupancy.png.amcl.pgm":
            metadata_path = image_path.with_name("occupancy.png.render.json")
        else:
            match = re.fullmatch(r"occupancy_step(\d+)\.amcl\.pgm", image_path.name)
            if match:
                sequence = int(match.group(1))
                metadata_path = image_path.with_name(
                    f"occupancy_step{sequence}.png.render.json"
                )
            else:
                png_metadata_path = image_path.with_name(image_path.name + ".render.json")
                metadata_path = (
                    png_metadata_path
                    if png_metadata_path.is_file()
                    else image_path.with_suffix(".render.json")
                )
    with Image.open(image_path) as image:
        grid = np.asarray(image.convert("L"), dtype=np.uint8)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_width = int(metadata["source_width"])
    expected_height = int(metadata.get("source_height", metadata["display_width"]))
    if grid.shape != (expected_height, expected_width):
        raise ValueError(
            f"occupancy grid {grid.shape[1]}x{grid.shape[0]} does not match "
            f"render source {expected_width}x{expected_height}"
        )
    return SlamOccupancyMap(
        image_path=image_path,
        render_metadata_path=metadata_path,
        grid=grid,
        render_metadata=metadata,
        sequence=sequence,
    )


def build_navigation_costmap(
    slam_map: SlamOccupancyMap,
    *,
    inflation_radius_m: float = 0.15,
    unknown_cost: float = 80.0,
    gradient_cost: float = 18.0,
    allow_unknown: bool = False,
    clear_around: Mapping[str, Any] | tuple[float, float] | None = None,
    clear_radius_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    grid = slam_map.grid
    occupied = grid == 0
    unknown = grid == 205
    free = grid == 254
    traversable = free | (unknown if allow_unknown else np.zeros_like(unknown, dtype=bool))
    blocked_seed = ~traversable | occupied
    blocked = _inflate_occupied(
        blocked_seed, max(0, int(math.ceil(inflation_radius_m / slam_map.resolution)))
    )
    cost = np.full(grid.shape, np.inf, dtype=np.float32)
    cost[free] = 0.0
    if allow_unknown:
        cost[unknown] = float(unknown_cost)
    cost[blocked] = np.inf
    if clear_around is not None and clear_radius_m > 0.0:
        _clear_costmap_disk(
            slam_map,
            cost,
            blocked,
            center=clear_around,
            radius_m=clear_radius_m,
        )
    if gradient_cost > 0:
        distance = _distance_from_obstacles(blocked)
        finite = np.isfinite(cost)
        cost[finite] += np.maximum(0.0, gradient_cost - distance[finite] * 3.0)
    return cost, blocked


def plan_slam_occupancy_path(
    slam_map: SlamOccupancyMap,
    *,
    start: Mapping[str, Any] | tuple[float, float],
    goal: Mapping[str, Any] | tuple[float, float],
    inflation_radius_m: float = 0.15,
    unknown_cost: float = 80.0,
    gradient_cost: float = 18.0,
    max_snap_radius_m: float = 1.0,
    allow_unknown: bool = False,
    start_clear_radius_m: float = 0.0,
) -> tuple[SlamPlan, np.ndarray]:
    start_xy = _xy(start)
    start_clear_radius_m = max(0.0, float(start_clear_radius_m))
    costmap, _ = build_navigation_costmap(
        slam_map,
        inflation_radius_m=inflation_radius_m,
        unknown_cost=unknown_cost,
        gradient_cost=gradient_cost,
        allow_unknown=allow_unknown,
        clear_around=start_xy,
        clear_radius_m=start_clear_radius_m,
    )
    goal_xy = _xy(goal)
    start_cell = slam_map.map_to_grid(*start_xy)
    goal_cell = slam_map.map_to_grid(*goal_xy)
    max_radius_cells = max(1, int(math.ceil(max_snap_radius_m / slam_map.resolution)))
    snapped_start = snap_to_traversable(costmap, start_cell, max_radius_cells)
    snapped_goal = snap_to_traversable(costmap, goal_cell, max_radius_cells)
    if snapped_start is None:
        return (
            SlamPlan(
                reachable=False,
                status="start_not_traversable",
                start_cell=start_cell,
                goal_cell=goal_cell,
                map_sequence=slam_map.sequence,
                start_clear_radius_m=start_clear_radius_m,
            ),
            costmap,
        )
    if snapped_goal is None:
        return (
            SlamPlan(
                reachable=False,
                status="goal_not_traversable",
                start_cell=start_cell,
                goal_cell=goal_cell,
                snapped_start_cell=snapped_start,
                map_sequence=slam_map.sequence,
                start_clear_radius_m=start_clear_radius_m,
            ),
            costmap,
        )
    path_cells = astar_costmap(costmap, snapped_start, snapped_goal)
    if not path_cells:
        return (
            SlamPlan(
                reachable=False,
                status="unreachable",
                start_cell=start_cell,
                goal_cell=goal_cell,
                snapped_start_cell=snapped_start,
                snapped_goal_cell=snapped_goal,
                map_sequence=slam_map.sequence,
                start_clear_radius_m=start_clear_radius_m,
            ),
            costmap,
        )
    points = [slam_map.grid_to_map(row, col) for row, col in path_cells]
    distance_m = path_length(points)
    return (
        SlamPlan(
            reachable=True,
            status="reachable",
            start_cell=start_cell,
            goal_cell=goal_cell,
            snapped_start_cell=snapped_start,
            snapped_goal_cell=snapped_goal,
            path_cells=path_cells,
            path_slam_points=points,
            distance_m=distance_m,
            map_sequence=slam_map.sequence,
            start_clear_radius_m=start_clear_radius_m,
        ),
        costmap,
    )


def astar_costmap(
    costmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    height, width = costmap.shape
    if not (_in_bounds(start, height, width) and _in_bounds(goal, height, width)):
        return None
    if not (np.isfinite(costmap[start]) and np.isfinite(costmap[goal])):
        return None
    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (_octile(start, goal), 0.0, start))
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    scores: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _, score, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(parents, current)
        closed.add(current)
        row, col = current
        for drow, dcol, move_cost in directions:
            neighbor = (row + drow, col + dcol)
            if not _in_bounds(neighbor, height, width) or neighbor in closed:
                continue
            if drow != 0 and dcol != 0:
                side_a = (row + drow, col)
                side_b = (row, col + dcol)
                if not (
                    _in_bounds(side_a, height, width)
                    and _in_bounds(side_b, height, width)
                    and np.isfinite(costmap[side_a])
                    and np.isfinite(costmap[side_b])
                ):
                    continue
            cell_cost = float(costmap[neighbor])
            if not math.isfinite(cell_cost):
                continue
            tentative = score + move_cost * (1.0 + cell_cost / 100.0)
            if tentative >= scores.get(neighbor, float("inf")):
                continue
            scores[neighbor] = tentative
            parents[neighbor] = current
            priority = tentative + _octile(neighbor, goal)
            heapq.heappush(open_heap, (priority, tentative, neighbor))
    return None


def snap_to_traversable(
    costmap: np.ndarray, cell: tuple[int, int], max_radius_cells: int
) -> tuple[int, int] | None:
    height, width = costmap.shape
    if _in_bounds(cell, height, width) and np.isfinite(costmap[cell]):
        return cell
    queue: deque[tuple[int, int, int]] = deque([(cell[0], cell[1], 0)])
    visited = {cell}
    best: tuple[int, int] | None = None
    best_cost = float("inf")
    while queue:
        row, col, radius = queue.popleft()
        if radius > max_radius_cells:
            continue
        if _in_bounds((row, col), height, width) and np.isfinite(costmap[row, col]):
            value = float(costmap[row, col])
            if value < best_cost:
                best = (row, col)
                best_cost = value
                if value <= 1e-6:
                    return best
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + drow, col + dcol)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor[0], neighbor[1], radius + 1))
    return best


def path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:])
        )
    )


def select_lookahead_point(
    points: list[tuple[float, float]],
    *,
    distance_m: float = 0.75,
) -> tuple[float, float] | None:
    if not points:
        return None
    origin = points[0]
    for point in points[1:]:
        if math.hypot(point[0] - origin[0], point[1] - origin[1]) >= distance_m:
            return point
    return points[-1]


def advance_slam_pose_by_action(
    pose: Mapping[str, float],
    *,
    action: str,
    distance_m: float = 0.25,
    degrees: float = 10.0,
) -> dict[str, float]:
    """Predict the next SLAM-frame pose for one discrete movement command."""
    x = float(pose["x"])
    y = float(pose["y"])
    yaw = float(pose["yaw_rad"])
    if action == "turn_left":
        yaw += math.radians(degrees)
    elif action == "turn_right":
        yaw -= math.radians(degrees)
    elif action == "move_forward":
        x += distance_m * math.cos(yaw)
        y += distance_m * math.sin(yaw)
    elif action == "move_backward":
        x -= distance_m * math.cos(yaw)
        y -= distance_m * math.sin(yaw)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    result = dict(pose)
    result.update({"x": x, "y": y, "yaw_rad": yaw})
    return result


def render_costmap_debug(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    output_path: str | Path,
    *,
    plan: SlamPlan | None = None,
) -> None:
    finite = np.isfinite(costmap)
    image = np.zeros((*costmap.shape, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    if np.any(finite):
        clipped = np.clip(costmap[finite], 0, 100)
        shade = (255 - clipped * 1.6).astype(np.uint8)
        image[finite] = np.stack([shade, shade, shade], axis=1)
    image[slam_map.grid == 205] = (150, 170, 190)
    image[~finite] = (15, 15, 15)
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    if plan is not None and plan.path_cells:
        xy = [(col, row) for row, col in plan.path_cells]
        if len(xy) > 1:
            draw.line(xy, fill=(240, 70, 30), width=3)
        for cell, color in (
            (plan.snapped_start_cell or plan.start_cell, (20, 220, 80)),
            (plan.snapped_goal_cell or plan.goal_cell, (40, 110, 255)),
        ):
            if cell is None:
                continue
            row, col = cell
            draw.ellipse((col - 4, row - 4, col + 4, row + 4), fill=color)
    pil.save(output_path)


def render_slam_occupancy_plan_overlay(
    slam_map: SlamOccupancyMap,
    output_path: str | Path,
    *,
    plan: SlamPlan | None = None,
    compiled_route: CompiledSlamRoute | None = None,
    display_frame: bool = False,
) -> None:
    if display_frame:
        _render_display_frame_plan_overlay(
            slam_map, output_path, plan=plan, compiled_route=compiled_route
        )
        return
    image = np.zeros((*slam_map.grid.shape, 3), dtype=np.uint8)
    image[slam_map.grid == 254] = (246, 246, 246)
    image[slam_map.grid == 205] = (178, 188, 196)
    image[slam_map.grid == 0] = (18, 18, 18)
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    if plan is not None and plan.path_cells:
        xy = [(col, row) for row, col in plan.path_cells]
        if len(xy) > 1:
            draw.line(xy, fill=(240, 70, 30), width=3)
        for cell, color in (
            (plan.snapped_start_cell or plan.start_cell, (20, 220, 80)),
            (plan.snapped_goal_cell or plan.goal_cell, (40, 110, 255)),
        ):
            if cell is None:
                continue
            row, col = cell
            draw.ellipse((col - 4, row - 4, col + 4, row + 4), fill=color)
    if compiled_route is not None and len(compiled_route.simulated_poses) > 1:
        xy = []
        for pose in compiled_route.simulated_poses:
            row, col = slam_map.map_to_grid(float(pose["x"]), float(pose["y"]))
            xy.append((col, row))
        draw.line(xy, fill=(35, 185, 225), width=2)
    pil.save(output_path)


def _render_display_frame_plan_overlay(
    slam_map: SlamOccupancyMap,
    output_path: str | Path,
    *,
    plan: SlamPlan | None = None,
    compiled_route: CompiledSlamRoute | None = None,
) -> None:
    metadata = slam_map.render_metadata
    raw = np.zeros((*slam_map.grid.shape, 3), dtype=np.uint8)
    raw[slam_map.grid == 254] = (246, 246, 246)
    raw[slam_map.grid == 205] = (178, 188, 196)
    raw[slam_map.grid == 0] = (18, 18, 18)
    display = np.flipud(np.transpose(raw, (1, 0, 2)))
    output_size = int(metadata.get("output_size", 1000))
    rendered = Image.new("RGB", (output_size, output_size), (205, 205, 205))
    display_image = Image.fromarray(display, mode="RGB")
    scale = float(metadata.get("scale", 1.0))
    resized_width = max(1, int(round(display_image.width * scale)))
    resized_height = max(1, int(round(display_image.height * scale)))
    display_image = display_image.resize(
        (resized_width, resized_height), Image.Resampling.NEAREST
    )
    paste_x = int(
        round(
            float(metadata.get("offset_x", 0.0))
            + float(metadata.get("left_padding", 0.0)) * scale
        )
    )
    paste_y = int(
        round(
            float(metadata.get("offset_y", 0.0))
            + float(metadata.get("top_padding", 0.0)) * scale
        )
    )
    rendered.paste(display_image, (paste_x, paste_y))
    draw = ImageDraw.Draw(rendered)

    def display_xy(cell: tuple[int, int] | None) -> tuple[float, float] | None:
        if cell is None:
            return None
        row, col = cell
        display_x = float(row)
        display_y = float(slam_map.grid.shape[1] - 1 - col)
        return (
            float(metadata.get("offset_x", 0.0))
            + (float(metadata.get("left_padding", 0.0)) + display_x) * scale,
            float(metadata.get("offset_y", 0.0))
            + (float(metadata.get("top_padding", 0.0)) + display_y) * scale,
        )

    if plan is not None and plan.path_cells:
        xy = [
            point
            for cell in plan.path_cells
            if (point := display_xy(cell)) is not None
        ]
        if len(xy) > 1:
            draw.line(xy, fill=(240, 70, 30), width=max(3, int(round(3 * scale))))
        marker_radius = max(5, int(round(4 * scale)))
        for cell, color in (
            (plan.snapped_start_cell or plan.start_cell, (20, 220, 80)),
            (plan.snapped_goal_cell or plan.goal_cell, (40, 110, 255)),
        ):
            center = display_xy(cell)
            if center is None:
                continue
            x, y = center
            draw.ellipse(
                (
                    x - marker_radius,
                    y - marker_radius,
                    x + marker_radius,
                    y + marker_radius,
                ),
                fill=color,
            )
    if compiled_route is not None and len(compiled_route.simulated_poses) > 1:
        xy = []
        for pose in compiled_route.simulated_poses:
            point = display_xy(
                slam_map.map_to_grid(float(pose["x"]), float(pose["y"]))
            )
            if point is not None:
                xy.append(point)
        if len(xy) > 1:
            compiled_width = max(2, int(round(2 * scale)))
            draw.line(xy, fill=(35, 185, 225), width=compiled_width)
    rendered.save(output_path)


def _simplify_path_cells_by_los(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    path_cells: list[tuple[int, int]],
) -> list[tuple[float, float]]:
    if not path_cells:
        return []
    simplified = [path_cells[0]]
    anchor_index = 0
    while anchor_index < len(path_cells) - 1:
        next_index = anchor_index + 1
        for candidate_index in range(len(path_cells) - 1, anchor_index, -1):
            if _grid_line_is_traversable(
                costmap, path_cells[anchor_index], path_cells[candidate_index]
            ):
                next_index = candidate_index
                break
        simplified.append(path_cells[next_index])
        anchor_index = next_index
    return [slam_map.grid_to_map(row, col) for row, col in simplified]


def _grid_line_is_traversable(
    costmap: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    height, width = costmap.shape
    for row, col in _bresenham_cells(start, end):
        if not _in_bounds((row, col), height, width):
            return False
        if not math.isfinite(float(costmap[row, col])):
            return False
    return True


def _bresenham_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = end
    dcol = abs(col1 - col0)
    drow = -abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = dcol + drow
    row, col = row0, col0
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            return cells
        twice_error = 2 * error
        if twice_error >= drow:
            error += drow
            col += step_col
        if twice_error <= dcol:
            error += dcol
            row += step_row


def _swept_segment_is_traversable(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> bool:
    distance = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    sample_step = max(1e-6, min(slam_map.resolution / 2.0, 0.025))
    sample_count = max(1, int(math.ceil(distance / sample_step)))
    height, width = costmap.shape
    for index in range(sample_count + 1):
        fraction = index / sample_count
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * fraction
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * fraction
        cell = slam_map.map_to_grid(x, y)
        if not _in_bounds(cell, height, width):
            return False
        if not math.isfinite(float(costmap[cell])):
            return False
    return True


def _xy(value: Mapping[str, Any] | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value["x"]), float(value["y"])
    return float(value[0]), float(value[1])


def _clear_costmap_disk(
    slam_map: SlamOccupancyMap,
    costmap: np.ndarray,
    blocked: np.ndarray,
    *,
    center: Mapping[str, Any] | tuple[float, float],
    radius_m: float,
) -> int:
    if radius_m <= 0.0:
        return 0
    try:
        center_x, center_y = _xy(center)
    except (KeyError, TypeError, ValueError):
        return 0
    center_row, center_col = slam_map.map_to_grid(center_x, center_y)
    height, width = costmap.shape
    if not _in_bounds((center_row, center_col), height, width):
        return 0

    radius_cells = max(0, int(math.ceil(float(radius_m) / slam_map.resolution)))
    cleared = 0
    for row in range(
        max(0, center_row - radius_cells),
        min(height, center_row + radius_cells + 1),
    ):
        for col in range(
            max(0, center_col - radius_cells),
            min(width, center_col + radius_cells + 1),
        ):
            dx = (float(col) - float(center_col)) * slam_map.resolution
            dy = (float(row) - float(center_row)) * slam_map.resolution
            if math.hypot(dx, dy) > radius_m + 1e-9:
                continue
            blocked[row, col] = False
            costmap[row, col] = 0.0
            cleared += 1
    return cleared


def _inflate_occupied(occupied: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return occupied.copy()
    try:
        from scipy import ndimage as ndi

        return ndi.binary_dilation(occupied, iterations=radius)
    except ImportError:
        rows, cols = np.nonzero(occupied)
        inflated = occupied.copy()
        height, width = occupied.shape
        for row, col in zip(rows, cols):
            row0, row1 = max(0, row - radius), min(height, row + radius + 1)
            col0, col1 = max(0, col - radius), min(width, col + radius + 1)
            inflated[row0:row1, col0:col1] = True
        return inflated


def _distance_from_obstacles(blocked: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi

        return ndi.distance_transform_edt(~blocked).astype(np.float32)
    except ImportError:
        distance = np.full(blocked.shape, 20.0, dtype=np.float32)
        rows, cols = np.nonzero(blocked)
        if len(rows) == 0:
            return distance
        for row in range(blocked.shape[0]):
            for col in range(blocked.shape[1]):
                distance[row, col] = float(np.min(np.hypot(rows - row, cols - col)))
        return distance


def _in_bounds(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    drow = abs(a[0] - b[0])
    dcol = abs(a[1] - b[1])
    return max(drow, dcol) + (math.sqrt(2.0) - 1.0) * min(drow, dcol)


def _reconstruct(
    parents: Mapping[tuple[int, int], tuple[int, int]], current: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [current]
    while current in parents:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path
