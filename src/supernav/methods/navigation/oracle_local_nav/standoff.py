"""Standoff point sampling, shortest-path queries, and candidate selection."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from habitat_contract.navigation import NavigationPathfinder

from supernav.methods.navigation.oracle_local_nav.coords import _as_vec3, _object_position, _scene_coordinate_system
from supernav.methods.navigation.oracle_local_nav.scene_graph import _label_match_score, _object_records, _object_target_ref
from supernav.methods.navigation.oracle_local_nav.types import _Candidate
from supernav.methods.navigation.oracle_local_nav.utils import _positive_float

def _horizontal_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(right, dtype=np.float32) - np.asarray(left, dtype=np.float32)
    delta[1] = 0.0
    return float(np.linalg.norm(delta))

def _bearing_degrees(
    *,
    current_position: np.ndarray,
    forward: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    direction = np.asarray(target, dtype=np.float32) - np.asarray(
        current_position,
        dtype=np.float32,
    )
    direction[1] = 0.0
    bearing = float(math.degrees(math.atan2(float(direction[2]), float(direction[0]))))
    heading = float(math.degrees(math.atan2(float(forward[2]), float(forward[0]))))
    turn_right = (bearing - heading + 180.0) % 360.0 - 180.0
    return round(bearing, 1), round(turn_right, 1)

def _best_reachable_standoff(
    *,
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    object_position: np.ndarray,
    standoff_m: float,
) -> tuple[np.ndarray | None, float | None, list[list[float]]]:
    best_target: np.ndarray | None = None
    best_distance: float | None = None
    best_path: list[list[float]] = []
    for point in _standoff_points(
        pathfinder,
        current_position,
        object_position,
        standoff_m,
    ):
        path = _shortest_path_proxy(pathfinder, current_position, point)
        if path is None:
            continue
        distance, path_points = path
        if best_distance is None or distance < best_distance:
            best_target = point
            best_distance = distance
            best_path = path_points
    return best_target, best_distance, best_path

def _select_candidate(
    *,
    scene_graph: Mapping[str, Any],
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    forward: np.ndarray,
    target_ref: str,
    instruction: str,
    target_label: str,
    horizon_m: float,
    standoff_m: float,
) -> _Candidate | None:
    candidates: list[_Candidate] = []
    target_ref_norm = str(target_ref or "").strip()
    for index, obj in enumerate(_object_records(scene_graph), start=1):
        label = str(obj.get("label") or obj.get("category") or "").strip()
        if not label:
            continue
        object_ref = _object_target_ref(obj, index)
        if target_ref_norm:
            if object_ref != target_ref_norm:
                continue
            label_score = 5.0
        else:
            label_score = _label_match_score(
                label,
                target_label=target_label,
                instruction=instruction,
            )
            if label_score <= 0.0:
                continue
        if not object_ref:
            continue
        object_position = _object_position(scene_graph, obj)
        if object_position is None:
            continue
        best = _best_standoff_candidate(
            pathfinder=pathfinder,
            current_position=current_position,
            forward=forward,
            object_position=object_position,
            label=label,
            object_id=str(obj.get("id") or obj.get("object_id") or ""),
            target_ref=object_ref,
            label_score=label_score,
            horizon_m=horizon_m,
            standoff_m=standoff_m,
        )
        if best is not None:
            candidates.append(best)

    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (
            -c.label_score,
            c.local_geodesic + 0.25 * (c.heading_angle_deg / 180.0),
            c.target_geodesic,
        )
    )
    return candidates[0]

def _best_standoff_candidate(
    *,
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    forward: np.ndarray,
    object_position: np.ndarray,
    label: str,
    object_id: str,
    target_ref: str,
    label_score: float,
    horizon_m: float,
    standoff_m: float,
) -> _Candidate | None:
    best: _Candidate | None = None
    for point in _standoff_points(pathfinder, current_position, object_position, standoff_m):
        path = _shortest_path_proxy(pathfinder, current_position, point)
        if path is None:
            continue
        target_geodesic, target_path_points = path
        local_target, local_path_points, local_geodesic = _limit_path_to_horizon(
            target_path_points,
            horizon_m=horizon_m,
        )
        angle = _heading_angle_deg(forward, object_position - current_position)
        candidate = _Candidate(
            label=label,
            object_id=object_id,
            object_position=object_position,
            target=point,
            target_ref=target_ref,
            target_path_points=target_path_points,
            local_target=local_target,
            local_path_points=local_path_points,
            local_geodesic=local_geodesic,
            target_geodesic=target_geodesic,
            label_score=label_score,
            heading_angle_deg=angle,
        )
        if best is None or (
            candidate.local_geodesic + 0.25 * (candidate.heading_angle_deg / 180.0)
            < best.local_geodesic + 0.25 * (best.heading_angle_deg / 180.0)
        ):
            best = candidate
    return best

def _standoff_points(
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    object_position: np.ndarray,
    standoff_m: float,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    radii = [standoff_m, max(standoff_m, 0.9), max(standoff_m, 1.2)]
    for radius in radii:
        for idx in range(16):
            theta = 2.0 * math.pi * (idx / 16.0)
            raw = np.array(
                [
                    float(object_position[0] + math.cos(theta) * radius),
                    float(current_position[1]),
                    float(object_position[2] + math.sin(theta) * radius),
                ],
                dtype=np.float32,
            )
            try:
                snapped = np.asarray(pathfinder.snap_point(raw), dtype=np.float32)
            except Exception:
                continue
            if np.any(np.isnan(snapped)):
                continue
            key = tuple(int(round(float(v) * 1000.0)) for v in snapped)
            if key in seen:
                continue
            seen.add(key)
            points.append(snapped)
    return points

def _shortest_path(
    pathfinder: NavigationPathfinder,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, list[list[float]]] | None:
    path = pathfinder.shortest_path(
        np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
    )
    if path is None:
        return None
    points = [np.asarray(point, dtype=np.float32).tolist() for point in path.points]
    return float(path.geodesic_distance), points


def _shortest_path_proxy(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the package-level ``_shortest_path``.

    This indirection lets tests monkeypatch ``oln._shortest_path`` and have the
    fake used by standoff helpers.
    """
    from supernav.methods.navigation.oracle_local_nav import _shortest_path

    return _shortest_path(*args, **kwargs)


def _limit_path_to_horizon(
    path_points: list[list[float]],
    *,
    horizon_m: float,
) -> tuple[np.ndarray, list[list[float]], float]:
    if not path_points:
        raise ValueError("path_points cannot be empty")
    if len(path_points) == 1 or horizon_m <= 0:
        point = np.asarray(path_points[-1], dtype=np.float32)
        return point, [point.tolist()], 0.0

    out = [np.asarray(path_points[0], dtype=np.float32)]
    travelled = 0.0
    for raw_next in path_points[1:]:
        nxt = np.asarray(raw_next, dtype=np.float32)
        prev = out[-1]
        segment = float(np.linalg.norm(nxt - prev))
        if travelled + segment >= horizon_m and segment > 1e-6:
            ratio = max(0.0, min(1.0, (horizon_m - travelled) / segment))
            limited = prev + (nxt - prev) * ratio
            out.append(limited.astype(np.float32))
            return limited.astype(np.float32), [point.tolist() for point in out], horizon_m
        travelled += segment
        out.append(nxt)
    return out[-1], [point.tolist() for point in out], travelled

def _heading_angle_deg(forward: np.ndarray, direction: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=np.float32)
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return 0.0
    unit = direction / norm
    dot = float(np.clip(np.dot(forward, unit), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))

def _remaining_geodesic(pathfinder: NavigationPathfinder, start: np.ndarray, end: np.ndarray) -> float | None:
    path = _shortest_path_proxy(pathfinder, start, end)
    return path[0] if path is not None else None
