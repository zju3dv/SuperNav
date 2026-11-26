"""Dataclasses used across the oracle local navigation package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

@dataclass(frozen=True)
class _Candidate:
    label: str
    object_id: str
    target_ref: str
    object_position: np.ndarray
    target: np.ndarray
    target_path_points: list[list[float]]
    local_target: np.ndarray
    local_path_points: list[list[float]]
    local_geodesic: float
    target_geodesic: float
    label_score: float
    heading_angle_deg: float

@dataclass(frozen=True)
class _LocalMapObject:
    map_index: int
    label: str
    object_id: str
    room_id: str | None
    object_position: np.ndarray
    distance_m: float
    geodesic_distance_m: float | None
    bearing_deg: float
    turn_right_deg: float
    reachable: bool
    reachable_target: np.ndarray | None
    path_points: list[list[float]]

@dataclass(frozen=True)
class _VisualMatchEvaluation:
    detection: Mapping[str, Any]
    projected: Mapping[str, Any]
    geometry_score: float
    semantic_score: float
    bbox_penalty: float
    final_score: float | None
    reject_reason: str | None

