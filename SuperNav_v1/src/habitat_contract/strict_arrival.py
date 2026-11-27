"""Shared strict arrival checks for bridge/agent task-quality guards.

The helpers in this module never expose hidden evaluator coordinates. They
only decide whether a terminal ``reached`` write should be recoverably rejected
so the agent can keep exploring or report ``blocked``.
"""

from __future__ import annotations

from typing import Any, Mapping

STRICT_MAPLESS_IF_ARRIVAL_ERROR = (
    "strict mapless arrival guard: cannot set status=reached for "
    "instruction_following because the evaluator success threshold is "
    "not satisfied. Continue exploring with visual evidence, or set "
    "status=blocked with a rationale if no safe progress is possible."
)


def strict_mapless_if_arrival_error(nav_status: Mapping[str, Any]) -> str | None:
    """Return a sanitized error when mapless IF is still outside threshold."""

    if nav_status.get("nav_mode") != "mapless":
        return None
    if nav_status.get("task_type") != "instruction_following":
        return None
    if not bool(nav_status.get("has_ground_truth")):
        return None

    distance = _final_distance(nav_status)
    if distance is None:
        return None
    threshold = _success_threshold(nav_status)
    if threshold is None:
        threshold = 0.5
    if distance < threshold:
        return None
    return STRICT_MAPLESS_IF_ARRIVAL_ERROR


def _final_distance(nav_status: Mapping[str, Any]) -> float | None:
    debug = (
        nav_status.get("_debug", {})
        if isinstance(nav_status.get("_debug"), Mapping)
        else {}
    )
    has_navmesh = bool(nav_status.get("has_navmesh"))
    keys = ["gt_distance_to_goal_m", "final_distance_m"]
    if has_navmesh:
        keys.extend(["gt_geodesic_distance", "gt_end_geodesic_distance"])
        keys.extend(["gt_euclidean_distance", "gt_end_euclidean_distance"])
    else:
        keys.extend(["gt_euclidean_distance", "gt_end_euclidean_distance"])
        keys.extend(["gt_geodesic_distance", "gt_end_geodesic_distance"])
    return _first_float([debug, nav_status], keys)


def _success_threshold(nav_status: Mapping[str, Any]) -> float | None:
    return _first_float(
        [nav_status],
        [
            "success_threshold_m",
            "success_distance_threshold_m",
            "success_distance_threshold",
            "distance_threshold_m",
            "distance_threshold",
        ],
    )


def _first_float(dicts: list[Mapping[str, Any]], keys: list[str]) -> float | None:
    for source in dicts:
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is None or isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


__all__ = [
    "STRICT_MAPLESS_IF_ARRIVAL_ERROR",
    "strict_mapless_if_arrival_error",
]
