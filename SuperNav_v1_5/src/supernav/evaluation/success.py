"""Shared ObjectNav success criterion for the InteriorGS global_task bench.

Current semantics (tasks.json ``success_criterion`` field, goal_radius 1.0 m):
success iff the agent's final XZ position lies within ``goal_radius`` of the
target object bbox's XZ surface::

    d = hypot(max(0, |px-cx| - sx/2), max(0, |pz-cz| - sz/2))
    success = d <= goal_radius

bbox center/size come from ``task["target_object"]`` (``bbox_center`` /
``bbox_size``, habitat world coordinates). Tasks without an annotated
target-object bbox fall back to the viewpoint criterion (3D euclidean
distance to ``goal_position`` <= goal_radius) and log a warning once.
"""

import logging
import math

log = logging.getLogger(__name__)

CRITERION_BBOX = "dist_xz_to_target_bbox_surface"
CRITERION_VIEWPOINT = "dist_to_goal_viewpoint"

_warned = set()


def bbox_surface_dist_xz(pos, bbox_center, bbox_size):
    """XZ-plane distance from pos to the bbox surface (0 inside the bbox)."""
    import numpy as np

    p = np.asarray(pos, dtype=np.float64)
    c = np.asarray(bbox_center, dtype=np.float64)
    s = np.asarray(bbox_size, dtype=np.float64)
    dx = max(0.0, abs(float(p[0] - c[0])) - float(s[0]) / 2.0)
    dz = max(0.0, abs(float(p[2] - c[2])) - float(s[2]) / 2.0)
    return math.hypot(dx, dz)


def goal_success(pos, task):
    """Evaluate episode success for final position ``pos``.

    Returns ``(success, dist, criterion)`` where ``dist`` is the distance the
    criterion actually thresholded (bbox-surface XZ distance when the task
    annotates ``target_object.bbox_center``/``bbox_size``, else the 3D
    euclidean distance to ``goal_position``).
    """
    goal_radius = float(task.get("goal_radius", 0.25))
    target = task.get("target_object") or {}
    center = target.get("bbox_center")
    size = target.get("bbox_size")
    if center is not None and size is not None:
        dist = bbox_surface_dist_xz(pos, center, size)
        return dist <= goal_radius, dist, CRITERION_BBOX
    key = (task.get("scene_id"), task.get("task_id"))
    if key not in _warned:
        _warned.add(key)
        log.warning("task %s/%s has no target_object bbox; falling back to "
                    "viewpoint success criterion",
                    task.get("scene_id"), task.get("task_id"))
    import numpy as np

    p = np.asarray(pos, dtype=np.float64)
    g = np.asarray(task["goal_position"], dtype=np.float64)
    dist = float(np.linalg.norm(p - g))
    return dist <= goal_radius, dist, CRITERION_VIEWPOINT
