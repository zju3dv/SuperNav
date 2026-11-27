"""Habitat NavMesh queries for offline scoring and replay."""
from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any, Optional

def find_navmesh(scenes_root: Path, split: str, scene_hash: str) -> Optional[Path]:
    # The episode split dir is authoritative; scene hashes are globally unique
    # in HM3D, so a cross-split glob fallback is unambiguous. This matters for
    # val_mini: its episodes name minival/<id>/... but a dataset copy without
    # the minival/ tree holds the same buildings under val/.
    for pattern in (
        str(scenes_root / split / f"*-{scene_hash}" / f"{scene_hash}.basis.navmesh"),
        str(scenes_root / "*" / f"*-{scene_hash}" / f"{scene_hash}.basis.navmesh"),
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    return None


# ----------------------------------------------------------------- geodesics
class NavmeshGeodesic:
    """Lazy per-scene PathFinder cache over official .basis.navmesh files.

    Only needed for the ``viewpoint_geodesic`` criterion; SPL's shortest-path
    term comes from the episode file, not from live navmesh queries.
    """

    def __init__(self, scenes_root: Path):
        self.scenes_root = Path(scenes_root)
        self._cache: dict[str, Any] = {}

    def geodesic_to_any(
        self, split: str, scene_hash: str, start: list[float], ends: list[list[float]]
    ) -> float:
        key = f"{split}/{scene_hash}"
        if key not in self._cache:
            navmesh = find_navmesh(self.scenes_root, split, scene_hash)
            if navmesh is None:
                self._cache[key] = None
            else:
                from supernav.backends.habitat.loader import prepare_simulator_import
                prepare_simulator_import()
                import habitat_sim  # lazy: module must import without native lib

                pf = habitat_sim.PathFinder()
                if not pf.load_nav_mesh(str(navmesh)):
                    self._cache[key] = None
                else:
                    self._cache[key] = pf
        pf = self._cache[key]
        if pf is None:
            return math.inf
        import habitat_sim

        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = ends
        pf.find_path(path)
        return float(path.geodesic_distance)


def navmesh_loads(navmesh: Path) -> bool:
    from supernav.backends.habitat.loader import prepare_simulator_import
    prepare_simulator_import()
    import habitat_sim

    return bool(habitat_sim.PathFinder().load_nav_mesh(str(navmesh)))


def objectnav_geometry(navmesh, start, viewpoints, viewpoint_index, viewpoint_selection, meters_per_pixel):
    from supernav.backends.habitat.loader import prepare_simulator_import

    prepare_simulator_import()
    import habitat_sim
    import numpy as np

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        return None
    if viewpoint_index is not None:
        if not 0 <= viewpoint_index < len(viewpoints):
            return None
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(
            viewpoints[viewpoint_index]["agent_state"]["position"],
            dtype=np.float32,
        )
        if not pathfinder.find_path(path):
            return None
        closest_index = viewpoint_index
    elif viewpoint_selection == "highest_iou":
        path = None
        closest_index = -1
        for index in sorted(
            range(len(viewpoints)),
            key=lambda item: float(viewpoints[item].get("iou") or -1.0),
            reverse=True,
        ):
            candidate = habitat_sim.ShortestPath()
            candidate.requested_start = np.asarray(start, dtype=np.float32)
            candidate.requested_end = np.asarray(
                viewpoints[index]["agent_state"]["position"], dtype=np.float32
            )
            if pathfinder.find_path(candidate):
                path = candidate
                closest_index = index
                break
        if path is None:
            return None
    else:
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_ends = np.asarray(
            [row["agent_state"]["position"] for row in viewpoints],
            dtype=np.float32,
        )
        if not pathfinder.find_path(path):
            return None
        closest_index = int(path.closest_end_point_index)
    topdown = pathfinder.get_topdown_view(meters_per_pixel, float(start[1]))
    lower, _ = pathfinder.get_bounds()
    return {
        "closest_index": closest_index,
        "topdown": topdown,
        "bounds_min_xz": [float(lower[0]), float(lower[2])],
        "path_points": [[float(p[0]), float(p[1]), float(p[2])] for p in path.points],
        "geodesic_distance_m": float(path.geodesic_distance),
    }
