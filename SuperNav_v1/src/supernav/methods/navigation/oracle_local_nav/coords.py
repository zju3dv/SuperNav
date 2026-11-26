"""Scene-graph coordinate system detection and conversion to Habitat coords."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from supernav.methods.navigation.oracle_local_nav.scene_graph import _object_records, _object_target_ref
from supernav.methods.navigation.oracle_local_nav.utils import _json_safe_value

def _object_coordinate_audit(
    scene_graph: Mapping[str, Any] | None,
    target_ref: str,
) -> dict[str, Any]:
    if not isinstance(scene_graph, Mapping):
        return {}
    target_ref_norm = str(target_ref or "").strip()
    if not target_ref_norm:
        return {}
    for index, obj in enumerate(_object_records(scene_graph), start=1):
        if _object_target_ref(obj, index) == target_ref_norm:
            return _object_coordinate_audit_for_record(scene_graph, obj)
    return {}

def _object_coordinate_audit_for_record(
    scene_graph: Mapping[str, Any],
    obj: Mapping[str, Any],
) -> dict[str, Any]:
    coordinate_system = _scene_coordinate_system(scene_graph)
    uses_semantic_floor = _uses_semantic_xy_floor_coordinates(coordinate_system)
    audit: dict[str, Any] = {
        "coordinate_system": _json_safe_value(coordinate_system),
        "semantic_xy_floor_y_flipped": bool(uses_semantic_floor),
    }
    for key in ("position_xyz", "position", "centroid", "center"):
        raw = _raw_mapping_xyz(obj.get(key))
        if raw is None:
            continue
        habitat = _as_vec3(obj.get(key), coordinate_system=coordinate_system)
        audit.update(
            {
                "source_key": key,
                "raw_xyz": raw,
                "habitat_xyz": habitat.tolist() if habitat is not None else None,
            }
        )
        return audit

    bbox_min_raw = _raw_mapping_xyz(obj.get("bbox_min_xyz"))
    bbox_max_raw = _raw_mapping_xyz(obj.get("bbox_max_xyz"))
    bbox_min = _as_vec3(obj.get("bbox_min_xyz"), coordinate_system=coordinate_system)
    bbox_max = _as_vec3(obj.get("bbox_max_xyz"), coordinate_system=coordinate_system)
    if bbox_min is not None and bbox_max is not None:
        center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
        raw_center = None
        if bbox_min_raw is not None and bbox_max_raw is not None:
            raw_center = [
                (bbox_min_raw[0] + bbox_max_raw[0]) * 0.5,
                (bbox_min_raw[1] + bbox_max_raw[1]) * 0.5,
                (bbox_min_raw[2] + bbox_max_raw[2]) * 0.5,
            ]
        audit.update(
            {
                "source_key": "bbox_center",
                "raw_xyz": raw_center,
                "habitat_xyz": center.tolist(),
                "bbox_raw_min_xyz": bbox_min_raw,
                "bbox_raw_max_xyz": bbox_max_raw,
                "bbox_habitat_min_xyz": bbox_min.tolist(),
                "bbox_habitat_max_xyz": bbox_max.tolist(),
            }
        )
    return audit

def _raw_mapping_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(axis in value for axis in ("x", "y", "z")):
        return None
    try:
        return [float(value["x"]), float(value["y"]), float(value["z"])]
    except (TypeError, ValueError):
        return None

def _object_position(
    scene_graph: Mapping[str, Any],
    obj: Mapping[str, Any],
) -> np.ndarray | None:
    coordinate_system = _scene_coordinate_system(scene_graph)
    for key in ("position_xyz", "position", "centroid", "center"):
        vec = _as_vec3(obj.get(key), coordinate_system=coordinate_system)
        if vec is not None:
            return vec
    bbox_min = _as_vec3(obj.get("bbox_min_xyz"), coordinate_system=coordinate_system)
    bbox_max = _as_vec3(obj.get("bbox_max_xyz"), coordinate_system=coordinate_system)
    if bbox_min is not None and bbox_max is not None:
        return ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    return None

def _scene_coordinate_system(scene_graph: Mapping[str, Any]) -> Mapping[str, Any]:
    coordinate_system = scene_graph.get("coordinate_system")
    return coordinate_system if isinstance(coordinate_system, Mapping) else {}

def _as_vec3(
    value: Any,
    *,
    coordinate_system: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    flip_semantic_floor_y = False
    if isinstance(value, Mapping):
        if not all(axis in value for axis in ("x", "y", "z")):
            return None
        if _uses_semantic_xy_floor_coordinates(coordinate_system):
            raw = [value.get("x"), value.get("z"), value.get("y")]
            flip_semantic_floor_y = True
        else:
            raw = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw = value
    else:
        return None
    try:
        arr = np.asarray(
            [float(raw[0]), float(raw[1]), float(raw[2])],
            dtype=np.float32,
        )
    except (TypeError, ValueError):
        return None
    if flip_semantic_floor_y:
        arr[2] = -arr[2]
    if np.any(np.isnan(arr)):
        return None
    return arr

def _uses_semantic_xy_floor_coordinates(
    coordinate_system: Mapping[str, Any] | None,
) -> bool:
    """Return true for InteriorGS semantic coordinates.

    Habitat positions are ``[x, vertical_y, horizontal_z]``. The generated
    InteriorGS semantic scene graph stores floor coordinates as ``x/y`` and
    height as ``z``. Empirical alignment against the Habitat navmesh for
    ``scenes_0236_840815`` showed the semantic floor ``y`` axis is opposite
    Habitat ``z``, so convert semantic ``(x, y, z_up)`` to Habitat
    ``[x, z_up, -y]`` before pathfinder queries.
    """
    if not isinstance(coordinate_system, Mapping):
        return False
    horizontal_axes = coordinate_system.get("horizontal_axes")
    vertical_axis = coordinate_system.get("vertical_axis")
    return list(horizontal_axes or []) == ["x", "y"] and vertical_axis == "z"

