"""Pure projection from audit results to the established model-visible fields."""
from __future__ import annotations
from typing import Any, Dict, List

_VISIBLE_NAV_TARGET_FIELDS = frozenset(
    {
        "target_ref",
        "label",
        "direction",
        "turn_right_deg",
        "bbox_px",
        "center_px",
        "depth_m",
        "visible_fraction",
        "visibility_basis",
    }
)

_MODEL_HIDDEN_RESULT_KEYS = frozenset(
    {
        "agent_state",
        "backend",
        "current_goal",
        "goal",
        "ground_truth",
        "heading_deg",
        "images",
        "map",
        "map_image",
        "metrics",
        "movement_frames",
        "nav_mode",
        "navmesh",
        "navmesh_loaded",
        "output_dir",
        "overlay_image",
        "overlay_path",
        "panorama_image_paths",
        "path",
        "position",
        "publish_hints",
        "rotation",
        "scene_dataset_config_file",
        "scene_graph",
        "start_position",
        "start_rotation",
        "state_summary",
        "topdown_map",
        "trace_frame_paths",
        "trajectory",
        "trajectory_log",
        "visuals",
    }
)

def _model_hidden_result_key(key: str) -> bool:
    return (
        key in _MODEL_HIDDEN_RESULT_KEYS
        or key == "dir"
        or key.endswith("_dir")
        or key.endswith("_path")
        or key.endswith("_paths")
    )

def _redact_global_task_result(value: Any) -> Any:
    """Remove absolute pose and filesystem capabilities from model-visible JSON."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if _model_hidden_result_key(key):
                if key == "overlay_image" and isinstance(child, str) and child:
                    out["overlay_available"] = True
                continue
            out[key] = _redact_global_task_result(child)
        return out
    if isinstance(value, list):
        return [_redact_global_task_result(item) for item in value]
    return value

def _strip_visual_ground_private_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_visual_ground_private_fields(child)
            for key, child in value.items()
            if key not in {"snapped_point", "_private"}
        }
    if isinstance(value, list):
        return [_strip_visual_ground_private_fields(item) for item in value]
    return value

def _normalize_visible_nav_targets(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = body.get("visible_nav_targets")
    if not isinstance(raw, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = {
            k: v
            for k, v in item.items()
            if k in _VISIBLE_NAV_TARGET_FIELDS and v is not None
        }
        if isinstance(row.get("target_ref"), str) and row["target_ref"]:
            rows.append(row)
    return rows
