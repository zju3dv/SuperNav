"""BBox math, JSON-safe conversion, and small coercion helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

import numpy as np

def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value

def _debug_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, tuple):
        return None
    x0, y0, x1, y1 = value
    return {
        "x": round(x0, 1),
        "y": round(y0, 1),
        "width": round(x1 - x0, 1),
        "height": round(y1 - y0, 1),
    }

def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))

def _projected_row_xyxy(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = row.get("bbox_px")
    if not isinstance(bbox, Mapping):
        return None
    try:
        x = float(bbox.get("x"))
        y = float(bbox.get("y"))
        width = float(bbox.get("width"))
        height = float(bbox.get("height"))
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return x, y, x + width, y + height

def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    left_area = max(0.0, (lx1 - lx0) * (ly1 - ly0))
    right_area = max(0.0, (rx1 - rx0) * (ry1 - ry0))
    union = left_area + right_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)

def _bbox_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))

def _bbox_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def _scale_bbox(
    box: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y

def _expand_bbox(
    box: tuple[float, float, float, float],
    *,
    ratio: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    pad_x = width * max(0.0, ratio)
    pad_y = height * max(0.0, ratio)
    expanded = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
    if bounds is None:
        return expanded
    bx0, by0, bx1, by1 = bounds
    return (
        max(bx0, expanded[0]),
        max(by0, expanded[1]),
        min(bx1, expanded[2]),
        min(by1, expanded[3]),
    )

def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5

def _point_in_bbox(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1

def _public_detection(detection: Mapping[str, Any]) -> dict[str, Any]:
    box = detection.get("box_xyxy")
    row: dict[str, Any] = {
        "direction": detection.get("direction"),
        "detection_confidence": round(float(detection.get("score") or 0.0), 3),
    }
    if isinstance(box, tuple):
        x0, y0, x1, y1 = box
        row["bbox_px"] = {
            "x": round(x0, 1),
            "y": round(y0, 1),
            "width": round(x1 - x0, 1),
            "height": round(y1 - y0, 1),
        }
    return {k: v for k, v in row.items() if v is not None}

def _append_unique_paths(target: list[str], paths: list[str]) -> None:
    seen = set(target)
    for path in paths:
        if path in seen:
            continue
        target.append(path)
        seen.add(path)

def _movement_frame_paths(nav: Mapping[str, Any]) -> list[str]:
    """Return RGB frame paths emitted during one navmesh follower burst."""
    paths: list[str] = []
    trace_paths = nav.get("trace_frame_paths")
    if isinstance(trace_paths, list):
        paths.extend(path for path in trace_paths if isinstance(path, str) and path)
    visuals = nav.get("visuals")
    if isinstance(visuals, Mapping):
        color = visuals.get("color_sensor")
        if isinstance(color, Mapping):
            path = color.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return paths

def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default

def _nonnegative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)

def _clamped_positive_float(value: Any, *, default: float, max_value: float) -> float:
    parsed = _positive_float(value, default=default)
    return min(parsed, max_value)

def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default

