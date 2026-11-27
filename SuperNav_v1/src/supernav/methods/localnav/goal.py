"""Goal-image snapshot and visual point-marker construction."""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from supernav.methods.localnav.contracts import GoalSnapshot, LocalNavInputError, NormalizedPoint

_DEFAULT_MARKER_RADIUS_PX = 12
_DEFAULT_MARKER_COLOR = (255, 64, 64)


def as_readonly_rgb(value: Any) -> np.ndarray:
    """Convert a path, PIL image, or array into an immutable uint8 RGB array."""
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
        if not os.path.isfile(path):
            raise LocalNavInputError(
                "goal_image_not_found", f"goal image does not exist: {path}"
            )
        array = np.asarray(Image.open(path).convert("RGB"))
    elif isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"))
    else:
        array = np.asarray(value)

    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise LocalNavInputError(
            "invalid_rgb", "RGB observation must have shape [H, W, 3|4]"
        )
    if array.shape[2] == 4:
        array = array[:, :, :3]
    try:
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    except Exception:  # noqa: BLE001 — malformed pixel payloads take many shapes
        raise LocalNavInputError(
            "invalid_rgb", "failed to decode RGB observation"
        ) from None
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def render_point_marker(
    image_array: Any,
    point: "Sequence[float] | NormalizedPoint",
    *,
    radius_px: int = _DEFAULT_MARKER_RADIUS_PX,
    color: tuple[int, int, int] = _DEFAULT_MARKER_COLOR,
) -> np.ndarray:
    """Return an immutable RGB copy with a ring and center dot at ``point``."""
    if int(radius_px) < 2:
        raise LocalNavInputError("invalid_marker_radius", "radius_px must be at least 2")
    if (
        not isinstance(color, (tuple, list))
        or len(color) != 3
        or not all(isinstance(channel, (int, float)) for channel in color)
    ):
        raise LocalNavInputError("invalid_marker_color", "marker color must be an RGB tuple")

    normalized = NormalizedPoint.parse(point)
    base = as_readonly_rgb(image_array)
    height, width = base.shape[:2]
    px_float, py_float = normalized.to_pixel(width, height)
    radius = int(radius_px)
    stroke = max(2, radius // 4)
    center_radius = max(2, radius // 3)
    rgb_color = tuple(int(channel) for channel in color)

    image = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [px_float - radius, py_float - radius, px_float + radius, py_float + radius],
        outline=rgb_color,
        width=stroke,
    )
    draw.ellipse(
        [
            px_float - center_radius,
            py_float - center_radius,
            px_float + center_radius,
            py_float + center_radius,
        ],
        fill=rgb_color,
    )
    marked = np.ascontiguousarray(np.asarray(image))
    marked.setflags(write=False)
    return marked


def build_goal_snapshot(
    image_ref: str,
    rgb_array: Any,
    point: "Sequence[float] | NormalizedPoint",
    *,
    goal_id: "str | None" = None,
    capture_seq: "int | None" = None,
    direction: "str | None" = None,
    hfov: "float | None" = None,
    marker_radius_px: int = _DEFAULT_MARKER_RADIUS_PX,
) -> GoalSnapshot:
    """Build the persistent internal goal used throughout one local episode."""
    if not str(image_ref or "").strip():
        raise LocalNavInputError("invalid_image_ref", "image_ref must be non-empty")
    normalized = NormalizedPoint.parse(point)
    rgb = as_readonly_rgb(rgb_array)
    marked = render_point_marker(rgb, normalized, radius_px=marker_radius_px)
    return GoalSnapshot(
        goal_id=goal_id or f"localgoal-{uuid.uuid4().hex}",
        image_ref=str(image_ref).strip(),
        point=normalized,
        rgb=rgb,
        marked_rgb=marked,
        capture_seq=capture_seq,
        direction=direction,
        hfov=hfov,
    )


def snapshot_goal_from_registry(
    registry: Mapping[str, Mapping[str, Any]],
    image_ref: str,
    latest_capture_seq: "int | None",
    point: "Sequence[float] | NormalizedPoint",
    *,
    goal_id: "str | None" = None,
    marker_radius_px: int = _DEFAULT_MARKER_RADIUS_PX,
) -> GoalSnapshot:
    """Resolve a fresh Habitat panorama ref and copy it into a persistent goal."""
    ref = str(image_ref or "").strip()
    entry = registry.get(ref) if isinstance(registry, Mapping) else None
    if not isinstance(entry, Mapping):
        raise LocalNavInputError(
            "invalid_image_ref", "image_ref is not valid for this session"
        )
    if latest_capture_seq is not None and entry.get("capture_seq") != latest_capture_seq:
        raise LocalNavInputError(
            "stale_image_ref", "image_ref is not from the latest visual capture"
        )
    rgb_source: Any = entry.get("rgb")
    if rgb_source is None:
        rgb_source = entry.get("path")
    if rgb_source is None:
        raise LocalNavInputError(
            "invalid_image_ref", "image_ref has no RGB data or path"
        )
    return build_goal_snapshot(
        ref,
        rgb_source,
        point,
        goal_id=goal_id,
        capture_seq=entry.get("capture_seq"),
        direction=entry.get("direction"),
        hfov=entry.get("hfov"),
        marker_radius_px=marker_radius_px,
    )
