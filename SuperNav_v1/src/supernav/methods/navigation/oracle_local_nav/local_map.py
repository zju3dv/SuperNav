"""Oracle local map generation and top-down rendering."""

from __future__ import annotations

from supernav.paths import workspace_root

import math
import os
from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend, NavigationPathfinder, NavigationSession

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supernav.methods.navigation.oracle_local_nav.coords import _object_position, _scene_coordinate_system
from supernav.methods.navigation.oracle_local_nav.scene_graph import _label_match_score, _normalize_text, _object_records
from supernav.methods.navigation.oracle_local_nav.standoff import _best_reachable_standoff, _bearing_degrees, _horizontal_distance
from supernav.methods.navigation.oracle_local_nav.types import _LocalMapObject
from supernav.methods.navigation.oracle_local_nav.utils import _clamped_positive_float, _coerce_bool, _positive_float

def _local_map_objects(
    *,
    scene_graph: Mapping[str, Any],
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    forward: np.ndarray,
    radius_m: float,
    max_objects: int,
    label_filter: str,
) -> list[_LocalMapObject]:
    label_filter_norm = _normalize_text(label_filter)
    rows: list[_LocalMapObject] = []
    seen: set[str] = set()
    for obj in _object_records(scene_graph):
        label = str(obj.get("label") or obj.get("category") or "").strip()
        if not label:
            continue
        if label_filter_norm and _label_match_score(
            label,
            target_label=label_filter,
            instruction="",
        ) <= 0.0:
            continue
        object_id = str(obj.get("id") or obj.get("object_id") or "")
        dedupe_key = object_id or f"{label}:{len(seen)}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        object_position = _object_position(scene_graph, obj)
        if object_position is None:
            continue
        distance = _horizontal_distance(current_position, object_position)
        if distance > radius_m:
            continue
        reachable_target, geodesic, path_points = _best_reachable_standoff(
            pathfinder=pathfinder,
            current_position=current_position,
            object_position=object_position,
            standoff_m=0.7,
        )
        bearing_abs, turn_right = _bearing_degrees(
            current_position=current_position,
            forward=forward,
            target=object_position,
        )
        rows.append(
            _LocalMapObject(
                map_index=0,
                label=label,
                object_id=object_id,
                room_id=_object_room_id(obj),
                object_position=object_position,
                distance_m=distance,
                geodesic_distance_m=geodesic,
                bearing_deg=bearing_abs,
                turn_right_deg=turn_right,
                reachable=reachable_target is not None,
                reachable_target=reachable_target,
                path_points=path_points,
            )
        )

    rows.sort(
        key=lambda row: (
            not row.reachable,
            (
                row.geodesic_distance_m
                if row.geodesic_distance_m is not None
                else row.distance_m
            ),
            row.distance_m,
            row.label,
        )
    )
    indexed: list[_LocalMapObject] = []
    for idx, row in enumerate(rows[:max_objects], start=1):
        indexed.append(
            _LocalMapObject(
                map_index=idx,
                label=row.label,
                object_id=row.object_id,
                room_id=row.room_id,
                object_position=row.object_position,
                distance_m=row.distance_m,
                geodesic_distance_m=row.geodesic_distance_m,
                bearing_deg=row.bearing_deg,
                turn_right_deg=row.turn_right_deg,
                reachable=row.reachable,
                reachable_target=row.reachable_target,
                path_points=row.path_points,
            )
        )
    return indexed

def _object_room_id(obj: Mapping[str, Any]) -> str | None:
    for key in ("room_id", "room", "region_id", "parent_room_id"):
        value = obj.get(key)
        if value is not None and str(value):
            return str(value)
    return None

def _local_map_object_public_row(obj: _LocalMapObject) -> dict[str, Any]:
    row: dict[str, Any] = {
        "map_index": obj.map_index,
        "label": obj.label,
        "object_id": obj.object_id,
        "distance_m": round(obj.distance_m, 3),
        "geodesic_distance_m": (
            round(obj.geodesic_distance_m, 3)
            if obj.geodesic_distance_m is not None
            else None
        ),
        "bearing_deg": obj.bearing_deg,
        "turn_right_deg": obj.turn_right_deg,
        "room_id": obj.room_id,
        "reachable": obj.reachable,
    }
    return {k: v for k, v in row.items() if v is not None}

def _local_map_object_debug_row(obj: _LocalMapObject) -> dict[str, Any]:
    return {
        **_local_map_object_public_row(obj),
        "object_position": obj.object_position.tolist(),
        "reachable_target": (
            obj.reachable_target.tolist() if obj.reachable_target is not None else None
        ),
        "path_points": obj.path_points,
    }

def _render_oracle_local_map(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    pathfinder: NavigationPathfinder,
    current_position: np.ndarray,
    forward: np.ndarray,
    objects: list[_LocalMapObject],
    radius_m: float,
    meters_per_pixel: float,
    output_dir: str,
) -> dict[str, Any]:
    raw_map = np.asarray(
        pathfinder.get_topdown_view(meters_per_pixel, float(current_position[1]))
    )
    image = adapter.to_rgb_topdown(raw_map)
    adapter.draw_topdown_marker(
        image, pathfinder, current_position, meters_per_pixel, (0, 255, 0), radius=4
    )
    adapter.draw_topdown_arrow(
        image,
        pathfinder,
        current_position,
        forward,
        meters_per_pixel,
        color=(0, 255, 255),
    )
    marker_positions: list[tuple[int, int, int]] = []
    for obj in objects:
        color = (255, 90, 60) if obj.reachable else (160, 160, 160)
        adapter.draw_topdown_marker(
            image,
            pathfinder,
            obj.object_position,
            meters_per_pixel,
            color,
            radius=4,
        )
        px, py = adapter.topdown_xy(pathfinder, obj.object_position, meters_per_pixel)
        marker_positions.append((obj.map_index, px, py))

    cropped, offset = _crop_topdown_local(
        image=image,
        center_xy=adapter.topdown_xy(pathfinder, current_position, meters_per_pixel),
        radius_m=radius_m,
        meters_per_pixel=meters_per_pixel,
    )
    _draw_marker_numbers(cropped, marker_positions, offset)

    session.capture_counter += 1
    session_dir = adapter.session_output_dir(output_dir, session.session_id)
    item = adapter.write_png_image(
        image=cropped,
        output_dir=session_dir,
        name="oracle_local_map",
        session_id=session.session_id,
        step_count=session.step_count,
        capture_seq=session.capture_counter,
    )
    item["kind"] = "oracle_local_map"
    item["radius_m"] = round(radius_m, 3)
    item["object_count"] = len(objects)
    return item

def _crop_topdown_local(
    *,
    image: np.ndarray,
    center_xy: tuple[int, int],
    radius_m: float,
    meters_per_pixel: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    pixel_radius = max(8, int(math.ceil(radius_m / meters_per_pixel)))
    cx, cy = center_xy
    height, width = image.shape[:2]
    x0 = max(0, cx - pixel_radius)
    x1 = min(width, cx + pixel_radius + 1)
    y0 = max(0, cy - pixel_radius)
    y1 = min(height, cy + pixel_radius + 1)
    return np.asarray(image[y0:y1, x0:x1, :]).copy(), (x0, y0)

def _draw_marker_numbers(
    image: np.ndarray,
    marker_positions: list[tuple[int, int, int]],
    offset: tuple[int, int],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = _load_marker_font(ImageFont)
    x_offset, y_offset = offset
    for idx, px, py in marker_positions:
        x = int(px - x_offset + 5)
        y = int(py - y_offset - 10)
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            continue
        text = str(idx)
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    image[:, :, :] = np.asarray(pil_image, dtype=np.uint8)

def _load_marker_font(ImageFont: Any) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, 13)
            except Exception:
                pass
    return ImageFont.load_default()


def get_oracle_local_map(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for ``hab_oracle_local_map``.

    The response is intentionally agent-facing and local: object labels,
    bearings, distances, reachability, and an optional numbered topdown crop.
    World coordinates stay in ``_debug`` for auditing and are stripped by MCP.
    """

    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)

    scene_graph = getattr(session, "scene_graph", None)
    if not isinstance(scene_graph, Mapping):
        return {
            "ok": False,
            "backend": "oracle_local_gt",
            "status": "scene_graph_unavailable",
            "error": "scene graph is not available for this session",
        }

    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    forward = adapter.agent_forward(session)
    radius_m = _clamped_positive_float(
        payload.get("radius_m"),
        default=5.0,
        max_value=20.0,
    )
    max_objects = max(1, min(100, int(payload.get("max_objects", 30))))
    meters_per_pixel = _clamped_positive_float(
        payload.get("meters_per_pixel"),
        default=0.05,
        max_value=0.5,
    )
    include_image = _coerce_bool(payload.get("include_image"), default=True)
    label_filter = str(payload.get("label_filter") or "").strip()

    objects = _local_map_objects(
        scene_graph=scene_graph,
        pathfinder=pathfinder,
        current_position=current_position,
        forward=forward,
        radius_m=radius_m,
        max_objects=max_objects,
        label_filter=label_filter,
    )

    map_image = None
    if include_image:
        output_dir = str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts"))
        map_image = _render_oracle_local_map(
            adapter=adapter,
            session=session,
            pathfinder=pathfinder,
            current_position=current_position,
            forward=forward,
            objects=objects,
            radius_m=radius_m,
            meters_per_pixel=meters_per_pixel,
            output_dir=output_dir,
        )

    object_rows = [_local_map_object_public_row(obj) for obj in objects]
    body: dict[str, Any] = {
        "ok": True,
        "backend": "oracle_local_gt",
        "status": "ok",
        "session_id": session.session_id,
        "radius_m": round(radius_m, 3),
        "meters_per_pixel": meters_per_pixel,
        "label_filter": label_filter,
        "agent_heading_deg": round(float(adapter.agent_heading_degrees(session)), 1),
        "objects": object_rows,
        "object_count": len(object_rows),
        "map_image": map_image,
        "_debug": {
            "agent_position": current_position.tolist(),
            "objects": [_local_map_object_debug_row(obj) for obj in objects],
        },
    }
    return {k: v for k, v in body.items() if v is not None}

