# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Licensed under the MIT license; see THIRD_PARTY_NOTICES.md in the repository root.

from __future__ import annotations

import math
import os
import re
import struct
import subprocess
import tempfile
import zlib
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

from supernav.backends.habitat.simulator import sdk as habitat_sim

from supernav.backends.habitat.simulator.types import (
    _DEFAULT_DEPTH_VIS_MAX,
    _DEFAULT_MAX_OBSERVATION_ELEMENTS,
    _DEFAULT_VIDEO_FPS,
    _DEFAULT_VISUAL_OUTPUT_DIR,
    _Session,
    HabitatAdapterError,
)


class HabitatAdapterVisualMediaMixin:
    """Observation serialization, visuals export, and replay/video tooling."""

    _DEFAULT_AGENT_IMAGE_MAX_SIZE = 0
    _DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS = 4
    _VISIBLE_TARGET_OVERLAY_NMS_IOU = 0.35
    _VISIBLE_TARGET_OVERLAY_CONTAINMENT = 0.72
    _PANORAMA_VIEW_LABELS = {
        "front": "FRONT",
        "right": "RIGHT",
        "back": "BACK",
        "left": "LEFT",
    }

    @staticmethod
    def _session_output_dir(output_dir: str, session_id: str) -> str:
        """Return session-scoped output directory, creating it if needed.

        If *output_dir* already ends with the session_id component (e.g.
        when the caller runs inside a session subdirectory),
        return it unchanged to avoid doubling the session path segment.
        """
        # Check if the last or second-to-last path component is the session_id
        tail = os.path.basename(os.path.normpath(output_dir))
        parent_tail = os.path.basename(os.path.dirname(os.path.normpath(output_dir)))
        if tail == session_id or parent_tail == session_id:
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _get_observation(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        refresh = self._coerce_bool(
            payload.get("refresh", False), field_name="payload.refresh"
        )
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )

        if refresh or session.last_sensor_obs is None:
            session.last_sensor_obs = self._capture_sensor_observations(session)

        return {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "observation": self._serialize_observation(
                session.last_sensor_obs,
                include_data=include_data,
                max_elements=max_elements,
            ),
        }

    def _get_visuals(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        refresh = self._coerce_bool(
            payload.get("refresh", False), field_name="payload.refresh"
        )
        include_metrics = self._coerce_bool(
            payload.get("include_metrics", False),
            field_name="payload.include_metrics",
        )
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if depth_max <= 0:
            raise HabitatAdapterError('Field "payload.depth_max" must be > 0')

        sensors_payload = payload.get("sensors")
        sensors: Optional[list[str]]
        if sensors_payload is None:
            sensors = None
        elif isinstance(sensors_payload, list) and all(
            isinstance(item, str) for item in sensors_payload
        ):
            sensors = sensors_payload
        else:
            raise HabitatAdapterError(
                'Field "payload.sensors" must be list[str] when provided'
            )

        if refresh or session.last_sensor_obs is None:
            session.last_sensor_obs = self._capture_sensor_observations(session)

        session_dir = self._session_output_dir(output_dir, session.session_id)
        session.capture_counter += 1
        visuals = self._export_visuals(
            observation=session.last_sensor_obs,
            output_dir=session_dir,
            sensors=sensors,
            depth_max=depth_max,
            session_id=session.session_id,
            step_count=session.step_count,
            capture_seq=session.capture_counter,
            agent_image_max_size=self._coerce_int(
                payload.get(
                    "agent_image_max_size", self._DEFAULT_AGENT_IMAGE_MAX_SIZE
                ),
                field_name="payload.agent_image_max_size",
            ),
        )
        response = {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "output_dir": session_dir,
            "visuals": visuals,
        }
        if include_metrics:
            response["metrics"] = self._build_metrics(session)
        return response

    _PANORAMA_DIRECTIONS = ("front", "right", "back", "left")
    _PANORAMA_TURN_RIGHT_DEGREES = {
        "front": 0,
        "right": 90,
        "back": 180,
        "left": 270,
    }


    def _build_visible_nav_target_overlays(
        self,
        *,
        images: list[dict[str, Any]],
        visible_nav_targets: list[dict[str, Any]],
        output_dir: str,
        pano_seq: int,
        max_targets_per_image: int = _DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        registry: dict[str, dict[str, dict[str, Any]]] = {}
        if not images:
            return registry
        targets_by_direction: dict[str, list[dict[str, Any]]] = {}
        for target in visible_nav_targets:
            if not isinstance(target, Mapping):
                continue
            direction = str(target.get("direction") or "").lower()
            if direction not in self._PANORAMA_DIRECTIONS:
                continue
            if not str(target.get("target_ref") or "").strip():
                continue
            targets_by_direction.setdefault(direction, []).append(dict(target))
        for direction, rows in list(targets_by_direction.items()):
            targets_by_direction[direction] = self._select_overlay_targets(
                rows,
                max_targets=max_targets_per_image,
            )

        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return registry

        font = self._overlay_font(ImageFont)
        for image in images:
            image_ref = str(image.get("image_ref") or "")
            direction = str(image.get("direction") or "").lower()
            image_path = image.get("path")
            if not image_ref or not isinstance(image_path, str) or not image_path:
                continue
            rows = targets_by_direction.get(direction, [])
            if not rows:
                continue
            aliases: dict[str, dict[str, Any]] = {}
            try:
                canvas = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            draw = ImageDraw.Draw(canvas)
            occupied_label_boxes: list[tuple[int, int, int, int]] = []
            for target in rows:
                alias = str(target.get("target_ref") or "").strip()
                if not alias or alias in aliases:
                    continue
                self._draw_visible_nav_target_alias(
                    draw=draw,
                    image=canvas,
                    target=target,
                    alias=alias,
                    font=font,
                    occupied_label_boxes=occupied_label_boxes,
                )
                aliases[alias] = {
                    "target_ref": str(target.get("target_ref") or ""),
                    "direction": direction,
                    "capture_seq": pano_seq,
                }
            if not aliases:
                continue
            try:
                overlay_path = os.path.join(
                    output_dir,
                    f"pano_{direction}_step{pano_seq:06d}_visible_overlay.png",
                )
                canvas.save(overlay_path)
            except Exception:
                continue
            image["overlay_path"] = overlay_path
            image["overlay_aliases"] = list(aliases)
            registry[image_ref] = aliases
        return registry

    def _overlay_objlist_from_images(
        self,
        images: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        objlist: dict[str, list[str]] = {}
        for image in images:
            direction = str(image.get("direction") or "").lower()
            if direction not in self._PANORAMA_DIRECTIONS:
                continue
            aliases = image.get("overlay_aliases")
            if not isinstance(aliases, list):
                aliases = []
            objlist[direction] = [str(alias) for alias in aliases if str(alias)]
        return objlist

    @staticmethod
    def _overlay_target_sort_key(
        row: Mapping[str, Any],
    ) -> tuple[float, float, float, str]:
        score = HabitatAdapterVisualMediaMixin._overlay_target_quality_score(row)
        try:
            depth = float(row.get("depth_m") or row.get("_sort_depth") or 1.0e9)
        except Exception:
            depth = 1.0e9
        bbox = row.get("bbox_px") if isinstance(row.get("bbox_px"), Mapping) else {}
        try:
            area = float(bbox.get("width") or 0.0) * float(bbox.get("height") or 0.0)
        except Exception:
            area = 0.0
        return (-score, depth, -area, str(row.get("target_ref") or ""))

    @classmethod
    def _select_overlay_targets(
        cls,
        rows: list[dict[str, Any]],
        *,
        max_targets: int,
    ) -> list[dict[str, Any]]:
        limit = max(0, int(max_targets))
        if limit <= 0:
            return []
        candidates = [row for row in rows if cls._overlay_bbox(row) is not None]
        candidates.sort(key=cls._overlay_target_sort_key)
        selected: list[dict[str, Any]] = []
        for row in candidates:
            bbox = cls._overlay_bbox(row)
            if bbox is None:
                continue
            if any(
                cls._overlay_bboxes_overlap_too_much(bbox, other)
                for other in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _overlay_bbox(
        row: Mapping[str, Any],
    ) -> tuple[float, float, float, float] | None:
        bbox = row.get("bbox_px") if isinstance(row.get("bbox_px"), Mapping) else {}
        try:
            x = float(bbox.get("x") or 0.0)
            y = float(bbox.get("y") or 0.0)
            width = float(bbox.get("width") or 0.0)
            height = float(bbox.get("height") or 0.0)
        except Exception:
            return None
        if width < 4.0 or height < 4.0:
            return None
        return (x, y, x + width, y + height)

    @classmethod
    def _overlay_bboxes_overlap_too_much(
        cls,
        bbox: tuple[float, float, float, float],
        selected_row: Mapping[str, Any],
    ) -> bool:
        other = cls._overlay_bbox(selected_row)
        if other is None:
            return False
        intersection = cls._overlay_bbox_intersection_area(bbox, other)
        if intersection <= 0.0:
            return False
        area = cls._overlay_bbox_area(bbox)
        other_area = cls._overlay_bbox_area(other)
        union = max(area + other_area - intersection, 1.0e-6)
        iou = intersection / union
        containment = intersection / max(min(area, other_area), 1.0e-6)
        return (
            iou >= cls._VISIBLE_TARGET_OVERLAY_NMS_IOU
            or containment >= cls._VISIBLE_TARGET_OVERLAY_CONTAINMENT
        )

    @staticmethod
    def _overlay_bbox_area(bbox: tuple[float, float, float, float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _overlay_bbox_intersection_area(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        x0 = max(left[0], right[0])
        y0 = max(left[1], right[1])
        x1 = min(left[2], right[2])
        y1 = min(left[3], right[3])
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @staticmethod
    def _overlay_target_quality_score(row: Mapping[str, Any]) -> float:
        bbox = HabitatAdapterVisualMediaMixin._overlay_bbox(row)
        area = (
            HabitatAdapterVisualMediaMixin._overlay_bbox_area(bbox) if bbox else 0.0
        )
        try:
            visible_fraction = float(row.get("visible_fraction") or 0.0)
        except Exception:
            visible_fraction = 0.0
        try:
            semantic_overlap = float(row.get("semantic_overlap") or 0.0)
        except Exception:
            semantic_overlap = 0.0
        try:
            depth = float(row.get("depth_m") or row.get("_sort_depth") or 1.0e9)
        except Exception:
            depth = 1.0e9
        area_score = min(area / 4096.0, 1.0)
        if visible_fraction > 0.45:
            area_score *= max(0.35, 1.0 - (visible_fraction - 0.45))
        depth_score = 1.0 / max(depth, 0.25)
        semantic_score = 2.0 * min(max(semantic_overlap, 0.0), 1.0)
        return (
            semantic_score
            + 1.5 * min(max(visible_fraction, 0.0), 1.0)
            + 0.35 * area_score
            + 0.10 * depth_score
        )

    @staticmethod
    def _overlay_font(image_font_module: Any) -> Any:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            if os.path.exists(path):
                try:
                    return image_font_module.truetype(path, size=11)
                except Exception:
                    pass
        return image_font_module.load_default()

    @staticmethod
    def _draw_visible_nav_target_alias(
        *,
        draw: Any,
        image: Any,
        target: Mapping[str, Any],
        alias: str,
        font: Any,
        occupied_label_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        bbox = target.get("bbox_px") if isinstance(target.get("bbox_px"), Mapping) else {}
        try:
            x = int(bbox.get("x") or 0)
            y = int(bbox.get("y") or 0)
            width = int(bbox.get("width") or 0)
            height = int(bbox.get("height") or 0)
        except Exception:
            return
        if width <= 0 or height <= 0:
            return
        x0 = max(0, min(int(image.width) - 1, x))
        y0 = max(0, min(int(image.height) - 1, y))
        x1 = max(0, min(int(image.width) - 1, x + width - 1))
        y1 = max(0, min(int(image.height) - 1, y + height - 1))
        if x1 <= x0 or y1 <= y0:
            return
        color = (255, 230, 0)
        outline = (0, 0, 0)
        px = int(round(x0 + (x1 - x0) * 0.5))
        py = int(round(y0 + (y1 - y0) * 0.88))
        px = max(0, min(int(image.width) - 1, px))
        py = max(0, min(int(image.height) - 1, py))

        try:
            text_box = draw.textbbox((0, 0), alias, font=font)
            text_w = int(text_box[2] - text_box[0])
            text_h = int(text_box[3] - text_box[1])
        except Exception:
            text_w = max(1, len(alias) * 7)
            text_h = 12
        pad_x = 4
        pad_y = 2
        label_w = text_w + 2 * pad_x
        label_h = text_h + 2 * pad_y
        occupied = occupied_label_boxes if occupied_label_boxes is not None else []
        tx, ty, line_to = HabitatAdapterVisualMediaMixin._overlay_label_position(
            image_width=int(image.width),
            image_height=int(image.height),
            pin_x=px,
            pin_y=py,
            label_w=label_w,
            label_h=label_h,
            occupied=occupied,
        )

        draw.line((px, py, line_to[0], line_to[1]), fill=color, width=1)
        radius = 4
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=color,
            outline=outline,
        )
        text_xy = (tx + pad_x, ty + pad_y)
        try:
            draw.text(
                text_xy,
                alias,
                fill=color,
                font=font,
                stroke_width=2,
                stroke_fill=outline,
            )
        except TypeError:
            # Older Pillow fallback: keep the label background transparent.
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text(
                    (text_xy[0] + dx, text_xy[1] + dy),
                    alias,
                    fill=outline,
                    font=font,
                )
            draw.text(text_xy, alias, fill=color, font=font)
        occupied.append((tx, ty, tx + label_w, ty + label_h))

    @staticmethod
    def _overlay_label_position(
        *,
        image_width: int,
        image_height: int,
        pin_x: int,
        pin_y: int,
        label_w: int,
        label_h: int,
        occupied: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, tuple[int, int]]:
        max_x = max(0, image_width - label_w - 1)
        max_y = max(0, image_height - label_h - 1)

        def clamp_box(x: int, y: int) -> tuple[int, int, int, int]:
            tx = max(0, min(x, max_x))
            ty = max(0, min(y, max_y))
            return tx, ty, tx + label_w, ty + label_h

        candidates = [
            clamp_box(pin_x - label_w // 2, pin_y - label_h - 8),
            clamp_box(pin_x + 8, pin_y - label_h // 2),
            clamp_box(pin_x - label_w - 8, pin_y - label_h // 2),
            clamp_box(pin_x - label_w // 2, pin_y + 8),
        ]
        chosen = candidates[0]
        for candidate in candidates:
            if not any(
                HabitatAdapterVisualMediaMixin._overlay_label_boxes_overlap(
                    candidate,
                    existing,
                )
                for existing in occupied
            ):
                chosen = candidate
                break
        tx0, ty0, tx1, ty1 = chosen
        if pin_y < ty0:
            line_to = (pin_x, ty0)
        elif pin_y > ty1:
            line_to = (pin_x, ty1)
        elif pin_x < tx0:
            line_to = (tx0, pin_y)
        else:
            line_to = (tx1, pin_y)
        return tx0, ty0, line_to

    @staticmethod
    def _overlay_label_boxes_overlap(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> bool:
        return not (
            left[2] < right[0]
            or right[2] < left[0]
            or left[3] < right[1]
            or right[3] < left[1]
        )

    @staticmethod
    def _depth_array_from_observation(
        obs: Mapping[str, Any],
    ) -> np.ndarray | None:
        if "depth_sensor" not in obs:
            return None
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            return None
        return depth

    @staticmethod
    def _semantic_array_from_observation(
        obs: Mapping[str, Any],
    ) -> np.ndarray | None:
        if "semantic_sensor" not in obs:
            return None
        semantic = np.asarray(obs["semantic_sensor"])
        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        if semantic.ndim != 2:
            return None
        return semantic


    @staticmethod
    def _aabb_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
        lo = np.minimum(bbox_min, bbox_max).astype(np.float32)
        hi = np.maximum(bbox_min, bbox_max).astype(np.float32)
        return np.asarray(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _project_aabb_to_image(
        *,
        corners: np.ndarray,
        camera_position: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[int, int, int, int, float, float, float] | None:
        points: list[tuple[float, float, float]] = []
        depths: list[float] = []
        for corner in corners:
            delta = np.asarray(corner, dtype=np.float32) - camera_position
            z_forward = float(np.dot(delta, forward))
            if z_forward <= 0.05:
                continue
            x_right = float(np.dot(delta, right))
            y_up = float(np.dot(delta, up))
            px = cx + fx * (x_right / z_forward)
            py = cy - fy * (y_up / z_forward)
            points.append((px, py, z_forward))
            depths.append(z_forward)
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if max(xs) < 0 or min(xs) >= width or max(ys) < 0 or min(ys) >= height:
            return None
        x0 = int(max(0, min(width - 1, math.floor(min(xs)))))
        y0 = int(max(0, min(height - 1, math.floor(min(ys)))))
        x1 = int(max(0, min(width - 1, math.ceil(max(xs)))))
        y1 = int(max(0, min(height - 1, math.ceil(max(ys)))))
        if x1 < x0 or y1 < y0:
            return None
        return (
            x0,
            y0,
            x1,
            y1,
            float(min(depths)),
            float(np.mean(depths)),
            float(max(depths)),
        )

    @staticmethod
    def _depth_visible_fraction(
        *,
        depth: np.ndarray,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        expected_depth_range_m: tuple[float, float],
    ) -> tuple[float, float | None]:
        x0, y0, x1, y1 = bbox
        depth_height, depth_width = depth.shape[:2]
        if depth_width <= 0 or depth_height <= 0:
            return 0.0, None
        xs = np.linspace(x0, x1, num=min(5, max(1, x1 - x0 + 1)))
        ys = np.linspace(y0, y1, num=min(5, max(1, y1 - y0 + 1)))
        valid = 0
        expected_near, expected_far = sorted(
            [float(expected_depth_range_m[0]), float(expected_depth_range_m[1])]
        )
        visible = 0
        visible_depths: list[float] = []
        for y in ys:
            for x in xs:
                dx = int(round(float(x) * (depth_width - 1) / max(1, image_width - 1)))
                dy = int(
                    round(float(y) * (depth_height - 1) / max(1, image_height - 1))
                )
                observed = float(depth[dy, dx])
                if not math.isfinite(observed) or observed <= 0.0:
                    continue
                valid += 1
                if expected_near - 0.25 <= observed <= expected_far + 0.25:
                    visible += 1
                    visible_depths.append(observed)
        if valid <= 0:
            return 0.0, None
        fraction = visible / valid
        if not visible_depths:
            return fraction, None
        return fraction, float(np.median(np.asarray(visible_depths, dtype=np.float32)))

    @staticmethod
    def _semantic_visible_fraction(
        *,
        semantic: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        object_record: Mapping[str, Any],
    ) -> float | None:
        if semantic is None:
            return None
        ids = HabitatAdapterVisualMediaMixin._semantic_ids_for_object(object_record)
        if not ids:
            return None
        x0, y0, x1, y1 = bbox
        sem_height, sem_width = semantic.shape[:2]
        if sem_width <= 0 or sem_height <= 0:
            return None
        sx0 = int(round(float(x0) * (sem_width - 1) / max(1, image_width - 1)))
        sx1 = int(round(float(x1) * (sem_width - 1) / max(1, image_width - 1)))
        sy0 = int(round(float(y0) * (sem_height - 1) / max(1, image_height - 1)))
        sy1 = int(round(float(y1) * (sem_height - 1) / max(1, image_height - 1)))
        sx0, sx1 = sorted((max(0, sx0), min(sem_width - 1, sx1)))
        sy0, sy1 = sorted((max(0, sy0), min(sem_height - 1, sy1)))
        region = np.asarray(semantic[sy0 : sy1 + 1, sx0 : sx1 + 1])
        if region.size <= 0:
            return None
        mask = np.isin(region.astype(np.int64, copy=False), list(ids))
        return float(np.count_nonzero(mask)) / float(region.size)

    @staticmethod
    def _semantic_ids_for_object(obj: Mapping[str, Any]) -> set[int]:
        ids: set[int] = set()

        def add_value(value: Any) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add_value(item)
                return
            if isinstance(value, Mapping):
                for key in ("id", "semantic_id", "instance_id", "ins_id"):
                    if key in value:
                        add_value(value[key])
                return
            if isinstance(value, str):
                match = re.fullmatch(r"(?:obj[_-])?(\d+)", value.strip())
                if not match:
                    return
                value = match.group(1)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed >= 0:
                ids.add(parsed)

        for key in (
            "semantic_id",
            "semantic_ids",
            "semantic_instance_id",
            "semantic_instance_ids",
            "instance_id",
            "ins_id",
        ):
            if key in obj:
                add_value(obj[key])
        return ids

    def _line_of_sight_reaches_aabb(
        self,
        *,
        session: _Session,
        camera_position: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> bool:
        simulator = getattr(session, "simulator", None)
        if simulator is None or not hasattr(simulator, "cast_ray"):
            return True
        lo = np.minimum(bbox_min, bbox_max).astype(np.float32)
        hi = np.maximum(bbox_min, bbox_max).astype(np.float32)
        center = ((lo + hi) * 0.5).astype(np.float32)
        samples = [
            center,
            np.asarray([center[0], center[1], lo[2]], dtype=np.float32),
            np.asarray([center[0], center[1], hi[2]], dtype=np.float32),
            np.asarray([lo[0], center[1], center[2]], dtype=np.float32),
            np.asarray([hi[0], center[1], center[2]], dtype=np.float32),
        ]
        for sample in samples:
            ray_vec = sample - camera_position
            distance = float(np.linalg.norm(ray_vec))
            if distance <= 1e-4:
                return True
            direction = (ray_vec / distance).astype(np.float32)
            interval = self._ray_aabb_interval(
                origin=camera_position,
                direction=direction,
                bbox_min=lo,
                bbox_max=hi,
            )
            if interval is None:
                continue
            entry, exit_ = interval
            if exit_ < 0.0:
                continue
            try:
                ray = habitat_sim.geo.Ray(camera_position, direction)
                hits = simulator.cast_ray(ray, max_distance=max(exit_ + 0.5, 0.5))
            except Exception:
                return True
            has_hits = bool(hits.has_hits()) if hasattr(hits, "has_hits") else bool(
                getattr(hits, "hits", None)
            )
            if not has_hits:
                return True
            hit_rows = list(getattr(hits, "hits", []) or [])
            if not hit_rows:
                return True
            first_distance = float(getattr(hit_rows[0], "ray_distance", 0.0))
            if first_distance + 0.25 >= max(0.0, entry):
                return True
        return False

    @staticmethod
    def _ray_aabb_interval(
        *,
        origin: np.ndarray,
        direction: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> tuple[float, float] | None:
        t_min = -math.inf
        t_max = math.inf
        for axis in range(3):
            d = float(direction[axis])
            o = float(origin[axis])
            lo = float(bbox_min[axis])
            hi = float(bbox_max[axis])
            if abs(d) < 1e-8:
                if o < lo or o > hi:
                    return None
                continue
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            near = min(t1, t2)
            far = max(t1, t2)
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return None
        return float(t_min), float(t_max)

    def _camera_basis_from_state(
        self,
        *,
        session: _Session,
        agent_state: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sensor_state = None
        sensor_states = getattr(agent_state, "sensor_states", None)
        if isinstance(sensor_states, Mapping):
            sensor_state = sensor_states.get("color_sensor")
        elif sensor_states is not None and hasattr(sensor_states, "get"):
            try:
                sensor_state = sensor_states.get("color_sensor")
            except Exception:
                sensor_state = None
        position = self._state_vec3(getattr(sensor_state, "position", None))
        if position is None:
            position = self._state_vec3(getattr(agent_state, "position", None))
            if position is None:
                position = np.zeros(3, dtype=np.float32)
            position = position.copy()
            position[1] += float(session.settings.get("sensor_height", 1.2) or 1.2)
        rotation = self._to_numeric_list(getattr(sensor_state, "rotation", None))
        if not isinstance(rotation, list) or len(rotation) != 4:
            rotation = self._to_numeric_list(getattr(agent_state, "rotation", None))
        if not isinstance(rotation, list) or len(rotation) != 4:
            rotation = [1.0, 0.0, 0.0, 0.0]
        right = self._quat_rotate(rotation, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        up = self._quat_rotate(rotation, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        forward = self._quat_rotate(
            rotation,
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        )
        return (
            position.astype(np.float32),
            self._unit_vector(right, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
            self._unit_vector(up, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
            self._unit_vector(forward, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)),
        )

    @staticmethod
    def _state_vec3(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.shape != (3,) or np.any(np.isnan(arr)):
            return None
        return arr

    @staticmethod
    def _quat_rotate(rotation: list[float], vector: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in rotation]
        q_vec = np.asarray([x, y, z], dtype=np.float32)
        vec = np.asarray(vector, dtype=np.float32)
        uv = np.cross(q_vec, vec)
        uuv = np.cross(q_vec, uv)
        return vec + 2.0 * (w * uv + uuv)

    @staticmethod
    def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return fallback.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def _dedupe_visible_nav_targets(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        direction_order = {
            direction: idx for idx, direction in enumerate(self._PANORAMA_DIRECTIONS)
        }
        rows.sort(
            key=lambda row: (
                direction_order.get(str(row.get("direction")), 999),
                -float(row.get("_sort_area") or 0.0),
                float(row.get("_sort_depth") or 1.0e9),
            )
        )
        seen: set[str] = set()
        public_rows: list[dict[str, Any]] = []
        for row in rows:
            ref = str(row.get("target_ref") or "")
            if not ref or ref in seen:
                continue
            seen.add(ref)
            public_rows.append(
                {k: v for k, v in row.items() if not str(k).startswith("_sort_")}
            )
        return public_rows

    def _export_video_trace(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )

        sensor = payload.get("sensor", "color_sensor")
        if not isinstance(sensor, str) or not sensor:
            raise HabitatAdapterError(
                'Field "payload.sensor" must be a non-empty str'
            )

        fps = self._coerce_float(
            payload.get("fps", _DEFAULT_VIDEO_FPS),
            field_name="payload.fps",
        )
        if fps <= 0:
            raise HabitatAdapterError('Field "payload.fps" must be > 0')

        step_start_raw = payload.get("step_start")
        step_end_raw = payload.get("step_end")
        step_start = (
            self._coerce_int(step_start_raw, "payload.step_start")
            if step_start_raw is not None
            else None
        )
        step_end = (
            self._coerce_int(step_end_raw, "payload.step_end")
            if step_end_raw is not None
            else None
        )
        if step_start is not None and step_end is not None and step_end < step_start:
            raise HabitatAdapterError(
                'Field "payload.step_end" must be >= "payload.step_start"'
            )

        include_metrics = self._coerce_bool(
            payload.get("include_metrics", True),
            field_name="payload.include_metrics",
        )
        include_publish_hints = self._coerce_bool(
            payload.get("include_publish_hints", True),
            field_name="payload.include_publish_hints",
        )
        frame_records = self._find_visual_frame_records(
            output_dir=output_dir,
            session_id=session.session_id,
            sensor=sensor,
            step_start=step_start,
            step_end=step_end,
        )
        if not frame_records:
            raise HabitatAdapterError(
                "No trace frames found for "
                f'session="{session.session_id}" sensor="{sensor}" '
                f'in "{output_dir}"'
            )

        first_step = frame_records[0][0]
        last_step = frame_records[-1][0]
        frame_paths = [path for _, path in frame_records]
        safe_sensor = self._safe_sensor_name(sensor)
        # Place video in the same directory where frames were found
        video_dir = os.path.dirname(frame_paths[0])
        video_path = os.path.join(
            video_dir,
            f"steps{first_step:06d}-{last_step:06d}_{safe_sensor}.mp4",
        )
        self._encode_video_trace(frame_paths=frame_paths, video_path=video_path, fps=fps)

        video_item: Dict[str, Any] = {
            "ok": True,
            "sensor": sensor,
            "path": video_path,
            "bytes": os.path.getsize(video_path),
            "mime_type": "video/mp4",
            "frame_count": len(frame_paths),
            "fps": fps,
            "duration_s": round(len(frame_paths) / fps, 3),
            "step_start": first_step,
            "step_end": last_step,
        }
        response: Dict[str, Any] = {
            "session_id": session.session_id,
            "output_dir": output_dir,
            "video": video_item,
        }
        if include_metrics:
            metrics = self._build_metrics(session)
            response["metrics"] = metrics
        else:
            metrics = None
        if include_publish_hints:
            response["publish_hints"] = self._build_video_publish_hints(
                session=session,
                video_item=video_item,
                metrics=metrics,
            )
        return response

    def _get_metrics(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        metrics = self._build_metrics(session)
        include_trajectory = self._coerce_bool(
            payload.get("include_trajectory", False),
            field_name="payload.include_trajectory",
        )
        if include_trajectory and not session.mapless:
            metrics["trajectory_points"] = list(session.trajectory)
            metrics["cumulative_path_length"] = float(session.cumulative_path_length)
        return metrics

    def _get_audit_metrics(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        metrics = self._build_metrics(session)
        include_trajectory = self._coerce_bool(
            payload.get("include_trajectory", False),
            field_name="payload.include_trajectory",
        )
        if include_trajectory:
            metrics["trajectory_points"] = list(
                session.audit_trajectory or session.trajectory
            )
            metrics["cumulative_path_length"] = float(session.cumulative_path_length)
        if session.mapless:
            agent = session.simulator.get_agent(session.agent_id)
            state = agent.get_state()
            metrics["agent_state"] = {
                "position": self._to_numeric_list(state.position),
                "rotation": self._to_numeric_list(state.rotation),
                "heading_deg": round(self._heading_degrees(session), 3),
            }
        return metrics

    def _build_metrics(self, session: _Session) -> Dict[str, Any]:
        agent = session.simulator.get_agent(session.agent_id)
        state = agent.get_state()
        step_time_s = getattr(session.simulator, "last_physics_step_time_s", None)

        metrics: Dict[str, Any] = {
            "session_id": session.session_id,
            "scene": session.scene,
            "step_count": session.step_count,
            "last_action": session.last_action,
            "available_actions": self._get_available_actions(session),
            "trajectory_length": len(session.trajectory),
            "collision_count": len(session.collision_points),
            "last_collision": session.last_collision,
            "state_summary": self._build_state_summary(session),
            "step_time_s": float(step_time_s) if step_time_s is not None else None,
        }
        if not session.mapless:
            metrics["agent_state"] = {
                "position": self._to_numeric_list(state.position),
                "rotation": self._to_numeric_list(state.rotation),
            }
            metrics["current_goal"] = session.last_goal
        else:
            metrics["agent_state"] = {
                "heading_deg": round(self._heading_degrees(session), 3),
            }
        return metrics

    def _build_publish_hints(
        self,
        session: _Session,
        action: str,
        collided: bool,
        visuals: Mapping[str, Mapping[str, Any]],
        metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        media_items: list[Dict[str, Any]] = []
        for sensor_name, item in visuals.items():
            if not item.get("ok", False):
                continue
            path = item.get("path")
            if not isinstance(path, str):
                continue
            media_items.append(
                {
                    "sensor": sensor_name,
                    "path": path,
                    "mime_type": item.get("mime_type", "image/png"),
                }
            )

        if metrics is not None:
            position = metrics.get("agent_state", {}).get("position")
            step_time_s = metrics.get("step_time_s")
        else:
            position = None
            step_time_s = None

        summary_parts = [
            f"scene={session.scene}",
            f"step={session.step_count}",
            f"action={action}",
            f"collided={collided}",
        ]
        if position is not None:
            summary_parts.append(f"position={position}")
        if step_time_s is not None:
            summary_parts.append(f"step_time_s={step_time_s:.6f}")
        summary_text = " | ".join(summary_parts)
        for item in media_items:
            item["label"] = f"{item['sensor']} frame"

        return {
            "summary_text": summary_text,
            "media_items": media_items,
            "agent_send_actions": self._build_agent_send_actions(
                summary_text=summary_text,
                media_items=media_items,
            ),
        }

    def _build_video_publish_hints(
        self,
        session: _Session,
        video_item: MutableMapping[str, Any],
        metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        position = None
        if metrics is not None:
            position = metrics.get("agent_state", {}).get("position")

        summary_parts = [
            f"scene={session.scene}",
            ("trace_steps=" f"{video_item['step_start']}-{video_item['step_end']}"),
            f"frames={video_item['frame_count']}",
            f"fps={video_item['fps']}",
            f"duration_s={video_item['duration_s']:.3f}",
        ]
        if position is not None:
            summary_parts.append(f"position={position}")
        summary_text = " | ".join(summary_parts)

        video_item["label"] = f"{video_item['sensor']} trace video"
        return {
            "summary_text": summary_text,
            "media_items": [dict(video_item)],
            "agent_send_actions": self._build_agent_send_actions(
                summary_text=summary_text,
                media_items=[video_item],
            ),
        }

    @staticmethod
    def _serialize_observation(
        observation: Optional[Mapping[str, Any]],
        include_data: bool,
        max_elements: int,
    ) -> Dict[str, Any]:
        if observation is None:
            return {}

        serialized: Dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                serialized[key] = HabitatAdapterVisualMediaMixin._serialize_array(
                    value, include_data=include_data, max_elements=max_elements
                )
            elif hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(
                value, "numpy"
            ):
                np_value = value.detach().cpu().numpy()
                serialized[key] = HabitatAdapterVisualMediaMixin._serialize_array(
                    np_value, include_data=include_data, max_elements=max_elements
                )
            elif isinstance(value, (bool, int, float, str)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def _serialize_array(
        array: np.ndarray, include_data: bool, max_elements: int
    ) -> Dict[str, Any]:
        np_array = np.asarray(array)
        result: Dict[str, Any] = {
            "dtype": str(np_array.dtype),
            "shape": list(np_array.shape),
            "num_elements": int(np_array.size),
        }

        if np_array.size > 0:
            try:
                result["min"] = float(np_array.min())
                result["max"] = float(np_array.max())
            except (TypeError, ValueError):
                pass

        if include_data:
            flat = np_array.reshape(-1)
            if flat.size > max_elements:
                result["data"] = flat[:max_elements].tolist()
                result["truncated"] = True
            else:
                result["data"] = np_array.tolist()
                result["truncated"] = False
        return result

    @staticmethod
    def _to_numeric_list(value: Any) -> Any:
        try:
            import quaternion  # type: ignore

            if isinstance(value, quaternion.quaternion):
                return quaternion.as_float_array(value).tolist()
        except ImportError:
            pass

        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        if hasattr(value, "__iter__"):
            try:
                return list(value)
            except TypeError:
                pass
        return value

    @staticmethod
    def _to_numpy_array(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if (
            hasattr(value, "detach")
            and hasattr(value, "cpu")
            and hasattr(value, "numpy")
        ):
            return value.detach().cpu().numpy()
        raise HabitatAdapterError("Observation is not a numpy-compatible array")

    @staticmethod
    def _to_uint8_color(array: np.ndarray) -> np.ndarray:
        np_array = np.asarray(array)
        if np_array.ndim != 3 or np_array.shape[2] not in (3, 4):
            raise HabitatAdapterError("Color visualization expects shape [H, W, 3|4]")

        if np.issubdtype(np_array.dtype, np.floating):
            np_array = np.nan_to_num(np_array, nan=0.0, posinf=1.0, neginf=0.0)
            max_val = float(np.max(np_array)) if np_array.size else 1.0
            if max_val <= 1.0:
                np_array = np_array * 255.0
            np_array = np.clip(np_array, 0.0, 255.0).astype(np.uint8)
        elif np_array.dtype != np.uint8:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        return np_array

    @staticmethod
    def _to_uint8_depth(array: np.ndarray, depth_max: float) -> np.ndarray:
        depth = np.asarray(array, dtype=np.float32)
        if depth.ndim != 2:
            raise HabitatAdapterError("Depth visualization expects shape [H, W]")
        depth = np.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, depth_max)
        normalized = 1.0 - (depth / depth_max)
        return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _downsample_image_nearest(image: np.ndarray, max_size: int) -> np.ndarray:
        if max_size <= 0:
            return image
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= max_size:
            return image
        scale = max_size / float(longest)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        y_idx = np.linspace(0, height - 1, new_h).round().astype(np.int64)
        x_idx = np.linspace(0, width - 1, new_w).round().astype(np.int64)
        return image[np.ix_(y_idx, x_idx)]

    @staticmethod
    def _encode_png_bytes(image: np.ndarray) -> bytes:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            raise HabitatAdapterError("PNG encoder expects uint8 arrays")

        if image_array.ndim == 2:
            height, width = image_array.shape
            color_type = 0
            raw_rows = [image_array[row].tobytes() for row in range(height)]
        elif image_array.ndim == 3 and image_array.shape[2] in (3, 4):
            height, width, channels = image_array.shape
            color_type = 2 if channels == 3 else 6
            raw_rows = [image_array[row].tobytes() for row in range(height)]
        else:
            raise HabitatAdapterError(
                "PNG encoder expects [H, W] or [H, W, 3|4] uint8 arrays"
            )

        scanlines = b"".join(b"\x00" + row_bytes for row_bytes in raw_rows)
        compressed = zlib.compress(scanlines, level=6)

        def chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
            length = struct.pack(">I", len(chunk_data))
            crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            return length + chunk_type + chunk_data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_signature = b"\x89PNG\r\n\x1a\n"
        return (
            png_signature
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )

    @staticmethod
    def _safe_sensor_name(sensor_name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", sensor_name).strip("_") or "sensor"

    def _parse_visual_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if depth_max <= 0:
            raise HabitatAdapterError('Field "payload.depth_max" must be > 0')

        sensors_payload = payload.get("sensors")
        if sensors_payload is None:
            sensors = None
        elif isinstance(sensors_payload, list) and all(
            isinstance(item, str) for item in sensors_payload
        ):
            sensors = sensors_payload
        else:
            raise HabitatAdapterError(
                'Field "payload.sensors" must be list[str] when provided'
            )

        return {
            "output_dir": output_dir,
            "depth_max": depth_max,
            "sensors": sensors,
            "agent_image_max_size": self._coerce_int(
                payload.get("agent_image_max_size", 0),
                field_name="payload.agent_image_max_size",
            ),
        }

    @staticmethod
    def _build_agent_send_actions(
        summary_text: str, media_items: list[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        agent_send_actions: list[Dict[str, Any]] = [
            {"action": "send", "message": summary_text}
        ]
        for item in media_items:
            media_path = item.get("path")
            if not isinstance(media_path, str):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label:
                label = "media"
            agent_send_actions.append(
                {
                    "action": "send",
                    "media": media_path,
                    "message": label,
                }
            )
        return agent_send_actions

    @staticmethod
    def _to_rgb_topdown(raw_map: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_map)
        if raw.ndim != 2:
            raise HabitatAdapterError("Topdown map must be a 2D array")
        binary = np.where(raw > 0, 255, 24).astype(np.uint8)
        return np.stack([binary, binary, binary], axis=-1)

    @staticmethod
    def _topdown_xy(
        pathfinder: Any, point: Any, meters_per_pixel: float
    ) -> tuple[int, int]:
        bounds = pathfinder.get_bounds()
        px = int(round((float(point[0]) - float(bounds[0][0])) / meters_per_pixel))
        py = int(round((float(point[2]) - float(bounds[0][2])) / meters_per_pixel))
        return px, py

    @staticmethod
    def _draw_circle(
        image: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]
    ) -> None:
        height, width = image.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    image[y, x, :3] = color

    @staticmethod
    def _draw_line(
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy), 1)
        for step in range(steps + 1):
            t = step / steps
            x = int(round(start[0] + dx * t))
            y = int(round(start[1] + dy * t))
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                image[y, x, :3] = color

    def _visual_robot_metadata_for_rgb(
        self,
        image: np.ndarray,
        observation: Optional[Mapping[str, Any]],
        session_id: str,
        sensor_name: str,
    ) -> Dict[str, Any]:
        del image, observation
        session = self._sessions.get(session_id)
        if session is None:
            return {}
        metadata: Dict[str, Any] = {
            "visual_robot_enabled": bool(session.visual_robot_enabled),
            "visual_robot_loaded": session.visual_robot_obj is not None,
        }
        if session.visual_robot_kind:
            metadata["visual_robot_kind"] = session.visual_robot_kind
        if sensor_name != "color_sensor":
            return metadata
        pitch = float(getattr(session, "camera_pitch_deg", 0.0))
        metadata["camera_pitch_deg"] = round(pitch, 3)
        metadata["visual_robot_first_person_hidden"] = self._env_bool(
            "HAB_VISUAL_ROBOT_HIDE_FROM_FIRST_PERSON", False
        )
        return metadata

    def _draw_topdown_marker(
        self,
        image: np.ndarray,
        pathfinder: Any,
        point: Any,
        meters_per_pixel: float,
        color: tuple[int, int, int],
        radius: int = 3,
    ) -> None:
        px, py = self._topdown_xy(pathfinder, point, meters_per_pixel)
        self._draw_circle(image, px, py, radius=radius, color=color)

    def _draw_topdown_path(
        self,
        image: np.ndarray,
        pathfinder: Any,
        points: list[list[float]],
        meters_per_pixel: float,
        color: tuple[int, int, int] = (255, 80, 80),
    ) -> None:
        if len(points) < 2:
            return
        topdown_points = [
            self._topdown_xy(pathfinder, point, meters_per_pixel) for point in points
        ]
        for start, end in zip(topdown_points[:-1], topdown_points[1:]):
            self._draw_line(image, start, end, color)

    @staticmethod
    def _fill_triangle(
        image: np.ndarray,
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        min_x = max(0, min(a[0], b[0], c[0]))
        max_x = min(image.shape[1] - 1, max(a[0], b[0], c[0]))
        min_y = max(0, min(a[1], b[1], c[1]))
        max_y = min(image.shape[0] - 1, max(a[1], b[1], c[1]))

        def _edge(p0: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int]) -> int:
            return (
                (p2[0] - p0[0]) * (p1[1] - p0[1])
                - (p2[1] - p0[1]) * (p1[0] - p0[0])
            )

        area = _edge(a, b, c)
        if area == 0:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x, y)
                w0 = _edge(b, c, p)
                w1 = _edge(c, a, p)
                w2 = _edge(a, b, p)
                if area > 0:
                    inside = w0 >= 0 and w1 >= 0 and w2 >= 0
                else:
                    inside = w0 <= 0 and w1 <= 0 and w2 <= 0
                if inside:
                    image[y, x, :3] = color

    def _draw_topdown_arrow(
        self,
        image: np.ndarray,
        pathfinder: Any,
        point: Any,
        direction: np.ndarray,
        meters_per_pixel: float,
        color: tuple[int, int, int],
    ) -> None:
        start = self._topdown_xy(pathfinder, point, meters_per_pixel)
        direction = np.asarray(direction, dtype=np.float32)
        tip_point = np.asarray(point, dtype=np.float32) + direction * 0.5
        tip = self._topdown_xy(pathfinder, tip_point, meters_per_pixel)
        screen_vec = np.array(
            [float(tip[0] - start[0]), float(tip[1] - start[1])], dtype=np.float32
        )
        norm = float(np.linalg.norm(screen_vec))
        if norm < 1.0:
            self._draw_circle(image, tip[0], tip[1], radius=2, color=color)
            return

        unit = screen_vec / norm
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        head_len = min(10.0, max(6.0, norm * 0.35))
        head_half_width = max(3.0, head_len * 0.45)
        base_center = np.array(tip, dtype=np.float32) - unit * head_len
        left_tip = tuple(np.rint(base_center + perp * head_half_width).astype(int))
        right_tip = tuple(np.rint(base_center - perp * head_half_width).astype(int))
        shaft_end = tuple(np.rint(base_center).astype(int))

        self._draw_line(image, start, shaft_end, color)
        self._fill_triangle(image, tip, left_tip, right_tip, color)

    def _write_png_image(
        self,
        image: np.ndarray,
        output_dir: str,
        name: str,
        session_id: str,
        step_count: int,
        capture_seq: Optional[int] = None,
        filename_prefix: str = "",
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        png_bytes = self._encode_png_bytes(image)
        safe_name = self._safe_sensor_name(name)
        # Use capture_seq (monotonic) to avoid overwriting when look is
        # called multiple times at the same step_count.
        seq = capture_seq if capture_seq is not None else step_count
        prefix_part = f"{filename_prefix}_" if filename_prefix else ""
        file_path = os.path.join(
            output_dir, f"{prefix_part}step{seq:06d}_{safe_name}.png"
        )
        with open(file_path, "wb") as file_handle:
            file_handle.write(png_bytes)

        height, width = image.shape[:2]
        item: Dict[str, Any] = {
            "ok": True,
            "sensor": name,
            "mode": "RGB" if image.ndim == 3 else "L",
            "width": int(width),
            "height": int(height),
            "mime_type": "image/png",
            "path": file_path,
            "bytes": len(png_bytes),
        }
        return item

    def _write_agent_downsampled_image(
        self,
        item: Dict[str, Any],
        image: np.ndarray,
        max_size: int,
    ) -> Dict[str, Any]:
        if max_size <= 0 or image.ndim != 3:
            return item
        agent_image = self._downsample_image_nearest(image, max_size=max_size)
        if agent_image.shape[:2] == image.shape[:2]:
            return item
        original_path = item.get("path")
        if not isinstance(original_path, str) or not original_path:
            return item
        stem, ext = os.path.splitext(original_path)
        agent_path = f"{stem}_agent{ext or '.png'}"
        png_bytes = self._encode_png_bytes(agent_image)
        with open(agent_path, "wb") as file_handle:
            file_handle.write(png_bytes)
        height, width = agent_image.shape[:2]
        downsampled = dict(item)
        downsampled["original_path"] = original_path
        downsampled["original_width"] = item.get("width")
        downsampled["original_height"] = item.get("height")
        downsampled["path"] = agent_path
        downsampled["width"] = int(width)
        downsampled["height"] = int(height)
        downsampled["bytes"] = len(png_bytes)
        downsampled["downsampled_for_agent"] = True
        downsampled["agent_image_max_size"] = int(max_size)
        return downsampled

    @staticmethod
    def _agent_panorama_label_font(size: int = 10):
        from PIL import ImageFont

        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def _annotate_agent_panorama_view_label(
        self,
        image_item: MutableMapping[str, Any],
        *,
        label: str,
    ) -> None:
        if not label or not image_item.get("downsampled_for_agent"):
            return
        path = image_item.get("path")
        original_path = image_item.get("original_path")
        if not isinstance(path, str) or not path or path == original_path:
            return
        try:
            from PIL import Image, ImageDraw

            with Image.open(path) as src:
                image = src.convert("RGB")
            draw = ImageDraw.Draw(image)
            font = self._agent_panorama_label_font(10)
            bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            text_w = bbox[2] - bbox[0]
            x = max(2, image.width - text_w - 4)
            y = 3
            draw.text(
                (x, y),
                label,
                font=font,
                fill=(20, 92, 210),
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )
            image.save(path)
            image_item["bytes"] = os.path.getsize(path)
        except Exception:
            return

    def _find_visual_frame_records(
        self,
        output_dir: str,
        session_id: str,
        sensor: str,
        step_start: Optional[int],
        step_end: Optional[int],
    ) -> list[tuple[int, str]]:
        """Find frame files for video export.

        The sequence number in filenames is capture_seq (monotonic per
        session), which may diverge from simulator step_count when look
        is called multiple times at the same step.  step_start/step_end
        filter on capture_seq, not simulator step_count.

        Searches in the canonical session subdirectory (output_dir/session_id/).
        """
        search_dir = os.path.join(output_dir, session_id)
        if not os.path.isdir(search_dir):
            raise HabitatAdapterError(
                f'Visual output directory does not exist: "{search_dir}"'
            )

        safe_sensor = self._safe_sensor_name(sensor)
        # Match both regular frames and panorama frames:
        #   regular:  step{N}_{sensor}.png
        #   panorama: pano_{dir}_step{N}_{sensor}.png
        # Note: {N} is capture_seq, not simulator step_count.
        pattern = re.compile(
            rf"^(?:pano_(?:front|right|back|left)_)?step(\d{{6}})_{re.escape(safe_sensor)}\.png$"
        )
        frame_records: list[tuple[int, str]] = []
        for file_name in os.listdir(search_dir):
            match = pattern.match(file_name)
            if match is None:
                continue
            seq = int(match.group(1))
            if step_start is not None and seq < step_start:
                continue
            if step_end is not None and seq > step_end:
                continue
            frame_records.append((seq, os.path.join(search_dir, file_name)))
        frame_records.sort(key=lambda item: item[0])
        return frame_records

    def _encode_video_trace(
        self, frame_paths: list[str], video_path: str, fps: float
    ) -> None:
        if not frame_paths:
            raise HabitatAdapterError("Cannot encode a video trace with no frames")

        output_dir = os.path.dirname(video_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = ""
        frame_duration_s = 1.0 / fps
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".ffconcat",
                prefix="video_trace_",
                dir=output_dir,
                delete=False,
            ) as manifest:
                manifest.write("ffconcat version 1.0\n")
                for frame_path in frame_paths[:-1]:
                    manifest.write(f"file {self._quote_ffconcat_path(frame_path)}\n")
                    manifest.write(f"duration {frame_duration_s:.6f}\n")
                last_frame_path = frame_paths[-1]
                manifest.write(f"file {self._quote_ffconcat_path(last_frame_path)}\n")
                manifest.write(f"duration {frame_duration_s:.6f}\n")
                manifest.write(f"file {self._quote_ffconcat_path(last_frame_path)}\n")
                manifest_path = manifest.name

            last_error_output = ""
            for codec in ("libx264", "mpeg4"):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            manifest_path,
                            "-an",
                            "-c:v",
                            codec,
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            video_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    last_error_output = ""
                    break
                except subprocess.CalledProcessError as exc:
                    last_error_output = (exc.stderr or exc.stdout or "").strip()
                    if os.path.exists(video_path):
                        os.unlink(video_path)
            else:
                raise HabitatAdapterError(
                    "Failed to encode video trace with ffmpeg"
                    + (f": {last_error_output}" if last_error_output else "")
                )
        except FileNotFoundError as exc:
            raise HabitatAdapterError(
                "ffmpeg is required to export video traces but was not found"
            ) from exc
        finally:
            if manifest_path and os.path.exists(manifest_path):
                os.unlink(manifest_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            raise HabitatAdapterError(
                f'Video trace export did not produce a valid file: "{video_path}"'
            )

    @staticmethod
    def _quote_ffconcat_path(path: str) -> str:
        escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
        return f"'{escaped}'"

    def _export_visuals(
        self,
        observation: Optional[Mapping[str, Any]],
        output_dir: str,
        sensors: Optional[list[str]],
        depth_max: float,
        session_id: str,
        step_count: int,
        capture_seq: Optional[int] = None,
        filename_prefix: str = "",
        agent_image_max_size: int = _DEFAULT_AGENT_IMAGE_MAX_SIZE,
    ) -> Dict[str, Dict[str, Any]]:
        if observation is None:
            raise HabitatAdapterError("No observation available for visualization")

        if sensors is None:
            sensor_names: list[str] = []
            for key, value in observation.items():
                try:
                    array_value = self._to_numpy_array(value)
                except HabitatAdapterError:
                    continue
                if array_value.ndim in (2, 3):
                    sensor_names.append(key)
        else:
            sensor_names = sensors

        visuals: Dict[str, Dict[str, Any]] = {}
        for sensor_name in sensor_names:
            if sensor_name not in observation:
                visuals[sensor_name] = {
                    "ok": False,
                    "error": f'Observation key "{sensor_name}" not found',
                }
                continue

            try:
                raw_value = observation[sensor_name]
                np_array = self._to_numpy_array(raw_value)

                if np_array.ndim == 3 and np_array.shape[2] in (3, 4):
                    image = self._to_uint8_color(np_array)
                    mode = "RGB" if image.shape[2] == 3 else "RGBA"
                    visual_metadata = self._visual_robot_metadata_for_rgb(
                        image=image,
                        observation=observation,
                        session_id=session_id,
                        sensor_name=sensor_name,
                    )
                elif np_array.ndim == 2:
                    image = self._to_uint8_depth(np_array, depth_max=depth_max)
                    mode = "L"
                    visual_metadata = {}
                else:
                    raise HabitatAdapterError(
                        f'Unsupported shape for sensor "{sensor_name}": '
                        f"{list(np_array.shape)}"
                    )
                item = self._write_png_image(
                    image=image,
                    output_dir=output_dir,
                    name=sensor_name,
                    session_id=session_id,
                    step_count=step_count,
                    capture_seq=capture_seq,
                    filename_prefix=filename_prefix,
                )
                if mode in ("RGB", "RGBA"):
                    item = self._write_agent_downsampled_image(
                        item, image, max_size=agent_image_max_size
                    )
                item["mode"] = mode
                item.update(visual_metadata)
                visuals[sensor_name] = item
            except Exception as exc:  # noqa: BLE001
                visuals[sensor_name] = {
                    "ok": False,
                    "error": str(exc),
                }

        return visuals
