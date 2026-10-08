"""Grounding DINO client and raw detection parsing."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from PIL import Image

from supernav.methods.navigation.oracle_local_nav.config import (
    _DEFAULT_CROP_EXPAND_RATIO,
    _DEFAULT_GROUNDING_SCORE_THRESHOLD,
    _DEFAULT_GROUNDING_TIMEOUT_S,
    _DEFAULT_GROUNDING_TOP_K,
    _DEFAULT_GROUNDING_URL,
    _DEFAULT_MAX_GROUNDING_CROPS,
)
from supernav.methods.navigation.oracle_local_nav.utils import _bbox_area, _debug_bbox, _expand_bbox, _json_safe_value, _point_in_bbox, _projected_row_xyxy, _scale_bbox

def _ground_detections_for_panorama(
    *,
    images: list[Mapping[str, Any]],
    phrases: list[str],
    grounding_url: str,
    score_threshold: float,
    timeout_s: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for image in images:
        image_source = _grounding_image_source(image)
        direction = str(image.get("direction") or "").lower()
        if image_source is None or not direction:
            continue
        for phrase in phrases:
            try:
                response = _call_grounding_service_proxy(
                    grounding_url=grounding_url,
                    image_path=image_source["path"],
                    phrase=phrase,
                    timeout_s=timeout_s,
                )
            except Exception:
                continue
            detections.extend(
                _response_detections(
                    response=response,
                    direction=direction,
                    query_phrase=phrase,
                    score_threshold=score_threshold,
                    source_width=image_source["source_width"],
                    source_height=image_source["source_height"],
                    target_width=image_source["target_width"],
                    target_height=image_source["target_height"],
                    offset_xy=(0.0, 0.0),
                    scale_to_target=True,
                    grounding_source="full_image",
                )
            )
    detections.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return detections

def _ground_detections_for_projection_crops(
    *,
    images: list[Mapping[str, Any]],
    projected_rows: list[Mapping[str, Any]],
    phrases: list[str],
    grounding_url: str,
    score_threshold: float,
    timeout_s: float,
    output_dir: str,
    crop_expand_ratio: float,
    max_crops: int,
) -> list[dict[str, Any]]:
    if max_crops <= 0:
        return []
    images_by_direction = {
        str(image.get("direction") or "").lower(): image
        for image in images
        if isinstance(image, Mapping)
    }
    crop_rows = sorted(
        [
            row
            for row in projected_rows
            if isinstance(row, Mapping) and _projected_row_xyxy(row) is not None
        ],
        key=lambda row: (
            float(row.get("visible_fraction") or 0.0),
            _bbox_area(_projected_row_xyxy(row) or (0.0, 0.0, 0.0, 0.0)),
        ),
        reverse=True,
    )[:max_crops]
    if not crop_rows:
        return []

    crop_dir = Path(output_dir) / "grounding_crops"
    try:
        crop_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    detections: list[dict[str, Any]] = []
    opened: dict[str, Image.Image] = {}
    try:
        for row in crop_rows:
            direction = str(row.get("direction") or "").lower()
            image = images_by_direction.get(direction)
            if image is None:
                continue
            image_source = _grounding_image_source(image)
            proj_box = _projected_row_xyxy(row)
            if image_source is None or proj_box is None:
                continue
            crop_box_target = _expand_bbox(
                proj_box,
                ratio=crop_expand_ratio,
                bounds=(
                    0.0,
                    0.0,
                    image_source["target_width"],
                    image_source["target_height"],
                ),
            )
            crop_box_source = _scale_bbox(
                crop_box_target,
                scale_x=image_source["source_width"] / image_source["target_width"],
                scale_y=image_source["source_height"] / image_source["target_height"],
            )
            sx0, sy0, sx1, sy1 = crop_box_source
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            source_path = image_source["path"]
            source_image = opened.get(source_path)
            if source_image is None:
                try:
                    source_image = Image.open(source_path).convert("RGB")
                except Exception:
                    continue
                opened[source_path] = source_image
            crop = source_image.crop((
                int(math.floor(sx0)),
                int(math.floor(sy0)),
                int(math.ceil(sx1)),
                int(math.ceil(sy1)),
            ))
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                prefix="ground_",
                dir=str(crop_dir),
                delete=False,
            ) as handle:
                crop_path = handle.name
            crop.save(crop_path)
            crop_width, crop_height = crop.size
            target_offset = (crop_box_target[0], crop_box_target[1])
            target_scale_x = (crop_box_target[2] - crop_box_target[0]) / max(
                1.0, float(crop_width)
            )
            target_scale_y = (crop_box_target[3] - crop_box_target[1]) / max(
                1.0, float(crop_height)
            )
            for phrase in phrases:
                try:
                    response = _call_grounding_service_proxy(
                        grounding_url=grounding_url,
                        image_path=crop_path,
                        phrase=phrase,
                        timeout_s=timeout_s,
                    )
                except Exception:
                    continue
                detections.extend(
                    _response_detections(
                        response=response,
                        direction=direction,
                        query_phrase=phrase,
                        score_threshold=score_threshold,
                        source_width=float(crop_width),
                        source_height=float(crop_height),
                        target_width=float(crop_width) * target_scale_x,
                        target_height=float(crop_height) * target_scale_y,
                        offset_xy=target_offset,
                        scale_to_target=True,
                        grounding_source="scene_graph_crop",
                        crop_target_ref=str(row.get("target_ref") or ""),
                    )
                )
    finally:
        for image in opened.values():
            image.close()
    detections.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return detections

def _response_detections(
    *,
    response: Mapping[str, Any],
    direction: str,
    query_phrase: str,
    score_threshold: float,
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
    offset_xy: tuple[float, float],
    scale_to_target: bool,
    grounding_source: str,
    crop_target_ref: str | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    scale_x = target_width / source_width if source_width > 0 else 1.0
    scale_y = target_height / source_height if source_height > 0 else 1.0
    for raw in response.get("detections") or []:
        if not isinstance(raw, Mapping):
            continue
        score = _detection_score(raw)
        box = _detection_xyxy(raw)
        if score is None or score < score_threshold or box is None:
            continue
        if scale_to_target:
            box = _scale_bbox(box, scale_x=scale_x, scale_y=scale_y)
        box = (
            box[0] + offset_xy[0],
            box[1] + offset_xy[1],
            box[2] + offset_xy[0],
            box[3] + offset_xy[1],
        )
        detections.append(
            {
                "direction": direction,
                "score": score,
                "box_xyxy": box,
                "image_width": target_width,
                "image_height": target_height,
                "label": str(raw.get("label") or ""),
                "query_phrase": query_phrase,
                "grounding_source": grounding_source,
                "crop_target_ref": crop_target_ref,
            }
        )
    return detections

def _grounding_image_source(image: Mapping[str, Any]) -> dict[str, Any] | None:
    original_path = str(image.get("original_path") or "")
    agent_path = str(image.get("path") or "")
    use_original = bool(original_path) and (
        os.path.exists(original_path) or not agent_path
    )
    path = original_path if use_original else agent_path
    if not path:
        return None
    try:
        target_width = float(image.get("width") or image.get("original_width") or 0)
        target_height = float(image.get("height") or image.get("original_height") or 0)
        source_width_value = (
            image.get("original_width") if use_original else image.get("width")
        )
        source_height_value = (
            image.get("original_height") if use_original else image.get("height")
        )
        source_width = float(source_width_value or 0)
        source_height = float(source_height_value or 0)
    except (TypeError, ValueError):
        return None
    if (
        target_width <= 0
        or target_height <= 0
        or source_width <= 0
        or source_height <= 0
    ):
        target_width = target_height = source_width = source_height = 1.0
    return {
        "path": path,
        "target_width": target_width,
        "target_height": target_height,
        "source_width": source_width,
        "source_height": source_height,
    }

def _call_grounding_service(
    *,
    grounding_url: str,
    image_path: str,
    phrase: str,
    timeout_s: float,
) -> dict[str, Any]:
    url = _normalize_grounding_url(grounding_url)
    request_body = json.dumps({"image_path": image_path, "phrases": phrase}).encode()
    req = urllib_request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("ok") is False:
        raise RuntimeError(str(parsed.get("error") if isinstance(parsed, dict) else parsed))
    return parsed


def _call_grounding_service_proxy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the package-level ``_call_grounding_service``.

    This indirection lets tests monkeypatch ``oln._call_grounding_service`` and
    have the fake used by the grounding client helpers.
    """
    from supernav.methods.navigation.oracle_local_nav import _call_grounding_service

    return _call_grounding_service(*args, **kwargs)


def _normalize_grounding_url(value: str) -> str:
    url = str(value or "").strip() or _DEFAULT_GROUNDING_URL
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/ground"):
        url = f"{url}/ground"
    return url

def _detection_score(raw: Mapping[str, Any]) -> float | None:
    for key in ("score", "confidence", "detection_confidence"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return None

def _detection_xyxy(raw: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    box = raw.get("box") or raw.get("bbox") or raw.get("box_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    if hi_x <= lo_x or hi_y <= lo_y:
        return None
    return lo_x, lo_y, hi_x, hi_y

def _json_safe_detection(detection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "direction": detection.get("direction"),
        "score": detection.get("score"),
        "bbox_px": _debug_bbox(detection.get("box_xyxy")),
        "label": detection.get("label"),
        "query_phrase": detection.get("query_phrase"),
        "grounding_source": detection.get("grounding_source"),
    }

