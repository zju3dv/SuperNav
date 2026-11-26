"""Minimal HTTP client for the LocateAnything grounding service.

The service is expected to run in a separate Python environment because
LocateAnything pins transformers versions that conflict with the Habitat
runtime.  The client speaks a small JSON protocol similar to the existing
Grounding DINO service, but supports both box and point modes.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

_DEFAULT_LOCATE_ANYTHING_URL = "http://127.0.0.1:8915/ground"
_DEFAULT_LOCATE_ANYTHING_TIMEOUT_S = 30.0
_DEFAULT_LOCATE_ANYTHING_MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _locate_anything_url() -> str:
    url = str(os.environ.get("NAV_LOCATE_ANYTHING_URL") or "").strip()
    if not url:
        url = _DEFAULT_LOCATE_ANYTHING_URL
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/ground"):
        url = f"{url}/ground"
    return url


def _call_locate_anything_service(
    *,
    image_path: str,
    phrase: str,
    mode: str = "box",
    url: str | None = None,
    timeout_s: float = _DEFAULT_LOCATE_ANYTHING_TIMEOUT_S,
) -> dict[str, Any]:
    """Call the LocateAnything service and return the raw response dict."""
    service_url = url or _locate_anything_url()
    request_body = _build_locate_anything_request_body(
        image_path=image_path,
        phrase=phrase,
        mode=mode,
    )
    req = urllib_request.Request(
        service_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(f"LocateAnything service error: {exc}") from exc
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"LocateAnything service returned non-dict: {parsed!r}")
    if parsed.get("ok") is False:
        raise RuntimeError(
            f"LocateAnything service returned error: {parsed.get('error')}"
        )
    return parsed


def _build_locate_anything_request_body(
    *,
    image_path: str,
    phrase: str,
    mode: str,
) -> bytes:
    """Build a path-independent JSON request for a local image."""
    path = Path(image_path)
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read LocateAnything image {image_path}: {exc}"
        ) from exc

    mime_type = mimetypes.guess_type(path.name)[0] or ""
    request_body = json.dumps(
        {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "image_mime_type": mime_type,
            "phrase": phrase,
            "mode": mode,
        }
    ).encode("utf-8")
    max_request_bytes = _positive_int_env(
        "NAV_LOCATE_ANYTHING_MAX_REQUEST_BYTES",
        _DEFAULT_LOCATE_ANYTHING_MAX_REQUEST_BYTES,
    )
    if len(request_body) > max_request_bytes:
        raise RuntimeError(
            "LocateAnything request body exceeds "
            f"{max_request_bytes} bytes: {len(request_body)}"
        )
    return request_body


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _locate_anything_detections(
    *,
    image_path: str,
    phrase: str,
    mode: str = "box",
    url: str | None = None,
    timeout_s: float = _DEFAULT_LOCATE_ANYTHING_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Return normalized detections from the LocateAnything service.

    For mode="box" each detection has keys:
        - box_xyxy: tuple[float, float, float, float] in pixel coordinates
        - score: float
        - label: str
    For mode="point" each detection has keys:
        - point_xy: tuple[float, float] in pixel coordinates
        - score: float
        - label: str
    """
    response = _call_locate_anything_service(
        image_path=image_path,
        phrase=phrase,
        mode=mode,
        url=url,
        timeout_s=timeout_s,
    )
    raw_detections = response.get("detections") or []
    detections: list[dict[str, Any]] = []
    for raw in raw_detections:
        if not isinstance(raw, Mapping):
            continue
        score = raw.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        label = str(raw.get("label") or phrase)
        if mode == "point":
            point = raw.get("point_xy") or raw.get("point")
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    px, py = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                detections.append(
                    {
                        "point_xy": (px, py),
                        "score": score,
                        "label": label,
                    }
                )
        else:
            box = raw.get("box_xyxy") or raw.get("box") or raw.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    x0, y0, x1, y1 = [float(v) for v in box[:4]]
                except (TypeError, ValueError):
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                detections.append(
                    {
                        "box_xyxy": (x0, y0, x1, y1),
                        "score": score,
                        "label": label,
                    }
                )
    detections.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return detections
