"""Image encoding helpers for LLM function-calling image inputs.

Kept in `runtime/` because agents and tools across the whole package need
them, and they have no dependencies beyond `base64` / `os`.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


SUPPORTED_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@dataclass(frozen=True)
class ImageReadResult:
    """Structured result for a single image read."""

    ok: bool
    content_part: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ImageBudgetExceeded(RuntimeError):
    """Raised before an LLM call would exceed the provider image cap."""


def read_image_as_data_url(path: str) -> Optional[str]:
    """Read image file and return as base64 data URL.

    Returns None on filesystem errors (missing files or permissions).
    Internal encoding errors propagate.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = SUPPORTED_IMAGE_MIME_BY_EXT.get(ext, "image/png")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except OSError:
        return None


def read_image_b64(path: str) -> Optional[str]:
    """Read image file and return raw base64 string (no data URL prefix).

    Returns None on filesystem errors.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def is_depth_image_path(path: str) -> bool:
    return "depth" in str(path).lower()


def select_llm_image_paths(
    image_paths: List[str],
    *,
    preserve_depth_paths: Optional[Set[str]] = None,
) -> List[str]:
    """Return image paths that should be exposed to the LLM.

    Depth images are filtered when color images are present, preserving
    the image selection priority. If all paths look depth-only, fall back
    to the raw list so depth-only observations remain debuggable.
    `preserve_depth_paths` keeps explicit read_image(path) requests from
    being filtered out when they are batched with automatic color captures.
    """

    if not image_paths:
        return []
    preserve_depth_paths = preserve_depth_paths or set()
    color_paths = [
        p
        for p in image_paths
        if not is_depth_image_path(p) or p in preserve_depth_paths
    ]
    return color_paths if color_paths else list(image_paths)


def read_image(path: str) -> ImageReadResult:
    """Read a single image path into an OpenAI-compatible content part."""

    if not path:
        return ImageReadResult(ok=False, error="empty_path")
    if not os.path.isfile(path):
        return ImageReadResult(ok=False, error="path_not_found")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_MIME_BY_EXT:
        return ImageReadResult(ok=False, error="unsupported_image_type")
    data_url = read_image_as_data_url(path)
    if not data_url:
        return ImageReadResult(ok=False, error="read_failed")
    return ImageReadResult(
        ok=True,
        content_part={"type": "image_url", "image_url": {"url": data_url}},
    )


def build_image_content_parts(
    image_paths: List[str],
    max_images: int = 4,
) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible image content parts from file paths.

    Caps at `max_images` to keep LLM input bounded. Paths that fail to
    load are silently skipped.

    For a complete user message with prompt text and panorama direction
    labels, use `build_visual_injection_message`.
    """
    parts: List[Dict[str, Any]] = []
    for path in image_paths[:max_images]:
        result = read_image(path)
        if result.ok and result.content_part:
            parts.append(result.content_part)
    return parts


_PANO_DIRECTION_RE = None  # lazy-compiled to avoid module-import-time regex cost


def build_visual_injection_payload(
    prompt_text: str,
    image_paths: List[str],
    *,
    preserve_depth_paths: Optional[Set[str]] = None,
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    """Build a one-shot image message and report paths actually loaded."""

    if not image_paths:
        return None, []

    color_paths = select_llm_image_paths(
        image_paths,
        preserve_depth_paths=preserve_depth_paths,
    )
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    global _PANO_DIRECTION_RE
    if _PANO_DIRECTION_RE is None:
        import re

        _PANO_DIRECTION_RE = re.compile(r"_pano_(front|right|back|left)_")

    loaded_paths: List[str] = []
    for path in color_paths:
        result = read_image(path)
        if not (result.ok and result.content_part):
            continue
        match = _PANO_DIRECTION_RE.search(path)
        if match:
            parts.append({"type": "text", "text": f"[{match.group(1).upper()} view]"})
        parts.append(result.content_part)
        loaded_paths.append(path)

    if not loaded_paths:
        return None, []
    return {"role": "user", "content": parts}, loaded_paths


def build_visual_injection_message(
    prompt_text: str,
    image_paths: List[str],
) -> Optional[Dict[str, Any]]:
    """Build a one-shot user message with text and image_url parts.

    This helper is for the transient "current observation / explicit
    read_image result" path only. It must not be used to replay a
    pending images from preceding turns across rounds.
    """

    message, _loaded_paths = build_visual_injection_payload(prompt_text, image_paths)
    return message


def count_image_url_parts(messages: List[Dict[str, Any]]) -> int:
    """Count OpenAI-style image_url content parts in a message list."""

    count = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                count += 1
    return count


__all__ = [
    "ImageBudgetExceeded",
    "ImageReadResult",
    "SUPPORTED_IMAGE_MIME_BY_EXT",
    "build_image_content_parts",
    "build_visual_injection_payload",
    "build_visual_injection_message",
    "count_image_url_parts",
    "is_depth_image_path",
    "read_image",
    "read_image_as_data_url",
    "read_image_b64",
    "select_llm_image_paths",
]
