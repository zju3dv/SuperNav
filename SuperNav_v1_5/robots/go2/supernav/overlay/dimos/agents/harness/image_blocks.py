# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for model-visible image content blocks.

The transport form for provider vision input is often a base64 data URL, but
multimodal providers price and compact images from pixel dimensions rather than
from the length of the base64 string. Keep that logic out of the general context
manager so the request assembly code stays readable.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from io import BytesIO
import re
from typing import Any

from PIL import Image as PILImage

from dimos.agents.harness.image_tokens import (
    ImageTokenEstimate,
    estimate_image_tokens,
)

_ARTIFACT_ID_RE = re.compile(r"\[artifact:(?P<artifact_id>[a-zA-Z0-9_-]+)\b")


def is_image_block(block: dict[str, Any]) -> bool:
    """Return whether *block* is a provider image content block."""

    return str(block.get("type") or "") in {"image", "image_url", "input_image"}


def image_block_mime_type(block: dict[str, Any]) -> str | None:
    """Return the MIME type declared by an image content block, if present."""

    block_type = str(block.get("type") or "")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            return _data_url_mime_type(str(image_url.get("url") or ""))
        if isinstance(image_url, str):
            return _data_url_mime_type(image_url)
    if block_type == "input_image":
        return _data_url_mime_type(str(block.get("image_url") or ""))

    source = block.get("source")
    if isinstance(source, dict):
        media_type = source.get("media_type") or source.get("mediaType") or source.get("mime_type")
        if media_type:
            return str(media_type)
    mime = block.get("mime_type") or block.get("mimeType") or block.get("mime")
    return str(mime) if mime else None


def image_block_data_url(block: dict[str, Any]) -> str | None:
    """Return a data URL representation when the image block carries one."""

    return _data_url_from_block(block)


def image_block_detail(block: dict[str, Any]) -> str | None:
    """Return provider detail level from an image block, if present."""

    block_type = str(block.get("type") or "")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            detail = image_url.get("detail") or block.get("detail")
            return str(detail) if detail else None
    detail = block.get("detail")
    return str(detail) if detail else None


def image_block_dimensions(block: dict[str, Any]) -> tuple[int, int] | None:
    """Best-effort image dimensions as ``(width, height)``."""

    explicit = _explicit_dimensions(block)
    if explicit is not None:
        return explicit

    data = _base64_payload_bytes(block)
    if data is None:
        return None
    try:
        with PILImage.open(BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def image_block_payload_bytes(block: dict[str, Any]) -> bytes | None:
    """Return decoded image payload bytes when the block carries base64 data."""

    return _base64_payload_bytes(block)


def image_block_token_estimate(
    block: dict[str, Any],
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
) -> int:
    """Estimate image tokens from resolution, never from base64 string length."""

    return image_block_token_estimate_details(
        block,
        target_provider=target_provider,
        target_model=target_model,
    ).tokens


def image_block_token_estimate_details(
    block: dict[str, Any],
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
) -> ImageTokenEstimate:
    """Return traceable image token estimate details for an image block."""

    dimensions = image_block_dimensions(block)
    width, height = dimensions if dimensions is not None else (None, None)
    return estimate_image_tokens(
        width,
        height,
        target_provider=target_provider,
        target_model=target_model,
        detail=image_block_detail(block),
    )


def image_token_estimate(
    width: int,
    height: int,
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
    detail: str | None = None,
) -> int:
    """Approximate provider/model image token cost from pixel area.

    Existing callers without provider/model arguments keep the previous neutral
    area estimate. Context assembly passes the configured provider/model for a
    more accurate budget.
    """

    return estimate_image_tokens(
        width,
        height,
        target_provider=target_provider,
        target_model=target_model,
        detail=detail,
    ).tokens


def artifact_ids_from_content(content: Any) -> list[str]:
    """Extract artifact ids from placeholder text in message content."""

    ids: list[str] = []
    for text in _text_fragments(content):
        ids.extend(match.group("artifact_id") for match in _ARTIFACT_ID_RE.finditer(text))
    return ids


def _text_fragments(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        yield content
        return
    if isinstance(content, list):
        for item in content:
            yield from _text_fragments(item)
        return
    if isinstance(content, dict):
        if content.get("type") == "text":
            yield str(content.get("text") or "")
        return


def _explicit_dimensions(block: dict[str, Any]) -> tuple[int, int] | None:
    width = block.get("width") or block.get("width_px")
    height = block.get("height") or block.get("height_px")
    if width is None or height is None:
        metadata = block.get("metadata")
        if isinstance(metadata, dict):
            width = width or metadata.get("width_px") or metadata.get("width")
            height = height or metadata.get("height_px") or metadata.get("height")
    try:
        if width is None or height is None:
            return None
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def _base64_payload_bytes(block: dict[str, Any]) -> bytes | None:
    data_url = _data_url_from_block(block)
    if data_url is None:
        return None
    if ";base64," not in data_url:
        return None
    payload = data_url.split(";base64,", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return None


def _data_url_from_block(block: dict[str, Any]) -> str | None:
    block_type = str(block.get("type") or "")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            return str(image_url.get("url") or "")
        if isinstance(image_url, str):
            return image_url
    if block_type == "input_image":
        return str(block.get("image_url") or "")

    source = block.get("source")
    if isinstance(source, dict):
        data = source.get("data")
        if data and str(source.get("type") or "") == "base64":
            media_type = (
                source.get("media_type")
                or source.get("mediaType")
                or source.get("mime_type")
                or "image/jpeg"
            )
            return f"data:{media_type};base64,{data}"
    data = block.get("data")
    if data and block_type == "image":
        media_type = image_block_mime_type(block) or "image/jpeg"
        return f"data:{media_type};base64,{data}"
    return None


def _data_url_mime_type(data_url: str) -> str | None:
    if data_url.startswith("data:") and ";base64," in data_url:
        return data_url.removeprefix("data:").split(";base64,", 1)[0]
    return None
