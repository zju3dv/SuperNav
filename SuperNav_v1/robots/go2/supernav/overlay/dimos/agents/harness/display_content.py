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

"""Display-safe rendering for rich agent message content."""

from __future__ import annotations

import json
import re
from typing import Any

from dimos.agents.harness.image_blocks import (
    image_block_dimensions,
    image_block_mime_type,
    is_image_block,
)

_IMAGE_DATA_URL_RE = re.compile(r"data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,[A-Za-z0-9+/=]+")


def content_to_display_text(content: Any) -> str:
    """Render message content for humans without raw rich-media payloads."""

    sanitized = redact_display_content(content)
    if isinstance(sanitized, str):
        return sanitized
    if isinstance(sanitized, list):
        parts: list[str] = []
        for item in sanitized:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(_json_display(item))
        return "\n".join(part for part in parts if part)
    return _json_display(sanitized)


def redact_display_content(content: Any) -> Any:
    """Return a JSON-like copy safe for terminal/UI display."""

    if isinstance(content, str):
        if _looks_like_base64_payload(content):
            return "[base64 payload redacted_from_display]"
        return _redact_data_urls(content)
    if isinstance(content, list):
        return [redact_display_content(item) for item in content]
    if isinstance(content, dict):
        if is_image_block(content):
            return _image_placeholder(content)
        if content.get("type") == "text":
            return {**content, "text": redact_display_content(str(content.get("text") or ""))}
        return {str(key): _redact_value(value) for key, value in content.items()}
    return content


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_like_base64_payload(value):
            return "[base64 payload redacted_from_display]"
        return _redact_data_urls(value)
    return redact_display_content(value)


def _redact_data_urls(text: str) -> str:
    return _IMAGE_DATA_URL_RE.sub(
        lambda match: f"[image mime={match.group('mime')} redacted_from_display]",
        text,
    )


def _looks_like_base64_payload(value: str) -> bool:
    if len(value) < 256:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", value))


def _image_placeholder(block: dict[str, Any]) -> str:
    mime = image_block_mime_type(block)
    dimensions = image_block_dimensions(block)
    artifact_id = _image_artifact_id(block)
    parts = ["[image"]
    if artifact_id is not None:
        parts.append(f"artifact={artifact_id}")
    if mime is not None:
        parts.append(f"mime={mime}")
    if dimensions is not None:
        parts.append(f"size={dimensions[0]}x{dimensions[1]}")
    parts.append("redacted_from_display]")
    return " ".join(parts)


def _image_artifact_id(block: dict[str, Any]) -> str | None:
    artifact_id = block.get("artifact_id") or block.get("id")
    if artifact_id:
        return str(artifact_id)
    metadata = block.get("metadata")
    if isinstance(metadata, dict):
        artifact_id = metadata.get("artifact_id")
        if artifact_id:
            return str(artifact_id)
    return None


def _json_display(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
