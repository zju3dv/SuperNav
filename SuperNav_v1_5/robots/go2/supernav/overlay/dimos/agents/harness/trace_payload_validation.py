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

"""Trace-audit checks for raw rich-media payload leakage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)

_IMAGE_DATA_URL_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")
_LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\s]{256,}")


def raw_trace_payload_absence_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate that harness traces do not carry raw base64/rich-media payloads."""

    leaks = []
    for event_index, event in enumerate(events):
        payload = event.get("payload")
        leaks.extend(
            {
                "event_index": event_index,
                "event_type": event.get("event_type"),
                **leak,
            }
            for leak in _raw_payload_leaks(payload)
        )

    if not leaks:
        return _pass(
            "raw_trace_payload_absence",
            "Harness trace payloads do not contain raw image data URLs or long base64 strings.",
            events_checked=len(events),
        )
    return _fail(
        "raw_trace_payload_absence",
        "Harness trace payloads contain raw rich-media data that should be summarized or redacted.",
        leaks=leaks[:20],
        leak_count=len(leaks),
    )


def _raw_payload_leaks(value: Any, *, path: str = "payload") -> Iterable[dict[str, Any]]:
    if isinstance(value, str):
        leak_kind = _raw_payload_kind(value)
        if leak_kind is not None:
            yield {"path": path, "kind": leak_kind, "preview": _preview(value)}
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _raw_payload_leaks(item, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _raw_payload_leaks(item, path=f"{path}[{index}]")


def _raw_payload_kind(text: str) -> str | None:
    if _IMAGE_DATA_URL_RE.search(text):
        return "image_data_url"
    stripped = "".join(text.split())
    if len(stripped) >= 256 and _LONG_BASE64_RE.fullmatch(stripped):
        return "long_base64_string"
    return None


def _preview(text: str, max_chars: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."
