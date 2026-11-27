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

"""Helpers for reading harness trace events from DimOS JSONL logs."""

from __future__ import annotations

import json
from typing import Any

from dimos.agents.harness.display_content import redact_display_content


def extract_harness_trace_event(raw: str) -> dict[str, Any] | None:
    """Parse one JSONL line into a harness event, if it contains one."""

    try:
        rec: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if rec.get("event") != "harness_trace_event":
        return None
    event = rec.get("harness_event")
    if not isinstance(event, dict):
        return None
    out = dict(event)
    if "timestamp" in rec:
        out["log_timestamp"] = rec["timestamp"]
    return out


def harness_trace_events_from_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Extract every harness trace event from JSONL lines."""

    events: list[dict[str, Any]] = []
    for raw in lines:
        event = extract_harness_trace_event(raw)
        if event is not None:
            events.append(event)
    return events


def format_harness_trace_event(event: dict[str, Any]) -> str:
    """Render one harness trace event for CLI display."""

    ts = str(event.get("log_timestamp") or "")
    hms = ts[11:19] if len(ts) >= 19 else ts
    event_type = str(event.get("event_type") or "?")
    payload = event.get("payload") or {}
    payload_text = json.dumps(
        redact_display_content(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{hms} {event_type} {payload_text}"
