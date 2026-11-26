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

"""Trace-audit checks for native-agent session lifecycle events."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def session_lifecycle_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate session_start/session_end trace presence and linkage."""

    starts = _events_of_type(events, "session_start")
    ends = _events_of_type(events, "session_end")
    if not starts or not ends:
        return _fail(
            "session_lifecycle",
            "Missing session_start or session_end trace events.",
            session_start=len(starts),
            session_end=len(ends),
        )

    start_ids = _event_ids(starts)
    end_ids = _event_ids(ends)
    missing_end = sorted(start_ids - end_ids)
    missing_start = sorted(end_ids - start_ids)
    malformed = _malformed_session_events(starts, ends)
    if not missing_end and not missing_start and not malformed:
        return _pass(
            "session_lifecycle",
            "Native session has start/end traces with matching session ids.",
            sessions=sorted(start_ids),
            session_start=len(starts),
            session_end=len(ends),
        )

    return _fail(
        "session_lifecycle",
        "Native session lifecycle traces are incomplete or malformed.",
        missing_end=missing_end,
        missing_start=missing_start,
        malformed=malformed,
    )


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _event_ids(events: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        session_id = _payload(event).get("session_id")
        if isinstance(session_id, str):
            ids.add(session_id)
    return ids


def _malformed_session_events(
    starts: list[dict[str, Any]],
    ends: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for event_type, events, required in (
        (
            "session_start",
            starts,
            ("session_id", "runtime_policy", "model", "mcp_server_url", "context"),
        ),
        ("session_end", ends, ("session_id", "reason")),
    ):
        for index, event in enumerate(events):
            payload = _payload(event)
            missing = [field for field in required if field not in payload]
            if missing:
                malformed.append({"event_type": event_type, "index": index, "missing": missing})
                continue
            if event_type == "session_start" and not isinstance(payload.get("model"), dict):
                malformed.append(
                    {
                        "event_type": event_type,
                        "index": index,
                        "missing": ["model:dict"],
                    }
                )
            if event_type == "session_start" and not isinstance(payload.get("context"), dict):
                malformed.append(
                    {
                        "event_type": event_type,
                        "index": index,
                        "missing": ["context:dict"],
                    }
                )
    return malformed
