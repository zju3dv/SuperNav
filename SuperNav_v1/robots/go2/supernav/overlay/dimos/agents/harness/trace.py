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

"""Small trace-event layer for DimOS harness decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


@dataclass(frozen=True)
class HarnessTraceEvent:
    """Structured event that can be emitted into the existing DimOS run log."""

    event_type: str
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def emit_trace_event(logger: Any, event_type: str, **payload: Any) -> HarnessTraceEvent:
    """Emit a structured harness event through the normal JSON logger."""

    event = HarnessTraceEvent(event_type=event_type, ts=time.time(), payload=payload)
    logger.info("harness_trace_event", harness_event=event.model_dump())
    return event
