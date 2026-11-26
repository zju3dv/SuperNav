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

"""Shared status and motion contracts for map-backed PixelGoal."""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from dimos.spec.utils import Spec
from .live_state import LiveStateWriter

GOAL_CACHE_SIZE = 16
_STATUS_EVENT_TAIL = 30

class UnitreeMotionSpec(Spec, Protocol):
    """Motion injection: odometry-closed primitives + hard stop."""

    def drive(self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0) -> Any: ...

    def rotate_in_place(
        self, degrees: float, max_duration_s: float = 5.0, angular_speed_limit: float = 0.4
    ) -> Any: ...

    def stop_movement(self) -> Any: ...


class NavLiveSink:
    """LiveStateWriter + in-memory status/events for the pixelgoal_status skill."""

    def __init__(self, writer: LiveStateWriter, event_tail: int = 200) -> None:
        self._writer = writer
        self._lock = threading.Lock()
        self.last_status: dict[str, Any] | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=event_tail)

    def write_frame(self, rgb: np.ndarray, *, goal: bool = False) -> None:
        self._writer.write_frame(rgb, goal=goal)

    def write_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self.last_status = dict(status)
        self._writer.write_status(status)

    def event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        entry = {"ts": time.time(), "type": event_type, "data": dict(data or {})}
        with self._lock:
            self.events.append(entry)
        self._writer.event(event_type, data)

    def snapshot(self, tail: int = _STATUS_EVENT_TAIL) -> dict[str, Any]:
        with self._lock:
            status = None if self.last_status is None else dict(self.last_status)
            events = list(self.events)[-tail:]
        return {"status": status, "events": events}


@dataclass(frozen=True)
class PixelGoalHopResult:
    """Structured outcome of one pixelgoal_navigate hop."""

    ok: bool
    status: str
    goal_id: str
    point: tuple[float, float]
    steps: int = 0
    duration_s: float = 0.0
    travelled_m: float = 0.0
    final_distance_m: float | None = None
    track_score: float | None = None
    needs_agent_reason: str | None = None
    error_code: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "goal_id": self.goal_id,
            "point": [round(self.point[0], 4), round(self.point[1], 4)],
            "steps": self.steps,
            "duration_s": round(self.duration_s, 3),
            "travelled_m": round(self.travelled_m, 3),
            "final_distance_m": (
                None if self.final_distance_m is None else round(self.final_distance_m, 3)
            ),
            "track_score": (None if self.track_score is None else round(self.track_score, 4)),
        }
        if self.needs_agent_reason:
            payload["needs_agent_reason"] = self.needs_agent_reason
        if self.error_code:
            payload["code"] = self.error_code
        if self.error:
            payload["error"] = self.error
        return payload
