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

"""Helpers for attaching measured durations to structured skill results."""

from __future__ import annotations

import time
from typing import TypeVar

from dimos.agents.skill_result import SkillResult

E = TypeVar("E", bound=str)


def record_skill_duration(
    result: SkillResult[E],
    started_at_s: float,
    *,
    finished_at_s: float | None = None,
) -> SkillResult[E]:
    """Set ``result.duration_ms`` from monotonic timestamps and return it."""

    finished_at_s = time.monotonic() if finished_at_s is None else finished_at_s
    result.duration_ms = max(0.0, (finished_at_s - started_at_s) * 1000.0)
    return result
