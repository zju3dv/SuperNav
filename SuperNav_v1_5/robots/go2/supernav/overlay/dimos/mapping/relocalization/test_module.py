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

import threading
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from reactivex import Subject

from dimos.mapping.relocalization.module import MIN_LOCAL_POINTS, RelocalizationModule


class _SmallMap:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count

    def __len__(self) -> int:
        return self.point_count


def _module_with_attempt_capture() -> tuple[RelocalizationModule, list[Any]]:
    module = object.__new__(RelocalizationModule)
    module._latest_global_map = None
    module._pending_forced_relocalization = False
    module._state_lock = threading.Lock()
    module._forced_relocalization = Subject()
    attempts: list[Any] = []
    module._forced_relocalization.subscribe(attempts.append)
    return module, attempts


def test_reported_prior_relocalizes_latest_map_below_normal_point_threshold() -> None:
    module, attempts = _module_with_attempt_capture()
    local_map = _SmallMap(MIN_LOCAL_POINTS - 1)
    assert module._has_enough_points(local_map) is False  # type: ignore[arg-type]
    module._remember_global_map(local_map)  # type: ignore[arg-type]

    module._on_relocalization_request(Bool(data=True))

    assert attempts == [local_map]


def test_reported_prior_waits_for_first_map_and_still_bypasses_threshold() -> None:
    module, attempts = _module_with_attempt_capture()

    module._on_relocalization_request(Bool(data=True))

    assert attempts == []
    assert module._pending_forced_relocalization is True

    local_map = _SmallMap(1)
    module._remember_global_map(local_map)  # type: ignore[arg-type]

    assert attempts == [local_map]
    assert module._pending_forced_relocalization is False
