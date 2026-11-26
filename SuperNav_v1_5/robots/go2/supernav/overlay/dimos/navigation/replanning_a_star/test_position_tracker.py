# Copyright 2025-2026 Dimensional Inc.
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

from unittest.mock import patch

import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.replanning_a_star.position_tracker import PositionTracker


@pytest.mark.parametrize("start", [100.0, 1789019724.931])
def test_stuck_window_excludes_prior_motion(start: float) -> None:
    tracker = PositionTracker(8.0, 0.4)
    for k in range(56 * 20):
        with patch("time.monotonic", return_value=start + k / 20):
            tracker.add_position(PoseStamped(position=[min(k / 200, 1.0), 0.0, 0.0]))
    with patch("time.monotonic", return_value=start + 56), patch("time.time", return_value=0.0):
        assert len(tracker._get_recent_positions()) == 160
        assert tracker.is_stuck()


def test_tracker_wraparound_and_expired_data() -> None:
    tracker = PositionTracker(1.0, 0.4)
    for k in range(1200):
        with patch("time.monotonic", return_value=k / 100):
            tracker.add_position(PoseStamped(position=[k / 100, 0.0, 0.0]))
    with patch("time.monotonic", return_value=12.0):
        assert len(tracker._get_recent_positions()) == 100
        assert not tracker.is_stuck()
    with patch("time.monotonic", return_value=14.0):
        assert not tracker.is_stuck()
