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

import math

import pytest

from dimos.core.global_config import GlobalConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner


def _pose(x: float, y: float, yaw_deg: float = 0.0) -> PoseStamped:
    return PoseStamped(
        frame_id="map",
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, math.radians(yaw_deg))),
    )


def test_low_progress_breaker_stops_replanning_after_one_retry() -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._current_goal = _pose(2.0, 0.0)
    planner._current_odom = _pose(0.05, 0.0)
    planner._goal_start_position = Vector3(0.0, 0.0, 0.0)
    planner._max_progress_from_goal_start_m = 0.05
    planner._replan_limiter.will_retry()

    planner._replan_path(reason="stuck")

    assert planner.get_last_failure() == "RETRY_LIMIT_EXCEEDED"
    assert planner._current_goal is None


def test_replanning_continues_when_robot_has_made_meaningful_progress(monkeypatch) -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._current_goal = _pose(2.0, 0.0)
    planner._current_odom = _pose(0.35, 0.0)
    planner._goal_start_position = Vector3(0.0, 0.0, 0.0)
    planner._max_progress_from_goal_start_m = 0.35
    planner._replan_limiter.will_retry()

    called = {"planned": 0}

    def _fake_plan_path() -> None:
        called["planned"] += 1

    monkeypatch.setattr(planner, "_plan_path", _fake_plan_path)

    planner._replan_path(reason="obstacle_found")

    assert called["planned"] == 1
    assert planner.get_last_failure() is None


def test_replanning_without_current_goal_is_ignored() -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._current_odom = _pose(0.0, 0.0)

    planner._replan_path(reason="stuck")

    assert planner._current_goal is None
    assert planner.get_last_failure() is None


def test_plan_path_without_current_goal_is_ignored() -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._current_odom = _pose(0.0, 0.0)

    planner._plan_path()

    assert planner._current_goal is None
    assert planner.get_last_failure() is None


def test_replanning_with_goal_but_missing_odom_records_failure() -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._current_goal = _pose(2.0, 0.0)

    planner._replan_path(reason="stuck")

    assert planner.get_last_failure() == "NO_ODOM"
    assert planner._current_goal is None


def test_missing_odom_records_failure_and_clears_goal() -> None:
    planner = GlobalPlanner(GlobalConfig())

    planner.handle_goal_request(_pose(1.0, 0.0))

    assert planner.get_last_failure() == "NO_ODOM"
    assert planner._current_goal is None


def test_record_motion_and_robot_status_store_latest_payloads() -> None:
    planner = GlobalPlanner(GlobalConfig())

    planner.record_motion_result({"precision_class": "short_stop", "status": "TIMEOUT"})
    planner.record_robot_status({"battery": {"state": "normal", "soc_percent": 88.0}})

    assert planner._latest_motion_result == {
        "precision_class": "short_stop",
        "status": "TIMEOUT",
    }
    assert planner._latest_robot_status == {
        "battery": {"state": "normal", "soc_percent": 88.0}
    }


def test_get_motion_diagnostics_exposes_latest_snapshot() -> None:
    planner = GlobalPlanner(GlobalConfig())

    planner.record_motion_result({"precision_class": "short_stop", "status": "TIMEOUT"})
    planner.record_robot_status({"battery": {"state": "normal", "soc_percent": 88.0}})
    planner._last_failure = "RETRY_LIMIT_EXCEEDED"
    planner._last_stop_reason = "obstacle_found"
    planner._last_stop_time = 123.0

    payload = planner.get_motion_diagnostics()

    assert payload["last_failure"] == "RETRY_LIMIT_EXCEEDED"
    assert payload["last_stop_reason"] == "obstacle_found"
    assert payload["last_stop_time"] == 123.0
    assert payload["measured_result"] == {
        "precision_class": "short_stop",
        "status": "TIMEOUT",
    }
    assert payload["robot_status"] == {
        "battery": {"state": "normal", "soc_percent": 88.0}
    }


@pytest.mark.parametrize(("end_error_deg", "expected_replans"), [(177.0, 0), (180.0, 1)])
def test_rotation_stuck_detection_uses_yaw_progress(
    monkeypatch, end_error_deg: float, expected_replans: int
) -> None:
    planner = GlobalPlanner(GlobalConfig())
    planner._stuck_time_window = 0.0
    planner._current_goal = _pose(2.0, 0.0)
    planner._current_odom = _pose(0.0, 0.0)
    rotation_errors = [math.pi, math.radians(end_error_deg)]
    replans: list[str | None] = []

    monkeypatch.setattr(planner._local_planner, "get_distance_to_path", lambda: None)
    monkeypatch.setattr(planner._local_planner, "get_unique_state", lambda: ("initial_rotation", 1))
    monkeypatch.setattr(planner._position_tracker, "is_stuck", lambda: True)

    def get_rotation_error() -> float:
        error = rotation_errors.pop(0)
        if not rotation_errors and expected_replans == 0:
            planner._stop_planner.set()
        return error

    def record_replan(*, reason: str | None = None) -> None:
        replans.append(reason)
        planner._stop_planner.set()

    monkeypatch.setattr(planner._local_planner, "get_rotation_error", get_rotation_error)
    monkeypatch.setattr(planner, "_replan_path", record_replan)

    planner._thread_entrypoint()

    assert replans == ["stuck"] * expected_replans
