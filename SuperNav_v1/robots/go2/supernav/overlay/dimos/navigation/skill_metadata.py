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

"""Shared metadata helpers for navigation-facing ``SkillResult`` payloads."""

from typing import Any

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.trigonometry import angle_diff


def vector_metadata(vector: Vector3 | None) -> dict[str, float] | None:
    """Serialize a vector with explicit meter units."""

    if vector is None:
        return None
    return {
        "x_m": vector.x,
        "y_m": vector.y,
        "z_m": vector.z,
    }


def pose_metadata(pose: PoseStamped | None) -> dict[str, Any] | None:
    """Serialize a pose with explicit frame and units for agent/tool metadata."""

    if pose is None:
        return None
    return {
        "frame_id": pose.frame_id,
        "position_m": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "yaw_rad": pose.yaw,
    }


def odom_delta_metadata(
    start: PoseStamped | None, end: PoseStamped | None
) -> dict[str, Any] | None:
    """Return measured odometry delta with explicit units when both poses exist."""

    if start is None or end is None:
        return None
    yaw_delta_rad = angle_diff(end.yaw, start.yaw)
    return {
        "linear_m": start.position.distance(end.position),
        "yaw_rad": yaw_delta_rad,
        "yaw_deg": yaw_delta_rad * 180.0 / 3.141592653589793,
    }
