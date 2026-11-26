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

"""Navigation-domain failure codes for ``SkillResult``.

Movement and navigation skills should return ``SkillResult[NavigationSkillError]``
instead of prose-only strings so the harness can distinguish success, refusal,
and measured failure without parsing English.
"""

from typing import Literal

from dimos.agents.skill_result import CommonSkillError

NavigationError = Literal[
    "NO_ODOM",
    "NO_PROGRESS",
    "NO_SAFE_GOAL",
    "NO_PATH",
    "TARGET_NOT_FOUND",
    "VISION_UNAVAILABLE",
    "ARRIVAL_NOT_CONFIRMED",
    "CANCELLED",
    "OBSTACLE",
    "TIMEOUT",
    "LEASE_DENIED",
    "RETRY_LIMIT_EXCEEDED",
    "YAW_SPEED_BELOW_EFFECTIVE_MIN",
]

NavigationSkillError = CommonSkillError | NavigationError
