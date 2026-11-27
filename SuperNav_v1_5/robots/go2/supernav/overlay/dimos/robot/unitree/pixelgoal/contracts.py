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

"""Image goal and input validation contracts for PixelGoal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


class LocalNavError(Exception):
    """Base error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


class LocalNavInputError(LocalNavError):
    """The agent-facing goal or RGB observation is invalid."""




@dataclass(frozen=True)
class NormalizedPoint:
    """Image point with ``(0, 0)`` at top-left and ``(1, 1)`` at bottom-right."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise LocalNavInputError("invalid_point", "point coordinates must be finite")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise LocalNavInputError(
                "invalid_point",
                "point coordinates must be normalized into the inclusive [0, 1] range",
            )

    @classmethod
    def parse(cls, value: Sequence[float] | NormalizedPoint) -> NormalizedPoint:
        if isinstance(value, NormalizedPoint):
            return value
        try:
            x_raw, y_raw = value  # type: ignore[misc]
        except (TypeError, ValueError):
            raise LocalNavInputError("invalid_point", "point must be [x, y]") from None
        try:
            return cls(float(x_raw), float(y_raw))
        except (TypeError, ValueError):
            raise LocalNavInputError(
                "invalid_point", "point must contain numeric coordinates"
            ) from None

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        if int(width) <= 0 or int(height) <= 0:
            raise LocalNavInputError("invalid_image_size", "image dimensions must be positive")
        px = round(self.x * (int(width) - 1))
        py = round(self.y * (int(height) - 1))
        return (px, py)




@dataclass(frozen=True)
class GoalSnapshot:
    """Persistent copy of an agent-selected image goal.

    ``image_ref`` only has to be fresh when the snapshot is created.  The RGB
    copy and marked image remain valid while the local navigation episode runs,
    even after the panorama registry advances to a newer capture.
    """

    goal_id: str
    image_ref: str
    point: NormalizedPoint
    rgb: np.ndarray
    marked_rgb: np.ndarray | None = None
    capture_seq: int | None = None
    direction: str | None = None
    hfov: float | None = None

    def __post_init__(self) -> None:
        if not self.goal_id or not self.image_ref:
            raise LocalNavInputError("invalid_goal", "goal_id and image_ref must be non-empty")
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise LocalNavInputError("invalid_goal_rgb", "goal RGB must have shape [H, W, 3]")
        if self.marked_rgb is not None:
            marked = np.asarray(self.marked_rgb)
            if marked.shape != rgb.shape:
                raise LocalNavInputError(
                    "invalid_marked_goal_rgb",
                    "marked goal RGB must have the same shape as the original RGB",
                )
