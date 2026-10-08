"""Detect bounded motion over a window of completed rounds.

record(position) tracks a fixed-size deque. is_blocked() checks whether
all positions fall within radius_m of the 3D centroid, enabling the
caller to replan or declare blocked.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple


Position = Tuple[float, float, float]


@dataclass
class BlockedDetector:
    window_size: int
    radius_m: float
    _history: Deque[Position] = field(init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0, got {self.window_size}"
            )
        if self.radius_m <= 0:
            raise ValueError(f"radius_m must be > 0, got {self.radius_m}")
        # init=False so mypy / IDEs don't flag the private field.
        # Using object.__setattr__ would be overkill; dataclass
        # permits direct assignment here since frozen=False.
        self._history = deque(maxlen=self.window_size)

    def record(self, position: Position) -> None:
        """Append a position to the sliding window."""

        x, y, z = position
        self._history.append((float(x), float(y), float(z)))

    def is_blocked(self) -> bool:
        """True when the window is full AND every sample lies
        within ``radius_m`` of the window centroid.

        Returns False if the window hasn't filled up yet, so the
        detector never flags ``blocked`` before it has enough data
        to be confident.
        """

        if len(self._history) < self.window_size:
            return False
        cx = sum(p[0] for p in self._history) / self.window_size
        cy = sum(p[1] for p in self._history) / self.window_size
        cz = sum(p[2] for p in self._history) / self.window_size
        for x, y, z in self._history:
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) > self.radius_m:
                return False
        return True

    def reset(self) -> None:
        """Clear the sliding window — call after the forward loop
        has decided to escalate so the next round starts fresh."""

        self._history.clear()


__all__ = ["BlockedDetector", "Position"]
