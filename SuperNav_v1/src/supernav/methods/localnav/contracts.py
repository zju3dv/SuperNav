"""Core contracts for RGB point-conditioned local navigation.

The contracts in this module deliberately contain no Habitat, torch, depth, or
navmesh dependency.  A learned policy can therefore be trained and served
independently, while simulator/robot adapters only need to translate the
physical velocity actions at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


class LocalNavError(Exception):
    """Base error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


class LocalNavInputError(LocalNavError):
    """The agent-facing goal or RGB observation is invalid."""


class LocalNavRuntimeError(LocalNavError):
    """The controller or policy violated the LocalNav runtime contract."""


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
    def parse(cls, value: "Sequence[float] | NormalizedPoint") -> "NormalizedPoint":
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
            raise LocalNavInputError(
                "invalid_image_size", "image dimensions must be positive"
            )
        px = int(round(self.x * (int(width) - 1)))
        py = int(round(self.y * (int(height) - 1)))
        return (px, py)


@dataclass(frozen=True)
class VelocityAction:
    """One continuous unicycle command in physical units.

    The policy predicts only ``[linear_mps, angular_rps]``.  Command duration
    is the controller's fixed ``control_dt_s`` and is not a model output.
    """

    linear_mps: float
    angular_rps: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.linear_mps, self.angular_rps)):
            raise LocalNavInputError(
                "invalid_action", "velocity action values must be finite"
            )

    def as_dict(self) -> dict[str, float]:
        return {"linear_mps": float(self.linear_mps), "angular_rps": float(self.angular_rps)}


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
    marked_rgb: "np.ndarray | None" = None
    capture_seq: "int | None" = None
    direction: "str | None" = None
    hfov: "float | None" = None

    def __post_init__(self) -> None:
        if not self.goal_id or not self.image_ref:
            raise LocalNavInputError(
                "invalid_goal", "goal_id and image_ref must be non-empty"
            )
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise LocalNavInputError(
                "invalid_goal_rgb", "goal RGB must have shape [H, W, 3]"
            )
        if self.marked_rgb is not None:
            marked = np.asarray(self.marked_rgb)
            if marked.shape != rgb.shape:
                raise LocalNavInputError(
                    "invalid_marked_goal_rgb",
                    "marked goal RGB must have the same shape as the original RGB",
                )


@dataclass(frozen=True)
class PolicyInput:
    """One inference request to a LocalNav policy."""

    goal: GoalSnapshot
    rgb_history: tuple[np.ndarray, ...]
    step_index: int
    previous_action: "VelocityAction | None" = None

    def __post_init__(self) -> None:
        if len(self.rgb_history) == 0:
            raise LocalNavInputError(
                "empty_rgb_history", "policy input needs at least one RGB frame"
            )
        if int(self.step_index) < 0:
            raise LocalNavInputError(
                "invalid_step_index", "step_index must be non-negative"
            )

    @property
    def current_rgb(self) -> np.ndarray:
        """The current observation is always the last history frame."""
        return self.rgb_history[-1]


@dataclass(frozen=True)
class PolicyPrediction:
    """Action chunk and terminal estimates returned by a policy."""

    actions: tuple[VelocityAction, ...]
    done_probability: float
    confidence: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("done_probability", self.done_probability),
            ("confidence", self.confidence),
        ):
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise LocalNavInputError(
                    "invalid_prediction", f"{name} must be finite and within [0, 1]"
                )


@dataclass(frozen=True)
class ControlDecision:
    """One controller decision for a simulator or robot adapter."""

    status: str
    action: "VelocityAction | None"
    replanned: bool
    queue_remaining: int


@runtime_checkable
class LocalNavPolicy(Protocol):
    """Backend contract implemented by a diffusion model or a test baseline."""

    @property
    def name(self) -> str: ...

    def reset(self, goal: GoalSnapshot) -> None: ...

    def predict(self, inputs: PolicyInput) -> PolicyPrediction: ...
