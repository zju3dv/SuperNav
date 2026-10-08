"""Dependency-free LocalNav policies for plumbing and simulator smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

from supernav.methods.localnav.contracts import GoalSnapshot, PolicyInput, PolicyPrediction, VelocityAction


@dataclass(frozen=True)
class DebugPointPolicyConfig:
    """Parameters for the non-learning debug policy."""

    center_tolerance: float = 0.08
    linear_mps: float = 0.5
    angular_rps: float = 0.3
    horizon: int = 8


class DebugPointPolicy:
    """Smoke-test policy that steers from the original point's horizontal bearing.

    It intentionally does not perform visual correspondence and is not a
    navigation baseline.  Its only purpose is to exercise goal snapshots,
    action chunks, receding-horizon scheduling, and downstream adapters before
    a learned RGB policy is available.
    """

    def __init__(self, config: "DebugPointPolicyConfig | None" = None) -> None:
        self._config = config or DebugPointPolicyConfig()
        self._goal: "GoalSnapshot | None" = None

    @property
    def name(self) -> str:
        return "debug_point_policy"

    def reset(self, goal: GoalSnapshot) -> None:
        self._goal = goal

    def predict(self, inputs: PolicyInput) -> PolicyPrediction:
        goal = self._goal or inputs.goal
        offset = goal.point.x - 0.5
        if abs(offset) > self._config.center_tolerance:
            # Point right of center → turn right (negative angular velocity).
            angular = -self._config.angular_rps if offset > 0 else self._config.angular_rps
            linear = 0.0
        else:
            angular = 0.0
            linear = self._config.linear_mps
        actions = tuple(
            VelocityAction(linear, angular) for _ in range(self._config.horizon)
        )
        return PolicyPrediction(
            actions=actions,
            done_probability=0.0,
            confidence=1.0,
            diagnostics={"backend": self.name, "goal_x_offset": offset},
        )
