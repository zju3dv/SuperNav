"""Stateful receding-horizon controller for LocalNav policies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from supernav.methods.localnav.contracts import (
    ControlDecision,
    GoalSnapshot,
    LocalNavPolicy,
    LocalNavRuntimeError,
    PolicyInput,
    PolicyPrediction,
    VelocityAction,
)
from supernav.methods.localnav.goal import as_readonly_rgb


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime cadence independent of the model architecture."""

    history_size: int = 4
    execute_actions: int = 3
    control_dt_s: float = 0.5
    done_threshold: float = 0.5
    confidence_threshold: float = 0.2

    def __post_init__(self) -> None:
        if int(self.history_size) < 1:
            raise ValueError("history_size must be >= 1")
        if int(self.execute_actions) < 1:
            raise ValueError("execute_actions must be >= 1")
        if not (math.isfinite(self.control_dt_s) and self.control_dt_s > 0.0):
            raise ValueError("control_dt_s must be positive and finite")
        for name, value in (
            ("done_threshold", self.done_threshold),
            ("confidence_threshold", self.confidence_threshold),
        ):
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0, 1]")


class RecedingHorizonController:
    """Keeps the goal image fixed while current RGB observations advance.

    The controller invokes the policy only when its action queue is empty,
    then executes at most ``execute_actions`` from the newly predicted chunk.
    This makes the Agent call low-frequency while retaining closed-loop local
    replanning at a configurable cadence.
    """

    def __init__(
        self, policy: LocalNavPolicy, config: "ControllerConfig | None" = None
    ) -> None:
        if not isinstance(policy, LocalNavPolicy):
            raise TypeError("policy must implement LocalNavPolicy")
        self._policy = policy
        self._config = config or ControllerConfig()
        self._goal: "GoalSnapshot | None" = None
        self._history: deque = deque(maxlen=self._config.history_size)
        self._queue: deque = deque()
        self._previous_action: "VelocityAction | None" = None
        self._latest_prediction: "PolicyPrediction | None" = None
        self._step_index = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def goal(self) -> "GoalSnapshot | None":
        return self._goal

    @property
    def control_dt_s(self) -> float:
        """Fixed duration for every continuous ``[v, omega]`` command."""
        return self._config.control_dt_s

    @property
    def latest_prediction(self) -> "PolicyPrediction | None":
        return self._latest_prediction

    def start(self, goal: GoalSnapshot) -> str:
        """Start a new local episode and return its persistent goal id."""
        self._goal = goal
        self._history.clear()
        self._queue.clear()
        self._previous_action = None
        self._latest_prediction = None
        self._step_index = 0
        self._active = True
        self._policy.reset(goal)
        return goal.goal_id

    def cancel(self) -> None:
        self._active = False
        self._queue.clear()

    def step(self, rgb) -> ControlDecision:
        """Consume the newest RGB frame and return at most one physical action."""
        if not self._active or self._goal is None:
            raise LocalNavRuntimeError(
                "localnav_not_started", "start(goal) must be called before step(rgb)"
            )
        self._history.append(as_readonly_rgb(rgb))

        replanned = False
        if not self._queue:
            inputs = PolicyInput(
                goal=self._goal,
                rgb_history=tuple(self._history),
                step_index=self._step_index,
                previous_action=self._previous_action,
            )
            prediction = self._policy.predict(inputs)
            if not isinstance(prediction, PolicyPrediction):
                raise LocalNavRuntimeError(
                    "invalid_prediction", "policy.predict must return PolicyPrediction"
                )
            self._latest_prediction = prediction
            replanned = True
            if prediction.done_probability >= self._config.done_threshold:
                self._active = False
                return self._decision("reached", None, replanned)
            if prediction.confidence < self._config.confidence_threshold:
                self._active = False
                return self._decision("needs_agent", None, replanned)
            if not prediction.actions:
                raise LocalNavRuntimeError(
                    "empty_action_chunk",
                    "a non-terminal LocalNav prediction must contain at least one action",
                )
            self._queue.extend(prediction.actions[: self._config.execute_actions])

        action = self._queue.popleft()
        self._previous_action = action
        self._step_index += 1
        return self._decision("running", action, replanned)

    def _decision(
        self, status: str, action: "VelocityAction | None", replanned: bool
    ) -> ControlDecision:
        return ControlDecision(
            status=status,
            action=action,
            replanned=replanned,
            queue_remaining=len(self._queue),
        )
