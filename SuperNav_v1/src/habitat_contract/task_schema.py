"""Task and evaluation contracts shared by bridge, agent and evaluators.

TaskCase stores task identity, start pose, prompt, goal, ground truth,
evaluation and provenance. validate() checks task-specific shapes and
required ground truth. EvaluationSpec stores metric thresholds,
answer-key references and hidden evaluation positions separately from
agent-facing goal fields; hidden evaluation data requires agent_visible=False.

These standard-library contracts describe data. Experiment configuration
and prompt construction belong to their callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Use the shared task type set for bridge and agent validation.
from habitat_contract.task_types import (  # noqa: E402
    SUPPORTED_TASK_TYPES,
    normalize_task_type,
)


# ---------------------------------------------------------------------------
# Coordinate validators
# ---------------------------------------------------------------------------


# Strict JSON helpers distinguish an absent key from a present value of the wrong type,
# preserving evaluation data types at deserialization.


def _as_str(payload: Dict[str, Any], key: str, *, default: str) -> str:
    if key not in payload:
        return default
    raw = payload[key]
    if raw is None:
        return default
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a string, got {type(raw).__name__}: {raw!r}")
    return raw


def _as_bool(payload: Dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in payload:
        return default
    raw = payload[key]
    if raw is None:
        return default
    # ``isinstance(True, int)`` is True; we want the reverse check —
    # reject ints / strings / lists etc.  Only exact bool survives.
    if not isinstance(raw, bool):
        raise ValueError(f"{key} must be a bool, got {type(raw).__name__}: {raw!r}")
    return raw


def _as_list(payload: Dict[str, Any], key: str) -> List[Any]:
    if key not in payload:
        return []
    raw = payload[key]
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list, got {type(raw).__name__}: {raw!r}")
    return list(raw)


def _as_dict(
    payload: Dict[str, Any], key: str, *, default: Dict[str, Any]
) -> Dict[str, Any]:
    if key not in payload:
        return dict(default)
    raw = payload[key]
    if raw is None:
        return dict(default)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object, got {type(raw).__name__}: {raw!r}")
    return dict(raw)


def _validate_vec(value: Any, length: int, field_name: str) -> None:
    """Validate a fixed-length numeric vector (3-vec for positions,
    4-vec for xyzw quaternions). Rotations and positions share the
    same strict shape check."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field_name} must be a list/tuple of {length} numbers, "
            f"got {type(value).__name__}: {value!r}"
        )
    if len(value) != length:
        raise ValueError(
            f"{field_name} must have length {length}, "
            f"got len={len(value)}: {value!r}"
        )
    for i, x in enumerate(value):
        # bool is a subclass of int — treat it as a non-numeric
        # type-safety smell rather than silently coercing to 0/1.
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(
                f"{field_name}[{i}] must be numeric (int/float), "
                f"got {type(x).__name__}: {x!r}"
            )


# ---------------------------------------------------------------------------
# EvaluationSpec
# ---------------------------------------------------------------------------


@dataclass
class EvaluationSpec:
    """How a :class:`TaskCase` should be scored.

    Separate from ``goal`` / ``prompt_text`` so evaluators can use
    anchors and answer keys without exposing them to the agent.

    Fields:
        metric: e.g. ``"success_rate"``, ``"success_distance"``,
            ``"answer_match"``. Free-form so task-type metrics
            can be added without bumping the schema; downstream
            consumers pick a handler by string.
        threshold: numeric cutoff for pass/fail where the metric
            is scalar (e.g. ``success_distance ≤ 1.0m``). Optional
            because answer-match metrics don't use a threshold.
        answer_key_ref: pointer to an external answer store —
            ``"answers/foo.jsonl#case-id"`` or similar. The contract
            treats this as opaque: the evaluator looks up the
            answer by ref at scoring time.
        hidden_eval_positions: list of 3-vectors the evaluator
            may use to score proximity-based success but which
            the *agent* must never see. Requires
            ``agent_visible=False``.
        vlm_fallback_allowed: whether the evaluator may call a
            VLM for answer extraction when the nav agent's
            output doesn't parse directly (EQA / IF tie-breaker).
        agent_visible: whether this EvaluationSpec is safe to
            render into the forward agent's prompt. Defaults to
            True for metric/threshold-only specs; MUST be False
            whenever ``hidden_eval_positions`` is non-empty.
    """

    metric: str
    threshold: Optional[float] = None
    answer_key_ref: str = ""
    hidden_eval_positions: List[List[float]] = field(default_factory=list)
    vlm_fallback_allowed: bool = False
    agent_visible: bool = True

    def validate(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("EvaluationSpec.metric must be a non-empty string")
        # answer_key_ref is an opaque string resolved during scoring. Require a string
        # before serialization, including for direct constructors.
        if not isinstance(self.answer_key_ref, str):
            raise ValueError(
                f"EvaluationSpec.answer_key_ref must be a string, got "
                f"{type(self.answer_key_ref).__name__}: "
                f"{self.answer_key_ref!r}"
            )
        if self.threshold is not None and (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
        ):
            raise ValueError(
                f"EvaluationSpec.threshold must be numeric, got "
                f"{type(self.threshold).__name__}: {self.threshold!r}"
            )
        # Require actual bool values for evaluation flags: truthiness coercion would
        # interpret the string "false" as True and could expose hidden data.
        for flag_name in ("vlm_fallback_allowed", "agent_visible"):
            flag_value = getattr(self, flag_name)
            if not isinstance(flag_value, bool):
                raise ValueError(
                    f"EvaluationSpec.{flag_name} must be a bool, got "
                    f"{type(flag_value).__name__}: {flag_value!r}"
                )
        # Validate the hidden_eval_positions list type before testing whether it is
        # empty, then check visibility and individual entries.
        if not isinstance(self.hidden_eval_positions, list):
            raise ValueError(
                "EvaluationSpec.hidden_eval_positions must be a list, "
                f"got {type(self.hidden_eval_positions).__name__}: "
                f"{self.hidden_eval_positions!r}"
            )
        if self.hidden_eval_positions:
            if self.agent_visible:
                raise ValueError(
                    "EvaluationSpec.hidden_eval_positions is non-empty "
                    "but agent_visible=True — hidden eval data must NOT "
                    "leak to the forward agent's prompt (set "
                    "agent_visible=False)."
                )
            for i, pos in enumerate(self.hidden_eval_positions):
                _validate_vec(pos, 3, f"EvaluationSpec.hidden_eval_positions[{i}]")
        # Answer-key references are evaluator-only metadata. A non-empty reference
        # requires agent_visible=False.
        if self.answer_key_ref and self.agent_visible:
            raise ValueError(
                "EvaluationSpec.answer_key_ref is set but "
                "agent_visible=True — answer-key metadata must NOT "
                "leak to the forward agent's prompt (set "
                "agent_visible=False)."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "answer_key_ref": self.answer_key_ref,
            "hidden_eval_positions": [list(p) for p in self.hidden_eval_positions],
            "vlm_fallback_allowed": bool(self.vlm_fallback_allowed),
            "agent_visible": bool(self.agent_visible),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationSpec":
        if not isinstance(payload, dict):
            raise ValueError(
                f"EvaluationSpec payload must be an object, "
                f"got {type(payload).__name__}"
            )
        # An absent hidden-position key defaults to an empty list. A present value must
        # be a list, including when it is falsy.
        if "hidden_eval_positions" in payload:
            hidden_raw = payload["hidden_eval_positions"]
            if hidden_raw is None:
                hidden_raw = []
            elif not isinstance(hidden_raw, list):
                raise ValueError(
                    "EvaluationSpec.hidden_eval_positions must be a list, "
                    f"got {type(hidden_raw).__name__}"
                )
        else:
            hidden_raw = []
        # Validate each position at deserialization so malformed vectors raise the
        # contract's ValueError.
        hidden: List[List[float]] = []
        for i, pos in enumerate(hidden_raw):
            _validate_vec(pos, 3, f"EvaluationSpec.hidden_eval_positions[{i}]")
            hidden.append(list(pos))
        # Absent or null thresholds remain optional; present values of the wrong type
        # raise a deserialization error.
        threshold: Optional[float]
        if "threshold" in payload:
            threshold_raw = payload["threshold"]
            if threshold_raw is None:
                threshold = None
            elif isinstance(threshold_raw, bool) or not isinstance(
                threshold_raw, (int, float)
            ):
                raise ValueError(
                    "EvaluationSpec.threshold must be numeric (or null), "
                    f"got {type(threshold_raw).__name__}: {threshold_raw!r}"
                )
            else:
                threshold = float(threshold_raw)
        else:
            threshold = None
        # Use typed helpers for strings and bools so malformed scalars fail at the JSON
        # boundary.
        spec = cls(
            metric=_as_str(payload, "metric", default=""),
            threshold=threshold,
            answer_key_ref=_as_str(payload, "answer_key_ref", default=""),
            hidden_eval_positions=hidden,
            vlm_fallback_allowed=_as_bool(
                payload, "vlm_fallback_allowed", default=False
            ),
            agent_visible=_as_bool(payload, "agent_visible", default=True),
        )
        # Validate visibility at deserialization before a consumer can render evaluator-
        # only positions or answer-key references.
        spec.validate()
        return spec


# ---------------------------------------------------------------------------
# TaskCase
# ---------------------------------------------------------------------------


@dataclass
class TaskCase:
    """Normalized cross-system task.

    Fields:
        case_id: stable identifier, non-empty. For experiment
            pipelines this typically looks like
            ``"<scene>-ep-<episode_id>"`` or
            ``"case-<index>"``; the contract doesn't enforce a
            naming scheme.
        task_type: one of :data:`SUPPORTED_TASK_TYPES`.
        scene_id: scene namespace. Empty string permitted for
            task types that are scene-agnostic, but the current
            five all require it; ``validate()`` enforces
            non-empty.
        start_position: agent's starting 3-vec position.
        start_rotation: agent's starting xyzw quaternion (4-vec).
        prompt_text: the task instruction rendered for the
            forward agent. This is the plain string the runner
            should show the agent — building it from raw episode
            data is a runner concern (and stays in
            ``tools/test_suite``).
        goal: task-type-specific goal payload:
                pointnav: ``{"position": [x, y, z]}``
                objectnav: ``{"object_category": str,
                              "goal_description": str?}``
                imagenav: ``{"reference_image": str}``
                instruction_following:
                    ``{"instruction": str,
                       "instruction_constraints": [str]?}``
                eqa: ``{"question": str}``
        ground_truth: task-type-specific evaluation reference.
            ``_validate_per_task`` enforces its per-type invariants.
        evaluation: optional :class:`EvaluationSpec`. Missing
            means the oracle handles scoring without extra
            metadata.
        provenance: free-form dict recording where the case came
            from — ``{"episode_id": "...", "source_file": "..."}``
            is typical. Not validated.
    """

    case_id: str
    task_type: str
    scene_id: str
    start_position: Sequence[float]
    start_rotation: Sequence[float]
    prompt_text: str
    goal: Dict[str, Any]
    ground_truth: Dict[str, Any]
    evaluation: Optional[EvaluationSpec] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ── storage normalisation ────────────────────────────────────

    def __post_init__(self) -> None:
        """Coerce TaskType enum values to plain strings on direct construction.

        Other strings are retained; validate() checks whether they are supported.
        """

        # Local import to avoid a module-level cycle between
        # task_types and task_schema.
        from habitat_contract.task_types import TaskType

        if isinstance(self.task_type, TaskType):
            # Use ``str(Enum.value)`` not ``str(Enum)`` — the
            # latter yields 'TaskType.POINTNAV' via Enum's default
            # __str__, not 'pointnav'.
            self.task_type = str(self.task_type.value)

    # ── validation ────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ``ValueError`` on any invariant violation."""

        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type {self.task_type!r} not supported; expected "
                f"one of {sorted(SUPPORTED_TASK_TYPES)}"
            )
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        _validate_vec(self.start_position, 3, "start_position")
        _validate_vec(self.start_rotation, 4, "start_rotation (xyzw quaternion)")
        if not isinstance(self.prompt_text, str):
            raise ValueError(
                f"prompt_text must be a string, got "
                f"{type(self.prompt_text).__name__}"
            )
        if not isinstance(self.goal, dict):
            raise ValueError(f"goal must be an object, got {type(self.goal).__name__}")
        if not isinstance(self.ground_truth, dict):
            raise ValueError(
                f"ground_truth must be an object, got "
                f"{type(self.ground_truth).__name__}"
            )
        self._validate_per_task()
        if self.evaluation is not None:
            self.evaluation.validate()

    def _validate_per_task(self) -> None:
        """Validate the goal and ground-truth shape for this task type."""

        t = self.task_type
        if t == "pointnav":
            # PointNav accepts a position goal or an instruction goal. Require position
            # ground truth only for position goals, and validate any supplied
            # coordinates.
            has_position_goal = "position" in self.goal
            has_instruction_goal = bool(
                self.goal.get("instruction") or self.goal.get("goal_description")
            )
            if not (has_position_goal or has_instruction_goal):
                raise ValueError(
                    "pointnav task requires goal.position=[x, y, z] "
                    "OR goal.instruction / goal.goal_description for "
                    "instruction-goal cases"
                )
            if has_position_goal:
                _validate_vec(self.goal["position"], 3, "goal.position")
                if "position" not in self.ground_truth:
                    raise ValueError(
                        "pointnav position-goal task requires "
                        "ground_truth.position=[x, y, z]"
                    )
                _validate_vec(self.ground_truth["position"], 3, "ground_truth.position")
            elif "position" in self.ground_truth:
                # Instruction-goal case shouldn't carry a
                # position GT, but if one is supplied we still
                # shape-check it (data-consistency guard).
                _validate_vec(self.ground_truth["position"], 3, "ground_truth.position")
        elif t == "objectnav":
            if "object_category" not in self.goal:
                raise ValueError("objectnav task requires goal.object_category")
            if "object_instances" not in self.ground_truth:
                raise ValueError(
                    "objectnav task requires "
                    "ground_truth.object_instances=[{position: [..]}, ...]"
                )
            instances = self.ground_truth.get("object_instances")
            if not isinstance(instances, (list, tuple)):
                raise ValueError(
                    "ground_truth.object_instances must be a list, "
                    f"got {type(instances).__name__}"
                )
            for i, inst in enumerate(instances):
                if not isinstance(inst, dict):
                    raise ValueError(
                        f"ground_truth.object_instances[{i}] must be a "
                        f"dict, got {type(inst).__name__}: {inst!r}"
                    )
                if "position" in inst:
                    _validate_vec(
                        inst["position"],
                        3,
                        f"ground_truth.object_instances[{i}].position",
                    )
        elif t == "imagenav":
            if "reference_image" not in self.goal:
                raise ValueError("imagenav task requires goal.reference_image")
            # ImageNav may use reference-image similarity without a target position.
            # Validate the position shape whenever coordinates are supplied.
            if "position" in self.ground_truth:
                _validate_vec(self.ground_truth["position"], 3, "ground_truth.position")
        elif t == "instruction_following":
            if "instruction" not in self.goal:
                raise ValueError("instruction_following task requires goal.instruction")
            # Instruction-following tasks may use constraint/report evaluation without
            # final-position ground truth. Validate any supplied final position.
            if "final_position" in self.ground_truth:
                _validate_vec(
                    self.ground_truth["final_position"],
                    3,
                    "ground_truth.final_position",
                )
        elif t == "eqa":
            if "question" not in self.goal:
                raise ValueError("eqa task requires goal.question")
            if "answer" not in self.ground_truth:
                raise ValueError("eqa task requires ground_truth.answer")

    # ── serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "scene_id": self.scene_id,
            "start_position": list(self.start_position),
            "start_rotation": list(self.start_rotation),
            "prompt_text": self.prompt_text,
            "goal": dict(self.goal),
            "ground_truth": dict(self.ground_truth),
            "evaluation": (
                self.evaluation.to_dict() if self.evaluation is not None else None
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskCase":
        if not isinstance(payload, dict):
            raise ValueError(
                f"TaskCase payload must be an object, " f"got {type(payload).__name__}"
            )

        # Absent or null evaluation stays optional; a present value must be a mapping so
        # malformed metadata cannot be discarded.
        if "evaluation" in payload:
            eval_raw = payload["evaluation"]
            if eval_raw is None:
                evaluation = None
            elif isinstance(eval_raw, dict):
                evaluation = EvaluationSpec.from_dict(eval_raw)
            else:
                raise ValueError(
                    "TaskCase.evaluation must be an object (or null), "
                    f"got {type(eval_raw).__name__}"
                )
        else:
            evaluation = None
        # Typed helpers distinguish absent keys from malformed values.
        # normalize_task_type accepts strings and enums and reports missing or unknown
        # types with ValueError.
        raw_task_type = _as_str(payload, "task_type", default="")
        normalized_task_type = normalize_task_type(raw_task_type)
        case = cls(
            case_id=_as_str(payload, "case_id", default=""),
            task_type=normalized_task_type,
            scene_id=_as_str(payload, "scene_id", default=""),
            start_position=_as_list(payload, "start_position"),
            start_rotation=_as_list(payload, "start_rotation"),
            prompt_text=_as_str(payload, "prompt_text", default=""),
            goal=_as_dict(payload, "goal", default={}),
            ground_truth=_as_dict(payload, "ground_truth", default={}),
            evaluation=evaluation,
            provenance=_as_dict(payload, "provenance", default={}),
        )
        # Validate task and ground-truth invariants before returning a deserialized
        # case.
        case.validate()
        return case


__all__ = [
    "SUPPORTED_TASK_TYPES",
    "EvaluationSpec",
    "TaskCase",
]
