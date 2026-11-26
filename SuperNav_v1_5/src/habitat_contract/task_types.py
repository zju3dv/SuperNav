"""Canonical task types and normalization at protocol boundaries.

TaskType supplies enum values for validation and completion. Stored
task_type fields use plain str, including when constructed from enums.
normalize_task_type accepts str or TaskType and raises ValueError for
missing, empty or unknown inputs. Constructors retain explicit validate()
semantics for other task-field checks.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Union


class TaskType(str, Enum):
    """Task types supported by the navigation protocol.

    str-Enum subclass means ``TaskType.POINTNAV == "pointnav"``
    and ``TaskType.POINTNAV.value == "pointnav"`` (same string).
    JSON encoders that see a str-Enum treat it as its string
    value; storage-as-str invariant in ``TaskCase`` prevents
    reliance on that coincidence at runtime.
    """

    POINTNAV = "pointnav"
    OBJECTNAV = "objectnav"
    IMAGENAV = "imagenav"
    EQA = "eqa"
    INSTRUCTION_FOLLOWING = "instruction_following"


SUPPORTED_TASK_TYPES: FrozenSet[str] = frozenset(t.value for t in TaskType)
"""Task-type values accepted at protocol boundaries."""


def normalize_task_type(value: Union[str, TaskType]) -> str:
    """Coerce ``str`` or ``TaskType`` to plain ``str`` and validate.

    Raises ``ValueError`` on:

    * empty string (treat as missing at the wire boundary — e.g.
      ``_as_str(payload, "task_type", default="")`` producing
      ``""`` when the key is absent)
    * unknown string not in ``SUPPORTED_TASK_TYPES``
    * non-str / non-TaskType input

    Returns plain ``str`` (the task_type value), **never a
    ``TaskType`` instance** — callers can freely assign to
    ``TaskCase.task_type`` without breaking the storage-as-str
    invariant.
    """

    if value is None:
        raise ValueError(
            "task_type is required; got None. At wire boundary "
            "this usually means the source payload omitted the "
            "task_type key."
        )
    if isinstance(value, TaskType):
        # Enum.value is already a plain str
        return str(value.value)
    if not isinstance(value, str):
        # Raise ValueError for malformed task types to match the deserialization
        # contract used by callers.
        raise ValueError(
            f"task_type must be str or TaskType, got "
            f"{type(value).__name__}: {value!r}"
        )
    if not value:
        raise ValueError(
            "task_type is required; got empty string. At wire "
            "boundary this usually means the source payload "
            "omitted the task_type key."
        )
    if value not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"task_type {value!r} is not supported; expected one "
            f"of {sorted(SUPPORTED_TASK_TYPES)}"
        )
    # Force plain str (defence against future str-subclass inputs
    # that satisfy isinstance(_, str) but would carry Enum / custom
    # identity).
    return str(value)


__all__ = ["TaskType", "SUPPORTED_TASK_TYPES", "normalize_task_type"]
