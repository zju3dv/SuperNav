"""Serializable training-sample contract for point-conditioned LocalNav."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from supernav.methods.localnav.contracts import LocalNavInputError, NormalizedPoint, VelocityAction


def load_samples_jsonl(path: str | Path) -> list["LocalNavTrainingSample"]:
    """Load the owned sample schema, preserving record order and metadata."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(LocalNavTrainingSample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, LocalNavInputError) as exc:
                raise ValueError(f"Invalid LocalNav sample at {path}:{number}: {exc}") from exc
    return records


@dataclass(frozen=True)
class LocalNavTrainingSample:
    """One sample generated from an expert trajectory.

    ``goal_frame_index`` is the frame on which the Agent-like point was
    selected.  ``current_frame_index`` may advance while the goal image stays
    fixed, which teaches the policy to finish a local hop without another
    high-level Agent call.
    """

    sample_id: str
    source: str
    episode_id: str
    goal_image_path: str
    goal_frame_index: int
    current_image_paths: tuple[str, ...]
    current_frame_index: int
    target_frame_index: int
    point: NormalizedPoint
    future_actions: tuple[VelocityAction, ...]
    control_dt_s: float
    done: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.episode_id or not self.goal_image_path:
            raise LocalNavInputError(
                "invalid_training_sample",
                "sample_id, episode_id, and goal_image_path must be non-empty",
            )
        if len(self.current_image_paths) < 1:
            raise LocalNavInputError(
                "invalid_training_sample",
                "current_image_paths must contain at least one frame",
            )
        if not (
            self.goal_frame_index <= self.current_frame_index <= self.target_frame_index
        ):
            raise LocalNavInputError(
                "invalid_training_sample",
                "expected goal_frame_index <= current_frame_index <= target_frame_index",
            )
        if not self.done and not self.future_actions:
            raise LocalNavInputError(
                "invalid_training_sample", "non-terminal samples require future_actions"
            )
        if not (math.isfinite(self.control_dt_s) and self.control_dt_s > 0.0):
            raise LocalNavInputError(
                "invalid_training_sample", "control_dt_s must be positive and finite"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "episode_id": self.episode_id,
            "goal_image_path": self.goal_image_path,
            "goal_frame_index": int(self.goal_frame_index),
            "current_image_paths": list(self.current_image_paths),
            "current_frame_index": int(self.current_frame_index),
            "target_frame_index": int(self.target_frame_index),
            "point": list(self.point.as_tuple()),
            "future_actions": [action.as_dict() for action in self.future_actions],
            "control_dt_s": float(self.control_dt_s),
            "done": bool(self.done),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalNavTrainingSample":
        return cls(
            sample_id=str(value["sample_id"]),
            source=str(value.get("source", "")),
            episode_id=str(value["episode_id"]),
            goal_image_path=str(value["goal_image_path"]),
            goal_frame_index=int(value["goal_frame_index"]),
            current_image_paths=tuple(
                str(path) for path in value.get("current_image_paths", ())
            ),
            current_frame_index=int(value["current_frame_index"]),
            target_frame_index=int(value["target_frame_index"]),
            point=NormalizedPoint.parse(value["point"]),
            future_actions=tuple(
                VelocityAction(
                    float(action["linear_mps"]), float(action["angular_rps"])
                )
                for action in value.get("future_actions", ())
            ),
            control_dt_s=float(value["control_dt_s"]),
            done=bool(value.get("done", False)),
            metadata=dict(value.get("metadata", {})),
        )
