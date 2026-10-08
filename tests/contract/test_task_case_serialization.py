"""TaskCase data serialization and validation."""
from __future__ import annotations

import copy
import json

import pytest

from habitat_contract.task_schema import TaskCase


def _payload(task_type, goal, ground_truth):
    return {
        "case_id": "schema-roundtrip",
        "task_type": task_type,
        "scene_id": "scene-a",
        "start_position": [1.0, 0.2, -3.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "prompt_text": "Keep the original instruction. 保留原文。",
        "goal": goal,
        "ground_truth": ground_truth,
        "evaluation": {
            "metric": "success_rate", "threshold": 0.25,
            "answer_key_ref": "answers/case.json",
            "hidden_eval_positions": [[9.0, 0.2, 7.0]],
            "vlm_fallback_allowed": False, "agent_visible": False,
        },
        "provenance": {"source_file": "original.json", "episode_id": "42"},
    }


@pytest.mark.parametrize("task_type,goal,ground_truth", [
    ("pointnav", {"position": [9, 0.2, 7]}, {"position": [9, 0.2, 7]}),
    ("pointnav", {"instruction": "Approach the chair"}, {}),
    ("objectnav", {"object_category": "chair"}, {"object_instances": [{"position": [9, 0.2, 7]}]}),
    ("imagenav", {"reference_image": "reference.png"}, {}),
    ("instruction_following", {"instruction": "Turn left", "instruction_constraints": ["Do not enter"]}, {}),
    ("eqa", {"question": "What color is the chair?"}, {"answer": "blue"}),
])
def test_task_variants_roundtrip_without_losing_ground_truth_or_hidden_evaluation(task_type, goal, ground_truth):
    original = _payload(task_type, goal, ground_truth)
    before = copy.deepcopy(original)
    case = TaskCase.from_dict(original)
    restored = TaskCase.from_dict(json.loads(json.dumps(case.to_dict())))
    assert original == before
    assert case.to_dict() == restored.to_dict() == original


@pytest.mark.parametrize("change,match", [
    ({"start_position": [True, 0, 0]}, "start_position"),
    ({"ground_truth": {}}, "ground_truth.position"),
    ({"evaluation": {"metric": "success_rate", "hidden_eval_positions": [[1, 2, 3]], "agent_visible": True}}, "agent_visible"),
])
def test_task_deserialization_still_rejects_invalid_coordinates_missing_gt_and_visible_hidden_anchors(change, match):
    payload = _payload("pointnav", {"position": [9, 0.2, 7]}, {"position": [9, 0.2, 7]})
    payload.update(change)
    with pytest.raises(ValueError, match=match):
        TaskCase.from_dict(payload)
