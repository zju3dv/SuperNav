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

"""Trace-audit checks for native request assembly contract shape."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)

_PAYLOAD_FIELDS = (
    "turn_id",
    "model",
    "assembly",
    "task_state",
    "policy",
    "tools_exposed",
    "tools_exposed_details",
    "sections",
)
_ASSEMBLY_FIELDS = (
    "total_history_messages",
    "included_messages",
    "estimated_tokens",
    "pinned_messages",
    "token_budget",
    "budget_exhausted",
    "failure_reason",
    "dropped",
    "artifacts_created",
    "images_included",
)
_SECTION_SOURCES = {
    "task_context": "task_state",
    "recent_history": "transcript",
    "visual_evidence": "recent_base64_images",
    "artifact_refs": "artifact_ledger",
}


def request_contract_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate the auditable shape of ``request_assembled`` trace events."""

    if not events:
        return _fail(
            "request_contract",
            "No request_assembled events found; request contract shape is not auditable.",
        )

    malformed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        problems = _request_contract_problems(index, event)
        if problems:
            malformed.append(problems)

    if not malformed:
        return _pass(
            "request_contract",
            "Every request_assembled event includes assembly, task, policy, tool, and section metadata.",
            requests=len(events),
        )
    return _fail(
        "request_contract",
        "Some request_assembled events lack the auditable native request contract.",
        malformed=malformed,
    )


def _request_contract_problems(index: int, event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    problems: dict[str, Any] = {"request_index": index}
    if not isinstance(payload, dict):
        problems["payload"] = "missing_or_not_object"
        return problems

    missing_payload = [field for field in _PAYLOAD_FIELDS if field not in payload]
    if missing_payload:
        problems["missing_payload_fields"] = missing_payload

    assembly = payload.get("assembly")
    if isinstance(assembly, dict):
        missing_assembly = [field for field in _ASSEMBLY_FIELDS if field not in assembly]
        if missing_assembly:
            problems["missing_assembly_fields"] = missing_assembly
    else:
        problems["assembly"] = "missing_or_not_object"

    _record_type_problem(problems, payload, "task_state", dict)
    _record_type_problem(problems, payload, "policy", dict)
    _record_string_list_problem(problems, payload, "tools_exposed")
    _record_tool_details_problem(problems, payload, "tools_exposed_details")
    sections = payload.get("sections")
    if isinstance(sections, list):
        section_problems = _section_contract_problems(assembly, sections)
        if section_problems:
            problems["sections"] = section_problems
    else:
        problems["sections"] = "missing_or_not_list"

    if len(problems) == 1:
        return {}
    return problems


def _record_type_problem(
    problems: dict[str, Any],
    payload: dict[str, Any],
    field: str,
    expected_type: type,
) -> None:
    if field in payload and not isinstance(payload.get(field), expected_type):
        problems[field] = f"not_{expected_type.__name__}"


def _record_string_list_problem(
    problems: dict[str, Any],
    payload: dict[str, Any],
    field: str,
) -> None:
    value = payload.get(field)
    if field not in payload:
        return
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        problems[field] = "not_list_of_strings"


def _record_tool_details_problem(
    problems: dict[str, Any],
    payload: dict[str, Any],
    field: str,
) -> None:
    value = payload.get(field)
    if field not in payload:
        return
    if not isinstance(value, list):
        problems[field] = "not_list"
        return
    malformed: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            malformed.append({"index": index, "reason": "not_object"})
            continue
        if not isinstance(item.get("name"), str) or not item.get("name"):
            malformed.append({"index": index, "reason": "missing_name"})
        uses = item.get("uses", [])
        if not isinstance(uses, list) or any(not isinstance(use, str) for use in uses):
            malformed.append({"index": index, "reason": "uses_not_string_list"})
    if malformed:
        problems[field] = malformed


def _section_contract_problems(
    assembly: Any,
    sections: list[Any],
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    section_by_id: dict[str, dict[str, Any]] = {}
    for section_index, section in enumerate(sections):
        if not isinstance(section, dict):
            problems.append({"section_index": section_index, "section": "not_object"})
            continue
        section_id = section.get("id")
        source = section.get("source")
        if not isinstance(section_id, str) or not section_id:
            problems.append({"section_index": section_index, "missing": ["id"]})
            continue
        if not isinstance(source, str) or not source:
            problems.append(
                {"section_index": section_index, "id": section_id, "missing": ["source"]}
            )
            continue
        expected_source = _SECTION_SOURCES.get(section_id)
        if expected_source is not None and source != expected_source:
            problems.append(
                {
                    "section_index": section_index,
                    "id": section_id,
                    "source": source,
                    "expected_source": expected_source,
                }
            )
        section_by_id[section_id] = section

    if isinstance(assembly, dict):
        problems.extend(_required_section_problems(assembly, section_by_id))
    return problems


def _required_section_problems(
    assembly: dict[str, Any],
    section_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    required: list[dict[str, Any]] = []
    pinned_messages = _int_or_none(assembly.get("pinned_messages"))
    if pinned_messages and pinned_messages > 0 and "task_context" not in section_by_id:
        required.append({"missing_section": "task_context", "reason": "pinned_messages"})

    images = assembly.get("images_included") or []
    if isinstance(images, list) and images and "visual_evidence" not in section_by_id:
        required.append({"missing_section": "visual_evidence", "reason": "images_included"})

    artifacts = assembly.get("artifacts_created") or []
    if isinstance(artifacts, list) and artifacts and "artifact_refs" not in section_by_id:
        required.append({"missing_section": "artifact_refs", "reason": "artifacts_created"})
    return required


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
