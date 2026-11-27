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

"""Trace-audit checks for request context and visual evidence budgets."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.validation_types import (
    LiveValidationCheck,
    fail_check as _fail,
    pass_check as _pass,
)


def request_assembly_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate that at least one request assembly trace exists."""

    if events:
        return _pass("request_assembled", f"Found {len(events)} request_assembled events.")
    return _fail(
        "request_assembled",
        "No request_assembled events found; model context growth is not auditable.",
    )


def context_budget_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate assembled requests do not exceed their configured token budget."""

    over_budget: list[dict[str, Any]] = []
    for event in events:
        assembly = assembly_payload(event)
        budget = assembly.get("token_budget")
        estimated = assembly.get("estimated_tokens")
        if budget is None or estimated is None:
            continue
        try:
            if int(estimated) > int(budget):
                over_budget.append({"estimated_tokens": estimated, "token_budget": budget})
        except (TypeError, ValueError):
            over_budget.append({"estimated_tokens": estimated, "token_budget": budget})

    if not over_budget:
        return _pass("context_budget", "No request_assembled event exceeds token budget.")
    return _fail(
        "context_budget",
        "At least one request_assembled event exceeds token budget.",
        over_budget=over_budget,
    )


def context_budget_exhaustion_trace_check(
    events: list[dict[str, Any]],
    request_events: list[dict[str, Any]],
) -> LiveValidationCheck:
    """Validate fail-closed budget exhaustion has a durable trace event."""

    exhausted = [
        assembly_payload(event)
        for event in request_events
        if assembly_payload(event).get("budget_exhausted") is True
    ]
    if not exhausted:
        return _pass("context_budget_exhaustion_trace", "No exhausted context assemblies found.")

    exhaustion_events = _events_of_type(events, "context_budget_exhausted")
    if exhaustion_events:
        return _pass(
            "context_budget_exhaustion_trace",
            f"Found {len(exhaustion_events)} context_budget_exhausted events.",
            exhausted=exhausted,
        )
    return _fail(
        "context_budget_exhaustion_trace",
        "request_assembled reported budget_exhausted but no context_budget_exhausted trace was emitted.",
        exhausted=exhausted,
    )


def visual_evidence_check(events: list[dict[str, Any]]) -> LiveValidationCheck:
    """Validate model-visible image evidence appears in request assembly traces."""

    included = []
    for event in events:
        images = assembly_payload(event).get("images_included") or []
        if images:
            included.extend(images)
    if included:
        return _pass(
            "visual_evidence",
            f"Found {len(included)} model-visible image injections.",
            images=included,
        )
    return _fail(
        "visual_evidence",
        "No model-visible image injections found in request_assembled events.",
    )


def context_token_growth_check(
    events: list[dict[str, Any]],
    max_growth: int,
) -> LiveValidationCheck:
    """Validate consecutive request token estimates stay within a growth budget."""

    if not events:
        return _fail(
            "context_token_growth",
            "No request_assembled events found; context token growth is not auditable.",
        )

    previous: int | None = None
    over_growth: list[dict[str, Any]] = []
    missing_estimates: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        estimated = _int_or_none(assembly_payload(event).get("estimated_tokens"))
        if estimated is None:
            missing_estimates.append({"request_index": index, "estimated_tokens": estimated})
            continue
        if previous is not None:
            growth = estimated - previous
            if growth > max_growth:
                over_growth.append(
                    {
                        "request_index": index,
                        "previous_tokens": previous,
                        "estimated_tokens": estimated,
                        "growth": growth,
                        "max_growth": max_growth,
                    }
                )
        previous = estimated

    if not over_growth and not missing_estimates:
        return _pass(
            "context_token_growth",
            f"No consecutive request_assembled event grows by more than {max_growth} tokens.",
        )
    return _fail(
        "context_token_growth",
        f"At least one request_assembled event grows by more than {max_growth} tokens.",
        over_growth=over_growth,
        missing_estimates=missing_estimates,
    )


def context_image_count_check(
    events: list[dict[str, Any]],
    max_images: int,
) -> LiveValidationCheck:
    """Validate per-request model-visible image counts stay within budget."""

    if not events:
        return _fail(
            "context_image_count",
            "No request_assembled events found; model-visible image count is not auditable.",
        )

    over_budget: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        images = assembly_payload(event).get("images_included")
        if images is None:
            images = []
        if not isinstance(images, list):
            malformed.append({"request_index": index, "images_included": images})
            continue
        if len(images) > max_images:
            over_budget.append(
                {
                    "request_index": index,
                    "image_count": len(images),
                    "max_images": max_images,
                }
            )

    if not over_budget and not malformed:
        return _pass(
            "context_image_count",
            f"Every request_assembled event includes at most {max_images} model-visible images.",
        )
    return _fail(
        "context_image_count",
        f"At least one request_assembled event exceeds {max_images} model-visible images.",
        over_budget=over_budget,
        malformed=malformed,
    )


def context_image_token_budget_check(
    events: list[dict[str, Any]],
    max_tokens: int,
) -> LiveValidationCheck:
    """Validate per-request model-visible image token estimates stay within budget."""

    if not events:
        return _fail(
            "context_image_tokens",
            "No request_assembled events found; model-visible image tokens are not auditable.",
        )

    over_budget: list[dict[str, Any]] = []
    missing_estimates: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for request_index, event in enumerate(events):
        images = assembly_payload(event).get("images_included")
        if images is None:
            images = []
        if not isinstance(images, list):
            malformed.append({"request_index": request_index, "images_included": images})
            continue

        total = 0
        for image_index, image in enumerate(images):
            if not isinstance(image, dict):
                malformed.append(
                    {
                        "request_index": request_index,
                        "image_index": image_index,
                        "image": image,
                    }
                )
                continue
            estimate = _int_or_none(image.get("estimated_tokens"))
            if estimate is None:
                missing_estimates.append(
                    {
                        "request_index": request_index,
                        "image_index": image_index,
                        "artifact_id": image.get("artifact_id"),
                    }
                )
                continue
            total += estimate
        if total > max_tokens:
            over_budget.append(
                {
                    "request_index": request_index,
                    "image_tokens": total,
                    "max_tokens": max_tokens,
                }
            )

    if not over_budget and not missing_estimates and not malformed:
        return _pass(
            "context_image_tokens",
            f"Every request_assembled event keeps image tokens within {max_tokens}.",
        )
    return _fail(
        "context_image_tokens",
        f"At least one request_assembled event exceeds {max_tokens} image tokens or lacks estimates.",
        over_budget=over_budget,
        missing_estimates=missing_estimates,
        malformed=malformed,
    )


def assembly_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Return the request assembly payload from a trace event, if present."""

    payload = event.get("payload") or {}
    assembly = payload.get("assembly") if isinstance(payload, dict) else None
    return assembly if isinstance(assembly, dict) else {}


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
