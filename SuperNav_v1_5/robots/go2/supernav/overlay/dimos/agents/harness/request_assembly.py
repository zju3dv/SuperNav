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

"""Native-agent model request assembly seam.

The native LangGraph client remains intentionally small. This module keeps the
robot harness request contract explicit so context policy, task pinning, and
trace payload shape can evolve without adding more turn-loop logic to
``McpClient``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.context_manager import ContextAssembly, ConversationContextManager
from dimos.agents.runtime_policy import AgentRuntimePolicy


class TaskContextProvider(Protocol):
    """Task-state subset needed by request assembly."""

    def context_messages(self) -> list[BaseMessage]: ...
    def snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NativeModelRequest:
    """A model-visible request and its traceable harness metadata."""

    context: ContextAssembly
    task_state: dict[str, Any]
    policy_state: dict[str, Any]
    tools_exposed: list[str]
    tools_exposed_details: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    turn_id: str | None = None
    model: str | None = None

    @property
    def messages(self) -> list[BaseMessage]:
        """Messages to pass to the model graph."""

        return self.context.messages

    def trace_payload(self) -> dict[str, Any]:
        """Payload fields for the ``request_assembled`` trace event."""

        payload = {
            "assembly": self.context.model_dump(),
            "task_state": self.task_state,
            "policy": self.policy_state,
            "tools_exposed": self.tools_exposed,
            "tools_exposed_details": self.tools_exposed_details,
            "sections": self.sections,
        }
        if self.turn_id is not None:
            payload["turn_id"] = self.turn_id
        if self.model is not None:
            payload["model"] = self.model
        return payload


class NativeRequestAssembler:
    """Build bounded model requests for the native DimOS MCP agent."""

    def __init__(
        self,
        *,
        context_manager: ConversationContextManager,
        task_context: TaskContextProvider | None = None,
        policy: AgentRuntimePolicy | None = None,
        tools_exposed: list[str] | None = None,
        tools_exposed_details: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        model: str | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._task_context = task_context
        self._policy = policy
        self._tools_exposed_details = _normalize_tool_details(tools_exposed_details or [])
        names_from_details = [tool["name"] for tool in self._tools_exposed_details]
        self._tools_exposed = sorted(tools_exposed or names_from_details)
        self._turn_id = turn_id
        self._model = model

    def assemble(self, history: list[BaseMessage]) -> NativeModelRequest:
        """Assemble current history plus pinned task context into a model request."""

        pinned_messages = (
            self._task_context.context_messages() if self._task_context is not None else []
        )
        task_state = self._task_context.snapshot() if self._task_context is not None else {}
        context = self._context_manager.assemble(history, pinned_messages=pinned_messages)
        policy_state = self._policy.model_dump() if self._policy is not None else {}
        sections = _request_sections(context, self._context_manager)
        return NativeModelRequest(
            context=context,
            task_state=task_state,
            policy_state=policy_state,
            tools_exposed=self._tools_exposed,
            tools_exposed_details=self._tools_exposed_details,
            sections=sections,
            turn_id=self._turn_id,
            model=self._model,
        )


def _normalize_tool_details(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return stable, trace-safe metadata for tools exposed to the model."""

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        meta = tool.get("_meta") if isinstance(tool.get("_meta"), dict) else {}
        uses = tool.get("uses")
        if uses is None:
            uses = meta.get("dimos/uses") if isinstance(meta, dict) else []
        normalized.append(
            {
                "name": name,
                "uses": sorted(str(use) for use in uses) if isinstance(uses, list | tuple) else [],
            }
        )
    return sorted(normalized, key=lambda item: item["name"])


def _request_sections(
    context: ContextAssembly,
    context_manager: ConversationContextManager,
) -> list[dict[str, Any]]:
    """Return traceable section summaries for one assembled model request."""

    messages = context.messages
    image_indices = {
        int(image["request_message_index"])
        for image in context.images_included
        if isinstance(image.get("request_message_index"), int)
    }
    sections: list[dict[str, Any]] = []

    pinned_messages = messages[: context.pinned_messages]
    if pinned_messages:
        sections.append(
            _message_section(
                "task_context",
                "task_state",
                pinned_messages,
                context_manager,
            )
        )

    history_messages = [
        message
        for index, message in enumerate(messages)
        if index >= context.pinned_messages and index not in image_indices
    ]
    if history_messages:
        sections.append(
            _message_section(
                "recent_history",
                "transcript",
                history_messages,
                context_manager,
            )
        )

    if context.images_included:
        sections.append(
            {
                "id": "visual_evidence",
                "source": "recent_base64_images",
                "images": len(context.images_included),
                "pinned_images": sum(
                    1 for image in context.images_included if image.get("pinned") is True
                ),
                "recent_images": sum(
                    1
                    for image in context.images_included
                    if image.get("inclusion_reason") == "recent_image"
                ),
                "tokens": sum(
                    int(image.get("estimated_tokens") or 0) for image in context.images_included
                ),
                "artifact_ids": [
                    image.get("artifact_id")
                    for image in context.images_included
                    if image.get("artifact_id")
                ],
                "inclusion_reasons": [
                    image.get("inclusion_reason")
                    for image in context.images_included
                    if image.get("inclusion_reason")
                ],
            }
        )

    if context.artifacts_created:
        sections.append(
            {
                "id": "artifact_refs",
                "source": "artifact_ledger",
                "artifacts": len(context.artifacts_created),
                "tokens": 0,
                "artifact_ids": [ref.artifact_id for ref in context.artifacts_created],
            }
        )

    return sections


def _message_section(
    section_id: str,
    source: str,
    messages: list[BaseMessage],
    context_manager: ConversationContextManager,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "source": source,
        "messages": len(messages),
        "tokens": context_manager._estimate_messages_tokens(messages),
    }
