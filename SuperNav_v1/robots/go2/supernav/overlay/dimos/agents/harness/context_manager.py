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

"""Conversation context assembly for DimOS harness agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.artifact_ledger import ArtifactLedger, ArtifactRef, default_artifact_dir
from dimos.agents.harness.image_blocks import (
    artifact_ids_from_content,
    image_block_token_estimate,
    image_block_token_estimate_details,
    is_image_block,
)
from dimos.agents.harness.image_placement import build_recent_image_placement


@dataclass(frozen=True)
class ContextAssembly:
    """A bounded request payload plus audit metadata."""

    messages: list[BaseMessage]
    total_history_messages: int
    included_messages: int
    estimated_tokens: int
    pinned_messages: int = 0
    token_budget: int | None = None
    budget_exhausted: bool = False
    failure_reason: str | None = None
    dropped: list[dict[str, Any]] = field(default_factory=list)
    artifacts_created: list[ArtifactRef] = field(default_factory=list)
    images_included: list[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("messages", None)
        return payload


@dataclass(frozen=True)
class _ImageInjection:
    anchor_index: int
    artifact_id: str
    message: HumanMessage
    metadata: dict[str, Any]
    pinned: bool = False


class ConversationContextManager:
    """Normalize rich history and assemble a bounded model request."""

    def __init__(
        self,
        *,
        artifact_ledger: ArtifactLedger | None = None,
        artifact_dir: str | Path | None = None,
        max_context_tokens: int | None = 64000,
        max_recent_messages: int = 80,
        max_tool_result_chars: int = 2000,
        max_image_blocks: int = 3,
        target_provider: str | None = None,
        target_model: str | None = None,
    ) -> None:
        self.artifact_ledger = artifact_ledger or ArtifactLedger(
            artifact_dir=artifact_dir or default_artifact_dir()
        )
        self.max_context_tokens = (
            max(1, int(max_context_tokens)) if max_context_tokens is not None else None
        )
        self.max_recent_messages = max(1, int(max_recent_messages))
        self.max_tool_result_chars = max(200, int(max_tool_result_chars))
        self.max_image_blocks = max(0, int(max_image_blocks))
        self.target_provider = target_provider
        self.target_model = target_model
        self._last_artifacts_created: list[ArtifactRef] = []

    def normalize_message(self, message: BaseMessage, *, source: str = "history") -> BaseMessage:
        """Return a transcript-safe copy of *message*.

        Image and other non-text blocks are moved into the artifact ledger and
        replaced by compact text placeholders in durable history. Request
        assembly can re-inject recent image artifacts as provider-visible image
        blocks without keeping raw base64 in every stored history message.
        """

        content = message.content
        message_type = str(getattr(message, "type", "message"))
        if isinstance(content, list):
            normalized_blocks: list[str | dict[str, Any]] = []
            changed = False
            for block in content:
                if isinstance(block, str):
                    normalized = self._truncate_if_needed(block, message_type)
                    normalized_blocks.append(normalized)
                    changed = changed or normalized != block
                    continue
                if not isinstance(block, dict):
                    normalized_blocks.append(str(block))
                    changed = True
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text") or "")
                    normalized = self._truncate_if_needed(text, message_type)
                    normalized_blocks.append({**block, "text": normalized})
                    changed = changed or normalized != text
                    continue

                ref = self.artifact_ledger.record_block(
                    block,
                    source=source,
                    message_type=message_type,
                )
                self._last_artifacts_created.append(ref)
                normalized_blocks.append({"type": "text", "text": ref.placeholder_text()})
                changed = True

            if changed:
                return message.model_copy(update={"content": normalized_blocks})
            return message

        if isinstance(content, str):
            normalized = self._truncate_if_needed(content, message_type)
            if normalized != content:
                return message.model_copy(update={"content": normalized})

        return message

    def normalize_messages(
        self, messages: list[BaseMessage], *, source: str = "history"
    ) -> list[BaseMessage]:
        return [self.normalize_message(message, source=source) for message in messages]

    def assemble(
        self,
        history: list[BaseMessage],
        *,
        pinned_messages: list[BaseMessage] | None = None,
    ) -> ContextAssembly:
        """Return the bounded request history and a traceable assembly summary."""

        self._last_artifacts_created = []
        pinned = self.normalize_messages(pinned_messages or [], source="pinned_context")
        normalized = self.normalize_messages(history, source="request_assembly")
        dropped: list[dict[str, Any]] = []
        if len(normalized) > self.max_recent_messages:
            drop_count = len(normalized) - self.max_recent_messages
            dropped.append(
                {
                    "id": "old_history",
                    "reason": "max_recent_messages",
                    "count": drop_count,
                }
            )
            normalized = normalized[-self.max_recent_messages :]

        normalized, token_drop_count = self._drop_old_messages_to_fit_token_budget(
            normalized,
            pinned_messages=pinned,
        )
        if token_drop_count:
            dropped.append(
                {
                    "id": "old_history",
                    "reason": "max_context_tokens",
                    "count": token_drop_count,
                }
            )

        assembly_base_messages = [*pinned, *normalized]
        image_injections = self._image_injections_for_messages(
            assembly_base_messages,
            pinned_message_count=len(pinned),
        )
        messages, images_included, image_drop_count = self._assemble_with_image_budget(
            assembly_base_messages,
            image_injections,
        )
        if image_drop_count:
            dropped.append(
                {
                    "id": "visual_evidence",
                    "reason": "image_budget_exceeded",
                    "count": image_drop_count,
                }
            )

        estimated_tokens = self._estimate_messages_tokens(messages)
        budget_exhausted = (
            self.max_context_tokens is not None and estimated_tokens > self.max_context_tokens
        )
        failure_reason = "single_turn_over_budget" if budget_exhausted else None
        if budget_exhausted:
            dropped.append(
                {
                    "id": "context_budget",
                    "reason": "budget_exhausted",
                    "estimated_tokens": estimated_tokens,
                    "token_budget": self.max_context_tokens,
                    "included_messages": len(normalized),
                }
            )

        return ContextAssembly(
            messages=messages,
            total_history_messages=len(history),
            included_messages=len(messages),
            estimated_tokens=estimated_tokens,
            pinned_messages=len(pinned),
            token_budget=self.max_context_tokens,
            budget_exhausted=budget_exhausted,
            failure_reason=failure_reason,
            dropped=dropped,
            artifacts_created=list(self._last_artifacts_created),
            images_included=images_included,
        )

    def _drop_old_messages_to_fit_token_budget(
        self,
        messages: list[BaseMessage],
        *,
        pinned_messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], int]:
        image_aware_trimmed = list(messages)
        image_aware_drop_count = 0
        while (
            self.max_context_tokens is not None
            and len(image_aware_trimmed) > 1
            and self._estimate_messages_tokens_with_selected_images(
                image_aware_trimmed,
                pinned_messages=pinned_messages,
            )
            > self.max_context_tokens
        ):
            image_aware_trimmed.pop(0)
            image_aware_drop_count += 1
        if (
            image_aware_drop_count
            and self.max_context_tokens is not None
            and self._estimate_messages_tokens_with_selected_images(
                image_aware_trimmed,
                pinned_messages=pinned_messages,
            )
            <= self.max_context_tokens
        ):
            return image_aware_trimmed, image_aware_drop_count

        # If the selected images cannot fit even after dropping history, leave
        # history intact here and let the image-budget pass compact images.
        trimmed = list(messages)
        drop_count = 0
        while (
            self.max_context_tokens is not None
            and len(trimmed) > 1
            and self._estimate_messages_tokens([*pinned_messages, *trimmed])
            > self.max_context_tokens
        ):
            trimmed.pop(0)
            drop_count += 1
        return trimmed, drop_count

    def _estimate_messages_tokens_with_selected_images(
        self,
        messages: list[BaseMessage],
        *,
        pinned_messages: list[BaseMessage],
    ) -> int:
        assembly_base_messages = [*pinned_messages, *messages]
        image_injections = self._image_injections_for_messages(
            assembly_base_messages,
            pinned_message_count=len(pinned_messages),
        )
        assembled, _ = self._interleave_image_injections(
            assembly_base_messages,
            image_injections,
        )
        return self._estimate_messages_tokens(assembled)

    def _image_injections_for_messages(
        self,
        messages: list[BaseMessage],
        *,
        pinned_message_count: int = 0,
    ) -> list[_ImageInjection]:
        if self.max_image_blocks <= 0:
            return []

        message_artifact_ids = [artifact_ids_from_content(message.content) for message in messages]
        pinned_image_ids = self._image_artifact_ids(message_artifact_ids[:pinned_message_count])
        recent_image_ids = self._recent_image_artifact_ids(
            message_artifact_ids[pinned_message_count:]
        )
        selected_image_ids = self._ordered_unique([*pinned_image_ids, *recent_image_ids])
        if not selected_image_ids:
            return []

        selected_set = set(selected_image_ids)
        pinned_set = set(pinned_image_ids)
        injections: list[_ImageInjection] = []
        included_once: set[str] = set()
        for anchor_index, artifact_ids in enumerate(message_artifact_ids):
            for artifact_id in artifact_ids:
                if artifact_id not in selected_set or artifact_id in included_once:
                    continue
                block = self.artifact_ledger.get_block(artifact_id)
                if not isinstance(block, dict) or not is_image_block(block):
                    continue
                ref = self.artifact_ledger.get_ref(artifact_id)
                pinned = artifact_id in pinned_set and anchor_index < pinned_message_count
                placement = build_recent_image_placement(
                    block=block,
                    artifact_id=artifact_id,
                    target_provider=self.target_provider,
                )
                token_estimate = image_block_token_estimate_details(
                    block,
                    target_provider=self.target_provider,
                    target_model=self.target_model,
                )
                injections.append(
                    _ImageInjection(
                        anchor_index=anchor_index,
                        artifact_id=artifact_id,
                        message=placement.message,
                        metadata={
                            **placement.metadata,
                            "artifact_id": artifact_id,
                            "anchor_message_index": anchor_index,
                            "mime_type": ref.mime_type if ref is not None else None,
                            "estimated_tokens": token_estimate.tokens,
                            "token_estimate": token_estimate.model_dump(),
                            "pinned": pinned,
                            "inclusion_reason": "pinned_image" if pinned else "recent_image",
                        },
                        pinned=pinned,
                    )
                )
                included_once.add(artifact_id)
        return injections

    def _assemble_with_image_budget(
        self,
        messages: list[BaseMessage],
        injections: list[_ImageInjection],
    ) -> tuple[list[BaseMessage], list[dict[str, Any]], int]:
        if not injections:
            return list(messages), [], 0

        selected = list(injections)
        image_drop_count = 0
        while selected:
            assembled, images_included = self._interleave_image_injections(
                messages,
                selected,
            )
            if (
                self.max_context_tokens is None
                or self._estimate_messages_tokens(assembled) <= self.max_context_tokens
            ):
                return assembled, images_included, image_drop_count
            selected.pop(self._next_image_drop_index(selected))
            image_drop_count += 1
        return list(messages), [], image_drop_count

    def _interleave_image_injections(
        self,
        messages: list[BaseMessage],
        injections: list[_ImageInjection],
        *,
        request_index_offset: int = 0,
    ) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
        by_anchor: dict[int, list[_ImageInjection]] = {}
        for injection in injections:
            by_anchor.setdefault(injection.anchor_index, []).append(injection)

        assembled: list[BaseMessage] = []
        images_included: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            assembled.append(message)
            for injection in by_anchor.get(index, []):
                request_message_index = request_index_offset + len(assembled)
                assembled.append(injection.message)
                images_included.append(
                    {
                        **injection.metadata,
                        "request_message_index": request_message_index,
                    }
                )
        return assembled, images_included

    def _recent_image_artifact_ids(self, message_artifact_ids: list[list[str]]) -> list[str]:
        return self._image_artifact_ids(message_artifact_ids)[-self.max_image_blocks :]

    def _image_artifact_ids(self, message_artifact_ids: list[list[str]]) -> list[str]:
        ordered_unique: list[str] = []
        seen: set[str] = set()
        for artifact_ids in message_artifact_ids:
            for artifact_id in artifact_ids:
                if artifact_id in seen:
                    continue
                block = self.artifact_ledger.get_block(artifact_id)
                if isinstance(block, dict) and is_image_block(block):
                    ordered_unique.append(artifact_id)
                    seen.add(artifact_id)
        return ordered_unique

    def _ordered_unique(self, artifact_ids: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for artifact_id in artifact_ids:
            if artifact_id in seen:
                continue
            ordered.append(artifact_id)
            seen.add(artifact_id)
        return ordered

    def _next_image_drop_index(self, selected: list[_ImageInjection]) -> int:
        for index, injection in enumerate(selected):
            if not injection.pinned:
                return index
        return 0

    def _truncate_if_needed(self, text: str, message_type: str) -> str:
        if not _is_tool_like_message(message_type, text):
            return text
        if len(text) <= self.max_tool_result_chars:
            return text
        return (
            text[: self.max_tool_result_chars]
            + f"... [truncated {len(text) - self.max_tool_result_chars} chars]"
        )

    def _estimate_messages_tokens(self, messages: list[BaseMessage]) -> int:
        return _estimate_messages_tokens(
            messages,
            target_provider=self.target_provider,
            target_model=self.target_model,
        )


def _is_tool_like_message(message_type: str, text: str) -> bool:
    if message_type in {"tool", "function"}:
        return True
    return text.startswith("[tool:")


def _estimate_messages_tokens(
    messages: list[BaseMessage],
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
) -> int:
    return sum(
        _estimate_content_tokens(
            message.content,
            target_provider=target_provider,
            target_model=target_model,
        )
        for message in messages
    )


def _estimate_content_tokens(
    content: Any,
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
) -> int:
    if isinstance(content, list):
        return sum(
            _estimate_content_tokens(
                item,
                target_provider=target_provider,
                target_model=target_model,
            )
            for item in content
        )
    if isinstance(content, dict) and is_image_block(content):
        return image_block_token_estimate(
            content,
            target_provider=target_provider,
            target_model=target_model,
        )
    chars = _content_char_count(content)
    return max(1, (chars + 3) // 4)


def _content_char_count(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_char_count(item) for item in content)
    if isinstance(content, dict):
        if is_image_block(content):
            # Image token pricing is handled by `_estimate_content_tokens`.
            return 0
        if content.get("type") == "text":
            return len(str(content.get("text") or ""))
        return len(str(content))
    return len(str(content))
