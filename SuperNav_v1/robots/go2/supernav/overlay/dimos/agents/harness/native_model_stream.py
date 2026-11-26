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

"""LangGraph model-stream handling for the native DimOS agent."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.image_blocks import artifact_ids_from_content
from dimos.agents.harness.native_agent_io import NativeAgentIO
from dimos.agents.harness.native_task_terminal import (
    NativeTaskTerminalDirective,
    terminal_directive_from_message,
)


class NativeStateGraph(Protocol):
    """Subset of LangGraph's compiled graph used by the native agent."""

    def stream(self, state: dict[str, Any], stream_mode: str) -> Iterable[Mapping[str, Any]]: ...


class ModelUpdateContextManager(Protocol):
    """Context-manager subset needed for model update normalization."""

    def normalize_message(self, message: BaseMessage, *, source: str) -> BaseMessage: ...


@dataclass(frozen=True)
class NativeModelStreamResult:
    """Compact summary of one native model stream."""

    update_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    terminal_directives: list[NativeTaskTerminalDirective] = field(default_factory=list)

    @property
    def tool_call_count(self) -> int:
        """Total tool calls requested by the streamed model messages."""

        return sum(len(message.get("tool_calls") or []) for message in self.messages)

    @property
    def tool_call_names(self) -> list[str]:
        """Ordered unique tool names requested by the streamed model messages."""

        names: list[str] = []
        seen: set[str] = set()
        for message in self.messages:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                name = call.get("name")
                if not isinstance(name, str) or not name or name in seen:
                    continue
                names.append(name)
                seen.add(name)
        return names

    @property
    def latest_terminal_directive(self) -> NativeTaskTerminalDirective | None:
        """Return the latest strict task-terminal directive emitted by the model."""

        if not self.terminal_directives:
            return None
        return self.terminal_directives[-1]


def publish_native_model_stream(
    state_graph: NativeStateGraph,
    *,
    request_messages: list[BaseMessage],
    context_manager: ModelUpdateContextManager,
    history: list[BaseMessage],
    agent_io: NativeAgentIO,
) -> NativeModelStreamResult:
    """Stream model updates, normalize them, append history, and publish them."""

    n_updates = 0
    summaries: list[dict[str, Any]] = []
    artifact_ids: list[str] = []
    artifact_ids_seen: set[str] = set()
    terminal_directives: list[NativeTaskTerminalDirective] = []
    for update in state_graph.stream({"messages": request_messages}, stream_mode="updates"):
        for node_output in update.values():
            if not isinstance(node_output, Mapping):
                continue
            messages = node_output.get("messages", [])
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, BaseMessage):
                    continue
                normalized_msg = context_manager.normalize_message(msg, source="model_update")
                terminal_directive = terminal_directive_from_message(
                    normalized_msg,
                    message_index=n_updates,
                )
                if terminal_directive is not None:
                    terminal_directives.append(terminal_directive)
                message_artifact_ids = artifact_ids_from_content(normalized_msg.content)
                for artifact_id in message_artifact_ids:
                    if artifact_id in artifact_ids_seen:
                        continue
                    artifact_ids.append(artifact_id)
                    artifact_ids_seen.add(artifact_id)
                history.append(normalized_msg)
                agent_io.publish_model_update(normalized_msg)
                summaries.append(
                    _summarize_model_message(
                        normalized_msg,
                        message_index=n_updates,
                        artifact_ids=message_artifact_ids,
                    )
                )
                n_updates += 1
    return NativeModelStreamResult(
        update_count=n_updates,
        messages=summaries,
        artifact_ids=artifact_ids,
        terminal_directives=terminal_directives,
    )


def _summarize_model_message(
    message: BaseMessage,
    *,
    message_index: int,
    artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact trace-safe summary of a model message."""

    content_summary = _summarize_content(getattr(message, "content", ""))
    response_metadata = getattr(message, "response_metadata", None)
    finish_reason = None
    model_name = None
    if isinstance(response_metadata, Mapping):
        finish_reason = response_metadata.get("finish_reason") or response_metadata.get(
            "stop_reason"
        )
        model_name = response_metadata.get("model_name") or response_metadata.get("model")

    summary: dict[str, Any] = {
        "message_index": message_index,
        "message_type": str(getattr(message, "type", message.__class__.__name__)),
        **content_summary,
        "tool_calls": _summarize_tool_calls(getattr(message, "tool_calls", [])),
    }
    if isinstance(finish_reason, str):
        summary["finish_reason"] = finish_reason
    if isinstance(model_name, str):
        summary["model_name"] = model_name
    if artifact_ids:
        summary["artifact_ids"] = artifact_ids
    return summary


def _summarize_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {
            "content_kind": "text",
            "text_chars": len(content),
            "text_preview": _safe_text_preview(content),
            "block_count": 0,
            "text_blocks": 1 if content else 0,
            "image_blocks": 0,
        }
    if isinstance(content, list):
        text_parts: list[str] = []
        image_blocks = 0
        text_blocks = 0
        for block in content:
            if not isinstance(block, Mapping):
                continue
            block_type = str(block.get("type") or "")
            if _is_image_block(block):
                image_blocks += 1
                continue
            text = block.get("text") if block_type in {"text", "input_text"} else None
            if isinstance(text, str):
                text_blocks += 1
                text_parts.append(text)
        text = "\n".join(text_parts)
        return {
            "content_kind": "blocks",
            "text_chars": len(text),
            "text_preview": _safe_text_preview(text),
            "block_count": len(content),
            "text_blocks": text_blocks,
            "image_blocks": image_blocks,
        }
    return {
        "content_kind": content.__class__.__name__,
        "text_chars": 0,
        "text_preview": "",
        "block_count": 0,
        "text_blocks": 0,
        "image_blocks": 0,
    }


def _is_image_block(block: Mapping[str, Any]) -> bool:
    block_type = str(block.get("type") or "")
    if block_type in {"image", "image_url", "input_image"}:
        return True
    return any(key in block for key in ("image_url", "source", "data"))


def _summarize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    summaries: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, Mapping):
            args = call.get("args")
            summaries.append(
                {
                    "index": index,
                    "name": call.get("name"),
                    "id": call.get("id"),
                    "arg_keys": sorted(args.keys()) if isinstance(args, Mapping) else [],
                }
            )
            continue

        name = getattr(call, "name", None)
        call_id = getattr(call, "id", None)
        args = getattr(call, "args", None)
        summaries.append(
            {
                "index": index,
                "name": name,
                "id": call_id,
                "arg_keys": sorted(args.keys()) if isinstance(args, Mapping) else [],
            }
        )
    return summaries


def _safe_text_preview(text: str, *, max_chars: int = 500) -> str:
    if "base64," in text or "data:image" in text:
        return "[redacted multimodal payload]"
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."
