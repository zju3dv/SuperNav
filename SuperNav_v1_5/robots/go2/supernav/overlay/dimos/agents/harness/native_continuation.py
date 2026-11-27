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

"""Continuation dispatch helpers for native DimOS harness tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from dimos.agents.harness.native_internal_message import native_internal_message

ToolCaller = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class NativeContinuationResult:
    """Result of resolving and dispatching a native tool continuation."""

    message_text: str
    executed: bool
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> HumanMessage:
        """Return the model-visible continuation event without user semantics."""

        return native_internal_message(self.message_text, source="continuation")


def dispatch_native_continuation(
    continuation: Mapping[str, Any],
    continuation_context: Mapping[str, Any],
    *,
    tool_registry: Mapping[str, Mapping[str, Any]],
    call_tool: ToolCaller,
) -> NativeContinuationResult:
    """Resolve a trigger-tool continuation and return the queue message text."""

    tool_name = continuation.get("tool")
    if not tool_name:
        return NativeContinuationResult(
            message_text=f"Continuation failed: missing 'tool' key in {dict(continuation)}",
            executed=False,
        )

    tool_name = str(tool_name)
    if tool_name not in tool_registry:
        return NativeContinuationResult(
            message_text=f"Continuation failed: tool '{tool_name}' not found",
            executed=False,
            tool_name=tool_name,
        )

    tool_args = _resolved_tool_args(continuation.get("args", {}), continuation_context)
    try:
        result = call_tool(tool_name, tool_args)
    except Exception as exc:
        return NativeContinuationResult(
            message_text=f"Continuation '{tool_name}' failed with error: {exc}",
            executed=False,
            tool_name=tool_name,
            tool_args=tool_args,
        )

    label = str(continuation_context.get("label", "unknown"))
    text = _text_content(result)
    return NativeContinuationResult(
        message_text=(
            f"Automatically executed '{tool_name}' as a continuation of lookout "
            f"detection (detected: {label}). Result: {text or 'started'}"
        ),
        executed=True,
        tool_name=tool_name,
        tool_args=tool_args,
    )


def _resolved_tool_args(
    raw_args: Any,
    continuation_context: Mapping[str, Any],
) -> dict[str, Any]:
    tool_args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    for key, value in tool_args.items():
        if isinstance(value, str) and value.startswith("$"):
            context_key = value[1:]
            if context_key in continuation_context:
                tool_args[key] = continuation_context[context_key]
    return tool_args


def _text_content(result: Mapping[str, Any]) -> str:
    content = result.get("content", [])
    if not isinstance(content, list):
        return ""
    parts = [
        str(item.get("text", ""))
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    return "\n".join(parts)
