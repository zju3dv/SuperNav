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

"""Typed status messages for asynchronous tool streams."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.tool_status_context import tool_status_context_from_meta
from dimos.agents.mcp import tool_stream

ToolStatusKind = Literal["progress", "message"]


class ToolStatusMessage(BaseMessage):
    """A UI/trace status event that must not be treated as user input."""

    type: Literal["dimos_tool_status"] = "dimos_tool_status"
    tool_name: str
    event_kind: ToolStatusKind
    progress_token: str | None = None
    progress: int | float | None = None
    status_context: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


def tool_status_from_stream_message(msg: dict[str, Any]) -> ToolStatusMessage | None:
    """Parse a tool-stream JSON-RPC notification into a typed status message."""

    method = msg.get("method")
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        return None

    if method == tool_stream.NOTIFICATIONS_PROGRESS_METHOD:
        text = params.get("message") or ""
        meta = params.get("_meta") or {}
        tool_name = meta.get("tool_name") or "tool" if isinstance(meta, dict) else "tool"
        progress = params.get("progress")
        return _make_status(
            text=text,
            tool_name=str(tool_name),
            event_kind="progress",
            progress_token=_optional_str(params.get("progressToken")),
            progress=progress if isinstance(progress, int | float) else None,
            status_context=tool_status_context_from_meta(meta),
            raw=msg,
        )

    if method == tool_stream.NOTIFICATIONS_MESSAGE_METHOD:
        text = params.get("data") or ""
        meta = params.get("_meta") or {}
        return _make_status(
            text=text,
            tool_name=str(params.get("logger") or "tool"),
            event_kind="message",
            status_context=tool_status_context_from_meta(meta),
            raw=msg,
        )

    return None


def _make_status(
    *,
    text: object,
    tool_name: str,
    event_kind: ToolStatusKind,
    progress_token: str | None = None,
    progress: int | float | None = None,
    status_context: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> ToolStatusMessage | None:
    content = str(text or "")
    if not content:
        return None
    return ToolStatusMessage(
        content=content,
        tool_name=tool_name,
        event_kind=event_kind,
        progress_token=progress_token,
        progress=progress,
        status_context=status_context,
        raw=raw,
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None
