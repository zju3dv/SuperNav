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

"""Bounded log tailing for operator-console agent runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

# Marker prepended to reasoning-summary events so the renderer can tell the
# agent's private "thinking" apart from its spoken agent_message narration
# (both otherwise arrive on the same "codex" source with bare text).
THINKING_PREFIX = "[thinking] "


@dataclass(frozen=True)
class LogStreamEvent:
    """One line read from a run log file."""

    source: str
    text: str


class LogStreamService:
    """Tail Codex/DimOS logs into bounded UI events without rendering decisions."""

    def __init__(
        self,
        paths: dict[str, Path] | None = None,
        *,
        max_events: int = 200,
    ) -> None:
        self._paths = dict(paths or {})
        self._offsets = {source: 0 for source in self._paths}
        self._events: deque[LogStreamEvent] = deque(maxlen=max_events)

    def set_paths(self, paths: dict[str, Path]) -> None:
        """Replace tailed paths while preserving offsets for unchanged sources."""

        self._paths = dict(paths)
        self._offsets = {source: self._offsets.get(source, 0) for source in self._paths}

    def poll(self) -> tuple[LogStreamEvent, ...]:
        """Read newly appended lines and return the bounded event snapshot."""

        for source, path in self._paths.items():
            self._poll_path(source, path)
        return tuple(self._events)

    def _poll_path(self, source: str, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        offset = self._offsets.get(source, 0)
        with path.open("r", encoding="utf-8", errors="replace") as file:
            file.seek(offset)
            for line in file:
                stripped = line.rstrip("\n")
                if stripped:
                    text = _format_codex_event(stripped) if source == "codex" else stripped
                    if text is not None:
                        self._events.append(LogStreamEvent(source=source, text=text))
            self._offsets[source] = file.tell()


def _format_codex_event(raw: str) -> str | None:
    try:
        event: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(event, dict):
        return raw

    event_type = str(event.get("type") or "")
    if event_type in {"thread.started", "turn.started", "turn.completed"}:
        return None
    item = event.get("item")
    if isinstance(item, dict):
        item_type = str(item.get("type") or "item")
        if item_type == "reasoning":
            summary = _reasoning_summary_text(item)
            return f"{THINKING_PREFIX}{summary}" if summary else None
        if item_type == "agent_message" and item.get("text"):
            return str(item["text"])
        if item_type == "mcp_tool_call":
            target = ".".join(
                str(value) for value in (item.get("server"), item.get("tool")) if value
            )
            if event_type == "item.started":
                return f"calling {target or 'MCP tool'}"
            if item.get("error"):
                return f"{target or 'MCP tool'} failed: {item['error']}"
            return f"{target or 'MCP tool'} {item.get('status') or 'completed'}"
        if item_type == "command_execution":
            status = (
                "started" if event_type == "item.started" else item.get("status") or "completed"
            )
            command = " ".join(str(item.get("command") or "").split())
            return f"command {status}: {command[:120]}".rstrip(": ")
        return f"{item_type.replace('_', ' ')} {item.get('status') or 'updated'}"

    if event.get("message"):
        return str(event["message"])
    return raw


def _reasoning_summary_text(item: dict[str, Any]) -> str | None:
    """Pull the reasoning-summary text codex exposes on a reasoning item.

    Codex's ``exec --json`` stream may carry the summary either as a flat
    ``text`` field or as a ``summary`` list of ``{type, text}`` parts; handle
    both so the console shows thinking regardless of which shape arrives.
    """

    text = item.get("text")
    if text:
        return str(text).strip() or None
    summary = item.get("summary")
    if isinstance(summary, list):
        parts = [
            str(part.get("text")).strip()
            for part in summary
            if isinstance(part, dict) and part.get("text")
        ]
        joined = "\n".join(part for part in parts if part)
        if joined:
            return joined
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None
