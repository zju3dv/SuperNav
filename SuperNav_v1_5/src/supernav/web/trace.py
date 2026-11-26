"""Bounded, incremental display of already-emitted agent messages and tool events.

This reader never asks a model for reasoning, decodes encrypted reasoning, or
modifies native/canonical evidence. It consumes only complete JSONL records.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from supernav.runtime.streams import _clean_tool_name, _text_from_content, _reasoning_summary_text


def _invalid_constant(value):
    raise ValueError(value)


def _fingerprint(name, arguments):
    return hashlib.sha256(json.dumps([name, display_value(arguments)], sort_keys=True).encode()).hexdigest()


def _seconds(value):
    try:
        if isinstance(value, (int, float)):
            return value / 1000 if value > 1e12 else value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None


def display_value(value, depth=0):
    if depth > 8:
        return "[nested content omitted]"
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                return display_value(json.loads(value, parse_constant=_invalid_constant), depth + 1)
            except ValueError:
                pass
        return re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[redacted]", value)[:12000]
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url", "input_image"}:
            return {"type": "image", "note": "Image shown in the observation panels"}
        return {k: ("[redacted]" if re.search(r"(?i)api.?key|authorization|password|secret|token|encrypted_content", k)
                    else display_value(v, depth + 1)) for k, v in list(value.items())[:100]
                if k not in {"data", "b64_json"}}
    if isinstance(value, list):
        return [display_value(v, depth + 1) for v in value[:40]]
    return value


def _failed(result):
    if isinstance(result, dict):
        return bool(result.get("error") or result.get("isError") or result.get("is_error") or result.get("ok") is False
                    or _failed(result.get("content")) or _failed(result.get("text")))
    if isinstance(result, list):
        return any(_failed(block) for block in result)
    return False


class TraceReader:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.offsets = {}
        self.items = OrderedDict()
        self.pending = {}
        self.mode = None
        self.primary_source = None
        self.timings = OrderedDict()
        self.revision = 0

    def _put(self, key, row):
        row["id"] = key
        self.items[key] = row
        while len(self.items) > 300:
            removed, _ = self.items.popitem(last=False)
            for name in list(self.pending):
                self.pending[name] = [key for key in self.pending[name] if key != removed]
                if not self.pending[name]:
                    del self.pending[name]
        self.revision += 1

    def _consume_timing(self, d, source, line):
        payload = d.get("payload") or {}
        typ, payload_type = d.get("type"), payload.get("type")
        stamp = d.get("timestamp")
        key = f"{source}:{line}"
        if typ == "response_item" and payload_type == "reasoning" and self.mode == "native":
            summary = _reasoning_summary_text(payload.get("summary"))
            if summary:
                self._put(key, {"source": source, "line": line, "timestamp": stamp,
                               "kind": "thought", "channel": "reasoning summary", "text": summary[:12000]})
        timing = None
        item = payload.get("item") or {}
        if typ == "event_msg" and item.get("type") == "McpToolCall":
            timing = {"fingerprint": _fingerprint(_clean_tool_name(item.get("tool", "")), item.get("arguments", {})), "timestamp": stamp}
        elif typ == "event_msg" and payload_type == "mcp_tool_call_end":
            invocation = payload.get("invocation") or {}
            timing = {"fingerprint": _fingerprint(_clean_tool_name(invocation.get("tool", "")), invocation.get("arguments", {})), "timestamp": stamp}
        elif typ == "response_item" and payload_type == "message" and payload.get("role") == "assistant":
            timing = {"text": _text_from_content(payload.get("content"))[:12000], "timestamp": stamp}
        if timing and stamp:
            self.timings[key] = {**timing, "source": source, "line": line}
            while len(self.timings) > 600:
                self.timings.popitem(last=False)

    def _consume(self, d, source, line):
        origin = {"source": source, "line": line, "timestamp": d.get("timestamp") or d.get("time_start") or d.get("time")}
        typ = d.get("type", "")
        key = f"{source}:{line}"
        if typ in {"assistant_text", "assistant_reasoning", "assistant_thinking"}:
            text = d.get("text", "")
            if text and text != "reasoning (encrypted)":
                self._put(key, {**origin, "kind": "thought", "channel": "agent message" if typ == "assistant_text" else "recorded reasoning", "text": str(text)[:12000]})
            return
        if typ == "tool_call":
            name = _clean_tool_name(str(d.get("name", "unknown")))
            self.pending.setdefault(name, []).append(key)
            self._put(key, {**origin, "kind": "tool", "name": name, "arguments": display_value(d.get("input", {})), "status": "running"})
            return
        if typ == "tool_result":
            name = _clean_tool_name(str(d.get("name", "unknown")))
            pending = self.pending.get(name, [])
            call_key = pending.pop(0) if pending else key
            row = dict(self.items.get(call_key, {**origin, "kind": "tool", "name": name, "arguments": {}}))
            result = display_value(d.get("content"))
            row.update(result=result, status="error" if _failed(result) else "completed", result_line=line,
                       completed_at=d.get("time_end") or d.get("timestamp"), audit_tool_seq=d.get("audit_tool_seq"))
            audit = d.get("audit") or {}
            anchor = audit.get("selected_anchor") or audit.get("original_annotation") or {}
            if anchor.get("image_ref"):
                row["image_ref"] = anchor["image_ref"]
            self._put(call_key, row)
            return
        item = d.get("item") if isinstance(d.get("item"), dict) else {}
        if item:
            item_type = item.get("type")
            item_key = f"{source}:item:{item.get('id', line)}"
            if item_type in {"agent_message", "reasoning", "thinking"} and item.get("text"):
                self._put(item_key, {**origin, "kind": "thought", "channel": "agent message" if item_type == "agent_message" else "recorded reasoning", "text": str(item["text"])[:12000]})
            elif item_type in {"mcp_tool_call", "command_execution"}:
                result = item.get("result") or item.get("aggregated_output") or item.get("error")
                status = "running" if typ in {"item.started", "item.updated"} else "completed"
                if item.get("error") or _failed(display_value(result)) or item.get("status") == "failed" or item.get("exit_code") not in (None, 0):
                    status = "error"
                previous = self.items.get(item_key, {})
                self._put(item_key, {**origin, "line": previous.get("line", line), "result_line": line,
                    "kind": "tool", "name": _clean_tool_name(str(item.get("tool", "shell"))),
                    "arguments": display_value(item.get("arguments") or {"command": item.get("command")}),
                    "status": status, "result": display_value(result)})
            return
        # OpenCode native JSONL.
        part = d.get("part") if isinstance(d.get("part"), dict) else {}
        if typ == "tool_use" or part.get("type") == "tool":
            state = part.get("state") or {}
            tool_key = f"{source}:tool:{part.get('callID') or part.get('id') or line}"
            self._put(tool_key, {**origin, "kind": "tool", "name": _clean_tool_name(str(part.get("tool", "unknown"))),
                "arguments": display_value(state.get("input", {})), "result": display_value(state.get("output") or state.get("error")),
                "status": state.get("status", "running")})
            return
        if typ in {"text", "thinking", "reasoning"} or part.get("type") in {"text", "thinking", "reasoning"}:
            text = part.get("text") or part.get("thinking") or part.get("reasoning") or d.get("text") or d.get("thinking") or d.get("reasoning")
            if text:
                self._put(key, {**origin, "kind": "thought", "channel": "agent message" if typ == "text" else "recorded reasoning", "text": str(text)[:12000]})
            return
        # Kimi stream-json assistant/tool messages and native wire events.
        if d.get("role") == "assistant":
            for index, block in enumerate(d.get("content", []) if isinstance(d.get("content"), list) else [{"text": d.get("content")} ]):
                if isinstance(block, dict):
                    text = block.get("think") or block.get("thinking") or block.get("text")
                    if text:
                        self._put(f"{key}:{index}", {**origin, "kind": "thought", "channel": "recorded reasoning" if block.get("think") or block.get("thinking") else "agent message", "text": str(text)[:12000]})
            for call in d.get("tool_calls", []):
                function = call.get("function") or {}
                tool_key = f"{source}:tool:{call.get('id', line)}"
                self._put(tool_key, {**origin, "kind": "tool", "name": _clean_tool_name(str(function.get("name", "unknown"))), "arguments": display_value(function.get("arguments", {})), "status": "running"})
        elif d.get("role") == "tool":
            tool_key = f"{source}:tool:{d.get('tool_call_id', line)}"
            row = dict(self.items.get(tool_key, {**origin, "kind": "tool", "name": "unknown", "arguments": {}}))
            result = display_value(d.get("content"))
            row.update(result=result, status="error" if _failed(result) else "completed", result_line=line)
            self._put(tool_key, row)
        elif typ == "context.append_loop_event":
            event = d.get("event") or {}
            event_type = event.get("type")
            call_id = event.get("toolCallId") or event.get("call_id") or event.get("uuid") or line
            tool_key = f"{source}:tool:{call_id}"
            if event_type == "tool.call":
                self._put(tool_key, {**origin, "kind": "tool", "name": _clean_tool_name(str(event.get("name") or event.get("tool") or "unknown")),
                    "arguments": display_value(event.get("args") or event.get("arguments") or {}), "status": "running"})
            elif event_type == "tool.result":
                row = dict(self.items.get(tool_key, {**origin, "kind": "tool", "name": "unknown", "arguments": {}}))
                result = display_value(event.get("result"))
                row.update(result=result, status="error" if _failed(result) else "completed", result_line=line, completed_at=origin["timestamp"])
                self._put(tool_key, row)
            elif event_type == "content.part":
                block = event.get("part") or {}
                for field in ("think", "thinking", "text"):
                    if block.get(field):
                        self._put(f"{key}:{field}", {**origin, "kind": "thought", "channel": "agent message" if field == "text" else "recorded reasoning", "text": str(block[field])[:12000]})

    def _read(self, path, *, timing=False):
        if not path.resolve().is_relative_to(self.run_dir) or not path.is_file():
            return
        source = str(path.relative_to(self.run_dir))
        stat = path.stat()
        offset, line, inode, skipping = self.offsets.get(source, (0, 0, stat.st_ino, False))
        if inode != stat.st_ino or stat.st_size < offset:
            offset, line, skipping = 0, 0, False
            self.items = OrderedDict((k, v) for k, v in self.items.items() if v["source"] != source)
            self.timings = OrderedDict((k, v) for k, v in self.timings.items() if v["source"] != source)
            self.pending.clear()
        consumed = 0
        with path.open("rb") as stream:
            stream.seek(offset)
            while consumed < 8 * 1024 * 1024:
                raw = stream.readline(8 * 1024 * 1024)
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    if len(raw) == 8 * 1024 * 1024 or skipping:
                        skipping = True
                        offset += len(raw)
                    break
                consumed += len(raw); offset += len(raw); line += 1
                if skipping:
                    skipping = False
                    continue
                try:
                    row = json.loads(raw, parse_constant=_invalid_constant)
                    if isinstance(row, dict):
                        if timing:
                            self._consume_timing(row, source, line)
                        else:
                            self._consume(row, source, line)
                except (ValueError, TypeError, AttributeError):
                    continue
        self.offsets[source] = (offset, line, stat.st_ino, skipping)

    def _events(self):
        rows = [dict(row) for row in self.items.values()]
        # Native Codex stdout has no clock. Match exact messages / tool arguments
        # to the isolated session's timestamped events, without changing evidence.
        groups = {}
        for row in rows:
            if row.get("status") == "running":
                continue
            signature = _fingerprint(row["name"], row["arguments"]) if row["kind"] == "tool" else row.get("text")
            groups.setdefault(signature, []).append(row)
        for signature, matches in groups.items():
            times = [t for t in self.timings.values() if t.get("fingerprint", t.get("text")) == signature]
            for row, timing in zip(reversed(matches), reversed(times)):
                if not row.get("timestamp"):
                    row["timestamp"] = timing["timestamp"]
                    row["timestamp_kind"] = "returned" if row["kind"] == "tool" else "emitted"
                    row["timing_source"] = f"{timing['source']}:{timing['line']}"
        # Some canonicalizers append reasoning after the stdout events. Restore
        # chronological display where native timestamps exist. Untimed rows keep
        # their source order between neighbouring timed rows; no clock is invented.
        primary = [r for r in rows if r.get("channel") not in {"recorded reasoning", "reasoning summary"} or _seconds(r.get("timestamp")) is None]
        thoughts = [r for r in rows if r not in primary]
        latest = max((_seconds(r.get("timestamp")) or 0 for r in rows), default=0)
        ordering = []
        for index, row in enumerate(primary):
            moment = _seconds(row.get("timestamp"))
            if moment is None:
                following = next(((_seconds(r.get("timestamp")), i) for i, r in enumerate(primary[index+1:], index+1)
                                  if _seconds(r.get("timestamp")) is not None), None)
                moment = following[0] - (following[1] - index) * 1e-6 if following else latest + (index + 1) * 1e-6
            ordering.append((moment, row))
        ordering.extend((_seconds(r.get("timestamp")) or latest, r) for r in thoughts)
        return [row for _, row in sorted(ordering, key=lambda pair: pair[0])]

    def snapshot(self):
        canonical = self.run_dir / "canonical.jsonl"
        mode = "canonical" if canonical.is_file() else "native"
        primary = canonical if mode == "canonical" else self.run_dir / "raw.jsonl"
        if mode == "native":
            wire = self.run_dir / "kimi_wire.jsonl"
            sessions = self.run_dir / "kimi_project/kimi_home/.kimi-code/sessions"
            if wire.is_file():
                primary = wire
            elif sessions.is_dir() and sessions.resolve().is_relative_to(self.run_dir):
                # Only this run's main agent; never traverse credentials or other homes.
                candidates = [p for p in sessions.glob("*/*/agents/main/wire.jsonl")
                              if p.resolve().is_relative_to(self.run_dir)]
                if candidates:
                    primary = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        if mode != self.mode or primary != self.primary_source:
            self.items.clear(); self.offsets.clear(); self.pending.clear(); self.timings.clear()
            self.mode = mode
            self.primary_source = primary
        paths = [primary]
        session_copy = self.run_dir / "codex_session.jsonl"
        if session_copy.is_file():
            paths.append(session_copy)
        else:
            # This is the isolated run's session directory, never the user's global home.
            sessions = self.run_dir / "codex_project/.codex_home/sessions"
            if sessions.is_dir() and sessions.resolve().is_relative_to(self.run_dir):
                paths.extend(sorted(sessions.glob("*/*/*/*.jsonl"))[-8:])
        for index, path in enumerate(paths):
            try:
                self._read(path, timing=index > 0)
            except OSError:
                continue
        return {"status": "available" if self.items else "waiting", "revision": self.revision,
                "events": self._events(), "limit": 300,
                "note": "Only messages and reasoning emitted by the agent are shown."}
