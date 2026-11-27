"""Parse run-local agent session logs into common observability records."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

RecordKind = Literal["llm_call", "frame"]
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _global_session_roots() -> tuple[Path, ...]:
    home = Path.home().resolve(strict=False)
    return (home / ".codex", home / ".kimi-code", home / ".claude")


@dataclass(frozen=True)
class ObservabilityTokens:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None
    ) -> "ObservabilityTokens | None":
        if not payload:
            return None
        values = {
            "input_tokens": _int_or_none(payload.get("input_tokens")),
            "cached_input_tokens": _int_or_none(payload.get("cached_input_tokens")),
            "output_tokens": _int_or_none(payload.get("output_tokens")),
            "reasoning_output_tokens": _int_or_none(
                payload.get("reasoning_output_tokens")
            ),
            "total_tokens": _int_or_none(payload.get("total_tokens")),
        }
        if values["total_tokens"] is None:
            input_tokens = values["input_tokens"] or 0
            output_tokens = values["output_tokens"] or 0
            if input_tokens or output_tokens:
                values["total_tokens"] = input_tokens + output_tokens
        if not any(value is not None for value in values.values()):
            return None
        return cls(**values)

    def to_json(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ObservabilityToolCall:
    name: str
    args: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "args": _json_safe(self.args)}


@dataclass(frozen=True)
class ObservabilityRecord:
    kind: RecordKind
    seq: int
    call_id: str
    reasoning: str = ""
    tool_calls: tuple[ObservabilityToolCall, ...] = ()
    tokens: ObservabilityTokens | None = None
    image_path: str | None = None
    pose: Mapping[str, Any] | None = None
    action: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "llm_call":
            return
        if not self.image_path:
            raise ValueError("frame record requires image_path")
        if self.action is None:
            raise ValueError("frame record requires action")
        if self.pose is None:
            object.__setattr__(self, "pose", {})

    def to_json(self) -> dict[str, Any]:
        if self.kind == "llm_call":
            return {
                "kind": "llm_call",
                "seq": self.seq,
                "call_id": self.call_id,
                "reasoning": self.reasoning,
                "tool_calls": [tool.to_json() for tool in self.tool_calls],
                "tokens": self.tokens.to_json() if self.tokens else None,
            }
        return {
            "kind": "frame",
            "seq": self.seq,
            "call_id": self.call_id,
            "image_path": self.image_path,
            "pose": _json_safe(dict(self.pose or {})),
            "action": self.action,
        }


class HarnessTraceAdapter(Protocol):
    def parse(self, source: Path | str) -> tuple[ObservabilityRecord, ...]:
        """Parse an explicit run-local/export source into common records."""


@dataclass
class _FrameDraft:
    image_path: str
    pose: Mapping[str, Any]
    action: str


@dataclass
class _CallDraft:
    call_id: str
    reasoning_parts: list[str]
    tool_calls: list[ObservabilityToolCall]
    frame_drafts: list[_FrameDraft]
    tokens: ObservabilityTokens | None = None

    def has_content(self) -> bool:
        return bool(
            self.reasoning_parts or self.tool_calls or self.frame_drafts or self.tokens
        )


@dataclass(frozen=True)
class _RawJsonlEntry:
    source_path: str
    line_no: int
    row: Mapping[str, Any] | None = None
    raw_line: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": self.source_path,
            "line_no": self.line_no,
        }
        if self.raw_line is not None:
            payload["raw_line"] = self.raw_line
        if self.row is not None:
            payload["row"] = _json_safe(self.row)
        else:
            payload.setdefault("raw_line", "")
        return payload


def parse_codex_session_trace(source: Path | str) -> tuple[ObservabilityRecord, ...]:
    return CodexSessionTraceAdapter().parse(source)


def parse_kimi_session_export(source: Path | str) -> tuple[ObservabilityRecord, ...]:
    return KimiSessionTraceAdapter().parse(source)


def parse_codex_raw_llm_calls(source: Path | str) -> tuple[dict[str, Any], ...]:
    """Preserve Codex run-local native session rows as TaskM raw-call rows."""

    return CodexSessionTraceAdapter().parse_raw_llm_calls(source)


def parse_kimi_raw_llm_calls(source: Path | str) -> tuple[dict[str, Any], ...]:
    """Preserve Kimi exported wire rows as TaskM raw-call rows."""

    return KimiSessionTraceAdapter().parse_raw_llm_calls(source)


def write_observability_jsonl(
    records: Sequence[ObservabilityRecord], path: Path | str
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(record.to_json(), sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return output


def write_raw_llm_calls_jsonl(
    rows: Sequence[Mapping[str, Any]], path: Path | str
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


class CodexSessionTraceAdapter:
    """Adapter for Codex run-local session rollout JSONL."""

    def parse(self, source: Path | str) -> tuple[ObservabilityRecord, ...]:
        source_path = Path(source)
        _reject_global_session_source(source_path, surface="Codex")
        paths = _codex_jsonl_paths(source_path)
        rows: list[Mapping[str, Any]] = []
        exec_stream: bool | None = None
        for path in paths:
            path_rows = tuple(_iter_jsonl_objects(path))
            path_exec_stream = _is_codex_exec_stream(path_rows)
            if exec_stream is None:
                exec_stream = path_exec_stream
            elif exec_stream != path_exec_stream:
                raise ValueError(
                    "Codex session source mixes rollout and exec JSONL formats"
                )
            rows.extend(path_rows)
        parser = _CodexRowsParser(seq_start=0)
        records = parser.parse(rows, exec_stream=bool(exec_stream))
        return tuple(_renumber_records(records))

    def parse_raw_llm_calls(self, source: Path | str) -> tuple[dict[str, Any], ...]:
        source_path = Path(source)
        _reject_global_session_source(source_path, surface="Codex")
        paths = _codex_jsonl_paths(source_path)
        entries: list[_RawJsonlEntry] = []
        exec_stream: bool | None = None
        for path in paths:
            path_entries = tuple(_iter_jsonl_raw_entries(path))
            path_rows = tuple(entry.row for entry in path_entries if entry.row is not None)
            path_exec_stream = _is_codex_exec_stream(path_rows)
            if exec_stream is None:
                exec_stream = path_exec_stream
            elif exec_stream != path_exec_stream:
                raise ValueError(
                    "Codex session source mixes rollout and exec JSONL formats"
                )
            entries.extend(path_entries)
        source_format = "codex_exec_events" if exec_stream else "codex_rollout"
        groups = (
            _codex_exec_raw_groups(entries)
            if exec_stream
            else _codex_rollout_raw_groups(entries)
        )
        return tuple(
            _raw_llm_call_row(
                call_id=f"codex-call-{index:06d}",
                seq=index - 1,
                wire_api="codex_cli",
                purpose="codex_cli_session",
                source_format=source_format,
                history=history,
                response=response,
                model=_model_from_entries(response) or _model_from_entries(history),
            )
            for index, (history, response) in enumerate(groups, start=1)
        )


class KimiSessionTraceAdapter:
    """Adapter for Kimi CLI exported sessions."""

    def parse(self, source: Path | str) -> tuple[ObservabilityRecord, ...]:
        source_path = Path(source)
        _reject_global_session_source(source_path, surface="Kimi")
        if source_path.is_file() and source_path.suffix == ".zip":
            rows, archive_path = _read_kimi_zip_wire(source_path)
            resolver = _zip_image_resolver(archive_path)
        else:
            export_root, wire_path = _kimi_export_paths(source_path)
            rows = tuple(_iter_jsonl_objects(wire_path))
            resolver = _directory_image_resolver(export_root)
        parser = _KimiRowsParser(image_resolver=resolver)
        return parser.parse(rows)

    def parse_raw_llm_calls(self, source: Path | str) -> tuple[dict[str, Any], ...]:
        source_path = Path(source)
        _reject_global_session_source(source_path, surface="Kimi")
        if source_path.is_file() and source_path.suffix == ".zip":
            entries, manifest = _read_kimi_zip_raw_entries(source_path)
        else:
            export_root, wire_path = _kimi_export_paths(source_path)
            entries = tuple(_iter_jsonl_raw_entries(wire_path))
            manifest = _read_kimi_manifest(export_root)
        groups = _kimi_raw_groups(entries)
        return tuple(
            _raw_llm_call_row(
                call_id=f"kimi-call-{index:06d}",
                seq=index - 1,
                wire_api="kimi_cli",
                purpose="kimi_cli_session",
                source_format="kimi_wire",
                history=history,
                response=response,
                model=_model_from_entries(response) or _model_from_entries(history),
                manifest=manifest,
            )
            for index, (history, response) in enumerate(groups, start=1)
        )


class _CodexRowsParser:
    def __init__(self, *, seq_start: int = 0) -> None:
        self._seq = seq_start
        self._call_index = 0
        self._records: list[ObservabilityRecord] = []
        self._current: _CallDraft | None = None
        self._seen_tool_calls: set[str] = set()
        self._seen_tool_results: set[str] = set()

    def parse(
        self, rows: Sequence[Mapping[str, Any]], *, exec_stream: bool
    ) -> tuple[ObservabilityRecord, ...]:
        if exec_stream:
            for row in rows:
                self._consume_exec_row(row)
        else:
            for row in rows:
                self._consume_rollout_row(row)
        self._finalize_current()
        return tuple(self._records)

    def _new_call(self) -> _CallDraft:
        self._call_index += 1
        return _CallDraft(
            call_id=f"codex-call-{self._call_index:06d}",
            reasoning_parts=[],
            tool_calls=[],
            frame_drafts=[],
        )

    def _ensure_call(self) -> _CallDraft:
        if self._current is None:
            self._current = self._new_call()
        return self._current

    def _consume_rollout_row(self, row: Mapping[str, Any]) -> None:
        row_type = row.get("type")
        payload = _mapping_or_empty(row.get("payload"))
        payload_type = payload.get("type")
        if row_type == "response_item" and payload_type == "message":
            if payload.get("role") == "assistant":
                text = _message_text(payload)
                if text:
                    self._ensure_call().reasoning_parts.append(text)
            return
        if row_type == "response_item" and payload_type == "function_call":
            self._add_tool_call(payload)
            return
        if row_type == "response_item" and payload_type == "function_call_output":
            native_id = _text_or_none(payload.get("call_id"))
            if native_id and native_id in self._seen_tool_results:
                return
            self._add_frames_from_tool_result(
                payload.get("output"),
                tool_name=None,
                native_call_id=native_id,
            )
            return
        if row_type == "event_msg" and payload.get("type") == "mcp_tool_call_end":
            native_id = _text_or_none(payload.get("call_id"))
            invocation = _mapping_or_empty(payload.get("invocation"))
            self._add_tool_call(
                {
                    "call_id": native_id,
                    "name": invocation.get("tool"),
                    "arguments": invocation.get("arguments"),
                }
            )
            self._add_frames_from_tool_result(
                payload.get("result"),
                tool_name=_text_or_none(invocation.get("tool")),
                native_call_id=native_id,
            )
            return
        if row_type == "event_msg" and payload.get("type") == "token_count":
            info = _mapping_or_empty(payload.get("info"))
            usage = _mapping_or_empty(
                info.get("last_token_usage") or info.get("total_token_usage")
            )
            self._ensure_call().tokens = ObservabilityTokens.from_mapping(usage)
            self._finalize_current()

    def _consume_exec_row(self, row: Mapping[str, Any]) -> None:
        row_type = row.get("type")
        if row_type == "turn.started":
            if self._current and self._current.has_content():
                self._finalize_current()
            self._current = self._new_call()
            return
        if row_type in {"item.started", "item.completed"}:
            item = _mapping_or_empty(row.get("item"))
            item_type = item.get("type")
            if item_type == "agent_message" and row_type == "item.completed":
                text = _text_or_none(item.get("text"))
                if text:
                    self._ensure_call().reasoning_parts.append(text)
                return
            if item_type == "mcp_tool_call":
                self._add_tool_call(item)
                if row_type == "item.completed":
                    self._add_frames_from_tool_result(
                        item.get("result"),
                        tool_name=_text_or_none(item.get("tool")),
                        native_call_id=_text_or_none(item.get("id")),
                    )
                return
        if row_type == "turn.completed":
            self._ensure_call().tokens = ObservabilityTokens.from_mapping(
                _mapping_or_empty(row.get("usage"))
            )
            self._finalize_current()

    def _add_tool_call(self, payload: Mapping[str, Any]) -> None:
        name = _text_or_none(payload.get("name") or payload.get("tool"))
        if not name:
            return
        native_id = _text_or_none(payload.get("call_id") or payload.get("id"))
        if native_id and native_id in self._seen_tool_calls:
            return
        args = _arguments_mapping(payload.get("arguments") or payload.get("args"))
        self._ensure_call().tool_calls.append(
            ObservabilityToolCall(name=name, args=args)
        )
        if native_id:
            self._seen_tool_calls.add(native_id)

    def _add_frames_from_tool_result(
        self,
        value: Any,
        *,
        tool_name: str | None,
        native_call_id: str | None,
    ) -> None:
        if native_call_id:
            self._seen_tool_results.add(native_call_id)
        current = self._ensure_call()
        for frame in _frame_drafts_from_result(value, tool_name=tool_name):
            current.frame_drafts.append(frame)

    def _finalize_current(self) -> None:
        current = self._current
        if current is None or not current.has_content():
            self._current = None
            return
        reasoning = "\n\n".join(part for part in current.reasoning_parts if part)
        self._records.append(
            ObservabilityRecord(
                kind="llm_call",
                seq=self._seq,
                call_id=current.call_id,
                reasoning=reasoning,
                tool_calls=tuple(current.tool_calls),
                tokens=current.tokens,
            )
        )
        self._seq += 1
        for frame in current.frame_drafts:
            self._records.append(
                ObservabilityRecord(
                    kind="frame",
                    seq=self._seq,
                    call_id=current.call_id,
                    image_path=frame.image_path,
                    pose=frame.pose,
                    action=frame.action,
                )
            )
            self._seq += 1
        self._current = None


class _KimiRowsParser:
    def __init__(self, *, image_resolver: "_ImageResolver") -> None:
        self._image_resolver = image_resolver
        self._seq = 0
        self._call_index = 0
        self._records: list[ObservabilityRecord] = []
        self._current: _CallDraft | None = None
        self._tool_names_by_id: dict[str, str] = {}

    def parse(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> tuple[ObservabilityRecord, ...]:
        for row in rows:
            self._consume_row(row)
        self._finalize_current()
        return tuple(self._records)

    def _new_call(self) -> _CallDraft:
        self._call_index += 1
        return _CallDraft(
            call_id=f"kimi-call-{self._call_index:06d}",
            reasoning_parts=[],
            tool_calls=[],
            frame_drafts=[],
        )

    def _ensure_call(self) -> _CallDraft:
        if self._current is None:
            self._current = self._new_call()
        return self._current

    def _consume_row(self, row: Mapping[str, Any]) -> None:
        if row.get("type") == "context.append_loop_event":
            self._consume_loop_event(_mapping_or_empty(row.get("event")))
            return

        payload = _mapping_or_empty(row.get("payload") or row)
        row_type = _text_or_none(payload.get("type") or row.get("type")) or ""
        role = _text_or_none(payload.get("role"))
        if row_type == "usage.record":
            usage = _usage_from_payload(payload)
            if usage and self._current and self._current.has_content():
                self._current.tokens = usage
            return
        if self._is_assistant_message(row_type, role):
            if self._current and self._current.has_content():
                self._finalize_current()
            current = self._ensure_call()
            text = _message_text(payload)
            if text:
                current.reasoning_parts.append(text)
            for tool_call in _extract_kimi_tool_calls(payload):
                current.tool_calls.append(tool_call)
            usage = _usage_from_payload(payload)
            if usage:
                current.tokens = usage
            return
        if row_type in {"tool_call", "function_call"}:
            for tool_call in _extract_kimi_tool_calls(payload):
                self._ensure_call().tool_calls.append(tool_call)
            return
        if row_type in {"tool_result", "tool_call_result", "function_call_output"}:
            tool_name = _text_or_none(payload.get("name") or payload.get("tool"))
            for frame in _frame_drafts_from_result(payload, tool_name=tool_name):
                resolved = _FrameDraft(
                    image_path=self._image_resolver(frame.image_path),
                    pose=frame.pose,
                    action=frame.action,
                )
                self._ensure_call().frame_drafts.append(resolved)
            return
        usage = _usage_from_payload(payload)
        if usage:
            self._ensure_call().tokens = usage
            if row_type in {"turn_end", "assistant_turn_end", "token_count"}:
                self._finalize_current()

    def _consume_loop_event(self, event: Mapping[str, Any]) -> None:
        event_type = _text_or_none(event.get("type")) or ""
        if event_type == "step.begin":
            if self._current and self._current.has_content():
                self._finalize_current()
            self._current = self._new_call()
            return
        if event_type == "content.part":
            part = _mapping_or_empty(event.get("part"))
            text = _text_or_none(part.get("think") or part.get("text"))
            if text:
                self._ensure_call().reasoning_parts.append(text)
            return
        if event_type == "tool.call":
            tool_call = _tool_call_from_payload(event)
            if tool_call:
                self._ensure_call().tool_calls.append(tool_call)
            tool_call_id = _text_or_none(
                event.get("toolCallId") or event.get("uuid") or event.get("id")
            )
            tool_name = _text_or_none(event.get("name") or event.get("tool"))
            if tool_call_id and tool_name:
                self._tool_names_by_id[tool_call_id] = tool_name
            return
        if event_type == "tool.result":
            tool_call_id = _text_or_none(
                event.get("toolCallId") or event.get("parentUuid")
            )
            tool_name = (
                self._tool_names_by_id.get(tool_call_id or "")
                or _text_or_none(event.get("name") or event.get("tool"))
            )
            for frame in _frame_drafts_from_result(
                event.get("result") if "result" in event else event,
                tool_name=tool_name,
            ):
                self._ensure_call().frame_drafts.append(
                    _FrameDraft(
                        image_path=self._image_resolver(frame.image_path),
                        pose=frame.pose,
                        action=frame.action,
                    )
                )
            return
        if event_type == "step.end":
            usage = _usage_from_payload(event)
            if usage:
                self._ensure_call().tokens = usage
            self._finalize_current()
            return

    @staticmethod
    def _is_assistant_message(row_type: str, role: str | None) -> bool:
        return row_type in {"assistant_message", "assistant"} or (
            row_type == "message" and role == "assistant"
        )

    def _finalize_current(self) -> None:
        current = self._current
        if current is None or not current.has_content():
            self._current = None
            return
        self._records.append(
            ObservabilityRecord(
                kind="llm_call",
                seq=self._seq,
                call_id=current.call_id,
                reasoning="\n\n".join(current.reasoning_parts),
                tool_calls=tuple(current.tool_calls),
                tokens=current.tokens,
            )
        )
        self._seq += 1
        for frame in current.frame_drafts:
            self._records.append(
                ObservabilityRecord(
                    kind="frame",
                    seq=self._seq,
                    call_id=current.call_id,
                    image_path=frame.image_path,
                    pose=frame.pose,
                    action=frame.action,
                )
            )
            self._seq += 1
        self._current = None


def _codex_jsonl_paths(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    if not source.exists():
        raise FileNotFoundError(f"Codex session source does not exist: {source}")
    if not source.is_dir():
        raise FileNotFoundError(
            f"Codex session source is not a file or directory: {source}"
        )
    patterns = (
        "codex_session_export/sessions/**/*.jsonl",
        "codex_home/sessions/**/*.jsonl",
        "sessions/**/*.jsonl",
        "**/rollout-*.jsonl",
        "codex_exec_events.jsonl",
        "**/codex_exec_events.jsonl",
    )
    for pattern in patterns:
        paths = tuple(sorted(path for path in source.glob(pattern) if path.is_file()))
        if paths:
            for path in paths:
                _reject_global_session_source(path, surface="Codex")
                _reject_source_escape(path, source, surface="Codex")
            return paths
    raise FileNotFoundError(f"No Codex session JSONL found under: {source}")


def _reject_global_session_source(source: Path, *, surface: str) -> None:
    resolved = source.expanduser().resolve(strict=False)
    for root in _global_session_roots():
        if resolved == root or _path_is_relative_to(resolved, root):
            raise ValueError(
                f"{surface} trace adapter requires an explicit run-local/export "
                f"source, not global session state: {root}"
            )


def _reject_global_kimi_export_candidate(path: Path) -> None:
    _reject_global_session_source(path, surface="Kimi")


def _reject_source_escape(path: Path, root: Path, *, surface: str) -> None:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    if not (
        resolved_path == resolved_root
        or _path_is_relative_to(resolved_path, resolved_root)
    ):
        raise ValueError(
            f"{surface} trace adapter requires run-local/export session files; "
            f"candidate escapes source root: {path}"
        )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _kimi_export_paths(source: Path) -> tuple[Path, Path]:
    if source.is_file():
        root = source.parent.parent.parent
        _require_kimi_manifest(root)
        _reject_global_kimi_export_candidate(source)
        _reject_source_escape(source, root, surface="Kimi")
        return root, source
    if not source.exists():
        raise FileNotFoundError(f"Kimi export source does not exist: {source}")
    if not source.is_dir():
        raise FileNotFoundError(
            f"Kimi export source is not a file or directory: {source}"
        )
    _require_kimi_manifest(source)
    preferred = source / "agents" / "main" / "wire.jsonl"
    if preferred.is_file():
        _reject_global_kimi_export_candidate(preferred)
        _reject_source_escape(preferred, source, surface="Kimi")
        return source, preferred
    matches = tuple(
        sorted(path for path in source.glob("**/wire.jsonl") if path.is_file())
    )
    if matches:
        wire = matches[0]
        _reject_global_kimi_export_candidate(wire)
        _reject_source_escape(wire, source, surface="Kimi")
        root = wire.parents[2] if len(wire.parents) >= 3 else source
        return root, wire
    raise FileNotFoundError(f"No Kimi agents/main/wire.jsonl found under: {source}")


def _require_kimi_manifest(root: Path) -> None:
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Kimi export is missing manifest.json: {root}")


def _read_kimi_manifest(root: Path) -> Mapping[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _read_kimi_zip_wire(source: Path) -> tuple[tuple[Mapping[str, Any], ...], Path]:
    if not source.is_file():
        raise FileNotFoundError(f"Kimi zip export does not exist: {source}")
    rows: list[Mapping[str, Any]] = []
    with zipfile.ZipFile(source) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise FileNotFoundError("Kimi zip export is missing manifest.json")
        wire_name = "agents/main/wire.jsonl"
        if wire_name not in names:
            candidates = sorted(name for name in names if name.endswith("/wire.jsonl"))
            if not candidates:
                raise FileNotFoundError("Kimi zip export is missing wire.jsonl")
            wire_name = candidates[0]
        for raw in zf.read(wire_name).decode("utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, Mapping):
                rows.append(obj)
    return tuple(rows), source


def _read_kimi_zip_raw_entries(
    source: Path,
) -> tuple[tuple[_RawJsonlEntry, ...], Mapping[str, Any] | None]:
    if not source.is_file():
        raise FileNotFoundError(f"Kimi zip export does not exist: {source}")
    with zipfile.ZipFile(source) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise FileNotFoundError("Kimi zip export is missing manifest.json")
        manifest = _decode_json_mapping(zf.read("manifest.json").decode("utf-8"))
        wire_name = "agents/main/wire.jsonl"
        if wire_name not in names:
            candidates = sorted(name for name in names if name.endswith("/wire.jsonl"))
            if not candidates:
                raise FileNotFoundError("Kimi zip export is missing wire.jsonl")
            wire_name = candidates[0]
        entries = _jsonl_raw_entries_from_text(
            zf.read(wire_name).decode("utf-8", errors="replace"),
            source_label=f"zip://{source.resolve(strict=False)}!/{wire_name}",
        )
    return tuple(entries), manifest


def _iter_jsonl_objects(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(payload)
    return tuple(rows)


def _iter_jsonl_raw_entries(path: Path) -> tuple[_RawJsonlEntry, ...]:
    return tuple(
        _jsonl_raw_entries_from_text(
            path.read_text(encoding="utf-8", errors="replace"),
            source_label=str(path),
        )
    )


def _jsonl_raw_entries_from_text(
    text: str, *, source_label: str
) -> tuple[_RawJsonlEntry, ...]:
    entries: list[_RawJsonlEntry] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            entries.append(
                _RawJsonlEntry(
                    source_path=source_label,
                    line_no=line_no,
                    raw_line=raw,
                )
            )
            continue
        if isinstance(payload, Mapping):
            entries.append(
                _RawJsonlEntry(
                    source_path=source_label,
                    line_no=line_no,
                    row=payload,
                    raw_line=raw,
                )
            )
        else:
            entries.append(
                _RawJsonlEntry(
                    source_path=source_label,
                    line_no=line_no,
                    raw_line=raw,
                )
            )
    return tuple(entries)


def _decode_json_mapping(text: str) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _is_codex_exec_stream(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        row_type = row.get("type")
        if row_type in {"turn.started", "turn.completed", "thread.started"}:
            return True
        if isinstance(row_type, str) and row_type.startswith("item."):
            return True
    return False


def _codex_exec_raw_groups(
    entries: Sequence[_RawJsonlEntry],
) -> tuple[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]], ...]:
    groups: list[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]]] = []
    history: list[_RawJsonlEntry] = []
    current: list[_RawJsonlEntry] = []
    current_history: list[_RawJsonlEntry] = []
    in_turn = False
    for entry in entries:
        row_type = entry.row.get("type") if entry.row is not None else None
        if row_type == "turn.started":
            if current:
                groups.append((tuple(current_history), tuple(current)))
                history.extend(current)
            current_history = list(history)
            current = [entry]
            in_turn = True
            continue
        if in_turn:
            current.append(entry)
            if row_type == "turn.completed":
                groups.append((tuple(current_history), tuple(current)))
                history.extend(current)
                current = []
                current_history = []
                in_turn = False
            continue
        history.append(entry)
    if current:
        groups.append((tuple(current_history), tuple(current)))
    elif not groups and history:
        groups.append(((), tuple(history)))
    return tuple(groups)


def _codex_rollout_raw_groups(
    entries: Sequence[_RawJsonlEntry],
) -> tuple[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]], ...]:
    groups: list[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]]] = []
    history: list[_RawJsonlEntry] = []
    current: list[_RawJsonlEntry] = []
    current_history: list[_RawJsonlEntry] = []
    for entry in entries:
        row_type = entry.row.get("type") if entry.row is not None else None
        payload = _mapping_or_empty(entry.row.get("payload") if entry.row else None)
        payload_type = payload.get("type")
        starts_call = (
            row_type == "response_item"
            and payload_type in {"message", "function_call"}
            and not current
        )
        if starts_call:
            current_history = list(history)
        if current or starts_call:
            current.append(entry)
            if row_type == "event_msg" and payload_type == "token_count":
                groups.append((tuple(current_history), tuple(current)))
                history.extend(current)
                current = []
                current_history = []
            continue
        history.append(entry)
    if current:
        groups.append((tuple(current_history), tuple(current)))
    elif not groups and history:
        groups.append(((), tuple(history)))
    return tuple(groups)


def _kimi_raw_groups(
    entries: Sequence[_RawJsonlEntry],
) -> tuple[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]], ...]:
    groups: list[tuple[tuple[_RawJsonlEntry, ...], tuple[_RawJsonlEntry, ...]]] = []
    history: list[_RawJsonlEntry] = []
    current: list[_RawJsonlEntry] = []
    current_history: list[_RawJsonlEntry] = []

    def finalize() -> None:
        nonlocal current, current_history
        if current:
            groups.append((tuple(current_history), tuple(current)))
            history.extend(current)
            current = []
            current_history = []

    for entry in entries:
        row = entry.row
        payload = _mapping_or_empty(row.get("payload") if row else None) or (
            _mapping_or_empty(row) if row else {}
        )
        row_type = _text_or_none(payload.get("type") or (row or {}).get("type")) or ""
        role = _text_or_none(payload.get("role"))
        event = _mapping_or_empty(row.get("event") if row else None)
        event_type = _text_or_none(event.get("type")) or ""
        starts_loop_call = (
            row_type == "context.append_loop_event" and event_type == "step.begin"
        )
        starts_message_call = _kimi_raw_is_assistant_message(row_type, role)
        if starts_loop_call or starts_message_call:
            finalize()
            current_history = list(history)
            current = [entry]
            if starts_message_call:
                continue
        elif current:
            current.append(entry)
        else:
            if row_type == "usage.record" and groups:
                previous_history, previous_response = groups[-1]
                groups[-1] = (previous_history, (*previous_response, entry))
            history.append(entry)
            continue
        if (
            row_type == "context.append_loop_event"
            and event_type == "step.end"
        ) or row_type in {"turn_end", "assistant_turn_end", "token_count"}:
            finalize()
    finalize()
    if not groups and history:
        groups.append(((), tuple(history)))
    return tuple(groups)


def _kimi_raw_is_assistant_message(row_type: str, role: str | None) -> bool:
    return row_type in {"assistant_message", "assistant"} or (
        row_type == "message" and role == "assistant"
    )


def _raw_llm_call_row(
    *,
    call_id: str,
    seq: int,
    wire_api: str,
    purpose: str,
    source_format: str,
    history: Sequence[_RawJsonlEntry],
    response: Sequence[_RawJsonlEntry],
    model: str | None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "source_format": source_format,
        "history_rows": [entry.to_json() for entry in history],
    }
    if manifest is not None:
        request["manifest"] = _json_safe(manifest)
    return {
        "schema_version": 1,
        "kind": "raw_llm_call",
        "round": None,
        "seq": seq,
        "call_id": call_id,
        "wire_api": wire_api,
        "purpose": purpose,
        "model": model,
        "request": request,
        "wire_request": None,
        "response": {
            "source_format": source_format,
            "rows": [entry.to_json() for entry in response],
        },
        "retry": None,
    }


def _model_from_entries(entries: Sequence[_RawJsonlEntry]) -> str | None:
    for entry in entries:
        if entry.row is None:
            continue
        model = _raw_entry_model(entry.row)
        if model:
            return model
    return None


def _raw_entry_model(row: Mapping[str, Any]) -> str | None:
    model = _text_or_none(row.get("model"))
    if model:
        return model
    payload = _mapping_or_empty(row.get("payload"))
    if _text_or_none(payload.get("type")) in {
        "session_meta",
        "token_count",
        "usage.record",
    }:
        return _text_or_none(payload.get("model"))
    return None


def _extract_kimi_tool_calls(
    payload: Mapping[str, Any],
) -> tuple[ObservabilityToolCall, ...]:
    calls: list[ObservabilityToolCall] = []
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
        for item in tool_calls:
            call = _tool_call_from_payload(_mapping_or_empty(item))
            if call:
                calls.append(call)
    for key in ("function_call", "tool_call"):
        if key in payload:
            call = _tool_call_from_payload(_mapping_or_empty(payload.get(key)))
            if call:
                calls.append(call)
    if payload.get("type") in {"tool_call", "function_call"}:
        call = _tool_call_from_payload(payload)
        if call:
            calls.append(call)
    return tuple(calls)


def _tool_call_from_payload(payload: Mapping[str, Any]) -> ObservabilityToolCall | None:
    function = _mapping_or_empty(payload.get("function"))
    name = _text_or_none(
        payload.get("name") or payload.get("tool") or function.get("name")
    )
    if not name:
        return None
    args_value = (
        payload.get("args")
        if "args" in payload
        else payload.get("arguments", function.get("arguments"))
    )
    return ObservabilityToolCall(name=name, args=_arguments_mapping(args_value))


def _usage_from_payload(payload: Mapping[str, Any]) -> ObservabilityTokens | None:
    for key in ("usage", "token_usage", "tokens"):
        usage = payload.get(key)
        if isinstance(usage, Mapping):
            return _tokens_from_usage_mapping(usage)
    if payload.get("type") == "token_count":
        info = _mapping_or_empty(payload.get("info"))
        usage = _mapping_or_empty(
            info.get("last_token_usage") or info.get("total_token_usage")
        )
        return _tokens_from_usage_mapping(usage)
    return None


def _tokens_from_usage_mapping(
    usage: Mapping[str, Any],
) -> ObservabilityTokens | None:
    if any(key in usage for key in ("inputOther", "inputCacheRead", "inputCacheCreation")):
        direct_input = _int_or_none(usage.get("input"))
        uncached_input = _int_or_none(usage.get("inputOther")) or 0
        cache_read = _int_or_none(usage.get("inputCacheRead")) or 0
        cache_creation = _int_or_none(usage.get("inputCacheCreation")) or 0
        input_tokens = direct_input
        if input_tokens is None:
            input_tokens = uncached_input + cache_read + cache_creation
        output_tokens = _int_or_none(usage.get("output"))
        total_tokens = input_tokens + (output_tokens or 0)
        return ObservabilityTokens(
            input_tokens=input_tokens,
            cached_input_tokens=cache_read or None,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    return ObservabilityTokens.from_mapping(usage)


def _frame_drafts_from_result(
    value: Any, *, tool_name: str | None
) -> tuple[_FrameDraft, ...]:
    expanded = _expand_jsonish(value)
    image_paths = _extract_image_paths(expanded)
    if not image_paths:
        return ()
    pose = _extract_pose(expanded)
    action = _extract_action(expanded, tool_name=tool_name)
    return tuple(
        _FrameDraft(image_path=image, pose=pose, action=action) for image in image_paths
    )


def _extract_image_paths(value: Any) -> tuple[str, ...]:
    images: list[str] = []

    def add(candidate: Any, *, force: bool = False) -> None:
        text = _text_or_none(candidate)
        if not text:
            return
        lower = text.lower().split("?", 1)[0]
        if (force or lower.endswith(_IMAGE_SUFFIXES)) and text not in images:
            images.append(text)

    def walk(node: Any) -> None:
        node = _expand_jsonish(node)
        if isinstance(node, Mapping):
            mime_type = _text_or_none(node.get("mime_type") or node.get("mimeType"))
            node_type = _text_or_none(node.get("type"))
            force_media = bool(
                (mime_type and mime_type.startswith("image/")) or node_type == "image"
            )
            for key in ("path", "url", "uri", "image", "image_path", "blob"):
                if key in node:
                    add(node.get(key), force=force_media)
            for key in ("images", "recallable_paths", "attachments", "blobs"):
                child = node.get(key)
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                    for item in child:
                        if isinstance(item, Mapping):
                            walk(item)
                        else:
                            add(item)
            for child in node.values():
                walk(child)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                walk(child)
            return
        add(node)

    walk(value)
    return tuple(images)


def _extract_pose(value: Any) -> Mapping[str, Any]:
    expanded = _expand_jsonish(value)
    for key in ("ego_pose", "pose"):
        direct_pose = _find_first_mapping_key(expanded, key)
        if direct_pose:
            return _json_safe(direct_pose)
    found: dict[str, Any] = {}

    def maybe_add(label: str, candidate: Any) -> None:
        if isinstance(candidate, Mapping) and candidate:
            found[label] = _json_safe(candidate)

    def walk(node: Any) -> None:
        node = _expand_jsonish(node)
        if isinstance(node, Mapping):
            maybe_add("ego_pose", node.get("ego_pose"))
            maybe_add("pose", node.get("pose"))
            maybe_add("agent_state", node.get("agent_state"))
            state_summary = node.get("state_summary")
            if isinstance(state_summary, Mapping):
                summary = {
                    key: state_summary[key]
                    for key in ("position", "heading_deg", "rotation", "collided")
                    if key in state_summary
                }
                maybe_add("state_summary", summary)
            direct = {
                key: node[key]
                for key in ("position", "rotation", "heading_deg")
                if key in node
            }
            if "position" not in direct and "rotation" not in direct:
                direct = {}
            maybe_add("direct", direct)
            for child in node.values():
                walk(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                walk(child)

    walk(expanded)
    if not found:
        return {}
    if "ego_pose" in found and len(found) == 1:
        return found["ego_pose"]
    if "pose" in found and len(found) == 1:
        return found["pose"]
    return found


def _find_first_mapping_key(value: Any, key: str) -> Mapping[str, Any] | None:
    value = _expand_jsonish(value)
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate
        for child in value.values():
            found = _find_first_mapping_key(child, key)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_first_mapping_key(child, key)
            if found:
                return found
    return None


def _extract_action(value: Any, *, tool_name: str | None) -> str:
    expanded = _expand_jsonish(value)
    for key in ("action", "last_action"):
        found = _find_first_text_key(expanded, key)
        if found:
            return found
    return tool_name or "unknown"


def _find_first_text_key(value: Any, key: str) -> str | None:
    value = _expand_jsonish(value)
    if isinstance(value, Mapping):
        if key in value:
            text = _text_or_none(value.get(key))
            if text:
                return text
        for child in value.values():
            found = _find_first_text_key(child, key)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_first_text_key(child, key)
            if found:
                return found
    return None


def _message_text(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            item_map = _mapping_or_empty(item)
            text = _text_or_none(item_map.get("text") or item_map.get("content"))
            if text and item_map.get("type") in {"output_text", "text", None}:
                parts.append(text)
        return "\n".join(parts)
    message = payload.get("message")
    if isinstance(message, Mapping):
        return _message_text(message)
    return _text_or_none(payload.get("text")) or ""


def _arguments_mapping(value: Any) -> Mapping[str, Any]:
    decoded = _expand_jsonish(value)
    if isinstance(decoded, Mapping):
        return decoded
    return {"value": decoded} if decoded not in (None, "") else {}


def _expand_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if "\nOutput:\n" in text:
            text = text.rsplit("\nOutput:\n", 1)[1].strip()
        decoded = _decode_json_text(text)
        if decoded is value or decoded == value:
            return value
        return _expand_jsonish(decoded)
    if isinstance(value, Mapping):
        return {str(key): _expand_jsonish(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_jsonish(child) for child in value]
    return value


def _decode_json_text(text: str) -> Any:
    if not text or text[0] not in '[{"':
        return text
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    return decoded


def _directory_image_resolver(root: Path) -> "_ImageResolver":
    resolved_root = root.resolve(strict=False)

    def resolve(path: str) -> str:
        candidate = Path(path)
        if candidate.is_absolute() or path.startswith(
            ("zip://", "http://", "https://")
        ):
            return path
        return str(resolved_root / candidate)

    return resolve


def _zip_image_resolver(archive_path: Path) -> "_ImageResolver":
    archive = archive_path.resolve(strict=False)

    def resolve(path: str) -> str:
        if path.startswith("zip://") or Path(path).is_absolute() or path.startswith(
            ("http://", "https://")
        ):
            return path
        return f"zip://{archive}!/{path.lstrip('/')}"

    return resolve


def _renumber_records(
    records: Sequence[ObservabilityRecord],
) -> tuple[ObservabilityRecord, ...]:
    renumbered: list[ObservabilityRecord] = []
    for seq, record in enumerate(records):
        if record.kind == "llm_call":
            renumbered.append(
                ObservabilityRecord(
                    kind="llm_call",
                    seq=seq,
                    call_id=record.call_id,
                    reasoning=record.reasoning,
                    tool_calls=record.tool_calls,
                    tokens=record.tokens,
                )
            )
        else:
            renumbered.append(
                ObservabilityRecord(
                    kind="frame",
                    seq=seq,
                    call_id=record.call_id,
                    image_path=record.image_path,
                    pose=record.pose,
                    action=record.action,
                )
            )
    return tuple(renumbered)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("observability record contains non-finite float")
        return value
    return str(value)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


_ImageResolver = Callable[[str], str]

__all__ = [
    "CodexSessionTraceAdapter",
    "HarnessTraceAdapter",
    "KimiSessionTraceAdapter",
    "ObservabilityRecord",
    "ObservabilityTokens",
    "ObservabilityToolCall",
    "parse_codex_raw_llm_calls",
    "parse_codex_session_trace",
    "parse_kimi_raw_llm_calls",
    "parse_kimi_session_export",
    "write_observability_jsonl",
    "write_raw_llm_calls_jsonl",
]
