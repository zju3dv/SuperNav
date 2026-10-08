from __future__ import annotations

import argparse
import html
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from supernav.runtime.config import write_json

SVG_FONT_STACK = "DejaVu Sans, Arial, Helvetica, sans-serif"
IMAGE_WIDTH = 1800
IMAGE_MARGIN = 32
ROW_GAP = 18
LABEL_FONT_SIZE = 24
CONTENT_FONT_SIZE = 20
MAX_IMAGE_CONTENT_CHARS = 1200
TYPE_COL_WIDTH = 240
TIME_COL_WIDTH = 180

TRACE_CATEGORY_ORDER = (
    "model_output",
    "model_reasoning",
    "function_call",
    "tool_call",
    "history_update",
    "build_prompt",
    "tool_call_result_recording",
    "request_dispatch",
    "time_to_first_token",
)


def _parse_iso_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _load_session_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        data["_line"] = idx
        data["_ts"] = _parse_iso_ts(str(data["timestamp"]))
        rows.append(data)
    return rows


def _load_trace_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        data["_line"] = idx
        data["_ts_ms"] = int(data.get("wall_time_unix_ms", 0))
        rows.append(data)
    return rows


def _assistant_text(payload: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for item in payload.get("content", []):
        if isinstance(item, Mapping) and item.get("type") in {"output_text", "input_text"}:
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def _reasoning_content(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary")
    encrypted = payload.get("encrypted_content")
    return {
        "summary": summary if isinstance(summary, list) else [],
        "encrypted_content": str(encrypted or ""),
    }


def _content_from_output_item(item: Mapping[str, Any]) -> Any:
    item_type = str(item.get("type", ""))
    if item_type == "message":
        return _assistant_text(item)
    if item_type == "reasoning":
        return _reasoning_content(item)
    if item_type == "function_call":
        name = str(item.get("name", ""))
        arguments = str(item.get("arguments", ""))
        return {"name": name, "arguments": arguments, "call_id": item.get("call_id")}
    return json.dumps(item, ensure_ascii=False)


def _trace_payload_path(trace_file: Path, ref: Mapping[str, Any]) -> Optional[Path]:
    payload_path = ref.get("path")
    if not payload_path:
        return None
    return trace_file.parent / str(payload_path)


def _pair_key(prefix: str, *parts: Any) -> str:
    return "|".join([prefix, *[str(part) for part in parts]])


def _trace_row_time_s(row: Mapping[str, Any]) -> float:
    return float(int(row.get("_ts_ms", 0))) / 1000.0


def _display_content(event: Mapping[str, Any], *, max_chars: int = MAX_IMAGE_CONTENT_CHARS) -> str:
    content: Any
    kind = str(event.get("kind", ""))
    if kind == "model_reasoning":
        payload = event.get("content", {})
        if isinstance(payload, Mapping):
            summary = payload.get("summary")
            encrypted = str(payload.get("encrypted_content", ""))
            content = json.dumps(summary, ensure_ascii=False) if isinstance(summary, list) and summary else encrypted
        else:
            content = str(payload)
    elif kind in {"function_call", "tool_call"}:
        payload = event.get("content", {})
        content = json.dumps(payload, ensure_ascii=False) if isinstance(payload, Mapping) else str(payload)
    else:
        content = event.get("content", "")
    text = str(content)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    omitted = len(text) - max_chars
    return f"{trimmed} ... [truncated {omitted} chars]"


def _wrap_text(text: str, width_chars: int) -> List[str]:
    if not text:
        return [""]
    wrapped = textwrap.wrap(
        text,
        width=max(20, width_chars),
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return wrapped or [text]


def render_timing_summary_image(
    summary: Mapping[str, Any],
    out_path: str | Path,
    *,
    max_content_chars: int = MAX_IMAGE_CONTENT_CHARS,
    width: int = IMAGE_WIDTH,
) -> Path:
    out_file = Path(out_path)
    content_x = IMAGE_MARGIN + TYPE_COL_WIDTH + TIME_COL_WIDTH
    content_width = width - content_x - IMAGE_MARGIN
    avg_char_width = max(8, int(CONTENT_FONT_SIZE * 0.58))
    width_chars = max(30, content_width // avg_char_width)
    line_height = CONTENT_FONT_SIZE + 8
    label_height = LABEL_FONT_SIZE + 8

    event_rows: List[Dict[str, Any]] = []
    total_height = IMAGE_MARGIN
    for event in summary.get("events", []):
        if not isinstance(event, Mapping):
            continue
        content = _display_content(event, max_chars=max_content_chars)
        wrapped = _wrap_text(content, width_chars)
        block_height = max(label_height, len(wrapped) * line_height)
        event_rows.append({"event": event, "wrapped": wrapped, "height": block_height})
        total_height += block_height + ROW_GAP
    total_height = max(total_height + IMAGE_MARGIN, 200)

    y = IMAGE_MARGIN
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        f'<rect x="0" y="0" width="{width}" height="{total_height}" fill="white" />',
        f'<style>text {{ font-family: {SVG_FONT_STACK}; fill: black; }}</style>',
    ]
    for row in event_rows:
        event = row["event"]
        kind = html.escape(str(event.get("kind", "")))
        time_text = html.escape(f"{float(event.get('time_s', 0.0)):.3f}s")
        svg_lines.append(
            f'<text x="{IMAGE_MARGIN}" y="{y + LABEL_FONT_SIZE}" font-size="{LABEL_FONT_SIZE}" font-weight="700">{kind}</text>'
        )
        svg_lines.append(
            f'<text x="{IMAGE_MARGIN + TYPE_COL_WIDTH}" y="{y + LABEL_FONT_SIZE}" font-size="{LABEL_FONT_SIZE}" font-weight="700">{time_text}</text>'
        )
        for idx, line in enumerate(row["wrapped"]):
            line_y = y + CONTENT_FONT_SIZE + idx * line_height
            svg_lines.append(
                f'<text x="{content_x}" y="{line_y}" font-size="{CONTENT_FONT_SIZE}">{html.escape(line)}</text>'
            )
        y += row["height"] + ROW_GAP
    svg_lines.append("</svg>")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    return out_file


def build_timing_summary_from_session(session_path: str | Path) -> Dict[str, Any]:
    source = Path(session_path)
    rows = _load_session_rows(source)
    if not rows:
        raise ValueError(f"no rows found in {source}")

    call_end_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("type") == "event_msg" and row.get("payload", {}).get("type") == "mcp_tool_call_end":
            call_id = str(row["payload"].get("call_id", ""))
            if call_id:
                call_end_by_id[call_id] = row

    prev_response_item: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = []
    totals = {"reasoning": 0.0, "model_output": 0.0, "tool_call": 0.0}
    counts = {"reasoning": 0, "model_output": 0, "tool_call": 0}
    tool_totals: Dict[str, float] = {}

    for row in rows:
        if row.get("type") != "response_item":
            continue
        if prev_response_item is None:
            prev_response_item = row
            continue

        payload = row.get("payload", {})
        kind = payload.get("type")
        anchor = prev_response_item
        anchor_ts = anchor["_ts"]

        if kind == "reasoning":
            duration_s = (row["_ts"] - anchor_ts).total_seconds()
            events.append(
                {
                    "kind": "reasoning",
                    "time_s": round(duration_s, 3),
                    "anchor_line": anchor["_line"],
                    "line": row["_line"],
                    "anchor_timestamp": anchor["timestamp"],
                    "timestamp": row["timestamp"],
                    "content": _reasoning_content(payload),
                }
            )
            totals["reasoning"] += duration_s
            counts["reasoning"] += 1
        elif kind == "message" and payload.get("role") == "assistant":
            duration_s = (row["_ts"] - anchor_ts).total_seconds()
            events.append(
                {
                    "kind": "model_output",
                    "time_s": round(duration_s, 3),
                    "phase": payload.get("phase"),
                    "anchor_line": anchor["_line"],
                    "line": row["_line"],
                    "anchor_timestamp": anchor["timestamp"],
                    "timestamp": row["timestamp"],
                    "content": _assistant_text(payload),
                }
            )
            totals["model_output"] += duration_s
            counts["model_output"] += 1
        elif kind == "function_call":
            call_id = str(payload.get("call_id", ""))
            end_row = call_end_by_id.get(call_id)
            if end_row is not None:
                duration_s = (end_row["_ts"] - anchor_ts).total_seconds()
                tool_name = str(payload.get("name", ""))
                events.append(
                    {
                        "kind": "tool_call",
                        "time_s": round(duration_s, 3),
                        "tool_name": tool_name,
                        "anchor_line": anchor["_line"],
                        "function_call_line": row["_line"],
                        "tool_end_line": end_row["_line"],
                        "anchor_timestamp": anchor["timestamp"],
                        "function_call_timestamp": row["timestamp"],
                        "tool_end_timestamp": end_row["timestamp"],
                    }
                )
                totals["tool_call"] += duration_s
                counts["tool_call"] += 1
                tool_totals[tool_name] = tool_totals.get(tool_name, 0.0) + duration_s

        prev_response_item = row

    elapsed_s = round((rows[-1]["_ts"] - rows[0]["_ts"]).total_seconds(), 3)
    summary = {
        "source_file": str(source),
        "source_kind": "codex_session",
        "timing_rule": {
            "reasoning": "previous response_item -> response_item.reasoning",
            "model_output": "previous response_item -> response_item.message (assistant)",
            "tool_call": "previous response_item -> response_item.function_call -> event_msg.mcp_tool_call_end",
        },
        "assumptions": [
            "codex_session.jsonl has no response_item.agent_message rows; model output uses response_item rows where payload.type=message and role=assistant",
            "times are attributed using previous response_item as the anchor, per fallback rule",
        ],
        "elapsed_s": elapsed_s,
        "counts": counts,
        "totals_s": {key: round(value, 3) for key, value in totals.items()},
        "tool_totals_s": {
            key: round(value, 3)
            for key, value in sorted(tool_totals.items(), key=lambda item: (-item[1], item[0]))
        },
        "shares_pct": {
            key: round((value * 100.0 / elapsed_s), 2) if elapsed_s else 0.0
            for key, value in totals.items()
        },
        "tool_shares_pct": {
            key: round((value * 100.0 / elapsed_s), 2) if elapsed_s else 0.0
            for key, value in tool_totals.items()
        },
    }
    return {"summary": summary, "events": events}


def build_timing_summary_from_trace(trace_path: str | Path) -> Dict[str, Any]:
    source = Path(trace_path)
    rows = _load_trace_rows(source)
    if not rows:
        raise ValueError(f"no rows found in {source}")

    payload_dir = source.parent / "payloads"
    item_details: Dict[str, Any] = {}
    tool_invocations: Dict[str, Any] = {}

    for row in rows:
        payload = row.get("payload", {})
        payload_type = payload.get("type")
        if payload_type == "inference_completed":
            response_ref = payload.get("response_payload")
            if isinstance(response_ref, Mapping):
                response_path = _trace_payload_path(source, response_ref)
                if response_path and response_path.is_file():
                    response_payload = _load_json(response_path)
                    for item in response_payload.get("output_items", []):
                        if isinstance(item, Mapping):
                            item_id = str(item.get("id", ""))
                            if item_id:
                                item_details[item_id] = item
        elif payload_type == "tool_call_started":
            invocation_ref = payload.get("invocation_payload")
            call_id = str(payload.get("tool_call_id", ""))
            if call_id and isinstance(invocation_ref, Mapping):
                invocation_path = _trace_payload_path(source, invocation_ref)
                if invocation_path and invocation_path.is_file():
                    tool_invocations[call_id] = _load_json(invocation_path)

    starts: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    totals = {key: 0.0 for key in TRACE_CATEGORY_ORDER}
    counts = {key: 0 for key in TRACE_CATEGORY_ORDER}
    tool_totals: Dict[str, float] = {}
    function_totals: Dict[str, float] = {}

    def begin(key: str, row: Dict[str, Any]) -> None:
        starts[key] = row

    def finish(key: str) -> Optional[Dict[str, Any]]:
        return starts.pop(key, None)

    def append_event(kind: str, start_row: Dict[str, Any], end_row: Dict[str, Any], content: Any) -> None:
        duration_s = (_trace_row_time_s(end_row) - _trace_row_time_s(start_row))
        entry = {
            "kind": kind,
            "time_s": round(duration_s, 3),
            "start_line": start_row["_line"],
            "end_line": end_row["_line"],
            "start_ts_ms": int(start_row["_ts_ms"]),
            "end_ts_ms": int(end_row["_ts_ms"]),
            "content": content,
        }
        events.append(entry)
        totals[kind] += duration_s
        counts[kind] += 1

    for row in rows:
        payload = row.get("payload", {})
        payload_type = payload.get("type")

        if payload_type == "inference_stream_event":
            event_kind = str(payload.get("event_kind", ""))
            item_id = str(payload.get("item_id", ""))
            item_type = str(payload.get("item_type", ""))
            call_id = str(payload.get("call_id", ""))

            if event_kind == "output_item_started":
                start_row = finish(_pair_key("time_to_first_token", payload.get("inference_call_id")))
                if start_row is not None:
                    append_event(
                        "time_to_first_token",
                        start_row,
                        row,
                        {
                            "inference_call_id": payload.get("inference_call_id"),
                            "item_id": payload.get("item_id"),
                            "item_type": payload.get("item_type"),
                            "call_id": payload.get("call_id"),
                        },
                    )

            if event_kind == "output_item_started" and item_type == "message":
                begin(_pair_key("model_output", item_id), row)
            elif event_kind == "output_item_completed" and item_type == "message":
                start_row = finish(_pair_key("model_output", item_id))
                if start_row is not None:
                    item = item_details.get(item_id, {})
                    append_event("model_output", start_row, row, _content_from_output_item(item))

            elif event_kind == "reasoning_item_started":
                begin(_pair_key("model_reasoning", item_id), row)
            elif event_kind == "reasoning_item_completed":
                start_row = finish(_pair_key("model_reasoning", item_id))
                if start_row is not None:
                    item = item_details.get(item_id, {})
                    append_event("model_reasoning", start_row, row, _content_from_output_item(item))

            elif event_kind == "function_call_item_started":
                begin(_pair_key("function_call", item_id), row)
            elif event_kind == "function_call_item_completed":
                start_row = finish(_pair_key("function_call", item_id))
                if start_row is not None:
                    item = item_details.get(item_id, {})
                    content = _content_from_output_item(item)
                    append_event("function_call", start_row, row, content)
                    if isinstance(content, Mapping):
                        name = str(content.get("name", ""))
                        if name:
                            function_totals[name] = function_totals.get(name, 0.0) + events[-1]["time_s"]

        elif payload_type == "tool_call_started":
            begin(_pair_key("tool_call", payload.get("tool_call_id")), row)
        elif payload_type in {"tool_call_completed", "tool_call_ended"}:
            start_row = finish(_pair_key("tool_call", payload.get("tool_call_id")))
            if start_row is not None:
                call_id = str(payload.get("tool_call_id", ""))
                invocation = tool_invocations.get(call_id, {})
                kind_payload = payload.get("kind", {}) if isinstance(payload.get("kind"), Mapping) else {}
                tool_name = str(kind_payload.get("name") or invocation.get("tool_name") or "")
                content = {
                    "tool_name": tool_name,
                    "tool_namespace": invocation.get("tool_namespace"),
                    "arguments": invocation.get("payload", {}).get("arguments") if isinstance(invocation.get("payload"), Mapping) else None,
                    "status": payload.get("status"),
                }
                append_event("tool_call", start_row, row, content)
                if tool_name:
                    tool_totals[tool_name] = tool_totals.get(tool_name, 0.0) + events[-1]["time_s"]

        elif payload_type == "inference_boundary_observed":
            inference_call_id = str(payload.get("inference_call_id", ""))
            boundary_name = str(payload.get("boundary_name", ""))
            if inference_call_id and boundary_name == "request_sent":
                dispatch_start = finish(_pair_key("request_dispatch", inference_call_id))
                if dispatch_start is not None:
                    append_event(
                        "request_dispatch",
                        dispatch_start,
                        row,
                        {
                            "inference_call_id": inference_call_id,
                            "boundary_name": boundary_name,
                        },
                    )
                begin(_pair_key("time_to_first_token", inference_call_id), row)

        elif payload_type == "boundary_observed":
            boundary_name = str(payload.get("boundary_name", ""))
            phase = str(payload.get("phase", ""))
            category = {
                "model_visible_history_update": "history_update",
                "next_prompt_input_rebuilt_from_history": "build_prompt",
                "drain_in_flight": "tool_call_result_recording",
            }.get(boundary_name)
            if category is None:
                continue
            key = _pair_key(category, boundary_name)
            if phase == "started":
                begin(key, row)
            elif phase == "completed":
                start_row = finish(key)
                if start_row is not None:
                    content = {
                        "boundary_name": boundary_name,
                        "item_count": payload.get("item_count"),
                        "pending_tool_calls": payload.get("pending_tool_calls"),
                    }
                    append_event(category, start_row, row, content)

        elif payload_type == "inference_started":
            inference_call_id = str(payload.get("inference_call_id", ""))
            if inference_call_id:
                begin(_pair_key("request_dispatch", inference_call_id), row)

    elapsed_s = round((_trace_row_time_s(rows[-1]) - _trace_row_time_s(rows[0])), 3)
    summary = {
        "source_file": str(source),
        "source_kind": "trace",
        "timing_rule": {
            "model_output": "output_item_started(message) -> output_item_completed(message)",
            "model_reasoning": "reasoning_item_started(reasoning) -> reasoning_item_completed(reasoning)",
            "function_call": "function_call_item_started -> function_call_item_completed",
            "tool_call": "tool_call_started -> tool_call_completed",
            "history_update": "model_visible_history_update.started -> model_visible_history_update.completed",
            "build_prompt": "next_prompt_input_rebuilt_from_history.started -> next_prompt_input_rebuilt_from_history.completed",
            "tool_call_result_recording": "drain_in_flight.started -> drain_in_flight.completed",
            "request_dispatch": "inference_started -> request_sent",
            "time_to_first_token": "request_sent -> output_item_started",
        },
        "assumptions": [
            "trace payloads are used to recover message, reasoning, function-call, and tool-call content",
            "tool_call_completed is accepted as tool_call_ended when present in the trace schema",
        ],
        "elapsed_s": elapsed_s,
        "counts": {key: counts[key] for key in TRACE_CATEGORY_ORDER},
        "totals_s": {key: round(totals[key], 3) for key in TRACE_CATEGORY_ORDER},
        "shares_pct": {
            key: round((totals[key] * 100.0 / elapsed_s), 2) if elapsed_s else 0.0 for key in TRACE_CATEGORY_ORDER
        },
        "tool_totals_s": {key: round(value, 3) for key, value in sorted(tool_totals.items(), key=lambda item: (-item[1], item[0]))},
        "function_totals_s": {key: round(value, 3) for key, value in sorted(function_totals.items(), key=lambda item: (-item[1], item[0]))},
    }
    return {"summary": summary, "events": events}


def build_timing_summary_from_canonical(canonical_path: str | Path) -> Dict[str, Any]:
    source = Path(canonical_path)
    rows = _load_canonical_rows(source)
    events: List[Dict[str, Any]] = []
    counts = {"model_output": 0, "model_reasoning": 0, "tool_call": 0}
    tool_totals: Dict[str, float] = {}
    first_ts = _canonical_ts(rows[0]) if rows else None
    last_ts = first_ts
    for row in rows:
        typ = str(row.get("type") or "")
        ts = _canonical_ts(row)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        rel = round((ts - first_ts), 3) if ts is not None and first_ts is not None else 0.0
        if typ == "assistant_text":
            events.append(
                {
                    "kind": "model_output",
                    "time_s": rel,
                    "line": row.get("_line"),
                    "timestamp": row.get("timestamp"),
                    "content": row.get("text", ""),
                }
            )
            counts["model_output"] += 1
        elif typ == "assistant_reasoning":
            events.append(
                {
                    "kind": "model_reasoning",
                    "time_s": rel,
                    "line": row.get("_line"),
                    "timestamp": row.get("timestamp"),
                    "content": row.get("text", ""),
                }
            )
            counts["model_reasoning"] += 1
        elif typ == "tool_call":
            tool_name = str(row.get("name") or "unknown")
            events.append(
                {
                    "kind": "tool_call",
                    "time_s": rel,
                    "line": row.get("_line"),
                    "timestamp": row.get("timestamp"),
                    "content": {
                        "tool_name": tool_name,
                        "arguments": row.get("input", {}),
                    },
                }
            )
            counts["tool_call"] += 1
            tool_totals.setdefault(tool_name, 0.0)
    elapsed_s = round((last_ts - first_ts), 3) if first_ts is not None and last_ts is not None else 0.0
    summary = {
        "source_file": str(source),
        "source_kind": "canonical",
        "timing_rule": {
            "model_output": "canonical assistant_text event timestamp when available",
            "model_reasoning": "canonical assistant_reasoning event timestamp when available",
            "tool_call": "canonical tool_call event timestamp when available",
        },
        "assumptions": [
            "canonical fallback preserves event order and relative timestamps when backend timestamps are present",
            "tool duration is unavailable in canonical-only mode, so per-tool totals are reported as zero",
        ],
        "elapsed_s": elapsed_s,
        "counts": counts,
        "totals_s": {"model_output": 0.0, "model_reasoning": 0.0, "tool_call": 0.0},
        "shares_pct": {"model_output": 0.0, "model_reasoning": 0.0, "tool_call": 0.0},
        "tool_totals_s": {key: 0.0 for key in sorted(tool_totals)},
        "tool_shares_pct": {key: 0.0 for key in sorted(tool_totals)},
    }
    return {"summary": summary, "events": events}


def _load_canonical_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            data["_line"] = idx
            rows.append(data)
    return rows


def _canonical_ts(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("timestamp")
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        return raw / 1000.0 if raw > 10_000_000_000 else raw
    if isinstance(value, str) and value.strip():
        try:
            return _parse_iso_ts(value).timestamp()
        except ValueError:
            return None
    return None


def build_timing_summary(source_path: str | Path) -> Dict[str, Any]:
    source = Path(source_path)
    if source.name == "trace.jsonl":
        return build_timing_summary_from_trace(source)
    if source.name == "canonical.jsonl":
        return build_timing_summary_from_canonical(source)
    return build_timing_summary_from_session(source)


def _discover_trace_file(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("trace/**/trace.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def generate_timing_summary_artifacts(
    run_dir: str | Path,
    *,
    session_path: str | Path | None = None,
    json_out: str | Path | None = None,
    image_out: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    run_path = Path(run_dir)
    explicit = Path(session_path) if session_path else None
    trace_file = explicit if explicit and explicit.name == "trace.jsonl" else _discover_trace_file(run_path)
    source_file = trace_file or explicit or (run_path / "codex_session.jsonl")
    if not source_file.is_file() and (run_path / "canonical.jsonl").is_file():
        source_file = run_path / "canonical.jsonl"
    if not source_file.is_file():
        return None
    result = build_timing_summary(source_file)
    json_path = Path(json_out) if json_out else run_path / "timing_summary.json"
    image_path = Path(image_out) if image_out else run_path / "timing_summary.svg"
    write_json(json_path, result)
    render_timing_summary_image(result, image_path)
    return {
        "source_path": str(source_file),
        "json_path": str(json_path),
        "image_path": str(image_path),
        "summary": result.get("summary", {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate timing summary JSON and image for a Codex run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--session", default=None, help="Optional explicit codex_session.jsonl or trace.jsonl path")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-image", default=None)
    args = parser.parse_args(argv)

    result = generate_timing_summary_artifacts(
        args.run_dir,
        session_path=args.session,
        json_out=args.out_json,
        image_out=args.out_image,
    )
    if result is None:
        raise SystemExit(f"no codex session or trace log found for run: {args.run_dir}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
