from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def iter_json_lines(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("text", "content", "thinking"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
        return "\n".join(parts)
    return ""


def _time_fields(time_obj: Any) -> Dict[str, Any]:
    if not isinstance(time_obj, dict):
        return {}
    fields: Dict[str, Any] = {}
    if time_obj.get("start") is not None:
        fields["time_start"] = time_obj.get("start")
    if time_obj.get("end") is not None:
        fields["time_end"] = time_obj.get("end")
    return fields


def _clean_tool_name(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return (
        name.replace("mcp__habitat-gs__", "")
        .replace("mcp__habitat_gs__", "")
        .replace("habitat-gs.", "")
        .replace("habitat-gs_", "")
        .replace("habitat_gs.", "")
        .replace("habitat_gs_", "")
    )


def _reasoning_summary_text(summary: Any) -> str:
    if isinstance(summary, str):
        return summary.strip()
    if not isinstance(summary, list):
        return ""
    parts: List[str] = []
    for item in summary:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        elif isinstance(item, dict):
            for key in ("text", "summary", "content"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
    return "\n".join(parts)


def session_id_from_events(events: Iterable[Mapping[str, Any]]) -> Optional[str]:
    rows = list(events)
    # Re-initialization can leave earlier simulator sessions behind. The
    # session explicitly returned by close_session is the episode that owns
    # the completed trajectory and replay frames.
    for event in reversed(rows):
        if (
            event.get("type") != "tool_result"
            or event.get("name") != "hab_close_session"
        ):
            continue
        text = json.dumps(event, ensure_ascii=False, default=str)
        match = UUID_RE.search(text)
        if match:
            return match.group(0)
    for event in rows:
        text = json.dumps(event, ensure_ascii=False, default=str)
        match = UUID_RE.search(text)
        if match:
            return match.group(0)
    return None


def tool_result_payload(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse the first JSON text block from one canonical tool result."""

    audit = event.get("audit")
    if isinstance(audit, dict):
        return dict(audit)
    content = str(event.get("content") or "").strip()
    if not content:
        return {}
    try:
        payload = json.loads(content.splitlines()[0])
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def augment_canonical_events(
    events: Iterable[Mapping[str, Any]],
    *,
    audit_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Merge private MCP audit payloads and materialize memory transitions.

    Pairing uses the exact model-visible result hash plus tool name.  A missing
    or ambiguous sidecar never causes positional guessing.
    """

    rows = [dict(event) for event in events]
    audit_rows = _load_audit_rows(audit_path)
    audit_queues: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for record in audit_rows:
        key = (
            str(record.get("tool_name") or ""),
            str(record.get("model_content_sha256") or ""),
        )
        if all(key):
            audit_queues.setdefault(key, []).append(record)

    used_tool_seqs: set[int] = set()
    existing_memory_ids = {
        str(event.get("event_id"))
        for event in rows
        if event.get("type") == "spatial_memory" and event.get("event_id")
    }
    merged: List[Dict[str, Any]] = []
    for event in rows:
        if event.get("type") != "tool_result":
            merged.append(event)
            continue
        content = str(event.get("content") or "").strip()
        model_text = content.splitlines()[0] if content else ""
        key = (
            str(event.get("name") or ""),
            hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        )
        candidates = audit_queues.get(key, [])
        if candidates:
            record = candidates.pop(0)
            result = record.get("result")
            if isinstance(result, dict):
                event["audit"] = result
            tool_seq = record.get("tool_seq")
            if isinstance(tool_seq, int):
                event["audit_tool_seq"] = tool_seq
                used_tool_seqs.add(tool_seq)
            event["audit_merge_status"] = "matched"
        elif audit_rows:
            event["audit_merge_status"] = "unmatched"
        merged.append(event)
        memory = tool_result_payload(event).get("spatial_memory")
        new_events = memory.get("new_events") if isinstance(memory, dict) else None
        if not isinstance(new_events, list):
            continue
        for memory_event in new_events:
            if not isinstance(memory_event, dict):
                continue
            event_id = str(memory_event.get("event_id") or "")
            if not event_id or event_id in existing_memory_ids:
                continue
            existing_memory_ids.add(event_id)
            materialized = {
                "type": "spatial_memory",
                **memory_event,
                "source_tool": event.get("name"),
                "source_raw_index": event.get("raw_index"),
                "timestamp": event.get("time_end") or event.get("timestamp"),
            }
            transition_time = event.get("time_end") or event.get("timestamp")
            if transition_time is not None:
                materialized["time_start"] = transition_time
                materialized["time_end"] = transition_time
            merged.append(materialized)

    unmatched = sorted(
        int(record["tool_seq"])
        for records in audit_queues.values()
        for record in records
        if isinstance(record.get("tool_seq"), int)
        and int(record["tool_seq"]) not in used_tool_seqs
    )
    if unmatched:
        merged.append(
            {
                "type": "audit_merge_diagnostic",
                "status": "unmatched_audit_records",
                "tool_seqs": unmatched,
                "timestamp": None,
            }
        )
    return merged


def write_canonical_events(
    path: str | Path, events: Iterable[Mapping[str, Any]]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(
                json.dumps(dict(event), ensure_ascii=False, default=str) + "\n"
            )


def _load_audit_rows(path: str | Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    audit_path = Path(path)
    if not audit_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for record in iter_json_lines(audit_path):
        if record.get("schema_version") == 1 and isinstance(record.get("result"), dict):
            rows.append(record)
    rows.sort(key=lambda row: int(row.get("tool_seq") or 0))
    return rows
