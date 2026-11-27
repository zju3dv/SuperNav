from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from supernav.runtime.support.file_io import append_jsonl_atomic

TOOL_DISPATCH_TRACE_FILENAME = "tool_dispatch_trace.jsonl"
TOOL_DISPATCH_TRACE_SCHEMA_VERSION = 1


def record_tool_dispatch_fail_open(
    *,
    ctx: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    tool: Any = None,
) -> None:
    """Record one dispatch row without letting capture affect execution."""

    try:
        path = _dispatch_trace_path(ctx)
        if not path:
            return
        append_jsonl_atomic(path, _build_record(ctx, tool_name, args, result, tool))
    except Exception:
        return


def _dispatch_trace_path(ctx: Any) -> str:
    output_dir = str(getattr(ctx, "output_dir", "") or "")
    is_mcp = str(getattr(ctx, "dispatch_source", "") or "") == "mcp"
    configured = str(getattr(ctx, "dispatch_trace_path", "") or "")
    if configured:
        if is_mcp and _path_is_inside(configured, output_dir):
            return ""
        return configured
    controller_dir = str(getattr(ctx, "controller_measurement_dir", "") or "")
    if controller_dir:
        path = str(Path(controller_dir) / TOOL_DISPATCH_TRACE_FILENAME)
        if is_mcp and _path_is_inside(path, output_dir):
            return ""
        return path
    if is_mcp:
        return ""
    if not output_dir:
        return ""
    return str(Path(output_dir) / TOOL_DISPATCH_TRACE_FILENAME)


def _path_is_inside(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def _build_record(
    ctx: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    tool: Any,
) -> dict[str, Any]:
    body = getattr(result, "body", {})
    metadata = getattr(tool, "metadata", None)
    return {
        "schema_version": TOOL_DISPATCH_TRACE_SCHEMA_VERSION,
        "event": "tool_dispatch",
        "ts": _utc_now_iso(),
        "tool": tool_name,
        "args": _json_safe(dict(args)),
        "caller": {
            "source": str(getattr(ctx, "dispatch_source", "in_process") or "unknown"),
            "harness": str(getattr(ctx, "harness", "") or ""),
            "nav_mode": str(getattr(ctx, "nav_mode", "") or ""),
            "task_type": str(getattr(ctx, "task_type", "") or ""),
            "session_id": str(getattr(ctx, "session_id", "") or ""),
            "loop_id": str(getattr(ctx, "loop_id", "") or ""),
        },
        "tool_metadata": _metadata_snapshot(metadata),
        "outcome": {
            "ok": bool(getattr(result, "ok", False)),
            "error": getattr(result, "error", None),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0),
            "body_keys": sorted(body.keys()) if isinstance(body, Mapping) else [],
            "captured_images": _json_safe(getattr(result, "captured_images", []) or []),
        },
        "result_body": _json_safe(body),
    }


def _metadata_snapshot(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    category = getattr(metadata, "category", None)
    permission = getattr(metadata, "permission", None)
    return {
        "category": getattr(category, "value", str(category)) if category else "",
        "permission": getattr(permission, "value", str(permission)) if permission else "",
        "allowed_nav_modes": sorted(getattr(metadata, "allowed_nav_modes", []) or []),
        "allowed_task_types": sorted(getattr(metadata, "allowed_task_types", []) or []),
        "allowed_harness_modes": sorted(
            getattr(metadata, "allowed_harness_modes", []) or []
        ),
        "mcp_visible": bool(getattr(metadata, "mcp_visible", False)),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
