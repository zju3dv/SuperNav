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

"""Trace helpers for MCP tool-call request and execution events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.agents.harness.native_turn_tracing import TraceEmitter
from dimos.agents.harness.trace import HarnessTraceEvent, emit_trace_event
from dimos.agents.skill_result import SkillResult


def emit_tool_call_requested_trace(
    logger: Any,
    *,
    invocation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    client_id: str,
    request_id: Any = None,
    progress_token: Any = None,
    uses: list[str] | None = None,
    lifecycle: str | None = None,
    found: bool = True,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a trace event for a raw MCP ``tools/call`` request."""

    return trace_emit(
        logger,
        "tool_call_requested",
        invocation_id=invocation_id,
        tool_name=tool_name,
        client_id=client_id,
        request_id=request_id,
        progress_token=_optional_str(progress_token),
        uses=uses or [],
        lifecycle=lifecycle,
        found=found,
        arguments=_argument_summary(arguments),
    )


def emit_tool_execution_start_trace(
    logger: Any,
    *,
    invocation_id: str,
    tool_name: str,
    client_id: str,
    uses: list[str],
    lifecycle: str,
    progress_token: Any = None,
    movement_ticket: Any = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a trace event immediately before invoking the RPC skill."""

    payload: dict[str, Any] = {
        "invocation_id": invocation_id,
        "tool_name": tool_name,
        "client_id": client_id,
        "uses": uses,
        "lifecycle": lifecycle,
        "progress_token": _optional_str(progress_token),
    }
    ticket_summary = _movement_ticket_summary(movement_ticket)
    if ticket_summary is not None:
        payload["movement_ticket"] = ticket_summary
    return trace_emit(logger, "tool_execution_start", **payload)


def emit_tool_execution_result_trace(
    logger: Any,
    *,
    invocation_id: str,
    tool_name: str,
    client_id: str,
    success: bool,
    duration_ms: float,
    result: Any = None,
    content_blocks: list[dict[str, Any]] | None = None,
    error: str | None = None,
    movement_ticket: Any = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a compact, trace-safe result for an executed MCP tool call."""

    tool_success = _effective_tool_success(rpc_success=success, result=result)
    payload: dict[str, Any] = {
        "invocation_id": invocation_id,
        "tool_name": tool_name,
        "client_id": client_id,
        "success": tool_success,
        "rpc_success": success,
        "duration_ms": round(duration_ms, 1),
        "result": _result_summary(result=result, content_blocks=content_blocks),
    }
    skill_outcome = _skill_outcome(result)
    if skill_outcome is not None:
        payload.update(skill_outcome)
    if error is not None:
        payload["error"] = _safe_preview(error)
    ticket_summary = _movement_ticket_summary(movement_ticket)
    if ticket_summary is not None:
        payload["movement_ticket"] = ticket_summary
    return trace_emit(logger, "tool_execution_result", **payload)


def emit_tool_call_refused_trace(
    logger: Any,
    *,
    invocation_id: str,
    tool_name: str,
    client_id: str,
    refusal_source: str,
    reason: str,
    request_id: Any = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a terminal trace event for a requested tool that was not executed."""

    return trace_emit(
        logger,
        "tool_call_refused",
        invocation_id=invocation_id,
        tool_name=tool_name,
        client_id=client_id,
        request_id=request_id,
        refusal_source=refusal_source,
        reason=_safe_preview(reason),
    )


def _argument_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "arg_keys": sorted(str(key) for key in arguments.keys()),
        "arg_count": len(arguments),
        "contains_multimodal_payload": _contains_multimodal_payload(arguments),
    }


def _result_summary(
    *,
    result: Any = None,
    content_blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if isinstance(result, SkillResult):
        return {
            "kind": "SkillResult",
            "skill_success": result.success,
            "error_code": result.error_code,
            "message_chars": len(result.message),
            "message_preview": _safe_preview(result.message),
            "metadata_keys": sorted(str(key) for key in result.metadata.keys()),
            "image_blocks": 0,
        }
    if content_blocks is not None:
        return {
            "kind": "content_blocks",
            "blocks": len(content_blocks),
            "text_chars": sum(_text_chars(block) for block in content_blocks),
            "image_blocks": sum(1 for block in content_blocks if _is_image_block(block)),
        }
    text = "" if result is None else str(result)
    return {
        "kind": result.__class__.__name__ if result is not None else "none",
        "text_chars": len(text),
        "text_preview": _safe_preview(text),
        "image_blocks": 0,
    }


def _effective_tool_success(*, rpc_success: bool, result: Any = None) -> bool:
    """Return the outcome the agent/auditor should trust for the tool call."""

    if not rpc_success:
        return False
    if isinstance(result, SkillResult):
        return result.success
    return True


def _skill_outcome(result: Any = None) -> dict[str, Any] | None:
    if not isinstance(result, SkillResult):
        return None
    return {
        "skill_success": result.success,
        "skill_error_code": result.error_code,
    }


def _text_chars(block: dict[str, Any]) -> int:
    text = block.get("text")
    return len(text) if isinstance(text, str) else 0


def _is_image_block(block: Mapping[str, Any]) -> bool:
    block_type = str(block.get("type") or "")
    if block_type in {"image", "image_url", "input_image"}:
        return True
    return any(key in block for key in ("image_url", "source", "data"))


def _contains_multimodal_payload(value: Any) -> bool:
    if isinstance(value, str):
        return "data:image" in value or "base64," in value
    if isinstance(value, Mapping):
        if _is_image_block(value):
            return True
        return any(_contains_multimodal_payload(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_multimodal_payload(item) for item in value)
    return False


def _safe_preview(text: str, *, max_chars: int = 500) -> str:
    if "data:image" in text or "base64," in text:
        return "[redacted multimodal payload]"
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _movement_ticket_summary(ticket: Any) -> dict[str, Any] | None:
    if ticket is None:
        return None
    if hasattr(ticket, "model_dump"):
        payload = ticket.model_dump()
    elif isinstance(ticket, Mapping):
        payload = dict(ticket)
    else:
        return None
    return {
        "client_id": payload.get("client_id"),
        "lease_id": payload.get("lease_id"),
        "epoch": payload.get("epoch"),
        "tool_name": payload.get("tool_name"),
        "cost": payload.get("cost"),
    }
