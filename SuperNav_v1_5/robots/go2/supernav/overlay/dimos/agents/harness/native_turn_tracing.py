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

"""Trace payload helpers for native DimOS agent turns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.context_manager import ContextAssembly
from dimos.agents.harness.native_model_stream import NativeModelStreamResult
from dimos.agents.harness.native_task_terminal import NativeTaskTerminalDirective
from dimos.agents.harness.native_turn_preparation import NativeTurnPreparation
from dimos.agents.harness.request_assembly import NativeModelRequest
from dimos.agents.harness.task_control import TaskControlDecision
from dimos.agents.harness.task_followup import task_followup_summary
from dimos.agents.harness.task_state import TaskStateUpdate
from dimos.agents.harness.tool_status import ToolStatusMessage
from dimos.agents.harness.trace import HarnessTraceEvent, emit_trace_event

TraceEmitter = Callable[..., HarnessTraceEvent]


def turn_start_trace_payload(
    *,
    turn_id: str,
    message: BaseMessage,
    queue_empty: bool,
    task_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return payload for a ``turn_start`` harness trace event."""

    return {
        "turn_id": turn_id,
        "message_type": _message_type(message),
        "message_source": _message_source(message),
        "queue_empty_at_start": queue_empty,
        "active_task": task_snapshot.get("active_task"),
    }


def turn_end_trace_payload(
    *,
    turn_id: str,
    outcome: str,
    queue_empty: bool,
    task_snapshot: dict[str, Any],
    model_updates: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    """Return payload for a ``turn_end`` harness trace event."""

    payload: dict[str, Any] = {
        "turn_id": turn_id,
        "outcome": outcome,
        "queue_empty_at_end": queue_empty,
        "model_updates": model_updates,
        "active_task": task_snapshot.get("active_task"),
        "follow_up": task_followup_summary(task_snapshot.get("active_task")),
    }
    if error:
        payload["error"] = error
    return payload


def task_state_trace_payload(
    update: TaskStateUpdate,
    *,
    task_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return payload for a ``task_state_updated`` harness trace event."""

    return {
        "update": update.model_dump(),
        "active_task": task_snapshot.get("active_task"),
    }


def task_control_trace_payload(
    update: TaskStateUpdate | None,
    decision: TaskControlDecision,
    *,
    task_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return payload for a ``task_control_command`` harness trace event."""

    return {
        "command": update.command if update is not None else None,
        "route": decision.route,
        "update": update.model_dump() if update is not None else None,
        "response_text": decision.response_text,
        "active_task": task_snapshot.get("active_task"),
    }


def task_terminal_directive_trace_payload(
    directive: NativeTaskTerminalDirective,
    update: TaskStateUpdate,
    *,
    turn_id: str,
    task_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return payload for a strict model-emitted task terminal directive."""

    return {
        "turn_id": turn_id,
        "directive": directive.model_dump(),
        "update": update.model_dump(),
        "active_task": task_snapshot.get("active_task"),
        "last_terminal_task": task_snapshot.get("last_terminal_task"),
    }


def request_assembled_trace_payload(request: NativeModelRequest) -> dict[str, Any]:
    """Return payload for a ``request_assembled`` harness trace event."""

    return request.trace_payload()


def context_budget_exhausted_trace_payload(assembly: ContextAssembly) -> dict[str, Any]:
    """Return payload for a ``context_budget_exhausted`` harness trace event."""

    return {
        "assembly": assembly.model_dump(),
        "reason": assembly.failure_reason,
    }


def context_compacted_trace_payload(
    assembly: ContextAssembly,
    *,
    turn_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return payload for a ``context_compacted`` harness trace event."""

    payload: dict[str, Any] = {
        "assembly": assembly.model_dump(),
        "dropped": assembly.dropped,
        "total_history_messages": assembly.total_history_messages,
        "included_messages": assembly.included_messages,
        "estimated_tokens": assembly.estimated_tokens,
        "token_budget": assembly.token_budget,
        "budget_exhausted": assembly.budget_exhausted,
    }
    if turn_id is not None:
        payload["turn_id"] = turn_id
    if model is not None:
        payload["model"] = model
    return payload


def model_response_trace_payload(
    *,
    turn_id: str,
    result: NativeModelStreamResult,
    model: str | None = None,
) -> dict[str, Any]:
    """Return payload for a compact ``model_response`` harness trace event."""

    payload: dict[str, Any] = {
        "turn_id": turn_id,
        "update_count": result.update_count,
        "tool_call_count": result.tool_call_count,
        "messages": result.messages,
    }
    if result.terminal_directives:
        payload["terminal_directives"] = [
            directive.model_dump() for directive in result.terminal_directives
        ]
    if model is not None:
        payload["model"] = model
    return payload


def emit_native_turn_preparation_traces(
    logger: Any,
    prepared: NativeTurnPreparation,
    *,
    task_snapshot: dict[str, Any],
    trace_emit: TraceEmitter = emit_trace_event,
) -> list[HarnessTraceEvent]:
    """Emit task-state and task-control traces for a prepared native turn."""

    events: list[HarnessTraceEvent] = []
    if prepared.task_update is not None:
        events.append(
            trace_emit(
                logger,
                "task_state_updated",
                **task_state_trace_payload(prepared.task_update, task_snapshot=task_snapshot),
            )
        )
    if prepared.control_decision is not None:
        events.append(
            trace_emit(
                logger,
                "task_control_command",
                **task_control_trace_payload(
                    prepared.task_update,
                    prepared.control_decision,
                    task_snapshot=task_snapshot,
                ),
            )
        )
    return events


def emit_task_terminal_directive_trace(
    logger: Any,
    directive: NativeTaskTerminalDirective,
    update: TaskStateUpdate,
    *,
    turn_id: str,
    task_snapshot: dict[str, Any],
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a task-terminal trace for a strict model directive."""

    return trace_emit(
        logger,
        "task_terminal_directive",
        **task_terminal_directive_trace_payload(
            directive,
            update,
            turn_id=turn_id,
            task_snapshot=task_snapshot,
        ),
    )


def emit_turn_start_trace(
    logger: Any,
    *,
    turn_id: str,
    message: BaseMessage,
    queue_empty: bool,
    task_snapshot: dict[str, Any],
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a native-agent turn-start trace event."""

    return trace_emit(
        logger,
        "turn_start",
        **turn_start_trace_payload(
            turn_id=turn_id,
            message=message,
            queue_empty=queue_empty,
            task_snapshot=task_snapshot,
        ),
    )


def emit_turn_end_trace(
    logger: Any,
    *,
    turn_id: str,
    outcome: str,
    queue_empty: bool,
    task_snapshot: dict[str, Any],
    model_updates: int = 0,
    error: str | None = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a native-agent turn-end trace event."""

    return trace_emit(
        logger,
        "turn_end",
        **turn_end_trace_payload(
            turn_id=turn_id,
            outcome=outcome,
            queue_empty=queue_empty,
            task_snapshot=task_snapshot,
            model_updates=model_updates,
            error=error,
        ),
    )


def emit_request_assembled_trace(
    logger: Any,
    request: NativeModelRequest,
    *,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a request-assembly trace event."""

    return trace_emit(
        logger,
        "request_assembled",
        **request_assembled_trace_payload(request),
    )


def emit_context_budget_exhausted_trace(
    logger: Any,
    assembly: ContextAssembly,
    *,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a context-budget fail-closed trace event."""

    return trace_emit(
        logger,
        "context_budget_exhausted",
        **context_budget_exhausted_trace_payload(assembly),
    )


def emit_context_compacted_trace(
    logger: Any,
    assembly: ContextAssembly,
    *,
    turn_id: str | None = None,
    model: str | None = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a trace event when request assembly drops/compacts context."""

    return trace_emit(
        logger,
        "context_compacted",
        **context_compacted_trace_payload(assembly, turn_id=turn_id, model=model),
    )


def emit_model_response_trace(
    logger: Any,
    *,
    turn_id: str,
    result: NativeModelStreamResult,
    model: str | None = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a compact model-response trace event for one native turn."""

    return trace_emit(
        logger,
        "model_response",
        **model_response_trace_payload(turn_id=turn_id, result=result, model=model),
    )


def _message_type(message: BaseMessage) -> str:
    return str(getattr(message, "type", message.__class__.__name__))


def _message_source(message: BaseMessage) -> str:
    if isinstance(message, ToolStatusMessage):
        return "tool_status"
    marker = getattr(message, "additional_kwargs", {}).get("dimos/native_source")
    if isinstance(marker, str) and marker.startswith("dimos_native:"):
        return marker
    return "external_user"
