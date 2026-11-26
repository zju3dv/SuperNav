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

"""Session lifecycle trace helpers for the native DimOS agent."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from dimos.agents.harness.model_provider import (
    NativeAgentModelConfig,
    native_agent_model_readiness,
)
from dimos.agents.harness.native_turn_tracing import TraceEmitter
from dimos.agents.harness.trace import HarnessTraceEvent, emit_trace_event
from dimos.agents.runtime_policy import AgentRuntimePolicy


def new_native_session_id() -> str:
    """Return a compact process-local native-agent session id."""

    return f"session_{uuid4().hex[:12]}"


def session_start_trace_payload(
    *,
    session_id: str,
    policy: AgentRuntimePolicy,
    model_config: NativeAgentModelConfig,
    mcp_server_url: str,
    context_manager: Any,
) -> dict[str, Any]:
    """Return trace payload for a native-agent session start."""

    readiness = native_agent_model_readiness(model_config)
    return {
        "session_id": session_id,
        "runtime_policy": policy.model_dump(),
        "model": {
            "name": model_config.model,
            "provider": model_config.model_provider,
            "context_target_provider": readiness.context_target_provider,
            "base_url_configured": readiness.base_url_configured,
            "base_url_summary": readiness.base_url_summary,
            "api_key_configured": readiness.api_key_configured,
            "fixture_configured": bool(model_config.model_fixture),
            "ready_for_native_agent": readiness.ready_for_native_agent,
            "ready_for_provider_vision_roundtrip": (readiness.ready_for_provider_vision_roundtrip),
            "readiness_checks": readiness.checks,
        },
        "mcp_server_url": mcp_server_url,
        "context": _context_config(context_manager),
    }


def session_end_trace_payload(
    *,
    session_id: str,
    reason: str,
    queue_size: int | None = None,
    task_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace payload for a native-agent session end."""

    payload: dict[str, Any] = {
        "session_id": session_id,
        "reason": reason,
    }
    if queue_size is not None:
        payload["queue_size"] = queue_size
    if task_snapshot is not None:
        payload["active_task"] = task_snapshot.get("active_task")
        payload["last_terminal_task"] = task_snapshot.get("last_terminal_task")
    return payload


def emit_native_session_start_trace(
    logger: Any,
    *,
    session_id: str,
    policy: AgentRuntimePolicy,
    model_config: NativeAgentModelConfig,
    mcp_server_url: str,
    context_manager: Any,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a native-agent session-start trace event."""

    return trace_emit(
        logger,
        "session_start",
        **session_start_trace_payload(
            session_id=session_id,
            policy=policy,
            model_config=model_config,
            mcp_server_url=mcp_server_url,
            context_manager=context_manager,
        ),
    )


def emit_native_session_end_trace(
    logger: Any,
    *,
    session_id: str,
    reason: str,
    queue_size: int | None = None,
    task_snapshot: dict[str, Any] | None = None,
    trace_emit: TraceEmitter = emit_trace_event,
) -> HarnessTraceEvent:
    """Emit a native-agent session-end trace event."""

    return trace_emit(
        logger,
        "session_end",
        **session_end_trace_payload(
            session_id=session_id,
            reason=reason,
            queue_size=queue_size,
            task_snapshot=task_snapshot,
        ),
    )


def _context_config(context_manager: Any) -> dict[str, Any]:
    return {
        "max_context_tokens": getattr(context_manager, "max_context_tokens", None),
        "max_recent_messages": getattr(context_manager, "max_recent_messages", None),
        "max_tool_result_chars": getattr(context_manager, "max_tool_result_chars", None),
        "max_image_blocks": getattr(context_manager, "max_image_blocks", None),
        "target_provider": getattr(context_manager, "target_provider", None),
        "target_model": getattr(context_manager, "target_model", None),
    }
