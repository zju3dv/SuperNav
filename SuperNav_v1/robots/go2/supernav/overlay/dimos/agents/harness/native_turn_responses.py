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

"""Local native-turn responses for DimOS fallback agent control paths."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.context_manager import ContextAssembly
from dimos.agents.harness.native_agent_io import NativeAgentIO
from dimos.agents.harness.native_turn_tracing import (
    TraceEmitter,
    emit_context_budget_exhausted_trace,
)
from dimos.agents.harness.task_control import TaskControlDecision

_CONTEXT_BUDGET_EXHAUSTED_TEXT = (
    "I cannot safely call the model because the assembled context exceeds "
    "the configured token budget. Please send a shorter request or clear "
    "older context before retrying."
)


def publish_task_control_response(
    *,
    context_manager: object,
    agent_io: NativeAgentIO,
    message: BaseMessage,
    decision: TaskControlDecision,
) -> AIMessage:
    """Publish a local task-control response without calling the model."""

    normalized_message = context_manager.normalize_message(message, source="input")
    return agent_io.publish_local_response(
        normalized_message,
        decision.response_text or "",
    )


def publish_context_budget_exhausted_response(
    *,
    logger: object,
    trace_emit: TraceEmitter,
    agent_io: NativeAgentIO,
    history: list[BaseMessage],
    assembly: ContextAssembly,
) -> AIMessage:
    """Trace and publish the fail-closed response for an over-budget request."""

    emit_context_budget_exhausted_trace(logger, assembly, trace_emit=trace_emit)
    response = agent_io.publish_text_response(_CONTEXT_BUDGET_EXHAUSTED_TEXT)
    history.append(response)
    agent_io.set_idle()
    return response
