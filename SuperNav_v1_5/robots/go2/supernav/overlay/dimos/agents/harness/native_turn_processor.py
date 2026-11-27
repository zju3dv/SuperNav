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

"""Native DimOS fallback-agent turn orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.context_manager import ConversationContextManager
from dimos.agents.harness.native_agent_io import NativeAgentIO
from dimos.agents.harness.native_auto_followup import (
    DEFAULT_MAX_AUTO_FOLLOWUPS_PER_TASK,
    DEFAULT_MAX_NAVIGATION_STATUS_FOLLOWUPS_PER_TASK,
    NativeAutoFollowupState,
    decide_native_auto_followup,
)
from dimos.agents.harness.native_input_history import record_native_input_turn
from dimos.agents.harness.native_model_stream import (
    NativeStateGraph,
    publish_native_model_stream,
)
from dimos.agents.harness.native_task_terminal import terminal_preemption_reason
from dimos.agents.harness.native_turn_preparation import prepare_native_turn
from dimos.agents.harness.native_turn_responses import (
    publish_context_budget_exhausted_response,
    publish_task_control_response,
)
from dimos.agents.harness.native_turn_tracing import (
    TraceEmitter,
    emit_context_compacted_trace,
    emit_model_response_trace,
    emit_native_turn_preparation_traces,
    emit_request_assembled_trace,
    emit_task_terminal_directive_trace,
    emit_turn_end_trace,
    emit_turn_start_trace,
)
from dimos.agents.harness.request_assembly import NativeRequestAssembler
from dimos.agents.harness.speech_policy import NativeSpeechGate
from dimos.agents.harness.task_state import TaskStateManager
from dimos.agents.harness.tool_status import ToolStatusMessage
from dimos.agents.harness.trace import emit_trace_event
from dimos.agents.runtime_policy import AgentRuntimePolicy


@dataclass(frozen=True)
class NativeTurnRuntimeState:
    """Mutable native-agent state summarized for one turn processor call."""

    last_user_message_text: str = ""
    movement_confirmed: bool = False
    movement_authorization: str | None = None
    auto_followup: NativeAutoFollowupState = field(default_factory=NativeAutoFollowupState)

    def with_auto_followup(
        self,
        auto_followup: NativeAutoFollowupState,
    ) -> NativeTurnRuntimeState:
        """Return this runtime state with updated auto-follow-up counters."""

        return NativeTurnRuntimeState(
            last_user_message_text=self.last_user_message_text,
            movement_confirmed=self.movement_confirmed,
            movement_authorization=self.movement_authorization,
            auto_followup=auto_followup,
        )


@dataclass(frozen=True)
class NativeTurnProcessResult:
    """Outcome of processing one queued native-agent message."""

    runtime_state: NativeTurnRuntimeState
    handled_status: bool = False
    local_response: bool = False
    budget_exhausted: bool = False
    model_updates: int = 0


def process_native_turn(
    *,
    state_graph: NativeStateGraph,
    message: BaseMessage,
    policy: AgentRuntimePolicy,
    speech_gate: NativeSpeechGate,
    task_state: TaskStateManager,
    context_manager: ConversationContextManager,
    history: list[BaseMessage],
    agent_io: NativeAgentIO,
    runtime_state: NativeTurnRuntimeState,
    queue_empty: Callable[[], bool],
    tool_names: list[str] | None,
    turn_id: str,
    logger: Any,
    tool_details: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
    runtime_state_update: Callable[[NativeTurnRuntimeState], None] | None = None,
    runtime_state_snapshot: Callable[[], NativeTurnRuntimeState] | None = None,
    enqueue_message: Callable[[BaseMessage], None] | None = None,
    terminal_task_preempt: Callable[[str], Any] | None = None,
    max_auto_followups_per_task: int = DEFAULT_MAX_AUTO_FOLLOWUPS_PER_TASK,
    max_navigation_status_followups_per_task: int = (
        DEFAULT_MAX_NAVIGATION_STATUS_FOLLOWUPS_PER_TASK
    ),
    trace_emit: TraceEmitter = emit_trace_event,
) -> NativeTurnProcessResult:
    """Process one queued native-agent message without owning transport details."""

    emit_turn_start_trace(
        logger,
        turn_id=turn_id,
        message=message,
        queue_empty=queue_empty(),
        task_snapshot=task_state.snapshot(),
        trace_emit=trace_emit,
    )

    def finish(
        outcome: str,
        *,
        result: NativeTurnProcessResult,
        model_updates: int = 0,
        error: str | None = None,
    ) -> NativeTurnProcessResult:
        emit_turn_end_trace(
            logger,
            turn_id=turn_id,
            outcome=outcome,
            queue_empty=queue_empty(),
            task_snapshot=task_state.snapshot(),
            model_updates=model_updates,
            error=error,
            trace_emit=trace_emit,
        )
        return result

    try:
        if isinstance(message, ToolStatusMessage):
            task_state.pin_tool_status(message)
            agent_io.publish_status(message)
            return finish(
                "tool_status",
                result=NativeTurnProcessResult(runtime_state=runtime_state, handled_status=True),
            )

        agent_io.set_busy()
        prepared = prepare_native_turn(
            message,
            policy=policy,
            speech_gate=speech_gate,
            task_state=task_state,
        )
        next_state = runtime_state
        if prepared.external_user_message:
            next_state = replace(
                runtime_state,
                last_user_message_text=prepared.user_message_text or "",
                movement_confirmed=prepared.movement_confirmed,
                movement_authorization=prepared.movement_authorization,
            )

        emit_native_turn_preparation_traces(
            logger,
            prepared,
            task_snapshot=task_state.snapshot(),
            trace_emit=trace_emit,
        )
        if (
            prepared.control_decision is not None
            and prepared.control_decision.route == "local_response"
        ):
            publish_task_control_response(
                context_manager=context_manager,
                agent_io=agent_io,
                message=message,
                decision=prepared.control_decision,
            )
            return finish(
                "local_response",
                result=NativeTurnProcessResult(
                    runtime_state=next_state,
                    local_response=True,
                ),
            )

        if runtime_state_update is not None:
            runtime_state_update(next_state)

        record_native_input_turn(
            context_manager=context_manager,
            history=history,
            model_message=prepared.model_message,
            original_message=message,
            agent_io=agent_io,
        )

        request = NativeRequestAssembler(
            context_manager=context_manager,
            task_context=task_state,
            policy=policy,
            tools_exposed=tool_names,
            tools_exposed_details=tool_details,
            turn_id=turn_id,
            model=model_name,
        ).assemble(history)
        emit_request_assembled_trace(logger, request, trace_emit=trace_emit)
        if request.context.dropped:
            emit_context_compacted_trace(
                logger,
                request.context,
                turn_id=turn_id,
                model=model_name,
                trace_emit=trace_emit,
            )
        if request.context.budget_exhausted:
            publish_context_budget_exhausted_response(
                logger=logger,
                trace_emit=trace_emit,
                agent_io=agent_io,
                history=history,
                assembly=request.context,
            )
            return finish(
                "context_budget_exhausted",
                result=NativeTurnProcessResult(
                    runtime_state=next_state,
                    budget_exhausted=True,
                ),
            )

        model_stream = publish_native_model_stream(
            state_graph,
            request_messages=request.messages,
            context_manager=context_manager,
            history=history,
            agent_io=agent_io,
        )
        task_state.pin_visual_artifacts(
            _image_artifact_ids_from_stream(
                context_manager,
                model_stream.artifact_ids,
            )
        )
        current_state = (
            runtime_state_snapshot() if runtime_state_snapshot is not None else next_state
        )
        emit_model_response_trace(
            logger,
            turn_id=turn_id,
            result=model_stream,
            model=model_name,
            trace_emit=trace_emit,
        )
        terminal_directive = model_stream.latest_terminal_directive
        if terminal_directive is not None:
            if terminal_task_preempt is not None:
                terminal_task_preempt(terminal_preemption_reason(terminal_directive.status))
            terminal_update = task_state.finish_from_model(
                terminal_directive.status,
                terminal_directive.reason,
            )
            if terminal_update.task is not None:
                current_state = replace(
                    current_state,
                    movement_confirmed=False,
                    movement_authorization=None,
                )
                if runtime_state_update is not None:
                    runtime_state_update(current_state)
                emit_task_terminal_directive_trace(
                    logger,
                    terminal_directive,
                    terminal_update,
                    turn_id=turn_id,
                    task_snapshot=task_state.snapshot(),
                    trace_emit=trace_emit,
                )
        if enqueue_message is not None:
            auto_followup = decide_native_auto_followup(
                task_snapshot=task_state.snapshot(),
                model_update_count=model_stream.update_count,
                model_tool_call_count=model_stream.tool_call_count,
                model_tool_call_names=model_stream.tool_call_names,
                queue_empty=queue_empty(),
                state=current_state.auto_followup,
                max_attempts=max_auto_followups_per_task,
                max_navigation_attempts=max_navigation_status_followups_per_task,
            )
            if auto_followup.should_enqueue and auto_followup.message is not None:
                current_state = current_state.with_auto_followup(auto_followup.state)
                if runtime_state_update is not None:
                    runtime_state_update(current_state)
                enqueue_message(auto_followup.message)
                trace_emit(
                    logger,
                    "auto_followup_queued",
                    **auto_followup.trace_payload(),
                )
            elif auto_followup.reason == "max_attempts_reached":
                trace_emit(
                    logger,
                    "auto_followup_skipped",
                    **auto_followup.trace_payload(),
                )

        if queue_empty():
            agent_io.set_idle()
        final_state = (
            runtime_state_snapshot() if runtime_state_snapshot is not None else current_state
        )
        return finish(
            "model_completed",
            result=NativeTurnProcessResult(
                runtime_state=final_state,
                model_updates=model_stream.update_count,
            ),
            model_updates=model_stream.update_count,
        )
    except Exception as e:
        finish(
            "error",
            result=NativeTurnProcessResult(runtime_state=runtime_state),
            error=f"{e.__class__.__name__}: {e}",
        )
        raise


def _image_artifact_ids_from_stream(
    context_manager: ConversationContextManager,
    artifact_ids: list[str],
) -> list[str]:
    """Return stream artifact ids that refer to image artifacts."""

    image_artifact_ids: list[str] = []
    for artifact_id in artifact_ids:
        ref = context_manager.artifact_ledger.get_ref(artifact_id)
        if ref is None or ref.kind != "image":
            continue
        image_artifact_ids.append(artifact_id)
    return image_artifact_ids
