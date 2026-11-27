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

"""Native user-turn preparation for the DimOS fallback agent."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.movement_intent import movement_authorization_reason
from dimos.agents.harness.native_internal_message import is_native_internal_message
from dimos.agents.harness.speech_policy import NativeSpeechGate
from dimos.agents.harness.task_control import TaskControlDecision, decide_task_control
from dimos.agents.harness.task_state import TaskStateManager, TaskStateUpdate
from dimos.agents.runtime_policy import AgentRuntimePolicy


@dataclass(frozen=True)
class NativeTurnPreparation:
    """Prepared native-agent turn metadata before request assembly."""

    model_message: BaseMessage
    external_user_message: bool
    user_message_text: str | None = None
    movement_authorization: str | None = None
    movement_confirmed: bool = False
    task_update: TaskStateUpdate | None = None
    control_decision: TaskControlDecision | None = None


def prepare_native_turn(
    message: BaseMessage,
    *,
    policy: AgentRuntimePolicy,
    speech_gate: NativeSpeechGate,
    task_state: TaskStateManager,
) -> NativeTurnPreparation:
    """Prepare task/speech/movement state for one native-agent input turn."""

    if not is_external_user_message(message):
        return NativeTurnPreparation(model_message=message, external_user_message=False)

    user_text = message_text(message)
    speech_gate.update_for_user_text(user_text)
    movement_reason = movement_authorization_reason(user_text, policy.movement_mode)
    task_update = task_state.update_from_user_message(
        user_text,
        movement_authorization=movement_reason,
    )
    control_decision = decide_task_control(task_update)
    model_message: BaseMessage = message
    if control_decision is not None and control_decision.model_text:
        model_message = HumanMessage(content=control_decision.model_text)
    prepared_movement_authorization = _prepared_movement_authorization(
        task_update,
        explicit_authorization=movement_reason,
    )

    return NativeTurnPreparation(
        model_message=model_message,
        external_user_message=True,
        user_message_text=user_text,
        movement_authorization=prepared_movement_authorization,
        movement_confirmed=prepared_movement_authorization is not None,
        task_update=task_update,
        control_decision=control_decision,
    )


def _prepared_movement_authorization(
    update: TaskStateUpdate,
    *,
    explicit_authorization: str | None,
) -> str | None:
    if explicit_authorization is not None:
        return explicit_authorization
    if update.command != "continue" or update.task is None:
        return None
    return update.task.movement_authorization


def message_text(message: BaseMessage) -> str:
    """Extract human-readable text from a LangChain message content block."""

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def is_external_user_message(message: BaseMessage) -> bool:
    """Return whether a message came from a human user rather than tool status."""

    if not isinstance(message, HumanMessage):
        return False
    if is_native_internal_message(message):
        return False
    text = message_text(message).strip()
    return not text.startswith("[tool:")
