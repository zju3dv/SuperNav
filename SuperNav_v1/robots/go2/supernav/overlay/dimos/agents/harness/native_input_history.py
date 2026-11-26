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

"""Input-history handling for native DimOS agent turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages.base import BaseMessage

from dimos.agents.harness.native_agent_io import NativeAgentIO


class InputHistoryContextManager(Protocol):
    """Context-manager subset needed for native input history handling."""

    def normalize_message(self, message: BaseMessage, *, source: str) -> BaseMessage: ...
    def normalize_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]: ...


@dataclass(frozen=True)
class NativeInputHistoryResult:
    """Normalized model/UI input messages after a native input turn is recorded."""

    model_message: BaseMessage
    ui_message: BaseMessage
    history: list[BaseMessage]


def record_native_input_turn(
    *,
    context_manager: InputHistoryContextManager,
    history: list[BaseMessage],
    model_message: BaseMessage,
    original_message: BaseMessage,
    agent_io: NativeAgentIO,
) -> NativeInputHistoryResult:
    """Normalize, store, compact, and publish one native-agent input turn."""

    normalized_model_message = context_manager.normalize_message(model_message, source="input")
    history.append(normalized_model_message)
    compacted_history = context_manager.normalize_messages(history)
    history[:] = compacted_history

    ui_message = (
        context_manager.normalize_message(original_message, source="input")
        if model_message is not original_message
        else normalized_model_message
    )
    agent_io.publish_input(normalized_model_message, ui_message=ui_message)
    return NativeInputHistoryResult(
        model_message=normalized_model_message,
        ui_message=ui_message,
        history=history,
    )
