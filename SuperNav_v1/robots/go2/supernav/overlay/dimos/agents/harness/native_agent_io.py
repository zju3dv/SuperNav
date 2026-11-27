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

"""UI/output publishing seam for the native DimOS agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage

from dimos.agents.utils import pretty_print_langchain_message


class MessagePublisher(Protocol):
    def publish(self, message: BaseMessage) -> None: ...


class IdlePublisher(Protocol):
    def publish(self, idle: bool) -> None: ...


PrettyPrinter = Callable[[BaseMessage], None]


@dataclass
class NativeAgentIO:
    """Publish native-agent messages and idle transitions in one place."""

    agent: MessagePublisher
    agent_idle: IdlePublisher
    pretty_print: PrettyPrinter = pretty_print_langchain_message

    def set_busy(self) -> None:
        self.agent_idle.publish(False)

    def set_idle(self) -> None:
        self.agent_idle.publish(True)

    def publish_status(self, message: BaseMessage) -> None:
        """Publish a status/control message without adding terminal pretty output."""

        self.agent.publish(message)

    def publish_input(
        self,
        model_visible_message: BaseMessage,
        *,
        ui_message: BaseMessage | None = None,
    ) -> None:
        """Publish a user/input turn while printing the model-visible form."""

        self.pretty_print(model_visible_message)
        self.agent.publish(ui_message or model_visible_message)

    def publish_model_update(self, message: BaseMessage) -> None:
        self.pretty_print(message)
        self.agent.publish(message)

    def publish_text_response(self, response_text: str) -> AIMessage:
        response = AIMessage(content=response_text)
        self.publish_model_update(response)
        return response

    def publish_local_response(
        self,
        input_message: BaseMessage,
        response_text: str,
    ) -> AIMessage:
        """Publish an input echo plus local assistant response and mark idle."""

        self.publish_input(input_message)
        response = self.publish_text_response(response_text)
        self.set_idle()
        return response
