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

"""Native-agent MCP tool-call metadata and authority accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from dimos.agents.harness.speech_policy import SPEECH_AUTHORIZED_META_KEY
from dimos.agents.harness.task_state import TaskRun
from dimos.agents.runtime_policy import is_movement_tool

_DEFAULT_CLIENT_ID = "dimos-agent"


@dataclass(frozen=True)
class NativeToolCallContext:
    """Metadata carried with native-agent MCP ``tools/call`` requests."""

    progress_token: str
    movement_confirmed: bool = False
    movement_authorization: str | None = None
    movement_authority_token: str | None = None
    user_message: str = ""
    task: TaskRun | None = None
    client_id: str = _DEFAULT_CLIENT_ID
    speech_authorized: bool = False

    @classmethod
    def create(
        cls,
        *,
        movement_confirmed: bool,
        movement_authorization: str | None,
        user_message: str,
        task: TaskRun | None,
        client_id: str = _DEFAULT_CLIENT_ID,
        speech_authorized: bool = False,
    ) -> NativeToolCallContext:
        """Create a context with a fresh MCP progress token."""

        return cls(
            progress_token=str(uuid.uuid4()),
            movement_confirmed=movement_confirmed,
            movement_authorization=movement_authorization,
            movement_authority_token=str(uuid.uuid4()) if movement_confirmed else None,
            user_message=user_message,
            task=task,
            client_id=client_id,
            speech_authorized=speech_authorized,
        )

    def meta(self) -> dict[str, Any]:
        """Return the MCP `_meta` payload for this native-agent tool call."""

        payload: dict[str, Any] = {
            "progressToken": self.progress_token,
            "dimos/agent": True,
            "dimos/client_id": self.client_id,
            "dimos/movement_confirmed": self.movement_confirmed,
            "dimos/movement_authorization": self.movement_authorization,
            "dimos/user_message": self.user_message[:1000],
        }
        if self.movement_authority_token is not None:
            payload["dimos/movement_authority_token"] = self.movement_authority_token
        if self.speech_authorized:
            payload[SPEECH_AUTHORIZED_META_KEY] = True
        if self.task is not None:
            payload.update(
                {
                    "dimos/task_id": self.task.task_id,
                    "dimos/task_goal": self.task.goal[:1000],
                    "dimos/task_origin": self.task.origin,
                }
            )
        return payload

    def params(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return JSON-RPC params for an MCP ``tools/call`` request."""

        return {
            "name": tool_name,
            "arguments": arguments,
            "_meta": self.meta(),
        }

    def consumes_movement_authority(self, tool_name: str) -> bool:
        """Return whether this call should consume the native turn authority flag."""

        return self.movement_confirmed and is_movement_tool(tool_name)
