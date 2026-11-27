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

"""External-harness movement-authority MCP metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from dimos.agents.mcp.external_harness_types import ExternalHarnessMcpAdapter

DEFAULT_EXTERNAL_HARNESS_CLIENT_ID = "external-harness"


@dataclass(frozen=True)
class ExternalHarnessMovementAuthority:
    """One-shot movement-authority metadata for an external MCP harness call."""

    client_id: str
    task_goal: str
    movement_authority_token: str
    progress_token: str

    @classmethod
    def create(
        cls,
        *,
        task_goal: str,
        client_id: str = DEFAULT_EXTERNAL_HARNESS_CLIENT_ID,
    ) -> ExternalHarnessMovementAuthority:
        """Create a fresh authority object for exactly one MCP tool call."""

        return cls(
            client_id=client_id,
            task_goal=task_goal,
            movement_authority_token=str(uuid.uuid4()),
            progress_token=str(uuid.uuid4()),
        )

    def meta(self) -> dict[str, Any]:
        """Return MCP ``_meta`` for a supervised external harness movement call."""

        return {
            "progressToken": self.progress_token,
            "dimos/external_harness": True,
            "dimos/client_id": self.client_id,
            "dimos/movement_confirmed": True,
            "dimos/movement_authority_token": self.movement_authority_token,
            "dimos/task_goal": self.task_goal[:1000],
        }


@dataclass(frozen=True)
class ExternalHarnessMovementCallResult:
    """Result plus the authority metadata used for an external harness tool call."""

    result: dict[str, Any]
    authority: ExternalHarnessMovementAuthority


def call_tool_with_external_movement_authority(
    adapter: ExternalHarnessMcpAdapter,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    task_goal: str,
    client_id: str = DEFAULT_EXTERNAL_HARNESS_CLIENT_ID,
) -> ExternalHarnessMovementCallResult:
    """Call one movement tool through the deliberate external-harness authority path."""

    authority = ExternalHarnessMovementAuthority.create(
        task_goal=task_goal,
        client_id=client_id,
    )
    result = adapter.call_tool(name, arguments or {}, meta=authority.meta())
    return ExternalHarnessMovementCallResult(result=result, authority=authority)


def untrusted_movement_confirmation_probe_meta(
    *,
    client_id: str = "external-harness-validation",
    task_goal: str = "validate movement authority refusal",
) -> dict[str, Any]:
    """Return MCP metadata that must be refused because it has no authority token."""

    return {
        "dimos/external_harness": True,
        "dimos/client_id": client_id,
        "dimos/movement_confirmed": True,
        "dimos/task_goal": task_goal[:1000],
    }


def external_harness_probe_meta(
    *,
    client_id: str = "external-harness-validation",
) -> dict[str, Any]:
    """Return MCP metadata for non-authorizing external harness probes."""

    return {
        "dimos/external_harness": True,
        "dimos/client_id": client_id,
    }
