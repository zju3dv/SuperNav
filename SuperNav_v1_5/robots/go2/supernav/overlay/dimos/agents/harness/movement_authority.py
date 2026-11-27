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

"""MCP movement-authority metadata validation.

The MCP server remains the authority for movement policy. Client-provided
``dimos/movement_confirmed`` is only a request to create or spend authority; it
is not trusted unless it is paired with the DimOS one-shot movement-authority
token used by the native harness path and deliberate external harness adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimos.agents.harness.movement_lease import MovementLeaseManager

_MISSING_TOKEN_REASON = (
    "Movement authority confirmation requires a one-shot DimOS movement authority token."
)
_REPLAYED_TOKEN_REASON = "Movement authority grant token was already consumed."


@dataclass(frozen=True)
class MovementAuthorityRequest:
    """Normalized movement-authority grant request from MCP ``_meta``."""

    confirmed: bool
    client_id: str
    task: str
    token: str
    agent_metadata: bool
    external_harness_metadata: bool

    @property
    def token_present(self) -> bool:
        return bool(self.token)


@dataclass(frozen=True)
class MovementAuthorityTokenCheck:
    """Result of consuming a movement-authority grant token."""

    allowed: bool
    reason: str | None = None
    token_present: bool = False
    replayed_token: bool = False


def movement_authority_request_from_meta(
    meta: dict[str, Any],
    *,
    client_id: str,
    fallback_task: str,
) -> MovementAuthorityRequest:
    """Extract a movement-authority grant request from untrusted MCP metadata."""

    task = str(meta.get("dimos/task_goal") or meta.get("dimos/user_message") or fallback_task)[:500]
    return MovementAuthorityRequest(
        confirmed=bool(meta.get("dimos/movement_confirmed")),
        client_id=client_id,
        task=task,
        token=str(meta.get("dimos/movement_authority_token") or "").strip(),
        agent_metadata=bool(meta.get("dimos/agent")),
        external_harness_metadata=bool(meta.get("dimos/external_harness")),
    )


def consume_movement_authority_token(
    leases: MovementLeaseManager,
    request: MovementAuthorityRequest,
) -> MovementAuthorityTokenCheck:
    """Consume the one-shot token required for a fresh movement-authority grant."""

    if not request.token_present:
        return MovementAuthorityTokenCheck(
            allowed=False,
            reason=_MISSING_TOKEN_REASON,
            token_present=False,
        )
    if not leases.consume_grant_token(client_id=request.client_id, token=request.token):
        return MovementAuthorityTokenCheck(
            allowed=False,
            reason=_REPLAYED_TOKEN_REASON,
            token_present=True,
            replayed_token=True,
        )
    return MovementAuthorityTokenCheck(allowed=True, token_present=True)


def movement_authority_refusal_message(tool_name: str, reason: str) -> str:
    """Return the user/model-facing refusal for an untrusted authority grant."""

    return (
        f"Runtime movement policy refused '{tool_name}': {reason} "
        "Use the native DimOS task-authority flow, a deliberate external harness "
        "adapter that sends DimOS movement-authority metadata, or enable explicit "
        "admin override for manual unsafe debugging."
    )
