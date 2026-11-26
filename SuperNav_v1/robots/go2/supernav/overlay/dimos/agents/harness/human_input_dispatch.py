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

"""Human-input dispatch that preserves the MCP preemption path when available."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import requests

from dimos.agents.harness.user_stop import movement_authority_preemption_intent
from dimos.agents.mcp.mcp_adapter import McpAdapter, McpError
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

HumanInputRoute = Literal["mcp_agent_send", "lcm_fallback"]


class HumanInputPublisher(Protocol):
    """Minimal publish surface shared by LCM transports and tests."""

    def publish(self, message: str) -> Any:
        """Publish a human input message."""


class McpAgentSendClient(Protocol):
    """Minimal MCP surface used by human input dispatch."""

    def call_tool_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call an MCP tool and return text content."""


@dataclass(frozen=True)
class HumanInputDispatchResult:
    """Result of routing one human input message."""

    route: HumanInputRoute
    message: str
    response: str | None = None
    fallback_reason: str | None = None
    authority_preemption_supported: bool = False


class HumanInputDispatchError(RuntimeError):
    """Raised when MCP is reachable but refuses or fails the input path."""


class HumanInputDispatcher:
    """Send human input through MCP first, falling back to direct LCM only if absent.

    `agent_send` is the authoritative native-agent user-stop path because it
    invalidates movement authority and calls stop tools before publishing to the
    `/human_input` topic.  Direct LCM is kept as a compatibility fallback for
    blueprints that do not include `McpServer`.
    """

    def __init__(
        self,
        publisher: HumanInputPublisher,
        *,
        mcp_client_factory: Callable[[], McpAgentSendClient] | None = None,
    ) -> None:
        self._publisher = publisher
        self._mcp_client_factory = mcp_client_factory or _default_mcp_client

    def send(self, message: str) -> HumanInputDispatchResult:
        """Route a human input message to the running agent."""

        normalized = str(message or "").strip()
        if not normalized:
            raise ValueError("Human input message cannot be empty")
        intent = movement_authority_preemption_intent(normalized)

        reason = "MCP agent_send unavailable"
        try:
            client = self._mcp_client_factory()
            response = client.call_tool_text("agent_send", {"message": normalized})
            return HumanInputDispatchResult(
                route="mcp_agent_send",
                message=normalized,
                response=response,
                authority_preemption_supported=True,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            reason = f"{type(e).__name__}: {e}"
            logger.debug("MCP agent_send unavailable; falling back to LCM", error=reason)
        except McpError as e:
            # Do not silently bypass a reachable MCP server's policy/error path.
            raise HumanInputDispatchError(str(e)) from e

        self._publisher.publish(normalized)
        if intent.should_preempt:
            logger.warning(
                "Human input sent through LCM fallback; server-side movement authority "
                "preemption was unavailable",
                message=normalized,
                preempted_by=intent.preempted_by,
            )
        return HumanInputDispatchResult(
            route="lcm_fallback",
            message=normalized,
            fallback_reason=reason,
            authority_preemption_supported=not intent.should_preempt,
        )


def _default_mcp_client() -> McpAgentSendClient:
    return McpAdapter.from_run_entry(timeout=1)
