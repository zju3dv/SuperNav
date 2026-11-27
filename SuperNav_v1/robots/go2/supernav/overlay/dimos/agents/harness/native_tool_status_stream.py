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

"""Native-agent handling for asynchronous MCP tool status streams."""

from __future__ import annotations

from typing import Any

from dimos.agents.harness.native_agent_io import NativeAgentIO
from dimos.agents.harness.native_turn_tracing import TraceEmitter
from dimos.agents.harness.tool_status import ToolStatusMessage, tool_status_from_stream_message
from dimos.agents.harness.trace import emit_trace_event


def publish_native_tool_status_stream_message(
    msg: dict[str, Any],
    *,
    logger: Any,
    agent_io: NativeAgentIO,
    trace_emit: TraceEmitter = emit_trace_event,
) -> ToolStatusMessage | None:
    """Parse, trace, and publish one asynchronous tool-stream frame."""

    status = tool_status_from_stream_message(msg)
    if status is None:
        return None

    trace_emit(
        logger,
        "tool_status",
        tool=status.tool_name,
        event_kind=status.event_kind,
        progress_token=status.progress_token,
        progress=status.progress,
        text=str(status.content)[:500],
        status_context=status.status_context,
        route="ui_status_only",
        model_history_appended=False,
        external_user_turn=False,
    )
    agent_io.publish_status(status)
    return status
