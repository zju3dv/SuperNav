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

"""Model-facing tool result contracts for native DimOS agents."""

from __future__ import annotations

from typing import Any

ModelToolResult = str | list[dict[str, Any]]


def model_visible_tool_result(tool_name: str, mcp_result: dict[str, Any]) -> ModelToolResult:
    """Normalize an MCP ``tools/call`` result for LangChain tool messages.

    Text-only MCP content becomes a compact string. Rich/multimodal content
    remains structured so the context manager can ledger artifacts and re-inject
    selected image blocks later. If a rich result has no text block, a compact
    explanatory text block is inserted so the model has a textual anchor for the
    visual payload.
    """

    content = mcp_result.get("content", [])
    if not isinstance(content, list):
        return str(content)

    blocks = [item for item in content if isinstance(item, dict)]
    text_parts = [item for item in content if isinstance(item, str)]
    has_rich_block = any(block.get("type") != "text" for block in blocks)
    if has_rich_block:
        if not any(block.get("type") == "text" and block.get("text") for block in blocks):
            blocks.insert(
                0,
                {
                    "type": "text",
                    "text": f"Tool '{tool_name}' returned visual content for this tool call.",
                },
            )
        return blocks

    parts = [str(block.get("text") or "") for block in blocks if block.get("type") == "text"]
    parts.extend(text_parts)
    return "\n".join(parts)
