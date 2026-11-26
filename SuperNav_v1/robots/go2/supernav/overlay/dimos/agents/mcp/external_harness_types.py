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

"""Shared types for external harness MCP validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Protocol

from dimos.agents.harness.display_content import redact_display_content

_MAX_RENDERED_DETAIL_CHARS = 500


class ExternalHarnessMcpAdapter(Protocol):
    """Subset of ``McpAdapter`` needed for external harness validation."""

    url: str

    def initialize(self) -> dict[str, Any]: ...
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def call_tool_text(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class HarnessValidationCheck:
    """One externally observable MCP readiness check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessValidationReport:
    """Validation result for an MCP server surface exposed to another harness."""

    url: str
    checks: list[HarnessValidationCheck]
    server_status: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def model_dump(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ok": self.ok,
            "checks": [check.model_dump() for check in self.checks],
            "server_status": self.server_status,
            "tools": self.tools,
        }

    def render_text(self) -> str:
        lines = [f"External harness MCP validation for {self.url}"]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"[{marker}] {check.name}: {check.message}")
            if not check.passed and check.details:
                lines.append(f"        details: {_render_details(check.details)}")
        lines.append(f"Overall: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


def pass_check(name: str, message: str, **details: Any) -> HarnessValidationCheck:
    """Build a passing external-harness validation check."""

    return HarnessValidationCheck(name=name, passed=True, message=message, details=details)


def fail_check(name: str, message: str, **details: Any) -> HarnessValidationCheck:
    """Build a failing external-harness validation check."""

    return HarnessValidationCheck(name=name, passed=False, message=message, details=details)


def _render_details(details: dict[str, Any]) -> str:
    safe_details = redact_display_content(details)
    try:
        text = json.dumps(safe_details, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(safe_details)
    if len(text) <= _MAX_RENDERED_DETAIL_CHARS:
        return text
    return f"{text[:_MAX_RENDERED_DETAIL_CHARS]}... [truncated]"
