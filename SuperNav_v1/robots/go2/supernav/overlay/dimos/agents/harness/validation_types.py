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

"""Shared result types for harness validation checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LiveValidationCheck:
    """One trace-backed validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveValidationReport:
    """Trace audit result for a DimOS live/replay run."""

    checks: list[LiveValidationCheck]
    event_count: int

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "event_count": self.event_count,
            "checks": [check.model_dump() for check in self.checks],
        }

    def render_text(self) -> str:
        lines = [f"Harness live trace audit ({self.event_count} events)"]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"[{marker}] {check.name}: {check.message}")
        lines.append(f"Overall: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


def pass_check(name: str, message: str, **details: Any) -> LiveValidationCheck:
    """Build a passing validation check."""

    return LiveValidationCheck(name=name, passed=True, message=message, details=details)


def fail_check(name: str, message: str, **details: Any) -> LiveValidationCheck:
    """Build a failing validation check."""

    return LiveValidationCheck(name=name, passed=False, message=message, details=details)
