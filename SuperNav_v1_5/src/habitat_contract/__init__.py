"""Shared bridge/agent protocol with standard-library dependencies.

SuperNav installs this package alongside its runtime. It defines
task/evaluation schemas and strict-arrival checks shared by bridge
and agent processes.
"""

from __future__ import annotations

from habitat_contract import strict_arrival, task_schema

__all__ = ["strict_arrival", "task_schema"]
