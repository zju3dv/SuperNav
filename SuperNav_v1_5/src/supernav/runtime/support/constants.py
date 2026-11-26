"""Navigation status and phase constants."""

from __future__ import annotations

from enum import Enum

# Terminal navigation status values.
TERMINAL_STATUSES = frozenset({"reached", "blocked", "error", "timeout"})


class NavPhaseEnum(str, Enum):
    """Navigation phases: planning, executing, verifying and terminal."""

    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    TERMINAL = "terminal"


__all__ = ["TERMINAL_STATUSES", "NavPhaseEnum"]
