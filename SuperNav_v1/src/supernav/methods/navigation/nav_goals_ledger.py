"""MCP-side ledger for multi-goal navigation episodes.

In multi-object navigation benchmark episodes the harness injects the
ordered goal list through the ``HAB_MCP_NAV_GOALS_JSON`` environment
variable (index + description only — never ground-truth coordinates).
This module keeps the per-process record of which goals the agent has
marked as reached so the ``nav_goals`` tool can answer ``status``
queries and validate ``mark`` calls. Ground-truth scoring happens
offline from the trajectory sidecar; the ledger holds no coordinates
and makes no success judgments of its own.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

from supernav.backends.habitat.trajectory import now_utc

ENV_NAV_GOALS_JSON = "HAB_MCP_NAV_GOALS_JSON"


class NavGoalsLedger:
    """Tracks goal indexes and mark state for one multi-goal episode."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.session_id: Optional[str] = None
        self.ordered: bool = True
        self._targets: List[Dict[str, Any]] = []
        self._marks: Dict[int, Dict[str, Any]] = {}

    @property
    def configured(self) -> bool:
        return bool(self._targets)

    def reset_from_env(self, session_id: Optional[str]) -> None:
        """(Re)load the goal list from ``HAB_MCP_NAV_GOALS_JSON``.

        Called when a new episode is initialized. A missing or malformed
        payload leaves an empty ledger so ``status`` can report the
        misconfiguration instead of failing silently.
        """
        targets: List[Dict[str, Any]] = []
        ordered = True
        raw = os.environ.get(ENV_NAV_GOALS_JSON, "").strip()
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                ordered = bool(payload.get("ordered", True))
                raw_targets = payload.get("targets")
                if isinstance(raw_targets, list):
                    for entry in raw_targets:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            index = int(entry.get("index"))
                        except (TypeError, ValueError):
                            continue
                        description = str(entry.get("description") or "").strip()
                        targets.append({"index": index, "description": description})
        with self._lock:
            self.session_id = session_id
            self.ordered = ordered
            self._targets = targets
            self._marks = {}

    def status(self) -> Dict[str, Any]:
        """Return the agent-visible ledger (indexes, descriptions, states)."""
        with self._lock:
            goals = [
                {
                    "index": target["index"],
                    "description": target["description"],
                    "state": "found" if target["index"] in self._marks else "pending",
                }
                for target in self._targets
            ]
            return {
                "ordered": self.ordered,
                "goals": goals,
                "found": len(self._marks),
                "total": len(self._targets),
            }

    def is_marked(self, target_index: int) -> bool:
        with self._lock:
            return target_index in self._marks

    def mark(self, target_index: int) -> Dict[str, Any]:
        """Record a goal as reached. Raises ``ValueError`` on invalid marks.

        A mark is one-shot per goal: re-marking an already-found goal or
        marking an unknown index is an error and never changes state.
        Order is not enforced here — ordered-episode compliance is
        scored offline from the recorded events.
        """
        with self._lock:
            if not self._targets:
                raise ValueError(
                    "No navigation goals are configured for this session; "
                    "this is not a multi-goal episode."
                )
            known = {target["index"] for target in self._targets}
            if target_index not in known:
                raise ValueError(
                    f"Unknown target_index {target_index}; "
                    f"valid indexes are {sorted(known)}."
                )
            if target_index in self._marks:
                raise ValueError(
                    f"Goal {target_index} is already marked as found; "
                    "marks cannot be revoked or repeated."
                )
            self._marks[target_index] = {"index": target_index, "marked_at": now_utc()}
            return self.status()


_LEDGER = NavGoalsLedger()


def get_nav_goals_ledger() -> NavGoalsLedger:
    """Process-wide ledger singleton (one MCP server serves one episode)."""
    return _LEDGER
