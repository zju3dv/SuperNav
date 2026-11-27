"""Private, append-only simulator evidence for the public GS benchmark.

Unlike the display trajectory this retains turns, collisions and tiny motions.
Only the bridge writes it; paths/positions are never added to policy responses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def _append(session, action: str, position=None, **extra) -> None:
    path = getattr(session, "primitive_trace_path", None)
    if not path or getattr(session, "primitive_trace_terminal", False):
        return
    event = {"action": action, "step_count": session.step_count,
             "time_ns": time.time_ns(), **extra}
    if position is not None:
        event["position"] = [float(v) for v in position]
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")
    if action in {"stop", "runtime_error"}:
        session.primitive_trace_terminal = True


def start_trace(session, position) -> None:
    root = os.environ.get("SUPERNAV_PRIMITIVE_TRACE_DIR")
    if not root:
        return
    path = Path(root) / f"{session.session_id}.primitive.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A reused session ID must not overwrite earlier evidence.
    with path.open("x"):
        pass
    session.primitive_trace_path = str(path)
    session.primitive_trace_last_step = 0
    _append(session, "start", position, scene=session.scene, schema="supernav_primitive_trace/v1")


def record_step(session, position) -> None:
    if not getattr(session, "primitive_trace_path", None):
        return
    if session.step_count == session.primitive_trace_last_step:
        return  # Some observation tools also ask for a pose without stepping.
    _append(session, str(session.last_action), position)
    session.primitive_trace_last_step = session.step_count


def stop_trace(session, position) -> None:
    _append(session, "stop", position)


def invalidate_trace(session, reason: str) -> None:
    _append(session, "runtime_error", reason=reason)
