"""Write navigation tool trace evidence without changing action outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from supernav.runtime.support.log import log


def append_trace(
    trace_file: str,
    kind: str,
    round_idx: int,
    **kwargs: Any,
) -> None:
    """Append a navigation tool trace entry.

    Log write failures without raising so trace I/O cannot change the
    navigation outcome.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry: Dict[str, Any] = {"ts": ts, "kind": kind, "round": round_idx}
    entry.update(kwargs)
    try:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"WARNING: append_trace failed: {exc!r}")


__all__ = ["append_trace"]
