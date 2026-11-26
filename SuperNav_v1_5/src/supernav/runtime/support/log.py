"""Timestamped stdout logging."""

from __future__ import annotations

from datetime import datetime, timezone

_LOG_PREFIX = "[supernav]"


def log(msg: str) -> None:
    """Print a UTC timestamp and message with the configured log prefix."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{_LOG_PREFIX}[{ts}] {msg}", flush=True)


__all__ = ["log"]
