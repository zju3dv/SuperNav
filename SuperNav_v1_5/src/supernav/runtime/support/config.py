"""Configuration loading helpers.

Currently only wraps python-dotenv for project-root `.env` discovery.
Kept as a separate module so the dependency on `dotenv` stays optional.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from supernav.paths import workspace_root

LOOPBACK_BIND_HOST = "127.0.0.1"


def restrict_bind_host_to_loopback(host: str | None) -> str:
    """Return a loopback-only bind host for local bridge services."""

    candidate = (host or "").strip()
    if candidate == "localhost":
        return candidate

    parse_candidate = candidate
    if parse_candidate.startswith("[") and parse_candidate.endswith("]"):
        parse_candidate = parse_candidate[1:-1]
    try:
        if ipaddress.ip_address(parse_candidate).is_loopback:
            return parse_candidate
    except ValueError:
        pass
    return LOOPBACK_BIND_HOST


def load_dotenv_from_project() -> None:
    """Auto-load .env from project root if present.

    No-op if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return

    env_path = workspace_root() / ".env"
    if env_path.is_file():
        _load(str(env_path), override=False)


__all__ = [
    "LOOPBACK_BIND_HOST",
    "load_dotenv_from_project",
    "restrict_bind_host_to_loopback",
]
