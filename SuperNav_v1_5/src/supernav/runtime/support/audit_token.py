"""Shared bridge-internal audit token helpers."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional


def audit_token_path(host: str, port: int) -> Path:
    safe_host = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in host
    )
    configured_runtime = os.environ.get("XDG_RUNTIME_DIR")
    user_runtime = Path(f"/run/user/{os.getuid()}")
    runtime_dir = (
        Path(configured_runtime)
        if configured_runtime
        else (
            user_runtime
            if user_runtime.is_dir()
            else Path(f"/tmp/habitat_agent_{os.getuid()}")
        )
    )
    return runtime_dir / f"bridge_audit_{safe_host}_{int(port)}.token"


def read_audit_token(host: str, port: int) -> Optional[str]:
    env_token = os.environ.get("NAV_BRIDGE_AUDIT_TOKEN")
    if env_token:
        return env_token
    path = audit_token_path(host, port)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def ensure_audit_token(host: str, port: int) -> str:
    token = read_audit_token(host, port)
    if token:
        return token
    token = secrets.token_urlsafe(32)
    path = audit_token_path(host, port)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token)
            fh.write("\n")
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return token
