"""Locate canonical experiment assets in a checkout or installed distribution."""
from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def asset_root() -> Path:
    configured = os.environ.get("SUPERNAV_ASSET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "configs").is_dir():
        return checkout
    return Path(sysconfig.get_path("data")) / "share" / "supernav"


def workspace_root() -> Path:
    configured = os.environ.get("SUPERNAV_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    assets = asset_root()
    return assets if (assets / "pyproject.toml").is_file() else Path.cwd()


def python_paths(root: Path | None = None) -> list[str]:
    source = (root or asset_root()) / "src"
    return [str(source)] if source.is_dir() else []


def resolve_asset_path(value: str | Path | None, *, base: Path) -> Path | None:
    """Resolve an explicit canonical asset path, with installed-asset fallback.

    Existing workspace files take precedence.
    """
    if value is None or not str(value).strip():
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    roots = (base.expanduser().resolve(), asset_root().resolve())
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / path).resolve()
