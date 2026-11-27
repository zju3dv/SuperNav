"""Compose named experiments with explicit file inheritance and arm semantics.

Only the top-level ``extends`` field refers to other files. Parents are applied
in declaration order, followed by the declaring file. Dictionaries merge
recursively; lists, scalars, and null replace their previous values. File
references are relative to the file declaring them. Other asset paths retain
the runtime's workspace-relative interpretation.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from supernav.paths import asset_root, resolve_asset_path


class ExperimentConfigError(ValueError):
    """An experiment recipe cannot be resolved or composed."""


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = (resolve_asset_path(path, base=Path.cwd()) or path).resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ExperimentConfigError(f"Experiment config inheritance cycle: {chain}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(f"Cannot load experiment config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperimentConfigError(f"Experiment config root must be an object: {path}")

    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) or not parent.strip() for parent in parents
    ):
        raise ExperimentConfigError(
            f"Experiment config 'extends' must be a non-empty path string or a list of non-empty path strings: {path}"
        )
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _merge(merged, _load(parent_path, (*stack, path)))
    return _merge(merged, data)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Compose and validate an experiment recipe."""
    from supernav.runtime.mcp import validate_mcp_config

    config = _load(Path(path), ())
    validate_mcp_config(config.get("mcp") or {})
    environments = [("mcp.environment", (config.get("mcp") or {}).get("environment") or {})]
    for name, arm in (config.get("arms") or {}).items():
        if isinstance(arm, dict):
            environments.append((f"arms.{name}.environment", arm.get("environment") or {}))
    for source, environment in environments:
        if "HAB_MCP_GLOBAL_TASK" in environment:
            raise ExperimentConfigError(
                f"{source}.HAB_MCP_GLOBAL_TASK is not supported; use benchmark_profile instead"
            )
    return config


def list_experiments() -> list[str]:
    """List recipe names from the checkout or installed package assets."""
    directory = asset_root() / "configs" / "experiments"
    return sorted(
        path.stem
        for path in directory.glob("*.json")
        if path.is_file() and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", path.stem)
    )


def resolve_experiment(name: str) -> Path:
    """Resolve a catalog name; explicit config paths use ``--config`` instead."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ExperimentConfigError(
            "Experiment name must contain lowercase letters, digits, '-' or '_', "
            "and start with a letter or digit; use --config for a file path"
        )
    path = asset_root() / "configs" / "experiments" / f"{name}.json"
    if not path.is_file():
        available = ", ".join(list_experiments()) or "(none)"
        raise ExperimentConfigError(f"Unknown experiment {name!r}; available: {available}")
    return path
