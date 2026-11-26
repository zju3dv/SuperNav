"""Apply local path bindings while leaving the frozen task manifests intact."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

_PATH_FIELDS = frozenset({"source_episode", "scene_dataset_config_file"})


def deploy_task(config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    deployment = config.get("deployment", {})
    if deployment is None:
        deployment = {}
    if not isinstance(deployment, dict):
        raise ValueError("deployment must be an object")
    prefixes = deployment.get("path_prefixes", {})
    if prefixes is None:
        prefixes = {}
    if not isinstance(prefixes, dict):
        raise ValueError("deployment.path_prefixes must be an object")
    if any(
        not isinstance(old, str) or not old.strip()
        or not isinstance(new, str) or not new.strip()
        for old, new in prefixes.items()
    ):
        raise ValueError("deployment.path_prefixes keys and values must be non-empty path strings")
    bindings = sorted(prefixes.items(), key=lambda item: len(item[0]), reverse=True)

    def bind(value):
        if not isinstance(value, str):
            return value
        expanded = os.path.expandvars(value)
        path = Path(expanded).expanduser()
        for old, new in bindings:
            prefix = Path(old)
            if path.is_relative_to(prefix):
                return str(Path(os.path.expandvars(str(new))).expanduser() / path.relative_to(prefix))
        return str(path)

    def visit(value):
        if isinstance(value, dict):
            return {key: bind(item) if key in _PATH_FIELDS else visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return copy.deepcopy(value)

    result = visit(dict(row))
    if deployment.get("scene_dataset_config_file"):
        result["scene_dataset_config_file"] = bind(deployment["scene_dataset_config_file"])
    return result
