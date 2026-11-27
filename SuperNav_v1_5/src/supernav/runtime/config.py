from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def repo_root_from_here() -> Path:
    from supernav.paths import workspace_root

    return workspace_root()


def load_json(path: str | Path) -> Dict[str, Any]:
    from supernav.paths import resolve_asset_path

    resolved = resolve_asset_path(path, base=Path.cwd())
    with (resolved or Path(path).expanduser()).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_path(value: Optional[str], *, base: Path) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def csv_list(value: str | Iterable[str] | None, default: Iterable[str]) -> List[str]:
    if value is None:
        return [str(x) for x in default]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def slugify(value: Any, *, default: str = "task") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = text.strip("-_.")
    return text or default


def merged_arm(config: Mapping[str, Any], arm_name: str) -> Dict[str, Any]:
    arms = config.get("arms")
    if not isinstance(arms, dict):
        raise ValueError('config field "arms" must be an object')
    if arm_name not in arms:
        raise ValueError(f"unknown arm {arm_name!r}; available={sorted(arms)}")
    raw = arms[arm_name]
    if not isinstance(raw, dict):
        raise ValueError(f"arm {arm_name!r} must be an object")
    parent_name = raw.get("extends")
    if parent_name:
        parent = merged_arm(config, str(parent_name))
        merged = dict(parent)
        merged.update({k: v for k, v in raw.items() if k != "extends"})
        return merged
    return dict(raw)


def resolve_prompts_dir(config: Mapping[str, Any], *, workspace_root: Path) -> Path:
    """Resolve the explicitly selected benchmark prompt resources."""
    from supernav.paths import resolve_asset_path

    value = config.get("prompts_dir")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Experiment config requires an explicit prompts_dir")
    return resolve_asset_path(value, base=workspace_root) or Path(value).expanduser()


def load_instruction_manifest(path: str | Path) -> Dict[str, Any]:
    """Expand relative manifest includes without filtering scoring evidence.

    Included rows precede local rows. Include defaults fill missing top-level
    fields only; an explicit value (including null) in a row always wins.
    """
    from supernav.paths import resolve_asset_path

    def expand(source: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
        source = (resolve_asset_path(source, base=Path.cwd()) or source).resolve()
        if source in stack:
            raise ValueError("Instruction manifest include cycle: " + " -> ".join(map(str, (*stack, source))))
        data = load_json(source)
        includes = data.pop("includes", [])
        if not isinstance(includes, list):
            raise ValueError(f"manifest includes must be a list: {source}")
        rows = []
        for entry in includes:
            if isinstance(entry, str):
                entry = {"path": entry}
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip():
                raise ValueError(f"manifest include requires a path: {source}")
            defaults = entry.get("defaults", {})
            if not isinstance(defaults, dict):
                raise ValueError(f"manifest include defaults must be an object: {source}")
            child = expand(source.parent / entry["path"], (*stack, source))
            rows.extend({**defaults, **row} if isinstance(row, dict) else row for row in child["instructions"])
        local_rows = data.get("instructions", [])
        if not isinstance(local_rows, list):
            raise ValueError('instructions file must contain list field "instructions"')
        return {**data, "instructions": rows + local_rows}

    return expand(Path(path), ())


def raw_instruction_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Return every object row, including scoring-only rows without prompt text."""
    return [row for row in load_instruction_manifest(path)["instructions"] if isinstance(row, dict)]


def instruction_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows = raw_instruction_rows(path)
    if not isinstance(rows, list):
        raise ValueError('instructions file must contain list field "instructions"')
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug", "")).strip()
        text = str(item.get("text", "")).strip()
        if not slug or not text:
            continue
        out.append(dict(item, slug=slug, text=text))
    return out


def row_task_id(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("task_id", "case_id", "episode_id", "scene_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return slugify(value)
    return None


def row_spawn(row: Mapping[str, Any], fallback: Mapping[str, Any]) -> Dict[str, Any]:
    raw = row.get("spawn")
    if isinstance(raw, Mapping):
        return dict(raw)
    out = dict(fallback)
    if isinstance(row.get("start_position"), list):
        out["start_position"] = list(row["start_position"])
    if isinstance(row.get("start_rotation"), list):
        out["start_rotation"] = list(row["start_rotation"])
    if row.get("sensor_height") is not None:
        out["sensor_height"] = row.get("sensor_height")
    return out


def row_ground_truth(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = row.get("ground_truth")
    if isinstance(raw, Mapping):
        return dict(raw)
    goal = row.get("goal")
    if isinstance(goal, Mapping) and isinstance(goal.get("ground_truth"), Mapping):
        return dict(goal["ground_truth"])
    return None
