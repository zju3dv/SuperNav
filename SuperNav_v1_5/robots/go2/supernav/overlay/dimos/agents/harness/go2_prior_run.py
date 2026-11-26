"""Resolve and seed Go2 prior-run artifacts for long-instruction runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class Go2PriorRunArtifacts:
    """Resolved artifacts under one prior Go2 run root."""

    run_root: Path
    graph_memory_path: Path | None = None
    slam_status_path: Path | None = None
    panoramas_path: Path | None = None
    relocalization_bundle_path: Path | None = None
    relocalization_manifest_path: Path | None = None
    relocalization_map_file: Path | None = None
    relocalization_costmap_file: Path | None = None
    missing: tuple[str, ...] = ()

    @property
    def map_available(self) -> bool:
        return self.relocalization_map_file is not None

    def model_dump(self) -> dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "graph_memory_path": _path_str(self.graph_memory_path),
            "slam_status_path": _path_str(self.slam_status_path),
            "panoramas_path": _path_str(self.panoramas_path),
            "relocalization_bundle_path": _path_str(self.relocalization_bundle_path),
            "relocalization_manifest_path": _path_str(self.relocalization_manifest_path),
            "relocalization_map_file": _path_str(self.relocalization_map_file),
            "relocalization_costmap_file": _path_str(self.relocalization_costmap_file),
            "map_available": self.map_available,
            "missing": list(self.missing),
        }


def resolve_go2_prior_run(run_root: Path | None) -> Go2PriorRunArtifacts | None:
    """Resolve Go2 artifacts below a harness root or native DimOS log root.

    Missing optional artifacts are recorded instead of failing; callers decide
    which capabilities to enable from the available paths.
    """

    if run_root is None:
        return None
    root = run_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"prior run root does not exist or is not a directory: {root}")

    # Prepared harness runs keep graph artifacts below ``logs/``. Native
    # ``dimos run`` directories are themselves log roots and keep the same
    # artifacts directly below the root. Prefer the harness layout when both
    # exist, then fall back to the native run layout.
    graph_memory = _existing_file(root / "logs" / "go2-graph-memory.json") or _existing_file(
        root / "go2-graph-memory.json"
    )
    slam_status = _existing_file(root / "logs" / "go2-slam-status.json") or _existing_file(
        root / "go2-slam-status.json"
    )
    panoramas = _existing_dir(root / "logs" / "panoramas") or _existing_dir(
        root / "panoramas"
    )
    bundle_dir = _existing_dir(root / "slam" / "relocalization")
    manifest = _existing_file(root / "slam" / "relocalization" / "manifest.json")
    map_file = _existing_file(root / "slam" / "relocalization" / "global_map.pc2.lcm")
    costmap_file = _existing_file(root / "slam" / "relocalization" / "global_costmap.occ.lcm")

    if manifest is not None:
        manifest_map = _manifest_map_file(manifest)
        if manifest_map is not None:
            map_file = manifest_map
        manifest_costmap = _manifest_costmap_file(manifest)
        if manifest_costmap is not None:
            costmap_file = manifest_costmap

    missing: list[str] = []
    if graph_memory is None:
        missing.append("logs/go2-graph-memory.json")
    if slam_status is None:
        missing.append("logs/go2-slam-status.json")
    if panoramas is None:
        missing.append("logs/panoramas")
    if manifest is None:
        missing.append("slam/relocalization/manifest.json")
    if map_file is None:
        missing.append("slam/relocalization/global_map.pc2.lcm")
    if costmap_file is None:
        missing.append("slam/relocalization/global_costmap.occ.lcm")

    return Go2PriorRunArtifacts(
        run_root=root,
        graph_memory_path=graph_memory,
        slam_status_path=slam_status,
        panoramas_path=panoramas,
        relocalization_bundle_path=bundle_dir,
        relocalization_manifest_path=manifest,
        relocalization_map_file=map_file,
        relocalization_costmap_file=costmap_file,
        missing=tuple(missing),
    )


def seed_go2_prior_graph_memory(
    artifacts: Go2PriorRunArtifacts | None,
    *,
    graph_memory_path: Path | None,
) -> dict[str, Any] | None:
    """Copy prior graph memory and panorama assets into the new run."""

    if artifacts is None or graph_memory_path is None:
        return None
    seeded: dict[str, Any] = {
        "source": "prior_run_root",
        "prior_run_root": str(artifacts.run_root),
        "graph_memory_seeded": False,
        "panoramas_seeded": False,
    }
    if artifacts.graph_memory_path is not None:
        graph_memory_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifacts.graph_memory_path, graph_memory_path)
        seeded["graph_memory_seeded"] = True
        seeded["graph_memory_source"] = str(artifacts.graph_memory_path)
        seeded["graph_memory_path"] = str(graph_memory_path)
    if artifacts.panoramas_path is not None:
        destination = graph_memory_path.parent / "panoramas"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(artifacts.panoramas_path, destination)
        seeded["panoramas_seeded"] = True
        seeded["panoramas_source"] = str(artifacts.panoramas_path)
        seeded["panoramas_path"] = str(destination)
    seeded["missing"] = list(artifacts.missing)
    return seeded


def _existing_file(path: Path) -> Path | None:
    return path.resolve() if path.is_file() else None


def _existing_dir(path: Path) -> Path | None:
    return path.resolve() if path.is_dir() else None


def _manifest_map_file(path: Path) -> Path | None:
    return _manifest_path(path, "global_map_file") or _manifest_path(path, "global_map_path")


def _manifest_costmap_file(path: Path) -> Path | None:
    return _manifest_path(path, "costmap_path")


def _manifest_path(path: Path, key: str) -> Path | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    return candidate.resolve() if candidate.is_file() else None


def _path_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None
