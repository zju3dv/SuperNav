"""Evaluator-only top-down replay data and rendering.

This module is imported only by the post-episode harness.  Its payloads must
never be copied into prompts or model-visible MCP results.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from PIL import Image, ImageDraw, ImageFont

MOTION_TOOLS = frozenset(
    {
        "hab_backward",
        "hab_forward",
        "hab_turn",
        "hab_navigate",
        "hab_visual_local_navigate",
        "hab_visual_ground_preview",
        "hab_visual_overlay_navigate",
        "hab_visual_point_navigate",
        "hab_oracle_local_navigate",
        "hab_navigate_wam",
        "hab_turn_then_navigate_wam",
        "hab_local_navigate",
    }
)


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


@lru_cache(maxsize=16)
def _map_background_rgb(path: str, source: str) -> Image.Image:
    with Image.open(path) as image:
        result = image.convert("RGB")
    if source != "habitat_pathfinder":
        return result
    # Canonical task maps contain a pre-rendered GT path plus start/goal/pose
    # markers. Keep the Pathfinder raster but remove those overlays; replay
    # redraws every dynamic or task-specific element itself.
    pixels = result.load()
    for y in range(result.height):
        for x in range(result.width):
            red, green, blue = pixels[x, y]
            if not (red == green == blue):
                pixels[x, y] = (255, 255, 255)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        result = float(value)
        return result / 1000.0 if result > 1e11 else result
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _vec(value: Any, length: int) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _pose_from_result(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    metrics = body.get("metrics") if isinstance(body.get("metrics"), Mapping) else {}
    agent = (
        metrics.get("agent_state")
        if isinstance(metrics.get("agent_state"), Mapping)
        else {}
    )
    summary = (
        body.get("state_summary")
        if isinstance(body.get("state_summary"), Mapping)
        else {}
    )
    position = _vec(agent.get("position"), 3) or _vec(summary.get("position"), 3)
    if position is None:
        return None
    pose: Dict[str, Any] = {"position": position}
    rotation = _vec(agent.get("rotation"), 4)
    if rotation is not None:
        pose["rotation_wxyz"] = rotation
    heading = summary.get("heading_deg", agent.get("heading_deg"))
    if isinstance(heading, (int, float)):
        pose["heading_deg"] = float(heading)
    return pose


def _tool_result_times(events: Iterable[Mapping[str, Any]]) -> Dict[int, Any]:
    result: Dict[int, Any] = {}
    for event in events:
        if event.get("type") != "tool_result":
            continue
        seq = event.get("audit_tool_seq")
        if isinstance(seq, int):
            result[seq] = event.get("time_end") or event.get("timestamp")
    return result


def _enrich_run_from_scene_tasks(run: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill missing ground_truth/goal from the scene's tasks.json.

    Episodes launched via run_one with only --task-id embed no instruction row,
    so run.json carries ground_truth=null even though the scene dataset keeps
    the same metadata at <dataset>/global_task/<scene>/tasks.json. The replay
    topdown needs it; fill the gaps without touching the on-disk run.json.
    """
    ground_truth = run.get("ground_truth")
    has_tasks = isinstance(ground_truth, Mapping) and bool(
        ground_truth.get("source_tasks_json")
    )
    goal = run.get("goal")
    has_goal = isinstance(goal, Mapping) and bool(goal.get("position"))
    if has_tasks and has_goal:
        return run
    scene = str(run.get("scene") or "")
    task_id = str(run.get("task_id") or "")
    cfg = str(run.get("scene_dataset_config_file") or "")
    if not scene or not task_id or not cfg:
        return run
    tasks_path = Path(cfg).expanduser().parent / "global_task" / scene / "tasks.json"
    if not tasks_path.is_file():
        return run
    try:
        doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return run
    task = next(
        (
            item
            for item in doc.get("tasks", [])
            if isinstance(item, Mapping) and item.get("task_id") == task_id
        ),
        None,
    )
    if task is None:
        return run
    enriched = dict(run)
    gt = dict(ground_truth) if isinstance(ground_truth, Mapping) else {}
    gt.setdefault("source_tasks_json", str(tasks_path))
    if task.get("ground_truth_path"):
        gt.setdefault("path_points", task["ground_truth_path"])
    enriched["ground_truth"] = gt
    if not has_goal and task.get("goal_position"):
        enriched["goal"] = {
            "position": task["goal_position"],
            "radius_m": task.get("goal_radius"),
        }
    return enriched


def _load_task_assets(run: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    ground_truth = run.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        return None
    tasks_path_value = ground_truth.get("source_tasks_json")
    task_id = str(run.get("task_id") or ground_truth.get("source_task_id") or "")
    if not isinstance(tasks_path_value, str) or not task_id:
        return None
    tasks_path = Path(tasks_path_value).expanduser()
    if not tasks_path.is_file() or len(tasks_path.parents) < 3:
        return None
    try:
        tasks_doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    task = next(
        (
            item
            for item in tasks_doc.get("tasks", [])
            if isinstance(item, Mapping) and item.get("task_id") == task_id
        ),
        None,
    )
    if task is None:
        return None
    dataset_root = tasks_path.parents[2]
    scene = str(run.get("scene") or tasks_doc.get("scene_id") or "")
    scene_dir = dataset_root / scene
    occupancy = scene_dir / "occupancy.png"
    occupancy_json = scene_dir / "occupancy.json"
    structure = scene_dir / "structure.json"
    labels = scene_dir / "labels.json"
    if not all(path.is_file() for path in (occupancy, occupancy_json, structure)):
        return None
    topdown_rel = (
        task.get("topdown", {}).get("image")
        if isinstance(task.get("topdown"), Mapping)
        else None
    )
    topdown = tasks_path.parent / str(topdown_rel) if topdown_rel else None
    return {
        "task": dict(task),
        "tasks_path": tasks_path,
        "occupancy": occupancy,
        "occupancy_json": occupancy_json,
        "structure": structure,
        "labels": labels if labels.is_file() else None,
        "topdown": topdown if topdown is not None and topdown.is_file() else None,
    }


def _objectnav_assets(
    run_dir: Path,
    run: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    viewpoint_selection: str = "nearest",
    viewpoint_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    source_episode = Path(str(ground_truth.get("source_episode") or "")).expanduser()
    goal_key = str(ground_truth.get("goal_key") or "")
    scene = str(run.get("scene") or "")
    config_path = Path(str(run.get("scene_dataset_config_file") or "")).expanduser()
    spawn = run.get("spawn") if isinstance(run.get("spawn"), Mapping) else {}
    start = _vec(spawn.get("start_position"), 3)
    if not source_episode.is_file() or not goal_key or not scene or start is None:
        return None
    try:
        opener = gzip.open if source_episode.suffix == ".gz" else open
        with opener(source_episode, "rt", encoding="utf-8") as stream:
            episode_doc = json.load(stream)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        goals = episode_doc.get("goals_by_category", {}).get(goal_key, [])
        viewpoints = [
            viewpoint
            for goal in goals
            if isinstance(goal, Mapping)
            for viewpoint in goal.get("view_points", [])
            if isinstance(viewpoint, Mapping)
            and isinstance(viewpoint.get("agent_state"), Mapping)
            and _vec(viewpoint["agent_state"].get("position"), 3) is not None
        ]
        navmesh_value = config.get("navmesh_instances", {}).get(scene)
        navmesh = config_path.parent / str(navmesh_value)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not viewpoints or not navmesh.is_file():
        return None

    try:
        from supernav.backends.habitat.geometry import objectnav_geometry

        meters_per_pixel = 0.05
        geometry = objectnav_geometry(
            navmesh, start, viewpoints, viewpoint_index, viewpoint_selection, meters_per_pixel
        )
        if geometry is None:
            return None
        closest_index = geometry["closest_index"]
        closest_state = viewpoints[closest_index]["agent_state"]
        goal = _vec(closest_state.get("position"), 3)
        goal_rotation = _vec(closest_state.get("rotation"), 4)
        if goal is None or goal_rotation is None:
            return None
        topdown_array = geometry["topdown"]
        topdown = run_dir / "objectnav_navmesh_topdown.png"
        Image.fromarray((topdown_array.astype("uint8") * 255)).convert("RGB").save(
            topdown
        )
    except (ImportError, RuntimeError, ValueError):
        return None
    return {
        "topdown": topdown,
        "meters_per_pixel": meters_per_pixel,
        "bounds_min_xz": geometry["bounds_min_xz"],
        "start": start,
        "goal": goal,
        "goal_rotation": goal_rotation,
        "goal_viewpoint_iou": viewpoints[closest_index].get("iou"),
        "goal_viewpoint_selection": (
            "reviewed" if viewpoint_index is not None else viewpoint_selection
        ),
        "path_points": geometry["path_points"],
        "geodesic_distance_m": geometry["geodesic_distance_m"],
        "goal_viewpoint_count": len(viewpoints),
        "closest_goal_viewpoint_index": closest_index,
        "navmesh": navmesh,
    }


def _build_objectnav_evaluator(
    run_dir: Path,
    run: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    viewpoint_selection: str = "nearest",
    viewpoint_index: Optional[int] = None,
) -> Dict[str, Any]:
    assets = _objectnav_assets(
        run_dir,
        run,
        ground_truth,
        viewpoint_selection=viewpoint_selection,
        viewpoint_index=viewpoint_index,
    )
    if assets is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["map_assets_unavailable"],
        }
    with Image.open(assets["topdown"]) as image:
        image_size = [image.width, image.height]
    meters_per_pixel = float(assets["meters_per_pixel"])
    bounds_min_xz = list(assets["bounds_min_xz"])
    status = "complete" if trajectory["status"] == "complete" else "partial"
    return {
        "schema_version": 1,
        "status": status,
        "badge": "EVAL-ONLY / OBJECTNAV GT",
        "trajectory": dict(trajectory),
        "map": {
            "background_image": str(assets["topdown"]),
            "background_source": "habitat_pathfinder",
            "background_sha256": _sha256(assets["topdown"]),
            "image_size": image_size,
            "meters_per_pixel": meters_per_pixel,
            "bounds_min_xz": bounds_min_xz,
            "bounds_max_xz_inferred": [
                bounds_min_xz[0] + image_size[0] * meters_per_pixel,
                bounds_min_xz[1] + image_size[1] * meters_per_pixel,
            ],
            "transform": {
                "version": "habitat_pathfinder_xz_v1",
                "pixel_x": "(habitat_x-bounds_min_x)/meters_per_pixel",
                "pixel_y": "(habitat_z-bounds_min_z)/meters_per_pixel",
            },
        },
        "ground_truth": {
            "path_points": list(assets["path_points"]),
            "start": list(assets["start"]),
            "ordered_goals": [
                {
                    "index": 1,
                    "position": list(assets["goal"]),
                    "kind": "nearest_reachable_objectnav_viewpoint",
                }
            ],
            "goal": list(assets["goal"]),
            "object_category": ground_truth.get("object_category"),
            "goal_viewpoint_count": assets["goal_viewpoint_count"],
            "goal_viewpoint_iou": assets["goal_viewpoint_iou"],
            "goal_viewpoint_selection": assets["goal_viewpoint_selection"],
            "recomputed_geodesic_distance_m": assets["geodesic_distance_m"],
            "episode_initial_geodesic_distance_m": ground_truth.get(
                "initial_geodesic_distance_m"
            ),
        },
        "structure": {"rooms": [], "walls": [], "holes": []},
        "semantics": {
            "target_object": None,
            "provenance": {
                key: ground_truth.get(key)
                for key in ("source_episode", "source_episode_id", "goal_key")
            },
        },
        "validation": {
            "status": "passed",
            "navmesh": str(assets["navmesh"]),
            "closest_goal_viewpoint_index": assets["closest_goal_viewpoint_index"],
            "goal_viewpoint_iou": assets["goal_viewpoint_iou"],
            "goal_viewpoint_selection": assets["goal_viewpoint_selection"],
        },
    }


def _sample_path(
    points: List[List[float]], interval_m: float = 0.02
) -> List[List[float]]:
    sampled: List[List[float]] = []
    for left, right in zip(points, points[1:]):
        distance = math.hypot(right[0] - left[0], right[2] - left[2])
        count = max(2, int(distance / interval_m))
        for index in range(count):
            ratio = index / count
            sampled.append(
                [left[axis] * (1 - ratio) + right[axis] * ratio for axis in range(3)]
            )
    if points:
        sampled.append(points[-1])
    return sampled


def _occupancy_score(
    image: Image.Image,
    meta: Mapping[str, Any],
    points: List[List[float]],
    *,
    semantic_sign: int,
    flip_row: bool,
) -> float:
    lower = _vec(meta.get("lower", meta.get("min")), 3)
    upper = _vec(meta.get("upper", meta.get("max")), 3)
    try:
        scale = float(meta.get("scale") or 0.0)
    except (TypeError, ValueError):
        scale = 0.0
    if lower is None or upper is None or scale <= 0:
        return 0.0
    pixels = image.convert("L")
    values: List[int] = []
    for point in _sample_path(points):
        semantic_y = semantic_sign * point[2]
        col = round((point[0] - lower[0]) / scale)
        row_value = upper[1] - semantic_y if flip_row else semantic_y - lower[1]
        row = round(row_value / scale)
        if 0 <= col < pixels.width and 0 <= row < pixels.height:
            values.append(pixels.getpixel((col, row)))
    return sum(value >= 250 for value in values) / len(values) if values else 0.0


def _navmesh_path_score(
    image_path: Path,
    points: List[List[float]],
    bounds_min_xz: List[float],
    meters_per_pixel: float,
) -> float:
    if meters_per_pixel <= 0:
        return 0.0
    try:
        pixels = _map_background_rgb(str(image_path), "habitat_pathfinder").convert("L")
    except OSError:
        return 0.0
    values: List[int] = []
    for point in _sample_path(points):
        col = round((point[0] - bounds_min_xz[0]) / meters_per_pixel)
        row = round((point[2] - bounds_min_xz[1]) / meters_per_pixel)
        if 0 <= col < pixels.width and 0 <= row < pixels.height:
            values.append(pixels.getpixel((col, row)))
    return sum(value >= 250 for value in values) / len(values) if values else 0.0


def _static_marker_validation(
    image_path: Optional[Path],
    start: List[float],
    goal: List[float],
    meters_per_pixel: float,
) -> Dict[str, Any]:
    if image_path is None or meters_per_pixel <= 0:
        return {"status": "unavailable"}
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return {"status": "failed", "reason": "static_topdown_invalid"}

    def center(color: tuple[int, int, int]) -> Optional[List[float]]:
        points = [
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if image.getpixel((x, y)) == color
        ]
        if not points:
            return None
        return [
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        ]

    start_px = center((0, 255, 0))
    goal_px = center((0, 0, 255))
    if start_px is None or goal_px is None:
        return {"status": "failed", "reason": "markers_not_found"}
    start_min = [
        start[0] - start_px[0] * meters_per_pixel,
        start[2] - start_px[1] * meters_per_pixel,
    ]
    goal_min = [
        goal[0] - goal_px[0] * meters_per_pixel,
        goal[2] - goal_px[1] * meters_per_pixel,
    ]
    error = math.hypot(start_min[0] - goal_min[0], start_min[1] - goal_min[1])
    return {
        "status": "passed" if error <= meters_per_pixel * 2.0 else "failed",
        "start_pixel": start_px,
        "goal_pixel": goal_px,
        "bounds_min_xz": [
            (start_min[0] + goal_min[0]) / 2,
            (start_min[1] + goal_min[1]) / 2,
        ],
        "agreement_error_m": error,
    }


def _trajectory_payload(
    sidecar_path: Path,
    events: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    result_times = _tool_result_times(events)
    if sidecar_path.is_file():
        try:
            doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        if doc.get("schema_version") == 2:
            dense = (
                doc.get("dense_trajectory")
                if isinstance(doc.get("dense_trajectory"), Mapping)
                else {}
            )
            reveals = []
            timing_issues: List[str] = []
            for reveal in dense.get("reveals", []):
                if not isinstance(reveal, Mapping):
                    continue
                row = dict(reveal)
                seq = row.get("tool_seq")
                row["reveal_at"] = result_times.get(seq)
                if row["reveal_at"] is None:
                    timing_issues.append(f"result_time_unavailable:{seq}")
                reveals.append(row)
            endpoints = []
            start_sample = doc.get("start_sample")
            if isinstance(start_sample, Mapping):
                endpoints.append(dict(start_sample))
            endpoints.extend(
                dict(sample)
                for sample in doc.get("endpoint_samples", [])
                if isinstance(sample, Mapping)
            )
            end_sample = doc.get("end_sample")
            if isinstance(end_sample, Mapping):
                endpoints.append(dict(end_sample))
            for sample in endpoints:
                seq = sample.get("tool_seq")
                sample["reveal_at"] = result_times.get(seq)
                if sample["reveal_at"] is None:
                    timing_issues.append(f"result_time_unavailable:{seq}")
            issues = list(doc.get("capture_issues", []))
            issues.extend(timing_issues)
            issues = list(dict.fromkeys(str(issue) for issue in issues))
            status = doc.get("capture_status", "partial")
            if timing_issues:
                status = "partial"
            return {
                "status": status,
                "schema_version": 2,
                "points": dense.get("points", []),
                "reveals": reveals,
                "endpoints": endpoints,
                "issues": issues,
                "source": "private_trajectory_v2",
                "source_path": str(sidecar_path),
                "source_sha256": _sha256(sidecar_path),
            }

    # Evaluator-only endpoint trace.  It is intentionally endpoint-only and
    # can never be presented as a complete trajectory.
    points: List[List[float]] = []
    reveals: List[Dict[str, Any]] = []
    endpoints: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") != "tool_result" or event.get("name") not in MOTION_TOOLS:
            continue
        audit = event.get("audit")
        if not isinstance(audit, Mapping):
            continue
        pose = _pose_from_result(audit)
        if pose is None:
            continue
        points.append(pose["position"])
        reveal_at = event.get("time_end") or event.get("timestamp")
        reveals.append(
            {
                "start_index": len(points) - 1,
                "end_index": len(points),
                "tool_seq": event.get("audit_tool_seq"),
                "reveal_at": reveal_at,
                "source": "legacy_benchmark_audit_endpoint",
            }
        )
        endpoints.append(
            {
                "tool": event.get("name"),
                "tool_seq": event.get("audit_tool_seq"),
                "pose": pose,
                "reveal_at": reveal_at,
            }
        )
    return {
        "status": "partial" if points else "unavailable",
        "schema_version": 1,
        "points": points,
        "reveals": reveals,
        "endpoints": endpoints,
        "issues": (
            ["legacy_endpoint_only", "start_end_dense_unavailable"]
            if points
            else ["trajectory_unavailable"]
        ),
        "source": "legacy_benchmark_audit" if points else "none",
    }


def build_evaluator_topdown(
    run_dir: Path,
    visuals_root: Path,
    events: List[Mapping[str, Any]],
    session_id: str,
) -> Dict[str, Any]:
    """Build the private evaluator payload embedded in a replay manifest."""
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["run_json_unavailable"],
        }
    run = _enrich_run_from_scene_tasks(run)
    ground_truth = (
        run.get("ground_truth") if isinstance(run.get("ground_truth"), Mapping) else {}
    )
    is_objectnav = bool(
        ground_truth.get("task_type") == "objectnav"
        and ground_truth.get("source_episode")
    )
    assets = None if is_objectnav else _load_task_assets(run)
    trajectory = _trajectory_payload(
        visuals_root / f"{session_id}.trajectory.json", events
    )
    if trajectory["status"] == "unavailable":
        reasons = list(trajectory.get("issues", []))
        if not is_objectnav and assets is None:
            reasons.insert(0, "map_assets_unavailable")
        return {"schema_version": 1, "status": "unavailable", "reasons": reasons}
    if is_objectnav:
        return _build_objectnav_evaluator(run_dir, run, ground_truth, trajectory)
    if assets is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["map_assets_unavailable"],
        }

    try:
        occupancy_meta = json.loads(
            assets["occupancy_json"].read_text(encoding="utf-8")
        )
        structure = json.loads(assets["structure"].read_text(encoding="utf-8"))
        occupancy_image = Image.open(assets["occupancy"])
    except (OSError, json.JSONDecodeError, ValueError):
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["map_assets_invalid"],
        }
    if not isinstance(occupancy_meta, Mapping) or not isinstance(structure, Mapping):
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["map_assets_invalid"],
        }
    gt_points = [
        point
        for point in ground_truth.get("path_points", [])
        if _vec(point, 3) is not None
    ]
    start = (
        _vec(run.get("spawn", {}).get("start_position"), 3)
        if isinstance(run.get("spawn"), Mapping)
        else None
    )
    goal = (
        _vec(run.get("goal", {}).get("position"), 3)
        if isinstance(run.get("goal"), Mapping)
        else None
    )
    if start is None or goal is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["ground_truth_unavailable"],
        }

    try:
        scale = float(occupancy_meta.get("scale") or 0.0)
    except (TypeError, ValueError):
        scale = 0.0
    lower = _vec(occupancy_meta.get("lower", occupancy_meta.get("min")), 3)
    upper = _vec(occupancy_meta.get("upper", occupancy_meta.get("max")), 3)
    dimensions_ok = bool(
        scale > 0
        and lower is not None
        and upper is not None
        and abs((upper[0] - lower[0]) / scale - occupancy_image.width) <= 2.0
        and abs((upper[1] - lower[1]) / scale - occupancy_image.height) <= 2.0
    )
    eval_points = [
        point for point in trajectory.get("points", []) if _vec(point, 3) is not None
    ]
    validation_points = gt_points or eval_points
    scores = {
        "semantic_y=-habitat_z,row=upper-y": _occupancy_score(
            occupancy_image,
            occupancy_meta,
            validation_points,
            semantic_sign=-1,
            flip_row=True,
        ),
        "semantic_y=habitat_z,row=y-lower": _occupancy_score(
            occupancy_image,
            occupancy_meta,
            validation_points,
            semantic_sign=1,
            flip_row=False,
        ),
        "semantic_y=habitat_z,row=upper-y": _occupancy_score(
            occupancy_image,
            occupancy_meta,
            validation_points,
            semantic_sign=1,
            flip_row=True,
        ),
        "semantic_y=-habitat_z,row=y-lower": _occupancy_score(
            occupancy_image,
            occupancy_meta,
            validation_points,
            semantic_sign=-1,
            flip_row=False,
        ),
    }
    task_topdown = (
        assets["task"].get("topdown")
        if isinstance(assets["task"].get("topdown"), Mapping)
        else {}
    )
    try:
        topdown_mpp = float(task_topdown.get("meters_per_pixel") or 0.0)
    except (TypeError, ValueError):
        topdown_mpp = 0.0
    static_validation = _static_marker_validation(
        assets["topdown"],
        start,
        goal,
        topdown_mpp,
    )
    bounds_min_xz = _vec(static_validation.get("bounds_min_xz"), 2)
    navmesh_image: Optional[Image.Image] = None
    if assets["topdown"] is not None:
        try:
            navmesh_image = _map_background_rgb(
                str(assets["topdown"]), "habitat_pathfinder"
            )
        except OSError:
            navmesh_image = None
    gt_navmesh_score = (
        _navmesh_path_score(assets["topdown"], gt_points, bounds_min_xz, topdown_mpp)
        if assets["topdown"] is not None and bounds_min_xz is not None and gt_points
        else None
    )
    eval_navmesh_score = (
        _navmesh_path_score(assets["topdown"], eval_points, bounds_min_xz, topdown_mpp)
        if assets["topdown"] is not None and bounds_min_xz is not None and eval_points
        else None
    )
    # Static markers and the GT path establish the transform when GT is present;
    # an executed trajectory may legitimately skim navmesh boundaries.
    evaluation_validation_ok = bool(gt_points) or (
        eval_navmesh_score is not None and eval_navmesh_score >= 0.9
    )
    transform_ok = bool(
        dimensions_ok
        and navmesh_image is not None
        and bounds_min_xz is not None
        and topdown_mpp > 0
        and static_validation.get("status") == "passed"
        and (gt_navmesh_score is None or gt_navmesh_score >= 0.9)
        and evaluation_validation_ok
    )
    if not transform_ok:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reasons": ["coordinate_transform_validation_failed"],
            "validation": {
                "dimensions_ok": dimensions_ok,
                "occupancy_scores": scores,
                "static_topdown": static_validation,
                "gt_navmesh_score": gt_navmesh_score,
                "evaluation_navmesh_score": eval_navmesh_score,
            },
        }

    rooms = [
        {"id": f"room_{index}", "profile": room.get("profile", [])}
        for index, room in enumerate(structure.get("rooms", []))
        if isinstance(room, Mapping)
    ]
    walls = [wall for wall in structure.get("walls", []) if isinstance(wall, Mapping)]
    holes = [
        hole
        for hole in structure.get("holes", [])
        if isinstance(hole, Mapping)
        and hole.get("type") in {"DOOR", "OPENING", "WINDOW"}
    ]
    target_object = None
    provenance = (
        run.get("goal", {}).get("provenance")
        if isinstance(run.get("goal"), Mapping)
        else None
    )
    instance_id = (
        str(provenance.get("semantic_instance_id") or "")
        if isinstance(provenance, Mapping)
        else ""
    )
    if assets["labels"] is not None and instance_id:
        try:
            labels = json.loads(assets["labels"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            labels = []
        target_object = next(
            (
                dict(item)
                for item in labels
                if isinstance(item, Mapping) and str(item.get("ins_id")) == instance_id
            ),
            None,
        )

    status = "complete" if trajectory["status"] == "complete" else "partial"
    return {
        "schema_version": 1,
        "status": status,
        "badge": "EVAL-ONLY / PRIVILEGED",
        "trajectory": trajectory,
        "map": {
            "background_image": str(assets["topdown"]),
            "background_source": "habitat_pathfinder",
            "background_sha256": _sha256(assets["topdown"]),
            "image_size": [navmesh_image.width, navmesh_image.height],
            "meters_per_pixel": topdown_mpp,
            "bounds_min_xz": bounds_min_xz,
            "bounds_max_xz_inferred": [
                bounds_min_xz[0] + navmesh_image.width * topdown_mpp,
                bounds_min_xz[1] + navmesh_image.height * topdown_mpp,
            ],
            "transform": {
                "version": "habitat_pathfinder_xz_v1",
                "habitat_from_semantic": "[semantic.x, semantic.z, -semantic.y]",
                "pixel_x": "(habitat_x-bounds_min_x)/meters_per_pixel",
                "pixel_y": "(habitat_z-bounds_min_z)/meters_per_pixel",
            },
            "semantic_occupancy": {
                "image": str(assets["occupancy"]),
                "image_sha256": _sha256(assets["occupancy"]),
                "metadata_sha256": _sha256(assets["occupancy_json"]),
                "image_size": [occupancy_image.width, occupancy_image.height],
                "scale": scale,
                "lower": lower,
                "upper": upper,
                "coordinate_authority": False,
            },
        },
        "ground_truth": {
            "path_points": gt_points,
            "start": start,
            "ordered_goals": [{"index": 1, "position": goal}],
            "goal": goal,
        },
        "structure": {
            "sha256": _sha256(assets["structure"]),
            "rooms": rooms,
            "walls": walls,
            "holes": holes,
            "room_semantics": "neutral_ids_only",
        },
        "semantics": {
            "target_object": target_object,
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
            "labels_sha256": (
                _sha256(assets["labels"]) if assets["labels"] is not None else None
            ),
            "room_labels": {
                "mode": "neutral_ids_only",
                "authoritative_room_semantics_consumed": False,
                "inferred": False,
                "source": str(assets["structure"]),
            },
        },
        "validation": {
            "dimensions_ok": dimensions_ok,
            "occupancy_scores": scores,
            "static_topdown": static_validation,
            "gt_navmesh_score": gt_navmesh_score,
            "evaluation_navmesh_score": eval_navmesh_score,
            "status": "passed",
        },
    }


def _semantic_pixel(
    point: Any, evaluator: Mapping[str, Any]
) -> Optional[tuple[float, float]]:
    vector = _vec(point, 2)
    if vector is None:
        return None
    # InteriorGS scene-graph ground coordinates are semantic x/y. Convert
    # them once into Habitat x/z, then use the exact Pathfinder pixel frame.
    return _habitat_xz_pixel([vector[0], -vector[1]], evaluator)


def _habitat_xz_pixel(
    point_xz: Any, evaluator: Mapping[str, Any]
) -> Optional[tuple[float, float]]:
    vector = _vec(point_xz, 2)
    map_data = evaluator.get("map") if isinstance(evaluator.get("map"), Mapping) else {}
    bounds_min_xz = _vec(map_data.get("bounds_min_xz"), 2)
    meters_per_pixel = float(map_data.get("meters_per_pixel") or 0.0)
    if vector is None or bounds_min_xz is None or meters_per_pixel <= 0:
        return None
    return (
        (vector[0] - bounds_min_xz[0]) / meters_per_pixel,
        (vector[1] - bounds_min_xz[1]) / meters_per_pixel,
    )


def _world_pixel(
    point: Any, evaluator: Mapping[str, Any]
) -> Optional[tuple[float, float]]:
    vector = _vec(point, 3)
    if vector is None:
        return None
    return _habitat_xz_pixel([vector[0], vector[2]], evaluator)


def _visible_state(evaluator: Mapping[str, Any], actual_time: float) -> Dict[str, Any]:
    trajectory = (
        evaluator.get("trajectory")
        if isinstance(evaluator.get("trajectory"), Mapping)
        else {}
    )
    points = (
        trajectory.get("points") if isinstance(trajectory.get("points"), list) else []
    )
    visible_end = 0
    for reveal in trajectory.get("reveals", []):
        if not isinstance(reveal, Mapping):
            continue
        when = _timestamp(reveal.get("reveal_at"))
        if when is not None and when <= actual_time + 1e-6:
            visible_end = max(visible_end, int(reveal.get("end_index") or 0))
    endpoints = [
        sample
        for sample in trajectory.get("endpoints", [])
        if isinstance(sample, Mapping)
        and _timestamp(sample.get("reveal_at")) is not None
        and float(_timestamp(sample.get("reveal_at")) or 0.0) <= actual_time + 1e-6
        and isinstance(sample.get("pose"), Mapping)
    ]
    endpoints.sort(key=lambda item: float(_timestamp(item.get("reveal_at")) or 0.0))
    final_visible = any(sample.get("event") == "end" for sample in endpoints)
    return {
        "points": points[:visible_end],
        "pose": endpoints[-1].get("pose") if endpoints else None,
        "final_visible": final_visible,
    }


def render_topdown_panel(
    evaluator: Mapping[str, Any], *, actual_time: float, width: int, height: int
) -> Image.Image:
    """Render one causal evaluator panel at an absolute replay timestamp."""
    panel = Image.new("RGB", (width, height), (12, 14, 20))
    map_data = evaluator.get("map") if isinstance(evaluator.get("map"), Mapping) else {}
    try:
        occupancy = _map_background_rgb(
            str(map_data.get("background_image")),
            str(map_data.get("background_source") or ""),
        )
    except Exception:
        return panel
    overlay = Image.new("RGBA", occupancy.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    structure = (
        evaluator.get("structure")
        if isinstance(evaluator.get("structure"), Mapping)
        else {}
    )
    room_colors = [
        (93, 178, 245, 35),
        (123, 196, 128, 35),
        (255, 193, 60, 30),
        (238, 91, 141, 28),
    ]
    room_font = _font(8)
    for index, room in enumerate(structure.get("rooms", [])):
        if not isinstance(room, Mapping):
            continue
        polygon = [
            pixel
            for point in room.get("profile", [])
            if (pixel := _semantic_pixel(point, evaluator)) is not None
        ]
        if len(polygon) < 3:
            continue
        draw.polygon(polygon, fill=room_colors[index % len(room_colors)])
        cx = sum(point[0] for point in polygon) / len(polygon)
        cy = sum(point[1] for point in polygon) / len(polygon)
        draw.text(
            (cx, cy),
            str(room.get("id")),
            font=room_font,
            fill=(20, 35, 50, 220),
            anchor="mm",
        )
    for wall in structure.get("walls", []):
        if not isinstance(wall, Mapping):
            continue
        line = [
            pixel
            for point in wall.get("location", [])
            if (pixel := _semantic_pixel(point, evaluator)) is not None
        ]
        if len(line) >= 2:
            draw.line(line, fill=(5, 5, 8, 255), width=2)
    hole_colors = {
        "DOOR": (180, 180, 180, 255),
        "OPENING": (45, 190, 205, 255),
        "WINDOW": (75, 150, 220, 210),
    }
    for hole in structure.get("holes", []):
        if not isinstance(hole, Mapping):
            continue
        raw = hole.get("profile", [])
        floor = []
        for point in raw:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                pixel = _semantic_pixel(point[:2], evaluator)
                if pixel is not None and pixel not in floor:
                    floor.append(pixel)
        if len(floor) >= 2:
            draw.line(
                floor[:2],
                fill=hole_colors.get(str(hole.get("type")), (180, 180, 180, 255)),
                width=3,
            )

    ground_truth = (
        evaluator.get("ground_truth")
        if isinstance(evaluator.get("ground_truth"), Mapping)
        else {}
    )
    gt = [
        pixel
        for point in ground_truth.get("path_points", [])
        if (pixel := _world_pixel(point, evaluator)) is not None
    ]
    if len(gt) >= 2:
        draw.line(gt, fill=(236, 151, 32, 255), width=3)
    state = _visible_state(evaluator, actual_time)
    eval_path = [
        pixel
        for point in state["points"]
        if (pixel := _world_pixel(point, evaluator)) is not None
    ]
    if len(eval_path) >= 2:
        draw.line(eval_path, fill=(29, 106, 230, 255), width=4)

    def marker(
        point: Any,
        color: tuple[int, int, int, int],
        radius: int,
        outline: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        pixel = _world_pixel(point, evaluator)
        if pixel is None:
            return
        x, y = pixel
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=outline,
            width=2,
        )

    marker(ground_truth.get("start"), (55, 190, 90, 255), 4)
    ordered_goals = ground_truth.get("ordered_goals", [])
    if not ordered_goals and ground_truth.get("goal") is not None:
        ordered_goals = [{"index": 1, "position": ground_truth.get("goal")}]
    for ordered_goal in ordered_goals:
        if not isinstance(ordered_goal, Mapping):
            continue
        goal_position = ordered_goal.get("position")
        marker(goal_position, (235, 82, 48, 255), 5, (80, 15, 10, 255))
        goal_pixel = _world_pixel(goal_position, evaluator)
        if goal_pixel is not None:
            draw.text(
                (goal_pixel[0] + 7, goal_pixel[1] - 7),
                str(ordered_goal.get("index") or ""),
                font=_font(9),
                fill=(90, 20, 10, 255),
            )
    pose = state.get("pose")
    if isinstance(pose, Mapping):
        marker(pose.get("position"), (29, 106, 230, 255), 4, (255, 255, 255, 255))
        current = _world_pixel(pose.get("position"), evaluator)
        heading = pose.get("heading_deg")
        if current is not None and isinstance(heading, (int, float)):
            angle = math.radians(float(heading))
            tip = (current[0] + math.cos(angle) * 12, current[1] + math.sin(angle) * 12)
            draw.line((current, tip), fill=(15, 55, 145, 255), width=3)
        if state.get("final_visible"):
            marker(pose.get("position"), (175, 85, 220, 180), 7, (255, 255, 255, 255))

    target = (
        evaluator.get("semantics", {}).get("target_object")
        if isinstance(evaluator.get("semantics"), Mapping)
        else None
    )
    if isinstance(target, Mapping):
        points = target.get("bounding_box", [])
        centers = [
            point
            for point in points
            if isinstance(point, Mapping)
            and isinstance(point.get("x"), (int, float))
            and isinstance(point.get("y"), (int, float))
        ]
        if centers:
            center = [
                sum(float(point["x"]) for point in centers) / len(centers),
                sum(float(point["y"]) for point in centers) / len(centers),
            ]
            pixel = _semantic_pixel(center, evaluator)
            if pixel is not None:
                draw.text(
                    pixel,
                    f"{target.get('label')} #{target.get('ins_id')}",
                    font=_font(8),
                    fill=(95, 20, 10, 255),
                    anchor="ls",
                )

    composed_map = Image.alpha_composite(occupancy.convert("RGBA"), overlay).convert(
        "RGB"
    )
    legend_h = 122
    map_h = max(1, height - legend_h - 34)
    ratio = min(width / composed_map.width, map_h / composed_map.height)
    resized = composed_map.resize(
        (
            max(1, round(composed_map.width * ratio)),
            max(1, round(composed_map.height * ratio)),
        ),
        Image.Resampling.LANCZOS,
    )
    x0 = (width - resized.width) // 2
    panel.paste(resized, (x0, 32))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rectangle((0, 0, width - 1, height - 1), outline=(55, 65, 78))
    panel_draw.text((10, 8), "Top-down audit", font=_font(15), fill=(230, 235, 240))
    badge = str(evaluator.get("badge") or "EVAL-ONLY / PRIVILEGED")
    panel_draw.text(
        (width - 10, 9), badge, font=_font(10), fill=(255, 180, 80), anchor="ra"
    )
    y = height - legend_h + 2
    legend = [
        *(
            [((236, 151, 32), "GT shortest path (full)")]
            if ground_truth.get("path_points")
            else []
        ),
        ((29, 106, 230), "evaluation trajectory (so far)"),
        ((55, 190, 90), "start"),
        ((235, 82, 48), "ordered target / goal"),
        ((175, 85, 220), "final pose (when available)"),
    ]
    for color, label in legend:
        panel_draw.line((12, y + 6, 32, y + 6), fill=color, width=4)
        panel_draw.text((40, y), label, font=_font(10), fill=(210, 220, 230))
        y += 18
    if evaluator.get("status") == "partial":
        panel_draw.text(
            (12, height - 20), "PARTIAL TRAJECTORY", font=_font(11), fill=(255, 110, 90)
        )
    return panel


__all__ = ["build_evaluator_topdown", "render_topdown_panel", "_visible_state"]
