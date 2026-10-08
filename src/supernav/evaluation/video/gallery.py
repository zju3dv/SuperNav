#!/usr/bin/env python3
"""Render ObjectNav shortest-path videos and an agent/GT comparison gallery."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

ANNOTATION_ERROR_OBJECTS = frozenset(
    {"cabinet", "potted plant", "refrigerator", "bookshelf"}
)
OFFICIAL_GOAL_DISTANCE_M = 1.5

from supernav.backends.habitat.bridge_client import BridgeClient  # noqa: E402
from supernav.runtime.config import instruction_rows  # noqa: E402
from supernav.evaluation.topdown_replay import (  # noqa: E402
    _build_objectnav_evaluator,
    _objectnav_assets,
)


def _sample_path(points: list[list[float]], spacing_m: float) -> list[list[float]]:
    sampled = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
        count = max(1, math.ceil(distance / spacing_m))
        sampled.extend(
            [start[i] + (end[i] - start[i]) * step / count for i in range(3)]
            for step in range(1, count + 1)
        )
    return sampled


def _path_rotation(current: list[float], following: list[float]) -> list[float]:
    dx = following[0] - current[0]
    dz = following[2] - current[2]
    yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _color_path(result: Mapping[str, Any]) -> Path:
    path = result.get("visuals", {}).get("color_sensor", {}).get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"get_visuals returned no color image: {result}")
    return Path(path)


def _write_video(frames: list[Path], output: Path, frame_s: float) -> None:
    concat = output.with_suffix(".concat.txt")
    lines: list[str] = []
    for frame in frames:
        lines.extend((f"file '{frame.resolve()}'", f"duration {frame_s:.4f}"))
    final_frame = frames[-1].resolve()
    lines.extend((f"file '{final_frame}'", "duration 1.5", f"file '{final_frame}'"))
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-vf",
                "fps=10,scale=640:480:force_original_aspect_ratio=decrease,"
                "pad=640:480:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    finally:
        concat.unlink(missing_ok=True)


def _render_gt_video(
    client: BridgeClient,
    run_dir: Path,
    output: Path,
    scratch: Path,
    spacing_m: float,
    frame_s: float,
    goal_selection: str,
    run: Mapping[str, Any] | None = None,
    viewpoint_index: int | None = None,
    pitch_down_degrees: int = 0,
) -> dict[str, Any]:
    run = run or json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ground_truth = run["ground_truth"]
    assets = _objectnav_assets(
        run_dir,
        run,
        ground_truth,
        viewpoint_selection=goal_selection,
        viewpoint_index=viewpoint_index,
    )
    if assets is None:
        raise RuntimeError("ObjectNav GT assets are unavailable")
    viewpoint_iou = assets["goal_viewpoint_iou"]
    if not isinstance(viewpoint_iou, (int, float)) or viewpoint_iou <= 0:
        raise RuntimeError(
            f"selected official goal viewpoint has invalid target IoU: {viewpoint_iou!r}"
        )
    points = _sample_path(assets["path_points"], spacing_m)
    client.session_id = None
    init = client.call(
        "init_scene",
        {
            "scene": run["scene"],
            "scene_dataset_config_file": run["scene_dataset_config_file"],
            "start_position": points[0],
            "start_rotation": run["spawn"]["start_rotation"],
            "sensor": {
                "width": 640,
                "height": 480,
                "sensor_height": run["spawn"].get("sensor_height", 1.25),
                "color_sensor": True,
                "depth_sensor": False,
                "semantic_sensor": False,
            },
        },
        timeout=300,
    )
    client.session_id = str(init["session_id"])
    frames: list[Path] = []
    try:
        for index, point in enumerate(points[:-1]):
            client.call(
                "set_agent_state",
                {
                    "position": point,
                    "rotation": _path_rotation(point, points[index + 1]),
                },
                timeout=120,
            )
            visual = client.call(
                "get_visuals",
                {
                    "output_dir": str(scratch),
                    "sensors": ["color_sensor"],
                    "agent_image_max_size": 0,
                },
                timeout=120,
            )
            frames.append(_color_path(visual))
        final_state = client.call(
            "set_agent_state",
            {"position": assets["goal"], "rotation": assets["goal_rotation"]},
            timeout=120,
        )["agent_state"]
        position_error = math.dist(final_state["position"], assets["goal"])
        expected_wxyz = [
            assets["goal_rotation"][3],
            *assets["goal_rotation"][:3],
        ]
        actual_wxyz = final_state["rotation"]
        rotation_error = min(
            math.dist(actual_wxyz, expected_wxyz),
            math.dist(actual_wxyz, [-value for value in expected_wxyz]),
        )
        if position_error > 1e-4 or rotation_error > 1e-4:
            raise RuntimeError(
                f"official viewpoint pose mismatch: position={position_error}, "
                f"rotation={rotation_error}"
            )
        if pitch_down_degrees:
            client.call(
                "step_and_capture",
                {
                    "action": "look_down",
                    "degrees": pitch_down_degrees,
                    "output_dir": str(scratch),
                    "sensors": ["color_sensor"],
                    "agent_image_max_size": 0,
                },
                timeout=120,
            )
        final_visual = client.call(
            "get_visuals",
            {
                "output_dir": str(scratch),
                "sensors": ["color_sensor"],
                "agent_image_max_size": 0,
            },
            timeout=120,
        )
        frames.append(_color_path(final_visual))
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_video(frames, output, frame_s)
    finally:
        try:
            client.call("close_session", {}, timeout=30)
        finally:
            client.session_id = None
    return {
        "gt_distance_m": float(assets["geodesic_distance_m"]),
        "gt_frame_count": len(frames),
        "gt_viewpoint_index": int(assets["closest_goal_viewpoint_index"]),
        "gt_viewpoint_iou": viewpoint_iou,
        "gt_viewpoint_selection": assets["goal_viewpoint_selection"],
        "gt_final_position_error_m": position_error,
        "gt_final_rotation_l2": rotation_error,
        "gt_final_pitch_down_degrees": pitch_down_degrees,
    }


def _render_instruction_gt(
    *,
    instructions: Path,
    output_dir: Path,
    bridge_port: int,
    spacing_m: float,
    frame_s: float,
    goal_selection: str,
    viewpoint_overrides: Mapping[str, Any],
    selected: set[str],
    overwrite: bool,
) -> int:
    rows = instruction_rows(instructions)
    available = {str(row.get("task_id") or "") for row in rows}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"unknown task IDs: {unknown}")

    episodes_path = output_dir / "episodes.json"
    completed = {
        str(row["task_id"]): row
        for row in (
            json.loads(episodes_path.read_text(encoding="utf-8"))
            if episodes_path.is_file()
            else []
        )
    }
    client = BridgeClient(port=bridge_port)
    scratch = output_dir / "_frames"
    failures: list[str] = []
    selected_rows = [
        row for row in rows if not selected or str(row.get("task_id") or "") in selected
    ]
    for index, row in enumerate(selected_rows, 1):
        task_id = str(row["task_id"])
        video = output_dir / "gt_videos" / f"{task_id}.mp4"
        print(f"[{index}/{len(selected_rows)}] {task_id}", flush=True)
        try:
            if overwrite or not video.is_file() or task_id not in completed:
                override = viewpoint_overrides.get(
                    f"{row['scene']}/{row['object_category']}", {}
                )
                if not isinstance(override, Mapping):
                    raise ValueError(f"invalid viewpoint override for {task_id}")
                run_dir = scratch / task_id
                run_dir.mkdir(parents=True, exist_ok=True)
                metadata = _render_gt_video(
                    client,
                    run_dir,
                    video,
                    run_dir,
                    spacing_m,
                    frame_s,
                    goal_selection,
                    run=row,
                    viewpoint_index=override.get("viewpoint_index"),
                    pitch_down_degrees=int(override.get("pitch_down_degrees") or 0),
                )
                completed[task_id] = {**row, **metadata}
                ordered = [
                    completed[str(item["task_id"])]
                    for item in rows
                    if str(item["task_id"]) in completed
                ]
                episodes_path.write_text(
                    json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(task_id)
            print(f"FAILED {task_id}: {exc}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(
        f"DONE rendered={len(selected_rows) - len(failures)} failures={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


def _link_agent_video(run_dir: Path, target: Path) -> None:
    source = run_dir / "replay_topdown_accelerated.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _sync_topdown_manifest(
    run_dir: Path, goal_selection: str, viewpoint_index: int | None = None
) -> str:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectory = manifest.get("evaluator_topdown", {}).get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise RuntimeError("manifest has no evaluator trajectory")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest["evaluator_topdown"] = _build_objectnav_evaluator(
        run_dir,
        run,
        run["ground_truth"],
        trajectory,
        viewpoint_selection=goal_selection,
        viewpoint_index=viewpoint_index,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return str(manifest["visuals_root"])


def _render_agent_topdown(run_dir: Path, visuals_root: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "supernav.evaluation.video.render",
            "--run-dir",
            str(run_dir),
            "--visuals-root",
            visuals_root,
            "--out",
            str(run_dir / "replay_topdown_accelerated.mp4"),
            "--accelerated",
            "--max-wait-s",
            "1",
        ],
        check=True,
    )


def _sync_agent_topdown(
    run_dir: Path, goal_selection: str, viewpoint_index: int | None = None
) -> None:
    _render_agent_topdown(
        run_dir,
        _sync_topdown_manifest(run_dir, goal_selection, viewpoint_index),
    )


def _is_annotation_error(row: Mapping[str, Any]) -> bool:
    ground_truth = row["ground_truth"]
    return bool(
        ground_truth["object_category"] in ANNOTATION_ERROR_OBJECTS
        and ground_truth.get("quality_status")
        != "semantic-coordinate-match-and-visual-gt-reviewed"
    )


def _official_goal_metrics(run_dir: Path) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    endpoints = manifest.get("evaluator_topdown", {}).get("trajectory", {}).get(
        "endpoints", []
    )
    final_position = next(
        (
            endpoint.get("pose", {}).get("position")
            for endpoint in reversed(endpoints)
            if endpoint.get("event") == "end"
        ),
        None,
    )
    ground_truth = run["ground_truth"]
    source = Path(str(ground_truth["source_episode"])).expanduser()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        episode = json.load(stream)
    goals = episode.get("goals_by_category", {}).get(ground_truth["goal_key"], [])
    viewpoints = [
        viewpoint["agent_state"]["position"]
        for goal in goals
        for viewpoint in goal.get("view_points", [])
        if isinstance(viewpoint, Mapping)
        and isinstance(viewpoint.get("agent_state"), Mapping)
        and isinstance(viewpoint["agent_state"].get("position"), list)
    ]
    distance = (
        min(math.dist(final_position, position) for position in viewpoints)
        if isinstance(final_position, list) and viewpoints
        else None
    )
    return {
        "official_goal_distance_m": round(distance, 3) if distance is not None else None,
        "official_goal_threshold_m": OFFICIAL_GOAL_DISTANCE_M,
        "reached_official_goal": (
            distance <= OFFICIAL_GOAL_DISTANCE_M if distance is not None else None
        ),
    }


def _write_gallery(
    rows: list[dict[str, Any]], output_dir: Path, runs_root: Path
) -> None:
    rows = [
        {**row, **_official_goal_metrics(runs_root / str(row["run_id"]))}
        for row in rows
    ]
    valid_rows = [row for row in rows if not _is_annotation_error(row)]
    achieved = sum(row["formal_task_outcome"] == "achieved" for row in valid_rows)
    blocked = sum(row["formal_task_outcome"] == "blocked" for row in valid_rows)
    achieved_gt = sum(
        row["formal_task_outcome"] == "achieved"
        and row["reached_official_goal"] is True
        for row in valid_rows
    )
    excluded = len(rows) - len(valid_rows)
    success_rate = achieved / len(valid_rows) * 100 if valid_rows else 0.0
    sections: list[str] = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        cards: list[str] = []
        for row in (item for item in rows if item["scene"] == scene):
            outcome = str(row["formal_task_outcome"])
            badge = "ok" if outcome == "achieved" else "bad"
            annotation_error = _is_annotation_error(row)
            badges = (
                '<span class="badge invalid">Annotation error</span>'
                if annotation_error
                else ""
            )
            reached_gt = row["reached_official_goal"]
            gt_badge = (
                '<span class="badge gt">Reached GT</span>'
                if reached_gt is True
                else '<span class="badge not-gt">GT not reached</span>'
                if reached_gt is False
                else '<span class="badge unknown">GT unknown</span>'
            )
            task_id = html.escape(str(row["task_id"]))
            category = html.escape(str(row["ground_truth"]["object_category"]))
            instruction = html.escape(
                str(row.get("text") or f"Find and approach a {category}.")
            )
            gt_filter_value = (
                "yes" if reached_gt is True else "no" if reached_gt is False else "unknown"
            )
            cards.append(f"""<article class="task" data-outcome="{html.escape(outcome)}" data-reached-gt="{gt_filter_value}">
  <div class="task-head"><div><h3>{category}</h3><p>{task_id}</p></div><div class="badges">{badges}<span class="badge {badge}">{outcome}</span>{gt_badge}</div></div>
  <div class="media-grid">
    <figure><figcaption>Accelerated agent replay</figcaption><video controls preload="metadata" src="agent_videos/{task_id}.mp4"></video></figure>
    <figure><figcaption>GT shortest path</figcaption><video controls preload="metadata" src="gt_videos/{task_id}.mp4"></video></figure>
  </div>
  <div class="meta"><span>{row['gt_distance_m']:.1f} m GT</span><span>{row['gt_frame_count']} frames</span><span>{int(row['total_tool_calls'])} tools</span></div>
  <p class="instruction">{instruction}</p>
</article>""")
        sections.append(
            f'<section class="scene"><h2>{html.escape(scene)}</h2><div class="tasks">'
            + "".join(cards)
            + "</div></section>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ObjectNav Agent / GT path comparison</title>
<style>
:root {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f4f7fb; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; min-height: 100vh; background: #f4f7fb; }}
main {{ width: min(1440px, calc(100% - 40px)); margin: 0 auto; padding: 44px 0 80px; }}
header {{ display: flex; align-items: end; justify-content: space-between; gap: 24px; padding-bottom: 28px; border-bottom: 1px solid #d7dfeb; }}
h1 {{ margin: 0; color: #111827; font-size: 34px; letter-spacing: 0; }} .summary {{ margin: 8px 0 0; color: #667085; font-size: 14px; }}
nav {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }} nav a {{ padding: 8px 10px; border: 1px solid #cfd9e8; border-radius: 4px; color: #315f72; background: #fff; font-size: 13px; font-weight: 650; text-decoration: none; }} nav a:hover {{ border-color: #7c93a3; }} nav a[aria-current="page"] {{ border-color: #315f72; color: #fff; background: #315f72; }}
.filters {{ display: flex; align-items: end; flex-wrap: wrap; gap: 12px; margin-top: 18px; padding: 14px 16px; border: 1px solid #d7dfeb; border-radius: 8px; background: #fff; }} .filter-field {{ display: grid; gap: 5px; color: #475467; font-size: 12px; font-weight: 650; }} .filter-field select {{ min-width: 150px; height: 36px; padding: 0 32px 0 10px; border: 1px solid #b8c4d4; border-radius: 4px; color: #172033; background: #fff; font: inherit; }} .filter-count {{ margin-left: auto; padding-bottom: 9px; color: #667085; font-size: 13px; font-weight: 650; }}
section {{ margin-top: 42px; }} h2 {{ margin: 0 0 16px; color: #344054; font-size: 20px; letter-spacing: 0; }}
.tasks {{ display: grid; gap: 18px; }} .task {{ overflow: hidden; border: 1px solid #d7dfeb; border-radius: 8px; background: #fff; box-shadow: 0 8px 24px rgba(43, 58, 85, .06); }}
.summary-card {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; margin-top: 24px; border: 1px solid #d7dfeb; border-radius: 8px; background: #d7dfeb; overflow: hidden; }} .summary-item {{ padding: 16px 18px; background: #fff; }} .summary-label {{ color: #667085; font-size: 12px; }} .summary-value {{ margin-top: 5px; color: #111827; font-size: 24px; font-weight: 750; }} .summary-note {{ margin-top: 4px; color: #667085; font-size: 11px; }}
.task-head {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 16px 18px 13px; }}
h3 {{ margin: 0; font-size: 18px; letter-spacing: 0; }} .task-head p {{ margin: 3px 0 0; color: #667085; font-size: 12px; overflow-wrap: anywhere; }}
.badges {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }} .badge {{ flex: 0 0 auto; padding: 3px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; }} .badge.ok {{ color: #067647; background: #dcfae6; }} .badge.bad {{ color: #b42318; background: #fee4e2; }} .badge.gt {{ color: #175cd3; background: #e0edff; }} .badge.not-gt, .badge.unknown {{ color: #475467; background: #eaecf0; }} .badge.invalid {{ color: #9a6700; background: #fff1c2; }}
.media-grid {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid #e4e9f1; border-bottom: 1px solid #e4e9f1; background: #0d121a; }}
figure {{ min-width: 0; margin: 0; }} figure + figure {{ border-left: 1px solid #344054; }} figcaption {{ padding: 8px 12px; color: #d0d5dd; background: #161d29; font-size: 12px; font-weight: 650; }}
video {{ display: block; width: 100%; height: clamp(240px, 29vw, 410px); object-fit: contain; background: #090d13; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 12px 18px 0; color: #667085; font-size: 12px; }} .instruction {{ margin: 7px 18px 15px; color: #475467; font-size: 14px; line-height: 1.5; }}
@media (max-width: 760px) {{ main {{ width: min(100% - 24px, 680px); padding-top: 28px; }} header {{ display: block; }} h1 {{ font-size: 27px; }} nav {{ justify-content: flex-start; margin-top: 14px; }} .filter-field {{ flex: 1 1 140px; }} .filter-field select {{ width: 100%; min-width: 0; }} .filter-count {{ flex-basis: 100%; margin-left: 0; padding-bottom: 0; }} section {{ margin-top: 34px; }} .media-grid {{ grid-template-columns: 1fr; }} figure + figure {{ border-left: 0; border-top: 1px solid #344054; }} video {{ height: auto; aspect-ratio: 4 / 3; }} }}
</style></head><body><main>
<header><div><h1>ObjectNav Agent / GT path comparison</h1><p class="summary">{len(rows)} episodes · GT sampled every 0.5 m</p></div><nav aria-label="Experiment pages"><a href="./" aria-current="page">HQ100 Agent / GT</a></nav></header>
<div class="summary-card" aria-label="Episode statistics"><div class="summary-item"><div class="summary-label">Total episodes</div><div class="summary-value">{len(rows)}</div><div class="summary-note">All episodes</div></div><div class="summary-item"><div class="summary-label">Valid episodes</div><div class="summary-value">{len(valid_rows)}</div><div class="summary-note">Excluded annotation errors: {excluded}</div></div><div class="summary-item"><div class="summary-label">Achieved</div><div class="summary-value">{achieved}</div><div class="summary-note">Valid episodes</div></div><div class="summary-item"><div class="summary-label">Blocked</div><div class="summary-value">{blocked}</div><div class="summary-note">Valid episodes</div></div><div class="summary-item"><div class="summary-label">Achieved and reached GT</div><div class="summary-value">{achieved_gt}</div><div class="summary-note">Any official goal ≤ {OFFICIAL_GOAL_DISTANCE_M:.1f} m</div></div><div class="summary-item"><div class="summary-label">Success rate</div><div class="summary-value">{success_rate:.1f}%</div><div class="summary-note">Achieved / valid episodes</div></div></div>
<div class="filters" aria-label="Episode filters"><label class="filter-field">Outcome<select id="outcome-filter"><option value="all">All</option><option value="achieved">Achieved</option><option value="blocked">Blocked</option></select></label><label class="filter-field">Official GT<select id="gt-filter"><option value="all">All</option><option value="yes">Reached GT</option><option value="no">GT not reached</option></select></label><span class="filter-count" id="filter-count" role="status" aria-live="polite">Showing {len(rows)} / {len(rows)}</span></div>
{''.join(sections)}
</main><script>
const tasks = [...document.querySelectorAll('.task')];
const outcomeFilter = document.querySelector('#outcome-filter');
const gtFilter = document.querySelector('#gt-filter');
const count = document.querySelector('#filter-count');
function applyFilters() {{
  let visible = 0;
  document.querySelectorAll('.scene').forEach(scene => {{
    let sceneVisible = 0;
    scene.querySelectorAll('.task').forEach(task => {{
      const show = (outcomeFilter.value === 'all' || task.dataset.outcome === outcomeFilter.value)
        && (gtFilter.value === 'all' || task.dataset.reachedGt === gtFilter.value);
      task.hidden = !show;
      if (show) {{ visible += 1; sceneVisible += 1; }}
    }});
    scene.hidden = sceneVisible === 0;
  }});
  count.textContent = `Showing ${{visible}} / ${{tasks.length}}`;
}}
outcomeFilter.addEventListener('change', applyFilters);
gtFilter.addEventListener('change', applyFilters);
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "episodes.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument(
        "--instructions",
        type=Path,
        help="Render GT-only videos directly from an instruction manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-port", type=int, default=8911)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--frame-s", type=float, default=0.12)
    parser.add_argument("--viewpoint-overrides", type=Path)
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--goal-selection", choices=("nearest", "highest_iou"), default="nearest"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sync-topdown-only", action="store_true")
    parser.add_argument("--video-workers", type=int, default=3)
    args = parser.parse_args()
    if (args.runs_root is None) == (args.instructions is None):
        parser.error("exactly one of --runs-root or --instructions is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("NO_PROXY", "no_proxy"):
        values = [value for value in os.environ.get(key, "").split(",") if value]
        os.environ[key] = ",".join(dict.fromkeys([*values, "127.0.0.1", "localhost"]))
    selected = {item for item in args.task_ids.split(",") if item}
    if args.instructions is not None:
        viewpoint_overrides = (
            json.loads(args.viewpoint_overrides.read_text(encoding="utf-8"))
            if args.viewpoint_overrides
            else {}
        )
        return _render_instruction_gt(
            instructions=args.instructions,
            output_dir=args.output_dir,
            bridge_port=args.bridge_port,
            spacing_m=args.spacing_m,
            frame_s=args.frame_s,
            goal_selection=args.goal_selection,
            viewpoint_overrides=viewpoint_overrides,
            selected=selected,
            overwrite=args.overwrite,
        )
    scratch = args.output_dir / "_frames"
    results = [
        json.loads(line)
        for line in (args.runs_root / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    episodes_path = args.output_dir / "episodes.json"
    existing = {
        row["task_id"]: row
        for row in (
            json.loads(episodes_path.read_text(encoding="utf-8"))
            if episodes_path.is_file()
            else []
        )
    }
    if args.sync_topdown_only:
        rows: list[dict[str, Any]] = []
        jobs: list[tuple[str, Path, Path, str]] = []
        failures: list[str] = []
        for index, row in enumerate(results, 1):
            task_id = str(row["task_id"])
            run_dir = args.runs_root / str(row["run_id"])
            target = args.output_dir / "agent_videos" / f"{task_id}.mp4"
            print(f"[{index}/{len(results)}] {task_id}", flush=True)
            rows.append({**row, **existing[task_id]})
            if not selected or task_id in selected:
                try:
                    visuals_root = _sync_topdown_manifest(
                        run_dir,
                        args.goal_selection,
                        int(existing[task_id]["gt_viewpoint_index"]),
                    )
                    jobs.append((task_id, run_dir, target, visuals_root))
                except Exception as exc:  # noqa: BLE001
                    failures.append(task_id)
                    print(f"FAILED {task_id}: {exc}", flush=True)
            else:
                _link_agent_video(run_dir, target)
        with ThreadPoolExecutor(max_workers=max(1, args.video_workers)) as executor:
            pending = {
                executor.submit(_render_agent_topdown, run_dir, visuals_root): (
                    task_id,
                    run_dir,
                    target,
                )
                for task_id, run_dir, target, visuals_root in jobs
            }
            for future in as_completed(pending):
                task_id, run_dir, target = pending[future]
                try:
                    future.result()
                    _link_agent_video(run_dir, target)
                    print(f"VIDEO {task_id}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failures.append(task_id)
                    print(f"FAILED {task_id}: {exc}", flush=True)
        _write_gallery(rows, args.output_dir, args.runs_root)
        print(f"DONE rendered={len(rows)} failures={len(failures)}", flush=True)
        return 1 if failures else 0
    client = BridgeClient(port=args.bridge_port)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, row in enumerate(results, 1):
        task_id = str(row["task_id"])
        run_dir = args.runs_root / str(row["run_id"])
        agent_video = args.output_dir / "agent_videos" / f"{task_id}.mp4"
        gt_video = args.output_dir / "gt_videos" / f"{task_id}.mp4"
        print(f"[{index}/{len(results)}] {task_id}", flush=True)
        try:
            selected_task = not selected or task_id in selected
            should_render = selected_task and (args.overwrite or not gt_video.is_file())
            if args.sync_topdown_only and selected_task:
                metadata = existing[task_id]
                _sync_agent_topdown(
                    run_dir,
                    args.goal_selection,
                    int(metadata["gt_viewpoint_index"]),
                )
            elif should_render:
                metadata = _render_gt_video(
                    client,
                    run_dir,
                    gt_video,
                    scratch / task_id,
                    args.spacing_m,
                    args.frame_s,
                    args.goal_selection,
                )
                _sync_agent_topdown(
                    run_dir,
                    args.goal_selection,
                    int(metadata["gt_viewpoint_index"]),
                )
            else:
                metadata = existing[task_id]
            _link_agent_video(run_dir, agent_video)
            rows.append({**row, **metadata})
        except Exception as exc:  # noqa: BLE001
            failures.append(task_id)
            print(f"FAILED {task_id}: {exc}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    _write_gallery(rows, args.output_dir, args.runs_root)
    print(f"DONE rendered={len(rows)} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
