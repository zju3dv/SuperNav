#!/usr/bin/env python3
"""Score multi-object-nav harness runs from recorded goal_marked events.

The online harness metrics for multi-goal episodes stay claim-based
(session closed with outcome='achieved'). This script recomputes per-goal
geometric success offline:

- For every run directory containing a run.json with a ``targets`` list,
  the agent's per-goal declarations are read from the session trajectory
  sidecar (``<session_id>.trajectory.json``, ``goal_marks`` entries written
  by the nav_goals tool).
- Each marked pose is scored against the manifest target: XZ distance to
  the target object bbox surface <= radius_m (bbox criterion), falling
  back to 3D viewpoint distance when the target has no bbox.
- Ordered episodes additionally check that marks arrived in index order.
- An episode succeeds when every goal is marked, every mark hits its
  criterion, and (for ordered episodes) the mark order is correct.

Usage:
    python -m supernav score-multi-objectnav \
        --runs-root data/runs/multi_global_task \
        [--visuals-root data/nav_artifacts]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from supernav.paths import workspace_root

RUNS_ROOT = workspace_root() / "data" / "runs"

from supernav.evaluation.success import bbox_surface_dist_xz  # noqa: E402


def _dist3d(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _path_length(points: list[list[float]]) -> float:
    return sum(_dist3d(a, b) for a, b in zip(points, points[1:]))


def _load_marks(
    trajectory_path: Path,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Return (goal_marks, trajectory positions) from a sidecar.

    Positions prefer the dense sim-step trajectory when available; the
    sparse per-tool endpoint samples are the fallback and systematically
    understate path length.
    """
    doc = json.loads(trajectory_path.read_text(encoding="utf-8"))
    marks = [
        entry
        for entry in doc.get("goal_marks", [])
        if isinstance(entry, dict) and entry.get("target_index") is not None
    ]
    marks.sort(key=lambda entry: str(entry.get("ts") or ""))
    dense = doc.get("dense_trajectory")
    dense_points = dense.get("points") if isinstance(dense, dict) else None
    if isinstance(dense_points, list):
        positions = [
            [float(v) for v in point[:3]]
            for point in dense_points
            if isinstance(point, list) and len(point) >= 3
        ]
        if len(positions) >= 2:
            return marks, positions
    positions = []
    for event in doc.get("trajectory", []):
        if not isinstance(event, dict):
            continue
        pose = event.get("pose")
        if isinstance(pose, dict) and isinstance(pose.get("position"), list):
            positions.append([float(v) for v in pose["position"][:3]])
    return marks, positions


def _score_goal(
    target: dict[str, Any], mark: Optional[dict[str, Any]]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": target.get("index"),
        "target_id": target.get("target_id"),
        "marked": mark is not None,
    }
    if mark is None:
        row["hit"] = False
        row["reason"] = "not_marked"
        return row
    position = mark.get("position")
    if not isinstance(position, list) or len(position) != 3:
        row["hit"] = False
        row["reason"] = "mark_has_no_position"
        return row
    radius = float(target.get("radius_m", 1.0))
    bbox = target.get("target_object") or {}
    center, size = bbox.get("bbox_center"), bbox.get("bbox_size")
    if center is not None and size is not None:
        dist = bbox_surface_dist_xz(position, center, size)
        row["criterion"] = "dist_xz_to_target_bbox_surface"
    else:
        viewpoint = target.get("position")
        if not isinstance(viewpoint, list) or len(viewpoint) != 3:
            row["hit"] = False
            row["reason"] = "no_bbox_and_no_viewpoint"
            return row
        dist = _dist3d(position, viewpoint)
        row["criterion"] = "dist_to_goal_viewpoint"
    row["mark_position"] = [round(float(v), 3) for v in position]
    row["distance_m"] = round(dist, 3)
    row["radius_m"] = radius
    row["hit"] = dist <= radius
    if not row["hit"]:
        row["reason"] = "out_of_radius"
    return row


def score_run(run_dir: Path, visuals_root: Path) -> Optional[dict[str, Any]]:
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        return None
    try:
        run = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    targets = run.get("targets")
    if not isinstance(targets, list) or not targets:
        return None

    metrics: dict[str, Any] = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            metrics = {}
    session_id = metrics.get("session_id")

    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task_id": run.get("task_id"),
        "scene": run.get("scene"),
        "ordered": bool(run.get("ordered", True)),
        "claim_success": metrics.get("success"),
    }
    trajectory_path = (
        visuals_root / f"{session_id}.trajectory.json" if session_id else None
    )
    if trajectory_path is None or not trajectory_path.is_file():
        record["error"] = "trajectory sidecar not found"
        return record

    marks, positions = _load_marks(trajectory_path)
    marks_by_index = {int(mark["target_index"]): mark for mark in marks}

    goals = [
        _score_goal(target, marks_by_index.get(int(target["index"])))
        for target in targets
    ]
    mark_order = [int(mark["target_index"]) for mark in marks]
    order_ok = mark_order == sorted(mark_order)

    ground_truth = run.get("ground_truth") or {}
    gt_total = ground_truth.get("geodesic_distance_m")
    actual_total = _path_length(positions) if len(positions) >= 2 else None
    spl = None
    if (
        actual_total is not None
        and isinstance(gt_total, (int, float))
        and float(gt_total) > 0
        and all(goal["hit"] for goal in goals)
    ):
        spl = float(gt_total) / max(actual_total, float(gt_total))

    record.update(
        {
            "goals": goals,
            "mark_order": mark_order,
            "order_ok": order_ok,
            "found": sum(1 for goal in goals if goal["marked"]),
            "hits": sum(1 for goal in goals if goal["hit"]),
            "total": len(goals),
            "path_length_m": (
                round(actual_total, 3) if actual_total is not None else None
            ),
            "gt_geodesic_distance_m": gt_total,
            "spl": round(spl, 4) if spl is not None else None,
        }
    )
    record["success"] = record["hits"] == record["total"] and (
        order_ok or not record["ordered"]
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT / "multi_global_task",
    )
    parser.add_argument(
        "--visuals-root",
        type=Path,
        default=Path("data/nav_artifacts"),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for run_json in sorted(args.runs_root.rglob("run.json")):
        try:
            record = score_run(run_json.parent, args.visuals_root)
        except Exception as exc:  # noqa: BLE001
            record = {
                "run_dir": str(run_json.parent),
                "task_id": None,
                "scene": None,
                "claim_success": None,
                "error": f"scoring failed: {exc}",
            }
        if record is not None:
            records.append(record)
    if not records:
        print(f"no multi-goal runs found under {args.runs_root}")
        return 1

    scored = [r for r in records if "error" not in r]
    n = len(scored)
    summary = {
        "runs_root": str(args.runs_root),
        "episodes": len(records),
        "scored": n,
        "unscored": len(records) - n,
        "episode_success": sum(1 for r in scored if r["success"]),
        "episode_success_rate": (
            round(sum(1 for r in scored if r["success"]) / n, 4) if n else None
        ),
        "goal_hit_rate": (
            round(sum(r["hits"] for r in scored) / sum(r["total"] for r in scored), 4)
            if n and sum(r["total"] for r in scored)
            else None
        ),
        "order_violations": sum(
            1 for r in scored if r["ordered"] and not r["order_ok"]
        ),
        "spl_mean": (
            round(
                sum(r["spl"] for r in scored if r["spl"] is not None)
                / sum(1 for r in scored if r["spl"] is not None),
                4,
            )
            if any(r["spl"] is not None for r in scored)
            else None
        ),
    }

    out_json = args.out_json or (args.runs_root / "multi_objectnav_score.json")
    out_json.write_text(
        json.dumps(
            {"summary": summary, "episodes": records}, indent=1, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Multi-object-nav offline score",
        "",
        f"- runs root: `{args.runs_root}`",
        f"- episodes: {summary['episodes']} "
        f"(scored {summary['scored']}, unscored {summary['unscored']})",
        f"- episode success: {summary['episode_success']}/{summary['scored']} "
        f"({summary['episode_success_rate']})",
        f"- per-goal hit rate: {summary['goal_hit_rate']}",
        f"- ordered-mark violations: {summary['order_violations']}",
        f"- mean SPL (fully successful episodes): {summary['spl_mean']}",
        "",
        "| task_id | scene | found | hits | total | order_ok | claim | success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "error" in r:
            lines.append(
                f"| {r['task_id']} | {r['scene']} | — | — | — | — | "
                f"{r['claim_success']} | ERROR: {r['error']} |"
            )
            continue
        lines.append(
            f"| {r['task_id']} | {r['scene']} | {r['found']} | {r['hits']} | "
            f"{r['total']} | {r['order_ok']} | {r['claim_success']} | "
            f"{r['success']} |"
        )
    out_md = args.out_md or (args.runs_root / "multi_objectnav_score.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md} and {out_json}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
