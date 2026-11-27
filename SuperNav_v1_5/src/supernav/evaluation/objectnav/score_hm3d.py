#!/usr/bin/env python3
"""SR/SPL rescoring for HM3D ObjectNav harness runs.

The harness's own ``metrics.success`` is claim-based (structured close with
outcome ``achieved``). This script recomputes success against the episode's
stored ``goal.view_points`` — not the object centers or bounding boxes:

- ``euclidean_viewpoint`` (default): the episode counts as success iff the
  agent formally stopped (``hab_close_session`` — the harness STOP action)
  AND the minimum euclidean distance from the stop position to any stored
  goal view point is <= 1.0 m. No stop call (e.g. timeout kill) is a failure
  even on the goal.
- ``viewpoint_geodesic``: the official habitat-lab measure implementation
  (task config ``distance_to: VIEW_POINTS`` / ``success_distance: 0.1``) —
  stopped AND geodesic distance on the scene's ``.basis.navmesh`` from the
  stop position to the nearest goal view point < 0.1 m. Stricter (implies
  oracle visibility); use for apples-to-apples with challenge-server numbers.

SPL and DTG use geodesic information: per episode ``S * l / max(p, l)``
where ``l`` is the episode's ``info.geodesic_distance`` (start to closest
goal instance, from the official episode file) and ``p`` is the path length
accumulated from the dense trajectory sidecar; DTG is the geodesic distance
on the official navmesh from the stop position to the nearest goal view
point. Dataset SPL/DTG are means over episodes.

Per run the inputs are: ``run.json`` (task_id), ``metrics.json``
(session_id, close_called), and ``<visuals_root>/<session_id>.trajectory.json``
(dense_trajectory points). Goal view points come from the episode's
``goals_by_category`` entry referenced by the instructions row.

Usage:

    python -m supernav score-objectnav \
      --runs-root data/runs/objectnav_hm3d_v2/val \
      --instructions configs/benchmarks/objectnav_hm3d/hm3d_v2_val.instructions.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from supernav.paths import asset_root, workspace_root

RUNS_ROOT = workspace_root() / "data" / "runs"

from supernav.runtime.config import load_instruction_manifest, slugify  # noqa: E402

DEFAULT_EPISODES_ROOT = os.environ.get("SUPERNAV_HM3D_EPISODES_ROOT")
DEFAULT_SCENES_ROOT = os.environ.get("SUPERNAV_HM3D_SCENES_ROOT")

CRITERION_EUCLIDEAN_VIEWPOINT = "euclidean_viewpoint"
CRITERION_VIEWPOINT_GEODESIC = "viewpoint_geodesic"
CRITERIA = (CRITERION_EUCLIDEAN_VIEWPOINT, CRITERION_VIEWPOINT_GEODESIC)

# ApexNav's published HM3Dv2 table uses criterion=viewpoint_geodesic with
# success_distance=0.2 (their yaml overrides the habitat default 0.1).
DEFAULT_SUCCESS_DISTANCE = {
    CRITERION_EUCLIDEAN_VIEWPOINT: 1.0,
    CRITERION_VIEWPOINT_GEODESIC: 0.1,
}

# SPL shortest-path term source: the episode file's stored
# info.geodesic_distance (to the goal object), or recomputed geodesic from
# the start position to the nearest goal VIEW POINT, which is what
# habitat-lab's SPL measure actually uses at episode reset.
L_SOURCE_EPISODE = "episode"
L_SOURCE_VIEWPOINT_GEODESIC = "viewpoint_geodesic"
L_SOURCES = (L_SOURCE_EPISODE, L_SOURCE_VIEWPOINT_GEODESIC)


# --------------------------------------------------------------------- goals
def load_goal_geometry(
    source_episode: str, goal_key: str
) -> tuple[list[list[float]], list[list[float]]]:
    """(instance positions, view-point positions) for ``goals_by_category[goal_key]``."""
    with gzip.open(source_episode, "rt") as f:
        data = json.load(f)
    positions: list[list[float]] = []
    view_points: list[list[float]] = []
    for goal in (data.get("goals_by_category") or {}).get(goal_key) or []:
        pos = goal.get("position")
        if isinstance(pos, list) and len(pos) == 3:
            positions.append([float(v) for v in pos])
        for vp in goal.get("view_points") or []:
            state = vp.get("agent_state") or {}
            vp_pos = state.get("position")
            if isinstance(vp_pos, list) and len(vp_pos) == 3:
                view_points.append([float(v) for v in vp_pos])
    return positions, view_points


from supernav.backends.habitat.geometry import NavmeshGeodesic, find_navmesh


def min_euclid_distance(pos: list[float], points: list[list[float]]) -> float:
    """Min 3D euclidean distance from pos to the given points.

    Goal view points and the agent stop position are both floor-level agent
    origins, so the 3D form matches the usual XZ reading to within the pose
    noise floor.
    """
    if not points:
        return math.inf
    return min(math.dist(pos, p) for p in points)


# ---------------------------------------------------------------- trajectory
def load_trajectory(sidecar: Path) -> list[list[float]]:
    """Dense sim-step positions preferred; sparse per-tool poses as fallback."""
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    dense = doc.get("dense_trajectory")
    points = dense.get("points") if isinstance(dense, dict) else None
    if isinstance(points, list):
        positions = [
            [float(v) for v in p[:3]]
            for p in points
            if isinstance(p, list) and len(p) >= 3
        ]
        if len(positions) >= 2:
            return positions
    positions = []
    for event in doc.get("trajectory", []):
        if not isinstance(event, dict):
            continue
        pose = event.get("pose")
        if isinstance(pose, dict) and isinstance(pose.get("position"), list):
            positions.append([float(v) for v in pose["position"][:3]])
    return positions


def path_length(points: list[list[float]]) -> float:
    # euclidean step accumulation, the official SPL convention
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


# MTU3D OVON replica clause: episodes whose goal view points are unreachable
# from the start position on the official .basis.navmesh are scored sr=1/spl=1
# by MTU3D's evaluator (hm3d-online/ovon-nav.py). Membership is
# (scene_hash, mtu3d_episode_index); both were recomputed and verified against
# the official navmeshes before being listed here.
MTU3D_UNREACHABLE_EPISODES = frozenset(
    {
        ("q5QZSEeHe5g", 121),  # exercise bike
        ("wcojb4TFT35", 11),  # picture
    }
)


# ------------------------------------------------------------------- scoring
def score_run(
    run_dir: Path,
    row: dict,
    view_points: list[list[float]],
    geodesic: Any,
    visuals_root: Path,
    criterion: str = CRITERION_EUCLIDEAN_VIEWPOINT,
    success_distance: Optional[float] = None,
    l_source: str = L_SOURCE_EPISODE,
    stop_required: bool = True,
    unreachable_freebie: bool = False,
) -> dict[str, Any]:
    """Score one harness run dir against its instructions row.

    Distances are measured to the episode's stored goal view points. When a
    ``geodesic`` provider (``geodesic_to_any(split, scene_hash, start,
    ends)``; NavmeshGeodesic in production, a stub in tests) is supplied,
    the geodesic DTG from the stop position to the nearest view point is
    always recorded; it is the success distance only under the
    viewpoint_geodesic criterion. With ``l_source=viewpoint_geodesic`` the
    SPL shortest-path term is recomputed from the episode start position to
    the nearest view point (habitat-lab's reset-time DistanceToGoal),
    falling back to the episode's stored value when unavailable.
    """
    if success_distance is None:
        success_distance = DEFAULT_SUCCESS_DISTANCE[criterion]
    record: dict[str, Any] = {"run_dir": str(run_dir), "task_id": row["task_id"]}
    gt = row["ground_truth"]
    l_episode = float(gt["initial_geodesic_distance_m"])
    record["geodesic_distance_gt_m"] = round(l_episode, 3)
    record["criterion"] = criterion

    metrics_path = run_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            metrics = {}
    record["claim_success"] = metrics.get("success")
    stopped = bool(metrics.get("close_called"))
    record["stopped"] = stopped

    session_id = metrics.get("session_id")
    sidecar = visuals_root / f"{session_id}.trajectory.json" if session_id else None
    positions: list[list[float]] = []
    if sidecar is not None and sidecar.is_file():
        try:
            positions = load_trajectory(sidecar)
        except Exception as exc:  # noqa: BLE001
            record["reason"] = f"trajectory_unreadable: {exc}"
    elif stopped:
        record["reason"] = "trajectory_sidecar_missing"

    p_len = path_length(positions) if len(positions) >= 2 else 0.0
    record["path_length_m"] = round(p_len, 3)

    start = (row.get("spawn") or {}).get("start_position")
    l_star = l_episode
    if l_source == L_SOURCE_VIEWPOINT_GEODESIC and geodesic is not None and start:
        l_vp = geodesic.geodesic_to_any(
            gt["hm3d_split"], row["scene"], start, view_points
        )
        if math.isfinite(l_vp) and l_vp > 0:
            l_star = l_vp
    record["l_source"] = l_source
    record["l_star_m"] = round(l_star, 3)

    dist_euclid = math.inf
    dtg_geodesic = None
    if positions and (stopped or not stop_required):
        dist_euclid = min_euclid_distance(positions[-1], view_points)
        if geodesic is not None:
            d = geodesic.geodesic_to_any(
                gt["hm3d_split"], row["scene"], positions[-1], view_points
            )
            dtg_geodesic = d if math.isfinite(d) else None
    record["success_distance_m"] = success_distance
    record["dist_view_point_euclid_m"] = (
        round(dist_euclid, 3) if math.isfinite(dist_euclid) else None
    )
    record["dtg_geodesic_m"] = round(dtg_geodesic, 3) if dtg_geodesic is not None else None

    if criterion == CRITERION_EUCLIDEAN_VIEWPOINT:
        dist = dist_euclid
    else:
        dist = dtg_geodesic if dtg_geodesic is not None else math.inf
    success = bool((stopped or not stop_required) and dist < success_distance)
    record["success"] = success
    if unreachable_freebie:
        # MTU3D unreachable-goal clause: start->view point has no path on the
        # official navmesh, so the episode is gifted sr=1/spl=1 regardless of
        # where the agent stopped.
        success = True
        record["success"] = True
        record["reason"] = "goal_unreachable_from_start_gifted"
    elif not stopped:
        record["reason"] = (
            "no_stop_called" if stop_required else "stopped_too_far"
        )
    elif not positions:
        record.setdefault("reason", "no_trajectory")
    elif not math.isfinite(dist):
        record["reason"] = "goal_unreachable_from_stop"
    elif not success:
        record["reason"] = "stopped_too_far"

    record["spl"] = (
        round(success * l_star / max(p_len, l_star), 4) if l_star > 0 else 0.0
    )
    if unreachable_freebie:
        record["spl"] = 1.0
    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    per_category: dict[str, dict[str, Any]] = {}
    cats: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cats.setdefault(r.get("object_category") or "?", []).append(r)
    for cat, rs in sorted(cats.items()):
        m = len(rs)
        per_category[cat] = {
            "episodes": m,
            "sr": round(sum(1 for r in rs if r["success"]) / m, 4) if m else None,
            "spl": round(sum(r["spl"] for r in rs) / m, 4) if m else None,
        }
    reasons: dict[str, int] = {}
    for r in records:
        if not r["success"]:
            reasons[r.get("reason") or "unknown"] = (
                reasons.get(r.get("reason") or "unknown", 0) + 1
            )
    dtgs = [r["dtg_geodesic_m"] for r in records if r.get("dtg_geodesic_m") is not None]
    return {
        "episodes": n,
        "sr": round(sum(1 for r in records if r["success"]) / n, 4) if n else None,
        "spl": round(sum(r["spl"] for r in records) / n, 4) if n else None,
        "dtg_geodesic_mean_m": round(sum(dtgs) / len(dtgs), 3) if dtgs else None,
        "stopped": sum(1 for r in records if r.get("stopped")),
        "failure_reasons": reasons,
        "per_category": per_category,
    }


def _latest_by_task(run_dirs: Iterable[Path]) -> dict[str, Path]:
    """Dedupe repeated runs of one task_id, keeping the newest metrics.json."""
    chosen: dict[str, tuple[float, Path]] = {}
    for d in run_dirs:
        try:
            run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        task_id = run.get("task_id")
        if not task_id:
            continue
        metrics = d / "metrics.json"
        stamp = metrics.stat().st_mtime if metrics.is_file() else (d / "run.json").stat().st_mtime
        if task_id not in chosen or stamp > chosen[task_id][0]:
            chosen[task_id] = (stamp, d)
    return {task_id: d for task_id, (stamp, d) in chosen.items()}


def discover_run_dirs(root: Path) -> list[Path]:
    """Include published lane links without recursively following arbitrary links."""
    directories = {p.parent for p in root.rglob("run.json") if p.parent.is_dir()}
    if root.is_dir():
        directories.update(
            p for p in root.iterdir() if p.is_symlink() and (p / "run.json").is_file()
        )
    return sorted(directories)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT / "objectnav_hm3d_v2" / "val",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="External instruction manifest.",
    )
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT, required=not DEFAULT_SCENES_ROOT)
    parser.add_argument("--visuals-root", type=Path, default=Path("data/nav_artifacts"))
    parser.add_argument("--criterion", choices=CRITERIA, default=CRITERION_EUCLIDEAN_VIEWPOINT)
    parser.add_argument(
        "--success-distance",
        type=float,
        default=None,
        help="override the criterion default (euclidean 1.0 m / geodesic 0.1 m; "
        "ApexNav's table uses viewpoint_geodesic with 0.2 m)",
    )
    parser.add_argument(
        "--l-source",
        choices=L_SOURCES,
        default=L_SOURCE_EPISODE,
        help="SPL shortest-path term: episode's stored geodesic, or recomputed "
        "start->nearest view point geodesic (habitat-lab/ApexNav reset-time value)",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument(
        "--mtu3d-replica",
        action="store_true",
        help="MTU3D OVON replica protocol: viewpoint_geodesic at 0.25 m, no "
        "explicit STOP required (trajectory-end position counts), and the two "
        "verified unreachable-goal episodes are gifted sr=1/spl=1. Writes "
        "mtu3d_replica_score.{json,md} by default.",
    )
    args = parser.parse_args()
    if args.mtu3d_replica:
        args.criterion = CRITERION_VIEWPOINT_GEODESIC
        if args.success_distance is None:
            args.success_distance = 0.25
        args.l_source = L_SOURCE_VIEWPOINT_GEODESIC

    instructions = load_instruction_manifest(args.instructions)
    # run.json task ids are slugified by the harness; normalize both sides so
    # mixed-case manifest ids still match.
    rows = {slugify(r["task_id"]): r for r in instructions["instructions"]}

    run_dirs = discover_run_dirs(args.runs_root)
    if not run_dirs:
        print(f"no runs found under {args.runs_root}")
        return 1
    by_task = _latest_by_task(run_dirs)
    duplicates = len(run_dirs) - len(by_task)

    # DTG (geodesic stop->nearest view point) is reported under every
    # criterion, so the navmesh provider is always wired up; it only loads
    # .basis.navmesh files for scenes that actually have scored runs.
    geodesic = NavmeshGeodesic(args.scenes_root)
    geometry_cache: dict[tuple[str, str], tuple[list, list]] = {}
    records: list[dict[str, Any]] = []
    missing_row = 0
    for task_id, run_dir in sorted(by_task.items()):
        row = rows.get(slugify(task_id))
        if row is None:
            missing_row += 1
            continue
        key = (row["source_episode"], row["goal_key"])
        if key not in geometry_cache:
            geometry_cache[key] = load_goal_geometry(*key)
        _goal_positions, view_points = geometry_cache[key]
        unreachable_freebie = False
        if args.mtu3d_replica:
            ep_idx = (row.get("ground_truth") or {}).get("mtu3d_episode_index")
            unreachable_freebie = (row["scene"], ep_idx) in MTU3D_UNREACHABLE_EPISODES
        record = score_run(
            run_dir,
            row,
            view_points,
            geodesic,
            args.visuals_root,
            criterion=args.criterion,
            success_distance=args.success_distance,
            l_source=args.l_source,
            stop_required=not args.mtu3d_replica,
            unreachable_freebie=unreachable_freebie,
        )
        record["object_category"] = row["object_category"]
        record["scene"] = row["scene"]
        records.append(record)

    if not records:
        print("no runs matched the instructions manifest")
        return 1
    summary = aggregate(records)
    summary["runs_root"] = str(args.runs_root)
    summary["criterion"] = args.criterion
    summary["success_distance_m"] = (
        args.success_distance
        if args.success_distance is not None
        else DEFAULT_SUCCESS_DISTANCE[args.criterion]
    )
    summary["duplicate_runs_ignored"] = duplicates
    summary["runs_without_manifest_row"] = missing_row
    summary["l_source"] = args.l_source

    out_json = args.out_json or (
        args.runs_root / "mtu3d_replica_score.json"
        if args.mtu3d_replica
        else args.runs_root / "official_score.json"
    )
    out_json.write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# HM3D ObjectNav SR/SPL rescore",
        "",
        f"- runs root: `{args.runs_root}`",
        f"- criterion: `{args.criterion}` "
        + (
            f"(no STOP required; dist <= {summary['success_distance_m']} m; "
            f"unreachable-goal gifts: {len(MTU3D_UNREACHABLE_EPISODES)}), "
            if args.mtu3d_replica
            else f"(stop called AND dist <= {summary['success_distance_m']} m), "
        )
        + f"SPL l: `{args.l_source}`",
        f"- episodes: {summary['episodes']} "
        f"(duplicates ignored: {duplicates}, without manifest row: {missing_row})",
        f"- **SR: {summary['sr']}**  **SPL: {summary['spl']}**  "
        f"**DTG(geodesic) mean: {summary['dtg_geodesic_mean_m']} m**",
        f"- stopped: {summary['stopped']}",
        f"- failure reasons: {summary['failure_reasons']}",
        "",
        "| category | episodes | SR | SPL |",
        "|---|---|---|---|",
    ]
    for cat, stats in summary["per_category"].items():
        lines.append(
            f"| {cat} | {stats['episodes']} | {stats['sr']} | {stats['spl']} |"
        )
    out_md = args.out_md or (
        args.runs_root / "mtu3d_replica_score.md"
        if args.mtu3d_replica
        else args.runs_root / "official_score.md"
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md} and {out_json}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
