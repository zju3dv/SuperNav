"""Score public HabitatGS v1 submissions, including ordered prefix SR1...SRk."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from supernav.evaluation.objectnav._habitatgs_v1 import aggregate, score_episode
from supernav.evaluation.objectnav.habitatgs_data import (
    TASKS, asset_path, load_release, navmesh_name, read_json, sha256, verify_file, write_new,
)


def stage_success_rates(results: list[dict]) -> list[dict]:
    """Prefix completion does not require STOP; use eligible evaluated episodes.

    Keep total/error denominators alongside the public contract's evaluated
    denominator. Missing submissions remain failures. Never count unordered hits.
    """
    rows = []
    for k in range(1, max((r["target_count"] for r in results), default=0) + 1):
        eligible = [r for r in results if r["target_count"] >= k]
        evaluated = [r for r in eligible if r["status"] != "runtime_error"]
        successes = sum(r["visited_targets"] >= k for r in evaluated)
        rows.append({"stage": k, "eligible_total": len(eligible), "evaluated": len(evaluated),
                     "runtime_errors": len(eligible) - len(evaluated), "successes": successes,
                     "success_rate": successes / len(evaluated) if evaluated else None})
    return rows


def score_submissions(episodes, submissions, geometry, reference="global_shortest"):
    """geometry(episode) supplies evaluator-only (geodesic-nearest, navigable)."""
    ids = {e["episode_id"] for e in episodes}
    by_id = {}
    for row in submissions["episodes"]:
        ident = row["episode_id"]
        if ident not in ids or ident in by_id:
            raise ValueError(f"unknown or duplicate submission ID: {ident}")
        by_id[ident] = row
    results = []
    for episode in episodes:
        submission = by_id.get(episode["episode_id"])
        # Missing submissions still use the full denominator but need no mesh.
        nearest, navigable = geometry(episode) if submission is not None else (None, None)
        results.append(score_episode(episode, submission, nearest, navigable, reference))
    payload = {"schema": "supernav_habitatgs_scores/v1", "summary": aggregate(results), "episodes": results}
    if any(e["ordered"] for e in episodes):
        payload["stage_sr"] = stage_success_rates(results)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=("global_shortest", "local_shortest"), default="global_shortest")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new file")
    episodes, assets, provenance = load_release(args.dataset_root, args.task)
    submissions = read_json(args.submissions)
    supplied = submissions.get("provenance", {})
    if supplied and any(supplied.get(k) != provenance[k] for k in ("episodes_sha256", "scene_manifest_sha256")):
        raise ValueError("submission and scoring dataset hashes differ")
    # No scene rendering or policy/model dependencies are needed for scoring.
    from supernav.backends.habitat.loader import prepare_simulator_import
    prepare_simulator_import()
    import habitat_sim

    current = None
    pf = habitat_sim.PathFinder()
    checked = set()

    def geometry(episode):
        nonlocal current
        name = navmesh_name(episode)
        path = asset_path(args.scene_root, name)
        if name not in checked:
            verify_file(path, assets[name])
            checked.add(name)
        if name != current:
            if not pf.load_nav_mesh(str(path)):
                raise ValueError(f"cannot load navmesh: {path}")
            current = name

        def nearest(start, goals):
            query = habitat_sim.MultiGoalShortestPath()
            query.requested_start, query.requested_ends = start, goals
            return float(query.geodesic_distance) if pf.find_path(query) else math.inf

        return nearest, pf.is_navigable

    payload = score_submissions(episodes, submissions, geometry, args.reference)
    payload["provenance"] = {**provenance, "submissions_sha256": sha256(args.submissions),
                             "reference": args.reference, "navmesh_checksums_verified": True,
                             "verified_navmeshes": sorted(checked)}
    write_new(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k in ("summary", "stage_sr")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
