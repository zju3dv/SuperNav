# Copyright (c) 2026 ZJU3DV Group at Zhejiang University. See LICENSE.
# Vendored scoring functions from yyh929/HabitatGS-Benchmark evaluation/score.py
# revision 4a8fe5cf654c1d19f0c3c992720d3c4f57844f12
# upstream file SHA256: 37bb98a11141eae72fa68b464a8c1a9ea32773b83e74ac5745a4211992b6716e
# Keep score_episode/aggregate behavior identical to that public contract.
"""Score recorded ObjectNav/ordered Multi-ObjectNav trajectories on their navmesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def position(value):
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in value
    ):
        raise ValueError("position must contain three finite numbers")
    return value


def score_episode(episode, submission, nearest, navigable, reference="global_shortest"):
    """Callbacks are evaluator-owned; no GT or target arrival feedback goes to policy."""
    if episode["schema"] != "habitat_gs/objectnav/v1" or episode["evaluation"]["success"] != {
        "metric": "distance_to_approved_viewpoint_set", "threshold_m": 0.1, "comparator": "<"
    }:
        raise ValueError("unsupported episode scoring contract")
    targets = episode["targets"]
    if not targets or (not episode["ordered"] and len(targets) != 1):
        raise ValueError("expected a single goal or ordered goals")
    goals = []
    for target in targets:
        views = target["viewpoints"]
        if not views or any(v["status"] != "approved" for v in views):
            raise ValueError("release contains missing or unapproved viewpoints")
        goals.append([position(v["pose"]["position"]) for v in views])
    reference_length = episode["ground_truth"]["variants"][reference]["total_geodesic_distance"]
    if not math.isfinite(reference_length) or reference_length < 0:
        raise ValueError("invalid reference distance")
    result = {
        "episode_id": episode["episode_id"], "status": "missing_submission",
        "success": False, "spl": 0.0, "visited_targets": 0,
        "target_count": len(targets), "path_length_m": 0.0,
        "reference_variant": reference, "reference_length_m": reference_length,
    }
    if submission is None:
        return result
    events = submission.get("events")
    if not isinstance(events, list) or not events or events[0].get("action") != "start":
        raise ValueError("events must begin with the actual start pose")
    previous = position(events[0]["position"])
    if math.dist(previous, position(episode["start"]["position"])) > 1e-4:
        raise ValueError("trajectory start differs from episode start")
    active, length = 0, 0.0
    result["status"] = "missing_stop"
    for index, event in enumerate(events):
        action = event.get("action")
        if not isinstance(action, str) or not action or (index and action == "start"):
            raise ValueError("each event needs an action; start may occur only once")
        if action in {"runtime_error", "budget_exhausted"}:
            result["status"] = action
            break
        current = position(event["position"])
        if not navigable(current):
            raise ValueError("recorded position is not on the evaluation navmesh")
        delta = math.dist(previous, current)
        if action == "stop" and delta > 1e-4:
            raise ValueError("STOP cannot move the agent")
        length += delta
        previous = current
        if episode["ordered"]:
            # Several successive target regions may overlap at the same observed pose.
            while active < len(goals):
                distance = nearest(current, goals[active])
                if not (math.isfinite(distance) and 0 <= distance < 0.1):
                    break
                active += 1
        if action == "stop":
            if episode["ordered"]:
                success = active == len(goals)
            else:
                distance = nearest(current, goals[0])
                success = math.isfinite(distance) and 0 <= distance < 0.1
                active = int(success)
            result.update(status="stopped", success=success)
            break
    result.update(visited_targets=active, path_length_m=length)
    if result["status"] == "runtime_error":
        result.update(success=None, spl=None)
    elif result["success"]:
        denominator = max(reference_length, length)
        result["spl"] = reference_length / denominator if denominator else 1.0
    return result


def aggregate(results):
    evaluated = [r for r in results if r["status"] != "runtime_error"]
    return {
        "total": len(results), "evaluated": len(evaluated),
        "runtime_errors": len(results) - len(evaluated),
        "successes": sum(bool(r["success"]) for r in evaluated),
        "success_rate": sum(bool(r["success"]) for r in evaluated) / len(evaluated) if evaluated else None,
        "spl": sum(r["spl"] for r in evaluated) / len(evaluated) if evaluated else None,
        "missing_submissions": sum(r["status"] == "missing_submission" for r in results),
    }
