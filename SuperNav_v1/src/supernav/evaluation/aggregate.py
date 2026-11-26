#!/usr/bin/env python3
"""Aggregate completion claims, costs and explicitly profiled objective scores.

Rows may carry objective_score containing the scorer's success/SPL and criterion,
success_distance_m, l_source, stop_required and unreachable_freebie settings.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from supernav.runtime.config import write_json

COST_FIELDS = ("llm_turns", "movement_tool_calls", "safety_calls", "nav_legs", "nav_steps_total")
PROFILE_FIELDS = ("criterion", "success_distance_m", "l_source", "stop_required", "unreachable_freebie")


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return st.mean(values) if values else None


def task_label(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("scene") or row.get("slug") or "?")


def completion_claim(row: dict[str, Any]) -> bool | None:
    claim = row.get("agent_terminal_claim")
    if isinstance(claim, dict) and claim.get("outcome") is not None:
        return claim["outcome"] == "achieved"
    return row.get("success")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [{"claim": completion_claim(row)} for row in rows]
    profiles = defaultdict(list)
    for row in rows:
        score = row.get("objective_score")
        if not isinstance(score, dict) or any(score.get(key) is None for key in PROFILE_FIELDS):
            continue
        profile = {key: score[key] for key in PROFILE_FIELDS}
        profile.update(protocol=score.get("protocol"), version=score.get("version"),
                       backend=row.get("backend"), benchmark_profile=row.get("benchmark_profile"))
        profiles[json.dumps(profile, sort_keys=True)].append(score)
    objective = []
    for key, scores in sorted(profiles.items()):
        objective.append({
            "profile": json.loads(key), "n": len(scores),
            "sr_scored_n": sum(score.get("success") is not None for score in scores),
            "spl_scored_n": sum(score.get("spl") is not None for score in scores),
            "sr": mean(scores, "success"), "spl": mean(scores, "spl"),
        })
    scored_n = sum(group["sr_scored_n"] for group in objective)
    return {
        "n": len(rows),
        "agent_completion_claims_n": sum(claim["claim"] is not None for claim in claims),
        "agent_completion_rate": mean(claims, "claim"),
        "objective_scored_n": scored_n, "objective_unscored_n": len(rows) - scored_n,
        "objective_sr": objective[0]["sr"] if len(objective) == 1 else None,
        "objective_spl": objective[0]["spl"] if len(objective) == 1 else None,
        "objective_profiles": objective,
        **{f"{key}_mean": mean(rows, key) for key in COST_FIELDS},
    }


def display(value: float | None, digits: int = 2) -> str:
    return "null" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/runs/main")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    results_path = runs_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()] if results_path.is_file() else []
    cells = defaultdict(list)
    for row in rows:
        cells[(task_label(row), str(row.get("slug", "?")), str(row.get("arm", "?")))].append(row)
    summary = {"runs": len(rows), "cells": {}, "arms": {}}
    print("=== per (task, instruction, arm) ===")
    print(f"{'task':42s} {'instr':32s} {'arm':22s} {'n':>3s} {'claim':>6s} {'SR':>6s} {'SPL':>6s} {'scored':>6s}")
    for (task, slug, arm), group in sorted(cells.items()):
        result = summarize(group)
        summary["cells"][f"{task}/{slug}/{arm}"] = {"task_id": task, "slug": slug, "arm": arm, **result}
        print(f"{task[:42]:42s} {slug[:32]:32s} {arm[:22]:22s} {len(group):3d} "
              f"{display(result['agent_completion_rate']):>6s} {display(result['objective_sr']):>6s} "
              f"{display(result['objective_spl']):>6s} {result['objective_scored_n']:6d}")
    print("\n=== per arm ===")
    for arm in sorted({str(row.get("arm", "?")) for row in rows}):
        result = summarize([row for row in rows if str(row.get("arm", "?")) == arm])
        summary["arms"][arm] = result
        print(json.dumps({arm: result}, ensure_ascii=False))
    write_json(runs_dir / "summary.json", summary)
    print("\nclaim = agent completion claim; SR/SPL = objective scores grouped by scoring settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
