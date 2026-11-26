"""Export one run per public GS episode using private primitive-action evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from supernav.evaluation.objectnav._habitatgs_v1 import position
from supernav.evaluation.objectnav.habitatgs_data import (
    TASKS, load_release, read_json, sha256, write_new,
)
from supernav.runtime.streams import tool_result_payload


def trace_events(path: Path, episode: dict) -> list[dict]:
    events, previous, step = [], None, 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            action = row.get("action")
            if not events:
                if action != "start" or row.get("schema") != "supernav_primitive_trace/v1":
                    raise ValueError("primitive trace must start with a versioned start event")
                if row.get("scene") != episode["scene"]["id"] or row.get("step_count") != 0:
                    raise ValueError("primitive trace scene/start counter mismatch")
                if math.dist(position(row["position"]), episode["start"]["position"]) > 1e-4:
                    raise ValueError("primitive trace start differs from published start")
            elif action == "runtime_error":
                # The bridge records invalid state changes explicitly.
                events.append({"action": action})
                break
            elif action == "stop":
                if row.get("step_count") != step or math.dist(position(row["position"]), previous) > 1e-4:
                    raise ValueError("STOP changed pose or has a discontinuous action counter")
            else:
                if action in (None, "", "start", "budget_exhausted") or not isinstance(action, str):
                    raise ValueError("invalid primitive action")
                step += 1
                if row.get("step_count") != step:
                    raise ValueError("primitive trace has missing/repeated actions")
            previous = position(row["position"])
            events.append({"action": action, "position": previous})
            if action == "stop":
                break  # The public contract ends at the first terminal event.
    if not events:
        raise ValueError("empty primitive trace")
    return events


def validate_session_binding(path: Path, session: str) -> None:
    """Do not select the final session from an episode with in-run restarts."""
    initialized = []
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("type") == "tool_result" and event.get("name") == "hab_init_scene":
                ident = tool_result_payload(event).get("session_id")
                if ident:
                    initialized.append(ident)
    if initialized != [session]:
        raise ValueError("expected exactly one initialized session matching metrics; audit missing evidence or in-run restarts")


def export_runs(*, episodes: list[dict], provenance: dict, runs_root: Path, visuals_root: Path) -> dict:
    lookup = {e["episode_id"]: e for e in episodes}
    submissions, sources, seen = [], [], set()
    if not runs_root.is_dir() or not visuals_root.is_dir():
        raise ValueError("runs root and visuals root must exist")
    # One selected method/repetition per directory. Never pick the best retry.
    for run_path in sorted(runs_root.glob("*/run.json")):
        directory = run_path.parent
        run = read_json(run_path)
        ident = run.get("task_id")
        if ident not in lookup or ident in seen:
            raise ValueError(f"unknown or duplicate run ID: {ident}; select one method/repetition")
        seen.add(ident)
        task = read_json(directory / "task.resolved.json")
        source = task.get("habitatgs", {})
        if source.get("episode_id") != ident or any(source.get(k) != provenance[k] for k in ("episodes_sha256", "scene_manifest_sha256")):
            raise ValueError(f"run is not bound to this public dataset: {ident}")
        execution = read_json(directory / "execution.resolved.json")
        if execution.get("dry_run"):
            raise ValueError("dry-run output is not navigation evidence")
        metrics = read_json(directory / "metrics.json")
        if metrics.get("task_id") != ident:
            raise ValueError("metrics/run ID mismatch")
        session = metrics.get("session_id")
        if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", session):
            raise ValueError(f"missing/invalid session ID for {ident}; preserve and audit infrastructure failures")
        validate_session_binding(directory / "canonical.jsonl", session)
        path = visuals_root / f"{session}.primitive.jsonl"
        # Do not infer primitive motion from sparse/deduplicated legacy paths.
        events = trace_events(path, lookup[ident])
        if events[-1]["action"] not in ("stop", "runtime_error"):
            if metrics.get("timed_out") or run.get("terminal_status") == "timeout":
                events.append({"action": "budget_exhausted"})
            elif metrics.get("returncode") is None:
                raise ValueError(f"run has no recorded process completion: {ident}")
            elif metrics["returncode"] != 0:
                events.append({"action": "runtime_error"})
            # A normal process exit without explicit STOP remains missing_stop.
        submissions.append({"episode_id": ident, "events": events})
        sources.append({"episode_id": ident, "run_dir": str(directory.absolute()),
                        "trace_sha256": sha256(path), "metrics_sha256": sha256(directory / "metrics.json"),
                        "task_sha256": sha256(directory / "task.resolved.json"),
                        "canonical_sha256": sha256(directory / "canonical.jsonl"),
                        "execution_sha256": sha256(directory / "execution.resolved.json")})
    if not submissions:
        raise ValueError("no run directories found")
    return {"episodes": submissions, "provenance": provenance, "sources": sources}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--visuals-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new file")
    episodes, _, provenance = load_release(args.dataset_root, args.task)
    payload = export_runs(episodes=episodes, provenance=provenance, runs_root=args.runs_root, visuals_root=args.visuals_root)
    write_new(args.output, payload)
    print(f"Exported {len(payload['episodes'])}/{len(episodes)} episodes; absent submissions score as failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
