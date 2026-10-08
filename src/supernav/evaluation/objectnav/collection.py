#!/usr/bin/env python3
"""Collect finalized ObjectNav results from local/SSH sources and rescore them."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supernav.runtime.config import load_instruction_manifest

# Copy result evidence, never a generated agent home containing login credentials.
ARTIFACTS = (
    "run.json",
    "metrics.json",
    "command.json",
    "canonical.jsonl",
    "raw.jsonl",
    "codex_session.jsonl",
    "stderr.log",
    "manifest.json",
    "prompt.txt",
    "skill_visibility_report.json",
    "timing_summary.json",
    "timing_summary.svg",
)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def inventory(runs_root: Path, visuals_root: Path, min_age_s: float) -> dict:
    """Only direct child run directories with stable, finalized metrics qualify."""
    cutoff = time.time() - min_age_s
    result: dict[str, Any] = {"runs": [], "pending": 0, "invalid": []}
    if not runs_root.is_dir() or not visuals_root.is_dir():
        raise FileNotFoundError("runs_root and visuals_root must both exist")
    for directory in sorted(runs_root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        if not (directory / "run.json").is_file():
            continue
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file() or metrics_path.stat().st_mtime > cutoff:
            result["pending"] += 1
            continue
        try:
            run = read_json(directory / "run.json")
            metrics = read_json(metrics_path)
            task_id = run["task_id"]
            if not task_id or metrics.get("task_id") != task_id:
                raise ValueError("run/metrics task_id mismatch")
            if not isinstance(metrics.get("close_called"), bool):
                raise ValueError("metrics lack a boolean close_called")
            if not SAFE_COMPONENT.fullmatch(directory.name):
                raise ValueError("unsafe run directory name")
            session_id = metrics.get("session_id")
            if session_id and not SAFE_COMPONENT.fullmatch(str(session_id)):
                raise ValueError("unsafe session_id")
            trajectory = f"{session_id}.trajectory.json" if session_id else None
            if trajectory and not (visuals_root / trajectory).is_file():
                trajectory = None
            if metrics["close_called"] and not trajectory:
                raise ValueError("STOP result is missing its trajectory")
            files = [
                name
                for name in ARTIFACTS
                if (directory / name).is_file() and not (directory / name).is_symlink()
            ]
            if not {"run.json", "metrics.json"}.issubset(files):
                raise ValueError("run/metrics must be regular files")
            result["runs"].append(
                {
                    "directory": directory.name,
                    "task_id": task_id,
                    "metrics_mtime": metrics_path.stat().st_mtime,
                    "files": files,
                    "trajectory": trajectory,
                    "model": metrics.get("model"),
                    "model_provider": metrics.get("model_provider"),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["invalid"].append({"directory": directory.name, "error": str(exc)})
    return result


def run_command(command: list[str], *, timeout: float = 900, **kwargs: Any) -> bytes:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}: "
            + result.stderr.decode(errors="replace")[-3000:]
        )
    return result.stdout


def source_inventory(source: dict, min_age_s: float) -> dict:
    if not source.get("host"):
        return inventory(
            Path(source["runs_root"]), Path(source["visuals_root"]), min_age_s
        )
    command = [
        source.get("python", "python3"),
        source["collector"],
        "--inventory-only",
        "--runs-root",
        source["runs_root"],
        "--visuals-root",
        source["visuals_root"],
        "--min-age-s",
        str(min_age_s),
    ]
    return json.loads(
        run_command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                source["host"],
                shlex.join(command),
            ],
            timeout=120,
        )
    )


def sync_files(
    source: dict, source_root: str, files: list[str], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not files:
        return
    origin = source_root.rstrip("/") + "/"
    if source.get("host"):
        origin = source["host"] + ":" + origin
    run_command(
        [
            "rsync",
            "-rt",
            "--checksum",
            "--protect-args",
            "--timeout=120",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=15",
            "--from0",
            "--files-from=-",
            origin,
            str(destination) + "/",
        ],
        input=("\0".join(files) + "\0").encode(),
    )


def collect_sources(config: dict, output: Path) -> tuple[dict, list[dict]]:
    inventories = {}
    candidates = []
    for source in config["sources"]:
        name = source["name"]
        if not SAFE_COMPONENT.fullmatch(name) or name in inventories:
            raise ValueError("source names must be unique safe path components")
        found = source_inventory(source, float(config.get("min_age_s", 60)))
        if found["invalid"]:
            raise ValueError(f"invalid completed results on {name}: {found['invalid']}")
        cache = output / "collected" / name
        sync_files(
            source,
            source["runs_root"],
            [
                f"{row['directory']}/{file}"
                for row in found["runs"]
                for file in row["files"]
            ],
            cache / "runs",
        )
        sync_files(
            source,
            source["visuals_root"],
            sorted({row["trajectory"] for row in found["runs"] if row["trajectory"]}),
            cache / "visuals",
        )
        inventories[name] = found
        for row in found["runs"]:
            candidates.append({**row, "source": name, "cache": str(cache)})
    return inventories, candidates


def select_latest(candidates: list[dict], task_ids: set[str]) -> list[dict]:
    selected: dict[str, dict] = {}
    for row in candidates:
        task_id = row["task_id"]
        if task_id not in task_ids:
            raise ValueError(f"task outside the experiment manifest: {task_id}")
        previous = selected.get(task_id)
        order = (row["metrics_mtime"], row["source"], row["directory"])
        if previous is None or order > (
            previous["metrics_mtime"],
            previous["source"],
            previous["directory"],
        ):
            selected[task_id] = row
    return [selected[key] for key in sorted(selected)]


def freeze_inputs(selected: list[dict], snapshot: Path) -> None:
    (snapshot / "runs").mkdir(parents=True)
    (snapshot / "visuals").mkdir()
    for row in selected:
        cache = Path(row["cache"])
        source = cache / "runs" / row["directory"]
        metrics = read_json(source / "metrics.json")
        if metrics.get("task_id") != row["task_id"]:
            raise ValueError("metrics changed during collection")
        if abs((source / "metrics.json").stat().st_mtime - row["metrics_mtime"]) > 1:
            raise ValueError("metrics changed during collection; retry next invocation")
        target = snapshot / "runs" / (row["source"] + "__" + row["directory"])
        target.mkdir()
        for name in ("run.json", "metrics.json"):
            shutil.copy2(source / name, target / name)
        if row["trajectory"]:
            trajectory = cache / "visuals" / row["trajectory"]
            # Validate before scoring; malformed sidecars must not become failures.
            from supernav.evaluation.objectnav.score_hm3d import load_trajectory

            positions = load_trajectory(trajectory)
            if metrics["close_called"] and not positions:
                raise ValueError(
                    f"STOP result has an empty trajectory: {row['task_id']}"
                )
            shutil.copy2(trajectory, snapshot / "visuals" / trajectory.name)


def score_snapshot(config: dict, selected: list[dict], snapshot: Path) -> dict:
    from supernav.evaluation.objectnav.score_hm3d import find_navmesh

    manifest = load_instruction_manifest(config["instructions"])
    rows = {row["task_id"]: row for row in manifest["instructions"]}
    scenes = Path(config["scenes_root"])
    checked = set()
    for task in selected:
        row = rows[task["task_id"]]
        if row["scene"] in checked:
            continue
        navmesh = find_navmesh(scenes, row["ground_truth"]["hm3d_split"], row["scene"])
        if navmesh is None:
            raise FileNotFoundError(f"missing official navmesh for {row['scene']}")
        from supernav.backends.habitat.geometry import navmesh_loads

        if not navmesh_loads(navmesh):
            raise ValueError(f"cannot load official navmesh: {navmesh}")
        checked.add(row["scene"])
    original = read_json(Path(config["instructions"]))
    if original == manifest:
        shutil.copy2(config["instructions"], snapshot / "instructions.json")
    else:
        # The scorer runs against a self-contained snapshot. Keep the original
        # recipe too, so its include/default declarations remain inspectable.
        shutil.copy2(config["instructions"], snapshot / "instructions.source.json")
        write_json(snapshot / "instructions.json", manifest)
    scores = {}
    for label, criterion, distance, l_source in (
        ("0.2m", "viewpoint_geodesic", "0.2", "viewpoint_geodesic"),
        ("1m", "euclidean_viewpoint", "1.0", "episode"),
    ):
        json_path = snapshot / f"score_{label}.json"
        if selected:
            stdout = run_command(
                [
                    sys.executable,
                    "-m",
                    "supernav.evaluation.objectnav.score_hm3d",
                    "--runs-root",
                    str(snapshot / "runs"),
                    "--instructions",
                    str(snapshot / "instructions.json"),
                    "--visuals-root",
                    str(snapshot / "visuals"),
                    "--scenes-root",
                    str(scenes),
                    "--criterion",
                    criterion,
                    "--success-distance",
                    distance,
                    "--l-source",
                    l_source,
                    "--out-json",
                    str(json_path),
                    "--out-md",
                    str(snapshot / f"score_{label}.md"),
                ],
                timeout=1800,
            )
            (snapshot / f"score_{label}.log").write_bytes(stdout)
            result = read_json(json_path)
            if result["summary"]["episodes"] != len(selected):
                raise ValueError(
                    "scorer denominator does not match completed task count"
                )
            scores[label] = {
                **result["summary"],
                "successes": sum(bool(row["success"]) for row in result["episodes"]),
            }
        else:
            scores[label] = {"episodes": 0, "successes": 0, "sr": None, "spl": None}
            write_json(json_path, {"summary": scores[label], "episodes": []})
    return scores


def collect_and_score(config: dict, output: Path) -> Path:
    started = datetime.now(timezone.utc)
    inventories, candidates = collect_sources(config, output)
    manifest = load_instruction_manifest(config["instructions"])["instructions"]
    task_ids = {row["task_id"] for row in manifest}
    if len(task_ids) != len(manifest):
        raise ValueError("instruction manifest contains duplicate task IDs")
    selected = select_latest(candidates, task_ids)
    snapshot = output / "history" / started.strftime("%Y%m%dT%H%M%S.%fZ")
    freeze_inputs(selected, snapshot)
    write_json(snapshot / "inventory.json", inventories)
    write_json(snapshot / "selected.json", selected)
    scores = score_snapshot(config, selected, snapshot)
    summary = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "completed": len(selected),
        "total": len(task_ids),
        "remaining": len(task_ids) - len(selected),
        "completed_by_source": {
            name: len(v["runs"]) for name, v in inventories.items()
        },
        "duplicates_ignored": len(candidates) - len(selected),
        "scores": scores,
        "definition": "Explicit STOP and distance < threshold; existing scorer semantics.",
    }
    write_json(snapshot / "summary.json", summary)
    lines = [
        "# HM3Dv2 completed-run scores",
        "",
        f"Updated (UTC): {summary['finished_at']}",
        f"Completed: **{len(selected)}/{len(task_ids)}**; remaining: {summary['remaining']}",
        f"Sources: {summary['completed_by_source']}; duplicates ignored: {summary['duplicates_ignored']}",
        "",
        "| Criterion | Successes / completed | SR | SPL |",
        "|---|---:|---:|---:|",
    ]
    for label, score in scores.items():
        rate = f"{score['sr']:.2%}" if score["sr"] is not None else "N/A"
        lines.append(
            f"| {label} | {score['successes']}/{len(selected)} | {rate} | {score['spl']} |"
        )
    lines += [
        "",
        "0.2m: viewpoint geodesic; 1m: viewpoint Euclidean. Both require STOP.",
        "The existing scorer uses strict < thresholds. Pending runs are excluded.",
        "Completed no-STOP outcomes count as failures. Rates describe the completed subset.",
    ]
    (snapshot / "summary.md").write_text("\n".join(lines) + "\n")
    # A single atomic pointer publishes both scores together, only after success.
    temporary = output / ".latest.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(snapshot.relative_to(output), target_is_directory=True)
    temporary.replace(output / "latest")
    (output / "last_error.json").unlink(missing_ok=True)
    print("\n".join(lines), flush=True)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--visuals-root", type=Path)
    parser.add_argument("--min-age-s", type=float, default=60)
    args = parser.parse_args()
    if args.inventory_only:
        if args.runs_root is None or args.visuals_root is None:
            parser.error("inventory requires --runs-root and --visuals-root")
        print(json.dumps(inventory(args.runs_root, args.visuals_root, args.min_age_s)))
        return 0
    if args.config is None:
        parser.error("--config is required")
    os.umask(0o077)
    config = read_json(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collect.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Previous collection is still running; skipped.", flush=True)
            return 0
        try:
            collect_and_score(config, output)
        except Exception as exc:
            write_json(
                output / "last_error.json",
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                },
            )
            print(
                f"Collection failed; previous latest report retained: {exc}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
