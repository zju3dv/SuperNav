#!/usr/bin/env python3
"""Run make_video.sh for harness runs listed by an instruction file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set

from supernav.paths import resolve_asset_path
from supernav.runtime.config import load_instruction_manifest


KNOWN_ARMS = ("primitive_skill", "wam_skill", "primitive", "wam")


def repo_root_from_here() -> Path:
    from supernav.paths import workspace_root

    return workspace_root()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def instruction_slugs(path: Path, include_hard: bool) -> Set[str]:
    spec = read_json(path)
    rows = load_instruction_manifest(path)["instructions"] if isinstance(spec, dict) else spec
    if not isinstance(rows, list):
        raise ValueError(f"instruction file must contain a list or an 'instructions' list: {path}")
    slugs: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("slug"):
            continue
        if row.get("hard") and not include_hard:
            continue
        slugs.add(str(row["slug"]))
    return slugs


def split_run_id(run_id: str) -> tuple[Optional[str], str]:
    base = run_id.split("__r", 1)[0]
    for arm in KNOWN_ARMS:
        prefix = arm + "_"
        if base.startswith(prefix):
            return arm, base[len(prefix) :]
    if "_" in base:
        arm, slug = base.split("_", 1)
        return arm, slug
    return None, base


def run_slug(run_dir: Path) -> str:
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            data = read_json(run_json)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and data.get("slug"):
            return str(data["slug"])
    return split_run_id(run_dir.name)[1]


def iter_matching_runs(runs_dir: Path, slugs: Set[str], arms: Optional[Set[str]]) -> Iterable[Path]:
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "canonical.jsonl").is_file() and not (run_dir / "run.json").is_file():
            continue
        arm, _ = split_run_id(run_dir.name)
        if arms is not None and arm not in arms:
            continue
        if run_slug(run_dir) in slugs:
            yield run_dir


def parse_arms(value: Optional[str]) -> Optional[Set[str]]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def run_make_video(
    script: Optional[Path], run_name: str, runs_dir: Path, visuals_root: Optional[Path], cwd: Path,
    dry_run: bool, *, accelerated: bool, max_wait_s: float,
) -> int:
    effective_visuals_root = visuals_root or cwd / "data" / "nav_artifacts"
    run_dir = runs_dir / run_name
    if script is None:
        output_name = "replay_accelerated.mp4" if accelerated else "replay_realtime.mp4"
        cmd: Sequence[str] = (
            sys.executable, "-m", "supernav.evaluation.video.render",
            "--run-dir", str(run_dir), "--visuals-root", str(effective_visuals_root),
            "--out", str(run_dir / output_name),
        )
    else:
        cmd = ("bash", str(script), run_name, str(runs_dir), str(effective_visuals_root))
    if accelerated:
        cmd = (*cmd, "--accelerated", "--max-wait-s", str(max_wait_s))
    if dry_run:
        print(" ".join(cmd))
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd))
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply make_video.sh to harness runs from an instruction file.")
    parser.add_argument("--instructions", required=True, help="External instruction manifest.")
    parser.add_argument("--runs-dir", default=None, help="Run discovery root. Defaults to data/runs/main.")
    parser.add_argument("--visuals-root", default=None, help="Visuals root directory. Defaults to data/nav_artifacts.")
    parser.add_argument("--script", default=None, help="Optional shell renderer; defaults to the installed SuperNav renderer.")
    parser.add_argument("--arms", default=None, help="Optional comma-separated arm filter, e.g. primitive,oracle.")
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with replay_realtime.mp4 already present.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after make_video.sh fails for one run.")
    parser.add_argument("--accelerated", action="store_true", help="Render compressed-wait replays.")
    parser.add_argument("--max-wait-s", type=float, default=1.0)
    args = parser.parse_args()

    root = repo_root_from_here()
    instructions = resolve_asset_path(args.instructions, base=root)
    assert instructions is not None
    script = Path(args.script).expanduser() if args.script else None
    if script is not None and not script.is_absolute():
        script = root / script
    if args.runs_dir:
        runs_dir = Path(args.runs_dir).expanduser()
        if not runs_dir.is_absolute():
            runs_dir = root / runs_dir
    else:
        runs_dir = root / "data" / "runs" / "main"

    visuals_root: Optional[Path] = None
    if args.visuals_root:
        visuals_root = Path(args.visuals_root).expanduser()
        if not visuals_root.is_absolute():
            visuals_root = root / visuals_root

    slugs = instruction_slugs(instructions, args.include_hard)
    arms = parse_arms(args.arms)
    runs = list(iter_matching_runs(runs_dir, slugs, arms))
    print(f"[make_all_videos] {len(slugs)} instruction slugs, {len(runs)} matching runs")

    failed: List[str] = []
    for idx, run_dir in enumerate(runs, start=1):
        out_name = "replay_accelerated.mp4" if args.accelerated else "replay_realtime.mp4"
        out_path = run_dir / out_name
        if args.skip_existing and out_path.is_file():
            print(f"[{idx}/{len(runs)}] skip existing {run_dir.name}")
            continue
        print(f"[{idx}/{len(runs)}] video {run_dir.name}")
        rc = run_make_video(script, run_dir.name, runs_dir, visuals_root, root, args.dry_run,
                            accelerated=args.accelerated, max_wait_s=args.max_wait_s)
        if rc != 0:
            failed.append(run_dir.name)
            print(f"[fail] {run_dir.name}: rc={rc}")
            if not args.keep_going:
                break

    if failed:
        print(f"[make_all_videos] failed {len(failed)} run(s): {', '.join(failed)}")
        return 1
    print("[make_all_videos] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
