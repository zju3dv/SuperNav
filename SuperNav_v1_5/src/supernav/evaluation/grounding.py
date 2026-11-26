#!/usr/bin/env python3
"""Grounding DINO objective arrival judge for harness runs.

A) Domain-gap probe: detection on the 6 look-around GS snapshots + precise-angle
   from box centers (90 deg hfov).
B) OBJECTIVE arrival judge: for each run, detect the per-task target phrase
   over the brain's LAST K observation frames with PER-TASK area gates. A run
   passes if ANY detection in ANY of those frames clears score+area gates.
   Compares against the brain's self-reported success.

The tool reads ``canonical.jsonl`` produced by the harness (not the raw agent
stream), so it works with any agent backend.

Usage:
    python -m supernav grounding-check \
        --runs-dir data/runs/main \
        --instructions configs/benchmarks/main/instructions.json \
        --visuals-root data/nav_artifacts \
        --out /tmp/grounding_check \
        --k 3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from supernav.runtime.streams import iter_json_lines  # noqa: E402

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SMOKE_SESSION = "a6793e08-0c20-4f03-8cb1-2b3ff93912a6"
SMOKE_PHRASES = "a bed. a window. a painting. a sofa. a door."
HFOV_DEG = 90.0

TARGETS: Dict[str, str] = {
    "sleep": "a bed.",
    "window": "a window.",
    "rest": "a sofa. an armchair. a bed.",
    "read": "an armchair. a chair. a sofa.",
    "art": "a painting. a framed picture.",
}
SCORE_T = 0.35
# Per-task minimum box-area ratio for "arrived": furniture is approached close
# (large box); windows are mid-distance; wall art is viewed from standoff.
AREA_T: Dict[str, float] = {
    "sleep": 0.08,
    "rest": 0.08,
    "window": 0.06,
    "read": 0.05,
    "art": 0.03,
}
ARMS = ("primitive_skill", "primitive", "wam_skill", "wam", "oracle", "oracle_skill")


def load_model():
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(dev).eval()
    return proc, model, dev


def detect(proc, model, dev, img: Image.Image, phrases: str):
    import torch

    with torch.no_grad():
        inputs = proc(images=img, text=phrases, return_tensors="pt").to(dev)
        out = model(**inputs)
        kw = dict(input_ids=inputs.input_ids, target_sizes=[img.size[::-1]])
        try:
            res = proc.post_process_grounded_object_detection(
                out, box_threshold=SCORE_T, text_threshold=0.25, **kw
            )[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(
                out, threshold=SCORE_T, text_threshold=0.25, **kw
            )[0]
        dets = []
        labels = res.get("text_labels", res.get("labels", []))
        for score, label, box in zip(res["scores"], labels, res["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box]
            W, H = img.size
            dets.append(
                {
                    "label": str(label),
                    "score": round(float(score), 3),
                    "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "area_ratio": round((x1 - x0) * (y1 - y0) / (W * H), 3),
                    "angle_off_deg": round(
                        math.degrees(
                            math.atan(
                                ((x0 + x1) / 2 - W / 2)
                                / (W / 2)
                                * math.tan(math.radians(HFOV_DEG / 2))
                            )
                        ),
                        1,
                    ),
                }
            )
        return sorted(dets, key=lambda d: -d["score"])


def annotate(img, dets, path):
    from PIL import ImageDraw, ImageFont

    im = img.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        fnt = ImageFont.load_default()
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        dr.rectangle([x0, y0, x1, y1], outline=(80, 230, 120), width=2)
        t = f"{d['label']} {d['score']:.2f}"
        ty = max(0.0, y0 - 18)
        dr.rectangle([x0, ty, x0 + dr.textlength(t, font=fnt) + 6, ty + 18], fill=(0, 0, 0))
        dr.text((x0 + 3, ty + 1), t, font=fnt, fill=(120, 255, 160))
    im.save(path)


def _image_paths_from_value(value: Any) -> List[str]:
    """Recursively collect image paths from a parsed JSON value."""
    paths: List[str] = []
    if isinstance(value, str):
        if value.endswith(".png") or value.endswith(".jpg") or value.endswith(".jpeg"):
            paths.append(value)
    elif isinstance(value, dict):
        for key in ("path", "original_path"):
            if isinstance(value.get(key), str):
                paths.append(value[key])
        for v in value.values():
            paths.extend(_image_paths_from_value(v))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_image_paths_from_value(item))
    return paths


def last_observation_frames(canonical_path: Path, visuals_root: Path, k: int) -> List[str]:
    """Return up to the last k observation frame paths from canonical.jsonl.

    Frame paths are deduplicated while preserving order. Relative paths are
    resolved against ``visuals_root``.
    """
    seen: set[str] = set()
    frames: List[str] = []
    for event in iter_json_lines(canonical_path):
        if event.get("type") != "tool_result":
            continue
        content = event.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for raw_path in _image_paths_from_value(payload):
            path = Path(raw_path)
            if not path.is_absolute():
                path = visuals_root / path
            path_str = str(path)
            if path_str in seen:
                continue
            seen.add(path_str)
            frames.append(path_str)
    # newest first
    return list(reversed(frames[-k:])) if frames else []


def resolve_visuals_root(run_dir: Path, visuals_root: Optional[Path]) -> Path:
    if visuals_root is not None:
        return visuals_root
    # Try to infer from canonical.jsonl frame paths (usually absolute).
    canonical = run_dir / "canonical.jsonl"
    if canonical.is_file():
        for event in iter_json_lines(canonical):
            if event.get("type") != "tool_result":
                continue
            content = event.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            for raw_path in _image_paths_from_value(payload):
                p = Path(raw_path)
                if p.is_absolute():
                    return p.parent.parent
    # Fall back to common defaults.
    for name in ("NAV_VISUAL_OUTPUT_DIR", "HAB_VISUAL_OUTPUT_DIR", "NAV_ARTIFACTS_DIR"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return Path("/tmp/habitat_gs_visuals")


def judge_run(
    proc,
    model,
    dev,
    run_dir: Path,
    slug: str,
    k: int,
    outdir: Path,
    visuals_root: Optional[Path],
):
    """Objective arrival verdict over the last k observation frames."""
    from PIL import Image

    vroot = resolve_visuals_root(run_dir, visuals_root)
    frames = last_observation_frames(run_dir / "canonical.jsonl", vroot, k=k)
    gate = AREA_T.get(slug, 0.08)
    best_hit, best_any, used = None, None, None
    for fp in frames:  # newest first
        if not os.path.isfile(fp):
            continue
        img = Image.open(fp).convert("RGB")
        dets = detect(proc, model, dev, img, TARGETS[slug])
        for d in dets:
            if best_any is None or d["score"] > best_any["score"]:
                best_any = d
        hit = next((d for d in dets if d["score"] >= SCORE_T and d["area_ratio"] >= gate), None)
        if hit and (best_hit is None or hit["score"] > best_hit["score"]):
            best_hit, used = hit, fp
            annotate(img, dets, str(outdir / f"judge_{run_dir.name}.png"))
    return {
        "run": run_dir.name,
        "objective": best_hit is not None,
        "hit": best_hit,
        "best_any": best_any,
        "frame": used,
        "n_frames_checked": len(frames),
    }


def split_run_name(run_id: str) -> Tuple[Optional[str], str]:
    base = run_id.split("__r", 1)[0]
    for arm in ARMS:
        prefix = arm + "_"
        if base.startswith(prefix):
            return arm, base[len(prefix) :]
    if "_" in base:
        arm, slug = base.split("_", 1)
        return arm, slug
    return None, base


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            return json.loads(run_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    # Fallback: parse run_id like arm_slug__r2
    arm, slug = split_run_name(run_dir.name)
    return {"arm": arm, "slug": slug, "rep": None}


def load_self_reports(runs_dir: Path) -> Dict[str, bool]:
    selfrep: Dict[str, bool] = {}
    results_path = runs_dir / "results.jsonl"
    if not results_path.is_file():
        return selfrep
    for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = f"{d['arm']}_{d['slug']}" + (f"__r{d['rep']}" if d.get("rep") else "")
        selfrep[key] = bool(d.get("success"))
    return selfrep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="data/runs/main")
    ap.add_argument("--instructions", default=None, help="instruction-set json with per-slug judge phrases+gates")
    ap.add_argument("--visuals-root", default=None)
    ap.add_argument("--out", default="/tmp/grounding_check")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--slugs", default=None, help="comma-separated slugs to judge; default uses TARGETS keys")
    args = ap.parse_args()
    from PIL import Image


    runs_dir = Path(args.runs_dir).expanduser().resolve()
    visuals_root = Path(args.visuals_root).expanduser().resolve() if args.visuals_root else None
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.instructions:
        spec = json.load(open(args.instructions))
        for ins in spec.get("instructions", []):
            if ins.get("judge"):
                TARGETS[ins["slug"]] = ins["judge"]
                AREA_T[ins["slug"]] = float(ins.get("gate", 0.05))

    allowed_slugs = set(args.slugs.split(",")) if args.slugs else None

    proc, model, dev = load_model()
    print(f"model loaded on {dev}", file=sys.stderr)
    results: Dict[str, Any] = {"smoke": [], "judge": []}

    if not args.skip_smoke:
        smoke_dir = f"/tmp/habitat_gs_visuals/{SMOKE_SESSION}"
        snaps = [f"{smoke_dir}/step{1+6*i:06d}_color_sensor.png" for i in range(6)]
        for i, fp in enumerate([p for p in snaps if os.path.isfile(p)]):
            img = Image.open(fp).convert("RGB")
            dets = detect(proc, model, dev, img, SMOKE_PHRASES)
            heading = i * 60
            annotate(img, dets, str(outdir / f"smoke_{heading:03d}deg.png"))
            results["smoke"].append({"heading_deg": heading, "detections": dets})

    selfrep = load_self_reports(runs_dir)

    print(f"\n{'run':22s} {'self':>4s} {'judge':>5s}  best detection")
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "canonical.jsonl").is_file():
            continue
        info = load_run_info(run_dir)
        slug = str(info.get("slug") or "")
        arm = str(info.get("arm") or "")
        rep = info.get("rep")
        slug_base = re.sub(r"__r\d+$", "", slug)
        if slug_base not in TARGETS:
            continue
        if allowed_slugs is not None and slug_base not in allowed_slugs:
            continue
        v = judge_run(proc, model, dev, run_dir, slug_base, args.k, outdir, visuals_root)
        key = f"{arm}_{slug}" + (f"__r{rep}" if rep else "")
        v["self_report"] = selfrep.get(key)
        results["judge"].append(v)
        d = v["hit"] or v["best_any"]
        b = f"{d['label']} {d['score']:.2f} area={d['area_ratio']:.2f}" if d else "(none)"
        print(f"{run_dir.name:22s} {str(v['self_report']):>4s} {str(v['objective']):>5s}  {b}")

    json.dump(results, open(outdir / "results.json", "w"), indent=1)
    n = len(results["judge"])
    agree = sum(1 for r in results["judge"] if r["self_report"] == r["objective"])
    print(f"\njudge vs self-report agreement: {agree}/{n}")
    print(f"outputs: {outdir}/")


if __name__ == "__main__":
    main()
