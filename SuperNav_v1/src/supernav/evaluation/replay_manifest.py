#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from supernav.runtime.config import write_json
from supernav.runtime.streams import (
    augment_canonical_events,
    iter_json_lines,
    session_id_from_events,
)
from supernav.evaluation.topdown_replay import build_evaluator_topdown

PANO_ORDER = ("front", "right", "back", "left")
STEP_FRAME_TOOLS_WITH_PANORAMA = {
    "hab_forward",
    "hab_backward",
    "hab_turn",
    "hab_init_scene",
    "hab_look_vertical",
    "hab_navigate",
    "hab_oracle_local_navigate",
    "hab_visual_overlay_navigate",
    "hab_visual_point_navigate",
    "hab_visual_ground_preview",
    "hab_set_pose",
    "hab_turn_then_navigate_wam",
}


def default_visuals_root() -> Path:
    for name in ("NAV_VISUAL_OUTPUT_DIR", "HAB_VISUAL_OUTPUT_DIR", "NAV_ARTIFACTS_DIR"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return Path("/tmp/habitat_gs_visuals")


def grouped_panorama_frames(session_dir: Path) -> List[Tuple[float, List[str]]]:
    """Panorama groups in step order, each paired with its earliest frame mtime.

    The step counter is not chronological across writers (bridge-side localnav
    captures vs simulator step tools), so consumers that need the views from a
    specific moment must select by timestamp, not by queue position."""
    groups: Dict[str, Dict[str, str]] = {}
    pattern = re.compile(r"pano_(front|left|back|right)_step(\d+)_color_sensor\.png$")
    for frame_path in glob.glob(str(session_dir / "pano_*_step*_color_sensor.png")):
        match = pattern.match(Path(frame_path).name)
        if not match:
            continue
        direction, step = match.groups()
        groups.setdefault(step, {})[direction] = frame_path
    ordered: List[Tuple[float, List[str]]] = []
    for step in sorted(groups, key=int):
        group = groups[step]
        frames = [
            group[direction] for direction in PANO_ORDER if direction in group
        ]
        if not frames:
            continue
        try:
            base_ts = min(os.path.getmtime(frame) for frame in frames)
        except OSError:
            base_ts = float("inf")
        ordered.append((base_ts, frames))
    return ordered


def ordered_panorama_frames(session_dir: Path) -> List[str]:
    return [
        frame
        for _base_ts, group in grouped_panorama_frames(session_dir)
        for frame in group
    ]


def _first_mtime(paths: List[str]) -> float:
    mtimes: List[float] = []
    for path in paths:
        try:
            mtimes.append(os.path.getmtime(path))
        except OSError:
            continue
    return min(mtimes) if mtimes else float("inf")


def _pano_group_index_before(
    pano_groups: List[Tuple[float, List[str]]], ts: float
) -> int:
    """Index of the newest pano group captured before ts (0 when none qualifies).

    A localnav hop's decision-basis views are the latest panorama captured
    before the hop's first movement frame. FIFO popping drifts out of sync
    whenever another writer (e.g. hab_turn) interleaves a pano group between
    hops, pairing the inset with views from the wrong decision point."""
    best: Optional[int] = None
    for idx, (group_ts, _frames) in enumerate(pano_groups):
        if group_ts < ts and (best is None or group_ts > pano_groups[best][0]):
            best = idx
    return best if best is not None else 0


_LOCALNAV_FRAME_PATTERN = re.compile(r"localnav_\d+_leg0*(\d+)_step0*(\d+)\.png$")
_LOCALNAV_GOAL_LEG_PATTERN = re.compile(r"localnav_goal_leg0*(\d+)\.png$")


def _localnav_leg_groups(frames_dir: Path) -> List[List[str]]:
    """Group localnav rollout frames by leg, ordered for per-call consumption.

    Frames live under the bridge-side NAV_ARTIFACTS_DIR (which may differ from
    the manifest's session_dir), so they are located through the frames_dir
    recorded in each tool result rather than a fixed visuals root. Leg numbers
    are assigned by the bridge loop in call order, so popping groups in order
    lines up with hab_local_navigate calls one-to-one (calls that error before
    the loop starts produce no leg and no frames_dir)."""
    by_leg: Dict[int, Dict[int, str]] = {}
    for frame_path in glob.glob(str(frames_dir / "localnav_*_leg*_step*.png")):
        match = _LOCALNAV_FRAME_PATTERN.match(Path(frame_path).name)
        if not match:
            continue
        leg, step = int(match.group(1)), int(match.group(2))
        by_leg.setdefault(leg, {})[step] = frame_path
    return [
        [by_leg[leg][step] for step in sorted(by_leg[leg])]
        for leg in sorted(by_leg)
    ]


def _localnav_leg_map(frames_dirs: List[Path]) -> Dict[int, List[str]]:
    """Leg number -> step-ordered rollout frames, merged across directories."""
    merged: Dict[int, Dict[int, str]] = {}
    for frames_dir in frames_dirs:
        for frame_path in glob.glob(str(frames_dir / "localnav_*_leg*_step*.png")):
            match = _LOCALNAV_FRAME_PATTERN.match(Path(frame_path).name)
            if not match:
                continue
            leg, step = int(match.group(1)), int(match.group(2))
            merged.setdefault(leg, {})[step] = frame_path
    return {
        leg: [merged[leg][step] for step in sorted(merged[leg])]
        for leg in sorted(merged)
    }


def _localnav_pairing_plan(
    events: List[Dict[str, Any]],
    session_dir: Path,
    recorded_visuals_root: Optional[Path],
) -> Dict[int, List[str]]:
    """Pair hab_local_navigate calls with rollout legs when results omit frames_dir.

    Bridges that do not record frames_dir leave leg artifacts on disk, possibly
    under a different visuals root than the panorama session directory (the
    localnav policy writer has its own artifact root). Legs pair with calls by
    the leg number embedded in the goal overlay filename (authoritative), or by
    call ordinal for calls that errored before writing an overlay — never by
    FIFO position, which silently shifts every later pairing once a call
    errors out mid-sequence."""
    calls: List[Tuple[int, Optional[str]]] = []
    for idx, event in enumerate(events):
        if event.get("type") != "tool_call" or event.get("name") != "hab_local_navigate":
            continue
        result = _result_payload(events, idx, "hab_local_navigate")
        frames_dir = result.get("frames_dir")
        if isinstance(frames_dir, str) and frames_dir:
            continue
        overlay = result.get("goal_overlay_image")
        calls.append((idx, overlay if isinstance(overlay, str) and overlay else None))
    if not calls:
        return {}
    frames_dirs: List[Path] = []
    for candidate in [session_dir, *(Path(overlay).parent for _idx, overlay in calls if overlay)]:
        if candidate not in frames_dirs:
            frames_dirs.append(candidate)
    if recorded_visuals_root is not None:
        recorded_dir = recorded_visuals_root / session_dir.name
        if recorded_dir not in frames_dirs:
            frames_dirs.append(recorded_dir)
    leg_map = _localnav_leg_map(frames_dirs)
    if not leg_map:
        return {}
    plan: Dict[int, List[str]] = {}
    claimed: set = set()
    ordinal_fallback: List[Tuple[int, int]] = []
    for ordinal, (idx, overlay) in enumerate(calls, start=1):
        leg: Optional[int] = None
        if overlay:
            match = _LOCALNAV_GOAL_LEG_PATTERN.search(Path(overlay).name)
            if match:
                leg = int(match.group(1))
        if leg is not None and leg in leg_map and leg not in claimed:
            plan[idx] = leg_map[leg]
            claimed.add(leg)
        else:
            ordinal_fallback.append((idx, ordinal))
    for idx, ordinal in ordinal_fallback:
        if ordinal in leg_map and ordinal not in claimed:
            plan[idx] = leg_map[ordinal]
            claimed.add(ordinal)
    return plan


def build_manifest(
    run_dir: Path, visuals_root: Path, *, write: bool = True
) -> Dict[str, Any]:
    canonical = run_dir / "canonical.jsonl"
    events = list(iter_json_lines(canonical))
    session_id = None
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            session_id = json.loads(metrics_path.read_text(encoding="utf-8")).get(
                "session_id"
            )
        except json.JSONDecodeError:
            session_id = None
    session_id = session_id or session_id_from_events(events)
    if not session_id:
        raise RuntimeError(f"Unable to infer session id from {run_dir}")
    events = augment_canonical_events(
        events,
        audit_path=visuals_root / f"{session_id}.benchmark_audit.jsonl",
    )
    session_dir = visuals_root / session_id
    step_frames = sorted(
        glob.glob(str(session_dir / "step*_color_sensor.png")), key=os.path.getmtime
    )
    wam_frames = sorted(
        glob.glob(str(session_dir / "wamnav_*.png")),
        key=lambda p: int((re.search(r"wamnav_0*(\d+)", Path(p).name) or [0, 0])[1]),
    )
    wam_by_leg: Dict[int, List[str]] = {}
    for frame in wam_frames:
        match = re.search(r"_leg0*(\d+)_step0*(\d+)", Path(frame).name)
        if match:
            wam_by_leg.setdefault(int(match.group(1)), []).append(frame)
    wam_legs = [wam_by_leg[k] for k in sorted(wam_by_leg)]
    pano_groups = grouped_panorama_frames(session_dir)
    frame_queue = list(step_frames)
    wam_queue = list(wam_frames)
    wam_leg_queue = list(wam_legs)
    pano_pool = list(pano_groups)
    localnav_frames_dir: Optional[str] = None
    localnav_leg_queue: List[List[str]] = []
    recorded_visuals_root: Optional[Path] = None
    stale_manifest_path = run_dir / "manifest.json"
    if stale_manifest_path.is_file():
        try:
            stale_root = json.loads(stale_manifest_path.read_text(encoding="utf-8")).get(
                "visuals_root"
            )
            if isinstance(stale_root, str) and stale_root.strip():
                recorded_visuals_root = Path(stale_root).expanduser()
        except json.JSONDecodeError:
            recorded_visuals_root = None
    localnav_plan = _localnav_pairing_plan(events, session_dir, recorded_visuals_root)
    manifest: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        localnav_overlay: Optional[str] = None
        if event.get("type") != "tool_call":
            continue
        name = str(event.get("name") or "unknown")
        frames: List[str] = []
        if name in ("hab_navigate_wam", "hab_turn_then_navigate_wam"):
            if wam_leg_queue:
                frames = wam_leg_queue.pop(0)
            else:
                frames = wam_queue
                wam_queue = []
            if pano_pool:
                frames = [*frames, *pano_pool.pop(0)[1]]
            if name in STEP_FRAME_TOOLS_WITH_PANORAMA and frame_queue:
                frame_queue.pop(0)
        elif name == "hab_local_navigate":
            result = _result_payload(events, idx, name)
            frames_dir = result.get("frames_dir")
            overlay = result.get("goal_overlay_image")
            # The marked-goal inset belongs to this call alone; a call that
            # errored before producing one must not inherit a stale overlay.
            localnav_overlay = overlay if isinstance(overlay, str) and overlay else None
            if isinstance(frames_dir, str) and frames_dir:
                if frames_dir != localnav_frames_dir:
                    localnav_frames_dir = frames_dir
                    localnav_leg_queue = _localnav_leg_groups(Path(frames_dir))
                if localnav_leg_queue:
                    frames = localnav_leg_queue.pop(0)
            else:
                frames = list(localnav_plan.get(idx, []))
            if pano_groups and frames:
                # Display-only pick from the full snapshot: the decision
                # views for this hop are the newest pano group captured
                # before its first movement frame. Interleaved step tools
                # (e.g. hab_turn) may legitimately show the same group.
                group_index = _pano_group_index_before(
                    pano_groups, _first_mtime(frames)
                )
                frames = [*frames, *pano_groups[group_index][1]]
        elif name == "hab_look_around":
            frames = [frame for _base_ts, group in pano_pool for frame in group]
            pano_pool.clear()
        elif name == "hab_panorama":
            if pano_pool:
                frames = pano_pool.pop(0)[1]
        elif name == "hab_oracle_local_map":
            frames = _result_image_paths(events, idx, name)
        elif name == "hab_visual_overlay_navigate":
            frames = _result_path_list(events, idx, name, "movement_frames")
            result_images = _result_image_paths(events, idx, name)
            frames.extend(path for path in result_images if path not in frames)
            if result_images and pano_pool:
                pano_pool.pop(0)
            elif pano_pool:
                frames = [*frames, *pano_pool.pop(0)[1]]
            if frame_queue:
                _drop_consumed_frames(frame_queue, frames)
        elif name == "hab_visual_ground_preview":
            result = _result_payload(events, idx, name)
            if result.get("status") == "preview_ready":
                overlay = _result_overlay_path(events, idx, name)
                if overlay:
                    frames = [overlay]
            else:
                frames = _result_path_list(events, idx, name, "movement_frames")
                result_images = _result_image_paths(events, idx, name)
                frames.extend(path for path in result_images if path not in frames)
                if result_images and pano_pool:
                    pano_pool.pop(0)
                elif pano_pool:
                    frames = [*frames, *pano_pool.pop(0)[1]]
                if frame_queue:
                    _drop_consumed_frames(frame_queue, frames)
        elif name in ("hab_oracle_local_navigate", "hab_visual_point_navigate"):
            frames = _result_path_list(events, idx, name, "movement_frames")
            result_images = _result_image_paths(events, idx, name)
            frames.extend(path for path in result_images if path not in frames)
            if result_images and pano_pool:
                pano_pool.pop(0)
            elif pano_pool:
                frames = [*frames, *pano_pool.pop(0)[1]]
            if frame_queue:
                _drop_consumed_frames(frame_queue, frames)
        elif name in STEP_FRAME_TOOLS_WITH_PANORAMA:
            frames = _result_image_paths(events, idx, name)
            if frames:
                if frame_queue:
                    _drop_consumed_frames(frame_queue, frames)
            elif pano_pool:
                frames = pano_pool.pop(0)[1]
                if frame_queue:
                    frame_queue.pop(0)
            elif frame_queue:
                frames = [frame_queue.pop(0)]
        elif name == "hab_see":
            if frame_queue:
                frames = [frame_queue.pop(0)]
        entry: Dict[str, Any] = {
            "tool": name,
            "input": event.get("input", {}),
            "frames": frames,
        }
        if name == "hab_visual_point_navigate":
            annotation = _visual_point_annotation(events, idx, event.get("input", {}))
            if annotation:
                entry["target_annotation"] = annotation
        elif name == "hab_visual_ground_preview":
            annotation = _visual_ground_preview_annotation(
                events, idx, event.get("input", {})
            )
            if annotation:
                entry["target_annotation"] = annotation
        elif name == "hab_local_navigate" and localnav_overlay:
            # Keep the marked-goal image as a persistent corner inset for the
            # whole hop instead of flashing it as one standalone frame.
            annotation = {"overlay_image": localnav_overlay}
            hop_input = event.get("input", {})
            if isinstance(hop_input, Mapping):
                for key in ("view", "point"):
                    if hop_input.get(key) is not None:
                        annotation[key] = hop_input[key]
            entry["target_annotation"] = annotation
        manifest.append(entry)
    payload = {
        "session_id": session_id,
        "pano_alignment": "decision_timestamp_v1",
        "visuals_root": str(visuals_root),
        "session_dir": str(session_dir),
        "entries": manifest,
        "evaluator_topdown": build_evaluator_topdown(
            run_dir, visuals_root, events, session_id
        ),
    }
    if write:
        write_json(run_dir / "manifest.json", payload)
    return payload


def _result_image_paths(
    events: List[Dict[str, Any]], call_index: int, tool_name: str
) -> List[str]:
    paths = _result_path_list(events, call_index, tool_name, "panorama_image_paths")
    if paths:
        return paths
    paths = _result_path_list(events, call_index, tool_name, "images")
    if paths:
        return paths
    map_path = _result_map_image_path(events, call_index, tool_name)
    return [map_path] if map_path else []


def _result_map_image_path(
    events: List[Dict[str, Any]], call_index: int, tool_name: str
) -> str:
    payload = _result_payload(events, call_index, tool_name)
    map_image = payload.get("map_image")
    if isinstance(map_image, dict) and isinstance(map_image.get("path"), str):
        return str(map_image["path"])
    return ""


def _result_payload(
    events: List[Dict[str, Any]],
    call_index: int,
    tool_name: str,
) -> Dict[str, Any]:
    for event in events[call_index + 1 :]:
        if event.get("type") == "tool_call":
            return {}
        if event.get("type") != "tool_result" or event.get("name") != tool_name:
            continue
        audit = event.get("audit")
        if isinstance(audit, dict):
            return dict(audit)
        content = str(event.get("content") or "").strip()
        if not content:
            return {}
        json_text = content.splitlines()[0]
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _result_overlay_path(
    events: List[Dict[str, Any]], call_index: int, tool_name: str
) -> str:
    result = _result_payload(events, call_index, tool_name)
    overlay = result.get("overlay_image")
    if isinstance(overlay, str) and overlay:
        return overlay
    return ""


def _visual_ground_preview_annotation(
    events: List[Dict[str, Any]],
    call_index: int,
    raw_input: Any,
) -> Dict[str, Any]:
    if not isinstance(raw_input, dict):
        raw_input = {}
    result = _result_payload(events, call_index, "hab_visual_ground_preview")
    annotation: Dict[str, Any] = {"kind": "visual_ground_preview"}
    for key in ("image_ref", "phrase", "mode", "candidate_id", "branch_id"):
        value = raw_input.get(key)
        if value is not None:
            annotation[key] = value
    for key in (
        "overlay_image",
        "candidate_count",
        "candidates",
        "detections",
        "status",
        "auto_confirmed",
        "selected_candidate_id",
        "selected_candidate",
        "planned_path_m",
        "displacement_m",
        "reason",
        "message",
    ):
        value = result.get(key)
        if value is not None:
            annotation[key] = value
    return annotation if len(annotation) > 1 else {}


def _visual_point_annotation(
    events: List[Dict[str, Any]],
    call_index: int,
    raw_input: Any,
) -> Dict[str, Any]:
    if not isinstance(raw_input, dict):
        raw_input = {}
    result = _result_payload(events, call_index, "hab_visual_point_navigate")
    annotation: Dict[str, Any] = {"kind": "visual_point_target"}
    for key in ("image_ref", "point", "bbox", "anchor", "intent"):
        value = raw_input.get(key)
        if value is not None:
            annotation[key] = value
    for key in ("direction", "selected_anchor_px", "overlay_image", "status"):
        value = result.get(key)
        if value is not None:
            annotation[key] = value
    for key in (
        "reason",
        "message",
        "planned_path_m",
        "displacement_m",
        "original_annotation",
        "refined_annotation",
        "refine_diagnostics",
    ):
        value = result.get(key)
        if value is not None:
            annotation[key] = value
    refined = annotation.get("refined_annotation")
    if isinstance(refined, dict):
        overlay = refined.get("overlay_image")
        if isinstance(overlay, str) and overlay and "overlay_image" not in annotation:
            annotation["overlay_image"] = overlay
        selected = refined.get("selected_anchor_px") or refined.get(
            "selected_sample_px"
        )
        if isinstance(selected, list) and "selected_anchor_px" not in annotation:
            annotation["selected_anchor_px"] = selected
    return annotation if len(annotation) > 1 else {}


def _result_path_list(
    events: List[Dict[str, Any]],
    call_index: int,
    tool_name: str,
    key: str,
) -> List[str]:
    payload = _result_payload(events, call_index, tool_name)
    paths = payload.get(key)
    if isinstance(paths, list):
        return [str(path) for path in paths if isinstance(path, str) and path]
    return []


def _drop_consumed_frames(frame_queue: List[str], frames: List[str]) -> None:
    consumed = set(frames)
    while frame_queue and frame_queue[0] in consumed:
        frame_queue.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build replay manifest for one harness-dev run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--visuals-root", default=None)
    args = parser.parse_args()
    root = (
        Path(args.visuals_root).expanduser().resolve()
        if args.visuals_root
        else default_visuals_root()
    )
    print(
        json.dumps(
            build_manifest(Path(args.run_dir).expanduser().resolve(), root),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
