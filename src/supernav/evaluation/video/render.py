#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from supernav.evaluation.replay_manifest import build_manifest, default_visuals_root
from supernav.evaluation.topdown_replay import build_evaluator_topdown, render_topdown_panel

TraceEntry = Tuple[float, str, str, Optional[float]]
StageEntry = Tuple[float, float, str]
ReplayFrame = Tuple[float, str, str, Optional[Dict[str, Any]]]
PANO_SEGMENTS = (
    ("front", None),
    ("left", None),
    ("back", None),
    ("right", None),
)
PANO_FRAME_PREFIX = "pano_concat::"
TRACE_CHUNK_CHARS = 220
SYNTHETIC_TOOL_STEP_S = 4.0
PANO_PATH_PATTERN = re.compile(
    r"pano_(front|left|back|right)_step\d+_(?:color_sensor(?:_agent)?|visible_overlay)\.png$"
)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit_image(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize image to fit inside box while preserving aspect ratio."""
    if im.width == 0 or im.height == 0:
        return Image.new("RGB", (max(1, box_w), max(1, box_h)), (0, 0, 0))
    ratio = min(box_w / im.width, box_h / im.height)
    new_w = max(1, int(round(im.width * ratio)))
    new_h = max(1, int(round(im.height * ratio)))
    return im.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _center_paste(canvas: Image.Image, im: Image.Image) -> None:
    """Paste image into the center of canvas."""
    x = (canvas.width - im.width) // 2
    y = (canvas.height - im.height) // 2
    canvas.paste(im, (x, y))


def load_image(path: str, size: int) -> Image.Image:
    """Load an image and fit it inside a size x size box, preserving aspect ratio."""
    if path.startswith(PANO_FRAME_PREFIX):
        try:
            parts = json.loads(path[len(PANO_FRAME_PREFIX) :])
        except json.JSONDecodeError:
            parts = []
        if not parts:
            return Image.new("RGB", (size, size), (0, 0, 0))
        # Fit each tile into a size x size box and concatenate horizontally.
        tiles: List[Image.Image] = []
        for part in parts:
            pano_path = str(part.get("path") or "")
            tile = load_image(pano_path, size)
            crop = part.get("crop")
            if crop == "left":
                tile = _fit_image(
                    tile.crop((0, 0, max(1, tile.width // 2), tile.height)), size, size
                )
            elif crop == "right":
                tile = _fit_image(
                    tile.crop((tile.width // 2, 0, tile.width, tile.height)), size, size
                )
            tiles.append(tile)
        strip_w = sum(t.width for t in tiles)
        strip_h = max((t.height for t in tiles), default=size)
        strip = Image.new("RGB", (strip_w, strip_h), (0, 0, 0))
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, (strip_h - tile.height) // 2))
            x += tile.width
        # Fit the whole strip into the output size box.
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        resized = _fit_image(strip, size, size)
        _center_paste(canvas, resized)
        return canvas
    try:
        im = Image.open(path).convert("RGB")
    except OSError:
        return Image.new("RGB", (size, size), (0, 0, 0))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    fitted = _fit_image(im, size, size)
    _center_paste(canvas, fitted)
    return canvas


def load_visual_panel(
    path: str,
    *,
    size: int,
    tool: str,
    annotation: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Render the replay visual area without privileged map/topdown panels."""
    visual_w = size * 2
    thumb_h = max(1, size // 3)
    visual_h = size + thumb_h
    if path.startswith(PANO_FRAME_PREFIX):
        return load_panorama_panel(path, size=size, tool=tool, annotation=annotation)
    panel = Image.new("RGB", (visual_w, visual_h), (10, 12, 18))
    if path:
        image = _fit_image(load_image(path, size), visual_w, size)
    else:
        image = _fit_image(_placeholder_panel(size, "Agent visual"), visual_w, size)
    front_canvas = Image.new("RGB", (visual_w, size), (0, 0, 0))
    _center_paste(front_canvas, image)
    panel.paste(
        _label_panel(front_canvas, f"front | {tool}", visual_w, height=size), (0, 0)
    )
    draw = ImageDraw.Draw(panel)
    f_small = font(13)
    for idx, label in enumerate(("left", "back", "right")):
        x0 = int(round(idx * visual_w / 3))
        x1 = int(round((idx + 1) * visual_w / 3))
        draw.rectangle(
            (x0, size, x1 - 1, visual_h - 1), fill=(16, 18, 24), outline=(42, 52, 66)
        )
        draw.text((x0 + 8, size + 8), label, font=f_small, fill=(150, 165, 180))
    _paste_visual_point_inset(panel, annotation, size=size)
    return panel


def load_panorama_panel(
    path: str,
    *,
    size: int,
    tool: str,
    annotation: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    visual_w = size * 2
    thumb_h = max(1, size // 3)
    visual_h = size + thumb_h
    try:
        parts = json.loads(path[len(PANO_FRAME_PREFIX) :])
    except json.JSONDecodeError:
        parts = []
    by_direction = {
        str(part.get("direction")): str(part.get("path") or "")
        for part in parts
        if isinstance(part, dict) and part.get("direction")
    }
    panel = Image.new("RGB", (visual_w, visual_h), (10, 12, 18))

    front_path = by_direction.get("front", "")
    if front_path:
        front = _fit_image(load_image(front_path, size), visual_w, size)
    else:
        front = _fit_image(_placeholder_panel(size, "front"), visual_w, size)
    front_canvas = Image.new("RGB", (visual_w, size), (0, 0, 0))
    _center_paste(front_canvas, front)
    panel.paste(
        _label_panel(front_canvas, f"front | {tool}", visual_w, height=size), (0, 0)
    )

    thumb_w = visual_w // 3
    for idx, direction in enumerate(("left", "back", "right")):
        x0 = idx * thumb_w
        width = visual_w - x0 if idx == 2 else thumb_w
        image_path = by_direction.get(direction, "")
        if image_path:
            image = _fit_image(load_image(image_path, size), width, thumb_h)
        else:
            image = _fit_image(_placeholder_panel(size, direction), width, thumb_h)
        thumb_canvas = Image.new("RGB", (width, thumb_h), (0, 0, 0))
        _center_paste(thumb_canvas, image)
        panel.paste(
            _label_panel(thumb_canvas, direction, width, height=thumb_h), (x0, size)
        )
    _paste_visual_point_inset(panel, annotation, size=size)
    return panel


def _paste_visual_point_inset(
    panel: Image.Image,
    annotation: Optional[Dict[str, Any]],
    *,
    size: int,
) -> None:
    inset = _visual_point_inset(annotation, size=max(96, size // 2))
    if inset is None:
        return
    margin = 10
    x = panel.width - inset.width - margin
    y = 34
    shadow = Image.new("RGB", (inset.width + 8, inset.height + 8), (0, 0, 0))
    panel.paste(shadow, (x - 4, y - 4))
    panel.paste(inset, (x, y))


def _visual_point_inset(
    annotation: Optional[Dict[str, Any]],
    *,
    size: int,
) -> Optional[Image.Image]:
    if not isinstance(annotation, dict):
        return None
    refined = annotation.get("refined_annotation")
    overlay = annotation.get("overlay_image")
    if not isinstance(overlay, str) and isinstance(refined, dict):
        overlay = refined.get("overlay_image")
    if not isinstance(overlay, str) or not overlay:
        return None
    try:
        image = Image.open(overlay).convert("RGB")
    except OSError:
        return None
    image = image.copy()
    draw = ImageDraw.Draw(image)
    _draw_visual_point_annotation(draw, image.size, annotation)
    fitted = _fit_image(image, size, size)
    canvas = Image.new("RGB", (size, size), (4, 6, 10))
    _center_paste(canvas, fitted)
    draw_canvas = ImageDraw.Draw(canvas)
    f_small = font(12)
    reason = annotation.get("reason")
    label = (
        f"self-check: {reason}"
        if isinstance(reason, str) and reason
        else "visual target"
    )
    label_w = int(draw_canvas.textlength(label, font=f_small)) + 14
    draw_canvas.rectangle((0, 0, min(size, label_w), 22), fill=(0, 0, 0))
    draw_canvas.text((7, 5), label, font=f_small, fill=(255, 236, 120))
    draw_canvas.rectangle(
        (0, 0, size - 1, size - 1),
        outline=(255, 210, 80),
        width=2,
    )
    return canvas


def _draw_visual_point_annotation(
    draw: ImageDraw.ImageDraw,
    image_size: Tuple[int, int],
    annotation: Dict[str, Any],
) -> None:
    width, height = image_size
    original = annotation.get("original_annotation")
    refined = annotation.get("refined_annotation")
    original = original if isinstance(original, dict) else annotation
    refined = refined if isinstance(refined, dict) else annotation

    bbox = _annotation_bbox(original.get("bbox"), width=width, height=height)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 210, 40), width=max(2, width // 128))
    point = _annotation_point(original, width=width, height=height)
    if point is None and bbox is not None:
        x0, y0, x1, y1 = bbox
        point = ((x0 + x1) / 2.0, y0 + 0.88 * (y1 - y0))
    radius = max(5, min(width, height) // 36)
    line = max(2, min(width, height) // 128)
    if point is not None:
        x, y = point
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255),
            width=line,
        )
        draw.line(
            (x - radius * 1.7, y, x + radius * 1.7, y),
            fill=(255, 64, 64),
            width=line,
        )
        draw.line(
            (x, y - radius * 1.7, x, y + radius * 1.7),
            fill=(255, 64, 64),
            width=line,
        )

    refined_point = _annotation_refined_point(refined, width=width, height=height)
    if refined_point is None:
        return
    rx, ry = refined_point
    refined_radius = max(4, radius - 2)
    draw.ellipse(
        (
            rx - refined_radius,
            ry - refined_radius,
            rx + refined_radius,
            ry + refined_radius,
        ),
        outline=(60, 255, 210),
        width=line,
    )
    draw.line(
        (rx - refined_radius * 1.6, ry, rx + refined_radius * 1.6, ry),
        fill=(60, 255, 210),
        width=line,
    )
    draw.line(
        (rx, ry - refined_radius * 1.6, rx, ry + refined_radius * 1.6),
        fill=(60, 255, 210),
        width=line,
    )


def _annotation_refined_point(
    annotation: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    for key in ("selected_sample_px", "selected_anchor_px", "anchor_px"):
        selected = annotation.get(key)
        if isinstance(selected, list) and len(selected) >= 2:
            try:
                return (
                    max(0.0, min(float(width - 1), float(selected[0]))),
                    max(0.0, min(float(height - 1), float(selected[1]))),
                )
            except (TypeError, ValueError):
                continue
    return None


def _annotation_point(
    annotation: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    selected = annotation.get("selected_anchor_px")
    if isinstance(selected, list) and len(selected) >= 2:
        try:
            return (
                max(0.0, min(float(width - 1), float(selected[0]))),
                max(0.0, min(float(height - 1), float(selected[1]))),
            )
        except (TypeError, ValueError):
            return None
    raw = annotation.get("point")
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            return (
                max(0.0, min(1.0, float(raw[0]))) * (width - 1),
                max(0.0, min(1.0, float(raw[1]))) * (height - 1),
            )
        except (TypeError, ValueError):
            return None
    return None


def _annotation_bbox(
    value: Any,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(value, dict):
        raw = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, list) and len(value) >= 4:
        raw = (value[0], value[1], value[2], value[3])
    else:
        return None
    try:
        x = max(0.0, min(1.0, float(raw[0])))
        y = max(0.0, min(1.0, float(raw[1])))
        w = max(0.0, min(1.0, float(raw[2])))
        h = max(0.0, min(1.0, float(raw[3])))
    except (TypeError, ValueError):
        return None
    x0 = x * (width - 1)
    y0 = y * (height - 1)
    x1 = min(width - 1.0, (x + w) * (width - 1))
    y1 = min(height - 1.0, (y + h) * (height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def timestamp_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds / 1000.0 if seconds > 1e11 else seconds
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            seconds = float(text)
            return seconds / 1000.0 if seconds > 1e11 else seconds
        except ValueError:
            pass
        try:
            from datetime import datetime

            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def event_time(event: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        ts = timestamp_seconds(event.get(key))
        if ts is not None:
            return ts
    return None


def canonical_events(run_dir: Path) -> List[Dict[str, Any]]:
    canonical = run_dir / "canonical.jsonl"
    if not canonical.is_file():
        return []
    events: List[Dict[str, Any]] = []
    for line in canonical.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def compact_text(value: Any) -> str:
    """Normalize event text without dropping content."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def chunk_text(text: str, max_chars: int = TRACE_CHUNK_CHARS) -> List[str]:
    """Split long model/reasoning text into display chunks at word boundaries."""
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = word
        else:
            chunks.append(word)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def append_text_trace(
    lines: List[TraceEntry],
    *,
    start_ts: Optional[float],
    end_ts: float,
    kind: str,
    label: str,
    text: str,
) -> None:
    chunks = chunk_text(text)
    if not chunks:
        return
    start = start_ts if start_ts is not None and start_ts <= end_ts else end_ts
    span = max(0.0, end_ts - start)
    duration_s = max(0.0, end_ts - start_ts) if start_ts is not None else None
    for idx, chunk in enumerate(chunks):
        if len(chunks) == 1 or span <= 0.0:
            ts = end_ts
        else:
            ts = start + span * (idx + 1) / len(chunks)
        prefix = f"{label}: " if idx == 0 else "  ... "
        lines.append((ts, kind, prefix + chunk, duration_s if idx == 0 else None))


def _duration_s(start_ts: Optional[float], end_ts: Optional[float]) -> Optional[float]:
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, end_ts - start_ts)


def _task_result_body(content: str) -> str:
    match = re.search(
        r"<task_result>\s*(.*?)\s*</task_result>", content, flags=re.DOTALL
    )
    if not match:
        return compact_text(content)
    return match.group(1).strip()


def _task_label(event: Dict[str, Any]) -> str:
    data = event.get("input") if isinstance(event.get("input"), dict) else {}
    subagent_type = str(data.get("subagent_type") or "subagent").strip()
    description = str(data.get("description") or "").strip()
    if description:
        return f"subagent {subagent_type}: {description}"
    return f"subagent {subagent_type}"


def _collab_subagent_label(event: Dict[str, Any]) -> str:
    data = event.get("input") if isinstance(event.get("input"), dict) else {}
    message = str(data.get("message") or data.get("prompt") or "").strip()
    agent_type = str(data.get("agent_type") or "subagent").strip() or "subagent"
    if message:
        return f"subagent {agent_type}: {message}"
    return f"subagent {agent_type}"


def _wait_status_summary(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return compact_text(content)
    if not isinstance(payload, dict):
        return compact_text(content)
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    timed_out = bool(payload.get("timed_out"))
    if timed_out and not status:
        return "subagent wait timed out"
    for agent_state in status.values():
        if not isinstance(agent_state, dict):
            continue
        completed = agent_state.get("completed")
        if isinstance(completed, str) and completed.strip():
            return completed.strip()
    if timed_out:
        return "subagent wait timed out"
    return compact_text(content)


def _tool_result_summary(event: Dict[str, Any]) -> str:
    name = str(event.get("name") or "tool")
    if not name.startswith("hab_"):
        return ""
    content = str(event.get("content") or "").strip()
    if not content:
        return f"< {name}: done"
    try:
        payload = json.loads(content.splitlines()[0])
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    if name == "hab_close_session":
        if payload.get("closed") is True:
            return f"< {name}: closed"
        return f"< {name}: done"
    parts: List[str] = []
    status = payload.get("status") or payload.get("nav_status")
    if isinstance(status, str) and status:
        parts.append(status)
    elif payload.get("ok") is False:
        parts.append("failed")
    elif payload.get("ok") is True:
        parts.append("ok")
    for key, label in (
        ("steps_executed", "steps"),
        ("steps_taken", "steps"),
        ("movement_frame_count", "frames"),
        ("candidate_count", "candidates"),
        ("selected_candidate_id", "candidate"),
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}={int(value)}")
    reason = payload.get("reason") or payload.get("message") or payload.get("error")
    if isinstance(reason, str) and reason and len(parts) < 4:
        parts.append(compact_text(reason)[:90])
    return f"< {name}: {' '.join(parts) if parts else 'done'}"


def trace_events(events: List[Dict[str, Any]]) -> List[TraceEntry]:
    lines: List[TraceEntry] = []
    for event in events:
        if event.get("type") == "tool_call":
            ts = event_time(event, "time_start", "timestamp")
            if ts is None:
                continue
            start_ts = event_time(event, "time_start")
            end_ts = event_time(event, "time_end", "timestamp")
            if event.get("name") == "task":
                lines.append(
                    (
                        ts,
                        "subagent",
                        f"> {_task_label(event)}",
                        _duration_s(start_ts, end_ts),
                    )
                )
            elif event.get("name") in {"spawn_agent", "wait_agent"}:
                lines.append(
                    (
                        ts,
                        "subagent",
                        f"> {_collab_subagent_label(event)}",
                        _duration_s(start_ts, end_ts),
                    )
                )
            else:
                lines.append(
                    (
                        ts,
                        "tool",
                        f"> {event.get('name')} {event.get('input') or {}}",
                        _duration_s(start_ts, end_ts),
                    )
                )
        elif event.get("type") == "tool_result" and event.get("name") in {
            "task",
            "wait_agent",
        }:
            end_ts = event_time(event, "time_end", "timestamp")
            if end_ts is None:
                continue
            content = str(event.get("content") or "")
            if event.get("name") == "task":
                text = compact_text(_task_result_body(content))
            else:
                text = compact_text(_wait_status_summary(content))
            append_text_trace(
                lines,
                start_ts=event_time(event, "time_start"),
                end_ts=end_ts,
                kind="subagent",
                label="subagent",
                text=text,
            )
        elif event.get("type") == "tool_result":
            end_ts = event_time(event, "time_end", "timestamp")
            if end_ts is None:
                continue
            summary = _tool_result_summary(event)
            if summary:
                lines.append(
                    (
                        end_ts,
                        "tool",
                        summary,
                        _duration_s(event_time(event, "time_start"), end_ts),
                    )
                )
        elif event.get("type") == "spatial_memory":
            ts = event_time(event, "time_end", "timestamp")
            if ts is None:
                continue
            action = str(event.get("action") or "transition")
            branch = event.get("branch_id")
            branch_ids = event.get("branch_ids")
            detail = f" {branch}" if branch else ""
            if isinstance(branch_ids, list) and branch_ids:
                detail = " " + ",".join(str(item) for item in branch_ids)
            lines.append((ts, "memory", f"# memory: {action}{detail}", None))
        elif event.get("type") == "assistant_text":
            end_ts = event_time(event, "time_end", "timestamp")
            if end_ts is None:
                continue
            append_text_trace(
                lines,
                start_ts=event_time(event, "time_start"),
                end_ts=end_ts,
                kind="text",
                label="model",
                text=compact_text(event.get("text", "")),
            )
        elif event.get("type") in ("assistant_thinking", "assistant_reasoning"):
            end_ts = event_time(event, "time_end", "timestamp")
            if end_ts is None:
                continue
            append_text_trace(
                lines,
                start_ts=event_time(event, "time_start"),
                end_ts=end_ts,
                kind="reasoning",
                label="reasoning",
                text=compact_text(event.get("text", "")),
            )
    return sorted(lines, key=lambda item: item[0])


def synthetic_trace_events(
    events: List[Dict[str, Any]],
    *,
    start_ts: float,
    end_ts: float,
    tool_step_s: float = SYNTHETIC_TOOL_STEP_S,
) -> List[TraceEntry]:
    lines: List[TraceEntry] = []
    tool_index = 0
    last_tool_time = start_ts
    for event in events:
        if event_time(event, "time_start", "time_end", "timestamp") is not None:
            continue
        typ = event.get("type")
        if typ == "tool_call":
            ts = start_ts + tool_index * tool_step_s
            if event.get("name") == "task":
                lines.append((ts, "subagent", f"> {_task_label(event)}", None))
            elif event.get("name") in {"spawn_agent", "wait_agent"}:
                lines.append(
                    (ts, "subagent", f"> {_collab_subagent_label(event)}", None)
                )
            else:
                lines.append(
                    (
                        ts,
                        "tool",
                        f"> {event.get('name')} {event.get('input') or {}}",
                        None,
                    )
                )
            last_tool_time = ts
            tool_index += 1
        elif typ == "tool_result" and event.get("name") in {"task", "wait_agent"}:
            content = str(event.get("content") or "")
            if event.get("name") == "task":
                text = compact_text(_task_result_body(content))
            else:
                text = compact_text(_wait_status_summary(content))
            for idx, chunk in enumerate(chunk_text(text)):
                lines.append(
                    (
                        last_tool_time + 0.8 + idx * 0.8,
                        "subagent",
                        f"subagent: {chunk}",
                        None,
                    )
                )
        elif typ == "tool_result":
            summary = _tool_result_summary(event)
            if summary:
                lines.append((last_tool_time + 0.8, "tool", summary, None))
        elif typ == "spatial_memory":
            action = str(event.get("action") or "transition")
            branch = event.get("branch_id")
            branch_ids = event.get("branch_ids")
            detail = f" {branch}" if branch else ""
            if isinstance(branch_ids, list) and branch_ids:
                detail = " " + ",".join(str(item) for item in branch_ids)
            lines.append(
                (
                    last_tool_time + 0.9,
                    "memory",
                    f"# memory: {action}{detail}",
                    None,
                )
            )
        elif typ == "assistant_text":
            base = start_ts + tool_index * tool_step_s
            for idx, chunk in enumerate(
                chunk_text(compact_text(event.get("text", "")))
            ):
                lines.append((base + idx * 0.8, "text", f"model: {chunk}", None))
        elif typ in ("assistant_thinking", "assistant_reasoning"):
            base = start_ts + tool_index * tool_step_s
            for idx, chunk in enumerate(
                chunk_text(compact_text(event.get("text", "")))
            ):
                lines.append(
                    (base + idx * 0.8, "reasoning", f"reasoning: {chunk}", None)
                )
    return sorted(lines, key=lambda item: item[0])


def stage_intervals(events: List[Dict[str, Any]]) -> List[StageEntry]:
    stages: List[StageEntry] = []
    for event in events:
        start = event_time(event, "time_start")
        end = event_time(event, "time_end")
        if start is None or end is None or end <= start:
            continue
        typ = event.get("type")
        if typ == "tool_call":
            stages.append((start, end, "tool"))
        elif typ == "assistant_text":
            stages.append((start, end, "text"))
        elif typ in ("assistant_thinking", "assistant_reasoning"):
            stages.append((start, end, "reasoning"))
    return sorted(stages, key=lambda item: (item[0], item[1]))


def active_stage(stages: List[StageEntry], ts: float) -> Optional[str]:
    active = [stage for stage in stages if stage[0] <= ts < stage[1]]
    if not active:
        return None
    return max(active, key=lambda item: item[0])[2]


def result_time(
    events: List[Dict[str, Any]], name: str, *, first: bool
) -> Optional[float]:
    matches = [
        event_time(event, "time_end", "timestamp")
        for event in events
        if event.get("type") == "tool_result" and event.get("name") == name
    ]
    matches = [ts for ts in matches if ts is not None]
    if not matches:
        return None
    return matches[0] if first else matches[-1]


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, width: int
) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font_obj) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word)
            current = ""
    if current:
        lines.append(current)
    return lines


def ordered_panorama_frames(session_dir: Path) -> List[Tuple[float, str, str]]:
    groups: Dict[str, Dict[str, Path]] = {}
    directions = "|".join(re.escape(direction) for direction, _ in PANO_SEGMENTS)
    pattern = re.compile(
        rf"pano_({directions})_step(\d+)_color_sensor(?:_agent)?\.png$"
    )
    for path in session_dir.glob("pano_*_step*_color_sensor.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        direction, step = match.groups()
        groups.setdefault(step, {})[direction] = path
    frames: List[Tuple[float, str, str]] = []
    for step in sorted(groups, key=int):
        group = groups[step]
        available = [
            group[direction]
            for direction in {direction for direction, _ in PANO_SEGMENTS}
            if direction in group
        ]
        if not available:
            continue
        base_ts = min(path.stat().st_mtime for path in available)
        strip_parts = []
        for direction, _crop in PANO_SEGMENTS:
            pano_path = group.get(direction)
            if pano_path is None:
                continue
            strip_parts.append({"direction": direction, "path": str(pano_path)})
        if strip_parts:
            frames.append(
                (base_ts, "panorama", PANO_FRAME_PREFIX + json.dumps(strip_parts))
            )
    return frames


def _panorama_direction(path: str) -> Optional[str]:
    match = PANO_PATH_PATTERN.match(Path(path).name)
    return str(match.group(1)) if match else None


def _panorama_concat_frame(paths: List[str]) -> Optional[str]:
    by_direction: Dict[str, str] = {}
    for path in paths:
        direction = _panorama_direction(path)
        if direction is not None:
            by_direction[direction] = path
    if not by_direction:
        return None
    strip_parts = []
    for direction, _crop in PANO_SEGMENTS:
        path = by_direction.get(direction)
        if path is not None:
            strip_parts.append({"direction": direction, "path": path})
    if not strip_parts:
        return None
    return PANO_FRAME_PREFIX + json.dumps(strip_parts)


def _panorama_concat_paths(frame_path: str) -> List[str]:
    if not frame_path.startswith(PANO_FRAME_PREFIX):
        return []
    try:
        parts = json.loads(frame_path[len(PANO_FRAME_PREFIX) :])
    except json.JSONDecodeError:
        return []
    if not isinstance(parts, list):
        return []
    paths: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        path = part.get("path")
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def _is_privileged_map_frame(tool: str, frame_path: str) -> bool:
    name = Path(frame_path).name.lower()
    return (
        "topdown" in name
        or "oracle_local_map" in name
        or tool in {"hab_topdown", "hab_oracle_local_map"}
    )


def _placeholder_panel(size: int, title: str) -> Image.Image:
    panel = Image.new("RGB", (size, size), (16, 18, 24))
    draw = ImageDraw.Draw(panel)
    f_title = font(18)
    f_small = font(13)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(55, 65, 78))
    draw.text((14, 14), title, font=f_title, fill=(150, 165, 180))
    draw.text((14, 42), "No frame yet", font=f_small, fill=(105, 115, 128))
    return panel


def _label_panel(
    image: Image.Image,
    label: str,
    width: int,
    *,
    height: Optional[int] = None,
) -> Image.Image:
    height = width if height is None else height
    panel = image.resize((width, height)).copy()
    draw = ImageDraw.Draw(panel)
    f_small = font(13)
    label_w = int(draw.textlength(label, font=f_small)) + 16
    draw.rectangle((0, 0, min(width, label_w), min(height, 26)), fill=(0, 0, 0))
    draw.text((8, 7), label, font=f_small, fill=(230, 235, 240))
    return panel


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textlength(text, font=font_obj) <= max_width:
        return text
    ellipsis = "..."
    trimmed = text
    while trimmed and draw.textlength(trimmed + ellipsis, font=font_obj) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed.rstrip() + ellipsis) if trimmed else ellipsis


def compose_frame(
    visual_panel: Image.Image,
    *,
    run_id: str,
    instruction: str = "",
    tool: str,
    trace: List[Tuple[str, str, Optional[float]]],
    status: Optional[str],
    size: int,
    actual_time_s: Optional[float] = None,
    topdown_panel: Optional[Image.Image] = None,
) -> Image.Image:
    header_h = 64 if instruction else 0
    visual_w = size * 2
    visual_h = visual_panel.height
    total_w = size * (4 if topdown_panel is not None else 3)
    trace_x = visual_w + (size if topdown_panel is not None else 0)
    out = Image.new("RGB", (total_w, visual_h + header_h), (12, 14, 20))
    draw = ImageDraw.Draw(out)
    f_small = font(13)
    f_instruction = font(15)
    if instruction:
        draw.rectangle((0, 0, total_w, header_h), fill=(5, 8, 13))
        label = "Instruction:"
        label_w = int(draw.textlength(label, font=f_instruction))
        draw.text((12, 10), label, font=f_instruction, fill=(120, 200, 255))
        text_x = 12 + label_w + 10
        text_w = total_w - text_x - 12
        lines = wrap_text(draw, instruction, f_instruction, text_w)
        if len(lines) > 2:
            lines = [
                lines[0],
                _fit_text_to_width(draw, lines[1], f_instruction, text_w),
            ]
        for idx, line in enumerate(lines[:2]):
            draw.text(
                (text_x, 10 + idx * 22),
                line,
                font=f_instruction,
                fill=(230, 235, 240),
            )
        draw.line((0, header_h - 1, total_w, header_h - 1), fill=(42, 52, 66), width=1)
    out.paste(visual_panel, (0, header_h))
    if topdown_panel is not None:
        out.paste(topdown_panel.resize((size, visual_h)), (visual_w, header_h))
    f_title = font(16)
    colors = {
        "tool": (130, 235, 160),
        "memory": (145, 210, 255),
        "text": (210, 220, 230),
        "reasoning": (240, 205, 130),
        "subagent": (255, 242, 170),
    }
    status_labels = {
        "tool": "TOOL CALL",
        "memory": "SPATIAL MEMORY",
        "text": "OUTPUT",
        "reasoning": "REASONING",
        "subagent": "SUBAGENT",
    }
    x0 = trace_x + 14
    max_width = size - 28
    draw.text((x0, header_h + 14), "Agent trace", font=f_title, fill=(120, 200, 255))
    if status:
        label = status_labels.get(status, status.upper())
        status_color = colors.get(status, (210, 220, 230))
        label_w = int(draw.textlength(label, font=f_small)) + 18
        badge_x = total_w - label_w - 14
        draw.rounded_rectangle(
            (badge_x, header_h + 12, badge_x + label_w, header_h + 34),
            radius=6,
            fill=(status_color[0] // 5, status_color[1] // 5, status_color[2] // 5),
            outline=status_color,
        )
        draw.text((badge_x + 9, header_h + 16), label, font=f_small, fill=status_color)
    y0 = header_h + 44
    line_height = 18
    max_lines = max(1, (header_h + visual_h - 18 - y0) // line_height)
    wrapped_lines: List[Tuple[str, str]] = []
    for kind, line, duration_s in trace:
        suffix = f" [{duration_s:.1f}s]" if duration_s is not None else ""
        full_line = line + suffix
        for wrapped in wrap_text(draw, full_line, f_small, max_width):
            wrapped_lines.append((kind, wrapped))
    scrolled = wrapped_lines[-max_lines:]
    if len(wrapped_lines) > max_lines:
        draw.text((x0, y0), "...", font=f_small, fill=(130, 140, 150))
        scrolled = scrolled[1:]
    y = y0
    for kind, line in scrolled:
        draw.text((x0, y), line, font=f_small, fill=colors.get(kind, (210, 220, 230)))
        y += line_height
    footer = f"{run_id} | {tool}"
    footer = _fit_text_to_width(draw, footer, f_small, visual_w - 16)
    footer_y = header_h + visual_h - 28
    draw.rectangle((0, footer_y, visual_w, header_h + visual_h), fill=(0, 0, 0))
    draw.text((8, footer_y + 6), footer, font=f_small, fill=(220, 230, 240))
    if actual_time_s is not None:
        label = f"actual time {actual_time_s:.1f}s"
        label_w = int(draw.textlength(label, font=f_small))
        x, y = visual_w - label_w - 16, header_h + 8
        draw.rounded_rectangle(
            (x - 8, y - 5, visual_w - 8, y + 19),
            radius=5,
            fill=(0, 0, 0),
            outline=(120, 200, 255),
        )
        draw.text((x, y), label, font=f_small, fill=(120, 200, 255))
    return out


def _evaluator_change_times(evaluator: Any) -> List[float]:
    if not isinstance(evaluator, dict):
        return []
    trajectory = evaluator.get("trajectory")
    if not isinstance(trajectory, dict):
        return []
    times: List[float] = []
    for collection in (trajectory.get("reveals", []), trajectory.get("endpoints", [])):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            value = timestamp_seconds(item.get("reveal_at"))
            if value is not None:
                times.append(value)
    return times


def _manifest_needs_refresh(manifest: Dict[str, Any], *, visuals_root: Path) -> bool:
    if manifest.get("pano_alignment") != "decision_timestamp_v1":
        return True
    if str(manifest.get("visuals_root") or "") != str(visuals_root):
        return True
    session_dir = manifest.get("session_dir")
    if not isinstance(session_dir, str) or not Path(session_dir).is_dir():
        return True
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if tool == "hab_local_navigate":
            # Rebuild manifests written before localnav frames were collected:
            # a hop that actually ran leaves localnav_* frames on disk, so an
            # entry without any is stale (or the run saved no frames at all —
            # the in-memory rebuild is cheap and idempotent in that case).
            frames = entry.get("frames")
            if not any("localnav_" in str(frame) for frame in frames or []):
                return True
            continue
        if tool != "hab_visual_point_navigate":
            continue
        annotation = entry.get("target_annotation")
        if not isinstance(annotation, dict):
            return True
        if (
            "original_annotation" not in annotation
            or "refined_annotation" not in annotation
        ):
            return True
    return False


def _synthetic_frame_time(
    entry_index: int, frame_index: int, frame_count: int
) -> float:
    start = entry_index * SYNTHETIC_TOOL_STEP_S
    if frame_count <= 1:
        return start + 1.2
    span = max(0.4, SYNTHETIC_TOOL_STEP_S - 1.0)
    return start + 0.7 + span * frame_index / max(1, frame_count - 1)


def _collect_replay_frames(
    manifest: Dict[str, Any], *, synthetic_time: bool = False
) -> List[ReplayFrame]:
    frames: List[ReplayFrame] = []
    seen_frame_paths = set()
    for entry_index, entry in enumerate(manifest.get("entries", [])):
        tool = str(entry.get("tool", "tool"))
        entry_frames = [str(frame) for frame in entry.get("frames", []) or []]
        annotation = (
            entry.get("target_annotation")
            if isinstance(entry.get("target_annotation"), dict)
            else None
        )
        if annotation and "kind" not in annotation:
            hop_input = entry.get("input")
            if isinstance(hop_input, dict) and hop_input.get("goal_id"):
                # goal_id retries navigate to the stored goal: the harness
                # ignores this call's point, and the paired overlay image
                # already carries the stored goal marker. Drawing the input
                # point on top would add a second, wrong target marker.
                annotation = {k: v for k, v in annotation.items() if k != "point"}
        pano_paths = [frame for frame in entry_frames if _panorama_direction(frame)]
        pano_frame = _panorama_concat_frame(pano_paths)
        if pano_frame is not None:
            for path in pano_paths:
                seen_frame_paths.add(path)
            if synthetic_time:
                ts = _synthetic_frame_time(entry_index, 0, max(1, len(entry_frames)))
            else:
                try:
                    ts = min(Path(path).stat().st_mtime for path in pano_paths)
                except OSError:
                    ts = math.inf
            frames.append((ts, tool, pano_frame, annotation))
        non_pano_frames = [
            frame for frame in entry_frames if not _panorama_direction(str(frame))
        ]
        for non_pano_index, frame in enumerate(non_pano_frames):
            frame_path = str(frame)
            if _is_privileged_map_frame(tool, frame_path):
                continue
            seen_frame_paths.add(frame_path)
            if synthetic_time:
                offset_index = non_pano_index + (1 if pano_frame is not None else 0)
                ts = _synthetic_frame_time(
                    entry_index, offset_index, max(1, len(entry_frames))
                )
            else:
                try:
                    ts = Path(frame_path).stat().st_mtime
                except OSError:
                    ts = math.inf
            frames.append((ts, tool, frame_path, annotation))
    if synthetic_time:
        return frames
    session_dir = manifest.get("session_dir")
    if session_dir:
        session_path = Path(session_dir)
        for ts, tool, frame_path in ordered_panorama_frames(session_path):
            pano_paths = _panorama_concat_paths(frame_path)
            if frame_path in seen_frame_paths or (
                pano_paths and all(path in seen_frame_paths for path in pano_paths)
            ):
                continue
            frames.append((ts, tool, frame_path, None))
            seen_frame_paths.add(frame_path)
            for path in pano_paths:
                seen_frame_paths.add(path)
        extra_patterns = ("step*_color_sensor.png", "wamnav_*.png", "localnav_*.png")
        for pattern in extra_patterns:
            for frame_path in glob.glob(str(session_path / pattern)):
                if frame_path in seen_frame_paths:
                    continue
                if _is_privileged_map_frame("visual", frame_path):
                    continue
                try:
                    ts = Path(frame_path).stat().st_mtime
                except OSError:
                    continue
                frames.append((ts, "visual", frame_path, None))
                seen_frame_paths.add(frame_path)
    return frames


def backfill_tool_event_times(
    events: List[Dict[str, Any]], manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Assign wall-clock times to canonical tool events that lack timestamps.

    Codex exec JSONL omits timestamps, so canonical tool_call/tool_result
    events carry null times and the trace/stages would silently drop every
    tool line. Per-call frame mtimes from the replay manifest give exact
    windows; remaining gaps interpolate between timed neighbours (assistant
    messages carry session timestamps). Display-only: the stored canonical is
    never rewritten."""
    windows: Dict[str, List[Tuple[float, float]]] = {}
    for entry in manifest.get("entries", []):
        name = str(entry.get("tool") or "")
        times: List[float] = []
        for frame in entry.get("frames", []) or []:
            try:
                times.append(Path(str(frame)).stat().st_mtime)
            except OSError:
                continue
        if times:
            windows.setdefault(name, []).append((min(times), max(times)))
    last_window: Dict[str, Tuple[float, float]] = {}
    for event in events:
        if event.get("type") not in ("tool_call", "tool_result"):
            continue
        if event_time(event, "time_start", "time_end", "timestamp") is not None:
            continue
        name = str(event.get("name") or "")
        window: Optional[Tuple[float, float]] = None
        if event.get("type") == "tool_call":
            queue = windows.get(name)
            if queue:
                window = queue.pop(0)
                last_window[name] = window
        else:
            window = last_window.get(name)
        if window is None:
            continue
        event["time_start"], event["time_end"] = window
    anchors = [
        (index, ts)
        for index, event in enumerate(events)
        if (ts := event_time(event, "time_start", "time_end", "timestamp")) is not None
    ]
    if not anchors:
        return events
    last_ts = 0.0
    for index, event in enumerate(events):
        if event.get("type") not in ("tool_call", "tool_result"):
            continue
        current = event_time(event, "time_start", "time_end", "timestamp")
        if current is not None:
            if current < last_ts:
                event["time_start"] = last_ts
                event["time_end"] = max(last_ts, event_time(event, "time_end") or last_ts)
                current = event["time_start"]
            last_ts = max(last_ts, current)
            continue
        prev = next(((j, ts) for j, ts in reversed(anchors) if j < index), None)
        nxt = next(((j, ts) for j, ts in anchors if j > index), None)
        if prev and nxt and nxt[1] > prev[1] and nxt[0] > prev[0]:
            est = prev[1] + (nxt[1] - prev[1]) * (index - prev[0]) / (nxt[0] - prev[0])
        elif prev:
            est = prev[1]
        elif nxt:
            est = nxt[1]
        else:
            continue
        est = max(est, last_ts)
        event["time_start"] = est
        event["time_end"] = est
        last_ts = est
    return events


def compressed_interval_s(
    duration_s: float, *, accelerated: bool, max_wait_s: float
) -> float:
    if duration_s <= 0.0 or not accelerated:
        return duration_s
    return min(duration_s, max_wait_s)


def make_video(
    run_dir: Path,
    out_path: Path,
    *,
    visuals_root: Path,
    fps: int,
    size: int,
    accelerated: bool = False,
    max_wait_s: float = 1.0,
) -> None:
    if max_wait_s <= 0.0:
        raise ValueError("max_wait_s must be greater than zero")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_needs_refresh(manifest, visuals_root=visuals_root):
            manifest = build_manifest(run_dir, visuals_root, write=False)
    else:
        manifest = build_manifest(run_dir, visuals_root, write=False)
    run_id = run_dir.name
    instruction = ""
    run_json_path = run_dir / "run.json"
    if run_json_path.is_file():
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
            instruction = str(run_payload.get("instruction") or "").strip()
        except json.JSONDecodeError:
            instruction = ""
    events = canonical_events(run_dir)
    events = backfill_tool_event_times(events, manifest)
    evaluator = manifest.get("evaluator_topdown")
    evaluator_map = (
        evaluator.get("map")
        if isinstance(evaluator, dict) and isinstance(evaluator.get("map"), dict)
        else {}
    )
    evaluator_transform = (
        evaluator_map.get("transform")
        if isinstance(evaluator_map.get("transform"), dict)
        else {}
    )
    if not isinstance(evaluator, dict) or evaluator_transform.get(
        "version"
    ) != "habitat_pathfinder_xz_v1":
        # Rebuild the evaluator payload in memory using the Pathfinder pixel contract.
        evaluator = build_evaluator_topdown(
            run_dir, visuals_root, events, str(manifest.get("session_id") or "")
        )
    evaluator_enabled = bool(
        isinstance(evaluator, dict)
        and evaluator.get("status") in {"complete", "partial"}
    )
    trace = trace_events(events)
    stages = stage_intervals(events)
    synthetic_time = not any(
        event_time(event, "time_start", "time_end", "timestamp") is not None
        for event in events
    )
    frames = _collect_replay_frames(manifest, synthetic_time=synthetic_time)
    frames = sorted(frames, key=lambda item: item[0])
    if not frames:
        raise RuntimeError(f"No replay frames found for {run_dir}")
    frame_times = [ts for ts, _, _, _ in frames if math.isfinite(ts)]
    trace_times = [ts for ts, _, _, _ in trace]
    stage_times = [ts for start, end, _ in stages for ts in (start, end)]
    evaluator_times = _evaluator_change_times(evaluator) if evaluator_enabled else []
    start_ts = result_time(events, "hab_set_pose", first=True)
    done_ts = result_time(events, "hab_close_session", first=False)
    if synthetic_time:
        t0 = 0.0
    else:
        t0_candidates = frame_times + ([start_ts] if start_ts is not None else [])
        t0 = min(t0_candidates)
    tend_candidates = frame_times + [
        ts for ts in trace_times + stage_times + evaluator_times if ts >= t0
    ]
    if done_ts is not None:
        tend_candidates.append(done_ts)
    tend = max(tend_candidates) + 1.5
    if not trace:
        trace = synthetic_trace_events(events, start_ts=t0, end_ts=tend)
        trace_times = [ts for ts, _, _, _ in trace]
        if trace_times:
            tend = max(tend, max(trace_times) + 1.5)
    changes: List[float] = sorted(
        {
            0.0,
            max(0.0, tend - t0),
            *[max(0.0, ts - t0) for ts in frame_times + trace_times + stage_times],
            *[max(0.0, ts - t0) for ts in evaluator_times],
        }
    )
    if len(changes) == 1:
        changes.append(changes[0] + 1.0)
    tmp = Path(tempfile.mkdtemp(prefix="harness_video_"))
    try:
        states: List[Tuple[Path, float]] = []
        current_tool = "agent"
        current_frame = ""
        current_annotation: Optional[Dict[str, Any]] = None
        frame_idx = 0
        trace_idx = 0
        visible_trace: List[Tuple[str, str, Optional[float]]] = []
        for idx, t in enumerate(changes[:-1], start=1):
            abs_t = t0 + t
            while (
                frame_idx < len(frames)
                and math.isfinite(frames[frame_idx][0])
                and frames[frame_idx][0] <= abs_t + 1e-6
            ):
                _, frame_tool, frame_path, frame_annotation = frames[frame_idx]
                current_tool = frame_tool
                current_frame = frame_path
                current_annotation = frame_annotation
                frame_idx += 1
            while trace_idx < len(trace) and trace[trace_idx][0] <= abs_t + 1e-6:
                _, kind, text, duration_s = trace[trace_idx]
                visible_trace.append((kind, text, duration_s))
                trace_idx += 1
            actual_duration = changes[idx] - t
            if actual_duration <= 1e-4:
                continue
            duration = compressed_interval_s(
                actual_duration, accelerated=accelerated, max_wait_s=max_wait_s
            )
            composed = compose_frame(
                load_visual_panel(
                    current_frame,
                    size=size,
                    tool=f"{current_tool} | t={t:.1f}s",
                    annotation=current_annotation,
                ),
                run_id=run_id,
                instruction=instruction,
                tool=f"{current_tool} | t={t:.1f}s",
                trace=visible_trace,
                status=active_stage(stages, abs_t),
                size=size,
                actual_time_s=t if accelerated else None,
                topdown_panel=(
                    render_topdown_panel(
                        evaluator,
                        actual_time=abs_t,
                        width=size,
                        height=size + max(1, size // 3),
                    )
                    if evaluator_enabled
                    else None
                ),
            )
            composed.save(tmp / f"{idx:06d}.png")
            states.append((tmp / f"{idx:06d}.png", duration))
        concat = tmp / "list.txt"
        with concat.open("w", encoding="utf-8") as f:
            for frame_path, duration in states:
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {duration:.3f}\n")
            f.write(f"file '{states[-1][0]}'\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        clock = (
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={fps},"
            f"drawtext=fontfile={font_path}:"
            "text='task clock %{eif\\:trunc(t)\\:d}.%{eif\\:trunc(mod(t*10,10))\\:d}s':"
            "x=w/2-tw-12:y=h-th-14:fontsize=20:fontcolor=0x78E6A0:"
            "box=1:boxcolor=black@0.65:boxborderw=6"
        )
        # Encoder selection: libx264 by default; set HAB_REPLAY_ENCODER
        # (e.g. h264_nvenc) to offload encoding to GPU hardware.
        encoder = os.environ.get("HAB_REPLAY_ENCODER", "libx264")
        codec_args = (
            ["-c:v", encoder, "-cq", "20"]
            if "nvenc" in encoder
            else ["-c:v", encoder, "-crf", "20"]
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-vf",
                clock,
                "-pix_fmt",
                "yuv420p",
                *codec_args,
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render one harness-dev run replay video."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--visuals-root", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--accelerated",
        action="store_true",
        help="Compress idle intervals while retaining every replay state.",
    )
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=1.0,
        help="Maximum displayed interval duration in accelerated mode.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    default_name = "replay_accelerated.mp4" if args.accelerated else "replay.mp4"
    out = Path(args.out).expanduser().resolve() if args.out else run_dir / default_name
    if args.visuals_root:
        visuals_root = Path(args.visuals_root).expanduser().resolve()
    else:
        # A replay manifest owns the artifact root used for that run. Prefer it
        # over process-local environment defaults so evaluator sidecars remain
        # discoverable when replay is rendered from a fresh shell.
        manifest_visuals_root: Optional[str] = None
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                value = manifest_payload.get("visuals_root")
                if isinstance(value, str) and value.strip():
                    manifest_visuals_root = value
            except json.JSONDecodeError:
                pass
        visuals_root = (
            Path(manifest_visuals_root).expanduser().resolve()
            if manifest_visuals_root
            else default_visuals_root()
        )
    make_video(
        run_dir,
        out,
        visuals_root=visuals_root,
        fps=args.fps,
        size=args.size,
        accelerated=args.accelerated,
        max_wait_s=args.max_wait_s,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
