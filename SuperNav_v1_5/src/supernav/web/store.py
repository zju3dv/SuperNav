"""Read-only view models over explicitly selected telemetry/evidence directories."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time

from supernav.web.trace import TraceReader

_FRAME = re.compile(r"(?:step\d+_color_sensor|pano_front_step\d+_color_sensor|localnav_\d+_leg\d+_step\d+)\.png$")
_PANO = re.compile(r"pano_(front|right|back|left)_step(\d+)_color_sensor(_agent)?\.png$")
_PRUNE = {"node_modules", ".git", "visuals", "skills", "codex_project", "opencode_project",
          "kimi_home", ".codex_home", "_bridge_logs", "__pycache__"}


def read_json(path: Path, limit: int = 16 * 1024 * 1024) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            return {}
        def invalid_constant(value):
            raise ValueError(value)
        value = json.loads(raw, parse_constant=invalid_constant)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def timestamp(value) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return 0.0


class SessionStore:
    def __init__(self, live_dir: Path, *, visuals_roots=(), runs_root: Path | None = None,
                 demo: bool = False):
        self.live_dir = live_dir.resolve()
        self.visuals_roots = [Path(p).resolve() for p in visuals_roots]
        self.runs_root = runs_root.resolve() if runs_root else None
        self.demo = demo
        self.demo_started = time.time()
        self.lock = threading.RLock()
        self._archives = {}
        self._metadata = {}
        self._next_scan = 0
        self._archive_cache = {}
        self._run_paths = {}
        self._session_runs = {}
        self._trace_readers = {}

    def _scan(self):
        if time.monotonic() < self._next_scan:
            return
        self._next_scan = time.monotonic() + 3
        self._archives = {}
        for root in self.visuals_roots:
            for path in sorted(root.glob("*.trajectory.json"))[:1000]:
                if contained(path, root):
                    key = "archive-" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
                    self._archives[key] = path
        if not self.runs_root:
            return
        metadata = {}
        run_paths, session_runs = {}, {}
        for index, (directory, dirs, files) in enumerate(os.walk(self.runs_root, followlinks=False)):
            if index >= 3000:
                break
            dirs[:] = sorted(d for d in dirs if d not in _PRUNE and not d.startswith("."))
            base = Path(directory)
            if len(base.relative_to(self.runs_root).parts) >= 8:
                dirs.clear()
            if "run.json" not in files or not contained(base, self.runs_root):
                continue
            def read(name):
                path = base / name
                return read_json(path) if contained(path, self.runs_root) else {}
            run, metrics, manifest = read("run.json"), read("metrics.json"), read("manifest.json")
            run_paths[str(run.get("run_id") or base.name)] = base
            sid = metrics.get("session_id") or manifest.get("session_id")
            if isinstance(sid, str):
                session_runs[sid] = base
                metadata[sid] = {
                    "instruction": run.get("instruction"), "agent": run.get("agent"),
                    "run_id": run.get("run_id"), "task_id": run.get("task_id"),
                    "agent_claim": metrics.get("agent_terminal_claim"),
                    # metrics.success is a completion claim, never an offline GT score.
                    "evaluation": None,
                }
        self._metadata = metadata
        self._run_paths, self._session_runs = run_paths, session_runs

    @staticmethod
    def _media_urls(doc, key):
        for group in doc.get("panoramas", []):
            for direction, view in group.get("views", {}).items():
                view["image_url"] = f"/api/sessions/{key}/frame?kind=panorama&index={group['id']}&view={direction}"
        for overlay in doc.get("overlays", []):
            overlay["image_url"] = f"/api/sessions/{key}/frame?kind=overlay&index={overlay['id']}"

    def _live(self, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key):
            return None
        path = self.live_dir / f"{key}.json"
        if not contained(path, self.live_dir):
            return None
        doc = read_json(path)
        if doc.get("schema_version") != 1 or doc.get("session_id") != key:
            return None
        # Only the view changes when the producer disappears; evidence is untouched.
        doc["source"] = "live"
        if doc.get("status") == "running" and doc.get("producer_host") == socket.gethostname():
            try:
                pid = int(doc.get("producer_pid", 0))
                if pid <= 0:
                    raise ProcessLookupError
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError):
                doc["status"] = "interrupted"
            except PermissionError:
                pass
        doc.pop("producer_pid", None)
        doc.pop("producer_host", None)
        doc.update(self._metadata.get(key, {}))
        doc["id"] = key
        if doc.get("frame_revision"):
            doc["image_url"] = f"/api/sessions/{key}/frame?v={doc['frame_revision']}"
        self._media_urls(doc, key)
        return doc

    def _archive(self, key):
        path = self._archives[key]
        if not contained(path, path.parent):
            return None
        try:
            revision = path.stat().st_mtime_ns
        except OSError:
            return None
        # Refresh while an old-style producer is still writing its evidence.
        cache_key = (revision, int(time.time() / 3))
        cached = self._archive_cache.get(key)
        if cached and cached[0][0] == revision and (cached[1]["status"] != "running" or cached[0] == cache_key):
            return {**cached[1], **self._metadata.get(cached[1]["session_id"], {})}
        raw = read_json(path)
        sid = raw.get("session_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sid):
            return None
        dense = raw.get("dense_trajectory")
        points = (dense.get("points") if isinstance(dense, dict) else None) or raw.get("trajectory_points") or []
        points = [p for p in points if isinstance(p, list) and len(p) == 3
                  and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)]
        samples = [s for s in (raw.get("trajectory") or []) if isinstance(s, dict)]
        events = [{"action": s.get("tool", "observation"), "phase": s.get("event"),
                   "timestamp": timestamp(s.get("ts")), "step": s.get("step_count"),
                   "ok": None} for s in raw.get("events", [])[-100:] if isinstance(s, dict)]
        folder = path.parent / sid
        frame_paths = []
        panorama_groups, media_paths = {}, {}
        if contained(folder, path.parent) and folder.is_dir():
            for p in folder.iterdir():
                match = _PANO.fullmatch(p.name)
                if match and contained(p, path.parent):
                    direction, capture_seq, agent = match.groups()
                    sequence = int(capture_seq)
                    group = panorama_groups.setdefault(sequence, {"id": sequence, "capture_seq": sequence,
                                                                   "timestamp": p.stat().st_mtime, "views": {}})
                    group["timestamp"] = min(group["timestamp"], p.stat().st_mtime)
                    asset_key = f"panorama:{sequence}:{direction}"
                    if agent or asset_key not in media_paths:
                        media_paths[asset_key] = p
                        group["views"][direction] = {"image_ref": f"pano:{sid}:{sequence}:{direction}"}
                if _FRAME.fullmatch(p.name) and contained(p, path.parent):
                    try:
                        frame_paths.append((p.stat().st_mtime, p))
                    except OSError:
                        pass
        frame_paths.sort(key=lambda row: (row[0], row[1].name))
        frame_paths = frame_paths[-10000:]
        frames = [{"url": f"/api/sessions/{key}/frame?index={i}", "timestamp": ts}
                  for i, (ts, _) in enumerate(frame_paths)]
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        overlays = []
        audit = path.parent / f"{sid}.benchmark_audit.jsonl"
        if contained(audit, path.parent) and audit.is_file():
            with audit.open("rb") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                        result = event.get("result") or {}
                        overlay_path = result.get("overlay_image")
                        if not overlay_path:
                            continue
                        image_path = Path(overlay_path)
                        if not image_path.is_absolute():
                            image_path = path.parent / image_path
                        if not contained(image_path, path.parent) or not image_path.is_file():
                            continue
                        anchor = result.get("selected_anchor") or result.get("original_annotation") or {}
                        ref = str(anchor.get("image_ref") or result.get("image_ref") or "")
                        overlay_id = int(event["tool_seq"])
                        capture_seq = anchor.get("capture_seq")
                        if capture_seq is None and ref.startswith("pano:"):
                            capture_seq = int(ref.split(":")[-2])
                        overlays.append({"id": overlay_id, "tool": event.get("tool_name"), "image_ref": ref,
                                         "capture_seq": capture_seq, "direction": anchor.get("direction") or ref.split(":")[-1],
                                         "point": anchor.get("point"), "anchor_px": anchor.get("anchor_px"),
                                         "timestamp": image_path.stat().st_mtime,
                                         "completed_at": timestamp(event.get("timestamp"))})
                        media_paths[f"overlay:{overlay_id}:"] = image_path
                    except (ValueError, TypeError, KeyError, AttributeError):
                        continue
        doc = {
            "id": key, "session_id": sid, "source": "recording", "status": raw.get("status", "unknown"),
            "scene": raw.get("scene") or raw.get("scene_id"), "revision": revision,
            "started_at": timestamp(raw.get("started_at")), "ended_at": timestamp(raw.get("ended_at")),
            "updated_at": path.stat().st_mtime, "frame_at": frames[-1]["timestamp"] if frames else None,
            "image_url": frames[-1]["url"] if frames else None, "frames": frames,
            "step": samples[-1].get("step_count") if samples else None,
            "pose": raw.get("end_pose") or (samples[-1].get("pose") if samples else None),
            "trajectory": points[::max(1, math.ceil(len(points) / 4000))],
            "trail_truncated": len(points) > 4000, "path_length_m": length,
            "action": events[-1]["action"] if events else "recorded observations", "events": events,
            "agent_claim": None, "evaluation": None,
            "panoramas": [panorama_groups[k] for k in sorted(panorama_groups)][-200:],
            "overlays": overlays[-200:],
            **self._metadata.get(sid, {}),
        }
        self._media_urls(doc, key)
        self._archive_cache[key] = (cache_key, doc, frame_paths, media_paths)
        return doc

    def _demo(self):
        elapsed = time.time() - self.demo_started
        step = int(elapsed * 4) % 160
        trail = [[round(2 + 4 * math.cos(i / 25), 3), 0, round(3 + 2.4 * math.sin(i / 25), 3)]
                 for i in range(step + 1)]
        actions = ["Observe surroundings", "Choose a waypoint", "Move toward the doorway", "Check the next view"]
        return {
            "id": "demo", "session_id": "demo", "source": "demo", "status": "running",
            "scene": "Illustrated apartment", "instruction": "Explore the room and find the orange chair.",
            "agent": "Sample agent", "revision": int(elapsed * 5), "started_at": self.demo_started,
            "updated_at": time.time(), "step": step,
            "pose": {"position": trail[-1], "heading_deg": math.degrees(step / 25) + 90},
            "trajectory": trail, "path_length_m": sum(math.dist(a,b) for a,b in zip(trail,trail[1:])),
            "action": actions[(step // 12) % 4], "evaluation": None, "agent_claim": None,
            "events": [{"action": actions[(i // 12) % 4], "phase": "sample", "ok": None,
                        "step": i, "timestamp": self.demo_started + i / 4}
                       for i in range(0, step + 1, 12)][-12:],
        }

    def sessions(self):
        with self.lock:
            self._scan()
            rows = []
            for path in sorted(self.live_dir.glob("*.json"))[:1000]:
                doc = self._live(path.stem)
                if doc:
                    rows.append(doc)
            for key in self._archives:
                doc = self._archive(key)
                if doc:
                    rows.append(doc)
            rows.sort(key=lambda d: (d.get("source") == "live" and d.get("status") == "running",
                                     d.get("updated_at", 0)), reverse=True)
            if self.demo:
                rows.append(self._demo())
            return [{k: row.get(k) for k in ("id", "session_id", "source", "status", "scene",
                                           "instruction", "agent", "updated_at", "step")}
                    for row in rows]

    def snapshot(self, key):
        with self.lock:
            self._scan()
            if key == "demo" and self.demo:
                return self._demo()
            if key in self._archives:
                return self._archive(key)
            return self._live(key)

    def trace(self, key):
        with self.lock:
            self._scan()
            doc = self.snapshot(key)
            if not doc:
                return None
            run_dir = self._session_runs.get(doc["session_id"]) or self._run_paths.get(str(doc.get("run_id")))
            if run_dir is None:
                return {"status": "unlinked", "events": [], "note": "Waiting for the session's agent run to be linked."}
            reader = self._trace_readers.setdefault(run_dir, TraceReader(run_dir))
            return reader.snapshot()

    def frame(self, key, index=0, *, kind="rgb", view=""):
        with self.lock:
            self._scan()
            if key in self._archives:
                if not self._archive(key):
                    return None
                paths = self._archive_cache[key][2]
                if kind == "rgb":
                    if not 0 <= index < len(paths):
                        return None
                    path = paths[index][1]
                else:
                    path = self._archive_cache[key][3].get(f"{kind}:{index}:{view}")
                    if path is None:
                        return None
                root = self._archives[key].parent
            elif doc := self._live(key):
                root = self.live_dir
                if kind == "rgb":
                    path = self.live_dir / f"{key}.jpg"
                elif kind == "panorama" and view in {"front", "right", "back", "left"}:
                    group = next((p for p in doc.get("panoramas", []) if p["id"] == index), None)
                    if not group or view not in group["views"]:
                        return None
                    path = self.live_dir / f"{key}.pano-{index}-{view}.png"
                elif kind == "overlay" and any(o["id"] == index for o in doc.get("overlays", [])):
                    path = self.live_dir / f"{key}.overlay-{index}.png"
                else:
                    return None
            else:
                return None
            if contained(path, root) and path.is_file():
                return path
            return None
