"""Optional, bounded spectator telemetry. No simulator imports or agent inputs.

The producer runs on the simulator thread; readers only see atomic snapshots.
This is deliberately separate from evidence logs and evaluation trajectories.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Callable


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class LivePublisher:
    """Publish latest RGB and state at a bounded rate, retaining a bounded trail."""

    def __init__(self, directory: Path, session_id: str, *, scene: str, fps: float = 5):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("Invalid live session id")
        if not math.isfinite(fps) or not 0 < fps <= 30:
            raise ValueError("Live FPS must be between 0 and 30")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.interval = 1 / fps
        self.next_capture = 0.0
        self.state = {
            "schema_version": 1, "session_id": session_id, "scene": scene,
            "source": "live", "status": "running", "started_at": time.time(),
            "producer_pid": os.getpid(), "producer_host": socket.gethostname(),
            "revision": 0, "frame_revision": 0, "frame_at": None,
            "step": 0, "pose": None, "trajectory": [], "trail_truncated": False,
            "path_length_m": 0.0, "action": "initializing", "events": [],
            "capture_error": None,
            "panoramas": [], "overlays": [],
        }
        self._panorama_revision = 0
        self._overlay_revision = 0

    def panorama(self, images: list[dict], capture_seq: int) -> None:
        self._panorama_revision += 1
        revision = self._panorama_revision
        views = {}
        for row in images:
            direction = row.get("direction")
            if direction not in {"front", "right", "back", "left"} or not row.get("path"):
                continue
            try:
                data = Path(row["path"]).read_bytes()
                atomic_write(self.directory / f"{self.session_id}.pano-{revision}-{direction}.png", data)
            except OSError:
                continue
            views[direction] = {"image_ref": row.get("image_ref"), "heading_deg": row.get("heading_deg")}
        self.state["panoramas"].append({"id": revision, "capture_seq": capture_seq,
                                        "timestamp": time.time(), "views": views})
        expired = self.state["panoramas"][:-12]
        del self.state["panoramas"][:-12]
        self.publish()
        for group in expired:
            for direction in group["views"]:
                (self.directory / f"{self.session_id}.pano-{group['id']}-{direction}.png").unlink(missing_ok=True)

    def overlay(self, path: str, metadata: dict) -> None:
        self._overlay_revision += 1
        revision = self._overlay_revision
        atomic_write(self.directory / f"{self.session_id}.overlay-{revision}.png", Path(path).read_bytes())
        self.state["overlays"].append({**metadata, "id": revision, "timestamp": time.time()})
        expired = self.state["overlays"][:-24]
        del self.state["overlays"][:-24]
        self.publish()
        for overlay in expired:
            (self.directory / f"{self.session_id}.overlay-{overlay['id']}.png").unlink(missing_ok=True)

    def due(self) -> bool:
        return time.monotonic() >= self.next_capture

    def sample(self, *, position: list[float], heading: float, step: int,
               path_length: float, capture: Callable, force: bool = False) -> None:
        if not force and not self.due():
            return
        self.next_capture = time.monotonic() + self.interval
        self.state.update(step=step, pose={"position": position, "heading_deg": heading},
                          path_length_m=path_length)
        trail = self.state["trajectory"]
        if not trail or position != trail[-1]:
            trail.append(position)
        if len(trail) > 4000:
            del trail[:-4000]
            self.state["trail_truncated"] = True
        try:
            # Pillow is a base SuperNav dependency, but is only needed by producers.
            from io import BytesIO
            from PIL import Image

            rgb = capture()
            if rgb is not None:
                frame = Image.fromarray(rgb).convert("RGB")
                frame.thumbnail((960, 720))
                buffer = BytesIO()
                frame.save(buffer, format="JPEG", quality=82)
                atomic_write(self.directory / f"{self.session_id}.jpg", buffer.getvalue())
                self.state["frame_revision"] += 1
                self.state["frame_at"] = time.time()
                self.state["capture_error"] = None
        except Exception as exc:
            # A spectator failing must never turn a successful move into a retry.
            self.state["capture_error"] = type(exc).__name__
        self.publish()

    def event(self, action: str, phase: str, *, ok: bool | None = None) -> None:
        self.state["action"] = action
        self.state["events"].append({"action": action, "phase": phase, "ok": ok,
                                     "timestamp": time.time(), "step": self.state["step"]})
        del self.state["events"][:-100]
        self.publish()

    def finish(self, status: str = "closed") -> None:
        self.state["status"] = status
        self.state["ended_at"] = time.time()
        self.publish()

    def publish(self) -> None:
        self.state["revision"] += 1
        self.state["updated_at"] = time.time()
        atomic_write(self.directory / f"{self.session_id}.json",
                     json.dumps(self.state, ensure_ascii=False, allow_nan=False).encode())
