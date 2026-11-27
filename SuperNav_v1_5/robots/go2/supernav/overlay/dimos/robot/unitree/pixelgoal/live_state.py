# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live-state sink for PixelGoal: frames + status + events on disk.

The closed loop and the MCP server write here; the live dashboard reads it.
Everything is best-effort — a dashboard write failure must never break a hop.
Writes are atomic (tmp file + rename) so readers never see torn JSON/JPEG;
``events.jsonl`` is append-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_LIVE_DIR = Path(__file__).resolve().parents[4] / "runs" / "pixelgoal-live"

_STATUS_FILE = "status.json"
_EVENTS_FILE = "events.jsonl"
_FRAME_FILE = "frame.jpg"
_GOAL_FRAME_FILE = "goal_frame.jpg"
_FRAMES_DIR = "frames"


class LiveStateWriter:
    """Best-effort atomic writer for the live directory contract."""

    def __init__(self, live_dir: str | Path, *, archive_frames: bool = True) -> None:
        self.live_dir = Path(live_dir)
        self.archive_frames = archive_frames
        try:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            if archive_frames:
                (self.live_dir / _FRAMES_DIR).mkdir(exist_ok=True)
        except OSError:
            pass  # every write is independently best-effort
        self._lock = threading.Lock()

    # -- writes ------------------------------------------------------------

    def write_frame(self, rgb: np.ndarray, *, goal: bool = False) -> None:
        """Latest camera frame (or the marked goal frame) as JPEG."""
        name = _GOAL_FRAME_FILE if goal else _FRAME_FILE
        try:
            image = Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)))
            fd, tmp = tempfile.mkstemp(dir=self.live_dir, prefix=f".{name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    image.save(fh, format="JPEG", quality=85)
                os.replace(tmp, self.live_dir / name)
            except BaseException:
                os.unlink(tmp)
                raise
            if self.archive_frames and not goal:
                image.save(
                    self.live_dir / _FRAMES_DIR / f"{time.time():.3f}.jpg",
                    format="JPEG",
                    quality=85,
                )
        except Exception:  # noqa: BLE001,S110 — telemetry must never break a hop
            pass

    def write_status(self, status: dict[str, Any]) -> None:
        payload = dict(status)
        payload["ts"] = time.time()
        self._atomic_text(_STATUS_FILE, json.dumps(payload))

    def event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        line = json.dumps({"ts": time.time(), "type": str(event_type), "data": dict(data or {})})
        try:
            with (
                self._lock,
                (self.live_dir / _EVENTS_FILE).open("a", encoding="utf-8") as fh,
            ):
                fh.write(line + "\n")
        except OSError:
            pass

    def clear_events(self) -> None:
        """Rotate the event log at demo start (fresh run, fresh timeline)."""
        try:
            (self.live_dir / _EVENTS_FILE).unlink(missing_ok=True)
        except OSError:
            pass

    # -- internals ----------------------------------------------------------

    def _atomic_text(self, name: str, text: str) -> None:
        try:
            fd, tmp = tempfile.mkstemp(dir=self.live_dir, prefix=f".{name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(tmp, self.live_dir / name)
            except BaseException:
                os.unlink(tmp)
                raise
        except Exception:  # noqa: BLE001,S110
            pass


def live_dir_from_env(env: dict[str, str] | None = None) -> Path:
    """Resolve DIMOS_PIXELGOAL_LIVE_DIR (default ``<repo>/runs/pixelgoal-live/``)."""
    environ = env if env is not None else os.environ
    raw = environ.get("DIMOS_PIXELGOAL_LIVE_DIR")
    return Path(raw) if raw else DEFAULT_LIVE_DIR
