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

"""Best-effort four-camera motion recording shared by Go2 navigation tools."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.go2.surround_camera import CAMERA_DIRECTIONS
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

MOTION_REC_HZ_ENV = "DIMOS_PIXELGOAL_MOTION_HZ"
MOTION_REC_DIR_ENV = "DIMOS_PIXELGOAL_MOTION_DIR"
MOTION_REC_JPEG_QUALITY = 80


class HopMotionRecorder:
    """Write rate-limited surround frames without slowing robot navigation."""

    def __init__(self, root: Path, goal_id: str, hz: float) -> None:
        tail = "".join(character for character in goal_id if character.isalnum())[-8:]
        self.dir = root / f"hop_{int(time.time())}_{tail or 'nogoal'}"
        self.goal_id = goal_id
        self.started_at = time.time()
        self.saved = 0
        self._interval = 1.0 / hz
        self._last_ts: dict[str, float] = {}
        self._queue: queue.Queue[tuple[str, float, np.ndarray] | None] = queue.Queue(
            maxsize=16
        )
        self._disabled = False
        self._thread = threading.Thread(
            target=self._writer,
            name="go2-motion-recorder",
            daemon=True,
        )

    def start(self) -> None:
        for direction in CAMERA_DIRECTIONS:
            (self.dir / direction).mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def offer(self, direction: str, image: Image) -> None:
        if self._disabled:
            return
        timestamp_s = float(image.ts or time.time())
        last_timestamp_s = self._last_ts.get(direction)
        if (
            last_timestamp_s is not None
            and timestamp_s - last_timestamp_s < self._interval
        ):
            return
        self._last_ts[direction] = timestamp_s
        try:
            rgb = np.asarray(image.to_rgb().data)
        except Exception:  # noqa: BLE001 - unconvertible frames are skipped
            return
        try:
            self._queue.put_nowait((direction, timestamp_s, rgb))
        except queue.Full:
            pass

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            direction, timestamp_s, rgb = item
            try:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, MOTION_REC_JPEG_QUALITY],
                )
                if not ok:
                    raise ValueError("cv2.imencode returned False")
                frame_path = (
                    self.dir / direction / f"{int(round(timestamp_s * 1000))}.jpg"
                )
                frame_path.write_bytes(buffer.tobytes())
                self.saved += 1
            except Exception as exc:  # noqa: BLE001 - recording is best-effort
                self._disabled = True
                logger.warning("Go2 motion recorder disabled: %s", exc)
                return

    def finish(self, metadata: dict[str, Any]) -> None:
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logger.warning("Go2 motion recorder queue did not drain before shutdown")
        self._thread.join(timeout=5.0)
        payload = {
            "goal_id": self.goal_id,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "frames_saved": self.saved,
            **metadata,
        }
        try:
            (self.dir / "meta.json").write_text(
                json.dumps(payload, indent=1, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            logger.warning("Go2 motion recorder metadata write failed: %s", exc)
