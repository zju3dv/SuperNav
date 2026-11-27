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

"""Four locally attached USB cameras for Go2 MCP observation."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import numpy as np

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.webcam import Webcam
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

CAMERA_DIRECTIONS = ("front", "right", "back", "left")
SURROUND_SHM_TOPICS = {direction: f"go2_surround_{direction}" for direction in CAMERA_DIRECTIONS}
RERUN_IMAGE_DOWNSAMPLE = 2
DEFAULT_CAMERA_IMAGE_MAX_AGE_S = 3.0
_ROTATIONS = {0, 90, 180, 270}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def surround_image_health(
    images: dict[str, Image],
    *,
    now_s: float | None = None,
    max_age_s: float | None = None,
) -> dict[str, Any]:
    """Classify cached surround frames before exposing them to an agent."""

    now_s = time.time() if now_s is None else now_s
    max_age_s = (
        float(os.environ.get("DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S", DEFAULT_CAMERA_IMAGE_MAX_AGE_S))
        if max_age_s is None
        else max_age_s
    )
    timestamps = {
        direction: float(images[direction].ts) if direction in images else None
        for direction in CAMERA_DIRECTIONS
    }
    age_s = {
        direction: max(0.0, now_s - ts) if ts is not None else None
        for direction, ts in timestamps.items()
    }
    missing = [direction for direction in CAMERA_DIRECTIONS if direction not in images]
    stale = [direction for direction, age in age_s.items() if age is not None and age > max_age_s]
    return {
        "ok": not missing and not stale,
        "fresh": [
            direction
            for direction in CAMERA_DIRECTIONS
            if direction in images and direction not in stale
        ],
        "missing": missing,
        "stale": stale,
        "timestamps": timestamps,
        "age_s": age_s,
        "max_age_s": max_age_s,
    }


class Go2SurroundCameraModule(Module):
    """Capture and cache four direction-labelled USB camera streams."""

    surround_front_image: Out[Image]
    surround_right_image: Out[Image]
    surround_back_image: Out[Image]
    surround_left_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        devices = {
            direction: os.environ.get(f"DIMOS_GO2_CAMERA_{direction.upper()}_DEVICE", "").strip()
            for direction in CAMERA_DIRECTIONS
        }
        rotations = {
            direction: _env_int(f"DIMOS_GO2_CAMERA_{direction.upper()}_ROTATION", 0)
            for direction in CAMERA_DIRECTIONS
        }
        invalid = {
            direction: rotation
            for direction, rotation in rotations.items()
            if rotation not in _ROTATIONS
        }
        if invalid:
            raise ValueError(f"camera rotations must be one of {_ROTATIONS}: {invalid}")
        super().__init__(**kwargs)
        self._width = _env_int("DIMOS_GO2_CAMERA_WIDTH", 1600)
        self._height = _env_int("DIMOS_GO2_CAMERA_HEIGHT", 1200)
        self._fourcc = os.environ.get("DIMOS_GO2_CAMERA_FOURCC", "MJPG").strip() or None
        self._front_fps = float(os.environ.get("DIMOS_GO2_CAMERA_FRONT_FPS", "15"))
        self._other_fps = float(os.environ.get("DIMOS_GO2_CAMERA_OTHER_FPS", "5"))
        self._devices = devices
        self._rotations = rotations
        self._latest: dict[str, Image] = {}
        self._lock = threading.Lock()
        self._cameras: dict[str, Webcam] = {}

    @rpc
    def start(self) -> None:
        super().start()
        for direction, device in self._devices.items():
            if not device:
                continue
            fps = self._front_fps if direction == "front" else self._other_fps
            camera = Webcam(
                camera_index=device,
                width=self._width,
                height=self._height,
                fps=fps,
                fourcc=self._fourcc,
                frame_id_prefix=f"go2/{direction}",
            )
            self._cameras[direction] = camera
            stream = camera.image_stream()
            disposable = stream.subscribe(
                on_next=lambda image, direction=direction: self._on_image(direction, image),
                on_error=lambda error, direction=direction: logger.error(
                    "Go2 %s camera failed: %s", direction, error
                ),
            )
            self.register_disposable(disposable)
        logger.info(
            "Go2 surround cameras started: configured=%s size=%dx%d front_fps=%s other_fps=%s fourcc=%s",
            [direction for direction, device in self._devices.items() if device],
            self._width,
            self._height,
            self._front_fps,
            self._other_fps,
            self._fourcc or "auto",
        )

    def _on_image(self, direction: str, image: Image) -> None:
        rotation = self._rotations[direction]
        if rotation:
            rgb = np.rot90(np.asarray(image.to_rgb().data), k=rotation // 90)
            image = Image.from_numpy(
                np.ascontiguousarray(rgb),
                format=ImageFormat.RGB,
                frame_id=f"go2/{direction}/camera_optical",
                ts=image.ts,
            )
        with self._lock:
            self._latest[direction] = image
        getattr(self, f"surround_{direction}_image").publish(image)

    def latest_images(self) -> dict[str, Image]:
        with self._lock:
            return dict(self._latest)

    @skill
    def refresh_rerun_images(self) -> str:
        """Publish one current frame from each USB camera to Rerun.

        Images remain frozen in Rerun until this skill is called again. Pixel
        clicks on the refreshed images continue to route to PixelGoal.
        """
        images = self.latest_images()
        health = surround_image_health(images)
        fresh = set(health["fresh"])
        published: list[str] = []
        publisher = LCM()
        try:
            for direction in CAMERA_DIRECTIONS:
                image = images.get(direction)
                if image is None or direction not in fresh:
                    continue
                snapshot = image.resize(
                    max(1, image.width // RERUN_IMAGE_DOWNSAMPLE),
                    max(1, image.height // RERUN_IMAGE_DOWNSAMPLE),
                )
                publisher.publish(Topic(f"/surround_{direction}_image", Image), snapshot)
                published.append(direction)
        finally:
            publisher.stop()
        return json.dumps(
            {
                **health,
                "published": published,
            }
        )

    @skill
    def observe_surround(self) -> list[dict[str, Any]]:
        """Return the latest front, right, back, and left USB camera frames.

        The four cameras are captured continuously. Cached frames older than
        ``DIMOS_GO2_CAMERA_IMAGE_MAX_AGE_S`` are reported but not attached.
        """
        images = self.latest_images()
        health = surround_image_health(images)
        fresh = set(health["fresh"])
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        **health,
                        "directions": list(CAMERA_DIRECTIONS),
                        "width": self._width,
                        "height": self._height,
                    }
                ),
            }
        ]
        for direction in CAMERA_DIRECTIONS:
            image = images.get(direction)
            if image is None or direction not in fresh:
                continue
            content.append({"type": "text", "text": direction})
            content.extend(image.agent_encode())
        return content


__all__ = [
    "CAMERA_DIRECTIONS",
    "DEFAULT_CAMERA_IMAGE_MAX_AGE_S",
    "RERUN_IMAGE_DOWNSAMPLE",
    "SURROUND_SHM_TOPICS",
    "Go2SurroundCameraModule",
    "surround_image_health",
]
