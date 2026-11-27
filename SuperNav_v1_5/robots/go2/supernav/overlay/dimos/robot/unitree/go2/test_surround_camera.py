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

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from reactivex.subject import Subject

from dimos.hardware.sensors.camera import webcam
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.unitree.go2 import surround_camera
from dimos.robot.unitree.go2.surround_camera import (
    CAMERA_DIRECTIONS,
    Go2SurroundCameraModule,
)


def _image(width: int = 8, height: int = 6, ts: float = 123.0) -> Image:
    return Image.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        format=ImageFormat.RGB,
        ts=ts,
    )


@pytest.fixture(autouse=True)
def camera_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surround_camera.time, "time", lambda: 123.5)
    for direction in CAMERA_DIRECTIONS:
        monkeypatch.delenv(f"DIMOS_GO2_CAMERA_{direction.upper()}_DEVICE", raising=False)
        monkeypatch.delenv(f"DIMOS_GO2_CAMERA_{direction.upper()}_ROTATION", raising=False)
    for name in (
        "DIMOS_GO2_CAMERA_WIDTH",
        "DIMOS_GO2_CAMERA_HEIGHT",
        "DIMOS_GO2_CAMERA_FOURCC",
        "DIMOS_GO2_CAMERA_FRONT_FPS",
        "DIMOS_GO2_CAMERA_OTHER_FPS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_observe_surround_reports_missing_frames() -> None:
    module = Go2SurroundCameraModule()
    try:
        content = module.observe_surround()
        payload = json.loads(content[0]["text"])
        assert payload == {
            "ok": False,
            "directions": ["front", "right", "back", "left"],
            "missing": ["front", "right", "back", "left"],
            "width": 1600,
            "height": 1200,
            "fresh": [],
            "timestamps": dict.fromkeys(CAMERA_DIRECTIONS),
            "age_s": dict.fromkeys(CAMERA_DIRECTIONS),
            "stale": [],
            "max_age_s": 3.0,
        }
    finally:
        module.stop()


def test_observe_surround_returns_labelled_images_in_direction_order() -> None:
    module = Go2SurroundCameraModule()
    try:
        for direction in reversed(CAMERA_DIRECTIONS):
            module._on_image(direction, _image())

        content = module.observe_surround()
        payload = json.loads(content[0]["text"])
        assert payload["ok"] is True
        assert payload["fresh"] == list(CAMERA_DIRECTIONS)
        assert payload["timestamps"] == dict.fromkeys(CAMERA_DIRECTIONS, 123.0)
        assert payload["age_s"] == dict.fromkeys(CAMERA_DIRECTIONS, 0.5)
        assert [block["text"] for block in content if block["type"] == "text"][1:] == list(
            CAMERA_DIRECTIONS
        )
        assert sum(block["type"] == "image_url" for block in content) == 4
    finally:
        module.stop()


def test_observe_surround_reports_and_omits_stale_frame() -> None:
    module = Go2SurroundCameraModule()
    try:
        for direction in CAMERA_DIRECTIONS:
            module._on_image(direction, _image(ts=100.0 if direction == "left" else 123.0))

        content = module.observe_surround()
        payload = json.loads(content[0]["text"])

        assert payload["ok"] is False
        assert payload["stale"] == ["left"]
        assert payload["age_s"]["left"] == 23.5
        assert "left" not in [block.get("text") for block in content]
        assert sum(block["type"] == "image_url" for block in content) == 3
    finally:
        module.stop()


def test_camera_rotation_is_applied_before_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIMOS_GO2_CAMERA_RIGHT_ROTATION", "90")
    module = Go2SurroundCameraModule()
    try:
        module._on_image("right", _image(width=8, height=6))
        rotated = module.latest_images()["right"]
        assert (rotated.width, rotated.height) == (6, 8)
        assert rotated.ts == 123.0
    finally:
        module.stop()


def test_invalid_camera_rotation_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIMOS_GO2_CAMERA_LEFT_ROTATION", "45")
    with pytest.raises(ValueError, match="camera rotations"):
        Go2SurroundCameraModule()


@pytest.mark.parametrize(("fourcc", "expected_fourcc"), [(None, "MJPG"), ("", None)])
def test_start_uses_camera_settings(
    monkeypatch: pytest.MonkeyPatch,
    fourcc: str | None,
    expected_fourcc: str | None,
) -> None:
    created: list[dict[str, Any]] = []

    class FakeWebcam:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)
            self._subject: Subject[Image] = Subject()

        def image_stream(self) -> Subject[Image]:
            return self._subject

    monkeypatch.setattr(surround_camera, "Webcam", FakeWebcam)
    for direction in CAMERA_DIRECTIONS:
        monkeypatch.setenv(f"DIMOS_GO2_CAMERA_{direction.upper()}_DEVICE", f"/dev/{direction}")
    if fourcc is not None:
        monkeypatch.setenv("DIMOS_GO2_CAMERA_FOURCC", fourcc)

    module = Go2SurroundCameraModule()
    module.start()
    try:
        assert [config["fps"] for config in created] == [15.0, 5.0, 5.0, 5.0]
        assert [config["camera_index"] for config in created] == [
            "/dev/front",
            "/dev/right",
            "/dev/back",
            "/dev/left",
        ]
        assert all(config["width"] == 1600 for config in created)
        assert all(config["height"] == 1200 for config in created)
        assert all(config["fourcc"] == expected_fourcc for config in created)
    finally:
        module.stop()


def test_webcam_applies_requested_fourcc(monkeypatch: pytest.MonkeyPatch) -> None:
    fourcc = webcam.cv2.VideoWriter_fourcc(*"MJPG")
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = float(fourcc)
    thread = MagicMock()
    thread.is_alive.return_value = False
    monkeypatch.setattr(webcam.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(webcam.threading, "Thread", lambda **kwargs: thread)
    camera = webcam.Webcam(camera_index="/dev/video-test", fourcc="MJPG")

    camera.start()

    capture.set.assert_any_call(webcam.cv2.CAP_PROP_FOURCC, fourcc)
    camera.stop()


def test_refresh_rerun_images_publishes_one_downsampled_snapshot_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLCM:
        instance: FakeLCM

        def __init__(self) -> None:
            self.published: list[tuple[Any, Image]] = []
            self.stopped = False
            FakeLCM.instance = self

        def publish(self, topic: Any, image: Image) -> None:
            self.published.append((topic, image))

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(surround_camera, "LCM", FakeLCM)
    module = Go2SurroundCameraModule()
    try:
        for direction in CAMERA_DIRECTIONS:
            module._on_image(direction, _image())

        payload = json.loads(module.refresh_rerun_images())
        fake = FakeLCM.instance

        assert payload == {
            "ok": True,
            "fresh": list(CAMERA_DIRECTIONS),
            "published": list(CAMERA_DIRECTIONS),
            "missing": [],
            "stale": [],
            "timestamps": dict.fromkeys(CAMERA_DIRECTIONS, 123.0),
            "age_s": dict.fromkeys(CAMERA_DIRECTIONS, 0.5),
            "max_age_s": 3.0,
        }
        assert [str(topic) for topic, _ in fake.published] == [
            f"/surround_{direction}_image#sensor_msgs.Image" for direction in CAMERA_DIRECTIONS
        ]
        assert [(image.width, image.height) for _, image in fake.published] == [(4, 3)] * 4
        assert fake.stopped is True
    finally:
        module.stop()
