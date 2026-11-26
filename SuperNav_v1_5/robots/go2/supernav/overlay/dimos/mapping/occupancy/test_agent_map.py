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

import base64
from io import BytesIO
import math

import numpy as np
from PIL import Image as PILImage

from dimos.mapping.occupancy.agent_map import (
    ARROW_FILL_RGB,
    FREE_RGB,
    OCCUPIED_RGB,
    TRACE_RGB,
    UNKNOWN_RGB,
    render_agent_occupancy_map,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid


def _pose(x: float, y: float, yaw_deg: float = 0.0) -> PoseStamped:
    return PoseStamped(
        frame_id="world",
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, math.radians(yaw_deg))),
    )


def _image_from_block(block: dict[str, object]) -> PILImage.Image:
    data = str(block["data"])
    return PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")


def test_render_agent_occupancy_map_uses_habitat_style_colors_and_overlays() -> None:
    grid = np.full((20, 20), -1, dtype=np.int8)
    grid[7:14, 7:14] = 0
    grid[12, 8] = 50
    grid[12, 12] = 100
    occupancy_grid = OccupancyGrid(grid=grid, resolution=1.0, frame_id="world")

    rendered = render_agent_occupancy_map(
        occupancy_grid,
        _pose(10.0, 10.0, yaw_deg=0.0),
        [_pose(8.0, 8.0), _pose(10.0, 10.0)],
        output_size_px=80,
        view_size_m=10.0,
    )

    image = _image_from_block(rendered.image_block)
    assert rendered.image_block["mimeType"] == "image/png"
    assert image.size == (80, 80)
    assert image.getpixel((12, 68)) == UNKNOWN_RGB
    assert image.getpixel((20, 36)) == FREE_RGB
    assert image.getpixel((28, 20)) == (127, 127, 127)
    assert image.getpixel((60, 20)) == OCCUPIED_RGB
    assert image.getpixel((28, 52)) == TRACE_RGB
    assert image.getpixel((60, 36)) == ARROW_FILL_RGB
    assert rendered.metadata["source"] == "global_costmap"
    assert rendered.metadata["trace_points"] == 2


def test_render_agent_occupancy_map_keeps_robot_visible_near_grid_boundary() -> None:
    occupancy_grid = OccupancyGrid(
        grid=np.zeros((12, 12), dtype=np.int8),
        resolution=1.0,
        frame_id="world",
    )

    rendered = render_agent_occupancy_map(
        occupancy_grid,
        _pose(1.0, 1.0, yaw_deg=90.0),
        output_size_px=60,
        view_size_m=10.0,
    )

    crop = rendered.metadata["crop_grid_bounds"]
    assert crop == {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
    image = _image_from_block(rendered.image_block)
    red_pixels = sum(
        1
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) == ARROW_FILL_RGB
    )
    assert red_pixels > 0


def test_render_agent_occupancy_map_draws_anchored_graph_node_badges() -> None:
    occupancy_grid = OccupancyGrid(
        grid=np.zeros((20, 20), dtype=np.int8),
        resolution=1.0,
        frame_id="world",
    )

    rendered = render_agent_occupancy_map(
        occupancy_grid,
        _pose(10.0, 10.0),
        graph_nodes=[
            {
                "id": "hall",
                "order_id": 1,
                "name": "hall",
                "kind": "area",
                "has_pano_images": True,
                "slam_pose": {"sequence": 1, "x": 9.0, "y": 9.0, "yaw_rad": 0.0},
            },
            {
                "id": "unanchored",
                "order_id": 2,
                "name": "unanchored",
                "kind": "area",
                "has_pano_images": True,
                "slam_pose": None,
            },
        ],
        output_size_px=80,
        view_size_m=10.0,
    )

    assert rendered.metadata["graph_overlay"] is True
    assert rendered.metadata["graph_nodes"] == [
        {"order_id": 1, "id": "hall", "name": "hall", "kind": "area"}
    ]
    image = _image_from_block(rendered.image_block)
    black_pixels = sum(
        1
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) == OCCUPIED_RGB
    )
    assert black_pixels > 0


def test_render_agent_occupancy_map_displaces_overlapping_graph_badges() -> None:
    occupancy_grid = OccupancyGrid(
        grid=np.zeros((30, 30), dtype=np.int8),
        resolution=1.0,
        frame_id="world",
    )

    rendered = render_agent_occupancy_map(
        occupancy_grid,
        _pose(15.0, 15.0),
        graph_nodes=[
            {
                "id": "a",
                "order_id": 1,
                "name": "a",
                "kind": "area",
                "has_pano_images": True,
                "slam_pose": {"sequence": 1, "x": 15.0, "y": 15.0, "yaw_rad": 0.0},
            },
            {
                "id": "b",
                "order_id": 2,
                "name": "b",
                "kind": "door",
                "has_pano_images": True,
                "slam_pose": {"sequence": 2, "x": 15.0, "y": 15.0, "yaw_rad": 0.0},
            },
        ],
        output_size_px=120,
        view_size_m=12.0,
    )

    assert [node["order_id"] for node in rendered.metadata["graph_nodes"]] == [1, 2]
    image = _image_from_block(rendered.image_block)
    white_pixels = sum(
        1
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) == FREE_RGB
    )
    assert white_pixels < image.width * image.height
