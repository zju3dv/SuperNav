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

import numpy as np
from PIL import Image as PILImage

from dimos.mapping.occupancy.floor_plan import (
    ARROW_FILL_RGB,
    FREE_RGB,
    WALL_RGB,
    FloorPlanConfig,
    clean_floor_plan_grid,
    render_clean_floor_plan,
)
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid


def _image_from_block(block: dict[str, object]) -> PILImage.Image:
    data = str(block["data"])
    return PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")


def _pose(x: float, y: float, yaw_deg: float = 0.0) -> PoseStamped:
    return PoseStamped(
        frame_id="world",
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, np.deg2rad(yaw_deg))),
    )


def test_clean_floor_plan_removes_small_clutter_and_keeps_walls() -> None:
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[:, 5] = CostValues.OCCUPIED
    grid[20, 5] = CostValues.FREE
    grid[10:12, 25:27] = CostValues.OCCUPIED
    grid[0:3, 0:3] = CostValues.UNKNOWN
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            closing_radius_m=0.15,
            min_component_area_m2=0.2,
            structural_wall_min_length_m=1.0,
            output_max_size_px=80,
        ),
    )

    assert cleaned.grid[19, 5] == CostValues.OCCUPIED
    assert cleaned.grid[21, 5] == CostValues.OCCUPIED
    assert np.all(cleaned.grid[10:12, 25:27] == CostValues.FREE)
    assert cleaned.grid[0, 0] == CostValues.UNKNOWN
    assert cleaned.frame_id == "world"
    assert cleaned.resolution == 0.1


def test_clean_floor_plan_fills_clutter_holes_and_draws_region_boundary() -> None:
    grid = np.full((30, 30), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:25, 5:25] = CostValues.FREE
    grid[13:16, 13:16] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            free_space_closing_radius_m=0.2,
            max_fill_hole_area_m2=0.2,
            boundary_wall_width_m=0.1,
            structural_wall_min_length_m=10.0,
            output_max_size_px=80,
        ),
    )

    assert cleaned.grid[14, 14] == CostValues.FREE
    assert cleaned.grid[5, 15] == CostValues.OCCUPIED
    assert cleaned.grid[0, 0] == CostValues.UNKNOWN


def test_clean_floor_plan_default_keeps_map_boundary_open() -> None:
    grid = np.full((30, 30), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:25, 5:25] = CostValues.FREE
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            structural_wall_min_length_m=10.0,
            output_max_size_px=80,
        ),
    )

    assert cleaned.grid[5, 15] == CostValues.FREE
    assert cleaned.grid[4, 15] == CostValues.UNKNOWN
    assert cleaned.grid[6, 15] == CostValues.OCCUPIED
    assert cleaned.grid[24, 15] == CostValues.OCCUPIED
    assert cleaned.grid[15, 5] == CostValues.OCCUPIED
    assert cleaned.grid[15, 24] == CostValues.OCCUPIED


def test_petslam_cleanup_opens_occupied_outer_boundary() -> None:
    grid = np.full((30, 30), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:25, 5:25] = CostValues.FREE
    grid[0, :] = CostValues.OCCUPIED
    grid[-1, :] = CostValues.OCCUPIED
    grid[:, 0] = CostValues.OCCUPIED
    grid[:, -1] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            structural_wall_min_length_m=10.0,
            output_max_size_px=80,
        ),
    )

    assert np.all(cleaned.grid[0, :] == CostValues.UNKNOWN)
    assert np.all(cleaned.grid[-1, :] == CostValues.UNKNOWN)
    assert np.all(cleaned.grid[:, 0] == CostValues.UNKNOWN)
    assert np.all(cleaned.grid[:, -1] == CostValues.UNKNOWN)


def test_petslam_cleanup_removes_walls_without_free_neighbor() -> None:
    grid = np.full((40, 40), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:35, 5:35] = CostValues.FREE
    grid[7:33, 20] = CostValues.OCCUPIED
    grid[0:6, 0:6] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            small_occupied_component_area_m2=0.02,
            structural_wall_min_length_m=1.0,
            output_max_size_px=80,
        ),
    )

    assert cleaned.grid[20, 20] == CostValues.OCCUPIED
    assert cleaned.grid[2, 2] == CostValues.UNKNOWN


def test_petslam_cleanup_keeps_structural_divider_while_removing_clutter() -> None:
    grid = np.full((50, 50), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:45, 5:45] = CostValues.FREE
    grid[8:42, 25] = CostValues.OCCUPIED
    grid[18:20, 35:37] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            small_occupied_component_area_m2=0.08,
            structural_wall_min_length_m=1.0,
            output_max_size_px=80,
        ),
    )

    assert cleaned.grid[25, 25] == CostValues.OCCUPIED
    assert np.all(cleaned.grid[18:20, 35:37] == CostValues.FREE)


def test_structural_line_rendering_is_disabled_by_default() -> None:
    grid = np.full((40, 40), CostValues.UNKNOWN, dtype=np.int8)
    grid[5:35, 5:35] = CostValues.FREE
    grid[8:32, 20] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    cleaned = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            include_cleaned_wall_evidence=False,
            structural_wall_min_length_m=1.0,
            output_max_size_px=80,
        ),
    )

    assert not np.any(cleaned.grid == CostValues.OCCUPIED)

    with_lines = clean_floor_plan_grid(
        occupancy_grid,
        config=FloorPlanConfig(
            include_cleaned_wall_evidence=False,
            include_structural_wall_lines=True,
            structural_wall_min_length_m=1.0,
            output_max_size_px=80,
        ),
    )

    assert with_lines.grid[20, 20] == CostValues.OCCUPIED


def test_render_clean_floor_plan_outputs_png_and_metadata() -> None:
    origin = Pose()
    origin.position.x = -1.0
    origin.position.y = -2.0
    grid = np.full((8, 10), CostValues.FREE, dtype=np.int8)
    grid[1, 1] = CostValues.UNKNOWN
    grid[2:7, 4] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.25, origin=origin, frame_id="map")

    rendered = render_clean_floor_plan(
        occupancy_grid,
        config=FloorPlanConfig(
            min_component_area_m2=0.1,
            structural_wall_min_length_m=0.5,
            output_max_size_px=40,
        ),
        finalized=True,
    )

    image = _image_from_block(rendered.image_block)
    assert rendered.image_block["mimeType"] == "image/png"
    assert image.size == (10, 8)
    pixels = {tuple(pixel) for row in np.asarray(image) for pixel in row}
    assert FREE_RGB in pixels
    assert WALL_RGB in pixels
    assert rendered.metadata["status"] == "finalized"
    assert rendered.metadata["source"] == "native_global_map_cleaned"
    assert rendered.metadata["origin_xy"] == [-1.0, -2.0]
    assert rendered.metadata["resolution_m_per_cell"] == 0.25
    assert "free_space_closing_radius_m" in rendered.metadata["cleaning"]
    assert rendered.metadata["cleaning"]["simplify_structural_walls"] is True


def test_render_clean_floor_plan_draws_pose_arrow_without_outer_border() -> None:
    grid = np.full((80, 80), CostValues.FREE, dtype=np.int8)
    grid[0, :] = CostValues.OCCUPIED
    grid[-1, :] = CostValues.OCCUPIED
    grid[:, 0] = CostValues.OCCUPIED
    grid[:, -1] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    rendered = render_clean_floor_plan(
        occupancy_grid,
        config=FloorPlanConfig(
            min_component_area_m2=0.1,
            free_space_closing_radius_m=0.0,
            max_fill_hole_area_m2=0.0,
            structural_wall_min_length_m=10.0,
            output_max_size_px=80,
        ),
        pose=_pose(4.0, 4.0, yaw_deg=0.0),
    )

    image = _image_from_block(rendered.image_block)
    pixels = np.asarray(image)
    assert not np.any(np.all(pixels[0, :, :] == WALL_RGB, axis=1))
    assert not np.any(np.all(pixels[-1, :, :] == WALL_RGB, axis=1))
    assert not np.any(np.all(pixels[:, 0, :] == WALL_RGB, axis=1))
    assert not np.any(np.all(pixels[:, -1, :] == WALL_RGB, axis=1))
    assert np.any(np.all(pixels == ARROW_FILL_RGB, axis=2))
    assert rendered.metadata["pose_overlay"]["status"] == "current"
