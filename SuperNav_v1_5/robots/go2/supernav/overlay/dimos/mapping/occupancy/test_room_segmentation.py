from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image as PILImage

from dimos.mapping.occupancy.room_segmentation import (
    RoomSegmentationConfig,
    render_room_segmentation,
    segment_room_areas,
)
from dimos.mapping.occupancy.habitat_area_segmentation import (
    analyze_habitat_area_segmentation,
    _class_code,
    OCCUPIED,
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


def test_segment_room_areas_finds_rooms_and_keeps_open_boundary() -> None:
    grid = np.full((80, 80), CostValues.UNKNOWN, dtype=np.int8)
    grid[10:70, 10:70] = CostValues.FREE
    grid[10:70, 40] = CostValues.OCCUPIED
    grid[20:22, 55:57] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    segmented = segment_room_areas(
        occupancy_grid,
        config=RoomSegmentationConfig(
            structural_wall_min_length_m=1.0,
            small_room_merge_area_m2=0.1,
            large_room_split_area_m2=999.0,
            output_max_size_px=80,
        ),
    )

    assert segmented.occupancy_grid.grid[0, 0] == CostValues.UNKNOWN
    assert segmented.metadata["room_count"] >= 2
    assert segmented.region_labels[30, 20] != segmented.region_labels[30, 60]
    assert np.max(segmented.region_labels) >= 2


def test_render_room_segmentation_draws_pose_and_colored_rooms() -> None:
    grid = np.full((80, 80), CostValues.UNKNOWN, dtype=np.int8)
    grid[10:70, 10:70] = CostValues.FREE
    grid[10:70, 40] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(
        grid=grid,
        resolution=0.1,
        origin=Pose(),
        frame_id="world",
    )

    rendered = render_room_segmentation(
        occupancy_grid,
        config=RoomSegmentationConfig(
            structural_wall_min_length_m=1.0,
            small_room_merge_area_m2=0.1,
            large_room_split_area_m2=999.0,
            output_max_size_px=80,
        ),
        pose=_pose(3.0, 3.0, yaw_deg=0.0),
        finalized=True,
    )

    image = _image_from_block(rendered.image_block)
    pixels = np.asarray(image)
    assert rendered.metadata["status"] == "finalized"
    assert rendered.metadata["pose_overlay"]["status"] == "current"
    assert np.any(np.all(pixels == (20, 20, 20), axis=2))
    assert not np.any(np.all(pixels[0, :, :] == (20, 20, 20), axis=1))
    assert not np.any(np.all(pixels[-1, :, :] == (20, 20, 20), axis=1))
    assert not np.any(np.all(pixels[:, 0, :] == (20, 20, 20), axis=1))
    assert not np.any(np.all(pixels[:, -1, :] == (20, 20, 20), axis=1))
    assert np.any(np.all(pixels == (245, 245, 245), axis=2))


def test_habitat_area_segmentation_inflates_wall_evidence() -> None:
    grid = np.full((11, 11), CostValues.FREE, dtype=np.int8)
    grid[5, 5] = CostValues.OCCUPIED
    occupancy_grid = OccupancyGrid(grid=grid, resolution=0.1, frame_id="world")

    _, base_class_grid, _ = analyze_habitat_area_segmentation(
        occupancy_grid,
        wall_inflation_m=0.0,
    )
    _, inflated_class_grid, _ = analyze_habitat_area_segmentation(
        occupancy_grid,
        wall_inflation_m=0.15,
    )

    occupied_code = _class_code(OCCUPIED)
    base_occupied = np.count_nonzero(base_class_grid == occupied_code)
    inflated_occupied = np.count_nonzero(inflated_class_grid == occupied_code)
    assert inflated_occupied > base_occupied
