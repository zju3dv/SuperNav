"""Room/area segmentation on a cleaned floor-plan occupancy grid."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import colorsys
import math
import time

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from scipy import ndimage

from dimos.mapping.occupancy.floor_plan import (
    ARROW_FILL_RGB,
    ARROW_OUTLINE_RGB,
    FREE_RGB,
    UNKNOWN_RGB,
    WALL_RGB,
    _clear_outer_boundary,
    _draw_pose_overlay,
    _extract_long_wall_lines,
    _remove_small_occupied_components,
    _remove_walls_without_free_neighbor,
    _remove_small_components,
    _fill_small_holes,
    _simplify_wall_components_to_lines,
    _disk_kernel,
    _radius_cells,
    clean_floor_plan_grid,
    FloorPlanConfig,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid


@dataclass(frozen=True)
class RoomSegmentationConfig:
    """Parameters for the Habitat-GS-style room segmentation pass."""

    occupied_threshold: int = 50
    open_outer_boundary: bool = True
    remove_wall_without_free_neighbor: bool = True
    small_occupied_component_area_m2: float = 0.08
    structural_wall_min_length_m: float = 1.2
    structural_wall_width_m: float = 0.05
    structural_wall_closing_radius_m: float = 0.05
    structural_wall_opening_radius_m: float = 0.0
    room_closing_radius_m: float = 0.10
    room_hole_fill_area_m2: float = 0.25
    small_room_merge_area_m2: float = 0.25
    large_room_split_area_m2: float = 24.0
    wall_inflation_m: float = 0.06
    output_max_size_px: int = 768


@dataclass(frozen=True)
class SegmentedRoomPlan:
    occupancy_grid: OccupancyGrid
    region_labels: np.ndarray
    image_block: dict[str, object]
    metadata: dict[str, object]


def render_room_segmentation(
    occupancy_grid: OccupancyGrid,
    *,
    floor_plan_config: FloorPlanConfig | None = None,
    config: RoomSegmentationConfig | None = None,
    finalized: bool = False,
    pose: PoseStamped | None = None,
) -> SegmentedRoomPlan:
    """Render a colorized room segmentation from a cleaned floor-plan grid."""

    floor_cfg = floor_plan_config or FloorPlanConfig()
    seg_cfg = config or RoomSegmentationConfig()
    cleaned = clean_floor_plan_grid(occupancy_grid, config=floor_cfg)
    segmented = segment_room_areas(cleaned, wall_source=occupancy_grid, config=seg_cfg)
    image = _render_segmented_image(segmented.occupancy_grid, segmented.region_labels, seg_cfg)
    pose_overlay = _draw_pose_overlay(image, segmented.occupancy_grid, pose)
    image_block = {
        "type": "image",
        "data": _png_base64(image),
        "mimeType": "image/png",
        "width_px": image.width,
        "height_px": image.height,
    }
    metadata: dict[str, object] = {
        "source": "native_global_map_area_segmented",
        "status": "finalized" if finalized else "current",
        "map_frame": segmented.occupancy_grid.frame_id,
        "grid_ts": segmented.occupancy_grid.ts,
        "resolution_m_per_cell": segmented.occupancy_grid.resolution,
        "origin_xy": [
            segmented.occupancy_grid.origin.position.x,
            segmented.occupancy_grid.origin.position.y,
        ],
        "grid_size_cells": [
            segmented.occupancy_grid.width,
            segmented.occupancy_grid.height,
        ],
        "image_size_px": [image.width, image.height],
        "room_count": int(segmented.metadata["room_count"]),
        "room_areas_m2": segmented.metadata["room_areas_m2"],
        "pose_overlay": pose_overlay,
        "cleaning": asdict(floor_cfg),
        "segmentation": asdict(seg_cfg),
        "created_at": time.time(),
    }
    return SegmentedRoomPlan(
        occupancy_grid=segmented.occupancy_grid,
        region_labels=segmented.region_labels,
        image_block=image_block,
        metadata=metadata,
    )


def segment_room_areas(
    occupancy_grid: OccupancyGrid,
    *,
    wall_source: OccupancyGrid | None = None,
    config: RoomSegmentationConfig | None = None,
) -> SegmentedRoomPlan:
    """Split a cleaned floor-plan map into room-like connected regions."""

    cfg = config or RoomSegmentationConfig()
    if occupancy_grid.grid.size == 0:
        raise ValueError("occupancy grid is empty")
    if occupancy_grid.resolution <= 0.0:
        raise ValueError("occupancy grid resolution must be positive")
    wall_grid = wall_source or occupancy_grid

    grid = np.asarray(occupancy_grid.grid, dtype=np.int8)
    wall_source_grid = np.asarray(wall_grid.grid, dtype=np.int8)
    known_mask = grid >= 0
    free_mask = known_mask & (grid < cfg.occupied_threshold)
    wall_mask = (wall_source_grid >= 0) & (wall_source_grid >= cfg.occupied_threshold)
    unknown_mask = grid < 0

    if cfg.open_outer_boundary:
        free_mask = _clear_outer_boundary(free_mask)
        wall_mask = _clear_outer_boundary(wall_mask)
        unknown_mask = _clear_outer_boundary(unknown_mask)

    if cfg.remove_wall_without_free_neighbor:
        wall_mask = _remove_walls_without_free_neighbor(wall_mask, free_mask)
    wall_mask = _remove_small_occupied_components(
        wall_mask,
        protected_mask=np.zeros_like(wall_mask, dtype=bool),
        max_remove_cells=int(
            round(cfg.small_occupied_component_area_m2 / occupancy_grid.resolution**2)
        ),
        free_mask=free_mask,
    )

    divider_seed = _extract_long_wall_lines(
        wall_mask,
        resolution_m=occupancy_grid.resolution,
        min_length_m=cfg.structural_wall_min_length_m,
        width_m=cfg.structural_wall_width_m,
    )
    if cfg.structural_wall_closing_radius_m > 0.0 and np.any(divider_seed):
        divider_seed = ndimage.binary_closing(
            divider_seed,
            structure=_disk_kernel(_radius_cells(cfg.structural_wall_closing_radius_m, occupancy_grid.resolution)),
        )
    if cfg.structural_wall_opening_radius_m > 0.0 and np.any(divider_seed):
        divider_seed = ndimage.binary_opening(
            divider_seed,
            structure=_disk_kernel(_radius_cells(cfg.structural_wall_opening_radius_m, occupancy_grid.resolution)),
        )
    if np.any(divider_seed):
        divider_seed = _simplify_wall_components_to_lines(
            divider_seed,
            resolution_m=occupancy_grid.resolution,
            min_length_m=cfg.structural_wall_min_length_m,
            width_m=cfg.structural_wall_width_m,
        )

    divider_mask = _extend_dividers_to_room_bounds(
        divider_seed,
        floor_mask=free_mask,
        resolution=occupancy_grid.resolution,
    )
    divider_mask = _trim_dividers_to_known_floor(divider_mask, free_mask)

    room_mask = free_mask & ~divider_mask
    room_closing_cells = _radius_cells(cfg.room_closing_radius_m, occupancy_grid.resolution)
    if room_closing_cells > 0 and np.any(room_mask):
        room_mask = ndimage.binary_closing(
            room_mask,
            structure=_disk_kernel(room_closing_cells),
        )
    room_mask = _remove_small_components(
        room_mask,
        min_cells=max(1, int(round(cfg.small_room_merge_area_m2 / occupancy_grid.resolution**2))),
    )
    hole_fill_cells = int(round(cfg.room_hole_fill_area_m2 / occupancy_grid.resolution**2))
    if hole_fill_cells > 0 and np.any(room_mask):
        room_mask = _fill_small_holes(room_mask, max_cells=hole_fill_cells)
    labels, count = ndimage.label(room_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        labels = np.zeros_like(grid, dtype=np.int32)
        room_areas: list[float] = []
    else:
        labels, room_areas = _merge_and_split_rooms(
            labels,
            room_mask=room_mask,
            divider_mask=divider_mask,
            occupancy_grid=occupancy_grid,
            config=cfg,
        )

    region_occupancy = np.full_like(grid, CostValues.UNKNOWN, dtype=np.int8)
    region_occupancy[free_mask] = CostValues.FREE
    region_occupancy[wall_mask | divider_mask] = CostValues.OCCUPIED
    if cfg.open_outer_boundary:
        region_occupancy = _clear_outer_boundary(region_occupancy)

    room_count = int(np.max(labels)) if labels.size else 0
    if not room_areas and room_count > 0:
        room_areas = [
            float(np.count_nonzero(labels == label) * occupancy_grid.resolution**2)
            for label in range(1, room_count + 1)
        ]
    metadata = {
        "room_count": room_count,
        "room_areas_m2": room_areas,
        "occupied_cells": int(np.count_nonzero(wall_mask | divider_mask)),
        "free_cells": int(np.count_nonzero(free_mask)),
        "unknown_cells": int(np.count_nonzero(region_occupancy < 0)),
    }
    return SegmentedRoomPlan(
        occupancy_grid=OccupancyGrid(
            grid=region_occupancy,
            resolution=occupancy_grid.resolution,
            origin=occupancy_grid.origin,
            frame_id=occupancy_grid.frame_id,
            ts=occupancy_grid.ts,
        ),
        region_labels=labels,
        image_block={},
        metadata=metadata,
    )


def _merge_and_split_rooms(
    labels: np.ndarray,
    *,
    room_mask: np.ndarray,
    divider_mask: np.ndarray,
    occupancy_grid: OccupancyGrid,
    config: RoomSegmentationConfig,
) -> tuple[np.ndarray, list[float]]:
    labels = labels.copy()
    min_room_cells = max(1, int(round(config.small_room_merge_area_m2 / occupancy_grid.resolution**2)))
    room_areas = _component_areas(labels, occupancy_grid.resolution)
    large_labels = {
        label_id
        for label_id, area_m2 in room_areas.items()
        if area_m2 >= config.small_room_merge_area_m2
    }
    if not large_labels and room_areas:
        large_labels = {max(room_areas, key=room_areas.get)}

    for _ in range(3):
        changed = False
        for label_id, area_m2 in sorted(room_areas.items(), key=lambda item: item[1]):
            if area_m2 >= config.small_room_merge_area_m2:
                continue
            component = labels == label_id
            if not np.any(component):
                continue
            neighbors = _neighbor_labels(labels, component)
            neighbors.discard(label_id)
            if neighbors:
                best_neighbor = max(
                    neighbors,
                    key=lambda neighbor: _shared_boundary_count(labels, component, neighbor),
                )
            else:
                best_neighbor = _nearest_large_label(
                    labels,
                    component,
                    preferred=large_labels,
                )
                if best_neighbor is None:
                    continue
            labels[component] = best_neighbor
            changed = True
        if not changed:
            break
        labels = _relabel_consecutive(labels)
        room_areas = _component_areas(labels, occupancy_grid.resolution)
        large_labels = {
            label_id
            for label_id, area_m2 in room_areas.items()
            if area_m2 >= config.small_room_merge_area_m2
        }
        if not large_labels and room_areas:
            large_labels = {max(room_areas, key=room_areas.get)}

    for _ in range(3):
        split_changed = False
        room_areas = _component_areas(labels, occupancy_grid.resolution)
        for label_id, area_m2 in list(room_areas.items()):
            if area_m2 < config.large_room_split_area_m2:
                continue
            component = labels == label_id
            split_line = _find_split_line(component, divider_mask, occupancy_grid.resolution)
            if split_line is None:
                continue
            labels[split_line] = 0
            labels = _relabel_consecutive(
                ndimage.label(labels > 0, structure=np.ones((3, 3), dtype=np.uint8))[0]
            )
            split_changed = True
            break
        if not split_changed:
            break

    room_areas_list = [
        float(room_areas[label_id])
        for label_id in sorted(room_areas)
        if room_areas[label_id] >= min_room_cells * occupancy_grid.resolution**2
    ]
    return labels, room_areas_list


def _component_areas(labels: np.ndarray, resolution: float) -> dict[int, float]:
    areas: dict[int, float] = {}
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        areas[int(label_id)] = float(np.count_nonzero(labels == label_id) * resolution**2)
    return areas


def _neighbor_labels(labels: np.ndarray, component: np.ndarray) -> set[int]:
    dilated = ndimage.binary_dilation(component, structure=np.ones((3, 3), dtype=np.uint8))
    neighbor_values = np.unique(labels[dilated & ~component])
    return {int(value) for value in neighbor_values if int(value) > 0}


def _shared_boundary_count(labels: np.ndarray, component: np.ndarray, neighbor_label: int) -> int:
    neighbor_mask = labels == neighbor_label
    boundary = ndimage.binary_dilation(component, structure=np.ones((3, 3), dtype=np.uint8))
    return int(np.count_nonzero(boundary & neighbor_mask))


def _nearest_large_label(
    labels: np.ndarray,
    component: np.ndarray,
    *,
    preferred: set[int],
) -> int | None:
    if not preferred:
        return None
    component_rows, component_cols = np.where(component)
    if component_rows.size == 0:
        return None
    component_center = np.array([component_rows.mean(), component_cols.mean()], dtype=np.float64)

    best_label: int | None = None
    best_distance = float("inf")
    for label_id in preferred:
        label_mask = labels == label_id
        if not np.any(label_mask):
            continue
        rows, cols = np.where(label_mask)
        if rows.size == 0:
            continue
        center = np.array([rows.mean(), cols.mean()], dtype=np.float64)
        distance = float(np.linalg.norm(center - component_center))
        if distance < best_distance:
            best_distance = distance
            best_label = int(label_id)
    return best_label


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        out[labels == label_id] = next_label
        next_label += 1
    return out


def _find_split_line(
    component: np.ndarray,
    divider_mask: np.ndarray,
    resolution: float,
) -> np.ndarray | None:
    rows, cols = np.where(component)
    if rows.size == 0:
        return None

    row_span = int(rows.max() - rows.min() + 1)
    col_span = int(cols.max() - cols.min() + 1)

    row_counts = np.sum(component, axis=1)
    col_counts = np.sum(component, axis=0)
    row_window = row_counts[rows.min() : rows.max() + 1]
    col_window = col_counts[cols.min() : cols.max() + 1]
    if row_window.size == 0 or col_window.size == 0:
        return None

    cut_row = int(rows.min() + np.argmin(row_window))
    cut_col = int(cols.min() + np.argmin(col_window))
    row_score = float(row_counts[cut_row]) / max(1.0, float(col_span))
    col_score = float(col_counts[cut_col]) / max(1.0, float(row_span))

    if row_score <= col_score and row_score <= 0.9:
        line = np.zeros_like(component, dtype=bool)
        line[cut_row, cols.min() : cols.max() + 1] = component[cut_row, cols.min() : cols.max() + 1]
        return line
    if col_score <= 0.9:
        line = np.zeros_like(component, dtype=bool)
        line[rows.min() : rows.max() + 1, cut_col] = component[rows.min() : rows.max() + 1, cut_col]
        return line
    return None


def _extend_dividers_to_room_bounds(
    divider_seed: np.ndarray,
    *,
    floor_mask: np.ndarray,
    resolution: float,
) -> np.ndarray:
    if not np.any(divider_seed):
        return divider_seed

    labels, count = ndimage.label(divider_seed, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return divider_seed

    extended = np.zeros_like(divider_seed, dtype=bool)
    for label_id in range(1, count + 1):
        component = labels == label_id
        rows, cols = np.where(component)
        if rows.size == 0:
            continue
        row_span = int(rows.max() - rows.min())
        col_span = int(cols.max() - cols.min())
        if col_span >= row_span:
            row = int(round(float(rows.mean())))
            if not (0 <= row < floor_mask.shape[0]):
                continue
            left = int(cols.min())
            right = int(cols.max())
            while left > 0 and floor_mask[row, left - 1]:
                left -= 1
            while right + 1 < floor_mask.shape[1] and floor_mask[row, right + 1]:
                right += 1
            extended[row, left : right + 1] = True
        else:
            col = int(round(float(cols.mean())))
            if not (0 <= col < floor_mask.shape[1]):
                continue
            top = int(rows.min())
            bottom = int(rows.max())
            while top > 0 and floor_mask[top - 1, col]:
                top -= 1
            while bottom + 1 < floor_mask.shape[0] and floor_mask[bottom + 1, col]:
                bottom += 1
            extended[top : bottom + 1, col] = True

    if np.any(extended):
        extended |= ndimage.binary_dilation(extended, structure=_disk_kernel(max(1, _radius_cells(0.05, resolution))))
    return extended


def _trim_dividers_to_known_floor(divider_mask: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
    if not np.any(divider_mask):
        return divider_mask
    return divider_mask & ndimage.binary_dilation(floor_mask, structure=np.ones((3, 3), dtype=np.uint8))


def _render_segmented_image(
    occupancy_grid: OccupancyGrid,
    labels: np.ndarray,
    config: RoomSegmentationConfig,
) -> PILImage.Image:
    grid = occupancy_grid.grid
    rgb = np.empty((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    rgb[grid < 0] = UNKNOWN_RGB
    rgb[grid == CostValues.OCCUPIED] = WALL_RGB
    rgb[grid == CostValues.FREE] = FREE_RGB

    max_label = int(np.max(labels)) if labels.size else 0
    if max_label > 0:
        for label_id in range(1, max_label + 1):
            color = _label_color(label_id, max_label)
            rgb[labels == label_id] = color

    image = PILImage.fromarray(rgb[::-1], mode="RGB")
    scale = min(
        config.output_max_size_px / max(1, image.width),
        config.output_max_size_px / max(1, image.height),
        1.0,
    )
    if scale < 1.0:
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        resampling = getattr(PILImage, "Resampling", PILImage)
        image = image.resize((width, height), resample=resampling.NEAREST)
    return image


def _label_color(label_id: int, max_label: int) -> tuple[int, int, int]:
    hue = (label_id * 0.6180339887498949) % 1.0
    sat = 0.22 + 0.06 * ((label_id % 3) / 2.0)
    val = 0.98
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def _png_base64(image: PILImage.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


from dimos.mapping.occupancy.habitat_area_segmentation import (
    analyze_habitat_area_segmentation,
    render_habitat_area_segmentation,
)


def segment_room_areas(
    occupancy_grid: OccupancyGrid,
    *,
    config: RoomSegmentationConfig | None = None,
) -> SegmentedRoomPlan:
    cfg = config or RoomSegmentationConfig()
    result, class_grid, area_grid = analyze_habitat_area_segmentation(
        occupancy_grid,
        min_area_px=max(1, int(getattr(cfg, "min_area_px", 80))),
        min_frontier_px=max(1, int(getattr(cfg, "min_frontier_px", 3))),
        split_connected_free_space=bool(getattr(cfg, "split_connected_free_space", True)),
        wall_inflation_m=max(0.0, float(cfg.wall_inflation_m)),
    )
    _ = class_grid  # keep parity with the exact segmentation analysis contract
    return SegmentedRoomPlan(
        occupancy_grid=occupancy_grid,
        region_labels=area_grid,
        image_block={},
        metadata=result,
    )


def render_room_segmentation(
    occupancy_grid: OccupancyGrid,
    *,
    floor_plan_config: FloorPlanConfig | None = None,
    config: RoomSegmentationConfig | None = None,
    finalized: bool = False,
    pose: PoseStamped | None = None,
) -> SegmentedRoomPlan:
    cfg = config or RoomSegmentationConfig()
    output_max_size_px = max(1, int(getattr(cfg, "output_max_size_px", 768)))
    result, class_grid, area_grid, image = render_habitat_area_segmentation(
        occupancy_grid,
        min_area_px=max(1, int(getattr(cfg, "min_area_px", 80))),
        min_frontier_px=max(1, int(getattr(cfg, "min_frontier_px", 3))),
        split_connected_free_space=bool(getattr(cfg, "split_connected_free_space", True)),
        wall_inflation_m=max(0.0, float(cfg.wall_inflation_m)),
        output_max_size_px=output_max_size_px,
        pose=pose,
    )
    image_block = {
        "type": "image",
        "data": _png_base64(image),
        "mimeType": "image/png",
        "width_px": image.width,
        "height_px": image.height,
    }
    metadata = {
        "source": "native_global_map_area_segmented",
        "status": "finalized" if finalized else "current",
        "map_frame": occupancy_grid.frame_id,
        "grid_ts": occupancy_grid.ts,
        "resolution_m_per_cell": occupancy_grid.resolution,
        "origin_xy": [
            occupancy_grid.origin.position.x,
            occupancy_grid.origin.position.y,
        ],
        "grid_size_cells": [occupancy_grid.width, occupancy_grid.height],
        "image_size_px": [image.width, image.height],
        "room_count": len(result["areas"]),
        "room_areas_m2": [
            float(area["pixel_count"]) * float(occupancy_grid.resolution) ** 2
            for area in result["areas"]
        ],
        "counts": result["counts"],
        "unknown_components": result["unknown_components"],
        "wall_inflation_m": float(cfg.wall_inflation_m),
        "pose_overlay": result.get("pose_overlay", {"status": "unavailable"}),
        "created_at": time.time(),
    }
    return SegmentedRoomPlan(
        occupancy_grid=occupancy_grid,
        region_labels=area_grid,
        image_block=image_block,
        metadata=metadata,
    )
