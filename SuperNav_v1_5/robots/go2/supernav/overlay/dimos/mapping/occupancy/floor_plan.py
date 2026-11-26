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

"""Clean floor-plan rendering from noisy occupancy/cost maps."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import math
import time

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from scipy import ndimage

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid

UNKNOWN_RGB = (225, 225, 225)
FREE_RGB = (255, 255, 255)
WALL_RGB = (0, 0, 0)
ARROW_FILL_RGB = (230, 0, 0)
ARROW_OUTLINE_RGB = (255, 230, 0)


@dataclass(frozen=True)
class FloorPlanConfig:
    occupied_threshold: int = 50
    petslam_cleanup_enabled: bool = True
    open_outer_boundary: bool = True
    remove_wall_without_free_neighbor: bool = True
    small_occupied_component_area_m2: float = 0.08
    include_cleaned_wall_evidence: bool = True
    include_structural_wall_lines: bool = False
    closing_radius_m: float = 0.05
    opening_radius_m: float = 0.0
    min_component_area_m2: float = 0.35
    free_space_closing_radius_m: float = 0.25
    max_fill_hole_area_m2: float = 2.5
    boundary_wall_width_m: float = 0.0
    structural_wall_min_length_m: float = 4.0
    structural_wall_width_m: float = 0.05
    simplify_structural_walls: bool = True
    output_max_size_px: int = 768


@dataclass(frozen=True)
class RenderedFloorPlan:
    occupancy_grid: OccupancyGrid
    image_block: dict[str, object]
    metadata: dict[str, object]


def render_clean_floor_plan(
    occupancy_grid: OccupancyGrid,
    *,
    config: FloorPlanConfig | None = None,
    finalized: bool = False,
    pose: PoseStamped | None = None,
) -> RenderedFloorPlan:
    """Render a cleaned structural floor plan from an occupancy/cost grid."""

    cfg = config or FloorPlanConfig()
    cleaned = clean_floor_plan_grid(occupancy_grid, config=cfg)
    image = _floor_plan_image(cleaned, cfg.output_max_size_px)
    pose_overlay = _draw_pose_overlay(image, cleaned, pose)
    image_block = {
        "type": "image",
        "data": _png_base64(image),
        "mimeType": "image/png",
        "width_px": image.width,
        "height_px": image.height,
    }
    metadata: dict[str, object] = {
        "source": "native_global_map_cleaned",
        "status": "finalized" if finalized else "current",
        "map_frame": cleaned.frame_id,
        "grid_ts": cleaned.ts,
        "resolution_m_per_cell": cleaned.resolution,
        "origin_xy": [
            cleaned.origin.position.x,
            cleaned.origin.position.y,
        ],
        "grid_size_cells": [cleaned.width, cleaned.height],
        "image_size_px": [image.width, image.height],
        "wall_cells": cleaned.occupied_cells,
        "free_cells": cleaned.free_cells,
        "unknown_cells": cleaned.unknown_cells,
        "cleaning": asdict(cfg),
        "pose_overlay": pose_overlay,
        "created_at": time.time(),
    }
    return RenderedFloorPlan(
        occupancy_grid=cleaned,
        image_block=image_block,
        metadata=metadata,
    )


def clean_floor_plan_grid(
    occupancy_grid: OccupancyGrid,
    *,
    config: FloorPlanConfig | None = None,
) -> OccupancyGrid:
    """Return a binary structural floor-plan OccupancyGrid.

    The cleaner favors an agent-facing floor plan over a raw obstacle map:
    traversable regions are smoothed, small interior obstacle holes are filled,
    and walls are rendered as thin region boundaries plus very long structural
    line evidence. This removes most furniture-sized lidar clutter while keeping
    stable room outlines.
    """

    cfg = config or FloorPlanConfig()
    if occupancy_grid.grid.size == 0:
        raise ValueError("occupancy grid is empty")
    if occupancy_grid.resolution <= 0.0:
        raise ValueError("occupancy grid resolution must be positive")

    grid = occupancy_grid.grid
    known_mask = grid >= 0
    free_seed = known_mask & (grid < cfg.occupied_threshold)
    wall_seed = known_mask & (grid >= cfg.occupied_threshold)
    if cfg.open_outer_boundary:
        free_seed = _clear_outer_boundary(free_seed)
        wall_seed = _clear_outer_boundary(wall_seed)

    if cfg.petslam_cleanup_enabled and cfg.remove_wall_without_free_neighbor:
        wall_seed = _remove_walls_without_free_neighbor(wall_seed, free_seed)
    if cfg.petslam_cleanup_enabled:
        wall_seed = _remove_small_occupied_components(
            wall_seed,
            protected_mask=np.zeros_like(wall_seed, dtype=bool),
            max_remove_cells=int(
                round(cfg.small_occupied_component_area_m2 / occupancy_grid.resolution**2)
            ),
            free_mask=free_seed,
        )

    floor_mask = free_seed
    free_closing_radius_cells = _radius_cells(
        cfg.free_space_closing_radius_m,
        occupancy_grid.resolution,
    )
    if free_closing_radius_cells > 0 and np.any(floor_mask):
        floor_mask = ndimage.binary_closing(
            floor_mask,
            structure=_disk_kernel(free_closing_radius_cells),
        )
    floor_mask = _remove_small_components(
        floor_mask,
        min_cells=max(1, int(round(cfg.min_component_area_m2 / occupancy_grid.resolution**2))),
    )
    max_fill_hole_cells = int(round(cfg.max_fill_hole_area_m2 / occupancy_grid.resolution**2))
    if max_fill_hole_cells > 0 and np.any(floor_mask):
        floor_mask = _fill_small_holes(floor_mask, max_cells=max_fill_hole_cells)

    boundary_wall_width_cells = _radius_cells(
        cfg.boundary_wall_width_m,
        occupancy_grid.resolution,
    )
    wall_mask = np.zeros_like(floor_mask, dtype=bool)
    if boundary_wall_width_cells > 0 and np.any(floor_mask):
        wall_mask |= ndimage.binary_dilation(
            floor_mask,
            structure=_disk_kernel(boundary_wall_width_cells),
        ) & ~ndimage.binary_erosion(
            floor_mask,
            structure=_disk_kernel(max(1, boundary_wall_width_cells // 2)),
        )

    closing_radius_cells = _radius_cells(cfg.closing_radius_m, occupancy_grid.resolution)
    structural_wall_mask = wall_seed
    if closing_radius_cells > 0 and np.any(structural_wall_mask):
        structural_wall_mask = ndimage.binary_closing(
            structural_wall_mask,
            structure=_disk_kernel(closing_radius_cells),
        )

    opening_radius_cells = _radius_cells(cfg.opening_radius_m, occupancy_grid.resolution)
    if opening_radius_cells > 0 and np.any(structural_wall_mask):
        structural_wall_mask = ndimage.binary_opening(
            structural_wall_mask,
            structure=_disk_kernel(opening_radius_cells),
        )

    structural_wall_mask = _extract_long_wall_lines(
        structural_wall_mask,
        resolution_m=occupancy_grid.resolution,
        min_length_m=cfg.structural_wall_min_length_m,
        width_m=cfg.structural_wall_width_m,
    )
    if cfg.simplify_structural_walls:
        structural_wall_mask = _simplify_wall_components_to_lines(
            structural_wall_mask,
            resolution_m=occupancy_grid.resolution,
            min_length_m=cfg.structural_wall_min_length_m,
            width_m=cfg.structural_wall_width_m,
        )
    if np.any(structural_wall_mask):
        structural_wall_mask &= ndimage.binary_dilation(
            floor_mask,
            structure=_disk_kernel(max(1, boundary_wall_width_cells + 1)),
        )
    if cfg.include_cleaned_wall_evidence:
        wall_mask |= wall_seed & ndimage.binary_dilation(
            floor_mask,
            structure=_disk_kernel(max(1, boundary_wall_width_cells + 1)),
        )
    if cfg.include_structural_wall_lines:
        wall_mask |= structural_wall_mask
    if cfg.open_outer_boundary:
        wall_mask = _clear_outer_boundary(wall_mask)

    clean_grid = np.full_like(grid, CostValues.UNKNOWN, dtype=np.int8)
    clean_grid[floor_mask] = CostValues.FREE
    clean_grid[wall_mask] = CostValues.OCCUPIED

    return OccupancyGrid(
        grid=clean_grid,
        resolution=occupancy_grid.resolution,
        origin=occupancy_grid.origin,
        frame_id=occupancy_grid.frame_id,
        ts=occupancy_grid.ts,
    )


def _clear_outer_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    opened = mask.copy()
    opened[0, :] = False
    opened[-1, :] = False
    opened[:, 0] = False
    opened[:, -1] = False
    return opened


def _remove_walls_without_free_neighbor(
    wall_mask: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(wall_mask):
        return wall_mask
    free_neighbors = ndimage.binary_dilation(
        free_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    return wall_mask & free_neighbors


def _remove_small_occupied_components(
    wall_mask: np.ndarray,
    *,
    protected_mask: np.ndarray,
    max_remove_cells: int,
    free_mask: np.ndarray,
) -> np.ndarray:
    if max_remove_cells <= 0 or not np.any(wall_mask):
        return wall_mask

    labels, count = ndimage.label(wall_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return wall_mask

    cleaned = wall_mask.copy()
    free_neighbors = ndimage.binary_dilation(
        free_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel())
    for label in range(1, count + 1):
        component = labels == label
        if np.any(component & protected_mask):
            continue
        size = int(sizes[label])
        if size <= max_remove_cells:
            cleaned[component] = False
            continue
        if size <= max_remove_cells * 2 and np.any(component & ~free_neighbors):
            cleaned[component] = False

    return cleaned


def _fill_small_holes(mask: np.ndarray, *, max_cells: int) -> np.ndarray:
    if max_cells <= 0 or not np.any(mask):
        return mask
    labels, count = ndimage.label(~mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_label_set = {int(label) for label in border_labels}
    sizes = np.bincount(labels.ravel())
    fill = np.zeros(count + 1, dtype=bool)
    for label in range(1, count + 1):
        if label not in border_label_set and sizes[label] <= max_cells:
            fill[label] = True

    filled = mask.copy()
    filled[fill[labels]] = True
    return filled


def _remove_small_components(mask: np.ndarray, *, min_cells: int) -> np.ndarray:
    if not np.any(mask):
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_cells
    keep[0] = False
    return keep[labels]


def _extract_long_wall_lines(
    mask: np.ndarray,
    *,
    resolution_m: float,
    min_length_m: float,
    width_m: float,
) -> np.ndarray:
    min_length_cells = _radius_cells(min_length_m, resolution_m)
    if min_length_cells <= 1 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    width_cells = max(1, _radius_cells(width_m, resolution_m))
    line_mask = np.zeros_like(mask, dtype=bool)
    for angle_deg in range(0, 180, 15):
        line_mask |= ndimage.binary_opening(
            mask,
            structure=_line_kernel(min_length_cells, angle_deg, width_cells),
        )

    if not np.any(line_mask):
        return line_mask

    line_mask = ndimage.binary_dilation(
        line_mask,
        structure=_disk_kernel(max(1, width_cells)),
    )
    return _remove_small_components(
        line_mask,
        min_cells=max(1, int(round((min_length_m * max(width_m, resolution_m)) / resolution_m**2))),
    )


def _simplify_wall_components_to_lines(
    mask: np.ndarray,
    *,
    resolution_m: float,
    min_length_m: float,
    width_m: float,
) -> np.ndarray:
    if not np.any(mask):
        return mask

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask

    min_length_cells = _radius_cells(min_length_m, resolution_m)
    width_cells = max(1, _radius_cells(width_m, resolution_m))
    simplified = np.zeros_like(mask, dtype=bool)
    for label in range(1, count + 1):
        rows, cols = np.where(labels == label)
        if rows.size < 2:
            continue

        coords = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
        center = coords.mean(axis=0)
        centered = coords - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        projections = centered @ direction
        start = center + direction * projections.min()
        end = center + direction * projections.max()
        length_cells = float(np.linalg.norm(end - start))
        if length_cells < min_length_cells:
            continue

        _draw_line_segment(simplified, start=start, end=end, width_cells=width_cells)

    return simplified


def _draw_line_segment(
    mask: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    width_cells: int,
) -> None:
    length_cells = max(1, int(round(float(np.linalg.norm(end - start)))))
    for alpha in np.linspace(0.0, 1.0, length_cells + 1):
        point = start * (1.0 - alpha) + end * alpha
        col = int(round(point[0]))
        row = int(round(point[1]))
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            mask[
                max(0, row - width_cells) : min(mask.shape[0], row + width_cells + 1),
                max(0, col - width_cells) : min(mask.shape[1], col + width_cells + 1),
            ] = True


def _line_kernel(length_cells: int, angle_deg: int, width_cells: int) -> np.ndarray:
    length_cells = max(1, int(length_cells))
    width_cells = max(1, int(width_cells))
    size = length_cells + 2 * width_cells + 3
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    for offset in np.linspace(
        -(length_cells - 1) / 2,
        (length_cells - 1) / 2,
        max(1, length_cells * 3),
    ):
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= y < size and 0 <= x < size:
            half_width = width_cells // 2
            kernel[
                max(0, y - half_width) : min(size, y + half_width + 1),
                max(0, x - half_width) : min(size, x + half_width + 1),
            ] = 1
    return kernel


def _floor_plan_image(occupancy_grid: OccupancyGrid, max_size_px: int) -> PILImage.Image:
    if max_size_px <= 0:
        raise ValueError("output_max_size_px must be positive")
    grid = occupancy_grid.grid
    rgb = np.empty((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    rgb[grid < 0] = UNKNOWN_RGB
    rgb[grid == CostValues.FREE] = FREE_RGB
    rgb[grid >= CostValues.OCCUPIED] = WALL_RGB

    image = PILImage.fromarray(rgb[::-1], mode="RGB")
    scale = min(max_size_px / max(1, image.width), max_size_px / max(1, image.height), 1.0)
    if scale < 1.0:
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        resampling = getattr(PILImage, "Resampling", PILImage)
        image = image.resize((width, height), resample=resampling.NEAREST)
    return image


def _draw_pose_overlay(
    image: PILImage.Image,
    occupancy_grid: OccupancyGrid,
    pose: PoseStamped | None,
) -> dict[str, object]:
    if pose is None:
        return {"status": "unavailable"}

    point = occupancy_grid.world_to_grid(pose.position)
    grid_x = float(point.x)
    grid_y = float(point.y)
    if not (0.0 <= grid_x < occupancy_grid.width and 0.0 <= grid_y < occupancy_grid.height):
        return {
            "status": "pose_outside_map",
            "pose_frame": pose.frame_id,
        }

    scale_x = image.width / max(1, occupancy_grid.width)
    scale_y = image.height / max(1, occupancy_grid.height)
    center = (
        (grid_x + 0.5) * scale_x,
        (occupancy_grid.height - grid_y - 0.5) * scale_y,
    )
    draw = ImageDraw.Draw(image)
    _draw_facing_arrow(draw, center, pose.yaw, max(image.width, image.height))
    return {
        "status": "current",
        "pose_frame": pose.frame_id,
        "pose_xy_yaw": [pose.x, pose.y, pose.yaw],
        "pixel_xy": [center[0], center[1]],
    }


def _draw_facing_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    yaw_rad: float,
    image_size_px: int,
) -> None:
    length = max(14.0, image_size_px * 0.085)
    half_width = max(5.0, image_size_px * 0.018)
    tail = max(4.0, image_size_px * 0.014)
    dx = math.cos(yaw_rad)
    dy = -math.sin(yaw_rad)
    px = -dy
    py = dx
    cx, cy = center
    tip = (cx + dx * length, cy + dy * length)
    base = (cx - dx * tail, cy - dy * tail)
    polygon = [
        tip,
        (base[0] + px * half_width, base[1] + py * half_width),
        (base[0] - px * half_width, base[1] - py * half_width),
    ]
    outline_width = max(2, image_size_px // 256)
    draw.polygon(polygon, fill=ARROW_FILL_RGB, outline=ARROW_OUTLINE_RGB)
    draw.line([polygon[-1], tip, polygon[1]], fill=ARROW_OUTLINE_RGB, width=outline_width)


def _disk_kernel(radius_cells: int) -> np.ndarray:
    y, x = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    return (x * x + y * y <= radius_cells * radius_cells).astype(np.uint8)


def _radius_cells(radius_m: float, resolution_m: float) -> int:
    if radius_m <= 0.0:
        return 0
    return max(1, int(round(radius_m / resolution_m)))


def _png_base64(image: PILImage.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
