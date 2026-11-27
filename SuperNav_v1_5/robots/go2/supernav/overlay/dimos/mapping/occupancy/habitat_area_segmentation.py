"""Habitat-GS-style segmentation of occupancy costmaps into free-space areas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from dimos.mapping.occupancy.floor_plan import _disk_kernel, _draw_pose_overlay, _radius_cells
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid

OCCUPIED = "occupied"
FREE = "free"
UNKNOWN = "unknown"
OVERLAY = "overlay"
OTHER = "other"

_CLASS_ORDER = (OCCUPIED, FREE, UNKNOWN, OVERLAY, OTHER)
_PALETTE = np.array(
    [
        (46, 134, 193),
        (39, 174, 96),
        (155, 89, 182),
        (230, 126, 34),
        (22, 160, 133),
        (192, 57, 43),
        (127, 140, 141),
        (241, 196, 15),
    ],
    dtype=np.uint8,
)


def analyze_habitat_area_segmentation(
    occupancy_grid: OccupancyGrid,
    *,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
    split_connected_free_space: bool = True,
    wall_inflation_m: float = 0.06,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    class_grid = _classify_costmap(occupancy_grid, wall_inflation_m=wall_inflation_m)
    counts = _count_classes(class_grid)
    area_grid, areas = _segment_free_areas(
        class_grid,
        min_area_px,
        split_connected_free_space=split_connected_free_space,
    )
    unknown_components = _segment_unknown_components(
        class_grid,
        min_frontier_px=min_frontier_px,
    )
    result: dict[str, Any] = {
        "width": int(occupancy_grid.width),
        "height": int(occupancy_grid.height),
        "counts": counts,
        "areas": areas,
        "unknown_components": unknown_components,
        "wall_inflation_m": float(wall_inflation_m),
    }
    return result, class_grid, area_grid


def build_slam_area_nodes(
    occupancy_grid: OccupancyGrid,
    area_grid: np.ndarray,
    *,
    min_frontier_pixels: int = 3,
) -> list[dict[str, Any]]:
    """Build ephemeral, planner-facing ``A<N>`` targets for one map snapshot.

    IDs are deliberately regenerated from the current segmentation and are not
    durable graph-memory identifiers.
    """
    nodes: list[dict[str, Any]] = []
    for area_id in sorted(int(value) for value in np.unique(area_grid) if value > 0):
        pixels = np.argwhere(area_grid == area_id)
        if len(pixels) == 0:
            continue
        rows, cols = pixels[:, 0], pixels[:, 1]
        # Segmentation rows are image-space (top-origin); OccupancyGrid uses
        # map-space coordinates, so use the grid's own conversion helpers.
        # Use the geometric bbox center, then snap to the nearest free pixel.
        # This gives the agent a stable center target without aiming at a wall
        # or an occupied cell.
        bbox_center_row = (float(rows.min()) + float(rows.max())) / 2.0
        bbox_center_col = (float(cols.min()) + float(cols.max())) / 2.0
        nearest = int(
            np.argmin(
                (rows.astype(np.float64) - bbox_center_row) ** 2
                + (cols.astype(np.float64) - bbox_center_col) ** 2
            )
        )
        target_row = float(rows[nearest])
        target_col = float(cols[nearest])
        world = occupancy_grid.grid_to_world((target_col, target_row, 0.0))
        target = {
            "x": float(world.x),
            "y": float(world.y),
            "yaw_rad": 0.0,
            "sequence": int(occupancy_grid.ts * 1000) if occupancy_grid.ts else 0,
        }
        nodes.append(
            {
                "id": f"slam_area_{area_id}",
                "label": f"A{area_id}",
                "area_id": area_id,
                "pixel_count": len(pixels),
                "centroid_grid": [bbox_center_row, bbox_center_col],
                "target_grid": [target_row, target_col],
                "bbox_grid": [
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()) + 1,
                    int(cols.max()) + 1,
                ],
                "target_slam_pose": target,
                "map_frame": occupancy_grid.frame_id,
                "grid_ts": occupancy_grid.ts,
            }
        )
    return nodes


def render_habitat_area_segmentation(
    occupancy_grid: OccupancyGrid,
    output_path: str | Path | None = None,
    *,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
    split_connected_free_space: bool = True,
    wall_inflation_m: float = 0.06,
    output_max_size_px: int = 768,
    pose: PoseStamped | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Image.Image]:
    result, class_grid, area_grid = analyze_habitat_area_segmentation(
        occupancy_grid,
        min_area_px=min_area_px,
        min_frontier_px=min_frontier_px,
        split_connected_free_space=split_connected_free_space,
        wall_inflation_m=wall_inflation_m,
    )
    image = _render_segmentation_overlay(
        class_grid,
        area_grid,
        output_max_size_px=output_max_size_px,
    )
    pose_overlay = _draw_pose_overlay(image, occupancy_grid, pose)
    result = {
        **result,
        "pose_overlay": pose_overlay,
    }
    if output_path is not None:
        image.save(output_path)
    return result, class_grid, area_grid, image


def _classify_costmap(
    occupancy_grid: OccupancyGrid,
    *,
    wall_inflation_m: float = 0.06,
) -> np.ndarray:
    grid = np.asarray(occupancy_grid.grid, dtype=np.int16)
    class_grid = np.full(grid.shape, _class_code(OTHER), dtype=np.uint8)
    class_grid[grid < 0] = _class_code(UNKNOWN)
    occupied_mask = grid >= CostValues.OCCUPIED
    inflation_cells = _radius_cells(wall_inflation_m, occupancy_grid.resolution)
    if inflation_cells > 0 and np.any(occupied_mask):
        occupied_mask = ndi.binary_dilation(
            occupied_mask,
            structure=_disk_kernel(inflation_cells),
        )
        occupied_mask &= grid >= 0
    class_grid[(grid >= 0) & (grid < CostValues.OCCUPIED)] = _class_code(FREE)
    class_grid[occupied_mask] = _class_code(OCCUPIED)
    return class_grid


def _render_segmentation_overlay(
    class_grid: np.ndarray,
    area_grid: np.ndarray,
    *,
    output_max_size_px: int,
) -> Image.Image:
    height, width = class_grid.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[class_grid == _class_code(FREE)] = (245, 245, 245)
    overlay[class_grid == _class_code(OCCUPIED)] = (20, 20, 20)
    overlay[class_grid == _class_code(UNKNOWN)] = (175, 185, 195)
    overlay[class_grid == _class_code(OVERLAY)] = (230, 80, 80)
    overlay[class_grid == _class_code(OTHER)] = (130, 130, 130)

    area_mask = area_grid > 0
    if np.any(area_mask):
        overlay[area_mask] = _PALETTE[(area_grid[area_mask] - 1) % len(_PALETTE)]

    boundary_mask = _area_boundary_mask(area_grid)
    overlay[boundary_mask] = (255, 255, 255)
    frontier_mask = _frontier_display_mask(class_grid)
    overlay[frontier_mask] = (0, 235, 255)
    overlay[class_grid == _class_code(OCCUPIED)] = (20, 20, 20)

    image = Image.fromarray(overlay[::-1], mode="RGB")
    scale = min(
        output_max_size_px / max(1, image.width),
        output_max_size_px / max(1, image.height),
        1.0,
    )
    if scale < 1.0:
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((width, height), resample=resampling.NEAREST)
    return image


def _segment_free_areas(
    class_grid: np.ndarray,
    min_area_px: int,
    *,
    split_connected_free_space: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = class_grid.shape
    free_code = _class_code(FREE)
    area_grid = np.zeros((height, width), dtype=np.int32)
    areas: list[dict[str, Any]] = []
    next_id = 1

    components, _ = ndi.label(class_grid == free_code, structure=np.ones((3, 3), dtype=bool))
    component_sizes = np.bincount(components.ravel())
    component_boxes = ndi.find_objects(components) if components.size else []
    for component_id, bounds in enumerate(component_boxes, start=1):
        if bounds is None or component_sizes[component_id] < min_area_px:
            continue
        local = np.argwhere(components[bounds] == component_id)
        local += np.array([bounds[0].start, bounds[1].start])
        segments = _split_free_component(
            class_grid,
            local,
            min_area_px,
            split_connected_free_space=split_connected_free_space,
        )
        for segment in segments:
            if len(segment) < min_area_px:
                continue
            positions = np.asarray(segment)
            area_grid[positions[:, 0], positions[:, 1]] = next_id
            next_id += 1

    counts, row_sums, col_sums, bounds = _label_geometry(area_grid)
    frontier_counts = np.bincount(area_grid[_frontier_mask(class_grid)], minlength=next_id)
    boundary_counts = np.bincount(area_grid[_area_boundary_mask(area_grid)], minlength=next_id)
    for area_id, box in enumerate(bounds, start=1):
        if box is None:
            continue
        count = int(counts[area_id])
        areas.append(
            {
                "id": area_id,
                "pixel_count": count,
                "bbox": [box[0].start, box[1].start, box[0].stop - 1, box[1].stop - 1],
                "centroid": [
                    round(float(row_sums[area_id]) / count, 3),
                    round(float(col_sums[area_id]) / count, 3),
                ],
                "frontier_pixel_count": int(frontier_counts[area_id]),
                "boundary_pixel_count": int(boundary_counts[area_id]),
            }
        )
    return area_grid, areas


def _split_free_component(
    class_grid: np.ndarray,
    pixels: np.ndarray,
    min_area_px: int,
    *,
    split_connected_free_space: bool,
) -> list[np.ndarray]:
    if not split_connected_free_space or len(pixels) < max(4 * min_area_px, 400):
        return [pixels]

    positions = np.asarray(pixels)
    rows, cols = positions[:, 0], positions[:, 1]
    row0 = max(0, int(rows.min()) - 1)
    row1 = min(class_grid.shape[0], int(rows.max()) + 2)
    col0 = max(0, int(cols.min()) - 1)
    col1 = min(class_grid.shape[1], int(cols.max()) + 2)
    mask = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    mask[rows - row0, cols - col0] = True

    distance = ndi.distance_transform_edt(mask)
    if float(distance.max()) < 3.0:
        return [pixels]

    smoothed_distance = ndi.gaussian_filter(distance, sigma=1.0)
    smoothed_distance[~mask] = 0.0
    peak_window = max(15, min(65, round(np.sqrt(len(pixels)) / 3) * 2 + 1))
    peaks = (
        mask
        & (smoothed_distance == ndi.maximum_filter(smoothed_distance, size=peak_window))
        & (distance >= 3.0)
    )
    seed_grid, seed_count = ndi.label(peaks, structure=np.ones((3, 3), dtype=bool))
    if seed_count <= 1:
        return [pixels]

    label_grid = _grow_watershed_labels(mask, smoothed_distance, seed_grid)
    label_grid = _merge_weak_watershed_segments(
        label_grid,
        mask,
        class_grid[row0:row1, col0:col1],
        distance,
        min_area_px,
    )
    segments: list[np.ndarray] = []
    for label in sorted(int(label) for label in np.unique(label_grid) if label > 0):
        segment = np.argwhere(label_grid == label) + np.array([row0, col0])
        if len(segment) >= min_area_px:
            segments.append(segment)
    return segments or [pixels]


def _merge_weak_watershed_segments(
    label_grid: np.ndarray,
    mask: np.ndarray,
    class_grid: np.ndarray,
    distance: np.ndarray,
    min_area_px: int,
) -> np.ndarray:
    labels = label_grid.copy()
    while True:
        stats = _watershed_segment_stats(labels, mask, class_grid, distance)
        if len(stats) <= 1:
            return labels

        weak_labels = [
            label
            for label, stat in stats.items()
            if not _is_stable_watershed_segment(stat, min_area_px)
        ]
        if not weak_labels:
            return labels
        if len(weak_labels) == len(stats):
            return np.where(mask, 1, 0).astype(np.int32)

        label = min(
            weak_labels,
            key=lambda item: (
                float(stats[item]["max_distance"]),
                int(stats[item]["area"]),
            ),
        )
        neighbors = stats[label]["neighbors"]
        if not neighbors:
            labels[labels == label] = 0
            continue
        stable_neighbors = {
            neighbor: contact
            for neighbor, contact in neighbors.items()
            if neighbor in stats and _is_stable_watershed_segment(stats[neighbor], min_area_px)
        }
        candidates = stable_neighbors or neighbors
        target = max(
            candidates,
            key=lambda item: (int(candidates[item]), int(stats[item]["area"])),
        )
        labels[labels == label] = target


def _watershed_segment_stats(
    label_grid: np.ndarray,
    mask: np.ndarray,
    class_grid: np.ndarray,
    distance: np.ndarray,
) -> dict[int, dict[str, Any]]:
    occupied_code = _class_code(OCCUPIED)
    unknown_code = _class_code(UNKNOWN)
    stats: dict[int, dict[str, Any]] = {}
    counts, _, _, boxes = _label_geometry(label_grid)
    size = len(counts)
    adjacent = _neighbor_values(label_grid, fill=-1)
    source = np.broadcast_to(label_grid[..., None], adjacent.shape)
    active = source > 0
    contacts = active & (adjacent > 0) & (adjacent != source)
    exterior = active & (adjacent != source) & (adjacent <= 0) & ~_neighbor_values(mask)
    perimeter = np.bincount(source[contacts | exterior], minlength=size)
    boundary_classes = _neighbor_values(class_grid, fill=-1)
    occupied = np.bincount(source[exterior & (boundary_classes == occupied_code)], minlength=size)
    unknown = np.bincount(source[exterior & (boundary_classes == unknown_code)], minlength=size)
    thick = np.bincount(label_grid[distance >= 5.0], minlength=size)
    maximum = np.zeros(size, dtype=np.float64)
    np.maximum.at(maximum, label_grid.ravel(), distance.ravel())

    neighbors_by_label: dict[int, dict[int, int]] = {}
    if np.any(contacts):
        # Flatten in row/column/offset order, matching the old loop's insertion
        # order. Equal-contact merge choices must retain the same tie-breaks.
        pairs = source[contacts].astype(np.int64) * size + adjacent[contacts]
        unique, first, contact_counts = np.unique(pairs, return_index=True, return_counts=True)
        for index in np.argsort(first):
            label, neighbor = divmod(int(unique[index]), size)
            neighbors_by_label.setdefault(label, {})[neighbor] = int(contact_counts[index])

    for label, box in enumerate(boxes, start=1):
        if box is None:
            continue
        neighbors = neighbors_by_label.get(label, {})
        bbox_area = (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
        area = int(counts[label])
        stats[label] = {
            "area": area,
            "max_distance": float(maximum[label]),
            "thick_count": int(thick[label]),
            "perimeter": int(perimeter[label]),
            "occupied_boundary": int(occupied[label]),
            "unknown_boundary": int(unknown[label]),
            "neighbors": neighbors,
            "max_neighbor_contact": max(neighbors.values(), default=0),
            "compactness": float(area / bbox_area) if bbox_area else 0.0,
        }
    return stats


def _is_stable_watershed_segment(stat: dict[str, Any], min_area_px: int) -> bool:
    area = int(stat["area"])
    perimeter = max(1, int(stat["perimeter"]))
    occupied_ratio = int(stat["occupied_boundary"]) / perimeter
    unknown_ratio = int(stat["unknown_boundary"]) / perimeter
    max_neighbor_contact = int(stat["max_neighbor_contact"])
    compactness = float(stat["compactness"])

    robust_open_area = (
        area >= max(3 * min_area_px, 220)
        and int(stat["thick_count"]) >= max(20, min_area_px // 4)
        and float(stat["max_distance"]) >= 5.0
    )
    small_enclosed_pocket = (
        area >= min_area_px
        and float(stat["max_distance"]) >= 3.0
        and occupied_ratio >= 0.30
        and unknown_ratio <= 0.25
        and compactness >= 0.25
        and max_neighbor_contact <= max(12, round(0.22 * perimeter))
    )
    return robust_open_area or small_enclosed_pocket


def _grow_watershed_labels(
    mask: np.ndarray, distance: np.ndarray, seed_grid: np.ndarray
) -> np.ndarray:
    import heapq

    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    heap: list[tuple[float, int, int, int]] = []
    for row, col in np.argwhere(seed_grid > 0):
        label = int(seed_grid[row, col])
        labels[row, col] = label
        heapq.heappush(heap, (-float(distance[row, col]), int(row), int(col), label))

    while heap:
        _, row, col, label = heapq.heappop(heap)
        if labels[row, col] != label:
            continue
        for drow, dcol in _four_offsets():
            next_row = row + drow
            next_col = col + dcol
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not mask[next_row, next_col] or labels[next_row, next_col] != 0:
                continue
            labels[next_row, next_col] = label
            heapq.heappush(
                heap,
                (
                    -float(distance[next_row, next_col]),
                    next_row,
                    next_col,
                    label,
                ),
            )
    return labels


def _segment_unknown_components(
    class_grid: np.ndarray, *, min_frontier_px: int
) -> list[dict[str, Any]]:
    unknown_code = _class_code(UNKNOWN)
    components: list[dict[str, Any]] = []
    component_id = 1
    labels, _ = ndi.label(class_grid == unknown_code, structure=np.ones((3, 3), dtype=bool))
    counts, row_sums, col_sums, boxes = _label_geometry(labels)
    frontier = (class_grid == unknown_code) & np.any(
        _neighbor_values(class_grid, fill=-1) == _class_code(FREE), axis=-1
    )
    frontier_counts = np.bincount(labels[frontier], minlength=len(counts))
    for label, box in enumerate(boxes, start=1):
        if box is None or frontier_counts[label] < min_frontier_px:
            continue
        count = int(counts[label])
        components.append(
            {
                "id": component_id,
                "pixel_count": count,
                "bbox": [box[0].start, box[1].start, box[0].stop - 1, box[1].stop - 1],
                "centroid": [
                    round(float(row_sums[label]) / count, 3),
                    round(float(col_sums[label]) / count, 3),
                ],
                "frontier_pixel_count": int(frontier_counts[label]),
            }
        )
        component_id += 1
    components.sort(
        key=lambda item: (-int(item["frontier_pixel_count"]), -int(item["pixel_count"]))
    )
    for idx, component in enumerate(components, start=1):
        component["id"] = idx
    return components


def _frontier_mask(class_grid: np.ndarray) -> np.ndarray:
    return (class_grid == _class_code(FREE)) & np.any(
        _neighbor_values(class_grid, fill=-1) == _class_code(UNKNOWN), axis=-1
    )


def _frontier_display_mask(class_grid: np.ndarray) -> np.ndarray:
    mask = _frontier_mask(class_grid)
    free_mask = class_grid == _class_code(FREE)
    return ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & free_mask


def _area_boundary_mask(area_grid: np.ndarray) -> np.ndarray:
    return (area_grid > 0) & np.any(
        _neighbor_values(area_grid, fill=-1) != area_grid[..., None], axis=-1
    )


def _neighbor_values(values: np.ndarray, fill: int | bool = False) -> np.ndarray:
    """Four neighbors in the original up/down/left/right iteration order."""
    dtype = np.result_type(values.dtype, np.min_scalar_type(fill))
    padded = np.pad(values.astype(dtype, copy=False), 1, constant_values=fill)
    return np.stack(
        (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]),
        axis=-1,
    )


def _label_geometry(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, ...] | None]]:
    """Accumulate area, coordinate sums and bounding boxes without per-pixel Python."""
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=1)
    rows, cols = np.indices(labels.shape)
    row_sums = np.bincount(flat, weights=rows.ravel(), minlength=len(counts))
    col_sums = np.bincount(flat, weights=cols.ravel(), minlength=len(counts))
    boxes = ndi.find_objects(labels) if labels.size else []
    return counts, row_sums, col_sums, boxes


def _count_classes(class_grid: np.ndarray) -> dict[str, int]:
    raw_counts = np.bincount(class_grid.ravel(), minlength=len(_CLASS_ORDER))
    return {class_name: int(raw_counts[_class_code(class_name)]) for class_name in _CLASS_ORDER}


def _class_code(class_name: str) -> int:
    return _CLASS_ORDER.index(class_name)


def _four_offsets() -> tuple[tuple[int, int], ...]:
    return ((-1, 0), (1, 0), (0, -1), (0, 1))
