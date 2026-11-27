from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ClassName = str

OCCUPIED: ClassName = "occupied"
FREE: ClassName = "free"
UNKNOWN: ClassName = "unknown"
OVERLAY: ClassName = "overlay"
OTHER: ClassName = "other"

_CLASS_ORDER = (OCCUPIED, FREE, UNKNOWN, OVERLAY, OTHER)


def segment_slam_occupancy(
    image_path: str | Path,
    *,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
    use_display_overlay_filter: bool = False,
    split_connected_free_space: bool = True,
    watershed_peak_window_scale: float = 1.0,
) -> dict[str, Any]:
    """Segment confirmed free-space areas and unknown frontiers in a SLAM map.

    The input may be either a raw Cartographer occupancy PNG or the displayed
    2D SLAM map with pose/path overlays. The returned payload is JSON-safe and
    intentionally omits per-pixel labels; use ``render_slam_segmentation_overlay``
    for a debug image.
    """
    result, _, _ = _analyze_slam_occupancy(
        image_path,
        min_area_px=min_area_px,
        min_frontier_px=min_frontier_px,
        use_display_overlay_filter=use_display_overlay_filter,
        split_connected_free_space=split_connected_free_space,
        watershed_peak_window_scale=watershed_peak_window_scale,
    )
    return result


def analyze_slam_occupancy(
    image_path: str | Path,
    *,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
    use_display_overlay_filter: bool = False,
    split_connected_free_space: bool = True,
    watershed_peak_window_scale: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return the JSON summary plus the class and area grids for a SLAM map."""

    return _analyze_slam_occupancy(
        image_path,
        min_area_px=min_area_px,
        min_frontier_px=min_frontier_px,
        use_display_overlay_filter=use_display_overlay_filter,
        split_connected_free_space=split_connected_free_space,
        watershed_peak_window_scale=watershed_peak_window_scale,
    )


def render_slam_segmentation_overlay(
    image_path: str | Path,
    output_path: str | Path,
    *,
    min_area_px: int = 80,
    min_frontier_px: int = 3,
    use_display_overlay_filter: bool = False,
    split_connected_free_space: bool = True,
    watershed_peak_window_scale: float = 1.0,
) -> dict[str, Any]:
    """Write a colorized segmentation debug image and return the JSON summary."""
    result, class_grid, area_grid = _analyze_slam_occupancy(
        image_path,
        min_area_px=min_area_px,
        min_frontier_px=min_frontier_px,
        use_display_overlay_filter=use_display_overlay_filter,
        split_connected_free_space=split_connected_free_space,
        watershed_peak_window_scale=watershed_peak_window_scale,
    )
    height, width = class_grid.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[class_grid == _class_code(FREE)] = (245, 245, 245)
    overlay[class_grid == _class_code(OCCUPIED)] = (20, 20, 20)
    overlay[class_grid == _class_code(UNKNOWN)] = (175, 185, 195)
    overlay[class_grid == _class_code(OVERLAY)] = (230, 80, 80)
    overlay[class_grid == _class_code(OTHER)] = (130, 130, 130)

    palette = np.array(
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
    area_mask = area_grid > 0
    if np.any(area_mask):
        overlay[area_mask] = palette[(area_grid[area_mask] - 1) % len(palette)]

    boundary_mask = _area_boundary_mask(area_grid)
    overlay[boundary_mask] = (255, 255, 255)
    frontier_mask = _frontier_display_mask(class_grid)
    overlay[frontier_mask] = (0, 235, 255)
    overlay[class_grid == _class_code(OCCUPIED)] = (20, 20, 20)

    Image.fromarray(overlay, mode="RGB").save(output_path)
    return result


def _analyze_slam_occupancy(
    image_path: str | Path,
    *,
    min_area_px: int,
    min_frontier_px: int,
    use_display_overlay_filter: bool,
    split_connected_free_space: bool,
    watershed_peak_window_scale: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if not np.isfinite(watershed_peak_window_scale) or watershed_peak_window_scale <= 0:
        raise ValueError("watershed_peak_window_scale must be finite and positive")
    path = Path(image_path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    class_grid = _classify_rgb(rgb, use_display_overlay_filter)
    counts = _count_classes(class_grid)
    area_grid, areas = _segment_free_areas(
        class_grid,
        min_area_px,
        split_connected_free_space=split_connected_free_space,
        watershed_peak_window_scale=watershed_peak_window_scale,
    )
    unknown_components = _segment_unknown_components(
        class_grid, min_frontier_px=min_frontier_px
    )
    result: dict[str, Any] = {
        "image": str(path),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "counts": counts,
        "areas": areas,
        "unknown_components": unknown_components,
    }
    return result, class_grid, area_grid


def _classify_rgb(rgb: np.ndarray, use_display_overlay_filter: bool) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    max_delta = np.maximum.reduce(
        (np.abs(red - green), np.abs(red - blue), np.abs(green - blue))
    )
    grayish = max_delta <= 3
    gray = ((red + green + blue) // 3).astype(np.int16)

    class_grid = np.full(rgb.shape[:2], _class_code(OTHER), dtype=np.uint8)
    if use_display_overlay_filter:
        red_overlay = (red > 150) & (green < 120) & (blue < 120)
        yellow_overlay = (red > 180) & (green > 140) & (green < 235) & (blue < 120)
        cyan_overlay = (red < 100) & (green > 120) & (blue > 120)
        overlay = red_overlay | yellow_overlay | cyan_overlay
        class_grid[overlay] = _class_code(OVERLAY)
        grayish = grayish & ~overlay

    class_grid[grayish & (gray < 80)] = _class_code(OCCUPIED)
    class_grid[grayish & (gray >= 245)] = _class_code(FREE)
    class_grid[grayish & (gray >= 80) & (gray < 245)] = _class_code(UNKNOWN)
    return class_grid


def _segment_free_areas(
    class_grid: np.ndarray,
    min_area_px: int,
    *,
    split_connected_free_space: bool,
    watershed_peak_window_scale: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = class_grid.shape
    free_code = _class_code(FREE)
    visited = np.zeros((height, width), dtype=bool)
    area_grid = np.zeros((height, width), dtype=np.int32)
    areas: list[dict[str, Any]] = []
    next_id = 1

    for row in range(height):
        for col in range(width):
            if visited[row, col] or class_grid[row, col] != free_code:
                continue
            pixels = _flood_fill(class_grid, visited, row, col, free_code, eight=True)
            if len(pixels) < min_area_px:
                continue
            segments = _split_free_component(
                class_grid,
                pixels,
                min_area_px,
                split_connected_free_space=split_connected_free_space,
                watershed_peak_window_scale=watershed_peak_window_scale,
            )
            for segment in segments:
                if len(segment) < min_area_px:
                    continue
                area_id = next_id
                next_id += 1
                for pixel_row, pixel_col in segment:
                    area_grid[pixel_row, pixel_col] = area_id

    for area_id in range(1, next_id):
        pixels = [
            (int(pixel[0]), int(pixel[1]))
            for pixel in np.argwhere(area_grid == area_id)
        ]
        if not pixels:
            continue
        rows = [pixel[0] for pixel in pixels]
        cols = [pixel[1] for pixel in pixels]
        frontier_count = _count_area_frontiers(class_grid, pixels)
        boundary_count = _count_area_boundary(area_grid, pixels, area_id)
        areas.append(
            {
                "id": area_id,
                "pixel_count": len(pixels),
                "bbox": [min(rows), min(cols), max(rows), max(cols)],
                "centroid": [
                    round(float(sum(rows)) / len(rows), 3),
                    round(float(sum(cols)) / len(cols), 3),
                ],
                "frontier_pixel_count": frontier_count,
                "boundary_pixel_count": boundary_count,
            }
        )
    return area_grid, areas


def _split_free_component(
    class_grid: np.ndarray,
    pixels: list[tuple[int, int]],
    min_area_px: int,
    *,
    split_connected_free_space: bool,
    watershed_peak_window_scale: float = 1.0,
) -> list[list[tuple[int, int]]]:
    if not split_connected_free_space or len(pixels) < max(4 * min_area_px, 400):
        return [pixels]

    try:
        from scipy import ndimage as ndi
    except ImportError:
        return [pixels]

    rows = [pixel[0] for pixel in pixels]
    cols = [pixel[1] for pixel in pixels]
    row0 = max(0, min(rows) - 1)
    row1 = min(class_grid.shape[0], max(rows) + 2)
    col0 = max(0, min(cols) - 1)
    col1 = min(class_grid.shape[1], max(cols) + 2)
    mask = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for row, col in pixels:
        mask[row - row0, col - col0] = True

    distance = ndi.distance_transform_edt(mask)
    if float(distance.max()) < 3.0:
        return [pixels]

    smoothed_distance = ndi.gaussian_filter(distance, sigma=1.0)
    smoothed_distance[~mask] = 0.0
    base_peak_window = max(
        15, min(65, int(round(np.sqrt(len(pixels)) / 3)) * 2 + 1)
    )
    peak_window = max(3, int(round(base_peak_window * watershed_peak_window_scale)))
    if peak_window % 2 == 0:
        peak_window += 1
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
    segments: list[list[tuple[int, int]]] = []
    for label in sorted(int(label) for label in np.unique(label_grid) if label > 0):
        segment = [
            (int(row + row0), int(col + col0))
            for row, col in np.argwhere(label_grid == label)
        ]
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
            if neighbor in stats
            and _is_stable_watershed_segment(stats[neighbor], min_area_px)
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
    height, width = label_grid.shape
    for label in (int(value) for value in np.unique(label_grid) if value > 0):
        positions = np.argwhere(label_grid == label)
        rows = positions[:, 0]
        cols = positions[:, 1]
        perimeter = 0
        occupied_boundary = 0
        unknown_boundary = 0
        neighbors: dict[int, int] = {}
        for row, col in ((int(item[0]), int(item[1])) for item in positions):
            for drow, dcol in _four_offsets():
                next_row = row + drow
                next_col = col + dcol
                if not (0 <= next_row < height and 0 <= next_col < width):
                    perimeter += 1
                    continue
                next_label = int(label_grid[next_row, next_col])
                if next_label == label:
                    continue
                if next_label > 0:
                    neighbors[next_label] = neighbors.get(next_label, 0) + 1
                    perimeter += 1
                    continue
                if not mask[next_row, next_col]:
                    perimeter += 1
                    boundary_class = int(class_grid[next_row, next_col])
                    if boundary_class == occupied_code:
                        occupied_boundary += 1
                    elif boundary_class == unknown_code:
                        unknown_boundary += 1
        bbox_area = (int(rows.max()) - int(rows.min()) + 1) * (
            int(cols.max()) - int(cols.min()) + 1
        )
        area = int(len(positions))
        stats[label] = {
            "area": area,
            "max_distance": float(distance[label_grid == label].max()),
            "thick_count": int(np.count_nonzero(distance[label_grid == label] >= 5.0)),
            "perimeter": perimeter,
            "occupied_boundary": occupied_boundary,
            "unknown_boundary": unknown_boundary,
            "neighbors": neighbors,
            "max_neighbor_contact": max(neighbors.values(), default=0),
            "compactness": float(area / bbox_area) if bbox_area else 0.0,
        }
    return stats


def _is_stable_watershed_segment(
    stat: dict[str, Any], min_area_px: int
) -> bool:
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
        and max_neighbor_contact <= max(12, int(round(0.22 * perimeter)))
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
    height, width = class_grid.shape
    unknown_code = _class_code(UNKNOWN)
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, Any]] = []
    component_id = 1
    for row in range(height):
        for col in range(width):
            if visited[row, col] or class_grid[row, col] != unknown_code:
                continue
            pixels = _flood_fill(
                class_grid, visited, row, col, unknown_code, eight=True
            )
            frontier_count = _count_unknown_frontiers(class_grid, pixels)
            if frontier_count < min_frontier_px:
                continue
            rows = [pixel[0] for pixel in pixels]
            cols = [pixel[1] for pixel in pixels]
            components.append(
                {
                    "id": component_id,
                    "pixel_count": len(pixels),
                    "bbox": [min(rows), min(cols), max(rows), max(cols)],
                    "centroid": [
                        round(float(sum(rows)) / len(rows), 3),
                        round(float(sum(cols)) / len(cols), 3),
                    ],
                    "frontier_pixel_count": frontier_count,
                }
            )
            component_id += 1
    components.sort(
        key=lambda item: (-int(item["frontier_pixel_count"]), -int(item["pixel_count"]))
    )
    for idx, component in enumerate(components, start=1):
        component["id"] = idx
    return components


def _flood_fill(
    class_grid: np.ndarray,
    visited: np.ndarray,
    start_row: int,
    start_col: int,
    target_code: int,
    *,
    eight: bool,
) -> list[tuple[int, int]]:
    offsets = _eight_offsets() if eight else _four_offsets()
    height, width = class_grid.shape
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    pixels: list[tuple[int, int]] = []
    while queue:
        row, col = queue.popleft()
        pixels.append((row, col))
        for drow, dcol in offsets:
            next_row = row + drow
            next_col = col + dcol
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if visited[next_row, next_col]:
                continue
            if class_grid[next_row, next_col] != target_code:
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col))
    return pixels


def _count_area_frontiers(
    class_grid: np.ndarray, pixels: list[tuple[int, int]]
) -> int:
    unknown_code = _class_code(UNKNOWN)
    height, width = class_grid.shape
    count = 0
    for row, col in pixels:
        for drow, dcol in _four_offsets():
            next_row = row + drow
            next_col = col + dcol
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and class_grid[next_row, next_col] == unknown_code
            ):
                count += 1
                break
    return count


def _count_unknown_frontiers(
    class_grid: np.ndarray, pixels: list[tuple[int, int]]
) -> int:
    free_code = _class_code(FREE)
    height, width = class_grid.shape
    count = 0
    for row, col in pixels:
        for drow, dcol in _four_offsets():
            next_row = row + drow
            next_col = col + dcol
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and class_grid[next_row, next_col] == free_code
            ):
                count += 1
                break
    return count


def _count_area_boundary(
    area_grid: np.ndarray, pixels: list[tuple[int, int]], area_id: int
) -> int:
    height, width = area_grid.shape
    count = 0
    for row, col in pixels:
        for drow, dcol in _four_offsets():
            next_row = row + drow
            next_col = col + dcol
            if not (0 <= next_row < height and 0 <= next_col < width):
                count += 1
                break
            if area_grid[next_row, next_col] != area_id:
                count += 1
                break
    return count


def _frontier_mask(class_grid: np.ndarray) -> np.ndarray:
    height, width = class_grid.shape
    mask = np.zeros((height, width), dtype=bool)
    free_code = _class_code(FREE)
    unknown_code = _class_code(UNKNOWN)
    for row in range(height):
        for col in range(width):
            if class_grid[row, col] != free_code:
                continue
            for drow, dcol in _four_offsets():
                next_row = row + drow
                next_col = col + dcol
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and class_grid[next_row, next_col] == unknown_code
                ):
                    mask[row, col] = True
                    break
    return mask


def _frontier_display_mask(class_grid: np.ndarray) -> np.ndarray:
    mask = _frontier_mask(class_grid)
    free_mask = class_grid == _class_code(FREE)
    occupied_mask = class_grid == _class_code(OCCUPIED)
    height, width = class_grid.shape
    expanded = mask.copy()
    for row, col in np.argwhere(mask):
        for drow, dcol in _eight_offsets():
            next_row = row + drow
            next_col = col + dcol
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and free_mask[next_row, next_col]
                and not occupied_mask[next_row, next_col]
            ):
                expanded[next_row, next_col] = True
    return expanded & free_mask


def _area_boundary_mask(area_grid: np.ndarray) -> np.ndarray:
    height, width = area_grid.shape
    mask = np.zeros((height, width), dtype=bool)
    for row in range(height):
        for col in range(width):
            area_id = area_grid[row, col]
            if area_id <= 0:
                continue
            for drow, dcol in _four_offsets():
                next_row = row + drow
                next_col = col + dcol
                if not (0 <= next_row < height and 0 <= next_col < width):
                    mask[row, col] = True
                    break
                if area_grid[next_row, next_col] != area_id:
                    mask[row, col] = True
                    break
    return mask


def _count_classes(class_grid: np.ndarray) -> dict[str, int]:
    raw_counts = Counter(int(value) for value in class_grid.ravel())
    return {
        class_name: int(raw_counts.get(_class_code(class_name), 0))
        for class_name in _CLASS_ORDER
    }


def _class_code(class_name: ClassName) -> int:
    return _CLASS_ORDER.index(class_name)


def _four_offsets() -> tuple[tuple[int, int], ...]:
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def _eight_offsets() -> tuple[tuple[int, int], ...]:
    return (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
