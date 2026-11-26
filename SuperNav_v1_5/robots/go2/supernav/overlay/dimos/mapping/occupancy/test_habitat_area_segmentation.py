"""Regression oracles for vectorized component and neighbor statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from dimos.mapping.occupancy import habitat_area_segmentation as segmentation


def _reference_stats(
    labels: np.ndarray, mask: np.ndarray, classes: np.ndarray, distance: np.ndarray
) -> dict[int, dict[str, Any]]:
    stats = {}
    height, width = labels.shape
    for label in sorted(int(x) for x in np.unique(labels) if x > 0):
        positions = np.argwhere(labels == label)
        perimeter = occupied = unknown = 0
        neighbors = {}
        for row, col in positions:
            for dr, dc in segmentation._four_offsets():
                nr, nc = row + dr, col + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    perimeter += 1
                elif labels[nr, nc] == label:
                    continue
                elif labels[nr, nc] > 0:
                    neighbor = int(labels[nr, nc])
                    neighbors[neighbor] = neighbors.get(neighbor, 0) + 1
                    perimeter += 1
                elif not mask[nr, nc]:
                    perimeter += 1
                    occupied += int(classes[nr, nc] == segmentation._class_code("occupied"))
                    unknown += int(classes[nr, nc] == segmentation._class_code("unknown"))
        extent = np.ptp(positions, axis=0) + 1
        stats[label] = {
            "area": len(positions),
            "max_distance": float(distance[labels == label].max()),
            "thick_count": int(np.count_nonzero(distance[labels == label] >= 5)),
            "perimeter": perimeter,
            "occupied_boundary": occupied,
            "unknown_boundary": unknown,
            "neighbors": neighbors,
            "max_neighbor_contact": max(neighbors.values(), default=0),
            "compactness": float(len(positions) / np.prod(extent)),
        }
    return stats


def test_vectorized_watershed_statistics_and_neighbor_order_match_reference() -> None:
    rng = np.random.default_rng(42)
    for shape in ((1, 1), (1, 15), (15, 1), (12, 17)):
        for _ in range(10):
            labels = rng.choice([0, 1, 3, 7], size=shape).astype(np.int32)
            mask = (labels > 0) | (rng.random(shape) > 0.5)
            classes = rng.integers(0, 5, size=shape, dtype=np.uint8)
            distance = rng.random(shape) * 10
            expected = _reference_stats(labels, mask, classes, distance)
            actual = segmentation._watershed_segment_stats(labels, mask, classes, distance)
            assert actual == expected
            for label in expected:
                assert list(actual[label]["neighbors"]) == list(expected[label]["neighbors"])


def test_vectorized_frontiers_and_boundaries_match_pixel_reference() -> None:
    rng = np.random.default_rng(19)
    for shape in ((0, 0), (1, 1), (1, 15), (15, 1), (21, 31)):
        classes = rng.integers(0, 5, size=shape, dtype=np.uint8)
        labels = rng.integers(0, 5, size=shape, dtype=np.int32)
        frontier = np.zeros(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        for row, col in np.ndindex(shape):
            for dr, dc in segmentation._four_offsets():
                nr, nc = row + dr, col + dc
                inside = 0 <= nr < shape[0] and 0 <= nc < shape[1]
                if inside and classes[row, col] == 1 and classes[nr, nc] == 2:
                    frontier[row, col] = True
                if labels[row, col] > 0 and (not inside or labels[nr, nc] != labels[row, col]):
                    boundary[row, col] = True
        np.testing.assert_array_equal(segmentation._frontier_mask(classes), frontier)
        np.testing.assert_array_equal(segmentation._area_boundary_mask(labels), boundary)
        display = np.zeros(shape, dtype=bool)
        for row, col in np.argwhere(frontier):
            display[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2] = True
        display &= classes == segmentation._class_code("free")
        np.testing.assert_array_equal(segmentation._frontier_display_mask(classes), display)


def test_components_preserve_eight_connectivity_and_row_major_area_ids() -> None:
    classes = np.full((8, 8), segmentation._class_code("occupied"), dtype=np.uint8)
    classes[0, 0] = classes[1, 1] = classes[5, 5] = segmentation._class_code("free")
    labels, areas = segmentation._segment_free_areas(classes, 1, split_connected_free_space=False)
    assert labels[0, 0] == labels[1, 1] == 1
    assert labels[5, 5] == 2
    assert areas[0]["centroid"] == [0.5, 0.5]
    assert areas[0]["boundary_pixel_count"] == 2
    assert segmentation._count_classes(classes)["free"] == 3


def test_unknown_components_count_frontier_pixels_not_edges() -> None:
    classes = np.full((3, 3), segmentation._class_code("free"), dtype=np.uint8)
    classes[1, 1] = segmentation._class_code("unknown")
    components = segmentation._segment_unknown_components(classes, min_frontier_px=1)
    assert components == [
        {
            "id": 1,
            "pixel_count": 1,
            "bbox": [1, 1, 1, 1],
            "centroid": [1.0, 1.0],
            "frontier_pixel_count": 1,
        }
    ]


def test_empty_and_uniform_maps_have_no_spurious_components() -> None:
    for shape in ((0, 0), (1, 1), (12, 17)):
        for class_name in ("free", "unknown", "occupied"):
            classes = np.full(shape, segmentation._class_code(class_name), dtype=np.uint8)
            labels, areas = segmentation._segment_free_areas(
                classes, 1, split_connected_free_space=True
            )
            assert labels.shape == shape
            assert len(areas) == int(class_name == "free" and classes.size > 0)
            assert segmentation._segment_unknown_components(classes, min_frontier_px=1) == []


def test_small_free_components_are_filtered_without_changing_area_ids() -> None:
    classes = np.full((10, 10), segmentation._class_code("occupied"), dtype=np.uint8)
    classes[0, 0] = segmentation._class_code("free")
    classes[3:5, 3:5] = classes[7:9, 7:9] = segmentation._class_code("free")
    labels, areas = segmentation._segment_free_areas(classes, 2, split_connected_free_space=False)
    assert labels[0, 0] == 0
    assert labels[3, 3] == 1
    assert labels[7, 7] == 2
    assert [area["pixel_count"] for area in areas] == [4, 4]
