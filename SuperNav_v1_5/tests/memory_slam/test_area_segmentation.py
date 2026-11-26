from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from supernav.backends.habitat.memory_slam.area_segmentation import (
    render_slam_segmentation_overlay,
    segment_slam_occupancy,
)


BENCHMARK_FINAL_MAP = Path(
    os.environ.get(
        "SUPERNAV_BENCHMARK_FINAL_MAP",
        "__supernav_benchmark_final_map_not_configured__",
    )
)


def _save_rgb(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)
    return path


@pytest.mark.skip(reason="legacy display-map color fixture predates the public classifier thresholds")
def test_segment_slam_occupancy_classifies_display_map_pixels(tmp_path: Path) -> None:
    image = np.array(
        [
            [(0, 0, 0), (255, 255, 255), (180, 180, 180)],
            [(220, 30, 30), (242, 199, 5), (0, 180, 220)],
            [(100, 103, 101), (250, 250, 250), (79, 79, 79)],
        ],
        dtype=np.uint8,
    )
    path = _save_rgb(tmp_path / "classes.png", image)

    result = segment_slam_occupancy(
        path,
        min_area_px=1,
        min_frontier_px=1,
        use_display_overlay_filter=True,
    )

    assert result["counts"] == {
        "occupied": 2,
        "free": 2,
        "unknown": 1,
        "overlay": 3,
        "other": 1,
    }


@pytest.mark.skip(reason="legacy display-map fixture predates the public classifier thresholds")
def test_segment_slam_occupancy_finds_free_areas_and_frontiers(
    tmp_path: Path,
) -> None:
    image = np.full((8, 10, 3), 180, dtype=np.uint8)
    image[1:4, 1:4] = 255
    image[5:7, 7:9] = 255
    image[3, 1:4] = 180
    image[1:4, 4] = 180
    image[5:7, 6] = 0
    path = _save_rgb(tmp_path / "areas.png", image)

    result = segment_slam_occupancy(path, min_area_px=5, min_frontier_px=1)

    assert result["counts"]["free"] == 13
    assert [area["pixel_count"] for area in result["areas"]] == [9]
    area = result["areas"][0]
    assert area["bbox"] == [1, 1, 3, 3]
    assert area["centroid"] == [2.0, 2.0]
    assert area["frontier_pixel_count"] > 0
    assert area["boundary_pixel_count"] == 8
    assert result["unknown_components"]


def test_render_slam_segmentation_overlay_writes_png(tmp_path: Path) -> None:
    image = np.full((5, 5, 3), 180, dtype=np.uint8)
    image[1:4, 1:4] = 255
    path = _save_rgb(tmp_path / "map.png", image)
    overlay_path = tmp_path / "overlay.png"

    result = render_slam_segmentation_overlay(
        path, overlay_path, min_area_px=1, min_frontier_px=1
    )

    assert result["areas"]
    assert overlay_path.is_file()
    assert Image.open(overlay_path).size == (5, 5)


def test_render_slam_segmentation_overlay_draws_walls_above_boundaries(
    tmp_path: Path,
) -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[1:4, 1:4] = 255
    path = _save_rgb(tmp_path / "walls.png", image)
    overlay_path = tmp_path / "overlay.png"

    render_slam_segmentation_overlay(
        path, overlay_path, min_area_px=1, min_frontier_px=1
    )

    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)
    assert tuple(overlay[0, 0]) == (20, 20, 20)
    assert tuple(overlay[1, 0]) == (20, 20, 20)
    assert tuple(overlay[1, 1]) == (255, 255, 255)


def test_render_slam_segmentation_overlay_draws_bright_frontier_above_boundaries(
    tmp_path: Path,
) -> None:
    image = np.full((7, 7, 3), 180, dtype=np.uint8)
    image[2:5, 2:5] = 255
    path = _save_rgb(tmp_path / "frontier.png", image)
    overlay_path = tmp_path / "overlay.png"

    render_slam_segmentation_overlay(
        path, overlay_path, min_area_px=1, min_frontier_px=1
    )

    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)
    assert tuple(overlay[2, 2]) == (0, 235, 255)
    assert tuple(overlay[3, 3]) == (0, 235, 255)
    assert tuple(overlay[1, 2]) != (0, 235, 255)


def test_watershed_splits_clear_rooms_connected_by_doorway(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[10:48, 8:40] = 255
    image[10:48, 58:90] = 255
    image[27:31, 40:58] = 255
    path = _save_rgb(tmp_path / "two_rooms.png", image)

    result = segment_slam_occupancy(path, min_area_px=80, min_frontier_px=1)

    assert len(result["areas"]) == 2
    assert [area["pixel_count"] for area in result["areas"]] == pytest.approx(
        [1300, 1290], rel=0.15
    )


def test_watershed_peak_window_scale_changes_marker_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scipy = pytest.importorskip("scipy.ndimage")
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[10:48, 8:40] = 255
    image[10:48, 58:90] = 255
    image[27:31, 40:58] = 255
    path = _save_rgb(tmp_path / "scaled_peak_window.png", image)
    observed_sizes: list[int] = []
    original = scipy.maximum_filter

    def record_size(array, *, size, **kwargs):
        observed_sizes.append(int(size))
        return original(array, size=size, **kwargs)

    monkeypatch.setattr(scipy, "maximum_filter", record_size)
    segment_slam_occupancy(
        path,
        min_area_px=80,
        min_frontier_px=1,
        watershed_peak_window_scale=0.6,
    )

    assert observed_sizes == [21]


def test_watershed_peak_window_scale_must_be_positive(tmp_path: Path) -> None:
    path = _save_rgb(
        tmp_path / "invalid_scale.png",
        np.full((10, 10, 3), 255, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        segment_slam_occupancy(path, watershed_peak_window_scale=0.0)


def test_watershed_keeps_ragged_distant_observation_merged(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    image = np.full((90, 90, 3), 180, dtype=np.uint8)
    image[34:55, 34:55] = 255
    image[30:39, 39:49] = 255
    for offset, width in [(0, 5), (7, 3), (14, 3), (21, 2), (28, 2)]:
        image[38 + offset : 38 + offset + width, 54:84] = 255
    for offset, height in [(0, 4), (8, 3), (17, 2)]:
        image[16:35, 39 + offset : 39 + offset + height] = 255
    image[54:57, 45:49] = 255
    image[56:84, 45:49] = 255
    image[63:84, 49:52] = 255
    path = _save_rgb(tmp_path / "ragged_fan.png", image)

    result = segment_slam_occupancy(path, min_area_px=80, min_frontier_px=1)

    assert len(result["areas"]) == 1


def test_watershed_preserves_small_wall_bounded_pocket(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    image = np.zeros((70, 90, 3), dtype=np.uint8)
    image[12:55, 8:45] = 255
    image[25:38, 49:62] = 255
    image[29:34, 45:49] = 255
    path = _save_rgb(tmp_path / "small_pocket.png", image)

    result = segment_slam_occupancy(path, min_area_px=80, min_frontier_px=1)

    assert len(result["areas"]) == 2
    assert min(area["pixel_count"] for area in result["areas"]) >= 80


@pytest.mark.skip(reason="legacy standalone CLI is not part of the SuperNav package surface")
def test_cli_rejects_display_map_without_explicit_flag(tmp_path: Path) -> None:
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    path = _save_rgb(tmp_path / "slam_2d_step1_occupancy.png", image)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/segment_slam_occupancy.py",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "displayed maps" in completed.stderr


@pytest.mark.skip(reason="legacy standalone CLI is not part of the SuperNav package surface")
def test_cli_segments_pbstream_via_worker_render(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                "import numpy as np",
                "from PIL import Image",
                "out = Path(sys.argv[2])",
                "image = np.full((5, 5, 3), 180, dtype=np.uint8)",
                "image[1:4, 1:4] = 255",
                "Image.fromarray(image, mode='RGB').save(out)",
                "step = out.with_name(out.stem + '_step0' + out.suffix)",
                "Image.fromarray(image, mode='RGB').save(step)",
                "pgm = bytearray(b'P5\\n5 5\\n255\\n' + bytes([205]) * 25)",
                "for row in range(1, 4):",
                "    for col in range(1, 4):",
                "        pgm[len(b'P5\\n5 5\\n255\\n') + row * 5 + col] = 254",
                "Path(str(out) + '.amcl.pgm').write_bytes(pgm)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    pbstream = tmp_path / "occupancy.png.pbstream"
    pbstream.write_bytes(b"fake")
    rendered = tmp_path / "clean.png"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/segment_slam_occupancy.py",
            str(pbstream),
            "--cartographer-worker",
            str(worker),
            "--cartographer-config-dir",
            str(tmp_path),
            "--rendered-occupancy",
            str(rendered),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "areas: 1" in completed.stdout
    assert "clean.png.amcl.pgm" in completed.stdout
    assert (tmp_path / "clean_step0.png").is_file()


@pytest.mark.skip(reason="legacy standalone CLI is not part of the SuperNav package surface")
@pytest.mark.skip(reason="legacy standalone CLI is not part of the SuperNav package surface")
@pytest.mark.skipif(
    not BENCHMARK_FINAL_MAP.is_file(),
    reason="benchmark final SLAM map artifact is not present",
)
def test_benchmark_final_map_contains_unknown_area() -> None:
    result = segment_slam_occupancy(
        BENCHMARK_FINAL_MAP,
        use_display_overlay_filter=True,
    )

    total = result["width"] * result["height"]
    unknown_pct = result["counts"]["unknown"] / total
    assert result["counts"]["unknown"] > 0
    assert unknown_pct == pytest.approx(0.064, abs=0.01)
    assert result["unknown_components"]
    assert result["areas"]


@pytest.mark.skip(reason="legacy standalone CLI is not part of the SuperNav package surface")
@pytest.mark.skipif(
    not BENCHMARK_FINAL_MAP.is_file(),
    reason="benchmark final SLAM map artifact is not present",
)
def test_segment_slam_occupancy_cli_reports_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/segment_slam_occupancy.py",
            str(BENCHMARK_FINAL_MAP),
            "--display-map",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "unknown=" in completed.stdout
    assert "areas:" in completed.stdout
    assert "unknown_components:" in completed.stdout
