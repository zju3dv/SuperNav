import json

import numpy as np

from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.robot.unitree.pixelgoal.area_captioning import Go2AreaCaptionWorker


def test_enrich_nodes_uses_native_grid_row_for_keyframe_pose(tmp_path):
    worker = Go2AreaCaptionWorker(tmp_path, "http://127.0.0.1:1")
    try:
        keyframe_id = "go2-kf-test"
        (tmp_path / "go2-slam-area-keyframes.jsonl").write_text(
            json.dumps(
                {
                    "keyframe_id": keyframe_id,
                    "step_count": 1,
                    "pose": {"x": 3.1, "y": 2.1},
                }
            )
            + "\n"
        )
        (tmp_path / "go2-slam-area-caption-results.jsonl").write_text(
            json.dumps(
                {
                    "job_id": f"{keyframe_id}:front",
                    "caption": "a test area",
                    "status": "ready",
                }
            )
            + "\n"
        )
        area_grid = np.zeros((10, 10), dtype=np.int32)
        area_grid[2, 3] = 1
        nodes = [{"area_id": 1, "centroid_grid": [2.0, 3.0]}]

        result = worker.enrich_nodes(
            nodes,
            OccupancyGrid(grid=np.zeros((10, 10), dtype=np.int8)),
            area_grid,
        )

        assert result[0]["caption_status"] == "ready"
        assert result[0]["caption_selection"] == {
            "strategy": "inside_area",
            "keyframe_id": keyframe_id,
        }
    finally:
        worker.close()


def test_enrich_nodes_uses_ray_cast_when_no_keyframe_is_inside(tmp_path):
    worker = Go2AreaCaptionWorker(tmp_path, "http://127.0.0.1:1")
    try:
        keyframe_id = "go2-kf-ray"
        (tmp_path / "go2-slam-area-keyframes.jsonl").write_text(
            json.dumps(
                {
                    "keyframe_id": keyframe_id,
                    "step_count": 1,
                    "pose": {"x": 1.0, "y": 2.0, "yaw_rad": 0.0},
                }
            )
            + "\n"
        )
        (tmp_path / "go2-slam-area-caption-results.jsonl").write_text(
            json.dumps(
                {
                    "job_id": f"{keyframe_id}:front",
                    "caption": "a ray-reached area",
                    "status": "ready",
                }
            )
            + "\n"
        )
        area_grid = np.zeros((10, 10), dtype=np.int32)
        area_grid[2, 5] = 1
        nodes = [{"area_id": 1, "centroid_grid": [2.0, 5.0]}]

        result = worker.enrich_nodes(
            nodes,
            OccupancyGrid(grid=np.zeros((10, 10), dtype=np.int8)),
            area_grid,
        )

        assert result[0]["caption_status"] == "ready"
        assert result[0]["caption_selection"]["strategy"] == "ray_cast"
        assert result[0]["caption_selection"]["view"] == "front"
    finally:
        worker.close()
