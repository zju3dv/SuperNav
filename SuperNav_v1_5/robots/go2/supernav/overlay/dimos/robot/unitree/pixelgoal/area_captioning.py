"""Best-effort asynchronous SLAM-area captioning client."""

from __future__ import annotations

import json
import base64
import math
import os
from pathlib import Path
import queue
import threading
import time
import urllib.request
import uuid
from typing import Any

import cv2
import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import get_run_log_dir

_VIEW_ORDER = ("front", "right", "back", "left")
_VIEW_YAW_OFFSETS_RAD = {
    "front": 0.0,
    "right": -math.pi / 2.0,
    "back": math.pi,
    "left": math.pi / 2.0,
}

DEFAULT_CAPTION_SERVER_URL = ""


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


class Go2AreaCaptionWorker:
    """Queue sampled four-way keyframes for a Habitat-compatible server."""

    def __init__(self, root: Path, backend_url: str, *, timeout_s: float = 60.0) -> None:
        self.root = Path(root)
        self.backend_url = backend_url.rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=16)
        self._closed = False
        self._processing = False
        self._processing_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="go2-area-caption", daemon=True)
        self._thread.start()

    @classmethod
    def from_environment(cls, root: Path) -> "Go2AreaCaptionWorker | None":
        url = os.environ.get("GO2_SLAM_AREA_CAPTION_URL", DEFAULT_CAPTION_SERVER_URL).strip()
        if not url:
            return None
        try:
            timeout = float(os.environ.get("GO2_SLAM_AREA_CAPTION_TIMEOUT_S", "60"))
        except ValueError:
            timeout = 60.0
        return cls(root, url, timeout_s=timeout)

    @property
    def keyframes_path(self) -> Path:
        return self.root / "go2-slam-area-keyframes.jsonl"

    @property
    def jobs_path(self) -> Path:
        return self.root / "go2-slam-area-caption-jobs.jsonl"

    @property
    def results_path(self) -> Path:
        return self.root / "go2-slam-area-caption-results.jsonl"

    def submit(self, images: dict[str, Image], pose: PoseStamped, step_count: int) -> str | None:
        keyframe_id = f"go2-kf-{uuid.uuid4().hex[:12]}"
        encoded: dict[str, bytes] = {}
        for direction, image in images.items():
            try:
                rgb = np.asarray(image.to_rgb().data)
                ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if ok:
                    encoded[direction] = buffer.tobytes()
            except Exception:  # noqa: BLE001 - one bad camera must not affect motion
                continue
        if not encoded:
            return None
        item = {
            "keyframe_id": keyframe_id,
            "step_count": int(step_count),
            "pose": {
                "x": float(pose.x),
                "y": float(pose.y),
                "yaw_rad": float(pose.yaw),
                "sequence": int(pose.ts * 1000),
            },
            # The caption service runs on a different host. Keep the local
            # path for auditability, but send bytes so the service does not
            # need access to Jetson's filesystem.
            "images": {
                direction: {
                    "image_path": str(self.root / "go2-caption-keyframes" / keyframe_id / f"{direction}.jpg"),
                    "data": data,
                }
                for direction, data in encoded.items()
            },
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            _append_jsonl(self.jobs_path, {"keyframe_id": keyframe_id, "status": "rejected", "reason": "queue_full"})
            return None
        return keyframe_id

    def enrich_nodes(
        self,
        nodes: list[dict[str, Any]],
        occupancy_grid: OccupancyGrid,
        area_grid: np.ndarray,
    ) -> list[dict[str, Any]]:
        keyframes = self._read_jsonl(self.keyframes_path)
        results = {
            str(row.get("job_id")): row
            for row in self._read_jsonl(self.results_path)
            if row.get("job_id")
        }
        for node in nodes:
            area_id = int(node.get("area_id", -1))
            center = node.get("centroid_grid") or [0.0, 0.0]
            candidates: list[tuple[float, int, dict[str, Any]]] = []
            for keyframe in keyframes:
                pose = keyframe.get("pose")
                if not isinstance(pose, dict):
                    continue
                try:
                    point = occupancy_grid.world_to_grid((float(pose["x"]), float(pose["y"]), 0.0))
                    # area_grid uses native OccupancyGrid indexing (row 0 is
                    # the grid origin). The renderer flips rows only when
                    # producing the human-facing map image.
                    row = int(round(float(point.y)))
                    col = int(round(float(point.x)))
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0 <= row < area_grid.shape[0] and 0 <= col < area_grid.shape[1]):
                    continue
                if int(area_grid[row, col]) != area_id:
                    continue
                distance = math.hypot(row - float(center[0]), col - float(center[1]))
                candidates.append((distance, -int(keyframe.get("step_count", 0)), keyframe))
            candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("keyframe_id"))))
            selected = candidates[0][2] if candidates else None
            selection: dict[str, Any] | None = None
            expected_views = list(_VIEW_ORDER)
            if selected:
                selection = {
                    "strategy": "inside_area",
                    "keyframe_id": selected["keyframe_id"],
                }
            else:
                selection = self._ray_cast_selection(
                    area_id=area_id,
                    center=center,
                    keyframes=keyframes,
                    occupancy_grid=occupancy_grid,
                    area_grid=area_grid,
                )
                if selection is not None:
                    selected = next(
                        keyframe
                        for keyframe in keyframes
                        if keyframe.get("keyframe_id") == selection["keyframe_id"]
                    )
                    expected_views = [str(selection["view"])]
            captions: list[dict[str, Any]] = []
            if selected:
                for direction in expected_views:
                    row = results.get(f"{selected['keyframe_id']}:{direction}")
                    if isinstance(row, dict) and isinstance(row.get("caption"), str):
                        captions.append({"keyframe_id": selected["keyframe_id"], "view": direction, "caption": row["caption"]})
            node["captions"] = captions
            node["caption_status"] = "ready" if captions else ("pending" if selected else "unavailable")
            node["caption_selection"] = selection or {"reason": "no_keyframe_reaches_area"}
        return nodes

    @staticmethod
    def _ray_cast_selection(
        *,
        area_id: int,
        center: list[Any],
        keyframes: list[dict[str, Any]],
        occupancy_grid: OccupancyGrid,
        area_grid: np.ndarray,
    ) -> dict[str, Any] | None:
        """Select the best view whose free-space ray reaches an area."""
        try:
            center_row, center_col = float(center[0]), float(center[1])
        except (IndexError, TypeError, ValueError):
            return None
        ray_candidates: list[tuple[int, float, int, str, str, dict[str, Any]]] = []
        max_steps = int(math.ceil(math.hypot(*area_grid.shape)))
        for keyframe in keyframes:
            pose = keyframe.get("pose")
            if not isinstance(pose, dict):
                continue
            try:
                point = occupancy_grid.world_to_grid(
                    (float(pose["x"]), float(pose["y"]), 0.0)
                )
                start = (int(round(float(point.y))), int(round(float(point.x))))
                yaw = float(pose["yaw_rad"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (
                0 <= start[0] < area_grid.shape[0]
                and 0 <= start[1] < area_grid.shape[1]
            ):
                continue
            for view, offset in _VIEW_YAW_OFFSETS_RAD.items():
                ray_count = 0
                heading = yaw + offset
                for spread in np.linspace(-math.pi / 4.0, math.pi / 4.0, num=9):
                    touched = False
                    ray_heading = heading + float(spread)
                    # Native Go2 grid rows increase with world Y. The map
                    # renderer flips rows only when producing the image.
                    ray_row_delta = math.sin(ray_heading)
                    ray_col_delta = math.cos(ray_heading)
                    for step in range(1, max_steps + 1):
                        row = int(round(start[0] + ray_row_delta * step))
                        col = int(round(start[1] + ray_col_delta * step))
                        if not (0 <= row < area_grid.shape[0] and 0 <= col < area_grid.shape[1]):
                            break
                        if int(occupancy_grid.grid[row, col]) >= CostValues.OCCUPIED:
                            break
                        if int(area_grid[row, col]) == area_id:
                            touched = True
                            break
                    if touched:
                        ray_count += 1
                if ray_count:
                    distance_m = math.hypot(start[0] - center_row, start[1] - center_col) * occupancy_grid.resolution
                    ray_candidates.append(
                        (
                            ray_count,
                            -distance_m,
                            int(keyframe.get("step_count", 0)),
                            str(keyframe.get("keyframe_id")),
                            view,
                            keyframe,
                        )
                    )
        if not ray_candidates:
            return None
        ray_count, negative_distance, _step, keyframe_id, view, keyframe = max(ray_candidates)
        return {
            "strategy": "ray_cast",
            "keyframe_id": keyframe_id,
            "view": view,
            "ray_count": int(ray_count),
            "distance_to_center_m": round(-negative_distance, 3),
        }

    def has_pending(self) -> bool:
        """Whether a sampled keyframe is queued or being sent to the server."""

        with self._processing_lock:
            processing = self._processing
        return processing or not self._queue.empty()

    def close(self) -> None:
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            with self._processing_lock:
                self._processing = True
            keyframe_id = str(item["keyframe_id"])
            image_paths: dict[str, str] = {}
            image_data: dict[str, bytes] = {}
            for direction, image in item["images"].items():
                data = image.get("data") if isinstance(image, dict) else image
                if not isinstance(data, bytes):
                    continue
                path = self.root / "go2-caption-keyframes" / keyframe_id / f"{direction}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    path.write_bytes(data)
                except OSError as exc:
                    _append_jsonl(self.jobs_path, {"keyframe_id": keyframe_id, "status": "error", "error": str(exc)})
                    continue
                image_paths[direction] = str(path)
                image_data[direction] = data
            keyframe = {"keyframe_id": keyframe_id, "step_count": item["step_count"], "pose": item["pose"], "images": image_paths}
            _append_jsonl(self.keyframes_path, keyframe)
            jobs = []
            request_jobs = []
            for direction, path in image_paths.items():
                job_id = f"{keyframe_id}:{direction}"
                job = {"job_id": job_id, "keyframe_id": keyframe_id, "session_id": "go2", "view": direction, "image_path": str(path), "image_data": base64.b64encode(image_data[direction]).decode("ascii"), "step_count": item["step_count"], "cartographer_sequence": item["pose"]["sequence"]}
                request_jobs.append(
                    {key: value for key, value in job.items() if key != "image_path"}
                )
                jobs.append({key: value for key, value in job.items() if key != "image_data"})
                _append_jsonl(self.jobs_path, {**jobs[-1], "status": "accepted"})
            if not jobs:
                with self._processing_lock:
                    self._processing = False
                continue
            try:
                request = urllib.request.Request(
                    self._batch_url(),
                    data=json.dumps({"items": request_jobs}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    document = json.loads(response.read().decode("utf-8"))
                rows = document.get("results", []) if isinstance(document, dict) else []
                by_id = {str(row.get("job_id")): row for row in rows if isinstance(row, dict)}
                for job in jobs:
                    row = by_id.get(job["job_id"], document if len(jobs) == 1 else {})
                    result_job = {key: value for key, value in job.items() if key != "image_path"}
                    if isinstance(row, dict) and isinstance(row.get("caption"), str) and row["caption"].strip():
                        _append_jsonl(self.results_path, {**result_job, "status": "ready", "caption": row["caption"].strip(), "completed_at_s": time.time()})
                    else:
                        _append_jsonl(self.results_path, {**result_job, "status": "error", "error": "missing caption"})
            except Exception as exc:  # noqa: BLE001 - captioning is best effort
                for job in jobs:
                    result_job = {key: value for key, value in job.items() if key != "image_path"}
                    _append_jsonl(self.results_path, {**result_job, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with self._processing_lock:
                    self._processing = False

    def _batch_url(self) -> str:
        if self.backend_url.endswith("/caption"):
            return self.backend_url[:-8] + "/caption_batch"
        return self.backend_url + "/caption_batch"

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
