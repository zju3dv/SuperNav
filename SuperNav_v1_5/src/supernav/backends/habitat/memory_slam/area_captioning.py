from __future__ import annotations

import base64
import json
import math
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .area_segmentation import OCCUPIED, _class_code
from .occupancy_navigation import SlamOccupancyMap, load_slam_occupancy_map


_VIEW_ORDER = ("front", "right", "back", "left")
_VIEW_YAW_OFFSETS_RAD = {
    "front": 0.0,
    "right": -math.pi / 2.0,
    "back": math.pi,
    "left": math.pi / 2.0,
}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


@dataclass(frozen=True)
class SlamAreaCaptionJob:
    job_id: str
    keyframe_id: str
    session_id: str
    view: str
    image_path: str
    heading_deg: float | None
    step_count: int
    cartographer_sequence: int | None


@dataclass(frozen=True)
class SlamAreaCaptionBatch:
    jobs: tuple[SlamAreaCaptionJob, ...]


class SlamAreaCaptionWorker:
    """Session-owned local worker for async key-frame image captioning."""

    def __init__(
        self,
        *,
        session_id: str,
        artifact_dir: Path,
        backend_url: str,
        queue_size: int = 64,
        timeout_s: float = 60.0,
        transport: str = "inline",
        check_health: bool = True,
        logger: Any = None,
    ) -> None:
        self.session_id = session_id
        self.artifact_dir = artifact_dir
        self.backend_url = backend_url.rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        if transport not in {"inline", "path"}:
            raise ValueError("caption transport must be inline or path")
        self.transport = transport
        self._logger = logger
        if check_health:
            self._require_healthy_backend()
        self._queue: queue.Queue[SlamAreaCaptionBatch | None] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"slam-area-caption-{session_id}",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_environment(
        cls,
        *,
        session_id: str,
        artifact_dir: Path,
        logger: Any = None,
    ) -> "SlamAreaCaptionWorker | None":
        backend_url = os.environ.get("HAB_SLAM_AREA_CAPTION_URL", "").strip()
        if not backend_url:
            return None
        try:
            queue_size = int(os.environ.get("HAB_SLAM_AREA_CAPTION_QUEUE_SIZE", "64"))
        except ValueError:
            queue_size = 64
        try:
            timeout_s = float(os.environ.get("HAB_SLAM_AREA_CAPTION_TIMEOUT_S", "60"))
        except ValueError:
            timeout_s = 60.0
        return cls(
            session_id=session_id,
            artifact_dir=artifact_dir,
            backend_url=backend_url,
            queue_size=queue_size,
            timeout_s=timeout_s,
            transport=os.environ.get(
                "HAB_SLAM_AREA_CAPTION_TRANSPORT", "inline"
            ).strip().lower(),
            logger=logger,
        )

    def _require_healthy_backend(self) -> None:
        health_url = self.backend_url.rstrip("/")
        if health_url.endswith("/caption"):
            health_url = health_url[: -len("/caption")]
        request = urllib.request.Request(health_url + "/healthz", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                document = json.loads(response.read().decode("utf-8") or "{}")
        except Exception as exc:
            raise RuntimeError(f"caption backend is unavailable: {exc}") from exc
        if not isinstance(document, Mapping) or document.get("ok") is not True:
            raise RuntimeError("caption backend health check did not report ok=true")

    @property
    def keyframes_path(self) -> Path:
        return self.artifact_dir / "slam_area_keyframes.jsonl"

    @property
    def jobs_path(self) -> Path:
        return self.artifact_dir / "slam_area_caption_jobs.jsonl"

    @property
    def results_path(self) -> Path:
        return self.artifact_dir / "slam_area_caption_results.jsonl"

    def enqueue_keyframe(self, keyframe: Mapping[str, Any]) -> list[str]:
        keyframe_id = str(keyframe.get("keyframe_id") or "")
        _append_jsonl(self.keyframes_path, keyframe)
        images = keyframe.get("images")
        if not keyframe_id or not isinstance(images, list):
            return []
        jobs: list[SlamAreaCaptionJob] = []
        for image in images:
            if not isinstance(image, Mapping):
                continue
            view = str(image.get("direction") or "").lower()
            image_path = image.get("path")
            if view not in _VIEW_ORDER or not isinstance(image_path, str) or not image_path:
                continue
            job_id = f"{keyframe_id}:{view}"
            try:
                heading = (
                    float(image["heading_deg"])
                    if image.get("heading_deg") is not None
                    else None
                )
            except (TypeError, ValueError):
                heading = None
            job = SlamAreaCaptionJob(
                job_id=job_id,
                keyframe_id=keyframe_id,
                session_id=self.session_id,
                view=view,
                image_path=image_path,
                heading_deg=heading,
                step_count=int(keyframe.get("step_count") or 0),
                cartographer_sequence=(
                    int(keyframe["cartographer_sequence"])
                    if keyframe.get("cartographer_sequence") is not None
                    else None
                ),
            )
            jobs.append(job)
        return self._submit_batch(SlamAreaCaptionBatch(tuple(jobs)))

    def _submit_job(self, job: SlamAreaCaptionJob) -> bool:
        return bool(self._submit_batch(SlamAreaCaptionBatch((job,))))

    def _submit_batch(self, batch: SlamAreaCaptionBatch) -> list[str]:
        if not batch.jobs:
            return []
        accepted: list[str] = []
        records = [self._job_record(job) for job in batch.jobs]
        if self._closed:
            for record in records:
                _append_jsonl(
                    self.jobs_path,
                    {**record, "status": "rejected", "reason": "closed"},
                )
            return accepted
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            for record in records:
                _append_jsonl(
                    self.jobs_path,
                    {**record, "status": "rejected", "reason": "queue_full"},
                )
            return accepted
        for record in records:
            _append_jsonl(self.jobs_path, {**record, "status": "accepted"})
            accepted.append(str(record["job_id"]))
        return accepted

    @staticmethod
    def _job_record(job: SlamAreaCaptionJob) -> dict[str, Any]:
        record = {
            "job_id": job.job_id,
            "keyframe_id": job.keyframe_id,
            "session_id": job.session_id,
            "view": job.view,
            "image_path": job.image_path,
            "heading_deg": job.heading_deg,
            "step_count": job.step_count,
            "cartographer_sequence": job.cartographer_sequence,
            "submitted_at_s": round(time.time(), 6),
        }
        return record

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            try:
                results = self._caption_batch(batch)
            except Exception as exc:  # noqa: BLE001 - worker records failures.
                if self._logger is not None:
                    self._logger.warning(
                        "SLAM area caption batch failed session=%s jobs=%s: %s",
                        self.session_id,
                        ",".join(job.job_id for job in batch.jobs),
                        exc,
                    )
                completed_at_s = round(time.time(), 6)
                results = [
                    {
                        "job_id": job.job_id,
                        "keyframe_id": job.keyframe_id,
                        "session_id": job.session_id,
                        "view": job.view,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_at_s": completed_at_s,
                    }
                    for job in batch.jobs
                ]
            for result in results:
                _append_jsonl(self.results_path, result)
            if self._closed and self._queue.empty():
                return

    def _caption(self, job: SlamAreaCaptionJob) -> dict[str, Any]:
        return self._caption_batch(SlamAreaCaptionBatch((job,)))[0]

    def _caption_batch(self, batch: SlamAreaCaptionBatch) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "items": [self._caption_payload(job) for job in batch.jobs],
        }
        request = urllib.request.Request(
            self._batch_backend_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"caption backend HTTP {exc.code}: {body[:200]}") from exc
        document = json.loads(raw) if raw.strip() else {}
        if not isinstance(document, Mapping):
            document = {}
        rows = document.get("results")
        if not isinstance(rows, list):
            rows = [document]
        by_job_id = {
            str(row.get("job_id") or ""): row
            for row in rows
            if isinstance(row, Mapping)
        }
        results: list[dict[str, Any]] = []
        completed_at_s = round(time.time(), 6)
        for job in batch.jobs:
            row = by_job_id.get(job.job_id)
            if row is None and len(batch.jobs) == 1 and isinstance(document, Mapping):
                row = document
            caption = row.get("caption", row.get("text")) if row is not None else None
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(f"caption backend response missing caption/text for {job.job_id}")
            results.append(
                {
                    "job_id": job.job_id,
                    "keyframe_id": job.keyframe_id,
                    "session_id": job.session_id,
                    "view": job.view,
                    "status": "ready",
                    "caption": caption.strip(),
                    "heading_deg": job.heading_deg,
                    "step_count": job.step_count,
                    "cartographer_sequence": job.cartographer_sequence,
                    "completed_at_s": completed_at_s,
                }
            )
        return results

    def _caption_payload(self, job: SlamAreaCaptionJob) -> dict[str, Any]:
        payload = {
            "job_id": job.job_id,
            "keyframe_id": job.keyframe_id,
            "session_id": job.session_id,
            "view": job.view,
            "heading_deg": job.heading_deg,
            "step_count": job.step_count,
            "cartographer_sequence": job.cartographer_sequence,
        }
        if self.transport == "path":
            payload["image_path"] = job.image_path
        else:
            try:
                payload["image_data"] = base64.b64encode(
                    Path(job.image_path).read_bytes()
                ).decode("ascii")
            except OSError as exc:
                raise RuntimeError(f"could not read caption image: {exc}") from exc
        return payload

    def _batch_backend_url(self) -> str:
        if self.backend_url.endswith("/caption"):
            return self.backend_url[: -len("/caption")] + "/caption_batch"
        return self.backend_url.rstrip("/") + "/caption_batch"

    def close(self, timeout_s: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Do not block session teardown on caption backlog.
            pass
        self._thread.join(timeout=max(0.0, float(timeout_s)))


def load_caption_results(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    path = artifact_dir / "slam_area_caption_results.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return results
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        job_id = row.get("job_id")
        if isinstance(job_id, str) and job_id:
            results[job_id] = row
    return results


def load_keyframes(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "slam_area_keyframes.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    keyframes: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            keyframes.append(row)
    return keyframes


def load_poses_by_sequence(artifact_dir: Path) -> dict[int, dict[str, Any]]:
    path = artifact_dir / "occupancy.png.poses.jsonl"
    poses: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return poses
    for line in lines:
        try:
            row = json.loads(line)
            sequence = int(row["sequence"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if isinstance(row, dict):
            poses[sequence] = row
    return poses


def enrich_slam_area_nodes_with_captions(
    *,
    artifact_dir: Path,
    inherited_sources: Sequence[Mapping[str, Any]] = (),
    segmentation_source: Path,
    segmentation_result: Mapping[str, Any],
    class_grid: np.ndarray,
    area_grid: np.ndarray,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not nodes:
        return nodes
    try:
        slam_map = load_slam_occupancy_map(segmentation_source)
    except Exception:
        for node in nodes:
            node["caption_status"] = "unavailable"
            node["caption_selection"] = {"reason": "raw_grid_pose_unavailable"}
            node["captions"] = []
        return nodes

    evidence_sources: list[tuple[Path, str, str | None, str]] = [
        (artifact_dir, "current", None, f"current:{artifact_dir.resolve()}")
    ]
    seen_dirs = {artifact_dir.resolve()}
    for source in inherited_sources:
        source_dir_value = source.get("artifact_dir")
        if not isinstance(source_dir_value, str) or not source_dir_value:
            continue
        source_dir = Path(source_dir_value)
        try:
            resolved = source_dir.resolve()
        except OSError:
            continue
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        session_id_value = source.get("source_session_id")
        source_session_id = (
            str(session_id_value) if session_id_value is not None else None
        )
        source_key = f"inherited:{source_session_id or resolved}:{resolved}"
        evidence_sources.append(
            (source_dir, "inherited", source_session_id, source_key)
        )

    keyframes: list[dict[str, Any]] = []
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for source_dir, origin, source_session_id, source_key in evidence_sources:
        source_results = load_caption_results(source_dir)
        source_keyframes = load_keyframes(source_dir)
        ready_keyframe_ids = {
            str(row.get("keyframe_id") or "")
            for row in source_results.values()
            if row.get("status") == "ready"
        }
        if origin == "inherited":
            # A pair snapshot is immutable. Only expose keyframes with at least
            # one completed caption instead of leaving retrieval permanently
            # pending on jobs that can no longer finish.
            source_keyframes = [
                row
                for row in source_keyframes
                if str(row.get("keyframe_id") or "") in ready_keyframe_ids
            ]
        source_keyframes = [
            {
                **row,
                "_caption_ready": str(row.get("keyframe_id") or "")
                in ready_keyframe_ids,
            }
            for row in source_keyframes
        ]
        keyframes.extend(
            _keyframes_with_pose(
                source_dir,
                slam_map,
                keyframes=source_keyframes,
                evidence_origin=origin,
                source_session_id=source_session_id,
                source_key=source_key,
            )
        )
        for job_id, row in source_results.items():
            results[(source_key, job_id)] = row

    if not keyframes:
        for node in nodes:
            node["caption_status"] = "unavailable"
            node["caption_selection"] = {"reason": "no_keyframes"}
            node["captions"] = []
        return nodes
    ready_keyframes = [row for row in keyframes if row.get("_caption_ready") is True]
    selection_keyframes = ready_keyframes or keyframes
    areas = {
        int(area["id"]): area
        for area in segmentation_result.get("areas", [])
        if isinstance(area, Mapping) and "id" in area
    }
    for node in nodes:
        try:
            area_id = int(node["area_id"])
        except (KeyError, TypeError, ValueError):
            continue
        area = areas.get(area_id)
        if area is None:
            node["caption_status"] = "unavailable"
            node["caption_selection"] = {"reason": "area_missing"}
            node["captions"] = []
            continue
        selected = _select_keyframe_for_area(
            slam_map=slam_map,
            class_grid=class_grid,
            area_grid=area_grid,
            area=area,
            area_id=area_id,
            keyframes=selection_keyframes,
        )
        if selected is None:
            node["caption_status"] = (
                "pending" if any(kf.get("pending_pose") for kf in keyframes) else "unavailable"
            )
            node["caption_selection"] = {"reason": "no_keyframe_with_pose"}
            node["captions"] = []
            continue
        captions, status = _captions_for_selection(selected, results)
        node["caption_status"] = status
        node["caption_selection"] = {
            key: value
            for key, value in selected.items()
            if key
            in {
                "strategy",
                "keyframe_id",
                "view",
                "ray_count",
                "distance_to_center_m",
                "cartographer_sequence",
                "evidence_origin",
                "source_session_id",
            }
        }
        node["captions"] = captions
    return nodes


def _keyframes_with_pose(
    artifact_dir: Path,
    slam_map: SlamOccupancyMap,
    *,
    keyframes: Sequence[Mapping[str, Any]] | None = None,
    evidence_origin: str = "current",
    source_session_id: str | None = None,
    source_key: str | None = None,
) -> list[dict[str, Any]]:
    poses = load_poses_by_sequence(artifact_dir)
    rows: list[dict[str, Any]] = []
    resolved_source_key = source_key or f"current:{artifact_dir.resolve()}"
    for keyframe in keyframes if keyframes is not None else load_keyframes(artifact_dir):
        keyframe_session_id = source_session_id or str(keyframe.get("session_id") or "") or None
        evidence_fields = {
            "evidence_origin": evidence_origin,
            "source_session_id": keyframe_session_id,
            "_caption_source_key": resolved_source_key,
        }
        sequence = keyframe.get("cartographer_sequence")
        try:
            sequence_int = int(sequence)
        except (TypeError, ValueError):
            rows.append({**keyframe, **evidence_fields, "pending_pose": True})
            continue
        pose = poses.get(sequence_int)
        if pose is None:
            rows.append({**keyframe, **evidence_fields, "pending_pose": True})
            continue
        try:
            row, col = slam_map.map_to_grid(float(pose["x"]), float(pose["y"]))
        except (KeyError, TypeError, ValueError):
            rows.append({**keyframe, **evidence_fields, "pending_pose": True})
            continue
        rows.append(
            {
                **keyframe,
                **evidence_fields,
                "pose": pose,
                "pose_grid_rc": [int(row), int(col)],
                "slam_yaw_rad": pose.get("yaw_rad"),
                "pending_pose": False,
            }
        )
    return rows


def _select_keyframe_for_area(
    *,
    slam_map: SlamOccupancyMap,
    class_grid: np.ndarray,
    area_grid: np.ndarray,
    area: Mapping[str, Any],
    area_id: int,
    keyframes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    centroid = area.get("centroid")
    if not isinstance(centroid, list) or len(centroid) < 2:
        return None
    center_row = float(centroid[0])
    center_col = float(centroid[1])
    inside: list[tuple[float, int, int, str, str, dict[str, Any]]] = []
    for keyframe in keyframes:
        cell = keyframe.get("pose_grid_rc")
        if not isinstance(cell, list) or len(cell) != 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < area_grid.shape[0] and 0 <= col < area_grid.shape[1]):
            continue
        if int(area_grid[row, col]) != area_id:
            continue
        distance_m = math.hypot(row - center_row, col - center_col) * slam_map.resolution
        inside.append(
            (
                distance_m,
                -int(keyframe.get("step_count") or 0),
                0 if keyframe.get("evidence_origin") == "current" else 1,
                str(keyframe.get("keyframe_id") or ""),
                str(keyframe.get("_caption_source_key") or ""),
                keyframe,
            )
        )
    if inside:
        distance_m, _neg_step, _origin_rank, keyframe_id, _source_key, keyframe = min(inside)
        return {
            "strategy": "inside_area",
            "keyframe_id": keyframe_id,
            "cartographer_sequence": keyframe.get("cartographer_sequence"),
            "distance_to_center_m": round(distance_m, 3),
            "expected_views": list(_VIEW_ORDER),
            "evidence_origin": keyframe.get("evidence_origin"),
            "source_session_id": keyframe.get("source_session_id"),
            "_caption_source_key": keyframe.get("_caption_source_key"),
        }

    ray_candidates: list[tuple[int, float, int, int, str, str, str, dict[str, Any]]] = []
    for keyframe in keyframes:
        cell = keyframe.get("pose_grid_rc")
        if not isinstance(cell, list) or len(cell) != 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < area_grid.shape[0] and 0 <= col < area_grid.shape[1]):
            continue
        distance_m = math.hypot(row - center_row, col - center_col) * slam_map.resolution
        view_counts = _ray_counts_by_view(
            keyframe=keyframe,
            class_grid=class_grid,
            area_grid=area_grid,
            area_id=area_id,
        )
        for view, ray_count in view_counts.items():
            ray_candidates.append(
                (
                    ray_count,
                    -distance_m,
                    int(keyframe.get("step_count") or 0),
                    1 if keyframe.get("evidence_origin") == "current" else 0,
                    str(keyframe.get("keyframe_id") or ""),
                    str(keyframe.get("_caption_source_key") or ""),
                    view,
                    keyframe,
                )
            )
    if not ray_candidates:
        return None
    (
        ray_count,
        neg_distance,
        _step,
        _origin_rank,
        keyframe_id,
        _source_key,
        view,
        keyframe,
    ) = max(ray_candidates)
    if ray_count <= 0:
        return None
    return {
        "strategy": "ray_cast",
        "keyframe_id": keyframe_id,
        "view": view,
        "ray_count": int(ray_count),
        "cartographer_sequence": keyframe.get("cartographer_sequence"),
        "distance_to_center_m": round(-neg_distance, 3),
        "expected_views": [view],
        "evidence_origin": keyframe.get("evidence_origin"),
        "source_session_id": keyframe.get("source_session_id"),
        "_caption_source_key": keyframe.get("_caption_source_key"),
    }


def _ray_counts_by_view(
    *,
    keyframe: Mapping[str, Any],
    class_grid: np.ndarray,
    area_grid: np.ndarray,
    area_id: int,
) -> dict[str, int]:
    images = keyframe.get("images")
    if not isinstance(images, list):
        return {}
    cell = keyframe.get("pose_grid_rc")
    if not isinstance(cell, list) or len(cell) != 2:
        return {}
    start = (int(cell[0]), int(cell[1]))
    counts: dict[str, int] = {}
    for image in images:
        if not isinstance(image, Mapping):
            continue
        view = str(image.get("direction") or "").lower()
        if view not in _VIEW_ORDER:
            continue
        try:
            heading = _slam_view_heading_rad(keyframe, view)
        except (KeyError, TypeError, ValueError):
            continue
        counts[view] = _count_rays_into_area(
            start=start,
            heading_rad=heading,
            class_grid=class_grid,
            area_grid=area_grid,
            area_id=area_id,
        )
    return counts


def _slam_view_heading_rad(keyframe: Mapping[str, Any], view: str) -> float:
    return float(keyframe["slam_yaw_rad"]) + _VIEW_YAW_OFFSETS_RAD[view]


def _count_rays_into_area(
    *,
    start: tuple[int, int],
    heading_rad: float,
    class_grid: np.ndarray,
    area_grid: np.ndarray,
    area_id: int,
) -> int:
    ray_count = 0
    occupied_code = _class_code(OCCUPIED)
    max_steps = int(math.ceil(math.hypot(area_grid.shape[0], area_grid.shape[1])))
    for offset in np.linspace(-math.pi / 4.0, math.pi / 4.0, num=9):
        angle = heading_rad + float(offset)
        row_delta = -math.sin(angle)
        col_delta = math.cos(angle)
        touched = False
        for step in range(1, max_steps + 1):
            row = int(round(start[0] + row_delta * step))
            col = int(round(start[1] + col_delta * step))
            if not (0 <= row < area_grid.shape[0] and 0 <= col < area_grid.shape[1]):
                break
            if int(class_grid[row, col]) == occupied_code:
                break
            if int(area_grid[row, col]) == area_id:
                touched = True
                break
        if touched:
            ray_count += 1
    return ray_count


def _captions_for_selection(
    selection: Mapping[str, Any],
    results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    keyframe_id = str(selection.get("keyframe_id") or "")
    source_key = str(selection.get("_caption_source_key") or "")
    expected_views = selection.get("expected_views")
    if not keyframe_id or not isinstance(expected_views, list):
        return [], "unavailable"
    captions: list[dict[str, Any]] = []
    saw_error = False
    ready_count = 0
    for view in expected_views:
        view_str = str(view)
        row = results.get((source_key, f"{keyframe_id}:{view_str}"))
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status") or "")
        if status == "error":
            saw_error = True
            continue
        caption = row.get("caption")
        if status == "ready" and isinstance(caption, str) and caption.strip():
            ready_count += 1
            captions.append(
                {
                    "view": view_str,
                    "heading_deg": row.get("heading_deg"),
                    "text": caption.strip(),
                    "status": "ready",
                    "evidence_origin": selection.get("evidence_origin"),
                    "source_session_id": selection.get("source_session_id"),
                }
            )
    if ready_count == len(expected_views):
        return captions, "ready"
    if ready_count > 0:
        return captions, "pending"
    if saw_error:
        return captions, "error"
    return captions, "pending"
