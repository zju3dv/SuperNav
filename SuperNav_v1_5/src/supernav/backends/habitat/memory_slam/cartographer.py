"""Asynchronous depth scan export for SuperNav's Cartographer 2D worker.

This SuperNav-1.5 method intentionally accepts only simulator-evaluator odometry.
Authoritative odometry is the default; Cartographer's regular odometry input is
retained as an explicit compatibility fallback.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_DEFAULT_DT_S = 1.0 / 60.0


def _write_memory_nodes(artifact_dirs: list[Path], nodes: list[dict[str, Any]]) -> None:
    payload = json.dumps({"nodes": nodes}, separators=(",", ":")) + "\n"
    for artifact_dir in artifact_dirs:
        path = artifact_dir / "memory_nodes.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)


def _write_reported_memory_node_prior(
    artifact_dir: Path,
    *,
    session_id: str,
    node: Mapping[str, Any],
    tool_seq: int,
) -> dict[str, Any]:
    """Publish the sequence-correlated node prior consumed by the worker."""

    node_id = node.get("id")
    slam_pose = node.get("slam_pose")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node.id must be a non-empty string")
    if not isinstance(slam_pose, Mapping):
        raise ValueError("node.slam_pose must be an object")
    try:
        sequence = slam_pose["sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError
        pose = {
            "sequence": sequence,
            "x": float(slam_pose["x"]),
            "y": float(slam_pose["y"]),
            "yaw_rad": float(slam_pose["yaw_rad"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "node.slam_pose must contain sequence, x, y, and yaw_rad"
        ) from exc
    radius_m = max(
        0.5, float(os.environ.get("HAB_CARTOGRAPHER_NODE_PRIOR_RADIUS_M", "4.0"))
    )
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "node_id": node_id,
        "node_kind": node.get("kind"),
        "node_name": node.get("name"),
        "slam_pose": pose,
        "region_bbox": [
            pose["x"] - radius_m,
            pose["y"] - radius_m,
            pose["x"] + radius_m,
            pose["y"] + radius_m,
        ],
        "region_radius_m": radius_m,
        "region_geometry": {
            "source": "pose_square_fallback",
            "fallback_radius_m": radius_m,
        },
        "tool_seq": tool_seq,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "reported_memory_node.json"
    temporary = artifact_dir / ".reported_memory_node.tmp"
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def _pbstream_available(output_path: Path, load_state_path: str | None) -> bool:
    """Return whether a final or immutable loaded Cartographer state is usable."""

    candidates = [output_path]
    if load_state_path:
        candidates.append(Path(load_state_path))
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _require_worker_capabilities(command: str, *, authoritative: bool) -> None:
    if not authoritative:
        return
    try:
        executable = shlex.split(command)[0]
        completed = subprocess.run(
            [executable, "--capabilities"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        capabilities = json.loads(completed.stdout)
    except Exception as exc:
        raise RuntimeError(
            "Cartographer authoritative mode requires a worker capability probe"
        ) from exc
    if not isinstance(capabilities, Mapping) or capabilities.get("authoritative_odometry") is not True:
        raise RuntimeError("Cartographer worker lacks authoritative_odometry capability")


def _require_relocalizer_dependencies(command: str | None) -> None:
    if not command:
        raise RuntimeError(
            "delayed Cartographer relocalization requires a relocalizer command"
        )
    try:
        completed = subprocess.run(
            [*shlex.split(command), "--check-open3d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to probe the Cartographer relocalizer runtime"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "Cartographer relocalization requires Open3D in the selected "
            "Habitat runtime; install the spatial-memory extra"
        )


@dataclass(frozen=True)
class CartographerScan:
    """One synthetic laser scan with optional declared-source odometry."""

    sequence: int
    timestamp_s: float
    ranges_m: list[float | None]
    odometry: tuple[float, float, float] | None = None
    hfov_deg: float = 90.0
    camera_hfov_deg: float = 90.0
    publish_snapshot: bool = True


@dataclass(frozen=True)
class CartographerControl:
    """One non-scan control message for the Cartographer worker."""

    type: str
    tool_seq: int | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class _QueuedScan:
    """One stream-labeled scan queued for a dual-stream worker."""

    stream: str
    scan: CartographerScan


@dataclass(frozen=True)
class _StreamConfig:
    """Per-input conversion and artifact settings for one worker stream."""

    artifact_dir: Path
    width: int
    hfov_deg: float
    sensor_height_m: float
    beam_count: int
    range_min_m: float
    range_max_m: float
    scan_source: str
    project_row_start_fraction: float
    project_pixel_stride: int
    min_obstacle_height_m: float
    max_obstacle_height_m: float
    load_state_path: str | None
    localization_mode: str | None
    initial_pose: tuple[float, float, float] | None


@dataclass
class CartographerStream:
    """Per-session bounded queue whose worker writes Cartographer input JSONL."""

    artifact_dir: Path
    width: int
    hfov_deg: float
    sensor_height_m: float
    beam_count: int
    range_min_m: float
    range_max_m: float
    queue_size: int
    use_odometry: bool = False
    odometry_source: str = "simulator_evaluator"
    odometry_mode: str = "authoritative"
    load_state_path: str | None = None
    localization_mode: str | None = None
    initial_pose: tuple[float, float, float] | None = None
    agent_guided_relocalization_enabled: bool = True
    relocalizer_command: str | None = None
    memory_nodes: list[dict[str, Any]] | None = None
    worker_command: str | None = None
    scan_source: str = "center_band"
    project_row_start_fraction: float = 0.5
    project_pixel_stride: int = 2
    min_obstacle_height_m: float = 0.15
    max_obstacle_height_m: float = 1.25
    _queue: queue.Queue[CartographerScan | CartographerControl | None] = field(
        init=False
    )
    _thread: threading.Thread = field(init=False)
    _worker: subprocess.Popen[str] | None = field(default=None, init=False)
    _input_fifo: Path | None = field(default=None, init=False)
    _fifo_stream: Any = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)
    submitted: int = field(default=0, init=False)
    processed: int = field(default=0, init=False)
    dropped: int = field(default=0, init=False)
    error: str | None = field(default=None, init=False)
    _odometry_origin: tuple[float, float, float] | None = field(
        default=None, init=False
    )
    _latest_odometry: tuple[float, float, float] | None = field(
        default=None, init=False
    )
    _latest_odometry_sequence: int | None = field(default=None, init=False)
    _last_map_source: Path | None = field(default=None, init=False)
    _last_localization_status: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_nodes is not None:
            (self.artifact_dir / "memory_nodes.json").write_text(
                json.dumps({"nodes": self.memory_nodes}, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._write_metadata()
        self._start_worker()
        self._thread = threading.Thread(
            target=self._run,
            name=f"cartographer-stream-{self.artifact_dir.name}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, scan: CartographerScan) -> None:
        if self._closed:
            return
        if self.use_odometry and scan.odometry is not None:
            if self._odometry_origin is None:
                self._odometry_origin = scan.odometry
            scan = dataclasses.replace(
                scan,
                odometry=_odometry_relative_to_origin(
                    scan.odometry, self._odometry_origin
                ),
            )
            self._latest_odometry = scan.odometry
            self._latest_odometry_sequence = scan.sequence
        try:
            self._queue.put_nowait(scan)
        except queue.Full:
            self.dropped += 1
            return
        self.submitted += 1

    def request_relocalization_prior_check(
        self, *, tool_seq: int | None = None, node_id: str | None = None
    ) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(
                CartographerControl(
                    type="check_relocalization_prior",
                    tool_seq=tool_seq,
                    node_id=node_id,
                )
            )
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def update_memory_nodes(self, nodes: list[dict[str, Any]]) -> None:
        self.memory_nodes = list(nodes)
        _write_memory_nodes([self.artifact_dir], self.memory_nodes)

    def close(self, timeout_s: float) -> dict[str, Any]:
        if not self._closed:
            self._closed = True
            # Closing is the only allowed blocking path: drain all accepted
            # inputs in order before destroying the session's simulator.
            self._queue.put(None, timeout=max(timeout_s, 0.0))
        self._thread.join(timeout=max(timeout_s, 0.0))
        if self._worker is not None:
            if self._fifo_stream is not None:
                self._fifo_stream.close()
                self._fifo_stream = None
            try:
                self._worker.wait(timeout=max(timeout_s, 0.0))
            except subprocess.TimeoutExpired:
                self._worker.kill()
                self.error = self.error or "Cartographer worker drain timed out"
        if self._thread.is_alive():
            self.error = self.error or "worker drain timed out"
        summary = self.status()
        (self.artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self._stream_mode(),
            "odometry": self.odometry_source if self.use_odometry else "omitted",
            "odometry_mode": self.odometry_mode if self.use_odometry else "omitted",
            "artifact_dir": str(self.artifact_dir),
            "submitted": self.submitted,
            "processed": self.processed,
            "dropped": self.dropped,
            "queue_depth": self._queue.qsize(),
            "error": self.error,
            "worker_returncode": (
                self._worker.poll() if self._worker is not None else None
            ),
            "occupancy_image": str(self.artifact_dir / "occupancy.png"),
            "worker_input_fifo": str(self._input_fifo) if self._input_fifo else None,
            "latest_odometry": self._latest_odometry,
            "latest_odometry_sequence": self._latest_odometry_sequence,
        }

    def wait_for_rendered_sequence(self, sequence: int, timeout_s: float) -> bool:
        """Wait until the worker has atomically published the requested map frame."""
        updates_path = self.artifact_dir / "occupancy.png.updates.jsonl"
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                with updates_path.open("r", encoding="utf-8") as updates:
                    for line in updates:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if int(record.get("sequence", -1)) >= sequence:
                            return True
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def wait_for_initial_map(self, timeout_s: float) -> bool:
        """Wait for either a persisted static map or the first live snapshot."""
        static_map = self.artifact_dir / "occupancy_step0.png"
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                if static_map.is_file() and static_map.stat().st_size > 0:
                    return True
            except OSError:
                pass
            if self.wait_for_rendered_sequence(0, 0.0):
                return True
            if self._worker is not None:
                returncode = self._worker.poll()
                if returncode not in (None, 0):
                    self.error = self.error or (
                        f"Cartographer worker exited before publishing initial map: "
                        f"returncode={returncode}"
                    )
                    return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _write_metadata(self) -> None:
        surround_scan = (
            os.environ.get("HAB_CARTOGRAPHER_SURROUND_SCAN", "0").strip() == "1"
        )
        scan_hfov_deg = 360.0 if surround_scan else self.hfov_deg
        angle_min = -math.radians(scan_hfov_deg) / 2.0
        angle_increment = math.radians(scan_hfov_deg) / (self.beam_count - 1)
        metadata = {
            "format": "supernav-cartographer-input/v1",
            "mode": self._stream_mode(),
            "odometry": self.odometry_source if self.use_odometry else "omitted",
            "odometry_mode": self.odometry_mode if self.use_odometry else "omitted",
            "odometry_authority": self.use_odometry
            and self.odometry_mode == "authoritative",
            "imu": "omitted",
            "sensor": {
                "frame_id": "depth_scan",
                "height_above_tracking_frame_m": self.sensor_height_m,
                "forward_axis": "Habitat -Z",
                "scan_plane": (
                    "lower-half projected depth floor-plane obstacle band"
                    if self.scan_source == "projected_depth"
                    else "horizontal optical-center band"
                ),
            },
            "laser_scan": {
                "beam_count": self.beam_count,
                "hfov_deg": scan_hfov_deg,
                "camera_hfov_deg": self.hfov_deg,
                "angle_min_rad": angle_min,
                "angle_increment_rad": angle_increment,
                "range_min_m": self.range_min_m,
                "range_max_m": self.range_max_m,
            },
            "projected_depth": {
                "enabled": self.scan_source == "projected_depth",
                "row_start_fraction": self.project_row_start_fraction,
                "pixel_stride": self.project_pixel_stride,
                "min_obstacle_height_m": self.min_obstacle_height_m,
                "max_obstacle_height_m": self.max_obstacle_height_m,
            },
            "surround_scan": surround_scan,
            "localization": {
                "mode": self.localization_mode
                or (
                    "frozen_localization_amcl_global"
                    if self.load_state_path
                    else "mapping"
                ),
                "load_state_path": self.load_state_path,
                "initial_pose": self.initial_pose,
            },
        }
        (self.artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _stream_mode(self) -> str:
        if not self.use_odometry:
            return "scan_matching_only"
        if self.odometry_mode == "authoritative":
            return "authoritative_odometry"
        return "scan_matching_with_odometry"

    def _start_worker(self) -> None:
        command = (
            self.worker_command
            if self.worker_command is not None
            else os.environ.get("HAB_CARTOGRAPHER_WORKER_COMMAND", "")
        ).strip()
        if not command:
            return
        _require_worker_capabilities(
            command,
            authoritative=self.use_odometry and self.odometry_mode == "authoritative",
        )
        if self.localization_mode == "delayed_fpfh_relocalization":
            _require_relocalizer_dependencies(self.relocalizer_command)
        self._input_fifo = self.artifact_dir / "cartographer_input.fifo"
        try:
            os.mkfifo(self._input_fifo)
        except FileExistsError:
            self._input_fifo.unlink()
            os.mkfifo(self._input_fifo)
        occupancy_path = self.artifact_dir / "occupancy.png"
        command = (
            command.replace("{occupancy_path}", str(occupancy_path))
            .replace("{input_fifo}", str(self._input_fifo))
            .replace("{session_id}", self.artifact_dir.name)
            .replace("{load_state_path}", self.load_state_path or "")
            .replace(
                "{initial_pose_json}",
                (
                    json.dumps(self.initial_pose, separators=(",", ":"))
                    if self.initial_pose is not None
                    else ""
                ),
            )
        )
        environment = os.environ.copy()
        environment["HAB_CARTOGRAPHER_USE_ODOMETRY"] = "1" if self.use_odometry else "0"
        environment["HAB_CARTOGRAPHER_ODOMETRY_SOURCE"] = self.odometry_source
        environment["HAB_CARTOGRAPHER_ODOMETRY_MODE"] = self.odometry_mode
        environment["HAB_CARTOGRAPHER_AGENT_GUIDED_RELOCALIZATION_ENABLED"] = (
            "1" if self.agent_guided_relocalization_enabled else "0"
        )
        if self.relocalizer_command is not None:
            environment["HAB_CARTOGRAPHER_RELOCALIZER_COMMAND"] = (
                self.relocalizer_command
            )
        if self.load_state_path is not None:
            environment["HAB_CARTOGRAPHER_LOAD_STATE"] = self.load_state_path
        if self.localization_mode is not None:
            environment["HAB_CARTOGRAPHER_LOCALIZATION_MODE"] = self.localization_mode
        if self.initial_pose is not None:
            environment["HAB_CARTOGRAPHER_INITIAL_POSE"] = json.dumps(
                self.initial_pose, separators=(",", ":")
            )
        self._worker = subprocess.Popen(
            command,
            shell=True,
            # The bridge-owned launcher reads the session FIFO itself.
            stdin=subprocess.DEVNULL,
            text=True,
            env=environment,
        )

    def _run(self) -> None:
        trace_path = self.artifact_dir / "scans.jsonl"
        try:
            with trace_path.open("w", encoding="utf-8") as stream:
                while True:
                    scan = self._queue.get()
                    if scan is None:
                        return
                    if isinstance(scan, CartographerControl):
                        record: dict[str, Any] = {"type": scan.type}
                        if scan.tool_seq is not None:
                            record["tool_seq"] = scan.tool_seq
                        if scan.node_id is not None:
                            record["node_id"] = scan.node_id
                    else:
                        record = {
                            "sequence": scan.sequence,
                            "timestamp_s": scan.timestamp_s,
                            "ranges_m": scan.ranges_m,
                            "hfov_deg": scan.hfov_deg,
                            "camera_hfov_deg": scan.camera_hfov_deg,
                            "publish_snapshot": scan.publish_snapshot,
                        }
                        if self.use_odometry and scan.odometry is not None:
                            record["odometry"] = {
                                "x": scan.odometry[0],
                                "y": scan.odometry[1],
                                "yaw_rad": scan.odometry[2],
                            }
                            if self._odometry_origin is not None:
                                metadata_path = self.artifact_dir / "metadata.json"
                                metadata = json.loads(
                                    metadata_path.read_text(encoding="utf-8")
                                )
                                metadata["odometry_origin"] = self._odometry_origin
                                metadata_path.write_text(
                                    json.dumps(metadata, indent=2, sort_keys=True)
                                    + "\n",
                                    encoding="utf-8",
                                )
                    stream.write(
                        json.dumps(
                            record,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    if self._input_fifo is not None and self._fifo_stream is None:
                        # The tmux worker opens the FIFO for reading. This blocks
                        # only the mapper queue thread, never a simulator action.
                        self._fifo_stream = self._input_fifo.open("w", encoding="utf-8")
                    if self._fifo_stream is not None:
                        self._fifo_stream.write(
                            json.dumps(
                                record,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        self._fifo_stream.flush()
                    self.processed += 1
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)


class _PlanningCartographerStreamProxy:
    """Compatibility surface for planning occupancy artifacts in a dual worker."""

    def __init__(self, parent: "DualCartographerStream") -> None:
        self._parent = parent

    @property
    def artifact_dir(self) -> Path:
        return self._parent.planning_artifact_dir

    @property
    def width(self) -> int:
        return self._parent.planning.width

    @property
    def hfov_deg(self) -> float:
        return self._parent.planning.hfov_deg

    @property
    def sensor_height_m(self) -> float:
        return self._parent.planning.sensor_height_m

    @property
    def beam_count(self) -> int:
        return self._parent.planning.beam_count

    @property
    def range_min_m(self) -> float:
        return self._parent.planning.range_min_m

    @property
    def range_max_m(self) -> float:
        return self._parent.planning.range_max_m

    @property
    def scan_source(self) -> str:
        return self._parent.planning.scan_source

    @property
    def project_row_start_fraction(self) -> float:
        return self._parent.planning.project_row_start_fraction

    @property
    def project_pixel_stride(self) -> int:
        return self._parent.planning.project_pixel_stride

    @property
    def min_obstacle_height_m(self) -> float:
        return self._parent.planning.min_obstacle_height_m

    @property
    def max_obstacle_height_m(self) -> float:
        return self._parent.planning.max_obstacle_height_m

    @property
    def localization_mode(self) -> str | None:
        return self._parent.planning.localization_mode

    def submit(self, scan: CartographerScan) -> None:
        self._parent.submit_planning(scan)

    def close(self, timeout_s: float) -> dict[str, Any]:
        return self._parent.close(timeout_s)

    def status(self) -> dict[str, Any]:
        status = self._parent.status()
        status["artifact_dir"] = str(self.artifact_dir)
        status["occupancy_image"] = str(self.artifact_dir / "occupancy.png")
        return status

    def wait_for_rendered_sequence(self, sequence: int, timeout_s: float) -> bool:
        return self._parent.wait_for_rendered_sequence(
            sequence, timeout_s, stream="planning"
        )

    def wait_for_initial_map(self, timeout_s: float) -> bool:
        return self._parent.wait_for_initial_map(timeout_s, stream="planning")


@dataclass
class DualCartographerStream:
    """One queue and worker process serving visual and planning Cartographer streams."""

    session_artifact_dir: Path
    visual: _StreamConfig
    planning: _StreamConfig
    queue_size: int
    use_odometry: bool = False
    odometry_source: str = "simulator_evaluator"
    odometry_mode: str = "authoritative"
    agent_guided_relocalization_enabled: bool = True
    relocalizer_command: str | None = None
    memory_nodes: list[dict[str, Any]] | None = None
    worker_command: str | None = None
    _queue: queue.Queue[_QueuedScan | CartographerControl | None] = field(init=False)
    _thread: threading.Thread = field(init=False)
    _worker: subprocess.Popen[str] | None = field(default=None, init=False)
    _input_fifo: Path | None = field(default=None, init=False)
    _fifo_stream: Any = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)
    submitted: int = field(default=0, init=False)
    processed: int = field(default=0, init=False)
    dropped: int = field(default=0, init=False)
    error: str | None = field(default=None, init=False)
    _odometry_origin: tuple[float, float, float] | None = field(
        default=None, init=False
    )
    _latest_odometry: tuple[float, float, float] | None = field(
        default=None, init=False
    )
    _latest_odometry_sequence: int | None = field(default=None, init=False)
    _last_map_source: Path | None = field(default=None, init=False)
    _last_localization_status: str | None = field(default=None, init=False)
    planning_proxy: _PlanningCartographerStreamProxy = field(init=False)

    def __post_init__(self) -> None:
        self.session_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.visual.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.planning.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_nodes is not None:
            payload = (
                json.dumps({"nodes": self.memory_nodes}, separators=(",", ":"))
                + "\n"
            )
            for artifact_dir in (self.visual.artifact_dir, self.planning.artifact_dir):
                (artifact_dir / "memory_nodes.json").write_text(
                    payload, encoding="utf-8"
                )
        self._queue = queue.Queue(maxsize=self.queue_size)
        self.planning_proxy = _PlanningCartographerStreamProxy(self)
        self._write_metadata()
        self._start_worker()
        self._thread = threading.Thread(
            target=self._run,
            name=f"dual-cartographer-stream-{self.session_artifact_dir.name}",
            daemon=True,
        )
        self._thread.start()

    @property
    def artifact_dir(self) -> Path:
        return self.visual.artifact_dir

    @property
    def planning_artifact_dir(self) -> Path:
        return self.planning.artifact_dir

    @property
    def width(self) -> int:
        return self.visual.width

    @property
    def hfov_deg(self) -> float:
        return self.visual.hfov_deg

    @property
    def sensor_height_m(self) -> float:
        return self.visual.sensor_height_m

    @property
    def beam_count(self) -> int:
        return self.visual.beam_count

    @property
    def range_min_m(self) -> float:
        return self.visual.range_min_m

    @property
    def range_max_m(self) -> float:
        return self.visual.range_max_m

    @property
    def scan_source(self) -> str:
        return self.visual.scan_source

    @property
    def project_row_start_fraction(self) -> float:
        return self.visual.project_row_start_fraction

    @property
    def project_pixel_stride(self) -> int:
        return self.visual.project_pixel_stride

    @property
    def min_obstacle_height_m(self) -> float:
        return self.visual.min_obstacle_height_m

    @property
    def max_obstacle_height_m(self) -> float:
        return self.visual.max_obstacle_height_m

    @property
    def localization_mode(self) -> str | None:
        return self.visual.localization_mode

    def submit(self, scan: CartographerScan) -> None:
        self._submit("visual", scan)

    def submit_planning(self, scan: CartographerScan) -> None:
        self._submit("planning", scan)

    def request_relocalization_prior_check(
        self, *, tool_seq: int | None = None, node_id: str | None = None
    ) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(
                CartographerControl(
                    type="check_relocalization_prior",
                    tool_seq=tool_seq,
                    node_id=node_id,
                )
            )
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def update_memory_nodes(self, nodes: list[dict[str, Any]]) -> None:
        self.memory_nodes = list(nodes)
        _write_memory_nodes(
            [self.visual.artifact_dir, self.planning.artifact_dir],
            self.memory_nodes,
        )

    def close(self, timeout_s: float) -> dict[str, Any]:
        if not self._closed:
            self._closed = True
            self._queue.put(None, timeout=max(timeout_s, 0.0))
        self._thread.join(timeout=max(timeout_s, 0.0))
        if self._fifo_stream is not None:
            self._fifo_stream.close()
            self._fifo_stream = None
        if self._worker is not None:
            try:
                self._worker.wait(timeout=max(timeout_s, 0.0))
            except subprocess.TimeoutExpired:
                self._worker.kill()
                self.error = self.error or "Cartographer worker drain timed out"
        if self._thread.is_alive():
            self.error = self.error or "worker drain timed out"
        final_artifacts_ready = self._wait_for_final_artifacts(timeout_s)
        if self.processed > 0 and not final_artifacts_ready:
            self.error = (
                self.error or "Cartographer worker did not publish final pbstreams"
            )
        summary = self.status()
        summary["final_artifacts_ready"] = final_artifacts_ready
        (self.session_artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self._stream_mode(),
            "dual_stream": True,
            "odometry": self.odometry_source if self.use_odometry else "omitted",
            "odometry_mode": self.odometry_mode if self.use_odometry else "omitted",
            "artifact_dir": str(self.artifact_dir),
            "planning_artifact_dir": str(self.planning_artifact_dir),
            "submitted": self.submitted,
            "processed": self.processed,
            "dropped": self.dropped,
            "queue_depth": self._queue.qsize(),
            "error": self.error,
            "worker_returncode": (
                self._worker.poll() if self._worker is not None else None
            ),
            "occupancy_image": str(self.artifact_dir / "occupancy.png"),
            "planning_occupancy_image": str(
                self.planning_artifact_dir / "occupancy.png"
            ),
            "worker_input_fifo": str(self._input_fifo) if self._input_fifo else None,
            "latest_odometry": self._latest_odometry,
            "latest_odometry_sequence": self._latest_odometry_sequence,
        }

    def wait_for_rendered_sequence(
        self, sequence: int, timeout_s: float, *, stream: str = "visual"
    ) -> bool:
        artifact_dir = self._artifact_dir_for_stream(stream)
        updates_path = artifact_dir / "occupancy.png.updates.jsonl"
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                with updates_path.open("r", encoding="utf-8") as updates:
                    for line in updates:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if int(record.get("sequence", -1)) >= sequence:
                            return True
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def wait_for_initial_map(self, timeout_s: float, *, stream: str = "visual") -> bool:
        artifact_dir = self._artifact_dir_for_stream(stream)
        static_map = artifact_dir / "occupancy_step0.png"
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                if static_map.is_file() and static_map.stat().st_size > 0:
                    return True
            except OSError:
                pass
            if self.wait_for_rendered_sequence(0, 0.0, stream=stream):
                return True
            if self._worker is not None:
                returncode = self._worker.poll()
                if returncode not in (None, 0):
                    self.error = self.error or (
                        f"Cartographer worker exited before publishing initial map: "
                        f"returncode={returncode}"
                    )
                    return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _submit(self, stream: str, scan: CartographerScan) -> None:
        if self._closed:
            return
        if self.use_odometry and scan.odometry is not None:
            if self._odometry_origin is None:
                self._odometry_origin = scan.odometry
            scan = dataclasses.replace(
                scan,
                odometry=_odometry_relative_to_origin(
                    scan.odometry, self._odometry_origin
                ),
            )
            self._latest_odometry = scan.odometry
            self._latest_odometry_sequence = scan.sequence
        try:
            self._queue.put_nowait(_QueuedScan(stream=stream, scan=scan))
        except queue.Full:
            self.dropped += 1
            return
        self.submitted += 1

    def _artifact_dir_for_stream(self, stream: str) -> Path:
        if stream == "visual":
            return self.visual.artifact_dir
        if stream == "planning":
            return self.planning.artifact_dir
        raise ValueError(f"unknown Cartographer stream: {stream}")

    def _wait_for_final_artifacts(self, timeout_s: float) -> bool:
        if self.processed <= 0:
            return False
        expected = [
            (
                self.visual.artifact_dir / "occupancy.png.pbstream",
                self.visual.load_state_path,
            ),
            (
                self.planning.artifact_dir / "occupancy.png.pbstream",
                self.planning.load_state_path,
            ),
        ]
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                if all(
                    _pbstream_available(output_path, load_state_path)
                    for output_path, load_state_path in expected
                ):
                    return True
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _write_metadata(self) -> None:
        metadata = {
            "format": "supernav-cartographer-dual-input/v1",
            "mode": self._stream_mode(),
            "odometry": self.odometry_source if self.use_odometry else "omitted",
            "odometry_mode": self.odometry_mode if self.use_odometry else "omitted",
            "odometry_authority": self.use_odometry
            and self.odometry_mode == "authoritative",
            "worker_input_fifo": "cartographer_input.fifo",
            "streams": {
                "visual": self._metadata_for_config(self.visual),
                "planning": self._metadata_for_config(
                    self.planning, localization_source="visual_transform"
                ),
            },
        }
        (self.session_artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.visual.artifact_dir / "metadata.json").write_text(
            json.dumps(metadata["streams"]["visual"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.planning.artifact_dir / "metadata.json").write_text(
            json.dumps(metadata["streams"]["planning"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def _metadata_for_config(
        self, config: _StreamConfig, *, localization_source: str | None = None
    ) -> dict[str, Any]:
        surround_scan = (
            os.environ.get("HAB_CARTOGRAPHER_SURROUND_SCAN", "0").strip() == "1"
        )
        scan_hfov_deg = 360.0 if surround_scan else config.hfov_deg
        angle_min = -math.radians(scan_hfov_deg) / 2.0
        angle_increment = math.radians(scan_hfov_deg) / (config.beam_count - 1)
        localization = {
            "mode": config.localization_mode
            or (
                "frozen_localization_amcl_global"
                if config.load_state_path
                else "mapping"
            ),
            "load_state_path": config.load_state_path,
            "initial_pose": config.initial_pose,
        }
        if localization_source is not None:
            localization["localization_source"] = localization_source
        return {
            "format": "supernav-cartographer-input/v1",
            "mode": self._stream_mode(),
            "odometry": self.odometry_source if self.use_odometry else "omitted",
            "odometry_mode": self.odometry_mode if self.use_odometry else "omitted",
            "odometry_authority": self.use_odometry
            and self.odometry_mode == "authoritative",
            "imu": "omitted",
            "sensor": {
                "frame_id": "depth_scan",
                "height_above_tracking_frame_m": config.sensor_height_m,
                "forward_axis": "Habitat -Z",
                "scan_plane": (
                    "lower-half projected depth floor-plane obstacle band"
                    if config.scan_source == "projected_depth"
                    else "horizontal optical-center band"
                ),
            },
            "laser_scan": {
                "beam_count": config.beam_count,
                "hfov_deg": scan_hfov_deg,
                "camera_hfov_deg": config.hfov_deg,
                "angle_min_rad": angle_min,
                "angle_increment_rad": angle_increment,
                "range_min_m": config.range_min_m,
                "range_max_m": config.range_max_m,
            },
            "projected_depth": {
                "enabled": config.scan_source == "projected_depth",
                "row_start_fraction": config.project_row_start_fraction,
                "pixel_stride": config.project_pixel_stride,
                "min_obstacle_height_m": config.min_obstacle_height_m,
                "max_obstacle_height_m": config.max_obstacle_height_m,
            },
            "surround_scan": surround_scan,
            "localization": localization,
        }

    def _stream_mode(self) -> str:
        if not self.use_odometry:
            return "scan_matching_only"
        if self.odometry_mode == "authoritative":
            return "authoritative_odometry"
        return "scan_matching_with_odometry"

    def _start_worker(self) -> None:
        command = (
            self.worker_command
            if self.worker_command is not None
            else os.environ.get("HAB_CARTOGRAPHER_WORKER_COMMAND", "")
        ).strip()
        if not command:
            return
        _require_worker_capabilities(
            command,
            authoritative=self.use_odometry and self.odometry_mode == "authoritative",
        )
        if self.visual.localization_mode == "delayed_fpfh_relocalization":
            _require_relocalizer_dependencies(self.relocalizer_command)
        self._input_fifo = self.session_artifact_dir / "cartographer_input.fifo"
        try:
            os.mkfifo(self._input_fifo)
        except FileExistsError:
            self._input_fifo.unlink()
            os.mkfifo(self._input_fifo)
        occupancy_path = self.visual.artifact_dir / "occupancy.png"
        planning_occupancy_path = self.planning.artifact_dir / "occupancy.png"
        command = (
            command.replace("{occupancy_path}", str(occupancy_path))
            .replace("{planning_occupancy_path}", str(planning_occupancy_path))
            .replace("{input_fifo}", str(self._input_fifo))
            .replace("{session_id}", self.session_artifact_dir.name)
            .replace("{load_state_path}", self.visual.load_state_path or "")
            .replace(
                "{initial_pose_json}",
                (
                    json.dumps(self.visual.initial_pose, separators=(",", ":"))
                    if self.visual.initial_pose is not None
                    else ""
                ),
            )
        )
        environment = os.environ.copy()
        environment["HAB_CARTOGRAPHER_USE_ODOMETRY"] = "1" if self.use_odometry else "0"
        environment["HAB_CARTOGRAPHER_ODOMETRY_SOURCE"] = self.odometry_source
        environment["HAB_CARTOGRAPHER_ODOMETRY_MODE"] = self.odometry_mode
        environment["HAB_CARTOGRAPHER_AGENT_GUIDED_RELOCALIZATION_ENABLED"] = (
            "1" if self.agent_guided_relocalization_enabled else "0"
        )
        if self.relocalizer_command is not None:
            environment["HAB_CARTOGRAPHER_RELOCALIZER_COMMAND"] = (
                self.relocalizer_command
            )
        environment["HAB_CARTOGRAPHER_DUAL_STREAM"] = "1"
        environment["HAB_CARTOGRAPHER_VISUAL_OUTPUT"] = str(occupancy_path)
        environment["HAB_CARTOGRAPHER_PLANNING_OUTPUT"] = str(planning_occupancy_path)
        if self.visual.load_state_path is not None:
            environment["HAB_CARTOGRAPHER_VISUAL_LOAD_STATE"] = (
                self.visual.load_state_path
            )
            environment["HAB_CARTOGRAPHER_LOAD_STATE"] = self.visual.load_state_path
        if self.planning.load_state_path is not None:
            environment["HAB_CARTOGRAPHER_PLANNING_LOAD_STATE"] = (
                self.planning.load_state_path
            )
        if self.visual.localization_mode is not None:
            environment["HAB_CARTOGRAPHER_VISUAL_LOCALIZATION_MODE"] = (
                self.visual.localization_mode
            )
            environment["HAB_CARTOGRAPHER_LOCALIZATION_MODE"] = (
                self.visual.localization_mode
            )
        if self.planning.localization_mode is not None:
            environment["HAB_CARTOGRAPHER_PLANNING_LOCALIZATION_MODE"] = (
                self.planning.localization_mode
            )
        if self.visual.initial_pose is not None:
            environment["HAB_CARTOGRAPHER_INITIAL_POSE"] = json.dumps(
                self.visual.initial_pose, separators=(",", ":")
            )
        self._worker = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            text=True,
            env=environment,
        )

    def _run(self) -> None:
        trace_path = self.session_artifact_dir / "scans.jsonl"
        try:
            with trace_path.open("w", encoding="utf-8") as stream:
                while True:
                    item = self._queue.get()
                    if item is None:
                        return
                    if isinstance(item, CartographerControl):
                        record: dict[str, Any] = {"type": item.type}
                        if item.tool_seq is not None:
                            record["tool_seq"] = item.tool_seq
                        if item.node_id is not None:
                            record["node_id"] = item.node_id
                    else:
                        scan = item.scan
                        record = {
                            "stream": item.stream,
                            "sequence": scan.sequence,
                            "timestamp_s": scan.timestamp_s,
                            "ranges_m": scan.ranges_m,
                            "hfov_deg": scan.hfov_deg,
                            "camera_hfov_deg": scan.camera_hfov_deg,
                            "publish_snapshot": scan.publish_snapshot,
                        }
                        if self.use_odometry and scan.odometry is not None:
                            record["odometry"] = {
                                "x": scan.odometry[0],
                                "y": scan.odometry[1],
                                "yaw_rad": scan.odometry[2],
                            }
                    payload = (
                        json.dumps(record, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                    stream.write(payload)
                    stream.flush()
                    if self._input_fifo is not None and self._fifo_stream is None:
                        self._fifo_stream = self._input_fifo.open("w", encoding="utf-8")
                    if self._fifo_stream is not None:
                        self._fifo_stream.write(payload)
                        self._fifo_stream.flush()
                    self.processed += 1
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)


def enabled_from_environment() -> bool:
    return os.environ.get("HAB_CARTOGRAPHER_ENABLED", "0").strip() == "1"


def _variant_key(variant: str | None) -> str | None:
    if variant is None:
        return None
    text = str(variant).strip().lower()
    return text or None


def _env_name(base: str, variant: str | None = None) -> str:
    variant_key = _variant_key(variant)
    if variant_key is None:
        return f"HAB_CARTOGRAPHER_{base}"
    return f"HAB_CARTOGRAPHER_{variant_key.upper()}_{base}"


def _setting_name(base: str, variant: str | None = None) -> str:
    variant_key = _variant_key(variant)
    if variant_key is None:
        return base
    return f"cartographer_{variant_key}_{base}"


def _env_int(name: str, default: int, *, fallback_name: str | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None and fallback_name is not None:
        raw = os.environ.get(fallback_name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (AttributeError, ValueError):
        return default


def _env_float(name: str, default: float, *, fallback_name: str | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None and fallback_name is not None:
        raw = os.environ.get(fallback_name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (AttributeError, ValueError):
        return default


def _env_bool(
    name: str, default: bool = False, *, fallback_name: str | None = None
) -> bool:
    raw = os.environ.get(name)
    if raw is None and fallback_name is not None:
        raw = os.environ.get(fallback_name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    return default


def stream_from_environment(
    session_id: str, settings: Mapping[str, Any], *, variant: str | None = None
) -> CartographerStream:
    (
        config,
        queue_size,
        use_odometry,
        odometry_source,
        odometry_mode,
        memory_node_labels,
        command,
    ) = _stream_config_from_environment(session_id, settings, variant=variant)
    return CartographerStream(
        artifact_dir=config.artifact_dir,
        width=config.width,
        hfov_deg=config.hfov_deg,
        sensor_height_m=config.sensor_height_m,
        beam_count=config.beam_count,
        range_min_m=config.range_min_m,
        range_max_m=config.range_max_m,
        queue_size=queue_size,
        use_odometry=use_odometry,
        odometry_source=odometry_source,
        odometry_mode=odometry_mode,
        load_state_path=config.load_state_path,
        localization_mode=config.localization_mode,
        initial_pose=config.initial_pose,
        agent_guided_relocalization_enabled=bool(
            settings.get("cartographer_agent_guided_relocalization_enabled", True)
        ),
        relocalizer_command=(
            str(settings["cartographer_relocalizer_command"])
            if settings.get("cartographer_relocalizer_command")
            else None
        ),
        memory_nodes=memory_node_labels,
        worker_command=command,
        scan_source=config.scan_source,
        project_row_start_fraction=config.project_row_start_fraction,
        project_pixel_stride=config.project_pixel_stride,
        min_obstacle_height_m=config.min_obstacle_height_m,
        max_obstacle_height_m=config.max_obstacle_height_m,
    )


def dual_stream_from_environment(
    session_id: str, settings: Mapping[str, Any]
) -> DualCartographerStream:
    (
        visual,
        queue_size,
        use_odometry,
        odometry_source,
        odometry_mode,
        memory_node_labels,
        command,
    ) = _stream_config_from_environment(session_id, settings)
    planning, planning_queue_size, _, _, _, _, _ = _stream_config_from_environment(
        session_id,
        settings,
        variant="occ",
        force_session_artifact_dir=visual.artifact_dir.parent / session_id / "planning",
    )
    if command is None or not str(command).strip():
        occ_worker_command = settings.get("cartographer_occ_worker_command")
        if isinstance(occ_worker_command, str) and occ_worker_command.strip():
            command = occ_worker_command
        else:
            env_occ_worker_command = os.environ.get(
                "HAB_CARTOGRAPHER_OCC_WORKER_COMMAND"
            )
            if env_occ_worker_command is not None and env_occ_worker_command.strip():
                command = env_occ_worker_command
    if visual.load_state_path is not None and planning.load_state_path is None:
        raise ValueError(
            "dual Cartographer retrieve requires cartographer_localization."
            "planning_load_state_path"
        )
    root = visual.artifact_dir.parent / session_id
    visual = dataclasses.replace(visual, artifact_dir=root / "visual")
    queue_size = max(queue_size, planning_queue_size)
    return DualCartographerStream(
        session_artifact_dir=root,
        visual=visual,
        planning=planning,
        queue_size=queue_size,
        use_odometry=use_odometry,
        odometry_source=odometry_source,
        odometry_mode=odometry_mode,
        agent_guided_relocalization_enabled=bool(
            settings.get("cartographer_agent_guided_relocalization_enabled", True)
        ),
        relocalizer_command=(
            str(settings["cartographer_relocalizer_command"])
            if settings.get("cartographer_relocalizer_command")
            else None
        ),
        memory_nodes=memory_node_labels,
        worker_command=command,
    )


def _stream_config_from_environment(
    session_id: str,
    settings: Mapping[str, Any],
    *,
    variant: str | None = None,
    force_session_artifact_dir: Path | None = None,
) -> tuple[
    _StreamConfig,
    int,
    bool,
    str,
    str,
    list[dict[str, Any]] | None,
    str | None,
]:
    variant_key = _variant_key(variant)
    default_scan_source = "projected_depth" if variant_key == "occ" else "center_band"
    scan_source = str(
        settings.get(
            _setting_name("scan_source", variant_key),
            os.environ.get(
                _env_name("SCAN_SOURCE", variant_key),
                default_scan_source,
            ),
        )
    ).strip()
    if scan_source not in {"center_band", "projected_depth"}:
        scan_source = default_scan_source
    output_setting = (
        "cartographer_output_dir"
        if variant_key is None
        else _setting_name("output_dir", variant_key)
    )
    output_dir = settings.get(output_setting)
    if not isinstance(output_dir, str) or not output_dir:
        output_dir = os.environ.get(
            _env_name("OUTPUT_DIR", variant_key),
            os.environ.get("HAB_CARTOGRAPHER_OUTPUT_DIR", "/tmp/habitat_cartographer"),
        )
    root = Path(output_dir)
    worker_setting = (
        "cartographer_worker_command"
        if variant_key is None
        else _setting_name("worker_command", variant_key)
    )
    worker_command = settings.get(worker_setting)
    if not isinstance(worker_command, str) or not worker_command.strip():
        worker_command = os.environ.get(
            _env_name("WORKER_COMMAND", variant_key),
            os.environ.get("HAB_CARTOGRAPHER_WORKER_COMMAND", ""),
        )
    beam_setting = settings.get(_setting_name("beams", variant_key))
    if beam_setting is None and variant_key is None:
        beam_setting = settings.get("cartographer_beams")
    beam_count = max(
        2,
        int(beam_setting)
        if beam_setting is not None
        else _env_int(
            _env_name("BEAMS", variant_key),
            256,
            fallback_name=None if variant_key is None else _env_name("BEAMS"),
        ),
    )
    queue_size = max(
        1,
        _env_int(
            _env_name("QUEUE_SIZE", variant_key),
            4096,
            fallback_name=None if variant_key is None else _env_name("QUEUE_SIZE"),
        ),
    )
    localization = settings.get("cartographer_localization")
    if not isinstance(localization, Mapping):
        localization = {}
    load_state_path = localization.get("load_state_path")
    localization_mode = localization.get("mode")
    relocalization_enabled = bool(
        settings.get("cartographer_relocalization_enabled", True)
    )
    if variant_key == "occ":
        planning_load_state_path = localization.get("planning_load_state_path")
        if isinstance(planning_load_state_path, str) and planning_load_state_path:
            load_state_path = planning_load_state_path
            localization_mode = (
                "static_map_no_relocalization"
                if localization_mode == "static_map_no_relocalization"
                else "localized_from_visual"
            )
        else:
            load_state_path = None
            localization_mode = None
    initial_pose = localization.get("initial_pose")
    if not isinstance(load_state_path, str) or not Path(load_state_path).is_file():
        load_state_path = None
        localization_mode = None
    elif localization_mode not in {
        "frozen_localization_amcl_global",
        "delayed_fpfh_relocalization",
        "localized_from_visual",
        "static_map_no_relocalization",
    }:
        localization_mode = "frozen_localization_amcl_global"
    if not relocalization_enabled and load_state_path is not None:
        if localization_mode != "static_map_no_relocalization":
            raise ValueError(
                "disabled Cartographer relocalization requires "
                "static_map_no_relocalization mode"
            )
    if not isinstance(initial_pose, (list, tuple)) or len(initial_pose) != 3:
        initial_pose = None
    else:
        initial_pose = tuple(float(value) for value in initial_pose)
    use_odometry_setting = settings.get(_setting_name("use_odometry", variant_key))
    if use_odometry_setting is None:
        use_odometry_setting = settings.get("cartographer_use_odometry")
    use_odometry = (
        bool(use_odometry_setting)
        if use_odometry_setting is not None
        else _env_bool(
            _env_name("USE_ODOMETRY", variant_key),
            default=True,
            fallback_name=None if variant_key is None else _env_name("USE_ODOMETRY"),
        )
    )
    odometry_source = str(
        settings.get(
            _setting_name("odometry_source", variant_key),
            settings.get(
                "cartographer_odometry_source",
                os.environ.get(
                    _env_name("ODOMETRY_SOURCE", variant_key),
                    os.environ.get(
                        "HAB_CARTOGRAPHER_ODOMETRY_SOURCE", "simulator_evaluator"
                    ),
                ),
            ),
        )
    ).strip()
    odometry_mode = str(
        settings.get(
            _setting_name("odometry_mode", variant_key),
            settings.get(
                "cartographer_odometry_mode",
                os.environ.get(
                    _env_name("ODOMETRY_MODE", variant_key),
                    os.environ.get(
                        "HAB_CARTOGRAPHER_ODOMETRY_MODE", "authoritative"
                    ),
                ),
            ),
        )
    ).strip()
    if use_odometry and odometry_source != "simulator_evaluator":
        raise ValueError(
            "HAB_CARTOGRAPHER_ODOMETRY_SOURCE must be simulator_evaluator"
        )
    if odometry_mode not in {"cartographer_input", "authoritative"}:
        raise ValueError(
            "HAB_CARTOGRAPHER_ODOMETRY_MODE must be cartographer_input or authoritative"
        )
    if odometry_mode == "authoritative" and not use_odometry:
        raise ValueError(
            "HAB_CARTOGRAPHER_ODOMETRY_MODE=authoritative requires "
            "HAB_CARTOGRAPHER_USE_ODOMETRY=1"
        )
    memory_nodes: list[dict[str, Any]] | None = None
    memory_nodes_path = settings.get(_setting_name("memory_nodes_path", variant_key))
    if not isinstance(memory_nodes_path, str) or not memory_nodes_path:
        memory_nodes_path = settings.get("cartographer_memory_nodes_path")
    if isinstance(memory_nodes_path, str) and memory_nodes_path:
        try:
            document = json.loads(Path(memory_nodes_path).read_text(encoding="utf-8"))
            raw_nodes = document.get("nodes") if isinstance(document, dict) else None
            if isinstance(raw_nodes, dict):
                memory_nodes = [
                    {
                        "id": str(node_id),
                        "order_id": node.get("order_id"),
                        "kind": node.get("kind"),
                        "name": node.get("name"),
                        "slam_pose": node.get("slam_pose"),
                        "has_pano_images": bool(node.get("pano_images")),
                    }
                    for node_id, node in raw_nodes.items()
                    if isinstance(node, dict)
                ]
                memory_nodes.sort(
                    key=lambda node: (
                        int(node["order_id"])
                        if isinstance(node.get("order_id"), int)
                        else -1
                    )
                )
        except (OSError, json.JSONDecodeError):
            pass
    artifact_dir = (
        force_session_artifact_dir
        if force_session_artifact_dir is not None
        else root / session_id
    )
    return (
        _StreamConfig(
            artifact_dir=artifact_dir,
            width=int(
                settings.get(
                    _setting_name("width", variant_key), settings.get("width", 512)
                )
            ),
            hfov_deg=float(
                settings.get(
                    _setting_name("hfov", variant_key), settings.get("hfov", 90.0)
                )
            ),
            sensor_height_m=float(
                settings.get(
                    _setting_name("sensor_height", variant_key),
                    settings.get("sensor_height", 1.5),
                )
            ),
            beam_count=beam_count,
            range_min_m=_env_float(
                _env_name("RANGE_MIN_M", variant_key),
                0.15,
                fallback_name=None if variant_key is None else _env_name("RANGE_MIN_M"),
            ),
            range_max_m=_env_float(
                _env_name("RANGE_MAX_M", variant_key),
                8.0,
                fallback_name=None if variant_key is None else _env_name("RANGE_MAX_M"),
            ),
            scan_source=scan_source,
            load_state_path=load_state_path,
            localization_mode=localization_mode,
            initial_pose=initial_pose,
            project_row_start_fraction=_env_float(
                _env_name("PROJECT_ROW_START_FRACTION", variant_key),
                float(
                    settings.get(
                        _setting_name("project_row_start_fraction", variant_key),
                        0.5,
                    )
                ),
                fallback_name=(
                    None
                    if variant_key is None
                    else _env_name("PROJECT_ROW_START_FRACTION")
                ),
            ),
            project_pixel_stride=max(
                1,
                _env_int(
                    _env_name("PROJECT_PIXEL_STRIDE", variant_key),
                    int(
                        settings.get(
                            _setting_name("project_pixel_stride", variant_key),
                            2,
                        )
                    ),
                    fallback_name=(
                        None
                        if variant_key is None
                        else _env_name("PROJECT_PIXEL_STRIDE")
                    ),
                ),
            ),
            min_obstacle_height_m=_env_float(
                _env_name("MIN_OBSTACLE_HEIGHT_M", variant_key),
                float(
                    settings.get(
                        _setting_name("min_obstacle_height_m", variant_key),
                        0.15,
                    )
                ),
                fallback_name=(
                    None if variant_key is None else _env_name("MIN_OBSTACLE_HEIGHT_M")
                ),
            ),
            max_obstacle_height_m=_env_float(
                _env_name("MAX_OBSTACLE_HEIGHT_M", variant_key),
                float(
                    settings.get(
                        _setting_name("max_obstacle_height_m", variant_key),
                        1.25,
                    )
                ),
                fallback_name=(
                    None if variant_key is None else _env_name("MAX_OBSTACLE_HEIGHT_M")
                ),
            ),
        ),
        queue_size,
        use_odometry,
        odometry_source,
        odometry_mode,
        memory_nodes,
        worker_command,
    )


def scan_from_depth(
    depth: Any,
    *,
    sequence: int,
    timestamp_s: float,
    beam_count: int,
    range_min_m: float,
    range_max_m: float,
    hfov_deg: float = 90.0,
) -> CartographerScan:
    """Project a level-camera center band to angularly spaced ranges."""
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError("depth_sensor must be a 2D array")
    center = image.shape[0] // 2
    band = image[max(0, center - 2) : min(image.shape[0], center + 3)]
    hfov_rad = math.radians(hfov_deg)
    focal_length_px = (image.shape[1] - 1) / (2.0 * math.tan(hfov_rad / 2.0))
    if not math.isfinite(focal_length_px) or focal_length_px <= 0.0:
        raise ValueError("hfov_deg must produce a valid pinhole focal length")
    principal_x_px = (image.shape[1] - 1) / 2.0
    if beam_count <= 0:
        raise ValueError("beam_count must be positive")
    if beam_count == 1:
        beam_angles = np.array([0.0], dtype=np.float64)
    else:
        beam_angles = np.linspace(-hfov_rad / 2.0, hfov_rad / 2.0, beam_count)
    columns = np.rint(principal_x_px - np.tan(beam_angles) * focal_length_px)
    columns = np.clip(columns, 0, image.shape[1] - 1).astype(np.intp)
    ranges: list[float | None] = []
    for angle, column in zip(beam_angles, columns):
        samples = band[:, column]
        valid = samples[
            np.isfinite(samples) & (samples >= range_min_m) & (samples <= range_max_m)
        ]
        # Keep each 256-beam NDJSON record below PIPE_BUF so one FIFO write is
        # atomic. Millimeter precision is more than sufficient for this scan.
        if not valid.size:
            ranges.append(None)
            continue
        # Habitat depth is measured along the camera optical axis. A pinhole
        # camera's pixel columns are not linearly spaced in ray angle, so beams
        # are generated in Cartographer's angular frame and projected to pixels.
        ranges.append(round(float(np.min(valid)) / math.cos(angle), 3))
    return CartographerScan(
        sequence=sequence,
        timestamp_s=timestamp_s,
        ranges_m=ranges,
        hfov_deg=hfov_deg,
        camera_hfov_deg=hfov_deg,
    )


def scan_from_projected_depth_floor_band(
    depth: Any,
    *,
    sequence: int,
    timestamp_s: float,
    beam_count: int,
    range_min_m: float,
    range_max_m: float,
    hfov_deg: float = 90.0,
    sensor_height_m: float = 1.5,
    row_start_fraction: float = 0.5,
    pixel_stride: int = 2,
    min_obstacle_height_m: float = 0.15,
    max_obstacle_height_m: float = 1.25,
) -> CartographerScan:
    """Project lower-image depth points to floor-plane lidar obstacle ranges."""
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError("depth_sensor must be a 2D array")
    height_px, width_px = image.shape
    if height_px == 0 or width_px == 0:
        raise ValueError("depth_sensor must be non-empty")
    if beam_count <= 0:
        raise ValueError("beam_count must be positive")

    row_start = int(
        np.clip(float(row_start_fraction), 0.0, 1.0) * max(0, height_px - 1)
    )
    stride = max(1, int(pixel_stride))
    rows = np.arange(row_start, height_px, stride, dtype=np.float32)
    cols = np.arange(0, width_px, stride, dtype=np.float32)
    if rows.size == 0 or cols.size == 0:
        return CartographerScan(
            sequence=sequence,
            timestamp_s=timestamp_s,
            ranges_m=[None] * beam_count,
            hfov_deg=hfov_deg,
            camera_hfov_deg=hfov_deg,
        )

    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    depth_samples = image[row_grid.astype(np.intp), col_grid.astype(np.intp)]
    finite = np.isfinite(depth_samples)
    in_range = (depth_samples >= range_min_m) & (depth_samples <= range_max_m)

    hfov_rad = math.radians(hfov_deg)
    focal_length_px = (width_px - 1) / (2.0 * math.tan(hfov_rad / 2.0))
    if not math.isfinite(focal_length_px) or focal_length_px <= 0.0:
        raise ValueError("hfov_deg must produce a valid pinhole focal length")
    principal_x_px = (width_px - 1) / 2.0
    principal_y_px = (height_px - 1) / 2.0

    forward_m = depth_samples
    right_m = ((col_grid - principal_x_px) / focal_length_px) * depth_samples
    down_m = ((row_grid - principal_y_px) / focal_length_px) * depth_samples
    point_height_m = float(sensor_height_m) - down_m
    height_mask = (point_height_m >= float(min_obstacle_height_m)) & (
        point_height_m <= float(max_obstacle_height_m)
    )
    valid = finite & in_range & height_mask

    ranges: list[float | None] = [None] * beam_count
    if np.any(valid):
        left_m = -right_m[valid]
        forward_valid_m = forward_m[valid]
        planar_ranges = np.hypot(forward_valid_m, left_m)
        planar_angles = np.arctan2(left_m, forward_valid_m)
        angle_min = -hfov_rad / 2.0
        angle_max = hfov_rad / 2.0
        angle_mask = (
            (planar_angles >= angle_min)
            & (planar_angles <= angle_max)
            & (planar_ranges >= range_min_m)
            & (planar_ranges <= range_max_m)
        )
        if np.any(angle_mask):
            span = max(angle_max - angle_min, np.finfo(np.float32).eps)
            # Keep the same angular convention as scan_from_depth() and the
            # worker: index 0 is the right/negative-yaw edge, while the last
            # index is the left/positive-yaw edge.
            bins = np.rint(
                (planar_angles[angle_mask] - angle_min) / span * (beam_count - 1)
            ).astype(np.intp)
            for bin_index, planar_range in zip(bins, planar_ranges[angle_mask]):
                value = round(float(planar_range), 3)
                previous = ranges[int(bin_index)]
                if previous is None or value < previous:
                    ranges[int(bin_index)] = value

    return CartographerScan(
        sequence=sequence,
        timestamp_s=timestamp_s,
        ranges_m=ranges,
        hfov_deg=hfov_deg,
        camera_hfov_deg=hfov_deg,
    )


def odometry_from_habitat_state(
    position_value: Any, rotation_value: Any
) -> tuple[float, float, float]:
    """Convert Habitat world X/Z and quaternion into Cartographer tracking XY."""
    position = np.asarray(position_value, dtype=np.float64)
    rotation = np.asarray(rotation_value, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or rotation.shape != (4,):
        raise ValueError("agent state must contain a 3D position and quaternion")
    w, x, y, z = rotation
    forward_x = -2.0 * (x * z + y * w)
    forward_z = -(1.0 - 2.0 * (x * x + y * y))
    # Cartographer scan X is forward and Y is left; Habitat forward is -Z
    # and its local +X axis is right. Negate Habitat X so a right turn's
    # negative Cartographer yaw and its subsequent forward translation agree.
    return (
        round(float(-position[2]), 6),
        round(float(-position[0]), 6),
        # Habitat turns right from -Z toward +X, while positive Cartographer
        # yaw turns left from +X toward +Y.
        round(float(math.atan2(-forward_x, -forward_z)), 6),
    )


def _odometry_relative_to_origin(
    odometry: tuple[float, float, float], origin: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Express an odometry source in Cartographer's first-sample local frame."""
    dx = odometry[0] - origin[0]
    dy = odometry[1] - origin[1]
    cos_yaw = math.cos(origin[2])
    sin_yaw = math.sin(origin[2])
    return (
        round(cos_yaw * dx + sin_yaw * dy, 6),
        round(-sin_yaw * dx + cos_yaw * dy, 6),
        round(
            math.atan2(
                math.sin(odometry[2] - origin[2]), math.cos(odometry[2] - origin[2])
            ),
            6,
        ),
    )


def timestamp_for_step(step_count: int, dt_s: float = _DEFAULT_DT_S) -> float:
    return float(step_count) * dt_s
