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

"""Jetson-resident pixel-goal MAP-backend navigation module (four-way USB + SLAM).

The agent marks a normalized point on any USB surround view; the clicked pixel is back-projected into a world ray at snapshot time
(pinhole USB model + mount extrinsics composed with the odom/TF base pose),
the ray's ROI is lifted out of the LiDAR global map (escalating radii,
nearest-quartile front-surface heuristic), and the resulting front-edge
contact goal is handed to the ReplanningAStarPlanner — real obstacle
avoidance. Clicks without projected returns fall back to a single short
costmap-directed exploration waypoint, then return fresh views to the agent.
The agent-facing contract uses polling
(``pixelgoal_navigate`` / ``pixelgoal_status`` / ``pixelgoal_stop``), and
Rerun is only an optional visualization of the frozen goal frame and markers.

Requires the SLAM stack (VoxelGridMapper / CostMapper / ReplanningAStarPlanner)
in the blueprint. With no LiDAR data flowing the module idles:
``pixelgoal_navigate`` fails cleanly with ``no_map_data`` / ``no_odom``.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
from dimos_lcm.std_msgs import Bool
import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.voxel_grid_mapper_spec import VoxelGridMapperSpec
from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.replanning_a_star.module_spec import ReplanningAStarPlannerSpec
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC
from dimos.robot.unitree.go2.surround_camera import (
    CAMERA_DIRECTIONS,
    RERUN_IMAGE_DOWNSAMPLE,
    surround_image_health,
)
from dimos.utils.logging_config import setup_logger

from .contracts import GoalSnapshot, LocalNavError, NormalizedPoint
from .goal import build_goal_snapshot, decode_click_frame
from .live_state import LiveStateWriter
from .pixel_goal_support import GOAL_CACHE_SIZE, NavLiveSink, UnitreeMotionSpec, PixelGoalHopResult
from .pixel_goal_map import (
    BASE_FRAME,
    ELEVATED_SURFACE_HEIGHT_M,
    EXPLORATION_STEP_M,
    MIN_RADIUS_PX,
    PixelGoalMapConfig,
    contact_goal,
    directed_waypoint,
    estimate_floor_height,
    floor_intersection_limit,
    front_surface_point,
    pinhole_camera_info,
    pixel_ray,
    projection_bbox,
    projection_diagnostics,
    projection_search_radii,
    select_surface_candidate,
    segment_crosses_obstacle,
    world_to_optical_from_odom,
)

logger = setup_logger()

FRONT_TARGET_TOLERANCE_M = 0.2
STOP_POINT_TOLERANCE_M = 0.25
NEAR_CONTACT_TOLERANCE_M = 0.6
NEAR_CONTACT_HEADING_TOLERANCE_RAD = math.radians(30.0)
NEAR_CONTACT_STABLE_TIME_S = 1.5
NEAR_CONTACT_PROGRESS_EPSILON_M = 0.03
FINAL_PROGRESS_LOG_INTERVAL_S = 1.0
EXPLORATION_FOOTPRINT_SCALE = 1.1
STATUS_UPDATE_INTERVAL_S = 1.0

_SESSION_STATUS = {
    "completed": ("reached", True),
    "exploration_complete": ("needs_agent", False),
    "exploration_failed": ("needs_agent", False),
    "no_progress": ("needs_agent", False),
    "movement_stopped": ("needs_agent", False),
    "final_goal_failed": ("needs_agent", False),
    "blocked_pixel_target": ("needs_agent", False),
    "recovery_exhausted": ("needs_agent", False),
    "map_reset": ("needs_agent", False),
    "cancelled_by_agent": ("needs_agent", False),
    "deadline_exceeded": ("timeout", False),
}

MOTION_REC_HZ_ENV = "DIMOS_PIXELGOAL_MOTION_HZ"
MOTION_REC_DIR_ENV = "DIMOS_PIXELGOAL_MOTION_DIR"
MOTION_REC_JPEG_QUALITY = 80


class _HopMotionRecorder:
    """Best-effort four-way JPEG recorder gated to one hop's lifetime.

    Recording must never slow or break navigation: offers are rate-limited per
    direction and handed to a bounded queue drained by a daemon writer thread;
    any write/encode failure disables the recorder with a single warning.
    Layout: ``<root>/hop_<start_epoch>_<goal_tail8>/<direction>/<epoch_ms>.jpg``
    plus a ``meta.json`` written by :meth:`finish`.
    """

    def __init__(self, root: Path, goal_id: str, hz: float) -> None:
        tail = "".join(c for c in goal_id if c.isalnum())[-8:] or "nogoal"
        self.dir = root / f"hop_{int(time.time())}_{tail}"
        self.goal_id = goal_id
        self.started_at = time.time()
        self.saved = 0
        self._interval = 1.0 / hz
        self._last_ts: dict[str, float] = {}
        self._queue: queue.Queue[tuple[str, float, np.ndarray] | None] = queue.Queue(maxsize=16)
        self._disabled = False
        self._thread = threading.Thread(
            target=self._writer, name="pixelgoal-motion-rec", daemon=True
        )

    def start(self) -> None:
        for direction in CAMERA_DIRECTIONS:
            (self.dir / direction).mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def offer(self, direction: str, image: Image) -> None:
        if self._disabled:
            return
        ts = float(image.ts or time.time())
        last = self._last_ts.get(direction)
        if last is not None and ts - last < self._interval:
            return
        self._last_ts[direction] = ts
        try:
            rgb = np.asarray(image.to_rgb().data)
        except Exception:
            return
        try:
            self._queue.put_nowait((direction, ts, rgb))
        except queue.Full:
            pass

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            direction, ts, rgb = item
            try:
                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, MOTION_REC_JPEG_QUALITY],
                )
                if not ok:
                    raise ValueError("cv2.imencode returned False")
                (self.dir / direction / f"{round(ts * 1000)}.jpg").write_bytes(buf.tobytes())
                self.saved += 1
            except Exception as exc:
                self._disabled = True
                logger.warning("pixel-goal motion recorder disabled: %s", exc)
                return

    def finish(self, meta: dict[str, Any]) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        payload = {
            "goal_id": self.goal_id,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "frames_saved": self.saved,
            **meta,
        }
        try:
            (self.dir / "meta.json").write_text(
                json.dumps(payload, indent=1, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("pixel-goal motion recorder meta write failed: %s", exc)


class PixelGoalMapNavConfig(ModuleConfig):
    camera_info: CameraInfo | None = None  # required for camera_model="fisheye"
    camera_model: str = "usb"  # "usb" (four-way surround) | "fisheye" (Go2 front)


class PixelGoalMapNavModule(Module):
    """Frozen-frame LiDAR pixel-goal backend with the pixelgoal_* MCP contract."""

    config: PixelGoalMapNavConfig

    surround_front_image: In[Image]
    surround_right_image: In[Image]
    surround_back_image: In[Image]
    surround_left_image: In[Image]
    color_image: In[Image]  # Go2 built-in fisheye front camera (fisheye model)
    global_map: In[PointCloud2]
    global_costmap: In[OccupancyGrid]
    odom: In[PoseStamped]
    goal_reached: In[Bool]
    stop_movement: In[Bool]
    pixel_click: In[PointStamped]

    goal_request: Out[PoseStamped]
    pixel_goal_image: Out[Image]
    pixel_goal_projection: Out[PointStamped]
    pixel_goal_target: Out[PointStamped]
    pixel_goal_stop: Out[PointStamped]

    _motion: UnitreeMotionSpec  # injected at blueprint build time
    _planner_spec: ReplanningAStarPlannerSpec  # injected at blueprint build time
    _mapper_spec: VoxelGridMapperSpec | None = None  # optional (map reset)

    def __init__(self, live_dir: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._live_dir = (
            live_dir
            or os.environ.get("DIMOS_PIXELGOAL_LIVE_DIR")
            or str(Path.home() / "pixelgoal-live" / "pixelgoal-map")
        )
        self._sink = NavLiveSink(LiveStateWriter(self._live_dir))
        self._frame_lock = threading.Condition()
        self._surround_images: dict[str, Image] = {}
        self._fisheye_image: Image | None = None
        self._goal_cache: OrderedDict[str, GoalSnapshot] = OrderedDict()
        self._hop_lock = threading.Lock()
        self._hop_cancel: threading.Event | None = None
        self._hop_thread: threading.Thread | None = None
        self._last_result: dict[str, Any] | None = None
        self._terminal_images_goal_id: str | None = None
        self._config: PixelGoalMapConfig | None = None
        # --- algorithm session state (ported from -stable pixel_goal.py) ---
        self._state_lock = threading.Lock()
        self._latest_map_frame_id = ""
        self._latest_map: PointCloud2 | None = None
        self._latest_costmap: OccupancyGrid | None = None
        self._latest_odom: PoseStamped | None = None
        self._goal_published = False
        self._pixel: tuple[int, int] | None = None
        self._ray_origin: np.ndarray | None = None
        self._ray_direction: np.ndarray | None = None
        self._radius_px = 8
        self._snapshot_camera_info: CameraInfo | None = None
        self._snapshot_transform: Transform | None = None
        self._snapshot_frame_id = ""
        self._exploration_goals = 0
        self._exploration_limit_m: float | None = None
        self._unknown_frontier_goal: np.ndarray | None = None
        self._unknown_frontier_map: OccupancyGrid | None = None
        self._final_target: np.ndarray | None = None
        self._final_stop: np.ndarray | None = None
        self._contact_completion_requested = False
        self._near_contact_started_at: float | None = None
        self._near_contact_reference_distance: float | None = None
        self._last_final_progress_log = 0.0
        self._phase = "idle"
        self._session_generation = 0
        # --- hop bookkeeping for the contract layer ---
        self._session_done = threading.Event()
        self._session_end_reason = ""
        self._session_stats: dict[str, Any] = {}
        self._active_goal: GoalSnapshot | None = None
        self._snapshot_image: Image | None = None
        self._travelled_m = 0.0
        self._last_odom_xy: np.ndarray | None = None
        self._last_status_write = 0.0
        self._progress_target: np.ndarray | None = None
        self._progress_best_distance: float | None = None
        self._progress_at = 0.0
        # --- hop motion recorder (demo replay frames; 0 Hz disables) ---
        self._motion_rec_hz = float(os.environ.get(MOTION_REC_HZ_ENV, "15.0") or 0.0)
        self._motion_rec_root = Path(
            os.environ.get(MOTION_REC_DIR_ENV) or str(Path(self._live_dir).parent / "motion")
        )
        self._motion_rec: _HopMotionRecorder | None = None

    # ------------------------------------------------------------------ lifecycle
    @rpc
    def start(self) -> None:
        super().start()
        for direction in CAMERA_DIRECTIONS:
            stream = getattr(self, f"surround_{direction}_image")
            self.register_disposable(
                Disposable(
                    stream.subscribe(
                        lambda image, direction=direction: self._on_surround_image(direction, image)
                    )
                )
            )
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_fisheye_image)))
        self.register_disposable(Disposable(self.global_map.subscribe(self._on_global_map)))
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_costmap)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.goal_reached.subscribe(self._on_goal_reached)))
        self.register_disposable(Disposable(self.pixel_click.subscribe(self._on_pixel_click)))
        if self.stop_movement.transport is not None:
            self.register_disposable(
                Disposable(self.stop_movement.subscribe(self._on_stop_movement))
            )
        logger.info(
            "PixelGoalMapNavModule started (camera_model=%s, live=%s)",
            self.config.camera_model,
            self._live_dir,
        )

    @rpc
    def stop(self) -> None:
        with self._hop_lock:
            cancel = self._hop_cancel
            thread = self._hop_thread
        if cancel is not None:
            cancel.set()
        self._finish_session("module_stopped")
        if thread is not None:
            thread.join(10.0)
        super().stop()

    # ------------------------------------------------------------------ inputs
    def _on_surround_image(self, direction: str, image: Image) -> None:
        with self._frame_lock:
            self._surround_images[direction] = image
        recorder = self._motion_rec
        if recorder is not None:
            recorder.offer(direction, image)

    def _on_fisheye_image(self, image: Image) -> None:
        with self._frame_lock:
            self._fisheye_image = image

    def _surround_image(self, direction: str) -> Image | None:
        with self._frame_lock:
            return self._surround_images.get(direction)

    def _surround_snapshot(self) -> dict[str, Image]:
        with self._frame_lock:
            return dict(self._surround_images)

    def _on_pixel_click(self, point: PointStamped) -> None:
        parts = set(point.frame_id.strip("/").split("/"))
        direction = next(
            (
                candidate
                for candidate in CAMERA_DIRECTIONS
                if f"surround_{candidate}_image" in parts
            ),
            None,
        )
        if direction is None:
            logger.warning("Ignored Rerun pixel click from unknown entity", entity=point.frame_id)
            return
        image = self._surround_image(direction)
        if image is None:
            logger.warning("Ignored Rerun pixel click before camera frame", direction=direction)
            return
        width = max(1, image.width // RERUN_IMAGE_DOWNSAMPLE)
        height = max(1, image.height // RERUN_IMAGE_DOWNSAMPLE)
        if not (0.0 <= point.x <= width and 0.0 <= point.y <= height):
            logger.warning(
                "Ignored out-of-bounds Rerun pixel click",
                direction=direction,
                point=(point.x, point.y),
                image_size=(width, height),
            )
            return
        result = self.pixelgoal_navigate(
            point=[min(point.x / width, 1.0), min(point.y / height, 1.0)],
            direction=direction,
        )
        logger.info("Rerun pixel click result", direction=direction, result=result)

    def _on_global_map(self, world_map: PointCloud2) -> None:
        with self._state_lock:
            if len(world_map) == 0:
                self._latest_map_frame_id = ""
                self._latest_map = None
                return
            if world_map.frame_id:
                self._latest_map_frame_id = world_map.frame_id
                self._latest_map = world_map

    def _on_costmap(self, costmap: OccupancyGrid) -> None:
        with self._state_lock:
            self._latest_costmap = costmap

    # ------------------------------------------------------------------ deps
    def _loop_config(self) -> PixelGoalMapConfig:
        if self._config is None:
            self._config = PixelGoalMapConfig.from_env()
        return self._config

    def _camera_model(self) -> str:
        return (self.config.camera_model or "usb").strip().lower()


    # ------------------------------------------------------------------ skills
    @skill(uses=[CAP_MOVEMENT])
    def pixelgoal_navigate(
        self,
        point: list[float],
        direction: str = "front",
        goal_id: str = "",
        frame_ts: float = 0.0,
        frame_jpeg_b64: str = "",
    ) -> str:
        """Start a pixel-goal hop to a visible target (non-blocking; poll pixelgoal_status).

        MAP backend: the clicked pixel is back-projected into a world ray at
        snapshot time, lifted out of the LiDAR global map (front-surface
        heuristic), and the planner (A* + costmap, real obstacle avoidance)
        drives the dog so its front edge reaches the projected surface. Side
        and back clicks need no pre-turn: the ray lives in the map frame and
        the planner handles the heading. Clicks without LiDAR returns advance
        through ray-aligned waypoints (at most 1.5 m per leg), with up to
        0.5 m of unknown extension from reachable free space. Another unknown
        leg requires new free evidence near the previous frontier,
        rechecking the target after each successful leg. The whole hop defaults
        to 60 seconds and a 10 m exploration budget. When exploration runs out,
        return needs_agent / exploration_complete with fresh views.
        Sustained lack of net approach ends the hop with no_progress.
        Known obstacles before the stopping point yield blocked_pixel_target.

        Args:
            point: Normalized [x, y] target ((0,0)=top-left, (1,1)=bottom-right)
                on the latest frame of the chosen direction. Prefer a clear solid
                surface in the lower half (y >= 0.5); floor is optional. Never
                choose glass, reflections, sky, ceiling, horizon,
                or anything visible through a transparent barrier. Ignored when
                goal_id is given.
            direction: USB camera view containing the point: front, right, back,
                or left. (camera_model=fisheye only accepts "front".)
            goal_id: Resume a goal cached by a previous call (retry after
                needs_agent/timeout without re-marking).
            frame_ts: Dashboard-internal timestamp for the exact displayed frame;
                normal callers should omit it.
            frame_jpeg_b64: Dashboard-internal JPEG paired with frame_ts; normal
                callers should omit it.

        Returns:
            JSON string: {"ok": true, "status": "started", "goal_id": ...} or
            an error code (invalid_point / invalid_direction / no_frame /
            goal_not_found / hop_already_running).
        """
        goal_id = str(goal_id or "").strip()
        goal: GoalSnapshot | None = None
        image: Image | None = None
        if goal_id:
            goal = self._goal_cache.get(goal_id)
            if goal is None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "goal_not_found",
                        "error": f"goal_id {goal_id!r} is not cached on the Jetson pixel-goal module",
                    }
                )
            self._goal_cache.move_to_end(goal_id)
        else:
            direction = str(direction or "front").strip().lower()
            if direction not in CAMERA_DIRECTIONS:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "invalid_direction",
                        "error": f"direction must be one of {CAMERA_DIRECTIONS}",
                    }
                )
            if frame_jpeg_b64 and self._camera_model() != "usb":
                return json.dumps(
                    {
                        "ok": False,
                        "code": "invalid_goal_frame",
                        "error": "an injected dashboard frame is only valid in USB camera mode",
                    }
                )
            if frame_jpeg_b64:
                try:
                    ts = float(frame_ts)
                    if not math.isfinite(ts) or ts <= 0:
                        raise LocalNavError(
                            "invalid_goal_frame", "clicked camera frame has no valid timestamp"
                        )
                    selected = Image.from_numpy(
                        decode_click_frame(frame_jpeg_b64),
                        format=ImageFormat.RGB,
                        frame_id=f"go2/{direction}/camera_optical",
                        ts=ts,
                    )
                except LocalNavError as exc:
                    return json.dumps({"ok": False, "code": exc.code, "error": exc.message})
            elif self._camera_model() == "fisheye":
                if direction != "front":
                    return json.dumps(
                        {
                            "ok": False,
                            "code": "invalid_direction",
                            "error": "camera_model=fisheye only supports the front (Go2) camera",
                        }
                    )
                with self._frame_lock:
                    selected = self._fisheye_image
            else:
                selected = self._surround_image(direction)
                health = surround_image_health(self._surround_snapshot())
                if direction in health["stale"]:
                    return json.dumps(
                        {
                            "ok": False,
                            "code": "stale_frame",
                            "error": f"{direction} camera frame is stale",
                            "camera_health": health,
                        }
                    )
            frame = None if selected is None else np.asarray(selected.to_rgb().data)
            if frame is None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "no_frame",
                        "error": f"no {direction} camera frame yet",
                    }
                )
            try:
                normalized = NormalizedPoint.parse(point)
            except LocalNavError as exc:
                return json.dumps({"ok": False, "code": exc.code, "error": exc.message})
            goal = build_goal_snapshot(
                f"pixelgoal-map-{direction}:{time.time():.3f}",
                frame,
                normalized,
                direction=direction,
            )
            self._cache_goal(goal)
            if goal.marked_rgb is not None:
                self._sink.write_frame(goal.marked_rgb, goal=True)
            image = selected
            if image is not None:
                self.pixel_goal_image.publish(image)

        with self._hop_lock:
            if self._hop_cancel is not None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "hop_already_running",
                        "error": "a pixel-goal hop is already running; poll pixelgoal_status "
                        "or pixelgoal_stop it",
                    }
                )
            cancel = threading.Event()
            self._hop_cancel = cancel

        self._sink.event(
            "hop_start",
            {
                "backend": "map",
                "goal_id": goal.goal_id,
                "point": list(goal.point.as_tuple()),
                "direction": goal.direction or "front",
                "resumed": bool(goal_id),
            },
        )
        self._sink.write_status(
            {
                "status": "starting",
                "hop_running": True,
                "goal_id": goal.goal_id,
                "direction": goal.direction or "front",
                "backend": "map",
            }
        )
        thread = threading.Thread(
            target=self._run_hop, args=(goal, image, cancel), name="pixelgoal-map-hop", daemon=True
        )
        with self._hop_lock:
            self._hop_thread = thread
            self._terminal_images_goal_id = None
        thread.start()
        return json.dumps(
            {
                "ok": True,
                "status": "started",
                "goal_id": goal.goal_id,
                "direction": goal.direction or "front",
                "backend": "map",
            }
        )

    @skill
    def pixelgoal_status(self, include_images: bool = True) -> str | list[dict[str, Any]]:
        """Poll the running (or most recent) pixel-goal hop.

        Args:
            include_images: Attach all four USB views once the hop is finished.

        Returns:
            JSON string with hop_running, the latest status fields (status,
            goal_id, phase, goals_published, target_kind, distance_to_target_m,
            travelled_m, elapsed), the finished hop's last_result, and the
            tail of the navigation event stream.
        """
        with self._hop_lock:
            running = self._hop_cancel is not None
        snap = self._sink.snapshot()
        status = dict(snap["status"] or {"status": None})
        status["hop_running"] = running
        payload: dict[str, Any] = {
            "ok": True,
            "backend": "map",
            "hop_running": running,
            "status": status,
            "last_result": self._last_result,
            "events": snap["events"],
        }
        if running or self._last_result is None or not include_images:
            return json.dumps(payload, default=str)
        goal_id = str(self._last_result.get("goal_id") or "")
        with self._hop_lock:
            if goal_id and self._terminal_images_goal_id == goal_id:
                return json.dumps(payload, default=str)
        images = self._surround_snapshot()
        health = surround_image_health(images)
        payload["camera_health"] = health
        fresh = set(health["fresh"])
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, default=str)}]
        for direction in CAMERA_DIRECTIONS:
            image = images.get(direction)
            if image is None or direction not in fresh:
                continue
            content.append({"type": "text", "text": direction})
            content.extend(image.agent_encode())
        if health["ok"]:
            with self._hop_lock:
                self._terminal_images_goal_id = goal_id
        return content

    @skill
    def pixelgoal_stop(self) -> str:
        """Abort the running hop: cooperative cancel + planner cancel + hard stop.

        Stop tools need no movement authority; always safe to call. The hop
        unwinds promptly and pixelgoal_status flips to hop_running=false with
        status needs_agent/cancelled_by_agent.

        Returns:
            JSON string: {"ok": true, "status": cancel_requested|no_hop_running,
            "robot_stop": sent|failed}.
        """
        with self._hop_lock:
            cancel = self._hop_cancel
        if cancel is not None:
            cancel.set()
        self._abort_session("cancelled_by_agent")
        robot_stop = "sent"
        try:
            self._motion.stop_movement()
        except Exception as exc:
            robot_stop = f"failed: {type(exc).__name__}: {exc}"
        return json.dumps(
            {
                "ok": True,
                "status": "cancel_requested" if cancel is not None else "no_hop_running",
                "robot_stop": robot_stop,
            }
        )

    @skill(requires_modules=["VoxelGridMapper", "CostMapper", "ReplanningAStarPlanner"])
    def reset_pixel_goal_map(self) -> str:
        """Stop navigation, clear the runtime SLAM map, and wait for a fresh map."""
        self._abort_session("map_reset")
        with self._state_lock:
            self._latest_map_frame_id = ""
            self._latest_map = None
            self._latest_costmap = None
        planner_reset = self._planner_spec.reset_navigation_map()
        mapper = self._mapper_spec
        mapper_reset = mapper.reset_voxel_map(timeout_s=5.0) if mapper is not None else False
        if not planner_reset or not mapper_reset:
            return (
                "Pixel goal map reset incomplete: navigation was stopped, but the mapper did not "
                "confirm a fresh map within 5 seconds."
            )
        return "Pixel goal map reset; waiting for a fresh map and camera frame."

    # ------------------------------------------------------------------ hop thread
    def _cache_goal(self, goal: GoalSnapshot) -> None:
        self._goal_cache[goal.goal_id] = goal
        self._goal_cache.move_to_end(goal.goal_id)
        while len(self._goal_cache) > GOAL_CACHE_SIZE:
            self._goal_cache.popitem(last=False)

    def _start_motion_recorder(self, goal: GoalSnapshot) -> None:
        if self._motion_rec_hz <= 0:
            return
        try:
            recorder = _HopMotionRecorder(self._motion_rec_root, goal.goal_id, self._motion_rec_hz)
            recorder.start()
        except Exception as exc:
            logger.warning("pixel-goal motion recorder unavailable: %s", exc)
            return
        self._motion_rec = recorder
        logger.info(
            "pixel-goal motion recorder on: %s (%.1f Hz)", recorder.dir, self._motion_rec_hz
        )

    def _finish_motion_recorder(self, goal: GoalSnapshot) -> None:
        recorder, self._motion_rec = self._motion_rec, None
        if recorder is None:
            return
        try:
            result = self._last_result or {}
            recorder.finish(
                {
                    "direction": goal.direction or "front",
                    "point": list(goal.point.as_tuple()),
                    "status": str(result.get("status") or ""),
                    "duration_s": result.get("duration_s"),
                    "travelled_m": result.get("travelled_m"),
                    "record_hz": self._motion_rec_hz,
                }
            )
            logger.info(
                "pixel-goal motion recorder off: %s frames=%s", recorder.dir, recorder.saved
            )
        except Exception as exc:
            logger.warning("pixel-goal motion recorder finish failed: %s", exc)

    def _run_hop(self, goal: GoalSnapshot, image: Image | None, cancel: threading.Event) -> None:
        started_at = time.monotonic()
        with self._state_lock:
            self._active_goal = goal
            # A new hop can fail geometry validation before _start_session resets
            # its state. Do not attach the previous hop's terminal metrics.
            self._session_stats = {}
            self._travelled_m = 0.0
        try:
            self._start_motion_recorder(goal)
            self._start_session(goal, image, cancel)
            result = self._await_session(goal, cancel, started_at)
        except LocalNavError as exc:
            result = PixelGoalHopResult(
                ok=False,
                status="needs_agent",
                goal_id=goal.goal_id,
                point=goal.point.as_tuple(),
                duration_s=time.monotonic() - started_at,
                needs_agent_reason=exc.code,
                error_code=exc.code,
                error=exc.message,
            )
        except Exception as exc:
            result = PixelGoalHopResult(
                ok=False,
                status="error",
                goal_id=goal.goal_id,
                point=goal.point.as_tuple(),
                duration_s=time.monotonic() - started_at,
                error_code="hop_internal_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        # Publish the terminal artifacts BEFORE clearing the running flag (same
        # rationale as the template backend: no stale "not running" gap).
        self._last_result = result.as_dict()
        self._last_result["direction"] = goal.direction or "front"
        self._last_result["backend"] = "map"
        with self._state_lock:
            self._last_result.update(self._session_stats)
        self._sink.event("hop_end", self._last_result)
        self._finish_motion_recorder(goal)
        self._sink.write_status({**self._last_result, "hop_running": False})
        with self._state_lock:
            self._active_goal = None
        with self._hop_lock:
            self._hop_cancel = None
        logger.info(
            "pixel-goal map hop done: status=%s reason=%s",
            result.status,
            result.needs_agent_reason or "-",
        )

    # ------------------------------------------------------------------ session start
    def _base_pose_at(self, frame_ts: float, map_frame_id: str) -> PoseStamped | None:
        """Base pose in the map frame: TF at the frame ts when available, else odom."""
        try:
            transform = self.tf.get(map_frame_id, BASE_FRAME, frame_ts, time_tolerance=0.75)
        except Exception:
            transform = None
        if transform is not None:
            return transform.to_pose()
        with self._state_lock:
            return self._latest_odom

    def _snapshot_geometry(
        self, goal: GoalSnapshot, frame: Image
    ) -> tuple[CameraInfo, Transform, tuple[int, int]]:
        """Camera model, snapshot-time world→optical transform, and the click pixel.

        Both camera models yield the pinhole-equivalent CameraInfo + pixel used
        for every later ROI re-projection of the frozen ray.
        """
        direction = goal.direction or "front"
        with self._state_lock:
            map_frame_id = self._latest_map_frame_id
        if not map_frame_id:
            raise LocalNavError("no_map_data", "no LiDAR global map yet (SLAM not ready)")
        pixel = goal.point.to_pixel(frame.width, frame.height)
        frame_ts = float(frame.ts or time.time())

        if self._camera_model() == "fisheye":
            raw_info = self.config.camera_info
            if raw_info is None:
                raise LocalNavError(
                    "no_camera_info", "camera_model=fisheye requires config.camera_info"
                )
            if (frame.width, frame.height) != (raw_info.width, raw_info.height):
                raise LocalNavError(
                    "camera_size_mismatch",
                    f"camera image {frame.width}x{frame.height} does not match calibration "
                    f"{raw_info.width}x{raw_info.height}",
                )
            model_info = pinhole_camera_info(raw_info)
            world_to_optical = self.tf.get(
                "camera_optical", map_frame_id, frame_ts, time_tolerance=1.0
            )
            if world_to_optical is None:
                raise LocalNavError(
                    "no_tf", "no TF from the map frame to camera_optical at the snapshot time"
                )
            # Warp the clicked pixel into the rectified pinhole frame so the
            # ray and every ROI projection share the pinhole model.
            distorted = np.array([[[float(pixel[0]), float(pixel[1])]]], dtype=np.float64)
            rectified = cv2.fisheye.undistortPoints(
                distorted,
                np.asarray(raw_info.K, dtype=np.float64).reshape(3, 3),
                np.asarray(raw_info.D, dtype=np.float64),
                R=np.asarray(raw_info.R, dtype=np.float64).reshape(3, 3),
                P=np.asarray(model_info.K, dtype=np.float64).reshape(3, 3),
            )
            ray_pixel = (round(float(rectified[0, 0, 0])), round(float(rectified[0, 0, 1])))
            return model_info, world_to_optical, ray_pixel

        rig = self._loop_config().rig
        odom = self._base_pose_at(frame_ts, map_frame_id)
        if odom is None:
            raise LocalNavError("no_odom", "no odometry snapshot for the frozen frame")
        model_info = rig.pinhole(frame.width, frame.height).to_camera_info(
            rig.optical_frame(direction)
        )
        world_to_optical = world_to_optical_from_odom(
            odom,
            rig.mount(direction),
            rig.optical_frame(direction),
            rig.base_height_m,
        )
        return model_info, world_to_optical, pixel

    def _start_session(
        self, goal: GoalSnapshot, image: Image | None, cancel: threading.Event
    ) -> None:
        config = self._loop_config()
        direction = goal.direction or "front"
        frame = image
        if frame is None:
            # goal_id resume: re-project against the freshest frame of the goal's
            # direction (the frozen normalized point is what persists).
            if self._camera_model() == "fisheye":
                with self._frame_lock:
                    frame = self._fisheye_image
            else:
                frame = self._surround_image(direction)
        if frame is None:
            raise LocalNavError("no_frame", f"no {direction} camera frame at hop start")

        camera_info, world_to_optical, ray_pixel = self._snapshot_geometry(goal, frame)
        map_frame_id = world_to_optical.child_frame_id

        try:
            world_map = self.global_map.get_next(timeout=config.map_timeout_s)
        except Exception as exc:
            raise LocalNavError("no_map_data", f"fresh map data unavailable ({exc})") from exc
        if not world_map.frame_id:
            raise LocalNavError("no_map_data", "global map has no frame_id")
        if world_map.frame_id != map_frame_id:
            raise LocalNavError(
                "map_frame_mismatch",
                f"current map frame {world_map.frame_id!r} does not match "
                f"snapshot frame {map_frame_id!r}",
            )
        if cancel.is_set():
            raise LocalNavError("cancelled_by_agent", "cancelled before the session started")

        try:
            ray_origin, ray_direction = pixel_ray(
                ray_pixel[0], ray_pixel[1], camera_info, world_to_optical
            )
        except Exception as exc:
            raise LocalNavError("projection_failed", f"ray back-projection failed ({exc})") from exc

        radius_px = config.radius_px
        if not MIN_RADIUS_PX <= radius_px <= 100:
            raise LocalNavError("invalid_radius", f"radius_px must be {MIN_RADIUS_PX}-100")
        with self._state_lock:
            self._session_generation += 1
            generation = self._session_generation
            self._goal_published = True
            self._pixel = ray_pixel
            self._ray_origin = ray_origin
            self._ray_direction = ray_direction
            self._radius_px = radius_px
            self._snapshot_camera_info = camera_info
            self._snapshot_transform = world_to_optical
            self._snapshot_frame_id = map_frame_id
            self._exploration_goals = 0
            self._unknown_frontier_goal = None
            self._unknown_frontier_map = None
            self._final_target = None
            self._final_stop = None
            self._contact_completion_requested = False
            self._near_contact_started_at = None
            self._near_contact_reference_distance = None
            self._phase = "starting"
            self._session_done = threading.Event()
            self._session_end_reason = ""
            self._session_stats = {}
            self._snapshot_image = frame
            self._travelled_m = 0.0
            self._last_odom_xy = None
        if cancel.is_set():
            with self._state_lock:
                self._goal_published = False
                self._phase = "idle"
            raise LocalNavError("cancelled_by_agent", "cancelled before the first goal")

        self._sink.event(
            "ray_projected",
            {
                "pixel": list(ray_pixel),
                "direction": direction,
                "camera_model": self._camera_model(),
                "ray_origin": ray_origin.round(3).tolist(),
                "ray_direction": ray_direction.round(4).tolist(),
                "map_frame_id": map_frame_id,
            },
        )

        map_points, _ = world_map.as_numpy()
        floor_z = estimate_floor_height(map_points, ray_origin)
        exploration_limit_m = floor_intersection_limit(ray_origin, ray_direction, floor_z)
        with self._state_lock:
            if generation != self._session_generation or not self._goal_published:
                raise LocalNavError("cancelled_by_agent", "the session was cancelled")
            self._exploration_limit_m = exploration_limit_m

        resolved = self._resolve_pixel_target(world_map)
        if resolved is not None:
            target, target_kind = resolved
            if self._resolved_goal_blocked(target):
                self._finish_session("blocked_pixel_target", expected_generation=generation)
                return
            if self._publish_resolved_goal(target, target_kind, generation=generation):
                return
        if self._publish_exploration_goal(expected_generation=generation):
            return
        with self._state_lock:
            self._goal_published = False
            self._phase = "idle"
        if cancel.is_set():
            raise LocalNavError("cancelled_by_agent", "cancelled during goal projection")
        raise LocalNavError(
            "no_reachable_target", "no projected surface or reachable free waypoint"
        )

    # ------------------------------------------------------------------ target resolution (ported)
    def _resolve_pixel_target(self, world_map: PointCloud2) -> tuple[np.ndarray, str] | None:
        with self._state_lock:
            camera_info = self._snapshot_camera_info
            transform = self._snapshot_transform
            pixel = self._pixel
            radius_px = self._radius_px
            ray_direction = self._ray_direction
            image = self._snapshot_image
        if camera_info is None or transform is None or pixel is None or ray_direction is None:
            return None
        if image is None:
            image = Image.from_numpy(
                np.zeros((camera_info.height, camera_info.width, 3), dtype=np.uint8)
            )

        map_points, _ = world_map.as_numpy()
        camera_position = transform.inverse().translation.to_numpy()
        floor_z = estimate_floor_height(map_points, camera_position)
        radii = projection_search_radii(camera_info, radius_px)
        config = self._loop_config()
        candidates: list[dict[str, Any]] = []
        for radius in radii:
            if radius < MIN_RADIUS_PX:
                continue
            bbox = projection_bbox(pixel[0], pixel[1], radius, camera_info, camera_info)
            detection = Detection3DPC.from_2d(
                Detection2DBBox(
                    bbox=bbox,
                    track_id=-1,
                    class_id=-1,
                    confidence=1.0,
                    name="pixel_goal",
                    ts=float(image.ts or time.time()),
                    image=image,
                ),
                world_pointcloud=world_map,
                camera_info=camera_info,
                world_to_optical_transform=transform,
                filters=[],
                bbox_margin_px=0,
            )
            if detection is None:
                continue
            points, _ = detection.pointcloud.as_numpy()
            try:
                target = front_surface_point(points, camera_position)
            except ValueError:
                continue
            distance = float(np.linalg.norm(target - camera_position))
            if not (
                math.isfinite(distance)
                and config.min_target_distance_m <= distance <= config.max_target_distance_m
            ):
                continue
            surface_height = float(target[2] - floor_z)
            target_kind = (
                "elevated_surface"
                if surface_height >= ELEVATED_SURFACE_HEIGHT_M
                else "ground_or_low_surface"
            )
            diagnostics = projection_diagnostics(
                points,
                camera_position,
                ray_direction,
                target,
            )
            candidate = {
                "radius_px": radius,
                "points": len(points),
                "target": target,
                "distance_m": distance,
                "surface_height_m": surface_height,
                "target_kind": target_kind,
                "ray_depth_m": float(diagnostics["target_ray_depth_m"]),
                "ray_miss_m": float(diagnostics["target_ray_miss_m"]),
            }
            candidates.append(candidate)
            logger.info(
                "Pixel surface candidate",
                pixel=pixel,
                radius_px=radius,
                points=len(points),
                target=target.round(3).tolist(),
                target_kind=target_kind,
            )

        if not candidates:
            return None
        selected = select_surface_candidate(candidates)
        target = np.asarray(selected["target"], dtype=np.float64)
        logger.info(
            "Resolved pixel surface",
            pixel=pixel,
            radius_px=selected["radius_px"],
            candidate_count=len(candidates),
            points=selected["points"],
            floor_z=round(floor_z, 3),
            target=target.tolist(),
            distance_m=selected["distance_m"],
            target_kind=selected["target_kind"],
        )
        return target, str(selected["target_kind"])

    # ------------------------------------------------------------------ goal publishing (ported)
    def _resolved_goal_blocked(self, target: np.ndarray) -> bool:
        with self._state_lock:
            costmap = self._latest_costmap
            ray_origin = self._ray_origin
        if costmap is None or ray_origin is None:
            return False
        inflated = simple_inflate(
            costmap, self.config.g.robot_width * EXPLORATION_FOOTPRINT_SCALE / 2
        )
        stop = contact_goal(
            target, ray_origin, offset_m=self._loop_config().robot_front_offset_m
        )
        return segment_crosses_obstacle(inflated, ray_origin, stop)

    def _publish_goal(self, target: np.ndarray, *, phase: str, generation: int) -> bool:
        with self._state_lock:
            if generation != self._session_generation or not self._goal_published:
                return False
            ray_direction = self._ray_direction
            frame_id = self._snapshot_frame_id
        assert ray_direction is not None
        yaw = math.atan2(float(ray_direction[1]), float(ray_direction[0]))
        goal = PoseStamped(
            position=target.tolist(),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
            ts=time.time(),
            frame_id=frame_id,
        )
        with self._state_lock:
            if generation != self._session_generation or not self._goal_published:
                return False
            self._phase = phase
            self._progress_target = target[:2].copy()
            self._progress_best_distance = (
                float(np.linalg.norm(target[:2] - self._latest_odom.position.to_numpy()[:2]))
                if self._latest_odom is not None
                else None
            )
            self._progress_at = time.monotonic()
            self._session_stats["goals_published"] = (
                int(self._session_stats.get("goals_published", 0)) + 1
            )
            self.goal_request.publish(goal)
        self._sink.event(
            "goal_publish", {"phase": phase, "goal": [round(v, 3) for v in target.tolist()]}
        )
        self._write_nav_status()
        logger.info("Published pixel navigation goal", phase=phase, goal=str(goal))
        return True

    def _publish_resolved_goal(
        self,
        target: np.ndarray,
        target_kind: str,
        *,
        generation: int,
    ) -> bool:
        with self._state_lock:
            ray_origin = self._ray_origin
        if ray_origin is None:
            return False
        projected_target = np.asarray(target, dtype=np.float64).copy()
        navigation_target = projected_target
        goal = contact_goal(
            navigation_target, ray_origin, offset_m=self._loop_config().robot_front_offset_m
        )
        if not self._publish_goal(goal, phase="final", generation=generation):
            return False
        with self._state_lock:
            frame_id = self._snapshot_frame_id
            self._final_target = navigation_target.copy()
            self._final_stop = goal.copy()
            self._contact_completion_requested = False
            self._near_contact_started_at = None
            self._near_contact_reference_distance = None
            self._last_final_progress_log = 0.0
            self._session_stats.update(
                {
                    "target_kind": target_kind,
                    "target": [round(float(v), 3) for v in navigation_target.tolist()],
                    "stop_point": [round(float(v), 3) for v in goal.tolist()],
                }
            )
        marker_ts = time.time()
        self.pixel_goal_projection.publish(
            PointStamped(
                x=float(projected_target[0]),
                y=float(projected_target[1]),
                z=float(projected_target[2]),
                ts=marker_ts,
                frame_id=frame_id,
            )
        )
        self.pixel_goal_target.publish(
            PointStamped(
                x=float(navigation_target[0]),
                y=float(navigation_target[1]),
                z=float(navigation_target[2]),
                ts=marker_ts,
                frame_id=frame_id,
            )
        )
        self.pixel_goal_stop.publish(
            PointStamped(
                x=float(goal[0]),
                y=float(goal[1]),
                z=float(goal[2]),
                ts=marker_ts,
                frame_id=frame_id,
            )
        )
        self._sink.event(
            "surface_resolved",
            {
                "target_kind": target_kind,
                "target": [round(float(v), 3) for v in navigation_target.tolist()],
                "stop_point": [round(float(v), 3) for v in goal.tolist()],
            },
        )
        return True

    def _publish_exploration_goal(self, *, expected_generation: int | None = None) -> bool:
        with self._state_lock:
            if not self._goal_published or (
                expected_generation is not None and expected_generation != self._session_generation
            ):
                return False
            costmap = self._latest_costmap
            odom = self._latest_odom
            ray_origin = self._ray_origin
            ray_direction = self._ray_direction
            generation = self._session_generation
            exploration_limit_m = self._exploration_limit_m
        if costmap is None or ray_origin is None or ray_direction is None:
            return False
        origin = odom.position.to_numpy() if odom is not None else ray_origin
        horizontal = np.asarray(ray_direction[:2], dtype=np.float64)
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-6:
            return False
        horizontal /= horizontal_norm
        config = self._loop_config()
        travelled = max(0.0, float((origin[:2] - ray_origin[:2]) @ horizontal))
        max_forward_m = config.max_exploration_distance_m
        if exploration_limit_m is not None:
            max_forward_m = min(max_forward_m, exploration_limit_m)
        remaining = max_forward_m - travelled
        step_m = min(config.exploration_step_m, remaining, EXPLORATION_STEP_M)
        if step_m < 0.5:
            logger.info(
                "Pixel exploration stopped because remaining forward distance is too short",
                travelled_m=round(travelled, 3),
                max_forward_m=round(max_forward_m, 3),
            )
            return False
        footprint_diameter_m = self.config.g.robot_width * EXPLORATION_FOOTPRINT_SCALE
        waypoint = directed_waypoint(
            costmap,
            origin,
            ray_direction,
            max_step_m=step_m,
            footprint_diameter_m=footprint_diameter_m,
        )
        if waypoint is None:
            logger.warning("No body-reachable pixel exploration waypoint found")
            return False
        unknown = costmap.cell_value(Vector3(*waypoint)) == CostValues.UNKNOWN
        with self._state_lock:
            previous_goal = self._unknown_frontier_goal
            previous_map = self._unknown_frontier_map
        if unknown and previous_goal is not None and previous_map is not None:
            # Only new free evidence near the previous frontier permits another
            # unknown step. A timestamp refresh or distant change is insufficient.
            expanded = False
            if previous_map.frame_id == costmap.frame_id:
                offsets = np.arange(-0.5, 0.51, costmap.resolution)
                for dx in offsets:
                    for dy in offsets:
                        point = Vector3(
                            float(previous_goal[0] + dx), float(previous_goal[1] + dy), 0.0
                        )
                        if (
                            previous_map.cell_value(point) == CostValues.UNKNOWN
                            and costmap.cell_value(point) == CostValues.FREE
                        ):
                            expanded = True
                            break
                    if expanded:
                        break
            if not expanded:
                logger.warning("Unknown frontier did not gain known-free coverage")
                return False
        with self._state_lock:
            if generation != self._session_generation or not self._goal_published:
                return False
            self._exploration_goals += 1
            self._unknown_frontier_goal = waypoint.copy() if unknown else None
            self._unknown_frontier_map = (
                OccupancyGrid(
                    grid=costmap.grid.copy(),
                    resolution=costmap.resolution,
                    origin=costmap.origin,
                    frame_id=costmap.frame_id,
                    ts=costmap.ts,
                )
                if unknown
                else None
            )
        self._sink.event(
            "exploration_waypoint",
            {"waypoint": waypoint.round(3).tolist(), "step_m": round(step_m, 3)},
        )
        return self._publish_goal(waypoint, phase="explore", generation=generation)

    # ------------------------------------------------------------------ session progress (ported)
    def _on_odom(self, odom: PoseStamped) -> None:
        with self._state_lock:
            self._latest_odom = odom
            active = self._goal_published
            phase = self._phase
            target = self._final_target
            stop = self._final_stop
            completion_requested = self._contact_completion_requested
            generation = self._session_generation
            # Travelled-distance bookkeeping for the hop result.
            xy = odom.position.to_numpy()[:2]
            if active and self._last_odom_xy is not None:
                self._travelled_m += float(np.linalg.norm(xy - self._last_odom_xy))
            if active:
                self._last_odom_xy = xy
        if not active:
            return
        now = time.monotonic()
        if now - self._last_status_write >= STATUS_UPDATE_INTERVAL_S:
            self._last_status_write = now
            self._write_nav_status()
        if phase != "final" or target is None or completion_requested:
            return

        center = xy
        robot_yaw = float(odom.orientation.euler[2])
        heading = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float64)
        front = center + heading * self._loop_config().robot_front_offset_m
        front_distance = float(np.linalg.norm(target[:2] - front))
        stop_distance = float(np.linalg.norm(stop[:2] - center)) if stop is not None else None
        target_delta = target[:2] - center
        target_bearing = math.atan2(float(target_delta[1]), float(target_delta[0]))
        heading_error = abs(
            math.atan2(
                math.sin(target_bearing - robot_yaw),
                math.cos(target_bearing - robot_yaw),
            )
        )
        with self._state_lock:
            if generation != self._session_generation or self._phase != "final":
                return
            contact_distance = max(
                front_distance,
                stop_distance if stop_distance is not None else front_distance,
            )
            near_contact = (
                front_distance <= NEAR_CONTACT_TOLERANCE_M
                and stop_distance is not None
                and stop_distance <= NEAR_CONTACT_TOLERANCE_M
                and heading_error <= NEAR_CONTACT_HEADING_TOLERANCE_RAD
            )
            if near_contact:
                if (
                    self._near_contact_started_at is None
                    or self._near_contact_reference_distance is None
                    or abs(contact_distance - self._near_contact_reference_distance)
                    > NEAR_CONTACT_PROGRESS_EPSILON_M
                ):
                    self._near_contact_started_at = now
                    self._near_contact_reference_distance = contact_distance
                near_contact_elapsed = now - self._near_contact_started_at
            else:
                self._near_contact_started_at = None
                self._near_contact_reference_distance = None
                near_contact_elapsed = 0.0
            if now - self._last_final_progress_log >= FINAL_PROGRESS_LOG_INTERVAL_S:
                self._last_final_progress_log = now
                should_log = True
            else:
                should_log = False
        if should_log:
            logger.info(
                "Pixel goal final approach progress",
                robot_center=center.round(3).tolist(),
                target=target[:2].round(3).tolist(),
                front_target_distance_m=round(front_distance, 3),
                center_stop_distance_m=(
                    round(stop_distance, 3) if stop_distance is not None else None
                ),
                near_contact_elapsed_s=round(near_contact_elapsed, 2),
            )
        front_reached = front_distance <= FRONT_TARGET_TOLERANCE_M
        stop_reached = stop_distance is not None and stop_distance <= STOP_POINT_TOLERANCE_M
        stable_near_contact = near_contact_elapsed >= NEAR_CONTACT_STABLE_TIME_S
        if not front_reached and not stop_reached and not stable_near_contact:
            return

        if stop_reached:
            completion_reason = "stop_point"
        elif front_reached:
            completion_reason = "front_edge"
        else:
            completion_reason = "stable_near_contact"

        with self._state_lock:
            if (
                generation != self._session_generation
                or self._phase != "final"
                or self._contact_completion_requested
            ):
                return
            self._contact_completion_requested = True
        self._sink.event(
            "front_edge_contact",
            {"criterion": completion_reason, "front_distance_m": round(front_distance, 3)},
        )

        def complete() -> None:
            if not self._planner_spec.complete_goal():
                logger.warning("Planner did not confirm pixel front-edge completion")

        threading.Thread(
            target=complete,
            daemon=True,
            name="pixel-goal-contact-completion",
        ).start()

    def _on_goal_reached(self, result: Bool) -> None:
        with self._state_lock:
            active = self._goal_published
            phase = self._phase
            generation = self._session_generation
            if not active or phase not in ("explore", "final"):
                return
            # Claim this completion before spawning work, so duplicate callbacks
            # cannot publish multiple continuation goals for the same leg.
            self._phase = "resolving"

        def advance() -> None:
            with self._state_lock:
                if generation != self._session_generation or not self._goal_published:
                    return
                world_map = self._latest_map
            if phase == "final":
                self._finish_session(
                    "completed" if result.data else "final_goal_failed",
                    expected_generation=generation,
                )
                return
            if not result.data:
                self._finish_session("exploration_failed", expected_generation=generation)
                return
            resolved = self._resolve_pixel_target(world_map) if world_map is not None else None
            if resolved is not None:
                target, target_kind = resolved
                if self._resolved_goal_blocked(target):
                    self._finish_session("blocked_pixel_target", expected_generation=generation)
                    return
                if self._publish_resolved_goal(target, target_kind, generation=generation):
                    return
            if self._publish_exploration_goal(expected_generation=generation):
                return
            self._finish_session("exploration_complete", expected_generation=generation)

        threading.Thread(target=advance, daemon=True, name="pixel-goal-recovery").start()

    def _on_stop_movement(self, stopped: Bool) -> None:
        if stopped.data:
            with self._state_lock:
                active = self._goal_published
            if active:
                self._finish_session("movement_stopped")

    # ------------------------------------------------------------------ session end
    def _reset_session(self, *, reason: str) -> None:
        with self._state_lock:
            self._session_generation += 1
            self._goal_published = False
            self._pixel = None
            self._ray_origin = None
            self._ray_direction = None
            self._snapshot_camera_info = None
            self._snapshot_transform = None
            self._snapshot_frame_id = ""
            self._snapshot_image = None
            self._exploration_goals = 0
            self._exploration_limit_m = None
            self._unknown_frontier_goal = None
            self._unknown_frontier_map = None
            self._final_target = None
            self._final_stop = None
            self._contact_completion_requested = False
            self._near_contact_started_at = None
            self._near_contact_reference_distance = None
            self._last_final_progress_log = 0.0
            self._phase = "idle"
            self._progress_target = None
            self._progress_best_distance = None
            self._progress_at = 0.0
        logger.info("Pixel goal map session reset", reason=reason)

    def _finish_session(self, reason: str, *, expected_generation: int | None = None) -> None:
        with self._state_lock:
            if self._session_done.is_set() or (
                expected_generation is not None and expected_generation != self._session_generation
            ):
                return
            self._session_end_reason = reason
            # Freeze the center-to-stop distance before _reset_session clears
            # the resolved goal. Later odometry must not change this result.
            stop = self._final_stop
            odom = self._latest_odom
            if stop is not None and odom is not None:
                distance = float(np.linalg.norm(stop[:2] - odom.position.to_numpy()[:2]))
                if math.isfinite(distance):
                    self._session_stats["final_distance_m"] = round(distance, 3)
                    self._session_stats["final_distance_basis"] = "robot_center_to_stop_point_xy"
        self._reset_session(reason=reason)
        self._session_done.set()

    def _abort_session(self, reason: str) -> None:
        # Invalidate callbacks before cancel_goal can emit goal_reached(False).
        with self._state_lock:
            if self._session_done.is_set():
                return
            self._goal_published = False
            self._session_generation += 1
        try:
            cancelled = self._planner_spec.cancel_goal()
        except Exception as exc:
            logger.warning("Planner cancel failed during %s: %s", reason, exc)
            cancelled = False
        if not cancelled:
            try:
                self._motion.stop_movement()
            except Exception as exc:
                logger.warning("Fallback stop failed during %s: %s", reason, exc)
        self._finish_session(reason)

    def _result_from_reason(
        self, goal: GoalSnapshot, reason: str, started_at: float
    ) -> PixelGoalHopResult:
        status, ok = _SESSION_STATUS.get(reason, ("needs_agent", False))
        error_code = None if ok else reason
        with self._state_lock:
            travelled_m = self._travelled_m
            goals_published = int(self._session_stats.get("goals_published", 0))
            final_distance_m = self._session_stats.get("final_distance_m")
        return PixelGoalHopResult(
            ok=ok,
            status=status,
            goal_id=goal.goal_id,
            point=goal.point.as_tuple(),
            steps=goals_published,
            duration_s=time.monotonic() - started_at,
            travelled_m=travelled_m,
            final_distance_m=final_distance_m,
            needs_agent_reason=None if ok else reason,
            error_code=error_code,
            error=None if ok else f"pixel-goal map session ended: {reason}",
        )

    def _check_goal_progress(self, now: float) -> bool:
        """Return True after sustained lack of net approach, even without new odom.

        The reference is a best-so-far distance, so oscillation or walking away
        and back cannot keep resetting the timer. Local replans cannot reset it.
        Run this from the hop wait loop, not a sensor callback.
        """
        config = self._loop_config()
        with self._state_lock:
            if not self._goal_published or self._progress_target is None:
                return False
            odom = self._latest_odom
            distance = (
                float(np.linalg.norm(self._progress_target - odom.position.to_numpy()[:2]))
                if odom is not None
                else None
            )
            if distance is not None and math.isfinite(distance):
                if self._progress_best_distance is None:
                    self._progress_best_distance = distance
                elif self._progress_best_distance - distance >= config.progress_epsilon_m:
                    self._progress_best_distance = distance
                    self._progress_at = now
            stalled_s = now - self._progress_at
            if stalled_s < config.no_progress_timeout_s:
                return False
            details = {
                "phase": self._phase,
                "goal": self._progress_target.tolist(),
                "distance_m": distance,
                "best_distance_m": self._progress_best_distance,
                "no_progress_s": round(stalled_s, 3),
            }
            self._session_stats["no_progress"] = details
        self._sink.event("no_progress", details)
        logger.warning("Pixel goal has no net progress; ending hop", **details)
        return True

    def _await_session(
        self, goal: GoalSnapshot, cancel: threading.Event, started_at: float
    ) -> PixelGoalHopResult:
        config = self._loop_config()
        while True:
            if self._session_done.wait(timeout=0.2):
                return self._result_from_reason(
                    goal, self._session_end_reason or "completed", started_at
                )
            if cancel.is_set():
                self._abort_session("cancelled_by_agent")
                # The done flag is now set; the next loop iteration returns.
            elif time.monotonic() - started_at > config.deadline_s:
                self._abort_session("deadline_exceeded")
            elif self._check_goal_progress(time.monotonic()):
                self._abort_session("no_progress")

    def _write_nav_status(self) -> None:
        with self._state_lock:
            phase = self._phase
            target = self._final_target
            odom = self._latest_odom
            travelled_m = self._travelled_m
            stats = dict(self._session_stats)
        distance_to_target_m = None
        if target is not None and odom is not None:
            distance_to_target_m = round(
                float(np.linalg.norm(target[:2] - odom.position.to_numpy()[:2])), 3
            )
        status_text = {
            "starting": "starting",
            "explore": "exploring",
            "final": "final_approach",
        }.get(phase, phase)
        with self._state_lock:
            goal = self._active_goal
        self._sink.write_status(
            {
                "status": status_text,
                "hop_running": True,
                "phase": phase,
                "backend": "map",
                "goal_id": None if goal is None else goal.goal_id,
                "direction": None if goal is None else (goal.direction or "front"),
                "goals_published": int(stats.get("goals_published", 0)),
                "target_kind": stats.get("target_kind"),
                "distance_to_target_m": distance_to_target_m,
                "travelled_m": round(travelled_m, 3),
            }
        )


__all__ = ["PixelGoalMapNavConfig", "PixelGoalMapNavModule"]
