# Copyright 2025-2026 Dimensional Inc.
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

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import functools
import inspect
import json
import math
import os
import random
import socket
import threading
import time
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.observable import Observable
from reactivex.subject import Subject
from unitree_webrtc_connect.constants import (
    DATA_CHANNEL_TYPE,
    RTC_TOPIC,
    SPORT_CMD,
    VUI_COLOR,
)

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.resource import Resource
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.type.lidar import (
    RawLidarMsg,
    pointcloud2_from_webrtc_lidar,
)
from dimos.robot.unitree.type.lowstate import LowStateMsg
from dimos.robot.unitree.type.odometry import Odometry, warm_odometry_dispatch
from dimos.types.timestamped import Timestamped
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure, callback_to_observable

VideoMessage: TypeAlias = NDArray[np.uint8]  # Shape: (height, width, 3)

logger = setup_logger()

_T = TypeVar("_T", bound=Timestamped)
DEFAULT_PUBLISH_REQUEST_TIMEOUT_S = 8.0
DEFAULT_CONNECT_READY_TIMEOUT_S = 30.0
DEFAULT_WEBRTC_UDP_RCVBUF_BYTES = 67_108_864
DEFAULT_WEBRTC_HEARTBEAT_TIMEOUT_S = 8.0
DEFAULT_WEBRTC_RECONNECT_DELAY_S = 2.0
DEFAULT_GO2_VIDEO_FPS = 4.0
PUBLISH_REQUEST_TIMEOUT_STATUS_CODE = -408


def time_is_now(x: _T) -> _T:
    x.ts = time.time()
    return x


def _odometry_from_msg_or_none(message: Any) -> Odometry | None:
    """Convert one ROBOTODOM frame without allowing it to kill the Rx stream."""
    try:
        return Odometry.from_msg(message)
    except Exception:
        logger.exception("Dropping invalid Go2 ROBOTODOM frame; odometry stream remains active")
        return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _video_frame_due(now_s: float, last_emit_s: float | None, fps: float) -> bool:
    return fps <= 0 or last_emit_s is None or now_s - last_emit_s >= 1.0 / fps


def _advance_video_frame_clock(
    now_s: float, last_emit_s: float | None, fps: float
) -> float:
    """Advance the ideal publication clock without accumulating arrival jitter."""
    if fps <= 0 or last_emit_s is None:
        return now_s
    period_s = 1.0 / fps
    elapsed_periods = max(1, math.floor((now_s - last_emit_s) / period_s))
    return last_emit_s + elapsed_periods * period_s


def _configure_aioice_udp_receive_buffer(size_bytes: int) -> bool:
    if size_bytes <= 0:
        return False
    try:
        from aioice import turn
    except ImportError:
        return False
    turn.UDP_SOCKET_BUFFER_SIZE = size_bytes
    return True


def _webrtc_local_ip(peer_ip: str) -> str | None:
    override = os.environ.get("DIMOS_GO2_WEBRTC_LOCAL_IP", "").strip()
    if override:
        return override
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((peer_ip, 9))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def _configure_aioice_host_address(local_ip: str | None) -> bool:
    if not local_ip:
        return False
    try:
        from aioice import ice
    except ImportError:
        return False

    def get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        del use_ipv6
        return [local_ip] if use_ipv4 else []

    ice.get_host_addresses = get_host_addresses
    return True


def _sport_mode_state_to_pose(
    message: Mapping[str, Any], *, ts: float | None = None
) -> PoseStamped | None:
    data = message.get("data")
    if not isinstance(data, Mapping):
        return None
    if "position" not in data and isinstance(data.get("data"), Mapping):
        data = data["data"]

    position = data.get("position")
    imu_state = data.get("imu_state")
    quaternion = imu_state.get("quaternion") if isinstance(imu_state, Mapping) else None
    if not isinstance(position, Sequence) or not isinstance(quaternion, Sequence):
        return None
    if len(position) < 3 or len(quaternion) < 4:
        return None

    try:
        px, py, pz = (float(position[index]) for index in range(3))
        qw, qx, qy, qz = (float(quaternion[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (px, py, pz, qw, qx, qy, qz)):
        return None

    return PoseStamped(
        ts=time.time() if ts is None else ts,
        frame_id="sport_mode_odom",
        position=Vector3(px, py, pz),
        orientation=Quaternion(qx, qy, qz, qw),
    )


@dataclass
class SerializableVideoFrame:
    """Pickleable wrapper for av.VideoFrame with all metadata"""

    data: np.ndarray
    pts: int | None = None
    time: float | None = None
    dts: int | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None

    @classmethod
    def from_av_frame(cls, frame):  # type: ignore[no-untyped-def]
        return cls(
            data=frame.to_ndarray(format="rgb24"),
            pts=frame.pts,
            time=frame.time,
            dts=frame.dts,
            width=frame.width,
            height=frame.height,
            format=frame.format.name if hasattr(frame, "format") and frame.format else None,
        )

    def to_ndarray(self, format=None):  # type: ignore[no-untyped-def]
        return self.data


class UnitreeWebRTCConnection(Resource):
    _SPORT_API_ID_RAGEMODE: int = 2059

    def __init__(self, ip: str, mode: str = "ai") -> None:
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection as LegionConnection,
            WebRTCConnectionMethod,
        )

        # Resolve Plum/Beartype's lazy Pose dispatch in this single-threaded
        # setup phase.  Otherwise the first live ROBOTODOM frame can race with
        # another Pose construction and permanently terminate the Rx stream.
        warm_odometry_dispatch()

        self.ip = ip
        self.mode = mode
        self.publish_request_timeout_s = _env_float(
            "DIMOS_GO2_PUBLISH_REQUEST_TIMEOUT_S", DEFAULT_PUBLISH_REQUEST_TIMEOUT_S
        )
        self.connect_ready_timeout_s = _env_float(
            "DIMOS_GO2_CONNECT_READY_TIMEOUT_S", DEFAULT_CONNECT_READY_TIMEOUT_S
        )
        self.webrtc_udp_rcvbuf_bytes = _env_int(
            "DIMOS_GO2_WEBRTC_RCVBUF_BYTES", DEFAULT_WEBRTC_UDP_RCVBUF_BYTES
        )
        self.video_fps = _env_float("DIMOS_GO2_VIDEO_FPS", DEFAULT_GO2_VIDEO_FPS)
        self.heartbeat_timeout_s = _env_float(
            "DIMOS_GO2_HEARTBEAT_TIMEOUT_S", DEFAULT_WEBRTC_HEARTBEAT_TIMEOUT_S
        )
        self.reconnect_delay_s = _env_float(
            "DIMOS_GO2_RECONNECT_DELAY_S", DEFAULT_WEBRTC_RECONNECT_DELAY_S
        )
        logger.info("Go2 WebRTC video publish FPS cap: %s", self.video_fps)
        _configure_aioice_udp_receive_buffer(self.webrtc_udp_rcvbuf_bytes)
        self.last_publish_request_diagnostic: dict[str, Any] | None = None
        self.connection_error: Exception | None = None
        self.stop_timer: threading.Timer | None = None
        self.cmd_vel_timeout = 0.2
        self._topic_callbacks: dict[str, Any] = {}
        self._video_track_callbacks: set[Any] = set()
        self._connected_at_s = 0.0
        self._stopping = False
        self.webrtc_local_ip = _webrtc_local_ip(self.ip)
        conn_kwargs: dict[str, Any] = {"ip": self.ip}
        aes_128_key = os.environ.get("UNITREE_AES_128_KEY") or os.environ.get(
            "DIMOS_UNITREE_AES_128_KEY"
        )
        if aes_128_key:
            if "aes_128_key" not in inspect.signature(LegionConnection).parameters:
                logger.warning(
                    "UNITREE_AES_128_KEY is set, but the installed "
                    "unitree_webrtc_connect package does not support aes_128_key; "
                    "continuing without passing the key."
                )
            else:
                conn_kwargs["aes_128_key"] = aes_128_key
        self.conn = LegionConnection(WebRTCConnectionMethod.LocalSTA, **conn_kwargs)
        self.connect()

    def connect(self) -> None:
        local_ip = getattr(self, "webrtc_local_ip", None)
        if _configure_aioice_host_address(local_ip):
            logger.info("Go2 WebRTC ICE local address: %s", local_ip)
        else:
            logger.warning(
                "Could not select a local WebRTC address for Go2 %s",
                getattr(self, "ip", "<unknown>"),
            )
        self.loop = asyncio.new_event_loop()
        self.task = None
        self.connected_event = asyncio.Event()
        self.connection_ready = threading.Event()
        self.connection_error = None
        self._stopping = False
        self._topic_callbacks = getattr(self, "_topic_callbacks", {})
        self._video_track_callbacks = getattr(self, "_video_track_callbacks", set())

        async def async_connect() -> None:
            try:
                await self._connect_session()

                self.connected_event.set()
                self.connection_ready.set()

                while not self._stopping:
                    await asyncio.sleep(1)
                    reason = self._connection_unhealthy_reason()
                    if reason:
                        await self._reconnect(reason)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connection_error = exc
                raise
            finally:
                self.connection_ready.set()

        def start_background_loop() -> None:
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        if not self.connection_ready.wait(timeout=self.connect_ready_timeout_s):
            self._stop_background_loop()
            raise TimeoutError(
                f"Timed out after {self.connect_ready_timeout_s}s waiting for Unitree WebRTC "
                "connection to become ready."
            )
        if self.connection_error is not None:
            self._stop_background_loop()
            raise RuntimeError("Unitree WebRTC connection failed before becoming ready.") from (
                self.connection_error
            )

    async def _connect_session(self) -> None:
        await self.conn.connect()
        await self.conn.datachannel.disableTrafficSaving(True)
        self.conn.datachannel.set_decoder(decoder_type="native")

        motion_switcher_request = {"api_id": 1002, "parameter": {"name": self.mode}}
        started_at_s = time.monotonic()
        try:
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"], motion_switcher_request
                ),
                timeout=self.publish_request_timeout_s,
            )
        except TimeoutError:
            self.last_publish_request_diagnostic = self._publish_request_timeout_result(
                topic=RTC_TOPIC["MOTION_SWITCHER"],
                data=motion_switcher_request,
                timeout_s=self.publish_request_timeout_s,
                started_at_s=started_at_s,
                phase="connect_warmup",
            )

        self._connected_at_s = time.monotonic()
        for topic, callback in tuple(self._topic_callbacks.items()):
            self.conn.datachannel.pub_sub.subscribe(topic, callback)
        for callback in tuple(self._video_track_callbacks):
            self.conn.video.add_track_callback(callback)
        if self._video_track_callbacks:
            self.conn.video.switchVideoChannel(True)

    def _connection_unhealthy_reason(self) -> str | None:
        pc = getattr(self.conn, "pc", None)
        datachannel = getattr(self.conn, "datachannel", None)
        channel = getattr(datachannel, "channel", None)
        peer_state = getattr(pc, "connectionState", None)
        ice_state = getattr(pc, "iceConnectionState", None)
        channel_state = getattr(channel, "readyState", None)

        if peer_state in {"closed", "disconnected", "failed"}:
            return f"peer_{peer_state}"
        if ice_state in {"closed", "disconnected", "failed"}:
            return f"ice_{ice_state}"
        if channel_state != "open":
            return f"datachannel_{channel_state or 'missing'}"
        if self.heartbeat_timeout_s <= 0:
            return None

        heartbeat = getattr(getattr(datachannel, "heartbeat", None), "heartbeat_response", None)
        if heartbeat is None:
            if time.monotonic() - self._connected_at_s > self.heartbeat_timeout_s:
                return "heartbeat_missing"
            return None
        age_s = time.time() - heartbeat
        if age_s > self.heartbeat_timeout_s:
            return f"heartbeat_stale_{age_s:.1f}s"
        return None

    async def _reconnect(self, reason: str) -> None:
        logger.warning("Go2 WebRTC unhealthy; reconnecting", reason=reason)
        while not self._stopping:
            try:
                try:
                    await self.conn.disconnect()
                except Exception:
                    logger.warning("Go2 WebRTC disconnect cleanup failed", exc_info=True)
                local_ip = _webrtc_local_ip(self.ip)
                if local_ip:
                    self.webrtc_local_ip = local_ip
                _configure_aioice_host_address(self.webrtc_local_ip)
                await self._connect_session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connection_error = exc
                logger.warning(
                    "Go2 WebRTC reconnect failed; retrying",
                    delay_s=self.reconnect_delay_s,
                    exc_info=True,
                )
                await asyncio.sleep(max(0.1, self.reconnect_delay_s))
                continue
            self.connection_error = None
            logger.warning(
                "Go2 WebRTC reconnected",
                topics=list(self._topic_callbacks),
            )
            return

    def _stop_background_loop(self) -> None:
        self._stopping = True
        if getattr(self, "loop", None) and self.loop.is_running():
            if getattr(self, "task", None) and not self.task.done():
                self.loop.call_soon_threadsafe(self.task.cancel)
            self.loop.call_soon_threadsafe(self.loop.stop)
        if getattr(self, "thread", None) and self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._stopping = True
        # Cancel timer
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if self.task:
            self.task.cancel()

        async def async_disconnect() -> None:
            try:
                # Send stop command directly since we're already in the event loop.
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": 0, "ly": 0, "rx": 0, "ry": 0},
                )
                await self.conn.disconnect()
            except Exception:
                pass

        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)

            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send movement command to the robot using Twist commands.

        Args:
            twist: Twist message with linear and angular velocities
            duration: How long to move (seconds). If 0, command is continuous

        Returns:
            bool: True if command was sent successfully
        """
        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        # DIMOS_GO2_MOVE_API selects the drive path:
        #   "1008"     -> SPORT_MOD Move (api_id 1008): parameter x/y/z are ACTUAL
        #                 velocities (m/s, m/s, rad/s) in the robot frame, so cmd_vel
        #                 maps 1:1 to real speed (NO joystick V_max scale/deadzone).
        #   "joystick" -> WIRELESS_CONTROLLER (default, legacy): the value is a
        #                 normalized joystick deflection [-1,1], real v = defl x V_max.
        move_api = os.environ.get("DIMOS_GO2_MOVE_API", "joystick").strip().lower()

        # WebRTC coordinate mapping (joystick path):
        # x - Positive right, negative left
        # y - positive forward, negative backwards
        # yaw - Positive rotate right, negative rotate left
        async def async_move() -> None:
            if move_api == "1008":
                # Sport Move: parameter {x=forward m/s, y=left m/s, z=yaw rad/s}
                # (robot frame; verify signs with a small drive_calib after enabling).
                gid = int(time.time() * 1000) % 2147483648 + random.randint(0, 1000)
                payload = {
                    "header": {"identity": {"id": gid, "api_id": SPORT_CMD["Move"]}},
                    "parameter": json.dumps({"x": x, "y": y, "z": yaw}),
                }
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["SPORT_MOD"], data=payload,
                    msg_type=DATA_CHANNEL_TYPE["REQUEST"],
                )
            else:
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": -y, "ly": x, "rx": -yaw, "ry": 0},
                )

        async def async_move_duration() -> None:
            """Send movement commands continuously for the specified duration."""
            start_time = time.time()
            sleep_time = 0.01

            while time.time() - start_time < duration:
                await async_move()
                await asyncio.sleep(sleep_time)

        # Cancel existing timer and start a new one
        if self.stop_timer:
            self.stop_timer.cancel()

        # Auto-stop after 0.5 seconds if no new commands
        self.stop_timer = threading.Timer(self.cmd_vel_timeout, self.stop_movement)
        self.stop_timer.daemon = True
        self.stop_timer.start()

        try:
            if duration > 0:
                # Send continuous move commands for the duration
                future = asyncio.run_coroutine_threadsafe(async_move_duration(), self.loop)
                future.result()
                # Stop after duration
                self.stop_movement()
            else:
                # Single command for continuous movement
                future = asyncio.run_coroutine_threadsafe(async_move(), self.loop)
                future.result()
            return True
        except Exception as e:
            print(f"Failed to send movement command: {e}")
            return False

    # Generic conversion of unitree subscription to Subject (used for all subs)
    def unitree_sub_stream(self, topic_name: str):  # type: ignore[no-untyped-def]
        def subscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            # Run the subscription in the background thread that has the event loop
            def run_subscription() -> None:
                self._topic_callbacks[topic_name] = cb
                self.conn.datachannel.pub_sub.subscribe(topic_name, cb)

            # Use call_soon_threadsafe to run in the background thread
            self.loop.call_soon_threadsafe(run_subscription)

        def unsubscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            # Run the unsubscription in the background thread that has the event loop
            def run_unsubscription() -> None:
                if self._topic_callbacks.get(topic_name) is cb:
                    self._topic_callbacks.pop(topic_name, None)
                self.conn.datachannel.pub_sub.unsubscribe(topic_name)

            # Use call_soon_threadsafe to run in the background thread
            self.loop.call_soon_threadsafe(run_unsubscription)

        return callback_to_observable(
            start=subscribe_in_thread,
            stop=unsubscribe_in_thread,
        )

    # Generic sync API call (we jump into the client thread)
    def publish_request(
        self, topic: str, data: dict[Any, Any], timeout_s: float | None = None
    ) -> Any:
        effective_timeout_s = self.publish_request_timeout_s if timeout_s is None else timeout_s
        started_at_s = time.monotonic()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.conn.datachannel.pub_sub.publish_request_new(topic, data), self.loop
            )
            return future.result(timeout=None if effective_timeout_s <= 0 else effective_timeout_s)
        except FutureTimeoutError:
            future.cancel()
            self.last_publish_request_diagnostic = self._publish_request_timeout_result(
                topic=topic,
                data=data,
                timeout_s=effective_timeout_s,
                started_at_s=started_at_s,
                phase="publish_request",
            )
            return self.last_publish_request_diagnostic
        except Exception as exc:
            self.last_publish_request_diagnostic = {
                "ok": False,
                "error": "publish_request_failed",
                "timed_out": False,
                "phase": "publish_request",
                "topic": topic,
                "request": data,
                "timeout_s": effective_timeout_s,
                "elapsed_s": round(time.monotonic() - started_at_s, 3),
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "command_may_have_been_accepted": False,
            }
            return self.last_publish_request_diagnostic

    def _publish_request_timeout_result(
        self,
        *,
        topic: str,
        data: dict[Any, Any],
        timeout_s: float,
        started_at_s: float,
        phase: str,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "publish_request_timeout",
            "timed_out": True,
            "phase": phase,
            "topic": topic,
            "request": data,
            "timeout_s": timeout_s,
            "duration_ms": round((time.monotonic() - started_at_s) * 1000, 1),
            "status": {
                "code": PUBLISH_REQUEST_TIMEOUT_STATUS_CODE,
                "message": "Unitree WebRTC publish_request did not return before timeout.",
            },
            "command_may_have_been_accepted": True,
            "recommended_next_action": (
                "Check observe/status before retrying; avoid blindly repeating the same command."
            ),
        }

    @simple_mcache
    def raw_lidar_stream(self) -> Observable[RawLidarMsg]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["ULIDAR_ARRAY"]))

    @simple_mcache
    def raw_odom_stream(self) -> Observable[Pose]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["ROBOTODOM"]))

    @simple_mcache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return backpressure(
            self.raw_lidar_stream().pipe(
                ops.map(pointcloud2_from_webrtc_lidar),
                ops.map(time_is_now),
                # repair_stale_ts(),
            )
        )

    @simple_mcache
    def tf_stream(self) -> Observable[Transform]:
        base_link = functools.partial(Transform.from_pose, "base_link")
        return backpressure(self.odom_stream().pipe(ops.map(base_link)))

    @simple_mcache
    def odom_stream(self) -> Observable[Pose]:
        return backpressure(
            self.raw_odom_stream().pipe(
                ops.map(_odometry_from_msg_or_none),
                ops.filter(lambda odom: odom is not None),
                ops.map(time_is_now),
            )
        )

    @simple_mcache
    def video_stream(self) -> Observable[Image]:
        return backpressure(
            self.raw_video_stream().pipe(
                ops.filter(lambda frame: frame is not None),
                ops.map(
                    lambda frame: Image.from_numpy(
                        # np.ascontiguousarray(frame.to_ndarray("rgb24")),
                        frame.to_ndarray(format="rgb24"),  # type: ignore[attr-defined]
                        format=ImageFormat.RGB,  # Frame is RGB24, not BGR
                        frame_id="camera_optical",
                    ),
                ),
                ops.map(time_is_now),
            )
        )

    @simple_mcache
    def lowstate_stream(self) -> Observable[LowStateMsg]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["LOW_STATE"]))

    @simple_mcache
    def sport_mode_state_stream(self) -> Observable[PoseStamped]:
        return backpressure(
            self.unitree_sub_stream(RTC_TOPIC["LF_SPORT_MOD_STATE"]).pipe(
                ops.map(_sport_mode_state_to_pose),
                ops.filter(lambda pose: pose is not None),
            )
        )

    def standup(self) -> bool:
        return self._request_succeeded(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]})
        )

    def balance_stand(self) -> bool:
        """Activate BalanceStand mode — enables WIRELESS_CONTROLLER joystick commands."""
        return self._request_succeeded(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        )

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self.publish_request(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"enable": int(enabled)}},
        )

    def free_walk(self) -> bool:
        """Activate FreeWalk locomotion mode — enables walking and velocity commands."""
        return self._request_succeeded(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["FreeWalk"]})
        )

    def enable_rage_mode(self) -> bool:
        """Enable Rage Mode on the Go2 via WebRTC.
        Assumes the robot is already in BalanceStand.
        """
        rage_ok = self._request_succeeded(
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": self._SPORT_API_ID_RAGEMODE, "parameter": {"data": True}},
            )
        )
        time.sleep(2.0)

        joystick_ok = self._request_succeeded(
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["SwitchJoystick"],
                    "parameter": {"data": True},
                },
            )
        )
        return rage_ok and joystick_ok

    def liedown(self) -> bool:
        return self._request_succeeded(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]})
        )

    def _request_succeeded(self, response: Any) -> bool:
        if isinstance(response, dict) and response.get("ok") is False:
            return False
        return bool(response)

    async def handstand(self):  # type: ignore[no-untyped-def]
        return self.publish_request(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Standup"], "parameter": {"data": True}},
        )

    def color(self, color: VUI_COLOR = VUI_COLOR.RED, colortime: int = 60) -> bool:
        return self.publish_request(  # type: ignore[no-any-return]
            RTC_TOPIC["VUI"],
            {
                "api_id": 1001,
                "parameter": {
                    "color": color,
                    "time": colortime,
                },
            },
        )

    @simple_mcache
    def raw_video_stream(self) -> Observable[VideoMessage]:
        subject: Subject[VideoMessage] = Subject()
        stop_event = threading.Event()
        last_emit_s: float | None = None

        from aiortc import MediaStreamTrack

        async def accept_track(track: MediaStreamTrack) -> None:
            nonlocal last_emit_s
            while True:
                if stop_event.is_set():
                    return
                frame = await track.recv()
                now_s = time.monotonic()
                if not _video_frame_due(now_s, last_emit_s, self.video_fps):
                    continue
                last_emit_s = _advance_video_frame_clock(
                    now_s, last_emit_s, self.video_fps
                )
                serializable_frame = SerializableVideoFrame.from_av_frame(frame)  # type: ignore[no-untyped-call]
                subject.on_next(serializable_frame)

        self.conn.video.add_track_callback(accept_track)
        self._video_track_callbacks.add(accept_track)

        # Run the video channel switching in the background thread
        def switch_video_channel() -> None:
            self.conn.video.switchVideoChannel(True)

        self.loop.call_soon_threadsafe(switch_video_channel)

        def stop() -> None:
            stop_event.set()  # Signal the loop to stop
            self._video_track_callbacks.discard(accept_track)
            if accept_track in self.conn.video.track_callbacks:
                self.conn.video.track_callbacks.remove(accept_track)

            # Run the video channel switching off in the background thread
            def switch_video_channel_off() -> None:
                self.conn.video.switchVideoChannel(False)

            self.loop.call_soon_threadsafe(switch_video_channel_off)

        return subject.pipe(ops.finally_action(stop))

    def get_video_stream(self, fps: int = 30) -> Observable[Image]:
        """Get the video stream from the robot's camera.

        Implements the AbstractRobot interface method.

        Args:
            fps: Frames per second. This parameter is included for API compatibility,
                 but doesn't affect the actual frame rate which is determined by the camera.

        Returns:
            Observable: An observable stream of video frames or None if video is not available.
        """
        return self.video_stream()

    def stop_movement(self) -> None:
        """Cancel the auto-stop timer (used by move() for continuous commands)."""
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    def disconnect(self) -> None:
        """Disconnect from the robot and clean up resources."""
        # Cancel timer
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if hasattr(self, "task") and self.task:
            self.task.cancel()
        if hasattr(self, "conn"):

            async def async_disconnect() -> None:
                try:
                    await self.conn.disconnect()
                except:
                    pass

            if hasattr(self, "loop") and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)

        if hasattr(self, "loop") and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
