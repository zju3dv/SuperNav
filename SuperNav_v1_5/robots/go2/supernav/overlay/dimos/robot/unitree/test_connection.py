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

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from reactivex.subject import Subject

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.robot.unitree.connection import (
    DEFAULT_GO2_VIDEO_FPS,
    UnitreeWebRTCConnection,
    _advance_video_frame_clock,
    _configure_aioice_host_address,
    _configure_aioice_udp_receive_buffer,
    _env_int,
    _odometry_from_msg_or_none,
    _sport_mode_state_to_pose,
    _video_frame_due,
    _webrtc_local_ip,
)
from dimos.robot.unitree.type.odometry import Odometry, raw_odometry_msg_sample


def test_default_go2_video_fps_matches_awm_control_rate() -> None:
    assert DEFAULT_GO2_VIDEO_FPS == 4.0


class BlockingPubSub:
    async def publish_request_new(self, topic: str, data: dict[Any, Any]) -> dict[str, Any]:
        await asyncio.sleep(30.0)
        return {"status": {"code": 0}}


class FailingPubSub:
    async def publish_request_new(self, topic: str, data: dict[Any, Any]) -> dict[str, Any]:
        raise RuntimeError("Data channel is not open")


class NeverConnectingLegion:
    async def connect(self) -> None:
        await asyncio.sleep(30.0)


class ReconnectPubSub:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []

    async def publish_request_new(self, topic: str, data: dict[Any, Any]) -> dict[str, Any]:
        return {"topic": topic, "data": data}

    def subscribe(self, topic: str, callback: Any) -> None:
        self.subscriptions.append((topic, callback))


class ReconnectLegion:
    def __init__(self) -> None:
        self.connect_count = 0
        self.disconnect_count = 0
        self.sessions: list[Any] = []

    async def connect(self) -> None:
        self.connect_count += 1
        pub_sub = ReconnectPubSub()
        self.datachannel = SimpleNamespace(
            channel=SimpleNamespace(readyState="open"),
            heartbeat=SimpleNamespace(heartbeat_response=time.time()),
            pub_sub=pub_sub,
            disableTrafficSaving=self._disable_traffic_saving,
            set_decoder=lambda decoder_type: None,
        )
        self.pc = SimpleNamespace(connectionState="connected", iceConnectionState="completed")
        self.video = SimpleNamespace(
            add_track_callback=lambda callback: None,
            switchVideoChannel=lambda enabled: None,
        )
        self.sessions.append(self.datachannel)

    async def disconnect(self) -> None:
        self.disconnect_count += 1

    async def _disable_traffic_saving(self, enabled: bool) -> bool:
        return enabled


def test_constructor_omits_aes_key_for_legacy_webrtc_driver(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    class LegacyLegion:
        def __init__(self, connection_method: object, *, ip: str) -> None:
            calls.append({"connection_method": connection_method, "ip": ip})

    fake_driver = types.ModuleType("unitree_webrtc_connect.webrtc_driver")
    fake_driver.UnitreeWebRTCConnection = LegacyLegion
    fake_driver.WebRTCConnectionMethod = SimpleNamespace(LocalSTA="local-sta")
    monkeypatch.setitem(sys.modules, "unitree_webrtc_connect.webrtc_driver", fake_driver)
    monkeypatch.setenv("UNITREE_AES_128_KEY", "0123456789abcdef0123456789abcdef")
    monkeypatch.setattr(UnitreeWebRTCConnection, "connect", lambda self: None)
    monkeypatch.setattr(
        "dimos.robot.unitree.connection._configure_aioice_udp_receive_buffer",
        lambda _: True,
    )

    with patch("dimos.robot.unitree.connection.logger.warning") as warning:
        UnitreeWebRTCConnection("192.168.123.161")

    assert calls == [{"connection_method": "local-sta", "ip": "192.168.123.161"}]
    warning.assert_called_once()
    assert "does not support aes_128_key" in warning.call_args.args[0]


def test_constructor_passes_aes_key_for_supported_webrtc_driver(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class AesLegion:
        def __init__(
            self,
            connection_method: object,
            *,
            ip: str,
            aes_128_key: str | None = None,
        ) -> None:
            calls.append(
                {
                    "connection_method": connection_method,
                    "ip": ip,
                    "aes_128_key": aes_128_key,
                }
            )

    fake_driver = types.ModuleType("unitree_webrtc_connect.webrtc_driver")
    fake_driver.UnitreeWebRTCConnection = AesLegion
    fake_driver.WebRTCConnectionMethod = SimpleNamespace(LocalSTA="local-sta")
    monkeypatch.setitem(sys.modules, "unitree_webrtc_connect.webrtc_driver", fake_driver)
    monkeypatch.setenv("UNITREE_AES_128_KEY", "0123456789abcdef0123456789abcdef")
    monkeypatch.setattr(UnitreeWebRTCConnection, "connect", lambda self: None)
    monkeypatch.setattr(
        "dimos.robot.unitree.connection._configure_aioice_udp_receive_buffer",
        lambda _: True,
    )

    UnitreeWebRTCConnection("192.168.123.161")

    assert calls == [
        {
            "connection_method": "local-sta",
            "ip": "192.168.123.161",
            "aes_128_key": "0123456789abcdef0123456789abcdef",
        }
    ]


def test_env_int_falls_back_on_bad_value() -> None:
    with patch.dict("os.environ", {"DIMOS_TEST_INT": "not-an-int"}):
        assert _env_int("DIMOS_TEST_INT", 123) == 123


def test_video_frame_due_honors_fps_cap() -> None:
    assert _video_frame_due(10.0, None, 2.0) is True
    assert _video_frame_due(10.4, 10.0, 2.0) is False
    assert _video_frame_due(10.5, 10.0, 2.0) is True
    assert _video_frame_due(10.1, 10.0, 0.0) is True


def test_video_frame_clock_does_not_accumulate_late_arrivals() -> None:
    clock = _advance_video_frame_clock(10.0, None, 4.0)
    assert clock == 10.0

    # Incoming frames are 17 ms late, but the logical 4 Hz clock remains on
    # exact 250 ms boundaries instead of drifting by 17 ms on every frame.
    for frame_time in (10.267, 10.517, 10.767, 11.017):
        assert _video_frame_due(frame_time, clock, 4.0)
        clock = _advance_video_frame_clock(frame_time, clock, 4.0)

    assert clock == 11.0


def test_video_frame_clock_skips_missed_periods_without_rephasing() -> None:
    assert _advance_video_frame_clock(11.31, 10.0, 4.0) == 11.25


def test_sport_mode_state_to_pose_uses_firmware_position_and_orientation() -> None:
    pose = _sport_mode_state_to_pose(
        {
            "topic": "rt/lf/sportmodestate",
            "data": {
                "position": [1.25, -0.5, 0.31],
                "velocity": [0.2, 0.0, 0.0],
                "yaw_speed": 0.1,
                "imu_state": {
                    "quaternion": [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
                    "rpy": [0.0, 0.0, math.pi / 2],
                },
            },
        },
        ts=123.0,
    )

    assert pose is not None
    assert pose.frame_id == "sport_mode_odom"
    assert pose.ts == 123.0
    assert pose.position.to_list() == [1.25, -0.5, 0.31]
    assert math.isclose(pose.yaw, math.pi / 2, abs_tol=1e-6)


def test_sport_mode_state_stream_subscribes_to_firmware_topic() -> None:
    connection = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    topics: list[str] = []
    upstream = Mock()
    upstream.pipe.return_value = upstream
    connection.unitree_sub_stream = lambda topic: topics.append(topic) or upstream  # type: ignore[method-assign]

    stream = connection.sport_mode_state_stream()

    assert topics == ["rt/lf/sportmodestate"]
    assert stream is not None


def test_odometry_stream_survives_one_conversion_failure(monkeypatch: Any) -> None:
    connection = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    upstream = Subject()
    connection.raw_odom_stream = lambda: upstream  # type: ignore[method-assign]
    converted = PoseStamped(position=[1.0, 2.0, 0.3], frame_id="world")
    calls = 0

    def convert(_message: Any) -> PoseStamped:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AssertionError("transient Plum annotation resolution failure")
        return converted

    monkeypatch.setattr(Odometry, "from_msg", staticmethod(convert))
    received: list[PoseStamped] = []
    errors: list[Exception] = []
    connection.odom_stream().subscribe(received.append, errors.append)

    upstream.on_next(raw_odometry_msg_sample)
    first_deadline = time.monotonic() + 1.0
    while calls < 1 and time.monotonic() < first_deadline:
        time.sleep(0.01)
    upstream.on_next(raw_odometry_msg_sample)
    second_deadline = time.monotonic() + 1.0
    while not received and time.monotonic() < second_deadline:
        time.sleep(0.01)

    assert errors == []
    assert received == [converted]


def test_odometry_conversion_guard_drops_failed_frame(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        Odometry,
        "from_msg",
        staticmethod(
            lambda _message: (_ for _ in ()).throw(AssertionError("dispatch failed"))
        ),
    )

    assert _odometry_from_msg_or_none(raw_odometry_msg_sample) is None


def test_configure_aioice_udp_receive_buffer() -> None:
    from aioice import turn

    previous = turn.UDP_SOCKET_BUFFER_SIZE
    try:
        assert _configure_aioice_udp_receive_buffer(1_234_567) is True
        assert turn.UDP_SOCKET_BUFFER_SIZE == 1_234_567
        assert _configure_aioice_udp_receive_buffer(0) is False
        assert turn.UDP_SOCKET_BUFFER_SIZE == 1_234_567
    finally:
        turn.UDP_SOCKET_BUFFER_SIZE = previous


def test_webrtc_ice_uses_route_selected_local_ip(monkeypatch: Any) -> None:
    from aioice import ice

    monkeypatch.delenv("DIMOS_GO2_WEBRTC_LOCAL_IP", raising=False)
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)
    sock.getsockname.return_value = ("10.42.0.1", 54321)
    monkeypatch.setattr("socket.socket", Mock(return_value=sock))

    previous = ice.get_host_addresses
    try:
        local_ip = _webrtc_local_ip("10.42.0.103")

        assert local_ip == "10.42.0.1"
        sock.connect.assert_called_once_with(("10.42.0.103", 9))
        assert _configure_aioice_host_address(local_ip) is True
        assert ice.get_host_addresses(use_ipv4=True, use_ipv6=True) == ["10.42.0.1"]
        assert ice.get_host_addresses(use_ipv4=False, use_ipv6=True) == []
    finally:
        ice.get_host_addresses = previous


def test_connect_times_out_when_connection_ready_never_sets() -> None:
    conn = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    conn.conn = NeverConnectingLegion()
    conn.mode = "ai"
    conn.publish_request_timeout_s = 8.0
    conn.connect_ready_timeout_s = 0.05
    conn.last_publish_request_diagnostic = None

    started_at_s = time.monotonic()
    try:
        try:
            conn.connect()
        except TimeoutError as exc:
            elapsed_s = time.monotonic() - started_at_s
            assert "Timed out" in str(exc)
        else:
            raise AssertionError("connect() should time out")
    finally:
        if hasattr(conn, "loop") and not conn.loop.is_closed():
            conn.loop.close()

    assert elapsed_s < 1.0
    assert not conn.thread.is_alive()


def test_stale_heartbeat_reconnects_and_restores_topic_subscriptions(
    monkeypatch: Any,
) -> None:
    callback = Mock()
    legion = ReconnectLegion()
    conn = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    conn.conn = legion
    conn.ip = "10.42.0.103"
    conn.webrtc_local_ip = "10.42.0.1"
    conn.mode = "ai"
    conn.publish_request_timeout_s = 0.1
    conn.heartbeat_timeout_s = 8.0
    conn.reconnect_delay_s = 0.0
    conn.last_publish_request_diagnostic = None
    conn.connection_error = None
    conn._topic_callbacks = {"rt/test": callback}
    conn._video_track_callbacks = set()
    conn._connected_at_s = 0.0
    conn._stopping = False
    selected_ips: list[str] = []
    configured_ips: list[str] = []
    monkeypatch.setattr(
        "dimos.robot.unitree.connection._webrtc_local_ip",
        lambda ip: selected_ips.append(ip) or "10.42.0.1",
    )
    monkeypatch.setattr(
        "dimos.robot.unitree.connection._configure_aioice_host_address",
        lambda ip: configured_ips.append(ip) or True,
    )

    async def run() -> None:
        await conn._connect_session()
        legion.datachannel.heartbeat.heartbeat_response = time.time() - 9.0
        reason = conn._connection_unhealthy_reason()
        assert reason is not None and reason.startswith("heartbeat_stale_")
        await conn._reconnect(reason)

    asyncio.run(run())

    assert legion.connect_count == 2
    assert legion.disconnect_count == 1
    assert legion.sessions[0].pub_sub.subscriptions == [("rt/test", callback)]
    assert legion.sessions[1].pub_sub.subscriptions == [("rt/test", callback)]
    assert selected_ips == ["10.42.0.103"]
    assert configured_ips == ["10.42.0.1"]
    assert conn._connection_unhealthy_reason() is None


def test_publish_request_returns_diagnostic_on_timeout() -> None:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(timeout=1.0)
    conn = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    conn.loop = loop
    conn.conn = SimpleNamespace(datachannel=SimpleNamespace(pub_sub=BlockingPubSub()))
    conn.publish_request_timeout_s = 0.05
    conn.last_publish_request_diagnostic = None

    try:
        started_at_s = time.monotonic()
        result = conn.publish_request("rt/sportmod", {"api_id": 1004})
        elapsed_s = time.monotonic() - started_at_s
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=1.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()

    assert elapsed_s < 1.0
    assert result["ok"] is False
    assert result["error"] == "publish_request_timeout"
    assert result["timed_out"] is True
    assert result["command_may_have_been_accepted"] is True
    assert result["topic"] == "rt/sportmod"
    assert result["request"] == {"api_id": 1004}
    assert conn.last_publish_request_diagnostic == result


def test_publish_request_returns_diagnostic_on_datachannel_error() -> None:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(timeout=1.0)
    conn = UnitreeWebRTCConnection.__new__(UnitreeWebRTCConnection)
    conn.loop = loop
    conn.conn = SimpleNamespace(datachannel=SimpleNamespace(pub_sub=FailingPubSub()))
    conn.publish_request_timeout_s = 8.0
    conn.last_publish_request_diagnostic = None

    try:
        result = conn.publish_request("rt/sportmod", {"api_id": 1004})
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=1.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()

    assert result["ok"] is False
    assert result["error"] == "publish_request_failed"
    assert result["timed_out"] is False
    assert result["exception_type"] == "RuntimeError"
    assert result["exception"] == "Data channel is not open"
    assert result["command_may_have_been_accepted"] is False
    assert conn.last_publish_request_diagnostic == result
