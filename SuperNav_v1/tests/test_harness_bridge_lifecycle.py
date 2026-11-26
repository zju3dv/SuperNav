from __future__ import annotations

from io import StringIO
from pathlib import Path
import socket
import sys

import pytest

from supernav.backends.habitat import bridge_lifecycle as lifecycle


def test_episode_bridge_rejects_occupied_port_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lifecycle.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with (
            pytest.raises(lifecycle.BridgeStartupError, match="occupied"),
            lifecycle.episode_bridge(
                root=tmp_path,
                config={"bridge": {"per_episode": True}},
                port=listener.getsockname()[1],
                log_path=tmp_path / "bridge.log",
                dry_run=False,
            ),
        ):
            pytest.fail("must not consume episode")


def test_episode_bridge_stops_owned_process_after_episode_error(tmp_path, monkeypatch):
    class Process:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            return 0

    process = Process()
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **k: process)

    class Opener:
        def open(self, *a, **k):
            return StringIO('{"ok": true}')

    monkeypatch.setattr(lifecycle.urllib.request, "build_opener", lambda *a: Opener())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with (
        pytest.raises(ValueError, match="episode failed"),
        lifecycle.episode_bridge(
            root=tmp_path,
            config={"bridge": {"per_episode": True}},
            port=port,
            log_path=tmp_path / "bridge.log",
            dry_run=False,
        ),
    ):
        raise ValueError("episode failed")
    assert process.stopped
