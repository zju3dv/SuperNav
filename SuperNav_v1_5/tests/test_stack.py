from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from supernav import stack


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _caption_config(tmp_path: Path, port: int, *, enabled: bool = True) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "mcp": {
                    "environment": {
                        "HAB_CARTOGRAPHER_ENABLED": "0",
                        "HAB_SLAM_AREA_CAPTION_ENABLED": "1" if enabled else "0",
                        "HAB_SLAM_AREA_CAPTION_URL": f"http://127.0.0.1:{port}/caption",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_caption_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_caption.py"
    path.write_text(
        """\
import json, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/healthz':
            self.send_response(404); self.end_headers(); return
        body = json.dumps({'ok': True, 'protocol': 'test/v1', 'loaded': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, format, *args):
        return

server = HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print('fake caption ready', flush=True)
server.serve_forever()
""",
        encoding="utf-8",
    )
    return path


def _managed_args(
    *, config: Path, state_dir: Path, script: Path, port: int
) -> list[str]:
    return [
        "--config",
        str(config),
        "--state-dir",
        str(state_dir),
        "--caption-url",
        f"http://127.0.0.1:{port}/caption",
        "--caption-command",
        f"{sys.executable} {script} {port}",
        "--health-timeout",
        "5",
        "--shutdown-timeout",
        "1",
    ]


def _wait_dead(pid: int) -> None:
    deadline = time.monotonic() + 3
    while stack._process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not stack._process_alive(pid)


def test_caption_defaults_to_habitat_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERNAV_CAPTION_COMMAND", raising=False)
    monkeypatch.delenv("SUPERNAV_CAPTION_PYTHON", raising=False)
    monkeypatch.setenv("SUPERNAV_HABITAT_PYTHON", "/runtime/habitat/bin/python")
    options = SimpleNamespace(
        caption_command=None,
        caption_python=None,
        caption_model="model",
        caption_device_map="cuda:0",
        caption_dtype="auto",
        caption_no_warmup=False,
    )

    command = stack._caption_command(options, host="127.0.0.1", port=8926)

    assert command[0] == "/runtime/habitat/bin/python"


def test_managed_up_status_logs_idempotence_and_down(tmp_path: Path, capsys) -> None:
    port = _free_port()
    state_dir = tmp_path / "state"
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    arguments = _managed_args(
        config=config, state_dir=state_dir, script=script, port=port
    )

    assert stack.main(["up", *arguments]) == 0
    state = json.loads((state_dir / "state.json").read_text())
    pid = state["services"]["caption"]["pid"]
    assert stack._process_alive(pid)
    assert state["services"]["caption"]["health_state"] == "healthy"

    capsys.readouterr()
    assert stack.main(["status", "--state-dir", str(state_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["services"]["caption"]["pid"] == pid

    assert stack.main(["logs", "--state-dir", str(state_dir)]) == 0
    assert "fake caption ready" in capsys.readouterr().out
    assert stack.main(["up", *arguments]) == 0
    assert "already running" in capsys.readouterr().out.lower()

    assert stack.main(["down", "--state-dir", str(state_dir), "--timeout", "1"]) == 0
    _wait_dead(pid)
    assert not (state_dir / "state.json").exists()


def test_managed_up_refuses_unowned_occupied_port(tmp_path: Path, capsys) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    state_dir = tmp_path / "state"
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        result = stack.main(
            [
                "up",
                *_managed_args(
                    config=config,
                    state_dir=state_dir,
                    script=script,
                    port=port,
                ),
            ]
        )

    assert result == 2
    assert "occupied by an unowned process" in capsys.readouterr().err
    assert not (state_dir / "state.json").exists()


def test_stale_pid_state_is_replaced_without_signaling_it(
    tmp_path: Path,
) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "service_mode": "managed",
                "services": {
                    "caption": {
                        "mode": "managed",
                        "pid": os.getpid(),
                        "process_start_token": "definitely-not-this-process",
                        "health_url": f"http://127.0.0.1:{port}/healthz",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = _managed_args(
        config=config, state_dir=state_dir, script=script, port=port
    )
    assert stack.main(["up", *arguments]) == 0
    state = json.loads((state_dir / "state.json").read_text())
    pid = state["services"]["caption"]["pid"]
    assert pid != os.getpid()
    assert stack.main(["down", "--state-dir", str(state_dir)]) == 0
    _wait_dead(pid)


def test_external_mode_validates_but_never_stops_service(tmp_path: Path) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    external = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/healthz"
        stack._wait_for_health(url=health_url, process=external, timeout_s=5)
        state_dir = tmp_path / "state"
        assert stack.main(
            [
                "up",
                "--config",
                str(config),
                "--state-dir",
                str(state_dir),
                "--service-mode",
                "external",
                "--health-timeout",
                "2",
            ]
        ) == 0
        state = json.loads((state_dir / "state.json").read_text())
        assert state["services"]["caption"]["mode"] == "external"
        assert "pid" not in state["services"]["caption"]
        assert stack.main(["down", "--state-dir", str(state_dir)]) == 0
        assert external.poll() is None
    finally:
        os.killpg(external.pid, signal.SIGTERM)
        external.wait(timeout=5)


def test_stack_run_cleans_up_owned_service_and_forwards_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    state_dir = tmp_path / "state"
    observed: dict[str, object] = {}

    def delegate(runner: str, arguments: list[str]) -> int:
        state = json.loads((state_dir / "state.json").read_text())
        observed["pid"] = state["services"]["caption"]["pid"]
        observed["runner"] = runner
        observed["arguments"] = arguments
        observed["url"] = os.environ.get("HAB_SLAM_AREA_CAPTION_URL")
        return 7

    monkeypatch.setattr(stack, "_delegate_runner", delegate)
    result = stack.main(
        [
            "run",
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            f"{sys.executable} {script} {port}",
            "--health-timeout",
            "5",
            "--shutdown-timeout",
            "1",
            "--config",
            str(config),
            "--arms",
            "default",
        ]
    )

    assert result == 7
    assert observed["runner"] == "run"
    assert observed["arguments"] == ["--config", str(config), "--arms", "default"]
    assert observed["url"] == f"http://127.0.0.1:{port}/caption"
    assert not (state_dir / "state.json").exists()
    _wait_dead(int(observed["pid"]))


def test_stack_run_paired_delegates_and_cleans_up_owned_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    state_dir = tmp_path / "state"
    observed: dict[str, object] = {}

    def delegate(runner: str, arguments: list[str]) -> int:
        state = json.loads((state_dir / "state.json").read_text())
        observed["pid"] = state["services"]["caption"]["pid"]
        observed["runner"] = runner
        observed["arguments"] = arguments
        observed["url"] = os.environ.get("HAB_SLAM_AREA_CAPTION_URL")
        return 0

    monkeypatch.setattr(stack, "_delegate_runner", delegate)
    result = stack.main(
        [
            "run-paired",
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            f"{sys.executable} {script} {port}",
            "--health-timeout",
            "5",
            "--shutdown-timeout",
            "1",
            "--config",
            str(config),
            "--task-ids",
            "task-1",
        ]
    )

    assert result == 0
    assert observed["runner"] == "run-paired"
    assert observed["arguments"] == [
        "--config",
        str(config),
        "--task-ids",
        "task-1",
    ]
    assert observed["url"] == f"http://127.0.0.1:{port}/caption"
    assert not (state_dir / "state.json").exists()
    _wait_dead(int(observed["pid"]))


def test_stack_run_signal_cleanup_and_keep_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    script = _fake_caption_script(tmp_path)
    state_dir = tmp_path / "state"
    command = f"{sys.executable} {script} {port}"

    monkeypatch.setattr(
        stack,
        "_delegate_runner",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    result = stack.main(
        [
            "run",
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            command,
            "--health-timeout",
            "5",
            "--shutdown-timeout",
            "1",
            "--config",
            str(config),
        ]
    )
    assert result == 130
    assert not (state_dir / "state.json").exists()

    monkeypatch.setattr(
        stack,
        "_delegate_runner",
        lambda *_: (_ for _ in ()).throw(stack._StackSignal(signal.SIGTERM)),
    )
    result = stack.main(
        [
            "run",
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            command,
            "--health-timeout",
            "5",
            "--shutdown-timeout",
            "1",
            "--config",
            str(config),
        ]
    )
    assert result == 128 + signal.SIGTERM
    assert not (state_dir / "state.json").exists()

    monkeypatch.setattr(stack, "_delegate_runner", lambda *_: 0)
    assert stack.main(
        [
            "run",
            "--keep-services",
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            command,
            "--health-timeout",
            "5",
            "--config",
            str(config),
        ]
    ) == 0
    state = json.loads((state_dir / "state.json").read_text())
    pid = state["services"]["caption"]["pid"]
    assert stack._process_alive(pid)
    assert stack.main(["down", "--state-dir", str(state_dir)]) == 0
    _wait_dead(pid)


def test_startup_failure_removes_state(tmp_path: Path, capsys) -> None:
    port = _free_port()
    config = _caption_config(tmp_path, port)
    failing = tmp_path / "fail.py"
    failing.write_text("raise SystemExit(9)\n", encoding="utf-8")
    state_dir = tmp_path / "state"

    result = stack.main(
        [
            "up",
            "--config",
            str(config),
            "--state-dir",
            str(state_dir),
            "--caption-url",
            f"http://127.0.0.1:{port}/caption",
            "--caption-command",
            f"{sys.executable} {failing}",
            "--health-timeout",
            "2",
        ]
    )

    assert result == 2
    assert "exited before becoming healthy" in capsys.readouterr().err
    assert not (state_dir / "state.json").exists()


def test_cartographer_capability_preflight_for_slam_only_stack(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"authoritative_odometry\":true,\"dual_stream\":true}'\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "mcp": {
                    "environment": {
                        "HAB_CARTOGRAPHER_ENABLED": "1",
                        "HAB_CARTOGRAPHER_ODOMETRY_MODE": "authoritative",
                        "HAB_CARTOGRAPHER_WORKER_COMMAND": str(worker),
                        "HAB_SLAM_AREA_CAPTION_ENABLED": "0",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"

    assert stack.main(
        ["up", "--config", str(config), "--state-dir", str(state_dir)]
    ) == 0
    state = json.loads((state_dir / "state.json").read_text())
    assert state["services"] == {}
    assert stack.main(["status", "--state-dir", str(state_dir)]) == 0
    assert stack.main(["down", "--state-dir", str(state_dir)]) == 0
