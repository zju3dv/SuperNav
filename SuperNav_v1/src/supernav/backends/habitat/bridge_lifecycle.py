"""Own a fresh local bridge process for each benchmark episode."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from supernav.backends.habitat.config import bridge_python, process_environment
from supernav.backends.base import BridgeStartupError


@contextmanager
def episode_bridge(
    *, root: Path, config: dict, port: int, log_path: Path, dry_run: bool,
    live_context: dict | None = None,
):
    bridge = config.get("bridge", {})
    if dry_run or not bridge.get("per_episode", False):
        yield
        return
    host = bridge.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise BridgeStartupError("per_episode bridge requires host 127.0.0.1")
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise BridgeStartupError(f"Bridge port {port} is already occupied")
    env = dict(os.environ)
    env.update(process_environment(config))
    if env.get("SUPERNAV_LIVE_DIR") and live_context:
        env["SUPERNAV_LIVE_CONTEXT"] = json.dumps(live_context, ensure_ascii=False)
    visuals = config.get("visuals_root")
    if visuals:
        env["NAV_ARTIFACTS_DIR"] = str((root / visuals).resolve())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [
                bridge_python(config),
                "-m", "supernav", "habitat-bridge",
                "--host",
                host,
                "--port",
                str(port),
                "--session-idle-timeout-s",
                "0",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + float(bridge.get("startup_timeout_s", 120))
            while True:
                if process.poll() is not None:
                    raise BridgeStartupError(f"Bridge exited; see {log_path}")
                try:
                    with opener.open(
                        f"http://{host}:{port}/healthz", timeout=2
                    ) as response:
                        health = json.load(response)
                    if health.get("ok") is True and process.poll() is None:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise BridgeStartupError(
                        f"Bridge startup timed out; see {log_path}"
                    )
                time.sleep(0.2)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
