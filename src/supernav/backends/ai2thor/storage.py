"""Process-local cache locations for the external simulator and MCP clients."""

import hashlib
import os
from pathlib import Path
import socket
import tempfile


def configure_storage(root: Path, namespace: str):
    root = root.resolve()
    base = root / "runtime" / "neednav_native" / socket.gethostname() / (namespace + "_" + str(os.getpid()))
    scratch = root / "runtime" / "tmp" / ("ddn_" + hashlib.sha256(str(base).encode()).hexdigest()[:8])
    for key, target in {
        "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch,
        "XDG_CACHE_HOME": base / "cache", "XDG_CONFIG_HOME": base / "config",
        "XDG_DATA_HOME": base / "data", "XDG_STATE_HOME": base / "state",
        "CUDA_CACHE_PATH": base / "cuda", "MPLCONFIGDIR": base / "matplotlib",
    }.items():
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Model-service requests are local IPC, not outbound provider traffic.
    # urllib honors lowercase proxy variables too, so configure both spellings.
    for key in ("NO_PROXY", "no_proxy"):
        values = [v for v in os.environ.get(key, "").split(",") if v]
        os.environ[key] = ",".join(dict.fromkeys(values + ["127.0.0.1", "localhost", "::1"]))
    graphics_tools = root / "runtime" / "neednav_native" / "tools" / "usr" / "bin"
    if graphics_tools.is_dir():
        os.environ["PATH"] = str(graphics_tools) + ":" + os.environ.get("PATH", "/usr/bin:/bin")
    tempfile.tempdir = str(scratch)
    return base

