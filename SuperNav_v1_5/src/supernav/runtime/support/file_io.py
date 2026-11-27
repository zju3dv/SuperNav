"""Cross-process file I/O primitives for SuperNav.

Provide sidecar locks, atomic snapshot copies and durable JSONL appends
for cooperating bridge and agent writers.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import time
from typing import Any, Dict, Iterator


# Sidecar lock-file extension. The lockfile lives next to the protected
# file (e.g. nav_status.json.lock) so it shares filesystem semantics with
# the file it guards. We use a sidecar instead of locking the file itself
# because the bridge writes via temp file + os.replace, which would unlink
# any flock held on the original inode.
_NAV_STATUS_LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def acquire_nav_status_lock(
    nav_status_path: str,
    *,
    timeout_s: float = 5.0,
) -> Iterator[None]:
    """Lock nav_status read-modify-write operations across processes.

    Use a sidecar because atomic replacement changes the status file inode.
    Read-only access observes complete snapshots without locking. Raise
    TimeoutError after timeout_s; release the lock on context exit or process death.
    """
    lock_path = nav_status_path + _NAV_STATUS_LOCK_SUFFIX
    parent = os.path.dirname(os.path.abspath(lock_path)) or "."
    os.makedirs(parent, exist_ok=True)
    # File locking permissions:
    #   - Mode 0o666 so a deployment with umask 0o002 ends up with a
    #     group-writable lock file; default umask 0o022 still yields
    #     0o644 which is fine for single-user dev.
    #   - O_RDONLY is intentional: fcntl.flock on Linux is advisory and
    #     does NOT require write access on the fd. Opening with O_RDWR
    #     would raise PermissionError whenever a second process (e.g.,
    #     a agent subprocess running under a different UID than the
    #     bridge) lacked write permission on an existing lock file.
    #     Read-only open works as long as the creator gave at least
    #     "other: read" (0o644 does).
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"acquire_nav_status_lock: failed to acquire {lock_path} "
                        f"within {timeout_s}s"
                    )
                time.sleep(0.02)
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def copy_file_atomic_under_lock(
    src_path: str,
    dst_path: str,
    *,
    lock_path: str,
) -> None:
    """Copy a small status JSON snapshot under its cross-process sidecar lock.

    Read the source while cooperating writers are locked, then replace
    the destination atomically. Raise OSError on I/O failure or TimeoutError
    when the five-second lock acquisition limit expires.
    """
    parent = os.path.dirname(os.path.abspath(dst_path)) or "."
    os.makedirs(parent, exist_ok=True)
    with acquire_nav_status_lock(lock_path):
        with open(src_path, "rb") as fsrc:
            data = fsrc.read()
        tmp_path = dst_path + ".tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dst_path)


def append_jsonl_atomic(path: str, record: Dict[str, Any]) -> None:
    """Append one JSONL record with O_APPEND and fsync.

    Write the record and newline, flush the descriptor and close it.
    Raise OSError for disk, permissions or descriptor failures; callers
    handle persistence errors.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Open with O_APPEND so the kernel handles concurrent appenders;
    # avoid Python's high-level open("a") so we can fsync the raw fd.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        data = line.encode("utf-8")
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if n <= 0:
                raise OSError(errno.EIO, f"append_jsonl_atomic: short write to {path}")
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_file_atomic(src_path: str, dst_path: str) -> None:
    """Atomically replace dst_path with src_path on the same filesystem.

    Raises FileNotFoundError for a missing source and OSError for failed renames.
    """

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"rename_file_atomic: source {src_path!r} does not exist"
        )
    parent = os.path.dirname(os.path.abspath(dst_path)) or "."
    os.makedirs(parent, exist_ok=True)
    os.replace(src_path, dst_path)


__all__ = [
    "_NAV_STATUS_LOCK_SUFFIX",
    "acquire_nav_status_lock",
    "copy_file_atomic_under_lock",
    "append_jsonl_atomic",
    "rename_file_atomic",
]
