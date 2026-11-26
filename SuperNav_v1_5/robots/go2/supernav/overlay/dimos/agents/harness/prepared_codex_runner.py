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

"""Execute a prepared external Codex harness run from its launch audit."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse

from dimos.agents.mcp.external_harness_codex import (
    DEFAULT_MCP_SMOKE_ATTEMPTS,
    DEFAULT_MCP_SMOKE_RETRY_DELAY_S,
    DEFAULT_MCP_SMOKE_TIMEOUT_S,
    DEFAULT_MCP_SMOKE_TRANSPORT,
    codex_mcp_smoke_log_result,
    run_direct_mcp_smoke_gate,
)
from dimos.agents.mcp.mcp_adapter import McpAdapter

SURROUND_DIRECTIONS = ("front", "right", "back", "left")
INITIAL_OBSERVE_ATTEMPTS = 3
INITIAL_OBSERVE_RETRY_S = 1.0


@dataclass(frozen=True)
class PreparedCodexExecutionResult:
    """Outcome of executing one prepared Codex harness run."""

    ok: bool
    returncode: int
    reason: str
    audit_path: Path
    codex_stdout_path: Path
    codex_stderr_path: Path
    proxy_stdout_path: Path | None = None
    proxy_stderr_path: Path | None = None
    result_path: Path | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "reason": self.reason,
            "audit_path": str(self.audit_path),
            "codex_stdout_path": str(self.codex_stdout_path),
            "codex_stderr_path": str(self.codex_stderr_path),
            "proxy_stdout_path": str(self.proxy_stdout_path) if self.proxy_stdout_path else None,
            "proxy_stderr_path": str(self.proxy_stderr_path) if self.proxy_stderr_path else None,
            "result_path": str(self.result_path) if self.result_path else None,
        }


def load_prepared_codex_run(audit_path: Path) -> dict[str, Any]:
    """Load and lightly validate one ``prepare-codex-run`` launch audit."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    required_keys = (
        "run_root",
        "workspace",
        "prompt_path",
        "command",
        "env",
        "mode",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"launch audit is missing required keys: {', '.join(missing)}")
    if not isinstance(payload["command"], list) or not payload["command"]:
        raise ValueError("launch audit command must be a non-empty list")
    if not isinstance(payload["env"], dict):
        raise ValueError("launch audit env must be a dict")
    return payload


def execute_prepared_codex_run(
    *,
    audit_path: Path,
    wait_for_proxy_s: float = 60.0,
    stop_event: threading.Event | None = None,
) -> PreparedCodexExecutionResult:
    """Run a prepared Codex harness launch and persist structured result files."""

    audit = load_prepared_codex_run(audit_path)
    run_root = Path(str(audit["run_root"]))
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    codex_stdout_path = logs_dir / "codex-main.jsonl"
    codex_stderr_path = logs_dir / "codex-main.stderr"
    proxy_stdout_path = logs_dir / "authority-proxy.stdout"
    proxy_stderr_path = logs_dir / "authority-proxy.stderr"
    result_path = logs_dir / "codex-run-result.json"

    prompt_path = Path(str(audit["prompt_path"]))
    command = [str(part) for part in audit["command"]]
    env = {**os.environ, **{str(k): str(v) for k, v in dict(audit["env"]).items()}}
    timeout_s = _optional_float(audit.get("codex_exec_timeout_s"))
    proxy_proc: subprocess.Popen[bytes] | None = None

    try:
        proxy_command = audit.get("proxy_command")
        proxy_url = _optional_str(audit.get("proxy_url"))
        if isinstance(proxy_command, list) and proxy_command:
            if proxy_url:
                existing_listener = proxy_endpoint_listener(proxy_url)
                if existing_listener is not None:
                    reason = (
                        f"refusing to start authority proxy because {proxy_url} "
                        f"was already listening before launch: {existing_listener}. "
                        "Stop the stale authority-proxy process or choose a free proxy port."
                    )
                    _append_text(proxy_stderr_path, reason)
                    result = PreparedCodexExecutionResult(
                        ok=False,
                        returncode=1,
                        reason=reason,
                        audit_path=audit_path,
                        codex_stdout_path=codex_stdout_path,
                        codex_stderr_path=codex_stderr_path,
                        proxy_stdout_path=proxy_stdout_path,
                        proxy_stderr_path=proxy_stderr_path,
                        result_path=result_path,
                    )
                    result_path.write_text(
                        json.dumps(result.model_dump(), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    return result
            proxy_proc = _start_proxy_process(
                command=[str(part) for part in proxy_command],
                stdout_path=proxy_stdout_path,
                stderr_path=proxy_stderr_path,
                env=env,
            )
            if proxy_url:
                wait_for_proxy_ready(proxy_url, timeout_s=wait_for_proxy_s)
        smoke_ok, smoke_reason = _run_mcp_smoke_if_needed(audit=audit, env=env)
        if not smoke_ok:
            result = PreparedCodexExecutionResult(
                ok=False,
                returncode=1,
                reason=smoke_reason,
                audit_path=audit_path,
                codex_stdout_path=codex_stdout_path,
                codex_stderr_path=codex_stderr_path,
                proxy_stdout_path=proxy_stdout_path if proxy_proc is not None else None,
                proxy_stderr_path=proxy_stderr_path if proxy_proc is not None else None,
                result_path=result_path,
            )
            result_path.write_text(
                json.dumps(result.model_dump(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return result

        observe_ok, observe_reason, image_paths = _prepare_initial_pixelgoal_observe(
            audit, logs_dir
        )
        if observe_reason:
            _append_text(proxy_stdout_path, observe_reason)
        if not observe_ok:
            result = PreparedCodexExecutionResult(
                ok=False,
                returncode=1,
                reason=observe_reason or "initial PixelGoal observe failed",
                audit_path=audit_path,
                codex_stdout_path=codex_stdout_path,
                codex_stderr_path=codex_stderr_path,
                proxy_stdout_path=proxy_stdout_path if proxy_proc is not None else None,
                proxy_stderr_path=proxy_stderr_path if proxy_proc is not None else None,
                result_path=result_path,
            )
            result_path.write_text(
                json.dumps(result.model_dump(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return result
        command = _command_with_images(command, image_paths)

        returncode, reason = _run_codex_command(
            command=command,
            prompt_path=prompt_path,
            stdout_path=codex_stdout_path,
            stderr_path=codex_stderr_path,
            env=env,
            timeout_s=timeout_s,
            stop_event=stop_event,
        )
        ok = returncode == 0
        result = PreparedCodexExecutionResult(
            ok=ok,
            returncode=returncode,
            reason=reason,
            audit_path=audit_path,
            codex_stdout_path=codex_stdout_path,
            codex_stderr_path=codex_stderr_path,
            proxy_stdout_path=proxy_stdout_path if proxy_proc is not None else None,
            proxy_stderr_path=proxy_stderr_path if proxy_proc is not None else None,
            result_path=result_path,
        )
        result_path.write_text(
            json.dumps(result.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
    finally:
        if proxy_proc is not None:
            _stop_process_group(proxy_proc)


def _prepare_initial_pixelgoal_observe(
    audit: dict[str, Any], logs_dir: Path
) -> tuple[bool, str | None, list[Path]]:
    envelope = audit.get("authority_envelope")
    allowed = envelope.get("allowed_movement_tools", []) if isinstance(envelope, dict) else []
    if "pixelgoal_navigate" not in allowed:
        return True, None, []

    mcp_url = _optional_str(audit.get("proxy_url") or audit.get("mcp_url"))
    artifact_path = logs_dir / "initial-observe.json"
    if mcp_url is None:
        reason = "Initial PixelGoal observe failed: no agent-facing MCP URL."
        artifact_path.write_text(json.dumps({"ok": False, "reason": reason}), encoding="utf-8")
        return False, reason, []

    adapter = McpAdapter(mcp_url, timeout=15)
    last_reason = "observe_surround returned no images"
    last_health: dict[str, Any] = {}
    for attempt in range(1, INITIAL_OBSERVE_ATTEMPTS + 1):
        try:
            adapter.initialize()
            visible = {str(tool.get("name")) for tool in adapter.list_tools()}
            if "observe_surround" not in visible:
                raise RuntimeError("observe_surround is not visible")
            result = adapter.call_tool("observe_surround", {})
            health = _surround_summary(result)
            if health is None or not (
                isinstance(health.get("ok"), bool)
                and isinstance(health.get("timestamps"), dict)
                and isinstance(health.get("age_s"), dict)
                and isinstance(health.get("stale"), list)
            ):
                raise RuntimeError("observe_surround did not report camera freshness")
            last_health = health
            images = _surround_image_bytes(result)
            attempt_dir = logs_dir / "initial-observe" / f"attempt-{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            paths: dict[str, Path] = {}
            for direction, (data, suffix) in images.items():
                path = attempt_dir / f"{direction}{suffix}"
                path.write_bytes(data)
                paths[direction] = path
            missing = [direction for direction in SURROUND_DIRECTIONS if direction not in paths]
            stale = [
                direction
                for direction in health.get("stale", [])
                if direction in SURROUND_DIRECTIONS
            ]
            if health.get("ok") is True and not missing and not stale:
                ordered = [paths[direction] for direction in SURROUND_DIRECTIONS]
                artifact_path.write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "attempts": attempt,
                            "directions": list(SURROUND_DIRECTIONS),
                            "image_paths": [str(path) for path in ordered],
                            "timestamps": health["timestamps"],
                            "age_s": health["age_s"],
                            "stale": [],
                            "max_age_s": health.get("max_age_s"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return (
                    True,
                    "Attached initial PixelGoal views in front/right/back/left order.",
                    ordered,
                )
            problems = []
            if missing:
                problems.append(f"missing: {', '.join(missing)}")
            if stale:
                problems.append(f"stale: {', '.join(stale)}")
            last_reason = f"observe_surround {'; '.join(problems) or 'reported unhealthy'}"
        except Exception as exc:  # noqa: BLE001 — any incomplete preflight must fail closed
            last_reason = f"{exc.__class__.__name__}: {exc}"
        if attempt < INITIAL_OBSERVE_ATTEMPTS:
            time.sleep(INITIAL_OBSERVE_RETRY_S)

    reason = (
        f"Initial PixelGoal observe failed after {INITIAL_OBSERVE_ATTEMPTS} attempts: "
        f"{last_reason}"
    )
    artifact_path.write_text(
        json.dumps(
            {
                "ok": False,
                "attempts": INITIAL_OBSERVE_ATTEMPTS,
                "reason": reason,
                "camera_health": last_health,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return False, reason, []


def _surround_image_bytes(result: dict[str, Any]) -> dict[str, tuple[bytes, str]]:
    images: dict[str, tuple[bytes, str]] = {}
    label = ""
    for block in result.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = str(block.get("text") or "").strip().lower()
            label = text if text in SURROUND_DIRECTIONS else ""
        elif block.get("type") == "image" and label:
            encoded = str(block.get("data") or "")
            suffix = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }.get(str(block.get("mimeType") or ""))
            if not encoded or suffix is None:
                continue
            data = base64.b64decode(encoded, validate=True)
            images[label] = (data, suffix)
            label = ""
    return images


def _surround_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    for block in result.get("content", []):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            payload = json.loads(str(block.get("text") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _command_with_images(command: list[str], paths: list[Path]) -> list[str]:
    if not paths:
        return command
    split = len(command) - 1 if command[-1] == "-" else len(command)
    image_args = [part for path in paths for part in ("--image", str(path))]
    return [*command[:split], *image_args, *command[split:]]


def wait_for_proxy_ready(proxy_url: str, *, timeout_s: float) -> None:
    """Poll the proxy TCP endpoint until it accepts connections."""

    host, port = _tcp_endpoint_from_url(proxy_url, label="proxy_url")
    deadline = time.monotonic() + timeout_s
    last_error = "proxy not ready"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.2)
    raise TimeoutError(f"proxy {proxy_url} was not ready within {timeout_s:.1f}s: {last_error}")


def proxy_endpoint_listener(proxy_url: str) -> str | None:
    """Return a diagnostic when the proxy endpoint is already bound."""

    host, port = _tcp_endpoint_from_url(proxy_url, label="proxy_url")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((host, port)) != 0:
            return None
    owner = _tcp_listener_owner(port)
    if owner:
        return owner
    return f"{host}:{port} accepts TCP connections"


def _tcp_endpoint_from_url(proxy_url: str, *, label: str) -> tuple[str, int]:
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported {label} scheme in {proxy_url!r}")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        raise ValueError(f"{label} is missing a hostname: {proxy_url!r}")
    return host, port


def _tcp_listener_owner(port: int) -> str | None:
    for command in (
        ["ss", "-ltnp"],
        ["netstat", "-ltnp"],
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
    ):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode not in {0, 1}:
            continue
        for line in completed.stdout.splitlines():
            if f":{port}" in line:
                return " ".join(line.split())
    return None


def _run_mcp_smoke_if_needed(*, audit: dict[str, Any], env: dict[str, str]) -> tuple[bool, str]:
    if not bool(audit.get("mcp_smoke_required")):
        return True, "mcp smoke not required"
    transport = str(audit.get("mcp_smoke_transport") or "codex_exec")
    if transport == DEFAULT_MCP_SMOKE_TRANSPORT:
        result = run_direct_mcp_smoke_gate(
            mcp_url=str(audit.get("mcp_url") or audit.get("proxy_url") or ""),
            server_name=str(audit.get("server_name") or "dimos"),
            required_tools=tuple(
                str(item) for item in audit.get("mcp_smoke_required_tool_names", [])
            ),
            log_path=Path(str(audit["mcp_smoke_log_path"])),
            stderr_path=Path(str(audit["mcp_smoke_stderr_path"])),
            result_path=Path(str(audit["mcp_smoke_result_path"])),
            timeout_s=(
                _optional_float(audit.get("mcp_smoke_timeout_s"))
                or DEFAULT_MCP_SMOKE_TIMEOUT_S
            ),
        )
        return result.ok, result.reason
    command = audit.get("mcp_smoke_command")
    if not isinstance(command, list) or not command:
        return False, "mcp smoke required but mcp_smoke_command is missing"
    prompt_path = Path(str(audit["mcp_smoke_prompt_path"]))
    stdout_path = Path(str(audit["mcp_smoke_log_path"]))
    stderr_path = Path(str(audit["mcp_smoke_stderr_path"]))
    result_path = Path(str(audit["mcp_smoke_result_path"]))
    timeout_s = _optional_float(audit.get("mcp_smoke_timeout_s")) or DEFAULT_MCP_SMOKE_TIMEOUT_S
    max_attempts = max(1, int(audit.get("mcp_smoke_attempts") or DEFAULT_MCP_SMOKE_ATTEMPTS))
    retry_delay_s = (
        _optional_float(audit.get("mcp_smoke_retry_delay_s"))
        if audit.get("mcp_smoke_retry_delay_s") is not None
        else DEFAULT_MCP_SMOKE_RETRY_DELAY_S
    )
    server_name = str(audit.get("server_name") or "dimos")
    required_tools = tuple(str(item) for item in audit.get("mcp_smoke_required_tool_names", []))

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        attempt_stdout_path = (
            stdout_path if max_attempts == 1 else _attempt_path(stdout_path, attempt)
        )
        attempt_stderr_path = (
            stderr_path if max_attempts == 1 else _attempt_path(stderr_path, attempt)
        )
        with prompt_path.open("rb") as stdin:
            with attempt_stdout_path.open("wb") as stdout:
                with attempt_stderr_path.open("wb") as stderr:
                    try:
                        completed = _run_child_process_group(
                            [str(part) for part in command],
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            env=env,
                            timeout=timeout_s,
                        )
                    except subprocess.TimeoutExpired:
                        reason = f"codex exec smoke timed out after {timeout_s:.1f}s"
                        _append_text(attempt_stderr_path, reason)
                        _copy_final_attempt_logs(
                            attempt_stdout_path=attempt_stdout_path,
                            attempt_stderr_path=attempt_stderr_path,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                        result_path.write_text(
                            json.dumps(
                                {
                                    "ok": False,
                                    "returncode": -1,
                                    "reason": reason,
                                    "required_mcp_tool_names": list(required_tools),
                                    "missing_mcp_tool_names": list(required_tools),
                                    "called_mcp_tool_names": [],
                                    "attempts": attempt,
                                },
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                        return False, reason

        log_text = attempt_stdout_path.read_text(encoding="utf-8", errors="replace")
        ok, reason, missing, called = codex_mcp_smoke_log_result(
            log_text,
            server_name=server_name,
            required_tools=required_tools,
            returncode=int(completed.returncode),
        )
        stderr_diagnostic = _mcp_smoke_stderr_diagnostic(attempt_stderr_path)
        if stderr_diagnostic and not ok:
            reason = f"{reason}: {stderr_diagnostic}"
        _copy_final_attempt_logs(
            attempt_stdout_path=attempt_stdout_path,
            attempt_stderr_path=attempt_stderr_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if ok or not _retryable_mcp_smoke_disconnect(
            reason=reason,
            returncode=int(completed.returncode),
            called_mcp_tool_names=called,
        ):
            break
        if attempt < max_attempts and retry_delay_s > 0:
            time.sleep(retry_delay_s)

    if attempt > 1:
        reason = f"{reason} (after {attempt} smoke attempts)"
    result_path.write_text(
        json.dumps(
            {
                "ok": ok,
                "returncode": int(completed.returncode),
                "reason": reason,
                "required_mcp_tool_names": [
                    f"mcp__{server_name}__{tool}" for tool in required_tools
                ],
                "missing_mcp_tool_names": missing,
                "called_mcp_tool_names": called,
                "attempts": attempt,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ok, reason


def _attempt_path(path: Path, attempt: int) -> Path:
    return path.with_name(f"{path.stem}.attempt-{attempt}{path.suffix}")


def _copy_final_attempt_logs(
    *,
    attempt_stdout_path: Path,
    attempt_stderr_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    if attempt_stdout_path != stdout_path:
        stdout_path.write_bytes(attempt_stdout_path.read_bytes())
    if attempt_stderr_path != stderr_path:
        stderr_path.write_bytes(attempt_stderr_path.read_bytes())


def _retryable_mcp_smoke_disconnect(
    *,
    reason: str,
    returncode: int,
    called_mcp_tool_names: list[str],
) -> bool:
    if returncode == 0 or called_mcp_tool_names:
        return False
    lowered = reason.lower()
    return (
        "stream disconnected before completion" in lowered
        or "error sending request for url" in lowered
    )


def _mcp_smoke_stderr_diagnostic(stderr_path: Path) -> str | None:
    stderr_text = (
        stderr_path.read_text(encoding="utf-8", errors="replace")
        if stderr_path.exists()
        else ""
    )
    if "OutputTextDelta without active item" in stderr_text:
        return (
            "provider emitted malformed Responses streaming text events "
            "(response.output_text.delta without an active output item); "
            "use a Responses-compatible provider/relay before live movement"
        )
    return None


def _run_codex_command(
    *,
    command: list[str],
    prompt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str],
    timeout_s: float | None,
    stop_event: threading.Event | None = None,
) -> tuple[int, str]:
    if stop_event is not None and stop_event.is_set():
        return -signal.SIGTERM, "codex exec stopped by operator"
    with prompt_path.open("rb") as stdin:
        with stdout_path.open("wb") as stdout:
            with stderr_path.open("wb") as stderr:
                try:
                    completed = _run_child_process_group(
                        command,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=stderr,
                        env=env,
                        timeout=timeout_s,
                        stop_event=stop_event,
                    )
                except subprocess.TimeoutExpired:
                    timeout_message = (
                        f"codex exec timed out after {timeout_s:.1f}s"
                        if timeout_s is not None
                        else "codex exec timed out"
                    )
                    previous_stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                    stderr_path.write_text(
                        (previous_stderr.rstrip() + "\n" if previous_stderr else "")
                        + timeout_message
                        + "\n",
                        encoding="utf-8",
                    )
                    return -1, timeout_message
    if stop_event is not None and stop_event.is_set():
        return completed.returncode, "codex exec stopped by operator"
    if completed.returncode != 0:
        return completed.returncode, f"codex exec exited with status {completed.returncode}"
    return 0, "codex exec completed successfully"


def _start_proxy_process(
    *,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    try:
        proc = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            start_new_session=True,
        )
        stdout_file.close()
        stderr_file.close()
        return proc
    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise


def _run_child_process_group(
    command: list[str],
    *,
    stdin: Any,
    stdout: Any,
    stderr: Any,
    env: dict[str, str],
    timeout: float | None,
    stop_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a nested Codex command in its own process group so timeouts clean up children."""

    proc = subprocess.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        env=env,
        start_new_session=True,
    )
    deadline = None if timeout is None else time.monotonic() + timeout
    while proc.poll() is None:
        if stop_event is not None and stop_event.wait(timeout=0.1):
            _stop_process_group(proc)
            break
        if deadline is not None and time.monotonic() >= deadline:
            _stop_process_group(proc)
            raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)
        if stop_event is None:
            try:
                proc.wait(
                    timeout=None if deadline is None else max(0.0, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired as exc:
                _stop_process_group(proc)
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout) from exc
    returncode = proc.returncode if proc.returncode is not None else proc.wait()
    return subprocess.CompletedProcess(args=command, returncode=returncode)


def _stop_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5.0)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=1.0)
        except Exception:
            pass


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _append_text(path: Path, message: str) -> None:
    previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    path.write_text(
        (previous.rstrip() + "\n" if previous else "") + message + "\n",
        encoding="utf-8",
    )
