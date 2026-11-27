"""Local service orchestration for SuperNav experiments.

The stack owns only long-lived support services. Habitat bridges remain
per-episode children and Cartographer remains a per-session bridge child.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from supernav.paths import workspace_root

STATE_VERSION = 1
DEFAULT_EXPERIMENT = "habitat-spatial-memory"
DEFAULT_CAPTION_URL = "http://127.0.0.1:8926/caption"
DEFAULT_CAPTION_MODEL = "Qwen/Qwen3.5-2B"


class StackError(RuntimeError):
    """A safe, user-facing stack lifecycle failure."""


class _StackSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(signum)


@contextmanager
def _termination_signal_handlers() -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise _StackSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _stack_dir(value: str | None = None) -> Path:
    configured = value or os.environ.get("SUPERNAV_STACK_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_root() / ".cache" / "stack"


def _state_path(root: Path) -> Path:
    return root / "state.json"


@contextmanager
def _state_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stack.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_state(root: Path) -> dict[str, Any] | None:
    path = _state_path(root)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StackError(f"Cannot read stack state {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != STATE_VERSION:
        raise StackError(f"Unsupported stack state in {path}")
    return document


def _write_state(root: Path, state: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = _state_path(root)
    temporary = root / ".state.json.tmp"
    temporary.write_text(
        json.dumps(dict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_state(root: Path) -> None:
    _state_path(root).unlink(missing_ok=True)


def _process_start_token(pid: int) -> str | None:
    """Return a PID-reuse-resistant Linux process identity token."""

    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = content.rfind(")")
        if closing < 0:
            return None
        fields = content[closing + 2 :].split()
        return fields[19] if len(fields) > 19 else None
    except (OSError, UnicodeError):
        return None


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = content.rfind(")")
        fields = content[closing + 2 :].split() if closing >= 0 else []
        if fields and fields[0] == "Z":
            return False
    except (OSError, UnicodeError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_owned_process(service: Mapping[str, Any]) -> bool:
    pid = service.get("pid")
    token = service.get("process_start_token")
    if not isinstance(pid, int) or not isinstance(token, str) or not token:
        return False
    return _process_alive(pid) and _process_start_token(pid) == token


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _caption_urls(endpoint: str) -> tuple[str, str, str, int]:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise StackError(f"Caption URL must be an HTTP(S) URL: {endpoint!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    for suffix in ("/caption_batch", "/caption", "/healthz"):
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    authority = parsed.hostname
    if ":" in authority and not authority.startswith("["):
        authority = f"[{authority}]"
    if parsed.port is not None:
        authority += f":{parsed.port}"
    base = urllib.parse.urlunparse(
        (parsed.scheme, authority, base_path, "", "", "")
    ).rstrip("/")
    return endpoint.rstrip("/"), base + "/healthz", parsed.hostname, port


def _health_document(url: str, timeout_s: float = 2.0) -> dict[str, Any] | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        request = urllib.request.Request(url, method="GET")
        with opener.open(request, timeout=max(0.1, timeout_s)) as response:
            document = json.loads(response.read().decode("utf-8") or "{}")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) and document.get("ok") is True else None


def _wait_for_health(
    *, url: str, process: subprocess.Popen[Any] | None, timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, timeout_s)
    while True:
        document = _health_document(url)
        if document is not None:
            return document
        if process is not None and process.poll() is not None:
            raise StackError(
                f"Caption service exited before becoming healthy (code {process.returncode})"
            )
        if time.monotonic() >= deadline:
            raise StackError(f"Caption service health check timed out: {url}")
        time.sleep(0.1)


def _terminate_owned_service(service: Mapping[str, Any], timeout_s: float) -> bool:
    """Stop only the recorded process identity; return whether a signal was sent."""

    if not _is_owned_process(service):
        return False
    pid = int(service["pid"])

    def send(sig: signal.Signals) -> None:
        if not _is_owned_process(service):
            return
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, timeout_s)
    while _is_owned_process(service) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _is_owned_process(service):
        send(signal.SIGKILL)
        kill_deadline = time.monotonic() + 2.0
        while _is_owned_process(service) and time.monotonic() < kill_deadline:
            time.sleep(0.05)
    return True


def _config_environment(config: Mapping[str, Any]) -> dict[str, str]:
    environment: dict[str, str] = {}
    mcp = config.get("mcp")
    if isinstance(mcp, Mapping):
        values = mcp.get("environment")
        if isinstance(values, Mapping):
            environment.update({str(key): str(value) for key, value in values.items()})
    return environment


def _load_selection(*, experiment: str | None, config_path: str | None) -> tuple[dict[str, Any], str]:
    from supernav.experiments.config import load_experiment_config, resolve_experiment

    if experiment and config_path:
        raise StackError("Select either --experiment or --config, not both")
    if config_path:
        source = str(Path(config_path).expanduser().resolve())
    else:
        selected = experiment or DEFAULT_EXPERIMENT
        source = str(resolve_experiment(selected))
    return load_experiment_config(source), source


def _runner_selection(arguments: Sequence[str]) -> tuple[str | None, str | None]:
    experiment: str | None = None
    config_path: str | None = None
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in {"--experiment", "--config"}:
            if index + 1 >= len(arguments):
                raise StackError(f"{value} requires a value")
            if value == "--experiment":
                experiment = arguments[index + 1]
            else:
                config_path = arguments[index + 1]
            index += 2
            continue
        if value.startswith("--experiment="):
            experiment = value.split("=", 1)[1]
        elif value.startswith("--config="):
            config_path = value.split("=", 1)[1]
        index += 1
    if experiment is None and config_path is None:
        raise StackError(
            "stack run requires the delegated runner's --experiment or --config"
        )
    return experiment, config_path


def _resolve_executable(command: str, root: Path) -> str:
    try:
        executable = shlex.split(command)[0]
    except (ValueError, IndexError) as exc:
        raise StackError("Cartographer worker command is invalid") from exc
    path = Path(os.path.expandvars(executable)).expanduser()
    if not path.is_absolute() and "/" in executable:
        path = root / path
    if "/" in executable and not path.is_file():
        raise StackError(f"Cartographer worker launcher is missing: {path}")
    return str(path) if "/" in executable else executable


def _preflight_cartographer(config: Mapping[str, Any]) -> dict[str, Any] | None:
    environment = _config_environment(config)
    if not _truthy(environment.get("HAB_CARTOGRAPHER_ENABLED"), False):
        return None
    command = environment.get("HAB_CARTOGRAPHER_WORKER_COMMAND", "").strip()
    if not command:
        raise StackError("HAB_CARTOGRAPHER_WORKER_COMMAND is required")
    root = workspace_root()
    executable = _resolve_executable(command, root)
    child_environment = dict(os.environ)
    child_environment.update(environment)
    try:
        completed = subprocess.run(
            [executable, "--capabilities"],
            cwd=root,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StackError(f"Cartographer capability probe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise StackError(
            f"Cartographer capability probe exited {completed.returncode}: {detail}"
        )
    try:
        capabilities = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise StackError("Cartographer capability probe returned invalid JSON") from exc
    if not isinstance(capabilities, dict):
        raise StackError("Cartographer capability probe returned a non-object")
    mode = environment.get("HAB_CARTOGRAPHER_ODOMETRY_MODE", "authoritative")
    if mode == "authoritative" and capabilities.get("authoritative_odometry") is not True:
        raise StackError("Cartographer worker lacks authoritative_odometry")
    if capabilities.get("dual_stream") is not True:
        raise StackError("Cartographer worker lacks dual_stream")
    return capabilities


def _caption_enabled(config: Mapping[str, Any]) -> bool:
    environment = _config_environment(config)
    return _truthy(environment.get("HAB_SLAM_AREA_CAPTION_ENABLED"), False)


def _caption_command(options: argparse.Namespace, *, host: str, port: int) -> list[str]:
    custom = options.caption_command or os.environ.get("SUPERNAV_CAPTION_COMMAND")
    if custom:
        try:
            command = shlex.split(custom)
        except ValueError as exc:
            raise StackError(f"Invalid caption command: {exc}") from exc
        if not command:
            raise StackError("Caption command cannot be empty")
        return command
    python = (
        options.caption_python
        or os.environ.get("SUPERNAV_CAPTION_PYTHON")
        or os.environ.get("SUPERNAV_HABITAT_PYTHON")
        or sys.executable
    )
    command = [
        python,
        "-m",
        "supernav.backends.habitat.memory_slam.qwen_caption_server",
        "--model",
        options.caption_model,
        "--host",
        host,
        "--port",
        str(port),
        "--device-map",
        options.caption_device_map,
        "--dtype",
        options.caption_dtype,
    ]
    if options.caption_no_warmup:
        command.append("--no-warmup")
    return command


def _state_service(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    services = state.get("services")
    if not isinstance(services, Mapping):
        return None
    caption = services.get("caption")
    return caption if isinstance(caption, Mapping) else None


def _existing_state_health(state: Mapping[str, Any]) -> tuple[bool, str]:
    service = _state_service(state)
    if service is None:
        return True, "ready (no managed services)"
    mode = service.get("mode")
    health_url = service.get("health_url")
    if not isinstance(health_url, str):
        return False, "invalid state"
    if mode == "managed" and not _is_owned_process(service):
        return False, "stale process identity"
    health = _health_document(health_url)
    return (health is not None, "healthy" if health is not None else "unhealthy")


def _start_or_validate(
    options: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    config_source: str,
) -> tuple[dict[str, Any], bool]:
    """Return state and whether this invocation created it."""

    root = _stack_dir(options.state_dir)
    _preflight_cartographer(config)
    caption_enabled = _caption_enabled(config)
    mode = options.service_mode
    environment = _config_environment(config)
    endpoint = (
        options.caption_url
        or environment.get("HAB_SLAM_AREA_CAPTION_URL")
        or os.environ.get("HAB_SLAM_AREA_CAPTION_URL")
        or DEFAULT_CAPTION_URL
    )
    endpoint, health_url, health_host, port = _caption_urls(endpoint)

    with _state_lock(root):
        existing = _read_state(root)
        if existing is not None:
            healthy, reason = _existing_state_health(existing)
            if healthy:
                existing_mode = existing.get("service_mode")
                existing_service = _state_service(existing)
                existing_url = existing_service.get("url") if existing_service else None
                if existing_mode != mode or (caption_enabled and existing_url != endpoint):
                    raise StackError(
                        "A different SuperNav stack is already active; run `supernav stack down` first"
                    )
                return existing, False
            service = _state_service(existing)
            if service is not None and service.get("mode") == "managed" and _is_owned_process(service):
                raise StackError(f"Owned caption service is running but {reason}; inspect stack logs")
            _remove_state(root)

        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "created_at": time.time(),
            "service_mode": mode,
            "config_source": config_source,
            "services": {},
        }
        if not caption_enabled:
            _write_state(root, state)
            return state, True

        if mode == "external":
            health = _wait_for_health(
                url=health_url, process=None, timeout_s=options.health_timeout
            )
            state["services"] = {
                "caption": {
                    "mode": "external",
                    "url": endpoint,
                    "health_url": health_url,
                    "port": port,
                    "health_state": "healthy",
                    "health": health,
                }
            }
            _write_state(root, state)
            return state, True

        if health_host not in {"127.0.0.1", "localhost", "::1"}:
            raise StackError(
                "Managed caption mode requires a loopback caption URL; use --service-mode external"
            )
        connect_host = "127.0.0.1" if health_host in {"localhost", "::1"} else health_host
        if _port_open(connect_host, port):
            raise StackError(
                f"Caption port {health_host}:{port} is occupied by an unowned process"
            )
        bind_host = "127.0.0.1" if health_host in {"localhost", "::1"} else health_host
        command = _caption_command(options, host=bind_host, port=port)
        log_path = root / "caption.log"
        child_environment = dict(os.environ)
        child_environment.update(
            {
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "no_proxy": "127.0.0.1,localhost,::1",
            }
        )
        process: subprocess.Popen[Any] | None = None
        try:
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=workspace_root(),
                    env=child_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            token = _process_start_token(process.pid)
            if token is None:
                raise StackError("Could not establish caption process identity")
            service = {
                "mode": "managed",
                "pid": process.pid,
                "process_start_token": token,
                "command": command,
                "url": endpoint,
                "health_url": health_url,
                "port": port,
                "log_path": str(log_path),
                "health_state": "starting",
            }
            state["services"] = {"caption": service}
            _write_state(root, state)
            health = _wait_for_health(
                url=health_url,
                process=process,
                timeout_s=options.health_timeout,
            )
            service["health_state"] = "healthy"
            service["health"] = health
            _write_state(root, state)
            return state, True
        except BaseException:
            if process is not None:
                token = _process_start_token(process.pid)
                if token is not None:
                    _terminate_owned_service(
                        {"pid": process.pid, "process_start_token": token},
                        options.shutdown_timeout,
                    )
            _remove_state(root)
            raise


def _down(root: Path, *, timeout_s: float) -> tuple[bool, str]:
    with _state_lock(root):
        state = _read_state(root)
        if state is None:
            return False, "Stack is already down"
        service = _state_service(state)
        stopped = False
        message = "Removed stack state"
        if service is not None and service.get("mode") == "managed":
            if _is_owned_process(service):
                stopped = _terminate_owned_service(service, timeout_s)
                message = "Stopped owned caption service"
            else:
                message = "Removed stale state without signaling an unowned process"
        elif service is not None:
            message = "Detached from external caption service"
        _remove_state(root)
        return stopped, message


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _delegate_runner(runner: str, arguments: Sequence[str]) -> int:
    from supernav.experiments import episode, paired, sweep

    entrypoints = {
        "run": sweep.main,
        "run-one": episode.main,
        "run-paired": paired.main,
    }
    entrypoint = entrypoints[runner]
    previous = sys.argv
    try:
        sys.argv = [f"supernav {runner}", *arguments]
        return int(entrypoint())
    finally:
        sys.argv = previous


def _print_state(state: Mapping[str, Any]) -> None:
    service = _state_service(state)
    if service is None:
        print("SuperNav stack ready (no long-lived services required)")
        return
    pid = f" pid={service['pid']}" if isinstance(service.get("pid"), int) else ""
    print(
        f"caption: {service.get('health_state', 'unknown')}"
        f" mode={service.get('mode')} port={service.get('port')}{pid}"
    )


def _add_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--experiment", default=None)
    selection.add_argument("--config", dest="config_path", default=None)


def _add_service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--service-mode",
        choices=("managed", "external"),
        default=os.environ.get("SUPERNAV_STACK_SERVICE_MODE", "managed"),
    )
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--caption-url", default=None)
    parser.add_argument("--caption-command", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--caption-python", default=None)
    parser.add_argument("--caption-model", default=os.environ.get("SLAM_AREA_CAPTION_MODEL", DEFAULT_CAPTION_MODEL))
    parser.add_argument("--caption-device-map", default=os.environ.get("SLAM_AREA_CAPTION_DEVICE_MAP", "auto"))
    parser.add_argument("--caption-dtype", choices=("auto", "bf16", "fp16", "fp32"), default=os.environ.get("SLAM_AREA_CAPTION_DTYPE", "auto"))
    parser.add_argument("--caption-no-warmup", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=10.0)


def _run_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="supernav stack run",
        description="Start required services, run an experiment, and clean up owned services.",
    )
    _add_service_options(parser)
    parser.add_argument("--keep-services", action="store_true")
    parser.add_argument("--runner", choices=("run", "run-one"), default="run")
    options, runner_args = parser.parse_known_args(list(argv))
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    experiment, config_path = _runner_selection(runner_args)
    config, source = _load_selection(experiment=experiment, config_path=config_path)
    created = False
    interrupted = False
    try:
        with _termination_signal_handlers():
            state, created = _start_or_validate(
                options, config=config, config_source=source
            )
            _print_state(state)
            service = _state_service(state)
            updates: dict[str, str] = {}
            if service is not None and isinstance(service.get("url"), str):
                updates["HAB_SLAM_AREA_CAPTION_URL"] = str(service["url"])
                updates["NO_PROXY"] = "127.0.0.1,localhost,::1"
                updates["no_proxy"] = "127.0.0.1,localhost,::1"
            with _temporary_environment(updates):
                return _delegate_runner(options.runner, runner_args)
    except KeyboardInterrupt:
        interrupted = True
        return 130
    except _StackSignal as exc:
        interrupted = True
        return 128 + exc.signum
    finally:
        if created and not options.keep_services:
            _, message = _down(
                _stack_dir(options.state_dir), timeout_s=options.shutdown_timeout
            )
            print(message, file=sys.stderr if interrupted else sys.stdout)


def _run_paired_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="supernav stack run-paired",
        description=(
            "Start required services, run paired exploration/retrieval sessions, "
            "and clean up owned services."
        ),
    )
    _add_service_options(parser)
    parser.add_argument("--keep-services", action="store_true")
    options, runner_args = parser.parse_known_args(list(argv))
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    experiment, config_path = _runner_selection(runner_args)
    config, source = _load_selection(experiment=experiment, config_path=config_path)
    created = False
    interrupted = False
    try:
        with _termination_signal_handlers():
            state, created = _start_or_validate(
                options, config=config, config_source=source
            )
            _print_state(state)
            service = _state_service(state)
            updates: dict[str, str] = {}
            if service is not None and isinstance(service.get("url"), str):
                updates["HAB_SLAM_AREA_CAPTION_URL"] = str(service["url"])
                updates["NO_PROXY"] = "127.0.0.1,localhost,::1"
                updates["no_proxy"] = "127.0.0.1,localhost,::1"
            with _temporary_environment(updates):
                return _delegate_runner("run-paired", runner_args)
    except KeyboardInterrupt:
        interrupted = True
        return 130
    except _StackSignal as exc:
        interrupted = True
        return 128 + exc.signum
    finally:
        if created and not options.keep_services:
            _, message = _down(
                _stack_dir(options.state_dir), timeout_s=options.shutdown_timeout
            )
            print(message, file=sys.stderr if interrupted else sys.stdout)


def _up_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="supernav stack up", description="Start and retain required support services.")
    _add_selection(parser)
    _add_service_options(parser)
    options = parser.parse_args(list(argv))
    config, source = _load_selection(
        experiment=options.experiment, config_path=options.config_path
    )
    state, created = _start_or_validate(options, config=config, config_source=source)
    _print_state(state)
    if not created:
        print("Stack was already running")
    return 0


def _status_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="supernav stack status", description="Show service ownership and health.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args(list(argv))
    root = _stack_dir(options.state_dir)
    with _state_lock(root):
        state = _read_state(root)
        if state is None:
            report = {"status": "down", "state_dir": str(root), "services": {}}
            healthy = False
        else:
            healthy, reason = _existing_state_health(state)
            report = {
                "status": "ready" if healthy else "degraded",
                "reason": reason,
                "state_dir": str(root),
                "service_mode": state.get("service_mode"),
                "services": state.get("services", {}),
            }
    if options.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif state is None:
        print("SuperNav stack is down")
    else:
        print(f"SuperNav stack: {report['status']} ({report['reason']})")
        _print_state(state)
    return 0 if healthy else 1


def _tail_lines(path: Path, count: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    except OSError as exc:
        raise StackError(f"Cannot read stack log {path}: {exc}") from exc
    return "".join(lines[-max(0, count) :])


def _logs_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="supernav stack logs", description="Read or follow managed service logs.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument("--follow", action="store_true")
    options = parser.parse_args(list(argv))
    root = _stack_dir(options.state_dir)
    with _state_lock(root):
        state = _read_state(root)
        if state is None:
            raise StackError("Stack is down; no active log is registered")
        service = _state_service(state)
        log_value = service.get("log_path") if service is not None else None
        if not isinstance(log_value, str):
            raise StackError("This stack has no managed service log")
        log_path = Path(log_value)
    sys.stdout.write(_tail_lines(log_path, options.lines))
    sys.stdout.flush()
    if not options.follow:
        return 0
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                if service is not None and service.get("mode") == "managed" and not _is_owned_process(service):
                    return 0
                time.sleep(0.2)
    except KeyboardInterrupt:
        return 130


def _down_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="supernav stack down", description="Stop only services owned by this stack.")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    options = parser.parse_args(list(argv))
    _, message = _down(_stack_dir(options.state_dir), timeout_s=options.timeout)
    print(message)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="supernav stack",
        description="Manage SuperNav's local spatial-memory support services.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("run", "Start services, delegate to run/run-one, then clean up."),
        (
            "run-paired",
            "Run exploration/retrieval pairs through immutable snapshots.",
        ),
        ("up", "Start services for a long-lived development session."),
        ("status", "Show process ownership and health."),
        ("logs", "Read or follow managed service logs."),
        ("down", "Stop only owned services and remove state."),
    ):
        actions.add_parser(action, add_help=False, help=help_text)
    selected, remaining = parser.parse_known_args(list(argv) if argv is not None else None)
    handlers = {
        "run": _run_command,
        "run-paired": _run_paired_command,
        "up": _up_command,
        "status": _status_command,
        "logs": _logs_command,
        "down": _down_command,
    }
    try:
        return handlers[selected.action](remaining)
    except (StackError, OSError, ValueError) as exc:
        print(f"supernav stack: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
