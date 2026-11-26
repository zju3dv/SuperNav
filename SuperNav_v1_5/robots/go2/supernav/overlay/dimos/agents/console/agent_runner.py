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

"""One-shot Codex harness task execution for the operator console."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading

from dimos.agents.console.config import ConsoleConfig
from dimos.agents.console.log_stream import LogStreamEvent, LogStreamService
from dimos.agents.harness.prepared_codex_runner import (
    PreparedCodexExecutionResult,
    execute_prepared_codex_run,
)
from dimos.agents.mcp.external_harness_codex import CodexHarnessRun, prepare_codex_harness_run

PrepareCodexRun = Callable[..., CodexHarnessRun]
ExecutePreparedRun = Callable[..., PreparedCodexExecutionResult]
RunIdFactory = Callable[[], str]


class AgentRunStatus(str, Enum):
    """Lifecycle state for one blocking Codex task."""

    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AgentRunArtifacts:
    """Artifact paths surfaced to the TUI."""

    run_root: Path
    audit_path: Path | None = None
    codex_stdout_path: Path | None = None
    codex_stderr_path: Path | None = None
    result_path: Path | None = None


@dataclass(frozen=True)
class AgentRunSnapshot:
    """Thread-safe state snapshot for rendering."""

    task: str
    status: AgentRunStatus
    mode: str
    tool_profile: str
    run_root: Path
    artifacts: AgentRunArtifacts
    events: tuple[LogStreamEvent, ...] = ()
    returncode: int | None = None
    reason: str | None = None
    ok: bool | None = None
    error: str | None = None
    stop_requested: bool = False

    @property
    def running(self) -> bool:
        """Return whether the Codex task is still in flight."""

        return self.status in {
            AgentRunStatus.PREPARING,
            AgentRunStatus.RUNNING,
            AgentRunStatus.STOP_REQUESTED,
        }


class AgentRunHandle:
    """Handle for one background Codex harness run."""

    def __init__(
        self,
        *,
        task: str,
        mode: str,
        tool_profile: str,
        run_root: Path,
        executor: ThreadPoolExecutor,
        log_stream: LogStreamService | None = None,
    ) -> None:
        self._task = task
        self._mode = mode
        self._tool_profile = tool_profile
        self._executor = executor
        self._log_stream = log_stream or LogStreamService()
        self._lock = threading.Lock()
        self._events: list[LogStreamEvent] = []
        self._artifacts = AgentRunArtifacts(run_root=run_root)
        self._status = AgentRunStatus.PREPARING
        self._returncode: int | None = None
        self._reason: str | None = None
        self._ok: bool | None = None
        self._error: str | None = None
        self._stop_requested = False
        self._stop_event = threading.Event()
        self._future: Future[None] | None = None

    @property
    def stop_event(self) -> threading.Event:
        """Signal the blocking Codex runner to terminate its child process."""

        return self._stop_event

    def bind_future(self, future: Future[None]) -> None:
        """Attach the worker future and shut down its executor after completion."""

        self._future = future
        future.add_done_callback(lambda _future: self._executor.shutdown(wait=False))

    def set_artifacts(self, artifacts: AgentRunArtifacts) -> None:
        """Publish artifact paths and start tailing known logs."""

        with self._lock:
            self._artifacts = artifacts
        paths: dict[str, Path] = {}
        if artifacts.codex_stdout_path is not None:
            paths["codex"] = artifacts.codex_stdout_path
        if artifacts.codex_stderr_path is not None:
            paths["stderr"] = artifacts.codex_stderr_path
        self._log_stream.set_paths(paths)

    def set_status(self, status: AgentRunStatus) -> None:
        """Update task status."""

        with self._lock:
            if self._stop_requested and status in {
                AgentRunStatus.PREPARING,
                AgentRunStatus.RUNNING,
            }:
                self._status = AgentRunStatus.STOP_REQUESTED
            else:
                self._status = status

    def add_event(self, source: str, text: str) -> None:
        """Append a synthetic lifecycle event."""

        with self._lock:
            self._events.append(LogStreamEvent(source=source, text=text))

    def request_stop(self) -> None:
        """Request immediate cancellation of the Codex worker."""

        with self._lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._stop_event.set()
            if self._status in {AgentRunStatus.PREPARING, AgentRunStatus.RUNNING}:
                self._status = AgentRunStatus.STOP_REQUESTED
            self._events.append(
                LogStreamEvent(
                    source="console",
                    text="Agent exit requested; terminating codex worker",
                )
            )

    def complete(self, result: PreparedCodexExecutionResult) -> None:
        """Record a completed prepared-run result."""

        artifacts = AgentRunArtifacts(
            run_root=self._artifacts.run_root,
            audit_path=result.audit_path,
            codex_stdout_path=result.codex_stdout_path,
            codex_stderr_path=result.codex_stderr_path,
            result_path=result.result_path,
        )
        self.set_artifacts(artifacts)
        with self._lock:
            self._status = (
                AgentRunStatus.STOPPED
                if self._stop_requested
                else AgentRunStatus.SUCCEEDED
                if result.ok
                else AgentRunStatus.FAILED
            )
            self._ok = result.ok
            self._returncode = result.returncode
            self._reason = result.reason

    def fail(self, error: Exception) -> None:
        """Record preparation/execution failure."""

        with self._lock:
            self._status = AgentRunStatus.STOPPED if self._stop_requested else AgentRunStatus.FAILED
            self._ok = False
            self._returncode = -1
            self._reason = str(error)
            self._error = str(error)

    def wait(self, timeout: float | None = None) -> AgentRunSnapshot:
        """Wait for worker completion and return a fresh snapshot."""

        if self._future is not None:
            self._future.result(timeout=timeout)
            self._executor.shutdown(wait=True)
        return self.snapshot()

    def snapshot(self) -> AgentRunSnapshot:
        """Return a snapshot with the latest tailed log events."""

        log_events = self._log_stream.poll()
        with self._lock:
            events = (*self._events, *log_events)
            return AgentRunSnapshot(
                task=self._task,
                status=self._status,
                mode=self._mode,
                tool_profile=self._tool_profile,
                run_root=self._artifacts.run_root,
                artifacts=self._artifacts,
                events=events[-200:],
                returncode=self._returncode,
                reason=self._reason,
                ok=self._ok,
                error=self._error,
                stop_requested=self._stop_requested,
            )


class AgentTaskRunner:
    """Prepare and execute one blocking Codex task in a background worker."""

    def __init__(
        self,
        *,
        config: ConsoleConfig,
        mcp_url: str,
        run_log_dir: str | None = None,
        prepare_run: PrepareCodexRun = prepare_codex_harness_run,
        execute_run: ExecutePreparedRun = execute_prepared_codex_run,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        self._config = config
        self._mcp_url = mcp_url
        self._run_log_dir = run_log_dir
        self._prepare_run = prepare_run
        self._execute_run = execute_run
        self._run_id_factory = run_id_factory or _timestamp_run_id

    def submit(self, task: str) -> AgentRunHandle:
        """Submit a complete objective; returns immediately with a background handle."""

        normalized = " ".join(str(task or "").split())
        if not normalized:
            raise ValueError("agent task must not be empty")
        run_root = self._agent_run_root()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dimos-operate-agent")
        handle = AgentRunHandle(
            task=normalized,
            mode=self._config.resolved_agent_mode.value,
            tool_profile=self._config.tool_profile,
            run_root=run_root,
            executor=executor,
        )
        future = executor.submit(self._run, handle, normalized, run_root)
        handle.bind_future(future)
        return handle

    def _run(self, handle: AgentRunHandle, task: str, run_root: Path) -> None:
        try:
            handle.add_event("console", "prepare_codex_harness_run starting")
            prepared = self._prepare_run(
                run_root=run_root,
                task=task,
                mcp_url=self._mcp_url,
                model=self._config.model or None,
                model_reasoning_effort=self._config.reasoning or None,
                mode=self._config.resolved_codex_harness_mode.value,
                prompt_mode=self._config.prompt_mode,
                tool_profile=self._config.tool_profile,
                authority_allowed_movement_tools=self._config.authority_tools,
            )
            artifacts = AgentRunArtifacts(
                run_root=Path(prepared.run_root),
                audit_path=Path(prepared.audit_path),
                codex_stdout_path=Path(prepared.run_root) / "logs" / "codex-main.jsonl",
                codex_stderr_path=Path(prepared.run_root) / "logs" / "codex-main.stderr",
                result_path=Path(prepared.run_root) / "logs" / "codex-run-result.json",
            )
            handle.set_artifacts(artifacts)
            handle.add_event("console", f"prepare_codex_harness_run ok: {prepared.audit_path}")
            handle.set_status(AgentRunStatus.RUNNING)
            result = self._execute_run(
                audit_path=Path(prepared.audit_path), stop_event=handle.stop_event
            )
            handle.complete(result)
            handle.add_event("console", f"codex task finished: {result.reason}")
        except Exception as exc:
            handle.fail(exc)
            handle.add_event("console", f"agent task failed: {exc}")

    def _agent_run_root(self) -> Path:
        base = self._config.run_root
        if base is None and self._run_log_dir:
            base = Path(self._run_log_dir) / "operator-console"
        if base is None:
            base = Path.cwd() / "runs" / "operator-console"
        return base.expanduser() / self._run_id_factory()


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
