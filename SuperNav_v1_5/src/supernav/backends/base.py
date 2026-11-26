"""Simulator-independent experiment contracts; implementations own their protocols."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class BridgeStartupError(RuntimeError):
    """An owned simulator process did not become healthy."""


@dataclass
class EpisodeTask:
    task_id: str | None
    instruction: str
    scene: str
    scene_config: str = ""
    spawn: dict = field(default_factory=dict)
    ground_truth: dict | None = None
    goal: dict | None = None
    warning: str | None = None


class EnvironmentBackend(Protocol):
    name: str
    default_port: int

    def tasks(self, config: dict, instructions: str | None, root: Path) -> list[dict]: ...
    def prepare_task(self, config: dict, args: Any, arm: dict, root: Path) -> EpisodeTask: ...
    def configure_agent(self, config: dict, arm: dict, task: EpisodeTask, root: Path, run_dir: Path) -> None: ...
    def prompt(self, *, task: EpisodeTask, arm_name: str, arm_cfg: dict, workspace_root: Path, prompts_dir: Path) -> str: ...
    def episode(self, *, root: Path, config: dict, task: EpisodeTask, run_dir: Path, dry_run: bool, live_context: dict): ...
    def collect(self, *, config: dict, root: Path, run_dir: Path, task: EpisodeTask, args: Any, result: Any, events: list[dict], session: Any) -> dict: ...
    def replay(self, run_dir: Path, config: dict, root: Path) -> dict: ...
