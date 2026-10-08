"""Task instructions supplied to an agent runtime."""

from pathlib import Path
from typing import Any, Mapping


def agent_instructions(config: Mapping[str, Any], backend: str) -> str:
    supplied = config.get("agent_instructions")
    if isinstance(supplied, str):
        return supplied
    if isinstance(supplied, Mapping):
        return str(supplied.get(backend, ""))
    if (config.get("environment") or {}).get("backend") == "habitat":
        from supernav.backends.habitat.instructions import habitat_instructions

        return habitat_instructions(backend)
    return "You are running a controlled agent benchmark. Follow the task prompt exactly.\n"


def agent_policy(
    config: Mapping[str, Any], *, native_skill_path: Path | None = None,
    arm_cfg: Mapping[str, Any] | None = None,
) -> str:
    if "agent_policy" in config:
        return str(config["agent_policy"])
    from supernav.methods.navigation.prompts import global_task_agent_policy

    return global_task_agent_policy(config, native_skill_path=native_skill_path, arm_cfg=arm_cfg)
