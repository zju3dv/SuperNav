"""Habitat bridge configuration and task-to-MCP binding."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from supernav.paths import python_paths
from supernav.runtime.config import resolve_path
from supernav.runtime.mcp import MCPServerSpec, validate_mcp_config
from supernav.methods.navigation.task_profile import resolve_task_profile


def task_profile_environment(config: Mapping[str, Any], arm_cfg: Mapping[str, Any]) -> dict[str, str]:
    """Compile the public task profile into the private MCP protocol setting."""
    profile = resolve_task_profile(config)
    key = "HAB_MCP_GLOBAL_TASK"
    for source, environment in (
        ("mcp.environment", (config.get("mcp") or {}).get("environment") or {}),
        ("arms.<selected>.environment", arm_cfg.get("environment") or {}),
    ):
        if key in environment:
            raise ValueError(
                f"{source}.{key} is not supported in experiment config; "
                "use benchmark_profile instead"
            )
    return {key: "1" if profile.is_global else "0"}


def habitat_root(config: Mapping[str, Any] | None = None, *, required: bool = False) -> Path | None:
    config = config or {}
    environment = config.get("environment", {})
    value = environment.get("habitat_root") if isinstance(environment, Mapping) else None
    value = value or os.environ.get("SUPERNAV_HABITAT_ROOT")
    if value:
        root = Path(os.path.expandvars(str(value))).expanduser().resolve()
        if not (root / "src_python" / "habitat_sim" / "__init__.py").is_file():
            raise ValueError(f"Habitat SDK package not found under {root}/src_python/habitat_sim")
        return root
    if required:
        raise ValueError("Set SUPERNAV_HABITAT_ROOT or environment.habitat_root to the Habitat-GS checkout")
    return None


def bridge_python(config: Mapping[str, Any]) -> str:
    environment = config.get("environment", {})
    value = environment.get("python") if isinstance(environment, Mapping) else None
    return str(value or os.environ.get("SUPERNAV_HABITAT_PYTHON") or sys.executable)


def scene_config_path(value: str | None, *, base: Path, config: Mapping[str, Any]) -> Path | None:
    """Resolve external scene data without relocating simulator assets."""
    path = resolve_path(value, base=base)
    if path is None or path.is_file() or Path(str(value)).is_absolute():
        return path
    root = habitat_root(config)
    if root is not None and (root / str(value)).is_file():
        return (root / str(value)).resolve()
    return path


def process_environment(config: Mapping[str, Any]) -> dict[str, str]:
    environment: dict[str, str] = {"PYTHONDONTWRITEBYTECODE": "1"}
    paths = python_paths()
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    root = habitat_root(config)
    if root is not None:
        environment["SUPERNAV_HABITAT_ROOT"] = str(root)
    return environment


def build_mcp_spec(
    *, workspace_root: Path, config: Mapping[str, Any], arm_cfg: Mapping[str, Any]
) -> MCPServerSpec:
    mcp = config.get("mcp", {})
    bridge = config.get("bridge", {})
    validate_mcp_config(mcp)
    whitelist = arm_cfg.get("tool_whitelist", [])
    if not isinstance(whitelist, list):
        raise ValueError("arm tool_whitelist must be a list")
    host = str(bridge.get("host", "127.0.0.1"))
    no_proxy = ",".join(dict.fromkeys([host, "127.0.0.1", "localhost", "::1"]))
    env = process_environment(config)
    env.update({
        "NAV_BRIDGE_HOST": host,
        "NAV_BRIDGE_PORT": str(bridge.get("port", 18911)),
        "HAB_SENSOR_DEPTH": "1" if bool(mcp.get("sensor_depth", True)) else "0",
        "HAB_MCP_TOOL_WHITELIST": ",".join(str(value) for value in whitelist),
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    })
    if config.get("visuals_root"):
        env["NAV_ARTIFACTS_DIR"] = str(resolve_path(config["visuals_root"], base=workspace_root))
    for extra in (mcp.get("environment"), arm_cfg.get("environment")):
        if isinstance(extra, Mapping):
            env.update({str(key): str(value) for key, value in extra.items()})
    env.update(task_profile_environment(config, arm_cfg))
    transport = str(mcp.get("transport", "stdio"))
    command = str(mcp.get("command") or sys.executable)
    args = tuple(str(value) for value in mcp.get(
        "args", ("-m", "supernav", "mcp", "--transport", transport)
    ))
    canonical_server = args[:3] == ("-m", "supernav", "mcp")
    http_args = mcp.get("http_args")
    if http_args is None and canonical_server:
        http_args = list(args)
        if "--transport" in http_args:
            index = http_args.index("--transport")
            if index + 1 >= len(http_args):
                raise ValueError("mcp.args --transport requires a value")
            http_args[index + 1] = "streamable-http"
        else:
            http_args.extend(("--transport", "streamable-http"))
    return MCPServerSpec(
        name=str(mcp.get("name", "habitat-gs")),
        command=command,
        args=args,
        environment=env,
        transport=transport,
        http_args=(tuple(str(value) for value in http_args) if http_args is not None else None),
    )
