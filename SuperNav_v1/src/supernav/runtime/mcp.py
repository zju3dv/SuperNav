"""An MCP process description consumed by agent clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    environment: Mapping[str, str]
    transport: str = "stdio"
    http_args: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_mcp_config(mcp: Mapping[str, Any]) -> None:
    """Reject unsupported process launch fields."""
    for field in ("server", "python", "module", "env"):
        if field in mcp:
            raise ValueError(f"mcp.{field} is not supported; use mcp.command/mcp.args and mcp.environment")
    for field in ("args", "http_args"):
        if mcp.get(field) is not None and not isinstance(mcp[field], (list, tuple)):
            raise ValueError(f"mcp.{field} must be an argument list")


def resolve_mcp_spec(
    *, workspace_root: Path, config: Mapping[str, Any], arm_cfg: Mapping[str, Any]
) -> MCPServerSpec:
    mcp = config.get("mcp") or {}
    validate_mcp_config(mcp)
    from supernav.backends import backend_callable

    name = (config.get("environment") or {}).get("backend")
    if name:
        builder = backend_callable(name, "mcp_spec")
        if builder is not None:
            return builder(workspace_root=workspace_root, config=config, arm_cfg=arm_cfg)
    prepared = config.get("_mcp_launch")
    if isinstance(prepared, Mapping):
        return MCPServerSpec(
            name=str(prepared["name"]),
            command=str(prepared["command"]),
            args=tuple(str(value) for value in prepared.get("args", ())),
            environment=dict(prepared.get("environment", {})),
            transport=str(prepared.get("transport", "stdio")),
            http_args=(
                tuple(str(value) for value in prepared["http_args"])
                if prepared.get("http_args") is not None
                else None
            ),
        )
    if not mcp.get("command"):
        raise ValueError("MCP requires mcp.command and mcp.args, or an explicit environment.backend")
    environment = {
        str(key): str(value)
        for key, value in dict(mcp.get("environment", {})).items()
    }
    environment.update(
        {str(key): str(value) for key, value in dict(arm_cfg.get("environment", {})).items()}
    )
    return MCPServerSpec(
        name=str(mcp.get("name", "environment")),
        command=str(mcp["command"]),
        args=tuple(str(value) for value in mcp.get("args", ())),
        environment=environment,
        transport=str(mcp.get("transport", "stdio")),
        http_args=(tuple(str(value) for value in mcp["http_args"]) if mcp.get("http_args") is not None else None),
    )
