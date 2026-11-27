# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin

import click
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
import requests
import typer

from dimos.agents.harness.artifact_ledger import (
    ArtifactLedger,
    ArtifactRef,
    ImageArtifactReadback,
    is_valid_artifact_id,
)
from dimos.agents.harness.display_content import content_to_display_text
from dimos.agents.harness.go2_harness_profiles import (
    GO2_STACK_DISABLED_MODULES,
    resolve_go2_profile_axes,
)
from dimos.agents.harness.go2_prior_run import (
    Go2PriorRunArtifacts,
    resolve_go2_prior_run,
    seed_go2_prior_graph_memory,
)
from dimos.agents.harness.image_blocks import (
    image_block_mime_type,
    image_block_payload_bytes,
    is_image_block,
)
from dimos.agents.mcp.external_harness_validation import validate_external_harness_surface
from dimos.agents.mcp.mcp_adapter import McpAdapter, McpError
from dimos.constants import CONFIG_DIR, LOG_DIR
from dimos.core.daemon import daemonize, install_signal_handlers
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.run_registry import get_most_recent, is_pid_alive, stop_entry
from dimos.mapping.utils.cli.map import main as _map_main
from dimos.mapping.utils.cli.pose_fill import main as _map_pose_fill_main
from dimos.mapping.utils.cli.rename import main as _map_rename_main
from dimos.mapping.utils.cli.replay import main as _map_replay_main
from dimos.mapping.utils.cli.replay_marker import main as _map_replay_marker_main
from dimos.mapping.utils.cli.summary import main as _map_summary_main
from dimos.robot.unitree.go2.cli.go2tool import app as go2tool_app
from dimos.utils.logging_config import setup_logger
from dimos.visualization.rerun.constants import RerunOpenOption

if TYPE_CHECKING:
    from dimos.core.coordination.blueprints import Blueprint, BlueprintAtom
    from dimos.core.module import ModuleBase

logger = setup_logger()

main = typer.Typer(
    help="Dimensional CLI",
    no_args_is_help=True,
)

load_dotenv()

SIMULATORS = ("mujoco", "dimsim")
_HIDDEN_AGENT_POLICY_OPTIONS = {
    "agent_control",
    "agent_movement_mode",
    "agent_profile",
    "agent_tool_policy_admin_override",
    "agent_tool_scope",
}
EDGE_PROFILE_ENV = "DIMOS_EDGE_PROFILE"
MOTION_ONLY_EDGE_PROFILE = "motion-only"
MOTION_ONLY_DISABLE_MODULES = GO2_STACK_DISABLED_MODULES["motion"]
NAVIGATION_LITE_EDGE_PROFILE = "navigation-lite"
NAVIGATION_LITE_DISABLE_MODULES = GO2_STACK_DISABLED_MODULES["mapping"]
GO2_RELOCALIZATION_BLUEPRINT = "unitree-go2-relocalization"
GO2_RELOCALIZATION_PRIOR_OPTIONS = {
    "relocalizationmodule.publish_loaded_map": "true",
    "relocalizationmodule.publish_static_map": "true",
}


def _normalize_simulation_argv(argv: list[str]) -> list[str]:
    """Keep `--simulation` backwards compatible.

    Without an argument it should be `mujoco`, but can be overridden.
    """
    out: list[str] = []
    for arg, nxt in zip(argv, [*argv[1:], None], strict=False):
        out.append(arg)
        if arg == "--simulation" and nxt not in SIMULATORS:
            out.append(SIMULATORS[0])
    return out


def cli_main() -> None:
    sys.argv = _normalize_simulation_argv(sys.argv)
    main()


def create_dynamic_callback():  # type: ignore[no-untyped-def]
    fields = GlobalConfig.model_fields

    # Build the function signature dynamically
    params = [
        inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context),
    ]

    # Create parameters for each field in GlobalConfig
    for field_name, field_info in fields.items():
        field_type = field_info.annotation

        # Handle Optional types
        # Check for Optional/Union with None
        if get_origin(field_type) is type(str | None):
            inner_types = get_args(field_type)
            if len(inner_types) == 2 and type(None) in inner_types:
                # It's Optional[T], get the actual type T
                actual_type = next(t for t in inner_types if t != type(None))
            else:
                actual_type = field_type
        else:
            actual_type = field_type

        # Convert field name from snake_case to kebab-case for CLI
        cli_option_name = field_name.replace("_", "-")

        # Special handling for boolean fields
        if actual_type is bool:
            # For boolean fields, create --flag/--no-flag pattern
            param = inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=typer.Option(
                    None,  # None means use the model's default if not provided
                    f"--{cli_option_name}/--no-{cli_option_name}",
                    help=f"Override {field_name} in GlobalConfig",
                    hidden=field_name in _HIDDEN_AGENT_POLICY_OPTIONS,
                ),
                annotation=bool | None,
            )
        else:
            # For non-boolean fields, use regular option
            param = inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=typer.Option(
                    None,  # None means use the model's default if not provided
                    f"--{cli_option_name}",
                    help=f"Override {field_name} in GlobalConfig",
                    hidden=field_name in _HIDDEN_AGENT_POLICY_OPTIONS,
                ),
                annotation=actual_type | None,
            )
        params.append(param)

    def callback(**kwargs) -> None:  # type: ignore[no-untyped-def]
        ctx = kwargs.pop("ctx")
        ctx.obj = {k: v for k, v in kwargs.items() if v is not None}

    callback.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    return callback


main.callback()(create_dynamic_callback())  # type: ignore[no-untyped-call]
main.add_typer(go2tool_app, name="go2tool")


def arg_help(
    config: type[BaseModel],
    blueprint: Blueprint,
    indent: str = "    ",
    module: str = "",
    _atom: BlueprintAtom | None = None,
) -> str:
    output = ""
    for k, info in config.model_fields.items():
        if k == "g":
            continue
        t = info.annotation
        if isinstance(t, types.GenericAlias):
            # Can't be specified on CLI
            continue

        # TODO(PY314): if isinstance(t, Union):
        if get_origin(t) in {Union, types.UnionType}:
            with suppress(StopIteration):
                t = next(u for u in get_args(t) if issubclass(u, BaseModel))

        if inspect.isclass(t) and issubclass(t, BaseModel):
            output += f"{indent}{module}{k}:\n"
            # Find blueprint atom
            bp = next(bp for bp in blueprint.blueprints if bp.module.name == k)
            output += arg_help(
                t, blueprint, indent=indent + "  ", module=module + k + ".", _atom=bp
            )
        else:
            assert _atom is not None
            # Use __name__ to avoid "<class 'int'>" style output on basic types.
            display_type = t.__name__ if isinstance(t, type) else t
            required = "[Required] " if info.is_required() and k not in _atom.kwargs else ""
            d = _atom.kwargs.get(k, info.default)
            default = f" (default: {d})" if d is not PydanticUndefined else ""
            output += f"{indent}* {required}{module}{k}: {display_type}{default}\n"
    return output


def load_config_args(config: type[BaseModel], args: Iterable[str], path: Path) -> dict[str, Any]:
    try:
        kwargs = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        kwargs = {}

    for k, v in os.environ.items():
        parts = k.lower().split("__")
        if parts[0] not in config.model_fields:
            continue
        d = kwargs
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v

    for arg in args:
        k, _, v = arg.partition("=")
        parts = k.split(".")
        d = kwargs
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v

    # We don't need this config, but this atleast validates the user input first.
    # This will help catch misspellings and similar mistakes.
    config(**kwargs)

    return kwargs  # type: ignore[no-any-return]


def _blueprint_arg_key(arg: str) -> str:
    return arg.partition("=")[0].strip().lower()


def _has_blueprint_arg(args: Iterable[str], key: str) -> bool:
    wanted = key.strip().lower()
    return any(_blueprint_arg_key(arg) == wanted for arg in args)


def _resolve_prior_run_startup(
    *,
    robot_types: Iterable[str],
    blueprint_args: Iterable[str],
    prior_run_root: Path | None,
) -> tuple[list[str], list[str], Go2PriorRunArtifacts | None, tuple[str, ...]]:
    resolved_robot_types = list(robot_types)
    resolved_blueprint_args = list(blueprint_args)
    if prior_run_root is None:
        return resolved_robot_types, resolved_blueprint_args, None, ()

    artifacts = resolve_go2_prior_run(prior_run_root)
    messages: list[str] = []
    if artifacts is None or artifacts.relocalization_map_file is None:
        messages.append(
            "Prior run root has no relocalization map; backend relocalization was not enabled."
        )
        return resolved_robot_types, resolved_blueprint_args, artifacts, tuple(messages)

    if GO2_RELOCALIZATION_BLUEPRINT not in resolved_robot_types:
        resolved_robot_types.append(GO2_RELOCALIZATION_BLUEPRINT)
        messages.append(f"Added {GO2_RELOCALIZATION_BLUEPRINT} for prior-run relocalization.")

    derived_options = {
        "relocalizationmodule.map_file": str(artifacts.relocalization_map_file),
        **GO2_RELOCALIZATION_PRIOR_OPTIONS,
    }
    for key, value in derived_options.items():
        if _has_blueprint_arg(resolved_blueprint_args, key):
            messages.append(f"Kept explicit -o {key}=... over prior-run default.")
            continue
        resolved_blueprint_args.append(f"{key}={value}")
    return resolved_robot_types, resolved_blueprint_args, artifacts, tuple(messages)


def _split_disable_args(disable: Iterable[str]) -> tuple[str, ...]:
    return tuple(name.strip() for value in disable for name in value.split(",") if name.strip())


def _resolve_disabled_module_classes(disable: Iterable[str]) -> tuple[type[ModuleBase], ...]:
    from dimos.robot.get_all_blueprints import class_name_to_registry_key, get_module_by_name

    modules: list[type[ModuleBase]] = []
    seen: set[type[ModuleBase]] = set()
    for name in _split_disable_args(disable):
        candidates = dict.fromkeys((name, class_name_to_registry_key(name)))
        errors: list[ValueError] = []
        for candidate in candidates:
            try:
                module = get_module_by_name(candidate).blueprints[0].module
                break
            except ValueError as exc:
                errors.append(exc)
        else:
            msg = f"Unknown module to disable: {name!r}. Use a module kebab-name or class name."
            if errors:
                msg += f" {errors[-1]}"
            raise ValueError(msg) from errors[-1] if errors else None

        if module not in seen:
            seen.add(module)
            modules.append(module)
    return tuple(modules)


def _edge_profile_disable_args(edge_profile: str | None) -> tuple[str, ...]:
    if not edge_profile or not edge_profile.strip():
        return ()

    return resolve_go2_profile_axes(edge_profile, "inspect").disabled_modules


def _disable_args_with_edge_profile(
    disable: Iterable[str], edge_profile: str | None
) -> tuple[str, ...]:
    return (*disable, *_edge_profile_disable_args(edge_profile))


def _disabled_dependency_errors(blueprint: Blueprint) -> tuple[str, ...]:
    from dimos.core.module import is_module_type
    from dimos.spec.utils import is_spec, spec_structural_compliance

    disabled = set(blueprint.disabled_modules_tuple)
    if not disabled:
        return ()

    errors: list[str] = []
    active = blueprint.active_blueprints
    for atom in active:
        for module_ref in atom.module_refs:
            if module_ref.optional:
                continue

            spec = blueprint.remapping_map.get((atom.module, module_ref.name), module_ref.spec)
            if is_module_type(spec):
                if spec in disabled:
                    errors.append(
                        f"{atom.module.__name__}.{module_ref.name} requires disabled module "
                        f"{spec.__name__}"
                    )
                continue

            if not is_spec(spec):
                continue

            has_active_provider = any(
                other != atom and spec_structural_compliance(other.module, spec) for other in active
            )
            if has_active_provider:
                continue

            disabled_providers = [
                other.module.__name__
                for other in blueprint.blueprints
                if other.module in disabled and spec_structural_compliance(other.module, spec)
            ]
            if disabled_providers:
                errors.append(
                    f"{atom.module.__name__}.{module_ref.name} requires disabled provider "
                    f"for {spec.__name__}: {', '.join(disabled_providers)}"
                )
    return tuple(errors)


@main.command()
def run(
    ctx: typer.Context,
    robot_types: list[str] = typer.Argument(..., help="Blueprints or modules to run"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run in background"),
    disable: list[str] = typer.Option(
        [], "--disable", help="Module names to disable; repeat or pass comma-separated names"
    ),
    stack: str = typer.Option(
        "",
        "--stack",
        envvar="DIMOS_STACK",
        help="Server modules: full, mapping, or motion.",
    ),
    control: str = typer.Option(
        "",
        "--control",
        envvar="DIMOS_CONTROL",
        help="Go2 tool authority: inspect, direct, local-planner, or frontier.",
    ),
    blueprint_args: list[str] = typer.Option((), "--option", "-o"),
    prior_run_root: Path | None = typer.Option(
        None,
        "--prior-run-root",
        help="Prior Go2 run root used to seed backend graph memory and SLAM relocalization.",
    ),
    config_path: Path = typer.Option(
        CONFIG_DIR / "dimos", "--config", "-c", help="Path to config file"
    ),
    show_help: bool = typer.Option(False, "--help"),
) -> None:
    """Start a robot blueprint"""
    axes = None
    legacy_stack = os.environ.get(EDGE_PROFILE_ENV)
    is_go2_run = any("go2" in name.casefold() for name in robot_types)
    if prior_run_root is not None and not is_go2_run:
        typer.echo(
            typer.style("--prior-run-root is supported for Go2 runs only.", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(1)
    if is_go2_run:
        try:
            axes = resolve_go2_profile_axes(
                stack or legacy_stack or "full",
                control or "local-planner",
            )
        except ValueError as exc:
            typer.echo(typer.style(str(exc), fg=typer.colors.RED), err=True)
            raise typer.Exit(1) from exc
    elif stack or control or legacy_stack:
        message = "--stack/--control currently support Go2 blueprints only."
        typer.echo(typer.style(message, fg=typer.colors.RED), err=True)
        raise typer.Exit(1)

    logger.info("Starting DimOS")

    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from dimos.core.coordination.process_lifecycle import (
        DIMOS_RUN_ID_ENV,
        spawn_watchdog,
    )
    from dimos.core.run_registry import (
        RunEntry,
        cleanup_stale,
        generate_run_id,
    )
    from dimos.robot.get_all_blueprints import get_by_name_or_exit
    from dimos.utils.logging_config import set_run_log_dir, setup_exception_handler

    setup_exception_handler()

    try:
        robot_types, blueprint_args, prior_artifacts, prior_messages = _resolve_prior_run_startup(
            robot_types=robot_types,
            blueprint_args=blueprint_args,
            prior_run_root=prior_run_root,
        )
    except ValueError as exc:
        typer.echo(typer.style(f"Error: {exc}", fg=typer.colors.RED), err=True)
        raise typer.Exit(1) from exc
    for message in prior_messages:
        logger.info(message)
        typer.echo(message)

    cli_config_overrides: dict[str, Any] = dict(ctx.obj)
    if axes is not None:
        cli_config_overrides.update(
            agent_profile="supervised_live",
            agent_control=axes.control,
            agent_movement_mode="task_authority",
            agent_tool_policy_admin_override=False,
            agent_tool_scope=axes.tool_scope,
        )

    # this is a workaround until we have a proper way to have delayed-module-choice in blueprints
    # ex: vis_module(viewer=global_config.viewer) is wrong (viewer will always be default value) without this patch
    global_config.update(**cli_config_overrides)

    # Clean stale registry entries
    stale = cleanup_stale()
    if stale:
        logger.info(f"Cleaned {stale} stale run entries")

    blueprint_name = "-".join(robot_types)
    run_id = generate_run_id(blueprint_name)
    log_dir = LOG_DIR / run_id

    # Tag every descendant with the run id so the watchdog and stale-run
    # cleanup can identify them via os.environ after main dies.
    os.environ[DIMOS_RUN_ID_ENV] = run_id

    # Route structured logs (main.jsonl) to the per-run directory.
    # Workers inherit DIMOS_RUN_LOG_DIR env var via forkserver.
    set_run_log_dir(log_dir)

    blueprint = autoconnect(*map(get_by_name_or_exit, robot_types))

    try:
        stack_disable = axes.disabled_modules if axes is not None else ()
        disable_args = (*disable, *stack_disable)
        disabled_classes = _resolve_disabled_module_classes(disable_args) if disable_args else ()
    except ValueError as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED), err=True)
        raise typer.Exit(1) from exc

    if disabled_classes:
        if stack_disable:
            assert axes is not None
            logger.info(
                "DimOS stack active",
                stack=axes.stack,
                control=axes.control,
                disable_modules=list(stack_disable),
            )
        logger.info(
            "Disabling DimOS modules",
            modules=[module.__name__ for module in disabled_classes],
        )
        blueprint = blueprint.disabled_modules(*disabled_classes)
        dependency_errors = _disabled_dependency_errors(blueprint)
        if dependency_errors:
            for error in dependency_errors:
                typer.echo(
                    typer.style(f"Cannot disable module: {error}", fg=typer.colors.RED),
                    err=True,
                )
            raise typer.Exit(1)

    if show_help:
        print("Blueprint arguments:")
        print(arg_help(blueprint.config(), blueprint))
        return

    prior_graph_seed = seed_go2_prior_graph_memory(
        prior_artifacts,
        graph_memory_path=log_dir / "go2-graph-memory.json",
    )
    if prior_graph_seed is not None:
        logger.info("Seeded prior-run graph memory", prior_graph_seed=prior_graph_seed)

    run_config_overrides: dict[str, Any] = dict(cli_config_overrides)
    if prior_artifacts is not None:
        run_config_overrides.update(
            {
                "prior_run_root": str(prior_artifacts.run_root),
                "prior_run_artifacts": prior_artifacts.model_dump(),
                "prior_graph_seed": prior_graph_seed,
            }
        )

    blueprint_config = blueprint.config()
    kwargs = load_config_args(blueprint_config, blueprint_args, config_path)
    if cli_config_overrides:
        kwargs["g"] = cli_config_overrides

    coordinator = ModuleCoordinator.build(blueprint, kwargs)

    if daemon:
        # Health check before daemonizing — catch early crashes
        if not coordinator.health_check():
            typer.echo("Error: health check failed — a worker process died.", err=True)
            coordinator.stop()
            raise typer.Exit(1)

        n_modules = coordinator.n_modules
        typer.echo(f"✓ All modules started ({n_modules} modules)")
        typer.echo("✓ Health check passed")
        typer.echo("✓ DimOS running in background\n")
        typer.echo(f"  Run ID:    {run_id}")
        typer.echo(f"  Log:       {log_dir}")
        typer.echo("  Stop:      dimos stop")
        typer.echo("  Status:    dimos status")

        coordinator.suppress_console()

        daemonize(log_dir)

        coordinator.start_rpc_service()  # After daemonize().
        entry = RunEntry(
            run_id=run_id,
            pid=os.getpid(),
            blueprint=blueprint_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_dir=str(log_dir),
            cli_args=list(robot_types),
            config_overrides=run_config_overrides,
            original_argv=sys.argv,
        )
        entry.save()
        spawn_watchdog(run_id, log_dir=log_dir)
        install_signal_handlers(entry, coordinator)
        coordinator.loop()
    else:
        coordinator.start_rpc_service()
        entry = RunEntry(
            run_id=run_id,
            pid=os.getpid(),
            blueprint=blueprint_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_dir=str(log_dir),
            cli_args=list(robot_types),
            config_overrides=run_config_overrides,
            original_argv=sys.argv,
        )
        entry.save()
        spawn_watchdog(run_id, log_dir=log_dir)
        # Foreground: only SIGTERM goes through the handler. SIGINT stays at
        # default so Ctrl+C raises KeyboardInterrupt and the try/finally below
        # runs with a visible traceback.
        install_signal_handlers(entry, coordinator, sigint=False)
        try:
            coordinator.loop()
        finally:
            entry.remove()


@main.command()
def status() -> None:
    """Show the running DimOS instance."""
    entry = get_most_recent(alive_only=True)
    if not entry:
        typer.echo("No running DimOS instance")
        return

    try:
        started = datetime.fromisoformat(entry.started_at)
        age = datetime.now(timezone.utc) - started
        hours, remainder = divmod(int(age.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m {seconds}s"
    except Exception:
        uptime = "unknown"

    typer.echo(f"  Run ID:    {entry.run_id}")
    typer.echo(f"  PID:       {entry.pid}")
    typer.echo(f"  Blueprint: {entry.blueprint}")
    typer.echo(f"  Uptime:    {uptime}")
    typer.echo(f"  Log:       {entry.log_dir}")


@main.command("operate")
def operate(
    blueprint: str = typer.Option(
        "unitree-go2-edge-mcp",
        "--blueprint",
        help="Blueprint to start after explicit operator confirmation.",
    ),
    attach: bool = typer.Option(
        True,
        "--attach/--start",
        help="Attach first by default; --start still requires operator confirmation.",
    ),
    mcp_url: str = typer.Option(
        "",
        "--mcp-url",
        envvar="DIMOS_MCP_URL",
        help="Explicit MCP URL. Overrides active-run discovery.",
    ),
    mcp_ready_timeout_s: float = typer.Option(
        10.0,
        "--mcp-ready-timeout-s",
        help="Seconds to wait for MCP initialize/tools readiness.",
    ),
    stack: str = typer.Option(
        "full",
        "--stack",
        "--profile",
        "--edge-profile",
        envvar="DIMOS_STACK",
        help=("Server modules: full, mapping, or motion. Old edge profile names remain aliases."),
    ),
    control: str = typer.Option(
        "local-planner",
        "--control",
        "--tool-profile",
        envvar="DIMOS_CONTROL",
        help=(
            "Tool authority: inspect, direct, local-planner, or frontier. "
            "Old tool profile names remain aliases."
        ),
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help=(
            "Agent mode: auto, readonly, authority-proxy, admin-override-debug, "
            "or agent-long-inst."
        ),
        hidden=True,
    ),
    authority_tool: list[str] = typer.Option(
        [],
        "--authority-tool",
        help="Restrict agent movement authority to these tools (repeatable).",
    ),
    model: str = typer.Option(
        "gpt-5.4",
        "--model",
        help="Codex model for future agent submissions.",
    ),
    model_reasoning_effort: str = typer.Option(
        "medium",
        "--model-reasoning-effort",
        help="Codex reasoning effort for future agent submissions.",
    ),
    run_root: Path | None = typer.Option(
        None,
        "--run-root",
        help="Optional operator-console artifact root override.",
    ),
    prior_run_root: Path | None = typer.Option(
        None,
        "--prior-run-root",
        help="Prior Go2 run root used to seed graph memory and SLAM relocalization.",
    ),
    viewer: str = typer.Option(
        "none",
        "--viewer",
        help="Viewer setting forwarded only if the console starts DimOS.",
    ),
    no_launch: bool = typer.Option(
        False,
        "--no-launch",
        help="Never start DimOS from the console; attach/probe only.",
    ),
) -> None:
    """Launch the DimOS operator console."""

    from dimos.agents.console.app import run_console
    from dimos.agents.console.config import ConsoleConfig, StartupPolicy

    try:
        config = ConsoleConfig(
            blueprint=blueprint,
            startup_policy=StartupPolicy.ATTACH if attach else StartupPolicy.START,
            no_launch=no_launch,
            mcp_url=mcp_url or None,
            mcp_ready_timeout_s=mcp_ready_timeout_s,
            stack=stack,
            control=control,
            mode=mode,
            model=model,
            reasoning=model_reasoning_effort,
            run_root=run_root,
            prior_run_root=prior_run_root,
            viewer=viewer,
            authority_tools=tuple(authority_tool),
        )
    except ValueError as exc:
        typer.echo(typer.style(f"Error: {exc}", fg=typer.colors.RED), err=True)
        raise typer.Exit(1) from exc

    run_console(config)


@main.command()
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)"),
) -> None:
    """Stop the running DimOS instance."""

    entry = get_most_recent(alive_only=True)
    if not entry:
        typer.echo("No running DimOS instance", err=True)
        raise typer.Exit(1)

    sig_name = "SIGKILL" if force else "SIGTERM"
    typer.echo(f"Stopping {entry.run_id} (PID {entry.pid}) with {sig_name}...")
    msg, _ok = stop_entry(entry, force=force)
    typer.echo(f"  {msg}")


@main.command("log")
def log_cmd(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_lines: bool = typer.Option(False, "--all", "-a", help="Show full log"),
    json_output: bool = typer.Option(False, "--json", help="Raw JSONL output"),
    run_id: str = typer.Option("", "--run", "-r", help="Specific run ID"),
) -> None:
    """View logs from a DimOS run."""
    from dimos.core.log_viewer import follow_log, format_line, read_log, resolve_log_path

    path = resolve_log_path(run_id)
    if not path:
        typer.echo("No log files found", err=True)
        raise typer.Exit(1)

    if follow:
        import signal

        _stop = False

        def _on_sigint(_sig: int, _frame: object) -> None:
            nonlocal _stop
            _stop = True

        prev = signal.signal(signal.SIGINT, _on_sigint)
        try:
            for line in follow_log(path, stop=lambda: _stop):
                typer.echo(format_line(line, json_output=json_output))
        finally:
            signal.signal(signal.SIGINT, prev)
    else:
        count = None if all_lines else lines
        for line in read_log(path, count):
            typer.echo(format_line(line, json_output=json_output))


@main.command("trace")
def trace_cmd(
    event_type: list[str] = typer.Option(
        [],
        "--type",
        "-t",
        help="Only show this harness event type. Can be repeated.",
    ),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to scan"),
    all_lines: bool = typer.Option(False, "--all", "-a", help="Scan the full log"),
    json_output: bool = typer.Option(False, "--json", help="Raw JSON events"),
    run_id: str = typer.Option("", "--run", "-r", help="Specific run ID"),
) -> None:
    """View structured harness trace events from a DimOS run."""

    from dimos.agents.harness.trace_log import (
        extract_harness_trace_event,
        format_harness_trace_event,
    )
    from dimos.core.log_viewer import read_log, resolve_log_path

    path = resolve_log_path(run_id)
    if not path:
        typer.echo("No log files found", err=True)
        raise typer.Exit(1)

    wanted = set(event_type)
    count = None if all_lines else lines
    found = 0
    for raw in read_log(path, count):
        event = extract_harness_trace_event(raw)
        if event is None:
            continue
        if wanted and event.get("event_type") not in wanted:
            continue
        found += 1
        if json_output:
            typer.echo(json.dumps(event, ensure_ascii=False))
        else:
            typer.echo(format_harness_trace_event(event))

    if found == 0:
        typer.echo("No harness trace events found")


@main.command("trace-audit")
def trace_audit_cmd(
    require_visual: bool = typer.Option(
        False,
        "--require-visual",
        help="Require model-visible image evidence in request_assembled events.",
    ),
    require_session_lifecycle: bool = typer.Option(
        False,
        "--require-session-lifecycle",
        help="Require native session_start/session_end traces with matching session ids.",
    ),
    require_turn_lifecycle: bool = typer.Option(
        False,
        "--require-turn-lifecycle",
        help="Require native turn_start/turn_end traces linked to request_assembled events.",
    ),
    require_model_turn: bool = typer.Option(
        False,
        "--require-model-turn",
        help="Require at least one native turn_end outcome=model_completed event.",
    ),
    require_model_response: bool = typer.Option(
        False,
        "--require-model-response",
        help="Require completed native model turns to emit compact model_response traces.",
    ),
    forbid_turn_errors: bool = typer.Option(
        False,
        "--forbid-turn-errors",
        help="Fail if any native turn_end event has outcome=error.",
    ),
    require_movement: bool = typer.Option(
        False,
        "--require-movement",
        help="Require movement policy and task-authority trace events.",
    ),
    require_navigation: bool = typer.Option(
        False,
        "--require-navigation",
        help="Require navigation task lifecycle trace events.",
    ),
    require_request_contract: bool = typer.Option(
        False,
        "--require-request-contract",
        help="Require request_assembled policy, exposed-tool, and section metadata.",
    ),
    require_request_tool_scope: bool = typer.Option(
        False,
        "--require-request-tool-scope",
        help="Require request_assembled tools_exposed to match runtime policy scope.",
    ),
    require_task_state: bool = typer.Option(
        False,
        "--require-task-state",
        help="Require pinned native task-state context in request_assembled events.",
    ),
    require_task_control: bool = typer.Option(
        False,
        "--require-task-control",
        help="Require native task-control command trace events.",
    ),
    require_task_followup: bool = typer.Option(
        False,
        "--require-task-followup",
        help="Require active-task follow-up traces and pending navigation follow-up.",
    ),
    require_auto_followup: bool = typer.Option(
        False,
        "--require-auto-followup",
        help="Require zero-tool pending task turns to queue and process auto follow-ups.",
    ),
    require_tool_status_isolated: bool = typer.Option(
        False,
        "--require-tool-status-isolated",
        help="Require typed tool-status events to stay out of model request history.",
    ),
    require_tool_status_context: bool = typer.Option(
        False,
        "--require-tool-status-context",
        help="Require tool-status traces to carry Human CLI panel context.",
    ),
    require_tool_execution: bool = typer.Option(
        False,
        "--require-tool-execution",
        help="Require MCP tool call request/start/result execution traces.",
    ),
    require_tool_call_terminal: bool = typer.Option(
        False,
        "--require-tool-call-terminal",
        help="Require every MCP tool call request to end in execution result or refusal trace.",
    ),
    forbid_raw_trace_payloads: bool = typer.Option(
        False,
        "--forbid-raw-trace-payloads",
        help="Fail if harness traces contain raw image data URLs or long base64 strings.",
    ),
    require_context_compaction: bool = typer.Option(
        False,
        "--require-context-compaction",
        help="Require dropped request context to have matching context_compacted traces.",
    ),
    require_failure_escalation: bool = typer.Option(
        False,
        "--require-failure-escalation",
        help="Require at least one structured failure_escalated trace event.",
    ),
    require_relative_move: bool = typer.Option(
        False,
        "--require-relative-move",
        help="Require relative_move result trace with measured odom and arrival evidence.",
    ),
    require_rotation: bool = typer.Option(
        False,
        "--require-rotation",
        help="Require rotate_in_place result trace with measured yaw evidence.",
    ),
    require_stop: bool = typer.Option(
        False,
        "--require-stop",
        help="Require stop_movement result trace with zero-velocity command evidence.",
    ),
    forbid_navigation_escalation: bool = typer.Option(
        False,
        "--forbid-navigation-escalation",
        help="Fail if navigation retry-limit escalation events are present.",
    ),
    require_preemption: bool = typer.Option(
        False,
        "--require-preemption",
        help="Require movement_preempted trace with stop result and duration evidence.",
    ),
    preemption_max_ms: float | None = typer.Option(
        None,
        "--preemption-max-ms",
        help="Fail if any movement_preempted trace exceeds this duration in ms.",
    ),
    expected_preempted_by: list[str] = typer.Option(
        [],
        "--expect-preempted-by",
        help=(
            "Require at least one movement_preempted trace with this preempted_by "
            "reason. Can be repeated."
        ),
    ),
    max_context_token_growth: int | None = typer.Option(
        None,
        "--max-context-token-growth",
        help=(
            "Fail if consecutive request_assembled estimated token counts grow by more "
            "than this amount."
        ),
    ),
    max_context_images: int | None = typer.Option(
        None,
        "--max-context-images",
        help="Fail if any request_assembled event includes more model-visible images.",
    ),
    max_context_image_tokens: int | None = typer.Option(
        None,
        "--max-context-image-tokens",
        help="Fail if any request_assembled event exceeds this image-token total.",
    ),
    profile: str = typer.Option(
        "",
        "--profile",
        help="Apply a named trace-audit gate bundle, e.g. go2-prelive or go2-live-motion.",
    ),
    list_profiles: bool = typer.Option(
        False,
        "--list-profiles",
        help="List named trace-audit profiles and exit.",
    ),
    lines: int = typer.Option(1000, "--lines", "-n", help="Number of log lines to scan"),
    all_lines: bool = typer.Option(False, "--all", "-a", help="Scan the full log"),
    json_output: bool = typer.Option(False, "--json", help="Raw JSON audit report"),
    run_id: str = typer.Option("", "--run", "-r", help="Specific run ID"),
) -> None:
    """Audit a DimOS run against trace-backed harness validation checks."""

    from dimos.agents.harness.live_validation import validate_live_trace_events
    from dimos.agents.harness.trace_audit_profiles import (
        available_trace_audit_profiles,
        merge_trace_audit_options,
    )
    from dimos.agents.harness.trace_log import harness_trace_events_from_lines
    from dimos.core.log_viewer import read_log, resolve_log_path

    if list_profiles:
        for item in available_trace_audit_profiles():
            typer.echo(f"{item.name}: {item.description}")
        return

    try:
        audit_options, applied_profile = merge_trace_audit_options(
            profile,
            {
                "require_visual": require_visual,
                "require_session_lifecycle": require_session_lifecycle,
                "require_turn_lifecycle": require_turn_lifecycle,
                "require_model_turn": require_model_turn,
                "require_model_response": require_model_response,
                "forbid_turn_errors": forbid_turn_errors,
                "require_movement": require_movement,
                "require_navigation": require_navigation,
                "require_request_contract": require_request_contract,
                "require_request_tool_scope": require_request_tool_scope,
                "require_task_state": require_task_state,
                "require_task_control": require_task_control,
                "require_task_followup": require_task_followup,
                "require_auto_followup": require_auto_followup,
                "require_tool_status_isolated": require_tool_status_isolated,
                "require_tool_status_context": require_tool_status_context,
                "require_tool_execution": require_tool_execution,
                "require_tool_call_terminal": require_tool_call_terminal,
                "forbid_raw_trace_payloads": forbid_raw_trace_payloads,
                "require_context_compaction": require_context_compaction,
                "require_failure_escalation": require_failure_escalation,
                "require_relative_move": require_relative_move,
                "require_rotation": require_rotation,
                "require_stop": require_stop,
                "require_preemption": require_preemption,
                "preemption_max_ms": preemption_max_ms,
                "expected_preempted_by": expected_preempted_by,
                "max_context_token_growth": max_context_token_growth,
                "max_context_images": max_context_images,
                "max_context_image_tokens": max_context_image_tokens,
                "forbid_navigation_escalation": forbid_navigation_escalation,
            },
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    path = resolve_log_path(run_id)
    if not path:
        typer.echo("No log files found", err=True)
        raise typer.Exit(1)

    count = None if all_lines else lines
    events = harness_trace_events_from_lines(read_log(path, count))
    report = validate_live_trace_events(events, **audit_options)
    if json_output:
        payload = report.model_dump()
        if applied_profile is not None:
            payload["profile"] = applied_profile.model_dump()
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if applied_profile is not None:
            typer.echo(f"Profile: {applied_profile.name} - {applied_profile.description}")
        typer.echo(report.render_text())
    if not report.ok:
        raise typer.Exit(1)


@main.command("agent-provider-status")
def agent_provider_status_cmd(
    provider: str = typer.Option(
        "",
        "--provider",
        help="Provider override; defaults to DIMOS_AGENT_PROVIDER or native model inference.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="Model override; defaults to DIMOS_AGENT_MODEL or gpt-4o.",
    ),
    base_url: str = typer.Option(
        "",
        "--base-url",
        help="OpenAI-compatible base URL; defaults to DIMOS_AGENT_BASE_URL or OPENAI_BASE_URL.",
    ),
    fixture: str = typer.Option(
        "",
        "--fixture",
        help="Fixture model path for offline native-agent smoke tests.",
    ),
    require_native: bool = typer.Option(
        False,
        "--require-native",
        help="Exit nonzero unless the native agent can build its model client.",
    ),
    require_vision_roundtrip: bool = typer.Option(
        False,
        "--require-vision-roundtrip",
        help="Exit nonzero unless the live provider visual round-trip is ready.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Raw JSON readiness report"),
) -> None:
    """Print a display-safe native agent model/provider readiness summary."""

    from pydantic import SecretStr

    from dimos.agents.harness.model_provider import (
        NativeAgentModelConfig,
        native_agent_model_readiness,
    )

    api_key = os.getenv("DIMOS_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    config = NativeAgentModelConfig(
        model=model or os.getenv("DIMOS_AGENT_MODEL") or "gpt-4o",
        model_provider=provider or os.getenv("DIMOS_AGENT_PROVIDER") or None,
        model_base_url=base_url
        or os.getenv("DIMOS_AGENT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL"),
        model_api_key=SecretStr(api_key) if api_key else None,
        model_fixture=fixture or None,
    )
    readiness = native_agent_model_readiness(config)
    exit_code = 0
    if require_native and not readiness.ready_for_native_agent:
        exit_code = 1
    if require_vision_roundtrip and not readiness.ready_for_provider_vision_roundtrip:
        exit_code = 1

    if json_output:
        typer.echo(json.dumps(readiness.model_dump(), ensure_ascii=False, indent=2))
    else:
        typer.echo(readiness.render_text())
        typer.echo("")
        typer.echo("Next checks:")
        typer.echo("  uv run dimos agent-vision-roundtrip --preflight")
        if readiness.ready_for_provider_vision_roundtrip:
            typer.echo("  uv run dimos agent-vision-roundtrip")
        else:
            typer.echo("  set DIMOS_AGENT_API_KEY/OPENAI_API_KEY before live provider round-trip")

    if exit_code:
        raise typer.Exit(exit_code)


@main.command("agent-vision-roundtrip")
def agent_vision_roundtrip_cmd(
    provider: str = typer.Option(
        "",
        "--provider",
        help="Provider to validate; defaults to DIMOS_AGENT_PROVIDER or openai-compatible.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="Model to validate; defaults to DIMOS_AGENT_MODEL or gpt-4o.",
    ),
    base_url: str = typer.Option(
        "",
        "--base-url",
        help="OpenAI-compatible base URL; defaults to DIMOS_AGENT_BASE_URL or OPENAI_BASE_URL.",
    ),
    expected_answer: str = typer.Option(
        "",
        "--expected-answer",
        help="Expected answer. Defaults to 'red' for synthetic checks and 'yes' for artifact checks.",
    ),
    question: str = typer.Option(
        "",
        "--question",
        help="Question to ask about the carried image evidence.",
    ),
    artifact_id: str = typer.Option(
        "",
        "--artifact-id",
        help="Use a persisted image artifact instead of the synthetic square image.",
    ),
    latest_image_artifact: bool = typer.Option(
        False,
        "--latest-image-artifact",
        help="Use the latest persisted image artifact from the active DimOS run.",
    ),
    artifact_dir: str = typer.Option(
        "",
        "--artifact-dir",
        help="Artifact directory override; defaults to DIMOS_AGENT_ARTIFACT_DIR or latest run.",
    ),
    preflight: bool = typer.Option(
        False,
        "--preflight",
        help="Only validate request assembly/image placement; do not call the model provider.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Raw JSON validation report"),
) -> None:
    """Run an opt-in provider vision round-trip against the configured model."""

    from dimos.agents.harness.visual_roundtrip import (
        VisualRoundTripConfig,
        VisualRoundTripImage,
        make_roundtrip_model,
        preflight_visual_roundtrip,
        run_visual_roundtrip,
    )

    resolved_provider = provider or os.getenv("DIMOS_AGENT_PROVIDER") or "openai-compatible"
    resolved_model = model or os.getenv("DIMOS_AGENT_MODEL") or "gpt-4o"
    resolved_base_url = (
        base_url or os.getenv("DIMOS_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )
    source_image: VisualRoundTripImage | None = None
    resolved_expected_answer = expected_answer or "red"
    resolved_question = question
    run_entry = _get_mcp_run_entry()

    if artifact_id or latest_image_artifact:
        readback = _read_cli_image_artifact(
            artifact_id=artifact_id or None,
            latest_image_artifact=latest_image_artifact,
            artifact_dir=artifact_dir or None,
            run_entry=run_entry,
        )
        if readback is None:
            raise typer.Exit(1)
        source_image = VisualRoundTripImage(
            block=readback.block,
            source="artifact",
            artifact_id=readback.ref.artifact_id,
            description="persisted DimOS visual artifact",
        )
        resolved_expected_answer = expected_answer or "yes"
        resolved_question = question or (
            "Does the earlier visual evidence contain a visible image? Answer exactly yes or no."
        )

    config = VisualRoundTripConfig(
        target_provider=resolved_provider,
        target_model=resolved_model,
        target_base_url=resolved_base_url or None,
        expected_answer=resolved_expected_answer,
        square_color=resolved_expected_answer,
        source_image=source_image,
        question=resolved_question or VisualRoundTripConfig().question,
    )
    api_key = os.getenv("DIMOS_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not preflight and not api_key:
        typer.echo(
            "Error: set DIMOS_AGENT_API_KEY or OPENAI_API_KEY before running the live "
            "provider visual round-trip. Use --preflight to validate request assembly "
            "without a provider call.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        if preflight:
            report = preflight_visual_roundtrip(config)
        else:
            live_model = make_roundtrip_model(
                provider=resolved_provider,
                model=resolved_model,
                api_key=api_key,
                base_url=resolved_base_url,
            )
            report = run_visual_roundtrip(live_model, config)
    except Exception as e:
        typer.echo(f"Error: provider visual round-trip failed to run: {e}", err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
    else:
        typer.echo(report.render_text())
    if not report.ok:
        raise typer.Exit(1)


@main.command("go2-harness-gates")
def go2_harness_gates_cmd(
    profile: str = typer.Option(
        "supervised-prelive",
        "--profile",
        help=(
            "Gate profile to render, e.g. readonly-prelive, supervised-prelive, "
            "navigation-live-audit, or live-motion-audit."
        ),
    ),
    list_profiles: bool = typer.Option(
        False,
        "--list-profiles",
        help="List Go2 harness gate profiles and exit.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the gate profile as JSON.",
    ),
) -> None:
    """Render repeatable Go2 harness gate commands without starting motion."""

    from dimos.agents.harness.go2_harness_gate import (
        available_go2_harness_gate_profiles,
        go2_harness_gate_profile,
    )

    if list_profiles:
        for item in available_go2_harness_gate_profiles():
            typer.echo(f"{item.name}: {item.description}")
        return

    try:
        gate = go2_harness_gate_profile(profile)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    if json_output:
        typer.echo(json.dumps(gate.model_dump(), indent=2))
    else:
        typer.echo(gate.render_text())


mcp_app = typer.Typer(help="Interact with the running MCP server")
main.add_typer(mcp_app, name="mcp")


def _get_mcp_run_entry() -> Any | None:
    """Return the latest live DimOS run entry, if one is registered."""

    return get_most_recent(alive_only=True)


def _get_adapter(entry: Any | None = None, mcp_url: str | None = None) -> McpAdapter:
    """Get an McpAdapter from the latest RunEntry or default URL."""
    if mcp_url:
        return McpAdapter(url=mcp_url)
    if entry is None:
        entry = _get_mcp_run_entry()
    return McpAdapter.from_run_entry(entry)


def _artifact_dir_for_run_entry(entry: Any | None) -> Path | None:
    config_overrides = getattr(entry, "config_overrides", {}) or {}
    configured = (
        config_overrides.get("agent_artifact_dir") if isinstance(config_overrides, dict) else None
    )
    if configured:
        return Path(str(configured)).expanduser()
    env_configured = os.environ.get("DIMOS_AGENT_ARTIFACT_DIR") or global_config.agent_artifact_dir
    if env_configured:
        return Path(str(env_configured)).expanduser()
    if entry is None:
        return None
    log_dir = getattr(entry, "log_dir", None)
    if log_dir:
        return Path(str(log_dir)).expanduser() / "artifacts"
    return None


def _latest_image_artifact_ref_for_run(entry: Any | None) -> ArtifactRef | None:
    artifact_dir = _artifact_dir_for_run_entry(entry)
    if artifact_dir is None:
        return None
    return ArtifactLedger(artifact_dir=artifact_dir).latest_image_ref()


def _read_cli_image_artifact(
    *,
    artifact_id: str | None,
    latest_image_artifact: bool,
    artifact_dir: str | None,
    run_entry: Any | None,
) -> ImageArtifactReadback | None:
    resolved_dir = (
        Path(artifact_dir).expanduser() if artifact_dir else _artifact_dir_for_run_entry(run_entry)
    )
    if resolved_dir is None:
        typer.echo(
            "Error: no artifact directory is available. Start DimOS, set "
            "DIMOS_AGENT_ARTIFACT_DIR, or pass --artifact-dir.",
            err=True,
        )
        return None

    ledger = ArtifactLedger(artifact_dir=resolved_dir)
    ref: ArtifactRef | None
    if artifact_id:
        if not is_valid_artifact_id(artifact_id):
            typer.echo(
                "Error: invalid --artifact-id. Expected an id like artifact_000001.",
                err=True,
            )
            return None
        ref = ledger.get_ref(artifact_id)
    elif latest_image_artifact:
        ref = ledger.latest_image_ref()
    else:
        return None

    if ref is None:
        typer.echo(
            f"Error: no image artifact found in {resolved_dir}. Run observe first "
            "or pass --artifact-id explicitly.",
            err=True,
        )
        return None

    readback = ledger.read_image(ref.artifact_id)
    if readback is None:
        typer.echo(
            f"Error: artifact {ref.artifact_id} in {resolved_dir} is not a readable image.",
            err=True,
        )
        return None
    return readback


def _save_first_image_content_block(content: Any, path: Path) -> dict[str, Any] | None:
    if not isinstance(content, list):
        return None
    for index, block in enumerate(content):
        if not isinstance(block, dict) or not is_image_block(block):
            continue
        payload = image_block_payload_bytes(block)
        if payload is None:
            continue
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": str(path),
            "bytes": len(payload),
            "mime_type": image_block_mime_type(block),
            "content_index": index,
        }
    return None


def _mcp_call_summary(
    *,
    tool_name: str,
    content: Any,
    saved_image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocks = content if isinstance(content, list) else []
    return {
        "tool_name": tool_name,
        "content_blocks": len(blocks),
        "image_blocks": sum(
            1 for block in blocks if isinstance(block, dict) and is_image_block(block)
        ),
        "text": content_to_display_text(content),
        "saved_image": saved_image,
    }


@mcp_app.command("list-tools")
def mcp_list_tools(
    mcp_url: str = typer.Option(
        "",
        "--mcp-url",
        envvar="DIMOS_MCP_URL",
        help="MCP URL to query. Defaults to DIMOS_MCP_URL, then the active DimOS run.",
    ),
) -> None:
    """List available MCP tools (skills)."""
    adapter = _get_adapter(mcp_url=mcp_url or None)
    try:
        tools = adapter.list_tools()
    except requests.ConnectionError:
        typer.echo(f"Error: no MCP server reachable at {adapter.url}", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(tools, indent=2))


class _KeyValueType(click.ParamType):
    """Parse KEY=VALUE arguments, auto-converting JSON values."""

    name = "KEY=VALUE"

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> tuple[str, Any]:
        if "=" not in value:
            self.fail(f"expected KEY=VALUE, got: {value}", param, ctx)
        key, val = value.split("=", 1)
        try:
            return (key, json.loads(val))
        except (json.JSONDecodeError, ValueError):
            return (key, val)


@mcp_app.command("call")
def mcp_call_tool(
    tool_name: str = typer.Argument(..., help="Tool name to call"),
    args: list[str] = typer.Option(
        [], "--arg", "-a", click_type=_KeyValueType(), help="Arguments as key=value"
    ),
    json_args: str = typer.Option("", "--json-args", "-j", help="Arguments as JSON string"),
    save_image: Path | None = typer.Option(
        None,
        "--save-image",
        help="Save the first image content block returned by the tool to this path.",
    ),
    json_summary: bool = typer.Option(
        False,
        "--json-summary",
        help="Print a compact JSON summary instead of display text.",
    ),
    mcp_url: str = typer.Option(
        "",
        "--mcp-url",
        envvar="DIMOS_MCP_URL",
        help="MCP URL to call. Defaults to DIMOS_MCP_URL, then the active DimOS run.",
    ),
) -> None:
    """Call an MCP tool by name."""
    arguments: dict[str, Any] = {}
    if json_args:
        try:
            arguments = json.loads(json_args)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: invalid JSON in --json-args: {e}", err=True)
            raise typer.Exit(1)
    else:
        # _KeyValueType.convert() returns (key, val) tuples at runtime
        arguments = dict(args)  # type: ignore[arg-type]

    adapter = _get_adapter(mcp_url=mcp_url or None)
    try:
        result = adapter.call_tool(tool_name, arguments)
    except requests.ConnectionError:
        typer.echo(f"Error: no MCP server reachable at {adapter.url}", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    content = result.get("content", [])
    if not content:
        typer.echo("(no output)")
        return

    saved_image = None
    if save_image is not None:
        saved_image = _save_first_image_content_block(content, save_image)
        if saved_image is None:
            typer.echo("Error: tool result did not include a readable image block.", err=True)
            raise typer.Exit(1)

    if json_summary:
        typer.echo(
            json.dumps(
                _mcp_call_summary(
                    tool_name=tool_name,
                    content=content,
                    saved_image=saved_image,
                ),
                indent=2,
            )
        )
        return

    typer.echo(content_to_display_text(content))
    if saved_image is not None:
        typer.echo(f"Saved image to {saved_image['path']}")


@mcp_app.command("validate-harness")
def mcp_validate_harness(
    expected_tool_scope: str = typer.Option(
        "",
        "--expected-tool-scope",
        help="Expected runtime tool scope before exposing MCP to an external harness.",
    ),
    profile: str = typer.Option(
        "",
        "--profile",
        help=(
            "Apply a named external harness validation profile, e.g. readonly, "
            "readonly-visual, or vision-lidar-supervised."
        ),
    ),
    list_profiles: bool = typer.Option(
        False,
        "--list-profiles",
        help="List named external harness validation profiles and exit.",
    ),
    artifact_id: str = typer.Option(
        "",
        "--artifact-id",
        help="Optional image artifact id to test with read_image.",
    ),
    latest_image_artifact: bool = typer.Option(
        False,
        "--latest-image-artifact",
        help="Use the latest persisted image artifact from the active DimOS run.",
    ),
    require_visual_readback: bool = typer.Option(
        False,
        "--require-visual-readback",
        help="Fail unless read_image returns metadata plus image content for --artifact-id.",
    ),
    require_movement_authority_refusal: bool = typer.Option(
        False,
        "--require-movement-authority-refusal",
        help=("Fail unless a tokenless external movement confirmation is refused before dispatch."),
    ),
    require_hidden_movement_refusal: bool = typer.Option(
        False,
        "--require-hidden-movement-refusal",
        help="Fail unless a hidden movement tool call is refused at MCP dispatch.",
    ),
    require_idle_runtime: bool = typer.Option(
        False,
        "--require-idle-runtime",
        help="Fail unless movement authority and capability holders are empty.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the validation report as JSON.",
    ),
) -> None:
    """Validate the MCP surface before attaching an external harness client."""
    from dimos.agents.mcp.external_harness_profiles import (
        available_external_harness_profiles,
        merge_external_harness_validation_options,
    )

    if list_profiles:
        for item in available_external_harness_profiles():
            typer.echo(f"{item.name}: {item.description}")
        return

    try:
        validation_options, applied_profile = merge_external_harness_validation_options(
            profile,
            {
                "expected_tool_scope": expected_tool_scope or None,
                "require_visual_readback": require_visual_readback,
                "require_movement_authority_refusal": require_movement_authority_refusal,
                "require_hidden_movement_refusal": require_hidden_movement_refusal,
                "require_idle_runtime": require_idle_runtime,
            },
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    run_entry = _get_mcp_run_entry()
    resolved_artifact_id = artifact_id or None
    if latest_image_artifact and resolved_artifact_id is None:
        latest_ref = _latest_image_artifact_ref_for_run(run_entry)
        if latest_ref is None:
            artifact_dir = _artifact_dir_for_run_entry(run_entry)
            location = str(artifact_dir) if artifact_dir is not None else "the latest run"
            typer.echo(
                "Error: --latest-image-artifact requested, but no persisted image "
                f"artifact was found for {location}. Run observe first or pass "
                "--artifact-id explicitly.",
                err=True,
            )
            raise typer.Exit(1)
        resolved_artifact_id = latest_ref.artifact_id

    try:
        report = validate_external_harness_surface(
            _get_adapter(run_entry),
            expected_tool_scope=validation_options["expected_tool_scope"],
            artifact_id=resolved_artifact_id,
            require_visual_readback=validation_options["require_visual_readback"],
            require_movement_authority_refusal=validation_options[
                "require_movement_authority_refusal"
            ],
            require_hidden_movement_refusal=validation_options["require_hidden_movement_refusal"],
            require_idle_runtime=validation_options["require_idle_runtime"],
        )
    except requests.ConnectionError:
        typer.echo("Error: no running MCP server (is DimOS running?)", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if json_output:
        payload = report.model_dump()
        if applied_profile is not None:
            payload["profile"] = applied_profile.model_dump()
        typer.echo(json.dumps(payload, indent=2))
    else:
        if applied_profile is not None:
            typer.echo(f"Profile: {applied_profile.name} - {applied_profile.description}")
        typer.echo(report.render_text())
    if not report.ok:
        raise typer.Exit(1)


@mcp_app.command("prepare-codex-run")
def mcp_prepare_codex_run(
    task: str = typer.Argument(..., help="Operator task for the isolated Codex run."),
    run_root: Path = typer.Option(
        ...,
        "--run-root",
        help="Directory where workspace, run-local CODEX_HOME, prompt, and audit files are written.",
    ),
    mcp_url: str = typer.Option(
        "",
        "--mcp-url",
        help="DimOS MCP URL. Defaults to the active run's MCP URL when available.",
    ),
    server_name: str = typer.Option(
        "dimos",
        "--server-name",
        help="MCP server name to expose in the run-local Codex config.",
    ),
    source_codex_home: Path | None = typer.Option(
        None,
        "--source-codex-home",
        help="Source Codex home for minimal provider/auth copy; defaults to ~/.codex.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="Optional model override written into the run-local Codex config.",
    ),
    model_reasoning_effort: str = typer.Option(
        "medium",
        "--model-reasoning-effort",
        help=(
            "Run-local Codex reasoning effort. Defaults to medium so live runs do not "
            "silently inherit a global high/xhigh profile."
        ),
    ),
    service_tier: str = typer.Option(
        "",
        "--service-tier",
        help="Optional Codex/OpenAI service tier config value, written only when provided.",
    ),
    codex_exec_timeout_s: float = typer.Option(
        1800.0,
        "--codex-exec-timeout-s",
        help="Operator-facing timeout budget for the main codex exec run, recorded in launch.json.",
    ),
    sandbox: str = typer.Option(
        "workspace-write",
        "--sandbox",
        help="Codex shell sandbox mode for the prepared command.",
    ),
    no_copy_auth: bool = typer.Option(
        False,
        "--no-copy-auth",
        help="Do not copy auth.json / installation_id into the run-local CODEX_HOME.",
    ),
    no_goal_tools: bool = typer.Option(
        False,
        "--no-goal-tools",
        help="Do not instruct Codex to call create_goal/get_goal/update_goal.",
    ),
    dangerously_bypass_sandbox: bool = typer.Option(
        False,
        "--dangerously-bypass-approvals-and-sandbox",
        help="Prepare a command that disables Codex approvals and sandboxing.",
    ),
    mode: str = typer.Option(
        "readonly",
        "--mode",
        help="Run mode: readonly, authority-proxy, or admin-override-debug.",
    ),
    prompt_mode: str = typer.Option(
        "default",
        "--prompt-mode",
        help="Prompt policy variant: default or long_instruction.",
    ),
    tool_profile: str = typer.Option(
        "",
        "--tool-profile",
        help=(
            "Go2 harness tool profile/audit label. Defaults to readonly_inspection "
            "for readonly runs and vision_odom_local_motion for movement runs."
        ),
    ),
    proxy_url: str = typer.Option(
        "",
        "--proxy-url",
        help="Codex-facing MCP proxy URL for authority-proxy mode.",
    ),
    authority_scope: str = typer.Option(
        "operator-approved live Go2 task",
        "--authority-scope",
        help="Operator-approved movement scope for authority-proxy mode.",
    ),
    authority_area: str = typer.Option(
        "operator-approved local test area",
        "--authority-area",
        help="Operator-approved physical area for authority-proxy mode.",
    ),
    authority_duration_s: float = typer.Option(
        3600.0,
        "--authority-duration-s",
        help="Movement authority envelope duration in seconds.",
    ),
    authority_tool: list[str] = typer.Option(
        [],
        "--authority-tool",
        help="Movement tool allowed by the authority envelope; repeat for multiple tools.",
    ),
    max_movement_calls: int | None = typer.Option(
        None,
        "--max-movement-calls",
        help="Optional maximum number of movement calls signed by the proxy envelope.",
    ),
    run_mcp_smoke: bool = typer.Option(
        False,
        "--run-mcp-smoke",
        help="Run the MCP-only nested Codex smoke gate and fail closed on missing tools.",
    ),
    skip_mcp_smoke: bool = typer.Option(
        False,
        "--skip-mcp-smoke",
        help="Skip the default MCP-only smoke gate for authority-proxy preparation.",
    ),
    defer_mcp_smoke: bool = typer.Option(
        False,
        "--defer-mcp-smoke",
        help=(
            "Record the default MCP smoke gate in launch.json but skip running it during "
            "prepare-codex-run; the prepared runner will execute it after starting the proxy."
        ),
    ),
    mcp_smoke_timeout_s: float | None = typer.Option(
        None,
        "--mcp-smoke-timeout-s",
        help="Timeout for the nested Codex MCP smoke gate.",
    ),
    prior_run_root: Path | None = typer.Option(
        None,
        "--prior-run-root",
        help="Prior Go2 run root used to seed graph memory and SLAM relocalization.",
    ),
    initial_slam_status_path: Path | None = typer.Option(
        None,
        "--initial-slam-status-path",
        help=(
            "Optional saved Go2 SLAM status JSON to seed long-instruction "
            "known-map relocation context at navigation startup."
        ),
    ),
    initial_relocalization_bundle: Path | None = typer.Option(
        None,
        "--initial-relocalization-bundle",
        help=(
            "Optional Go2 relocalization bundle directory or manifest.json to seed "
            "known-map context explicitly; does not auto-publish localization."
        ),
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print prepared run details as JSON.",
    ),
) -> None:
    """Prepare a run-local Codex exec harness for the active DimOS MCP server."""
    from dimos.agents.mcp.external_harness_codex import (
        DEFAULT_MCP_SMOKE_TIMEOUT_S,
        prepare_codex_harness_run,
        run_codex_mcp_smoke_gate,
    )

    if skip_mcp_smoke and defer_mcp_smoke:
        typer.echo("Error: --skip-mcp-smoke and --defer-mcp-smoke are mutually exclusive", err=True)
        raise typer.Exit(2)

    resolved_mcp_url = mcp_url
    if not resolved_mcp_url:
        try:
            resolved_mcp_url = _get_mcp_run_entry().mcp_url
        except SystemExit:
            raise
        except Exception as exc:
            typer.echo(f"Error: could not resolve active MCP URL: {exc}", err=True)
            raise typer.Exit(1) from exc

    try:
        effective_mcp_smoke_timeout_s = (
            DEFAULT_MCP_SMOKE_TIMEOUT_S
            if mcp_smoke_timeout_s is None
            else mcp_smoke_timeout_s
        )
        prepared = prepare_codex_harness_run(
            run_root=run_root,
            task=task,
            mcp_url=resolved_mcp_url,
            source_codex_home=source_codex_home,
            server_name=server_name,
            model=model or None,
            model_reasoning_effort=model_reasoning_effort or None,
            service_tier=service_tier or None,
            codex_exec_timeout_s=codex_exec_timeout_s,
            mcp_smoke_timeout_s=effective_mcp_smoke_timeout_s,
            sandbox=sandbox,
            copy_auth=not no_copy_auth,
            use_goal_tools=not no_goal_tools,
            dangerously_bypass_sandbox=dangerously_bypass_sandbox,
            mode=mode,  # type: ignore[arg-type]
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            tool_profile=tool_profile,
            proxy_url=proxy_url or None,
            authority_scope=authority_scope,
            authority_area=authority_area,
            authority_duration_s=authority_duration_s,
            authority_allowed_movement_tools=tuple(authority_tool) if authority_tool else (),
            max_movement_calls=max_movement_calls,
            mcp_smoke_required=False if skip_mcp_smoke else None,
            prior_run_root=prior_run_root,
            initial_slam_status_path=initial_slam_status_path,
            initial_relocalization_bundle=initial_relocalization_bundle,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc

    smoke_result = None
    should_run_smoke = run_mcp_smoke or (
        prepared.mcp_smoke_required and not skip_mcp_smoke and not defer_mcp_smoke
    )
    if should_run_smoke:
        smoke_result = run_codex_mcp_smoke_gate(
            prepared,
            timeout_s=effective_mcp_smoke_timeout_s,
        )
        if not smoke_result.ok:
            typer.echo(
                f"MCP-only nested Codex smoke failed before live movement: {smoke_result.reason}",
                err=True,
            )
            typer.echo(f"Smoke log: {smoke_result.log_path}", err=True)
            typer.echo(f"Smoke stderr: {smoke_result.stderr_path}", err=True)
            typer.echo(f"Smoke result: {smoke_result.result_path}", err=True)
            raise typer.Exit(1)

    payload = prepared.model_dump()
    if smoke_result is not None:
        payload["mcp_smoke_result"] = smoke_result.model_dump()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    typer.echo("Prepared isolated Codex harness run.")
    typer.echo(f"Run root: {prepared.run_root}")
    typer.echo(f"Workspace: {prepared.workspace}")
    typer.echo(f"CODEX_HOME: {prepared.codex_home}")
    typer.echo(f"Config: {prepared.config_path}")
    typer.echo(f"Prompt: {prepared.prompt_path}")
    typer.echo(f"Audit: {prepared.audit_path}")
    typer.echo(f"Shareable summary: {prepared.shareable_summary_path}")
    typer.echo(f"Mode: {prepared.mode}")
    profile_name = (
        prepared.tool_profile.get("name") if isinstance(prepared.tool_profile, dict) else None
    )
    if profile_name:
        typer.echo(f"Tool profile: {profile_name}")
    typer.echo(f"MCP URL: {prepared.mcp_url}")
    typer.echo(f"Model: {prepared.model or '(config/default)'}")
    typer.echo(f"Reasoning effort: {prepared.model_reasoning_effort or '(codex default)'}")
    typer.echo(f"Codex exec timeout: {prepared.codex_exec_timeout_s}s")
    if prepared.service_tier:
        typer.echo(f"Service tier: {prepared.service_tier}")
    typer.echo(f"MCP smoke prompt: {prepared.mcp_smoke_prompt_path}")
    typer.echo("MCP-only smoke command:")
    typer.echo(prepared.mcp_smoke_shell_command)
    if smoke_result is not None:
        typer.echo(f"MCP-only smoke: PASS ({smoke_result.reason})")
    if prepared.proxy_shell_command is not None:
        typer.echo("Proxy command:")
        typer.echo(prepared.proxy_shell_command)
    typer.echo("Command:")
    typer.echo(prepared.shell_command)


@mcp_app.command("authority-proxy")
def mcp_authority_proxy(
    task: str = typer.Option(..., "--task", help="Operator-approved task goal."),
    upstream_mcp_url: str = typer.Option(
        ...,
        "--upstream-mcp-url",
        help="Real DimOS MCP URL that remains the final movement authority.",
    ),
    proxy_host: str = typer.Option(
        "127.0.0.1",
        "--proxy-host",
        help="Host for the local authority proxy.",
    ),
    proxy_port: int = typer.Option(
        9992,
        "--proxy-port",
        help="Port for the local authority proxy.",
    ),
    scope: str = typer.Option(
        "operator-approved live Go2 task",
        "--scope",
        help="Operator-approved movement scope.",
    ),
    area: str = typer.Option(
        "operator-approved local test area",
        "--area",
        help="Operator-approved physical area.",
    ),
    duration_s: float = typer.Option(
        3600.0,
        "--duration-s",
        help="Movement authority envelope duration in seconds.",
    ),
    authority_tool: list[str] = typer.Option(
        [],
        "--authority-tool",
        help="Movement tool allowed by the authority envelope; repeat for multiple tools.",
    ),
    tool_profile: str = typer.Option(
        "",
        "--tool-profile",
        help=(
            "Go2 harness tool profile used to choose default authority tools when "
            "--authority-tool is omitted."
        ),
    ),
    max_movement_calls: int | None = typer.Option(
        None,
        "--max-movement-calls",
        help="Optional maximum number of movement calls signed by the proxy envelope.",
    ),
    graph_memory_path: Path | None = typer.Option(
        None,
        "--graph-memory-path",
        help="Optional run-local JSON store path enabling Go2 graph-memory MCP tools.",
    ),
    slam_status_path: Path | None = typer.Option(
        None,
        "--slam-status-path",
        help="Optional run-local JSON path where cached Go2 SLAM status is saved.",
    ),
    initial_slam_status_path: Path | None = typer.Option(
        None,
        "--initial-slam-status-path",
        help="Optional saved Go2 SLAM status JSON loaded before the first movement/map call.",
    ),
    initial_relocalization_bundle: Path | None = typer.Option(
        None,
        "--initial-relocalization-bundle",
        help=(
            "Optional Go2 relocalization bundle directory or manifest.json loaded as "
            "read-only startup context."
        ),
    ),
) -> None:
    """Run a local MCP proxy that signs movement calls inside a bounded envelope."""
    import uvicorn

    from dimos.agents.harness.go2_harness_profiles import go2_harness_tool_profile
    from dimos.agents.mcp.external_harness_authority_proxy import (
        MovementAuthorityEnvelope,
        create_authority_proxy_app,
    )

    resolved_profile = go2_harness_tool_profile(tool_profile or "vision_odom_local_motion")
    allowed_tools = tuple(authority_tool) or resolved_profile.default_authority_movement_tools
    if not allowed_tools:
        typer.echo(
            f"Error: Go2 harness tool profile '{resolved_profile.name}' grants no movement tools; "
            "use readonly mode or pass --authority-tool.",
            err=True,
        )
        raise typer.Exit(2)

    envelope = MovementAuthorityEnvelope.create(
        task_goal=task,
        scope=scope,
        area=area,
        duration_s=duration_s,
        tool_profile=resolved_profile.name,
        allowed_movement_tools=allowed_tools,
        max_movement_calls=max_movement_calls,
    )
    proxy_url = f"http://{proxy_host}:{proxy_port}/mcp"
    typer.echo("Starting DimOS external-harness movement authority proxy.")
    typer.echo(f"Proxy URL: {proxy_url}")
    typer.echo(f"Upstream MCP URL: {upstream_mcp_url}")
    typer.echo(f"Tool profile: {resolved_profile.name}")
    typer.echo(f"Envelope: {json.dumps(envelope.model_dump(), ensure_ascii=False)}")
    app = create_authority_proxy_app(
        upstream_url=upstream_mcp_url,
        envelope=envelope,
        proxy_url=proxy_url,
        graph_memory_path=graph_memory_path,
        slam_status_path=slam_status_path,
        initial_slam_status_path=initial_slam_status_path,
        initial_relocalization_bundle=initial_relocalization_bundle,
    )
    uvicorn.run(app, host=proxy_host, port=proxy_port)


@mcp_app.command("status")
def mcp_status(
    mcp_url: str = typer.Option(
        "",
        "--mcp-url",
        envvar="DIMOS_MCP_URL",
        help="MCP URL to query. Defaults to DIMOS_MCP_URL, then the active DimOS run.",
    ),
) -> None:
    """Show MCP server status (modules, skills)."""
    adapter = _get_adapter(mcp_url=mcp_url or None)
    try:
        data = adapter.call_tool_text("server_status")
    except requests.ConnectionError:
        typer.echo(f"Error: no MCP server reachable at {adapter.url}", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    # server_status returns JSON string -- pretty-print it
    try:
        typer.echo(json.dumps(json.loads(data), indent=2))
    except (json.JSONDecodeError, ValueError):
        typer.echo(data)


@mcp_app.command("modules")
def mcp_modules() -> None:
    """List deployed modules and their skills."""
    try:
        data = _get_adapter().call_tool_text("list_modules")
    except requests.ConnectionError:
        typer.echo("Error: no running MCP server (is DimOS running?)", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    try:
        typer.echo(json.dumps(json.loads(data), indent=2))
    except (json.JSONDecodeError, ValueError):
        typer.echo(data)


@main.command("agent-send")
def agent_send_cmd(
    message: str = typer.Argument(..., help="Message to send to the running agent"),
) -> None:
    """Send a message to the running DimOS agent via MCP."""
    try:
        text = _get_adapter().call_tool_text("agent_send", {"message": message})
    except requests.ConnectionError:
        typer.echo("Error: no running MCP server (is DimOS running?)", err=True)
        raise typer.Exit(1)
    except McpError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(text)


@main.command()
def restart(
    force: bool = typer.Option(False, "--force", "-f", help="Force kill before restarting"),
) -> None:
    """Restart the running DimOS instance with the same arguments."""
    entry = get_most_recent(alive_only=True)
    if not entry:
        typer.echo("No running DimOS instance to restart", err=True)
        raise typer.Exit(1)

    if not entry.original_argv:
        typer.echo("Cannot restart: run entry missing original command", err=True)
        raise typer.Exit(1)

    # Save argv and pid before stopping (stop removes the entry)
    argv = entry.original_argv
    old_pid = entry.pid

    typer.echo(f"Restarting {entry.run_id} ({entry.blueprint})...")
    msg, _ok = stop_entry(entry, force=force)
    typer.echo(f"  {msg}")

    # Wait for the old process to fully exit so ports are released.
    for _ in range(20):  # up to 2s
        if not is_pid_alive(old_pid):
            break
        time.sleep(0.1)

    typer.echo(f"  Running: {' '.join(argv)}")
    try:
        os.execvp(argv[0], argv)
    except OSError as exc:
        typer.echo(f"Error: failed to restart — {exc}", err=True)
        raise typer.Exit(1)


@main.command()
def show_config(ctx: typer.Context) -> None:
    """Show current config settings and their values."""

    cli_config_overrides: dict[str, Any] = ctx.obj
    global_config.update(**cli_config_overrides)

    for field_name, value in global_config.model_dump().items():
        typer.echo(f"{field_name}: {value}")


@main.command(name="list")
def list_blueprints() -> None:
    """List all available blueprints."""
    from dimos.robot.all_blueprints import all_blueprints

    blueprints = [name for name in all_blueprints.keys() if not name.startswith("demo-")]
    for blueprint_name in sorted(blueprints):
        typer.echo(blueprint_name)


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lcmspy(ctx: typer.Context) -> None:
    """LCM spy tool for monitoring LCM messages."""
    from dimos.utils.cli.lcmspy.run_lcmspy import main as lcmspy_main

    sys.argv = ["lcmspy", *ctx.args]
    lcmspy_main()


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def agentspy(ctx: typer.Context) -> None:
    """Agent spy tool for monitoring agents."""
    from dimos.utils.cli.agentspy.agentspy import main as agentspy_main

    sys.argv = ["agentspy", *ctx.args]
    agentspy_main()


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def humancli(ctx: typer.Context) -> None:
    """Interface interacting with agents."""
    from dimos.utils.cli.human.humanclianim import main as humancli_main

    sys.argv = ["humancli", *ctx.args]
    humancli_main()


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def top(ctx: typer.Context) -> None:
    """Live resource monitor TUI."""
    from dimos.utils.cli.dtop import main as dtop_main

    sys.argv = ["dtop", *ctx.args]
    dtop_main()


topic_app = typer.Typer(help="Topic commands for pub/sub")
main.add_typer(topic_app, name="topic")


@topic_app.command()
def echo(
    topic: str = typer.Argument(..., help="Topic name to listen on (e.g., /goal_request)"),
    type_name: str | None = typer.Argument(
        None,
        help="Optional message type (e.g., PoseStamped). If omitted, infer from '/topic#pkg.Msg'.",
    ),
) -> None:
    from dimos.robot.cli.topic import topic_echo

    topic_echo(topic, type_name)


@topic_app.command()
def send(
    topic: str = typer.Argument(..., help="Topic name to send to (e.g., /goal_request)"),
    message_expr: str = typer.Argument(..., help="Python expression for the message"),
) -> None:
    from dimos.robot.cli.topic import topic_send

    topic_send(topic, message_expr)


map_app = typer.Typer(help="Voxel-map tools over recorded sqlite datasets")
main.add_typer(map_app, name="map")
map_app.command("global")(_map_main)
map_app.command("summary")(_map_summary_main)
map_app.command("rename")(_map_rename_main)
map_app.command("pose-fill")(_map_pose_fill_main)
map_app.command("replay")(_map_replay_main)
map_app.command("replay-marker")(_map_replay_marker_main)

from dimos.memory2.cli.app import mem_app

main.add_typer(mem_app, name="mem")


@main.command()
def cameracalibrate(
    source: str = typer.Option(..., "--source", help="Frame source: webcam, folder, or topic"),
    device_index: int = typer.Option(0, "--device-index", help="Webcam device index"),
    images: Path | None = typer.Option(
        None, "--images", help="Directory of calibration images for --source folder"
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        help=(
            "Pubsub URI for --source topic (proto:channel), "
            "e.g. 'jpeg_lcm:/color_image' or 'pshm:color_image'."
        ),
    ),
    topic_timeout_sec: float = typer.Option(
        60.0,
        "--topic-timeout-sec",
        help="Abort --source topic if no frames arrive within this many seconds.",
    ),
    cols: int = typer.Option(..., "--cols", help="Inner chessboard corner columns"),
    rows: int = typer.Option(..., "--rows", help="Inner chessboard corner rows"),
    square_size_m: float = typer.Option(
        ..., "--square-size-m", help="Chessboard square size in meters"
    ),
    out: Path | None = typer.Option(None, "--out", help="Optional ROS CameraInfo YAML output path"),
    preview_out: Path | None = typer.Argument(
        None, help="Optional preview PNG output path. Requires --out."
    ),
    camera_name: str = typer.Option("webcam", "--camera-name", help="Camera name in YAML"),
    target_count: int = typer.Option(20, "--target-count", help="Accepted webcam frame count"),
    no_display: bool = typer.Option(False, "--no-display", help="Disable OpenCV preview windows"),
    distortion_model: str = typer.Option(
        "plumb_bob",
        "--distortion-model",
        help=(
            "Lens model: 'plumb_bob' (5 coeffs, near-pinhole) or 'fisheye' "
            "(4 coeffs, wide-angle / fisheye; written as ROS 'equidistant')."
        ),
    ),
) -> None:
    """Calibrate camera intrinsics and write ROS CameraInfo YAML."""
    from dimos.utils.cli.cameracalibrate.cameracalibrate import run_calibration

    if preview_out is not None and out is None:
        raise typer.BadParameter("preview output requires --out")

    try:
        result = run_calibration(
            source=source,
            device_index=device_index,
            images=images,
            topic=topic,
            topic_timeout_sec=topic_timeout_sec,
            cols=cols,
            rows=rows,
            square_size_m=square_size_m,
            out=out,
            preview_out=preview_out,
            camera_name=camera_name,
            target_count=target_count,
            no_display=no_display,
            distortion_model=distortion_model,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"RMS: {float(result['rms']):.6f} px ({int(result['n_used'])} frame(s) used)")
    typer.echo(
        f"Detected pattern: {tuple(result.get('pattern_size', (cols, rows)))} "
        f"({result.get('pattern_label', 'requested inner corners')})"
    )
    if out is not None:
        typer.echo(f"Wrote camera info YAML to {out}")
    if preview_out is not None:
        typer.echo(f"Wrote preview overlay PNG to {preview_out}")


@main.command()
def apriltag(
    out: Path = typer.Option(Path("apriltags.pdf"), "--out", "-o", help="Output PDF path"),
    ids: str = typer.Option("0-11", "--ids", help="ID spec, e.g. '0-49' or '0,1,5,10-20'"),
    size_mm: float = typer.Option(
        50.0, "--size-mm", "-s", help="Tag black-border edge size in mm (typical: 50 or 100)"
    ),
    page_size: str = typer.Option(
        "a4", "--page-size", "-p", help="Page size: a0..a8 (ISO A series) or letter"
    ),
    pack: bool = typer.Option(
        True, "--pack/--no-pack", help="Pack as many tags per page as fit (vs one per page)"
    ),
    family: str = typer.Option(
        "tag36h11",
        "--family",
        help=(
            "Tag family: AprilTag (tag36h11, tag25h9, tag16h5) or "
            "ArUco (aruco_original, aruco_mip_36h12, aruco_{4x4,5x5,6x6,7x7}_{50,100,250,1000})"
        ),
    ),
) -> None:
    """Generate a printable AprilTag/ArUco PDF with calibration ruler."""
    from dimos.utils.cli.apriltag import generate_pdf, parse_id_spec

    id_list = parse_id_spec(ids)
    path = generate_pdf(
        id_list, out, family=family, size_mm=size_mm, page_size=page_size, pack=pack
    )
    typer.echo(f"Wrote {len(id_list)} tag(s) to {path}")


def _default_rerun_record_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"rerun-record-{timestamp}.rrd"


def _parse_rerun_record_max_hz(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise click.BadParameter("--max-hz must use ENTITY=HZ, e.g. world/global_map=0.5")
        entity, hz_text = value.split("=", 1)
        entity = entity.strip()
        if not entity:
            raise click.BadParameter("--max-hz entity cannot be empty")
        try:
            hz = float(hz_text)
        except ValueError as exc:
            raise click.BadParameter(f"--max-hz value must be numeric: {value}") from exc
        if hz < 0:
            raise click.BadParameter("--max-hz cannot be negative")
        if entity.startswith("/"):
            entity = f"world{entity}"
        parsed[entity] = hz
    return parsed


@main.command(name="rerun-record")
def rerun_record_cmd(
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output .rrd path. Defaults to logs/rerun-record-<timestamp>.rrd",
    ),
    include: list[str] = typer.Option(
        [],
        "--include",
        help="Topic or Rerun entity allowlist pattern. Repeatable.",
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        help="Topic or Rerun entity blocklist pattern. Repeatable.",
    ),
    max_hz: list[str] = typer.Option(
        [],
        "--max-hz",
        help="Throttle a Rerun entity as ENTITY=HZ. Repeatable.",
    ),
    memory_limit: str = typer.Option(
        "25%",
        "--memory-limit",
        help="Accepted for parity with rerun-bridge; no live viewer is started.",
    ),
) -> None:
    """Record Rerun data to a .rrd file until interrupted."""
    from dimos.visualization.rerun.bridge import run_recording

    run_recording(
        out or _default_rerun_record_path(),
        include=tuple(include),
        exclude=tuple(exclude),
        max_hz=_parse_rerun_record_max_hz(max_hz),
        memory_limit=memory_limit,
    )


@main.command(name="rerun-bridge")
def rerun_bridge_cmd(
    memory_limit: str = typer.Option(
        "25%", help="Memory limit for Rerun viewer (e.g., '4GB', '16GB', '25%')"
    ),
    rerun_open: str = typer.Option("native", help="How to open Rerun: native, web, both, none"),
    rerun_web: bool = typer.Option(
        True, "--rerun-web/--no-rerun-web", help="Enable/Disable Rerun web server"
    ),
) -> None:
    """Launch the Rerun visualization bridge."""
    from dimos.visualization.rerun.bridge import run_bridge

    run_bridge(
        memory_limit=memory_limit,
        rerun_open=cast("RerunOpenOption", rerun_open),
        rerun_web=rerun_web,
    )


if __name__ == "__main__":
    cli_main()
