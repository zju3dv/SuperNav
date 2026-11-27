"""Tool protocols, metadata, contexts and a shared registry.

ToolMetadata declares schemas and gates. RoundState owns mutable
per-round image, collision and action state. ToolContext supplies the
bridge, session and asset handles. ToolResult carries body, images,
latency and errors. Stateless tools register at import time; callers
use ToolRegistry to build schemas and dispatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from supernav.evaluation.measurement.tool_capture import record_tool_dispatch_fail_open


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolCategory(Enum):
    """High-level grouping for telemetry and UI."""

    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MAPPING = "mapping"
    STATUS = "status"
    SESSION = "session"
    MEMORY = "memory"


class PermissionLevel(Enum):
    """Permission level used to classify tool risk."""

    READ_ONLY = "read_only"  # does not change world state
    MUTATING = "mutating"  # changes state but is recoverable
    DESTRUCTIVE = "destructive"  # cannot be undone


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ToolMetadata:
    """Declarative tool schemas, availability gates and runtime hints."""

    # Identity
    name: str  # e.g. "forward"
    category: ToolCategory
    description: str  # LLM-facing help text

    # LLM interface — JSON Schema for OpenAI function calling
    parameters_schema: Dict[str, Any] = field(default_factory=dict)

    # Availability constraints. Empty `allowed_task_types` means "any
    # task type"; we use a sentinel rather than None so that the
    # filter logic stays a simple set-membership check.
    allowed_nav_modes: Set[str] = field(default_factory=lambda: {"navmesh", "mapless"})
    allowed_task_types: Optional[Set[str]] = None  # None → any task type

    # MCP aliases.
    legacy_names: Set[str] = field(default_factory=set)

    # Whether this tool should be exposed via the dynamic MCP server
    # registration. Orthogonal to allowed_nav_modes / allowed_task_types —
    # those control
    # agent execution context, this controls external API exposure.
    mcp_visible: bool = True

    # Harness availability is enforced at schema construction and dispatch.
    allowed_harness_modes: Set[str] = field(
        default_factory=lambda: {"forward", "backward"}
    )

    # Tier 2 catalog hint.
    when_to_use: str = ""

    # Runtime hints (telemetry / planning)
    typical_latency_ms: int = 0
    typical_token_cost: int = 0
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    requires_session: bool = True


    def __post_init__(self) -> None:
        # Sanity-check the most easily-broken fields. Catching these at
        # registration time is much more useful than catching them
        # at the first dispatch.
        if not self.name:
            raise ValueError("ToolMetadata.name must be a non-empty string")
        if not isinstance(self.category, ToolCategory):
            raise TypeError(
                f"ToolMetadata.category must be ToolCategory, "
                f"got {type(self.category).__name__}"
            )
        if not self.description:
            raise ValueError(
                f"ToolMetadata.description for {self.name!r} must not be empty"
            )
        if not isinstance(self.allowed_nav_modes, set):
            raise TypeError(
                f"ToolMetadata.allowed_nav_modes must be a set, "
                f"got {type(self.allowed_nav_modes).__name__}"
            )
        if not self.allowed_nav_modes:
            raise ValueError(
                f"ToolMetadata.allowed_nav_modes for {self.name!r} cannot be empty"
            )
        if self.allowed_task_types is not None and not isinstance(
            self.allowed_task_types, set
        ):
            raise TypeError(
                f"ToolMetadata.allowed_task_types must be a set or None, "
                f"got {type(self.allowed_task_types).__name__}"
            )
        if not isinstance(self.legacy_names, set):
            raise TypeError(
                f"ToolMetadata.legacy_names must be a set, "
                f"got {type(self.legacy_names).__name__}"
            )
        if not isinstance(self.allowed_harness_modes, set):
            raise TypeError(
                f"ToolMetadata.allowed_harness_modes must be a set, "
                f"got {type(self.allowed_harness_modes).__name__}"
            )
        if not self.allowed_harness_modes:
            raise ValueError(
                f"ToolMetadata.allowed_harness_modes for {self.name!r} "
                f"cannot be empty"
            )


# ---------------------------------------------------------------------------
# RoundState — per-round mutable tracking
# ---------------------------------------------------------------------------


@dataclass
class RoundState:
    """Mutable state shared by tool calls within a round.

    Perception and movement tools append captured images and visual paths.
    The agent clears images between calls; status tools consume collision
    and movement history and clear round_actions on commit. Collision
    recovery state gates movement until the required inspection completes.
    """

    captured_images: List[str] = field(default_factory=list)
    last_visual_path: Optional[str] = None
    last_visual_metadata: Optional[Dict[str, Any]] = None
    last_collided: bool = False
    last_movement_action: Optional[str] = None
    # Human-readable per-step action descriptions
    # (e.g. "forward(0.5m)!", "turn_left(45°)"). Joined with " → "
    # by update_nav_status to produce the action chain string that
    # gets auto-injected into action_history_append entries.
    round_actions: List[str] = field(default_factory=list)
    collision_recovery_stage: Optional[str] = None
    collision_look_down_degrees: float = 0.0

    def require_collision_backward(self) -> None:
        self.collision_recovery_stage = _COLLISION_NEED_BACKWARD
        self.collision_look_down_degrees = 0.0

    def mark_collision_backward_done(self) -> None:
        if self.collision_recovery_stage == _COLLISION_NEED_BACKWARD:
            self.collision_recovery_stage = _COLLISION_NEED_DEPTH

    def mark_collision_depth_checked(self) -> None:
        if self.collision_recovery_stage == _COLLISION_NEED_DEPTH:
            self.collision_recovery_stage = _COLLISION_NEED_LOOK_DOWN

    def mark_collision_look_vertical(self, direction: str, degrees: float) -> None:
        if degrees <= 0:
            return
        if self.collision_recovery_stage == _COLLISION_NEED_LOOK_DOWN:
            if direction == "down":
                self.collision_look_down_degrees += degrees
                if self.collision_look_down_degrees >= _COLLISION_LOOK_DOWN_MIN_DEGREES:
                    self.collision_recovery_stage = None
                    self.collision_look_down_degrees = 0.0


# ---------------------------------------------------------------------------
# ToolContext — what every tool dispatch sees
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Bridge, session and per-round state supplied to tool dispatch."""

    bridge: Any  # BridgeClient at runtime
    session_id: str
    output_dir: str
    nav_mode: str
    task_type: str
    workspace_host: str = ""
    is_gaussian: bool = False  # mutable: InitSceneTool sets this
    round_state: RoundState = field(default_factory=RoundState)
    harness: str = "forward"  # "forward" | "backward"
    dispatch_source: str = "in_process"
    dispatch_trace_path: str = ""
    controller_measurement_dir: str = ""


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Standardized tool output.

    Tools should return `ToolResult.ok=False` with an `error` string for
    expected failures (bridge errors, validation failures, missing
    sessions). Unexpected exceptions are caught by `ToolRegistry.dispatch`
    and converted to the same shape, so callers never need to wrap
    dispatch in try/except.
    """

    ok: bool
    body: Dict[str, Any] = field(default_factory=dict)
    captured_images: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """A capability the agent can invoke.

    Subclass-via-Protocol: any class with a `metadata` class attribute
    and an `execute(args, ctx)` method satisfies this. We use
    `runtime_checkable` so `isinstance(x, Tool)` works in tests.
    """

    metadata: ToolMetadata

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Perform the action and return a ToolResult."""
        ...


_COLLISION_NEED_BACKWARD = "need_backward"
_COLLISION_NEED_DEPTH = "need_depth_analyze"
_COLLISION_NEED_LOOK_DOWN = "need_look_down"
_COLLISION_LOOK_DOWN_MIN_DEGREES = 60.0


def _collision_recovery_error(name: str, args: Dict[str, Any], ctx: ToolContext) -> Optional[str]:
    if name in {"init_scene", "close_session"}:
        return None
    stage = ctx.round_state.collision_recovery_stage
    if stage == _COLLISION_NEED_BACKWARD:
        if name == "backward":
            return None
        return (
            "collision recovery required: call backward with a short "
            "distance before depth_analyze after a collided forward step"
        )
    if stage == _COLLISION_NEED_DEPTH:
        if name in {"depth_analyze", "depth_grid"}:
            return None
        return (
            "collision recovery required: call depth_analyze or depth_grid before "
            "any further movement or perception after a collided forward step"
        )
    if stage == _COLLISION_NEED_LOOK_DOWN:
        if name == "look_vertical" and args.get("direction") == "down":
            return None
        return (
            "collision recovery required: perform a substantial downward "
            "inspection after depth_analyze. Call look_vertical with "
            f"direction='down' until at least {_COLLISION_LOOK_DOWN_MIN_DEGREES:g} "
            "total degrees have been inspected; shallow symbolic glances do not count."
        )
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of stateless tools used for schema construction and dispatch."""

    _tools: Dict[str, Tool] = {}

    # ── registration ──────────────────────────────────────────────

    @classmethod
    def register(cls, tool: Tool) -> None:
        """Register a tool, replacing any existing entry with the same name."""
        if not hasattr(tool, "metadata"):
            raise TypeError(
                f"register() expects an object with a `metadata` attribute, "
                f"got {type(tool).__name__}"
            )
        if not hasattr(tool, "execute"):
            raise TypeError(
                f"register() expects an object with an `execute` method, "
                f"got {type(tool).__name__}"
            )
        cls._tools[tool.metadata.name] = tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a tool. Mainly used by tests to keep the global
        registry isolated between cases."""
        cls._tools.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry. Tests use this in fixtures."""
        cls._tools.clear()

    # ── lookup ────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> Optional[Tool]:
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[Tool]:
        return list(cls._tools.values())

    @classmethod
    def available_for(
        cls,
        nav_mode: str,
        task_type: str,
        *,
        harness: str = "forward",
        ctx: Optional["ToolContext"] = None,
    ) -> List[Tool]:
        """Return tools accepted by navigation, task and harness gates."""
        out: List[Tool] = []
        for tool in cls._tools.values():
            if nav_mode not in tool.metadata.allowed_nav_modes:
                continue
            if (
                tool.metadata.allowed_task_types is not None
                and task_type not in tool.metadata.allowed_task_types
            ):
                continue
            if harness not in tool.metadata.allowed_harness_modes:
                continue
            out.append(tool)
        return out

    # ── schema generation ─────────────────────────────────────────

    @classmethod
    def build_openai_schemas(
        cls,
        nav_mode: str,
        task_type: str,
        *,
        harness: str = "forward",
        ctx: Optional["ToolContext"] = None,
    ) -> List[Dict[str, Any]]:
        """Generate function-calling schemas from filtered tool metadata.

        Apply the same navigation, task and harness gates as dispatch.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters_schema
                    or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for tool in cls.available_for(nav_mode, task_type, harness=harness, ctx=ctx)
        ]

    # ── dispatch ──────────────────────────────────────────────────

    @classmethod
    def dispatch(
        cls,
        name: str,
        args: Dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        """Look up `name` and execute it with `args` against `ctx`.

        Always returns a ToolResult — never raises. Unknown tools and
        unexpected exceptions are converted to ``ok=False`` with an
        ``error`` string so callers don't need to wrap dispatch in
        try/except. The latency_ms field is populated even on failure
        so telemetry stays consistent.
        """
        # Start the latency clock before dispatch validation so success, exceptions and
        # gate rejections all carry measured latency.
        start = time.perf_counter()

        def _elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000.0

        def _finish(result: ToolResult, *, capture_tool: Optional[Tool] = None) -> ToolResult:
            record_tool_dispatch_fail_open(
                ctx=ctx,
                tool_name=name,
                args=args,
                result=result,
                tool=capture_tool,
            )
            return result

        tool = cls.get(name)
        if tool is None:
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=f"Unknown tool: {name}",
                    latency_ms=_elapsed_ms(),
                )
            )


        # Defense-in-depth availability gates. `available_for()` filters
        # what the LLM sees in its schemas, but a hallucinated tool name
        # or a direct call from test code could otherwise bypass that
        # filter. Reject at dispatch time as well, mirroring the exact
        # rules in `available_for`.
        if ctx.nav_mode not in tool.metadata.allowed_nav_modes:
            allowed = ",".join(sorted(tool.metadata.allowed_nav_modes))
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=(
                        f"Tool {name!r} not allowed in nav_mode={ctx.nav_mode!r} "
                        f"(allowed_nav_modes={{{allowed}}})"
                    ),
                    latency_ms=_elapsed_ms(),
                ),
                capture_tool=tool,
            )
        if (
            tool.metadata.allowed_task_types is not None
            and ctx.task_type not in tool.metadata.allowed_task_types
        ):
            allowed = ",".join(sorted(tool.metadata.allowed_task_types))
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=(
                        f"Tool {name!r} not allowed for task_type={ctx.task_type!r} "
                        f"(allowed_task_types={{{allowed}}})"
                    ),
                    latency_ms=_elapsed_ms(),
                ),
                capture_tool=tool,
            )
        # Apply the schema's harness gate at dispatch too, including direct calls to
        # forward-only tools.
        if ctx.harness not in tool.metadata.allowed_harness_modes:
            allowed = ",".join(sorted(tool.metadata.allowed_harness_modes))
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=(
                        f"Tool {name!r} not allowed for harness={ctx.harness!r} "
                        f"(allowed_harness_modes={{{allowed}}})"
                    ),
                    latency_ms=_elapsed_ms(),
                ),
                capture_tool=tool,
            )

        collision_recovery_error = _collision_recovery_error(name, args, ctx)
        if collision_recovery_error is not None:
            return ToolResult(
                ok=False,
                body={},
                error=collision_recovery_error,
                latency_ms=_elapsed_ms(),
            )

        try:
            result = tool.execute(args, ctx)
        except Exception as exc:
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=_elapsed_ms(),
                ),
                capture_tool=tool,
            )

        # Convert unexpected return values to a failed ToolResult, preserving dispatch
        # error handling even for malformed tool implementations.
        if not isinstance(result, ToolResult):
            return _finish(
                ToolResult(
                    ok=False,
                    body={},
                    error=(
                        f"Tool {name!r} returned {type(result).__name__}, "
                        f"expected ToolResult"
                    ),
                    latency_ms=_elapsed_ms(),
                ),
                capture_tool=tool,
            )

        # Tools may have set their own latency_ms. Only fill it in if
        # they didn't, so handcrafted measurements (e.g. excluding
        # post-processing) win over our outer measurement.
        if result.latency_ms == 0.0:
            result.latency_ms = _elapsed_ms()
        return _finish(result, capture_tool=tool)


__all__ = [
    "ToolCategory",
    "PermissionLevel",
    "ToolMetadata",
    "RoundState",
    "ToolContext",
    "ToolResult",
    "Tool",
    "ToolRegistry",
]
