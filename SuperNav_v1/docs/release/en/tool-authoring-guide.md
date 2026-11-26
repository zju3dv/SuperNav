[English](tool-authoring-guide.md) | [简体中文](../zh/tool-authoring-guide.md)

# Tool authoring guide

Navigation tools belong to SuperNav. Add their implementation under
`src/supernav/methods/navigation/tools/`; platform process startup and simulator
SDK bindings belong under `src/supernav/backends/`. Keep external Habitat-GS
checkouts read-only when adding a SuperNav tool. See [Architecture](architecture.md)
for the ownership boundaries.

The shared registry supplies schemas and dispatch for the navigation
MCP server. Public execution starts with `supernav mcp` or a configured experiment.

## Add a tool

1. Choose the appropriate `ToolCategory` for the operation.
2. Define a class with `metadata: ToolMetadata` and
   `execute(self, args, ctx) -> ToolResult` in the corresponding tools module.
3. Register it once with `ToolRegistry.register(YourTool())`, and ensure the module
   is imported by `src/supernav/methods/navigation/tools/__init__.py`.
4. Add focused tests for schema, bridge payloads, returned evidence, errors, and
   context restrictions. Use the
   [MCP contract tests](../../../tests/contract/v2/test_mcp_inline_rgb_images.py)
   and the relevant runtime tests as references.

Use one canonical tool name. Preserve model-visible names, schemas, native result
shapes, and outcome semantics as experiment contracts when reorganizing code.

## Definitions and state

Import common types from the canonical package:

```python
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)
```

[`ForwardTool`](../../../src/supernav/methods/navigation/tools/navigation.py) is a
concrete reference. It builds a bridge request, collects images, updates collision
and motion state, and returns the original result
inside `ToolResult`. Use its current schema and implementation as a reference.

| Type | Responsibility |
| --- | --- |
| `ToolMetadata` | Canonical name, description, JSON Schema, availability, and exposure |
| `ToolContext` | Bridge, session identifier, navigation and task modes, and round state |
| `RoundState` | Captured images, collision state, and movement history for the current round |
| `ToolResult` | `ok`, original result body, captured images, latency, and error |
| `ToolRegistry` | Registration, schema generation, context filtering, and dispatch |

Store mutable session state on `ctx.round_state` or another explicit context
object. This keeps the shared registry reusable across callers. Follow the caller's
image-clearing and round-state lifecycle to scope data to each call.

Key metadata fields:

- `name`, `category`, `description`, and `parameters_schema` define the model-facing API.
- `allowed_nav_modes` and `allowed_task_types` restrict execution contexts.
- `allowed_harness_modes` distinguishes forward and backward execution policies.
- `requires_session=False` is appropriate only for operations that establish a session.
- `mcp_visible=False` excludes an operation from MCP discovery.
- `permission` classifies the action.

## Bridge calls and results

Send simulator operations through `ctx.bridge.call(action, payload)`. Put SDK
integration and platform initialization in the backend, and deployment paths in
local configuration. Check actions against the SuperNav backend and the external
simulator API.

Return expected failures explicitly:

```python
return ToolResult(ok=False, body={}, error="A session is required")
```

The registry catches unexpected exceptions, but deliberate validation gives clearer
errors. Successful requests should preserve native result fields and capture their
images through the shared helpers. Propagate bridge errors with `ok=False`.
Record agent claims and session lifecycle events separately from ground-truth scoring.

Images and action history are recorded on the round context. Return a copy of the
captured image list to preserve each result as a snapshot.

## Filtering and MCP exposure

`ToolRegistry.available_for(nav_mode, task_type)` filters the available tools.
`ToolRegistry.dispatch(name, args, ctx)` checks the context again during execution.
Tests should cover both discovery and attempted dispatch in disallowed contexts.

The navigation MCP server in
[`methods/navigation/mcp_server.py`](../../../src/supernav/methods/navigation/mcp_server.py)
creates `hab_<tool-name>` wrappers for exposed tools. `_extract_params` translates
JSON Schema types, required arguments, and defaults into wrapper signatures.
`_MCP_RESPONSE_SHAPERS` handles established result layouts where needed; preserve
existing native outcomes when changing these projections.

Expose agent operations as tools and supporting data as MCP resources. Put reusable
utilities in shared helpers and keep backend maintenance operations on the bridge API.

Select the task protocol through the experiment's `benchmark_profile`. The backend
supplies the corresponding internal `HAB_MCP_GLOBAL_TASK` value. See
[Configuration](configuration.md) for experiment and arm composition.

## Verification

Use a fake bridge for deterministic checks of the requested action and exact payload.
A minimal fixture records calls and returns controlled responses:

```python
class FakeBridge:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}
        self.session_id = "s1"

    def call(self, action, payload=None):
        self.calls.append((action, dict(payload or {})))
        return self.responses.get(action, {})
```

Check the normal path, bridge failures, invalid arguments, session preconditions,
context gates, and preserved result/image fields. For a protocol-sensitive change,
compare model-visible schemas and native output with the verified contract.
Run the relevant existing tests and check imports in an environment with the
installed base package. Keep original tool outcomes and logs available for inspection.

## Reference modules

- [tools/base.py](../../../src/supernav/methods/navigation/tools/base.py): types and registry.
- [tools/navigation.py](../../../src/supernav/methods/navigation/tools/navigation.py): movement tools.
- [tools/perception.py](../../../src/supernav/methods/navigation/tools/perception.py): retained depth analysis and query tools.
- [tools/status.py](../../../src/supernav/methods/navigation/tools/status.py): state updates.
- [tools/session.py](../../../src/supernav/methods/navigation/tools/session.py): session validation.
- [tools/_common.py](../../../src/supernav/methods/navigation/tools/_common.py): visual payload, image collection, and boolean parsing helpers.
- [result_projection.py](../../../src/supernav/methods/navigation/result_projection.py): result projection.
- [backends/habitat/](../../../src/supernav/backends/habitat/): simulator boundary and lifecycle.
