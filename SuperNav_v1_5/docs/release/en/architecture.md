[English](architecture.md) | [简体中文](../zh/architecture.md)

# Architecture and responsibilities

SuperNav owns agent execution, navigation methods, skills, experiment definitions,
evidence collection, and evaluation. Habitat-GS and AI2-THOR provide simulator
backends. The public entry points are `supernav` and `python -m supernav`.
See [Repository layout](layout.md) for file ownership and
[Configuration](configuration.md) for experiment composition.

```text
supernav
  ├─ config list / show
  ├─ run / run-one
  │    └─ experiments: configuration → manifest → tasks × arms × repetitions
  │         ├─ runtime: agents, MCP processes, skills, events
  │         ├─ methods: navigation tools, prompts, policies, memory
  │         ├─ backends: simulator processes and protocol adapters
  │         └─ evaluation: evidence, metrics, offline scoring, replay
  ├─ prepare: task and experiment input preparation
  ├─ score-* / collect-objectnav / aggregate / metrics / timing / video*
  └─ web / viewer: observation and presentation
```

## Ownership boundaries

| Layer | Location | Responsibility |
| --- | --- | --- |
| CLI | `src/supernav/cli.py` | Thin routing; subcommand modules own their arguments and implementation |
| Experiments | `src/supernav/experiments/` | Configuration composition, task selection, sweeps, episodes, and matrices |
| Runtime | `src/supernav/runtime/` | Simulator-independent agent clients, MCP process descriptions, skills, and original events |
| Methods | `src/supernav/methods/` | Navigation tools, prompts, policies, and memory |
| Backends | `src/supernav/backends/` | Simulator lifecycle, protocol bindings, task adaptation, and native evidence |
| Evaluation | `src/supernav/evaluation/` | Offline scoring, measurements, trajectories, videos, and replay |
| Presentation | `src/supernav/web/` | Read-only visualization of run evidence |
| Shared contracts | `src/habitat_contract/` | SDK-independent navigation and task protocols |

Experiments coordinate platforms through `EnvironmentBackend`; agent clients
consume the common `MCPServerSpec`. Backends also select method prompts and assemble
evaluation evidence. Simulator SDK imports and platform initialization belong in
backends; deployment paths belong in local configuration, and matrices in experiments.

Use canonical `supernav.*` imports and the shared `habitat_contract` package.
Development and audit helpers live in `scripts/`; reusable implementation belongs
in the package.

## Experiment lifecycle

```text
--experiment <name> or --config <file>
  → compose configuration through extends
  → expand manifest includes and apply explicit local deployment settings
  → select tasks, arms, and repetitions; save resolved inputs
  → freeze skills and validate visibility for native-skill arms
  → backend starts the simulator; runtime starts the agent
  → preserve native logs; derive canonical events, metrics, and replay
  → backend cleans up episode processes
```

`experiments/config.py` handles discovery and composition; `arguments.py` requires
an explicit configuration selection. `sweep.py` orchestrates batches, `episode.py`
owns individual episodes, and `deployment.py` relocates only designated path
fields.

Each sweep writes `experiment.resolved.json`, `instructions.resolved.json`, and
`selection.json`. Each episode saves its task, experiment, and effective execution
configuration. Offline scoring uses that run's resolved task snapshot. Run output
defaults to `data/runs/`; callers can explicitly select another output directory.
Retain the original evidence with each run's code and configuration provenance.

`config show` displays file composition. Run-command overrides and credential
resolution apply during experiment execution. `--dry-run` stops after preparing
the run artifacts.

Client timeouts produce `timed_out=true` and return code 124. Event and metric
collection still runs, retaining the error and native logs, and the CLI exits
nonzero. Agent completion claims and offline ground-truth scores are recorded
separately. AI2-THOR currently uses `success_scoring=withheld`, with `success` and
`spl` set to `null`.

## Configuration and task protocols

| Location | Content |
| --- | --- |
| `configs/experiments/` | Named experiment recipes |
| `configs/agents/`, `models/`, `backends/`, `methods/` | Reusable client, model, platform, and method fragments |
| `configs/benchmarks/` | General prompt templates |
| `configs/runtime/`, `evaluation/`, `providers/`, `viewers/` | Consumer-specific policies and settings |
| `configs/local/` | Local deployment overlays and external task-manifest paths |

Task manifests, scene/spawn annotations, and evaluation GT are supplied externally
through `--instructions` or a local overlay's `instructions_file`.
Every experiment explicitly supplies `prompts_dir`. Preserve prompt contents,
tool schemas, task ordering, and arm merge semantics when reorganizing inputs.

For Habitat, `benchmark_profile` declares the task protocol. `global_task` selects
the global-task protocol; an omitted profile uses the standard protocol. Prompt
constraints, initialization, tools, and metrics consume the same profile. The
backend derives the internal `HAB_MCP_GLOBAL_TASK` value and enforces the declared
profile for launched processes. Configure the protocol through `benchmark_profile`.

Custom MCP processes use `command`, `args`, `http_args`, `transport`, and
`environment`. Standard backend launches use canonical Python modules.

## Implementation map

The Habitat bridge uses SuperNav's
[`backends/habitat/adapter.py`](../../../src/supernav/backends/habitat/adapter.py)
and binds the public `habitat_sim` SDK through `backends/habitat/simulator/`.
[`habitat_contract/navigation.py`](../../../src/habitat_contract/navigation.py)
defines session, observation, movement, pathfinding, and ray interfaces. Methods
access simulator operations through these contracts, with SDK integration handled
by the backend. See [Habitat setup](habitat.md) for the external SDK requirements.

Navigation state validation, spatial memory, and task constraints live in
`methods/navigation/state.py`; shared state fields are defined in
`habitat_contract/navigation_state.py`. Long-running processes, sessions, and
native evidence remain backend responsibilities.

| Component | Implementation |
| --- | --- |
| Agent clients and runtime helpers | `runtime/agents.py`, `runtime/support/` |
| Navigation tools, memory, and prompts | `methods/navigation/` |
| Learned local execution and training geometry | `methods/localnav/`, `methods/localnav_policy/rollout_geometry.py` |
| Demand-driven navigation | `methods/demand_driven/` |
| Task preparation | `experiments/preparation/` |
| ObjectNav scoring and videos | `evaluation/objectnav/`, `evaluation/video/` |
| Rerun viewer | `web/rerun_viewer.py` |

Under
`methods/navigation/`, `passability.py` handles depth-based passability and
`result_projection.py` projects tool results; `mcp_server.py` owns session state,
dynamic schemas, and MCP registration. Extend modules along these behavior boundaries.

## Development verification

Changes to entry points or configuration should check composition rules, invalid
input, canonical imports, prompt equivalence, task protocols, and both backend
execution contracts. Tool changes should also follow the
[Tool authoring guide](tool-authoring-guide.md).

Run targeted pytest checks and validate an installed wheel from outside the source
tree using
[`scripts/check_installed_package.py`](../../../scripts/check_installed_package.py).
The installed-package check covers CLI access, recipes, prompts, Skills, and
dry-runs with synthetic external tasks using only the base package dependencies.

For simulator validation, record the tested code, configuration, commands, native
evidence, and limitations. Keep external simulator checkouts read-only and retain
failed attempts alongside successful ones.
