# Experiment Configuration

**English** | [简体中文](../zh/configuration.md)

Shared recipes in `configs/experiments/` are example templates. To run an
experiment, prepare:

1. Scene assets or datasets and a compatible simulator environment.
2. A configuration adapted to your machine: model, credentials, paths, and services.
3. Task instructions or a manifest with the matching scenes, spawn poses, and goals.

Provide GT when required by the selected method or scoring procedure. Use
`--experiment <name>` with your environment settings and CLI options, or save
local overrides in ignored `configs/local/` files and run with `--config <path>`.

```text
configs/
├── experiments/   Named experiment recipes
├── agents/        Client selection and timeouts
├── models/        Model names and reasoning settings
├── backends/      Simulator and protocol settings
├── methods/       Navigation methods, arms, tools, and Skills
├── benchmarks/    General prompt templates
└── local/         Local overrides; not committed or packaged
```

```bash
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

## Available recipes

| Recipe | Purpose |
| --- | --- |
| `habitat-geo-based-executor` | Habitat navigation with a geometry-based executor and native Skills |
| `habitat-learned-executor` | Habitat navigation with a learned local policy and native Skills; weights not yet released |
| `ai2thor-primitive` | AI2-THOR demand-driven navigation with primitive actions |

Public recipes set `instructions_file` to `null`. Supply a task manifest through
`--instructions /path/to/tasks.json` or a local overlay's `instructions_file`.
The paper's task lists and ground truth are excluded from the release. Prepare
the matching scene assets and external datasets. Each recipe has one `default`
arm; omit `--arms` or use `--arms default`. Habitat recipes load native Skills.
The geometry-based executor uses oracle depth and NavMesh paths. The learned
executor requires a policy service and its checkpoint. Its weights are not yet
released, and no download is currently available. See the
[policy service requirements](habitat.md#learned-executor-weights) before selecting
this recipe for a real run.

## Models and credentials

| Setting | Location and behavior |
| --- | --- |
| Agent client | `agent` and `agents.<client>` in the agent fragment or local overlay |
| Model and reasoning | `model.name=gpt-6-astra`; Codex `extra_args` set `model_reasoning_effort="medium"` |
| Codex provider mode | Shared recipes set `agents.codex.provider_mode=user`, using the current client's provider/auth |
| Per-run model | `--model` overrides the configured model name |
| Credentials | Environment variables or local client credential files |

All bundled recipes use the shared
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json)
fragment: model `gpt-6-astra` with `medium` reasoning. Local overlays inherit these
settings unless explicitly overridden. `--model` changes the model name and keeps
the configured reasoning setting.

API clients select a provider through `agents.<client>.provider` in a local
experiment configuration. Credentials use the environment variable named by
`env_key`; see the [configuration guide](configuration.md).

Codex `provider_mode=user` continues to use the client configuration and login.
For explicit `provider_mode=experiment`, set `agents.codex.provider.id`, `base_url` and
`env_key` in a local experiment overlay; there is no built-in endpoint. The
optional `HAB_BENCH_RELAY_BASE_URL` environment variable overrides that URL.
The image-prune proxy requires `--upstream` or `HAB_BENCH_RELAY_UPSTREAM`.


## Local deployment and runs

Shared Habitat recipes use `${SUPERNAV_SCENE_DATASET_CONFIG}` for their scene
configuration. Create `configs/local/habitat.json`:

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "model": {"name": "gpt-6-astra"},
  "output_dir": "data/runs/habitat",
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}",
    "path_prefixes": {
      "/datasets/hm3d_ovon": "${SUPERNAV_OVON_ROOT}"
    }
  }
}
```

Replace the paths with your installation and `example-task` with a `task_id`
from your manifest:

```bash
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/scene_dataset_config.json
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python

supernav config show --config configs/local/habitat.json
supernav run --config configs/local/habitat.json \
  --task-ids example-task --dry-run \
  --output-dir data/runs/habitat-dry

supernav run --config configs/local/habitat.json \
  --task-ids example-task \
  --output-dir data/runs/habitat
```

Prepare the simulator and datasets before the real run; see [Habitat](habitat.md).
Use distinct directories for dry-runs and real runs: existing episodes are skipped
by default. `--run-tag` can distinguish repeated runs. `config show` displays file
composition with credential and deployment variables preserved as placeholders.
Run commands apply CLI overrides and resolve deployment variables. A dry-run ends
after preparing prompts, projects, and Skill snapshots; supply any task data
required by the backend.

To use the learned executor with your own compatible checkpoint and policy
service, extend `habitat-learned-executor.json`. Until weights are released,
use `habitat-geo-based-executor` if you do not have these resources.
`benchmark_profile` declares the Habitat task
protocol: `global_task` enables the global-task protocol; an omitted profile uses
standard mode.

External formats such as OVON require matching scene assets, spawn poses, goals,
and source episodes. Use `deployment.path_prefixes` to map dataset path prefixes
in your manifest to local directories. Relocation preserves instructions, goals,
spawn poses, and GT values.

## Backend processes

Shared Habitat recipes set `bridge.per_episode=true` and a 120-second startup
budget. Each episode owns its bridge process; configure ports in the experiment
or local overlay. Navigation MCP normally uses the current interpreter with
`python -m supernav mcp --transport stdio`.

For custom MCP processes, use `mcp.command`, `mcp.args`, `mcp.http_args`,
`mcp.transport`, and `mcp.environment`. `environment.python` selects the simulator
interpreter; the agent and navigation MCP retain their own interpreters.

The Habitat backend normally imports the SDK installed in that interpreter.
`environment.habitat_root` / `SUPERNAV_HABITAT_ROOT` optionally selects a public
SDK checkout with compatible native extensions already installed. See
[Habitat](habitat.md) and [AI2-THOR](ai2thor.md) for backend-specific setup.

## Composition rules

1. Top-level `extends` accepts a path or a list of paths, resolved relative to
   the file declaring them.
2. Parents merge from left to right; the current file overrides them last.
   Dictionaries merge recursively; lists, scalars, and `null` replace their
   previous values. An empty dictionary retains the inherited dictionary.
3. Top-level `extends` disappears after loading. Cycles, missing parents, and
   invalid types are errors.
4. `arms.<name>.extends` references an arm name and uses shallow overrides within
   the composed arm table.
5. Resource paths use `workspace_root`; read-only assets can resolve from the
   installed package. Task, scene, output, and Skill path values are preserved during inheritance.
6. Every recipe explicitly declares `prompts_dir`.
7. Run options such as `--backend`, `--model`, and `--output-dir` apply after composition.

Select exactly one of `--experiment` or `--config` for each run.

## Task manifests

Store task manifests and GT outside the public repository. An external manifest
can compose other files through ordered `includes`, for example:

```json
{
  "includes": [
    {
      "path": "./scene-a.json",
      "defaults": {"scene": "scene-a"}
    }
  ]
}
```

Each `path` is relative to its declaring manifest. Recursive includes are allowed
and cycles are rejected. `defaults` fills missing top-level task fields;
a task's own `spawn` takes precedence. Goal and GT objects are treated as complete
field values.
Included tasks retain their order, followed by the current manifest's `instructions`.
Use `--task-ids` to select tasks from a multi-scene recipe.

## Other configuration directories

`configs/runtime/`, `evaluation/`, `providers/`, and `viewers/` contain settings
for their consumers, each with its own file format and loader.
Add new distributable assets to `pyproject.toml` and verify the installed wheel.
See [Architecture](architecture.md) and [Directory layout](layout.md).
