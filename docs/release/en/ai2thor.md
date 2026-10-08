[English](ai2thor.md) | [简体中文](../zh/ai2thor.md)

# AI2-THOR backend

Use `ai2thor-primitive` as an example template. Before running
an experiment, prepare:

1. Scene and episode data, including house data, and a compatible AI2-THOR/Unity environment.
2. A configuration adapted to your machine: model, credentials, paths, and services.
3. Task instructions or a manifest with the matching scenes, spawn poses, and goals.

Provide GT when required by the selected method or scoring procedure. Use
`--experiment` with your environment settings and CLI options, or `--config`
with a local overlay. Select task data through `--instructions` or the overlay's
`instructions_file`.

`environment.backend` selects the simulator and `agent` selects the client.
Use tasks and arms compatible with the chosen `--backend`. SuperNav manages
task selection, agents, Skills, simulator processes, logs, and run directories
through the shared experiment runner.

## Prerequisites

Use Python 3.12 on Linux with NVIDIA/Vulkan support. Prepare:

- AI2-THOR **5.0.0**, installed by the `ai2thor` extra.
- CloudRendering build **`f0825767cd50d69f666c7f282e54abfe58f1e917`**. Place its
  executable and metadata in
  `thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917/` under your releases
  directory.
- The frozen demand-driven dataset, obtained separately, including `audit.json`,
  `test/episodes/*.json`, and
  `test/reproducibility/<episode_id>/house_data.json`. Use the external dataset's task
  and provenance files together with its house data.
- The selected agent CLI and authentication for real model runs.

The dataset uses model-conditioned selection, recorded as
`model_conditioned_selection=true`. Report this selection protocol with results
for the dataset. The runner uses each task's original request; an `--instructions`
subset must match the external dataset's task IDs and request text.

## Install and run

After preparing the dataset and CloudRendering build, run these commands from the
SuperNav repository root. Replace paths with your setup:

```bash
python -m pip install -e '.[ai2thor,agents,evaluation]'
export SUPERNAV_AI2THOR_DATASET=/path/to/frozen-demand-driven-dataset
export SUPERNAV_AI2THOR_RELEASES=/path/to/ai2thor/releases

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task --dry-run \
  --output-dir data/runs/ai2thor-dry

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task \
  --model gpt-6-astra --output-dir data/runs/ai2thor-experiment
```

Replace `example-task` with an ID from your external manifest.
Use `--dry-run` to validate the dataset and prepare prompts and agent configuration.
This check requires the external dataset. Use separate output directories for
dry-runs and real runs; existing episode directories are skipped by default.
For real runs, the runner starts and stops the bridge and Unity processes.

The recipe uses Codex with the local client's provider and authentication, and
`gpt-6-astra` with `medium` reasoning from
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json).
Select Kimi or OpenCode through `agent`, `agents`, and `model`; see the
[configuration guide](configuration.md).

Use `--task-ids` to select a subset or omit it to run all tasks in the supplied
manifest. The shared
`--arms`, `--reps`, `--rep-start`, `--model`, `--codex-provider`, `--run-tag`,
`--output-dir`, `--dry-run`, and Skill snapshot options are also available.
The recipe has one `default` arm; omit `--arms` or use `--arms default`.
List available recipes with `supernav config list`.

## Backend configuration

Create the ignored `configs/local/` directory and save local overrides in
`configs/local/ai2thor.json`:

```json
{
  "extends": "../experiments/ai2thor-primitive.json",
  "instructions_file": "/path/to/tasks/ai2thor.json",
  "environment": {
    "backend": "ai2thor",
    "dataset": "/path/to/frozen-demand-driven-dataset",
    "releases": "/path/to/ai2thor/releases",
    "python": "/path/to/simulator-env/bin/python",
    "views": "four",
    "gpu": 0
  },
  "bridge": {"host": "127.0.0.1", "port": 0, "startup_timeout_s": 120}
}
```

Run it with `supernav run --config configs/local/ai2thor.json` and the task/model
options above. You can also set `dataset`, `releases`, and `python` through
`SUPERNAV_AI2THOR_DATASET`, `SUPERNAV_AI2THOR_RELEASES`, and
`SUPERNAV_AI2THOR_PYTHON`; explicit configuration takes precedence. The interpreter
defaults to the runner's Python. Install the same SuperNav version and AI2-THOR
dependencies when using a separate simulator environment.

Custom MCP processes use `mcp.command`, `mcp.args`, and optional `mcp.http_args`.
The runner configures stdio MCP for Codex/OpenCode and HTTP MCP for Kimi.
Set the bridge port to `0` for automatic selection, or specify an available port.

## Observations and navigation

Select four-direction observations with `environment.views=four` (the default),
or front-only observations with `front`.

| Setting | `views=front` | `views=four` |
| --- | --- | --- |
| Agent observations | Front RGB | Front/right/back/left RGB |
| Resolution | 640 × 480 | 640 × 480 per image |
| Field of view | 120° vertical | 90° horizontal, approximately 73.74° vertical |
| Camera height | Native camera; may set `camera_height=1.25` | Fixed at 1.25 m above the floor |
| Action budget | 500 actions | 500 units; each primitive action or STOP costs one |
| Wall-clock budget | Agent timeout | 3600 seconds after simulator initialization, also subject to agent timeout |

The four cameras share a position and face yaw offsets of 0°, 90°, 180°, and −90°.
Observations and resource claims cost zero action units. Report results separately
for each view mode because their fields of view differ.

Each capture has an `observation_id`, such as `frame_000005`. MCP returns it with
the direction labels and images. Image references for one capture are
`frame_000005:front`, `:right`, `:back`, and `:left`. Repeated observe calls reuse
the current capture ID. `action_index` records the action count.

The recipe's `default` arm uses 0.1 m movement and 10° turn/look actions. To use
NoMaD, deploy the policy service and checkpoint, set the arm's `movement` to `nomad`,
and configure `environment.policy_url`. Four-view NoMaD uses
[`fourview_instructions.md`](../../../src/supernav/methods/demand_driven/fourview_instructions.md);
front-view NoMaD also requires `environment.camera_height=1.25`.

Directional local navigation saves the selected target image, then turns toward
it: right/left require nine 10° steps, and back requires eighteen. These turns
consume action budget. The policy then uses current front RGB and the saved goal
image. Reaching the budget ends the episode; local arrival is recorded as a
navigation event.

Select arms that expose the AI2-THOR tools: `ddn_observe`, `ddn_step`,
`ddn_local_navigate`, `ddn_claim_resource`, and `ddn_stop`. Configure task
instructions through native Skills or explicit `agent_instructions`.

## Live viewing

Before starting the viewer and runner, set the same absolute directory in both
terminals. From the same repository root:

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
```

Start the viewer in one terminal:

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

Run the experiment in the other. You can also set the publishing directory with
`environment.live_dir`. Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and
select `AI2-THOR · <scene_id>` to see images, actions, XZ trajectories, directional
points, and recorded agent/tool output. The runner links the run ID, instruction,
and logs automatically. The viewer retains the final frame and each point's
capture/direction association. See [Web Viewer](web-viewer.md) for details.

## Evidence and scoring

Runs save `run.json`, `prompt.txt`, raw agent logs, `canonical.jsonl`, `command.json`,
`metrics.json`, timing summaries, and the parent `results.jsonl`. Backend evidence
includes `backend.json`, `simulator.log`, `simulator-process.json`, and a
`simulator-<attempt>/` directory for each attempt. Caches, FIFO files, PlayerPrefs,
and Unity logs also go into the run directory.

Native evidence includes `protocol.json`, dataset provenance, frames, evaluator
metadata, resource claims, camera synchronization, turn costs, and termination
reasons. A client timeout records `terminal_status=timeout` and return code `124`;
an agent return before STOP records `agent_returned_without_stop`. Startup failures,
agent errors, and timeouts return a nonzero CLI status, preserve evidence collected
so far, and trigger process cleanup.

AI2-THOR's current evaluation mode is **`success_scoring=withheld`**, with
`success` and `spl` set to `null`. Report trajectories, resource claims, local
arrival, and STOP as collected execution evidence. Task SR/SPL requires a separate
validated scorer.

Agent tools provide task requests, RGB observations, and action results. For
formal evaluation with agents that have shell/file access, isolate evaluator data
from the agent's filesystem: house data, task DAGs, candidate objects, pose, and
ground-truth success feedback belong on the evaluator side. See
[architecture](architecture.md) for module responsibilities and the
[documentation index](README.md) for other guides.
