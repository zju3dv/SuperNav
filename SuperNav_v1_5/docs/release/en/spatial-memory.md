# SuperNav-1.5: Structured Spatial Memory For Agentic Navigation Systems

**English** | [简体中文](../zh/spatial-memory.md)

SuperNav-1.5 is a subsequent work following SuperNav. It gives a navigation agent
structured spatial memory through persistent **memory nodes**, Cartographer 2D
mapping, occupancy navigation, area segmentation, map relocalization, and
optional visual captions.

This guide covers the released Habitat configuration. For the shortest runnable
example, see [Run with Structured Spatial Memory](../../../README.md#run-navigation-experiments).

## Before you start

Run all commands from the `SuperNav_v1_5` directory. You need:

- a Linux machine with an NVIDIA GPU and a working Habitat-GS installation;
- Conda for the native Cartographer build;
- a Habitat scene dataset configuration and a matching task manifest;
- access to the agent model selected by your experiment configuration; and
- enough GPU memory for Habitat-GS and, for the full recipe, the caption model.

The Cartographer build environment and the Habitat runtime environment are
separate. Keep both environments after installation.

<a id="install-and-build"></a>

## 1. Build Cartographer and the worker

Build the pinned, patched Cartographer source included with this release:

```bash
git submodule update --init third_party/cartographer
conda env create -f environment-cartographer.yml
conda activate supernav-cartographer
SUPERNAV_CARTOGRAPHER_BUILD_JOBS=4 scripts/build_cartographer_worker.sh
conda deactivate
```

The build command compiles Cartographer and the SuperNav worker, then runs their
release checks. The resulting worker is launched through
`scripts/run_cartographer_worker.sh`. Do not replace it with a system
Cartographer installation or a worker from another checkout.

## 2. Install the Habitat runtime

Activate the Python environment that already runs Habitat-GS, then install
SuperNav-1.5 and its spatial-memory dependencies:

```bash
conda activate <habitat-gs-runtime-environment>
python -m pip install -e '.[agents,spatial-memory,caption]'
```

Configure the simulator, scene dataset, GPU, and worker launcher:

```bash
export SUPERNAV_HABITAT_ROOT=/absolute/path/to/habitat-gs
export SUPERNAV_HABITAT_PYTHON=/absolute/path/to/habitat-gs-runtime-environment/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/absolute/path/to/global_navigation.scene_dataset_config.json
export HAB_GPU_DEVICE_ID=0
export HAB_CARTOGRAPHER_WORKER_COMMAND="scripts/run_cartographer_worker.sh {input_fifo} {occupancy_path}"
```

`HAB_GPU_DEVICE_ID` is the logical device visible after any
`CUDA_VISIBLE_DEVICES` filtering. No additional EGL setting is normally
required: SuperNav launches the bridge with the same renderer configuration as
the working Habitat-GS environment. The process must still have access to the
NVIDIA driver and selected GPU device.

The released Cartographer integration uses simulator-evaluator odometry with
authoritative poses.

## 3. Choose a memory recipe

Two Habitat recipes are included:

| Experiment | Configuration |
| --- | --- |
| `habitat-spatial-memory` | Memory nodes, 2D SLAM, occupancy navigation, relocalization, and Qwen area captions |
| `habitat-spatial-memory-slam-only` | The same memory and SLAM pipeline with area captions disabled |

Inspect the resolved full configuration before running:

```bash
supernav config show --experiment habitat-spatial-memory
```

Both recipes enable SuperNav-1.5 spatial memory and disable SuperNav v1's
bounded spatial-fact ledger. The agent therefore uses memory nodes as its
spatial memory rather than running both memory systems together.

Both recipes use the dedicated spatial-memory policy: one-call visual-point
movement for visible local targets and preview/confirmation occupancy routing
for off-screen memory nodes or current SLAM-area frontiers. Before acting, the
agent must load all three cooperating native Skills: `memory-nodes`, `2d-slam`,
and `spatial-memory-navigation`. The no-caption recipe changes only the caption
service setting; it uses the same prompt, tools, and Skills.

<a id="run-with-the-service-stack"></a>

## 4. Run an experiment

The full recipe uses a managed caption service. Model weights are not bundled;
by default the stack loads `Qwen/Qwen3.5-2B` from the local Hugging Face cache
or downloads it on first use.

```bash
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory
```

Replace `example-task` with a `task_id` from your manifest. You can select a
different Hugging Face model or a local model directory with
`--caption-model MODEL_OR_PATH`.

`stack run` checks the Cartographer worker, starts the caption service, waits
until it is ready, and forwards the remaining arguments to the standard
SuperNav experiment runner. The runner starts a Habitat bridge for each episode,
and the bridge starts the Cartographer worker for that session. When the run
finishes, the stack stops the caption process it started.

To run without a caption model:

```bash
supernav stack run \
  --experiment habitat-spatial-memory-slam-only \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory-slam-only
```

Task manifests, task selection, repetitions, run tags, output layout,
visualization, and evaluation follow the
[SuperNav v1 experiment workflow](../../../../SuperNav_v1/README.md#run-navigation-experiments).

## 5. Keep services running between experiments

For several consecutive full-memory runs, start the caption service once:

```bash
supernav stack up --experiment habitat-spatial-memory
supernav stack status

supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default

supernav stack logs
supernav stack down
```

Use `supernav stack logs --follow` in another terminal to follow the caption
service while an experiment is running. `stack up` starts the long-lived
caption service; the Habitat bridge and Cartographer worker are still created
only when `stack run` launches an episode.

## 6. Use an external caption service

The caption service can run in another environment, container, or host. Start
it with a Hugging Face model ID or local model path:

```bash
python -m supernav.backends.habitat.memory_slam.qwen_caption_server \
  --model Qwen/Qwen3.5-2B \
  --host 0.0.0.0 \
  --port 8926
```

Then tell the stack to validate and use that endpoint:

```bash
supernav stack run \
  --service-mode external \
  --caption-url http://caption-host:8926/caption \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default
```

Images are sent inline, so the caption service does not need access to the
experiment filesystem.

## Troubleshooting

- **Cartographer preflight fails:** activate or restore the
  `supernav-cartographer` environment and rerun
  `scripts/build_cartographer_worker.sh`.
- **Caption startup fails:** confirm that the `caption` dependencies are
  installed, the model is available, and the caption GPU has sufficient memory.
  Inspect the service output with `supernav stack logs`.
- **EGL or CUDA device errors:** first verify Habitat-GS in the same runtime
  environment, then check `CUDA_VISIBLE_DEVICES`, `HAB_GPU_DEVICE_ID`, the
  NVIDIA driver, and device access.
- **A service is already running:** check `supernav stack status`. Stop a
  managed stack with `supernav stack down`, or use external mode with the
  existing caption endpoint.
