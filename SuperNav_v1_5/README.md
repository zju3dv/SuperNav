<div align="center">

<img src="docs/release/assets/supernav1.5-wordmark.svg" alt="SuperNav-1.5" width="580">

<h1>Structured Spatial Memory For Agentic Navigation Systems</h1>

<p><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a></p>

<p>
  <a href="https://zju3dv.github.io/SuperNav-1.5/"><img src="https://img.shields.io/badge/Project-SuperNav--1.5-e34b4b" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2611.29912"><img src="https://img.shields.io/badge/Paper-arXiv%3A2611.29912-2f6feb" alt="SuperNav-1.5 paper"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/Code-SuperNav--1.5-black?logo=github" alt="SuperNav-1.5 code"></a>
  <a href="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="Overview video"></a>
</p>

<p>
  Jiarui Guo<sup>1</sup> ·
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Yongchen Zai<sup>2</sup> ·
  Futang Zhu<sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>3</sup> ·<br>
  <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;·&nbsp;
  <sup>2</sup> BYD New Technology Research Institute &nbsp;·&nbsp;
  <sup>3</sup> Shenzhen University
</p>

<video src="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4" poster="docs/release/assets/supernav-1.5-overview.webp" controls muted loop width="920"></video>

</div>

## 📢 News

> **[2026-10-21]** 🤖 The **Unitree Go2 deployment** is released, with structured spatial memory, four-view RGB, LiDAR-based map-backed PixelGoal navigation, area captions, and cross-session relocalization. See the [deployment guide](docs/release/en/go2.md) for Jetson installation and operation.

> **[2026-10-13]** 🎉 The InteriorGS **ObjectNav and Multi-ObjectNav test episodes** (150 each, with reference trajectories) are released in [HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark). See the [benchmark guide](https://github.com/zju3dv/habitat-gs/blob/main/docs/benchmark/README.md) for data and evaluation.

## 🧭 What Is SuperNav-1.5?

**SuperNav-1.5** gives navigation agents structured spatial memory: an
area-segmented occupancy map with agent-maintained **memory nodes**. It connects
scene recall to executable routes, helping the agent explore efficiently, reach
successive goals, and reuse memory across sessions. It is subsequent work in the
SuperNav series. See [original SuperNav](../SuperNav_v1/README.md) for the
preceding method.


This repository provides the agent runtime, navigation tools and Skills,
experiment recipes, and evaluation utilities for **Habitat-GS** and **AI2-THOR**.
With SuperNav-1.5, you can:

- **Build persistent spatial memory** from memory nodes, metric SLAM, and captions.
- **Replace the legacy ledger** whenever a spatial-memory experiment is selected.
- **Run navigation agents** with Codex, OpenCode, or Kimi.
- **Compose tasks, tools, and Skills** through explicit experiment configurations.
- **Watch execution live**, including RGB observations, trajectories, and tool calls.
- **Inspect and evaluate runs** using saved prompts, tool results, native logs,
  and independent task metrics.

See the [SuperNav-1.5 project page](https://zju3dv.github.io/SuperNav-1.5/) for
real-world and simulation videos, the method overview, and research results.
The [Unitree Go2 deployment source](docs/release/en/go2.md) provides the same
structured spatial-memory workflow on a real robot: four-view RGB, annotated
LiDAR occupancy maps, memory nodes, optional area captions, map-backed PixelGoal
navigation, and cross-session relocalization. It is assembled on a pinned public
DimOS submodule. Further real-world demonstrations are on the project page.

## 📖 Table of Contents

- [Release Status](#release-status)
- [Installation](#installation)
- [Run with Structured Spatial Memory](#run-navigation-experiments)
- [Shared SuperNav v1 Workflows](#visualization-and-evaluation)
- [Documentation and Development](#documentation-and-development)
- [Citation](#citation)
- [Acknowledgements and License](#acknowledgements-and-license)

<a id="release-status"></a>

## 📦 Release Status

- [x] Agent runtime, navigation tools and Skills, experiment recipes, and evaluation
- [x] Structured spatial memory, patched Cartographer worker, and stack lifecycle CLI
- [ ] Unitree Go2 real-robot deployment: structured spatial memory, map-backed PixelGoal, four-view RGB, and LiDAR

<a id="installation"></a>

## 🛠️ Installation

The canonical SuperNav-1.5 release setup uses **Habitat-GS with Structured
Spatial Memory**. Follow the
[complete spatial-memory installation and run guide](docs/release/en/spatial-memory.md)
for prerequisites, GPU configuration, external caption services, and
troubleshooting.

The installation uses two environments:

| Environment | Purpose |
| --- | --- |
| `supernav-cartographer` | Builds the pinned Cartographer source and provides its native libraries |
| Your Habitat-GS Python 3.12 environment | Runs `supernav stack`, the Habitat bridge, spatial memory, and the managed caption service |

Keep both environments after installation. Start from the project directory:

```bash
git clone https://github.com/zju3dv/SuperNav.git
cd SuperNav/SuperNav_v1_5
```

First build the patched Cartographer worker in its dedicated Conda environment:

```bash
git submodule update --init third_party/cartographer
conda env create -f environment-cartographer.yml
conda activate supernav-cartographer
SUPERNAV_CARTOGRAPHER_BUILD_JOBS=4 scripts/build_cartographer_worker.sh
conda deactivate
```

Next, activate the Python 3.12 environment in which Habitat-GS already works
and install the complete SuperNav-1.5 runtime:

```bash
conda activate <habitat-gs-runtime-environment>
python -m pip install --upgrade pip
python -m pip install -e '.[agents,spatial-memory,caption]'
```

If Habitat-GS is not installed yet, complete the
[Habitat-GS setup](docs/release/en/habitat.md#install-and-select-the-simulator)
before this runtime installation. Install and authenticate the selected agent
CLI separately. The bundled Codex recipe uses `gpt-6-astra` with medium
reasoning; provider and credential settings are documented in the
[configuration guide](docs/release/en/configuration.md).

Other workflows are documented separately:

| Workflow | Guide |
| --- | --- |
| Original SuperNav baseline experiments | [SuperNav v1](../SuperNav_v1/README.md#installation) |
| AI2-THOR | [AI2-THOR setup](docs/release/en/ai2thor.md) |
| Unitree Go2 | [Go2 deployment](docs/release/en/go2.md) |
| Live and recorded visualization | [Web viewer](docs/release/en/web-viewer.md) |

<a id="quick-start"></a>
<a id="run-navigation-experiments"></a>

## 🧠 Run with Structured Spatial Memory

SuperNav-1.5 uses the same task manifests, task selection, repetitions, scoring,
and evaluation workflow as
[SuperNav v1](../SuperNav_v1/README.md#run-navigation-experiments). Structured
spatial memory can be evaluated in two execution modes:

| Execution mode | Purpose | Entry point |
| --- | --- | --- |
| **Single-session benchmark** | Run the usual Habitat benchmark suites—including single-object, multi-object, HM3D ObjectNav, and HM3D-OVON—with spatial memory built within each episode. Each episode starts with empty memory and a new SLAM map. | `supernav stack run` |
| **Paired-session benchmark** | Evaluate memory reuse across sessions. An exploration session builds memory. A retrieval session then starts from an immutable snapshot of that memory and its SLAM state, optionally from a different pose. | `supernav stack run-paired` |


The execution mode is independent of the spatial-memory recipe:

| Experiment recipe | Memory configuration |
| --- | --- |
| `habitat-spatial-memory` | Memory nodes, Cartographer 2D SLAM, occupancy navigation, and Qwen area captions |
| `habitat-spatial-memory-slam-only` | The same memory-node and SLAM pipeline with area captioning disabled |

### Common setup

First complete the
[native Cartographer build and runtime installation](docs/release/en/spatial-memory.md#install-and-build)
and the [Habitat-GS setup](docs/release/en/habitat.md). Prepare the task manifest
for the selected benchmark using the
[same data preparation workflow as SuperNav v1](../SuperNav_v1/README.md#run-navigation-experiments).
Then configure the runtime paths for your machine:

```bash
export SUPERNAV_HABITAT_ROOT=/absolute/path/to/habitat-gs
export SUPERNAV_HABITAT_PYTHON=/absolute/path/to/habitat-environment/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/absolute/path/to/global_navigation.scene_dataset_config.json
export HAB_GPU_DEVICE_ID=0
export HAB_CARTOGRAPHER_WORKER_COMMAND="scripts/run_cartographer_worker.sh {input_fifo} {occupancy_path}"
```

`SUPERNAV_HABITAT_PYTHON` is also the default interpreter for the managed Qwen
caption service and the Open3D relocalizer, so Torch, Open3D, and the
`spatial-memory`/`caption` extras must be installed in that environment. To use
an existing local Qwen checkout instead of downloading the public model ID, set
`SLAM_AREA_CAPTION_MODEL=/absolute/path/to/Qwen3.5-2B`. The explicit
`SUPERNAV_CAPTION_PYTHON` and `HAB_CARTOGRAPHER_RELOCALIZER_COMMAND` variables
remain available as advanced overrides.

Inspect either recipe before running it:

```bash
supernav config show --experiment habitat-spatial-memory
supernav config show --experiment habitat-spatial-memory-slam-only
```

The full recipe is used below. Substitute
`habitat-spatial-memory-slam-only` in either mode when captions are not needed.

### Mode 1: single-session benchmarks

Use `stack run` for ordinary single-session benchmark manifests. The same entry
point applies whether the selected manifest contains Habitat-GS single-object,
multi-object, HM3D ObjectNav, or HM3D-OVON tasks:

```bash
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory
```

Replace `example-task` with a `task_id` from the manifest. The recipe already
selects `gpt-6-astra` with medium reasoning; normal runner options such as
`--model`, `--reps`, `--run-tag`, and `--dry-run` are forwarded unchanged.
The stack starts the required support service, delegates to the normal
experiment runner, and starts and stops the Habitat bridge for each episode.

Use the SuperNav v1 guides to prepare each benchmark's data and task manifest;
replace the v1 `supernav run` invocation with the `supernav stack run` command
above so the spatial-memory services are available:

| Benchmark workflow | Preparation and scoring guide |
| --- | --- |
| Habitat-GS single-object and multi-object | [HabitatGS-Benchmark](../SuperNav_v1/docs/release/en/habitatgs-benchmark.md) |
| HM3D ObjectNav and HM3D-OVON | [HM3D v2 and OVON](../SuperNav_v1/docs/release/en/hm3d-and-ovon.md) |
| Task selection, repetitions, arms, run tags, and output directories | [Run navigation experiments](../SuperNav_v1/README.md#run-navigation-experiments) |

Every single-session episode starts with fresh memory. The spatial-memory recipe
disables SuperNav v1's bounded spatial-fact ledger, so the agent sees one
spatial-memory system rather than two competing memories.

### Mode 2: paired-session benchmark

Use `stack run-paired` only when the benchmark condition is intended to measure
reuse of memory from a prior session:

```bash
supernav stack run-paired \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory-paired
```

`run-paired` performs two runs of the selected task row:

1. **Exploration** starts at `spawn` and writes memory nodes plus visual and
   planning Cartographer state.
2. The runner creates an immutable, hash-audited snapshot containing
   `memory_nodes.json`, both `.pbstream` files, the final visual occupancy
   raster with its AMCL/render/node sidecars, and available caption evidence.
3. **Retrieval** starts at the row's optional `retrieval_spawn` (falling back to
   `spawn`), receives a separate writable copy of the memory nodes, and loads
   both frozen maps in `delayed_fpfh_relocalization` mode. Agent calls to
   `hab_report_memory_pos` provide node priors to the source-faithful
   relocalizer while background accumulated-scan matching remains active.

Before retrieval has a verified localized pose, map-bearing tool responses show
the immutable prior-session map with `2d_slam_map.status="static"` and no pose
marker. After relocalization and a synchronized render, they switch to the
current map and pose marker. `hab_navigate_to_memory_node_occ` first returns a
map-only route preview and confirmation token; execution then returns the
latest front/right/back/left RGB views plus the final annotated occupancy map.
On collision, the executor does not assume the requested forward primitive was
completed. Habitat may still apply a partial sliding displacement when sliding
is enabled, so the executor waits for the actual post-collision Cartographer
pose before replanning and returns `blocked` when its configured replan budget
is exhausted.

The instruction, policy templates, model, arm, and native Skill snapshot are
shared by the two sessions; only the benchmark-owned start pose and persisted
runtime state differ. A task row may specify the two starts directly:

```json
{
  "task_id": "example-task",
  "slug": "example-task",
  "text": "Find the requested object.",
  "scene": "interior_0405_840145",
  "spawn": {"start_position": [0.0, 0.2, 0.0], "start_rotation": [0, 0, 0, 1]},
  "retrieval_spawn": {"start_position": [3.0, 0.2, -2.0], "start_rotation": [0, 0.707, 0, 0.707]}
}
```

The stack preflights the patched worker, starts and health-checks the caption
service, and starts a fresh Habitat bridge and Cartographer worker for each
session. It stops only the caption process it owns.

The recipe binds a dedicated spatial-memory prompt and jointly requires the
`memory-nodes`, `2d-slam`, and `spatial-memory-navigation` Skills. Visible local
targets use immediate one-call visual-point movement; off-screen memory nodes
and current SLAM-area frontiers use preview/confirmation occupancy routing.

Pair state is written below `<output-dir>/_pairs/<pair-id>/`. `pair.json`
records both run IDs and status; `snapshot/manifest.json` records artifact
hashes; normal episode artifacts live below `episodes/`. To retry retrieval
without rerunning a valid exploration snapshot:

```bash
supernav stack run-paired \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task --arms default --resume \
  --output-dir data/runs/habitat-spatial-memory-paired
```

Resume fails closed if an immutable snapshot artifact is absent or its hash has
changed. Use `--overwrite` to archive the prior pair under a timestamped stale
directory and start a new pair. `--retrieve-spawn '{...}'` can override the
retrieval start when exactly one task is selected.

### Reusable configuration and services

For either execution mode, a reusable machine-local configuration can keep the
manifest and output paths out of the command. Create the ignored file
`configs/local/habitat-spatial-memory.json`:

```json
{
  "extends": "../experiments/habitat-spatial-memory.json",
  "instructions_file": "/absolute/path/to/tasks/habitat.json",
  "output_dir": "data/runs/habitat-spatial-memory"
}
```

```bash
supernav config show --config configs/local/habitat-spatial-memory.json
supernav stack run \
  --config configs/local/habitat-spatial-memory.json \
  --task-ids example-task --arms default
```

Use the same configuration with `run-paired` for the paired-session condition:

```bash
supernav stack run-paired \
  --config configs/local/habitat-spatial-memory.json \
  --task-ids example-task --arms default
```

By default, either entry point starts the required support service and cleans up
the service process it owns after the benchmark finishes. For repeated
experiments, retain the caption service across both single-session and paired
runs:

```bash
supernav stack up --experiment habitat-spatial-memory
supernav stack status
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default
# Additional `stack run` or `stack run-paired` commands can reuse the service.
supernav stack logs --follow
supernav stack down
```

`stack up` starts the long-lived caption service, not the Habitat bridge.
`stack run` starts a bridge for each ordinary episode; `stack run-paired` starts
a fresh bridge for each of its exploration and retrieval sessions. See the
[spatial-memory guide](docs/release/en/spatial-memory.md#run-with-the-service-stack)
for external caption services, service ownership, logs, and shutdown behavior.

<a id="visualization-and-evaluation"></a>

## 🔁 Shared SuperNav v1 Workflows

Yes—the experiment lifecycle after service startup is the same as SuperNav v1.
Use the v1 documentation for actions that SuperNav-1.5 does not change:

| Action | Guidance |
| --- | --- |
| Task manifests, task selection, repetitions, arms, run tags, and output directories | [Run navigation experiments in SuperNav v1](../SuperNav_v1/README.md#run-navigation-experiments) |
| HM3D ObjectNav and OVON data/task preparation | [Run an HM3D ObjectNav episode in SuperNav v1](../SuperNav_v1/README.md#run-hm3d-objectnav) |
| Live viewing, saved artifacts, scoring, aggregation, and video export | [Visualization and evaluation in SuperNav v1](../SuperNav_v1/README.md#visualization-and-evaluation) |
| Baseline SuperNav experiments and AI2-THOR | [SuperNav v1 README](../SuperNav_v1/README.md) |

The spatial-memory recipes currently target the Habitat backend. For a baseline
v1 recipe such as `habitat-geo-based-executor`, continue to use
`supernav run`; `supernav stack run` is needed only when the selected recipe
requires the SuperNav-1.5 service stack.

<a id="documentation-and-development"></a>

## 📚 Documentation and Development

Deployment and simulation guides are available in [English](docs/release/en/README.md)
and [简体中文](docs/release/zh/README.md).

| Guide | Contents |
| --- | --- |
| [Configuration](docs/release/en/configuration.md) | Experiments, models, local settings, and task manifests |
| [Scene assets and Skills](docs/release/en/assets-and-skills.md) | Backend requirements, external assets, and navigation Skills |
| [Habitat-GS](docs/release/en/habitat.md) | Simulator installation and scene integration |
| [HabitatGS-Benchmark](docs/release/en/habitatgs-benchmark.md) | Public GS dataset conversion, navigation, SR/SPL and stage SR |
| [HM3D v2 and OVON](docs/release/en/hm3d-and-ovon.md) | Scene and episode data, task preparation, navigation, and offline scoring |
| [AI2-THOR](docs/release/en/ai2thor.md) | Dataset, Unity build, observations, and execution |
| [Unitree Go2](docs/release/en/go2.md) | Jetson installation, spatial-memory operation, paired sessions, source maintenance, and tests |
| [Web viewer](docs/release/en/web-viewer.md) | Live observation and recorded runs |
| [Architecture](docs/release/en/architecture.md) | Runtime, methods, backends, experiments, and evaluation |
| [Directory layout](docs/release/en/layout.md) | Source, configuration, asset, and output locations |

For development:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

CI builds a wheel and checks its CLI, configuration, prompt, and Skill resources
in a clean environment with only the base dependencies.

<a id="citation"></a>

## 📄 Citation

Please cite SuperNav-1.5 as follows:

```bibtex
@misc{guo2026supernav15structuredspatialmemory,
  title={SuperNav-1.5: Structured Spatial Memory For Agentic Navigation Systems},
  author={Jiarui Guo and Jinkai Zhang and Yuanhong Yu and Yongchen Zai and Futang Zhu and Ruizhen Hu and Sida Peng and Xiaowei Zhou},
  year={2026},
  eprint={2611.29912},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2611.29912},
}
```

<a id="acknowledgements-and-license"></a>

## 🙏 Acknowledgements and License

SuperNav uses [Habitat-GS](https://github.com/zju3dv/habitat-gs),
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim), and
[AI2-THOR](https://github.com/allenai/ai2thor) as simulator foundations. We thank
their authors and the contributors to the datasets and tools used in this project.

SuperNav is distributed under the [Project Registration License (PRL) v1.0](LICENSE).
Use solely for academic research, education, evaluation, or personal purposes is
free and does not require registration, including publication of research results.

Organizational project use is free but requires registration before use, including
commercial and non-commercial projects, internal R&D, testing, deployment, and
production. Complete and accurate registration automatically grants permission;
no approval or fee is required. Register each materially distinct project once.

**Project use registration:** [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSff5QjjANlJ2hnXYqBSAzKWxEM58gIxPLh17-t3j3-MqO1a5w/viewform).

Third-party code and bundled fonts retain their original licenses and copyright
notices. See [Third-party notices](THIRD_PARTY_NOTICES.md). External simulators,
datasets, and model weights retain their respective licenses.
