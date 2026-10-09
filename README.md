<div align="center">

<h1>
  <img src="docs/release/assets/supernav-wordmark.svg" alt="SuperNav" width="460"><br>
  <sub>An Agentic Navigation System for Any Task in Any Scene</sub>
</h1>

<p>
  <a href="https://arxiv.org/abs/2610.12126v1"><img src="https://img.shields.io/badge/arXiv-2610.12126-b31b1b" alt="arXiv: 2610.12126"></a>
  <a href="https://huggingface.co/papers/2610.12126"><img src="https://img.shields.io/badge/Hugging_Face-Paper-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Paper"></a>
  <a href="https://zju3dv.github.io/SuperNav/"><img src="https://img.shields.io/badge/Project_Page-SuperNav-orange" alt="Project Page"></a>
</p>

<p><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a></p>

<p>
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://echo636.github.io/">Jingyi Xu</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Jiarui Guo<sup>1</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>2</sup> ·<br>
  <a href="https://person.zju.edu.cn/0093140">Hujun Bao</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup> ·
  <a href="https://pengsida.net/">Sida Peng</a><sup>1,3†</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;&nbsp; <sup>2</sup> Shenzhen University &nbsp;&nbsp; <sup>3</sup> Causa Robotics<br>
  <sup>†</sup> Corresponding author
</p>

<h3>Overview Video</h3>

<video src="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d" controls width="800"></video>

<p>
  <a href="https://zju3dv.github.io/SuperNav/">Watch on the project page</a> ·
  <a href="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d">Open MP4</a>
</p>

</div>

## 🧭 What Is SuperNav?

**SuperNav** is an agentic navigation system that uses a pretrained multimodal
model to solve diverse navigation tasks through tool use, task-progress tracking,
and context management, without navigation-specific fine-tuning of the model.
Tasks range from finding a single object and visiting multiple targets to
fulfilling high-level requests.

This repository provides the agent runtime, navigation tools and Skills,
experiment recipes, and evaluation utilities for **Habitat-GS** and **AI2-THOR**.
With SuperNav, you can:

- **Run navigation agents** with Codex, OpenCode, or Kimi.
- **Compose tasks, tools, and Skills** through explicit experiment configurations.
- **Watch execution live**, including RGB observations, trajectories, and tool calls.
- **Inspect and evaluate runs** using saved prompts, tool results, native logs,
  and independent task metrics.

See the [project page](https://zju3dv.github.io/SuperNav/) for simulation
videos, the method overview, and research results.
Real-world robot demonstrations are on the project page; this repository currently
contains no real-robot drivers.

## 📖 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Run Navigation Experiments](#run-navigation-experiments)
- [Visualization and Evaluation](#visualization-and-evaluation)
- [Documentation and Development](#documentation-and-development)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgements and License](#acknowledgements-and-license)

<a id="installation"></a>

## 🛠️ Installation

Use **Python 3.12**. Start with an independent environment for SuperNav:

```bash
git clone https://github.com/zju3dv/SuperNav.git
cd SuperNav

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base installation supports configuration inspection, Habitat dry-runs, and
the browser demo. Install the extras needed for your workflow:

| Workflow | Install command |
| --- | --- |
| Agents and evaluation | `python -m pip install -e '.[agents,evaluation]'` |
| AI2-THOR experiments | `python -m pip install -e '.[ai2thor,agents,evaluation]'` |
| Rerun viewer | `python -m pip install -e '.[viewer]'` |

For Codex, OpenCode, or Kimi runs, install and authenticate the selected CLI
separately. Shared Codex recipes use the client's configured provider and login;
all bundled recipes use `gpt-6-astra` with `medium` reasoning from
[`configs/models/gpt-6-astra-medium.json`](configs/models/gpt-6-astra-medium.json).
Keep credentials in environment variables or local client configuration.

API clients select a provider through `agents.<client>.provider` in a local
experiment configuration. Credentials use the environment variable named by
`env_key`; see the [configuration guide](docs/release/en/configuration.md).

For Habitat, use the [Habitat setup guide](docs/release/en/habitat.md#install-in-a-dedicated-venv)
to install the public CUDA-enabled SDK in a dedicated environment.
A bridge health response confirms server startup; RGB/depth, motion, session close,
and independent GT scoring require a real episode.

<a id="quick-start"></a>

## ⚡ Quick Start

Start the built-in browser demo:

```bash
supernav web --demo
```

Open **http://127.0.0.1:8765** to explore a generated room and sample trajectory.
The demo is included in the base installation. Press **Ctrl+C** to stop it before
continuing, or use another terminal.

**Before running an experiment, prepare these three inputs yourself:**

- **Data:** scene assets and datasets for the selected simulator.
- **Configuration:** local data paths, simulator environment, model access, and
  any services required by the selected method. Adapt the example recipes in
  `configs/experiments/` to your setup.
- **Instructions:** a task manifest containing your requests and matching scene,
  spawn, and goal information.

SuperNav provides the code, general prompts, and configuration templates. Supply
matching ground truth when required for scoring or by the selected method.
The paper's task lists and ground truth are excluded from this release.

Inspect the example experiment configurations:

```bash
supernav --help
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

After preparing your configuration and task manifest, replace `example-task`
below with a `task_id` from your manifest, then prepare a dry-run:

```bash
supernav run --experiment habitat-geo-based-executor \
  --instructions /path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default --dry-run \
  --output-dir data/runs/quickstart-dry
```

`supernav` and `python -m supernav` expose the same command tree. Runs require
an explicit `--experiment` or `--config` selection.

<a id="run-navigation-experiments"></a>

## 🚀 Run Navigation Experiments

### Run a Habitat-GS episode

Install the agent dependencies, then follow the [Habitat setup guide](docs/release/en/habitat.md)
to install Habitat-GS and the same SuperNav version in the simulator environment.
The agent and simulator may use separate Python environments.

```bash
python -m pip install -e '.[agents,evaluation]'

export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/global_navigation.scene_dataset_config.json

supernav run --experiment habitat-geo-based-executor \
  --instructions /path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default --model gpt-6-astra \
  --output-dir data/runs/quickstart
```

Replace the paths with your installation. The recipe supplies `medium` reasoning
for `gpt-6-astra`. The task manifest supplies the scene and spawn. The selected
method uses visual-point navigation with native global and local navigation Skills.
SuperNav starts and stops the Habitat bridge for each episode.

**Use separate output directories for dry-runs and real runs.** Existing episode
directories are skipped by default; choose a new directory or `--run-tag` for
another run.

### Choose an experiment

Example recipes in [`configs/experiments`](configs/experiments) combine reusable
agent, model, backend, and method settings. Adapt a recipe to your setup and
supply your own data and instructions:

| Experiment | Purpose |
| --- | --- |
| `habitat-geo-based-executor` | Habitat navigation with a geometry-based executor and native Skills |
| `habitat-learned-executor` | Habitat navigation with a learned local policy and native Skills; weights not yet released |
| `ai2thor-primitive` | AI2-THOR demand-driven navigation with primitive actions |

**Learned Executor weights are not yet released, and no download is available.**
The `habitat-learned-executor` recipe requires compatible trained weights and a
running policy service; installing SuperNav alone does not make it ready to run.
If you already have compatible weights, see the
[policy service requirements](docs/release/en/habitat.md#learned-executor-weights).
Otherwise, use `habitat-geo-based-executor` while the weight release is pending.
Download and setup instructions will be added here when the weights are released.

Each recipe has one `default` arm; omit `--arms` or use `--arms default`.
Use `--task-ids` and `--reps` to select tasks and repetitions. The geometry-based
executor uses oracle geometry, including depth and NavMesh paths; account for
this information when comparing methods.

Keep machine-specific settings in ignored `configs/local/` files. For example,
create `configs/local/habitat.json` to select your task manifest and customize a recipe:

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "model": {"name": "gpt-6-astra"},
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}"
  },
  "output_dir": "data/runs/habitat"
}
```

```bash
supernav config show --config configs/local/habitat.json
supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default
```

See the [configuration guide](docs/release/en/configuration.md) for composition rules, model
selection, and external dataset path mappings. Each recipe declares its prompt
directory explicitly.

### Run an AI2-THOR episode

After preparing the dataset and Unity build described in the
[AI2-THOR guide](docs/release/en/ai2thor.md):

```bash
python -m pip install -e '.[ai2thor,agents,evaluation]'
export SUPERNAV_AI2THOR_DATASET=/path/to/frozen-demand-driven-dataset
export SUPERNAV_AI2THOR_RELEASES=/path/to/ai2thor/releases

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task \
  --model gpt-6-astra \
  --output-dir data/runs/ai2thor-example
```

Use a task ID and matching request text from your external dataset.
This backend records actions, resource claims, and STOP events. Its current
evaluation mode is `withheld`: SR and SPL remain `null`.

<a id="visualization-and-evaluation"></a>

## 🖥️ Visualization and Evaluation

To watch a real run, set the same absolute live directory in both the viewer
terminal and the experiment terminal **before** starting either process:

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
```

Start the viewer in one terminal, then run an experiment in the other:

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

The viewer shows observations, trajectories, and recorded agent/tool activity.
See the [viewer guide](docs/release/en/web-viewer.md) for remote access and existing recordings.

Runs save configuration and task snapshots, prompts, raw agent logs, tool events,
and metrics under `data/runs/` or the selected output directory. Inspect the
evaluation and export commands with:

```bash
supernav score-objectnav --help
supernav score-multi-objectnav --help
supernav aggregate --help
supernav video --help
```

Compute SR/SPL with the task's offline scorer and ground truth. Agent completion
claims, STOP events, and process status are recorded separately.

<a id="documentation-and-development"></a>

## 📚 Documentation and Development

All release guides are available in [English](docs/release/en/README.md) and
[简体中文](docs/release/zh/README.md).

| Guide | Contents |
| --- | --- |
| [Configuration](docs/release/en/configuration.md) | Experiments, models, local settings, and task manifests |
| [Scene assets and Skills](docs/release/en/assets-and-skills.md) | Backend requirements, external assets, and navigation Skills |
| [Habitat-GS](docs/release/en/habitat.md) | Simulator installation and scene integration |
| [AI2-THOR](docs/release/en/ai2thor.md) | Dataset, Unity build, observations, and execution |
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

<a id="roadmap"></a>

## 🗺️ Roadmap

Parts of the SuperNav work that are not yet released in this repository:

- **Learned Executor weights:** trained checkpoints and their download and setup
  instructions for `habitat-learned-executor`.
- **Real-robot deployment:** the Unitree Go2 deployment used in the paper
  (four-view RGB with LiDAR-based geometric execution).
- **Benchmark task data:** the InteriorGS single- and multi-object instance
  navigation tasks and the AI2-THOR demand-driven tasks used in the paper.

<a id="citation"></a>

## 📄 Citation

If you find SuperNav useful in your research, please cite our
[paper](https://arxiv.org/abs/2610.12126v1):

```bibtex
@misc{zhang2026supernav,
  title={{SuperNav}: An Agentic Navigation System for Any Task in Any Scene},
  author={Jinkai Zhang and Jingyi Xu and Yuanhong Yu and Jiarui Guo and Ruizhen Hu and Hujun Bao and Xiaowei Zhou and Sida Peng},
  year={2026},
  eprint={2610.12126},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.12126v1}
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
notices; see [Third-party notices](THIRD_PARTY_NOTICES.md). External simulators,
datasets, and model weights retain their respective licenses.
