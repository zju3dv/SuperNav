[English](habitat.md) | [简体中文](../zh/habitat.md)

# Habitat-GS backend

SuperNav uses the `habitat_sim` SDK from
[Habitat-GS](https://github.com/zju3dv/habitat-gs) to render scenes and execute
navigation actions. SuperNav manages agents, navigation tools, experiments, and
evaluation.

Before running an experiment, prepare:

1. Scene datasets and assets, including NavMeshes, and a compatible Habitat-GS environment.
2. A configuration adapted to your machine: model, credentials, paths, and services.
3. Task instructions or a manifest with the matching scenes, spawn poses, and goals.

Provide GT when required by the selected method or scoring procedure. Shared
recipes are example templates: adapt them through environment settings and CLI
options with `--experiment`, or use a local overlay with `--config`.

Run the commands below from the SuperNav repository root. See the
[documentation index](README.md) and [configuration guide](configuration.md)
for shared commands and settings.

For API clients, configure provider fields and credential environment variables
as described in the [configuration guide](configuration.md). Shared Codex
recipes use the client configuration and login.

## Install and select the simulator

Use Python 3.12 for SuperNav. Follow the upstream
[installation instructions](https://github.com/zju3dv/habitat-gs#-install-habitat-gs)
to install Habitat-GS and its native/CUDA dependencies and prepare scene assets.

### Install in a dedicated venv

The upstream example creates a conda environment. For a venv deployment, use
Python 3.12 and a CUDA toolkit with `nvcc` available on `PATH`, then install the
same CUDA-enabled SDK from the public repository:

```bash
python3.12 -m venv .cache/venv-habitat
export SUPERNAV_HABITAT_PYTHON="$PWD/.cache/venv-habitat/bin/python"
"$SUPERNAV_HABITAT_PYTHON" -m pip install --upgrade pip
"$SUPERNAV_HABITAT_PYTHON" -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
"$SUPERNAV_HABITAT_PYTHON" -m pip install --upgrade cmake ninja setuptools wheel

# chumpy 0.70 imports pip during setup; prepare it in the target environment.
"$SUPERNAV_HABITAT_PYTHON" -m pip install --no-build-isolation chumpy

git clone --recursive https://github.com/zju3dv/habitat-gs.git .cache/habitat-gs
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF CMAKE_BUILD_PARALLEL_LEVEL=2 \
  "$SUPERNAV_HABITAT_PYTHON" -m pip install .cache/habitat-gs
"$SUPERNAV_HABITAT_PYTHON" -m pip install '.[agents,evaluation]'
"$SUPERNAV_HABITAT_PYTHON" -m pip check
"$SUPERNAV_HABITAT_PYTHON" -c 'import habitat_sim, supernav; print(habitat_sim.__file__, supernav.__file__)'
```

A standard isolated build of `chumpy` 0.70 can fail with
`ModuleNotFoundError: No module named 'pip'`. The preinstallation above supplies
its build requirements in the venv; the Habitat-GS checkout stays unchanged.
`CMAKE_BUILD_PARALLEL_LEVEL` limits CPU compilation concurrency. Match the
PyTorch CUDA wheel and native CUDA toolkit to the upstream requirements.

The agent and simulator can use separate Python environments. Install the same
SuperNav version in both:

```bash
python -m pip install -e '.[agents,evaluation]'
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-gs-env/bin/python
"$SUPERNAV_HABITAT_PYTHON" -m pip install --no-deps .
```

Use `--no-deps` when the simulator environment already has the required dependencies.
Otherwise, install them first. You can also install the same built SuperNav wheel
in both environments.

Install and authenticate your agent CLI. Shared Codex recipes use the client's
provider and login, with `gpt-6-astra` and `medium` reasoning from
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json).

`environment.python` / `SUPERNAV_HABITAT_PYTHON` selects the bridge interpreter.
By default, SuperNav uses the SDK installed in that environment. To check bridge
startup manually:

```bash
unset SUPERNAV_HABITAT_ROOT
"$SUPERNAV_HABITAT_PYTHON" -B -m supernav habitat-bridge \
  --host 127.0.0.1 --port 18911
```

Press Ctrl+C after the check. Experiment runs start and stop their own bridge.

To use a specific source checkout, set
`SUPERNAV_HABITAT_ROOT=/path/to/habitat-gs` or `environment.habitat_root` in a local
configuration. Install native extensions compatible with that checkout's Python SDK
in the selected simulator environment.

## Prepare matching scenes

Prepare the scene dataset configuration, scene assets, NavMeshes, semantic
information, and spawn poses required by your selected tasks. Match scene IDs to
the task definitions. For GS assets, see the upstream
[asset instructions](https://github.com/zju3dv/habitat-gs#-download-gs-asset).

In each scene instance, use `stage_instance.template_name` to reference its stage.
Save local scene configuration copies under `configs/local/`, keeping the stage,
NavMesh, spawn, and ground truth aligned with the task. Check rendered RGB at the
task's initial pose.

## Run one episode

Supply an external task manifest with instructions, scene and spawn settings,
and goals. Add GT as required by your method or scorer. This example uses the
`habitat-geo-based-executor` recipe with native global and local navigation Skills.
It uses oracle depth and NavMesh geometry to execute visual-point actions.
Create the ignored `configs/local/` directory and save this configuration as
`configs/local/habitat.json`:

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}"
  },
  "output_dir": "data/runs/habitat"
}
```

```bash
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/global_navigation.scene_dataset_config.json

supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --dry-run --output-dir data/runs/habitat-dry

supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --model gpt-6-astra --output-dir data/runs/habitat
```

Replace the paths with your installation and `example-task` with an ID from your
manifest. The recipe supplies `medium` reasoning for `gpt-6-astra`. Use `--dry-run` for a configuration-only check: it writes the
prompt, episode configuration, and native Skill snapshots.
Use separate output directories for dry-runs and real runs. Existing episode
directories are skipped by default; choose a new directory or `--run-tag` to run again.

The Habitat backend starts and stops a bridge for each episode, with a 120-second
startup timeout. Reserve an available port for the runner through `bridge.port`
or `--bridge-port`.

Both Habitat examples have one `default` arm; omit `--arms` or use `--arms default`.

External OVON tasks require matching HM3D/OVON assets. Use
`deployment.scene_dataset_config_file` and `deployment.path_prefixes` to map
external task paths to your local data; see the [configuration guide](configuration.md).
For dataset preparation, task conversion, navigation, and scoring, follow the
[HM3D v2 and OVON guide](hm3d-and-ovon.md).

<a id="learned-executor-weights"></a>

## Learned Executor weights and policy service

The released checkpoint is hosted at
[the0xka1/SuperNav-Learned-Executor](https://huggingface.co/the0xka1/SuperNav-Learned-Executor).
It contains the learned local policy; the high-level agent, Habitat-GS, scene
assets, and task manifests remain separate requirements.

```bash
python -m pip install -U huggingface_hub
hf download the0xka1/SuperNav-Learned-Executor \
  ckpt_latest125.pt CHECKSUMS.sha256 --local-dir /path/to/checkpoints
(cd /path/to/checkpoints && sha256sum -c CHECKSUMS.sha256)
```

In the policy-service environment, use Python 3.12 and install compatible
PyTorch and torchvision builds for your device, plus NumPy and Pillow. Install
SuperNav with `python -m pip install -e '.[agents,evaluation]'` from the repository
root. SuperNav's `nomad` backend reads the architecture settings from the checkpoint.

For CUDA 12.1, this example installs PyTorch `2.5.1` and torchvision
`0.20.1` from the
[official PyTorch wheels](https://pytorch.org/get-started/previous-versions/#v251):

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Start the service with the downloaded checkpoint's absolute path:

```bash
export TORCH_HOME=/path/to/torch-cache
export NAV_LOCALNAV_CKPT=/path/to/checkpoints/ckpt_latest125.pt
python -m supernav.methods.localnav.server \
  --backend nomad --checkpoint "$NAV_LOCALNAV_CKPT" \
  --host 127.0.0.1 --port 18914 --device cuda
```

Replace the placeholder paths with writable directories on your machine. On
first startup, the DINOv2 loader downloads its upstream code and initial backbone
weights through Torch Hub, so allow access to GitHub and `dl.fbaipublicfiles.com`,
or prepare the cache under `TORCH_HOME` in advance. xFormers is optional for this
inference path. Use `--device cpu` for CPU inference.

Keep the service running. In the terminal used to launch the agent, configure
the URL and check the service:

```bash
export NAV_LOCALNAV_URL=http://127.0.0.1:18914
curl --fail "$NAV_LOCALNAV_URL/healthz"
```

The service configuration can also be supplied through environment variables:

| Setting | Requirement |
| --- | --- |
| `NAV_LOCALNAV_BACKEND` | Set to `nomad` in the policy service environment to load the learned backend |
| `NAV_LOCALNAV_CKPT` | Absolute checkpoint path on the policy service host |
| `NAV_LOCALNAV_URL` | Policy service URL reachable from the Habitat bridge; defaults to `http://127.0.0.1:18914` |

The service's `GET /healthz` response reports its `backend` and `ckpt`; verify
that it reports `nomad` and the downloaded checkpoint before running an experiment.
If the Habitat bridge runs on another host, set `NAV_LOCALNAV_URL` to a reachable
service address and adjust the service's `--host` binding. When using an HTTP
proxy locally, include `127.0.0.1,localhost` in `NO_PROXY` and `no_proxy`.
The service defaults to the `debug` backend, which exercises the protocol only.
Starting it without learned weights does not provide a trained executor.
Extend `habitat-learned-executor.json` in your local configuration and run a task
with your scene assets and external manifest, following the episode commands
above. `--dry-run` only prepares run inputs and does not verify policy loading
or task execution.

## Navigation settings

- Use `benchmark_profile` to select the task protocol. The backend sets
  `HAB_MCP_GLOBAL_TASK` internally.
- Set `HAB_GPU_DEVICE_ID` to select the simulator GPU. `HAB_ALLOW_SLIDING=0`
  disables collision sliding; a session's explicit `allow_sliding` takes precedence.
- GS uses the supplied NavMesh, with automatic NavMesh rebuilding and physics
  disabled by default.
- For NoMaD or LocateAnything methods, prepare the policy or grounding service
  and its required weights.
- Tool calls use the `habitat-gs/v1` protocol. Offline evaluation computes SR/SPL
  from task ground truth; agent completion claims are recorded separately.

Use the [Web Viewer guide](web-viewer.md) to observe runs and the
[tool authoring guide](tool-authoring-guide.md) to extend navigation tools.

## Development checks

Install test dependencies and run the backend checks:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_public_habitat_backend.py
```

To validate a package, install the built wheel and its base dependencies in a
clean virtual environment outside the checkout. Run
[`scripts/check_installed_package.py`](../../../scripts/check_installed_package.py)
with that environment's interpreter to check the CLI and packaged resources
without simulator SDKs.

Record the Python SDK revision and native extension build when validating a
deployment. See [architecture](architecture.md) for module responsibilities.
