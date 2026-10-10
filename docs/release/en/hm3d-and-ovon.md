[English](hm3d-and-ovon.md) | [简体中文](../zh/hm3d-and-ovon.md)

# Run HM3D v2 ObjectNav and HM3D-OVON

This guide prepares external episodes, runs SuperNav with the geometry-based
executor, and scores saved trajectories. Complete the [Habitat-GS setup](habitat.md)
and [agent configuration](configuration.md#models-and-credentials) first. Run the
commands from the SuperNav repository root, with `supernav` and `python` available
in the agent environment. Install the same SuperNav version in the simulator
environment.

HM3D mesh scenes load through the same Habitat-GS `habitat_sim` SDK used by
SuperNav's Habitat backend.

## Choose the data

Scenes and navigation episodes are separate downloads:

| Input | Contents | Source |
| --- | --- | --- |
| HM3D/HM3DSem **v0.2** scenes | Renderable scenes, scene-instance configurations, semantic assets, and NavMeshes | [Matterport HM3D downloads](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02) |
| `objectnav_hm3d_v2` episodes | ObjectNav tasks that use HM3DSem v0.2 scenes | [Habitat-Lab task datasets](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets) |
| HM3D-OVON episodes | Open-vocabulary tasks using HM3D scenes | [OVON author instructions](https://github.com/naokiyokoyama/ovon#-downloading-the-datasets) and [episode archive](https://huggingface.co/datasets/nyokoyama/hm3d_ovon/tree/main) |
| OVON selection manifest | The external `our-set/ovon_full_set.json` selection used by the current converter | [MTU3D data instructions](https://github.com/MTU3D/MTU3D#prepare-data) and [benchmark archive](https://huggingface.co/datasets/bigai/MTU3D/tree/main) |

Here, “HM3D v2” means the **ObjectNav v2 episode dataset paired with v0.2
scenes**.

`supernav prepare objectnav` reads the per-scene ObjectNav split.
`supernav prepare ovon` reads OVON episodes selected by an external MTU3D-format
manifest; the example below uses its `val_unseen` selection.

## Prepare storage and assets

Choose absolute paths on a data volume with enough space. Use a new work directory
for each experiment and keep task manifests, goals, and generated evidence outside
the public source tree. Existing installations can use their own paths.

```bash
export SUPERNAV_DATA_ROOT=/path/to/storage/datasets
export SUPERNAV_WORK_DIR=/path/to/storage/supernav-hm3d-ovon
export SUPERNAV_HABITAT_PYTHON=/path/to/storage/habitat-env/bin/python

mkdir -p "$SUPERNAV_DATA_ROOT" "$SUPERNAV_WORK_DIR/.cache" \
  "$SUPERNAV_WORK_DIR/.cache/tmp" \
  "$SUPERNAV_WORK_DIR/private" "$SUPERNAV_WORK_DIR/configs" \
  "$SUPERNAV_WORK_DIR/runs"
export XDG_CACHE_HOME="$SUPERNAV_WORK_DIR/.cache"
export PIP_CACHE_DIR="$SUPERNAV_WORK_DIR/.cache/pip"
export HF_HOME="$SUPERNAV_WORK_DIR/.cache/huggingface"
export TORCH_HOME="$SUPERNAV_WORK_DIR/.cache/torch"
export TMPDIR="$SUPERNAV_WORK_DIR/.cache/tmp"

"$SUPERNAV_HABITAT_PYTHON" -c 'import habitat_sim, supernav; print(habitat_sim.__file__, supernav.__file__)'
```

Obtain access to HM3D through the [dataset provider](https://github.com/matterport/habitat-matterport-3dresearch).
With your Matterport token ID and secret already set in the following environment
variables, download the v0.2 validation scenes using Habitat's downloader:

```bash
"$SUPERNAV_HABITAT_PYTHON" -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 --data-path "$SUPERNAV_DATA_ROOT"
```

The `hm3d_val_v0.2` group includes the Habitat scene assets, configurations, and
semantic assets. Download `hm3d_train_v0.2` as well when your selected episodes use
training scenes. Keep the original `.basis.navmesh` files.

For ObjectNav, download `objectnav_hm3d_v2.zip` from the Habitat-Lab dataset table
and extract it so that the chosen episode root contains `val/content/`. For OVON,
extract the author's `hm3d.tar.gz` so that the episode root contains
`val_unseen/content/`. Obtain `our-set/ovon_full_set.json` from MTU3D's
`embodied_bench_data.tar.gz` if using the OVON selection below. Keep that selection
file in private storage.

Arrange or point the variables at this structure; the directory names above each
split can differ:

```text
datasets/
├── scene_datasets/hm3d/
│   ├── hm3d_annotated_basis.scene_dataset_config.json
│   └── val/<number>-<scene_hash>/
│       ├── <scene_hash>.basis.glb
│       ├── <scene_hash>.basis.navmesh
│       ├── <scene_hash>.basis.scene_instance.json
│       └── ... semantic and stage configuration assets ...
├── objectnav_hm3d_v2/
│   └── val/content/<scene_hash>.json.gz
└── hm3d_ovon/hm3d/
    └── val_unseen/content/<scene_hash>.json.gz

supernav-hm3d-ovon/private/our-set/ovon_full_set.json
```

```bash
export SUPERNAV_HM3D_SCENES_ROOT="$SUPERNAV_DATA_ROOT/scene_datasets/hm3d"
export SUPERNAV_SCENE_DATASET_CONFIG="$SUPERNAV_HM3D_SCENES_ROOT/hm3d_annotated_basis.scene_dataset_config.json"
test -f "$SUPERNAV_SCENE_DATASET_CONFIG"
```

The scene dataset configuration must register the selected scenes with their
NavMeshes and semantic assets. The converters retain the case-sensitive scene
hash for scene-instance lookup; do not replace it with a `.glb` path. A bare mesh
load can bypass the registered NavMesh.

## Prepare HM3D v2 ObjectNav tasks

```bash
export SUPERNAV_TASK_SET=hm3d-v2
export SUPERNAV_SPLIT=val
export SUPERNAV_EPISODES_ROOT="$SUPERNAV_DATA_ROOT/objectnav_hm3d_v2"

supernav prepare objectnav \
  --episodes-root "$SUPERNAV_EPISODES_ROOT" \
  --split "$SUPERNAV_SPLIT" \
  --scene-dataset-config "$SUPERNAV_SCENE_DATASET_CONFIG" \
  --sensor-height 1.25 \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --instructions-dir "$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET" \
  --config-dir "$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET"

export SUPERNAV_INSTRUCTIONS="$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET/hm3d_v2_${SUPERNAV_SPLIT}.instructions.json"
export SUPERNAV_PREPARED_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/hm3d-${SUPERNAV_SPLIT}-localnav.json"
```

For a smaller initial task list, add `--max-episodes 1`. If only some scenes are
installed, also add `--scenes "$SUPERNAV_SCENE_HASH"`, after setting that variable
to an installed scene hash. `--scenes` accepts comma-separated hashes. Remove
these limits to prepare the full split, writing to a new private directory when
preserving an existing experiment.

The camera height is 1.25 m; adjust it with `--sensor-height`.

Continue with **Run one task, then a batch** below. To run OVON instead, use the
following preparation block.

## Prepare the OVON selection

The selection file contains a list under `val_unseen`. Each entry supplies
`scan_id_suffix`, `episode_index`, and `object_category`. `episode_index` is the
zero-based array position in the per-scene episode file, not `episode_id`. The
converter checks that the selected category matches the source episode.

Use episodes with an empty or absent `children_object_categories` field for this
workflow. The converter and scorer use viewpoints for the episode's own
`object_category`.

```bash
export SUPERNAV_TASK_SET=ovon
export SUPERNAV_SPLIT=val_unseen
export SUPERNAV_EPISODES_ROOT="$SUPERNAV_DATA_ROOT/hm3d_ovon/hm3d"
export SUPERNAV_OVON_SELECTION="$SUPERNAV_WORK_DIR/private/our-set/ovon_full_set.json"

supernav prepare ovon \
  --episodes-root "$SUPERNAV_EPISODES_ROOT" \
  --manifest "$SUPERNAV_OVON_SELECTION" \
  --split "$SUPERNAV_SPLIT" \
  --scene-dataset-config "$SUPERNAV_SCENE_DATASET_CONFIG" \
  --sensor-height 1.25 \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --instructions-dir "$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET" \
  --config-dir "$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET" \
  --output-dir "$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-batch"

export SUPERNAV_INSTRUCTIONS="$SUPERNAV_WORK_DIR/private/$SUPERNAV_TASK_SET/ovon_${SUPERNAV_SPLIT}_mtu3d120.instructions.json"
export SUPERNAV_PREPARED_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/ovon-${SUPERNAV_SPLIT}-mtu3d120-localnav.json"
```

The `--template-config` choice determines the executor.

## Run one task, then a batch

Use this block after either preparation section. The geometry-based template
already supplies `prompts_dir`, native navigation Skills, the `default` arm, and
the model/client settings. It uses oracle depth and NavMesh geometry. Preparation
sets
`HAB_DEFAULT_AGENT_NAVMESH=0` to retain the dataset's NavMesh.

Write a local overlay with explicit output paths. `visuals_root` places images,
audit logs, and trajectory sidecars on the data volume and is also used by the
scorer below.

```bash
export SUPERNAV_CONFIG="$SUPERNAV_WORK_DIR/configs/$SUPERNAV_TASK_SET/run.json"
export SUPERNAV_VISUALS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-visuals"
export SUPERNAV_RUNS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-one"
export SUPERNAV_SWEEP_ID=one

python - <<'PY'
import json
import os
from pathlib import Path

config = {
    "extends": os.environ["SUPERNAV_PREPARED_CONFIG"],
    "output_dir": os.environ["SUPERNAV_RUNS_ROOT"],
    "visuals_root": os.environ["SUPERNAV_VISUALS_ROOT"],
}
Path(os.environ["SUPERNAV_CONFIG"]).write_text(json.dumps(config, indent=2) + "\n")
PY

export SUPERNAV_TASK_ID=$(python - <<'PY'
import json
import os
from pathlib import Path

rows = json.loads(Path(os.environ["SUPERNAV_INSTRUCTIONS"]).read_text())["instructions"]
print(rows[0]["task_id"])
PY
)

supernav run --config "$SUPERNAV_CONFIG" \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --task-ids "$SUPERNAV_TASK_ID" --arms default \
  --dry-run --output-dir "$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-dry"

supernav run --config "$SUPERNAV_CONFIG" \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --task-ids "$SUPERNAV_TASK_ID" --arms default \
  --output-dir "$SUPERNAV_RUNS_ROOT"
```

The script selects the first task in the generated manifest; set
`SUPERNAV_TASK_ID` to another manifest ID to select another task. A dry-run prepares
configuration, prompts, and Skills but does not load a scene or execute navigation.
Inspect the real run's logs and RGB observations before starting a larger batch.
Use an unused bridge port with `--bridge-port` if another run is active.

To run every prepared task sequentially, omit `--task-ids`:

```bash
export SUPERNAV_RUNS_ROOT="$SUPERNAV_WORK_DIR/runs/$SUPERNAV_TASK_SET-batch"
export SUPERNAV_SWEEP_ID=batch
supernav run --config "$SUPERNAV_CONFIG" --arms default \
  --sweep-id "$SUPERNAV_SWEEP_ID" \
  --output-dir "$SUPERNAV_RUNS_ROOT"
```

A manifest prepared with `--max-episodes 1` still contains only one task. Existing
episode directories are skipped by default. Use a new output directory or
`--run-tag` for a new attempt, and retain earlier evidence. Assign a new
`SUPERNAV_SWEEP_ID` when reusing an output directory so that the saved sweep
configuration and task snapshot are preserved as well.

To use the Learned Executor, first follow the
[weights and policy-service instructions](habitat.md#learned-executor-weights).
Prepare a separate configuration using
`configs/experiments/habitat-learned-executor.json` as `--template-config`, then
repeat the run steps with separate output directories.

## Score saved runs

Run the scorer with the simulator interpreter to query the original scene
NavMeshes. `SUPERNAV_RUNS_ROOT` must point to the real single-task or batch
directory being scored. Use the
resolved manifest saved by that run and the same `visuals_root` as above:

```bash
"$SUPERNAV_HABITAT_PYTHON" -m supernav score-objectnav \
  --runs-root "$SUPERNAV_RUNS_ROOT" \
  --instructions "$SUPERNAV_RUNS_ROOT/_sweeps/$SUPERNAV_SWEEP_ID/instructions.resolved.json" \
  --scenes-root "$SUPERNAV_HM3D_SCENES_ROOT" \
  --visuals-root "$SUPERNAV_VISUALS_ROOT" \
  --criterion viewpoint_geodesic \
  --success-distance 0.2 \
  --l-source viewpoint_geodesic \
  --out-json "$SUPERNAV_RUNS_ROOT/score-geodesic.json" \
  --out-md "$SUPERNAV_RUNS_ROOT/score-geodesic.md"
```

The scoring parameters are:

| Parameter | Example value | Meaning |
| --- | --- | --- |
| `--criterion` | `viewpoint_geodesic` | NavMesh geodesic distance from the trajectory endpoint to the nearest goal viewpoint |
| `--success-distance` | `0.2` | Distance threshold in meters, using a strict less-than comparison |
| `--l-source` | `viewpoint_geodesic` | Recompute the start-to-goal viewpoint shortest distance for SPL |

SPL is `S × L / max(P, L)`: `S` is the independently scored success indicator,
`P` is accumulated trajectory length, and `L` is the start-to-goal viewpoint
geodesic distance. If recomputing `L` is unavailable, the scorer falls back to the
episode's stored distance. SR and SPL are means over scored episodes. The output
records the agent's completion claim as `claim_success`, while `success` and `spl`
come from independent scoring.

The scorer deduplicates runs by `task_id`, selecting the newest by metrics file
modification time. Use separate scoring roots for different methods, arms, and
repetitions, and check the number of scored tasks in `summary.episodes`.

Retain source episode files, task snapshots, native logs, and
`<session_id>.trajectory.json` sidecars. The scorer reads goal viewpoints from the
source episode paths recorded in the task snapshot; keep those files accessible.

For missing trajectories or unreachable goals, check `visuals_root`, the matching
`.basis.navmesh`, and source episode paths. See the [Web Viewer guide](web-viewer.md)
to inspect saved observations and trajectories.
