**English** | [简体中文](../zh/habitatgs-benchmark.md)

# Run HabitatGS-Benchmark: ObjectNav and ordered Multi-ObjectNav

The [public dataset](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark)
contains 150 ObjectNav and 150 ordered Multi-ObjectNav test episodes in a union of
70 InteriorGS scenes. This guide supports the public `habitat_gs/objectnav/v1`
contract. Historical `new2` and `multi_objectnav_global_task/v2` files use their
own historical workflows.

**Do not pass the public `episodes.json` directly to `supernav run --instructions`.**
Convert it with `supernav prepare habitatgs`, run navigation, export primitive
motion evidence, then score independently. `score-objectnav` is for HM3D/OVON;
legacy `score-multi-objectnav` evaluates goal marks with bbox/viewpoint Euclidean
criteria. Neither command implements the public GS contract.

## 1. Install and download

Complete the [Habitat-GS setup](habitat.md) and [agent configuration](configuration.md).
Install this updated SuperNav version in both agent and simulator environments:
the primitive-action recorder runs in the simulator process. From the checkout:

```bash
python -m pip install -e '.[agents,evaluation]'
python -m pip install huggingface_hub
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python
"$SUPERNAV_HABITAT_PYTHON" -m pip install '.[evaluation]'

export GS_DATA=/path/to/storage/habitatgs-benchmark
export GS_SCENES=/path/to/storage/habitatgs-scenes
export GS_WORK=/path/to/storage/supernav-gs
export GS_REVISION=4a8fe5cf654c1d19f0c3c992720d3c4f57844f12
hf download yyh929/HabitatGS-Benchmark --repo-type dataset \
  --revision "$GS_REVISION" --local-dir "$GS_DATA"
python "$GS_DATA/evaluation/verify.py" "$GS_DATA"
```

This revision is the adapter's verified release. Preserve the actual full dataset
commit SHA and manifest hashes. Revalidate schema and scoring compatibility when
updating. Keep downloaded annotations/GT and outputs outside the source checkout;
do not grant the evaluated policy access to evaluator files. Converted manifests
contain instructions, starts, scenes, ordered target descriptions and provenance,
without bboxes, viewpoints, instance labels or reference routes.

## 2. Prepare matching scenes

HF includes annotations, evaluation code and asset hashes, **not scene binaries**.
Obtain runtime assets matching `scene-assets/v1/manifest.json`, under the
[InteriorGS terms](https://github.com/manycore-research/InteriorGS):

```text
habitatgs-scenes/
└── interior_<scene-id>/
    ├── <scene-id>.gs.ply
    └── <scene-id>.navmesh
```

Local symlinks to existing assets are supported; preparation verifies file sizes
and SHA256 contents. The public release does not yet provide the complete
source-to-runtime conversion and matching-navmesh distribution route. Annotations
alone cannot run the simulator. Renaming compressed source PLY files or arbitrarily
rebaking a navmesh does not produce verified runtime assets.

The following upstream check requires all 140 scene files under `GS_SCENES`.
It rejects symlinks that resolve outside that directory. For external symlinks or
a subset of scenes, continue to preparation instead: `supernav prepare habitatgs`
verifies the size and SHA256 of every selected PLY/NavMesh before writing outputs.

```bash
python "$GS_DATA/evaluation/verify.py" "$GS_DATA" --scene-root "$GS_SCENES"
```

## 3. Convert tasks and configuration

Set `GS_TASK` to `objectnav` or `multi-objectnav`:

```bash
export GS_TASK=multi-objectnav
supernav prepare habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" \
  --dataset-revision "$GS_REVISION" --scene-root "$GS_SCENES" \
  --template-config configs/experiments/habitat-geo-based-executor.json \
  --sensor-height 1.25 --output-dir "$GS_WORK/$GS_TASK"
```

Preparation validates annotation/schema/asset-manifest hashes and selected scenes'
PLY/navmesh contents. It produces:

| Output | Contents |
| --- | --- |
| `instructions.json` | SuperNav tasks preserving episode IDs, instructions, starts and target order |
| `run.json` | Resolved experiment, input/output paths and primitive trace configuration |
| `scenes.scene_dataset_config.json`, `stages/`, `scenes/` | Habitat configurations referencing local assets |
| `preparation.json` | Revision, hashes, selected IDs, sensor height and asset verification |

Zero-based target indexes become one-based indexes for `hab_nav_goals`, preserving
order. Marks are agent memory; **scores do not depend on marks**. The converter
enables `hab_nav_goals` for multi-object tasks and selects the Geo or learned
multi-goal native skill, including its base global and local skill dependencies.
Multi-object preparation supports the native Geo and learned executor templates;
unsupported arm/skill combinations are rejected before output is written. The converter
preserves the dataset navmesh and disables automatic agent-specific rebaking.
The preparation directory must not already exist.

The explicit 1.25 m camera height is an example, not a universal dataset setting.
Report model, image settings, primitive step/turn sizes, sliding, executor and all
budgets. For the Learned Executor, start the [policy service](habitat.md#learned-executor-weights),
use `habitat-learned-executor.json` as template, and choose a separate output directory.
To prepare only one task in an installed scene, add `--task-ids <episode-id>`;
the default is the complete task release.

## 4. Preflight, single episode and full run

```bash
export GS_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["instructions"][0]["task_id"])' \
  "$GS_WORK/$GS_TASK/instructions.json")

supernav run --config "$GS_WORK/$GS_TASK/run.json" --task-ids "$GS_ID" \
  --dry-run --output-dir "$GS_WORK/$GS_TASK/dry-run"
supernav run --config "$GS_WORK/$GS_TASK/run.json" --task-ids "$GS_ID" \
  --output-dir "$GS_WORK/$GS_TASK/smoke-runs"

# After verifying observations, actions and logs from the real smoke run:
supernav run --config "$GS_WORK/$GS_TASK/run.json"
```

A dry-run prepares inputs without calling a model or simulator; it does not test
navigation or weight loading. Keep one method, arm and repetition per scoring
directory. Use separate directories for retries: the exporter rejects duplicate
episode IDs instead of selecting a successful retry. Parallel runners also need
separate bridge ports.
The exporter also checks `canonical.jsonl` for exactly one initialized session
matching the metrics; restarting a session inside a run cannot silently replace its trajectory.

The simulator writes `visuals/<session_id>.primitive.jsonl`: actual start position,
every primitive-action position, and STOP on explicit session closure. Turns,
collisions and tiny displacements are retained. Cleanup/process exit does not
invent STOP. Mid-episode pose resets record a runtime error. This trace is private
evaluator evidence and its path/positions are not added to policy responses.
Primitive traces are excluded from both MCP artifact listings and artifact reads.

## 5. Export and score

These commands evaluate the full `runs/` directory. Use `smoke-runs/` for the
single-episode check:

```bash
supernav export-habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" \
  --runs-root "$GS_WORK/$GS_TASK/runs" \
  --visuals-root "$GS_WORK/$GS_TASK/visuals" \
  --output "$GS_WORK/$GS_TASK/submissions.json"

"$SUPERNAV_HABITAT_PYTHON" -m supernav score-habitatgs \
  --dataset-root "$GS_DATA" --task "$GS_TASK" --scene-root "$GS_SCENES" \
  --submissions "$GS_WORK/$GS_TASK/submissions.json" \
  --reference global_shortest --output "$GS_WORK/$GS_TASK/scores.json"
```

The submission also works with the pinned HF `evaluation/score.py`. That upstream
command requires the full scene layout, even for missing submissions, and rejects
asset symlinks that resolve outside its scene root. SuperNav
preserves its episode SR/SPL contract and adds ordered `stage_sr` aggregates.
Scoring uses verified navmeshes, without rendering or model calls. Historical
runs with only tool endpoints or deduplicated trajectories cannot be exported as
complete primitive evidence. Collect new traces instead of inventing missing
actions. Missing/inconsistent files are errors; do not delete failed tasks or
substitute `goal_marks` to make export pass.

Use new output filenames. Unsubmitted episodes remain failures in the full
release denominator. A one-episode smoke submission therefore reports 149 missing
submissions and is not a full benchmark performance evaluation.
SuperNav needs navmeshes only for submitted episodes and lists every verified mesh
in the output provenance. Infrastructure failures without a session/trace require
audited error evidence; the exporter stops for review rather than guessing their status.

## 6. Success, SPL and stage SR

- **Single:** STOP with navmesh geodesic distance strictly `<0.1m` to any approved viewpoint.
- **Multi:** visit all target regions in order along the recorded trajectory, then
  STOP. The terminal pose need not stay at the final target. Consecutive overlapping
  regions can be completed at the same pose. Bboxes, a single reference endpoint
  and Euclidean distance are not substitutes for viewpoint-set geodesic distance.
- **SPL:** `S * L / max(L, P)`. `L` defaults to the released `global_shortest`
  reference; `P` is primitive-action path length. Failures contribute zero;
  successful `L=P=0` contributes one. Report `local_shortest` explicitly if selected.
- **SRk:** among episodes with at least k targets, the fraction that visits the
  first k targets in order. Prefix completion does not require STOP. Variable
  target counts mean SR5 need not equal overall SR.

For the full Multi150 release:

| Metric | SR1 | SR2 | SR3 | SR4 | SR5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eligible_total` | 150 | 150 | 112 | 75 | 38 |

The public contract excludes runtime errors from evaluated means (null success/SPL).
Budget exhaustion, missing STOP and missing submissions are failures. Stage SR
uses the same error exclusion and reports `eligible_total`, `evaluated`,
`runtime_errors`, `successes`, and `success_rate` per stage. Always report totals,
actual denominators and error counts alongside rates. Events after the first STOP,
budget exhaustion or runtime error are ignored.

Runtime `metrics.json.success` describes agent claims/engineering completion;
use independent `scores.json` for benchmark metrics. HM3D/OVON are single-goal
tasks: their conversion and SR/SPL commands are documented in the
[HM3D v2 and OVON guide](hm3d-and-ovon.md), with no multi-goal SR2–SR5.
