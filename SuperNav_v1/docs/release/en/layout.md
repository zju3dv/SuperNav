[English](layout.md) | [简体中文](../zh/layout.md)

# Repository layout

Start with the [README](../../../README.md). See [Architecture](architecture.md)
for runtime responsibilities and [Configuration](configuration.md) for experiment
composition.

```text
SuperNav/
├── src/
│   ├── supernav/             CLI, experiments, runtime, methods, backends, evaluation, views
│   └── habitat_contract/     Simulator-independent shared protocols
├── configs/
│   ├── experiments/          Named experiment recipes
│   ├── agents/               Client and timeout settings
│   ├── models/               Model names and reasoning settings
│   ├── backends/             Simulator backend settings
│   ├── methods/              Navigation methods
│   ├── benchmarks/           General prompt templates
│   ├── runtime/              Runtime policies
│   ├── evaluation/           Evaluation policies
│   ├── providers/            Provider catalog and credential variable examples
│   ├── viewers/              Viewer settings, including Rerun
│   └── local/                Machine-specific overlays for local use
├── robots/
│   └── go2/                  Independent map-backed PixelGoal deployment
│       ├── dimos/            Public DimOS submodule, pinned commit
│       ├── supernav/         Assembly tools, manifest, overlay/, patches/
│       ├── deploy/           Installers, launchers, config templates, dependency locks
│       ├── tests/            Deployment and resource tests
│       └── runtime/          Generated source and local run outputs (ignored)
├── skills/                   Navigation skill sources, snapshotted for runs
├── scripts/                  Development, validation, and audit helpers
├── tests/                    Automated tests and fixtures
├── docs/
│   ├── release/
│   │   ├── en/               English guides, including go2*.md deployment tutorials
│   │   └── zh/               Matching Simplified Chinese guides
├── data/
│   ├── embodiments/          Small embodiment parameter snapshots
│   ├── test_assets/          Small input assets
│   └── runs/                 Run evidence; may point to external storage
├── .cache/                   Build and test caches
├── README.md                 English introduction
└── README.zh-CN.md            Simplified Chinese introduction
```

The simulation runtime lives in `src/` and uses the `supernav` command tree.
The [Go2 deployment](go2.md) has its own installation
root at `robots/go2/`, including an independent `.venv` and dependency locks.
Run its assembly commands from that directory; its `supernav/` directory contains
assembly tools and DimOS changes, separate from the simulation Python package.
The public submodule plus overlays and patches assemble into
`robots/go2/runtime/dimos/`.

Rerun is implemented in `src/supernav/web/rerun_viewer.py` and configured through
`configs/viewers/rerun.yaml`.

## Inputs and generated artifacts

| Location | Purpose | Committed? |
| --- | --- | --- |
| `robots/go2/dimos/` | Public DimOS submodule | Pinned gitlink |
| `robots/go2/supernav/`, `robots/go2/deploy/`, `robots/go2/tests/` | Go2 integration source, installers, locks, templates, and tests | Yes |
| `docs/release/en/go2*.md`, `docs/release/zh/go2*.md` | Bilingual Go2 deployment tutorials | Yes |
| `robots/go2/runtime/`, `robots/go2/.venv/`, `robots/go2/deploy/config/*.env` | Generated runtime, environment, logs, and real robot configuration | No |
| `configs/benchmarks/` | General prompt templates | Yes |
| `data/embodiments/` | Embodiment parameter snapshots | Yes |
| `data/test_assets/` | Small warmup images and other inputs | Yes |
| `tests/fixtures/` | Synthetic test inputs and fixture documentation | Yes |
| `data/runs/` | Experiment evidence | No |
| `data/nav_artifacts*/` | Local navigation logs, observations, and measurements | No |
| `configs/local/` | Local deployment settings and external task-manifest paths | No |
| `.cache/` | Reproducible build, test, and packaging caches | No |
| `src/*.egg-info/`, `.venv/` | Package metadata and local Python environments | No |

Current run output defaults belong under `data/runs/`. Preserve its external
storage link, if present, and existing run evidence. Prepare scenes, model weights,
task manifests, and evaluation ground truth separately, then locate them through
environment variables or explicit local configuration. Real benchmark tasks and
GT are excluded from public source exports and packages.

## Where to add files

- Maintain Go2 changes in `robots/go2/supernav/overlay/` and `patches/`, update the manifest, and rebuild the runtime. Keep navigation rules in the deployment's Markdown skills.
- Put implementation in the responsible `src/supernav/` module and keep CLI routing thin.
- Add named experiments in `configs/experiments/` and shared settings in the corresponding fragments. Select external task manifests through local overlays.
- Put general prompt templates in `configs/benchmarks/<suite>/`. Keep real tasks, annotations, and GT in private storage and exclude them from release artifacts.
- Put development and validation scripts in `scripts/`; move reusable logic into the canonical package.
- Use synthetic test inputs in `tests/fixtures/` or generate them in test temporary directories. Keep real task data private.
- Maintain public guides as matching pages in `docs/release/en/` and `docs/release/zh/`.
- Store run evidence under `data/runs/` or the explicitly selected output directory, retaining each run's original contents and recorded paths.

Keep the repository root for project-level files. `pyproject.toml` defines
dependencies, packages, packaged resources, and the CLI; `setup.cfg` sets the build
location. Build and test caches belong under `.cache/`.

## Export a release snapshot

After reviewing and committing the release changes, export the current commit:

```bash
python scripts/export_release_source.py \
  --output .cache/release/supernav-source.tar.gz
```

This export and the root Python wheel/sdist cover the simulation runtime; they
exclude `robots/go2/` and its submodule configuration. Get Go2 through a full Git
clone, then initialize `robots/go2/dimos` as described in its
[installation guide](go2.md).

The archive contains the simulation source and bilingual guides. Extract it into a
new directory to install the simulation runtime.

Keep credentials in local client files or environment variables. The ignore and
packaging rules cover environment files, authentication files, and client
configuration directories; empty `.env.example` templates remain distributable.
