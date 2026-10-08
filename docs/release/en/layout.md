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
├── skills/                   Navigation skill sources, snapshotted for runs
├── scripts/                  Development, validation, and audit helpers
├── tests/                    Automated tests and fixtures
├── docs/
│   ├── release/
│   │   ├── en/               English public documentation
│   │   └── zh/               Simplified Chinese public documentation
├── data/
│   ├── embodiments/          Small embodiment parameter snapshots
│   ├── test_assets/          Small input assets
│   └── runs/                 Run evidence; may point to external storage
├── .cache/                   Build and test caches
├── README.md                 English introduction
└── README.zh-CN.md            Simplified Chinese introduction
```

Implementation lives in `src/` and public operations use the `supernav` command tree.

Rerun is implemented in `src/supernav/web/rerun_viewer.py` and configured through
`configs/viewers/rerun.yaml`.

## Inputs and generated artifacts

| Location | Purpose | Committed? |
| --- | --- | --- |
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

- Put implementation in the responsible `src/supernav/` module and keep CLI routing thin.
- Add named experiments in `configs/experiments/` and shared settings in the corresponding fragments. Select external task manifests through local overlays.
- Put general prompt templates in `configs/benchmarks/<suite>/`. Keep real tasks, annotations, and GT in private storage and exclude them from release artifacts.
- Put development and validation scripts in `scripts/`; move reusable logic into the canonical package.
- Use synthetic test inputs in `tests/fixtures/` or generate them in test temporary directories. Keep real task data private.
- Maintain public guides as matching pages in `docs/release/en/` and `docs/release/zh/`. Store internal handoffs and validation records in a separate internal documentation repository.
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

The archive contains the public source and bilingual guides, with development-only
`AGENTS.md` files, internal handoffs, real benchmark tasks and GT, local configuration, credentials, and run
outputs excluded. The export contains no Git history. Extract it into a new directory and initialize the new
public repository there; pushing an existing branch also transfers its history.

Keep credentials in local client files or environment variables. The ignore and
packaging rules cover environment files, authentication files, and client
configuration directories; empty `.env.example` templates remain distributable.
