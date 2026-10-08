# SuperNav Configurations

**English** | [简体中文](README.zh-CN.md)

Recipes in `experiments/` are example templates. Before running an experiment,
prepare:

1. Scene assets or datasets and a compatible simulator environment.
2. A configuration adapted to your machine: model, credentials, paths, and services.
3. Task instructions or a manifest with the matching scenes, spawn poses, and goals.

Provide GT when required by the selected method or scoring procedure. Discover
templates with `supernav config list`. Run with `--experiment <name>` plus your
environment settings and CLI options, or save an ignored `local/` overlay and
run with `--config <path>`. Set `instructions_file` in the overlay or pass
`--instructions /path/to/tasks.json` to select your task manifest.
`benchmarks/` provides general prompt templates.

| Example recipe | Purpose |
| --- | --- |
| `habitat-geo-based-executor` | Habitat navigation with a geometry-based executor and native Skills |
| `habitat-learned-executor` | Habitat navigation with a learned local policy and native Skills; requires a policy service |
| `ai2thor-primitive` | AI2-THOR demand-driven navigation with primitive actions |

Each recipe has one `default` arm. Omit `--arms` or use `--arms default`.
Reusable method fragments are `methods/geo-based-executor.json`,
`methods/learned-executor.json`, and `methods/primitive.json`.

All bundled recipes use the shared
[`models/gpt-6-astra-medium.json`](models/gpt-6-astra-medium.json) fragment:
model `gpt-6-astra` with `medium` reasoning. Keep credentials in environment
variables or local client configuration.

See the [experiment configuration guide](../docs/release/en/configuration.md)
for composition rules, external task inputs, local paths, and running examples,
or browse the [release documentation](../docs/release/en/README.md).
