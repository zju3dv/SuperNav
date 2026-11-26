# InteriorGS Multi-Target Prompts

**English** | [简体中文](README.zh-CN.md)

This directory provides prompt templates for multi-target Habitat tasks.
Set `prompts_dir` to `configs/benchmarks/multi_global_task/prompts` in a local
configuration, and choose compatible navigation Skills and tools.
Supply an external task manifest through `instructions_file` or `--instructions`.

Each task specifies `ordered` and `targets`, together with its scene and spawn
settings. The agent receives target numbers and descriptions; keep target
coordinates, distances, and reference paths in separate evaluation inputs.
Agent completion claims are recorded separately from offline SR/SPL scores.

Prepare the Habitat-GS backend, scene assets, and local configuration following
the [configuration guide](../../../docs/release/en/configuration.md). Use
`--task-ids` to select IDs from your own manifest.

For the public HabitatGS-Benchmark, use `supernav prepare habitatgs`,
`export-habitatgs` and `score-habitatgs`. The
[GS benchmark guide](../../../docs/release/en/habitatgs-benchmark.md)
defines ordered SR1–SR5, their eligible denominators, STOP and SPL.
The legacy `score-multi-objectnav` goal-mark/bbox scorer implements a different protocol.
