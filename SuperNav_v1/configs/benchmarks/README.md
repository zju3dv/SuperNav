# Benchmark Prompt Templates

**English** | [简体中文](README.zh-CN.md)

This directory provides general prompt templates for the experiment recipes in
[`configs/experiments/`](../experiments/).

- [InteriorGS single-target navigation](global_task/README.md)
- [InteriorGS multi-target navigation](multi_global_task/README.md)

Task instructions, episode selections, scene annotations, and ground truth are
external inputs. Set `instructions_file` in an ignored `configs/local/` overlay
or pass `--instructions /path/to/tasks.json` when running a recipe.
The public HabitatGS-Benchmark is distributed separately. Its `episodes.json`
requires conversion with `supernav prepare habitatgs`; see the
[GS conversion, run and scoring guide](../../docs/release/en/habitatgs-benchmark.md).

Discover recipes and inspect their settings from the repository root:

```bash
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

See the [configuration guide](../../docs/release/en/configuration.md) for
external task manifests, scene assets, backend dependencies, and local settings.
