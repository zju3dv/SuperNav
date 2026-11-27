# InteriorGS Single-Target Prompts

**English** | [简体中文](README.zh-CN.md)

This directory provides prompt templates for the
[`habitat-geo-based-executor`](../../experiments/habitat-geo-based-executor.json)
and [`habitat-learned-executor`](../../experiments/habitat-learned-executor.json)
recipes.

Supply your task manifest with instructions, scene and spawn settings.
Keep scoring ground truth separate from policy inputs. Set the task path as `instructions_file` in a
`configs/local/` overlay that extends the chosen recipe. Select episodes with
`--task-ids` using IDs from your manifest.

Prepare the Habitat-GS backend and scene assets, then run the local configuration:

```bash
supernav run --config configs/local/habitat.json
```

The [configuration guide](../../../docs/release/en/configuration.md) shows the
overlay format and external task setup. Shared recipe fragments define the
agent, model, and navigation method.

For the public HabitatGS-Benchmark `episodes.json`, first use
`supernav prepare habitatgs`. Follow the
[GS benchmark guide](../../../docs/release/en/habitatgs-benchmark.md)
for conversion, primitive trajectory export and SR/SPL scoring.
