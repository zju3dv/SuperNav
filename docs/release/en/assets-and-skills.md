[English](assets-and-skills.md) | [简体中文](../zh/assets-and-skills.md)

# Assets and Navigation Skills

## Scene Assets and Backends

SuperNav includes experiment recipes, general prompts, and Skills. Supply task
manifests and evaluation ground truth externally. Prepare simulator installations,
scene assets, external datasets, and policy checkpoints for your experiment.

| Backend | Required resources | Setup guide |
| --- | --- | --- |
| [Habitat-GS](https://github.com/zju3dv/habitat-gs) | Compatible `habitat_sim` SDK, scene dataset configuration, scene assets, and NavMeshes | [Habitat-GS](habitat.md) |
| [AI2-THOR](https://github.com/allenai/ai2thor) | AI2-THOR 5.0.0, the pinned CloudRendering build, frozen demand-driven dataset, and Linux NVIDIA/Vulkan support | [AI2-THOR](ai2thor.md) |

For Gaussian Splatting scenes, follow the
[Habitat-GS asset instructions](https://github.com/zju3dv/habitat-gs#-download-gs-asset).
Match scene IDs, NavMeshes, task annotations, and spawn poses to the experiment.
OVON uses the corresponding HM3D/OVON data; AI2-THOR uses the frozen demand-driven
dataset described in its setup guide. Configure local paths using the
[configuration guide](configuration.md).

Methods using NoMaD or LocateAnything also require the corresponding policy or
grounding service and its model weights.

## Navigation Skills

[`skills/`](../../../skills) provides instructions for exploration, target search,
local navigation, recovery, and stopping. The two Habitat recipes use these native Skills:

- `habitat-geo-based-executor`:
  [`global-navigation-geo-based-executor`](../../../skills/global-navigation-geo-based-executor/SKILL.md)
  and [`localnav-pointnav-geo-based-executor`](../../../skills/localnav-pointnav-geo-based-executor/SKILL.md).
  Its executor uses oracle depth and NavMesh geometry.
- `habitat-learned-executor`:
  [`global-navigation-learned-executor`](../../../skills/global-navigation-learned-executor/SKILL.md)
  and [`localnav-pointnav`](../../../skills/localnav-pointnav/SKILL.md).
  Its weights are not yet released, and no download is available. Real execution
  requires your own compatible checkpoint and running policy service; see
  [policy service requirements](habitat.md#learned-executor-weights).

Native-Skill runs save a frozen Skill snapshot and verify its visibility to the
selected client. See [tool authoring](tool-authoring-guide.md) to add or change tools.
