[English](../en/assets-and-skills.md) | [简体中文](assets-and-skills.md)

# 资产与导航 Skills

<a id="scene-assets-and-backends"></a>

## 场景资产与后端

SuperNav 包含实验配方、通用提示和 Skills。任务清单和评测 GT 由外部提供。
请按所选实验准备仿真器、场景资产、外部数据集及策略权重。

| 后端 | 所需资源 | 接入指南 |
| --- | --- | --- |
| [Habitat-GS](https://github.com/zju3dv/habitat-gs) | 兼容的 `habitat_sim` SDK、场景数据集配置、场景资产及 NavMesh | [Habitat-GS](habitat.md) |
| [AI2-THOR](https://github.com/allenai/ai2thor) | AI2-THOR 5.0.0、固定 CloudRendering 构建、冻结需求驱动数据集及 Linux NVIDIA/Vulkan 环境 | [AI2-THOR](ai2thor.md) |

Gaussian Splatting 场景按
[Habitat-GS 资产说明](https://github.com/zju3dv/habitat-gs#-download-gs-asset) 准备。
场景 ID、NavMesh、任务标注和起始位姿应与实验匹配。OVON 使用对应的 HM3D/OVON 数据；
AI2-THOR 使用接入指南中说明的冻结需求驱动数据集。本机路径设置见[配置指南](configuration.md)。

使用 NoMaD 或 LocateAnything 的方法时，还需准备相应的策略或 grounding 服务及模型权重。

<a id="navigation-skills"></a>

## 导航 Skills

[`skills/`](../../../skills) 提供场景探索、目标搜索、局部导航、恢复和停止的操作指导。
两个 Habitat 配方使用以下原生 Skills：

- `habitat-geo-based-executor`：
  [`global-navigation-geo-based-executor`](../../../skills/global-navigation-geo-based-executor/SKILL.md)
  与 [`localnav-pointnav-geo-based-executor`](../../../skills/localnav-pointnav-geo-based-executor/SKILL.md)。
  执行器使用 oracle 深度和 NavMesh 几何。
- `habitat-learned-executor`：
  [`global-navigation-learned-executor`](../../../skills/global-navigation-learned-executor/SKILL.md)
  与 [`localnav-pointnav`](../../../skills/localnav-pointnav/SKILL.md)。
  权重尚未发布，目前没有下载地址。真实运行需自备兼容 checkpoint 和已启动的策略服务，
  详见[策略服务要求](habitat.md#learned-executor-weights)。

原生 Skill 模式会保存冻结的 Skill 快照，并验证其对所选客户端的可见性。
新增或修改工具见[工具编写指南](tool-authoring-guide.md)。
