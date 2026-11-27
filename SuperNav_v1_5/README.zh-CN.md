<div align="center">

<img src="docs/release/assets/supernav1.5-wordmark.svg" alt="SuperNav-1.5" width="580">

<h1>Structured Spatial Memory For Agentic Navigation Systems</h1>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

<p>
  <a href="https://zju3dv.github.io/SuperNav-1.5/"><img src="https://img.shields.io/badge/Project-SuperNav--1.5-e34b4b" alt="项目主页"></a>
  <a href="https://arxiv.org/abs/2611.29912"><img src="https://img.shields.io/badge/Paper-arXiv%3A2611.29912-2f6feb" alt="SuperNav-1.5 论文"></a>
  <a href="../README.zh-CN.md"><img src="https://img.shields.io/badge/Code-SuperNav--1.5-black?logo=github" alt="SuperNav-1.5 代码"></a>
  <a href="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4"><img src="https://img.shields.io/badge/Demo-Overview_Video-c79a00" alt="概览视频"></a>
</p>

<p>
  Jiarui Guo<sup>1</sup> ·
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Yongchen Zai<sup>2</sup> ·
  Futang Zhu<sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>3</sup> ·<br>
  <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;·&nbsp;
  <sup>2</sup> BYD New Technology Research Institute &nbsp;·&nbsp;
  <sup>3</sup> Shenzhen University
</p>

<video src="https://raw.githubusercontent.com/bugByteMe/SuperNav-1.5-assets/main/assets/overview/supernav-1.5-overview.mp4" poster="docs/release/assets/supernav-1.5-overview.webp" controls muted loop width="920"></video>

</div>

## 📢 News

> **[2026-10-21]** 🤖 **Unitree Go2 真机部署**已发布，提供结构化空间记忆、四路 RGB、基于 LiDAR 地图的 PixelGoal 导航、区域描述和跨会话重定位。Jetson 安装与使用方法见[部署教程](docs/release/zh/go2.md)。

> **[2026-10-13]** 🎉 InteriorGS **ObjectNav 与 Multi-ObjectNav 测试回合**（各 150 个，含参考轨迹）已在 [HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark) 发布，数据与评测方式见 [benchmark 指南](https://github.com/zju3dv/habitat-gs/blob/main/docs/benchmark/README.md)。

## 🧭 SuperNav-1.5 是什么？

**SuperNav-1.5** 为导航智能体提供结构化空间记忆：由分区占据地图与智能体维护的
**记忆节点**组成。它将场景回忆连接到可执行路线，帮助智能体高效探索、连续到达
多个目标，并跨会话复用记忆。它是 SuperNav 系列的后续工作；前序方法见
[原始 SuperNav](../SuperNav_v1/README.zh-CN.md)。

本仓库提供用于 **Habitat-GS** 和 **AI2-THOR** 的智能体运行时、导航工具与 Skills、
实验配置和评测工具。使用 SuperNav-1.5 可以：

- **构建持久空间记忆**：结合记忆节点、度量 SLAM 与图像描述。
- **替代旧账本**：选择空间记忆实验时自动停用原有空间事实账本。
- **运行导航智能体**：支持 Codex、OpenCode 和 Kimi。
- **组合任务、工具与 Skills**：通过显式实验配置控制智能体能力。
- **实时观察执行**：查看 RGB 观测、运动轨迹和工具调用。
- **检查并评测运行结果**：保留提示、工具结果、原生日志和独立任务指标。

[SuperNav-1.5 项目主页](https://zju3dv.github.io/SuperNav-1.5/)提供真机与仿真视频、
方法介绍和研究结果。
[Unitree Go2 部署源码](docs/release/zh/go2.md) 将同一套结构化空间记忆流程用于真机：
四路 RGB、带标注的 LiDAR 占据地图、记忆节点、可选区域描述、基于地图的 PixelGoal
导航，以及跨会话重定位。该部署基于固定版本的公开 DimOS submodule 组装。
更多真机演示见项目主页。

## 📖 目录

- [发布状态](#release-status)
- [安装](#installation)
- [使用结构化空间记忆运行](#run-navigation-experiments)
- [复用 SuperNav v1 工作流](#visualization-and-evaluation)
- [文档与开发](#documentation-and-development)
- [引用](#citation)
- [致谢与许可证](#acknowledgements-and-license)

<a id="release-status"></a>

## 📦 发布状态

- [x] 智能体运行时、导航工具与 Skills、实验配方及评测
- [x] 结构化空间记忆、定制 Cartographer worker 与 stack 生命周期命令
- [ ] Unitree Go2 真机部署：结构化空间记忆、基于地图的 PixelGoal、四路 RGB 和 LiDAR

<a id="installation"></a>

## 🛠️ 安装

SuperNav-1.5 的标准发布配置是使用**结构化空间记忆的 Habitat-GS**。请按
[完整的空间记忆安装与运行指南](docs/release/zh/spatial-memory.md)准备依赖、配置 GPU、
设置外部描述服务并排查问题。

安装过程使用两个环境：

| 环境 | 用途 |
| --- | --- |
| `supernav-cartographer` | 构建固定版本的 Cartographer 源码并提供其原生库 |
| 你的 Habitat-GS Python 3.12 环境 | 运行 `supernav stack`、Habitat bridge、空间记忆和受管描述服务 |

安装完成后请保留两个环境。先进入项目目录：

```bash
git clone https://github.com/zju3dv/SuperNav.git
cd SuperNav/SuperNav_v1_5
```

首先在专用 Conda 环境中构建补丁版 Cartographer worker：

```bash
git submodule update --init third_party/cartographer
conda env create -f environment-cartographer.yml
conda activate supernav-cartographer
SUPERNAV_CARTOGRAPHER_BUILD_JOBS=4 scripts/build_cartographer_worker.sh
conda deactivate
```

然后激活已经可以运行 Habitat-GS 的 Python 3.12 环境，安装完整的 SuperNav-1.5
运行时：

```bash
conda activate <habitat-gs-runtime-environment>
python -m pip install --upgrade pip
python -m pip install -e '.[agents,spatial-memory,caption]'
```

如果尚未安装 Habitat-GS，请先完成
[Habitat-GS 配置](docs/release/zh/habitat.md#安装与选择仿真器)，再安装上述运行时。
所选智能体 CLI 需要单独安装和认证。内置 Codex 配方使用 `gpt-6-astra` 和 medium
推理强度；provider 与凭据设置见[配置指南](docs/release/zh/configuration.md)。

其他工作流分别记录在以下指南中：

| 工作流 | 指南 |
| --- | --- |
| 原始 SuperNav 基线实验 | [SuperNav v1](../SuperNav_v1/README.zh-CN.md#installation) |
| AI2-THOR | [AI2-THOR 配置](docs/release/zh/ai2thor.md) |
| Unitree Go2 | [Go2 部署](docs/release/zh/go2.md) |
| 实时与记录可视化 | [Web 查看器](docs/release/zh/web-viewer.md) |

<a id="quick-start"></a>
<a id="run-navigation-experiments"></a>

## 🧠 使用结构化空间记忆运行

SuperNav-1.5 与 [SuperNav v1](../SuperNav_v1/README.zh-CN.md#run-navigation-experiments)
使用相同的任务清单、任务选择、重复次数、评分和评测流程。结构化空间记忆可以通过
两种执行模式评测：

| 执行模式 | 用途 | 入口命令 |
| --- | --- | --- |
| **单会话 benchmark** | 使用空间记忆运行常规 Habitat benchmark，包括单目标、多目标、HM3D ObjectNav 和 HM3D-OVON。每个 episode 都从空记忆和新 SLAM 地图开始。 | `supernav stack run` |
| **配对会话 benchmark** | 评测跨会话记忆复用。探索会话先构建记忆，检索会话随后从该记忆及其 SLAM 状态的不可变快照启动，并可使用不同初始位姿。 | `supernav stack run-paired` |

配对会话是与单目标、多目标、HM3D 等并列的额外 benchmark 条件。
`stack run` 是常规单会话 benchmark 的标准入口，并非诊断命令。

执行模式与空间记忆配方相互独立：

| 实验配方 | 记忆配置 |
| --- | --- |
| `habitat-spatial-memory` | 记忆节点、Cartographer 二维 SLAM、占据栅格导航和 Qwen 区域描述 |
| `habitat-spatial-memory-slam-only` | 相同的记忆节点与 SLAM 流程，但关闭区域描述 |

### 通用配置

先完成[原生 Cartographer 构建与运行环境安装](docs/release/zh/spatial-memory.md#安装与构建)
和 [Habitat-GS 配置](docs/release/zh/habitat.md)。按照
[SuperNav v1 相同的数据准备流程](../SuperNav_v1/README.zh-CN.md#run-navigation-experiments)
为所选 benchmark 准备任务清单。之后按本机路径配置运行环境：

```bash
export SUPERNAV_HABITAT_ROOT=/absolute/path/to/habitat-gs
export SUPERNAV_HABITAT_PYTHON=/absolute/path/to/habitat-environment/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/absolute/path/to/global_navigation.scene_dataset_config.json
export HAB_GPU_DEVICE_ID=0
export HAB_CARTOGRAPHER_WORKER_COMMAND="scripts/run_cartographer_worker.sh {input_fifo} {occupancy_path}"
```

`SUPERNAV_HABITAT_PYTHON` 同时是受管 Qwen 描述服务和 Open3D 重定位器的默认
解释器，因此该环境必须安装 Torch、Open3D 以及 `spatial-memory`/`caption` extras。
如需使用已有的本地 Qwen checkout，而不是下载公开模型 ID，请设置
`SLAM_AREA_CAPTION_MODEL=/absolute/path/to/Qwen3.5-2B`。高级用户仍可通过
`SUPERNAV_CAPTION_PYTHON` 和 `HAB_CARTOGRAPHER_RELOCALIZER_COMMAND` 显式覆盖默认值。

运行前可检查任一配方：

```bash
supernav config show --experiment habitat-spatial-memory
supernav config show --experiment habitat-spatial-memory-slam-only
```

下文使用完整配方。如不需要区域描述，可在任一模式中将其替换为
`habitat-spatial-memory-slam-only`。

### 模式 1：单会话 benchmark

普通单会话 benchmark 任务清单使用 `stack run`。无论任务清单包含 Habitat-GS
单目标、多目标、HM3D ObjectNav 还是 HM3D-OVON，入口命令都相同：

```bash
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory
```

将 `example-task` 替换为任务清单中的 `task_id`。配方已选择使用 medium 推理强度的
`gpt-6-astra`；`--model`、`--reps`、`--run-tag` 和 `--dry-run` 等普通运行参数会
原样转发。stack 会启动所需的支持服务，委托普通实验运行器执行，并为每个 episode
启动和关闭 Habitat bridge。

使用 SuperNav v1 指南准备各 benchmark 的数据和任务清单；将 v1 指南中的
`supernav run` 替换为上面的 `supernav stack run`，以确保空间记忆服务可用：

| Benchmark 工作流 | 准备与评分指南 |
| --- | --- |
| Habitat-GS 单目标与多目标 | [HabitatGS-Benchmark](../SuperNav_v1/docs/release/zh/habitatgs-benchmark.md) |
| HM3D ObjectNav 与 HM3D-OVON | [HM3D v2 与 OVON](../SuperNav_v1/docs/release/zh/hm3d-and-ovon.md) |
| 任务选择、重复次数、arms、run tag 和输出目录 | [运行导航实验](../SuperNav_v1/README.zh-CN.md#run-navigation-experiments) |

每个单会话 episode 都从全新记忆开始。空间记忆配方会关闭 SuperNav v1 的有界
空间事实账本，确保智能体只看到一个空间记忆系统，而不是两套相互竞争的记忆。

### 模式 2：配对会话 benchmark

仅当 benchmark 条件需要度量先前会话的记忆复用时，才使用 `stack run-paired`：

```bash
supernav stack run-paired \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory-paired
```

`run-paired` 会对选中的任务行执行两次运行：

1. **探索会话**从 `spawn` 启动，写入记忆节点以及视觉、规划 Cartographer 状态。
2. 运行器创建经过哈希审计的不可变快照，其中包含 `memory_nodes.json`、两个
   `.pbstream` 文件和可用的描述证据。
3. **检索会话**从任务行可选的 `retrieval_spawn` 启动（未提供时回退到 `spawn`），
   获得记忆节点的独立可写副本，并以 `delayed_fpfh_relocalization` 模式加载两张
   冻结地图。智能体调用 `hab_report_memory_pos` 时会向与源实现一致的重定位器提供
   节点先验，同时后台累积扫描匹配保持运行。

两个会话共享指令、策略模板、模型、arm 和原生 Skill 快照；只有 benchmark 管理的
初始位姿与持久运行状态不同。任务行可以直接指定两个起点：

```json
{
  "task_id": "example-task",
  "slug": "example-task",
  "text": "Find the requested object.",
  "scene": "interior_0405_840145",
  "spawn": {"start_position": [0.0, 0.2, 0.0], "start_rotation": [0, 0, 0, 1]},
  "retrieval_spawn": {"start_position": [3.0, 0.2, -2.0], "start_rotation": [0, 0.707, 0, 0.707]}
}
```

stack 会预检补丁版 worker，启动描述服务并等待健康，然后为每个会话分别启动新的
Habitat bridge 和 Cartographer worker。它只停止自己启动的描述进程。

配方绑定专用空间记忆提示，并共同要求 `memory-nodes`、`2d-slam` 和
`spatial-memory-navigation` Skills。可见的局部目标使用即时单次调用的视觉点移动；
屏幕外的记忆节点和当前 SLAM 区域前沿使用预览/确认式占据地图路由。

配对状态写入 `<output-dir>/_pairs/<pair-id>/`。`pair.json` 记录两个 run ID 和状态；
`snapshot/manifest.json` 记录产物哈希；普通 episode 产物位于 `episodes/` 下。
如需在不重新运行有效探索快照的情况下重试检索：

```bash
supernav stack run-paired \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task --arms default --resume \
  --output-dir data/runs/habitat-spatial-memory-paired
```

如果不可变快照产物缺失或哈希已变化，resume 会以失败关闭。使用 `--overwrite`
可将之前的配对归档到带时间戳的 stale 目录并启动新配对。仅选择一个任务时，
可用 `--retrieve-spawn '{...}'` 覆盖检索起点。

### 可复用配置与服务

任一执行模式都可以使用本机可复用配置，将任务清单和输出路径从命令中移出。
创建已被 Git 忽略的 `configs/local/habitat-spatial-memory.json`：

```json
{
  "extends": "../experiments/habitat-spatial-memory.json",
  "instructions_file": "/absolute/path/to/tasks/habitat.json",
  "output_dir": "data/runs/habitat-spatial-memory"
}
```

```bash
supernav config show --config configs/local/habitat-spatial-memory.json
supernav stack run \
  --config configs/local/habitat-spatial-memory.json \
  --task-ids example-task --arms default
```

配对会话条件使用相同配置和 `run-paired`：

```bash
supernav stack run-paired \
  --config configs/local/habitat-spatial-memory.json \
  --task-ids example-task --arms default
```

默认情况下，任一入口命令都会启动所需支持服务，并在 benchmark 结束后清理它自己
创建的服务进程。重复运行实验时，可以让描述服务跨单会话和配对会话运行保持常驻：

```bash
supernav stack up --experiment habitat-spatial-memory
supernav stack status
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default
# 后续 `stack run` 或 `stack run-paired` 命令可以复用该服务。
supernav stack logs --follow
supernav stack down
```

`stack up` 启动长期运行的描述服务，而不是 Habitat bridge。`stack run` 为每个普通
episode 启动一个 bridge；`stack run-paired` 分别为探索和检索会话启动新的 bridge。
外部描述服务、进程归属、日志与关闭行为见
[空间记忆指南](docs/release/zh/spatial-memory.md#使用服务栈运行)。

<a id="visualization-and-evaluation"></a>

## 🔁 复用 SuperNav v1 工作流

是的，服务启动后的实验生命周期与 SuperNav v1 相同。以下未被 SuperNav-1.5
改变的操作直接参考 v1 文档：

| 操作 | 指南 |
| --- | --- |
| 任务清单、任务选择、重复次数、arms、run tag 和输出目录 | [SuperNav v1 运行导航实验](../SuperNav_v1/README.zh-CN.md#run-navigation-experiments) |
| HM3D ObjectNav 和 OVON 数据与任务准备 | [SuperNav v1 运行 HM3D ObjectNav](../SuperNav_v1/README.zh-CN.md#run-hm3d-objectnav) |
| 实时查看、运行产物、评分、聚合和视频导出 | [SuperNav v1 可视化与评测](../SuperNav_v1/README.zh-CN.md#visualization-and-evaluation) |
| SuperNav 基线实验与 AI2-THOR | [SuperNav v1 README](../SuperNav_v1/README.zh-CN.md) |

空间记忆配方目前面向 Habitat 后端。运行 `habitat-geo-based-executor` 等 v1
基线配方时，继续使用 `supernav run`；只有所选配方需要 SuperNav-1.5 服务栈时，
才使用 `supernav stack run`。

<a id="documentation-and-development"></a>

## 📚 文档与开发

部署与仿真指南提供 [English](docs/release/en/README.md) 和
[简体中文](docs/release/zh/README.md) 两个版本。

| 指南 | 内容 |
| --- | --- |
| [配置](docs/release/zh/configuration.md) | 实验、模型、本机设置和任务 manifest |
| [场景资产与 Skills](docs/release/zh/assets-and-skills.md) | 后端要求、外部资产和导航 Skills |
| [Habitat-GS](docs/release/zh/habitat.md) | 仿真器安装与场景接入 |
| [HabitatGS-Benchmark](docs/release/zh/habitatgs-benchmark.md) | 公开 GS 数据转换、运行、SR/SPL 与分阶段 SR |
| [HM3D v2 与 OVON](docs/release/zh/hm3d-and-ovon.md) | 场景与任务数据、任务转换、导航运行和离线评分 |
| [AI2-THOR](docs/release/zh/ai2thor.md) | 数据集、Unity 构建、观测和执行 |
| [Unitree Go2](docs/release/zh/go2.md) | Jetson 安装、空间记忆操作、配对会话、源码维护与测试 |
| [Web 查看器](docs/release/zh/web-viewer.md) | 实时观察与已有记录 |
| [架构](docs/release/zh/architecture.md) | 运行时、方法、后端、实验与评测 |
| [目录布局](docs/release/zh/layout.md) | 源码、配置、资产和输出位置 |

开发环境：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

CI 会构建 wheel，并在仅安装基础依赖的独立环境中检查 CLI、配置、提示与 Skill 资源。

<a id="citation"></a>

## 📄 引用

SuperNav-1.5 的引用信息如下：

```bibtex
@misc{guo2026supernav15structuredspatialmemory,
  title={SuperNav-1.5: Structured Spatial Memory For Agentic Navigation Systems},
  author={Jiarui Guo and Jinkai Zhang and Yuanhong Yu and Yongchen Zai and Futang Zhu and Ruizhen Hu and Sida Peng and Xiaowei Zhou},
  year={2026},
  eprint={2611.29912},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2611.29912},
}
```

<a id="acknowledgements-and-license"></a>

## 🙏 致谢与许可证

SuperNav 使用 [Habitat-GS](https://github.com/zju3dv/habitat-gs)、
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) 和
[AI2-THOR](https://github.com/allenai/ai2thor) 作为仿真基础。
感谢这些项目的作者，以及本项目使用的数据集和工具的贡献者。

SuperNav 采用 [Project Registration License（PRL）v1.0](LICENSE)。
仅用于学术研究、教育、评估或个人用途时，可免费使用且无需登记；发表研究结果也无需登记。

组织项目使用同样免费，但须在使用前登记，包括商业与非商业项目、内部研发、测试、部署和生产。
提交完整、准确的登记信息后自动获得许可，无需审批或付费；每个实质不同的项目分别登记一次。

**项目使用登记：**[Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSff5QjjANlJ2hnXYqBSAzKWxEM58gIxPLh17-t3j3-MqO1a5w/viewform)。

第三方代码和随包字体保留其原有许可证及版权声明，详见[第三方许可声明](THIRD_PARTY_NOTICES.md)。
外部仿真器、数据集和模型权重分别遵循其自身许可证。
