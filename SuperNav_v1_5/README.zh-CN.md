<div align="center">

<img src="docs/release/assets/supernav1.5-wordmark.svg" alt="SuperNav-1.5" width="580">

<h1>Structured Spatial Memory For Agentic Navigation Systems</h1>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

<p>
  <a href="https://zju3dv.github.io/SuperNav-1.5/"><img src="https://img.shields.io/badge/Project-SuperNav--1.5-e34b4b" alt="项目主页"></a>
  <img src="https://img.shields.io/badge/Paper-Coming_Soon-lightgrey" alt="论文即将发布">
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

SuperNav-1.5 为导航智能体提供结构化空间记忆：由分区占据地图与智能体维护的
**记忆节点**组成。它将场景回忆连接到可执行路线，帮助智能体高效探索、连续到达
多个目标，并跨会话复用记忆。它是 SuperNav 系列的后续工作；前序方法见
[原始 SuperNav](../SuperNav_v1/README.zh-CN.md)。

## 📢 News

> **[2026-10-21]** 🤖 **Unitree Go2 真机部署**已发布，提供四路 RGB 与 LiDAR 支持的 map-backed PixelGoal 导航。Jetson 安装与使用方法见[部署教程](docs/release/zh/go2.md)。

> **[2026-10-13]** 🎉 InteriorGS **ObjectNav 与 Multi-ObjectNav 测试回合**（各 150 个，含参考轨迹）已在 [HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark) 发布，数据与评测方式见 [benchmark 指南](https://github.com/zju3dv/habitat-gs/blob/main/docs/benchmark/README.md)。

原始场景数据请从 [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) 下载。

公开 `habitat_gs/objectnav/v1` 文件需要先转换才能交给 `supernav run`。
转换、运行、逐动作轨迹导出及 SR/SPL、SR1～SR5 评分见
[SuperNav GS benchmark 指南](docs/release/zh/habitatgs-benchmark.md)。

## 🧭 SuperNav-1.5 是什么？

**SuperNav-1.5** 是面向多种任务与场景的智能体导航系统。预训练多模态模型通过工具调用、
任务进度跟踪和上下文管理完成导航，无需对该模型进行导航任务专用微调。
任务涵盖寻找单个物体、依次访问多个目标，以及完成高层需求。

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
[Unitree Go2 部署源码](docs/release/zh/go2.md) 通过固定版本的公开 DimOS submodule，
提供四路 RGB 与 LiDAR 支持的 map-backed PixelGoal 导航。更多真机演示见项目主页。

<a id="release-status"></a>

## 📦 发布状态

- [x] 智能体运行时、导航工具与 Skills、实验配方及评测
- [x] 结构化空间记忆、定制 Cartographer worker 与 stack 生命周期命令
- [x] Learned Executor 权重（[Hugging Face](https://huggingface.co/the0xka1/SuperNav-Learned-Executor)）
- [x] InteriorGS 单目标/多目标导航 benchmark（[HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark)）
- [ ] AI2-THOR 需求驱动导航 benchmark
- [x] [Unitree Go2 真机部署](docs/release/zh/go2.md)已发布：四路 RGB 与 LiDAR 支持的 map-backed PixelGoal

## 📖 目录

- [发布状态](#release-status)
- [安装](#installation)
- [使用结构化空间记忆运行](#run-navigation-experiments)
- [复用 SuperNav v1 工作流](#visualization-and-evaluation)
- [文档与开发](#documentation-and-development)
- [引用](#citation)
- [致谢与许可证](#acknowledgements-and-license)

<a id="installation"></a>

## 🛠️ 安装

Unitree Go2 请按[部署指南](docs/release/zh/go2.md)初始化公开 DimOS submodule，
在 `robots/go2/` 使用依赖锁安装独立环境。以下命令安装仿真运行时。

需要 **Python 3.12**。首先为 SuperNav 创建独立环境：

```bash
git clone https://github.com/zju3dv/SuperNav.git
cd SuperNav/SuperNav_v1_5

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

空间记忆实验还需要定制 Cartographer worker。请按
[安装与构建指南](docs/release/zh/spatial-memory.md#安装与构建)操作；该文档也说明了
`environment-cartographer.yml` 与 pip 运行时安装的分工。

基础安装支持配置查看、Habitat dry-run 和浏览器演示。按需安装可选依赖：

| 用途 | 安装命令 |
| --- | --- |
| 智能体与评测 | `python -m pip install -e '.[agents,evaluation]'` |
| AI2-THOR 实验 | `python -m pip install -e '.[ai2thor,agents,evaluation]'` |
| Rerun 查看器 | `python -m pip install -e '.[viewer]'` |

使用 Codex、OpenCode 或 Kimi 时，需单独安装并认证相应 CLI。
共享 Codex 配置使用客户端已配置的 provider 和登录态；所有内置配方统一使用
[`configs/models/gpt-6-astra-medium.json`](configs/models/gpt-6-astra-medium.json)
中的 `gpt-6-astra` 模型和 `medium` 推理设置。
凭证存放在环境变量或本机客户端配置中。

API 客户端通过本地实验配置中的 `agents.<client>.provider` 选择供应商；
凭据使用 `env_key` 指定环境变量。配置字段见[配置指南](docs/release/zh/configuration.md)。

Habitat-GS 的安装见 [独立 venv 指南](docs/release/zh/habitat.md#在独立-venv-中安装)，
使用公开仓库的 CUDA 构建并选择专用环境。
bridge 健康响应只证明服务器启动；RGB/深度、动作、关闭和独立 GT 评分需要真实 episode 验证。

<a id="quick-start"></a>
<a id="run-navigation-experiments"></a>

## 🧠 使用结构化空间记忆运行

SuperNav-1.5 与 SuperNav v1 使用相同的任务清单格式和实验运行参数。区别在于
方法配方及其依赖服务：运行空间记忆实验时，应使用 `supernav stack run`，
而不是直接调用 `supernav run`。

发布版本提供两个配方：

| 实验 | 空间记忆配置 |
| --- | --- |
| `habitat-spatial-memory` | 记忆节点、Cartographer 二维 SLAM、占据栅格导航和 Qwen 区域描述 |
| `habitat-spatial-memory-slam-only` | 相同的记忆节点和 SLAM 流程，但关闭区域描述 |

先完成[原生 Cartographer 构建与运行环境安装](docs/release/zh/spatial-memory.md#安装与构建)
和 [Habitat-GS 配置](docs/release/zh/habitat.md)。任务清单使用
[SuperNav v1 相同的格式和数据准备流程](../SuperNav_v1/README.zh-CN.md#quick-start)。
之后在当前目录按本机路径运行：

```bash
export SUPERNAV_HABITAT_ROOT=/absolute/path/to/habitat-gs
export SUPERNAV_HABITAT_PYTHON=/absolute/path/to/habitat-environment/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/absolute/path/to/global_navigation.scene_dataset_config.json
export HAB_GPU_DEVICE_ID=0
export HAB_CARTOGRAPHER_WORKER_COMMAND="scripts/run_cartographer_worker.sh {input_fifo} {occupancy_path}"

supernav config show --experiment habitat-spatial-memory

supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory
```

将 `example-task` 替换为任务清单中的 `task_id`。配方已选择使用 medium
推理强度的 `gpt-6-astra`；`--model`、`--reps`、`--run-tag` 和
`--dry-run` 等普通运行参数会原样转发。

该命令会预检补丁版工作进程、启动图像描述服务并等待其健康，再调用普通实验
运行器。运行器为每个 episode 启动 Habitat bridge，bridge 再创建会话级
Cartographer 工作进程。结束时，服务栈只停止由它启动的图像描述进程。空间记忆
配方还会关闭 SuperNav v1 的有界空间事实账本，避免智能体同时使用两套相互竞争
的记忆。

如需复用本机配置，可创建已被 Git 忽略的
`configs/local/habitat-spatial-memory.json`：

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

不使用图像描述模型时，只需切换配方：

```bash
supernav stack run \
  --experiment habitat-spatial-memory-slam-only \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default \
  --output-dir data/runs/habitat-spatial-memory-slam-only
```

重复运行多个实验时，可以让图像描述服务保持运行：

```bash
supernav stack up --experiment habitat-spatial-memory
supernav stack status
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions /absolute/path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default
supernav stack logs --follow
supernav stack down
```

`stack up` 只启动常驻的图像描述服务，不启动 Habitat bridge；Habitat bridge
由 `stack run` 调用的实验运行器启动。外部图像描述服务、进程归属、日志与关闭
行为见[空间记忆指南](docs/release/zh/spatial-memory.md#使用服务栈运行)。

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
| [Unitree Go2](docs/release/zh/go2.md) | Jetson 安装、PixelGoal 使用、源码维护与测试 |
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

SuperNav-1.5 的引用信息将在论文发布时提供。对于本工作所基于的原始 SuperNav
系统，请引用 [SuperNav 论文](https://arxiv.org/abs/2610.12126v1)：

```bibtex
@misc{zhang2026supernav,
  title={{SuperNav}: An Agentic Navigation System for Any Task in Any Scene},
  author={Jinkai Zhang and Jingyi Xu and Yuanhong Yu and Jiarui Guo and Ruizhen Hu and Hujun Bao and Xiaowei Zhou and Sida Peng},
  year={2026},
  eprint={2610.12126},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.12126v1}
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
