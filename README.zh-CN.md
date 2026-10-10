<div align="center">

<h1>
  <img src="docs/release/assets/supernav-wordmark.svg" alt="SuperNav" width="460"><br>
  <sub>An Agentic Navigation System for Any Task in Any Scene</sub>
</h1>

<p>
  <a href="https://arxiv.org/abs/2610.12126v1"><img src="https://img.shields.io/badge/arXiv-2610.12126-b31b1b" alt="arXiv: 2610.12126"></a>
  <a href="https://huggingface.co/papers/2610.12126"><img src="https://img.shields.io/badge/Hugging_Face-Paper-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Paper"></a>
  <a href="https://zju3dv.github.io/SuperNav/"><img src="https://img.shields.io/badge/Project_Page-SuperNav-orange" alt="Project Page"></a>
</p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

<p>
  <a href="https://the0xka1.cc/">Jinkai Zhang</a><sup>1</sup> ·
  <a href="https://echo636.github.io/">Jingyi Xu</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  Jiarui Guo<sup>1</sup> ·
  <a href="https://csse.szu.edu.cn/pages/user/index?id=531">Ruizhen Hu</a><sup>2</sup> ·<br>
  <a href="https://person.zju.edu.cn/0093140">Hujun Bao</a><sup>1</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup> ·
  <a href="https://pengsida.net/">Sida Peng</a><sup>1,3†</sup>
</p>

<p>
  <sup>1</sup> Zhejiang University &nbsp;&nbsp; <sup>2</sup> Shenzhen University &nbsp;&nbsp; <sup>3</sup> Causa Robotics<br>
  <sup>†</sup> 通讯作者
</p>

<h3>Overview Video</h3>

<video src="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d" controls width="800"></video>

<p>
  <a href="https://zju3dv.github.io/SuperNav/">在项目主页观看</a> ·
  <a href="https://github.com/user-attachments/assets/d415a2ec-89de-4f6d-a2e5-eeec8993d05d">打开 MP4</a>
</p>

</div>

## 🧭 SuperNav 是什么？

**SuperNav** 是面向多种任务与场景的智能体导航系统。预训练多模态模型通过工具调用、
任务进度跟踪和上下文管理完成导航，无需对该模型进行导航任务专用微调。
任务涵盖寻找单个物体、依次访问多个目标，以及完成高层需求。

本仓库提供用于 **Habitat-GS** 和 **AI2-THOR** 的智能体运行时、导航工具与 Skills、
实验配置和评测工具。使用 SuperNav 可以：

- **运行导航智能体**：支持 Codex、OpenCode 和 Kimi。
- **组合任务、工具与 Skills**：通过显式实验配置控制智能体能力。
- **实时观察执行**：查看 RGB 观测、运动轨迹和工具调用。
- **检查并评测运行结果**：保留提示、工具结果、原生日志和独立任务指标。

[项目主页](https://zju3dv.github.io/SuperNav/) 提供仿真视频、方法介绍和研究结果。
真机演示见项目主页；当前仓库不包含真机驱动。

<a id="release-status"></a>

## 📦 发布状态

- [x] 智能体运行时、导航工具与 Skills、实验配方及评测
- [x] Learned Executor 权重（[Hugging Face](https://huggingface.co/the0xka1/SuperNav-Learned-Executor)）
- [ ] InteriorGS 单目标/多目标导航 benchmark
- [ ] AI2-THOR 需求驱动导航 benchmark
- [ ] 真机部署（Unitree Go2）：四路 RGB，基于 LiDAR 的几何执行

## 📖 目录

- [发布状态](#release-status)
- [安装](#installation)
- [快速开始](#quick-start)
- [运行导航实验](#run-navigation-experiments)
- [可视化与评测](#visualization-and-evaluation)
- [文档与开发](#documentation-and-development)
- [引用](#citation)
- [致谢与许可证](#acknowledgements-and-license)

<a id="installation"></a>

## 🛠️ 安装

需要 **Python 3.12**。首先为 SuperNav 创建独立环境：

```bash
git clone https://github.com/zju3dv/SuperNav.git
cd SuperNav

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

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

## ⚡ 快速开始

启动内置的浏览器演示：

```bash
supernav web --demo
```

打开 **http://127.0.0.1:8765**，查看程序生成的房间和示例轨迹。基础安装已包含演示所需资源。
继续后续操作前按 **Ctrl+C** 关闭演示，或使用另一个终端。

**运行实验前，用户必须自行准备以下三项：**

- **数据**：所选仿真器需要的场景资源和数据集。
- **配置**：本机数据路径、仿真环境、模型访问配置，以及所选方法依赖的服务。
  可基于 `configs/experiments/` 中的示例配方修改。
- **指令**：包含任务指令及对应场景、初始位姿和目标信息的任务清单。

SuperNav 提供代码、通用 prompts 和配置模板。执行评分或使用依赖 GT 的方法时，
还需准备对应的 GT。论文任务清单和 GT 不包含在本次发行中。

查看实验配置示例：

```bash
supernav --help
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

准备好运行配置和任务清单后，将下方 `example-task` 替换为自备清单中的
`task_id`，然后准备 dry-run：

```bash
supernav run --experiment habitat-geo-based-executor \
  --instructions /path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default --dry-run \
  --output-dir data/runs/quickstart-dry
```

`supernav` 与 `python -m supernav` 提供相同的命令树。
运行时必须通过 `--experiment` 或 `--config` 显式选择配置。

<a id="run-navigation-experiments"></a>

## 🚀 运行导航实验

### 运行一个 Habitat-GS episode

安装智能体依赖，然后按照 [Habitat 接入指南](docs/release/zh/habitat.md) 安装 Habitat-GS，
并在仿真器环境中安装同一版本的 SuperNav。智能体与仿真器可以使用不同的 Python 环境。

```bash
python -m pip install -e '.[agents,evaluation]'

export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/global_navigation.scene_dataset_config.json

supernav run --experiment habitat-geo-based-executor \
  --instructions /path/to/tasks/habitat.json \
  --task-ids example-task \
  --arms default --model gpt-6-astra \
  --output-dir data/runs/quickstart
```

将路径替换为自己的配置。配方为 `gpt-6-astra` 提供 `medium` 推理设置。
场景和初始位姿由任务清单提供，所选方法采用视觉点导航及原生全局、局部导航 Skills。
SuperNav 会为每个 episode 启动并关闭 Habitat bridge。

**dry-run 与正式运行应使用不同的输出目录。** 默认跳过已存在的 episode 目录；
再次运行时请选择新目录或指定新的 `--run-tag`。

### 选择实验

[`configs/experiments`](configs/experiments) 中的示例配方组合了共享的智能体、模型、后端和方法设置。
用户需按本机环境修改配置，并提供自己的数据和任务指令：

| 实验 | 用途 |
| --- | --- |
| `habitat-geo-based-executor` | 使用几何执行器和原生 Skills 的 Habitat 导航 |
| `habitat-learned-executor` | 使用学习型局部策略和原生 Skills 的 Habitat 导航 |
| `ai2thor-primitive` | 使用原子动作的 AI2-THOR 需求驱动导航 |

**Learned Executor 权重已发布到
[Hugging Face](https://huggingface.co/the0xka1/SuperNav-Learned-Executor)。**
下载 checkpoint 并校验文件：

```bash
python -m pip install -U huggingface_hub
hf download the0xka1/SuperNav-Learned-Executor \
  ckpt_latest125.pt CHECKSUMS.sha256 --local-dir /path/to/checkpoints
(cd /path/to/checkpoints && sha256sum -c CHECKSUMS.sha256)
```

将 `/path/to/checkpoints` 替换为下载目录。运行 `habitat-learned-executor` 前，
按照[策略服务部署说明](docs/release/zh/habitat.md#learned-executor-weights)
使用 `nomad` 后端加载 checkpoint，并设置 `NAV_LOCALNAV_URL`。
场景资产和任务指令仍需单独准备。

每个配方只有一个 `default` arm，可省略 `--arms` 或使用 `--arms default`。
通过 `--task-ids` 和 `--reps` 选择任务与重复次数。几何执行器使用深度和 NavMesh
路径等 oracle 几何信息，比较方法时需计入这一信息条件。

本机设置放入忽略提交的 `configs/local/`。例如创建 `configs/local/habitat.json`，
指定自备任务清单并定制配方：

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "model": {"name": "gpt-6-astra"},
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}"
  },
  "output_dir": "data/runs/habitat"
}
```

```bash
supernav config show --config configs/local/habitat.json
supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default
```

组合规则、模型选择和外部数据路径映射见 [配置指南](docs/release/zh/configuration.md)。
每份配方都显式声明提示目录。

### 运行一个 AI2-THOR episode

按照 [AI2-THOR 指南](docs/release/zh/ai2thor.md) 准备数据集与 Unity 构建后运行：

```bash
python -m pip install -e '.[ai2thor,agents,evaluation]'
export SUPERNAV_AI2THOR_DATASET=/path/to/frozen-demand-driven-dataset
export SUPERNAV_AI2THOR_RELEASES=/path/to/ai2thor/releases

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task \
  --model gpt-6-astra \
  --output-dir data/runs/ai2thor-example
```

使用外部数据集中的任务 ID 和匹配的原始需求文本。
该后端记录动作、资源声明与 STOP 事件。当前评测模式为 `withheld`，SR 和 SPL 保持 `null`。

<a id="visualization-and-evaluation"></a>

## 🖥️ 可视化与评测

观察真实运行时，请在启动查看器与实验进程**之前**，为两个终端设置相同的绝对路径：

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
```

在一个终端启动查看器，在另一个终端运行实验：

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

查看器展示观测、轨迹以及已记录的智能体和工具活动。
远程访问与已有记录的查看方式见 [查看器指南](docs/release/zh/web-viewer.md)。

运行产物保存在 `data/runs/` 或指定输出目录中，包括配置与任务快照、提示、
智能体原始日志、工具事件和指标。通过以下命令查看评测与导出参数：

```bash
supernav score-objectnav --help
supernav score-multi-objectnav --help
supernav aggregate --help
supernav video --help
```

使用任务对应的离线评分器和真值计算 SR/SPL。智能体完成声明、STOP 事件和进程状态单独记录。

<a id="documentation-and-development"></a>

## 📚 文档与开发

所有发布指南均提供 [English](docs/release/en/README.md) 和
[简体中文](docs/release/zh/README.md) 两个版本。

| 指南 | 内容 |
| --- | --- |
| [配置](docs/release/zh/configuration.md) | 实验、模型、本机设置和任务 manifest |
| [场景资产与 Skills](docs/release/zh/assets-and-skills.md) | 后端要求、外部资产和导航 Skills |
| [Habitat-GS](docs/release/zh/habitat.md) | 仿真器安装与场景接入 |
| [HM3D v2 与 OVON](docs/release/zh/hm3d-and-ovon.md) | 场景与任务数据、任务转换、导航运行和离线评分 |
| [AI2-THOR](docs/release/zh/ai2thor.md) | 数据集、Unity 构建、观测和执行 |
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

如果 SuperNav 对你的研究有帮助，请引用我们的
[论文](https://arxiv.org/abs/2610.12126v1)：

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
