[English](../en/ai2thor.md) | [简体中文](ai2thor.md)

# AI2-THOR 后端

以 `ai2thor-primitive` 为示例模板。运行实验前，需自行准备：

1. 场景和 episode 数据（包括房屋数据），以及兼容的 AI2-THOR/Unity 环境。
2. 适配本机的运行配置，包括模型、凭证、路径和服务。
3. 任务指令或清单，以及对应的场景、初始位姿和目标。

GT 按所选方法或评分要求提供。可以使用 `--experiment` 配合环境设置和 CLI 选项，
也可以用 `--config` 加载本机覆盖。通过 `--instructions` 或覆盖中的 `instructions_file`
指定任务数据。

`environment.backend` 选择仿真器，`agent` 选择客户端。任务和 arm 应与所选 `--backend`
匹配。SuperNav 通过统一实验 runner 管理任务选择、Agent、Skills、仿真器进程、日志和运行目录。

## 前置条件

使用 Python 3.12，以及支持 NVIDIA/Vulkan 的 Linux 环境。准备以下依赖：

- AI2-THOR **5.0.0**，通过 `ai2thor` extra 安装。
- CloudRendering 构建 **`f0825767cd50d69f666c7f282e54abfe58f1e917`**。将可执行文件和
  构建元数据放到 releases 目录下的
  `thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917/`。
- 单独获取冻结的 demand-driven 数据集，包含 `audit.json`、`test/episodes/*.json` 和
  `test/reproducibility/<episode_id>/house_data.json`。任务、来源记录和房屋数据
  应来自同一份外部数据集。
- 真实模型运行所需的 Agent CLI 及其认证。

数据集采用与模型有关的筛选方式，记录为 `model_conditioned_selection=true`。
报告这组数据的实验结果时，注明其筛选方式。runner 使用任务的原始需求；通过
`--instructions` 提供的选集必须与外部数据集中的任务 ID 和需求文本一致。

## 安装与运行

准备好数据集和 CloudRendering 构建后，从 SuperNav 仓库根目录执行以下命令。
将路径替换为本机可用配置：

```bash
python -m pip install -e '.[ai2thor,agents,evaluation]'
export SUPERNAV_AI2THOR_DATASET=/path/to/frozen-demand-driven-dataset
export SUPERNAV_AI2THOR_RELEASES=/path/to/ai2thor/releases

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task --dry-run \
  --output-dir data/runs/ai2thor-dry

supernav run --experiment ai2thor-primitive \
  --instructions /path/to/tasks/ai2thor.json \
  --task-ids example-task \
  --model gpt-6-astra --output-dir data/runs/ai2thor-experiment
```

将 `example-task` 替换为外部清单中的任务 ID。
使用 `--dry-run` 验证数据集并准备提示和 Agent 配置，这项预检需要外部数据集。
dry-run 和真实运行使用不同输出目录，已有 episode 目录默认跳过。
真实运行时，runner 自动启动并回收 bridge 和 Unity 进程。

配方使用 Codex 和本机客户端的 provider/认证，以及
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json)
中的 `gpt-6-astra` 模型和 `medium` 推理设置。通过 `agent`、`agents` 和 `model`
可选择 Kimi 或 OpenCode，详见[配置指南](configuration.md)。

使用 `--task-ids` 选择任务子集，省略该参数时运行所提供清单中的全部任务。还可使用统一的
`--arms`、`--reps`、`--rep-start`、`--model`、`--codex-provider`、`--run-tag`、
`--output-dir`、`--dry-run` 和 Skill 快照参数。
配方只有一个 `default` arm，可省略 `--arms` 或使用 `--arms default`。
通过 `supernav config list` 查看可用配方。

## 后端配置

创建被忽略的 `configs/local/` 目录，将本机覆盖保存为 `configs/local/ai2thor.json`：

```json
{
  "extends": "../experiments/ai2thor-primitive.json",
  "instructions_file": "/path/to/tasks/ai2thor.json",
  "environment": {
    "backend": "ai2thor",
    "dataset": "/path/to/frozen-demand-driven-dataset",
    "releases": "/path/to/ai2thor/releases",
    "python": "/path/to/simulator-env/bin/python",
    "views": "four",
    "gpu": 0
  },
  "bridge": {"host": "127.0.0.1", "port": 0, "startup_timeout_s": 120}
}
```

通过 `supernav run --config configs/local/ai2thor.json` 加上前述任务/模型参数运行。
`dataset`、`releases`、`python` 也可分别由 `SUPERNAV_AI2THOR_DATASET`、
`SUPERNAV_AI2THOR_RELEASES`、`SUPERNAV_AI2THOR_PYTHON` 设置，显式配置优先。
解释器默认使用 runner 自身的 Python。使用独立仿真器环境时，安装同版本 SuperNav
和 AI2-THOR 依赖。

自定义 MCP 进程使用 `mcp.command`、`mcp.args` 和可选 `mcp.http_args`。
runner 为 Codex/OpenCode 配置 stdio MCP，为 Kimi 配置 HTTP MCP。
bridge 端口设为 `0` 时自动选择空闲端口，也可显式指定可用端口。

## 观察与导航

通过 `environment.views=four` 使用默认的四向观察，也可设为 `front` 使用前视观察。

| 设置 | `views=front` | `views=four` |
| --- | --- | --- |
| Agent 观察 | 前视 RGB | 前/右/后/左 RGB |
| 分辨率 | 640 × 480 | 每张 640 × 480 |
| 视场角 | 垂直 120° | 水平 90°，垂直约 73.74° |
| 相机高度 | 原生相机，可设 `camera_height=1.25` | 固定离地 1.25 米 |
| 动作预算 | 500 次 | 500 单位，每个原子动作或 STOP 各 1 单位 |
| 墙钟预算 | Agent timeout | 仿真初始化后 3600 秒，同时受 Agent timeout 约束 |

四个相机位于同一位置，yaw 偏移分别为 0°、90°、180°、−90°。
观察和资源声明计 0 个动作单位。不同观察模式的视场角不同，结果应分组报告。

每次采集生成 `frame_000005` 这样的 `observation_id`，MCP 将其与方向标签和图像一同
交给模型。同次采集的图片引用为 `frame_000005:front`、`:right`、`:back`、`:left`。
重复 observe 复用当前采集编号，`action_index` 记录动作次数。

配方的 `default` arm 使用 0.1 米移动和 10° 转向/仰俯动作。使用 NoMaD 时，
部署策略服务和 checkpoint，将 arm 的 `movement` 设为 `nomad`，
并配置 `environment.policy_url`。四向 NoMaD 使用
[`fourview_instructions.md`](../../../src/supernav/methods/demand_driven/fourview_instructions.md)，
前视 NoMaD 还要求 `environment.camera_height=1.25`。

方向局部导航先保存所选目标图，再转向目标：right/left 各需九个 10° 步，
back 需十八步。这些转向消耗动作预算，随后策略使用当前前视 RGB 和保存的目标图。
达到预算时 episode 结束，局部到达作为导航事件记录。

选择提供 AI2-THOR 工具的 arm：`ddn_observe`、`ddn_step`、`ddn_local_navigate`、
`ddn_claim_resource` 和 `ddn_stop`。通过原生 Skills 或显式 `agent_instructions`
配置任务提示。

## 实时观测

启动 viewer 和 runner 前，在两个终端设置同一个绝对路径。从相同仓库根目录执行：

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
```

在一个终端启动 viewer：

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

另一个终端运行实验。也可通过 `environment.live_dir` 设置发布目录。
打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，选择 `AI2-THOR · <scene_id>`，
查看图像、动作、XZ 轨迹、方向打点和已记录的 Agent/工具输出。
runner 自动关联 run ID、任务指令和日志，viewer 保留最后画面及每个打点的
采集/方向关联。详见 [Web Viewer](web-viewer.md)。

## 证据与评分

运行产物包括 `run.json`、`prompt.txt`、Agent 原始日志、`canonical.jsonl`、
`command.json`、`metrics.json`、计时汇总及父目录的 `results.jsonl`。后端证据包括
`backend.json`、`simulator.log`、`simulator-process.json`，每次尝试各有一个
`simulator-<attempt>/` 目录。缓存、FIFO 文件、PlayerPrefs 和 Unity 日志也写入运行目录。

原生证据包含 `protocol.json`、数据来源、帧、evaluator 元数据、资源声明、相机同步、
转向费用和终止原因。客户端超时记录 `terminal_status=timeout` 和返回码 `124`；
Agent 在 STOP 前返回时记录 `agent_returned_without_stop`。
启动失败、Agent 错误和超时使 CLI 返回非零，保留已采集的证据并触发进程清理。

AI2-THOR 当前的评测模式为 **`success_scoring=withheld`**，`success` 和 `spl` 均为 `null`。
轨迹、资源声明、局部到达和 STOP 作为执行证据报告，任务 SR/SPL 需要独立且经过验证的评分器。

Agent 工具提供任务需求、RGB 观察和动作结果。对具有 shell/文件权限的 Agent 做正式评测时，
将 evaluator 数据与 Agent 文件系统隔离：house 数据、任务 DAG、候选对象、位姿和
GT 成功反馈保留在评测端。模块职责见[架构指南](architecture.md)，其他指南见
[文档索引](README.md)。
