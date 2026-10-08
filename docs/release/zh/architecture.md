[English](../en/architecture.md) | [简体中文](architecture.md)

# 架构与职责

SuperNav 管理 Agent 执行、导航方法、Skills、实验定义、证据采集和评测。
Habitat-GS 与 AI2-THOR 提供仿真后端。公开入口是 `supernav` 和
`python -m supernav`。文件归属见[目录布局](layout.md)，实验组合规则见
[配置指南](configuration.md)。

```text
supernav
  ├─ config list / show
  ├─ run / run-one
  │    └─ experiments：配置 → manifest → tasks × arms × repetitions
  │         ├─ runtime：Agent、MCP 进程、Skills、事件
  │         ├─ methods：导航工具、提示、策略、记忆
  │         ├─ backends：仿真器进程与协议适配
  │         └─ evaluation：证据、指标、离线评分、回放
  ├─ prepare：任务与实验输入准备
  ├─ score-* / collect-objectnav / aggregate / metrics / timing / video*
  └─ web / viewer：观察与展示
```

## 职责边界

| 层 | 位置 | 职责 |
| --- | --- | --- |
| CLI | `src/supernav/cli.py` | 薄路由；子命令模块管理具体参数和实现 |
| 实验 | `src/supernav/experiments/` | 配置组合、任务选择、sweep、episode 与矩阵编排 |
| 执行 | `src/supernav/runtime/` | 独立于仿真器的 Agent 客户端、MCP 进程描述、Skills 与原始事件 |
| 方法 | `src/supernav/methods/` | 导航工具、提示、策略和记忆 |
| 后端 | `src/supernav/backends/` | 仿真器生命周期、协议绑定、任务适配与原生证据 |
| 评测 | `src/supernav/evaluation/` | 离线评分、测量、轨迹、视频与回放 |
| 展示 | `src/supernav/web/` | 运行证据的只读可视化 |
| 共享契约 | `src/habitat_contract/` | 不依赖仿真 SDK 的导航与任务协议 |

实验通过 `EnvironmentBackend` 协调平台，Agent 客户端消费通用 `MCPServerSpec`。
后端也会选择方法提示并组合评测证据。仿真 SDK 导入和平台初始化放在后端，部署路径放在
本机配置，矩阵编排放在实验层。

使用规范的 `supernav.*` 导入和共享 `habitat_contract` 包。
开发与审计辅助脚本位于 `scripts/`，可复用实现应进入规范包。

## 实验生命周期

```text
--experiment <name> 或 --config <file>
  → 通过 extends 组合配置
  → 展开 manifest includes，应用显式本机部署设置
  → 选择任务、arms 和重复次数，保存展开后的输入
  → 冻结 Skills 并验证可见性（native-skill arms）
  → 后端启动仿真器，runtime 启动 Agent
  → 保留原生日志，提取规范事件、指标和回放
  → 后端回收本集进程
```

`experiments/config.py` 处理发现与组合，`arguments.py` 要求显式选择配置。
`sweep.py` 编排批次，`episode.py` 管理单集，`deployment.py` 只重定位明确的路径字段。


每次 sweep 保存 `experiment.resolved.json`、`instructions.resolved.json` 和
`selection.json`；每集保存任务、实验与有效执行配置。离线评分使用该次运行展开的任务快照。
运行输出默认位于 `data/runs/`，也可显式指定其他目录。原始证据与每次运行的代码和配置
来源一同保留。

`config show` 展示文件组合结果，运行命令的覆盖与凭证解析在实验执行时生效。
`--dry-run` 完成运行产物准备后即结束。

客户端超时会记录 `timed_out=true` 和返回码 124；仍执行事件与指标收集，保留错误和原生日志，
CLI 以非零状态退出。Agent 完成声明与基于真值的离线评分分别记录。
AI2-THOR 当前使用 `success_scoring=withheld`，`success` 和 `spl` 为 `null`。

## 配置与任务协议

| 位置 | 内容 |
| --- | --- |
| `configs/experiments/` | 命名实验方案 |
| `configs/agents/`、`models/`、`backends/`、`methods/` | 可复用的客户端、模型、平台与方法片段 |
| `configs/benchmarks/` | 通用提示模板 |
| `configs/runtime/`、`evaluation/`、`providers/`、`viewers/` | 各消费方的策略与设置 |
| `configs/local/` | 本机部署覆盖与外部任务清单路径 |

任务清单、场景与初始位姿标注、评测 GT 通过 `--instructions` 或本机覆盖的
`instructions_file` 从外部提供。每份实验显式提供 `prompts_dir`。重组输入时，保留提示内容、工具 schema、任务顺序和
arm 合并语义。

Habitat 的任务协议由 `benchmark_profile` 声明：`global_task` 选择全局任务协议，
未设置时使用标准协议。提示约束、初始化、工具与指标消费同一 profile。
后端派生内部 `HAB_MCP_GLOBAL_TASK` 值，并让启动的进程遵循声明的 profile。
公开配置通过 `benchmark_profile` 选择协议。

自定义 MCP 进程使用 `command`、`args`、`http_args`、`transport` 和 `environment`。
后端标准启动使用规范 Python 模块。

## 实现位置

Habitat bridge 使用 SuperNav 自有的
[`backends/habitat/adapter.py`](../../../src/supernav/backends/habitat/adapter.py)，
通过 `backends/habitat/simulator/` 绑定公开 `habitat_sim` SDK。
[`habitat_contract/navigation.py`](../../../src/habitat_contract/navigation.py)
定义会话、观察、运动、寻路与射线接口。方法层通过这些契约调用仿真操作，SDK 接入由后端
处理。外部 SDK 的要求见 [Habitat 接入指南](habitat.md)。

导航状态校验、空间记忆和任务约束位于 `methods/navigation/state.py`，共享状态字段在
`habitat_contract/navigation_state.py`。长循环进程、会话与原生证据由后端管理。

| 组件 | 实现位置 |
| --- | --- |
| Agent 客户端与运行辅助 | `runtime/agents.py`、`runtime/support/` |
| 导航工具、记忆与提示 | `methods/navigation/` |
| 学习型局部执行与训练几何 | `methods/localnav/`、`methods/localnav_policy/rollout_geometry.py` |
| Demand-driven 导航 | `methods/demand_driven/` |
| 任务准备 | `experiments/preparation/` |
| ObjectNav 评分与视频 | `evaluation/objectnav/`、`evaluation/video/` |
| Rerun 查看器 | `web/rerun_viewer.py` |

`methods/navigation/` 下的
`passability.py` 处理基于深度的通行判断，`result_projection.py` 投影工具结果，
`mcp_server.py` 管理会话状态、动态 schema 和 MCP 注册。扩展代码时沿这些行为边界组织职责。

## 开发验证

修改入口或配置时，应检查组合规则、错误输入、规范导入、提示等价性、任务协议和两种后端的
执行契约。工具变更还应遵循[工具开发指南](tool-authoring-guide.md)。

运行定向 pytest，并用
[`scripts/check_installed_package.py`](../../../scripts/check_installed_package.py)
从源码目录外验证安装后的 wheel。安装检查在仅有基础包依赖的环境中覆盖 CLI、
配方、提示、Skills，以及使用外部合成任务的 dry-run。

仿真验证应记录实际代码、配置、命令、原始证据与限制。保持外部仿真器 checkout 只读，
同时保留失败和成功尝试。
