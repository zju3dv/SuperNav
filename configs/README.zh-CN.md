# SuperNav 配置

[English](README.md) | **简体中文**

`experiments/` 中的配方是示例模板。运行实验前，需自行准备：

1. 场景资产或数据集，以及兼容的仿真环境。
2. 适配本机的运行配置，包括模型、凭证、路径和服务。
3. 任务指令或清单，以及对应的场景、初始位姿和目标。

GT 按所选方法或评分要求提供。通过 `supernav config list` 查看模板。
可以使用 `--experiment <name>` 配合环境设置和 CLI 选项，也可以将覆盖配置保存到
忽略提交的 `local/`，再用 `--config <path>` 运行。通过覆盖中的 `instructions_file`
或 `--instructions /path/to/tasks.json` 选择自备任务清单。
`benchmarks/` 提供通用提示模板。

| 示例配方 | 用途 |
| --- | --- |
| `habitat-geo-based-executor` | 使用几何执行器和原生 Skills 的 Habitat 导航 |
| `habitat-learned-executor` | 使用学习型局部策略和原生 Skills 的 Habitat 导航；需部署策略服务 |
| `ai2thor-primitive` | 使用原子动作的 AI2-THOR 需求驱动导航 |

每个配方只有一个 `default` arm，可省略 `--arms` 或使用 `--arms default`。
可复用方法片段为 `methods/geo-based-executor.json`、
`methods/learned-executor.json` 和 `methods/primitive.json`。

所有内置配方使用共享片段
[`models/gpt-6-astra-medium.json`](models/gpt-6-astra-medium.json)：
模型为 `gpt-6-astra`，推理设置为 `medium`。凭证保存在环境变量或本机客户端配置中。

组合规则、外部任务输入、本机路径和运行示例见
[实验配置指南](../docs/release/zh/configuration.md)，或浏览
[发布文档](../docs/release/zh/README.md)。
