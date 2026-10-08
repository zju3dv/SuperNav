# InteriorGS 多目标提示

[English](README.md) | **简体中文**

本目录提供 Habitat 多目标任务的提示模板。在本机配置中将 `prompts_dir` 设为
`configs/benchmarks/multi_global_task/prompts`，并选择兼容的导航 Skills 和工具。
通过 `instructions_file` 或 `--instructions` 提供外部任务清单。

每个任务包含 `ordered`、`targets`、`ground_truth`，以及场景和初始位姿设置。
Agent 接收目标序号及描述；目标坐标、距离和参考轨迹用于评测。
Agent 的完成声明与离线 SR/SPL 成绩分别记录。

按照[配置指南](../../../docs/release/zh/configuration.md)准备 Habitat-GS 后端、
场景资产和本机配置。通过 `--task-ids` 选择自备清单中的任务 ID。
