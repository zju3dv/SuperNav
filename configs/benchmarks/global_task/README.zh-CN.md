# InteriorGS 单目标提示

[English](README.md) | **简体中文**

本目录为
[`habitat-geo-based-executor`](../../experiments/habitat-geo-based-executor.json)
和 [`habitat-learned-executor`](../../experiments/habitat-learned-executor.json)
配方提供提示模板。

单独准备包含指令、场景、初始位姿和目标的任务清单，按方法或评分要求提供 GT。
创建继承所选配方的 `configs/local/` 覆盖，将清单路径设为 `instructions_file`。
通过 `--task-ids` 选择清单中的任务 ID。

准备 Habitat-GS 后端和场景资产后，运行本机配置：

```bash
supernav run --config configs/local/habitat.json
```

覆盖格式和外部任务设置见[配置指南](../../../docs/release/zh/configuration.md)。
配方引用的共享片段定义 Agent、模型和导航方法。
