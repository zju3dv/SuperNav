# 基准提示模板

[English](README.md) | **简体中文**

本目录为 [`configs/experiments/`](../experiments/) 中的实验配方提供通用提示模板。

- [InteriorGS 单目标导航](global_task/README.zh-CN.md)
- [InteriorGS 多目标导航](multi_global_task/README.zh-CN.md)

任务指令、episode 选集、场景标注和 GT 由外部提供。运行配方前，在忽略提交的
`configs/local/` 覆盖中设置 `instructions_file`，或传入
`--instructions /path/to/tasks.json`。论文任务清单和 GT 不包含在公开发行内容中。

在仓库根目录列出配方并查看设置：

```bash
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

外部任务清单、场景资产、后端依赖和本机设置见
[配置指南](../../docs/release/zh/configuration.md)。
