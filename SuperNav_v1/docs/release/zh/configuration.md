# 实验配置

[English](../en/configuration.md) | **简体中文**

`configs/experiments/` 中的共享配方是示例模板。运行实验前，需自行准备：

1. 场景资产或数据集，以及兼容的仿真环境。
2. 适配本机的运行配置，包括模型、凭证、路径和服务。
3. 任务指令或清单，以及对应的场景、初始位姿和目标。

GT 按所选方法或评分要求提供。可以使用 `--experiment <name>` 配合环境设置和
CLI 选项，也可以将本机覆盖保存到忽略提交的 `configs/local/`，再用 `--config <path>` 运行。

```text
configs/
├── experiments/   命名实验配方
├── agents/        客户端选择与超时
├── models/        模型名与推理设置
├── backends/      仿真器与协议设置
├── methods/       导航方法、arms、工具和 Skills
├── benchmarks/    通用提示模板
└── local/         本机覆盖；不提交、不打包
```

```bash
supernav config list
supernav config show --experiment habitat-geo-based-executor
```

## 可用配方

| 配方 | 用途 |
| --- | --- |
| `habitat-geo-based-executor` | 使用几何执行器和原生 Skills 的 Habitat 导航 |
| `habitat-learned-executor` | 使用学习型局部策略和原生 Skills 的 Habitat 导航 |
| `ai2thor-primitive` | 使用原子动作的 AI2-THOR 需求驱动导航 |

公开配方将 `instructions_file` 设为 `null`。通过 `--instructions /path/to/tasks.json`
或本机覆盖中的 `instructions_file` 提供任务清单。论文任务清单和 GT 不包含在发行内容中。
按任务准备匹配的场景资产和外部数据集。每个配方只有一个 `default` arm，
可省略 `--arms` 或使用 `--arms default`。Habitat 配方加载原生 Skills。
几何执行器使用 oracle 深度和 NavMesh 路径；学习型执行器需要策略服务及其权重。
从 [Hugging Face](https://huggingface.co/the0xka1/SuperNav-Learned-Executor)
下载权重，并在选择该配方进行真实运行前完成
[策略服务部署](habitat.md#learned-executor-weights)。

## 模型与凭证

| 设置 | 位置与行为 |
| --- | --- |
| 智能体客户端 | 智能体片段或本机覆盖中的 `agent` 与 `agents.<client>` |
| 模型与推理 | `model.name=gpt-6-astra`；Codex `extra_args` 设置 `model_reasoning_effort="medium"` |
| Codex provider 模式 | 共享配方设置 `agents.codex.provider_mode=user`，使用当前客户端的 provider/auth |
| 本次模型 | `--model` 覆盖配置中的模型名 |
| 凭证 | 环境变量或本机客户端凭证文件 |

所有内置配方统一使用共享片段
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json)：
模型为 `gpt-6-astra`，推理设置为 `medium`。本机覆盖未显式修改时继承这些设置。
`--model` 改变模型名，并保留已配置的推理设置。
API 客户端通过本地实验配置中的 `agents.<client>.provider` 选择供应商；
凭据使用 `env_key` 指定环境变量。配置字段见[配置指南](configuration.md)。

Codex 的 `provider_mode=user` 继续使用客户端配置和登录态。
显式选择 `provider_mode=experiment` 时，在本地实验覆盖中设置
`agents.codex.provider.id`、`base_url` 和 `env_key`；代码不提供默认供应商地址。
可通过 `HAB_BENCH_RELAY_BASE_URL` 覆盖该地址。image-prune proxy 必须设置
`--upstream` 或 `HAB_BENCH_RELAY_UPSTREAM`。


## 本机部署与运行

共享 Habitat 配方使用 `${SUPERNAV_SCENE_DATASET_CONFIG}` 指定场景配置。
创建 `configs/local/habitat.json`：

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "model": {"name": "gpt-6-astra"},
  "output_dir": "data/runs/habitat",
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}",
    "path_prefixes": {
      "/datasets/hm3d_ovon": "${SUPERNAV_OVON_ROOT}"
    }
  }
}
```

将路径替换为本机配置，`example-task` 替换为自备清单中的 `task_id`：

```bash
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/scene_dataset_config.json
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-env/bin/python

supernav config show --config configs/local/habitat.json
supernav run --config configs/local/habitat.json \
  --task-ids example-task --dry-run \
  --output-dir data/runs/habitat-dry

supernav run --config configs/local/habitat.json \
  --task-ids example-task \
  --output-dir data/runs/habitat
```

正式运行前需准备仿真器和数据，见 [Habitat 接入](habitat.md)。dry-run 与正式运行
使用不同目录：默认跳过已存在的 episode。也可通过 `--run-tag` 区分重复运行。
`config show` 展示文件组合，凭证和部署变量保留为占位符。
运行命令应用 CLI 覆盖并解析部署变量。dry-run 完成提示、项目和 Skill 快照准备后结束；
请提供后端所需的任务数据。

已自备兼容 checkpoint 和策略服务时，可继承 `habitat-learned-executor.json`
使用学习型执行器。权重发布前，未持有这些资源的用户请使用 `habitat-geo-based-executor`。
`benchmark_profile` 声明 Habitat 任务协议：`global_task` 启用全局任务协议，
省略时使用标准模式。

OVON 等外部格式需要匹配的场景资产、初始位姿、目标和源 episode。使用
`deployment.path_prefixes` 将清单中的数据路径前缀映射到本机目录。
路径重定位保留指令、目标、初始位姿和 GT 数值。

## 后端进程

共享 Habitat 配方设置 `bridge.per_episode=true`，启动预算为 120 秒。
每个 episode 拥有自己的 bridge 进程；端口在实验或本机覆盖中配置。
导航 MCP 默认使用当前解释器运行 `python -m supernav mcp --transport stdio`。

自定义 MCP 进程使用 `mcp.command`、`mcp.args`、`mcp.http_args`、`mcp.transport`
和 `mcp.environment`。
`environment.python` 选择仿真器解释器；智能体和导航 MCP 使用各自的解释器。

Habitat 后端默认导入该解释器已安装的 SDK。可用 `environment.habitat_root` /
`SUPERNAV_HABITAT_ROOT` 选择公开 SDK checkout，所需的兼容原生扩展应预先安装。
各后端安装方式见 [Habitat](habitat.md) 与 [AI2-THOR](ai2thor.md)。

## 组合规则

1. 顶层 `extends` 接受路径或路径列表，相对路径以声明它的文件为基准。
2. 父文件从左到右合并，当前文件最后覆盖。字典递归合并；列表、标量和 `null`
   整体替换。空字典保留继承的字典。
3. 加载后移除顶层 `extends`。循环继承、缺失父文件和错误类型会报错。
4. `arms.<name>.extends` 引用 arm 名称，在组合后的 arm 表内浅覆盖。
5. 资源路径基于 `workspace_root`；只读资产可以从安装包定位。继承时保留任务、
   场景、输出和 Skill 路径的原值。
6. 每份配方显式声明 `prompts_dir`。
7. `--backend`、`--model` 和 `--output-dir` 等运行参数在组合后生效。

每次运行必须从 `--experiment` 与 `--config` 中选择一个。

## 任务 manifest

任务清单和 GT 保存在公开仓库之外。外部 manifest 可通过有序 `includes` 组合其他文件，例如：

```json
{
  "includes": [
    {
      "path": "./scene-a.json",
      "defaults": {"scene": "scene-a"}
    }
  ]
}
```

每个 `path` 相对于声明它的 manifest。允许递归 includes，并拒绝循环。
`defaults` 填充缺失的顶层任务字段；任务自己的 `spawn` 优先。目标和 GT 对象作为完整字段值处理。
被引用任务保留顺序，随后追加当前 manifest 的 `instructions`。
使用 `--task-ids` 从多场景配方中选择任务。

## 其他配置目录

`configs/runtime/`、`evaluation/`、`providers/` 和 `viewers/` 为各自消费方提供设置，
各自使用独立的文件格式和加载器。
新增需分发的资产要加入 `pyproject.toml`，并检查安装后的 wheel。
整体职责见 [架构](architecture.md) 与 [目录布局](layout.md)。
