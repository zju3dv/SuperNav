[English](../en/tool-authoring-guide.md) | [简体中文](tool-authoring-guide.md)

# 工具开发指南

导航工具由 SuperNav 管理，实现放在 `src/supernav/methods/navigation/tools/`。
平台进程启动与仿真 SDK 绑定放在 `src/supernav/backends/`。新增 SuperNav 工具时，
保持外部 Habitat-GS checkout 只读。职责边界见[架构说明](architecture.md)。

共享注册表为导航 MCP 服务提供 schema 与分发能力。
公开执行入口是 `supernav mcp` 或显式配置的实验。

## 添加工具

1. 为操作选择适合的 `ToolCategory`。
2. 在对应工具模块中定义类，提供 `metadata: ToolMetadata` 和
   `execute(self, args, ctx) -> ToolResult`。
3. 使用 `ToolRegistry.register(YourTool())` 注册一次，并确保
   `src/supernav/methods/navigation/tools/__init__.py` 导入该模块。
4. 针对 schema、bridge 请求、返回证据、错误和上下文限制添加定向测试。
   参考 [MCP 契约测试](../../../tests/contract/v2/test_mcp_inline_rgb_images.py)和相关 runtime 测试。

每个工具使用一个规范名称。重组代码时，将模型可见名称、schema、原生结果结构和结果语义
作为实验契约保留。

## 类型与状态

从规范包导入公共类型：

```python
from supernav.methods.navigation.tools.base import (
    PermissionLevel,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)
```

[`ForwardTool`](../../../src/supernav/methods/navigation/tools/navigation.py)
是一个具体参考：它构造 bridge 请求，采集图像，更新碰撞与运动状态，
并将原始结果保存在 `ToolResult` 内。可参考它当前的 schema 与实现。

| 类型 | 职责 |
| --- | --- |
| `ToolMetadata` | 规范名称、描述、JSON Schema、可用条件和暴露设置 |
| `ToolContext` | Bridge、会话标识、导航与任务模式和本轮状态 |
| `RoundState` | 本轮采集图像、碰撞状态和运动历史 |
| `ToolResult` | `ok`、原始结果体、采集图像、耗时和错误 |
| `ToolRegistry` | 注册、schema 生成、上下文过滤和分发 |

可变会话状态放在 `ctx.round_state` 或其他显式上下文对象中，让共享注册表可供多个调用方复用。
遵循调用方的图像清理与本轮状态生命周期，将数据限定在对应调用内。

主要元数据字段：

- `name`、`category`、`description` 和 `parameters_schema` 定义模型可见 API。
- `allowed_nav_modes` 和 `allowed_task_types` 限制执行上下文。
- `allowed_harness_modes` 区分 forward 与 backward 执行策略。
- 只有建立会话的操作才应使用 `requires_session=False`。
- `mcp_visible=False` 将操作从 MCP 工具发现中排除。
- `permission` 对动作分类。

## Bridge 请求与结果

通过 `ctx.bridge.call(action, payload)` 发送仿真操作。SDK 接入与平台初始化放在后端，
部署路径放在本机配置。根据 SuperNav 后端与外部仿真 API 检查动作和请求字段。

显式返回预期失败：

```python
return ToolResult(ok=False, body={}, error="A session is required")
```

注册表会捕获意外异常，但主动校验能给出更清晰的错误。成功请求应保留原生结果字段，
并通过共享辅助函数采集图像。Bridge 错误通过 `ok=False` 返回。
Agent 声明、会话生命周期事件与基于真值的评分分别记录。

图像与动作历史记录在本轮上下文中。返回采集图像列表的副本，将每次结果保留为独立快照。

## 过滤与 MCP 暴露

`ToolRegistry.available_for(nav_mode, task_type)` 过滤可用工具，
`ToolRegistry.dispatch(name, args, ctx)` 在实际执行时再次检查上下文。
测试应同时覆盖工具发现和在禁止上下文中尝试调用的行为。

导航 MCP 服务位于
[`methods/navigation/mcp_server.py`](../../../src/supernav/methods/navigation/mcp_server.py)，
为对外工具生成 `hab_<tool-name>` 包装函数。`_extract_params` 将 JSON Schema 类型、
必填参数和默认值转换为包装函数签名。`_MCP_RESPONSE_SHAPERS` 在需要时处理约定的结果结构；
修改这些投影时应保留已有的原生结果语义。

将 Agent 操作暴露为工具，辅助数据作为 MCP 资源提供。可复用函数放在共享辅助模块，
后端维护操作保留在 bridge API 中。

通过实验的 `benchmark_profile` 选择任务协议，后端提供对应的内部 `HAB_MCP_GLOBAL_TASK` 值。
实验和 arm 的组合方式见[配置指南](configuration.md)。

## 验证

使用假 bridge 对动作与精确请求字段进行确定性检查。最小 fixture 可以记录调用并返回可控响应：

```python
class FakeBridge:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}
        self.session_id = "s1"

    def call(self, action, payload=None):
        self.calls.append((action, dict(payload or {})))
        return self.responses.get(action, {})
```

检查正常路径、bridge 失败、非法参数、会话前置条件、上下文限制，以及原始结果和图像字段的保留。
涉及协议的变更，应对照已验证契约比较模型可见 schema 和原生输出。
运行相关已有测试，并在安装了基础包的环境中进行导入检查。保留原始工具结果和日志，
便于后续核查。

## 参考模块

- [tools/base.py](../../../src/supernav/methods/navigation/tools/base.py)：类型与注册表。
- [tools/navigation.py](../../../src/supernav/methods/navigation/tools/navigation.py)：运动工具。
- [tools/perception.py](../../../src/supernav/methods/navigation/tools/perception.py)：保留的深度分析与查询工具。
- [tools/status.py](../../../src/supernav/methods/navigation/tools/status.py)：状态更新。
- [tools/session.py](../../../src/supernav/methods/navigation/tools/session.py)：会话校验。
- [tools/_common.py](../../../src/supernav/methods/navigation/tools/_common.py)：视觉请求、图像采集和布尔值解析辅助函数。
- [result_projection.py](../../../src/supernav/methods/navigation/result_projection.py)：结果投影。
- [backends/habitat/](../../../src/supernav/backends/habitat/)：仿真边界与生命周期。
