[English](../en/go2-testing.md) | [简体中文](go2-testing.md)

# Go2 测试

本文命令在 SuperNav 的 `robots/go2/` 目录执行。

## 源码与部署脚本

使用 Git 和 Python 3.12 运行：

```bash
python3 -m supernav.prepare
python3 -m supernav.prepare --check
./deploy/bin/check.sh --static
python3 -m unittest discover -s tests -v
```

这组测试检查源码组装、安装器预览、启动参数、进程清理以及 Markdown 资源加载与任务副本，使用本地临时目录和模拟进程。

## 安装测试环境

Linux x86_64 工作站使用 Python 3.12、uv 0.11.11 和 CPU 测试锁。系统依赖包括编译器、PortAudio、TurboJPEG 和图像库，安装方法见 [安装教程](go2.md)。

在空的 `runtime/test-venv` 路径创建环境：

```bash
uv venv --python 3.12 runtime/test-venv
uv pip install --python runtime/test-venv/bin/python --require-hashes \
  -r deploy/requirements-build.lock
uv pip install --python runtime/test-venv/bin/python --require-hashes \
  -r deploy/requirements-test.lock --torch-backend cpu \
  --build-constraints deploy/requirements-build.in
uv pip install --python runtime/test-venv/bin/python --no-deps --no-build-isolation \
  -e ./runtime/dimos
uv pip check --python runtime/test-venv/bin/python
```

本地解包的原生库和头文件可通过 `CFLAGS`、`LDFLAGS`、`LD_LIBRARY_PATH` 提供给构建和测试进程。

## 运行功能测试

在测试终端设置内存 LCM 通道和临时状态目录：

```bash
export DIMOS_SKIP_SYSTEM_CONFIG=1 USE_TF=0 USE_FLAX=0
export LCM_DEFAULT_URL=memq://
export PYTHONDONTWRITEBYTECODE=1
export XDG_STATE_HOME="$(mktemp -d /tmp/supernav-go2-test-state.XXXXXX)"

runtime/test-venv/bin/python -m pytest \
  --noconftest -c /dev/null -p no:cacheprovider \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map.py \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py \
  runtime/dimos/dimos/agents/mcp/test_pixelgoal_handoff_contract.py \
  runtime/dimos/dimos/agents/mcp/test_memory_free_operator.py \
  runtime/dimos/dimos/agents/console/test_robot_status.py
```

测试覆盖地图投影、目标选择、导航会话、blueprint 组成、移动授权、停止、结果交接、工具隔离和控制台状态。`--noconftest` 避免加载上游整机测试 fixture；代理测试使用模拟 MCP 服务。

检查 CLI、blueprint 和原生扩展：

```bash
runtime/test-venv/bin/python -m dimos.robot.cli.dimos --help
runtime/test-venv/bin/python -c 'import importlib; importlib.import_module("dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal"); importlib.import_module("dimos.navigation.replanning_a_star.min_cost_astar_ext")'
```

上述环境变量用于本地测试。运行机器人时打开新终端，按 [使用教程](go2-usage.md) 启动部署环境。

## 首次安装后的功能检查

在设备独占使用期间，按[使用说明](go2-usage.md)启动栈和操作端：

1. 确认前、右、后、左四路图像方向、分辨率和标定与相机安装一致，图像持续更新。
2. 检查机器人状态、电量、位姿和实时地图，确认 MCP 提供 `pixelgoal_navigate`、`pixelgoal_status` 和 `pixelgoal_stop`。
3. 在开阔场地选择近处可通行地面，观察目标方向、到达状态和停止行为；测试过程中保持硬件停止手段可用。
4. 检查 `pixelgoal_stop`、ESC/F8、地图重置和退出后的进程清理。
5. 使用 `1008` 或 Rerun 时，分别检查对应运动接口或可视化连接。
