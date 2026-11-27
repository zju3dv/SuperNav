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

这组测试检查源码组装、审计 replacement 声明、安装器预览、启动参数、prior-run 参数传递、快照地图保存默认值、进程清理和 release profile。审计回归测试会故意遗漏 canonical skill、修改共享依赖及 blueprint 保存设置，验证这些漂移能被检测。测试使用本地临时目录和模拟进程。

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
  runtime/dimos/dimos/agents/mcp/test_mcp_adapter.py \
  runtime/dimos/dimos/core/coordination/test_supernav_snapshot_stop.py \
  runtime/dimos/dimos/mapping/relocalization/test_module.py \
  runtime/dimos/dimos/navigation/replanning_a_star/test_global_planner.py \
  runtime/dimos/dimos/navigation/replanning_a_star/test_controllers.py \
  runtime/dimos/dimos/mapping/occupancy/test_habitat_area_segmentation.py \
  runtime/dimos/dimos/mapping/occupancy/test_agent_map.py::test_render_agent_occupancy_map_draws_anchored_graph_node_badges \
  runtime/dimos/dimos/mapping/occupancy/test_agent_map.py::test_render_agent_occupancy_map_displaces_overlapping_graph_badges \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_status_attaches_four_images_once \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_slam_map_rejects_odom_in_different_frame \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_slam_map_requires_robot_pose \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_graph_overlay_suppresses_area_label_inside_region \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_agent_navigation_result_hides_internal_slam_geometry_without_changing_image \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_graph_overlay_uses_grid_row_without_vertical_mirroring \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_proxy_returns_slam_map_with_graph_route_context
```

这些 release-critical 测试覆盖当前 Habitat 风格分割、原生 grid-row 方向、重复区域标签抑制、terminal 和 route-context 地图附件、导航专用请求超时、安全目标选择、prior 触发重定位，以及停止前同步保存状态。持久节点 badge 保持在实测 SLAM 坐标处；代理不会为避让标签而移动节点。`--noconftest` 避免加载上游整仓 fixture；代理测试使用模拟 MCP 服务。

机器人 skill fixture 还在不控制硬件的情况下检查地图/位姿验证、静态图、prior 上报、运动记录与重定位 bundle 保存：

```bash
runtime/test-venv/bin/python -m pytest \
  --noconftest -c /dev/null -p no:cacheprovider \
  runtime/dimos/dimos/agents/skills/test_unitree_skill_container.py \
  -k 'memory_navigation or static_slam_snapshot or fresh_map_snapshot or map_attachment_rejects or prepare_stop or report_relocalization_prior or memory_node_navigation'
```

## 与批准快照比较

维护者可将 `SNAPSHOT_DIR` 设为包含 `supernav/manifest.json` 所列 base commit 和三个批准 backport 的完整源码树，不要直接使用任意更新版本。重建并安装 runtime 后执行：

```bash
runtime/test-venv/bin/python -m supernav.audit_snapshot \
  --snapshot "$SNAPSHOT_DIR" --behavior \
  --report runtime/snapshot-audit.json
```

审计比较所有共享的生产 Python/C++ 模块、canonical skill 资源、保留的支持定义和 map-only blueprint，逐项检查声明过的包名/路径适配。随后在两个独立进程中比较实际生成的 prompt 和五份 skill、分割标签及原始/标注 PNG 哈希、工具可见性与地图附件规则、持久/临时目标解析、十种记忆节点导航结果、caption 编码/批量请求/选择/保存、48 个 Python/C++ A* 用例、强制 prior 触发和 prior 裁剪，以及停止顺序。必须构建 C++ 扩展，不能静默使用 Python fallback。

测试不会控制机器人或调用模型服务。传输、规划完成状态、时钟和保存位置受控，其余实际实现正常运行；它不验证真实 SLAM/ICP 收敛、相机标定或物理导航。仓库内的 `supernav/snapshot-audit.json` 记录本次离线审计；上述命令输出完整的本地报告。

本次记录通过 914 项源码/资源检查、20 组差分用例、43 个文档功能测试、9 个机器人 skill 测试及 28 个打包/部署/审计测试。更广泛比较中，191 个共享快照测试的结果完全一致：两边均为 166 通过、25 失败。另一个快照独有的 policy-backend 选择测试不属于本 release 的 map-only 范围。快照中也有与当前实现不一致的旧断言，例如旧 door-node schema 或错误假定的 grid 默认分辨率；加载整仓 fixture 时还可能出现运动记录线程退出的时序错误。应先对同一快照比较失败项，不要为了通过旧断言改变生产行为。本次使用本地 Python 3.12 CPU 测试环境，不代表从依赖锁全新安装或完成实机验证。

检查 CLI、blueprint 和原生扩展：

```bash
runtime/test-venv/bin/python -m dimos.robot.cli.dimos --help
runtime/test-venv/bin/python -c 'import importlib; importlib.import_module("dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal"); importlib.import_module("dimos.navigation.replanning_a_star.min_cost_astar_ext")'
```

上述环境变量用于本地测试。运行机器人时打开新终端，按 [使用教程](go2-usage.md) 启动部署环境。

## 首次安装后的功能检查

在设备独占使用期间，按[使用说明](go2-usage.md)启动栈和操作端：

1. 确认前、右、后、左四路图像方向、分辨率和标定与相机安装一致，图像持续更新。
2. 检查机器人状态、电量、位姿和实时地图。上游 MCP 提供导航、状态和停止工具；agent-facing authority proxy 隐藏 `pixelgoal_stop`、`stop_movement` 及 operator-only 地图/prior 工具。PixelGoal 的 terminal 结果包含区域分割 PNG 与 `2d_slam_map` metadata。
3. 确认 memory badge 锚定实测位姿、纵向方向正确，重复的临时区域标签被抑制；不要为了图面整洁移动持久节点 badge。
4. 在开阔场地选择近处可通行地面，观察目标方向、到达状态和停止行为；测试过程中保持硬件停止手段可用。
5. 检查 `pixelgoal_stop`、ESC/F8、地图重置和退出后的进程清理。
6. 第二次会话使用第一次运行的 `--prior-run-root`，确认 operator 报告 memory-node prior、重定位收敛，并且返回的带标注地图使用当前会话位姿。
7. 使用 `1008` 或 Rerun 时，分别检查对应运动接口或可视化连接。
