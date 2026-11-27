# SuperNav-1.5：面向智能体导航系统的结构化空间记忆

[English](../en/spatial-memory.md) | **简体中文**

SuperNav-1.5 是 SuperNav 的独立后续工作，并非小幅功能叠加。该方法将智能体
编写的**记忆节点**、Cartographer 二维建图、占据栅格导航、几何区域分割、
冻结地图延迟重定位和可选的 Qwen 图像描述证据统一为空间记忆系统。

发布材料统一使用“空间记忆”指代完整方法，使用“记忆节点”指代持久化的区域
和门节点。源码中为保持迁移保真度而保留的 `memory_slam`、`graph_memory` 等
Python 或协议标识仅属于内部实现，不是方法名称。

## 安装与构建

必须从锁定的子模块源码构建 Cartographer 和工作进程，不能使用系统中已有的
Cartographer 或旧工作进程：

```bash
git submodule update --init third_party/cartographer
conda env create -f environment-cartographer.yml
conda activate supernav-cartographer
SUPERNAV_CARTOGRAPHER_BUILD_JOBS=4 scripts/build_cartographer_worker.sh
conda deactivate
```

构建脚本会检查锁定的源码版本、幂等应用发布补丁、将 Cartographer 安装到
`build/cartographer-prefix`、将工作进程构建到 `build/cartographer-worker`，并运行
原生测试和能力检查。该构建环境刻意不包含预编译 Cartographer 包。构建完成后
不要删除此环境，因为工作进程会从中解析原生动态库。需要自定义输出目录时可设置
`SUPERNAV_CARTOGRAPHER_BUILD_DIR`、`HAB_CARTOGRAPHER_PREFIX` 和
`SUPERNAV_CARTOGRAPHER_WORKER_BUILD_DIR`。

随后在能够运行 Habitat-GS 的现有 Python 环境中安装 SuperNav；运行环境与上述
原生构建环境相互独立：

```bash
conda activate <habitat-gs-runtime-environment>
python -m pip install -e '.[agents,spatial-memory,caption]'
export HAB_CARTOGRAPHER_WORKER_COMMAND="scripts/run_cartographer_worker.sh {input_fifo} {occupancy_path}"
```

公开实现只接受 `simulator_evaluator` 里程计。默认使用补丁提供的
`authoritative` 模式；`cartographer_input` 仅作为显式兼容选项。

## Habitat GPU 与 EGL 配置

SuperNav 应使用可正常运行 Habitat-GS 的同一个 Python 环境，并显式选择渲染
设备：

```bash
export SUPERNAV_HABITAT_ROOT=/path/to/habitat-gs
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-gs-environment/bin/python
export HAB_GPU_DEVICE_ID=0
```

`HAB_GPU_DEVICE_ID` 是经过 `CUDA_VISIBLE_DEVICES` 过滤后桥接进程可见的逻辑
设备编号。Habitat-GS 并未设置特殊 EGL 模式：启动器继承进程环境，只设置
`HABITAT_SIM_LOG`、`MAGNUM_LOG`，SDK 配置选择 GPU 0；SuperNav 现在遵循同一
约定。

进程还必须能访问 NVIDIA 驱动和设备文件。在禁止 GPU 设备访问的沙箱内运行
模拟器探针，即使宿主机配置正确，也可能报告 EGL/CUDA 设备错误。GPU 验证应
在正常 Habitat 环境中运行，或在显式暴露所选 NVIDIA 设备的容器中运行。

## 图像描述服务

模型权重不随项目发布。服务可使用 Hugging Face 模型标识或本地路径：

```bash
python -m supernav.backends.habitat.memory_slam.qwen_caption_server \
  --model Qwen/Qwen3.5-2B --port 8926
export HAB_SLAM_AREA_CAPTION_URL=http://127.0.0.1:8926
```

默认以内联字节传图，适用于远程服务或容器。只有共享文件系统时才设置
`HAB_SLAM_AREA_CAPTION_TRANSPORT=path`。启用描述的配置会在健康检查失败时
终止初始化，不会静默降级。

## 使用服务栈运行

`supernav stack` 会预检 Cartographer 补丁工作进程、启动并等待图像描述服务
通过 `/healthz`，再调用现有实验运行器；结束时只停止本次启动且身份匹配的进程：

```bash
supernav stack run \
  --experiment habitat-spatial-memory \
  --instructions configs/local/tasks.json \
  --arms default
```

使用 `--runner run-one` 可转发给 `supernav run-one`，使用
`--keep-services` 可保留本次新启动的服务。长期开发会话可使用：

```bash
supernav stack up --experiment habitat-spatial-memory
supernav stack status
supernav stack logs --follow
supernav stack run --experiment habitat-spatial-memory --instructions configs/local/tasks.json --arms default
supernav stack down
```

默认状态、PID 身份和日志写入 `.cache/stack/`。若描述端口已被未登记进程占用，
栈不会直接接管；`down` 只会向 PID 与进程启动身份均匹配的托管进程发送有界
TERM/KILL。Cartographer 仍由每个会话的桥接进程创建，不作为常驻守护进程。

Docker Compose 或远程描述服务使用 `external` 模式，仅验证健康状态，不启动或
停止外部服务：

```bash
supernav stack up --service-mode external \
  --caption-url http://caption-host:8926/caption
```

SLAM-only 配置仍会预检 Cartographer，但不会启动常驻服务。

## 方法配置与数据边界

使用 `habitat-spatial-memory` 实验；不需要描述服务时使用
`habitat-spatial-memory-slam-only`。智能体会获得 `memory-nodes`、`2d-slam`
和 `spatial-memory-navigation` Skills，以及 `hab_*` 记忆节点工具。

这些配置会显式启用结构化空间记忆，并禁用 SuperNav 旧有的有界路口/分支事实
账本。即使误将旧版 `HAB_MCP_SPATIAL_MEMORY` 标志设为启用，运行时也以结构化
空间记忆标志为准。

记忆节点采用 v7 格式，可读取 v1-v6；迁移时丢弃旧邻接/路径字段和私有的
`navigation_goal`。节点导航只使用节点的 SLAM 位姿与实时占据地图，不提供
GT/NavMesh 后备路径。

冻结地图重定位通过内部 `memory_slam.localization` 配置使用延迟几何匹配、
占用/空闲冲突门控和智能体报告的节点先验。阈值仍属实验配置；诊断产物用于
审计接受或拒绝，不应宣称已完成基准校准。
