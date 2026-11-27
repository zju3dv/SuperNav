[English](../en/go2-usage.md) | [简体中文](go2-usage.md)

# Go2 操作教程

完成[安装](go2.md)后，在 `SuperNav/robots/go2/` 执行以下命令。同一台机器人只运行一个栈。启动器不自动站立，操作者需先完成姿态准备。

## 启动机器人栈

```bash
source .venv/bin/activate
./deploy/bin/check.sh
./deploy/bin/start-go2-pixel-goal.sh
```

默认使用 `joystick`。选择 `1008` 接口时运行：

```bash
./deploy/bin/start-go2-pixel-goal.sh 1008
```

两种接口的最低转向角速度分别为 0.2 和 0.5 rad/s，可用 `DIMOS_ASTAR_MIN_ANGULAR_RPS` 覆盖。

启动器等待四路 GoPro bridge 就绪，再启动 `unitree-go2-pixelgoal`，固定使用 `mapping` 和 `local-planner`。GoPro 视频经 ffmpeg 写入 `/dev/video42–45`，MCP 使用端口 9990。

默认关闭 Rerun。默认设置下不传 `--prior-run-root` 时，每次启动从空地图开始；正常停止时保留快照默认启用的 voxel-map 保存行为。运行期间，SuperNav-1.5 会对占据地图进行区域分割，把带标注的地图附加到导航工具结果，并根据直接观测维护记忆节点。

## 启动操作端

操作端需要已安装并认证的 Codex CLI。在 `robot.env` 中设置账户可用的 `SUPERNAV_MODEL` 和 `SUPERNAV_REASONING_EFFORT`，并确认 `command -v codex` 能找到 CLI。

另开终端，在同一部署目录执行：

```bash
source .venv/bin/activate
./deploy/bin/operate.sh
```

操作端使用部署目录的 `.venv`，连接 `http://127.0.0.1:9990/mcp`。每个任务会在工作区写入与实际源码一致的运行时 skill：PixelGoal 选点、全局导航、空间记忆层级、2D SLAM 解读，以及记忆与 SLAM 路线融合。任务审计会记录 prompt 与 skill 快照。

SuperNav-1.5 的空间记忆独占本次运行的空间状态；原 SuperNav 的有界空间事实账本不会同时启用。

## 第二次会话复用记忆

正常关闭时，操作端会完成当前 SLAM 重定位 bundle。复用已完成的运行时，机器人栈和操作端必须传入同一个历史任务目录：

```bash
./deploy/bin/start-go2-pixel-goal.sh --prior-run-root /absolute/path/to/prior-run
```

在操作端终端运行：

```bash
./deploy/bin/operate.sh --prior-run-root /absolute/path/to/prior-run
```

历史运行必须包含 `logs/go2-graph-memory.json` 与 `slam/relocalization/manifest.json`。机器人栈加载已保存地图并启用重定位；操作端加载历史记忆节点，并可在视觉证据充分时把可信节点作为重定位先验。历史运行不完整时会明确报错，不会静默启动残缺的记忆会话。

## 可选区域描述服务

在 `deploy/config/robot.env` 中把 `GO2_SLAM_AREA_CAPTION_URL` 设置为兼容的 `/caption` 服务。关键帧异步入队，服务不可用或失败不会阻塞导航。留空即可关闭描述服务。

### 模型服务代理

需要代理时，在操作端终端设置。将 `PROXY_HOST:PORT` 替换为可访问的 HTTP 代理地址：

```bash
export HTTPS_PROXY=http://PROXY_HOST:PORT
export HTTP_PROXY="$HTTPS_PROXY" ALL_PROXY="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1
./deploy/bin/operate.sh
```

启动脚本会将机器人 IP 加入 `NO_PROXY`。机器人栈使用独立终端。

### 从工作站操作

工作站需组装相同源码并安装[工作站依赖](go2-testing.md)，通过 SSH 转发访问 Jetson MCP；将 `user@jetson-host` 替换为实际 SSH 用户与主机：

```bash
ssh -N -L 9990:127.0.0.1:9990 user@jetson-host
```

`operate.sh` 固定使用部署目录的 `.venv`；在工作站安装时，将测试教程中的环境路径改为 `.venv`。

## 控制台操作

| 按键 | 操作 |
| --- | --- |
| F2 | 进入 Agent 模式，输入导航任务 |
| F7 | Pixel Goal 面板：`refresh`、`reset`、`stop`；图像显示和点击依赖 Rerun |
| ESC / F8 | 仅操作者可用的停止操作；确认停止后按 Enter 继续 |

Agent 从四向图像选择归一化目标点，每段导航完成后根据新图像继续。优先选择可通行地面；稳定不透明立面按 PixelGoal skill 的条件选择。

控制台每 2 秒异步读取机器人状态和电量：

- `ROBOT STATUS OK`：最近一次低层状态读取成功。
- `UNVERIFIED`：MCP 已连接，低层状态尚未读取成功。
- `STALE` / `UNAVAILABLE`：状态读取失败，旧电量已清除。

状态查询与输入、停止操作独立。姿态和运动环境由操作者确认。

## PixelGoal 接口与结果

| MCP 工具 | 用途 |
| --- | --- |
| `pixelgoal_navigate` | 提交归一化图像点 `[x, y]`（0–1）与 `direction`（front/right/back/left） |
| `pixelgoal_status` | 查询导航状态和结果 |
| `pixelgoal_stop` | 仅操作者可用，取消当前导航 |

导航需要有效地图、位姿和新鲜图像。Agent-facing `pixelgoal_navigate` 只接受 `point` 与 `direction`，等待执行结束后返回终态，正常操作不轮询 status。冻结帧参数属于上游 operator 接口，不属于 agent schema。每段结束后读取终态及新鲜图像，再决定下一步。

地图尚未覆盖目标时，每段探索最多 1.5 米，方向贴近点击方向。从可达已知自由区向未知区最多延伸 0.5 米；下一段进入未知区需要前次边界附近新增自由空间证据。

已知障碍挡住目标接近线或停靠点时返回 `blocked_pixel_target`。默认 15 秒内净接近未累计达到 0.1 米时停止并返回 `no_progress`。遇到这些结果后，依据新图像重新选点。

`final_distance_m` 是会话结束时机器人中心到最终停靠点的 XY 平面距离，单位米。结果在结束时冻结；尚未形成有效停靠点或缺少有效位姿时为 `null`。

### 地图附件、记忆目标与工具可见性

地图附件遵循实现快照，并受地图/位姿可用性限制；返回失败 metadata 不代表一定附图。

| 工具/结果 | 地图图像 | 其他图像 |
| --- | --- | --- |
| Agent `pixelgoal_navigate` 终态 | 一张完整区域分割占据图，由代理叠加记忆标注 | 新鲜的前/右/后/左视图 |
| `pixelgoal_status(include_images=true)` | 导航终态地图；运行中不附图。健康的终态图像通常每目标只返回一次，caption pending 时可刷新 | 新鲜四向视图 |
| `go2_navigate_to_memory_node` | 一张 canonical 分割图；导航失败但地图/位姿有效时仍可返回 | 可用的最新相机图；代理还会请求新鲜四向视图 |
| 普通运动（`drive`、`relative_move`、`rotate_in_place`） | 清理后的 floor plan、区域分割图和机器人中心占据图 | 可用的最新相机图 |
| `go2_query_route_context` / `go2_report_graph_position` | 可用的最新缓存标注图；route query 先刷新地图 | 按请求返回存储节点全景；position report 包含全景 |
| `go2_inspect_graph_node` | 不附 SLAM 图 | 按请求返回存储节点全景 |
| Map-context/status 读取、graph 写入、普通观测 | 不额外附 SLAM 图 | 对应工具的观测 |

Route query 的 `return_images=false` 只关闭存储全景，不关闭 SLAM 图。仅 operator 可用的 `go2_slam_map_snapshot` 和 `go2_report_relocalization_prior` 也可返回完整分割图；代理对 agent 隐藏这两个工具、地图重置、`pixelgoal_stop` 和 `stop_movement`。运动工具可见性受 operator 批准的当前 envelope 限制。

持久记忆节点的数字 badge 固定在实测 SLAM 位姿，不为避让标签而移动；同一原生 grid 区域已有持久节点时，抑制临时 `A<N>` 标签。原生 grid 行只在显示时翻转。记忆节点导航解析存储节点或当前临时区域目标，要求地图/位姿当前有效且坐标系一致，调用原生规划器，不回退到模拟器 NavMesh。超时和规划拒绝会明确返回，不视为到达成功。

## 可选 Rerun

在 `robot.env` 中设置并重启栈：

```bash
SUPERNAV_RERUN=1
SUPERNAV_RERUN_CORS_ORIGIN=http://127.0.0.1:9878
```

启动器运行端口 9877 的 gRPC viewer，端口 3030 用于可视化控制 WebSocket。浏览器 viewer 另行启动，CORS origin 与浏览器实际来源保持一致。

## 停止与日志

先通过控制台或 `pixelgoal_stop` 停止运动，确认停止后，在启动终端按 Ctrl-C 退出。启动器回收本次创建的进程组；任一主进程退出时，同次启动的其余进程也会结束。自动趴下和硬件急停需通过机器人的相应控制执行。

| 路径 | 内容 |
| --- | --- |
| `runtime/logs/<时间>/` | `gopro.log`、`dimos.log`、可选 `rerun.log` |
| `runtime/pixelgoal-live/` | PixelGoal 帧、状态和事件 |
| `runtime/gopro-bridge/` | 相机桥状态 |
| `runtime/state/` | 运行注册表 |
| `runtime/operator-runs/` | 操作端任务产物 |
| `runtime/operator-runs/<run>/logs/go2-graph-memory.json` | 空间记忆节点与观测记录 |
| `runtime/operator-runs/<run>/slam/relocalization/` | 后续会话使用的地图与重定位清单 |
| `runtime/dimos/logs/` | 使用源码默认路径生成的 DimOS 日志 |

## 故障定位

| 现象 | 检查 |
| --- | --- |
| 源码校验失败 | `git -C dimos status`、`python3 -m supernav.prepare --inputs-only` |
| 生成源码过期 | 停止栈后按[维护说明](go2-maintenance.md)重建并重装 |
| 缺少 video42–45 | loopback 模块、内核 headers、video 组 |
| 相机启动超时 | USB DHCP、序列号顺序、相机模式、`gopro.log` |
| 缺少硬件 decoder | NVIDIA ffmpeg 的 `h264_nvv4l2dec`；选择软件解码后检查四路性能 |
| 端口或视频设备占用 | 先停止占用资源的旧栈或视频 producer |
| WebRTC 无法连接 | 机器人 IP、路由、AES key、`NO_PROXY` |
| 目标投影偏移 | 分辨率、Linear 视场、相机内外参 |
| 找不到 Codex CLI | 安装路径；若安装在 `~/.npm-global/bin`，设置 `export PATH="$HOME/.npm-global/bin:$PATH"` |
