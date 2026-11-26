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

默认关闭 Rerun。每次启动从空地图开始，PixelGoal 使用本次运行的实时地图执行导航。

## 启动操作端

操作端需要已安装并认证的 Codex CLI。在 `robot.env` 中设置账户可用的 `SUPERNAV_MODEL` 和 `SUPERNAV_REASONING_EFFORT`，并确认 `command -v codex` 能找到 CLI。

另开终端，在同一部署目录执行：

```bash
source .venv/bin/activate
./deploy/bin/operate.sh
```

操作端使用部署目录的 `.venv`，连接 `http://127.0.0.1:9990/mcp`。每个任务加载 DimOS MCP 和两份导航 skill：[PixelGoal 选点](../../../robots/go2/supernav/overlay/dimos/agents/skills/go2-pixelgoal/SKILL.md)负责单段选点，[全局导航](../../../robots/go2/supernav/overlay/dimos/agents/skills/global-navigation-pixelgoal/SKILL.md)负责搜索和路线选择。

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
| ESC / F8 | 请求停止运动，确认停止后按 Enter 继续 |

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
| `pixelgoal_stop` | 取消当前导航 |

导航需要有效地图、位姿和新鲜图像。需要对应点击时刻的图像时，传入接口支持的冻结帧数据。每段结束后读取终态及返回的四向图像，再决定下一步。

地图尚未覆盖目标时，每段探索最多 1.5 米，方向贴近点击方向。从可达已知自由区向未知区最多延伸 0.5 米；下一段进入未知区需要前次边界附近新增自由空间证据。

已知障碍挡住目标接近线或停靠点时返回 `blocked_pixel_target`。默认 15 秒内净接近未累计达到 0.1 米时停止并返回 `no_progress`。遇到这些结果后，依据新图像重新选点。

`final_distance_m` 是会话结束时机器人中心到最终停靠点的 XY 平面距离，单位米。结果在结束时冻结；尚未形成有效停靠点或缺少有效位姿时为 `null`。

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
