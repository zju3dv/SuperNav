[English](../en/web-viewer.md) | [简体中文](web-viewer.md)

# SuperNav Live Viewer

浏览器观测器以只读方式展示第一视角 RGB、前/右/后/左观察、打点 overlay、世界坐标轨迹，
以及已记录的 Agent 消息和工具调用。

## 体验 Demo

从 SuperNav 仓库根目录执行：

```bash
python -m pip install -e .
supernav web --demo
```

打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，查看程序绘制的房间和示例轨迹。
安装后的 Python 包直接提供 viewer，随包静态资源和字体支持离线使用。
`python -m supernav web` 是等价入口。

继续操作前可按 Ctrl+C 停止 Demo，或使用另一个终端。若要在同一端口启动新 viewer，
请先停止已有进程。

## 观看真实运行

先按 [Habitat 指南](habitat.md)或 [AI2-THOR 指南](ai2thor.md)准备可运行的实验，包括
仿真资产、Agent CLI 和凭证。启动 viewer 和 bridge/实验 runner 前，在两个终端设置
**同一个绝对路径**。两个终端都位于仓库根目录时：

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
export SUPERNAV_LIVE_FPS=5
```

帧率设置可选，默认为 5，允许大于 0 且不超过 30。示例用 `$PWD` 获取仓库根目录；
若两个终端的工作目录不同，请显式填写同一绝对路径。

在 viewer 终端执行：

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

在实验终端按照 [Habitat 指南](habitat.md)完成配置，在 `configs/local/habitat.json`
中指定外部任务清单。将 `example-task` 替换为清单中的任务 ID，然后运行：

```bash
supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --model gpt-6-astra --output-dir data/runs/habitat-live
```

示例从
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json)
继承 `gpt-6-astra` 模型的 `medium` 推理设置。再次运行时使用新输出目录或 `--run-tag`，
因为已有 episode 默认跳过。

`bridge.per_episode=true` 时，新 bridge 会继承采集设置，并附带任务指令、Agent、run ID
和 task ID。手动启动 bridge 时，在启动前设置 `SUPERNAV_LIVE_DIR`；补充该变量后重启
bridge。每个手动启动的 bridge 应使用空闲端口。

手动 bridge 可通过 `SUPERNAV_LIVE_CONTEXT` 提供 JSON，其中包含 `instruction`、
`agent`、`run_id`、`task_id`；未提供时显示场景和通用 Agent 名称。其 `run_id` 应与
episode 的 `run.json` 一致。viewer 用该 ID 关联日志，即使 `metrics.json` 尚未生成
也能读取 Agent 时间线。

`--runs-root` 默认为相对当前工作目录的 `data/runs`。若实验输出到其他位置，应传入
包含 episode 的父目录。live 目录独立于运行证据目录，保存观测快照。

新会话自动出现在选择器中，当前没有选择时会自动连接。多个实验可共享 live 目录，
每个仿真器 session 使用自己的 UUID。

## 远端访问

服务默认绑定 loopback。远端实验可通过 SSH 转发端口：

```bash
ssh -L 8765:127.0.0.1:8765 user@experiment-host
```

随后在本机打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。
`--host` 和 `--port` 可覆盖监听地址。服务没有账户认证，应通过可信网络或 SSH 隧道使用。

## AI2-THOR 观察

`ai2thor-primitive` 也支持 `SUPERNAV_LIVE_DIR`。为 runner 和 viewer
设置同一目录，再选择 `AI2-THOR · <scene_id>`；runner 自动关联 run ID 与 Agent 日志。
前置配置见 [AI2-THOR](ai2thor.md)。

页面显示已渲染的 RGB、任务指令、动作 Activity、水平轨迹和资源声明/NoMaD 打点。
默认 `environment.views=four`：四个面板显示 640 × 480 的前/右/后/左图像，打点关联
所选方向与 capture。`front` 模式下其他方向标注为未采集。思考面板展示已有模型日志，
资源声明和 STOP 显示在 Agent 活动中。

## 控件与状态

- **Pause view / Follow live** 冻结或恢复浏览器画面，Agent 继续运行。
- **Four-direction observation** 同时展示同一次 capture 的图像，可点击放大。
  直接复用交给 Agent 的 panorama。`front_only` 模式下展示前向图像，其他方向标注未采集。
- **Agent's point overlay** 在文件写出后立即展示。页面保留原始
  overlay 像素、方向、坐标、图片引用和采集序号。选择历史打点时同步展示它所依据的
  观察，**Latest** 返回最新观察。
- **Agent reasoning & tool trace** 展示已输出的解释和思考摘要，以及可展开的工具参数、
  状态、结果和来源文件/行号。条目按时间戳和原始事件顺序排列；可筛选思考或工具，
  匹配的调用可跳转到 overlay。日志每秒增量读取，界面保留最近 300 项。最新 Agent/
  思考消息也显示在实时图片下方。
- **Trajectory** 将采样的世界坐标显示为橙色路径，配有空心起点、朝向箭头以及缩放与适配控件。
  X 向右、Z 向下，单位为米。
- **Activity** 展示动作开始、结果与错误，保留最近 100 项。开启 live capture 后，
  长动作内部也可更新 RGB/位姿。
- **连接状态** 区分等待首帧、采集失败、正常关闭与 producer 消失。SSE 自动重连并取得
  最新完整快照，断线时保留最后画面。
- **完成与评分** 展示 Agent 声明和会话状态。`closed` 表示会话关闭。
  `metrics.json.success` 是运行元数据，基于任务真值的 SR/SPL 由离线评分得到。

## 浏览已有记录

```bash
supernav web \
  --visuals-root path/to/experiment/visuals \
  --runs-root path/to/experiment/episodes
```

多个目录可重复传入 `--visuals-root`。各目录顶层应有 `<session_id>.trajectory.json`
sidecar，`<session_id>/` 子目录应有 RGB PNG。支持 `step*_color_sensor.png`、
`pano_front_step*_color_sensor.png` 和 LocalNav 帧。深度图暂不支持。四向面板按 capture
分组读取 `pano_{front,right,back,left}_step*.png`，优先采用表示 Agent 实际观察的
`_agent` 图片。通过该会话的 `<session_id>.benchmark_audit.jsonl` 匹配 overlay。
RGB 按文件时间排列，可拖动或以 8 fps 播放。

`--runs-root` 通过 `run.json`、`metrics.json` 和/或 `manifest.json` 关联任务指令、
客户端和 Agent 声明。时间线读取 `canonical.jsonl`，运行中增量读取 `raw.jsonl`，支持
Codex、Kimi 和 OpenCode 格式。Codex 可从该 run 的 `codex_session.jsonl` 或隔离的
`codex_project/.codex_home/sessions/` 补充已输出的思考和时间戳。Kimi 优先读取该 run
隔离的 `kimi_project/kimi_home/.kimi-code/sessions/` 中主 Agent 的 `wire.jsonl`。

时间线排除图片二进制、密钥字段与加密思考；过长输出会截断展示，完整证据仍保留在
原始日志中。历史浏览的轨迹和统计展示整个会话的最终值，RGB 滑条切换图片与关联的
观察/打点，暂不支持逐帧精确位姿重建。未启用 live capture 的 sidecar 在工具完成时更新。

## 采集限制与证据

采集默认关闭。开启后 Habitat 按设置的采集频率渲染，AI2-THOR 复用已有 RGB；发布的
JPEG 最多缩放至 960 × 720。采集可能增加墙钟耗时，严格性能对比应统一采集设置或关闭采集。

每个 session 替换最新 JPEG/JSON，最多保留 12 组四向 PNG、24 张 overlay、4000 个
轨迹采样点和 100 项动作事件；更旧的 live 副本自动清理。完整历史仍在原始图像产物和
日志中。图像失败时保留上一帧并报告采集错误。

高频更新时图片可能比位姿快一个采样。精确时序以原始证据为准，评测使用离线评分器。

采集独立于浏览器运行。关闭后保留最终快照，清理 live 目录会移除相应会话。
producer 存活检查适用于同一主机；跨主机共享目录显示记录的状态和最后画面时间。

## 开发检查

在源码 checkout 中安装测试依赖后运行：

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_web_viewer.py tests/test_harness_bridge_lifecycle.py tests/test_supernav_boundary.py tests/demand_driven/test_live.py
```

这些检查覆盖 SSE/重连、生命周期和采集错误、工具结果与动作数完整性、图像与 overlay
关联及保留边界、增量日志、资源路径、已有记录与 claim/GT 分离。字体许可证位于随包提供的
[`static/fonts/`](../../../src/supernav/web/static/fonts)。

viewer 位于 [`src/supernav/web/`](../../../src/supernav/web)，采集发布器位于 runtime 与
仿真后端。更多说明见[架构](architecture.md)、[目录布局](layout.md)与[文档索引](README.md)。
