[English](../en/habitat.md) | [简体中文](habitat.md)

# Habitat-GS 后端

SuperNav 使用 [Habitat-GS](https://github.com/zju3dv/habitat-gs) 的 `habitat_sim` SDK
渲染场景并执行导航动作，负责 Agent、导航工具、实验和评测。

运行实验前，需自行准备：

1. 场景数据集与资产（包括 NavMesh），以及兼容的 Habitat-GS 环境。
2. 适配本机的运行配置，包括模型、凭证、路径和服务。
3. 任务指令或清单，以及对应的场景、初始位姿和目标。

GT 按所选方法或评分要求提供。共享配方是示例模板：可以使用 `--experiment`
配合环境设置和 CLI 选项完成适配，也可以用 `--config` 加载本机覆盖。

以下命令均从 SuperNav 仓库根目录运行。统一命令和配置方式见
[文档索引](README.md)与[配置指南](configuration.md)。

API 客户端的供应商字段与凭据环境变量见[配置指南](configuration.md)。
共享 Codex 配方使用客户端配置和登录态。

## 安装与选择仿真器

SuperNav 需要 Python 3.12。按照上游的
[安装说明](https://github.com/zju3dv/habitat-gs#-install-habitat-gs) 安装 Habitat-GS
及其原生/CUDA 依赖，并准备场景资产。

### 在独立 venv 中安装

上游示例使用 conda。采用 venv 部署时，使用 Python 3.12，并确保 CUDA toolkit 的
`nvcc` 位于 `PATH`，然后从公开仓库安装启用 CUDA 的 SDK：

```bash
python3.12 -m venv .cache/venv-habitat
export SUPERNAV_HABITAT_PYTHON="$PWD/.cache/venv-habitat/bin/python"
"$SUPERNAV_HABITAT_PYTHON" -m pip install --upgrade pip
"$SUPERNAV_HABITAT_PYTHON" -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
"$SUPERNAV_HABITAT_PYTHON" -m pip install --upgrade cmake ninja setuptools wheel

# chumpy 0.70 的安装脚本会导入 pip，先在目标环境中准备该依赖。
"$SUPERNAV_HABITAT_PYTHON" -m pip install --no-build-isolation chumpy

git clone --recursive https://github.com/zju3dv/habitat-gs.git .cache/habitat-gs
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF CMAKE_BUILD_PARALLEL_LEVEL=2 \
  "$SUPERNAV_HABITAT_PYTHON" -m pip install .cache/habitat-gs
"$SUPERNAV_HABITAT_PYTHON" -m pip install '.[agents,evaluation]'
"$SUPERNAV_HABITAT_PYTHON" -m pip check
"$SUPERNAV_HABITAT_PYTHON" -c 'import habitat_sim, supernav; print(habitat_sim.__file__, supernav.__file__)'
```

`chumpy` 0.70 的标准隔离构建可能报
`ModuleNotFoundError: No module named 'pip'`。上述预安装在 venv 中提供其构建依赖，
Habitat-GS checkout 保持原样。`CMAKE_BUILD_PARALLEL_LEVEL` 限制 CPU 编译并发；
PyTorch CUDA wheel 与原生 CUDA toolkit 应符合上游要求。

Agent 和仿真器可以使用独立的 Python 环境。在两个环境中安装同一版本 SuperNav：

```bash
python -m pip install -e '.[agents,evaluation]'
export SUPERNAV_HABITAT_PYTHON=/path/to/habitat-gs-env/bin/python
"$SUPERNAV_HABITAT_PYTHON" -m pip install --no-deps .
```

仿真器环境已有所需依赖时使用 `--no-deps`，否则先安装依赖。
也可以在两个环境中安装同一份构建好的 SuperNav wheel。

安装 Agent CLI 并完成认证。共享 Codex 配方使用客户端的 provider 和登录信息，以及
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json)
中的 `gpt-6-astra` 模型和 `medium` 推理设置。

`environment.python` / `SUPERNAV_HABITAT_PYTHON` 选择 bridge 的解释器，
SuperNav 默认使用该环境中已安装的 SDK。可手动检查 bridge 启动：

```bash
unset SUPERNAV_HABITAT_ROOT
"$SUPERNAV_HABITAT_PYTHON" -B -m supernav habitat-bridge \
  --host 127.0.0.1 --port 18911
```

检查后按 Ctrl+C 停止进程。运行实验时，runner 会自动启动和回收 bridge。

使用特定源码 checkout 时，设置 `SUPERNAV_HABITAT_ROOT=/path/to/habitat-gs`，
或在本机配置中设置 `environment.habitat_root`。在所选仿真器环境中安装
与该 checkout 的 Python SDK 匹配的原生扩展。

## 准备匹配的场景

按所选任务准备场景数据集配置、场景资产、NavMesh、语义信息和初始位姿，
场景 ID 应与任务定义一致。GS 资产准备方式见上游
[资产说明](https://github.com/zju3dv/habitat-gs#-download-gs-asset)。

每个 scene instance 使用 `stage_instance.template_name` 引用 stage。
将本机场景配置副本保存到 `configs/local/`，使 stage、NavMesh、spawn 和 GT
与任务保持一致，并检查任务初始位姿下渲染的 RGB。

## 运行单个 episode

单独准备包含指令、场景、初始位姿和目标的外部任务清单，按方法或评分要求补充 GT。
示例使用 `habitat-geo-based-executor` 配方及原生全局、局部导航 Skills，
通过 oracle 深度和 NavMesh 几何执行视觉点动作。创建被忽略的 `configs/local/` 目录，
将以下配置保存为 `configs/local/habitat.json`：

```json
{
  "extends": "../experiments/habitat-geo-based-executor.json",
  "instructions_file": "/path/to/tasks/habitat.json",
  "deployment": {
    "scene_dataset_config_file": "${SUPERNAV_SCENE_DATASET_CONFIG}"
  },
  "output_dir": "data/runs/habitat"
}
```

```bash
export SUPERNAV_SCENE_DATASET_CONFIG=/path/to/global_navigation.scene_dataset_config.json

supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --dry-run --output-dir data/runs/habitat-dry

supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --model gpt-6-astra --output-dir data/runs/habitat
```

将路径替换为本机可用配置，`example-task` 替换为自备清单中的任务 ID。配方为 `gpt-6-astra` 提供 `medium` 推理设置。
使用 `--dry-run` 做配置预检，生成提示、episode 配置和原生 Skill 快照。
dry-run 和真实运行使用不同的输出目录。已有 episode 目录默认跳过，
再次运行时选择新目录或 `--run-tag`。

Habitat 后端为每个 episode 启动并回收 bridge，启动超时为 120 秒。
通过 `bridge.port` 或 `--bridge-port` 为 runner 指定可用端口。

两个 Habitat 示例都只有一个 `default` arm，可省略 `--arms` 或使用 `--arms default`。

外部 OVON 任务需要匹配的 HM3D/OVON 资产。设置 `deployment.scene_dataset_config_file`
和 `deployment.path_prefixes`，将外部任务中的路径映射到本机数据，详见[配置指南](configuration.md)。
HM3D v2 ObjectNav 和 OVON 的数据准备、任务转换、运行与评分步骤见
[HM3D v2 与 OVON 指南](hm3d-and-ovon.md)。

<a id="learned-executor-weights"></a>

## Learned Executor 权重与策略服务

已发布的 checkpoint 位于
[the0xka1/SuperNav-Learned-Executor](https://huggingface.co/the0xka1/SuperNav-Learned-Executor)。
它包含学习型局部策略；高层 Agent、Habitat-GS、场景资产和任务清单仍需单独准备。

```bash
python -m pip install -U huggingface_hub
hf download the0xka1/SuperNav-Learned-Executor \
  ckpt_latest125.pt CHECKSUMS.sha256 --local-dir /path/to/checkpoints
(cd /path/to/checkpoints && sha256sum -c CHECKSUMS.sha256)
```

策略服务环境使用 Python 3.12，并安装适合设备的 PyTorch、torchvision，以及 NumPy 和 Pillow。
在仓库根目录执行 `python -m pip install -e '.[agents,evaluation]'` 安装 SuperNav。
SuperNav 的 `nomad` 后端会从 checkpoint 读取模型架构设置。

对于 CUDA 12.1，以下示例通过 [PyTorch 官方 wheel](https://pytorch.org/get-started/previous-versions/#v251)
安装 PyTorch `2.5.1` 和 torchvision `0.20.1`：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

使用下载文件的绝对路径启动服务：

```bash
export TORCH_HOME=/path/to/torch-cache
export NAV_LOCALNAV_CKPT=/path/to/checkpoints/ckpt_latest125.pt
python -m supernav.methods.localnav.server \
  --backend nomad --checkpoint "$NAV_LOCALNAV_CKPT" \
  --host 127.0.0.1 --port 18914 --device cuda
```

将占位路径替换为本机可写目录。DINOv2 加载器在首次启动时通过 Torch Hub 下载上游代码和
初始骨干网络权重，因此需要访问 GitHub 和 `dl.fbaipublicfiles.com`，或提前在 `TORCH_HOME`
下准备缓存。该推理路径不要求安装 xFormers。CPU 推理可使用 `--device cpu`。

保持服务运行，在启动 Agent 的终端中设置 URL 并检查服务：

```bash
export NAV_LOCALNAV_URL=http://127.0.0.1:18914
curl --fail "$NAV_LOCALNAV_URL/healthz"
```

策略服务也可通过以下环境变量配置：

| 设置 | 要求 |
| --- | --- |
| `NAV_LOCALNAV_BACKEND` | 在策略服务环境中设为 `nomad`，加载学习型后端 |
| `NAV_LOCALNAV_CKPT` | 策略服务所在机器上的 checkpoint 绝对路径 |
| `NAV_LOCALNAV_URL` | Habitat bridge 可访问的策略服务 URL；默认 `http://127.0.0.1:18914` |

服务的 `GET /healthz` 响应会报告 `backend` 和 `ckpt`。运行实验前，确认其分别为
`nomad` 和下载的 checkpoint。如果 Habitat bridge 位于另一台机器，需将 `NAV_LOCALNAV_URL`
设为它能访问的服务地址，并调整服务的 `--host` 监听地址。使用本机 HTTP 代理时，在
`NO_PROXY` 和 `no_proxy` 中加入 `127.0.0.1,localhost`。
服务默认使用 `debug` 后端，仅用于检验通信协议；
未加载学习型权重的服务不能提供训练好的执行器。
在本机配置中继承 `habitat-learned-executor.json`，准备场景资产和外部任务清单，
再按上文的 episode 命令运行任务。`--dry-run` 只准备运行输入，不会验证权重加载或任务执行。

## 导航设置

- 使用 `benchmark_profile` 选择任务协议，后端在内部设置 `HAB_MCP_GLOBAL_TASK`。
- `HAB_GPU_DEVICE_ID` 选择仿真器 GPU；`HAB_ALLOW_SLIDING=0` 关闭碰撞滑动，
  会话中显式指定的 `allow_sliding` 优先。
- GS 使用场景提供的 NavMesh，默认关闭自动 NavMesh 重建和物理模拟。
- 使用 NoMaD 或 LocateAnything 方法时，准备对应的策略或 grounding 服务及其权重。
- 工具调用使用 `habitat-gs/v1` 协议。离线评测根据任务 GT 计算 SR/SPL，
  Agent 完成声明单独记录。

观测实验见 [Web Viewer 指南](web-viewer.md)，扩展导航工具见
[工具开发指南](tool-authoring-guide.md)。

## 开发检查

安装测试依赖并运行后端检查：

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_public_habitat_backend.py
```

验证安装包时，在 checkout 外的新虚拟环境中安装构建好的 wheel 和基础依赖，
用该环境的解释器运行
[`scripts/check_installed_package.py`](../../../scripts/check_installed_package.py)，
检查 CLI 和打包资源在没有仿真器 SDK 时的运行情况。

验证部署时，记录 Python SDK 的版本和原生扩展的构建信息。
模块职责见[架构指南](architecture.md)。
