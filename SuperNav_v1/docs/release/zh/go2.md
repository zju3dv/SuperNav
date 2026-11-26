[English](../en/go2.md) | [简体中文](go2.md)

# Unitree Go2 安装

SuperNav 的 map-backed PixelGoal 使用四路 GoPro 图像、实时 LiDAR 地图和局部规划器执行图像目标导航。安装后继续阅读[操作教程](go2-usage.md)；源码修改和功能测试分别见[维护](go2-maintenance.md)与[测试](go2-testing.md)。

部署环境：Jetson aarch64、L4T R36.4 系列、Python 3.12、四路 GoPro HERO8 USB 网络相机及 NVIDIA ffmpeg。为部署创建独立目录和 Python 环境；运行时独占相机、MCP 端口及机器人 LCM 通道。

## 1. 获取并组装源码

准备 Git、Python 3.12 和 uv 0.11.11，确保 `python3 --version` 为 3.12。已有 uv 时，可用独立工具环境提供该版本：

```bash
uv venv --python 3.12 /tmp/supernav-build-tools
source /tmp/supernav-build-tools/bin/activate
```

首次获取源码：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/zju3dv/SuperNav.git
cd SuperNav
```

已有 checkout 时，在 SuperNav 仓库根目录初始化 submodule：

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive -- robots/go2/dimos
```

从仓库根目录进入部署目录并组装：

```bash
cd robots/go2
git -C dimos rev-parse HEAD
python3 -m supernav.prepare
./deploy/bin/check.sh --static
```

DimOS HEAD 应为 `79c9ed163ec6b1abd278a9161f0c74f4562f1231`。`prepare` 将固定版本的公开源码、补丁和新增文件组装到 `runtime/dimos/`，生成 blueprint 注册表。LFS 演示数据保留为指针即可。

**后续命令和路径均以 `SuperNav/robots/go2/` 为部署根目录。**

## 2. 准备系统依赖

目标机需要以下组件：

- Bash、Git、Python 3.12、uv。
- `curl`、`iproute2`、`iputils-ping`、`util-linux`、`procps`、`udev`、NetworkManager。
- `v4l-utils`、`v4l2loopback-dkms` 和与当前 Jetson 内核匹配的 headers。
- 支持 `h264_nvv4l2dec` 的 NVIDIA ffmpeg。
- `build-essential`、`pkg-config`、`libturbojpeg`、`libgl1`、`libglib2.0-0`、`portaudio19-dev`。

按目标机缺少的组件安装，例如：

```bash
sudo apt-get install build-essential pkg-config git curl iproute2 iputils-ping \
  util-linux procps v4l-utils v4l2loopback-dkms \
  libturbojpeg libgl1 libglib2.0-0 portaudio19-dev
modinfo v4l2loopback
ffmpeg -hide_banner -decoders | grep nvv4l2
```

保留 NVIDIA ffmpeg，避免用通用包替换。DKMS 构建失败时先修复内核 headers 和模块安装。选择 `DIMOS_GOPRO_DECODER=software` 可使用软件解码，需检查四路视频的解码性能。

### 可选：在部署目录提供 TurboJPEG

Ubuntu 22.04 aarch64 可将 TurboJPEG 解包到部署目录：

```bash
mkdir -p runtime/native/packages
(cd runtime/native/packages && apt-get download libturbojpeg=2.1.2-0ubuntu1)
for package in runtime/native/packages/*.deb; do
  dpkg-deb --extract "$package" runtime/native
done
ln -s libturbojpeg.so.0 runtime/native/usr/lib/aarch64-linux-gnu/libturbojpeg.so
```

部署脚本会将 `runtime/native/usr/lib/$(uname -m)-linux-gnu` 加入 `LD_LIBRARY_PATH`。手动调用 Python 时设置：

```bash
export LD_LIBRARY_PATH="$PWD/runtime/native/usr/lib/$(uname -m)-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

其他 Ubuntu 版本使用对应软件源中的包版本。

## 3. 安装系统配置

先预览，或将系统文件展开到空目录：

```bash
./deploy/bin/install-system.sh --plan
./deploy/bin/install-system.sh --stage /tmp/supernav-system-review
```

安装前备份下表中已有的同名文件，停止旧 GoPro 视频服务。已加载的 loopback 若设备编号或 `exclusive_caps` 不匹配，先安排重启。安装器会检查这些条件。

在目标 Jetson 上执行：

```bash
sudo ./deploy/bin/install-system.sh --apply
sudo usermod -aG video "$USER"
```

重新登录使 video 组生效，再插拔相机应用 USB DHCP 配置。安装器重载系统规则、加载 loopback 模块、启用 LCM 路由并设置接收缓冲区；视频和机器人栈由[启动器](go2-usage.md)启动。

| 安装位置 | 用途 |
| --- | --- |
| `/usr/local/sbin/gopro-usb-reconcile` | 按 USB 序列号配置相机 DHCP，禁止成为默认路由 |
| `/etc/systemd/system/gopro-usb-reconcile.service` | USB 热插拔触发网络配置 |
| `/etc/udev/rules.d/70-gopro-webcam.rules` | video42–45 的权限及稳定别名 |
| `/etc/udev/rules.d/71-gopro-usb.rules` | GoPro USB 网络热插拔规则 |
| `/etc/modprobe.d/gopro-v4l2loopback.conf` | 四个视频节点，`exclusive_caps=0` |
| `/etc/modules-load.d/gopro-v4l2loopback.conf` | 开机加载 loopback |
| `/etc/systemd/system/supernav-lcm-route.service` | 将 `239.255.76.67/32` 路由到 lo |
| `/etc/sysctl.d/90-supernav-lcm.conf` | LCM 接收缓冲区 |

## 4. 安装 Python 环境

在尚无 `.venv` 的部署目录执行：

```bash
./deploy/bin/install-python.sh --plan
./deploy/bin/install-python.sh --apply
```

`--apply` 用于 aarch64 Jetson。它创建 Python 3.12 `.venv`，按 SHA256 校验安装 `requirements-build.lock` 和 `requirements-jetson.lock`，再编译安装 `runtime/dimos/`。完成后检查依赖、PixelGoal blueprint、A* 扩展和 TurboJPEG，安装版本写入 `runtime/installed-requirements.txt`。

Jetson 锁针对 Python 3.12、Linux aarch64、glibc 2.35，包含以下兼容约束：

- PyTorch/torchvision 使用 CPU 版本；手动安装锁文件时同样传入 `--torch-backend cpu`。相机使用 NVIDIA ffmpeg 硬件解码。
- `unitree-webrtc-connect==2.2.0`。
- `beartype==0.22.9` 与 `plum-dispatch==2.5.7` 配套，保证 Quaternion 构造分派兼容。
- `rerun-sdk==0.32.0a1` 使用锁文件中的公开 wheel URL 和 SHA256。Rerun 的 Python 依赖随运行环境安装，viewer 按配置启动。

安装使用现有锁文件。更新依赖及工作站测试环境见[维护](go2-maintenance.md)和[测试](go2-testing.md)。安装失败后，解决对应包或 ABI 问题，再移走本次创建的 `.venv` 重试。

## 5. 填写机器配置

```bash
cp deploy/config/robot.env.example deploy/config/robot.env
cp deploy/config/secrets.env.example deploy/config/secrets.env
chmod 600 deploy/config/secrets.env
```

在 `robot.env` 填写机器人 IP、四个 GoPro 序列号及相机标定参数；序列号按前、右、后、左排列。模板对应 1280×720 Linear 视场，内参、安装高度和朝向需与实物一致。在 `secrets.env` 填写 `UNITREE_AES_128_KEY`。

配置按 Bash 文件加载。也可通过 `SUPERNAV_CONFIG`、`SUPERNAV_SECRETS` 指定配置的绝对路径。真实配置保存在本地，Git 已忽略这些 `.env` 文件。

默认配置为 `joystick`、`SUPERNAV_RERUN=0`，机器人姿态由操作者准备，启动器不自动站立。

## 6. 检查与启动

保持机器人栈停止，执行：

```bash
./deploy/bin/start-go2-pixel-goal.sh --dry-run
./deploy/bin/check.sh
```

`--dry-run` 显示启动参数；`check.sh` 检查设备、路由、空闲端口及依赖。通过后按[操作教程](go2-usage.md)启动。

回退时先停止当前栈，再切换到上一份独立部署目录；系统配置从安装前备份恢复。
