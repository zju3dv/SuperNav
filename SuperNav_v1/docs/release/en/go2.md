[English](go2.md) | [简体中文](../zh/go2.md)

# Unitree Go2 installation

SuperNav's map-backed PixelGoal executes image goals using four GoPro views, a live LiDAR map, and a local planner. After installation, follow the [operation guide](go2-usage.md); see [maintenance](go2-maintenance.md) for source changes and [testing](go2-testing.md) for functional checks.

Use a Jetson aarch64 device with L4T R36.4, Python 3.12, four GoPro HERO8 USB network cameras, and NVIDIA ffmpeg. Create a dedicated deployment directory and Python environment. The running stack needs exclusive access to the cameras, MCP port, and robot LCM channel.

## 1. Get and assemble the source

Prepare Git, Python 3.12, and uv 0.11.11. Check that `python3 --version` reports 3.12. With uv available, a separate tool environment can provide this interpreter:

```bash
uv venv --python 3.12 /tmp/supernav-build-tools
source /tmp/supernav-build-tools/bin/activate
```

For a new checkout:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/zju3dv/SuperNav.git
cd SuperNav
```

For an existing checkout, initialize the submodule from the SuperNav repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive -- robots/go2/dimos
```

From the repository root, enter the deployment directory and assemble the source:

```bash
cd robots/go2
git -C dimos rev-parse HEAD
python3 -m supernav.prepare
./deploy/bin/check.sh --static
```

DimOS HEAD should be `79c9ed163ec6b1abd278a9161f0c74f4562f1231`. `prepare` combines the pinned public source, patches, and new files in `runtime/dimos/`, then generates the blueprint registry. LFS demo assets can remain as pointers.

**Run all remaining commands from `SuperNav/robots/go2/`; paths are relative to this deployment root.**

## 2. Install system dependencies

The target device needs:

- Bash, Git, Python 3.12, and uv.
- `curl`, `iproute2`, `iputils-ping`, `util-linux`, `procps`, `udev`, and NetworkManager.
- `v4l-utils`, `v4l2loopback-dkms`, and headers matching the running Jetson kernel.
- NVIDIA ffmpeg with the `h264_nvv4l2dec` decoder.
- `build-essential`, `pkg-config`, `libturbojpeg`, `libgl1`, `libglib2.0-0`, and `portaudio19-dev`.

Install the missing components on the target device, for example:

```bash
sudo apt-get install build-essential pkg-config git curl iproute2 iputils-ping \
  util-linux procps v4l-utils v4l2loopback-dkms \
  libturbojpeg libgl1 libglib2.0-0 portaudio19-dev
modinfo v4l2loopback
ffmpeg -hide_banner -decoders | grep nvv4l2
```

Keep NVIDIA ffmpeg installed instead of replacing it with a generic package. If DKMS fails to build, fix the kernel headers and module installation first. Set `DIMOS_GOPRO_DECODER=software` to use software decoding, and check its performance with all four video streams.

### Optional: provide TurboJPEG inside the deployment

On Ubuntu 22.04 aarch64, unpack TurboJPEG into the deployment directory:

```bash
mkdir -p runtime/native/packages
(cd runtime/native/packages && apt-get download libturbojpeg=2.1.2-0ubuntu1)
for package in runtime/native/packages/*.deb; do
  dpkg-deb --extract "$package" runtime/native
done
ln -s libturbojpeg.so.0 runtime/native/usr/lib/aarch64-linux-gnu/libturbojpeg.so
```

Deployment scripts add `runtime/native/usr/lib/$(uname -m)-linux-gnu` to `LD_LIBRARY_PATH`. Set it explicitly when invoking Python yourself:

```bash
export LD_LIBRARY_PATH="$PWD/runtime/native/usr/lib/$(uname -m)-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For other Ubuntu versions, use the package version from the corresponding repository.

## 3. Install system configuration

Preview the changes or stage the system files in an empty directory:

```bash
./deploy/bin/install-system.sh --plan
./deploy/bin/install-system.sh --stage /tmp/supernav-system-review
```

Back up existing files at the destinations below and stop old GoPro video services. If a loaded loopback module has different device numbers or `exclusive_caps`, schedule a reboot first. The installer checks these conditions.

Run on the target Jetson:

```bash
sudo ./deploy/bin/install-system.sh --apply
sudo usermod -aG video "$USER"
```

Log in again to activate video group membership, then reconnect the cameras to apply USB DHCP configuration. The installer reloads system rules, loads the loopback module, enables the LCM route, and configures receive buffers. Use the [launcher](go2-usage.md) to start video streams and the robot stack.

| Destination | Purpose |
| --- | --- |
| `/usr/local/sbin/gopro-usb-reconcile` | Configure camera DHCP by USB serial number and prevent a camera interface from becoming the default route |
| `/etc/systemd/system/gopro-usb-reconcile.service` | Configure networking on USB hotplug |
| `/etc/udev/rules.d/70-gopro-webcam.rules` | Set video42–45 permissions and stable aliases |
| `/etc/udev/rules.d/71-gopro-usb.rules` | Apply GoPro USB network hotplug rules |
| `/etc/modprobe.d/gopro-v4l2loopback.conf` | Configure four video nodes with `exclusive_caps=0` |
| `/etc/modules-load.d/gopro-v4l2loopback.conf` | Load loopback at boot |
| `/etc/systemd/system/supernav-lcm-route.service` | Route `239.255.76.67/32` through lo |
| `/etc/sysctl.d/90-supernav-lcm.conf` | Configure LCM receive buffers |

## 4. Install the Python environment

Run in a deployment directory that has no `.venv` yet:

```bash
./deploy/bin/install-python.sh --plan
./deploy/bin/install-python.sh --apply
```

`--apply` targets aarch64 Jetson. It creates a Python 3.12 `.venv`, installs `requirements-build.lock` and `requirements-jetson.lock` with SHA256 verification, then compiles and installs `runtime/dimos/`. It checks dependencies, the PixelGoal blueprint, the A* extension, and TurboJPEG, and writes installed versions to `runtime/installed-requirements.txt`.

The Jetson lock targets Python 3.12, Linux aarch64, and glibc 2.35, with these compatibility constraints:

- PyTorch/torchvision use CPU builds. Pass `--torch-backend cpu` when installing the lock manually. Cameras use NVIDIA ffmpeg hardware decoding.
- `unitree-webrtc-connect==2.2.0`.
- `beartype==0.22.9` is paired with `plum-dispatch==2.5.7` for compatible Quaternion constructor dispatch.
- `rerun-sdk==0.32.0a1` uses the public wheel URL and SHA256 in the lock. Its Python dependencies are installed with the environment; configuration controls viewer startup.

Installation uses the existing locks. See [maintenance](go2-maintenance.md) for dependency updates and [testing](go2-testing.md) for the workstation environment. If installation fails, resolve the package or ABI issue, move aside the `.venv` created by this attempt, and retry.

## 5. Configure the robot and cameras

```bash
cp deploy/config/robot.env.example deploy/config/robot.env
cp deploy/config/secrets.env.example deploy/config/secrets.env
chmod 600 deploy/config/secrets.env
```

In `robot.env`, set the robot IP, four GoPro serial numbers, and camera calibration parameters. List serial numbers in front, right, back, left order. The template uses a 1280×720 Linear field of view; intrinsics, mounting height, and orientation must match the physical cameras. Set `UNITREE_AES_128_KEY` in `secrets.env`.

The scripts load configuration as Bash files. Set `SUPERNAV_CONFIG` and `SUPERNAV_SECRETS` to use absolute paths to other configuration files. Keep actual configuration local; Git ignores these `.env` files.

The defaults are `joystick` and `SUPERNAV_RERUN=0`. The operator prepares the robot's posture; the launcher does not stand it up automatically.

## 6. Check and start

With the robot stack stopped, run:

```bash
./deploy/bin/start-go2-pixel-goal.sh --dry-run
./deploy/bin/check.sh
```

`--dry-run` displays launch arguments. `check.sh` checks devices, routes, available ports, and dependencies. Once it passes, follow the [operation guide](go2-usage.md).

To roll back, stop the current stack, switch to the previous deployment directory, and restore system configuration from the installation backups.
