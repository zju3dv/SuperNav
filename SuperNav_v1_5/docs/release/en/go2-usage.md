[English](go2-usage.md) | [简体中文](../zh/go2-usage.md)

# Operating Go2

Complete [installation](go2.md), then run the commands below from `SuperNav/robots/go2/`. Run one stack per robot. Prepare the robot's stance before launch; the launcher does not make it stand automatically.

## Start the robot stack

```bash
source .venv/bin/activate
./deploy/bin/check.sh
./deploy/bin/start-go2-pixel-goal.sh
```

The default movement interface is `joystick`. To use `1008`, run:

```bash
./deploy/bin/start-go2-pixel-goal.sh 1008
```

The minimum turning speeds are 0.2 rad/s for `joystick` and 0.5 rad/s for `1008`. Override them with `DIMOS_ASTAR_MIN_ANGULAR_RPS`.

The launcher waits for all four GoPro bridges, then starts `unitree-go2-pixelgoal` with `mapping` and `local-planner`. ffmpeg writes the GoPro video streams to `/dev/video42–45`, and MCP uses port 9990.

Rerun is off by default. Each launch starts with an empty map; PixelGoal navigates using the live map built during that run.

## Start the operator console

Install and authenticate the Codex CLI. Set `SUPERNAV_MODEL` and `SUPERNAV_REASONING_EFFORT` in `robot.env` to values supported by your account, and confirm that `command -v codex` finds the CLI.

Open another terminal in the deployment directory:

```bash
source .venv/bin/activate
./deploy/bin/operate.sh
```

The console uses the deployment's `.venv` and connects to `http://127.0.0.1:9990/mcp`. Each task loads DimOS MCP and two navigation skills: [PixelGoal](../../../robots/go2/supernav/overlay/dimos/agents/skills/go2-pixelgoal/SKILL.md) governs point selection for each hop, and [global navigation](../../../robots/go2/supernav/overlay/dimos/agents/skills/global-navigation-pixelgoal/SKILL.md) governs search and route selection.

### Model service proxy

If your model service requires a proxy, configure it in the console terminal. Replace `PROXY_HOST:PORT` with a reachable HTTP proxy:

```bash
export HTTPS_PROXY=http://PROXY_HOST:PORT
export HTTP_PROXY="$HTTPS_PROXY" ALL_PROXY="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1
./deploy/bin/operate.sh
```

The launcher adds the robot's IP address to `NO_PROXY`. Run the robot stack in its separate terminal.

### Operate from a workstation

Assemble the same source on the workstation and install the [workstation dependencies](go2-testing.md). Forward the Jetson MCP port over SSH, replacing `user@jetson-host` with your SSH user and host:

```bash
ssh -N -L 9990:127.0.0.1:9990 user@jetson-host
```

`operate.sh` uses the deployment directory's `.venv`. When following the testing guide to install workstation dependencies, use `.venv` as the environment path.

## Console controls

| Key | Action |
| --- | --- |
| F2 | Enter Agent mode and type a navigation task |
| F7 | Open the Pixel Goal panel: `refresh`, `reset`, and `stop`; image display and clicking require Rerun |
| ESC / F8 | Request a motion stop; confirm the robot has stopped, then press Enter to continue |

The agent selects a normalized target point from the four camera views and uses fresh images after each hop. Prefer traversable ground; select stable opaque vertical surfaces according to the PixelGoal skill.

The console reads robot status and battery level asynchronously every two seconds:

- `ROBOT STATUS OK`: the latest low-level status read succeeded.
- `UNVERIFIED`: MCP is connected, and a low-level status read has yet to succeed.
- `STALE` / `UNAVAILABLE`: the status read failed, and the old battery value was cleared.

Status queries run independently of input and stop controls. The operator checks the robot's stance and surrounding space.

## PixelGoal tools and results

| MCP tool | Purpose |
| --- | --- |
| `pixelgoal_navigate` | Submit a normalized image point `[x, y]` in the range 0–1 and a `direction`: front, right, back, or left |
| `pixelgoal_status` | Read navigation status and results |
| `pixelgoal_stop` | Cancel the current navigation hop |

Navigation requires a valid map, pose, and fresh image. To target a specific captured frame, supply the frozen-frame data supported by the interface. Read the terminal status and returned four camera views after each hop before choosing the next action.

When the map has yet to cover the target, each exploration leg is at most 1.5 m and stays close to the clicked direction. A leg may extend at most 0.5 m into unknown space from reachable known free space. Another leg into unknown space requires new free-space evidence near the previous frontier.

`blocked_pixel_target` means a known obstacle blocks the target's approach line or stopping point. By default, navigation stops with `no_progress` after 15 seconds without at least 0.1 m of net approach. Use fresh images to select another point after either result.

`final_distance_m` is the XY distance in meters from the robot's center to the final stopping point when the session ends. The value is frozen at completion. It is `null` if a valid stopping point or pose is unavailable.

## Optional Rerun

Set these values in `robot.env`, then restart the stack:

```bash
SUPERNAV_RERUN=1
SUPERNAV_RERUN_CORS_ORIGIN=http://127.0.0.1:9878
```

The launcher starts a gRPC viewer on port 9877 and a visualization control WebSocket on port 3030. Start the browser viewer separately and match the CORS origin to its actual origin.

## Stop and inspect logs

Stop motion through the console or `pixelgoal_stop`. Once the robot has stopped, press Ctrl-C in the stack terminal to exit. The launcher cleans up the process groups it created; if a main process exits, the other processes from that launch also stop. Use the robot's own controls for lying down or a hardware emergency stop.

| Path | Contents |
| --- | --- |
| `runtime/logs/<timestamp>/` | `gopro.log`, `dimos.log`, and optional `rerun.log` |
| `runtime/pixelgoal-live/` | PixelGoal frames, status, and events |
| `runtime/gopro-bridge/` | Camera bridge status |
| `runtime/state/` | Runtime registry |
| `runtime/operator-runs/` | Operator task outputs |
| `runtime/dimos/logs/` | DimOS logs written to the default source directory |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Source validation fails | `git -C dimos status` and `python3 -m supernav.prepare --inputs-only` |
| Generated source is outdated | Stop the stack, then rebuild and reinstall using the [maintenance guide](go2-maintenance.md) |
| video42–45 are missing | loopback module, kernel headers, and membership in the video group |
| Camera startup times out | USB DHCP, serial number order, camera mode, and `gopro.log` |
| Hardware decoder is missing | NVIDIA ffmpeg's `h264_nvv4l2dec`; check all four streams' performance if selecting software decoding |
| A port or video device is occupied | Stop the old stack or video producer using it |
| WebRTC cannot connect | Robot IP address, routing, AES key, and `NO_PROXY` |
| Projected targets are offset | Image resolution, Linear field of view, and camera intrinsics/extrinsics |
| Codex CLI is not found | Installation path; for `~/.npm-global/bin`, set `export PATH="$HOME/.npm-global/bin:$PATH"` |
