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

Rerun is off by default. With default settings and no `--prior-run-root`, each launch starts with an empty map; voxel-map saving on normal shutdown remains enabled, as in the implementation snapshot. During the run, SuperNav-1.5 segments the occupancy map into areas, attaches the annotated map to navigation results, and maintains memory nodes from direct observations.

## Start the operator console

Install and authenticate the Codex CLI. Set `SUPERNAV_MODEL` and `SUPERNAV_REASONING_EFFORT` in `robot.env` to values supported by your account, and confirm that `command -v codex` finds the CLI.

Open another terminal in the deployment directory:

```bash
source .venv/bin/activate
./deploy/bin/operate.sh
```

The console uses the deployment's `.venv` and connects to `http://127.0.0.1:9990/mcp`. Each task writes the source-canonical run-local skills into its workspace: PixelGoal point selection, global navigation, spatial-memory hierarchy, 2D SLAM interpretation, and memory/SLAM route fusion. The prompt and skill snapshots are recorded in the task audit.

The SuperNav-1.5 spatial memory owns spatial state for this run. The original bounded spatial-fact ledger is not enabled at the same time.

## Reuse memory in a second session

At normal shutdown, the operator finalizes the current SLAM relocalization bundle. To reuse one completed run, pass the exact same prior task directory to both processes:

```bash
./deploy/bin/start-go2-pixel-goal.sh --prior-run-root /absolute/path/to/prior-run
```

In the operator terminal:

```bash
./deploy/bin/operate.sh --prior-run-root /absolute/path/to/prior-run
```

The prior run must contain `logs/go2-graph-memory.json` and `slam/relocalization/manifest.json`. The stack loads the saved map and enables relocalization; the operator seeds the prior memory nodes and may report a trusted node as a relocalization prior when visual evidence supports it. A missing or incomplete prior run is reported explicitly instead of silently starting a partial memory session.

## Optional area captions

Set `GO2_SLAM_AREA_CAPTION_URL` in `deploy/config/robot.env` to a compatible `/caption` endpoint. Keyframes are queued asynchronously and navigation continues if the service is absent or fails. Leave the value empty to disable captioning.

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
| ESC / F8 | Operator-only motion stop; confirm the robot has stopped, then press Enter to continue |

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
| `pixelgoal_stop` | Operator-only cancellation of the current navigation hop |

Navigation requires a valid map, pose, and fresh image. The agent-facing `pixelgoal_navigate` accepts only `point` and `direction`, waits for the hop, and returns the terminal result; normal agent operation does not poll status. Frozen-frame arguments belong to the upstream operator interface, not the agent schema. Read the terminal status and fresh camera views after each hop before choosing the next action.

When the map has yet to cover the target, each exploration leg is at most 1.5 m and stays close to the clicked direction. A leg may extend at most 0.5 m into unknown space from reachable known free space. Another leg into unknown space requires new free-space evidence near the previous frontier.

`blocked_pixel_target` means a known obstacle blocks the target's approach line or stopping point. By default, navigation stops with `no_progress` after 15 seconds without at least 0.1 m of net approach. Use fresh images to select another point after either result.

`final_distance_m` is the XY distance in meters from the robot's center to the final stopping point when the session ends. The value is frozen at completion. It is `null` if a valid stopping point or pose is unavailable.

### Maps, memory targets, and tool visibility

Map attachments follow the implementation snapshot. They are conditional on map/pose availability; failure metadata does not imply an image was attached.

| Tool/result | Map image | Other images |
| --- | --- | --- |
| Agent `pixelgoal_navigate` terminal result | One full segmented occupancy map, annotated by the proxy | Fresh front/right/back/left views |
| `pixelgoal_status(include_images=true)` | Terminal map, not during an active hop; healthy terminal images are normally emitted once per goal, with caption-pending refreshes | Fresh surround views |
| `go2_navigate_to_memory_node` | One canonical segmented map, including failed navigation when map/pose remain valid | Latest camera if available; proxy also requests fresh surround views |
| Generic movement (`drive`, `relative_move`, `rotate_in_place`) | Clean floor plan, segmented map, and robot-centered occupancy view | Latest camera if available |
| `go2_query_route_context` / `go2_report_graph_position` | Latest cached annotated map when available; route query refreshes it first | Stored node panoramas when requested; position report includes them |
| `go2_inspect_graph_node` | No SLAM-map attachment | Stored node panoramas when requested |
| Map-context/status reads, graph writes, ordinary observations | No additional SLAM-map attachment | Tool-specific observations only |

For route queries, `return_images=false` suppresses stored panoramas, not the SLAM map. Operator-only `go2_slam_map_snapshot` and `go2_report_relocalization_prior` can also return the full segmented map. The proxy hides those tools, map reset, `pixelgoal_stop`, and `stop_movement` from the agent; movement visibility is limited by the active operator-approved envelope.

Numbered persistent memory badges remain at their measured SLAM poses, without collision-driven displacement. Ephemeral `A<N>` labels are suppressed where a persistent node covers the same native-grid region. Native grid rows are flipped only for display. A memory-node navigation request resolves either a stored node or a current ephemeral area target, requires a current frame-consistent map/pose, and uses the native planner; it does not fall back to a simulator NavMesh. Timeout and planner rejection remain explicit results, not successful arrivals.

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
| `runtime/operator-runs/<run>/logs/go2-graph-memory.json` | Spatial-memory nodes and their observations |
| `runtime/operator-runs/<run>/slam/relocalization/` | Saved map and relocalization manifest for a later session |
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
