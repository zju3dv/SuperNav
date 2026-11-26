[English](web-viewer.md) | [简体中文](../zh/web-viewer.md)

# SuperNav Live Viewer

The browser viewer provides a read-only view of first-person RGB,
front/right/back/left observations, point overlays, world-coordinate trajectories,
and recorded agent messages and tool calls.

## Try the demo

From the SuperNav repository root:

```bash
python -m pip install -e .
supernav web --demo
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) to view a procedurally drawn
room and sample trajectory. The installed Python package serves the viewer
directly, with bundled static assets and fonts for offline use.
`python -m supernav web` is equivalent.

Press Ctrl+C to stop the demo before continuing, or use another terminal. Stop an
existing viewer before starting another on the same port.

## Watch a real run

First prepare a working experiment using the [Habitat guide](habitat.md) or
[AI2-THOR guide](ai2thor.md), including simulator assets, agent CLI, and credentials.
Before starting the viewer and bridge/experiment runner, set **the same absolute
path** in both terminals. When both terminals are at the repository root:

```bash
export SUPERNAV_LIVE_DIR="$PWD/data/nav_artifacts/live"
export SUPERNAV_LIVE_FPS=5
```

The frame-rate setting is optional: the default is 5, and allowed values are
greater than 0 and at most 30. `$PWD` supplies the repository root in this example;
when the terminals use different working directories, set the absolute path explicitly.

In the viewer terminal:

```bash
supernav web --live-dir "$SUPERNAV_LIVE_DIR" --runs-root data/runs
```

In the experiment terminal, complete the [Habitat setup](habitat.md), including
an external task manifest in `configs/local/habitat.json`. Replace `example-task`
with an ID from that manifest and run:

```bash
supernav run --config configs/local/habitat.json \
  --task-ids example-task --arms default \
  --model gpt-6-astra --output-dir data/runs/habitat-live
```

The example inherits `medium` reasoning for `gpt-6-astra` from
[`configs/models/gpt-6-astra-medium.json`](../../../configs/models/gpt-6-astra-medium.json).
Use a new output directory or `--run-tag` when repeating the example, because
existing episodes are skipped by default.

With `bridge.per_episode=true`, new bridges inherit capture settings and include
the task instruction, agent, run ID, and task ID. For a manually started bridge,
set `SUPERNAV_LIVE_DIR` before startup; restart the bridge after adding it.
Choose a free port for each manually started bridge.

A manual bridge may supply JSON in `SUPERNAV_LIVE_CONTEXT` with `instruction`,
`agent`, `run_id`, and `task_id`; otherwise the viewer displays the scene and a
generic agent name. Its `run_id` should match the episode's `run.json`. The viewer
uses that ID to associate agent logs even before `metrics.json` exists.

`--runs-root` defaults to `data/runs` relative to the current working directory.
For another output location, pass the parent directory containing the episodes.
The live directory stores observation snapshots independently of the run evidence
directory.

New sessions appear automatically in the selector. If none is selected, the viewer
connects automatically. Multiple experiments may share a live directory; each
simulator session has its own UUID.

## Remote access

The server binds to loopback by default. For a remote experiment, forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 user@experiment-host
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) on your local machine.
`--host` and `--port` override the listening address. The server has no account
authentication; use a trusted network or SSH tunnel.

## AI2-THOR observations

`ai2thor-primitive` also supports `SUPERNAV_LIVE_DIR`. Set the same
directory for the runner and viewer, then select `AI2-THOR · <scene_id>`. The runner
associates run IDs and agent logs automatically. See [AI2-THOR](ai2thor.md) for setup.

The viewer displays rendered RGB, task instructions, action activity, horizontal
trajectories, and resource-claim/NoMaD point overlays. `environment.views=four` is
the default: the four panels show 640 × 480 front/right/back/left images and associate
points with their selected direction and capture. In `front` mode, other directions
are marked as not captured. The reasoning panel displays available model logs;
resource claims and STOP events appear in agent activity.

## Controls and status

- **Pause view / Follow live** freezes or resumes the browser view; the agent
  continues running.
- **Four-direction observation** shows images from the same capture and supports
  enlargement. It reuses the panorama given to the agent. In `front_only` mode,
  the front image is shown and other directions are marked as not captured.
- **Agent's point overlay** appears as soon as it is written. The viewer preserves
  the original overlay pixels,
  direction, coordinates, image reference, and capture sequence. Selecting a
  historical point shows the observations on which it was based; **Latest** returns
  to the newest observation.
- **Agent reasoning & tool trace** shows emitted explanations and reasoning
  summaries, expandable tool arguments, status, outcomes, and source file/line
  references. Entries follow timestamps and original event order. Filters select
  reasoning or tools, and matching calls link to overlays. Logs are read
  incrementally every second, with the latest 300 entries retained in the view.
  The latest agent/reasoning message also appears below the live image.
- **Trajectory** shows sampled world coordinates as an orange path, with a hollow
  start marker, a heading arrow, and zoom and fit controls. X points right and Z
  down, in meters.
- **Activity** shows action starts, results, and errors, retaining the latest 100
  events. Live capture can also update RGB/pose during long actions.
- **Connection status** distinguishes waiting for the first frame, capture
  failures, normal closure, and a missing producer. SSE reconnects automatically
  with a complete current snapshot; disconnection leaves the last frame visible.
- **Completion and scoring** displays agent claims and session status. `closed`
  marks session closure. Read `metrics.json.success` as run metadata; obtain
  task-ground-truth SR/SPL from offline scoring.

## Browse recorded runs

```bash
supernav web \
  --visuals-root path/to/experiment/visuals \
  --runs-root path/to/experiment/episodes
```

Repeat `--visuals-root` for multiple directories. Each should contain top-level
`<session_id>.trajectory.json` sidecars and RGB PNG files under `<session_id>/`.
Supported images include `step*_color_sensor.png`,
`pano_front_step*_color_sensor.png`, and LocalNav frames. Depth display is unsupported.
The four-view panel groups `pano_{front,right,back,left}_step*.png` by capture and
prefers `_agent` images representing the agent's actual observations. Overlays are
matched through the session's `<session_id>.benchmark_audit.jsonl`. RGB frames
are ordered by file time and can be scrubbed or played at 8 fps.

`--runs-root` associates instructions, client information, and agent claims through
`run.json`, `metrics.json`, and/or `manifest.json`. The timeline reads
`canonical.jsonl` and incrementally reads `raw.jsonl` while a run is active,
supporting Codex, Kimi, and OpenCode formats. Codex session JSONL can provide emitted
reasoning and timestamps from that run's `codex_session.jsonl` or isolated
`codex_project/.codex_home/sessions/`. Kimi prefers the main agent's `wire.jsonl`
under the run's isolated `kimi_project/kimi_home/.kimi-code/sessions/`.

The timeline excludes image binary data, secret fields, and encrypted reasoning.
Long outputs are truncated for display; complete evidence remains in the original
logs. During historical browsing, trajectories and statistics show the final
whole-session values. The RGB slider selects images and associated observations/
points; exact per-frame pose reconstruction is unavailable. Sidecars recorded
without live capture update at tool completion.

## Capture limits and evidence

Capture is disabled by default. When enabled, Habitat renders at the configured
capture rate, while AI2-THOR reuses existing RGB. Published JPEGs are resized to at
most 960 × 720. Capture can increase wall-clock time; use consistent settings or
disable it for strict performance comparisons.

Each session replaces its latest JPEG/JSON and retains up to 12 four-view PNG
groups, 24 overlays, 4000 trajectory samples, and 100 action events. Older live
copies are removed automatically. Full history remains in the original visual
artifacts and logs. Image failures preserve the previous frame and report a capture
error.

During frequent updates, the image can lead the pose by one sample. Use original
evidence for precise timing and offline scorers for evaluation.

Capture runs independently of the browser. Shutdown preserves the final snapshot;
clearing the live directory removes its sessions. Producer liveness checks apply
on the same host; shared directories across hosts show recorded status and the
last frame time.

## Development checks

From a source checkout with test dependencies installed:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_web_viewer.py tests/test_harness_bridge_lifecycle.py tests/test_supernav_boundary.py tests/demand_driven/test_live.py
```

These checks cover SSE/reconnection, lifecycle and capture errors, tool-result and
action-count integrity, image/overlay associations and retention, incremental logs,
resource paths, recorded runs, and claim/ground-truth separation. Font licenses
are bundled in
[`static/fonts/`](../../../src/supernav/web/static/fonts).

The viewer lives in [`src/supernav/web/`](../../../src/supernav/web), with capture
publishers in the runtime and simulator backends. See [architecture](architecture.md),
[directory layout](layout.md), and the [documentation index](README.md).
