[English](go2-testing.md) | [简体中文](../zh/go2-testing.md)

# Go2 tests

Run these commands from `SuperNav/robots/go2/`.

## Source and deployment scripts

Use Git and Python 3.12:

```bash
python3 -m supernav.prepare
python3 -m supernav.prepare --check
./deploy/bin/check.sh --static
python3 -m unittest discover -s tests -v
```

These tests cover source assembly, audited replacement declarations, installer previews, launch arguments, prior-run forwarding, snapshot map-persistence defaults, process cleanup, and the release profile. Audit regressions deliberately remove canonical skills and change shared dependencies and blueprint persistence settings to verify that these changes are detected. They use local temporary directories and simulated processes.

## Install a test environment

On Linux x86_64, use Python 3.12, uv 0.11.11, and the CPU test lock. System dependencies include a compiler, PortAudio, TurboJPEG, and image libraries; see the [installation guide](go2.md).

Create the environment at an empty `runtime/test-venv` path:

```bash
uv venv --python 3.12 runtime/test-venv
uv pip install --python runtime/test-venv/bin/python --require-hashes \
  -r deploy/requirements-build.lock
uv pip install --python runtime/test-venv/bin/python --require-hashes \
  -r deploy/requirements-test.lock --torch-backend cpu \
  --build-constraints deploy/requirements-build.in
uv pip install --python runtime/test-venv/bin/python --no-deps --no-build-isolation \
  -e ./runtime/dimos
uv pip check --python runtime/test-venv/bin/python
```

Use `CFLAGS`, `LDFLAGS`, and `LD_LIBRARY_PATH` to provide locally unpacked native libraries and headers to build and test processes.

## Run functional tests

In a terminal dedicated to testing, select the in-memory LCM channel and a temporary state directory:

```bash
export DIMOS_SKIP_SYSTEM_CONFIG=1 USE_TF=0 USE_FLAX=0
export LCM_DEFAULT_URL=memq://
export PYTHONDONTWRITEBYTECODE=1
export XDG_STATE_HOME="$(mktemp -d /tmp/supernav-go2-test-state.XXXXXX)"

runtime/test-venv/bin/python -m pytest \
  --noconftest -c /dev/null -p no:cacheprovider \
  runtime/dimos/dimos/agents/mcp/test_mcp_adapter.py \
  runtime/dimos/dimos/core/coordination/test_supernav_snapshot_stop.py \
  runtime/dimos/dimos/mapping/relocalization/test_module.py \
  runtime/dimos/dimos/navigation/replanning_a_star/test_global_planner.py \
  runtime/dimos/dimos/navigation/replanning_a_star/test_controllers.py \
  runtime/dimos/dimos/mapping/occupancy/test_habitat_area_segmentation.py \
  runtime/dimos/dimos/mapping/occupancy/test_agent_map.py::test_render_agent_occupancy_map_draws_anchored_graph_node_badges \
  runtime/dimos/dimos/mapping/occupancy/test_agent_map.py::test_render_agent_occupancy_map_displaces_overlapping_graph_badges \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_status_attaches_four_images_once \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_slam_map_rejects_odom_in_different_frame \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py::test_terminal_slam_map_requires_robot_pose \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_graph_overlay_suppresses_area_label_inside_region \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_agent_navigation_result_hides_internal_slam_geometry_without_changing_image \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_graph_overlay_uses_grid_row_without_vertical_mirroring \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py::test_proxy_returns_slam_map_with_graph_route_context
```

These release-critical tests exercise active Habitat-style segmentation, native grid-row orientation, area-label suppression, terminal and route-context map attachments, request-specific navigation timeouts, safe-goal selection, prior-triggered relocalization, and synchronous state saving before shutdown. Persistent badges stay at measured SLAM coordinates; the proxy does not move them to avoid collisions. `--noconftest` bypasses upstream whole-repository fixtures; proxy tests use a mock MCP service.

The robot-skill fixtures also test map/pose validation, static-map rendering, prior reporting, motion recording, and relocalization-bundle saving without commanding hardware:

```bash
runtime/test-venv/bin/python -m pytest \
  --noconftest -c /dev/null -p no:cacheprovider \
  runtime/dimos/dimos/agents/skills/test_unitree_skill_container.py \
  -k 'memory_navigation or static_slam_snapshot or fresh_map_snapshot or map_attachment_rejects or prepare_stop or report_relocalization_prior or memory_node_navigation'
```

## Compare with the approved snapshot

For maintainers with access to the implementation snapshot, set `SNAPSHOT_DIR` to a materialized tree containing the base commit and the three approved backports listed in `supernav/manifest.json`. Do not use an arbitrary newer checkout. After rebuilding and installing the runtime:

```bash
runtime/test-venv/bin/python -m supernav.audit_snapshot \
  --snapshot "$SNAPSHOT_DIR" --behavior \
  --report runtime/snapshot-audit.json
```

The audit compares every shared production Python/C++ module, all canonical skill assets, retained support definitions, and the map-only blueprint composition. It checks documented package/path adaptations explicitly. Two isolated processes then compare generated prompts and all five skills; segmentation labels and raw/annotated PNG hashes; tool visibility and map attachment rules; persistent and ephemeral target resolution; ten memory-navigation outcomes; caption encoding, batch requests, selection, and persistence; 48 Python/C++ A* cases; forced-prior triggering and prior cropping; and shutdown ordering. The C++ extension must be built, not silently replaced by Python fallback.

The fixtures do not command a robot or call a model service. Transport, planner completion, clocks, and artifact destinations are controlled; the underlying implementations run normally. They do not validate live SLAM/ICP convergence, camera calibration, or physical navigation. The checked-in `supernav/snapshot-audit.json` records the offline audit; the command above produces a full local report.

The recorded audit passed 914 source/asset checks, 20 differential groups, 43 documented functional tests, 9 focused robot-skill tests, and 28 packaging/deployment/audit tests. A broader comparison found identical outcomes for 191 shared snapshot tests: 166 passed and 25 failed in both trees. One snapshot-only policy-backend selector test is outside this release's map-only scope. The snapshot contains older assertions that disagree with its current implementation (for example, door-node schema and default grid resolution), and timing-sensitive motion-recorder teardown errors can appear with its whole-repository fixtures. Compare failures against the same snapshot before treating them as transfer regressions. Do not change production behavior merely to satisfy those older assertions. This validation used a local Python 3.12 CPU environment; it is not a fresh dependency-lock installation or a physical robot trial.

Check the CLI, blueprint, and native extension:

```bash
runtime/test-venv/bin/python -m dimos.robot.cli.dimos --help
runtime/test-venv/bin/python -c 'import importlib; importlib.import_module("dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal"); importlib.import_module("dimos.navigation.replanning_a_star.min_cost_astar_ext")'
```

These environment variables configure local tests. Open a new terminal for robot operation and follow the [usage guide](go2-usage.md).

## Check a new installation

With exclusive access to the equipment, start the stack and operator as described in the [usage guide](go2-usage.md):

1. Check that the front, right, back, and left views match the camera mounting, resolution, and calibration, and that images update continuously.
2. Check robot status, battery, pose, and live mapping. Upstream MCP exposes navigation/status/stop tools; the agent-facing authority proxy hides `pixelgoal_stop`, `stop_movement`, and operator-only map/prior tools. Terminal PixelGoal results include a segmented PNG map plus `2d_slam_map` metadata.
3. Verify that memory badges are anchored to measured poses with the correct vertical orientation, and duplicate ephemeral area labels are suppressed. Do not relocate persistent badges to make the image look cleaner.
4. In an open area, select a nearby traversable floor point and observe goal direction, arrival status, and stopping. Keep the hardware stop control available throughout the test.
5. Check `pixelgoal_stop`, ESC/F8, map reset, and process cleanup after exit.
6. Run a second session with the first run's `--prior-run-root`; verify that the operator reports a memory-node prior, relocalization converges, and the returned annotated map uses the current-session pose.
7. When using `1008` or Rerun, check the selected motion interface or visualization connection separately.
