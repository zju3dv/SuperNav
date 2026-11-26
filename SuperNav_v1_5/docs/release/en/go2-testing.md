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

These tests cover source assembly, installer previews, launch arguments, process cleanup, and Markdown resource loading and task copies. They use local temporary directories and simulated processes.

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
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map.py \
  runtime/dimos/dimos/robot/unitree/pixelgoal/test_pixel_goal_map_module.py \
  runtime/dimos/dimos/agents/mcp/test_external_harness_authority_proxy.py \
  runtime/dimos/dimos/agents/mcp/test_pixelgoal_handoff_contract.py \
  runtime/dimos/dimos/agents/mcp/test_memory_free_operator.py \
  runtime/dimos/dimos/agents/console/test_robot_status.py
```

The tests cover map projection, goal selection, navigation sessions, blueprint composition, movement authorization, stopping, result handoffs, tool isolation, and console status. `--noconftest` bypasses upstream fixtures for full robot tests; proxy tests use a mock MCP service.

Check the CLI, blueprint, and native extension:

```bash
runtime/test-venv/bin/python -m dimos.robot.cli.dimos --help
runtime/test-venv/bin/python -c 'import importlib; importlib.import_module("dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal"); importlib.import_module("dimos.navigation.replanning_a_star.min_cost_astar_ext")'
```

These environment variables configure local tests. Open a new terminal for robot operation and follow the [usage guide](go2-usage.md).

## Check a new installation

With exclusive access to the equipment, start the stack and operator as described in the [usage guide](go2-usage.md):

1. Check that the front, right, back, and left views match the camera mounting, resolution, and calibration, and that images update continuously.
2. Check robot status, battery, pose, and live mapping. Confirm that MCP exposes `pixelgoal_navigate`, `pixelgoal_status`, and `pixelgoal_stop`.
3. In an open area, select a nearby traversable floor point and observe goal direction, arrival status, and stopping. Keep the hardware stop control available throughout the test.
4. Check `pixelgoal_stop`, ESC/F8, map reset, and process cleanup after exit.
5. When using `1008` or Rerun, check the selected motion interface or visualization connection separately.
