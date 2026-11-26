#!/usr/bin/env bash
# SuperNav Go2 launcher. Upstream notices: ../../LICENSE.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
dry_run=0
move_api=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1;;
    --help|-h) echo 'Usage: start-go2-pixel-goal.sh [joystick|1008] [--dry-run]'; exit 0;;
    joystick|1008) [[ -z "$move_api" ]] || die 'specify only one move API'; move_api="$arg";;
    *) die "unknown argument: $arg";;
  esac
done
verify_source
load_config
set_move_api "$move_api"
viewer_args=(--viewer none)
if [[ "$SUPERNAV_RERUN" == 1 ]]; then
  viewer_args=(--viewer rerun --rerun-open none --no-rerun-web)
fi
command_args=("$DIMOS_PYTHON" -m dimos.robot.cli.dimos
  --robot-ip "${UNITREE_GO2_IP:-CHANGE_ME}" "${viewer_args[@]}"
  --mcp-port 9990 --no-go2-auto-stand
  run unitree-go2-pixelgoal --stack mapping --control local-planner)
if [[ "$dry_run" == 1 ]]; then
  printf 'Runtime: %s\nMove API: %s\nRerun: %s\nPolicy: disabled\n' \
    "$DIMOS_REPO" "$DIMOS_GO2_MOVE_API" "$SUPERNAV_RERUN"
  printf 'Command: '; printf '%q ' "${command_args[@]}"; printf '\n'
  exit 0
fi
validate_robot_config
verify_runtime
"$SUPERNAV_ROOT/deploy/bin/check.sh"
if [[ "$SUPERNAV_RERUN" == 1 ]]; then
  [[ -x "$SUPERNAV_ROOT/.venv/bin/dimos-viewer" ]] || die 'missing dimos-viewer'
fi
mkdir -p "$SUPERNAV_ROOT/runtime/logs" "$SUPERNAV_ROOT/runtime/work" "$DIMOS_PIXELGOAL_LIVE_DIR"
exec 8>"$SUPERNAV_ROOT/runtime/stack.lock"
flock -n 8 || die 'this release is already running'
log_dir="$SUPERNAV_ROOT/runtime/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$log_dir"
children=()
cleanup() {
  local pid attempt alive
  trap - EXIT INT TERM
  # Each child owns a new process group; never kill a process we did not start.
  for pid in "${children[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for attempt in {1..15}; do
    alive=0
    for pid in "${children[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 1
  done
  for pid in "${children[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid "$DIMOS_REPO/scripts/go2_gopro_bridge.sh" run >"$log_dir/gopro.log" 2>&1 &
bridge_pid=$!
children+=("$bridge_pid")
ready=0
for attempt in {1..60}; do
  kill -0 "$bridge_pid" 2>/dev/null || die "GoPro bridge exited; see $log_dir/gopro.log"
  if grep -q 'four YU12 streams ready' "$log_dir/gopro.log"; then ready=1; break; fi
  sleep 1
done
[[ "$ready" == 1 ]] || die "GoPro bridge timeout; see $log_dir/gopro.log"
if [[ "$SUPERNAV_RERUN" == 1 ]]; then
  setsid "$SUPERNAV_ROOT/.venv/bin/dimos-viewer" --serve-grpc --port 9877 \
    --server-memory-limit 256MB \
    --cors-allow-origin "${SUPERNAV_RERUN_CORS_ORIGIN:-http://127.0.0.1:9878}" \
    >"$log_dir/rerun.log" 2>&1 &
  viewer_pid=$!
  children+=("$viewer_pid")
  ready=0
  for attempt in {1..20}; do
    kill -0 "$viewer_pid" 2>/dev/null || die "Rerun exited; see $log_dir/rerun.log"
    if [[ -n "$(ss -H -ltn 'sport = :9877')" ]]; then ready=1; break; fi
    sleep 0.25
  done
  [[ "$ready" == 1 ]] || die 'Rerun did not listen on port 9877'
fi
cd "$SUPERNAV_ROOT/runtime/work"
printf 'Starting SuperNav; logs: %s; Ctrl-C stops this process group.\n' "$log_dir"
setsid "${command_args[@]}" >"$log_dir/dimos.log" 2>&1 &
robot_pid=$!
children+=("$robot_pid")
# Fail the whole stack if its bridge, viewer, or robot process exits.
set +e
wait -n "${children[@]}"
status=$?
set -e
exit "$status"
