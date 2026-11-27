#!/usr/bin/env bash
# Shared deployment configuration; never source the old ~/go2-deploy env.
set -euo pipefail
SUPERNAV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly SUPERNAV_ROOT

# Optional native libraries unpacked into this release, without system installs.
native_lib="$SUPERNAV_ROOT/runtime/native/usr/lib/$(uname -m)-linux-gnu"
if [[ -d "$native_lib" ]]; then
  export LD_LIBRARY_PATH="$native_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset native_lib

die() { printf 'supernav: %s\n' "$*" >&2; exit 1; }

verify_source() {
  (cd "$SUPERNAV_ROOT" && python3 -m supernav.prepare --inputs-only >/dev/null)
}

verify_runtime() {
  (cd "$SUPERNAV_ROOT" && python3 -m supernav.prepare --check >/dev/null)
}

load_config() {
  local config secrets
  config="${SUPERNAV_CONFIG:-$SUPERNAV_ROOT/deploy/config/robot.env}"
  secrets="${SUPERNAV_SECRETS:-$SUPERNAV_ROOT/deploy/config/secrets.env}"
  [[ -f "$config" ]] || die "copy deploy/config/robot.env.example to robot.env and configure it"
  set -a
  # shellcheck disable=SC1090
  source "$config"
  if [[ -f "$secrets" ]]; then
    # shellcheck disable=SC1090
    source "$secrets"
  fi
  set +a

  # Paths and profile are fixed to this release, regardless of the parent shell.
  export DIMOS_REPO="$SUPERNAV_ROOT/runtime/dimos"
  export DIMOS_LIVE_DIR="$DIMOS_REPO"
  export DIMOS_VENV="$SUPERNAV_ROOT/.venv"
  export DIMOS_PYTHON="$DIMOS_VENV/bin/python"
  export PYTHONPATH="$DIMOS_REPO"
  export PATH="$SUPERNAV_ROOT/.venv/bin:$PATH"
  export XDG_STATE_HOME="$SUPERNAV_ROOT/runtime/state"
  export DIMOS_PIXELGOAL_LIVE_DIR="$SUPERNAV_ROOT/runtime/pixelgoal-live"
  export DIMOS_GOPRO_STATE_DIR="$SUPERNAV_ROOT/runtime/gopro-bridge"
  export DIMOS_SKIP_SYSTEM_CONFIG=1
  export DIMOS_GO2_AUTO_STAND=false
  export DIMOS_VOXEL_MAP_LOAD_ON_START=none DIMOS_VOXEL_MAP_SAVE_ON_STOP=0
  export DIMOS_GO2_CAMERA_BACKEND=gopro
  export DIMOS_GOPRO_VIDEO_NRS="42 43 44 45"
  export DIMOS_GO2_CAMERA_FRONT_DEVICE=/dev/video42
  export DIMOS_GO2_CAMERA_RIGHT_DEVICE=/dev/video43
  export DIMOS_GO2_CAMERA_BACK_DEVICE=/dev/video44
  export DIMOS_GO2_CAMERA_LEFT_DEVICE=/dev/video45
  export DIMOS_GO2_CAMERA_WIDTH=1280 DIMOS_GO2_CAMERA_HEIGHT=720
  export DIMOS_GO2_CAMERA_FOURCC=""
  export DIMOS_GO2_CAMERA_FRONT_FPS=5 DIMOS_GO2_CAMERA_OTHER_FPS=5
  export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1
  unset DIMOS_EDGE_PROFILE DIMOS_DISABLE_MODULES
  export SUPERNAV_RERUN="${SUPERNAV_RERUN:-0}"
  export DIMOS_GO2_MOVE_API="${DIMOS_GO2_MOVE_API:-joystick}"
  export NO_PROXY="${NO_PROXY:-${no_proxy:-}},127.0.0.1,localhost,${UNITREE_GO2_IP:-}"
  export no_proxy="$NO_PROXY"
  case "$SUPERNAV_RERUN" in 0|1) ;; *) die "SUPERNAV_RERUN must be 0 or 1";; esac
}

set_move_api() {
  export DIMOS_GO2_MOVE_API="${1:-$DIMOS_GO2_MOVE_API}"
  case "$DIMOS_GO2_MOVE_API" in
    joystick) export DIMOS_ASTAR_MIN_ANGULAR_RPS="${DIMOS_ASTAR_MIN_ANGULAR_RPS:-0.2}";;
    1008) export DIMOS_ASTAR_MIN_ANGULAR_RPS="${DIMOS_ASTAR_MIN_ANGULAR_RPS:-0.5}";;
    *) die "move API must be joystick or 1008";;
  esac
}

validate_robot_config() {
  [[ -n "${UNITREE_GO2_IP:-}" && "$UNITREE_GO2_IP" != CHANGE_ME ]] || die "configure UNITREE_GO2_IP"
  [[ -n "${UNITREE_AES_128_KEY:-}" ]] || die "configure UNITREE_AES_128_KEY in secrets.env"
  local -a serials
  read -ra serials <<< "${DIMOS_GOPRO_SERIALS:-}"
  [[ ${#serials[@]} == 4 ]] || die "configure four GoPro serials in front/right/back/left order"
  local serial
  for serial in "${serials[@]}"; do
    [[ "$serial" != *_SERIAL && "$serial" =~ ^[[:alnum:]]+$ ]] || die "replace GoPro serial placeholders"
  done
  [[ "$(printf '%s\n' "${serials[@]}" | sort -u | wc -l)" == 4 ]] || die "GoPro serials must be unique"
}
