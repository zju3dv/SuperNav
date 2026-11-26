#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: operate.sh [additional dimos operate arguments]'; exit 0
fi
verify_source
load_config
verify_runtime
[[ -x "$DIMOS_PYTHON" ]] || die 'run install-python.sh --apply on the target first'
command -v codex >/dev/null || die 'the operator host needs an authenticated codex CLI'
mkdir -p "$SUPERNAV_ROOT/runtime/operator"
cd "$SUPERNAV_ROOT/runtime/operator"
model_args=()
[[ -z "${SUPERNAV_MODEL:-}" ]] || model_args=(--model "$SUPERNAV_MODEL")
exec "$DIMOS_PYTHON" -m dimos.robot.cli.dimos operate \
  --blueprint unitree-go2-pixelgoal --stack mapping --control local-planner \
  --authority-tool pixelgoal_navigate --no-launch \
  --mcp-url http://127.0.0.1:9990/mcp \
  --model-reasoning-effort "${SUPERNAV_REASONING_EFFORT:-high}" \
  --run-root "$SUPERNAV_ROOT/runtime/operator-runs" "${model_args[@]}" "$@"
