#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
verify_source
if [[ "${1:-}" == --static ]]; then
  python3 "$SUPERNAV_ROOT/deploy/lib/verify-package.py"
  exit 0
fi
[[ $# == 0 ]] || die 'usage: check.sh [--static]'
load_config
validate_robot_config
verify_runtime
set_move_api
for cmd in curl ffmpeg flock ip ping udevadm v4l2-ctl setsid ss; do
  command -v "$cmd" >/dev/null || die "missing command: $cmd"
done
[[ -x "$DIMOS_PYTHON" ]] || die 'missing release .venv; install Python dependencies first'
for n in 42 43 44 45; do
  [[ -r /dev/video$n && -w /dev/video$n ]] || die "missing access to /dev/video$n; check loopback and video group"
done
[[ "$(ip route get 239.255.76.67)" == *'dev lo'* ]] || die 'LCM narrow loopback route is missing'
[[ "$(ip link show lo)" == *MULTICAST* ]] || die 'loopback multicast is disabled'
for setting in rmem_max rmem_default; do
  read -r buffer_bytes <"/proc/sys/net/core/$setting"
  [[ "$buffer_bytes" -ge 67108864 ]] || die "net.core.$setting must be at least 67108864; see deploy/system/90-supernav-lcm.conf"
done
[[ "$(ip route get "$UNITREE_GO2_IP")" != *'dev lo'* ]] || die 'robot route incorrectly uses loopback'
[[ "$(ip route get 224.0.0.1)" != *'dev lo'* ]] || die 'broad multicast route incorrectly uses loopback'
decoder="${DIMOS_GOPRO_DECODER:-h264_nvv4l2dec}"
if [[ "$decoder" != software && "$decoder" != none ]]; then
  decoders="$(ffmpeg -hide_banner -decoders 2>/dev/null)"
  [[ "$decoders" == *" $decoder "* ]] || die "ffmpeg decoder $decoder missing; preserve NVIDIA ffmpeg or select software explicitly"
fi
"$DIMOS_PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
import socket
import subprocess

assert metadata.version('unitree-webrtc-connect') == '2.2.0', 'unexpected WebRTC version'
importlib.import_module('dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal')
importlib.import_module('dimos.navigation.replanning_a_star.min_cost_astar_ext')
from turbojpeg import TurboJPEG
TurboJPEG()  # Loading Image alone does not load its native JPEG dependency.
ports = [9990] + ([9877, 3030] if os.environ['SUPERNAV_RERUN'] == '1' else [])
for port in ports:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
# Refuse to take over cameras already used by the old stack.
processes = subprocess.run(['ps', '-eo', 'args='], check=True, text=True, capture_output=True).stdout
for line in processes.splitlines():
    if 'ffmpeg' in line and any('/dev/video' + str(n) in line for n in range(42, 46)):
        raise SystemExit('GoPro video producer already active; stop the old stack before starting this release')
print('Source, Python imports, network routes and free ports checked.')
PY
echo 'No motion commands or camera START/STOP calls were made. Camera reachability and live navigation still require validation.'
