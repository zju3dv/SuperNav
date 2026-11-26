#!/usr/bin/env bash
# Bridge four USB-connected GoPro Webcam streams to v4l2loopback devices.
set -euo pipefail

ACTION="${1:-run}"
# GoPro Webcam has a fixed ~30 FPS source; lower resolution reduces bitrate and
# decode work while the loopback contract stays 1280x720. Output frame rate is
# DIMOS_GOPRO_FPS (default 15); decode cost follows the source rate, so higher
# output rates only add scale/convert work on the Jetson.
START_DELAY_S="${DIMOS_GOPRO_START_DELAY_S:-5}"
SOURCE_RES="${DIMOS_GOPRO_SOURCE_RES:-480}"
OUTPUT_FPS="${DIMOS_GOPRO_FPS:-15}"
# Jetson's hardware decoder avoids four software H.264 decoders contending at 7W.
# Set DIMOS_GOPRO_DECODER=software to use the old software path.
DECODER="${DIMOS_GOPRO_DECODER:-h264_nvv4l2dec}"
if [[ "$DECODER" == "software" || "$DECODER" == "none" ]]; then
  DECODER_ARGS=()
else
  DECODER_ARGS=(-c:v "$DECODER")
fi
if [[ "$ACTION" != "run" && "$ACTION" != "check" ]]; then
  echo "usage: $0 [run|check]" >&2
  exit 2
fi

for command in curl ffmpeg flock ip ping udevadm v4l2-ctl; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 1; }
done

read -ra SERIALS <<< "${DIMOS_GOPRO_SERIALS:?Set four GoPro serials in front/right/back/left order}"
read -ra VIDEO_NRS <<< "${DIMOS_GOPRO_VIDEO_NRS:-42 43 44 45}"
if (( ${#SERIALS[@]} != 4 || ${#VIDEO_NRS[@]} != 4 )); then
  echo "DIMOS_GOPRO_SERIALS and DIMOS_GOPRO_VIDEO_NRS must each contain four values" >&2
  exit 1
fi
for video_nr in "${VIDEO_NRS[@]}"; do
  [[ -e "/dev/video$video_nr" ]] || {
    echo "missing /dev/video$video_nr; load v4l2loopback first" >&2
    exit 1
  }
done

interface_for_serial() {
  local serial="$1" net_path current
  for net_path in /sys/class/net/enx*; do
    [[ -e "$net_path" ]] || continue
    current="$(udevadm info -q property -p "$net_path" 2>/dev/null | sed -n 's/^ID_SERIAL_SHORT=//p')"
    [[ "$current" == "$serial" ]] && { basename "$net_path"; return 0; }
  done
  return 1
}

discover_all() {
  local serial interface host_ip camera_ip
  INTERFACES=()
  HOST_IPS=()
  CAMERA_IPS=()
  for serial in "${SERIALS[@]}"; do
    interface="$(interface_for_serial "$serial")" || return 1
    host_ip="$(ip -4 -o addr show dev "$interface" | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}')"
    [[ -n "$host_ip" ]] || return 1
    camera_ip="${host_ip%.*}.51"
    ping -c 1 -W 1 "$camera_ip" >/dev/null || return 1
    INTERFACES+=("$interface")
    HOST_IPS+=("$host_ip")
    CAMERA_IPS+=("$camera_ip")
  done
}

print_map() {
  local index
  for index in 0 1 2 3; do
    printf '%s %s host=%s camera=%s /dev/video%s\n' \
      "${SERIALS[$index]}" "${INTERFACES[$index]}" "${HOST_IPS[$index]}" \
      "${CAMERA_IPS[$index]}" "${VIDEO_NRS[$index]}"
  done
}

if [[ "$ACTION" == "check" ]]; then
  discover_all || { echo "not all configured GoPros are reachable" >&2; exit 1; }
  print_map
  exit 0
fi

STATE_DIR="${DIMOS_GOPRO_STATE_DIR:-$HOME/.local/state/dimos/gopro-bridge}"
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/bridge.lock"
flock -n 9 || { echo "another GoPro bridge is already running" >&2; exit 1; }
for serial in "${SERIALS[@]}"; do
  : >"$STATE_DIR/$serial.log"
done

CHILDREN=()
stop_children() {
  local pid attempt alive
  for pid in "${CHILDREN[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  for attempt in {1..10}; do
    alive=0
    for pid in "${CHILDREN[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive == 0 )) && break
    sleep 0.2
  done
  for pid in "${CHILDREN[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  CHILDREN=()
}

stop_cameras() {
  local camera_ip pid
  local pids=()
  for camera_ip in "${CAMERA_IPS[@]:-}"; do
    curl --noproxy '*' --silent --max-time 2 "http://$camera_ip/gp/gpWebcam/STOP" >/dev/null &
    pids+=($!)
  done
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}

shutdown() {
  local status=$?
  trap - EXIT INT TERM
  stop_children
  stop_cameras
  exit "$status"
}
trap shutdown EXIT INT TERM

start_round() {
  local index serial video_nr camera_ip
  stop_cameras
  sleep 2
  for index in 0 1 2 3; do
    serial="${SERIALS[$index]}"
    video_nr="${VIDEO_NRS[$index]}"
    ffmpeg -nostdin -hide_banner -loglevel warning -threads 1 \
      "${DECODER_ARGS[@]}" \
      -i "udp://0.0.0.0:8554?localaddr=${HOST_IPS[$index]}&reuse=0&overrun_nonfatal=1&timeout=15000000" \
      -map 0:v:0 -an -vf "fps=${OUTPUT_FPS},scale=1280:720,format=yuv420p" \
      -f v4l2 "/dev/video$video_nr" >>"$STATE_DIR/$serial.log" 2>&1 &
    CHILDREN+=($!)
    sleep 1
    camera_ip="${CAMERA_IPS[$index]}"
    curl --noproxy '*' --silent --fail --show-error --max-time 8 \
      "http://$camera_ip/gp/gpWebcam/START?res=${SOURCE_RES}&fov=4&port=8554" >/dev/null
    echo "[gopro-bridge] started $serial"
    if (( index < 3 )); then
      sleep "$START_DELAY_S"
    fi
  done
}

wait_until_ready() {
  local attempt video_nr pid format ready
  for attempt in {1..30}; do
    ready=1
    for video_nr in "${VIDEO_NRS[@]}"; do
      format="$(v4l2-ctl -d "/dev/video$video_nr" --get-fmt-video 2>&1)"
      grep -q "1280/720" <<<"$format" || ready=0
      grep -q "'YU12'" <<<"$format" || ready=0
    done
    for pid in "${CHILDREN[@]}"; do kill -0 "$pid" 2>/dev/null || ready=0; done
    (( ready == 1 )) && return 0
    sleep 1
  done
  return 1
}

while true; do
  until discover_all; do
    echo "[gopro-bridge] waiting for all four GoPros" >&2
    sleep 5
  done
  print_map
  if start_round && wait_until_ready; then
    echo "[gopro-bridge] four YU12 streams ready"
    set +e
    wait -n "${CHILDREN[@]}"
    status=$?
    set -e
    echo "[gopro-bridge] ffmpeg exited ($status); restarting all streams" >&2
  else
    echo "[gopro-bridge] startup failed; retrying" >&2
  fi
  stop_children
  stop_cameras
  sleep 2
done
