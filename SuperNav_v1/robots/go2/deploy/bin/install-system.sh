#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
mode="${1:---plan}"
case "$mode" in
  --plan)
    printf '%s\n' 'Install GoPro USB DHCP helper, udev rules and four v4l2loopback devices (42-45).' \
      'Install the narrow LCM loopback route and receive-buffer configuration.' \
      'Does not install/start legacy webcam producers or start DimOS.' \
      'Use --stage DIRECTORY to inspect files; --apply only on the intended Jetson.'
    exit 0;;
  --stage)
    [[ $# == 2 && -n "$2" && "$2" != / ]] || die 'usage: install-system.sh --stage EMPTY_DIRECTORY'
    mkdir -p "$2"
    target="$(cd -- "$2" && pwd)"
    [[ "$target" != / && -z "$(ls -A "$target")" ]] || die 'stage destination must be empty and not /';;
  --apply)
    [[ $EUID == 0 ]] || die 'run with sudo on the target Jetson'
    [[ "$(uname -m)" == aarch64 && -f /etc/nv_tegra_release ]] || die 'target must be an aarch64 Jetson'
    for cmd in systemctl udevadm modprobe nmcli ip sysctl v4l2-ctl ffmpeg; do
      command -v "$cmd" >/dev/null || die "missing $cmd; see docs/release/en/go2.md in the SuperNav repository"
    done
    modinfo v4l2loopback >/dev/null || die 'v4l2loopback is not built for the running kernel'
    [[ -z "$(systemctl list-units --state=active --no-legend 'gopro-webcam@*.service')" ]] ||
      die 'legacy GoPro video services are active; stop them during a maintenance window first'
    if [[ -d /sys/module/v4l2loopback ]]; then
      for n in 42 43 44 45; do
        [[ -e /dev/video$n ]] || die 'v4l2loopback is already loaded with different devices; schedule a reboot'
      done
      caps="$(cut -d, -f1-4 /sys/module/v4l2loopback/parameters/exclusive_caps)"
      [[ "$caps" == N,N,N,N ]] || die 'loaded v4l2loopback has different exclusive_caps; schedule a reboot'
    fi
    target="";;
  *) die 'usage: install-system.sh [--plan|--stage EMPTY_DIRECTORY|--apply]';;
esac
src="$SUPERNAV_ROOT/deploy/system"
install -Dm755 "$src/gopro/gopro-usb-reconcile" "$target/usr/local/sbin/gopro-usb-reconcile"
install -Dm644 "$src/gopro/gopro-usb-reconcile.service" "$target/etc/systemd/system/gopro-usb-reconcile.service"
install -Dm644 "$src/gopro/70-gopro-webcam.rules" "$target/etc/udev/rules.d/70-gopro-webcam.rules"
install -Dm644 "$src/gopro/71-gopro-usb.rules" "$target/etc/udev/rules.d/71-gopro-usb.rules"
install -Dm644 "$src/gopro/gopro-v4l2loopback.conf" "$target/etc/modprobe.d/gopro-v4l2loopback.conf"
install -Dm644 "$src/gopro/gopro-v4l2loopback.modules" "$target/etc/modules-load.d/gopro-v4l2loopback.conf"
install -Dm644 "$src/supernav-lcm-route.service" "$target/etc/systemd/system/supernav-lcm-route.service"
install -Dm644 "$src/90-supernav-lcm.conf" "$target/etc/sysctl.d/90-supernav-lcm.conf"
if [[ "$mode" == --apply ]]; then
  systemctl daemon-reload
  udevadm control --reload-rules
  modprobe v4l2loopback
  udevadm trigger --subsystem-match=video4linux --action=add
  systemctl enable --now supernav-lcm-route.service
  sysctl -p /etc/sysctl.d/90-supernav-lcm.conf
  echo 'Installed. Reconnect cameras to apply USB DHCP rules; no video/robot process was started.'
else
  printf 'Staged system files under %s\n' "$target"
fi
