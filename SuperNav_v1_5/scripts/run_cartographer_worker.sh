#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_build="${SUPERNAV_CARTOGRAPHER_WORKER_BUILD_DIR:-${repo_root}/build/cartographer-worker}"
worker="${HAB_CARTOGRAPHER_WORKER:-${worker_build}/habitat_cartographer_worker}"
cartographer_prefix="${HAB_CARTOGRAPHER_PREFIX:-${repo_root}/build/cartographer-prefix}"

if [[ ! -x "${worker}" ]]; then
  echo "Cartographer worker is missing or not executable: ${worker}" >&2
  exit 2
fi
if [[ "${1:-}" == "--capabilities" ]]; then
  exec "${worker}" --capabilities
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <input-fifo> <occupancy-png>" >&2
  exit 2
fi

config_dir="${cartographer_prefix}/share/cartographer/configuration_files"
if [[ ! -d "${config_dir}" ]]; then
  echo "Cartographer configuration directory is missing: ${config_dir}" >&2
  exit 2
fi
exec "${worker}" "${config_dir}" "$2" < "$1"
