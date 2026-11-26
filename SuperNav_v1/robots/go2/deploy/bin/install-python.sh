#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
case "${1:---plan}" in
  --plan) printf '%s\n' 'Assemble public DimOS + SuperNav into runtime/dimos; create an isolated Python 3.12 .venv.' 'Install hashed ARM64 dependency and build locks, build the local runtime, and check imports.'; exit 0;;
  --apply) ;;
  *) die 'usage: install-python.sh [--plan|--apply]';;
esac
verify_source
[[ "$(uname -m)" == aarch64 && -f /etc/nv_tegra_release ]] || die 'installation target must be an aarch64 Jetson; this workstation is for packaging'
command -v uv >/dev/null || die 'install uv on the target first (see docs/release/en/go2.md in the SuperNav repository)'
[[ ! -e "$SUPERNAV_ROOT/.venv" ]] || die '.venv already exists; use a new release directory for a fresh install'
cd "$SUPERNAV_ROOT"
python3 -m supernav.prepare
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r deploy/requirements-build.lock
uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu \
  -r deploy/requirements-jetson.lock --build-constraints deploy/requirements-build.in
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e ./runtime/dimos
uv pip check --python .venv/bin/python
mkdir -p runtime
uv pip freeze --python .venv/bin/python > runtime/installed-requirements.txt
.venv/bin/python -c 'import importlib; importlib.import_module("dimos.robot.unitree.go2.blueprints.pixelgoal.unitree_go2_pixelgoal"); importlib.import_module("dimos.navigation.replanning_a_star.min_cost_astar_ext"); from turbojpeg import TurboJPEG; TurboJPEG()'
printf '%s\n' 'Python dependencies and blueprint import checked. Hardware operation is not tested by this installer.'
