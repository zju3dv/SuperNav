#!/usr/bin/env bash
# Maintainer operation: resolve public packages without touching a robot or venv.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
case "${1:---help}" in
  --help) echo 'Usage: lock-python.sh --update (uv 0.11.11; assemble runtime first)'; exit 0;;
  --update) ;;
  *) die 'usage: lock-python.sh --update';;
esac
verify_runtime
case "$(uv --version)" in
  'uv 0.11.11'|'uv 0.11.11 '*) ;;
  *) die 'use uv 0.11.11 to regenerate release locks';;
esac
cd "$SUPERNAV_ROOT"
uv pip compile deploy/requirements-build.in --python-version 3.12 --universal \
  --generate-hashes --no-header -o deploy/requirements-build.lock
common=(--python-version 3.12 -c deploy/constraints-jetson.txt \
  -c deploy/requirements-build.in --no-emit-package dimos --generate-hashes --no-header)
uv pip compile deploy/requirements-jetson.in "${common[@]}" \
  --python-platform aarch64-manylinux_2_35 --torch-backend cpu --emit-index-url \
  -o deploy/requirements-jetson.lock
uv pip compile deploy/requirements-test.in "${common[@]}" \
  --python-platform x86_64-manylinux_2_35 --torch-backend cpu --emit-index-url \
  -o deploy/requirements-test.lock
