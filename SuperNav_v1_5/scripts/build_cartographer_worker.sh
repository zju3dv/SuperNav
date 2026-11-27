#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cartographer_source="${SUPERNAV_CARTOGRAPHER_SOURCE:-${repo_root}/third_party/cartographer}"
cartographer_build="${SUPERNAV_CARTOGRAPHER_BUILD_DIR:-${repo_root}/build/cartographer}"
cartographer_prefix="${HAB_CARTOGRAPHER_PREFIX:-${repo_root}/build/cartographer-prefix}"
worker_build="${SUPERNAV_CARTOGRAPHER_WORKER_BUILD_DIR:-${repo_root}/build/cartographer-worker}"
patch_file="${repo_root}/native/cartographer/patches/authoritative_odometry.patch"
build_jobs="${SUPERNAV_CARTOGRAPHER_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-4}}"

if [[ ! "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUPERNAV_CARTOGRAPHER_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi

expected_commit="877157a0d91788a7700221d87232d412cb3c1ef4"
actual_commit="$(git -C "${cartographer_source}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Cartographer must be pinned to ${expected_commit}; found ${actual_commit}" >&2
  exit 2
fi

if git -C "${cartographer_source}" apply --check "${patch_file}" \
  >/dev/null 2>&1; then
  git -C "${cartographer_source}" apply "${patch_file}"
elif ! git -C "${cartographer_source}" apply --reverse --check "${patch_file}" \
  >/dev/null 2>&1; then
  echo "Authoritative-odometry patch does not apply cleanly" >&2
  exit 2
fi

if ! git -C "${cartographer_source}" diff --cached --quiet || \
  [[ -n "$(git -C "${cartographer_source}" ls-files --others --exclude-standard)" ]] || \
  ! cmp -s \
    <(git -C "${cartographer_source}" diff --no-ext-diff --binary) \
    "${patch_file}"; then
  echo "Cartographer source contains changes beyond the release patch" >&2
  exit 2
fi

cmake -S "${cartographer_source}" -B "${cartographer_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${cartographer_prefix}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
cmake --build "${cartographer_build}" --parallel "${build_jobs}"
cmake --install "${cartographer_build}"

HAB_CARTOGRAPHER_PREFIX="${cartographer_prefix}" \
cmake -S "${repo_root}/native/cartographer" -B "${worker_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${cartographer_prefix}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
cmake --build "${worker_build}" --parallel "${build_jobs}"
ctest --test-dir "${worker_build}" --output-on-failure
"${worker_build}/habitat_cartographer_worker" --capabilities

echo "Worker: ${worker_build}/habitat_cartographer_worker"
