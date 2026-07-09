#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
CLEAN="${CLEAN:-0}"

PRESETS=(
  native-release
  native-debug
  riscv-release
  riscv-debug
  riscv-gemmini-release
  riscv-gemmini-debug
)

if [[ "$#" -gt 0 ]]; then
  PRESETS=("$@")
fi

if [[ "${CLEAN}" == "1" ]]; then
  echo "Removing build directories for selected presets..."
  for preset in "${PRESETS[@]}"; do
    rm -rf "build-${preset}"
  done
fi

for preset in "${PRESETS[@]}"; do
  echo
  echo "================================================================"
  echo "Configuring preset: ${preset}"
  echo "================================================================"
  cmake --preset "${preset}"

  echo
  echo "================================================================"
  echo "Building preset: ${preset}"
  echo "================================================================"
  cmake --build --preset "${preset}" --parallel "${JOBS}"
done

echo
echo "Done."