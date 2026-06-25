#!/usr/bin/env bash
set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/maca/tools/cu-bridge/CUDA_DIR"

HOME="${tmpdir}/home" \
  MACA_PATH= \
  CUDA_PATH= \
  CUCC_PATH= \
  bash -c "source env.sh '${tmpdir}/maca' && test \"\${CUDA_PATH}\" = '${tmpdir}/maca/tools/cu-bridge/CUDA_DIR'"

mkdir -p "${tmpdir}/custom-cuda"
HOME="${tmpdir}/home" \
  CUDA_PATH="${tmpdir}/custom-cuda" \
  bash -c "source env.sh '${tmpdir}/maca' && test \"\${CUDA_PATH}\" = '${tmpdir}/custom-cuda'"

echo "env_sh_paths_ok"
