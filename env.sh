# setup MACA path
DEFAULT_DIR="/opt/maca"
export MACA_PATH=${1:-$DEFAULT_DIR}

# cu-bridge
export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
if [ -z "${CUDA_PATH}" ]; then
  if [ -d "${CUCC_PATH}/CUDA_DIR" ]; then
    export CUDA_PATH="${CUCC_PATH}/CUDA_DIR"
  elif [ -d "${HOME}/cu-bridge/CUDA_DIR" ]; then
    export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
  else
    export CUDA_PATH="${CUCC_PATH}"
  fi
fi
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1
