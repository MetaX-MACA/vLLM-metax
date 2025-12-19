# setup MACA path
DEFAULT_DIR="/opt/maca"
export MACA_PATH=${1:-$DEFAULT_DIR}

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH=/root/cu-bridge/CUDA_DIR
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1