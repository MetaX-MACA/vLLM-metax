## 1 拉取镜像

docker pull pub-registry1.metax-tech.com/ai-opentest/release/maca/vllm-metax:0.18.0-maca.ai3.5.3.401-torch2.8-py310-ubuntu22.04-amd64



## 2

```bash
# setup MACA path
export MACA_PATH="/opt/maca"

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH="${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}"

export USE_PRECOMPILED_KERNEL=0


apt update
apt install -y git

git config --global user.name ""
git config --global user.email ""
1. 安装 vLLM-Metax
git clone --branch ai2-dev
 https://github.com/MetaX-MACA/vLLM-metax
cd vLLM-metax

python use_existing_metax.py
pip install -r requirements/build.txt
pip install -v -e . --no-build-isolation
2. 安装 vLLM
git clone --depth 1 --branch v0.19.1rc0   https://github.com/vllm-project/vllm
cd vllm

git config --global --add safe.directory "$(pwd)"
VLLM_TARGET_DEVICE=empty pip install . --no-build-isolation
```
## 3 运行

```bash
export MACA_SMALL_PAGESIZE_ENABLE=1
export MACA_DIRECT_DISPATCH=1
export TRITON_ENABLE_MACA_COMPILER_INT8_OPT=True
export TRITON_ENABLE_ELEMENTWISE_PK_FMA_OPT=True
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/xxy/vLLM-metax:$PYTHONPATH #换成自己的插件目录
跑server或者benchmark都要把上面这个环境变量加上
