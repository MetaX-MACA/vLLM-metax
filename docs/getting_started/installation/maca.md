# Installation

## Requirements

- OS: Linux
- Python: 3.10 -- 3.12

## Set up using pip (without UV)

### Build wheel from source

!!! note
    If using pip, all the build and installation steps are based on *corresponding docker images*. You can find them on [quick start](../quickstart.md).
    We need to add `-no-build-isolation` flag (or an equivalent one) during package building, since all the requirements are already pre-installed in released docker image.

#### Setup environment variables

```bash
# setup MACA path
export MACA_PATH="/opt/maca"

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH=/root/cu-bridge/CUDA_DIR
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1
```

#### Build vllm

Clone vllm project:

```bash 
git clone  --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm
cd vllm
```

Build with *empty device*:

```bash
python use_existing_torch.py
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v . --no-build-isolation
```

#### Build plugin

Install the build requirments first:

```bash
python use_existing_metax.py
pip install -r requirements/build.txt
```

Build and install vLLM:

```bash
pip install . -v --no-build-isolation
```

If you want to develop vLLM, install it in editable mode instead.

```bash
pip install . -e -v --no-build-isolation
```

Optionally, build a portable wheel which you can then install elsewhere:

```bash
python -m build -w -n
pip install dist/*.whl
``` 

!!! note
    vllm-metax need to manually copy the .so files to vllm's site-package folder.

    This additional behavior has been fixed and removed in v0.11.1 or newer. But **before v0.11.1**, you may still need execute the command: 

    ```bash
    vllm_metax_init
    ```

    after your installation. And It's also **not recommended** to install plugin via editable mode. (This is usually unstable for library copy)


## Set up using UV (experimental)

Todo

## Extra information
