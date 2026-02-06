# Installation

!!! warning "Breaking Change Notice"
    After v0.11.2, vLLM-MetaX moved its `_C` and `_moe_C` kernel into a separate package named `mcoplib`. 
    
    mcoplib is open-sourced at [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) and would maintain its own release cycle. vllm-metax's release rely on its corresponding version of mcoplib. Check it at the [Release Page](../quickstart.md#releases).

    Though the *csrc* folder is still kept in this repo for development convenience, and there is no guarantee that the code is always in sync with mcoplib. Not only the performance but also the correctness may differ from mcoplib. 

    To build and use the vllm-metax csrc , you need to set: 

    ```bash    
    export USE_PRECOMPILED_KERNEL=0
    ```
    
    in both *build* and *runtime* environment variables.
    
    **Please always use mcoplib for production usage.**

## Requirements

- OS: Linux
- Python: 3.10 -- 3.12


## Build from source

### Prepare environment
```bash
# setup MACA path
export MACA_PATH="/opt/maca"

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2

# update PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
```

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:prepare-env"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:prepare-env"

### Build vllm

Clone vllm project:

```bash 
git clone  --depth 1 --branch releases/v0.13.0 https://github.com/vllm-project/vllm 
cd vllm
```

Build with *empty device*:

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:build-vllm"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:build-vllm"

### Build plugin

Clone vllm-metax project:

```bash 
git clone --branch releases/v0.13.0 https://github.com/MetaX-MACA/vLLM-metax
cd vLLM-metax
```

Build the plugin:

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:build-vllm-metax"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:build-vllm-metax"

## Extra information