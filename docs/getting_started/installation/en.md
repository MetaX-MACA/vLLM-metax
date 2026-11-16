# Installation

[中文](/getting_started/installation/zh.md){ .md-button }

## Build from source

!!! warning "Breaking Change Notice"
    After v0.11.2, vLLM-MetaX moved its `_C` and `_moe_C` kernel into a separate package named `mcoplib`.

    mcoplib is open-sourced at [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) and would maintain its own release cycle. vllm-metax's release rely on its corresponding version of mcoplib. Check it at the [Release Page](../quickstart/en.md#releases).

    Though the *csrc* folder is still kept in this repo for development convenience, and there is no guarantee that the code is always in sync with mcoplib. Not only the performance but also the correctness may differ from mcoplib. 

    To build and use the vllm-metax csrc , you need to set: 

    ```bash    
    export USE_PRECOMPILED_KERNEL=0
    ```
    
    in both *build* and *runtime* environment variables.
    
    **Please always use mcoplib for production usage.**

### Requirements

- OS: Linux
- Python: 3.10 -- 3.12
- Hardware: MetaX C-series
- SDK: MACA-SDK

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

### Build vllm-metax plugin

Clone vllm-metax project:

```bash
git clone https://github.com/MetaX-MACA/vLLM-metax
cd vLLM-metax
```

Build the plugin:

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:build-vllm-metax"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:build-vllm-metax"

### Build vllm

!!! warning

    It's ***not recommended*** to install vllm directly from PYPI with:

    ```
    uv pip install vllm==X.Y.Z
    ```

    This would install a *cuda-build* vllm which carried a lot of cuda-related dependencies and kernel files together with internal vllm_flash_attn and triton_kernels, which may wrongly cause runtime errors on some checking. (E.g. Some pre-conditions that should **only** be detected on `cuda` may also passed to `maca` backend by mistake in this kind of vllm.)

Clone vllm project:

```bash
git clone  --depth 1 --branch releases/v0.32.0 https://github.com/vllm-project/vllm 
cd vllm
```

Build with *empty device*:

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:build-vllm"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:build-vllm"

### Post build (Significant)

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.inc.md:post-build"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.inc.md:post-build"

## Install the Release Wheel

vLLM-MetaX also ships released wheels that can be installed directly with pip.

### Requirements

- OS: Linux
- Python: 3.10 or 3.12
- Hardware: MetaX C-series
- SDK: MACA-SDK

### Find a version

List the available vllm-metax versions from the MetaX package repository:

```bash
pip index versions vllm-metax -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
```

This example installs `0.28.0+g666b8f.d20260929.maca3.8.2.401.torch2.10.0`, which corresponds to plugin v0.28.0.

### Install version-specific dependencies

The wheels are tightly coupled to specific PyTorch / MACA versions. Depending on the plugin version, PyTorch and mcoplib must either be installed manually **before** installing vllm, or they are pulled in automatically when vllm-metax is installed. The MACA and mcoplib versions per plugin version are listed in the [releases table](../quickstart/en.md#releases). Choose your version below:

=== "v0.22.0 – v0.24.0"

    PyTorch and mcoplib must be installed manually **before** installing vllm.

    **1. Install PyTorch from the offline maca-pytorch package**

    The offline package name follows the format `maca-pytorch{torch_version}-py{310|312}-{maca version}-{architecture}.tar.xz`. To build the search string:

    1. PyTorch version: since v0.22.0, vllm-metax uses torch 2.10, so use `2.10`.
    2. Python version: `3.10` or `3.12` (e.g. `py312`).
    3. MACA version: read it from the [releases table](../quickstart/en.md#releases) and take the first three fields. For example, v0.23.0 maps to `maca3.8.0.x`, so use `3.8.0`.

    For example, the v0.23.0 / Python 3.12 search string is `maca-pytorch2.10-py312-3.8.0`. Search it at the [MetaX software center](https://developer.metax-tech.com/softnova/search?package_name=maca-pytorch2.10-py312-3.8.0) and download the archive matching your architecture (`x86_64` or `aarch64`). Extract it and install all contained wheels (apex, torchaudio, torchvision, etc.):

    ```bash
    tar -xf maca-pytorch2.10-py312-3.8.0-*.tar.xz
    pip install *.whl
    ```

    **2. Install mcoplib**

    List the available versions:

    ```bash
    pip index versions mcoplib -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
    ```

    The wheel version format is `{mcoplib_version}+maca{maca version}.torch{torch version}`. Both numbers come from the [releases table](../quickstart/en.md#releases). For example, for v0.23.0: `0.4.8+maca3.8.0.25.torch2.10`:

    ```bash
    pip install mcoplib==0.4.8+maca3.8.0.25.torch2.10 \
        -i https://repos.metax-tech.com/r/maca-pypi/simple \
        --extra-index-url https://pypi.org/simple \
        --trusted-host repos.metax-tech.com
    ```

=== "v0.25.0 – v0.27.0"

    PyTorch must still be installed manually from the offline maca-pytorch package (see the v0.22.0 – v0.24.0 tab above).

    mcoplib does **not** need to be installed separately anymore: it is pulled in automatically when vllm-metax is installed with pip.

=== "v0.28.0 and newer"

    Nothing to install manually: PyTorch, mcoplib, and the other dependencies are pulled in automatically when vllm-metax is installed.

    Verify the pulled-in versions after the wheel install — see the note under [Install vllm-metax](#install-vllm-metax).

### Install vllm

Install the matching upstream vllm version first. The `--no-deps` flag is required: the vllm-metax wheel resolves the MACA-compatible dependencies later.

```bash
pip install vllm==0.28.0 --no-deps
```

### Install vllm-metax

Then install the plugin wheel:

```bash
pip install vllm-metax==0.28.0+g666b8f.d20260929.maca3.8.2.401.torch2.10.0 \
    -i https://repos.metax-tech.com/r/maca-pypi/simple \
    --extra-index-url https://pypi.org/simple \
    --trusted-host repos.metax-tech.com
```

!!! note "v0.28.0 and newer: verify the pulled-in dependency versions"

    For v0.28.0 and newer, the install command above also pulls in PyTorch, mcoplib, and the other dependencies. Some of them may resolve to wrong versions, i.e. their version strings lack the `torch2.10` suffix. The affected packages are:

    `apex`, `block_sparse_attn`, `torchaudio`, `torchcodec`, `torchvision`, `triton`

    For example, `apex==0.1+metax3.8.2.2` is wrong. The correct version is `0.1+metax3.8.2.2torch2.10`. Uninstall and reinstall each affected package that has no `torch2.10` suffix:

    ```bash
    pip uninstall apex
    pip install apex==0.1+metax3.8.2.2torch2.10 \
        -i https://repos.metax-tech.com/r/maca-pypi/simple \
        --extra-index-url https://pypi.org/simple \
        --trusted-host repos.metax-tech.com
    ```

### Check the environment

Verify the installation:

```bash
python -c 'import vllm_metax; print(vllm_metax.__file__)'
python -c 'import torch; print(torch.cuda.is_available())'
```

Then start a model to test the installation — see [Start the inference service](../deployment/en.md#4-start-the-inference-service) in the Deployment guide.
