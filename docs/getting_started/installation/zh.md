# 安装

[English](/getting_started/installation/en.md){ .md-button }

## 从源码构建 {#build-from-source}

!!! warning "破坏性变更提示"
    v0.11.2 之后，vLLM-MetaX 将其 `_C` 和 `_moe_C` 内核迁出到独立包 `mcoplib`。

    mcoplib 已在 [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) 开源，并维护自己的发布周期。vllm-metax 的发布依赖其对应版本的 mcoplib，可在[版本映射](../quickstart/zh.md#releases)中查看。

    仓库中仍保留 *csrc* 目录以方便开发，但代码不保证与 mcoplib 同步，性能乃至正确性都可能存在差异。

    如需构建并使用 vllm-metax 的 csrc，需要在**构建**与**运行**环境中同时设置：

    ```bash
    export USE_PRECOMPILED_KERNEL=0
    ```

    **生产环境请始终使用 mcoplib。**

### 环境要求

- 操作系统：Linux
- Python：3.10 -- 3.12
- 硬件：MetaX C 系列
- SDK：MACA-SDK

### 准备环境

```bash
# 设置 MACA 路径
export MACA_PATH="/opt/maca"

# cu-bridge
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2

# 更新 PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
```

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.zh.inc.md:prepare-env"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.zh.inc.md:prepare-env"

### 构建 vllm-metax 插件

克隆 vllm-metax 项目：

```bash
git clone https://github.com/MetaX-MACA/vLLM-metax
cd vLLM-metax
```

构建插件：

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.zh.inc.md:build-vllm-metax"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.zh.inc.md:build-vllm-metax"

### 构建 vllm

!!! warning

    ***不建议***直接从 PYPI 安装 vllm：

    ```
    uv pip install vllm==X.Y.Z
    ```

    这样会装到 *cuda 构建* 的 vllm，其中携带大量 cuda 相关依赖和内核文件，以及内部的 vllm_flash_attn 和 triton_kernels，可能在某些检查中错误触发运行时错误。（例如，某些**只应**在 `cuda` 上检测的前置条件，可能在这种 vllm 中被错误传递到 `maca` 后端。）

克隆 vllm 项目：

```bash
git clone  --depth 1 --branch releases/v0.32.0 https://github.com/vllm-project/vllm 
cd vllm
```

使用 *empty device* 构建：

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.zh.inc.md:build-vllm"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.zh.inc.md:build-vllm"

### 构建后（重要）

=== "PIP"
    --8<-- "docs/getting_started/installation/pip.zh.inc.md:post-build"
=== "UV"
    --8<-- "docs/getting_started/installation/uv.zh.inc.md:post-build"

## 使用发布 Wheel 安装 {#install-the-release-wheel}

vLLM-MetaX 也发布可直接通过 pip 安装的 wheel。

### 环境要求

- 操作系统：Linux
- Python：3.10 或 3.12
- 硬件：MetaX C 系列
- SDK：MACA-SDK

### 查找版本

从 MetaX 包仓库列出可用的 vllm-metax 版本：

```bash
pip index versions vllm-metax -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
```

以下示例安装 `0.28.0+g666b8f.d20260929.maca3.8.2.401.torch2.10.0`，对应插件 v0.28.0。

### 安装版本相关依赖

wheel 与特定的 PyTorch / MACA 版本强耦合。根据插件版本，PyTorch 和 mcoplib 要么需要在安装 vllm **之前**手动安装，要么会在安装 vllm-metax 时自动带入。各插件版本对应的 MACA 与 mcoplib 版本见[版本映射表](../quickstart/zh.md#releases)。请按你的版本选择：

=== "v0.22.0 – v0.24.0"

    PyTorch 和 mcoplib 必须在安装 vllm **之前**手动安装。

    **1. 通过离线 maca-pytorch 包安装 PyTorch**

    离线包名称遵循 `maca-pytorch{torch_version}-py{310|312}-{maca version}-{architecture}.tar.xz` 格式。构造搜索字符串：

    1. PyTorch 版本：v0.22.0 起 vllm-metax 使用 torch 2.10，故取 `2.10`。
    2. Python 版本：`3.10` 或 `3.12`（例如 `py312`）。
    3. MACA 版本：从[版本映射表](../quickstart/zh.md#releases)读取并取前三位。例如 v0.23.0 对应 `maca3.8.0.x`，取 `3.8.0`。

    以 v0.23.0 / Python 3.12 为例，搜索字符串为 `maca-pytorch2.10-py312-3.8.0`。在 [MetaX 软件中心](https://developer.metax-tech.com/softnova/search?package_name=maca-pytorch2.10-py312-3.8.0) 搜索并下载与架构匹配（`x86_64` 或 `aarch64`）的离线包。解压后安装其中全部 wheel（apex、torchaudio、torchvision 等）：

    ```bash
    tar -xf maca-pytorch2.10-py312-3.8.0-*.tar.xz
    pip install *.whl
    ```

    **2. 安装 mcoplib**

    列出可用版本：

    ```bash
    pip index versions mcoplib -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
    ```

    wheel 版本格式为 `{mcoplib_version}+maca{maca version}.torch{torch version}`，两个版本号均来自[版本映射表](../quickstart/zh.md#releases)。例如 v0.23.0 对应 `0.4.8+maca3.8.0.25.torch2.10`：

    ```bash
    pip install mcoplib==0.4.8+maca3.8.0.25.torch2.10 \
        -i https://repos.metax-tech.com/r/maca-pypi/simple \
        --extra-index-url https://pypi.org/simple \
        --trusted-host repos.metax-tech.com
    ```

=== "v0.25.0 – v0.27.0"

    PyTorch 仍需通过离线 maca-pytorch 包手动安装（见上方 v0.22.0 – v0.24.0 页签）。

    mcoplib **无需**再单独安装：使用 pip 安装 vllm-metax 时会自动带入匹配的版本。

=== "v0.28.0 及更高版本"

    无需手动安装任何依赖：PyTorch、mcoplib 及其他依赖会在安装 vllm-metax 时自动带入。

    wheel 安装完成后请核对带入的版本——见[安装 vllm-metax](#install-vllm-metax)下方的提示。

### 安装 vllm

先安装匹配的上游 vllm 版本。`--no-deps` 参数必填：MACA 兼容的依赖由 vllm-metax 的 wheel 随后解析。

```bash
pip install vllm==0.28.0 --no-deps
```

### 安装 vllm-metax {#install-vllm-metax}

然后安装插件 wheel：

```bash
pip install vllm-metax==0.28.0+g666b8f.d20260929.maca3.8.2.401.torch2.10.0 \
    -i https://repos.metax-tech.com/r/maca-pypi/simple \
    --extra-index-url https://pypi.org/simple \
    --trusted-host repos.metax-tech.com
```

!!! note "v0.28.0 及更高版本：核对自动带入的依赖版本"

    v0.28.0 起，上面的安装命令还会带入 PyTorch、mcoplib 及其他依赖。其中部分包可能解析出错误的版本，即版本号缺少 `torch2.10` 后缀。受影响的包包括：

    `apex`、`block_sparse_attn`、`torchaudio`、`torchcodec`、`torchvision`、`triton`

    例如 `apex==0.1+metax3.8.2.2` 是错误的，正确版本为 `0.1+metax3.8.2.2torch2.10`。对每个缺少 `torch2.10` 后缀的受影响包，执行卸载并重新安装：

    ```bash
    pip uninstall apex
    pip install apex==0.1+metax3.8.2.2torch2.10 \
        -i https://repos.metax-tech.com/r/maca-pypi/simple \
        --extra-index-url https://pypi.org/simple \
        --trusted-host repos.metax-tech.com
    ```

### 检查环境

验证安装：

```bash
python -c 'import vllm_metax; print(vllm_metax.__file__)'
python -c 'import torch; print(torch.cuda.is_available())'
```

之后就可以启动一个模型来测试，参考部署指南中的[启动推理服务](../deployment/zh.md#4-start-the-inference-service)。