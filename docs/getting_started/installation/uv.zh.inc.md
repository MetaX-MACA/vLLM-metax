<!-- markdownlint-disable MD025 -->

# --8<-- [start:prepare-env]

!!! note

    UV **不依赖** docker 中预装的任何包，会从零开始在虚拟环境中安装全部依赖。

    ??? console "UV 安装指南"
        建议先用 pip 安装 uv（非强制要求）：

        ```bash
        pip install uv
        ```

        然后用 python 3.10 及以上版本创建虚拟环境：

        ```bash
        uv venv /opt/venv --python python3.10
        ```

        并激活虚拟环境：

        ```bash
        source /opt/venv/bin/activate
        ```

    安装过程中需要手动设置 MetaX PyPi 仓库，以获取 *maca 相关* 依赖。

    ```
    export UV_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
    export UV_INDEX_STRATEGY=unsafe-best-match
    ```

    ??? console "可选：更换默认 PyPi 镜像"
        可将阿里云 PyPi 镜像设为默认源，加速 *非 metax 相关* 包的安装：

        ```bash
        export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
        ```

# --8<-- [end:prepare-env]

# --8<-- [start:build-vllm-metax]

!!! note

    ```bash
    uv pip install -r requirements/build.txt
    uv pip install . 
    ```

    ??? console "其他安装选项"
        如果你想开发 vLLM，请改用可编辑模式安装。

        ```bash
        uv pip install -v -e .
        ```

        可选：构建便携 wheel。

        ```bash 
        uv build --wheel
        ```

# --8<-- [end:build-vllm-metax]

# --8<-- [start:build-vllm]

!!! note "使用本地 uv 环境构建 vLLM"

    ```bash
    VLLM_TARGET_DEVICE=empty uv pip install . --no-build-isolation
    ```

    ??? note "关于隔离"
        `--no-build-isolation` 为可选参数，添加它是为了加速安装。
        即使设置了 `VLLM_TARGET_DEVICE=empty`，uv 在构建过程中仍会尝试下载 cuda 相关包，耗时可能较长。

# --8<-- [end:build-vllm]

# --8<-- [start:post-build]

!!! note
    暂无

# --8<-- [end:post-build]