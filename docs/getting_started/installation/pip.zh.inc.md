<!-- markdownlint-disable MD025 -->

# --8<-- [start:prepare-env]

!!! note
    使用 pip 时，所有构建与安装步骤均***基于对应的 docker 镜像***。镜像可在[快速开始页面](../quickstart/zh.md)找到。
    整个包构建期间需添加 `--no-build-isolation` 参数，因为需要依赖发布 docker 镜像中预装的全部依赖。

# --8<-- [end:prepare-env]

# --8<-- [start:build-vllm-metax]

!!! note

    ```bash
    python use_existing_metax.py
    pip install -r requirements/build.txt
    pip install .  --no-build-isolation
    ```

    ??? console "其他安装选项"
        如果你想开发 vllm-metax，请改用**可编辑模式**安装。

        ```bash
        pip install -v -e . --no-build-isolation
        ```

        可选：构建一个便携 wheel，供其他环境安装。

        ```bash 
        python -m build -w -n 
        pip install dist/*.whl
        ```

# --8<-- [end:build-vllm-metax]

# --8<-- [start:build-vllm]

!!! note "使用已安装的 PyTorch 构建 vllm-metax"

    ```bash
    python use_existing_pytorch.py
    pip install -r requirements/build/cuda.txt
    VLLM_TARGET_DEVICE=empty pip install . --no-build-isolation
    ```

# --8<-- [end:build-vllm]

# --8<-- [start:post-build]

!!! note
    暂无

# --8<-- [end:post-build]
