# --8<-- [start:prepare-env]
!!! note

    UV **does not rely** on any pre-installed packages in the docker, and would install all the dependencies in a virtual environment from scratch.

    ??? console "UV installation guide"
        We'd recommend install uv with pip (this is not forcibly required):

        ```bash
        pip install uv
        ```

        Then create the virtual environment with python 3.10 or above:

        ```bash
        uv venv .venv --python python3.10
        ```

        And activate the virtual environment:

        ```bash
        source .venv/bin/activate
        ```

    You need to manually set Metax PyPi repo to download maca-related dependencies.

    ```
    export UV_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
    export UV_INDEX_STRATEGY=unsafe-best-match
    ```

    ??? console "Optional: Change PyPi default mirror"
        You could set Aliyun PyPi mirror as default to speed up package downloading:

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

    ??? console "Additional installation options"
        If you want to develop vLLM, install it in editable mode instead.

        ```bash
        uv pip install -v -e .
        ```

        Optionally, build a portable wheel which you can then install elsewhere.

        ```bash 
        uv build --wheel
        ```
# --8<-- [end:build-vllm-metax]



# --8<-- [start:build-vllm]
!!! note "To build vLLM using local uv environment"

    ```bash
    VLLM_TARGET_DEVICE=empty uv pip install . --no-build-isolation
    ```

    ??? note "About isolation"
        `--no-build-isolation` is optional. we add this option for speeding up installation.
        uv would still trying to download cuda-related packages during build even if you set 
        `VLLM_TARGET_DEVICE=empty`, which may take a long time.

# --8<-- [end:build-vllm]