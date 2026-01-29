# --8<-- [start:prepare-env]
!!! note
    If using pip, all the build and installation steps are ***based on corresponding docker images***. You can find them on [QuickStart page](../quickstart.md).
    We need to add `--no-build-isolation` flag during the whole package building since we need all the requirements that were pre-installed in released docker image.
# --8<-- [end:prepare-env]

# --8<-- [start:build-vllm-metax]
!!! note

    ```bash
    python use_existing_metax.py
    pip install -r requirements/build.txt
    pip install .  --no-build-isolation
    ```

    ??? console "Additional installation options"
        If you want to develop vllm-metax, install it in **editable mode** instead.

        ```bash
        pip install -v -e . --no-build-isolation
        ```

        Optionally, build a portable wheel which you can then install elsewhere.

        ```bash 
        python -m build -w -n 
        pip install dist/*.whl
        ```
# --8<-- [end:build-vllm-metax]

# --8<-- [start:build-vllm]
!!! note "To build vllm-metax using an existing PyTorch installation"

    ```bash
    python use_existing_pytorch.py
    pip install -r requirements/build.txt
    VLLM_TARGET_DEVICE=empty pip install . --no-build-isolation
    ```
# --8<-- [end:build-vllm]
