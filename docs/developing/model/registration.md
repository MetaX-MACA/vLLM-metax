# Registering a Model

vllm-metax basically reuses the models that are already registered by vLLM.

If your model is not on this list, or you have customized a registered model, you must register it to vLLM.
This page provides detailed instructions on how to do so.

## Out-of-tree models

You can load an external model [using a plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) without modifying the vLLM codebase.

To register the model, use the following code:

```python
# The entrypoint of your plugin
def register():
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

If your model imports modules that initialize CUDA, consider lazy-importing it to avoid errors like `RuntimeError: Cannot re-initialize CUDA in forked subprocess`:

```python
# The entrypoint of your plugin
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM",
    )
```

!!! Note
    This is best practice on vllm-metax. Please review it [here](https://github.com/MetaX-MACA/vLLM-metax/blob/master/vllm_metax/models).
