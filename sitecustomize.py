try:
    import vllm_metax
    vllm_metax.register_early_patch()
except Exception as e:
    print(f"[sitecustomize] failed to apply metax early patch: {e}")