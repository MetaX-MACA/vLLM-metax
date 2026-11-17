# Online Hadamard rotation

The TileLang kernel cache includes the tensor GPU index. Compilation and
execution run in that GPU context, and kernel elaboration explicitly binds
its compiler thread to the same GPU. This prevents a kernel first compiled
on one GPU from being reused on another GPU and restores the caller's device.
The transform, tuning candidates, and normalization are unchanged.

`tests/kernels/test_online_rotation_device.py` checks FP16/BF16 forward and
inverse results against an independent CPU float64 Hadamard calculation. It
covers compiler threads, GPU 0/1 cache entries, return to GPU 0, caller device
restoration, and warm graph capture/replay on both GPUs.

Run in the matching source-provider MetaX environment:

```bash
USE_PRECOMPILED_KERNEL=0 python -m pytest --confcutdir=tests/kernels \
  tests/kernels/test_online_rotation_device.py
```
