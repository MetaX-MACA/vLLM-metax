# Quickstart

Currently the recommended way to start ***vLLM-MetaX*** is via *docker*.

You could get the docker image at [MetaX develop community](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&arch=amd64&system=ubuntu).

!!! note
    After v0.11.2, vllm-metax moved its `_C` and `_moe_C` kernels into a separate package named `mcoplib`. 
    
    **mcoplib** is open-sourced at [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) and would maintain its own release cycle. Please always install the corresponding version of mcoplib when using vLLM-MetaX.

    Though the *csrc* folder is still kept in this repo for development convenience, and there is no guarantee that the code is always in sync with mcoplib. Not only the performance but also the correctness may differ from mcoplib. 

    If you want build the latest vllm-metax, please refer to [installation](./installation/maca.md) to build from source.

    ***Please always use mcoplib for production usage.***

## Releases

*Below is version mapping to released plugin and mcoplib with maca*:

| plugin version | maca version | mcoplib version | docker image url |
|:--------------:|:------------:|:-----------------------:|:-----------------------:|
|v0.8.5          |maca2.33.1.13 | N/A | [vllm:0.8.5](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai2.33.0.13-torch2.6-py310-ubuntu22.04-amd64) |
|v0.9.1          |maca3.0.0.5   | N/A | [vllm:0.9.1](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64) |
|v0.10.2         |maca3.2.1.7   | N/A | [vllm-metax:0.10.2](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64) |
|v0.11.0         |maca3.3.0.x   | 0.1.1 | [vllm-metax:0.11.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.11.0-torch2.6) |
|v0.11.2         |maca3.3.0.x   | 0.2.0 | [vllm-metax:0.11.2](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.11.2-torch2.8) |
|v0.12.0         |maca3.3.0.x   | 0.3.0 | [vllm-metax:0.12.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.12.0-torch2.8) |
|v0.13.0         |maca3.3.0.x   | 0.3.1 | [vllm-metax:0.13.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.13.0-torch2.8) |
|v0.14.0         |maca3.5.3.x   | 0.4.0 | [vllm-metax:0.14.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.14.0-torch2.8) |
|v0.15.0         |maca3.5.3.x   | 0.4.1 | [vllm-metax:0.15.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.15.0-torch2.8) |
|v0.16.0         | N/A | N/A | **Skipped** |
|v0.17.0         |maca3.5.3.x   | 0.4.2 | [vllm-metax:0.17.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.17.0-torch2.8) |
|v0.18.0         |maca3.5.3.x   | 0.4.3 | [vllm-metax:0.18.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.18.0-torch2.8) |
|v0.19.0         |maca3.5.3.x   | 0.4.4 | [vllm-metax:0.19.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.19.0-torch2.8) |
|v0.20.0         |maca3.7.0.x   | 0.4.5 | [vllm-metax:0.20.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.20.0-torch2.8) |

!!! warning "Usage Warning"
    **vLLM-MetaX is out of box via docker images provided above.**

    All the vllm tests are based on the related maca version. Using incorresponding version of maca for vllm may cause unexpected bugs or errors. This is not guaranteed.
## Install via pip (pre-built wheels)

If you are on a cloud instance that already ships the MACA SDK and a MetaX build
of PyTorch (e.g. Gitee.AI / 模力方舟 images), you can install vllm-metax from the
MetaX PyPI index instead of using docker:

```bash
pip install "vllm-metax==<version>" \
    -i https://repos.metax-tech.com/r/maca-pypi/simple \
    --trusted-host repos.metax-tech.com
```

Pick the wheel that matches the **MACA runtime** (`mx-smi | grep "MACA Version"`,
major.minor must match), the **torch build** (`pip show torch`) and your Python:

| vllm-metax wheel | MACA runtime | torch | mcoplib |
|:---|:---|:---|:---|
| 0.13.0+...maca3.3.0.15.torch2.8 | 3.3.x | 2.8.0+metax3.3.x | bundled in wheel |
| 0.17.0+...maca3.5.3.20.torch2.8 | 3.5.x | 2.8.0+metax3.5.x | 0.4.2 (install separately) |
| 0.19.0+...maca3.5.3.20.torch2.8 | 3.5.x | 2.8.0+metax3.5.x | 0.4.4 (install separately) |
| 0.20.0+...maca3.7.0.37.torch2.8 | 3.7.x | 2.8.0+metax3.7.x | 0.4.5 (install separately) |

Wheels up to v0.13.0 bundle `_C`/`_moe_C`; from v0.14.0 on, install the matching
`mcoplib` from the same index (see the release table above). mcoplib checks the
MACA runtime at import time and only requires major.minor to match.

The upstream `vllm` package is not hosted on the MetaX index; install the
matching version from PyPI without its CUDA dependency set:

```bash
pip download vllm==<version> --no-deps -d /tmp/vllm-whl -i https://pypi.org/simple
pip install /tmp/vllm-whl/vllm-*.whl --no-deps
```

### Common pitfalls

- **`No matching distribution found for torch==2.7.0` while resolving
  `arctic-inference`**: pip picked a vllm-metax wheel built for a different
  torch than the installed MetaX torch. Pin the full wheel version (including
  the `+...macaX.Y.Z.torchA.B` suffix) instead of letting pip resolve the
  latest.
- **mcoplib refuses to load (`Check the current MACA version`)**: the MACA
  runtime major.minor differs from the mcoplib build. Switch the instance
  image rather than mixing versions; e.g. mcoplib 0.4.x will not run on a
  3.3.x runtime.
- **Wrong device count on some single-GPU instances**: torch can report an
  extra device; set `CUDA_VISIBLE_DEVICES=0`.
- **Plain `pip install vllm` is a cuda build** with extra dependencies and
  preconditions that may pass cuda-only checks on maca; install with
  `--no-deps` as above (see warning in Installation).
