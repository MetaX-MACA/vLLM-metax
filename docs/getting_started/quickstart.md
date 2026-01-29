# Quickstart

Currently the recommanded way to start ***vLLM-MetaX*** is via *docker*.

You could get the docker image at [MetaX develop community](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax&arch=amd64&system=ubuntu).

!!! note
    After v0.11.2, vllm-metax moved its `_C` and `_moe_C` kernel into a separate package named `mcoplib`. 
    
    **mcoplib** is open-sourced at [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) and would maintain its own release cycle. Please always install the corresponding version of mcoplib when using vLLM-MetaX.

    Though the *csrc* folder is still kept in this repo for development convenience, and there is no guarantee that the code is always in sync with mcoplib. Not only the performance but also the correctness may differ from mcoplib. 

    If you want build the latest vllm-metax, please refer to [installation](./installation/maca.md) to build from source.

    **Please always use mcoplib for production usage.**

## Releases

*Belows is version mapping to released plugin and mcoplib with maca*:

| plugin version | maca version | mcoplib version | docker distribution tag |
|:--------------:|:------------:|:-----------------------:|:-----------------------:|
|v0.8.5          |maca2.33.1.13 | N/A | vllm:maca.ai2.33.1.13-torch2.6-py310-ubuntu22.04-amd64 |
|v0.9.1          |maca3.0.0.5   | N/A | vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64 |
|v0.10.2         |maca3.2.1.7   | N/A | vllm-metax:0.10.2-maca.ai3.2.1.7-torch2.6-py310-ubuntu22.04-amd64 |
|v0.11.0         |maca3.3.0.15   | 0.1.1 | vllm-metax:0.11.0-maca.ai3.3.0.11-torch2.6-py312-ubuntu22.04-amd64 |
|v0.11.2         |maca3.3.0.15   | 0.2.0 | vllm-metax:0.11.2-maca.ai3.3.0.103-torch2.8-py312-ubuntu22.04-amd64 |
|v0.12.0         |maca3.3.0.15   | 0.3.0 | vllm-metax:0.12.0-maca.ai3.3.0.204-torch2.8-py312-ubuntu22.04-amd64 |
|master          |maca3.3.0.15   | >=0.3.0 | not released |

!!! warning "Usage Warning"
    **vLLM-MetaX is out of box via these docker images.**

    All the vllm tests are based on the related maca version. Using incorresponding version of maca for vllm may cause unexpected bugs or errors. This is not garanteed.
