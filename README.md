<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/MetaX-MACA/vLLM-metax/master/docs/assets/logos/vllm-metax-logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM MetaX Plugin
</h3>

<p align="center">
| <a href="https://www.metax-tech.com/en/"><b>About MetaX</b></a> | <a href="https://vllm-metax.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#sig-maca</b></a> </a> |
</p>

---

*Latest News* 🔥
- [2026/1] Released vllm-metax **v0.11.2** 👻 — aligned with vLLM *v0.11.2*, supported more models and improved performance.
- [2025/11] Released vllm-metax **v0.10.2** 🎉 — aligned with vLLM *v0.10.2*, improved model performance, and fixed key decoding bugs.
- [2025/11] We hosted [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/xSrYXjNgr1HbCP4ExYNG1w) focusing on distributed inference and diverse accelerator support with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1nQJ8ZkLSjKxvu36sSHaceVXtttbLvvu-?usp=drive_link).
- [2025/08] We hosted [vLLM Shanghai Meetup](https://mp.weixin.qq.com/s/pDmAXHcN7Iqc8sUKgJgGtg) focusing on building, developing, and integrating with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1OvLx39wnCGy_WKq8SiVKf7YcxxYI3WCH).


## About

vLLM MetaX is a hardware plugin for running vLLM seamlessly on MetaX GPU, which is a cuda_alike backend and provided near-native CUDA experiences on MetaX Hardware with [*MACA*](https://www.metax-tech.com/en/goods/platform.html?cid=4).

It is the recommended approach for supporting the MetaX backend within the vLLM community. 

The plugin follows the vLLM plugin RFCs by default:
 - [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)
 - [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161)

Which ensured the hardware features and functionality support on integration of the MetaX GPU with vLLM.

## Prerequisites

- Hardware: MetaX C-series
- OS: Linux
- Software:
  - Python >= 3.9, < 3.12
  - vLLM (the same version as vllm-metax)
  - Docker support

## Getting Started

vLLM MetaX currently only support starting on docker images release by [MetaX develop community](https://developer.metax-tech.com/softnova/docker) which is out of box. (DockerFile for other OS is undertesting)

If you want to develop, debug or test the newest feature on vllm-metax, you may need to build from scratch and follow this [*source build tutorial*](https://vllm-metax.readthedocs.io/en/latest/getting_started/installation/maca.html). 

## Branch

vllm-metax has master branch and dev branch.

- **master**: main branch，catching up with main branch of vLLM upstream.
- **vX.Y.Z-dev**: development branch, created with part of new releases of vLLM. For example, `v0.10.2-dev` is the dev branch for vLLM `v0.10.2` version.

Below is maintained branches:

| Branch      | Status       | Note                                 |
|-------------|--------------|--------------------------------------|
| master      | N/A | trying to support vllm main, no gurantee on functionality |
| v0.14.0-dev | N/A | under testing |
| v0.13.0-dev | N/A | under testing |
| v0.12.0-dev | N/A | under testing |
| releases/v0.11.2 | Released | related to vllm release v0.11.2 |
| releases/v0.10.2 | Released | related to vllm release v0.10.2 |

Please check [here](https://vllm-metax.readthedocs.io/en/latest/getting_started/quickstart.html) for more details .

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.

