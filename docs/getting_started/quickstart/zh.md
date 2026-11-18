# 快速开始

[English](/getting_started/quickstart/en.md){ .md-button }

目前推荐使用 ***docker*** 方式启动 ***vLLM-MetaX***。

- 如果使用 docker 部署，请参考 [部署指南](../deployment/zh.md) 完成从拉取镜像、启动容器到验证服务的完整步骤。
- 如果要构建最新版的 vllm-metax，请参考[从源码构建](../installation/zh.md#build-from-source)。
- 如果要安装 vllm-metax 的发布 wheel，请参考[使用发布 Wheel 安装](../installation/zh.md#install-the-release-wheel)。

!!! note
    v0.11.2 之后，vllm-metax 将 `_C` 和 `_moe_C` 内核迁移到了独立包 `mcoplib`。

    **mcoplib** 已在 [MetaX-mcoplib](https://github.com/MetaX-MACA/mcoplib) 开源，并维护自己的发布周期。使用 vLLM-MetaX 时，请始终安装对应版本的 mcoplib。

    仓库中仍保留 *csrc* 目录以方便开发，但代码不保证与 mcoplib 同步。其性能乃至正确性都可能与 mcoplib 存在差异。

    ***生产环境请始终使用 mcoplib。***

## 版本映射 {#releases}

*以下是已发布插件、mcoplib 与 maca 的版本对应关系*：

| 插件版本 | maca 版本 | mcoplib 版本 | docker 镜像地址 |
| :---: | :---: | :---: | :---: |
| v0.28.0 | maca3.8.2.x | 0.4.13 | [vllm-metax:0.28.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.28.0-torch2.10) |
| v0.27.0 | maca3.8.2.x | 0.4.12 | [vllm-metax:0.27.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.27.0-torch2.10) |
| v0.26.0 | maca3.8.2.x | 0.4.11 | [vllm-metax:0.26.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.26.0-torch2.10) |
| v0.25.0 | maca3.8.2.x | 0.4.10 | [vllm-metax:0.25.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.25.0-torch2.10) |
| v0.24.0 | maca3.8.2.x | 0.4.9 | [vllm-metax:0.24.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.24.0-torch2.10) |
| v0.23.0 | maca3.8.0.x | 0.4.8 | [vllm-metax:0.23.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.23.0-torch2.10) |
| v0.22.0 | maca3.8.0.x | 0.4.7 | [vllm-metax:0.22.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.22.0-torch2.10) |
| v0.21.0 | maca3.7.1.x | 0.4.6 | [vllm-metax:0.21.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.21.0-torch2.8) |
| v0.20.0 | maca3.7.0.x | 0.4.5 | [vllm-metax:0.20.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.20.0-torch2.8) |
| v0.19.0 | maca3.5.3.x | 0.4.4 | [vllm-metax:0.19.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.19.0-torch2.8) |
| v0.18.0 | maca3.5.3.x | 0.4.3 | [vllm-metax:0.18.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.18.0-torch2.8) |
| v0.17.0 | maca3.5.3.x | 0.4.2 | [vllm-metax:0.17.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.17.0-torch2.8) |
| v0.16.0 | N/A | N/A | Skipped |
| v0.15.0 | maca3.5.3.x | 0.4.1 | [vllm-metax:0.15.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.15.0-torch2.8) |
| v0.14.0 | maca3.5.3.x | 0.4.0 | [vllm-metax:0.14.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.14.0-torch2.8) |
| v0.13.0 | maca3.3.0.x | 0.3.1 | [vllm-metax:0.13.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.13.0-torch2.8) |
| v0.12.0 | maca3.3.0.x | 0.3.0 | [vllm-metax:0.12.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.12.0-torch2.8) |
| v0.11.2 | maca3.3.0.x | 0.2.0 | [vllm-metax:0.11.2](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.11.2-torch2.8) |
| v0.11.0 | maca3.3.0.x | 0.1.1 | [vllm-metax:0.11.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.11.0-torch2.6) |
| v0.10.2 | maca3.2.1.7 | N/A | [vllm-metax:0.10.2](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64) |
| v0.9.1 | maca3.0.0.5 | N/A | [vllm:0.9.1](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64) |
| v0.8.5 | maca2.33.1.13 | N/A | [vllm:0.8.5](https://developer.metax-tech.com/softnova/docker?package_name=vllm:maca.ai2.33.0.13-torch2.6-py310-ubuntu22.04-amd64) |

!!! warning "使用警告"
    **上述 docker 镜像已开箱即用（out of box）。**

    所有 vllm 测试均基于对应的 maca 版本。为 vllm 使用不匹配的 maca 版本可能导致意外 bug 或错误，无法保证可靠性。
