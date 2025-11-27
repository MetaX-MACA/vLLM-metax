# Quickstart

Currently the recommanded way to start ***vLLM-MetaX*** is via *docker*.

You could get the docker image at [MetaX develop community](https://developer.metax-tech.com/softnova/docker).

*Belows is version mapping to released plugin and maca*:

| plugin version | maca version | docker distribution tag |
|:--------------:|:------------:|:-----------------------:|
|v0.8.5          |maca2.33.1.13 | vllm:maca.ai2.33.1.13-torch2.6-py310-ubuntu22.04-amd64 |
|v0.9.1          |maca3.0.0.5   | vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64 |
|v0.10.1.1 (dev) |maca3.1.0.7   | not released |
|v0.10.2         |maca3.2.1.7   | vllm-metax:0.10.2-maca.ai3.2.1.7-torch2.6-py310-ubuntu22.04-amd64 |
|v0.11.0         |maca3.3.x.x   | not released |
|v0.11.1rc6      |maca3.3.x.x   | not released |
|master          |maca3.4.x.x.  | not released |

> Note: All the vllm tests are based on the related maca version. Using incorresponding version of maca for vllm may cause unexpected bugs or errors. This is not garanteed.

vLLM-MetaX is out of box via these docker images.

## Offline Batched Inference

## OpenAI-Compatible Server

## On Attention Backends

