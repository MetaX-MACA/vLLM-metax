---
hide:
  - navigation
  - toc
---

# Welcome to vLLM-MetaX

<figure markdown="span">
  ![](./assets/logos/vllm-metax-logo.png){ align="center" alt="vLLM Light" class="logo-light" width="60%" }
  <!-- ![](./assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM Dark" class="logo-dark" width="60%" } -->
</figure>

<p style="text-align:center">
<strong>MetaX Hardware Backend Plugin
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/MetaX-MACA/vLLM-metax" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/MetaX-MACA/vLLM-metax/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/MetaX-MACA/vLLM-metax/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>


vLLM MetaX is a hardware plugin for running vLLM seamlessly on MetaX GPU, which is a cuda_alike backend and provided near-native CUDA experiences on MetaX Hardware with [*MACA*](https://www.metax-tech.com/en/goods/platform.html?cid=4).

It is the recommended approach for supporting the MetaX backend within the vLLM community. 

The plugin follows the vLLM plugin RFCs by default:

 - [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)
 - [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161)

Which ensured the hardware features and functionality support on integration of the MetaX GPU with vLLM.