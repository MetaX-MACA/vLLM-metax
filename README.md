# vLLM-MetaX

<p align="center">
  <b>在沐曦 GPU 上流畅运行 vLLM</b>
</p>

<p align="center">
  <a href="https://vllm-metax.readthedocs.io">
    <img src="https://readthedocs.org/projects/vllm-metax/badge/?version=latest" alt="Docs"/>
  </a>
  <a href="https://pypi.org/project/vllm-metax/">
    <img src="https://img.shields.io/pypi/v/vllm-metax?color=%23orange" alt="PyPI"/>
  </a>
  <a href="https://github.com/MetaX-MACA/vLLM-metax/stargazers">
    <img src="https://img.shields.io/github/stars/MetaX-MACA/vllm-metax" alt="Stars"/>
  </a>
  <a href="https://github.com/MetaX-MACA/vLLM-metax/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/MetaX-MACA/vllm-metax" alt="License"/>
  </a>
</p>

---

## ✨ 特性

- 🚀 **高性能推理** - 沐曦 C 系列 GPU 原生支持
- 🔄 **无缝集成** - 遵循 vLLM 插件规范，即插即用
- 🐳 **开箱即用** - 提供官方 Docker 镜像
- 📖 **完整文档** - 配套中文/英文教程
- 🏠 **社区活跃** - 北京/上海 Meetup 已举办多场

---

## 🏃 快速开始

### 方式一：Docker（推荐）

```bash
# 拉取镜像
docker pull metax/vllm-metax:v0.13.0

# 运行
docker run --gpus all -v /path/to/models:/models \
  metax/vllm-metax:v0.13.0 \
  vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1
```

### 方式二：pip 安装

```bash
pip install vllm-metax
```

### 方式三：源码编译

详见 [安装指南](https://vllm-metax.readthedocs.io/en/latest/getting_started/installation/maca.html)

---

## 🖥️ 支持的模型

| 模型系列 | 状态 | 示例 |
|---------|------|------|
| Qwen | ✅ 支持 | Qwen2.5-7B-Instruct |
| Llama | ✅ 支持 | Llama-3-8B |
| DeepSeek | ✅ 支持 | DeepSeek-V2 |
| Yi | ✅ 支持 | Yi-6B |
| Baichuan | ✅ 支持 | Baichuan2-7B |

> 完整支持列表见 [模型文档](https://vllm-metax.readthedocs.io/en/latest/models/supported_models.html)

---

## 💻 硬件支持

| 型号 | 显存 | 状态 | 推荐场景 |
|-----|------|------|---------|
| 沐曦 C500 | 32GB | ✅ 生产可用 | 大模型推理 |
| 沐曦 C400 | 16GB | ✅ 实验阶段 | 开发测试 |
| 沐曦 C300 | 8GB | 🔄 开发中 | 轻量推理 |

---

## 📖 文档

- [📚 官方文档](https://vllm-metax.readthedocs.io)
- [🚀 快速开始](https://vllm-metax.readthedocs.io/en/latest/getting_started/quickstart.html)
- [🐛 常见问题](https://vllm-metax.readthedocs.io/en/latest/faq.html)
- [💬 GitHub Discussions](https://github.com/MetaX-MACA/vLLM-metax/discussions)

---

## 🆕 更新日志

### 2026.02 v0.13.0
- 🎉 对齐 vLLM v0.13.0
- ✨ 新增支持 DeepSeek-V2 系列
- ⚡ 性能优化：推理吞吐量提升 15%

### 2026.01 v0.12.0
- ✨ 新增 Qwen2.5 系列支持
- 🐛 修复多项 bug

[查看完整更新日志 →](CHANGELOG.md)

---

## 🤝 贡献

欢迎贡献！请阅读 [贡献指南](CONTRIBUTING.md)。

```bash
# 克隆项目
git clone https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax

# 创建开发分支
git checkout -b feature/your-feature

# 提交 PR
```

---

## 📞 社区

- 💬 Slack: [#sig-maca](https://slack.vllm.ai)
- 🐙 GitHub Issues: [提 Bug/需求](https://github.com/MetaX-MACA/vLLM-metax/issues)
- 📧 邮件列表: dev@metax-tech.com

---

## 📄 许可证

Apache License 2.0 - 查看 [LICENSE](LICENSE)

---

<p align="center">
  <sub>Made with ❤️ by MetaX Team</sub>
</p>
