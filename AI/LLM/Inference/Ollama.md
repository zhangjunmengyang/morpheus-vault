---
title: "Ollama"
brief: "本地运行 LLM 的轻量工具：一行命令拉取 Llama/Qwen/Mistral 等模型并在 CPU/GPU 上推理。支持 REST API（OpenAI 兼容）、Modelfile 自定义、多模型并发。是个人开发者和快速原型阶段的事实标准本地推理工具。"
type: concept
domain: ai/llm/inference
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/inference
  - type/concept
---
# Ollama

## 概述

Ollama 是一个本地 LLM 推理工具，目标是让运行大模型像用 Docker 一样简单。一行命令拉取模型、一行命令启动推理，面向的是个人开发者和本地部署场景。

底层推理引擎基于 **llama.cpp**（C/C++ 实现的 LLM 推理库），Ollama 在其上封装了模型管理、API 服务、量化模型分发等功能。

## 核心设计

### Modelfile

类比 Dockerfile，Ollama 用 Modelfile 定义模型配置：

```dockerfile
FROM llama3.1:8b

# 系统提示
SYSTEM """You are a helpful coding assistant."""

# 推理参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

# 模板（chat template）
TEMPLATE """{{ if .System }}<|system|>{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>{{ .Prompt }}<|end|>
<|assistant|>{{ end }}{{ .Response }}"""
```

### 模型管理

```bash
# 拉取模型（自动选择适合硬件的量化版本）
ollama pull qwen2.5:7b

# 查看本地模型
ollama list

# 运行（交互式）
ollama run qwen2.5:7b

# 从 Modelfile 创建自定义模型
ollama create my-model -f Modelfile

# 删除
ollama rm qwen2.5:7b
```

### API 服务

Ollama 启动后自动监听 `localhost:11434`，提供 REST API：

```bash
# Generate（非 chat 模式）
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat（对话模式）
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'

# Embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "The quick brown fox"
}'
```

也可以通过 OpenAI-compatible endpoint 调用（`/v1/chat/completions`），方便对接现有代码。

## 量化与硬件适配

Ollama 默认分发 **GGUF 格式**的量化模型（llama.cpp 原生格式）：

| 量化级别 | 模型大小 (7B) | 质量 | 推荐场景 |
|---------|-------------|------|---------|
| Q2_K | ~2.7 GB | 较低 | 极度显存受限 |
| Q4_K_M | ~4.1 GB | 不错 | **日常推荐** |
| Q5_K_M | ~4.8 GB | 好 | 质量优先 |
| Q8_0 | ~7.2 GB | 接近 FP16 | 有充裕显存 |
| FP16 | ~14 GB | 无损 | 评测/对比用 |

硬件支持：
- **Apple Silicon**：原生 Metal 加速，M1/M2/M3 都支持，统一内存对大模型友好
- **NVIDIA GPU**：CUDA 加速
- **AMD GPU**：ROCm 支持（相对没那么成熟）
- **纯 CPU**：可以跑，但很慢

### Apple Silicon 的优势

在 Mac 上跑 LLM，Ollama 可能是最佳选择：
- 统一内存架构意味着 CPU 和 GPU 共享内存池
- M4 Pro 36GB 可以流畅跑 Q4 量化的 32B 模型
- Metal Performance Shaders 提供 GPU 加速

## 实际使用场景

### 1. 本地开发与测试

```python
# Python SDK
import ollama

response = ollama.chat(model='qwen2.5:7b', messages=[
    {'role': 'user', 'content': 'Write a Python quicksort'}
])
print(response['message']['content'])
```

### 2. 作为 Agent 的本地 LLM 后端

很多 Agent 框架支持 Ollama 作为 backend：
- LangChain / LlamaIndex 原生支持
- 配合 Open WebUI 提供 ChatGPT 式界面
- 适合隐私敏感场景（数据不出本地）

### 3. RAG 应用

Ollama 同时提供 chat 和 embedding API，可以搭建完整的本地 RAG pipeline：

```python
# Embedding
embedding = ollama.embeddings(model='nomic-embed-text', prompt='your text')

# 结合向量数据库（如 ChromaDB）做检索
# 然后用 chat API 做生成
```

## Ollama vs vLLM

| 维度 | Ollama | vLLM |
|------|--------|------|
| 定位 | 个人/本地 | 生产/服务端 |
| 吞吐量 | 中等 | 高（continuous batching） |
| 并发 | 有限 | 强 |
| 量化 | GGUF (llama.cpp) | AWQ/GPTQ/FP8 |
| 硬件 | Apple Silicon 友好 | NVIDIA GPU 为主 |
| 模型管理 | 内置（类 Docker） | 依赖 HuggingFace |
| 学习成本 | 极低 | 中等 |

简单说：**本地玩 → Ollama，线上部署 → vLLM**。

## 局限性

1. **不适合高并发**：底层 llama.cpp 的 batching 能力远不如 vLLM
2. **模型格式受限**：主要支持 GGUF，不是所有模型都有现成的 GGUF 版本
3. **功能较简单**：没有 speculative decoding、prefix caching 等高级优化
4. **自定义模型繁琐**：需要先转成 GGUF 格式

## See Also

**本地推理生态**
- [[AI/LLM/Inference/vLLM|vLLM]] — 生产级推理引擎对比（Ollama 的进阶替代）
- [[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]] — vLLM 核心机制手撕（Continuous Batching / PagedKVCache）
- [[AI/LLM/Inference/LLM-推理优化-2026-全景|LLM推理优化2026全景]] — 推理优化技术全貌

**量化与模型格式**
- [[AI/LLM/Frameworks/Unsloth/量化|量化]] — GGUF 量化技术详解
- [[AI/LLM/Application/Embedding/Embedding|Embedding]] — Ollama 的 embedding 功能

## 推荐阅读

1. **官方文档**：[ollama.ai](https://ollama.ai) — 模型库 + API 参考
2. **对比测试**：Ollama vs vLLM throughput benchmarks — 理解使用场景边界
