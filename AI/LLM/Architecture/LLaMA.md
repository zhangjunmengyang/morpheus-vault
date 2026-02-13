---
title: "LLaMA"
type: paper
domain: ai/llm/architecture
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/architecture
  - type/paper
---
# LLaMA

LLaMA（Large Language Model Meta AI）是 Meta 于 2023 年发布的开源大语言模型系列，从根本上改变了开源 LLM 的格局。如果说 GPT 系列是闭源模型的标杆，LLaMA 就是开源世界的基石——几乎所有主流的开源模型（Qwen、DeepSeek、Mistral 等）都直接或间接受到 LLaMA 架构的影响。

## LLaMA 1：开源的起点

LLaMA 1（2023.02）的核心贡献是证明了一个关键命题：**在给定计算预算下，用更多数据训练更小的模型，比训练更大的模型效果更好**。

这一洞察来自 Chinchilla scaling laws（DeepMind）——Chinchilla 用 1.4T tokens 训练 70B 模型，效果优于用 300B tokens 训练的 280B Gopher。LLaMA 将这个思路推到极致：用 **1.4T tokens** 训练 7B/13B/33B/65B 四个规模的模型。

关键架构选择：
- **Pre-RMSNorm**：用 RMSNorm 替代 LayerNorm，放在每个 sub-layer 之前（Pre-Norm 结构），训练更稳定
- **SwiGLU 激活函数**：替代 ReLU/GELU，来自 PaLM 的实践
- **Rotary Position Embedding（RoPE）**：旋转位置编码，天然支持相对位置建模和长度外推

$$\text{RoPE}(x_m, m) = x_m e^{im\theta}$$

其中 $\theta_j = 10000^{-2j/d}$，本质上是将位置信息编码为旋转角度。

- **无 bias**：所有线性层都不使用 bias term

这套架构选择现在被视为"现代 LLM 的标准配置"——Qwen、DeepSeek、Mistral 基本都采用了相同的组合。

## LLaMA 2：对话与对齐

LLaMA 2（2023.07）的重要升级：

1. **更大的训练数据**：从 1.4T 增加到 **2T tokens**
2. **更长的上下文**：从 2048 扩展到 **4096**
3. **Grouped Query Attention（GQA）**：34B 和 70B 版本使用 GQA 替代标准 MHA，推理效率显著提升

GQA 的关键思想：在 KV heads 的数量上做折中。MHA 每个注意力头都有独立的 K、V，GQA 让多个 Q heads 共享同一组 K、V：

```
MHA:  Q头数=K头数=V头数=32
MQA:  Q头数=32, K头数=V头数=1
GQA:  Q头数=32, K头数=V头数=8  ← LLaMA 2 的选择
```

4. **LLaMA 2-Chat**：通过 RLHF 训练的对话版本，使用了超过 100 万条人类标注数据

Meta 同时发布了一份极其详细的训练技术报告，包括 RLHF 的细节、安全评估等，对开源社区的价值巨大。

## LLaMA 3 / 3.1 / 3.2：全面扩展

LLaMA 3（2024.04）是一次全面的升级：

- **Tokenizer 升级**：从 SentencePiece（32K 词汇表）换到 tiktoken（**128K 词汇表**），编码效率提升约 15%
- **训练数据**：超过 **15T tokens**，数据质量大幅提升
- **上下文窗口**：逐步扩展到 **128K**
- **模型规模**：8B / 70B / 405B

LLaMA 3.1（2024.07）的 405B 版本是第一个在多项基准上接近 GPT-4 水平的开源模型。

LLaMA 3.2 进一步引入了多模态（视觉）和小模型（1B / 3B）。

## 技术影响

LLaMA 对开源生态的影响是革命性的：

### 1. 定义了现代开源 LLM 的标准架构

```
RMSNorm + SwiGLU + RoPE + GQA + No Bias
```

这个组合已经成为行业标准，被 Qwen、DeepSeek、Mistral、Yi 等几乎所有主流模型采用。

### 2. 催生了微调生态

LLaMA 权重的开放直接催生了：
- **Alpaca**：Stanford 用 52K 指令数据微调 LLaMA 7B
- **Vicuna**：用 ShareGPT 数据微调
- **QLoRA**：4-bit 量化 + LoRA 微调，让 65B 模型可以在单卡上训练

### 3. 推动了推理优化

LLaMA 的广泛使用推动了 llama.cpp、vLLM、Ollama 等推理框架的快速发展。

### 4. 开源许可证的演进

LLaMA 1 的研究许可证限制了商用，LLaMA 2 转向更开放的社区许可证，LLaMA 3 进一步放宽限制。这个演进路径说明 Meta 逐渐认识到**开放生态的网络效应比封闭控制更有价值**。

## 一个工程师视角的思考

LLaMA 的成功给我最大的启发是：**架构创新远不如数据和工程重要**。LLaMA 并没有提出任何新的架构组件——RMSNorm、SwiGLU、RoPE 都是别人先提出的。它的贡献在于：

1. 把已知最好的组件拼在一起
2. 用精心策划的数据训练
3. 开源出来让所有人受益

这种"集成创新 + 开源"的路线，可能比追求单点突破更有影响力。

## 相关

- [[GPT]]
- [[BERT]]
- [[AI/LLM/SFT/LoRA|LoRA]]
- [[AI/LLM/Inference/vLLM|vLLM]]
- [[AI/LLM/Inference/Ollama|Ollama]]
- [[AI/LLM/Infra/分布式训练|分布式训练]]
- [[AI/LLM/Infra/FSDP|FSDP]]
