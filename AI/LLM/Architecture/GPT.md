---
title: "GPT"
type: paper
domain: ai/llm/architecture
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/architecture
  - type/paper
---
# GPT

GPT（Generative Pre-trained Transformer）是 OpenAI 提出的一系列基于 Transformer Decoder 的自回归语言模型，从 GPT-1 到 GPT-4 逐步定义了现代大语言模型的范式。理解 GPT 的演进路线，本质上就是理解整个 LLM 领域是怎么走到今天的。

## GPT-1：预训练 + 微调范式的开创

GPT-1（2018）的核心贡献是验证了一个简单但深远的想法：**在大规模无标注文本上做语言建模预训练，然后在下游任务上微调，效果可以超过专门设计的模型**。

架构上，GPT-1 使用了 12 层 Transformer Decoder，约 1.17 亿参数。训练数据是 BooksCorpus（约 7000 本书）。预训练目标是标准的 causal language modeling：

$$L_1(\mathcal{U}) = \sum_i \log P(u_i | u_{i-k}, \ldots, u_{i-1}; \Theta)$$

微调阶段则加上任务特定的分类头，同时保留语言模型损失作为辅助目标：

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})$$

这个设计现在看来稀松平常，但在 2018 年，它和 ELMo、ULMFiT 一起开创了 NLP 的预训练时代。

## GPT-2：规模即能力

GPT-2（2019）的核心信念是 **scaling**：不需要微调，足够大的语言模型应该能 zero-shot 完成各种任务。

主要变化：
- 参数量从 1.17 亿增加到 **15 亿**（最大版本）
- 训练数据从 BooksCorpus 升级到 WebText（约 40GB，800 万网页）
- 引入了 task conditioning 的概念：通过 prompt 来指定任务

架构微调：
- Layer Norm 移到每个 sub-block 的输入端（Pre-Norm），而非输出端
- 最终的 self-attention block 后增加了一个额外的 Layer Norm
- 词汇表扩大到 50257（BPE 编码）
- 上下文窗口从 512 扩展到 1024

GPT-2 证明了一个重要的 scaling 假设：**模型越大，zero-shot 能力越强**。这为后续的 GPT-3 和整个 scaling laws 的研究奠定了基础。

## GPT-3：Few-Shot Learning 与 In-Context Learning

GPT-3（2020）是真正让世界注意到大模型潜力的里程碑。1750 亿参数，训练数据约 570GB（混合了 Common Crawl、WebText2、Books、Wikipedia）。

GPT-3 最重要的贡献不是架构创新（架构基本和 GPT-2 一样），而是展示了 **In-Context Learning（ICL）** 的能力：

- **Zero-shot**：只给任务描述
- **One-shot**：给一个示例
- **Few-shot**：给几个示例

关键发现：
1. ICL 能力随模型规模增长而涌现
2. Few-shot 可以接近甚至超过微调的效果
3. 模型不需要梯度更新就能"学习"新任务

这直接催生了 prompt engineering 这个领域，也让人们重新思考"什么是学习"。

## GPT-3.5 与 InstructGPT：对齐的转折点

InstructGPT（2022）是 GPT 系列最重要的方法论转变。它引入了 **RLHF（Reinforcement Learning from Human Feedback）** 三阶段训练：

1. **SFT（Supervised Fine-Tuning）**：用人工标注的高质量指令-回答对微调
2. **Reward Model 训练**：用人类偏好数据训练奖励模型
3. **PPO 优化**：用 RL 让模型输出对齐人类偏好

核心洞察：**一个 1.3B 的 InstructGPT 在人类评估中优于 175B 的 GPT-3**。这说明对齐（alignment）比纯粹的规模更重要。

## GPT-4：多模态与能力天花板

GPT-4（2023）是目前公开信息最少的版本。已知的关键特性：
- 多模态输入（文本 + 图像）
- 更长的上下文窗口（8K / 32K / 128K）
- 在各种专业考试中达到人类顶尖水平
- 传闻使用了 Mixture of Experts 架构

## 技术遗产与影响

GPT 系列对整个领域的影响：

1. **Decoder-Only 架构成为主流**：相比 BERT 的 Encoder-Only 和 T5 的 Encoder-Decoder，GPT 证明了 Decoder-Only 在生成任务上的优势，这一架构被 LLaMA、Qwen、DeepSeek 等后续模型广泛采用
2. **Scaling Laws**：GPT 系列是 Kaplan scaling laws 的主要验证者
3. **RLHF 范式**：InstructGPT 开创的三阶段训练成为行业标准
4. **In-Context Learning**：改变了人们使用模型的方式
5. **API 经济**：GPT-3 的 API 模式开创了 LLM 的商业化路径

## 关键反思

GPT 的成功不在于某个单一的技术突破，而在于持续押注几个正确的方向：

- **简单架构 + 大规模数据 + 大量算力** 胜过精巧的架构设计
- **通用能力** 比任务特定能力更有价值
- **对齐** 是从"能力"到"产品"的关键桥梁

## 相关

- [[BERT]]
- [[LLaMA]]
- [[T5]]
- [[AI/LLM/Prompt-Engineering/Prompt engineering 概述|Prompt engineering 概述]]
- [[AI/LLM/Infra/分布式训练|分布式训练]]
