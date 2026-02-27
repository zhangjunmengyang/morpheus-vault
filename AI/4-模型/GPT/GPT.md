---
title: GPT 系列：从预训练到 AGI 的演进之路
brief: GPT 系列是 OpenAI 基于 Decoder-Only Transformer 的自回归语言模型，从 GPT-1（117M）到 GPT-4（~1.7T MoE）定义了现代 LLM 的核心范式。关键里程碑：GPT-2 验证了 scaling 假设，GPT-3 展示了 In-Context Learning 涌现，InstructGPT 开创 RLHF 三阶段对齐。理解 GPT 演进就是理解整个 LLM 领域的技术主线。
type: paper
domain: ai/llm/architecture
created: 2026-02-13
updated: 2026-02-22
tags:
  - ai/llm/architecture
  - type/paper
status: complete
sources:
  - Radford et al. *Improving Language Understanding by Generative Pre-Training* OpenAI 2018 (GPT-1)
  - Radford et al. *Language Models are Unsupervised Multitask Learners* OpenAI 2019 (GPT-2)
  - Brown et al. *Language Models are Few-Shot Learners* arXiv:2005.14165 (GPT-3)
  - Ouyang et al. *Training language models to follow instructions with human feedback* arXiv:2203.02155 (InstructGPT)
  - OpenAI. *GPT-4 Technical Report* arXiv:2303.08774
related:
  - "[[AI/3-LLM/Architecture/BERT|BERT]]"
  - "[[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]]"
  - "[[Transformer 通识|Transformer 通识]]"
  - "[[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]"
---
# GPT

GPT（Generative Pre-trained Transformer）是 OpenAI 提出的一系列基于 Transformer Decoder 的自回归语言模型，从 GPT-1 到 GPT-4 逐步定义了现代大语言模型的范式。理解 GPT 的演进路线，本质上就是理解整个 LLM 领域是怎么走到今天的。

## GPT-1：预训练 + 微调范式的开创

GPT-1（2018）的核心贡献是验证了一个简单但深远的想法：**在大规模无标注文本上做语言建模预训练，然后在下游任务上微调，效果可以超过专门设计的模型**。

> 来源：Radford et al. *Improving Language Understanding by Generative Pre-Training* OpenAI 2018

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

> 来源：Radford et al. *Language Models are Unsupervised Multitask Learners* OpenAI 2019

## GPT-3：Few-Shot Learning 与 In-Context Learning

GPT-3（2020, arXiv:2005.14165）是真正让世界注意到大模型潜力的里程碑。1750 亿参数，训练数据约 570GB（混合了 Common Crawl、WebText2、Books、Wikipedia）。

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

InstructGPT（2022, arXiv:2203.02155）是 GPT 系列最重要的方法论转变。它引入了 **RLHF（Reinforcement Learning from Human Feedback）** 三阶段训练：

1. **SFT（Supervised Fine-Tuning）**：用人工标注的高质量指令-回答对微调
2. **Reward Model 训练**：用人类偏好数据训练奖励模型
3. **PPO 优化**：用 RL 让模型输出对齐人类偏好

核心洞察：**一个 1.3B 的 InstructGPT 在人类评估中优于 175B 的 GPT-3**。这说明对齐（alignment）比纯粹的规模更重要。

## GPT-4：多模态与能力天花板

GPT-4（2023, arXiv:2303.08774）是目前公开信息最少的版本。已知的关键特性：
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

## 📚 推荐阅读

### 原始论文
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1 原文，预训练+微调的起点
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 论文，In-Context Learning 的里程碑
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT，RLHF 三阶段训练的奠基之作
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — GPT-4 技术报告，多模态+超强推理

### 深度解读
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar 可视化解读 ⭐⭐⭐⭐⭐
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. GPT 背后的 Scaling Laws 理论基础

### 实践资源
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 实现，学习架构的最佳入口
- [OpenAI API Documentation](https://platform.openai.com/docs/) — GPT-4 系列的 API 使用指南

## 🔧 落地应用

### 直接可用场景
- **通用对话 / ChatBot**：GPT 系列定义了 Chat 模型的标准，几乎所有后续模型（LLaMA-Chat、Qwen-Chat）都遵循 GPT 确立的 SFT → RLHF 训练范式
- **In-Context Learning / Few-shot**：GPT-3+ 无需微调，通过 prompt 中的示例即可完成新任务。核心公式：$P(y|x, \text{examples}) = \prod_t P(y_t | y_{<t}, x, \text{examples})$
- **代码生成（Codex/GPT-4）**：GPT-4 在 HumanEval 上 pass@1 达到 67%，是 GitHub Copilot 的底层引擎

### 工程实现要点
- **KV Cache**：Decoder-Only 推理的核心优化，缓存已计算的 Key/Value，将每步生成从 $O(n)$ 降到 $O(1)$ 注意力计算
- **温度与 Top-p**：$T \to 0$ 趋近贪心，$T \to \infty$ 趋近均匀分布；Top-p (nucleus sampling) 通常设 0.9-0.95
- **Pre-Norm vs Post-Norm**：GPT-2 起采用 Pre-Norm（LayerNorm 在残差之前），训练更稳定

### 面试高频问法
- Q: 为什么 1.3B 的 InstructGPT 能在人类评估中优于 175B 的 GPT-3？
  A: 因为"能力"和"对齐"是两个维度。GPT-3 有能力但不知道人类想要什么格式的输出；RLHF 让小模型学会了"以人类期望的方式回答"，这比原始能力更影响用户体验。

## 💡 启发与思考

### So What？对老板意味着什么
- **GPT 的成功路径 = 简单架构 + 大规模数据 + 对齐**。这个公式对任何想训练自己模型的团队都是指导原则——不要在架构上过度创新，把精力花在数据质量和对齐上
- **In-Context Learning 改变了"使用模型"的方式**：从"训练一个专用模型"变成"写一个好 prompt"，这是 AI 工程范式的根本转变

### 未解问题与局限
- GPT-4 的 MoE 架构细节至今未公开，社区复现（如 [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]）在逐步逼近但仍有差距
- In-Context Learning 为何只在足够大的自回归模型中涌现？理论解释仍不充分（有 meta-learning 假说但未定论）
- RLHF 的 reward hacking 问题：模型学会"讨好" reward model 而非真正对齐人类偏好

### 脑暴：如果往下延伸
- GPT 的 RLHF 三阶段 vs [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] 的纯 RL 路线：哪条路更有前途？R1 证明 RL 可以激发推理，但 GPT 的对齐更全面
- 如果把 [[AI/3-LLM/Architecture/Mamba-SSM|Mamba]] 的线性复杂度和 GPT 的自回归范式结合，能否突破 Transformer 的长度瓶颈？

## 相关

- [[AI/3-LLM/Architecture/BERT|BERT]] — Encoder-Only 路线对比，两条技术路线的分野
- [[LLaMA|LLaMA]] — 开源 GPT 路线的代表
- [[AI/3-LLM/Architecture/T5|T5]] — Encoder-Decoder 路线对比（T5 笔记待建）
- [[AI/3-LLM/Application/Prompt-Engineering-概述|Prompt engineering 概述]] — ICL 能力的工程化应用
- [[AI/3-LLM/Infra/分布式训练|分布式训练]] — GPT-3/4 训练所需的基础设施
- [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — 纯 RL 路线对 GPT RLHF 范式的挑战
- [[Transformer 通识|Transformer 通识]]
- [[AI/3-LLM/Architecture/Attention 变体综述|Attention 详解]]
- [[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]] — InstructGPT 使用的 RL 算法
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]] — RLHF 三阶段的第一步
