---
title: DeepSeek-R1：纯 RL 驱动推理能力涌现的里程碑
brief: DeepSeek-R1 基于 DeepSeek-V3（671B MoE，37B 激活）底座，通过四阶段训练（Cold Start → GRPO RL → Rejection Sampling SFT → 二次 RL）实现推理能力涌现。核心洞察：纯 RL + 极简 outcome-based reward 即可激发自我反思和长链推理，且蒸馏效率惊人——1.5B 蒸馏模型在 MATH-500 打到 83.9%。
type: paper
domain: ai/llm/architecture
created: 2026-02-13
updated: 2026-02-22
tags:
  - ai/llm/architecture
  - type/paper
  - ai/llm/rl
status: complete
sources:
  - "DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* arXiv:2501.12948"
  - DeepSeek-AI. *DeepSeek-V3 Technical Report* arXiv:2412.19437
  - "Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* arXiv:2402.03300"
related:
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]"
  - "[[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]]"
  - "[[AI/3-LLM/Architecture/GPT|GPT]]"
  - "[[MoE 基础|MoE 基础]]"
---
# Deepseek-R1

## 概述

DeepSeek-R1（2025 年 1 月）是 DeepSeek 发布的推理模型，核心卖点是**纯 RL 驱动的推理能力涌现**。论文标题 *"Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"* 已经说明了一切：不靠人工标注的 Chain-of-Thought 数据，而是让模型通过强化学习自己"学会"推理。

> 来源：DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* arXiv:2501.12948

这篇论文对行业的冲击不在于模型架构本身（底座就是 DeepSeek-V3），而在于训练方法论：它证明了 RL 可以从一个普通的 base model 中激发出 reasoning 能力，而且涌现出的行为模式（自我验证、反思、长链推理）是 SFT 很难教会的。

## 架构基础

DeepSeek-R1 的底座架构与 DeepSeek-V3 完全一致：

- **MoE (Mixture of Experts)**：671B 总参数，每个 token 激活约 37B
- **Multi-Head Latent Attention (MLA)**：将 KV Cache 压缩到潜空间，显著降低推理时显存占用
- **DeepSeekMoE**：细粒度专家 + 共享专家的混合架构，比传统 MoE 路由更高效
- **辅助损失无关的负载均衡**：用 bias 项而非辅助 loss 来平衡专家负载，避免对主损失的干扰

关键数据：
- 128K context length
- 预训练数据：14.8T tokens
- FP8 混合精度训练

## 训练流程

这是 R1 最核心的贡献。整个训练分四个阶段：

### Stage 1: Cold Start（冷启动）

收集少量高质量的长 CoT 数据（几千条），对 DeepSeek-V3 做 SFT。目的不是教会模型推理，而是给它一个好的初始化点，让后续 RL 更稳定。

这一步解决了 R1-Zero 的几个问题：
- 语言混杂（中英文混着来）
- 格式混乱（没有清晰的思考/回答分界）
- 输出可读性差

### Stage 2: 推理导向 RL（Reasoning RL）

这是核心阶段。使用 **GRPO（Group Relative Policy Optimization）** 对模型做 RL 训练。GRPO 的目标函数：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q)} \mathbb{E}_{\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i, \text{clip}(\cdot, 1\pm\epsilon) \hat{A}_i \right) \right]$$

其中 $\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ 是组内相对优势（无需 critic model）。

> 来源：GRPO 最初由 DeepSeekMath（arXiv:2402.03300）提出，R1 论文 Sec. 3.1 详述

**Reward 设计**非常克制：
- 数学题：答案正确性（rule-based 验证）
- 代码题：测试用例通过率
- **没有 process reward，没有人工偏好标注**

这个阶段只在数学和代码任务上训练，但发现推理能力能**泛化**到其他领域。

### Stage 3: Rejection Sampling + SFT

从 RL checkpoint 采样大量 response，用规则和 DeepSeek-V3 作为 judge 筛选高质量数据，再做一轮 SFT。这一步扩展到通用任务（写作、问答、翻译等），同时保留推理能力。

### Stage 4: 二次 RL

在 SFT 模型上再做一轮 RL，同时优化：
- 推理任务的正确性
- 通用任务的 helpfulness 和 safety
- 格式规范性

## DeepSeek-R1-Zero：纯 RL 的极限实验

R1-Zero 是论文中最令人兴奋的实验——直接在 base model 上做 RL，不用任何 SFT 数据。

观察到的涌现行为：
1. **"Aha moment"**：训练过程中模型突然学会了自我反思（"Wait, let me reconsider..."）
2. **自我验证**：主动检查自己的推理步骤
3. **思考时间自适应**：难题自动生成更长的推理链
4. **探索与利用**：尝试多种解题路径后选择最优

但 R1-Zero 也暴露了纯 RL 的局限：
- 输出格式不稳定
- 语言混杂严重
- 可读性差
- 在 AIME 2024 上 pass@1 = 71%，低于最终 R1 的 79.8%

## 蒸馏（Distillation）

另一个重要贡献：将 R1 的推理能力蒸馏到小模型。

| 模型 | 基座 | AIME 2024 | MATH-500 |
|------|------|-----------|----------|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9% | 83.9% |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5% | 92.8% |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7% | 93.9% |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 72.6% | 94.3% |
| R1-Distill-Llama-70B | Llama-3.3-70B | 70.0% | 94.5% |

一个关键发现：**蒸馏 > 小模型自己做 RL**。在相同基座上，直接用 R1 的 CoT 数据做 SFT，效果远超小模型自己跑 RL。这说明推理能力的涌现可能需要足够大的模型容量。

> 来源：arXiv:2501.12948, Sec. 5 (Distillation) — "distilling from DeepSeek-R1 is significantly better than RL on the small models"

## 我的观察

1. **RL 的 reward 设计极简**：没用 process reward model，没用人工偏好，就是最简单的 outcome-based reward。这反而避免了 reward hacking。
2. **GRPO 代替 PPO** 的选择很务实：省掉 critic model 意味着少一半显存，对 671B 的模型来说是刚需。
3. **Cold start 的必要性**：R1-Zero 虽然能涌现推理，但格式问题严重影响实用性。这说明 RL 能教会"怎么想"，但"怎么表达"还是需要 SFT。
4. **蒸馏的效率惊人**：1.5B 的蒸馏模型在 MATH-500 上打到 83.9%，这对边缘部署意义重大。
5. **开源策略**：模型权重全部开源（MIT license），但训练细节有意留白，这让 OpenR1 等复现项目应运而生。

## 📚 推荐阅读

### 原始论文
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) — R1 论文原文，必读的四阶段训练流程
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — GRPO 算法首次提出
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — R1 的底座架构细节（MLA、DeepSeekMoE）

### 深度解读
- [OpenR1 Project](https://github.com/huggingface/open-r1) — HuggingFace 主导的社区复现，跟踪复现进展 ⭐⭐⭐⭐
- [DeepSeek-R1 解读合集 (知乎)](https://www.zhihu.com/topic/30114558) — 中文社区深度讨论

### 实践资源
- [DeepSeek-R1 模型权重](https://huggingface.co/deepseek-ai/DeepSeek-R1) — MIT License 全开源
- [verl 框架](https://github.com/volcengine/verl) — 用于复现 GRPO 训练的 RL 框架

## 🔧 落地应用

### 直接可用场景
- **数学/代码推理**：R1 在 AIME 2024 上 pass@1 = 79.8%，直接可用于复杂数学问题求解
- **蒸馏到小模型**：R1-Distill-Qwen-7B 在 MATH-500 达到 92.8%，适合边缘部署的推理场景
- **训练范式参考**：四阶段训练流程（Cold Start → RL → RS-SFT → 二次 RL）可直接用于自研模型的推理能力训练

### 工程实现要点
- **GRPO 显存优势**：省掉 critic model，对 671B 的模型意味着省下约 $37B \times 2$ bytes = 74GB 显存
- **Reward 设计极简**：数学用 rule-based 验证，代码用测试用例——刻意不用 process reward model，避免 reward hacking
- **Cold Start 数据量**：只需几千条高质量长 CoT 数据即可，但质量 >> 数量

### 面试高频问法
- Q: R1-Zero 和 R1 的核心区别？为什么需要 Cold Start？
  A: R1-Zero 直接在 base model 上做 RL，可以涌现推理但格式混乱、语言混杂。Cold Start 用少量 SFT 数据解决"怎么表达"的问题，让后续 RL 聚焦于"怎么想"。AIME 上 R1-Zero 71% vs R1 79.8%。

## 💡 启发与思考

### So What？对老板意味着什么
- **RL 不仅是对齐工具，更是能力激发工具**。传统认知中 RL 用于 alignment（InstructGPT），R1 证明 RL 可以直接激发 base model 中潜在的推理能力——这改变了"能力来自哪里"的认知
- **蒸馏路线的实用价值巨大**：不需要所有人都训 671B 模型，用 R1 的 CoT 数据蒸馏小模型，7B 就能达到 92.8% MATH-500，这才是大多数团队能落地的路径

### 未解问题与局限
- "Aha moment" 的涌现条件是什么？必须是 671B 这个量级才行，还是 70B 也能出现？目前没有严格的 scaling 实验
- 纯 RL 路线能否扩展到非 verifiable 任务（如创意写作、开放式对话）？目前 R1 的成功严重依赖 rule-based reward
- 开源了权重但训练细节有意留白：RL 训练的具体数据配比、reward shaping 细节不完全透明

### 脑暴：如果往下延伸
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO]] + [[AI/3-LLM/Architecture/Mamba-SSM|Mamba]] 的组合：SSM 的线性推理效率 + RL 的推理能力激发，能否做出推理效率极高的小模型？
- R1 的蒸馏成功说明 CoT 是可迁移的"知识"。那 Agent 的 ReAct 轨迹能否类似地蒸馏？这对  领域有重大意义

## 相关

- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — R1 使用的核心 RL 算法
- [[AI/3-LLM/RL/GRPO/DeepSeek R1 学习笔记|DeepSeek R1 学习笔记]] — 更详细的学习记录
- [[AI/3-LLM/RL/GRPO/DeepSeek-Math|DeepSeek-Math]] — GRPO 最初提出的论文
- [[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]] — GRPO 的前身对比
- [[AI/3-LLM/RL/Fundamentals/RL 概览|RL 概览]] — 强化学习基础
- [[AI/3-LLM/Infra/分布式训练|分布式训练]] — R1 训练所需的基础设施
- [[AI/3-LLM/RL/GRPO/OpenR1|OpenR1]] — 社区复现项目
- [[AI/3-LLM/Architecture/LLaMA|LLaMA]]
- [[AI/3-LLM/Architecture/GPT|GPT]] — RLHF 路线 vs 纯 RL 路线的对比
- [[MoE 基础|MoE 基础]] — R1 底座的 MoE 架构
- [[AI/3-LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/3-LLM/Frameworks/TRL/TRL 概述|TRL 概述]]

**代码手撕（理论 → 代码）：**
- [[AI/3-LLM/Architecture/DeepSeek-V3-手撕实操|DeepSeek-V3-手撕实操]] ⭐⭐⭐⭐⭐ — MLA（KV cache 降 16x）+ MoE + mHC 架构从零实现，面试高频考点
- [[AI/3-LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]] ⭐⭐⭐⭐⭐ — R1 核心训练算法从零实现
- [[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解|MA-RLHF 核心代码注解]] — PPO/GRPO 在 LLM RLHF 中的完整训练框架
