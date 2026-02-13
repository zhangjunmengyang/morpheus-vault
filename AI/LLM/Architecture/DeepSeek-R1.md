---
title: "Deepseek-R1"
type: paper
domain: ai/llm/architecture
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/architecture
  - type/paper
---
# Deepseek-R1

## 概述

DeepSeek-R1（2025 年 1 月）是 DeepSeek 发布的推理模型，核心卖点是**纯 RL 驱动的推理能力涌现**。论文标题 *"Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"* 已经说明了一切：不靠人工标注的 Chain-of-Thought 数据，而是让模型通过强化学习自己"学会"推理。

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

这是核心阶段。使用 **GRPO（Group Relative Policy Optimization）** 对模型做 RL 训练：

```
# GRPO 的核心思路
对每个 prompt，采样一组 response（比如 64 个）
用 rule-based reward 打分
组内相对排序得到 advantage
不需要额外的 value model / critic
```

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

## 我的观察

1. **RL 的 reward 设计极简**：没用 process reward model，没用人工偏好，就是最简单的 outcome-based reward。这反而避免了 reward hacking。
2. **GRPO 代替 PPO** 的选择很务实：省掉 critic model 意味着少一半显存，对 671B 的模型来说是刚需。
3. **Cold start 的必要性**：R1-Zero 虽然能涌现推理，但格式问题严重影响实用性。这说明 RL 能教会"怎么想"，但"怎么表达"还是需要 SFT。
4. **蒸馏的效率惊人**：1.5B 的蒸馏模型在 MATH-500 上打到 83.9%，这对边缘部署意义重大。
5. **开源策略**：模型权重全部开源（MIT license），但训练细节有意留白，这让 OpenR1 等复现项目应运而生。

## 相关

- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — R1 使用的核心 RL 算法
- [[AI/LLM/RL/GRPO/DeepSeek R1 学习笔记|DeepSeek R1 学习笔记]] — 更详细的学习记录
- [[AI/LLM/RL/GRPO/DeepSeek-Math|DeepSeek-Math]] — GRPO 最初提出的论文
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] — GRPO 的前身对比
- [[AI/LLM/RL/Fundamentals/RL 概览|RL 概览]] — 强化学习基础
- [[AI/LLM/Infra/分布式训练|分布式训练]] — R1 训练所需的基础设施
- [[AI/LLM/RL/GRPO/OpenR1|OpenR1]] — 社区复现项目
- [[AI/LLM/Architecture/LLaMA|LLaMA]]
- [[AI/LLM/Architecture/GPT|GPT]]
- [[AI/Foundations/DL-Basics/MoE 基础|MoE 基础]]
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL 概述]]
