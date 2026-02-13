---
title: "Deepseek-Math"
type: paper
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/paper
---
# Deepseek-Math

## 概述

DeepSeek-Math（2024 年 2 月）是 DeepSeek 发布的数学推理专用模型，论文 *"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"*。它有两个核心贡献：

1. **高质量数学预训练数据的挖掘方法**
2. **GRPO（Group Relative Policy Optimization）**——一种不需要 critic model 的 RL 算法

GRPO 后来被 DeepSeek-R1 采用，成为推理模型训练的关键算法。从这个角度看，DeepSeek-Math 是 R1 的技术先驱。

## 数据工程

DeepSeek-Math 的数据工作量可能比算法创新更大。他们从 Common Crawl 中挖掘了 **120B tokens** 的高质量数学数据（DeepSeekMath Corpus）。

### 数据挖掘流程

```
Common Crawl (大量网页)
    ↓ 用 seed 数学网页（OpenWebMath 等）训练 fastText 分类器
    ↓ 第一轮筛选
    ↓ 人工标注 + 迭代训练分类器
    ↓ 多轮迭代精化
    ↓
DeepSeekMath Corpus (120B tokens)
```

关键技巧：
- **迭代式挖掘**：每轮用上一轮的高质量数据来改进分类器，类似 self-training
- **URL 级别聚合**：同一域名下的页面质量往往一致，利用这个先验提升召回率
- **多语言覆盖**：不只是英文，也包含中文数学内容

### 与 OpenWebMath 对比

OpenWebMath 用的是更精细的 HTML 解析 + LaTeX 提取，质量更高但规模只有 14.7B tokens。DeepSeek-Math 证明了**规模 > 精度**的数据哲学——120B 的"还行"数据好于 14.7B 的"精品"数据。

## 模型训练

### 基座模型

基于 DeepSeek-Coder-v1.5-7B，继续用数学数据做 **continued pre-training**（500B tokens，其中数学数据 120B，通用数据做混合防止遗忘）。

### 训练流程

```
DeepSeek-Coder-v1.5-7B
    ↓ Continued Pre-training (数学 + 通用)
    → DeepSeek-Math-Base 7B
    ↓ SFT (数学 CoT 数据)
    → DeepSeek-Math-Instruct 7B
    ↓ RL (GRPO)
    → DeepSeek-Math-RL 7B
```

## GRPO 算法

这是论文最重要的算法贡献。GRPO 的核心动机：**PPO 的 critic model 太贵了**。

### PPO 的问题

标准 RLHF 用 PPO 需要 4 个模型：
1. Policy model（要训练的模型）
2. Reference model（KL 约束）
3. Reward model（打分）
4. **Value/Critic model**（估计 baseline）

其中 critic model 和 policy model 同等规模，训练时显存翻倍。

### GRPO 的解法

GRPO 去掉 critic model，用**组内相对排名**替代 value baseline：

对于 prompt $q$，采样一组 response $\{o_1, o_2, ..., o_G\}$，计算各自的 reward $\{r_1, r_2, ..., r_G\}$。

**Advantage 计算**：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,...,r_G\})}{\text{std}(\{r_1,...,r_G\})}$$

**GRPO 目标函数**：
$$\mathcal{L}_{GRPO} = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} \hat{A}_i, \; \text{clip}(\cdot) \hat{A}_i\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

其中 clip 操作与 PPO 相同（防止策略更新过大），$\beta$ 控制与 reference model 的 KL 散度惩罚。

### token 级别 KL

GRPO 的 KL 散度是在 **token 级别**计算的，不是 sequence 级别：

$$D_{KL} = \sum_{t=1}^{T} \left( \frac{\pi_{\text{ref}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} - \log \frac{\pi_{\text{ref}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} - 1 \right)$$

注意这里用的是**反向 KL**（$\pi_{\text{ref}} / \pi_\theta$），而不是标准的 $\pi_\theta / \pi_{\text{ref}}$。这是一个工程上的选择，具体原因论文没有详细解释。

### 与 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic model | 需要 | 不需要 |
| Baseline | Value function | 组内均值 |
| 显存开销 | 4 个模型 | 3 个模型 |
| 采样策略 | 每个 prompt 1 个 response | 每个 prompt G 个 response |
| Advantage 估计 | GAE (Generalized Advantage Estimation) | 组内 z-score |

### 实现伪代码

```python
for batch in training_data:
    prompts = batch["prompts"]
    
    # 1. 采样：每个 prompt 生成 G 个 response
    responses = []
    for q in prompts:
        group = [policy.generate(q) for _ in range(G)]
        responses.append(group)
    
    # 2. 打分
    rewards = [[reward_fn(q, o) for o in group] for q, group in zip(prompts, responses)]
    
    # 3. 计算 advantage（组内标准化）
    advantages = []
    for group_rewards in rewards:
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        adv = [(r - mean_r) / (std_r + eps) for r in group_rewards]
        advantages.append(adv)
    
    # 4. PPO-style clipped 更新
    loss = clipped_surrogate_loss(policy, old_policy, advantages)
    loss += beta * kl_divergence(policy, ref_policy)
    loss.backward()
    optimizer.step()
```

## 实验结果

在 7B 规模上，DeepSeek-Math-RL 达到了当时的开源 SOTA：

- **MATH**: 51.7%（对比 Minerva 540B 的 33.6%）
- **GSM8K**: 88.2%
- **竞赛级数学（AIME等）**：表现也不错，但论文主要 focus 在 MATH/GSM8K

一个重要消融：**GRPO > PPO > RFT (Rejection Sampling Fine-tuning) > SFT-only**，且 GRPO 训练更稳定、收敛更快。

## 对后续工作的影响

1. **DeepSeek-R1**：直接使用 GRPO 作为核心 RL 算法，证明了在 671B 规模上的可行性
2. **DAPO**：在 GRPO 基础上的改进（去掉 KL 惩罚等）
3. **TRL 框架**：HuggingFace 的 TRL 库集成了 GRPO trainer
4. **开源社区**：GRPO 成为中小团队做 RLHF 的首选（因为省 critic）

## 相关

- [[GRPO 深度理解]] — GRPO 算法的深入分析
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — GRPO 的大规模应用
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] — GRPO 的对比基线
- [[AI/LLM/RL/Fundamentals/策略梯度方法|策略梯度方法]] — RL 基础
- [[AI/LLM/RL/Fundamentals/KL散度|KL散度]] — GRPO 中的 KL 正则化
- [[AI/LLM/RL/Fundamentals/为什么 PPO 优于 PG|为什么 PPO 优于 PG]] — 从 PG 到 PPO 到 GRPO 的演进
- [[GRPO-TRL实践]] — TRL 中的 GRPO 实现
- [[GRPO-verl实践]] — verl 中的 GRPO 实现
