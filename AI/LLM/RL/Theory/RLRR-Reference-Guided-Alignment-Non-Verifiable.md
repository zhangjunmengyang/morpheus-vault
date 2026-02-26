---
brief: "RLRR（arXiv:2602.16802，ICLR 2026）——Reference-Guided RL for Non-Verifiable Domains；用高质量参考答案作为对齐信号替代 outcome verifier，解决创意写作/开放问答等无法自动评分任务的 RLHF 问题。"
title: "RLRR: Reference-Guided RL Alignment for Non-Verifiable Domains"
date: 2026-02-18
tags: [alignment, RLHF, LLM-as-judge, non-verifiable, reference-guided, self-improvement, DPO, ICLR2026]
domain: AI/LLM/RL/Theory
arxiv: "2602.16802"
rating: 4
status: permanent
see-also:
  - "[[AI/LLM/RL/GRPO/ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]]"
  - "[[AI/Safety/对齐技术总结|对齐技术总结]]"
  - "[[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]]"
---

# RLRR: References Improve LLM Alignment in Non-Verifiable Domains

> arXiv: 2602.16802 | ICLR 2026 Camera Ready
> 作者: Kejian Shi, Yixin Liu, Peifeng Wang, Alexander R. Fabbri, Shafiq Joty, Arman Cohan
> 机构: Yale University + Meta + Scale AI + Salesforce Research + NTU
> 代码: https://github.com/yale-nlp/RLRR

## 评分: ★★★★☆

## 一句话

RLVR 只能用在有 ground-truth verifier 的任务（数学/代码）。但对齐任务没有 verifier。这篇论文问：能不能用 reference output + LLM-as-judge 当软 verifier？答案是可以，而且性能接近有专门训练的 reward model（ArmoRM）。

---

## 问题：RLVR 的边界

```
RLVR 能做：数学（答案对错）、代码（运行通过）
RLVR 不能做：open-ended alignment（"哪个回答更好？"——没有 gold answer）
```

现有 non-verifiable 对齐方案：
- RLHF：人类标注 preference → 贵，慢
- RLAIF：LLM-as-judge → 准确率不够，positional bias，verbosity bias
- SFT distillation：强模型生成数据直接学 → 牺牲 generality

这篇论文的核心 claim：**有高质量 reference output 的情况下，LLM-judge 的准确率能大幅提升**——足以作为对齐训练的 soft verifier。

---

## 方法

### Part 1：Reference-Guided LLM Judge

三种评估策略（从弱到强）：

**RefFree（Ours baseline）**：无 reference，强化版 prompt，指令要求 judge 同时评估 instruction-following、factuality、verbosity。比现有 reference-free baseline 已经更好。

**RefEval（核心方法）**：
```
Prompt 要求 judge：
1. 检查哪个回答更接近 reference 的质量和内容
2. 验证 factual accuracy（与 reference 对比）
3. 检查是否有 reference 之外的多余内容
4. 参考 reference 理解"什么是好回答"再做比较
```
关键：**明确告知 judge 如何使用 reference**。之前的工作（HREF-Ref、LLMBar-Ref）只是把 reference 塞进 prompt，没有明确指导，效果提升有限。

**RefMatch**：更极端，让 judge 主要判断哪个回答在语义和风格上更接近 reference（相似度匹配）。在某些任务上略逊于 RefEval。

### 评估结果（11 个 open-source judge，5 个 benchmark）

| 方法 | Nat | Adv | MT | Ins | HREF | **Avg** |
|------|-----|-----|-----|-----|------|---------|
| LLMBar-Base | 83.1 | 61.7 | 74.6 | 70.2 | 72.0 | 72.3 |
| CoT | 82.0 | 60.1 | 75.4 | 69.1 | 69.6 | 71.2 |
| RefMatch | 84.6 | 74.1 | 76.3 | 72.9 | 80.4 | 77.7 |
| **RefEval** | **86.8** | **74.9** | **76.7** | **74.5** | **82.7** | **79.1** |

RefEval vs 最强 baseline：**+6.8% 平均准确率**。所有 baseline 与 RefEval 的差异均显著（p<0.05）。

关键发现：**naive 地把 reference 加进 prompt 效果一般**（LLMBar-Ref avg=74.0，与没有 reference 相当）。**明确引导 judge 如何使用 reference** 才是关键。

---

### Part 2：Reference-Guided Self-Improvement for Alignment

训练流程（两阶段）：

**阶段 1：SFT on reference outputs**
- 用 DeepSeek-V3 为 UltraFeedback dataset 生成高质量 reference
- 直接 SFT 蒸馏到 Llama-3-8B / Qwen2.5-7B

**阶段 2：DPO with reference-guided self-judge**
- 模型自己生成多个 responses
- 用自身作为 RefEval judge，判断哪个更接近 reference
- 用 judge 选出的 preference pairs 做 DPO

整个流程**不需要额外的人工标注或外部 AI 反馈**——只需要高质量 reference（这里用 DeepSeek-V3 生成）。

### 训练结果

| 方法 | AlpacaEval | Arena-Hard |
|------|-----------|-----------|
| SFT on reference | baseline | baseline |
| DPO + ArmoRM (专训 RM) | +19.2 | +16.5 |
| **DPO + RefEval self-judge** | **接近 ArmoRM** | **接近 ArmoRM** |
| DPO + reference-free self-judge | 弱于 RefEval | 弱于 RefEval |

Llama-3-8B：AlpacaEval **73.1%**，Arena-Hard **58.7%**（vs SFT baseline +20.2/+17.1）
Qwen2.5-7B：AlpacaEval **70.0%**，Arena-Hard **74.1%**

**关键发现**：reference-guided self-judge 的 DPO 性能接近使用专门 fine-tuned reward model（ArmoRM-Llama3-8B）的 DPO，**但不需要训练 reward model**。

---

## 我的分析

### 为什么这个方向重要？

这篇论文在填一个真实的工程缺口：

RLVR 的成功（DeepSeek-R1/o1）让所有人都想用 RL 做 alignment，但 alignment 任务没有 verifiable reward。这篇论文给了一条路：用高质量 reference + 精心设计的 judge prompt，能获得足够准确的软 verifier。

**实用价值**：如果你有强模型生成的 reference（现在 DeepSeek-V3/GPT-4o 都能做），就能做高质量 alignment fine-tuning，**不需要**：
1. 人工标注
2. 专门训练 reward model
3. 外部 API 调用（只需要 self-judging）

### 核心洞察的 critique

**成立的前提**：你得有高质量 reference。这个 "free lunch" 是假的——DeepSeek-V3 生成 reference 本身就是成本。

但话说回来：
- Reference 生成是一次性成本（offline）
- Reward model 训练需要人工标注（更贵）
- 所以 reference 路线总体更便宜

**更深的问题**：self-improvement 有上限。judge 用的是同一个模型，会强化 judge 自己的偏好。Reference 的作用是给 self-judge 一个外部锚点，减缓 self-amplification。但 reference 本身（DeepSeek-V3 生成的）也有 bias。这个 distribution 最终会收敛到 reference 生成模型的偏好上。

**与 RLVR 的本质区别保留**：即使用了 reference-guided judge，这仍然是软 verifier，不是硬 verifier。准确率 79.1% 意味着 ~21% 的时候 judge 会给错误的 preference signal。在数学 RLVR 里 verifier 准确率接近 100%。这个差距是 non-verifiable 领域的天花板。

### 与 Blockwise-Advantage 和 OpenRS 的联系

Vault 里有 OpenRS（pairwise adaptive rubric），也是在非验证域做 RL。两者路线不同：
- OpenRS：动态生成 rubric → LLM 按 rubric 打分
- RLRR（这篇）：用 reference output 引导 LLM judge → pairwise preference

RLRR 更系统（ICLR 2026，5 个 benchmark，11 个 judge），结论更可信。

---

## 连接

- 前驱：RLVR（GRPO/PPO 系），LLM-as-Judge（MT-Bench/Arena-Hard）
- 竞争：OpenRS（rubric-based），ArmoRM（专训 RM）
- 应用：alignment fine-tuning without human feedback

## see-also

- [[AI/LLM/RL/RLHF 全链路|RLHF 全链路]] — RLVR 的传统对齐路线（RLRR 是其 non-verifiable 扩展）
- [[AI/Safety/对齐技术总结|对齐技术总结]] — RLHF/DPO/Constitutional AI 对齐方法全景
- [[AI/Safety/AI安全与对齐-2026技术全景|AI 安全与对齐 2026 全景]] — Scalable Oversight 等与 RLRR 问题动机相关
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — RLVR 在 verifiable 域的改进全景（与 RLRR non-verifiable 互补）
- [[AI/LLM/RL/GRPO/ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]] — RLVR 信号质量改进；RLRR 解决的是 RLVR 能力边界（无 verifier 场景）

## Tags
`#RLHF` `#alignment` `#LLM-as-judge` `#non-verifiable` `#reference-guided` `#self-improvement` `#DPO` `#ICLR-2026` `#2026-02`
