---
brief: "RARL Reward Modeling Survey——系统综述 RL-based LLM 推理中的奖励模型设计，涵盖 outcome RM/process RM/LLM-as-judge/rule-based 四类范式及其优劣；是奖励工程选型的快速参考。"
title: "RARL: Reward Modeling for RL-Based LLM Reasoning"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - type/paper
  - type/survey
created: 2026-02-16
---

# RARL: Reward Modeling for RL-Based LLM Reasoning

> 论文：[arXiv:2602.09305](https://arxiv.org/abs/2602.09305) (2026-02-10)
> 类型：综述
> 关键词：reward modeling, RLHF, RLVR, reasoning alignment, reward hacking

## 核心论点

**Reward modeling 不是 implementation detail，而是 reasoning alignment 的 central architect。** 它决定了：
- 模型学到什么
- 如何 generalize
- 输出是否可信

## RARL Framework

Reasoning-Aligned Reinforcement Learning — 统一框架，系统化了多步推理的 reward 范式。

### Reward 机制分类

```
Reward Paradigms
├── Outcome-Based Rewards (ORM)
│   ├── Binary correctness (math/code)
│   ├── Partial credit scoring
│   └── Verifiable rewards (RLVR)
├── Process-Based Rewards (PRM)
│   ├── Step-level verification
│   ├── Human-annotated process rewards
│   └── Automated PRM (Monte Carlo estimation)
└── Hybrid Approaches
    ├── ORM + PRM combination
    └── Self-verification signals
```

### Reward Hacking — 核心失败模式

模型学会 exploit reward function 而非真正提升能力：
- **Shortcut exploitation** — 找到 reward function 的漏洞
- **Distribution shift** — 模型生成 out-of-distribution 的输出骗过 reward model
- **Reward model overoptimization** — 过度优化导致 reward model 失效

## Reward Design 与核心挑战的关系

| 挑战 | Reward 如何影响 |
|------|----------------|
| **Evaluation bias** | Reward model 的 bias 直接传导到 policy |
| **Hallucination** | Reward 不惩罚幻觉 → 模型学会编造 |
| **Distribution shift** | On-policy vs off-policy reward estimation |
| **Inference-time scaling** | Reward signal 决定 TTC 的 search 方向 |

## 关键 Insight

### 1. RLVR vs RLHF 的本质区别

| | RLHF | RLVR |
|--|------|------|
| Reward 来源 | Human preference | Programmatic verification |
| Noise | 高（annotator disagreement） | 低（deterministic） |
| 适用域 | Open-ended（chat, creative） | Structured（math, code, logic） |
| Scalability | 受限于标注成本 | 可大规模 on-policy |
| ICLR 2026 论文数 | 54 | 125 |

### 2. Benchmark 的脆弱性

- **Data contamination** — benchmark 数据泄露到训练集
- **Reward misalignment** — benchmark metric ≠ 真实能力
- 需要 custom eval pipeline 而非依赖公开 benchmark

### 3. Reward 统一视角

论文认为 inference-time scaling 和 hallucination mitigation 本质上都是 reward design 问题：
- TTC 的 self-verification 本质是 **implicit process reward**
- Hallucination 的解决需要 **factuality reward signal**

## 面试要点

1. **ORM vs PRM 的 trade-off**
   - ORM: 简单、scalable，但 credit assignment 粗
   - PRM: 精细，但标注成本高或自动化有 noise

2. **Reward hacking 怎么防**
   - KL penalty (PPO/GRPO 的标配)
   - Reward model ensemble
   - Reward model 定期更新
   - Constrained optimization

3. **为什么 RLVR 在 reasoning 上赢了 RLHF**
   - Clean reward signal
   - On-policy scalable
   - 不需要 human in the loop

## 关联笔记

- [[AI/3-LLM/RL/RLHF-工程全栈|RLHF 全链路]]
- [[AI/3-LLM/RL/对齐技术综述|对齐技术综述]]
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/3-LLM/Evaluation/ICLR-2026-趋势分析|ICLR-2026-趋势分析]]

---
*Created: 2026-02-16 by Scholar heartbeat*
