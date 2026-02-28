---
brief: "Blockwise Advantage Estimation——将 GRPO 的优势估计从 outcome 级延伸到 block（段落）级，实现多目标 RL 的细粒度信用分配；解决长链推理中奖励稀疏导致的更新信号弱问题。"
title: "Blockwise Advantage Estimation for Multi-Objective RL"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm/rl/grpo
  - type/paper
created: 2026-02-16
---

# Blockwise Advantage Estimation for Multi-Objective RL

> 论文：[arXiv:2602.10231](https://arxiv.org/abs/2602.10231) (2026-02-10)
> 关键词：GRPO, credit assignment, multi-objective RL, verifiable rewards

## 问题

标准 GRPO 给整个 completion 一个 scalar advantage → 所有 token 共享同一个 advantage 值。

对于**有结构化 segments 的生成**（例如 reasoning chain + final answer + confidence estimation），这带来：
- **Objective interference** — 不同 segment 的 reward signal 耦合
- **Credit misattribution** — 模型不知道 advantage 该归因到哪个 segment

## 核心方案

### Blockwise Advantage Estimation

把 completion 按语义切成 blocks，每个 block 对应一个 objective：

```
Block 1: [reasoning tokens]  ← reward_correctness
Block 2: [answer tokens]     ← reward_format  
Block 3: [confidence tokens]  ← reward_calibration
```

每个 objective 有自己的 advantage，**只作用于对应 block 的 tokens**。

### 关键难点：Later Blocks 的 Advantage 估计

后面 block 的 reward 依赖前面 block 的采样结果（prefix-conditioned），标准无偏方法需要从中间状态做 nested rollouts，计算开销很大。

### 解法：Outcome-Conditioned Baseline

用 **within-group statistics** 近似中间状态 value：
- 按 prefix 产生的 intermediate outcome 对 samples 分层（stratify）
- 在每个 stratum 内计算 baseline
- 不需要额外 rollout，完全复用 GRPO 已有的采样

## 技术评价

### 优点
1. **GRPO-compatible** — 不改变 GRPO 框架，是扩展而非替代
2. **No value model** — 不引入额外模型，用 group statistics 搞定
3. **Modular** — 自然 scale 到更多 objectives，不需要手工设计 scalar reward 权重
4. **保留 test-time gains** — confidence-weighted ensembling 仍然 work

### 局限
1. 需要 completion 有明确的 block 结构（不是所有任务都有）
2. Outcome-Conditioned Baseline 是近似，非无偏
3. 实验主要在 math + uncertainty estimation，generalization 待验证

## 与已有工作的关系

| 方法 | Credit Assignment 粒度 | 额外模型 |
|------|----------------------|----------|
| PPO | token-level (value model) | ✅ critic |
| GRPO | completion-level | ❌ |
| **Blockwise** | **block-level** | **❌** |
| Token-level GRPO variants | token-level | 取决于实现 |

**定位：** 在 GRPO 的 completion-level 和 PPO 的 token-level 之间找到了一个 sweet spot。

## 面试要点

- GRPO 的核心局限之一就是 **coarse-grained credit assignment**
- Blockwise 方案展示了不用 value model 也能做 finer-grained advantage 的路线
- 关键 insight：利用 **group statistics + stratification** 替代 value function

## 关联笔记

- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]] — Blockwise 是 GRPO completion-level advantage 的 block-level 升级
- [[Projects/DeepSeek-R1-学习笔记|DeepSeek R1 学习笔记]]
- [[AI/3-LLM/RL/实践/对齐技术综述|对齐技术综述]]

> **see-also（Agent 场景）**：Block-level credit assignment 与 Agent 场景的 step-level credit assignment 解决同一根问题（粗粒度 advantage → 细粒度），路线不同：
> - [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]] — Anchor State Grouping（无额外 rollout 实现 step-level CA，与 Blockwise 的 group statistics 思路同源）
> - [[AI/2-Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — step-level CA 全景，Blockwise 可作为 block-level 中间层

---
*Created: 2026-02-16 by Scholar heartbeat*
