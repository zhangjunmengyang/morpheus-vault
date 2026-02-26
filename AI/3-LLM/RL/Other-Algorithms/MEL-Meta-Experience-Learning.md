---
brief: "MEL（arXiv:2602.10224，Meta-Experience Learning for RLVR）——元学习框架改善 RLVR 的 credit assignment；将历史经验元信息注入当前训练，解决单次 rollout 奖励信号稀疏导致的低样本效率问题。"
title: "MEL: Meta-Experience Learning for RLVR"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - type/paper
  - rl/rlvr
  - rl/credit-assignment
created: 2026-02-16
arxiv: "2602.10224"
---

# MEL: Meta-Experience Learning for RLVR

> **Paper:** Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models
> **ArXiv:** 2602.10224 (2026-02-10)
> **Authors:** Shiting Huang et al. (USTC, Feng Zhao 组)
> **一句话:** 在 RLVR 基础上加一个「错误归因 → 经验抽象 → 参数内化」的闭环，把 contrastive pair 中的 bifurcation point 提炼为可复用的 meta-experience，通过 NLL 写入模型参数。

---

## 核心问题

RLVR (如 GRPO) 的 credit assignment 粒度太粗：
- 只有 outcome-level binary reward (对/错)
- 不知道 **哪一步** 错了，**为什么** 错了
- 无法形成可复用的知识结构

PRM 路线虽然提供 dense signal，但依赖 trained proxy → reward hacking 风险，与 RLVR 的 verifiable reward 理念冲突。

## MEL 框架

### 1. Explorative Rollout (标准 GRPO)
- Query x → 生成 G 条 trajectory → verifier 分为 Y⁺ (正确) 和 Y⁻ (错误)
- 只处理同时有正负样本的 query（gradient-informative samples）

### 2. Meta-Experience Construction（核心创新）

**Step 1: 定位 Bifurcation Point s***
- 构建 contrastive pair (y⁺, y⁻)
- 用 policy model 自身做 discriminative task：找到推理链分叉的具体步骤
- 关键 insight: 判别分叉点比从头推理更简单

**Step 2: Deep Diagnosis → Critique C**
- 以 s* 为锚点，对比正负 trajectory 在该步的策略差异
- 输出: 错误归因 + 策略差距 + 修正原则

**Step 3: Abstraction → Heuristic H**
- 将 instance-specific critique 抽象为 generalizable heuristic
- 去除上下文依赖变量，映射到通用的 precondition-response 空间
- 包含: 问题类型分类 + 推理原则 + 易错边界

**Meta-Experience Tuple:** M = (s*, C, H)

**Step 4: Empirical Validation via Replay**
- 将 M 作为 in-context hint 重新解题
- 只保留能通过 verifier 的 meta-experience → D_M*
- 过滤掉 hallucination 和 causal misalignment

### 3. Internalization Mechanism

核心思路：**把 meta-experience 从 context window 转移到 parametric memory**

- NLL loss 对 verified meta-experience 做 fine-tuning
- 条件: retrospective context C_retro = [I, x, y⁺, y⁻]
- 训练目标: 生成 M* 的 token-averaged NLL

### 4. Joint Training Objective

L_total = L_GRPO + λ · L_NLL

- L_GRPO: 标准 policy optimization（trajectory-level）
- L_NLL: meta-experience internalization（knowledge-level）
- 双重优化：行为层 + 知识层

---

## 实验结果

| Model | Benchmark | Base → GRPO → MEL |
|-------|-----------|-------------------|
| 4B | MATH-500 | - → baseline → +3.92% Pass@1 |
| 8B | MATH-500 | - → baseline → +4.73% Pass@1 |
| 14B | 5 benchmarks | consistent improvement |

- 在 Pass@1、Avg@8、Pass@8 上一致提升
- 兼容 RFT、GRPO、REINFORCE++
- 随模型规模扩大效果更显著（scalability）

---

## 技术分析

### 优点
1. **不需要 PRM** — 完全基于 outcome reward + self-verification，保持 RLVR 范式纯净
2. **Fine-grained credit assignment** — 从 trajectory-level 提升到 step-level + knowledge-level
3. **Inference-time free** — meta-experience 写入参数后推理时无额外开销（vs RAG/retrieval 方案）
4. **Self-distillation** — 不依赖外部强模型，用自身能力做 contrastive analysis
5. **Validation loop** — replay 验证过滤低质量 meta-experience

### 局限 & 疑问
1. **计算开销** — 每个 training step 需要额外的 contrastive analysis + replay，compute cost 至少 2-3x
2. **Self-verification 可靠性** — 模型自身做错误归因，accuracy 多高？paper 没有报 bifurcation point 定位准确率
3. **Meta-experience 质量天花板** — 受限于 policy model 的 critique 能力，小模型可能产生低质量 M
4. **Scalability to non-math** — 所有实验都在数学推理上，coding/logic 等 domain 未验证
5. **NLL vs RL signal 的冲突** — 同时做 NLL fine-tuning 和 RL 可能有 objective 冲突

### 与相关工作的对比
- **vs StepHint:** MEL 内化到参数，StepHint 作为 off-policy hint → MEL 避免 distributional mismatch
- **vs Scaf-GRPO:** MEL 不需要 inference-time prefix → 无 train-test gap
- **vs PRM:** MEL 不需要额外 reward model → 无 reward hacking
- **vs Blockwise Advantage (我们之前看的):** Blockwise 改进 GRPO advantage estimation 粒度，MEL 改进 knowledge internalization — 正交且可组合

---

## 定位判断

**创新性:** ⭐⭐⭐⭐ — meta-experience = (s*, C, H) 三元组设计 elegant，replay validation 是个好 trick
**实用性:** ⭐⭐⭐ — 计算开销大，但无 inference overhead 是实际部署的卖点
**影响力:** ⭐⭐⭐ — RLVR credit assignment 是 hot topic，但 3-5% 的 gain 不够 breakthrough

**与老板方向的关联:**
- 直接相关: RLVR/GRPO 生态的重要扩展
- 面试价值: "如何在 RLVR 框架内做 fine-grained credit assignment 而不依赖 PRM" — 这是一个好问题，MEL 提供了一个 elegant answer

---

## Tags
#RL #RLVR #GRPO #credit-assignment #meta-learning #self-distillation #reasoning #2026

---

## See Also

- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — MEL 在 Sample Efficiency 维度的位置
- [[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 同在 Other-Algorithms，均为 policy optimization 改进
- [[AI/3-LLM/RL/目录|RL MOC]] — LLM 强化学习全图谱
- [[强化学习的数学原理|强化学习数学原理]] — MEL 的理论基础
