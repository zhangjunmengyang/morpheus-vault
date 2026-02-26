---
brief: "GEPA（ICLR 2026 Oral）——反思性 Prompt 进化通过自我评估-改写循环优化 prompt，在多项任务上超越 RL 微调；关键证据：post-training 的方法论分界线（Prompt 优化 vs 权重更新）；与 E-SPL/CTA 三向双向链接。"
title: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm
  - topic/rl
  - topic/prompt-optimization
  - topic/evolutionary-algorithm
  - type/paper
  - rating/5star
arxiv: "2507.19457"
venue: "ICLR 2026 Oral"
created: 2026-02-20
authors: "Lakshya A Agrawal et al."
affiliation: "UC Berkeley + Stanford + Databricks + MIT"
see-also:
  - "[[AI/LLM/RL/Other-Algorithms/E-SPL-Evolutionary-System-Prompt-Learning]]"
  - "[[AI/Agent/Agentic-RL/Calibrate-Then-Act-Cost-Aware-Exploration]]"
  - "[[AI/LLM/RL/Theory/RLVR-Edge-of-Competence]]"
---

# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**arXiv**: 2507.19457v2 | **Date**: 2025-07-25 (v2: 2026-02-14) | **Venue**: ICLR 2026 **Oral**  
**Code**: https://github.com/gepa-ai/gepa  
**Authors**: Lakshya A Agrawal et al.  
**Affiliation**: UC Berkeley + Stanford + Databricks + MIT (Omar Khattab)  
**Rating**: ★★★★★

---

## 一句话

**Reflective prompt evolution**（不更新权重，只演化 system prompt）在大多数任务上可以击败 GRPO，使用的 rollout 减少 35x。

---

## 核心主张与挑战

GRPO 等 RLVR 方法的问题：
- 需要数万次 rollout 才能收敛
- 从稀疏标量 reward 中提取 policy gradient 信息密度低
- 无法应用于不能 finetune 的模型（最强 closed-source API）
- Tool call 昂贵时 rollout cost 爆炸

GEPA 的核心论点：**LLM 的执行 trace 是天然的自然语言学习信号，信息密度远高于标量 reward。** 与其用标量 reward 估计梯度，不如让 LLM 直接反思轨迹、提炼规则、更新 prompt。

---

## 问题形式化

复合 AI 系统：`Φ = (M, C, X, Y)`，其中 `M = ⟨M_1,...,M_|M|⟩` 是语言模块，每个模块 `M_i = (π_i, θ_i, X_i, Y_i)`。

可学习参数：`⟨Π, Θ⟩_Φ`（prompts + weights）。

GEPA 只优化 `Π_Φ`（prompts），`Θ_Φ` 冻结——这是和 E-SPL 的本质区别。

优化目标带预算约束：
```
argmax_Π E[(x,m)~T] μ(Φ(x; Π), m)
s.t. #rollouts ≤ B
```

---

## 方法

### 3.1 核心循环：Reflective Prompt Mutation

每次迭代：
1. **选 candidate**（Pareto-based，见 3.2）
2. **选 module**（round-robin 轮流更新多模块系统中的某一个）
3. **在 minibatch 上执行**，收集 (trajectory, score, feedback_text)
4. **LLM 反思**：输入「当前 prompt + 执行轨迹 + reward + 文本反馈」→ 诊断错误 → 提出修改
5. **更新 prompt** → 在 minibatch 上验证，如改进则加入 candidate pool
6. **全量验证**（Dpareto）→ 记录 score matrix

关键：`feedback_text` 是第二类诊断信号。除了执行 trace，评估过程中的自然语言输出（如编译错误信息、失败 rubric 说明）也被直接喂给反思 LLM。这让 GEPA 在 code optimization 等任务上有天然优势。

### 3.2 Pareto-Based Candidate Selection（核心创新）

**问题**：贪婪选「全局最优 candidate」→ 局部最优陷阱，budget 消耗在无效搜索上。

**GEPA 的解法**：维护 Pareto frontier。
- 对每个 training instance `i`，记录所有 candidate 中在该 instance 上的最高分
- 一个 candidate 只要在「至少一个 instance」上达到最优，就进入 Pareto 前沿
- 被任何其他 candidate 在所有 instance 上支配的才被剪除
- 从 Pareto 前沿按「领跑 instance 数量」加权采样

**效果**：不同 candidate 可能擅长不同类型问题，Pareto 策略保持 diversity，避免早熟收敛。

```
实验对比（Qwen3 8B, aggregate improvement）:
- SelectBestCandidate（贪婪）: +6.05%
- BeamSearch (N=4): +5.11%
- GEPA (Pareto): +12.44%
```

### 3.3 System-Aware Merge（Crossover）

对于多模块系统，不同演化分支可能在不同模块上学到互补策略。Merge 操作：
- 识别不同分支中表现最好的 module prompt
- 跨分支选最优模块组合 → 新 candidate

GEPA+Merge 额外提升 ~2%（GPT-4.1 Mini 上更显著，Qwen3 8B 上有时反而降性能——对 crossover 时机敏感）。

---

## 实验结果

**Qwen3 8B 基准**（Table 1）：

| 方法 | HotpotQA | IFBench | HoVer | PUPA | AIME-25 | LiveBench-M | Agg. |
|------|----------|---------|-------|------|---------|-------------|------|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | **38.00** | 51.26 | 48.91 |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 20.00 | 46.60 | 47.84 |
| GEPA | **62.33** | **38.61** | **52.33** | **91.85** | 32.00 | **51.95** | **54.85** |
| GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | 32.00 | 51.95 | 52.40 |

**GEPA rollout 数**：平均 3936；GRPO rollout 数：24000（约 6x 差距）

AIME-2025 是 GEPA 唯一不敌 GRPO 的任务（32% vs 38%）——纯数学推理对权重更新的依赖更强，prompt 优化天花板较低。

**GPT-4.1 Mini 基准**（Table 2，针对 closed model）：

| 方法 | Aggregate | vs Baseline |
|------|-----------|------------|
| Trace/OptoPrime | 56.30 | +3.27% |
| MIPROv2-No-Demos | 57.14 | +4.11% |
| MIPROv2 | 58.67 | +5.64% |
| TextGrad | 59.14 | +6.11% |
| GEPA | 65.22 | +12.19% |
| GEPA+Merge | **66.36** | **+13.33%** |

**Cross-model 泛化**：用 Qwen3-8B 优化的 prompt 直接用于 GPT-4.1-Mini → +9.00%，超过所有专门为 GPT-4.1-Mini 优化的 baseline。

---

## 关键 Observations

**O1：极端 sample efficiency**：
- GEPA 匹配 GRPO 最优 validation 性能只需 102–1179 次 rollout（GRPO 用 24000）
- 若只算 train rollout（不含 validation），有任务只需 6–32 次即超过 GRPO

**O2：纯 instruction 优化超越 instruction+few-shot**：
- GEPA 不优化 few-shot demonstration，只改 instruction
- 但 aggregate 性能超过联合优化 instruction+few-shot 的 MIPROv2
- 原因：现代 LLM 的 instruction following 能力已足够强，declarative rule 比 few-shot 更 generalizable

**O3：Pareto selection 是关键**（见上方 ablation）

**O4：GEPA prompt 比 MIPROv2 prompt 短 9.2x**：
- 更短 = 推理时计算更低
- 更短 = 泛化性更好（没有 overfitting 到 few-shot case 的模式）

**O5：Crossover（Merge）效果 model-dependent**

**O6：Cross-model 泛化**：用弱模型优化的 prompt 在强模型上 transfer，超过专门为强模型优化的 baseline

---

## 我的评价

### 为什么是 ICLR 2026 Oral

这篇论文的贡献是**实证性**的，而且实证结果很 striking：

1. **颠覆预期**：「prompt 优化不能 update 权重，应该比 RL 弱得多」是行业默认假设。GEPA 在 5/6 任务上打败 GRPO，而 rollout cost 是 GRPO 的 1/6，这个结果本身就是 claim。

2. **Pareto frontier** 的想法非常 elegant：把 multi-task 性能矩阵转化为多元 Pareto 问题，保持 candidate 多样性。这比「选最好的 candidate」和「beam search」都更本质——因为不同策略在不同任务类型上有互补优势。

3. **实用性极强**：GEPA 对 closed-source model 直接适用（不需要权重访问），对 compound AI system（多模块）直接适用，对推理时 search 也适用。这让它的 impact surface 远大于 RL post-training 方法。

### 和 E-SPL 的关系

E-SPL 的 Evolution-only 条件就是 GEPA 的简化版（E-SPL 论文明确提到「our EA is very similar to GEPA」）。区别：
- **GEPA**：冻结权重，只演化 prompt，支持 compound multi-module system
- **E-SPL**：同时演化 prompt + 更新权重，只针对单模型 post-training

两篇论文在 AIME 任务上的分歧有意义：GEPA 在 AIME 上不敌 GRPO，而 E-SPL 的 AIME 增益就是靠 RL 权重更新贡献的。这说明**纯数学推理仍然是 RL 的主场，RL 在这类任务的隐式程序性知识编码上有不可替代的优势**。

### 边界条件

1. **什么时候 prompt evolution 够用**：任务可以通过更好的「策略说明」改善时（多跳 QA、指令遵循、隐私保护）。任务需要内化新的「直觉/程序性知识」时（硬数学推理）→ 需要 RL。

2. **Compound system 假设**：GEPA 最适合有明确模块划分的系统（如 RAG pipeline、multi-hop agent），对 single-model 端到端系统的优势可能减弱。

3. **Validation cost**：GEPA 的 rollout 大部分花在 Dpareto 评估（candidate selection），不是 training signal。如果 eval 很贵，实际效率优势缩水。

4. **Crossover 时机**：Merge 在 Qwen3-8B 上有时反而降性能，说明 crossover 的超参需要 task-aware 调整，还不够 plug-and-play。

### 深层联系：三篇论文的统一框架

| 论文 | 主张 | 机制 |
|------|------|------|
| CTA (2602.16699) | RL 无法学 meta-exploration 策略 | 显式 prior 注入（cost/difficulty estimation） |
| GEPA (2507.19457) | Prompt evolution 可替代 RL（多数任务） | 自然语言反思 + Pareto selection |
| E-SPL (2602.14697) | Prompt evolution + RL 联合优化最强 | 声明性/程序性知识分工 |

三者都指向同一个核心：**RL 的 policy gradient 是低带宽、高 sample cost 的学习信号；自然语言反思是高带宽、低 sample cost 的学习信号**。当任务可以通过显式策略表达时，language-based learning 更优；当需要内化隐式直觉时，RL 不可或缺。

这是 2026 年 AI post-training 的一条重要方法论分界线。

---

## 相关论文

- [[AI/LLM/RL/Other-Algorithms/E-SPL-Evolutionary-System-Prompt-Learning|E-SPL]] (arXiv:2602.14697) — 在 GEPA 基础上加上 RL weight update，联合优化；E-SPL 论文明确说「our EA is very similar to GEPA」
- [[AI/Agent/Agentic-RL/Calibrate-Then-Act-Cost-Aware-Exploration|Calibrate-Then-Act]] (arXiv:2602.16699) — 显式先验注入，探索策略不能从 RL 自动涌现；与 GEPA 结论三角互证
- [[AI/LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR Edge of Competence]] — RL 在能力边界附近才有效；GEPA 补充了"软边界内 language 更优"的视角
- **MIPROv2** — Instruction + few-shot 联合优化，GEPA 的主要 baseline
- **TextGrad** — 基于文本 gradient 的 prompt 优化
- **FunSearch/AlphaEvolve** — 代码层面的演化搜索，同类思路不同对象

---

*笔记日期: 2026-02-20*
