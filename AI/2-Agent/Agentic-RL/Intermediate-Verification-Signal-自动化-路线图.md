---
title: "Intermediate Verification Signal 自动化：路线图（从评测到训练）"
created: 2026-03-02
tags:
  - type/wisdom
  - topic/agent-rl
  - topic/verification
  - topic/reward-shaping
  - topic/llm-as-judge
---

## 这条线到底在解决什么？
长时序 / tool-use / long-form 任务的核心痛点不是“优化器不会写”，而是**终局奖励太稀疏、且不可解释**。

所谓 *Intermediate Verification Signal*，就是把“最终成功/偏好”拆成一组 **中间可验证约束**（subgoal/constraint/checklist/path-structure），让训练/选择/自我改写都能吃到密集信号。

关键：
- 信号必须 **可判定（verifiable-ish）**，否则只是在做另一种风格的打分，极易 Goodhart。
- 信号必须 **可自动生成**，否则人力写 rubric/checklist 无法规模化。

## 统一闭环（我现在认为最有前途的范式）

instruction → (auto) checklist/constraints → verify →
- **evaluation**：TICK（结构化 judge）
- **test-time improvement**：STICK/self-refine + Best-of-N（用 checklist 做搜索/筛选）
- **training-time RL**：ACE-RL / CM2（把 verify 结果当 reward / shaping）

> 这条闭环最像“把 reward model 从一个标量，拆成一组可审计的 predicate”。

## 信号来源的几种“自动化路径”（按结构性从强到弱）

### A) Instruction 分解 → Checklist（最直接、最工程化）
- **TICK (2410.03608)**：judge LLM 生成 instruction-specific YES/NO checklist，用于评测；exact agreement 46.4%→52.2%。
- **ACE-RL (2509.04903)**：从真实指令里自动生成 constraints checklist + verifier/RM 逐条验证，用作 RL reward（长文生成）。
- **CM2 (2602.12268)**：multi-turn tool-use 的 checklist reward（偏人工设计，但提供了“checklist=structured verification”范式）。

适用：长文/指令跟随/工具型 agent 的 reward 设计。

### B) 结构约束（temporal order / stage mapping）→ Potential Shaping（理论更干净）
- **STO-RL (2601.08107)**：LLM 给出有序 subgoals + state→stage 映射，用 potential-based shaping 把 sparse 终局奖励变 dense。

适用：offline RL / 环境可定义 stage 的长任务。

### C) Mid-level supervision / schema（把“过程正确性”写成结构）
- **SCRIBE (2601.03555)**：结构化中层监督（偏“写清楚中间状态/步骤格式”），属于把监督信号结构化的一类。

适用：tool-use 行为轨迹需要中层规范的场景。

## 我现在最关心的 3 个 failure modes（也是选型分水岭）
1) **Coverage failure（漏项）**：checklist 没覆盖到最关键隐含意图，优化会被带偏。
2) **Verifier hacking（可欺骗）**：模型学会迎合 verifier（写出“看似满足”的证据），而不是达成真实目标。
3) **Correlation trap（相关≠因果）**：某些 checklist 条目只是“表象相关”，优化它会伤害总体质量。

## 一个可执行的对照实验清单（给后续工程落地）
- Checklist 生成：同一 instruction、多次采样 → 计算 checklist 的稳定性（Jaccard/语义聚类）。
- Verifier 鲁棒性：对抗 prompt / style hacks / irrelevant but persuasive evidence，测 false positive。
- Goodhart 测试：用 checklist 做 BoN selection vs 直接 judge 打分 selection，观察是否出现“高 checklist 低人评”。
- 迁移：在 unseen domain/新体裁/新工具上复用 checklist pipeline，测泛化与退化模式。

## 判断（为什么这条线值得追）
- 它把“对齐/训练”问题从黑盒标量 reward，推进到 **可审计的结构化证据链**；这在工程上可 debug，在研究上可累计。
- 但真正的技术护城河不是“会生成 checklist”，而是：
  - checklist 的 **生成质量控制**（coverage + consistency）
  - verifier 的 **可信度与对抗鲁棒性**
  - 以及如何把 verify 信号注入到 RL/搜索而不引发新一轮 reward hacking。

## See Also
- [[AI/3-LLM/Evaluation/TICK-Generated-Checklists-Improve-LLM-Evaluation-and-Generation-2410.03608.md]]
- [[AI/3-LLM/RL/Fundamentals/ACE-RL-Adaptive-Constraint-Enhanced-Reward-2509.04903.md]]
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL.md]]
- [[AI/2-Agent/Agentic-RL/STO-RL-Offline-RL-under-Sparse-Rewards-via-LLM-Guided-Subgoal-Temporal-Order-2601.08107.md]]
- [[AI/2-Agent/Agentic-RL/SCRIBE-Structured-Mid-Level-Supervision-for-Tool-Using-LMs-2601.03555.md]]
