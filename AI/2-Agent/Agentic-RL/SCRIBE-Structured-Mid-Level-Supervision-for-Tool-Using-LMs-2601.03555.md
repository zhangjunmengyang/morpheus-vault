---
title: "SCRIBE: Structured Mid-Level Supervision for Tool-Using Language Models"
authors:
  - Yuxuan Jiang
  - Francis Ferraro
nyear: 2026
source: "https://arxiv.org/abs/2601.03555"
tags:
  - agentic-rl
  - tool-use
  - reward-modeling
  - credit-assignment
  - process-supervision
  - rubric
---

## TL;DR（我的判断）
SCRIBE 把“过程监督 / step-level credit assignment”的难点，**从 open-ended LLM judge 变成 constrained verification**：先把 trajectory 切成「subgoal」，再把每个 subgoal 路由到一个 **skill prototype 库**，用 prototype 自带的结构化 rubric 来评判，从而显著降噪。它不是在追求更强 judge，而是改变 judge 的输入结构。

## 论文问题意识
- Tool-augmented agent 的 RL 训练难：multi-step reasoning 的 credit assignment。
- 现有 process-level RM / LLM judge 容易 **noisy & inconsistent**，原因不是模型不够大，而是缺少“细粒度、任务特定的 rubrics”，导致 judge 既要理解高层 plan，又要管低层执行，误差被放大。

## 方法：Skill-Conditioned Reward with Intermediate Behavioral Evaluation
（根据 arXiv 摘要信息重建，高置信；细节需后续精读 PDF）

核心组件：
1. **Skill Prototypes（原型库）**
   - 人为/半自动整理的一组中层技能原型（可以理解为“子任务模板 + 评价维度”）。
2. **Subgoal routing（子目标路由）**
   - 把一段复杂行为分解为 subgoals，并将每个 subgoal 映射到某个 prototype。
3. **Constrained verification（约束验证）**
   - judge 不再自由发挥，而是在 prototype 给定的 rubric 上打分/判定，从而降低方差。
4. **Mid-level intervention（中层抽象）**
   - 不是 token-level 也不是 trajectory-level，而是“介于 plan 与 tool-execution 之间”的 mid-level。

## 关键实验信号（来自 arXiv 摘要）
- 在 reasoning + tool-use benchmarks 上 SOTA。
- 例子：Qwen3-4B 的 AIME25 accuracy **43.3% → 63.3%**（+20.0 绝对百分点）。
- 多轮 tool interaction 成功率显著提升。
- 训练动力学分析：出现“跨抽象层 co-evolution”——**mid-level skills 的 mastery 往往先于高层 planning 行为的涌现**。

## 我认为它解决了什么“元问题”
### 1) 把 credit assignment 的难点从“更精确的数值回归”转为“更可判定的结构化验证”
这和我们在 CM2（checklist）里看到的方向一致：
- 与其幻想一个万能 RM/LLM judge 能对开放式行为给稳定分数，不如把 supervision 变成一组 **可验证的原子条件**。

区别在于：
- CM2 更像「turn-level checklist」；
- SCRIBE 在 turn 内引入 **subgoal × prototype** 的路由，使 rubric 更“条件化”（conditioned）。

### 2) 它是“Intermediate Verification Signal 自动化”缺口的一块拼图
INBOX 里那个 #gap 的本质：
- 我们缺的不是“用 checklist 当 reward”，而是：**如何自动/半自动生成这些 intermediate signals**，避免人手工写 checklist。

SCRIBE 的启发：
- intermediate signal 可以来自一个“可复用的原型库”，而不是每个任务从零写 rubric。
- 真正的自动化入口可能在：
  1) prototype 库如何扩展（自动聚类/归纳新 prototype）；
  2) subgoal 识别与路由如何更鲁棒（弱监督/自训练）；
  3) rubric 的表达形式如何标准化（schema / DSL）。

## 局限与我会质疑的点
- **Prototype 库的构建成本**：如果库主要靠人工 curating，它会把 reward engineering 成本从“每任务写 checklist”转移到“维护 prototype taxonomy”。
- **Domain shift**：原型库是否跨领域可迁移？还是会像技能树一样强 domain-specific？
- **路由误差**：subgoal → prototype 的错配可能造成系统性误奖惩（尤其是多工具、多意图混合的 subgoal）。

## 与 Vault 现有框架的连接（建议补链）
- Credit Assignment 谱系：它属于“step/turn 之间的 mid-level CA”，与 HiPER（segment-level）/ GiGPO（step-level）形成互补轴。
- Tool Use RL 的「token masking」规范：SCRIBE 仍应遵守 observation 不参与梯度（需要看论文训练细节确认）。

## See Also
- CM2（Checklist Rewards，arXiv:2602.12268）
- iStar（Implicit Step Rewards，arXiv:2509.19199）
- AgentPRM / GiGPO（step-level credit assignment）
