---
title: "RL 训练信号利用效率：三条正交路径"
type: 思考
date: 2026-02-27
tags: [RL, GRPO, PolicyOptimization, AgentRL, 认知框架]
related:
  - "[[AI/3-LLM/RL/算法/SSPO-Soft-Sequence-Policy-Optimization]]"
  - "[[AI/3-LLM/RL/算法/GC-RL-Second-Order-Rollout-Generation-Critique]]"
  - "[[AI/2-Agent/Agentic-RL/Search-P1-Path-Centric-Reward-Agentic-RAG]]"
  - "[[AI/2-Agent/Agentic-RL/SORL-Stabilizing-Off-Policy-RL-Long-Horizon-Agent]]"
---

# RL 训练信号利用效率：三条正交路径

> 写于 2026-02-27，整合当日精读的三篇论文

---

今天读了三篇表面方向不同的工作：Search-P1（路径级奖励密度）、GC-RL（二阶 rollout）、SSPO（IS 粒度校正）。读完发现它们说的是同一件事的不同侧面——**在不增加训练数据量的前提下，提升 RL 训练信号的利用效率**。

这值得整理成框架，因为这个问题会越来越重要。随着 base model 越来越强，RL 能改善的空间越来越依赖训练信号的质量和密度，而不是数据量。

---

## 问题的本质

标准 GRPO 的训练过程：问题 q → 采样 k 个回答 → 计算 advantage → 梯度更新。这个过程在三个维度上存在信号浪费：

1. **信号密度低**：整条轨迹只得到 binary reward（对或错）。失败轨迹中 90% 的步骤可能是对的，但 sparse 0 reward 使这些贡献为零。

2. **信号维度窄**：采样了 k 个回答，训练了生成能力，但模型从未被要求评判这些回答质量。"评判"这个维度的训练信号被完全忽略。

3. **信号传递失真**：多次 mini-batch update 后，behavior policy 和 update policy 出现分布漂移。token-level IS ratio 对 sequence-level off-policy 程度的校正不准确，有效信号被噪声稀释。

---

## 三条正交路径

### 路径 A：提升信号密度（Search-P1，arXiv:2602.22576）

解法：**Dual-Track Path Scoring**——不只看最终答案对不对，也看路径本身质量：
- Self-Consistency Track：模型是否忠实执行了自己的计划？
- Reference-Alignment Track：路径是否与参考解对齐？

关键是 **Soft Outcome Scoring**：失败轨迹不再是 reward=0，而是根据路径质量给 soft score。信号密度从 binary 变为连续，失败轨迹也能贡献梯度。

**本质**：把 reward 的时间粒度从"最终结果"推进到"路径质量"，soft scoring 消灭二值化。

### 路径 B：拓展信号维度（GC-RL，arXiv:2602.22765）

解法：**Second-Order Rollout**——在 response 之上再做一层 critique rollout。对每个 response，再采样 m 个 critique，让模型判断这个 response 对不对。

**Critique reward 是免费的**：因为 response 的 ground truth 已知，"critique 是否正确"可以自动计算，不需要额外标注。同样训练数据产生两个维度的信号：生成维度 + 评判维度。

**本质**：不增加训练数据，增加训练任务的维度。同一组样本既训练生成能力，也训练元认知能力。

### 路径 C：修复信号传递（SSPO，arXiv:2602.19327）

解法：**Geometric Mean IS + Soft Gate**——几何均值把 token-level IS ratio 聚合成 sequence-level IS（理论等价 length-normalized sequence IS），sigmoid soft gate 替代硬 clip，避免高偏差 token 的梯度被直接截断。

**本质**：确保信号能被更准确地传递到参数更新。不增加信号，减少传递过程中的失真。

---

## 三条路径的正交性

```
┌──────────────────────────────────────────────┐
│         RL 训练信号利用效率                   │
│                                              │
│  信号密度 ─── Search-P1（路径级 soft score） │
│  信号维度 ─── GC-RL（二阶 rollout）          │
│  信号传递 ─── SSPO（IS 粒度校正）            │
│                                              │
│  + SORL：multi-turn 场景下的信号传递稳定性   │
└──────────────────────────────────────────────┘
```

三条路径修改训练过程的不同环节，可以同时应用。理想训练栈：SSPO（确保传递准确）+ Search-P1（增加路径级密度）+ GC-RL（增加评判维度）。

SORL 是 multi-turn 扩展——turn-level IS 修复时间维度的信号失真，与 SSPO 的空间维度（token vs sequence）互补。

---

## 判断

**哪条路径最值得跟进？**

路径 C（SSPO/信号传递）是基础性的。信号传递有失真，路径 A/B 的效果都会打折扣。SSPO 类工作理论贡献清晰（等价性证明），工程成本低（不增加额外 rollout），应该最先被采用。

路径 A（Search-P1/信号密度）在 multi-step agentic 任务中价值最大。单步问答里路径级信号不如 outcome 准确（路径可能错但结果碰巧对）；agent 任务中"路径质量"才真正重要。这是 Search-P1 在 AD-QA（+20.6）远高于标准问答的原因。

路径 B（GC-RL/信号维度）最有创意但也最难验证。"双能力正反馈回路"有道理，但 critique 能力提升是否真能反哺生成能力，需要更严格的消融实验。

**更大的格局**

这三条路径背后的共同直觉：**已有训练数据比通常认为的含有更多信息**。Binary reward、单维度任务、粗粒度 IS，都是在用宽刷刷一幅可以用细笔的画。

这和预训练数据工程的哲学一样：FineWeb/D4/DoReMi 都不是在收集更多数据，而是在问"如何从已有数据提取更多信息"。RL 阶段现在走到了同样的岔路口。

"数据够用了，现在的问题是怎么用"——这一判断在 2026 年越来越被接受。

---

*整合性思考，非综述。有错请指正。*
