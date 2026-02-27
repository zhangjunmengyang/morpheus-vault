---
title: "SNDT: 扩散语言模型的测试时缩放——跨轨迹奖励引导拼接"
brief: "Stitching Noisy Diffusion Thoughts（SNDT）利用 masked diffusion LM 廉价并行采样多条推理轨迹，用 PRM 逐步评分，再跨轨迹拼接最优步骤构建合成 rationale。在 LLaDA-2.0（8B）上 Math500 +5pt（48.2→53.2），step count 几乎不变——实现「免费」的测试时增益。"
arxiv: "2602.22871"
date: 2026-02-28
rating: ★★★☆
tags:
  - ai/llm/inference
  - test-time-scaling
  - diffusion-lm
  - PRM
  - self-consistency
  - type/paper
related:
  - "[[AI/3-LLM/Architecture/LaViDa-R1-Diffusion-LLM-Reasoning]]"
  - "[[AI/3-LLM/Inference/Sink-Aware-Pruning-Diffusion-LLM]]"
  - "[[AI/3-LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention]]"
---

# SNDT: Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching

> arXiv:2602.22871 | 2026-02-27 | ★★★☆

---

## 一句话定位

把 masked diffusion LM 的**非自回归并行采样**特性变成 test-time scaling 的资产：采样多条多样推理轨迹（低成本），用 off-the-shelf PRM 逐步评分，跨轨迹拼接最高质量步骤序列——不增加 diffusion steps，获得 +5pt Math500 提升。

---

## 背景：Diffusion LLM 做推理的两个阶段

已有工作（**LaViDa-R1**，arXiv:2602.14147）解决**训练时**：把 GRPO 风格的 RL 搬到 dLLM，让 diffusion 模型学会推理过程。

SNDT 的切入点是**测试时**：已经有了能推理的 dLLM（LLaDA-2.0），如何进一步提升性能？

关键观察：Diffusion LM 采样**天然多样**——同一 prompt 的多次采样轨迹差异大（不同 masked token 还原路径不同），采样成本相对低（固定步数 budget 内并行）。与自回归 LLM 不同，dLLM 可以做**跨轨迹拼接**——因为生成不依赖严格的左→右因果关系。

---

## SNDT 三步流程

```
Step 1: 多轨迹采样（廉价多样性）
  给定 problem prompt
  → masked diffusion LM 采样 N 条推理轨迹（N=10~30）
  → 轨迹间差异显著（不同 masking 路径→不同中间状态）
  → 总计算量 ≈ 1 次 AR beam search

Step 2: PRM 逐步评分（step-level 质量估计）
  每条轨迹的每个推理步骤 → off-the-shelf PRM 打分
  → N 条轨迹 × M 步 = N×M 个候选步骤，各带 PRM 分数

Step 3: 跨轨迹拼接（奖励引导组合）
  贪心策略：每个步骤位置选 PRM 分数最高的步骤
  → 从不同轨迹中拼出一条「最优合成轨迹」
  → 这条轨迹不一定存在于任何原始轨迹中
```

**为什么 Diffusion LM 可以拼接而 AR LM 不行？**

AR LM 每个 token 依赖之前所有 token（因果掩码）。跨轨迹拼接会导致语义断裂——轨迹 A 的 step 3 在轨迹 B 的 step 1,2 之后会前后不一致。

Diffusion LM 的生成是**全局去噪**：一次前向传播同时处理整个序列，不存在严格因果依赖。拼接后可再做少量扩散去噪"修复"拼接边界。这是 SNDT 的核心 insight：**dLLM 的非因果性 = 天然的 stitching 接受能力**。

---

## 实验结果

| 模型 | 基准 | 基线 | SNDT | 提升 |
|------|------|------|------|------|
| LLaDA-2.0（8B） | Math500 | 48.2 | **53.2** | +5.0pt |
| LLaDA-2.0（8B） | GSM8K | 强 | 保持强 | 无退化 |
| LLaDA-2.0（8B） | Step count | 256 | 267 | +4%（极小） |

**关键**：+5pt Math500，只多用 4% steps——几乎"免费"。收益来自轨迹间多样性：不同轨迹覆盖不同子问题的正确路径，拼接聚合了这些正确片段。

---

## 深度分析

### 与 Self-Consistency 的本质区别

**SC（Wang et al. 2022）**：AR LLM 采样 N 条答案，多数投票。粒度：**trajectory-level**。不改变推理过程，只选结果。

**SNDT**：step-level 拼接。粒度：**step-level**，构建实际上不存在于任何原始轨迹的"合成最优轨迹"，充分利用 N 条轨迹的全部信息。

类比：SC 是"选最好的参赛选手"，SNDT 是"把每位选手最好的表现片段剪辑成全明星集锦"。

### 信息利用率视角

- SC / Best-of-N：利用约 1/N 的总信息（只看胜出轨迹）
- SNDT：每步都选最优 → 理论上可利用接近 N 倍总信息量

实际受限于步骤间的上下文依赖和拼接一致性，但 5pt 提升说明信息利用有实质改善。

### 为什么 step count 几乎不变？

dLLM 推理在固定步数 budget（256 steps）内完成去噪。Stitching 是对已有轨迹的后处理选择，不额外增加去噪步骤。+11 steps 仅来自拼接边界的局部修复。这与 AR LLM test-time compute（更多 tokens = 更多 FLOPS）根本不同：SNDT 的"额外计算"主要是 PRM 推理（相对轻量）。

---

## 局限与批判

**适用范围窄**：依赖 dLLM 的非因果特性，**不适用于标准 AR LLM**（GPT/Llama/Qwen 等）。当前生产主流是 AR LLM。

**基础模型性能差距**：LLaDA-2.0 Math500 基线 48.2，远低于同规模 AR 推理模型（Qwen2.5-7B-Instruct 约 75+）。SNDT 后 53.2 仍有代差。这是在**相对弱的基础**上的改进，绝对性能不具竞争力。

**PRM 质量依赖**：数学推理有成熟 PRM，但代码/Agent 任务等场景 PRM 质量未知。

**拼接一致性**：不同轨迹的步骤拼接后可能有语义断层（前步假设 x=5，后步不知道这个前提）。后处理去噪可缓解但不能完全消除。

---

## 在推理优化体系中的位置

```
Test-Time Scaling 方法谱系：

AR LLM：
  Best-of-N → SC（多数投票）→ MCTS → TSR（树搜索替换 rollout）
  粒度：trajectory → trajectory → step（需重新生成）

Diffusion LLM（SNDT 开辟的路径）：
  vanilla decoding → SNDT（跨轨迹 step-level 拼接）
  粒度：trajectory → step（重新组合，无需额外生成）

关键区别：
  AR LLM 的 step-level 方法需要"重新生成"（rollout 候选步骤）
  SNDT 的 step-level 是"重新组合"（已有轨迹的步骤库）
  → SNDT 计算效率更高（前提：已有多条轨迹）
```

**与 LaViDa-R1 的关系（正交互补）**：
- LaViDa-R1：训练时 RL → dLLM 获得推理能力（能力获取）
- SNDT：测试时拼接 → 在已有推理能力基础上进一步提升（能力释放）
- 两者可叠加：RL 训练好的 dLLM + SNDT

---

## See Also

- [[AI/3-LLM/Inference/Thinking-by-Subtraction-Confidence-Contrastive-Decoding-Reasoning|Thinking by Subtraction]] — 单轨迹精炼（与 SNDT 多轨迹拼接互补）：compute 受限用前者，充足用后者
- [[AI/3-LLM/Architecture/LaViDa-R1-Diffusion-LLM-Reasoning]] — 训练时维度：diffu-GRPO 让 dLLM 获得推理能力，与 SNDT 形成训练/测试两端互补
- [[AI/3-LLM/Inference/Sink-Aware-Pruning-Diffusion-LLM]] — dLLM 推理优化，侧重稀疏性/剪枝
- [[AI/3-LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention]] — Block diffusion + sparse attention，dLLM 推理加速
- [[AI/2-Agent/Agentic-RL/Search-P1-Path-Centric-Reward-Agentic-RAG]] — 同为"路径结构→密集信号"思路，但用于 agent RL 训练；SNDT 是 test-time 的"路径拼接"，Search-P1 是训练��的"路径评分"
