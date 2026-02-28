---
title: "Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems"
brief: "Dr. MAS（arXiv:2602.08847，NTU Singapore）——多智能体 GRPO 训练的稳定性问题：全局 reward 归一化在角色异构的多 agent 场景下导致梯度范数爆炸；Agent-Wise Advantage Normalization 用 per-agent (μₖ,σₖ) 替换全局统计量，理论证明梯度二阶矩不爆炸；搜索任务 +15.2%，与 RAGEN（单 agent Echo Trap）正交互补。"
arxiv: "2602.08847"
authors: "Lang Feng, Longtao Zheng, Shuo He, Fuxiang Zhang, Bo An (NTU Singapore)"
date: 2026-02-09
venue: "arXiv preprint"
tags: [multi-agent-rl, grpo, training-stability, gradient-norm, policy-optimization]
rating: ★★★★☆
sources:
  - "Dr. MAS: Feng et al., arXiv:2602.08847, NTU Singapore, 2026-02-09"
  - "RAGEN: Wang et al., arXiv:2504.20073（对比：单 agent 内部 Echo Trap）"
  - "MARS2: arXiv:2602.11455（对比：多 agent 性能 Scaling）"
---

## 一句话

多智能体 GRPO 训练中，全局归一化 baseline 对不同 agent 的 reward 分布不对齐，导致梯度范数爆炸；**每个 agent 用自己的 reward 统计量 (μₖ, σₖ) 做归一化**，把梯度二阶矩压回正常范围，+15.2% avg@16 on search。

---

## 问题：全局归一化在多智能体场景下失效

GRPO 的 advantage normalization 假设 group 内所有 rollout 来自同一分布：

$$A^i_\text{global} = \frac{R^i - \mu}{\sigma}$$

但在多智能体系统中：
- 不同 agent 承担不同角色（规划者 vs 执行者，检索者 vs 合成者）
- 各 agent 的 reward 分布 **天然不同**：mean (μₖ) 和 variance (σₖ²) 都可能偏离全局统计量

**Lemma 4.2（梯度二阶矩分解）**：对 agent k 的梯度二阶矩：

$$\mathbb{E}\left[\|\tilde{g}_k^\text{global}\|^2\right] = \mathbb{E}\left[\|z_{i,t}^{(k)}\|^2\right] \cdot \frac{\sigma_k^2 + (\mu_k - \mu)^2}{\sigma^2} + \Delta_k$$

其中 $\frac{\sigma_k^2 + (\mu_k - \mu)^2}{\sigma^2}$ 是放大因子：
- 当 agent 的 mean reward 偏离全局均值（μₖ ≠ μ）→ 项 (μₖ - μ)² 放大梯度
- 当 agent 的 reward 方差远大于全局方差（σₖ² >> σ²）→ 比例失调
- **随着 agent 专业化程度增加，各 agent 的 reward 分布必然越来越异质，全局归一化越来越不稳定**

**Proposition 4.3（梯度范数爆炸）**：当 |μₖ - μ|/σ 或 σₖ²/σ² 增大时，梯度二阶矩至少线性增长 → 梯度范数爆炸。

---

## 解法：Agent-Wise Advantage Normalization

极简修改：用每个 agent 自己的 reward 统计量替换全局统计量：

$$A^{i,k}_\text{agent} = \frac{R^i - \mu_k}{\sigma_k}$$

其中 μₖ 和 σₖ² 仅在该 agent 被激活的步骤上计算（即 $\mathcal{Y}_k = \{a_t^i \mid k_t^i = k\}$）。

**效果**：放大因子变为 $(\sigma_k^2 + (\mu_k - \mu_k)^2)/\sigma_k^2 = 1$，梯度二阶矩回归到：

$$\mathbb{E}\left[\|\tilde{g}_k^\text{agent}\|^2\right] = \mathbb{E}\left[\|z_{i,t}^{(k)}\|^2\right] + \Delta_k$$

仅由 score function 本身决定，不再受跨 agent reward 分布差异影响。

**直觉**：每个 agent 在自己的 reward 空间里比较，而不是用全局均值作参照，避免了"规划者总是高分 / 执行者总是低分"导致的梯度量纲失衡。

---

## 系统框架：Dr. MAS End-to-End Training

除算法外，论文还提供了完整的多智能体 RL 训练框架：

### 三大组件

1. **Multi-Agent Orchestrator**：
   - 动态选择激活哪个 agent
   - 支持条件路由（基于当前状态或前一 agent 输出）
   - 与 veRL-agent / VerlTool 系列兼容

2. **Agent-Model Assignment**：
   - 逻辑 agent (1...K) → 物理 LLM worker group 的映射
   - 支持 **LLM 共享**（多 agent 用同一模型，用 role prompt 区分）
   - 支持 **异构模型**（高层规划者用 7B，低层执行者用 3B）

3. **Shared Resource Pooling**：
   - GPU 资源池化，所有 actor backend 映射到 ActorRollout 角色
   - 推理引擎：sglang（高吞吐低延迟）
   - 支持 per-agent 独立优化配置（学习率、KL 系数等）

### 关键设计：LLM 共享 vs 异构

| 模式 | 适用场景 | 优点 |
|------|----------|------|
| Shared LLM | agent 能力要求相近，role 差异靠 prompt | 参数共享，显存效率高 |
| Heterogeneous | 高层规划 vs 低层执行能力差异大 | 灵活分配算力，小模型做低层任务 |

---

## 实验结果

**基准**：Qwen2.5 + Qwen3 系列，多智能体数学推理 + 多轮搜索

| 指标 | GRPO（基线） | Dr. MAS | 提升 |
|------|-------------|---------|------|
| 数学 avg@16 | baseline | +5.6% | +5.6% |
| 数学 pass@16 | baseline | +4.6% | +4.6% |
| 搜索 avg@16 | baseline | +15.2% | **+15.2%** |
| 搜索 pass@16 | baseline | +13.1% | +13.1% |
| 梯度 spike | 频繁 | 基本消除 | ✓ |

**搜索任务提升明显**（15.2% vs 5.6%）：搜索任务中检索者和合成者的 reward 分布差异更大，全局归一化失效更严重 → per-agent normalization 收益更大。

**异构模型设置**（7B planner + 3B executor）：Dr. MAS 仍然有效，体现 per-agent normalization 对模型规模差异的鲁棒性。

---

## 与相关工作的关系

### 与 RAGEN/StarPO 的互补
| 问题 | RAGEN/StarPO | Dr. MAS |
|------|-------------|---------|
| 根因 | Multi-turn reward 方差坍塌（单 agent 内部） | 跨 agent reward 分布异质（多 agent 间） |
| 症状 | Echo Trap（梯度趋零） | 梯度范数爆炸（梯度过大）|
| 解法 | filtering + critic baselining + decoupled clipping | per-agent advantage normalization |

两者是 Multi-Agent RL 稳定性的**不同维度**：
- RAGEN 处理训练过程中的**动态退化**（单智能体多轮 RL 的 Echo Trap）
- Dr. MAS 处理训练初始就存在的**异质性**（多智能体角色专业化的 reward 分布差异）

### 与 Dr. GRPO 的关系
Dr. GRPO（单智能体）识别了 GRPO 中 normalization 的另一个问题（重复 query 的 bias），Dr. MAS 识别了多智能体场景的特有问题。命名上有意对应，核心都是"更准确的 baseline 归一化"。

### 与 MARS2 的关系
MARS2 研究的是多智能体的**性能 scaling**（异构 agent 协作超过单一大模型），Dr. MAS 解决的是多智能体的**训练稳定性**。两者是同一方向的两个维度。

---

## 关键洞察

### 洞察 1：专业化越深，全局归一化越危险
随着 multi-agent system 中 agent 专业化程度提高，各 agent reward 分布的异质性必然增加。这是一个随 agent 分工演进而自动加剧的问题，不是偶发 bug。

### 洞察 2：改动极小但效果显著
Agent-wise normalization 只是把全局 (μ, σ) 换成 per-agent (μₖ, σₖ)，代码改动几行，却能从理论上保证梯度二阶矩不爆炸。这是"identify the right problem → minimal fix"的经典路径。

### 洞察 3：Multi-Agent RL 缺少基础设施
论文的系统框架部分（Section 4.3）揭示了一个行业空白：现有 RL 框架（veRL/OpenRLHF/ROLL）基本是单 LLM 设计，多智能体场景的 native 支持很弱。Dr. MAS 是少数同时解决算法和基础设施的工作。

### 洞察 4：搜索任务是多智能体 RL 的 stress test
搜索任务（检索者 + 合成者）比数学任务（多步推理）更适合暴露多智能体训练的不稳定性，因为 reward 分布差异更极端：检索者 reward ≈ 0/1 二元，合成者 reward 连续且范围宽。

---

## 局限与待解

1. **实验规模**：仅测了 2 种任务（数学 + 搜索），agent 数量未超过 3 个，规模化（K > 5）的行为未知
2. **Reward 异质性根因**：per-agent normalization 解决了症状，但没回答"如何设计使各 agent reward 分布更接近的 task structure"
3. **与 GiGPO 的对比**：GiGPO 做 step-level credit assignment，Dr. MAS 做 agent-level normalization，两者可以叠加但未实验
4. **异步 rollout**：论文在同步 rollout 下测试，GLM-5 的 slime 框架做异步 RL，两者结合未知

---

## Multi-Agent RL 知识地图更新

```
Multi-Agent RL 训练
├── 训练稳定性
│   ├── 单 agent 内部（RAGEN/StarPO）— Echo Trap 三联征
│   └── 跨 agent 间（Dr. MAS）— Reward 分布异质 → 梯度范数爆炸  ← 本文
├── 性能 Scaling
│   └── MARS2 — 异构 2×32B > 单体 72B
├── 训练内部路由
│   └── PA-MoE — phase-level expert routing
└── 推理时编排
    └── AdaptOrch — 四拓扑 DAG 路由
```

---

---

## See Also

**Multi-Agent RL 训练稳定性谱系**
- [[AI/2-Agent/Multi-Agent/FlexMARL-Rollout-Training-CoDesign-MARL-System|FlexMARL]] — MARL 基础设施层：Rollout-Training Co-Design，7.3x 加速（与算法层互补）
- [[AI/2-Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution|RAGEN & StarPO]] — **正交互补**：RAGEN 处理单 agent 内部多轮训练的 Echo Trap（梯度趋零），Dr. MAS 处理跨 agent reward 异质导致的梯度范数爆炸；两者覆盖多 agent RL 稳定性的不同维度
- [[AI/2-Agent/Agentic-RL/Multi-Agent-RL-训练专题|Multi-Agent RL 训练专题]] — AT-GRPO/MAGRPO/MARS2 全图谱；Dr. MAS 补稳定性维度

**性能 Scaling 对比**
- [[AI/2-Agent/Agentic-RL/MARS2-Multi-Agent-Scaling-Law-RL-Code-Generation|MARS2]] — 多 agent 性能 scaling（异构组合 > 单体大模型）；Dr. MAS 解决训练稳定性，MARS2 研究 scaling 规律，两者都关注多 agent 系统的优越性来源

**同名系列（Normalization 问题）**
- [[AI/3-LLM/RL/算法/Dr-GRPO-Unbiased-Optimization|Dr. GRPO]] — 单 agent 场景的 GRPO normalization 问题（重复 query bias）；Dr. MAS 是多 agent 扩展版本，命名对应

**训练拓扑与编排**
- [[AI/2-Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — agent 内部 phase-level expert routing；Dr. MAS 解决跨 agent reward 归一化
- [[AI/2-Agent/Multi-Agent/AdaptOrch-Task-Adaptive-Multi-Agent-Orchestration|AdaptOrch]] — 推理时多 agent 编排拓扑路由；Dr. MAS 解决训练时多 agent 梯度稳定性
