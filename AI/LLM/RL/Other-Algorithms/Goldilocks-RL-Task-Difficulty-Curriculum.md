---
brief: "Goldilocks RL——课程学习：任务难度既不能太易（无学习信号）也不能太难（全失败）；自适应调度任务难度分布，始终保持在 RLVR 能产生有效梯度的 competence 边界；与 RLVR-Edge-of-Competence 互补印证。"
title: "Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards"
type: paper
domain: ai/llm/rl
tags:
  - rl
  - curriculum-learning
  - sample-efficiency
  - sparse-reward
  - difficulty-scheduling
  - RLVR
created: 2026-02-16
status: v1
---

# Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards

> arXiv: 2602.14868 | Apple + EPFL | 2026-02-16
> Authors: Ilia Mahrooghi, Aryo Lotfi, Emmanuel Abbe

## 一句话

RL 训练中梯度信号只有在 p_q ≈ 0.5 时才最大，Goldilocks 用一个 Teacher 模型在线预测每道题对 Student 的难度，专挑"不太难不太容易"的题训练——在同等算力下显著加速收敛。

---

## 问题：sparse reward 下的样本低效

GRPO/RLVR 训练有个根本问题：reward 是 binary（对/错），sparse reward 导致：
- 太简单的题：Student 全对 → p_q = 1 → 梯度为零，纯浪费算力
- 太难的题：Student 全错 → p_q = 0 → 梯度也为零，同样浪费
- 只有 p_q ≈ 0.5 的题才有最大 learning signal

现有 Curriculum Learning 方法的问题：
1. **History-based（需要多次重访同一样本）**：大规模训练不现实，数据集比 checkpoint 还大
2. **Category-based（按题型分组）**：假设难度只取决于类别，而不是具体样本

---

## 核心数学：梯度与难度的关系

**Theorem（来自 Section 2）**：

对于 GRPO + verifiable reward (Bernoulli(p_q))，policy gradient 的梯度 norm 为：

$$\|\nabla_\theta L_{PG}\| = \sqrt{p_q(1-p_q)} \times \|E[\log\pi|\text{correct}] - E[\log\pi|\text{wrong}]\|$$

**关键结论**：
- 梯度 norm 正比于 **√(p_q(1-p_q))**
- 这是 Bernoulli 方差的平方根
- **p_q = 0 或 1 → 梯度为零**
- **p_q = 0.5 → 梯度最大**

这不只是直觉，是严格推导。Goldilocks 的目标：让 Teacher 找出那些 p_q 接近 0.5 的题——即 Student 处于"能力边界"的题。

这与 RLVR-Edge-of-Competence 论文高度相关（那篇分析的是 KL 约束边界，这篇是梯度信号角度）。

---

## 系统设计：Teacher-Student Framework

### 架构

```
Dataset (3M+ problems)
    ↓ 随机采样 K_candidate 道候选题
Teacher (LM backbone + linear head)
    → 预测每道题的 utility score: f_φ(q) = 0.5·σ(w^T·MeanPool(Embed(q)))
    → ε-greedy 选择 utility 最高的题 q*
Student (policy LM, trained with GRPO)
    → 生成 G 个 rollout
    → 计算 empirical p_q = success_rate
    → 更新 GRPO
Teacher 异步更新:
    → target y_q = √(p̂_q(1-p̂_q))  ← 实测方差作为监督信号
    → MSE loss: minimize (f_φ(q) - y_q)²
    → Replay buffer (sliding window)
```

### Teacher 的巧妙之处

Teacher 学习的不是"这题答案是什么"，而是"这题对当前 Student 有多大 learning value"。

监督信号 y_q = √(p̂_q(1-p̂_q)) 来自 Student 的真实表现，Teacher 用 MSE 拟合这个目标。随着训练推进：
- Student 能力提升 → 越来越多题变 easy (p_q → 1)
- Teacher 的预测均值 μ 自动下降（动态适应 Student 的成长）
- Teacher 对 unseen 数据的 MAE 持续下降（泛化能力验证）

### ε-greedy 数据选择

- 以概率 ε：随机从候选池选题（保证覆盖）
- 以概率 1-ε：选 Teacher 打分最高的题（利用 curriculum）

### 计算资源分配

- 2 GPU 给 Teacher（难度预测）
- 6 GPU 给 Student（GRPO 训练）
- 公平比较：GRPO baseline 8 GPU，步数按 8/6 换算

---

## 实验结果

**数据集**：OpenMathReasoning（3M+ CoT problems）

**模型**：
- Qwen2.5-1.5B (Teacher = 同模型 or Qwen3-1.7B)
- Qwen3-4B (Teacher = Qwen3-1.7B，跨模型 curriculum)
- Phi-4-mini-instruct 4B
- Olmo2-1B

**主要结果**（Table 1）：
- Goldilocks 在同等算力预算下，所有模型均优于 GRPO baseline
- 学习速度更快（收敛步数更少），最终精度更高

**关键现象**：
- Training reward std 全程更高（Teacher 确实在选 p ≈ 0.5 的题）
- 零方差样本比例大幅减少（梯度不再被浪费）
- Gradient norm 持续更高（优化更高效）

**Ablation**：
- DAPO loss + Goldilocks > DAPO baseline
- GRPO + Entropy Regularization + Goldilocks > GRPO + Entropy Reg
→ Goldilocks 的 curriculum 效果与 loss function 正交，可叠加

---

## 我的评价

**★★★★☆（理论清晰，实用性强，但规模验证不足）**

### 为什么好

1. **数学根基扎实**：从梯度 norm = √(p(1-p)) × signal_gap 出发，curriculum 的方向不是猜测，是定理推导的必然结论
2. **Teacher 泛化能力**：不是 history-based（不需要重访同一道题），对未见数据也能预测 utility——这是工程上真正可行的关键
3. **正交叠加性**：Ablation 证明 Goldilocks 与 DAPO/entropy reg 正交，任何 RL recipe 都可以叠加
4. **动态适应**：Teacher 的均值预测随 Student 成长自然下降，自动追踪能力边界

### 我的质疑

1. **规模验证缺失**：最大模型是 4B，没有在 7B/13B/70B 验证——scaling 行为未知
2. **Teacher 的计算开销**：2/8 GPU 给 Teacher，约 25% overhead，对大规模训练来说代价不小，能否用更小模型做 Teacher？
3. **OpenMathReasoning 的分布特异性**：这个数据集本身就是专门为 RL reasoning 设计的，在代码/通用推理领域能否复现？
4. **p_q 估计的噪声**：G=16 个 rollout 估计 p_q，方差不小（例如真实 p_q = 0.3，16 个样本估计误差约 ±0.11）。Teacher 学的是带噪信号。
5. **与 DAPO 的 dynamic sampling 关系**：DAPO 已经要求 batch 里有对有错（0 < correct < G），效果有多少重叠？

### 与相关工作的关系

| 方法 | Curriculum 类型 | 是否需重访数据 | 是否泛化新数据 |
|------|----------------|--------------|--------------|
| Beta Distribution Sampling (Qu et al.) | target p_q = γ* | 是 | 否 |
| Thompson Sampling (Shen et al.) | uncertainty-based | 是 | 否 |
| DAPO Dynamic Sampling | 过滤全对/全错 | 否 | 部分 |
| **Goldilocks** | Teacher 预测 utility | **否** | **是** |

Goldilocks 在 generalization 维度上是真正的创新——不需要重访，不依赖类别标签。

### 大局判断

**√(p(1-p)) 最大化**这个目标实际上就是在最大化"outcome variance"——这和强化学习经典的 exploration-exploitation trade-off 在本质上是同一问题。Goldilocks 的贡献是把它变成了一个 online learning 问题：Teacher 学习预测 p_q，无需重访，无需类别先验。

这个方向有两个重要推论：
1. 在数据选择之外，能否用同样逻辑做 **训练时长分配**（给 medium-difficulty 题分配更多 rollout budget）？
2. 能否应用于 **预训练数据筛选**（在 pretraining 阶段用 Teacher 过滤 "太简单或太难" 的文本）？第 9 节 Future Work 提到了这个方向，值得跟进。

---

## 连接

- 数学基础同源：[[AI/LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]]（边界 vs 梯度视角，互补）
- 同类 RL 稳定性：[[AI/LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO-Spurious-Token-Aware-Policy-Optimization]]（token 级稳定 vs batch 级 curriculum，正交）
- 同族：[[AI/LLM/RL/Other-Algorithms/Stable-Asynchrony-VCPO-Off-Policy-RL|Stable-Asynchrony-VCPO-Off-Policy-RL]]（系统级稳定，与 Goldilocks 样本级正交）
- 统一框架：[[AI/LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL-Training-Stability-2026-Unified-Analysis]]（Token/样本/系统三分法，覆盖 Goldilocks）
- 相关算法：[[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO]]、[[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO]]
- Credit Assignment：[[AI/LLM/RL/GRPO/Blockwise-Advantage-Estimation|Blockwise-Advantage-Estimation]]（同样是"哪些信号更有价值"的问题）
- [[AI/LLM/RL/Other-Algorithms/PACED-RL-Partition-Function-Difficulty-Scheduler|PACED-RL]] ⭐ — **独立多路验证**：PACED-RL 用 GFlowNet Z_φ 估计准确率做难度调度，与 Goldilocks 的 Teacher LM 路径完全不同，但收敛到同一规律（中间难度最优）；两篇合读让这个规律从 empirical 升级为 robust finding

---

*写于 2026-02-20 | Scholar*
