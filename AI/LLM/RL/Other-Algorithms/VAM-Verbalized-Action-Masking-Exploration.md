# VAM: Verbalized Action Masking for Controllable Exploration in RL Post-Training

**论文**: VAM: Verbalized Action Masking for Controllable Exploration in RL Post-Training — A Chess Case Study  
**arXiv**: 2602.16833 (cs.CL)  
**投稿**: ICML 2026  
**提交**: 2026-02-18  
**作者**: Zhicheng Zhang, Ziyan Wang, Yali Du, Fei Fang  
**评分**: ★★★☆☆（方法有创意，但领域限制较大）

---

## 核心问题：GRPO 探索的 Within-State Collapse

RLVR 的探索失败有两个层次：
1. **跨 state 探索**：模型在不同 prompt 上的行为多样性（已有不少研究）
2. **Within-state 探索**：对**同一个 prompt**，采 G 个 rollout，它们是否真的不同？

**VAM 聚焦第二个问题**：

GRPO 的核心机制依赖 group 内 reward 差异来计算 advantage：
```
A_i = (r_i - mean(r)) / std(r)
```

如果同一 prompt 下采样的 G 个 rollout 都是同样几个高概率动作，variance → 0，学习信号 → 消失。额外的采样纯属计算浪费。

这是 GRPO 探索的**隐藏瓶颈**：增加 group size 没有帮助，因为瓶颈在 action 多样性，不在 sample 数量。

---

## 方法：Verbalized Action Masking

### 核心思路

**不用 logit-level masking**（会破坏 on-policy 采样）。

改为 **prompt-level masking**：把 admissible action set 直接写进 prompt，用 verifier 检查输出是否合法。

```
[Prompt 结构]
State description: {task prompt}
Format specification: <think>...</think> <action>...</action>  
Allowed actions: [act1, act2, ..., actk]
```

out-of-mask action → 固定惩罚 + terminate rollout（Penalty 可以趋于无穷 = hard constraint）。

**关键优点**：每个 rollout 仍然是在当前策略下的 on-policy 采样（条件于 mask-conditioned prompt），与标准 GRPO 完全兼容。

### MDP 形式化

Augmented state space：
```
S̃ = {(s, M) : s ∈ S, M ⊆ A(s)}
```

Augmented reward：
```
R̃((s,M), a) = 1[a∈M]·R(s,a) + (1-1[a∈M])·(-Penalty)
```

当 Penalty→∞ 时退化为 hard action masking；有限 Penalty 是 soft constraint。

### 迭代动作空间剪枝（Iterative Action-Space Pruning）

这是 VAM 的核心探索机制：

```
Algorithm PruneAndSample(π, s, a*):
  M ← A(s)         # 初始 mask = 全部合法动作
  D_s ← ∅
  for r = 1 to R_max:
    采样 G 个 rollout {o_i} ~ π(·|s, M)
    计算 reward {r_i}
    D_s ← D_s ∪ {(M, {(o_i, r_i)})}
    V ← {已采到的有效动作}
    if a* ∈ V: break         # 找到目标动作，结束
    M ← M \ V               # 剪掉已采到的动作，强制探索新动作
  return D_s
```

**直觉**：如果模型一直采同一个动作，就把这个动作从候选集里删掉，迫使它去探索其他动作。

目标动作 a* = argmax_{a∈M} μ(s,a)（verifier 给分最高的动作）。

### 两种训练制式

| 制式 | State 来源 | Verifier |
|------|-----------|---------|
| Fixed-dataset | 预先建好的棋局数据集 | 预计算象棋引擎分数 |
| Engine-play | 与引擎对弈，on-policy 生成棋局状态 | 实时查询引擎 |

两种制式共用同一套 PruneAndSample 机制，只是 state 生成方式不同。

---

## 实验设置：国际象棋作为 Testbed

为什么选国际象棋？
1. **有限可枚举的 action set**：每个棋局位置的合法走法是精确定义的（通常 20-40 种）
2. **完美 verifier**：象棋引擎（Stockfish 等）可以给每步棋的质量打精确分数
3. **干净的探索失败观测**：先前工作发现，不提供合法走法列表时模型经常走非法棋，interface 选择主导了性能

评估指标：
- **Chess puzzles**：单步最优走法选择（pass@1）
- **全局对弈**：Average Centipawn Loss (ACPL)，值越小越好（损失的棋力越少）

---

## 实验结果

- VAM 在 chess puzzles 和全局对弈（ACPL）上均优于 strong baselines
- **Fixed-dataset 和 engine-play 两种制式均有效**，验证 pruning 机制的通用性
- 学习效率提升：相同训练步数下收敛更快

（具体数值 paper 中有但 HTML 截断，定性结论：一致性改进）

---

## 批判性分析

### 亮点

1. **问题定义精准**：within-state exploration collapse 是 GRPO 真实存在的问题，之前没有被系统讨论
2. **Prompt-level 而非 logit-level**：保持 on-policy 采样的设计决策是对的，logit-level masking 会破坏 autoregressive 分布
3. **MDP 形式化**：把 action masking 形式化为 augmented MDP 是清晰的理论贡献
4. **Chess 是好 testbed**：有限 action set + 完美 verifier + 清晰的失败模式

### 疑问与局限

1. **Domain 限制严重**：国际象棋有有限可枚举的 action set，但大多数 LLM RL 任务（数学、代码、推理）的 action space 是无限的字符串空间——如何定义 mask？如何枚举候选集？
2. **候选集来自哪里**：对于通用推理任务，"allowed action set"怎么构造？需要另一个模型来预先生成候选集 → 引入新的开销
3. **Pruning 开销**：每个 state 要跑 R_max 轮，增加了多少计算量？对于实际 LLM 训练可能是 2-5x 开销
4. **泛化性存疑**：论文自己也说是"chess case study"——对 AIME/coding 等主流 benchmark 是否有效，论文未验证

### 我的判断

**方法论上有价值，实用性受限**。

VAM 正确指出了 GRPO within-state collapse 这个被忽视的问题，这是真实 insight。但国际象棋的有限 action set 与通用 LLM 推理任务差距太大，直接迁移的路径不明确。

更有意思的推广方向：**对于数学问题，能否用 solution sketch（大纲）作为 action space 的结构化约束？** 或者用 LLM 本身先生成候选集，再用 VAM 强制覆盖？

**与 DEEP-GRPO 对比**：两者都在解决 within-trajectory 探索，但切入点不同：
- DEEP-GRPO：找 pivot 点重分支（从推理过程的关键决策节点出发）
- VAM：在 action space 层面强制覆盖（从候选集外部限制出发）

---

## 与 GRPO Panorama 的关系

VAM 属于 **Exploration 维度**，但与 DEEP-GRPO 正交：

| | DEEP-GRPO | VAM |
|-|-----------|-----|
| 问题层次 | 跨不同推理路径 | 同一 state 的 action 多样性 |
| 机制 | pivot resampling（从内部找决策点） | action space pruning（从外部限制候选集） |
| 适用范围 | 通用推理（数学/代码） | 有限 action space（chess 等） |
| 计算开销 | 按需 resampling | 固定轮次 × R_max |

---

## 元数据

- **Tags**: #exploration #GRPO #action-masking #chess #RLVR
- **关联笔记**: [[DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling]] [[GRPO-Improvement-Panorama-2026]] [[PACED-RL-Partition-Function-Difficulty-Scheduler]]
- **写于**: 2026-02-21
