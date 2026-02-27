---
brief: "RePO（arXiv:2602.10819）——通过 Rephrasing 缓解 off-policy 分布漂移；对难样本进行语义等价改写后重新采样，在 on-policy 和 off-policy 之间找到平衡；hard-sample mining 的工程实践方向。"
title: "RePO: Rephrasing Policy Optimization"
date: 2026-02-11
tags: [策略优化, off-policy, on-policy, hard-sample, 分布漂移]
domain: AI/LLM/RL/Other-Algorithms
arxiv: "2602.10819"
rating: 3
status: permanent
---

# RePO: Rephrasing Policy Optimization

> arXiv: 2602.10819 | 提交: 2026-02-11
> 作者: Linxuan Xia, Xiaolong Yang, Yongyuan Chen, Enyue Zhao, Deng Cai, Yasheng Wang, Boxi Wu
> 机构: 未标注（cs.LG）

## 评分: ★★★☆☆

## 一句话

On-policy RL 碰不到 hard sample（自己采不出来），Off-policy 直接喂又导致分布漂移崩溃。RePO 的解法：让模型先「读懂」专家解法，再用自己的话重写一遍——把 off-policy 知识变成 on-policy 兼容的轨迹，然后才注入训练。

---

## 问题：三角困境

```
SFT  → 学到了知识，但通用性退化（分布外泛化差）
On-policy RL  → 通用性好，但 hard sample 采不出来（能力边界以外的问题永远碰不到正确路径）
Off-policy RL → 理论上能给 hard sample 学习信号，但强制分布偏移 → 训练不稳定
```

LUFFY（直接从更强模型 sample 的 off-policy 方法）的问题：
- 强模型和训练模型词表不完全一致
- 强制 token 映射 → 梯度信号失真
- Hard sample 上尤其严重（正是最想学的地方最容易崩）

---

## 方法

### 三个阶段

**Phase 1：Knowledge Internalization（知识内化）**

构造 rephrasing prompt `P(q, k)`：
- `q` = 原始问题
- `k` = off-policy 知识（专家解法或 ground-truth 路径）
- Prompt 要求模型：先理解 `k` 的逻辑，然后用自己的方式重写

```
o_rep ~ π_θ(· | P(q, k))
```

关键：重写的是**模型自己**，所以生成的 `o_rep` 在模型自身的参数分布内——on-policy 兼容。

**Phase 2：Dynamic Injection（动态注入）**

不是无脑注入，而是基于当前 group 的失败率决定是否注入：

```python
γ_fail = (# of failed rollouts) / G

if γ_fail >= ρ:  # 模型在这道题上确实挣扎
    替换 group 中 reward 最低的那个 rollout 为 o_rep
else:  # 模型自己能解，保留纯 on-policy
    不注入
```

`δ` = 失败阈值（R < δ 算失败）
`ρ` = 最小失败率（超过才触发注入）

**Phase 3：Standard GRPO Optimization**

用改造后的 `O_final` 正常跑 GRPO。

---

## 稳定性分析

与 LUFFY 对比：
- LUFFY 的 token 强制映射 → importance ratio `r_{i,t}` 在 hard sample 处产生极端值 → 梯度爆炸
- RePO 的 o_rep 是模型自己生成的 → 词表一致 → importance ratio 始终在正常范围

数学上：RePO 的 `r_{i,t} = π_θ(o_{rep,t} | ·) / π_{θ_old}(o_{rep,t} | ·)` 比 LUFFY 的对应值方差小得多，clip 机制能正常工作。

---

## 实验结果

数学推理、代码生成、通识 QA 均优于 GRPO baseline 和 LUFFY。
（论文未给出具体数字表格内容，Abstract 说"state-of-the-art performance"）

---

## 我的分析

### 核心 insight 的本质

RePO 在做什么？**一种新的知识蒸馏形式**。

传统蒸馏：teacher 直接给 logits，student 拟合。
RePO 蒸馏：teacher 给解题路径，student 先理解再改写，然后从改写后的轨迹学。

这其实是**认知层面的蒸馏**而非分布层面的蒸馏——先理解，再内化。这个框架更接近人的学习方式。

### 真实的技术贡献？

诚实评价：
- **核心机制并不复杂**：rephrasing prompt + 动态替换低质量 rollout
- **真正的贡献**：识别出 off-policy stability 的根因（不是数据质量，是分布漂移），并给了一个 elegant 的工程解法
- **缺点**：额外计算 rephrasing 步骤；off-policy knowledge 来源需要外部资源（更强模型或 ground-truth 路径）

### 局限性

1. 需要 off-policy knowledge（从哪来？ground-truth 路径是否总是存在？）
2. Rephrasing 本身依赖模型能「读懂」专家解法——如果 hard sample 超出理解能力太多，rephrasing 质量差
3. `ρ` 和 `δ` 的设置需要调参

### 适用场景

- 有明确 ground-truth 路径的领域：数学、代码
- 需要突破 on-policy 能力边界时
- 不适合：奖励信号模糊或开放式任务（无 off-policy reference 可用）

---

## 与其他方法对比

| 方法 | Hard Sample 利用 | 训练稳定性 | 额外资源需求 |
|------|-----------------|----------|------------|
| GRPO | 差（采不到） | 高 | 无 |
| LUFFY | 好 | 低（分布偏移） | 外部模型 |
| **RePO** | 好 | 高（on-policy 兼容） | 外部知识 + 额外推理 |
| SFT | 好（直接喂） | 中 | 标注数据 |

---

## 连接

- 竞争：LUFFY（off-policy RL，RePO 直接 compare）
- 前驱：GRPO（backbone）
- 相关：KLong（也在解决 hard sample 的 RL 利用问题，但方向不同——轨迹切割 vs 知识重述）

## Tags
`#GRPO` `#off-policy` `#on-policy` `#knowledge-distillation` `#hard-sample` `#training-stability` `#2026-02`

---

## See Also

- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 2026 全景]] — RePO 在 Diversity 维度（与 ProGRPO 共同构成该维度）
- [[AI/3-LLM/RL/算法/ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]] — 同在 Diversity 维度，两者互补：RePO 改输入多样性，ProGRPO 改权重分布
-  — LLM 强化学习全图谱
- [[AI/3-LLM/RL/Theory/MARS-Margin-Aware-Reward-Modeling-Self-Refinement|MARS]] — 多样性视角：RePO 在 response 层增多样性，MARS 在 reward 层聚焦低 margin
