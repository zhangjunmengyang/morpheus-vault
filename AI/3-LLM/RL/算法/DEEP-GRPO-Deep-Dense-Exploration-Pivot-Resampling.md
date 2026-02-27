---
brief: "DEEP-GRPO——通过 Pivot-Driven Resampling 解决 GRPO 探索塌缩；识别高质量 pivot 轨迹后进行密集重采样，避免 root saturation（模型趋向同一起点）；在 RLVR 数学任务上覆盖更广推理路径。"
title: "DEEP-GRPO: Deep Dense Exploration via Pivot-Driven Resampling"
type: paper
domain: ai/llm/rl
tags:
  - rl
  - GRPO
  - exploration
  - pivot-resampling
  - root-saturation
  - RLVR
created: 2026-02-14
status: v1
---

# DEEP-GRPO: Deep Dense Exploration via Pivot-Driven Resampling

> Deep Dense Exploration for LLM Reinforcement Learning via Pivot-Driven Resampling

**arXiv**: 2602.14169  
**作者**: Yiran Guo, Zhongjian Qiao, Yingqi Xie, Jie Liu, Dan Ye, Ruiqing Zhang, Shuang Qiu*, Lijie Xu*  
**提交日期**: 2026-02-15  
**投稿目标**: ICML（abstract 结尾标注）  
**标签**: #GRPO改进 #探索策略 #RL训练 #轨迹采样 #树搜索

---

## 核心问题：GRPO 的探索缺陷

### 三个维度的失效

**1. GRPO 的 Root Saturation 问题**

GRPO 每次都从问题根节点（root）重新采样完整轨迹。问题：
- 随着训练推进，policy 对高概率路径越来越自信
- 大量 rollout budget 浪费在已掌握的高概率轨迹上
- 深层的 error-prone states 被累积概率衰减挡在门外
- **实验验证**：把 GRPO 的 rollout 数量从 N=8 扩大到 N=64，性能从 64.1% 到 66.2%，但 N=32→64 几乎没有增益（diminishing returns）

**2. Tree-Based 方法的 Sample Dispersion 问题**

TreeRL、AttnRL、FR3E 等从中间状态分支，但：
- 有限的 budget 分散在大量 intermediate states 上
- 每个分支点的 local sample 数量极少 → local advantage 估计不稳定
- 混合了 global policy 分布和人工探索路径 → 训练不稳定

**3. 现有探索启发式的错误目标**

- TreeRL：高熵 token → 实际上很多是语言上的同义词，不是逻辑不确定性
- AttnRL：高 attention score → 表示步骤重要性，不是错误易发性
- 这些启发式可能 branch 在浅层，而浅层已经被 root sampling 覆盖了

---

## 方法：Deep Dense Exploration (DDE) / DEEP-GRPO

**核心 insight**：

> (1) 失败轨迹中有很多 valid reasoning prefix——从这些位置 resample 可能找到正确 suffix，形成高质量对比对。
> (2) 深层 states 通过 root sampling 指数级难以到达，但这些 states 往往有高不确定性，需要密集优化。
> → 找到"深而可恢复"的 pivot states，在那里密集 resample。

### 三个创新组件

---

#### 组件 1：Utility-Guided Pivot Selection

**Pivot Sampling Distribution**：

```
Q(t) ∝ P_φ(success | s_{<t}) × (t/T)^γ
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
            Recoverability              Depth Bias
```

- `P_φ(success | s_{<t})`：给定前缀，估计能找到正确 suffix 的概率
- `(t/T)^γ`：深度偏置，γ 控制倾向深层的程度（默认 γ=2）

**关键设计**：两个因素相乘创造 "sweet spot"——
- 太浅：root sampling 已经覆盖，branch 价值低
- 太深：recoverability 太低，几乎不可能纠正
- **最优 pivot = 深度足够（root sampling 难以到达）+ 可恢复性足够（还能找到正确路径）**

**Recoverability 估计：Online Logistic Regression**

不是复杂的 value model，而是一个极轻量的 logistic 回归：

```
P_φ(success | r_t) = σ(w · r_t + b)
```

- `r_t = t/T`：当前 step 在轨迹中的归一化位置
- 维护 experience buffer M，存储 (r_t, y_t)
- `y_t = 1` 若 K 个 branch 中至少一个找到正确答案，否则 0
- 在线更新（binary cross-entropy），无需额外模型参数
- **关键假设**：Recoverability 主要由轨迹中的相对位置决定，是跨问题的结构性规律

---

#### 组件 2：Local Dense Resampling

**两阶段层级生成**：

1. **Main Chain**: 从 root 标准 GRPO 采样 G 条轨迹，识别失败的 T_fail
2. **Auxiliary Chain**: 对每条失败轨迹 τ^i ∈ T_fail：
   - 用 Q(t) 采样一个 pivot t*_i
   - 在 prefix s_{<t*_i} 上密集生成 K=8 条续写 → 形成 "bifurcation"
   - 总 budget = G (root) + |T_fail| × K (local) ≈ G + failed_ratio × G × K

**重要区别**：tree-based 方法把 budget 分散在所有中间节点，DEEP-GRPO 把 K 个 branch 全部集中在**同一个 pivot**上 → 保证 local advantage 估计的稳定性。

---

#### 组件 3：Dual-Stream Optimization

**问题**：如果直接把 main chains 和 auxiliary chains 合并，sample 量不均衡会导致梯度贡献比例不稳定。

**解决**：解耦两个流：

```
J(θ) = E_{x~D}[
  (1/G) Σ L_main(τ^i)                            ← 全局策略学习
  + λ × (1/|T_fail|) Σ (1/K) Σ L_aux(τ̂^{i,k})  ← 局部纠错
]
```

**Main Chain Loss (L_main)**：
- 用 T_main 的 group-wise 统计计算 global advantage A_global
- 标准 GRPO clip + KL regularization

**Auxiliary Chain Loss (L_aux)**：
- 用同一 pivot 的 K 个 sibling branches 计算 **local advantage** A_local
- **Gradient Masking**：frozen prefix s_{<t*}，只对生成的 suffix y^{i,k} 计算梯度
- 原因：shared prefix 对两个流都有梯度，直接更新会造成干扰

**λ = 1 by default**（两个 stream 等权）

---

## 实验结果

### GSM8K（Qwen2.5-0.5B-Instruct）

| 方法 | 准确率 |
|------|--------|
| GRPO (N=8) | 64.1% |
| GRPO (N=64) | 66.2% |
| TreeRL | 65.5% |
| AttnRL | 67.0% |
| **DEEP-GRPO** | **67.7%** |

- DEEP-GRPO vs GRPO(N=64)：**+1.5%**（同等总 budget，但用了 root=8 + local K=8）
- DEEP-GRPO vs AttnRL：**+0.7%**
- 训练过程中 policy entropy 持续更高（避免了 premature convergence）
- 生成的 response 更长（探索了更深的推理）

### 数学推理 Benchmarks（1.5B & 7B）

| 模型 | AIME24 | AMC | MATH500 | Minerva | Oly. | **Avg** |
|------|--------|-----|---------|---------|------|---------|
| Oat-Zero-1.5B (Dr.GRPO) | 20.0 | 53.0 | 74.2 | 25.7 | 37.6 | 42.1 |
| **DEEP-GRPO-1.5B** | **26.7** | 50.6 | **75.2** | **27.2** | 36.7 | **43.3** |
| Oat-Zero-7B (Dr.GRPO) | 43.3 | 62.7 | 80.0 | 30.1 | 41.0 | 51.4 |
| **DEEP-GRPO-7B** | **46.7** | **65.1** | **81.6** | **33.8** | **42.6** | **54.0** |

- 7B 平均 **+2.6%** vs Dr.GRPO baseline
- AIME24 增益最大（1.5B: +6.7%，7B: +3.4%）——困难任务上探索优势更明显

---

## 批判性分析

### 为什么这个 insight 是对的

**理论依据**：
- 在 MDP 中，轨迹的 "bifurcation point"（分叉点）是信息量最高的位置——在这里一步差异决定了成功/失败
- 深层 bifurcation points 对 policy 的贡献被 root sampling 严重低估（概率衰减是指数级的）
- Logistic regression 的在线学习能在训练过程中动态跟踪 policy 的 recoverability 演变

**实验支撑**：
- DEEP-GRPO 在整个训练过程中 entropy 保持高位 → 说明真的在探索新区域
- 困难 benchmark（AIME24）上收益最大 → 难题有更多深层 error-prone states

### 我的疑问

1. **Logistic Regression 是否 oversimplify**？
   假设 recoverability 仅取决于轨迹位置（而非实际推理状态）可能太强。不同问题的 pivot 位置可能差异很大——一道简单题的 depth=0.8 可能远比难题的 depth=0.5 更可恢复。这个假设在 ablation 中有没有被检验？

2. **K=8 的选择**：
   Budget 对比：DEEP-GRPO (p1b8) = G root rollouts + 1 failed trajectory × 8 branch = G+8（如果 fail rate=1/G）。但实际失败率会更高，真实 budget 可能比 GRPO N=64 高很多——论文如何控制公平对比？

3. **auxiliary loss 的 gradient masking 细节**：
   如果 prefix 不更新，policy 对 pivot 之前的 token 没有反馈信号。在长 prefix 情况下，大量 token 的梯度被完全丢弃，这是否限制了 policy 改进的范围？

4. **与 STAPO 的互补性**：
   STAPO 在 token 级别过滤 spurious gradients，DEEP-GRPO 在 trajectory 级别优化探索点。两者结合应该有更好的效果，论文没有测试。

### 与相关工作的对比

| 方法 | 解决问题 | 机制层次 |
|------|---------|---------|
| GRPO | 基础 RL | Trajectory-level |
| STAPO | Spurious token 梯度 | Token-level |
| Goldilocks | 任务难度课程 | Sample-level |
| TreeRL/AttnRL | 探索深度 | State-level（分散）|
| **DEEP-GRPO** | 探索深度+密度 | State-level（集中）|
| MASPO | Trust region + 信号 | Sample+Token-level |

DEEP-GRPO 最接近 tree-based 方法，但关键创新是把 "dense local resampling at one pivot" 替代 "sparse branching at many states"——这个 insight 简单但有效。

### 我的评分

**★★★★☆**

- 问题识别准确：GRPO 的 root saturation 是真实瓶颈，有数据支撑
- 解决方案优雅：Pivot selection 的 utility function 设计巧妙，offline logistic regression 轻量实用
- Dual-stream optimization 解耦避免了 weight imbalance，是工程上重要的 insight
- 结果可信：多模型（0.5B/1.5B/7B）多 benchmark 验证，ICML 投稿
- 扣一星：recoverability 的 position-only 假设太强；与 STAPO/Goldilocks 的联合实验缺失；budget 对比不够清晰

---

## 深层洞察：探索策略的本质

**GRPO、树搜索、DEEP-GRPO 代表了三种不同的探索哲学：**

- **GRPO**：广度优先，从全局分布采样——在分布内探索，越来越窄
- **树搜索**：结构化探索，追求全覆盖——预算太分散，信号太弱
- **DEEP-GRPO**：目标化深度探索——找到"边缘状态"密集攻坚

这三种哲学对应了 exploration-exploitation 问题的三种答案：
1. Exploit（GRPO：不断利用已知好路径）
2. Explore broadly（树搜索：尽量覆盖）
3. **Explore strategically**（DEEP-GRPO：找 pivot，集中火力）

真正的进展来自第三种——因为 LLM 的错误往往不是随机的，而是集中在特定的推理失败模式上。Pivot 就是这些模式的"反转点"。

---

## 关联

- **前序工作**：Yiran Guo 同组有 **Segment Policy Optimization (SPO)**（2025-05-29），处理 segment-level credit assignment——这篇 DEEP-GRPO 是更新的探索方向延伸
- **与 HEARTBEAT 追踪的论文群**：STAPO（token级）+ Goldilocks（sample级）+ DEEP-GRPO（exploration）+ MASPO（trust region）构成了 GRPO 改进的四条独立路线
- **统一框架**：[[AI/3-LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL-Training-Stability-2026-Unified-Analysis]]（v3 已纳入 DEEP-GRPO 探索维度）
- **同族**：[[AI/3-LLM/RL/算法/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO-Spurious-Token-Aware-Policy-Optimization]]、[[AI/3-LLM/RL/算法/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks-RL-Task-Difficulty-Curriculum]]、[[AI/3-LLM/RL/算法/Stable-Asynchrony-VCPO-Off-Policy-RL|Stable-Asynchrony-VCPO-Off-Policy-RL]]、[[AI/3-LLM/RL/算法/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO-Mass-Adaptive-Soft-Policy-Optimization]]
