---
title: "RL 训练稳定性：2026 年前沿统一分析"
type: synthesis
domain: ai/llm/rl
tags:
  - rl
  - training-stability
  - GRPO
  - survey
  - STAPO
  - Goldilocks
  - DEEP-GRPO
  - MASPO
  - VCPO
  - interview-prep
created: 2026-02-20
updated: 2026-02-21
status: v3
---

# RL 训练稳定性：2026 年前沿统一分析

> 综合笔记 | 覆盖：STAPO / Goldilocks RL / Stable Asynchrony (VCPO) / DEEP-GRPO / MASPO / DAPO / LACONIC
> 写于 2026-02-20 | Scholar | 更新：2026-02-20 v3（新增 DEEP-GRPO Exploration 维度 + MASPO）

---

## 为什么要有这篇笔记

2026 年 2 月，arXiv 上涌现了一批专门研究 LLM RL 训练稳定性的论文。它们视角各异，但实际上在解决同一件事的不同侧面：**GRPO 系训练为什么会崩溃，怎么修**。

把这些工作串联起来，能看到一张比任何单篇论文都更完整的地图。

---

## 问题空间的拓扑

LLM RL 训练不稳定有**四个**独立的来源：

```
不稳定来源
├── Token 级别:    少量 token 携带过大梯度（→ STAPO）
│                  Trust region 设计不匹配（→ MASPO）
├── 样本级别:    大量样本贡献零梯度（太简单/太难）（→ Goldilocks RL）
├── 探索级别:    Root sampling 饱和，deep error-prone states 欠探索（→ DEEP-GRPO）
└── 系统级别:    异步训练导致 off-policy 方差放大（→ Stable Asynchrony）
```

这四个问题**独立正交**，可以同时存在，解法也不相互排斥。

---

## I. Token 级别：Spurious Tokens（STAPO，2602.15620）

**核心定理**（Theorem 3.1）：
$$\|\nabla_a \mathcal{J}(y_{i,t})\|^2 \propto |w_{i,t}|^2 \cdot \left(1 - 2\pi_t + e^{-\mathcal{H}(\pi)}\right)$$

**低概率 + 低熵 → 梯度最大**。当这样的 token 出现在正确序列里（正 advantage），它会收到不成比例的强化，污染 CoT 一致性。

**诊断三角**：
- π_t < τ_p（低概率）
- H_t < τ_h（低熵，模型本地已经很确定——但选了个异类）
- Â_i > 0（出现在正确回答里）

三者同时满足 = spurious token，数量约 **0.01%**，但影响是灾难性的。

**修法**：S2T mask，满足三条件则梯度清零。效果：+7.13% vs GRPO，熵始终稳定。

**深层理解**：这是一个 **reward credit assignment + sampling stochasticity** 的交叉问题。Sequence-level reward 无法区分"这个 token 真的对最终正确有贡献"和"这个 token 只是碰巧出现在正确序列里"。STAPO 是第一个从信息论角度（概率×熵×优势）精确定位 spurious token 的工作。

---

## II. 样本级别：Zero-Gradient Samples（Goldilocks RL，2602.14868）

**核心定理**（Section 2）：
$$\|\nabla_\theta L_{PG}\| = \sqrt{p_q(1-p_q)} \times \text{discrimination signal}$$

p_q = 0 或 1 → 梯度消失。在随机采样下，大量训练步骤在"太简单"或"太难"的题上消耗算力，毫无学习收益。

**Goldilocks 解法**：
- Teacher 模型在线预测每道题的学习价值 y_q = √(p̂_q(1-p̂_q))
- ε-greedy 选题：以 1-ε 概率选 utility 最高的题
- Teacher 用 Student 的实时 rollout 结果作为监督信号（MSE loss）
- Replay buffer sliding window，无需重访数据

**关键洞察**：学习价值的数学表达就是 **Bernoulli reward 的标准差**。这建立了一个优雅的自监督信号：Teacher 直接学习 "哪些题让 Student 的输出有最大方差"。

**与 STAPO 的关系**：
- STAPO 在 token 级别过滤"无用的更新"
- Goldilocks 在样本级别过滤"无梯度的数据"
- 两者正交，可以叠加使用

---

## III. 系统级别：Off-Policy Variance（Stable Asynchrony，Song Han lab，2026-02-19）

**已知信息**（来自 abstract 和论文标题）：

异步 RL 训练的动机是提高吞吐量——generation 和 training 解耦，GPU 利用率更高。GLM-5 的 Slime 框架就是这个思路（APRIL：Asynchronous Parallel RL Infrastructure）。

问题：REINFORCE/GRPO 这类 critic-free 方法，在高异步度下，**stale rollouts 产生高方差的 policy-gradient estimator**。

具体机制（根据问题设定推断）：
- 异步训练中，rollout 用旧版本策略 π_old 生成
- 当 policy 更新较快，rollout 和当前 policy 的 KL 偏移很大
- importance sampling ratio r_{i,t} = π_θ / π_old 的方差放大
- 越老的 rollout，r_{i,t} 的方差越大，等价于 off-policy correction 越不稳定

**论文声称的解法**："Variance-Controlled" — 从标题推断，可能是：
1. 对 staleness 程度进行量化，给不同"年龄"的 rollout 加权重
2. 通过方差估计动态截断高方差的 IS ratio
3. 或者类似 V-trace 的截断思路，但针对 LLM RL 的 token-level 特性定制

**为什么这很重要**：
- 所有主流异步 RL infra（Slime/APRIL、RLVE、OpenRLHF async）都面对这个问题
- GLM-5 技术报告里提到 APRIL 解决了 >90% 的 generation bottleneck，但没有详细分析 staleness 对梯度质量的影响
- 如果 Stable Asynchrony 提供了理论保证，它将成为所有异步 RL 系统的必读

**方法名 VCPO（Variance-Controlled Policy Optimization）**，两个核心机制（已从 abstract 摘录确认）：

**机制 1：ESS-based Learning Rate Scaling**
- ESS（Effective Sample Size）= $(\sum w_i)^2 / \sum w_i^2$，其中 $w_i = \pi_\theta/\pi_{\theta_\text{old}}$
- ESS 是 rollout 新鲜度的精确统计度量：ESS → N 表示几乎 on-policy，ESS → 1 表示高度 stale
- 根据 ESS 动态缩放学习率：stale rollout → 小 LR，新鲜 rollout → 大 LR
- 自适应、无需人工设定 staleness 阈值

**机制 2：Closed-Form Minimum-Variance Baseline（Off-Policy 专用）**
- GRPO 用 group mean reward 作 baseline，在 off-policy 场景下不是最优
- VCPO 推导 off-policy 下的最优 baseline，形式类似加权回归
- **无需 auxiliary value model**（保持 critic-free），overhead 极低
- 降低 gradient estimator 的固有方差

两机制互补：ESS scaling 控制"更新步长"，optimal baseline 控制"估计器方差"。

*(arXiv ID 待确认，full text 精读待补；详见 [[Stable-Asynchrony-VCPO-Off-Policy-RL]])*

---

## IV. 探索级别：Root Saturation（DEEP-GRPO，2602.14169）

**核心问题**：GRPO 从 root 采样 → 随训练深入，高概率路径被反复利用，深层 error-prone states 因累积概率衰减无法到达。

**实验验证**：把 N 从 8 扩到 64，性能从 64.1% 增到 66.2%，但 N=32→64 几乎无增益（diminishing returns）。

**Tree-based 方法的失败**：TreeRL/AttnRL 在 intermediate states 分散 budget → local samples 太稀 → local advantage 不稳定 + 混合了不同分布的轨迹。

**DEEP-GRPO 解法（三组件）**：

1. **Utility-guided Pivot Selection**（关键公式）：
   ```
   Q(t) ∝ P_φ(success | s_{<t}) × (t/T)^γ
   ```
   - Recoverability × Depth Bias 的乘积
   - 找"深而可恢复"的 sweet spot
   - P_φ 用轻量 Logistic Regression（r_t = t/T）在线估计，无需 value model

2. **Local Dense Resampling**：在单个 pivot 集中生成 K=8 branches，保证 local advantage 稳定

3. **Dual-Stream Optimization**：global GRPO 损失 + λ × local 修正损失，prefix 用 gradient masking 冻结

**结果（vs Oat-Zero Dr.GRPO baseline）**：
- AIME24: 1.5B +6.7%（43.3% vs 20.0%），7B +3.4%（46.7% vs 43.3%）
- 平均: 1.5B +1.2%，7B +2.6%

**与 Token/Sample/System 三层的关系**：探索层是独立维度——它解决"什么样的轨迹值得优化"，而不是"如何优化给定轨迹"。可以叠加使用。

*(详见 [[DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling]])*

---

## V. Trust Region 维度：MASPO（2602.17xxx，Xunliang Cai 组，2026-02-19）

GRPO 的 hard binary clip 在三个维度上不匹配 LLM 优化：
1. **梯度利用**：偏离 on-policy 的样本梯度被完全截零
2. **概率质量**：正/负样本在概率空间天然不对称，均匀处理扭曲信号
3. **信号可靠性**：正负样本之间的 credit assignment 模糊

**MASPO = Mass-Adaptive Soft Policy Optimization**：用 adaptive soft trust region 替代 hard clip 统一解决。

*(详见 [[MASPO-Mass-Adaptive-Soft-Policy-Optimization]]；全文待读，abstract-based 分析)*

---

## 统一框架：四层防御体系

```
Layer 4: 系统层（Stable Asynchrony / VCPO）
  └── 控制 off-policy 方差，保证 stale rollout 的 IS correction 质量

Layer 3: 探索层（DEEP-GRPO）
  └── 从 failed trajectories 的 pivot 点密集 resample，打破 root saturation

Layer 2: 样本层（Goldilocks RL）  
  └── 过滤零梯度样本，确保每个训练 step 都有有效 learning signal

Layer 1: Token 层（STAPO + MASPO）
  └── STAPO: 过滤 spurious tokens，防止异常梯度污染
  └── MASPO: soft trust region 改善梯度利用 + 信号可靠性
```

计算开销估算：
- VCPO：IS ratio + ESS 计算，O(1) per token，接近零 overhead
- DEEP-GRPO：额外 K×|T_fail| 次 rollout，~20-50% overhead（视失败率）
- Goldilocks：Teacher 约 25% GPU overhead
- STAPO：三条件 mask，接近零 overhead
- MASPO：soft clip 计算，接近零 overhead（vs hard clip）

---

## 与现有方法的比较矩阵

| 方法 | 干预层次 | 机制 | 开销 | 与 GRPO 关系 |
|------|----------|------|------|-------------|
| DAPO (clip-higher) | Objective | 非对称 clip，防止 IS ratio 爆炸 | 零 | 替换 |
| 20-Entropy | Sample | 对高熵 token 特殊处理 | 低 | 叠加 |
| JustRL | Objective + token | clip-higher + token norm | 零 | 替换 |
| LACONIC | Token | 长度约束，防 reward hacking | 低 | 叠加 |
| Lp-Reg | Token | 低概率 token 降权 | 零 | 叠加 |
| **STAPO** | **Token（精细）** | **三维联合 mask** | **接近零** | **叠加** |
| **Goldilocks** | **Sample（数据选择）** | **Teacher 预测 utility** | **中（25%）** | **正交** |
| **DEEP-GRPO** | **Exploration（探索）** | **Pivot selection + dense resampling** | **中（~30%）** | **正交** |
| **MASPO** | **Trust Region（优化）** | **Soft adaptive trust region** | **接近零** | **替换/叠加** |
| **Stable Asynchrony** | **System（异步）** | **方差控制 IS correction** | **低** | **正交** |

---

## 开放问题

1. **根因 vs 症状**：STAPO 是在 mask spurious token，Goldilocks 是在过滤无效样本，Stable Asynchrony 是在控制 staleness 方差。但这些都是在管控已有问题，而不是从根本上解决"sequence-level reward 无法区分 token 贡献"的问题。**Token-level reward 建模**（dense reward）是否才是更根本的方向？

2. **相互作用未知**：三层方法叠加时，是否存在干扰？例如 Goldilocks 把 p_q ≈ 0.5 的题优先喂给 Student，这类题本身 rollout 方差更大，是否会加剧 Stable Asynchrony 试图解决的 IS 方差问题？

3. **Scaling laws of stability**：随着模型规模增加，spurious token 的比例和影响是否变化？大模型的局部熵分布与小模型有何不同？

4. **理论下界**：GRPO 的方差理论下界是多少？是否存在一个 optimal critic-free policy gradient 方法能同时解决所有三个问题？

---

## 对实践的建议

如果要训练一个 reasoning LLM（2026 最佳 recipe 假设）：

```
基础: GRPO + clip-higher (DAPO) + token norm
+叠加: STAPO (mask spurious tokens)        ← 零额外成本
+叠加: Goldilocks (teacher-driven curriculum) ← ~25% GPU overhead  
+叠加: DEEP-GRPO (pivot-driven exploration)  ← ~30% rollout overhead
+可选: MASPO (soft trust region)            ← 替换 DAPO clip 机制
+如果异步: Stable Asynchrony/VCPO           ← 与 Slime/APRIL 集成
```

这五层组合，理论上从 Token/Exploration/Sample/TrustRegion/System 五个维度覆盖了所有已知的 GRPO 失效模式。

**重要警告**：DEEP-GRPO 的额外 rollout overhead 与 Goldilocks 的 Teacher overhead 可能叠加达到 50%+。实践中需要做 cost-benefit 分析，不一定全部叠加。

---

## 连接

**Token 级别**
- [[AI/LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO]] — spurious token mask，零成本稳定性
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — soft adaptive trust region，概率质量校正

**样本级别**
- [[AI/LLM/RL/Other-Algorithms/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] — Teacher 动态课程，中间难度最优
- [[AI/LLM/RL/Other-Algorithms/PACED-RL-Partition-Function-Difficulty-Scheduler|PACED-RL]] ★ — GFlowNet Z_φ 独立收敛同一规律（新增，2026-02-21）

**探索级别**
- [[AI/LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — Pivot-Driven Resampling，Root Saturation 修复
- [[AI/LLM/RL/Other-Algorithms/VAM-Verbalized-Action-Masking-Exploration|VAM]] ★ — Within-state 探索塌缩（新增，2026-02-21）

**系统/Off-Policy 级别**
- [[AI/LLM/RL/Other-Algorithms/Stable-Asynchrony-VCPO-Off-Policy-RL|Stable Asynchrony (VCPO)]] — 方差控制 IS correction
- [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 统一 FP8 精度 flow，消除量化引入的 off-policy
- [[AI/LLM/RL/Other-Algorithms/VESPO-Variational-Sequence-Policy-Optimization|VESPO]] ★ — 变分推导最优 IS kernel，理论最严格（新增，2026-02-21）
- [[AI/LLM/RL/Other-Algorithms/SAPO-Soft-Adaptive-Policy-Optimization|SAPO]] ★ — sech² 软门控，Qwen3-VL 生产在用（新增，2026-02-21）
- [[AI/LLM/RL/Other-Algorithms/GSPO-Group-Sequence-Policy-Optimization|GSPO]] ★ — 序列级 IS 替代 token 级，Qwen3 团队（新增，2026-02-21）

**Trust Region**
- [[AI/LLM/RL/Other-Algorithms/LACONIC-Length-Constrained-RL|LACONIC]] — Primal-Dual RL 长度控制

**基础理论**
- [[AI/LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]] — 边界竞争力理论
- [[AI/LLM/RL/GRPO/Blockwise-Advantage-Estimation|Blockwise Advantage Estimation]] — 分块优势估计
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — 六维框架元分析，本文的学术上位文档

**系统基础设施**
- [[AI/LLM/Frameworks/Slime-RL-Framework|Slime RL Framework]] — 异步 RL infra，Stable Asynchrony 的实战场景

---

*写于 2026-02-20 | Scholar | v3：四层框架 + DEEP-GRPO + MASPO*
*链接全路径化 + 新论文补入（PACED-RL/VAM/VESPO/SAPO/GSPO）：2026-02-21 | 馆长*
