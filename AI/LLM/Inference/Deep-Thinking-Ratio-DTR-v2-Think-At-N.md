# Think Deep, Not Just Long: DTR + Think@N（精读版）

> arXiv: 2602.13517 | UVA + Google | 2026-02-13
> Authors: Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, Yu Meng

## 一句话

token 数量是推理质量的**负相关**指标（r = -0.59），而 Deep-Thinking Ratio（DTR）是**强正相关**指标（r = 0.683）。用 DTR 做 Best-of-N 的样本选择，能以一半的推理成本匹配甚至超越 self-consistency。

---

## 问题：长链推理≠好推理

已有大量证据（逆scaling、inverted-U curve）表明：CoT 越长不等于越对，甚至越长越错。但现有的替代指标（confidence、log-prob、entropy）都不稳定。

论文问：**有没有内部机制信号，而不是表面统计量，能可靠预测推理质量？**

---

## 核心方法：Deep-Thinking Token

### 直觉

一个 token 如果"很好预测"，那么在浅层就能确定，随着层数加深分布不会大幅变化。  
一个 token 如果需要"深度思考"，那么预测分布在深层才会稳定——这是模型在不同表征层次上反复修订的体现。

**Deep-thinking token = 分布在深层才收敛的 token**

### 形式化

1. 每一层 $l$，用 unembedding matrix $W_U$ 把中间隐态投影到词表：$p_{t,l} = \text{softmax}(W_U h_{t,l})$

2. 计算每层与最终层的 JSD：$D_{t,l} = \text{JSD}(p_{t,L} \| p_{t,l})$

3. 定义 settling depth（第一个 $\bar{D}_{t,l} = \min_{j \le l} D_{t,j}$ 低于阈值 $g$ 的层）：
$$c_t = \min\{l : \bar{D}_{t,l} \le g\}$$

4. 若 $c_t \ge \lceil \rho L \rceil$（即在深层才收敛），则该 token 是 deep-thinking token

5. DTR = deep-thinking token 的比例：
$$\text{DTR}(S) = \frac{1}{T}\sum_{t=1}^T \mathbf{1}[c_t \in \mathcal{L}_{\text{deep-thinking}}]$$

超参推荐：$g = 0.5$，$\rho = 0.85$（最后 15% 的层为 "深层"）

---

## 实验结果：DTR vs. 所有 baseline

模型：GPT-OSS-20B/120B（low/mid/high reasoning），DeepSeek-R1-70B，Qwen3-30B-Thinking  
Benchmark：AIME 24/25，HMMT 25，GPQA-Diamond（8 × 4 = 32 组设置）

**Pearson 相关系数（越高越好）**：

| 指标 | 平均 r |
|------|--------|
| Token count | **-0.59**（负相关！） |
| Reverse token count | +0.59（统计镜像，无机制意义） |
| Log probability | ~+0.3 |
| Negative entropy | ~+0.4 |
| Self-Certainty | ~+0.6 |
| **DTR** | **+0.683**（最高，最稳定） |

**关键发现**：
- Token count 在 30/32 组设置中都是负相关——越长越错是普遍现象，不是特例
- Self-Certainty 表现不错但不稳定（有些设置变成负相关）
- DTR 在 32 组中只有 2 组出现橙色（负值），是最稳定的指标

---

## Think@N：用 DTR 做高效 Best-of-N

### 思想

从 N 个候选中，选 DTR 最高的 50%，然后做 majority voting。

关键 trick：**只用前 50 个 token 的 prefix 估计 DTR** —— 不用等生成完成，就能做 early rejection。

### 算法
```
1. 并行生成 N=48 个候选
2. 对每个候选的前 50 token 计算 DTR
3. 选 DTR 最高的 50%（即 24 个）
4. 用这 24 个做 majority voting（Cons@N）
5. 剩余的 50% 在 prefix 后 early stop
```

### 实验结果（GPT-OSS-120B-medium vs Cons@N=48）

| 方法 | AIME25 Acc | 推理 Cost | Cost 节省 |
|------|-----------|---------|---------|
| Cons@N（baseline） | 92.7% | 307.6k tokens | — |
| Short@N | 87.3% | 255.7k | -17% |
| Self-Certainty@N | 87.3% | 150.6k | **-51%** |
| **Think@N** | **94.7%** | **155.4k** | **-49%** |

- Think@N 比 Cons@N **准确率更高**（+2%），同时成本减半
- Self-Certainty@N 节省成本相近，但准确率低 7.4%
- Short@N 更差——进一步证明长度是坏指标

**最惊人的发现**：仅用 50 token prefix 的 DTR，比使用完整序列的 DTR 效果还好（Table 3）。
- Think@N(prefix=50) = 94.7%
- Think@N(prefix=all) = 94.0%
- 说明：推理的"深度"在早期就已经决定了

---

## 我的评价

**★★★★★（机制优雅，实用性强，值得深挖）**

### 为什么令我兴奋

1. **内部机制而非表面统计**：DTR 来自模型的 layer-wise 信息流，不是 token 数或 confidence score 这类外部统计量。这和 mechanistic interpretability 的思路一脉相承。

2. **50 token prefix = 完整序列**：这个结果非常 counterintuitive——推理质量在开头 50 个 token 就已经被 DTR 预测了。这意味着模型在推理开始时的"思考深度"已经决定了最终结果。这是关于 LLM 推理的一个深刻发现。

3. **零训练开销**：只需要访问中间层的 hidden states（大多数框架支持），不需要任何微调或额外模型。

4. **Pareto 最优**：在 accuracy-cost 权衡上，Think@N 严格优于所有 baseline。

### 我的质疑

1. **访问中间层的工程代价**：需要存储和处理所有 $L$ 层的 hidden states，对内存压力很大。在 serving 场景下实际可行吗？

2. **g 和 ρ 的 task-specific 敏感性**：论文的最优超参 (g=0.5, ρ=0.85) 是在数学 benchmark 上找的。在代码生成或通用 QA 上会不会不同？

3. **只在 math/science benchmark 验证**：这些 benchmark 有明确的对/错答案，而且 long-form reasoning 特别明显。在 open-ended task 上，DTR 还有效吗？

4. **Mechanistic 理解缺失**：论文确认了 DTR 与正确性的相关性，但没有解释**为什么** deep-thinking tokens 会在开头 50 token 就体现出来。这是最有趣的开放问题。

### 与 DTR（之前版本）的关系

我之前写过一篇 DTR 相关的笔记（AI/LLM/Inference/Deep-Thinking-Ratio-DTR.md），那是基于更早的版本。这篇论文（2602.13517）是更完整的版本，加入了：
- 完整的 Think@N 算法
- 与更多 baseline 的对比（Self-Certainty 等）
- 50 token prefix 的关键发现
- 多模型、多 benchmark 的系统验证

### 大局判断

DTR 解决的是 **"怎么知道哪个推理轨迹是好的？"** 这个问题，而不需要 verifier 或 outcome reward。

结合当前的 TTC scaling 趋势：
- **Best-of-N + Think@N** = 用 DTR 提前淘汰坏轨迹，只对好轨迹做 full decode
- **Goldilocks RL + Think@N** = 训练时选对难度的题，推理时选 DTR 高的轨迹
- 两者组合，训练和推理都在最大化"真正有效的计算"

这个方向（**内部信号驱动的高效 TTC**）值得持续追踪。

---

## 连接

- 直接相关：[[Deep-Thinking-Ratio-DTR]]（旧笔记，本篇是完整版）
- TTC 框架：[[Gemini-3-Deep-Think]]（ARC-AGI-2 84.6%，TTC 大胜利）
- 样本选择：[[Goldilocks-RL-Task-Difficulty-Curriculum]]（训练时 curriculum，推理时 Think@N，思路互补）
- Interpretability 基础：DoLA (Chuang et al. 2023) — 分层对比 logit 的先驱工作

---

*写于 2026-02-20 | Scholar*
