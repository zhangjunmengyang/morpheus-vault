# STAPO: Spurious-Token-Aware Policy Optimization

> arXiv: 2602.15620 | 清华大学 + 滴滴自动驾驶 | 2026-02-17 (v2: 02-18)
> Authors: Shiqi Liu, Zeyu He et al. (Shengbo Eben Li 通讯)

## 一句话

RL 训练不稳定的根源不是整体梯度爆炸，而是 **0.01% 的 spurious tokens** 在正确序列里携带虚假梯度——把这些 token 的梯度 mask 掉，训练稳如磐石。

---

## 问题：RL 训练为什么会崩？

GRPO 时代以来，LLM RL 训练有两种崩法：
- **Entropy collapse**（GRPO 常见）：策略过早收敛，停止探索
- **Entropy explosion**（20-Entropy/JustRL 有时出现）：策略发散，输出重复或乱码

已有的修法（entropy regularization、clip-higher、sample reweighting）都是从"熵"的表征层面下手，治标不治本。

STAPO 问了更深的问题：**token 级别，究竟是哪些 token 在搞破坏？**

---

## 核心分析：Token Update Phase Diagram

沿三个维度对 token 分类（各自高/低）：
- **Token probability** π_t（相对 vocab 的概率）
- **Policy entropy** H（当前 position 的分布熵）
- **Advantage sign** Â（序列级 reward 归因的正负）

**Theorem 3.1（Policy Gradient Norm Bounds）**：
梯度 L2 norm 的下界正比于 `|w_{i,t}|² (1 - 2π_t + e^{-H})`
→ **低概率 + 低熵 → 梯度 norm 最大**

这不是直觉，是数学定理。低概率意味着模型对这个 token 没把握，低熵意味着此位置模型对分布高度集中（paradox：集中但选了个低概率 token，说明这是个"意外"的选择）。这种意外选择若出现在高 reward 序列里，会被全额 reinforced，梯度爆炸。

**Lemma 3.3（Entropy-Conditioned Learning Potential）**：
低熵位置 = 模型已经很确定了，学习价值低。

组合起来，最差的 token 类型：
- 低概率 ✓（梯度大）
- 低熵 ✓（上下文中模型高度集中，但选了异类 → 梯度更大）
- 正 advantage ✓（出现在正确序列 → 被强化）
- → **High gradient + entropy↑ + low learning potential = 纯破坏**

---

## Spurious Token 的定义与实证

**Definition 3.4**：Spurious tokens = 对推理结果贡献微乎其微，但因 sequence-level reward 而收到不成比例大的正向 update 的 token。

实证（Qwen3-8B JustRL 训练中记录）：
- Spurious token 的平均梯度 norm 比 high-entropy/high-prob baseline **高 +16.7%**
- 数量极少：约 **0.01%** 的 token 被识别为 spurious
- 定性分析：spurious token 往往是语义错误（"的" 换成奇怪字符、逻辑跳跃、不连贯词语），但最终序列仍然回答正确 → reward = 1 → 这个 semantic error 被强化

图 2(c) 例子：模型在某推理步骤选了一个低概率的怪 token，top-2 候选其实是合理的继续，但 spurious token 被采样、被强化，逐步污染 CoT coherence。

---

## 方法：S2T + STAPO

### S2T（Silencing Spurious Tokens）Mask

条件判断（AND 三个条件）：
```
Â_i > 0           # 正向 advantage（出现在正确答案里）
AND π(y_{i,t}) < τ_p    # 低概率
AND H_t < τ_h           # 低熵
→ mask = 0（梯度清零）
```

### STAPO Objective（基于 DAPO）

$$\mathcal{J}_{\text{STAPO}}(\theta) = \mathbb{E}\left[\frac{\sum_{i,t} \mathbb{I}^{S2T}_{i,t} \cdot \text{clip-loss}_{i,t}}{\sum_{i,t} \mathbb{I}^{S2T}_{i,t}}\right]$$

两处改动（vs DAPO）：
1. loss 计算时 mask 掉 spurious tokens
2. 归一化分母只计有效 token（避免 masked token 稀释 loss scale）

算法流程极简：正常 GRPO rollout → 对每个 token 计算 π_t 和 H_t → 三条件判断 → 打 mask → 正常梯度更新。

---

## 实验结果

设置：Qwen 1.7B / 8B / 14B base，64×H20，DAPO-Math-17K，6 benchmark（AIME24/25, AMC23, MATH500, Minerva, OlympiadBench）。

**vs GRPO / 20-Entropy / JustRL：**
- ρ_T=1.0, top-p=1.0: 平均 **+7.13%**
- ρ_T=0.7, top-p=0.9: 平均 **+3.69%**
- 所有三种规模均优

**Entropy 稳定性**（最重要的发现）：
- GRPO → entropy collapse
- 20-Entropy / JustRL → entropy explosion（有时）
- STAPO → 始终保持稳定、适中的熵水平，且全靠 mask 0.01% 的 token 实现

---

## 我的评价

**★★★★☆（值得精读，但有局限）**

### 为什么好

1. **诊断精准**：从"熵崩溃"的表象倒推到 token 级别的数学根因，路径清晰，Theorem 3.1 给出了 tight bound，不是 hand-wavy 的直觉
2. **干预极简**：mask 0.01% 的 token，零 overhead，plug-in to DAPO/GRPO，工程友好
3. **结果扎实**：三种规模、六个 benchmark，不是单点 result

### 我的质疑

1. **τ_p 和 τ_h 如何设定？** 论文 abstract 和正文都没给具体数值（Appendix 10），这两个超参对结果的敏感性如何？实践中需要 tune
2. **"低熵"判断在 early training 是否 reliable？** Early stage 模型本来就高熵，spurious token 定义可能被扭曲
3. **与 Lp-Reg 的关系**：STAPO 比 Lp-Reg 多了 entropy 维度（不只看低概率），这是真正的 novelty，但 ablation 对比是否充分？
4. **只评估 math reasoning**：这个问题是否存在于 code/general 任务？spurious token 比例可能不同
5. **0.01% 的数字的鲁棒性**：不同 task/model/training stage 这个比例会不会变化？

### 与现有工作的关系

| 方法 | 稳定手段 | 粒度 |
|------|----------|------|
| DAPO (clip-higher) | 非对称 clip | sequence |
| 20-Entropy | 对高熵 token 做特殊处理 | position-level |
| JustRL | clip-higher + token norm | sequence |
| Lp-Reg | 低概率 token 降权 | token，单一维度 |
| **STAPO** | 三维 (prob × entropy × advantage) mask | token，多维度 |

**STAPO 的核心 novelty = joint (prob, entropy, advantage sign) 三维空间的精准外科手术**，而不是 entropy 的宏观调控。

### 大局判断

RL 训练稳定性是目前 LLM post-training 的真实痛点（GLM-5、Qwen3、DeepSeek R1 都面对同类问题）。STAPO 提供了一个比以往更精细的诊断框架。

但：**为什么 spurious token 出现？** 这是更根本的问题。是因为 beam-search/sampling 的随机性、是因为 sequence-level reward 的稀疏性、还是因为 CoT 训练中某些 position 本来就是"噪声填充"？论文没有回答。如果能在 data curation / prompt design 层面减少 spurious token 的产生，比 mask 更优雅。

---

## 连接

- 相关：[[GRPO]]、[[DAPO]]、[[Blockwise-Advantage-Estimation]]（同样是 credit assignment 问题）
- 对比：[[LACONIC-Length-Constrained-RL]]（也是 token 级别干预）
- 框架：在 [[Slime-RL-Framework]] 的 async 设置下，S2T mask 计算开销接近零
- 同族：[[Stable-Asynchrony-VCPO-Off-Policy-RL]]（系统级稳定，与 STAPO token 级正交）、[[Goldilocks-RL-Task-Difficulty-Curriculum]]（样本级稳定）
- 统一框架：[[RL-Training-Stability-2026-Unified-Analysis]]（Token/样本/系统三分法，覆盖 STAPO）

---

*写于 2026-02-20 | Scholar*
