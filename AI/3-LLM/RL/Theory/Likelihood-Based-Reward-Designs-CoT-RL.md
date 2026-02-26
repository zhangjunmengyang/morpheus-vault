---
brief: "Likelihood-Based Reward Design——用模型自身的 log-probability 作为 CoT 推理的奖励信号，无需 verifier；实验显示在开放域推理上优于 outcome-only reward，是 non-verifiable 任务的奖励设计新方向。"
title: "Likelihood-Based Reward Designs for General LLM Reasoning"
date: 2026-02-22
tags:
  - ai/llm/rl
  - reward-design
  - log-probability
  - cot
  - verifier-free
  - non-verifiable
  - meta-fair
domain: ai/llm/rl/theory
arxiv: "2602.03979"
rating: ★★★★☆
status: active
---

# Likelihood-Based Reward Designs：Log-Prob 是通用 CoT 训练的答案

**arXiv**: 2602.03979  
**机构**: Meta FAIR + University of Amsterdam + NYU  
**作者**: Ariel Kwiatkowski 等（Joint senior authors）  
**提交**: 2026-02-03  
**评分**: ★★★★☆  
**一句话**: 系统比较六种概率/log-概率衍生的 reward 设计，发现**log-prob reward**是唯一在可验证（数学）和不可验证（长文本）场景下都表现良好的方法，消除了"每个任务都要设计专用 reward"的需求。

## See Also

- [[AI/3-LLM/RL/Theory/Rationale-Consistency-GenRM-Deceptive-Alignment|Rationale Consistency]] — reward model 的推理质量问题；本文是 reward 信号的密度/通用性问题——两者都在问"如何让 reward signal 更好"，角度互补（RC=判断质量，本文=信号设计）
- [[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR Edge of Competence]] — RLVR 的能力边界理论；本文的 non-verifiable 分析与该理论直接对话：边界附近的问题恰好是 binary reward 最稀疏的区域
- [[AI/3-LLM/RL/Other-Algorithms/PACED-RL-Partition-Function-Difficulty-Scheduler|PACED-RL]] — GFlowNet + difficulty scheduler；本文的 log-prob reward 与 partition function 有数学联系（JEPO 的 log-mean-exp 结构与 GFlowNet 目标函数同构）
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO全景]] — 本文使用 RLOO（GRPO 的无偏版本）作为 optimizer；log-prob reward 是对 GRPO 七维框架中 Token 维度问题的新解法
- [[AI/3-LLM/RL/Theory/RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]] — 同样针对 non-verifiable 场景：RLRR 用高质量 reference + RefEval judge 造软 verifier；本文用 log-prob 完全绕过 verifier——两者是 non-verifiable reward 问题的两条路径（RLRR=构造verifier，本文=绕过verifier）

---

## 核心问题

标准 RL 训练 LLM 的范式：采样 CoT + 检查答案对错 = 0/1 binary reward。

这有两个根本限制：
1. **每个任务需要一个专用 verifier**（数学用 sympy，代码用 unit tests，开放问题？没有）
2. **binary reward 极其稀疏**：早期训练中模型几乎不输出正确答案，梯度信号接近零

**本文的 insight**：能不能用"模型给参考答案赋予多高的概率"来作为 reward？这样：
- 不需要 verifier（只需要参考答案）
- 提供连续的密集信号（不是 0/1，是连续值）
- 与预训练的 log-likelihood 目标保持一致

---

## 六种 Reward 设计的系统比较

### 方法定义

设 `p` = 问题，`z` = 生成的 CoT，`a` = 生成的答案，`a*` = 参考答案，`π_θ` = 模型

| 方法 | 公式 | 关键特征 |
|------|------|---------|
| **SFT** | cross-entropy on `a*` | 无 CoT，直接监督 |
| **Base RL** | `R = I[a = a*]` | 0/1 binary，需要采样 |
| **Probability (VeriFree)** | `R = π_θ(a* \| p,z)` | 连续 [0,1]，无需采样 |
| **AvgProb (RLPR)** | `R = (1/\|a*\|) Σ_t π_θ(a_t*\|...)` | per-token 平均概率 |
| **Log-prob** | `R = log π_θ(a* \| p,z)` | **主角**，对数概率 |
| **AvgLogprob** | `R = (1/\|a*\|) log π_θ(a* \| p,z)` | per-token 平均 log 概率 |
| **JEPO** | `R = log (1/G) Σ_i π_θ(a*\|p,z_i)` | log-mean-exp 组级奖励 |

### 关键数学区别：Probability vs Log-Probability

当 `π_θ(a* | p,z)` 极小时（模型认为参考答案很不可能）：
- **Probability (VeriFree)**：reward ≈ 0，梯度信号消失，**学习停滞**
- **Log-prob**：reward = log(极小值) = 很大的负数，但**梯度仍然存在**，可以学习

这就是为什么在 non-verifiable 的长文本域，VeriFree 崩溃而 log-prob 还能工作。

### JEPO 的设计动机

VeriFree 的期望 `E_z[π(a*|p,z)]` 是 `log π^CoT(a*|p)` 的低估（因为 log 是凹函数，Jensen 不等式）。JEPO 修正了这个低估：

```
R_JEPO = log (1/G) Σ_i π_θ(a* | p, z_i)
       = log 均值（G 个采样）
```

vs log-prob：

```
R_logprob = log π_θ(a* | p, z)
           = 单个采样的 log 概率
```

JEPO 理论上更精确，但实现复杂度更高（G 个样本的奖励无法独立计算）。

---

## 实验结果

### 测试设置

- **模型**：Llama-3.2-3B-Instruct + Qwen-2.5-3B-Instruct
- **可验证域**：MATH（~7k 问题）+ DeepScaleR（~39k 问题）
- **不可验证域**：Alpaca（~50k 长文本）+ NuminaProof（~50k 证明题）
- **Optimizer**：RLOO（GRPO 的无偏版本），G=4 / G=32

### 主要结论

**结论 1：在可验证域，log-prob 和 binary RL 接近，但 log-prob 的 perplexity 更好**

```
在 MATH (G=32)：
  Greedy success rate：所有 RL 变体相似
  Log-prob reward：perplexity 明显优于 Base RL（更接近 SFT 水平）
  VeriFree：greedy success 相似，但 perplexity 介于两者之间
```

这意味着：log-prob reward 在提高正确率的同时，还在维持模型的语言质量。Base RL 则往往会破坏 perplexity（reward hacking 的表现）。

**结论 2：在不可验证域，只有 log-prob 能 work**

```
Non-verifiable 域（Alpaca, NuminaProof）：
  Base RL：崩溃（采样答案与参考答案不匹配，reward = 0）
  VeriFree：崩溃（长文本的精确匹配概率 ≈ 0，学习停滞）
  Log-prob：与 SFT 相当（是的，不比 SFT 强，但也不崩溃）
```

**结论 3：log-prob reward 会初始缩短 CoT，之后行为分化**

```
可验证域：CoT 先缩短，之后恢复到正常长度
不可验证域：CoT 缩短后不恢复，基本退化为 SFT 行为
```

这是个有趣的现象，论文提出了假设：在 non-verifiable 域，模型发现"减少 CoT = 减少不确定性 = 提高 log-prob"，这是一种形式的 shortcut learning。缓解策略（length penalty、KL divergence 约束）都会恢复 CoT 长度，但会损害性能。

**结论 4：计算优势**

log-prob reward 不需要采样答案 `a`，只需要一次 forward pass 在 `a*` 上。这在 G=32 时有明显的训练速度优势。

---

## 我的分析

### 这篇论文的真正贡献是什么

**贡献是澄清（clarification），不是发明（invention）**。

VeriFree、JEPO、RLPR 这些工作都已经在用 likelihood-based rewards，但没有人系统地比较"为什么有的 work，有的不 work，边界条件在哪里"。

这篇论文的价值在于提供了一个**干净的实验矩阵**：2 个模型 × 4 个数据集 × 6 种方法 × 多种指标。结论清晰，不模糊。这种"系统清理"的工作对于领域有很高价值，虽然不如 novel 的方法论文那么 flashy。

### Log-Prob Reward 的理论意义

`R = log π_θ(a* | p, z)` 有一个非常干净的解释：

**它就是预训练目标的延续**。预训练优化 `Σ_t log π(x_t | x_{<t})`，相当于最大化所有 token 的 log-likelihood。Log-prob reward 在 RL fine-tuning 阶段继续优化"给定 CoT，参考答案的 log-likelihood"。

这样 RL 和预训练之间的目标是连续的，不是突然从"预测下一个 token"切换到"回答是否正确"。这解释了为什么 log-prob 的 perplexity 更好：它没有破坏预训练建立的语言模型基础。

### VeriFree 的失效模式是个重要教训

VeriFree 把 reward 设为 `π_θ(a* | p, z)`（不加 log）。

在数学/代码这类任务上，当 a* 很短（比如"42"），这个概率还可以；但对于长文本，`π_θ("整篇参考答案" | p, z)` 会因为乘法效应而趋近于零——即使模型每步的 token 概率都不低，但 100 个 token 相乘后也会指数级衰减。

Log 的作用是把乘法变加法：`log Π_t p_t = Σ_t log p_t`，这样长答案不会因为长度而有梯度消失问题。

这是 log-prob 在 non-verifiable 域成功的根本数学原因，不是什么神奇，就是数值稳定性。

### Non-Verifiable 域的悲观结论

在非可验证域，log-prob reward 最好的情况是"与 SFT 持平"——不比 SFT 好。

这其实是一个重要的 negative result：**RL fine-tuning 在非可验证域可能不比 SFT 更好**。这跟 RLVR Edge of Competence 的理论一致——verifiable reward 是 RLVR 真正有效的前提条件。

这个结论对于 LLM 研究实践有直接影响：
- 不要期望用 RL 提升 open-ended 生成任务（写作、摘要等）
- 如果一定要用 RL，最好找一个 verifiable 的代理任务
- 或者退回到 RLHF（用人类偏好作为 reward），但那是另一套 pipeline

### 与 Rationale Consistency（上轮）的对话

上轮（2602.04649）的问题：**reward 的推理质量（为什么这个 reward 是对的）**  
本轮（2602.03979）的问题：**reward 的类型选择（用什么 reward）**

两篇论文形成了对 reward design 的互补视角：

| 维度 | 2602.04649（RC） | 2602.03979（Log-prob）|
|------|-------------------|----------------------|
| 问的问题 | "reward 的推理过程对不对" | "reward 信号的类型选哪种" |
| 关注场景 | 有 human annotation 的设置 | verifiable/non-verifiable 通用 |
| 对 binary RL 的批评 | 推理过程可能是错的 | 梯度太稀疏，non-verifiable 崩溃 |
| 解决方案 | rationale supervision + hybrid reward | log-prob reward（密集、通用）|

两篇一起读，给出了一个更完整的 reward design 图景：reward 信号要既**密集**（log-prob 解决），又**正确**（rationale consistency 解决）。

### 局限

1. **模型规模小**：只测了 3B 参数模型，是否 scale 到更大模型是个问题
2. **Non-verifiable 结论偏悲观**：只测了 Alpaca 和数学证明，可能有其他 non-verifiable 场景表现更好
3. **CoT 缩短现象未完全解释**：论文提出了假设但没有 conclusive 的解释
4. **没有测 RLHF 下游**：只测了 benchmark accuracy，没有像 RC 论文那样做 Arena Hard v2

---

## 关键公式

**Log-prob Reward（主推方法）**：
```
R(z, a) = log π_θ(a* | p, z)

梯度分解（tang2025beyond 证明）：
∇J_θ = E_{p,z,a} [log π_θ(a*|p,z) · ∇log π_θ(z|p) + ∇log π_θ(a*|p,z)]
       = Reinforce项（用 log-prob 作 reward）+ SFT项（直接监督 a*）
```

这个梯度分解非常优雅：log-prob RL 在做的事情，**同时**是一个 Reinforce 算法（用 log-prob 作 reward）和一个 SFT（直接最大化参考答案的 log-likelihood）。它是两者的自然结合。

**VeriFree（失败案例的对比）**：
```
R_VeriFree(z, a) = π_θ(a* | p, z)    # 不加 log → 长文本趋近于 0 → 梯度消失
```

**结论对比**：
```
可验证域：  Log-prob ≥ Base RL（greedy success），且 perplexity 更好
不可验证域：Log-prob ≈ SFT（唯一不崩溃的），VeriFree/Base RL 均崩溃
CoT 长度：  Log-prob 初始缩短，verifiable 域恢复，non-verifiable 域不恢复
```

---

## Tags
#RewardDesign #LogProbability #VerifierFree #ChainOfThought #NonVerifiable #MetaFAIR #RLOO #CoTTraining #LikelihoodReward #RL后训练 #VeriFree #JEPO
