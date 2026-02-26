---
brief: "Scaling Laws——Kaplan/Chinchilla Scaling Laws 的核心公式：Loss ∝ N^α × D^β；Chinchilla 最优：compute budget 的 50% 给模型，50% 给数据；Test-Time Compute Scaling（o1/R1）作为新维度的延伸；LLM 路线图判断的理论基础。"
title: "Scaling Laws：从 Chinchilla 到 Inference-time Compute Scaling"
date: 2026-02-13
tags:
  - ai/foundations
  - ai/llm/training
  - ai/scaling
  - type/deep-dive
  - interview/hot
status: active
---

# Scaling Laws

> Scaling Laws 是 LLM 时代的"摩尔定律"——它揭示了模型性能如何随参数量、数据量和计算量可预测地提升。从 Kaplan 到 Chinchilla 再到 inference-time scaling，这条主线贯穿了 GPT-3 → o1 → DeepSeek-R1 的整个发展史。

## 1. 基本概念：幂律关系

LLM 的 cross-entropy loss $L$ 与三个变量呈幂律关系：

$$L(N, D, C) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

- $N$ = 模型参数量（non-embedding）
- $D$ = 训练数据量（tokens）
- $C$ = 计算预算（FLOPs），近似 $C \approx 6ND$
- $L_\infty$ = 不可约 loss（数据本身的熵）

**核心洞察**：在 log-log 图上，loss 随 N、D、C 近似线性下降，可以用小实验预测大模型的性能。

## 2. Kaplan Scaling Laws (OpenAI, 2020)

OpenAI 最早系统研究，关键结论：

1. **幂律指数**：$\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$ → 参数量对 loss 的边际收益更大
2. **最优分配**：给定计算预算 C，应该把更多预算给参数量 N（大模型 + 少数据）
3. **具体比例**：参数量增加 8× 时，数据量只需增加 ~5×

```
Kaplan 的建议：
C ↑ 10× → N ↑ 5.5× , D ↑ 1.8×
即：大力出奇迹，模型越大越好
```

这直接催生了 GPT-3 (175B) 的训练策略：巨大模型 + 相对较少的数据 (300B tokens)。

## 3. Chinchilla Scaling Laws (DeepMind, 2022)

Hoffmann et al. 修正了 Kaplan 的结论：

### 3.1 核心发现

**Kaplan 低估了数据的重要性。** 最优分配应该是参数量和数据量 **等比增长**：

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

具体比例：**每个参数大约需要 20 个训练 token**。

$$D^* \approx 20 \times N$$

### 3.2 Chinchilla 实验

| 模型 | 参数量 | 训练 Tokens | Tokens/Param | 结论 |
|------|--------|------------|-------------|------|
| **Gopher** | 280B | 300B | 1.1 | 严重欠训练 ❌ |
| **Chinchilla** | 70B | 1.4T | 20 | 计算最优 ✅ |

Chinchilla (70B) 用 1/4 的参数和 4× 的数据，**在几乎所有基准上击败了 Gopher (280B)**。

### 3.3 公式形式

三种估计方法取平均：

$$L(N, D) = \frac{A}{N^{0.34}} + \frac{B}{D^{0.28}} + E$$

给定计算预算 C = 6ND，最优分配：

$$N_{\text{opt}} = G \cdot C^a, \quad D_{\text{opt}} = G' \cdot C^b \quad (a \approx b \approx 0.5)$$

## 4. 后 Chinchilla 时代：Over-training

现实中，**推理成本远大于训练成本**。更小的模型在推理阶段更便宜，因此工业界选择 **over-train**——用远超 Chinchilla 比例的数据训练较小模型：

| 模型 | 参数量 | 训练 Tokens | Tokens/Param | vs Chinchilla |
|------|--------|------------|-------------|---------------|
| Chinchilla | 70B | 1.4T | 20× | 1× (基线) |
| LLaMA-1 7B | 7B | 1T | 143× | 7× |
| LLaMA-2 7B | 7B | 2T | 286× | 14× |
| LLaMA-3 8B | 8B | 15T | 1875× | **94×** |
| Mistral 7B | 7B | 8T+ | 1143×+ | 57×+ |

**Over-training 的经济学**：

$$\text{Total Cost} = C_{\text{train}} + C_{\text{inference}} \times \text{Requests}$$

当预期推理请求量足够大时，多花训练成本换更小模型是划算的。LLaMA-3 8B 的训练 compute 相当于一个 70B Chinchilla-optimal 模型，但推理成本低 ~10×。

```python
# 估算 Chinchilla-optimal 配置
def chinchilla_optimal(compute_budget_flops):
    """给定 FLOPs 预算，返回最优 N 和 D"""
    # C ≈ 6ND
    # N_opt ≈ D_opt / 20
    # C ≈ 6 * (D/20) * D = 0.3 * D^2
    D_opt = (compute_budget_flops / 0.3) ** 0.5
    N_opt = D_opt / 20
    return int(N_opt), int(D_opt)

# 估算 inference-aware optimal
def inference_aware_optimal(compute_budget_flops, expected_inference_tokens):
    """考虑推理成本的最优配置"""
    # 较小模型 + 更多训练数据
    # 参考 Sardana & Frankle (2024) 的公式
    N_opt, D_chinchilla = chinchilla_optimal(compute_budget_flops)

    # 如果推理量很大，降低 N，增加 D
    inference_ratio = expected_inference_tokens / D_chinchilla
    if inference_ratio > 10:
        N_adjusted = N_opt * 0.5  # 缩小模型
        D_adjusted = compute_budget_flops / (6 * N_adjusted)  # 增加数据
        return int(N_adjusted), int(D_adjusted)
    return int(N_opt), int(D_chinchilla)
```

## 5. Inference-time Compute Scaling

2024 年最重要的新范式：**不只在训练时 scale，也在推理时 scale**。

### 5.1 核心思想

传统 scaling：更多的训练 FLOPs → 更好的模型
Inference-time scaling：更多的推理 FLOPs → 更好的回答

$$\text{Quality}(x) = f(\text{train compute}) + g(\text{test-time compute}(x))$$

### 5.2 实现方式

| 方法 | 机制 | 代表 |
|------|------|------|
| **Chain-of-Thought** | 更长的推理链 = 更多 token | CoT Prompting |
| **Best-of-N Sampling** | 生成 N 个答案，选最好的 | 需要 verifier |
| **Tree Search** | MCTS / Beam Search 在推理空间搜索 | AlphaProof |
| **长思考链 (Long CoT)** | 训练模型使用更多 token 做深度推理 | **o1, o3, DeepSeek-R1** |
| **Self-Refinement** | 模型迭代改进自己的回答 | Self-Refine |

### 5.3 o1 与 DeepSeek-R1 的 Test-time Compute

**OpenAI o1 系列**：
- 通过 RL 训练模型使用 "thinking tokens"——在输出答案前进行大量内部推理
- 推理时间越长（thinking tokens 越多），答案质量越高
- 在 AIME 2024 数学竞赛中，o1 的分数随 test-time compute 近似 log-linear 增长

**DeepSeek-R1**：
- 开源证明了 inference-time scaling 的可行性
- 使用 [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO]]（Group Relative Policy Optimization）训练
- 关键发现：**纯 RL 训练（不需要 SFT 冷启动）也能涌现出 long CoT 能力**
- R1-Zero 展示了自发的思考行为：回溯、验证、自我纠错

```
Inference-time Scaling Law (实证):

Score(task) ∝ log(test_time_compute)

其中 test_time_compute ≈ thinking_tokens × model_FLOPs_per_token

这意味着：
- 增加 10× 推理计算 → 分数提升约 Δ
- 再增加 10× → 再提升约 Δ（对数关系）
```

### 5.4 两种 Scaling 的互补

```
                    ┌──────────────────────────────────┐
                    │        Performance               │
                    │            │                      │
                    │   Train    │    Inference         │
                    │   Scaling  │    Scaling           │
                    │     ╱      │      ╱               │
                    │    ╱       │     ╱  (o1, R1)      │
                    │   ╱        │    ╱                 │
                    │  ╱         │   ╱                  │
                    │ ╱          │  ╱                   │
                    │╱           │ ╱                    │
                    ├────────────┼──────────────────────│
                    │  Compute   │  Compute             │
                    │ (Training) │  (Inference)         │
                    └──────────────────────────────────┘

一个关键 insight: 当 training scaling 遇到瓶颈（数据墙）,
inference-time scaling 提供了新的提升路径。
```

## 6. 实践意义

### 6.1 训练决策

1. **预算分配**：用 Chinchilla 公式估算 N 和 D 的起点，根据推理需求调整
2. **小实验预测**：在 1/1000 规模上训练，用 scaling law 外推大模型性能
3. **何时停训**：loss 曲线与 scaling law 预测对比，判断是否需要更多数据

### 6.2 产品决策

- **计算密集型任务**（数学、代码）：优先 inference-time scaling（用 o1 类模型）
- **延迟敏感任务**（聊天、补全）：用 over-trained 小模型
- **混合策略**：Router 根据问题难度动态分配 inference-time compute

## 7. 面试常考题

### Q1: Chinchilla Scaling Law 的核心发现是什么？它如何改变了 LLM 训练策略？
**答**：Chinchilla 发现最优训练策略是参数量和数据量等比增长，每个参数约需 20 个 token。这修正了 Kaplan 的结论（偏重大模型少数据）。直接影响：GPT-3 (175B, 300B tokens) 是严重欠训练的；Chinchilla (70B, 1.4T tokens) 用 1/4 参数击败了 Gopher (280B)。后续 LLaMA 系列更进一步，用远超 20× 的 token/param 比例 over-train 小模型以降低推理成本。

### Q2: 为什么 LLaMA-3 8B 用了 1875 tokens/param，远超 Chinchilla 比例？
**答**：Chinchilla-optimal 只考虑训练成本。但实际部署中，推理成本 = 模型大小 × 请求量，通常远超训练成本。Over-training 小模型（多花训练 FLOPs，换来更小的推理模型）在总 TCO 上更优。LLaMA-3 8B 的训练计算量相当于一个 ~70B Chinchilla-optimal 模型，但推理成本低约 10 倍，几亿次请求后就收回了"多训练"的成本。

### Q3: Inference-time Compute Scaling 是什么？有哪些实现方式？
**答**：通过在推理阶段投入更多计算来提升输出质量。实现方式：(1) Best-of-N Sampling——生成 N 个候选选最好的；(2) Tree Search——在解空间做 MCTS/Beam Search；(3) Long CoT (o1/R1 方式)——训练模型使用更多 thinking tokens 做深度推理，推理质量随 thinking tokens 近似 log-linear 增长。后者是目前最成功的范式。

### Q4: DeepSeek-R1 的训练有什么独特之处？
**答**：关键创新：(1) 使用 GRPO 而非 PPO，不需要 value model，显存更省；(2) R1-Zero 证明纯 RL（不需要 SFT 冷启动）也能涌现 long CoT 能力——模型自发学会回溯、验证、自我纠错等推理行为；(3) 蒸馏——将 R1 的 long CoT 能力蒸馏到 Qwen/LLaMA 小模型，7B 模型也能具备一定的深度推理能力。

### Q5: 如何用 Scaling Law 来做训练预算规划？
**答**：实操步骤：(1) 在小规模（如 1B 以下）训练多个 (N, D) 组合，拟合幂律曲线；(2) 外推到目标 compute 预算，估算最优 N 和 D；(3) 如果预期推理量大（>10× 训练 tokens），偏向 over-train——缩小 N、增大 D；(4) 训练时实时对比 loss 曲线与预测值，如果偏离明显可能是数据质量问题或架构问题；(5) 用 Emergent Abilities 论文中的关键 benchmark 作为验证点。注意 scaling law 预测的是 loss，不直接预测下游任务表现。
