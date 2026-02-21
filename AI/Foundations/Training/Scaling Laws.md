---
title: "Scaling Laws"
date: 2026-02-14
tags:
  - training
  - scaling-laws
  - chinchilla
  - interview
type: note
---

# Scaling Laws

## 1. Kaplan Scaling Laws（2020）

### 原始论文

OpenAI, *Scaling Laws for Neural Language Models*（Kaplan et al., 2020）

### 核心结论

模型的 **test loss** 与三个因素呈**幂律关系**（power law），且彼此近似独立：

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

其中 $N$ = 模型参数量，$D$ = 数据量（tokens），$C$ = 计算量（FLOPs）。

### 关键发现

1. **模型规模最重要**：固定计算预算时，应优先增大模型（$N$），即使训练不充分（few epochs）也比小模型训练更久好
2. **最优分配**：约 $N \propto C^{0.73}$，大部分算力应分配给更大的模型
3. **数据次要**：他们认为数据量的指数 $\alpha_D$ 较小，增加数据的收益不如增加模型大
4. **架构不敏感**：在控制参数量后，不同 Transformer 变体的 scaling 行为相似

### 实际影响

这直接导致了 GPT-3 的策略：**造一个巨大的模型，用相对少的数据训练相对少的步数**。GPT-3 175B 只训练了约 300B tokens。

## 2. Chinchilla Scaling Laws（Hoffmann 2022）

### 论文

DeepMind, *Training Compute-Optimal Large Language Models*（Hoffmann et al., 2022）

### 核心修正

Kaplan 的结论有偏差。Chinchilla 通过更严格的实验发现：

$$\text{给定计算预算 } C, \text{最优分配为 } N \propto C^{0.5}, \quad D \propto C^{0.5}$$

即：**模型参数量和训练数据量应该同比例扩展**。具体比例约为：

> **最优 tokens ≈ 20 × 参数量**

### 关键结论

| 对比 | Kaplan | Chinchilla |
|------|--------|------------|
| 最优策略 | 大模型 + 少数据 | 模型和数据同步扩展 |
| 数据:参数比 | 不强调 | ≈ 20:1 |
| GPT-3 评价 | 遵循其建议 | **严重欠训练**（175B 参数仅 300B tokens，应 3.5T） |

### Chinchilla 的验证

用 70B 参数 + 1.4T tokens 训练的 Chinchilla，**性能全面超越** 280B 参数的 Gopher —— 用 1/4 的参数量，关键是数据给够了。

## 3. 对预训练策略的实际影响

### 为什么 LLaMA 选择 Over-train

LLaMA（Meta, 2023）明确违反了 Chinchilla 最优比：

| 模型 | 参数量 | 训练 tokens | Chinchilla 最优 | 实际比例 |
|------|--------|------------|----------------|---------|
| LLaMA-7B | 7B | 1T | 140B | 7× over-train |
| LLaMA-13B | 13B | 1T | 260B | 4× over-train |
| LLaMA-65B | 65B | 1.4T | 1.3T | ≈ 1× |

**为什么故意 over-train？**

1. **推理成本主导**：Chinchilla 最优是针对**训练成本**的。但实际部署中，模型要服务数百万请求，**推理成本远超训练成本**
2. **小模型 + 多数据 > 大模型 + 少数据**：7B 模型推理成本远低于 70B，如果通过 over-train 让 7B 接近 13B 的性能，总 TCO 更优
3. **Inference-Aware Scaling**：给定总预算（训练 + 推理），最优策略往往是训练一个**比 Chinchilla 最优更小、但训练更久**的模型

> **LLaMA 的启示**：Chinchilla 最优是训练效率最优，不是部署效率最优。

### LLaMA-3 的极端 Over-train

LLaMA-3-8B 训练了 **15T tokens**（Chinchilla 最优约 160B），over-train 接近 **100×**。结果是 8B 模型性能媲美 LLaMA-2-70B —— 充分验证了 over-train 策略在推理经济学下的合理性。

## 4. 后 Chinchilla 时代

### 4.1 Inference-Time Compute Scaling

传统 scaling laws 关注**训练时**的计算分配。2024-2025 年的新范式：

**在推理时投入更多计算来提升性能。**

核心方法：

| 方法 | 代表 | 思路 |
|------|------|------|
| **Chain-of-Thought** | Wei et al. 2022 | 让模型"思考"更多步 |
| **Self-Consistency** | Wang et al. 2023 | 多次采样取多数票 |
| **Tree-of-Thought** | Yao et al. 2023 | 搜索式推理 |
| **Verifier / ORM** | Cobbe et al. 2021 | 生成多个答案，用验证器选最优 |

### 4.2 Test-Time Scaling（o1 范式）

OpenAI o1 / o3 系列开创了 **test-time compute scaling**：

- 模型在推理时通过**内部 chain-of-thought**进行长时间"思考"
- 计算量可以从几秒到几分钟动态调整
- 呈现出类似训练时的 **scaling law**：推理计算 ↑ → 性能 ↑（幂律关系）

**关键洞察**：

$$L(\text{test-time compute}) \propto C_{\text{inference}}^{-\alpha}$$

这意味着 scaling 的维度不再局限于 N、D、C_train，还包括 $C_{\text{inference}}$ —— 给模型更多"思考时间"等价于更大的模型。

### 4.3 超越单一 Scaling 维度

当前理解的 scaling 维度全景：

```
Training-time:   N (参数) × D (数据) × C_train (算力)
                              ↓
Post-training:   RLHF / DPO 对齐 × SFT 数据质量
                              ↓
Inference-time:  C_inference (思考时间) × search (搜索策略) × tools (工具调用)
```

## 5. 与 nanochat 的联系

### $43K → $72 的成本降幅中 Scaling Laws 的角色

nanochat 项目复现了 GPT-2 级别模型，训练成本从 2019 年的约 $43K 降到 2024 年的 $72。Scaling Laws 在这个降幅中扮演了关键角色：

#### 硬件层面（≈ 100×）
- GPU 算力：V100 → H100 ≈ 10-15× FLOPs 提升
- 价格/性能比改善 + 云计算竞争

#### 算法层面（≈ 5-10×，Scaling Laws 直接相关）
- **正确的模型-数据配比**：按 Chinchilla 法则分配，同样算力下 loss 更低
- **不需要重复实验调参**：Scaling Laws 提供了可预测的训练行为，减少了试错成本
- **架构选择的理性化**：知道 scaling 行为后，可以提前预测大模型性能，避免无效尝试

#### 工程层面（≈ 5-10×）
- FlashAttention：内存效率提升 5-20×
- 混合精度训练（BF16）：计算效率 2×
- 编译优化（torch.compile）：额外 10-30% 加速
- 更高效的数据 pipeline

#### Scaling Laws 的特殊价值

> Scaling Laws 最大的贡献不是让训练更快，而是让训练**可预测** —— 你可以在小规模实验中预测大规模结果，避免了天文数字的试错成本。

这就是为什么 nanochat 能用 $72 复现：不需要从零探索，直接站在 scaling laws 指导下选择正确的配置。

## 6. 面试常见问题及回答要点

### Q1: 解释 Scaling Laws 的核心结论

**要点**：Loss 与 N/D/C 呈幂律关系。关键区分 Kaplan vs Chinchilla：Kaplan 认为应优先放大模型，Chinchilla 修正为模型和数据应同步扩展（约 20:1）。强调 Chinchilla 证明 GPT-3 是欠训练的。

### Q2: Chinchilla 最优 vs LLaMA 的 Over-train，谁对？

**要点**：两者回答不同问题。Chinchilla 最优化**训练效率**（固定算力下最低 loss），LLaMA 最优化**部署 TCO**（训练 + 推理总成本）。在推理成本主导的场景下，over-train 小模型是经济理性的选择。用 LLaMA-3-8B 的数据佐证：15T tokens over-train 100×，性能媲美 LLaMA-2-70B。

### Q3: Scaling Laws 能预测 emergent abilities 吗？

**要点**：这是有争议的。传统观点（Wei et al., 2022）认为存在能力涌现的"相变"，scaling laws 无法预测。但 Schaeffer et al.（2023）反驳说"涌现"可能是评估指标的不连续造成的假象 —— 如果用连续指标，性能提升是平滑的。面试中展现对两方观点的了解即可。

### Q4: 什么是 Test-Time Scaling？和传统 Scaling Laws 的关系？

**要点**：传统 Scaling Laws 关注训练时的 N/D/C 三个维度。Test-Time Scaling 新增了推理计算 $C_{\text{inference}}$ 维度 —— 给模型更多思考时间（如 o1 的内部 CoT）同样呈幂律提升。这意味着不一定要训练更大的模型，可以在推理时"按需加算力"。类比：训练是"学习"，test-time compute 是"考试时多想想"。

### Q5: 如果你要预训练一个模型，怎么利用 Scaling Laws？

**要点**：
1. 先用**小模型（10M-1B）**做一系列 scaling 实验，拟合幂律曲线
2. 根据曲线**外推**目标规模的预期 loss
3. 根据部署场景选择策略：
   - 如果推理量小 → Chinchilla 最优
   - 如果推理量大 → Over-train 更小的模型
4. 选择 learning rate schedule 时参考 scaling laws 对 lr 的建议（$\mu P$ 等）
5. **Scaling laws 是规划工具**，帮你在花大钱之前做出理性决策

### Q6: 为什么 Scaling Laws 对行业如此重要？

**要点**：因为大模型训练极其昂贵（GPT-4 估计 $100M+），不可能反复试错。Scaling Laws 使训练变得**可预测**：小实验 → 拟合曲线 → 预测大模型表现 → 理性投资决策。本质上是把 AI 训练从"炼丹"变成了"工程"。

---

## See Also

- [[AI/Foundations/Training/Training Loss 分析|Training Loss 分析]] — Scaling Law 中 loss 曲线的微观解读
- [[AI/LLM/Pretraining/LLM预训练与分布式训练2026全景|LLM 预训练 2026 全景]] — Scaling Law 在工程实践中的落地：Chinchilla 最优 vs Over-train
- [[AI/Foundations/Math/向量微积分|向量微积分]] — 幂律拟合的数学直觉
- [[AI/Foundations/_MOC|Foundations MOC]] — 训练基础全图谱
