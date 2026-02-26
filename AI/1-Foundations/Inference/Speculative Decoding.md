---
brief: "Speculative Decoding——用小草稿模型预测多个 token，大模型批量验证；理论加速比 = 接受率/(1-接受率×每步加速比)；Self-Speculative/Medusa/EAGLE 等变体；推理吞吐量 2-3x 提升的关键技术。"
title: "Speculative Decoding 投机采样"
date: 2026-02-14
tags:
  - inference
  - speculative-decoding
  - optimization
  - interview
type: note
---

> [!info] 📖 版本说明
> 本篇为**面试速查版**（简洁直接）。深度工程版：[[AI/3-LLM/Inference/Speculative Decoding|Speculative Decoding 工程深度版]]

# Speculative Decoding 投机采样

## 1 动机：自回归解码的串行瓶颈

LLM 推理的核心瓶颈不在计算，而在 **内存带宽（memory-bound）**：

- 自回归生成每步只产出 **1 个 token**，但需要加载模型的全部参数。
- 一个 70B 模型（FP16）约 140GB，即使在 8×A100（总带宽 ~16TB/s）上，每次 forward pass 的理论下限也在 ~9ms。
- GPU 的算力利用率极低——大量 FLOPS 空闲等待数据从 HBM 搬到计算单元。

**核心矛盾**：每步生成 1 个 token 的"工作量"太小，无法充分利用 GPU 并行计算能力，但自回归结构要求串行。

> **一句话**：Speculative Decoding 的本质是把"串行生成 → 并行验证"，用计算换时间。

---

## 2 核心思想

### 2.1 Draft-then-Verify 框架

1. **Draft 阶段**：用一个小而快的 **Draft Model** $M_q$ 自回归生成 $\gamma$ 个 candidate tokens：$x_1, x_2, \ldots, x_\gamma$
2. **Verify 阶段**：将这 $\gamma$ 个 token 连同原始 prefix **一次性** 送入大模型 $M_p$ 做 forward pass，并行计算每个位置的概率分布
3. **Accept/Reject**：从左到右逐个检查，接受与大模型分布一致的 token，在第一个 reject 位置停下并从大模型分布中重新采样

```
Draft Model (fast):    [t1] → [t2] → [t3] → [t4]     （串行，但很快）
                          ↓      ↓      ↓      ↓
Target Model (slow):   [ ----  并行验证 4 个 token  ---- ]  （1 次 forward pass）
                          ✓      ✓      ✗
Result:                 t1     t2     t3'               （接受2个 + 修正1个 = 3个token/step）
```

**关键洞察**：Transformer 的 forward pass 对 sequence length 几乎是并行的（KV Cache 下验证 $\gamma$ 个 token 的成本接近生成 1 个 token），所以"一次验证多个"几乎是免费的。

### 2.2 为什么 Draft Model 要小？

- Draft Model 的目的是 **快速猜测**，不需要与大模型完全一致
- 越小越快 → 更多 candidate tokens → 更大的"投机"幅度
- 即使猜错也没关系，有 rejection sampling 兜底

---

## 3 数学保证：Rejection Sampling

这是 Speculative Decoding 最优雅的部分：**无论 Draft Model 质量如何，最终输出分布与单独使用 Target Model 完全一致**。

### 3.1 接受-拒绝机制

对于 draft token $x$ 在位置 $t$：

- Draft Model 给出概率 $q(x)$，Target Model 给出概率 $p(x)$
- 以概率 $\min\left(1, \frac{p(x)}{q(x)}\right)$ 接受该 token
- 若拒绝，从修正分布中采样：$p'(x) = \text{norm}\left(\max(0, p(x) - q(x))\right)$

### 3.2 为什么这保证了分布一致性？

对于任意 token $x$，其最终被选中的概率为：

$$P(\text{accept } x) + P(\text{reject}) \cdot p'(x)$$

$$= q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_x q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)\right) \cdot p'(x) = p(x)$$

这是经典的 rejection sampling 理论，不依赖任何近似。

> **面试关键**：必须能说清"输出分布完全等价于 target model"——这是 Speculative Decoding 与其他近似加速方法的本质区别。

---

## 4 Acceptance Rate 与加速比

### 4.1 Acceptance Rate $\alpha$

$$\alpha = \mathbb{E}\left[\min\left(1, \frac{p(x)}{q(x)}\right)\right]$$

$\alpha$ 取决于 draft model 与 target model 分布的接近程度。

### 4.2 加速比分析

假设 draft model 生成 $\gamma$ 个 token 的时间可忽略，target model 一次 forward pass 的时间为 $T_p$：

- **无 Speculative Decoding**：每个 token 需要 $T_p$
- **有 Speculative Decoding**：每次 verify step 期望接受 $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ 个 token（含修正采样的那个），花费 $\approx T_p$

**典型加速比**：

| Draft Model 质量 | $\alpha$ | $\gamma$ | 加速比 |
|------------------|----------|----------|--------|
| 很好（同系列小模型） | 0.8 | 5 | ~2.5-3x |
| 一般 | 0.6 | 4 | ~1.8-2.2x |
| 较差 | 0.4 | 3 | ~1.3-1.5x |

> **实际经验**：好的 draft model + 合适的 $\gamma$ 通常能达到 **2-3x** 加速，且对生成质量 **零损失**。

---

## 5 主要变体

### 5.1 Self-Speculative Decoding

- **思路**：不用外部 draft model，而是让 target model 自身的浅层（early exit）或跳层作为 draft
- **优势**：不需要额外模型，显存零开销，draft 与 target 天然对齐
- **代表**：Draft & Verify (2023)、LayerSkip (Meta, 2024)

### 5.2 Medusa（多头预测）

> Cai et al., 2024

- **思路**：在 LLM 最后一层之上添加多个 **Medusa Head**，每个 head 预测未来第 $k$ 步的 token
- **并行 draft**：所有 head 同时预测，无需串行自回归
- **Tree Attention**：将多个候选路径组织成树结构，一次 forward pass 验证整棵树
- **优势**：无需独立 draft model，训练成本低（只训练 head）
- **局限**：需要额外训练 Medusa Head，不是即插即用

### 5.3 Eagle / Eagle-2

> Li et al., 2024

- **思路**：用一个轻量级自回归 head 结合 target model 的 hidden states 做 draft
- **特点**：比 Medusa 的独立 head 预测更准确（因为利用了隐状态的上下文信息）
- **Eagle-2**：引入动态 draft 树结构，根据 confidence 动态调整投机深度
- **加速比**：在多个 benchmark 上达到 3-4x

### 5.4 Lookahead Decoding

> Fu et al., 2024

- **思路**：利用 Jacobi iteration 将自回归生成转化为并行求解不动点问题
- **无需 draft model**：通过并行猜测多个位置并迭代修正来加速
- **优势**：完全无损，不需要任何额外模型或训练
- **局限**：加速比通常不如 draft-based 方法

### 变体对比

| 方法 | 需要额外模型 | 需要训练 | 加速比 | 即插即用 |
|------|------------|---------|--------|---------|
| Standard Speculative | ✅ Draft model | ❌ | 2-3x | ✅ |
| Self-Speculative | ❌ | ❌ | 1.5-2x | ✅ |
| Medusa | ❌（额外 head） | ✅ Head 训练 | 2-3x | ❌ |
| Eagle | ❌（额外 head） | ✅ Head 训练 | 3-4x | ❌ |
| Lookahead | ❌ | ❌ | 1.5-2x | ✅ |

---

## 6 与其他推理优化技术的关系

### 6.1 与 KV Cache 的关系

- Speculative Decoding **依赖** KV Cache——验证阶段需要快速计算 target model 的概率
- Draft model 也有自己的 KV Cache，需要额外显存
- **KV Cache 管理**是实现高效 Speculative Decoding 的工程关键

### 6.2 与量化（Quantization）的关系

- 两者 **正交且互补**：
  - 量化减少每次 forward pass 的开销 → $T_p$ 变小
  - Speculative Decoding 减少 forward pass 次数 → 调用 $T_p$ 的次数变少
- 可以同时使用：量化后的大模型 + 量化后的小 draft model
- 注意：量化可能改变 target model 的分布，影响 acceptance rate

### 6.3 与 Continuous Batching 的关系

- **Speculative Decoding 在 batch 场景下收益递减**：
  - Batch 推理已经部分解决了 memory-bound 问题（更多 token 共享一次参数加载）
  - 不同请求的 draft 长度和 accept 位置不同，难以对齐
  - Batch 越大，单次 forward 的计算密度越高，投机的边际收益越小
- **但在低并发/在线推理场景下，Speculative Decoding 仍然非常有效**

### 6.4 技术栈位置

```
┌─────────────────────────────────────┐
│        应用层（Chat / Agent）         │
├─────────────────────────────────────┤
│   Speculative Decoding（减少步数）    │  ← 算法层
├─────────────────────────────────────┤
│   KV Cache / PagedAttention（显存）   │  ← 系统层
├─────────────────────────────────────┤
│   量化 / 蒸馏（减小模型）             │  ← 模型层
├─────────────────────────────────────┤
│   CUDA / Triton Kernel（加速计算）    │  ← 算子层
└─────────────────────────────────────┘
```

---

## 7 实际部署考虑

### 7.1 Draft Model 选择

| 策略 | 示例 | 优缺点 |
|------|------|--------|
| 同系列小模型 | Llama-70B + Llama-7B | $\alpha$ 高，但需要额外显存 |
| 同模型量化版 | FP16 + INT4 版本 | 分布接近，显存友好 |
| 专门训练的小模型 | 针对 target 蒸馏 | 最佳 $\alpha$，但需要训练成本 |
| N-gram / 检索 | 从 prompt 中检索 | 零成本，适合重复性任务 |

### 7.2 $\gamma$（Draft Length）选择

- 太小：加速比不够，overhead 占比过高
- 太大：后面的 draft token 越来越不准，reject 浪费计算
- **自适应 $\gamma$**：根据运行时 acceptance rate 动态调整，是当前的最佳实践
- 经验值：$\gamma = 3 \sim 7$，取决于 draft model 质量

### 7.3 Batch 场景

- **低并发（≤8）**：Speculative Decoding 收益明显
- **高并发（≥64）**：收益递减甚至可能为负（draft model 的额外开销 > 节省的验证时间）
- **混合策略**：低负载时开启 speculative，高负载时关闭——需要框架层面支持

### 7.4 主流框架支持

| 框架 | 支持情况 |
|------|---------|
| vLLM | ✅ 原生支持，含多种 draft 策略 |
| TensorRT-LLM | ✅ 支持 draft model + Medusa |
| HuggingFace TGI | ✅ 支持 |
| llama.cpp | ✅ 支持 speculative sampling |
| SGLang | ✅ 支持 Eagle 等变体 |

---

## 8 面试常见问题及回答要点

### Q1: Speculative Decoding 为什么能加速，它的前提假设是什么？

**答**：LLM 推理是 memory-bound 的——每步生成 1 个 token，但需要加载全部参数。GPU 的计算能力远未饱和。Speculative Decoding 利用这一点：让小模型串行猜测 $\gamma$ 个 token（几乎瞬间完成），然后大模型一次 forward pass 并行验证所有 $\gamma$ 个位置（成本约等于生成 1 个 token）。**前提假设**是验证 $\gamma$ 个 token 的成本 ≈ 生成 1 个 token 的成本，这在 KV Cache + memory-bound 条件下成立。

### Q2: 如何证明 Speculative Decoding 的输出分布与原始模型完全一致？

**答**：通过 Rejection Sampling。对于 draft token $x$，以 $\min(1, p(x)/q(x))$ 的概率接受；若拒绝，从修正分布 $\text{norm}(\max(0, p(x) - q(x)))$ 中采样。可以数学证明最终每个 token 的选中概率恰好等于 $p(x)$——这是标准的 rejection sampling，不依赖任何近似。

### Q3: Speculative Decoding 在什么场景下效果不好？

**答**：（1）**高并发/大 batch**：GPU 计算利用率已经较高，投机的边际收益小；（2）**Draft model 与 target 差距太大**：acceptance rate 低，频繁 reject 反而浪费；（3）**生成内容高度不可预测**：如随机创意写作，draft 命中率低；（4）**短生成**：overhead 占比过高。最佳场景是低并发 + 长生成 + 可预测内容（如代码、结构化输出）。

### Q4: Medusa 和标准 Speculative Decoding 的本质区别是什么？

**答**：标准方法用独立的 draft model **串行**生成候选 token；Medusa 在 target model 自身上添加多个预测 head，**并行**预测未来多个位置。Medusa 的优势是不需要额外的 draft model（节省显存），但需要训练额外的 head，不是即插即用的。此外，Medusa 使用 tree attention 来验证多条候选路径，比线性验证更高效。

### Q5: 如果面试官问"你会怎么在生产环境中部署 Speculative Decoding"？

**答**：我会从以下维度考虑——（1）**选择 draft model**：同系列小模型或量化版本，平衡 $\alpha$ 和显存；（2）**调参**：通过 profiling 确定最佳 $\gamma$，或使用自适应策略；（3）**框架**：基于 vLLM 或 TensorRT-LLM，它们有成熟的实现；（4）**流量感知**：低并发时开启 speculative，高并发时降级为普通自回归；（5）**监控**：持续监控 acceptance rate 和 latency，作为调优依据。

### Q6: Speculative Decoding 能和量化一起用吗？

**答**：可以，而且推荐一起用。量化减小模型体积、降低每次 forward pass 延迟；Speculative Decoding 减少 forward pass 次数。两者正交互补。一种常见做法是用 FP16 模型做 target、INT4 版本做 draft（同一个模型的不同量化版本，分布非常接近，$\alpha$ 很高）。

---

## 参考资料

- Leviathan et al. (2023). *Fast Inference from Transformers via Speculative Decoding* — 奠基论文
- Chen et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling* — DeepMind 独立提出
- Cai et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
- Li et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
- Fu et al. (2024). *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*

---

## See Also

- [[AI/3-LLM/Inference/Speculative Decoding|Speculative Decoding（LLM 深度版）]] — 本文面试版，深度版含 Medusa/EAGLE/Lookahead 实现细节
- [[AI/1-Foundations/Inference/KV Cache|KV Cache]] — 推理加速双轮：Speculative Decoding 降 latency，KV Cache 优化降显存；两者互补
- [[AI/1-Foundations/Inference/采样策略|采样策略]] — Speculative Decoding 的 draft token 验收机制依赖采样策略（temperature/top-p）；草稿模型用贪心，验证阶段用修正采样
- [[AI/3-LLM/Inference/LLM推理优化2026全景|LLM 推理优化 2026 全景]] ⭐ — 推理加速工程全景；Speculative Decoding 是其中 compute-bound 场景的核心技术
