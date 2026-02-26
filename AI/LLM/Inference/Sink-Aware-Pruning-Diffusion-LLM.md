---
title: "Sink-Aware Pruning for Diffusion Language Models"
brief: "MBZUAI VILA Lab（arXiv:2602.17664）在 Diffusion LLM 剪枝中识别并保留 Attention Sink token（接受大量注意力的特殊位置），其他位置正常剪枝；与 StreamingLLM 中 sink 保留策略相对应，但应用于 masked diffusion 范式。可在 50% 参数下保持 90%+ 生成质量。"
date: 2026-02-20
type: paper
domain: ai/llm/inference
rating: ★★★★☆
arxiv: "2602.17664"
github: "VILA-Lab/Sink-Aware-Pruning"
institution: MBZUAI (VILA Lab)
tags:
  - diffusion-LLM
  - pruning
  - attention-sink
  - inference-optimization
  - LLaDA
institution: VILA Lab, MBZUAI
---

# Sink-Aware Pruning：DLM 的 attention sink 不是永久的锚

> **一句话**：AR LLM 的 attention sink（BOS/前缀 token 吸引大量注意力）是稳定的全局锚，剪枝时必须保留。但 Diffusion LLM 的 sink 是**短暂漂移的**——去噪过程中 sink 位置频繁迁移，是暂时的副产品而非结构支柱。Sink-Aware Pruning 识别并剪掉这些不稳定 sink，无需重训，在相同算力下超越 Wanda/SparseGPT 等强 baseline。

---

## 背景：Attention Sink 是什么

**Attention sink**（Xiao et al., 2023）：LLM 中有一小组 token 位置（通常是 BOS、系统提示等前缀 token），在几乎所有层和注意力头中都吸引着不成比例的大量 attention mass，但这些位置本身语义价值有限。

**为什么存在**：Softmax 归一化必须把注意力分配给某些位置。当一个 query 在上下文中找不到强匹配时，attention 就会"溢出"到这些全局可见的早期 token——它们成了"注意力的垃圾桶"，但同时也扮演着稳定 residual stream 的角色。

**AR LLM 中 sink 的重要性**：
- 位置高度持久：一旦某个 sink 出现（通常是前缀），它在后续所有 token 生成和所有层中保持 sink 地位
- 是因果、前缀条件计算图中的结构性稳定器
- 剪枝时必须保留（否则性能崩溃）

这个"必须保留 sink"的原则已成为 AR LLM 剪枝的 de facto 标准。

---

## 核心发现：DLM 的 sink 与 AR 完全不同

### 两个方差统计量

论文引入两个互补指标量化 sink 的时间动态：

**Spatial Variance**（空间方差）：
```
m̄(i) = (1/T) Σ_t m_t(i)    # 每个位置的平均 attention mass
σ²_spatial = Var_i(m̄(i))   # 这些平均值的方差
```
大 → 少数位置长期主导（有 sink）

**Temporal Variance**（时间方差）：
```
c_t = Σ_{i∈S_t} m_t(i)·i / Σ_{i∈S_t} m_t(i)    # 当前步 sink 质心
σ²_temporal = Var_t(c_t)
```
近零 → sink 位置固定；大 → sink 在步与步之间大幅漂移

### 实验对比（LLaDA/Dream vs LLaMA-3.1/Qwen-2.5）

| 模型类型 | Spatial Variance | Temporal Variance |
|---------|-----------------|------------------|
| AR（LLaMA/Qwen） | **高**（少数位置主导） | **近零**（sink 固定不动） |
| DLM（LLaDA/Dream） | 较低（注意力分布更均匀） | **高**（sink 位置频繁漂移） |

**关键结论**：
- AR：spatial 高 + temporal 低 → **稳定的结构性 sink**，必须保留
- DLM：spatial 低 + temporal 高 → **暂时的漂移性 sink**，可以（应该）剪掉

### 为什么 DLM 的 sink 会漂移？

DLM 的去噪过程有不同的计算需求：
- **早期步（高噪声）**：需要解决全局结构，某些位置成为全局协调点 → 形成 sink
- **晚期步（低噪声）**：精化局部语法和语义，全局协调需求减弱 → 原 sink 失去重要性，新 sink 可能在其他位置形成

这与 AR 根本不同：AR 的因果前缀条件让早期 token 永久重要；DLM 的全序列并行去噪让重要性在时序上是流动的。

同时，DLM 的**双向 attention**（可以参考任意位置）提供了替代的信息聚合路径，使得 sink 的"结构性稳定器"作用变弱——即使移除某个 sink，模型有其他路径完成信息传递。

---

## Sink-Aware Pruning 方法（四步，无需重训）

见 Fig.3，是一个**即插即用的权重重要性修正框架**，在 Wanda 或 SparseGPT 之上加 sink-aware 修正：

**Step 1: 计算 attention mass**
- 收集 calibration 数据集上的 attention 矩阵 A^(t)（每个 diffusion 步）
- 计算每个位置 i 在每步 t 接收的 incoming attention mass：`m_t(i) = Σ_j A^(t)_{j,i}`

**Step 2: 识别不稳定 sink，计算软降权因子**
- 基于阈值（如 attention mass > μ + kσ）识别 sink 位置
- 计算跨步的 temporal variance，标记高方差 sink 为"不稳定"
- 对不稳定 sink 位置的激活值计算软降权因子：`ω = 1 - s`（s 是 sink 程度的量化）
- 修正激活：`X̃ = X * ω`（在 sink 位置降权激活）

**Step 3: 更新重要性分数**
- 把修正后的激活 X̃ 代入 Wanda 或 SparseGPT 的重要性计算：
  - Wanda: `S_ij = |W_ij| · ||X̃_·j||₂`（不稳定 sink 激活被压缩 → 相关权重重要性下降）
  - SparseGPT: 用 X̃ 更新 Hessian 估计

**Step 4: 剪枝决策**
- 基于更新后的重要性分数做最终剪枝
- 低重要性的 sink 相关权重被优先移除

**关键设计**：这不是直接"删除 sink token"，而是通过**降权 sink 位置的激活值**来影响权重重要性估计。无需修改推理逻辑，只影响剪枝决策。

---

## 实验结果

**基线**：Wanda、SparseGPT（AR LLM 剪枝标准方法）

**被测 DLM**：LLaDA、Dream（主流 masked diffusion LLM）

**结论**：
- 在相同稀疏度（sparsity ratio）下，Sink-Aware Pruning 的 perplexity 和下游任务得分均优于标准 Wanda/SparseGPT
- 差距在高稀疏度时更显著（AR heuristic 在高稀疏度时的 sink 保护反而成为错误的约束）
- 无需重训，calibration-only，可直接应用于已有 DLM

---

## 批判性分析

### 这个观察真正重要在哪里

这篇论文的核心贡献不只是一个更好的剪枝方法，而是揭示了一个更基础的事实：**AR LLM 的归纳经验不能无脑迁移到 DLM**。

"保留 sink"这个在 AR 中 empirically 坚实的原则，在 DLM 中是错的——不只是次优，而是方向相反。这说明 DLM 是一个质性不同的计算范式，需要专属的分析和优化框架。

### 与 MAGE 的关联（昨日读）

MAGE 利用了 DLM 的另一个独特属性：All-[MASK] 第一步 attention 的 temporal consistency（KV indices 跨步稳定），做 sparse attention。

Sink-Aware Pruning 发现了恰恰相反的现象：sink 位置的 temporal instability。

两者并不矛盾——稳定的是**哪些非 mask 上下文 token 对生成 token 重要**（MAGE 利用），不稳定的是**哪些位置成为全局注意力汇集点**（sink，Sink-Aware Pruning 处理）。这两个维度在 DLM 中独立变化。

### 方法的局限

1. **Calibration 数据依赖**：需要代表性 calibration set 来准确估计 sink variance，域外数据可能得到错误的 sink 分类
2. **软降权因子 ω 的选择**：论文描述为 `ω = 1 - s`，但 s 的定量化方式（阈值怎么定、是否需要 per-layer 调整）细节不够充分
3. **只做了权重剪枝**：没有和 MAGE 类型的 attention 稀疏化组合——两者本可以互补（先 MAGE 做 KV 稀疏，再 Sink-Aware 做权重稀疏）
4. **评估 DLM 规模有限**：LLaDA 和 Dream 是当前主流，但 MMaDA、MDLM 等其他架构是否适用未验证

### 为什么这方向值得追踪

DLM 的推理加速是 2026 年最活跃的方向之一。已有：MAGE（注意力稀疏），Sparrow（Video LLM speculative decoding），Sink-Aware Pruning（权重剪枝）。三者各有不同工具：
- MAGE：减少每步内的 attention 计算量
- Sink-Aware：减少模型参数量
- Speculative Decoding（DLM 版本）：减少去噪步数

这三条路最终需要组合起来，才能真正让 DLM 达到 AR LLM 的推理效率。

---

## 与 Vault 其他笔记的连接

- → [[AI/LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention|MAGE (DLM 稀疏优化，互补视角)]]
- → [[AI/LLM/Inference/LaViDa-R1-Diffusion-LLM-Reasoning|LaViDa-R1 (DLM 能力边界)]]
- → [[AI/LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow (DLM 推理加速，不同工具)]]
- → [[AI/LLM/Inference/LLM-推理优化-2026-全景|LLM 推理优化 2026 全景 (全局视角)]]
- → [[AI/LLM/Inference/剪枝与蒸馏|剪枝与蒸馏 (权重压缩背景)]]
