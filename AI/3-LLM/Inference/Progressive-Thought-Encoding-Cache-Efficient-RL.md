---
title: "Progressive Thought Encoding (PTE): Cache-Efficient RL"
brief: "（arXiv:2602.16839，ICLR 2026）在 RL 训练中复用多轮 CoT 的 KV Cache：将前几步推理编码压缩后拼接给后续步骤，避免重复计算 Transformer 前向传播；在长推理 RL 训练中节省 50-70% 显存，使超长 horizon 训练成为可能。"
date: 2026-02-18
tags: [推理效率, KV缓存, RL训练, 内存优化, 长推理, ICLR2026]
domain: AI/LLM/Inference
arxiv: "2602.16839"
rating: 4
status: permanent
---

# Progressive Thought Encoding: Cache-Efficient RL for Large Reasoning Models

> Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**arXiv**: 2602.16839  
**作者**: Zeliang Zhang (Rochester), Xiaodong Liu†, Hao Cheng, Hao Sun, Chenliang Xu, **Jianfeng Gao**（微软研究院）  
**提交日期**: 2026-02-18  
**发表**: **ICLR 2026**（已录用，15 pages）  
**标签**: #推理效率 #KV缓存 #LoRA #RL训练 #内存优化 #长推理

---

## 核心问题：RL 训练的 KV Cache 瓶颈

### 问题链

1. **LRM 推理需要长 CoT**：困难任务需要几千到几万 token 的推理轨迹
2. **RL 训练依赖 rollout**：outcome-based reward 必须等轨迹完成才能计算
3. **KV Cache 线性增长**：每个 token 都要 cache 其 key/value，内存随 rollout 长度线性增长
4. **内存成为实际瓶颈**：8B 模型在 3072 token 的 rollout 下峰值内存达 89% A100

### 已有解法的失败

**Sliding Window Cache**：固定 window 大小，evict 旧 token → 推理质量下降

实验验证：
- Qwen2.5-7B LoRA（全 cache）：38.1% 平均准确率
- Qwen2.5-7B LoRA + Sliding Window：**36.7%**（-1.4%）
- 在 AIME 这类困难任务上，差距更大

原因：推理过程中**被 evict 的早期 token 往往是关键推理步骤**，丢掉它们导致模型"失忆"

---

## 方法：Progressive Thought Encoding

**核心 insight**：不要丢掉被 evict 的 token，而是在 evict 之前**学习**它们，把信息压缩进 LoRA 参数里。

### 数学形式化

**Cache-constrained 策略**：

```
π_θ^D(y|p) = ∏_{t=1}^T π_θ(y_t | C_t^D)
```

其中 `C_t^D = CachePrune_D(p, y_{<t})` 是 pruned context。

**目标**：训练出来的 `π_θ_g^D(y|p) ≈ π_θ*(y|p)`（受约束的策略 ≈ 全 cache 策略）

### 核心更新规则

当 KV cache 满时，被 evict 的 token `{y_{e1}, ..., y_{em}}` 的 key/value 为 `K_e, V_e`。

用一个**global query vector** `q_g` 聚合这些 evicted token 的信息：

```
ΔW = A · S_e · B

其中 S_e = (W_Q^a q_g)(W_K^a K_e)^T (W_K^a V_e)
         = Cross-attention(q_g → evicted tokens)
```

- `A, B`：LoRA 的 up/down projection 矩阵（固定 rank=32）
- `S_e`：被 evict token 的压缩 context state
- `q_g`：全局查询向量，是 evicted 信息的聚合器

**Progressive update**：每次 cache 满并 evict 时：
```
S_e ← Normalize(S_e + S_e_new)
ΔW ← A · S_e · B
θ' = θ + ΔW
```

模型继续在 `θ' = θ + ΔW` 下 decode，把 evicted 历史"内化"进了 LoRA。

### Eviction 策略设计

- **问题 token 永久保留**（类比 sink tokens，锚定 prompt context）
- **推理 token 用 sliding window evict**（25% of tokens evicted when cache full）
- 每次 evict 前先学习，再 evict

### 初始化技巧

Global tokens `h_g` 初始化 `S_e`，让 `q_g` 从一开始就有语义意义（避免零初始化的冷启动问题）。

---

## 实验结果

### 主表（Table 1）：准确率 vs 内存效率

| 模型 | 方法 | 平均准确率 | Peak Memory | FLOPs |
|------|------|-----------|------------|-------|
| Qwen2.5-3B | Baseline | - | 100% | 基准 |
| Qwen2.5-3B | LoRA（全 cache）| 28.2% | 83% | 4.2T |
| Qwen2.5-3B | LoRA + Sliding Window | 25.6% | 38% | - |
| Qwen2.5-3B | **PTE（ours）** | **30.1%** | **45%** | **2.7T** |
| Qwen2.5-7B | LoRA | 38.1% | 85.8% | 5.7T |
| Qwen2.5-7B | LoRA + SW | 36.7% | 63.1% | - |
| Qwen2.5-7B | **PTE（ours）** | **39.6%** | - | **3.6T** |
| DeepSeek-R1-8B | LoRA | 34.9% | 89% | 7.4T |
| DeepSeek-R1-8B | LoRA + SW | 34.2% | - | - |
| DeepSeek-R1-8B | **PTE（ours）** | **45.6%** | 59.8% | **4.6T** |

**DeepSeek-R1-8B 的 AIME 结果（pass@16）**：
- AIME2024: **+33.4%** vs LoRA baseline
- AIME2025: **+23.3%** vs LoRA baseline

### 变长 rollout 的扩展性（Figure 5）

在 1K context window 约束下，把 max generation 从 3K 扩到 **64K**：
- LoRA：在 ~8K 时性能开始 plateau
- **PTE**：**持续增长到 64K**，无 plateau

这说明 PTE 不只是解决了内存问题，还让模型真正学会了"在受限 cache 下长程推理"。

---

## 批判性分析

### 为什么这个思路有效

**关键 insight**：被 evict 的推理 token 不是随机的——它们是时序上较早的推理步骤。LRM 的 long CoT 存在**层次性**：早期步骤是问题分解和初步分析，晚期步骤是基于前面的细化。如果把早期步骤直接丢掉，后面的推理就失去了 grounding。

PTE 通过 cross-attention（global query → evicted tokens）把这些早期步骤的精华**压缩进 LoRA 参数**，让模型用 ΔW 来"记住"它本该看到但因 cache 限制而看不到的内容。

这本质上是一个**在线蒸馏**（online distillation）：
- Teacher：全 cache 的 π_θ* 
- Student：受限 cache 的 π_θ_g
- 蒸馏媒介：LoRA adapter（ΔW），用 evicted token 的 cross-attention 计算更新

### 我的疑问

1. **LoRA 参数更新的频率问题**：每次 cache 满都更新 ΔW，在 64K 生成的情况下这可能发生几十次。每次更新会不会导致灾难性遗忘（early evicted 信息被 later updates 覆盖）？论文用 Normalize(S_e + S_e') 累积，但这是否足够？

2. **Global query q_g 的学习**：q_g 是 learnable 参数。在不同问题类型（代数/几何/组合）上，q_g 是否能泛化？还是说每类问题需要不同的 q_g？

3. **Eviction 策略的 bias**：使用 sliding window eviction（evict 最旧的 token），假设早期 token 重要性低于近期 token。但在长推理中，"问题理解"步骤（往往在最前面）可能比中间的计算步骤更重要。这个 bias 会导致什么？

4. **与 FlashAttention / PagedAttention 的兼容性**：论文是否考虑了与这些高效 attention 实现的集成？ΔW 的更新需要访问 evicted K/V，这在 paged 系统中可能需要额外设计。

5. **计算开销**：每次 eviction 时要做 cross-attention(q_g → K_e, V_e) + 矩阵乘法更新 ΔW。对于长 rollout（>8K），这个 overhead 多大？论文给了 FLOPs 数字但没有 wall-time 对比。

### 与相关工作的关系

| 工作 | 解决的问题 | 与 PTE 的关系 |
|------|----------|------------|
| Jet-RL (2601.14243) | FP8 量化的 training/rollout 不一致 | 正交——PTE 解决内存，Jet-RL 解决精度一致性 |
| StreamingLLM | Sink token + sliding window | PTE 在 sliding window 基础上 "学习" evicted content |
| LoRA | 高效 fine-tuning | PTE 是动态 LoRA 更新（test-time LoRA） |
| Memory efficient attention | Recompute activations | 另一个方向，PTE 是 compress 信息 |

### 最大 novelty

**在 token eviction 发生前先学习**这个 insight 很简单但很有力。现有的所有 cache 压缩方法（sliding window、H2O、ScissorHands）都是在 eviction 后就把 token 丢弃，PTE 是第一个让 eviction 变成学习机会的方法。

---

## 核心贡献总结

1. **明确了 RL 训练的 cache-constrained optimization 形式化**：把问题定义为让 π_θ_g^D 近似 π_θ*，这个形式化使后续方法的设计有了清晰目标
2. **Progressive Thought Encoding 方法**：online cross-attention → LoRA update → eviction，循环压缩推理历史
3. **训练和推理双重受益**：训练时省内存（-40% peak），推理时也省内存（constant cache 不增长）
4. **AIME 上的大幅收益**：+23-33% on AIME，说明困难推理任务受益最大

---

## 评分

**★★★★★（5/5）**

罕见的 5 星。理由：
- 问题真实重要（RL 训练内存是实际瓶颈）
- Insight 简单且正确（eviction 前学习）
- 方法工程上可实现（LoRA，rank=32，不需要大的 overhead）
- 结果显著（AIME +33%，内存 -40%）
- ICLR 2026 录用，社区认可
- 泛化性好：原理上适用于任何 KV cache + RL 的场景

**唯一遗憾**：没有 wall-time breakdown（训练总时间 vs LoRA 的对比），只有 FLOPs 和 memory。

---

## 对老板的关键 Takeaways

### 实践意义
如果在做 LRM 的 RL 训练，**PTE 解决了一个真实痛点**：
- 现有框架（VeRL/SLIME）在长 rollout 时内存爆炸
- PTE 可以让同样的 GPU 支持 2x 的 rollout 长度
- 更重要的是：更长的 rollout → 更好的推理能力（见 Figure 5）

### 面试话术
"LRM 的 RL 训练有个 fundamental tension：越难的任务需要越长的 CoT，越长的 CoT 需要越大的 KV cache，在有限内存下 sliding window eviction 会让模型失忆。Progressive Thought Encoding 的思路是，在 evict 一个 token 之前，先用 cross-attention 把它的信息学进 LoRA adapter 里，让模型用 ΔW 来补偿它本该看到但已经丢失的 token。这本质上是个 online self-distillation。"

### 与 Jet-RL 的关系
Jet-RL 和 PTE 是同一个"RL 训练效率"大问题的两个子问题：
- Jet-RL：量化精度不一致导致 off-policy → 统一 FP8 flow
- PTE：KV cache 限制导致信息丢失 → 压缩 evicted token 到 LoRA

两者可以组合：FP8 rollout（Jet-RL）+ Progressive Thought Encoding（PTE）→ 既快又省内存还不信息丢失

---

## see-also

- [[Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 同一"RL 训练效率"问题的另一维度：FP8 量化精度不一致→统一 precision flow（NVIDIA+MIT HAN Lab，arXiv:2601.14243）
- [[AI/3-LLM/Inference/KV Cache]] — KV Cache 核心机制
- [[AI/3-LLM/Inference/KV Cache|KV Cache]] — KV Cache 优化综述
- [[Deep-Thinking-Ratio-DTR-v2-Think-At-N|DTR v2 + Think@N]] — 推理质量评估维度，与 PTE 正交（PTE 解决训练效率，DTR 解决推理轨迹选择）
- [[Accordion-Thinking-Self-Regulated-Step-Summaries|Accordion-Thinking]] — 同类问题的主动压缩路径：模型学会生成 summary 后 fold，与 PTE 的被动 evict 补救互补；三者组合（Accordion+PTE+Jet-RL）= 完整高效长推理 pipeline（arXiv:2602.03249）★★★★☆
