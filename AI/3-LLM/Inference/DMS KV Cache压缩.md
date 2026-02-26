---
title: "DMS: 动态内存稀疏化 KV Cache 压缩"
brief: "NVIDIA + Edinburgh（arXiv:2506.05345，NeurIPS 2025）提出 DMS：~1K 步轻量 post-training 让 LLM 获得 8× KV Cache 压缩能力，reasoning 任务几乎无精度损失；在固定显存预算下允许运行更多更长推理链，实测反而提升准确率。"
date: 2026-02-15
tags: [inference, kv-cache, compression, sparsification, interview]
type: note
---

# DMS: 动态内存稀疏化 KV Cache 压缩

> **论文**: [Inference-Time Hyper-Scaling with KV Cache Compression](https://arxiv.org/abs/2506.05345) (arXiv 2506.05345)
> **作者**: Adrian Łańcucki, Konrad Staniszewski (NVIDIA), Piotr Nawrot, Edoardo M. Ponti (University of Edinburgh)
> **发表**: NeurIPS 2025
> **一句话总结**: 仅需 ~1K 步轻量 post-training，即可让现有 LLM 获得 **8× KV cache 压缩能力**，在 reasoning 任务上几乎无精度损失，甚至因为能在相同显存预算下跑更多/更长的 reasoning chain，反而提升了推理准确率。

---

## 1. KV Cache 瓶颈回顾

### 1.1 KV Cache 是什么

在 Transformer 的自回归生成（auto-regressive generation）中，每一步解码都需要对所有先前 token 做 attention。为了避免重复计算，我们把每一层每个 attention head 产生的 Key 和 Value 向量缓存下来，这就是 **KV Cache**。

### 1.2 显存占用量级

以 Llama-2 70B 为例（80 层, 8 KV heads, head_dim=128, FP16）：

```
KV Cache per token = 2 (K+V) × 80 layers × 8 heads × 128 dim × 2 bytes
                   = 327,680 bytes ≈ 320 KB/token
```

对于 32K 上下文窗口：`320 KB × 32,768 = 10 GB`。这还只是 **一条序列** 的缓存量。如果做 batch serving（比如 batch_size=64），KV cache 就直接占到 640 GB，远超单张 GPU 的显存。

### 1.3 为什么 KV Cache 是推理瓶颈

1. **显存容量限制**: KV cache 线性增长于序列长度 × batch size，长序列场景（128K+）下是首要 OOM 原因
2. **内存带宽瓶颈**: 解码阶段是典型的 memory-bound 操作——每一步生成一个 token，却需要读取整个 KV cache。在 Qwen-R1 1.5B/7B 模型上，8K-32K 序列长度下 KV cache 读取占推理延迟的 80%-90%
3. **Inference-Time Scaling 的障碍**: 当代 reasoning 模型（o1, R1, QwQ）通过生成更长/更多的推理链来提升准确率，但 KV cache 的膨胀使得"思考更久"的成本急剧上升

### 1.4 核心矛盾

> **Inference-time scaling** 要求模型"想得更多"（更长序列 + 更多并行推理链），但 KV cache 是限制"想得更多"的硬性瓶颈。

这就是 DMS 论文提出 **"Inference-Time Hyper-Scaling"** 概念的出发点：如果能压缩 KV cache，就能在同等显存/延迟预算下探索更多推理路径，从而提升最终答案质量。

---

## 2. 现有压缩方案对比

### 2.1 Training-Free 稀疏方法

| 方法 | 核心思路 | 局限 |
|------|---------|------|
| **H2O** (Heavy-Hitter Oracle) | 保留累积 attention weight 最高的 token + 滑动窗口内的最近 token | 高压缩比下精度显著下降；启发式规则无法适应不同 head 的差异化需求 |
| **TOVA** (Token Omission via Attention) | 每步驱逐当前 attention weight 最低的 token | 同上；在 GQA 架构下尤其有害，因为多个 query head 共享同一组 KV |
| **StreamingLLM** | 保留 attention sink（开头几个 token）+ 滑动窗口 | 只适用于流式场景，中间重要信息会丢失 |
| **Quest** | 不驱逐 token，只选择性地从显存中读取最相关的 page | 加速了推理但**不减少显存占用**，甚至因为额外存储 page 索引有轻微开销 |

**共同问题**: 这些方法在 4× 以上压缩比时精度掉得很厉害。原因是它们用的都是固定的启发式规则（如 attention score 排序），没有让模型本身学会"该丢什么"。

### 2.2 需要训练的压缩方法

| 方法 | 核心思路 | 局限 |
|------|---------|------|
| **DMC** (Dynamic Memory Compression) | 每个 attention head 独立决定 append 还是 merge（加权平均）KV 到 cache | 训练成本高（需要大量步数）；prefill 阶段不加速；实现复杂 |
| **量化** (KV Cache Quantization) | 将 KV cache 从 FP16 压缩到 INT8/INT4 | 通常只能 2-4× 压缩；与稀疏化方法正交，可叠加使用 |
| **MLA** (Multi-Latent Attention) | DeepSeek-V2 的架构级方案，用低秩投影压缩 KV | 需要从头训练；不能直接 retrofit 到现有模型 |

### 2.3 PagedAttention

vLLM 的 PagedAttention 解决的是 KV cache 的**内存碎片化**问题（类似操作系统的虚拟内存分页），而非真正减少 KV 的数量。它是一个正交的优化，DMS 实际上可以和 PagedAttention 结合使用——DMS 稀疏化后的 KV cache 以 page 为单位存储，被驱逐的 token 的 page 可以被回收。

### 2.4 DMS 的定位

DMS 填补了一个关键空白：**用极低训练成本（~1K 步）实现高压缩比（8×），同时几乎不损失精度**。它是 post-training 方案，可以 retrofit 到任意已有模型。

---

## 3. DMS 核心方法：动态稀疏化机制

### 3.1 核心思想

DMS 的本质是教会模型一个**自适应的 token 驱逐策略**：对于每个 attention head 的每个 token，模型自己预测一个二元决策 α ∈ {0, 1}——这个 KV 要不要在未来被驱逐。

关键创新在两点：
1. **Learned Eviction（学习驱逐）**: 不靠启发式规则，而是让模型通过梯度下降学会哪些 token 的 KV 可以丢弃
2. **Delayed Eviction（延迟驱逐）**: 被标记为驱逐的 token 不立即删除，而是在一个滑动窗口 w 内仍然可见，给模型时间"吸收"该 token 的信息后再真正丢弃

### 3.2 数学细节

#### 驱逐决策的生成

给定 attention 层的输入隐藏状态 **h_t**，DMS 为每个 KV head 预测驱逐概率：

```
α_t ~ Gumbel-sigmoid(h_t · w^T + b, τ) ∈ [0, 1]
```

其中：
- **w** ∈ ℝ^d 是可训练的权重向量（初始化为全零）
- **b = -5** 是偏置（初始化为负值，确保训练初期 α_t ≈ 0，即默认不驱逐，避免早期 loss spike 导致灾难性遗忘）
- **τ** 是温度参数（设为低值以鼓励离散决策）
- Gumbel-sigmoid 提供了可微分的离散近似（类似 Gumbel-Softmax trick）

**关键设计**: 实现上 DMS 复用 query 或 key 投影矩阵中的一个 neuron 来预测 α，**不引入任何额外参数**。这是比 DMC 更加轻量的设计。

#### 训练时的注意力掩码

训练时，驱逐决策 α_1:T 被编码为一个加性注意力掩码 M_α：
- 未被驱逐的 token: mask 值 = 0（正常参与 attention）
- 被驱逐但仍在滑动窗口内的 token: mask 值 = log(1 - α)（部分衰减）
- 超出滑动窗口的已驱逐 token: mask 值 = -∞（完全屏蔽）

这个设计使得驱逐是一个"渐进淡出"的过程，而非突然消失。

#### 推理时的行为

推理时将连续决策硬量化为二元值：

```
α_t^bin = round(sigmoid(h_t · w^T + b)) ∈ {0, 1}
```

如果 α_t^bin = 1，则 (k_t, v_t) 在 t+w 步后被驱逐。被驱逐的 token 的存储空间直接被新 token 覆写，无需额外的删除操作。

### 3.3 延迟驱逐为什么重要

论文的消融实验明确验证了：**立即驱逐（w=0）会导致所有滑动窗口大小下的快速性能退化**，而延迟驱逐则保持稳定训练。

直觉理解：decoder-only 模型强烈依赖近期 token 的 attention（recency bias）。如果一个 token 刚被写入 cache 就被立即驱逐，模型根本没有机会从它那里提取信息。延迟驱逐给了模型一个缓冲期——"我知道你要走了，让我先把你的信息吸收完"。

默认滑动窗口大小 w=256。

### 3.4 每个 Head 的独立决策

DMS 允许不同 attention head 采用不同的压缩率。有些 head 可能需要保留大部分 token（如关注局部语法的 head），有些 head 可能可以大量稀疏化（如关注全局语义的 head）。这种灵活性是比 uniform pruning 更高效的关键。

### 3.5 与 DMC 的关键区别

| 维度 | DMC (Dynamic Memory Compression) | DMS (Dynamic Memory Sparsification) |
|------|----------------------------------|--------------------------------------|
| 压缩方式 | Merge（加权平均合并 token） | Evict（直接丢弃 token） |
| 额外参数 | 每个 head 有额外参数 | **零额外参数**（复用现有 neuron） |
| 训练成本 | 数千步 + 大量数据 | **~1K 步** |
| Prefill 加速 | ❌ 不加速（保留中间累积 token） | ✅ 可加速（驱逐的 token 在 prefill 也可跳过） |
| 实现复杂度 | 高（需要维护变长的 merged KV） | 低（标准稀疏 attention） |

---

## 4. 训练流程：为什么只需 1K 步

### 4.1 训练配置

- **训练方式**: Logit Distillation（原始模型作 teacher，DMS retrofit 后的模型作 student）
- **训练步数**: 约 1K 步（具体公式：每 100 步增加 1× 压缩比，8× 压缩需要 ~700-800 步，加上 warmup 约 1K 步）
- **压缩比调度**: 线性退火 CR(t) = t/100 + 1，从 1×（无压缩）逐步增加到目标压缩比
- **损失函数**: L = L_D (logit distillation loss) + L_aux (压缩约束 loss)
- **滑动窗口**: w = 256 tokens

### 4.2 为什么能这么快

三个关键设计使得 DMS 训练极其高效：

1. **零额外参数**: 不需要从零学习新的网络结构，只需要调整已有 neuron 的行为。这大幅减小了搜索空间。

2. **安全初始化**: b=-5 确保训练初期模型行为几乎与原始模型一致（α ≈ 0，不驱逐任何 token）。然后通过线性退火逐步增加压缩率，模型有足够时间平滑适应。

3. **Logit Distillation**: 相比标准 language modeling loss，logit distillation 对训练数据分布不敏感（因为 teacher 提供了 soft label），且对小模型特别有效。这避免了需要精确复制原始预训练数据配比的问题。

### 4.3 训练数据

论文使用了公开数据集进行 distillation（具体配比见论文 Appendix C）。关键点是**不需要原始预训练数据**——任何通用文本数据都可以用于 distillation 训练。

### 4.4 训练成本估算

以 Qwen-R1 7B 为例：
- 1K 步 × batch_size × seq_len 的训练数据量
- 在单节点 8×A100 上大约几小时可以完成
- 这比完整的 continued pre-training（通常需要数十到数百 GPU-days）便宜了 2-3 个数量级

---

## 5. 实验结果：压缩比 vs 精度 Tradeoff

### 5.1 Reasoning 任务（核心结果）

论文在以下推理密集型 benchmark 上评测：

| Benchmark | 任务类型 |
|-----------|---------|
| AIME 2024 | 高难度竞赛数学 |
| MATH-500 | 数学问题求解 |
| GPQA Diamond | 硬科学 QA（物理/化学/生物） |
| LiveCodeBench | 代码生成 |

**核心数字**（Qwen-R1 32B + DMS 8× 压缩 vs 原始模型在等价推理预算下）：

- **AIME 24**: +12.0 points（从约 54% → 66%）
- **GPQA Diamond**: +8.6 points
- **LiveCodeBench**: +9.7 points

这些不是"压缩后不掉分"的结果——而是 **压缩后反而涨分**。原因是 8× 压缩后，相同显存预算可以跑 8× 更多的 parallel reasoning chains，通过 majority voting 获得更好的最终答案。

### 5.2 跨模型规模的一致性

| 模型 | AIME 24 提升 | GPQA 提升 | LiveCodeBench 提升 |
|------|-------------|----------|-------------------|
| Qwen-R1 1.5B | 显著提升 | 显著提升 | 显著提升 |
| Qwen-R1 7B | 显著提升 | 显著提升 | 显著提升 |
| Qwen-R1 32B | +12.0 | +8.6 | +9.7 |

跨所有模型规模（1.5B、7B、32B），DMS 都一致性地改善了 Pareto 前沿。

### 5.3 与其他方法的对比

在相同压缩比下的精度排名（从高到低）：

```
DMS > Quest > TOVA > H2O
```

- **DMS vs Quest**: DMS 在精度上全面胜出，且 Quest 不减少显存占用（只减少读取量），DMS 既减少显存又减少读取量
- **DMS vs DMC**: DMS 精度相当或更优，但训练成本低 1-2 个数量级
- **DMS vs TOVA/H2O**: 在 8× 压缩下差距非常显著，training-free 方法在高压缩比下基本不可用

### 5.4 通用任务表现

在非推理任务上（短上下文 benchmark）：
- **4× 压缩**: MMLU、GSM8K、HellaSwag 等精度损失约 3.5 分，基本可接受
- **8× 压缩**: 短上下文任务有一定精度损失，但长上下文任务（Needle-in-a-Haystack、Variable Tracking）**甚至超过原始模型**

长上下文任务反而变好的原因：稀疏化可能缓解了 attention 中的 "information over-squashing" 问题——当 KV cache 太大时，attention weight 被稀释，真正重要的 token 反而得不到足够关注。

### 5.5 关键观察

1. **Compression 不是 pure trade-off**: 传统观念认为压缩必然损失精度。DMS 证明在 reasoning-heavy 场景下，压缩反而释放了更多推理预算，净效果是正的。
2. **Training-free vs Trained**: 启发式方法在低压缩比（2×）时尚可，但高压缩比必须用 learned 方法。
3. **GQA 的影响**: 现代模型普遍使用 GQA（多个 query head 共享 KV），这使得 token 驱逐的影响更大（一个 KV token 被驱逐影响多个 query head），更需要精细的 learned eviction。

---

## 6. 与 Inference-Time Scaling 的结合

### 6.1 Hyper-Scaling 概念

论文提出的核心概念是 **Inference-Time Hyper-Scaling**：

```
传统 Inference-Time Scaling: 花更多 compute → 生成更多 token → 更好的推理

Hyper-Scaling: 压缩 KV cache → 同等 compute 下生成更多 token → 更好的推理
```

这不只是一个效率优化——它改变了 scaling 的 Pareto 前沿。

### 6.2 Sequential vs Parallel Scaling

推理时间缩放有两个维度：
- **Sequential Scaling**: 增加单条推理链的最大长度（L）—— "想得更深"
- **Parallel Scaling**: 增加并行推理链的数量（W）—— "想得更广"

DMS 在两个维度上都有帮助：
- Sequential: 8× 压缩后，同等显存下可以支持 8× 更长的推理链
- Parallel: 同等显存下可以跑 8× 更多的并行 chain，通过 majority voting 提升正确率

### 6.3 配置空间

论文引入了 **L-W-CR** 三元组表示推理配置：
- L: 序列长度（× 1024 tokens）
- W: 并行推理链数量
- CR: 压缩比

例如：
- `32-1-1`: 32K 序列，1 条 chain，无压缩（原始模型的默认配置）
- `32-8-8`: 32K 序列，8 条 chain，8× 压缩（DMS 配置，显存/延迟与 32-1-1 相当）

实验表明 `32-8-8` 在 reasoning benchmark 上显著优于 `32-1-1`。

### 6.4 与 Best-of-N 的协同

DMS 天然适配 Best-of-N（majority voting / reward model reranking）策略：
- 压缩 8× → 同等显存跑 8× chains → 8× 更大的 candidate pool
- 在 AIME 24 这类有明确正确答案的任务上，更多 candidate = 更高的覆盖率

### 6.5 实际意义

对于一个部署 reasoning 模型的团队来说，DMS 提供了一个新的选择：

**原来**: 买更多 GPU → 跑更多 batch → 更快出结果
**现在**: DMS 压缩 → 同等 GPU 跑 8× 更多 chain → 更高质量的结果 + 更低延迟

这是推理效率的 **帕累托改进**。

---

## 7. 工程落地思路

### 7.1 Retrofit 流程

1. **准备 Teacher 模型**: 加载原始模型权重
2. **初始化 Student 模型**: 复制原始权重，为每个 KV head 复用一个 neuron 作为 eviction predictor（w=0, b=-5）
3. **Distillation 训练**: ~1K 步，线性退火压缩目标
4. **导出**: 保存 retrofit 后的权重（模型结构不变，只有权重变化）

### 7.2 推理集成

#### 与 vLLM / TensorRT-LLM 的集成

DMS 的推理逻辑可以分解为：
1. 在每个 attention layer 的 forward 中，额外执行一次 sigmoid 计算得到 eviction decision
2. 在 KV cache manager 中实现 delayed eviction 逻辑（维护一个 eviction queue）
3. 被驱逐 token 的 page 标记为可回收

这些改动主要在 attention kernel 和 cache manager 层面，不需要改变模型的整体架构。

#### PagedAttention 兼容性

DMS 原生兼容 PagedAttention：
- 每个 attention head 可以有不同的 KV cache 长度
- 被驱逐的 token 释放的 page 可以立即被新 token 使用
- 论文明确提到了与 PagedAttention 的集成方案

#### 与量化的叠加

DMS（稀疏化）和 KV cache 量化是正交的优化：
- DMS 8× 稀疏 + INT4 量化 = 理论 32× 压缩
- 具体叠加效果需要验证，但方向上是完全兼容的

### 7.3 已开源的模型

NVIDIA 已在 HuggingFace 上发布了 DMS retrofit 后的模型，例如：
- `nvidia/Qwen3-8B-DMS-8x`

可以直接使用这些模型进行推理，无需自己 retrofit。

### 7.4 Prefill 阶段优化

DMS 的一个相比 DMC 的优势是可以加速 prefill 阶段：
- 在 prefill 时就执行 eviction decision
- 被标记驱逐的 token 在 prefill 的后续 chunk 中可以跳过计算
- 这对长 prompt 场景（如 RAG、长文档问答）尤其有价值

### 7.5 动态压缩比

实际部署中可以根据场景动态调整压缩比：
- 简单任务（短对话）: 低压缩或不压缩
- 复杂推理任务: 高压缩 + 更多 parallel chains
- 长文档场景: 中等压缩，确保长距离依赖不丢失

### 7.6 Serving 框架改动清单

要在生产环境中落地 DMS，主要需要以下改动：

1. **Model 层**: 修改 attention forward，增加 eviction prediction（~10 行代码/layer）
2. **KV Cache Manager**: 实现 per-head variable-length cache + delayed eviction queue
3. **Scheduler**: 利用压缩后更小的 cache footprint，增加 batch size 或 parallel chains
4. **Kernel**: 支持稀疏 KV cache 的高效 attention 计算（可复用 FlexAttention 等现有工具）

---

## 8. 面试高频题

### 题目 1: DMS 为什么只需要 ~1K 步训练就能达到 8× 压缩？与 DMC 相比效率提升的根本原因是什么？

**回答要点**:

DMS 训练效率高的根本原因在于三个设计选择：

1. **零额外参数**: DMS 复用模型已有的 query/key 投影矩阵中的一个 neuron 来预测 eviction decision，不引入任何新参数。这意味着模型需要学习的不是一个新能力，而是微调一个已有的投影方向。相比之下 DMC 需要为每个 head 添加额外的决策参数并学习 merge 操作。

2. **任务简单性**: DMS 学习的是一个二元分类任务（保留 or 驱逐），而 DMC 学习的是一个更复杂的决策（如何加权合并 token）。二元分类的参数空间远小于连续混合权重。

3. **安全训练动态**: 通过 b=-5 初始化（训练初期 α≈0，即不驱逐任何 token）和线性退火的压缩目标（每 100 步增加 1× 压缩比），模型从"完全不压缩"平滑过渡到"目标压缩比"，避免了 loss spike 和灾难性遗忘。

4. **Logit Distillation**: 使用原始模型作为 teacher 的 soft label 训练，比标准 language modeling loss 更稳定，且不需要精确复制预训练数据分布。

本质上，DMS 证明了**稀疏化（直接丢弃）比压缩（合并）更容易学习**——只要配合延迟驱逐机制让模型有时间"消化"即将被丢弃的信息。

---

### 题目 2: Delayed Eviction（延迟驱逐）的机制是什么？为什么它对保持高压缩比下的精度至关重要？

**回答要点**:

**机制描述**:
- 在时间步 t，模型预测 token t 的驱逐决策 α_t
- 如果 α_t = 1（驱逐），该 token 的 KV pair 不会立即被删除
- 它会在接下来的 w 步内（滑动窗口 w=256）仍然参与 attention 计算
- 直到时间步 t+w 时才被真正从 cache 中移除

**训练时的实现**:
通过加性注意力掩码 M_α 实现渐进效果：被标记驱逐的 token 在滑动窗口内获得 log(1-α) 的 partial mask（α 接近 1 时接近 -∞），窗口外获得 -∞ 的 full mask。

**为什么重要**:
1. **Recency Bias**: Decoder-only 模型高度依赖近期 token 的 attention。如果立即驱逐刚写入的 token，模型还来不及从中提取信息。
2. **隐式信息传递**: 在延迟窗口内，后续 token 可以通过 attention 机制"吸收"即将被驱逐的 token 的信息，并将其编码到自身的表示中。这相当于一种隐式的信息合并。
3. **消融实验证据**: 论文明确展示了 w=0（立即驱逐）在所有 cache 大小设置下都导致快速性能退化，而 w=256 的延迟驱逐维持了稳定的训练和高精度。

**面试加分点**: 可以类比人类记忆——与其突然忘记一个信息，不如在知道即将遗忘时先把关键要点"总结"到当前思维中。延迟驱逐给了模型这个"总结"的缓冲期。

---

### 题目 3: 在实际部署中，DMS 如何与 Inference-Time Scaling（如 Best-of-N）策略配合使用？请分析一个具体的部署场景。

**回答要点**:

**场景**: 部署一个 32B reasoning 模型用于数学竞赛题解答，服务器有 2×A100 80GB。

**不用 DMS 的方案（Baseline）**:
- KV cache 占用约 ~40GB（32K 序列长度 × 32B 模型）
- 只能跑 1-2 条并行推理链
- Majority voting 的 candidate pool 太小，效果有限
- AIME 24 准确率约 54%

**使用 DMS 8× 的方案**:
- KV cache 压缩到 ~5GB
- 同等显存下可以跑 8-16 条并行推理链
- Majority voting 从 8-16 个 candidate 中选最终答案
- AIME 24 准确率提升到约 66%（+12 points）

**具体配合方式**:
1. **并行生成**: 对同一问题启动 N=16 条独立推理链，每条链用 DMS 8× 压缩
2. **Majority Voting**: 提取每条链的最终答案，取多数票
3. **或 Reward Model Reranking**: 用 PRM/ORM 对每条链打分，选最高分的答案

**Latency 分析**:
- 单条链的延迟：DMS 减少了 KV cache 读取量 8×，attention 阶段加速约 5-7×（考虑非 attention 部分的占比）
- 但并行 16 条链需要更多计算 → 总延迟取决于 GPU 利用率
- 在 memory-bound 场景下（长序列），DMS 的延迟优势非常显著

**面试加分点**:
- 提及 DMS 可以与量化叠加（8× sparse + 4bit quant = 32× 理论压缩）
- 提及不同任务可以用不同压缩比（简单对话 1×-2×，复杂推理 8×）
- 提及 DMS 在 prefill 阶段也能加速，对 RAG 场景（长 prompt）尤其有价值
- 提及 NVIDIA 已开源 DMS 模型（如 `nvidia/Qwen3-8B-DMS-8x`），可以直接使用

---

## 附录

### A. 关键数字速记

| 指标 | 数值 |
|------|------|
| 训练步数 | ~1K steps |
| 最大压缩比 | 8× |
| 滑动窗口大小 | 256 tokens |
| 额外参数量 | 0 |
| AIME 24 提升 (32B) | +12.0 points |
| GPQA 提升 (32B) | +8.6 points |
| LiveCodeBench 提升 (32B) | +9.7 points |
| 短上下文精度损失 (4×) | ~3.5 points |

### B. 论文时间线

- 2025-06-05: v1 上传 arXiv
- 2025-06-11: 媒体报道（MarkTechPost, VentureBeat 等）
- 2025-11-07: v2 更新（NeurIPS camera-ready）
- NeurIPS 2025 接收

### C. 相关工作速查

| 方法 | 类型 | 压缩方式 | 需要训练 | 减少显存 | 减少延迟 |
|------|------|---------|---------|---------|---------|
| H2O | Sparse | 驱逐低 attention token | ❌ | ✅ | ✅ |
| TOVA | Sparse | 驱逐低 attention token | ❌ | ✅ | ✅ |
| StreamingLLM | Sparse | Sink + 滑动窗口 | ❌ | ✅ | ✅ |
| Quest | Retrieval | 选择性读取 page | ❌ | ❌ | ✅ |
| DMC | Compression | 加权合并 token | ✅（贵） | ✅ | ❌ prefill |
| **DMS** | **Sparse** | **Learned 延迟驱逐** | **✅（便宜）** | **✅** | **✅** |
| KV Quantization | Quantization | INT8/INT4 | 可选 | ✅ | ✅ |
| MLA | Architecture | 低秩投影 | ✅（从头训） | ✅ | ✅ |
| PagedAttention | Memory Mgmt | 分页管理 | ❌ | 减少碎片 | ❌ |

### D. 思考题

1. **DMS + MLA**: 如果将 DMS 应用于已经使用 MLA 的模型（如 DeepSeek-V2），效果会叠加吗？MLA 的低秩 KV 表示是否让 eviction 更困难（因为每个 KV 已经是压缩后的）？

2. **Adaptive CR**: 论文中压缩比在训练时固定。是否可以设计一个在推理时动态调整 CR 的机制？比如当检测到"困难段落"时降低压缩比？

3. **Speculative Decoding 结合**: DMS 的 8× 压缩释放的显存可以用来加载一个 draft model 做 speculative decoding，这样的组合是否比单独使用任一方法更优？

---

## See Also

- [[AI/3-LLM/Inference/KV Cache|KV Cache 优化]] — KV Cache 优化全貌
- [[AI/3-LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow 推测解码]] — 推理加速的另一路线
- [[模型量化综述|模型量化综述]] — 量化与 KV Cache 压缩的结合
-  — 大语言模型知识全图谱
