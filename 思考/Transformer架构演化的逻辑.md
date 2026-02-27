---
title: Transformer 架构演化的逻辑
type: 思考
date: 2026-02-26
tags:
  - ai/llm/architecture
  - 思考
  - Transformer
  - Attention
  - MoE
  - SSM
---

# Transformer 架构演化的逻辑

> Transformer 从 2017 到 2026 的每一次演化，都不是发明新东西，而是在解决注意力机制的某个具体缺陷。理解这条逻辑链，比记住任何单个技术都重要。

---

## 一、Attention 的原罪与三条演化主线

Attention 的核心公式 `softmax(QK^T/√d)V` 简洁优美，但藏着三个原罪：

1. **O(N²) 计算和显存** —— 序列越长越爆
2. **KV Cache 线性增长** —— 推理时每个 token 的历史都要缓存
3. **全连接 = 无差别计算** —— 每个 token 都花同样的算力，不管难度

2017 年之后的每一项重大架构创新，都是在治这三个病：

```
原罪①: O(N²)
  → FlashAttention（不改复杂度，改 IO）
  → Sparse Attention / Sliding Window（真砍计算）
  → SSM/Mamba（彻底换架构，O(N)）

原罪②: KV Cache 爆炸
  → MQA / GQA（减少 KV head）
  → MLA（低秩压缩 + absorption trick）
  → KV Cache 量化 / Sliding Window

原罪③: 无差别计算
  → MoE（条件选择 expert）
  → Mixture of Depths（条件选择是否计算）
  → Token Merging（合并冗余 token）
```

---

## 二、注意力本身的优化：不是算得更少，是搬得更少

### FlashAttention：一个被广泛误解的技术

FlashAttention **没有降低计算复杂度**——仍然是 O(N²d)。它降低的是 IO 复杂度。

核心做法：把 Q、K、V 分成小块，全部在 GPU SRAM（~20MB）里完成 attention 计算，中间那个 N×N 的 attention 矩阵**永远不写回 HBM**。技术手段是 online softmax——不需要看到完整一行就能增量地算 softmax。

为什么这能加速 2-4×？因为 attention 是 memory-bound 操作。标准实现里，光是把 N×N 矩阵在 HBM 里来回搬就吃掉了大部分时间。FlashAttention 把这个搬运量砍掉了。

**这给我的启示：** 很多时候瓶颈不是算力，是数据搬运。优化 IO 比优化 FLOPS 更重要。

### Sparse Attention：真的砍计算

FlashAttention 让 O(N²) 跑得很快，但物理极限在那里。真要处理 128K+ 序列，必须降复杂度。

- **Sliding Window**（Mistral）：每个 token 只看前 W 个邻居。简单粗暴，但配合多层堆叠后感受野 = L×W（32 层 × 4K = 128K）。
- **Attention Sink + Sliding Window**（StreamingLLM）：保留前 4 个 token + 滑动窗口，支持理论无限长推理。关键发现：前几个 token 承担了 attention 的「锚点」功能，丢不得。
- **动态稀疏**（DeepSeek-V3）：不按位置决定看谁，按内容决定——先用少量锚点估计注意力分布，再只对 top-k 重要的 KV 做精确 attention。比固定模式更聪明。

**我的判断：** Sliding Window + Attention Sink 是长序列推理的实用选择。动态稀疏是更好的方向，但工程实现复杂，目前只有 DeepSeek 级别的团队在做。

---

## 三、KV Cache 压缩：从 GQA 到 MLA 的三级跳

这是推理成本的核心战场。KV Cache 的大小直接决定你能并发多少请求，也就是决定你的服务器能赚多少钱。

**演化路径和内在逻辑：**

| 阶段 | 方法 | 思路 | KV Cache 压缩 | 质量影响 |
|------|------|------|--------------|---------|
| 原始 | MHA | 每个 head 独立 K、V | 1× (baseline) | — |
| 第一步 | MQA | 所有 head 共享 1 组 K、V | h× (如 32×) | 明显下降 |
| 平衡点 | GQA | 分组共享（LLaMA-2 用 g=8） | h/g× (如 4×) | 接近 MHA |
| 飞跃 | MLA | 低秩压缩 + absorption trick | ~28× | 不降反升 |

**MLA 为什么是飞跃？**

GQA 的思路是「减少冗余 head」——直觉上合理，但本质是在表达力和显存之间线性 trade-off。

MLA 换了一个思路：不缓存 K 和 V 本身，缓存一个低维 latent vector（512 维 vs MHA 的 16384 维）。推理时把恢复矩阵 W^UK 吸收进 query 投影，直接在压缩空间做 attention——**根本不需要展开 K**。

这是数学上的优雅，不是工程上的妥协。但有一个实际问题：RoPE 和 absorption trick 不兼容（旋转矩阵无法被吸收），DeepSeek 的解法是 decoupled RoPE keys——额外分出 64 维专门承载位置信息。

**我的判断：** MLA 是 KV Cache 压缩的当前最优解。但它和 DeepSeek 的整体架构深度耦合，其他模型不太容易照搬。对大多数团队来说，GQA + KV Cache INT8 量化是最实际的组合。

---

## 四、MoE：不是「多个专家」，是条件计算

MoE 的本质经常被表面理解错。它不是「多个专家投票」，它是**条件计算（conditional computation）**——根据输入内容决定激活哪部分参数。

**为什么这重要？** 因为它打破了「参数量 = 计算量」的绑定。DeepSeek-V3 总参数 671B，但每个 token 只激活 37B——用 670B 的容量装知识，用 37B 的成本做推理。

### 演化的内在逻辑

| 阶段 | 方法 | 核心改进 |
|------|------|----------|
| 早期 | Switch Transformer (top-1) | 极简路由，但 expert 坍塌严重 |
| 稳定 | GShard (top-2 + auxiliary loss) | 用 loss 强制均衡，但干扰训练 |
| 反转 | Expert Choice | expert 选 token 而非 token 选 expert，天然均衡 |
| 精细化 | DeepSeek-MoE (256 小 expert + shared expert) | 细粒度知识组合 + 通用基线 |
| 去噪 | DeepSeek-V3 (bias-based balancing) | 用 bias 调整路由，不污染梯度 |

**MoE 的核心难题是负载均衡。** 少数 expert 过载 → 成为延迟瓶颈 → expert 坍塌。三代解法的演进很清晰：auxiliary loss（有效但干扰训练）→ expert choice（天然均衡但 token 覆盖不确定）→ bias-based（不干扰梯度）。

DeepSeek-V3 的方案是目前最优雅的：一个可学习的 bias 项动态把 token 引向负载不足的 expert，完全不影响 gating 的梯度信号。加上 shared expert（始终激活，保底通用知识），整个系统既灵活又稳定。

---

## 五、SSM/Mamba：O(N) 的诱惑和代价

Mamba 用 O(N) 的序列建模替代 O(N²) 的 attention。核心创新是 **Selective SSM**——让状态转移矩阵依赖输入，实现「选择性记忆」（类比 LSTM 的门控，但可以并行 scan）。

**Mamba 的硬伤：** 固定维度的隐状态是信息瓶颈。它是序列历史的有损压缩——无法精确回忆任意历史 token。在 needle-in-a-haystack 测试中，Mamba 明显弱于 Attention。

**2026 的共识是混合架构：** Jamba 用 3:1 的 Mamba:Attention 比例——大部分层用 Mamba 高效处理长序列，每 4 层插一个 Attention 层做精确检索兜底。

**我的判断：** 纯 SSM 替代 Attention 是个伪命题。真正的问题是「什么时候需要精确检索，什么时候压缩记忆就够了」。混合架构是正确方向，但最优混合比可能因任务而异——代码生成可能需要更多 Attention 层（精确 copy），对话可能更偏 SSM。

---

## 六、2026 前沿：条件计算的深化

三个值得关注的方向，都是条件计算思想的延伸：

**Mixture of Depths（MoD）：** MoE 选择「用哪个 expert」，MoD 选择「是否计算」。每层用一个 routing score 决定哪些 token 走子层计算、哪些直接 skip。不同 token 难度不同——标点和常见词可能不需要经过每一层。潜力：相同性能下 FLOPs 减少 12-50%。

**Hyper-Connections：** 标准残差连接只有一条恒等路径，深层时浅层信息被稀释。Hyper-Connections 把隐状态展开成多个「分身」，用可学习的矩阵控制信息流——浅层信息可以通过专用通道直达深层。类比 DenseNet 但更轻量。

**Token Merging：** 很多 token 经过几层后变得高度相似——信息冗余。动态合并相似 token，减少后续计算量。在 VLM 中特别有价值（图像 token 通常大量冗余）。

---

## 七、底层零件的收敛

架构演化不只在顶层设计，底层组件也在收敛到最优解：

| 组件 | 2017 | 2026 共识 | 为什么 |
|------|------|-----------|--------|
| 归一化 | Post-LayerNorm | Pre-RMSNorm | 梯度直通 + 省去 mean centering |
| 激活函数 | ReLU → GELU | SwiGLU | 门控选择性过滤 + Swish 平滑非单调 |
| 位置编码 | Sinusoidal / Learned | RoPE + YaRN | 相对位置 + 长度可扩展 |
| FFN 宽度 | 4d | 8d/3（SwiGLU 补偿） | 三矩阵结构参数量对齐 |
| 训练精度 | FP32 | BF16 forward + FP32 optimizer | 范围够大，无需 loss scaling |

**RoPE 的长度外推** 值得特别说：RoPE 理论上对任意位置有定义，但实际超出训练长度就崩。YaRN 的解法是分频段处理——高频维度（编码局部关系）不动，低频维度（编码远程关系）做插值缩放——400 步微调就能从 4K 扩展到 128K+。

---

## 八、我的判断：架构演化的 Meta-Pattern

看完整条演化线，我认为有三个 meta-pattern：

**1. 所有优化的底层逻辑只有两个：减少内存访问 + 条件计算。**

FlashAttention、KV Cache 压缩、量化——都是减少数据搬运。MoE、MoD、Sparse Attention、Token Merging——都是条件计算。理解这两个方向，新技术出来就知道它属于哪一类、解决什么问题。

**2. 工程妥协和架构革命的边界在模糊。**

GQA 是工程妥协（牺牲表达力换显存）。MLA 是架构革命（用数学技巧同时拿到两头）。FlashAttention 是纯工程（不改模型，改实现）。Mamba 是架构替代（换掉 attention）。但最终胜出的方案几乎都是——**用工程方法把理论突破落地**。纯理论优美但无法高效实现的方案（如某些 linear attention 变体），都死在了这一步。

**3. Decoder-only + 条件计算 是当前的架构收敛方向。**

Encoder-Decoder 退场的原因不是它差，而是 Decoder-only 更统一、更适合 scaling。在 Decoder-only 的骨架上，MoE 负责「参数量 ≠ 计算量」，MoD 负责「不同 token 不同算力」，Sparse Attention 负责「不同距离不同精度」。这三者组合，就是 2026 年最强架构的样子——DeepSeek-V3 已经走在这条路上。

**下一步的猜测：** MLA + MoE + MoD 全组合，1T 参数、<50B 活跃——这可能是通向 AGI 级模型的可行路径。但关键瓶颈已经不在架构本身，而在训练数据和 RLHF 对齐。

---

> 🔗 Related: [[思考/LLM推理优化的本质|LLM 推理优化的本质]] · [[AI/3-LLM/Architecture/MoE 深度解析|MoE 深度解析]] · [[AI/3-LLM/Architecture/FlashAttention|FlashAttention]] · [[AI/3-LLM/Architecture/_MOC|Architecture MOC]]
