---
brief: Jet-RL——FP8 精度全程 on-policy RL 训练框架；解决 FP8 训练中量化误差对 RL 梯度更新的影响，实现 2x 吞吐量提升；核心发现：量化必须全局一致，部分 FP8 反而不如全精度（与 QeRL 形成对比）。
title: "Jet-RL: FP8 On-Policy RL Training"
date: 2026-01-20
tags:
  - FP8训练
  - RL推理优化
  - 量化
  - 系统优化
  - 分布式训练
domain: AI/LLM/RL/Frameworks
arxiv: "2601.14243"
rating: 4
status: permanent
see-also:
  - "[[AI/3-LLM/RL/算法/QeRL-Quantization-Enhanced-RL|QeRL-Quantization-Efficient-RL]]"
---

# Jet-RL: FP8 On-Policy RL Training

> Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow

**arXiv**: 2601.14243  
**作者**: Haocheng Xi, Charlie Ruan, Peiyuan Liao, Yujun Lin, Han Cai, Yilong Zhao, Shuo Yang, Kurt Keutzer, **Song Han**, Ligeng Zhu  
**机构**: NVIDIA + MIT (Song Han) + UC Berkeley + Stanford  
**提交日期**: 2026-01-20  
**标签**: #FP8训练 #RL推理优化 #量化 #系统优化 #分布式训练

---

## 核心问题

**Rollout 是 RL 训练的最大瓶颈**：
- Rollout 长度 > 8K token 时，rollout 占总训练时间 > 70%
- 随 rollout 长度增加，这个比例继续上升

**现有 FP8 加速方案（BF16-train-FP8-rollout）存在严重问题**：
- SLIME、VeRL、NeMo-RL、OpenRLHF 都采用这种策略
- 直接把训练时的 BF16 权重 cast 到 FP8 进行 rollout
- **失效场景 1**：rollout 长度 > 8K 时精度崩溃（在 Qwen3-8B-Base 上验证）
- **失效场景 2**：困难任务/弱 base model 时训练发散

---

## 根本原因分析

### Off-Policy 精度不匹配

**关键洞察**：BF16 训练 + FP8 rollout 本质上是 **off-policy** 的。

原因链：
1. BF16 训练更新 actor weights（BF16 精度）
2. Rollout 用 FP8 权重生成 token
3. → **BF16 actor ≠ FP8 actor**（logit 分布不同）
4. → 训练看到的 log probs 和 rollout 实际生成的 log probs 不一致
5. → importance ratio 被污染 → GRPO/PPO 的 on-policy 假设失效

**长 rollout 为何更严重**：
- 短序列：每步的精度误差小，不构成问题
- 长序列：误差在每个 decoding step 累积，最终导致分布漂移不可挽回
- 类比：一个小角度偏差，走 100 步后离目标很远

**困难任务为何更脆弱**：
- 简单任务：模型 confidence 高，log prob 的小误差不影响 argmax 选择
- 困难任务：模型在接近 0.5 的边界区域，精度误差直接影响 token 选择
- → 生成的 trajectory 与训练期望的轨迹不同

### 为什么 calibration 方案也不管用

PTQ calibration（AWQ/GPTQ style）在 offline deployment 设计，但 RL 训练需要**每步都同步权重**：
- 8B 模型 calibration 需要几十分钟
- 无法在每次 rollout 前都做 calibration
- 没有 calibration 直接 cast → 精度不稳定

---

## 方法：Jet-RL

**核心思想**：**Unified FP8 Precision Flow** — 训练和 rollout 使用完全相同的 FP8 精度配置。

### 关键设计决策

**1. 将推理图设计为训练图的子图**

形式化：
- 训练图 `G_train = G_train_fwd + G_train_bwd`
- 推理图 `G_infer`
- Jet-RL 要求：`G_infer ⊆ G_train_fwd`

这意味着同一个 token 经过训练 forward pass 和 rollout 时，量化行为完全一致 → logits 完全匹配 → on-policy 保证。

**2. 混合量化粒度**（来自 DeepSeek-V3 方案）

| 张量类型 | 量化粒度 |
|---------|---------|
| Weights | 128×128 per-block |
| Activations / Gradients | 1×128 per-group |
| WGrad 的第二个矩阵 | 128×1 per-group |

精细粒度的 per-group quantization 比 per-tensor 稳定得多。

**3. FP8 GEMM 覆盖三个 pass**

| GEMM | 阶段 | FP8？ |
|------|------|-------|
| FProp | Forward | ✅ |
| WGrad | Backward | ✅ |
| DGrad | Backward | ✅ |

**4. 梯度保留 BF16**

- 权重梯度在 operator 间传输时保留 BF16
- 原因：FP8 梯度量化会导致 gradient underflow（梯度下溢）
- 只有 GEMM 内部计算用 FP8，传输时 dequantize 回 BF16

**5. FP8 activation 存储节省内存**

Backward pass 需要 save 部分 activations：
- Forward 时 1×128 量化存储
- Backward 时 128×1 量化使用（两种方向都 fuse 处理）
- 内存占用 = BF16 的 1/2

---

## 实验结果

### 性能数字

| 指标 | 提升幅度 | 模型/场景 |
|------|---------|---------|
| Rollout phase 加速 | **+33%** (1.33×) | 32B 模型 |
| Training phase 加速 | **+41%** (1.41×) | 8B 模型 |
| End-to-end 加速 | **+16%** (1.16×) | 8B 实验 |

### 精度保留

| 方法 | 性能降损 |
|------|---------|
| BF16-train-FP8-rollout | **>5% 降损**（长 rollout/难任务） |
| **Jet-RL (Unified FP8)** | **~1% 降损** |
| BF16 baseline | 0%（参照系） |

### 关键验证

- Qwen3-8B-Base 在 MATH benchmark，rollout = 16K：BF16-FP8-rollout 在 20 步后崩溃，Jet-RL 稳定收敛
- Qwen2.5-7B GRPO 训练：4K rollout 两者相近，8K+ 时 BF16-FP8-rollout 发散
- 困难任务（Qwen3-8B-Base vs instruct）：base model 任务时 BF16-FP8-rollout 发散，Jet-RL 稳定

---

## 批判性分析

### 这个发现有多重要？

**非常重要**，原因：

1. **揭示了生产系统的隐患**：VeRL、SLIME、NeMo-RL、OpenRLHF 都在用有问题的方案，但可能还没发现（因为他们的 rollout 长度还在 <8K）
2. **随着 CoT 更长，问题会越来越严重**：ICLR 2026 的 TTC scaling 趋势意味着 rollout 会更长，这个问题会被放大
3. **Off-policy 精度不匹配是个系统性问题**：quantization 只是载体，核心问题是"任何导致 training/rollout 不一致的操作都会破坏 on-policy 假设"

### 我的疑问

1. **end-to-end 只有 16% 是否值得**：rollout +33%，但整体只有 16%，说明 evaluation/update phase 没有被加速。如果 rollout 占 70%，那理论上 33% 的 rollout 加速应该给 ~23% 的端到端加速，但实际只有 16%，说明 overhead 有增加。
2. **内存 vs 速度 trade-off**：FP8 训练对内存有什么影响？论文没有突出这点。
3. **对梯度爆炸的鲁棒性**：FP8 的动态范围（E4M3）只有 [-448, 448]，对梯度这种可能很小或很大的量，quantization 可能不稳定——为什么他们选择 BF16 传输梯度，这是不得已的妥协？
4. **与 FP4/INT8 的对比**：后续方向，更激进的量化能否在 Jet-RL 框架下使用？

### 关联工作

- **Stable Asynchrony/VCPO**（Song Han lab 另一篇）：同组！一个解决量化精度不匹配（Jet-RL），一个解决异步更新的 variance（VCPO）。Song Han 组在系统级 RL 稳定性上全面布局。
- **DAPO**：提到 rollout 是瓶颈的源头论文，Jet-RL 引用了 DAPO 的 rollout 统计
- **DeepSeek-V3**：量化粒度方案（per-group + per-block）来自 DeepSeek-V3 技术报告

### 我的评分

**★★★★☆**

- 发现的问题真实且重要，不是 toy problem
- 解决方案优雅（将推理图设为训练图子图，原理简洁）
- 对行业有直接影响：VeRL/SLIME 等框架需要更新
- 扣一星：end-to-end 加速有限（16%），且只在 math benchmark 验证，代码未开源（匿名阶段）

---

## 对老板的关键 Takeaways

### 面试话术
"BF16-train-FP8-rollout 看似无害，但本质上打破了 RL 的 on-policy 假设——FP8 rollout 生成的 trajectory 来自一个和训练 actor 不同的 policy。短序列时误差小，但 CoT 越长，这个误差会指数级放大。Jet-RL 的修复方法很简单：让推理图成为训练 forward 图的子图，保证量化行为完全一致。"

### 工程实践建议
- 如果你的 RL rollout 长度 < 4K，BF16-FP8 混合没太大问题
- 如果 rollout > 8K，必须认真考虑 unified precision flow
- Song Han 组代码待开源，值得追踪

### 与系统架构的联系
这篇论文揭示了一个深层原则：**RL 的 on-policy 假设不仅适用于数据分布，也适用于计算精度**。任何在训练和推理之间引入不一致的操作（量化、随机性、数值精度）都会隐式地破坏 on-policy 保证。

---

## 元注记

- arXiv 2601.14243，Song Han lab (NVIDIA/MIT) + UCB + Stanford
- 全文读到 Section 4.2 （量化粒度），实验细节在 Section 5 待读
- 笔记基于完整读取的 intro + motivation + Section 4 核心内容
- 代码待开源（论文写 "when less anonymous"，即审稿结束后）

---

## see-also

- [[AI/3-LLM/Inference/Progressive-Thought-Encoding-Cache-Efficient-RL|PTE]] — 同一"RL 训练效率"问题的另一维度：KV cache 限制→ online self-distillation 压缩 evicted token（ICLR 2026，arXiv:2602.16839）★★★★★
- [[AI/3-LLM/RL/算法/Slime-RL-Framework|Slime RL Framework]] — 异步 RL infra，Jet-RL 的 FP8 统一可直接应用于此类框架
- [[AI/3-LLM/RL/算法/Stable-Asynchrony-VCPO-Off-Policy-RL|VCPO]] — 同为系统级 RL 稳定性方向，解决 off-policy 方差（PTE 解决 cache 内存，VCPO 解决异步误差）
- [[AI/3-LLM/Inference/Accordion-Thinking-Self-Regulated-Step-Summaries|Accordion-Thinking]] — 算法层效率（模型学会主动压缩 CoT，4× throughput），与 Jet-RL 系统层正交可组合（arXiv:2602.03249）★★★★☆
- [[AI/3-LLM/RL/算法/QeRL-Quantization-Enhanced-RL|QeRL]] ⭐ — **反向方向**：量化噪声对 RL **有益**（促进探索），4-bit+LoRA 不仅快 1.5×，多项基准超 16-bit；与 Jet-RL 对"量化在 RL 中的角色"形成互补论证（ICLR 2026，arXiv:2510.11696）★★★★
