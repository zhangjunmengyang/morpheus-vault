---
title: "Speculative Decoding 深度解析"
brief: "推测解码：用小 Draft Model 快速生成候选 token，大 Target Model 并行验证，等效降低大模型自回归延迟 2-3×。接受率 α 是核心指标（推测正确才保留）。Medusa/EAGLE/Jacobi Decoding 等变体进一步优化。面试必考推理加速技术。"
tags: [llm, inference, speculative-decoding, optimization, interview-prep]
created: 2026-02-14
status: draft
---

> [!info] 另有面试版
> Foundations 精简版：[[AI/1-Foundations/Inference/Speculative Decoding]]

# Speculative Decoding（投机采样）

## 核心思想

Speculative Decoding 是一种 LLM 推理加速技术，核心思想是让小模型（draft model）快速生成多个候选 token，然后大模型（target model）并行验证这些候选。这样可以充分利用 GPU 的并行计算能力，显著减少推理时的序列化步骤。

传统的自回归解码每次只能生成一个 token，而 Speculative Decoding 通过"猜测-验证"的两阶段模式，理论上可以一次处理多个 token，实现 **2-3x** 的加速比。

### 工作流程

1. **Draft 阶段**：小模型快速生成 K 个候选 token 序列
2. **Verification 阶段**：大模型并行计算所有候选位置的概率分布
3. **Acceptance 阶段**：通过 rejection sampling 决定接受多少个 token

关键是这个过程与 [[AI/3-LLM/Inference/采样策略|采样策略]] 完全等价，不会改变最终的输出分布。

## 数学保证：Rejection Sampling

### 核心定理

设大模型的真实分布为 $p(x_t|x_{<t})$，小模型的近似分布为 $q(x_t|x_{<t})$，通过以下 rejection sampling 过程可以保证输出分布不变：

对于候选 token $x_t$：
- **接受概率**：$\alpha(x_t) = \min(1, \frac{p(x_t|x_{<t})}{q(x_t|x_{<t})})$
- **修正采样**：若拒绝，从修正分布 $\tilde{p}(x_t) = \frac{\max(0, p(x_t) - q(x_t))}{\sum_{x'} \max(0, p(x') - q(x'))}$ 重新采样

### 数学推导

证明这种采样等价于直接从 $p(x_t|x_{<t})$ 采样：

对于任意 token $x_t$，被选中的总概率为：
$$P(\text{选中} \, x_t) = q(x_t) \cdot \alpha(x_t) + \left(1 - \sum_{x'} q(x') \alpha(x')\right) \cdot \tilde{p}(x_t)$$

当 $q(x_t) \leq p(x_t)$ 时，$\alpha(x_t) = 1$，$\tilde{p}(x_t) = \frac{p(x_t) - q(x_t)}{1 - \sum_{x'} q(x')}$

代入可得：
$$P(\text{选中} \, x_t) = q(x_t) + (1 - \sum_{x'} q(x')) \cdot \frac{p(x_t) - q(x_t)}{1 - \sum_{x'} q(x')} = p(x_t)$$

这保证了 **数学上的等价性**。

## 主要变体

### Medusa：多头并行预测

Medusa 在原模型基础上添加多个"medusa heads"，每个头预测不同未来位置的 token。优势是：
- **无需额外模型**：直接扩展现有模型
- **并行预测**：一次前向传播预测多个位置
- **Tree-based sampling**：构建候选树而非序列

与 [[AI/3-LLM/Inference/KV Cache|KV Cache]] 配合时需要小心处理分支状态。

### Eagle/Eagle-2：自回归草稿头

Eagle 系列使用轻量级的自回归头作为 draft model：
- **特征级预测**：重用主模型的中间特征
- **更好的 acceptance rate**：draft model 与 target model 共享表示
- **Eagle-2** 进一步优化了训练策略和架构设计

### Lookahead Decoding：无需 Draft Model

基于 Jacobi 迭代的思想，利用模型的并行预测能力：
- **无外部依赖**：只需原模型
- **Jacobi 固定点**：通过迭代收敛到正确序列
- **适合长序列**：特别在生成任务中表现良好

### SPEED：稀疏化加速

通过稀疏化技术减少计算量：
- **稀疏注意力**：只计算重要的注意力权重
- **动态剪枝**：运行时决定计算路径
- **与 speculative decoding 正交**：可以叠加使用

## 加速比分析

### 理论加速比

设候选长度为 K，acceptance rate 为 α，则期望加速比为：
$$\text{Speedup} = \frac{K \cdot \alpha + (1-\alpha)}{1 + \gamma \cdot K}$$

其中 γ 是 draft model 与 target model 的成本比。

### 实际性能

- **典型场景**：2-3x speedup
- **影响因素**：
  - Draft model 质量（影响 acceptance rate）
  - 任务类型（代码生成 > 对话 > 创意写作）
  - 序列长度（长序列效果更好）

### Acceptance Rate 分析

关键指标是平均接受的 token 数：
- **高质量 draft model**：acceptance rate > 80%
- **模型大小差距过大**：acceptance rate 下降
- **领域适配**：在相关领域微调 draft model 可提升效果

## 兼容性挑战

### 与 Continuous Batching 的冲突

[[AI/3-LLM/Inference/推理服务架构|推理服务架构]] 中的 continuous batching 假设所有请求按 token 同步推进，但 speculative decoding 会导致：
- **不同步进度**：某些请求可能一次处理多个 token
- **调度复杂性**：需要重新设计 batch 调度算法
- **内存管理**：KV cache 的增长模式不再可预测

### KV Cache 挑战

与 [[AI/3-LLM/Inference/KV Cache|KV Cache]] 的集成面临：
- **分支状态管理**：需要为候选序列维护多个 cache 分支
- **内存开销**：worst-case 内存需求显著增加
- **cache 一致性**：rejection 后需要正确回退 cache 状态

### 解决方案

1. **Adaptive batching**：动态调整 batch 内的同步策略
2. **Layered cache**：分层管理确定和候选状态
3. **Hybrid scheduling**：根据请求特性选择是否启用 speculative decoding

## 面试常见问题

### Q1: Speculative Decoding 为什么不会改变输出分布？

**A**: 通过 rejection sampling 的数学保证。核心是接受概率 α(x) = min(1, p(x)/q(x))，被拒绝时从修正分布重新采样。这确保了每个 token 的最终选中概率仍然等于大模型的原始概率 p(x)。

### Q2: 什么情况下 Speculative Decoding 效果最好？

**A**: 
- Draft model 与 target model 的分布差距不大（高 acceptance rate）
- 生成较长序列（摊薄 overhead）
- 计算密集型任务（充分利用并行性）
- GPU 内存充足（可以同时加载两个模型）

### Q3: 如何选择合适的 draft model？

**A**: 考虑三个维度：
- **质量**：在目标任务上的分布匹配程度
- **速度**：推理延迟应显著低于 target model
- **兼容性**：词表、tokenizer 需要兼容

通常选择同系列的小模型，或者专门训练的轻量级版本。

### Q4: Speculative Decoding 的主要开销在哪里？

**A**: 
- **内存开销**：需要同时加载两个模型
- **KV Cache 管理**：维护多分支状态的复杂性
- **调度开销**：rejection sampling 的额外计算
- **通信开销**：分布式部署时的模型间通信

### Q5: 在生产环境中部署的主要挑战？

**A**: 
- **服务稳定性**：rejection 导致的不确定延迟
- **资源调度**：与现有 batching 策略的兼容性
- **监控指标**：需要新的性能监控维度（acceptance rate 等）
- **成本控制**：双模型部署的资源成本

---

## 相关笔记

- [[AI/3-LLM/Inference/KV Cache|KV Cache]] - 缓存机制与内存优化
- [[AI/3-LLM/Inference/推理服务架构|推理服务架构]] - 生产环境的服务设计
- [[AI/3-LLM/Inference/采样策略|采样策略]] - 各种解码算法
- [[模型并行策略|模型并行策略]] - 大模型分布式推理
- [[推理优化|推理优化]] - 其他加速技术