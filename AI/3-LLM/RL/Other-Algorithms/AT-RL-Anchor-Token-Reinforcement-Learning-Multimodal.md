---
brief: "AT-RL（Anchor Token RL）——识别跨模态连接的关键锚点 token 并集中信用分配；解决多模态 RLVR 中视觉 token 贡献被平均淡化的问题；相比 GRPO 均匀分配，anchor token 获得更高梯度权重，推理准确率提升。"
title: "AT-RL: Anchor Token Reinforcement Learning for Multimodal"
type: note
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - multimodal
  - credit-assignment
  - rlvr
  - type/paper
date: 2026-02-20
---

# AT-RL: Credit Where It is Due — Cross-Modality Connectivity Drives Precise RL for MLLM Reasoning

**arXiv**: 2602.11455  
**提交日期**: 2026-02-12  
**作者**: Zhengbo Jiao 等  
**评分**: ★★★★☆

---

## 一句话

多模态 RLVR 中只有约 **15% 的 token** 具有强视觉-文本耦合（"视觉锚点"）；这些锚点 token 才是 credit assignment 的真正载体。AT-RL 用图聚类识别并选择性强化这些锚点，32B 模型在 MathVista 上达到 80.2，超越 72B-Instruct baseline，仅增加 1.2% 开销。

---

## 核心问题

RLVR 对 MLLM 推理有显著提升，但一个基本问题未被回答：**推理过程中视觉证据是如何被整合的？**

具体来说：当给模型一张图片 + 数学题，生成 chain-of-thought 推理链时，推理链里哪些 token 真的在"看"图片，哪些只是在重复语言模式？

如果大多数 token 只是语言模式——那么对所有 token 施加相同的序列级奖励，实际上是在用大量"视觉无关 token"的梯度稀释少数"真正重要的视觉锚点"的信号。

---

## 关键发现

### 1. 视觉锚点 token 只占 ~15%

通过分析跨模态注意力连通性（cross-modal attention connectivity），发现：
- **~15% 的 token** 具有强视觉-文本耦合（high-connectivity tokens）
- 这些 token 作为"锚点"，将推理过程接地（grounded）到图像中
- **~85% 的 token** 主要遵循语言模式，视觉依赖度低

这个 15% 不是均匀分布的——它们集中在关键的视觉描述步骤（如"图中可见..."、"根据图表..."）而非推理展开步骤。

### 2. RLVR 训练自然强化锚点

在普通 RLVR 训练过程中，credit assignment **自然地**集中到这些锚点 token 上——梯度更新让它们的视觉接地变得更精确。但这个过程是隐式的，效率较低。

### 3. 消融实验：只训练低连通性 token → 严重退化

如果反过来，只对低视觉连通性的 token 施加强化学习更新，模型表现**严重退化**。

这个消融实验提供了强力证据：**多模态 RL 的关键不在于 token 数量，而在于正确识别并强化视觉锚点**。

---

## 方法：AT-RL

**Anchor-Token Reinforcement Learning** 的三个步骤：

1. **连通性分析**：计算每个 token 对视觉 token 的注意力强度，得到 cross-modal attention connectivity 矩阵
2. **图聚类**：用 attention topology 的图聚类方法识别高连通性 token 集合（视觉锚点）
3. **选择性强化**：在 RLVR 更新时，只对视觉锚点 token 施加强化，其他 token 正常梯度或降权

**开销**：仅 1.2% 额外计算（主要来自连通性计算）

---

## 实验结果

| Model | Baseline | AT-RL | 提升 |
|-------|---------|-------|------|
| 32B | 72B-Instruct (80.2是AT-RL的32B达到的) | 80.2 | 超越 72B-Instruct baseline |
| 3B-32B 系列 | - | 一致增益 | STEM / 视频 / 通用任务均有改善 |

**关键数字**：AT-RL 让 **32B 模型超越 72B-Instruct baseline**，说明精确 credit assignment 的效果等价于参数量翻倍的提升。

---

## 我的分析

### 与 GRPO 全景综述的直接连接

GRPO 全景综述里的核心论点：

> 所有 GRPO 改进都指向同一根因：序列级奖励训练 token 级决策，所有 token 被均匀对待，但实际高度异构。真正的解法是 token 级密集奖励。

AT-RL 是这个论点的**多模态实例化**：

- 在纯语言 RL 中：不同位置的 token 有不同的 credit（STAPO/MASPO/DEEP-GRPO 都在解决这个问题）
- 在多模态 RL 中：不同 modality 的 token 有不同的 credit——视觉锚点的梯度比纯语言 token 更关键

AT-RL 提供了一个**操作性的实现**：用注意力连通性作为 proxy 来识别高 credit token，然后选择性强化。

### 这比纯语言更简单在哪里？

多模态场景下识别"重要 token"比纯语言场景更容易：
- **视觉锚点**有明确的操作性定义（cross-modal attention connectivity）
- **可以直接测量**哪些文本 token 在关注视觉 token

在纯语言场景，"哪些 token 贡献了最终答案的正确性"没有直接的结构化信号——这是为什么 token 级 credit assignment 在纯语言 GRPO 中更难解决（需要额外的 value model）。

AT-RL 绕开了这个困难：**用跨模态注意力作为 credit 的 proxy**，不需要学习一个 value model。

### 质疑点

1. **15% 这个比例是否稳定？** 论文没有给出这个比例在不同任务/模型规模下的变化。数学推理和图像描述任务的视觉锚点分布可能差异很大。

2. **图聚类的超参数**：cluster 数量 / threshold 如何确定？注意力 topology 的图聚类是否引入了新的超参数敏感性？

3. **与 STAPO/MASPO 的组合**：AT-RL 聚焦多模态的视觉-文本分离，STAPO/MASPO 聚焦纯语言的梯度方差，两者可以叠加吗？

4. **训练 vs 推理时使用锚点**：锚点识别是在训练时做还是推理时做？如果是训练时，那锚点分布会随训练进展而变化——是否需要动态更新？

### 多模态 RLVR 的共同问题

AT-RL 提出的问题实际上指向一个更大的洞：**多模态 RL 的 credit assignment 问题比纯语言更严重**。

原因：GRPO 等方法的序列级奖励把文本 token 和视觉 token 完全混在一起。但视觉 token 的"贡献"方式根本不同——它们不生成文字，但它们决定了文本 token 的视觉接地。奖励信号无法直接区分"这个推理步骤写对了因为它正确利用了视觉信息"和"这个推理步骤写对了因为语言模式偏好"。

AT-RL 用注意力图提供了一个方向，但更根本的解决方案可能需要**多模态 token 级 reward model**——这是一个尚未被很好解决的方向。

---

## 与 GRPO 全景综述的更新

这篇论文为 Token 维度增加了一个新的视角：**多模态场景下 credit assignment 的结构化解法**

| 维度 | 新增 |
|------|------|
| Token（多模态扩展）| AT-RL：~15% 视觉锚点 token 携带核心 credit，选择性强化 32B 超越 72B |

---

## 关键词连接

- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO-Improvement-Panorama-2026]] — Token 维度的多模态扩展
- [[AI/3-LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO-Spurious-Token-Aware-Policy-Optimization]] — 纯语言场景下的 token 级 credit 问题
- [[AI/3-LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling]] — 探索维度的 credit 问题

---

## 总结

AT-RL 是 GRPO 全景综述"token 级 credit assignment"开放问题在多模态场景下的一个优雅解答：用跨模态注意力连通性作为 credit 的结构化 proxy，用图聚类识别视觉锚点，选择性强化。

关键结论：**推理质量由跨模态锚定的精确度决定，而非 token 数量**。这和 GRPO 全景综述的核心论点完美呼应——序列级奖励让所有 token 等权重，而实际上 credit 极度不均匀。

AT-RL 提供的是多模态版本的答案。纯语言的 token 级 dense reward 仍然是未解的核心问题。

---

**See also**：[[AI/3-LLM/RL/Other-Algorithms/VPPO-Visually-Perceptive-Policy-Optimization|VPPO]] — 同样识别"关键感知 token"，但用 KL 散度（有无图像）定义视觉依赖度，而非图聚类；AT-RL 关注跨模态锚定点，VPPO 关注视觉依赖度稀疏分布，互补视角
