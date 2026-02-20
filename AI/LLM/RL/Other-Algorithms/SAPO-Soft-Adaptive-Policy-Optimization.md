---
title: "SAPO: Soft Adaptive Policy Optimization"
type: paper
domain: ai/llm/rl
tags:
  - rl
  - GRPO
  - soft-clipping
  - trust-region
  - MoE
  - Qwen3-VL
  - importance-sampling
created: 2026-02-21
status: v1
---

# SAPO: Soft Adaptive Policy Optimization

**arXiv**: [2511.20347](https://arxiv.org/abs/2511.20347)  
**提交日期**: 2025-11-25  
**作者**: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin（Qwen 团队，Alibaba）  
**机构**: Alibaba DAMO Academy / Qwen Team  
**标签**: `RL` `GRPO` `信任域` `软裁剪` `MoE稳定性` `Qwen3-VL`

---

## 一句话

用 sigmoid 软门控替代硬裁剪——梯度权重 = **sech²(τ/2 · (r−1))**，在 on-policy 点 (r=1) 保持满权重，随偏差平滑衰减，消除断崖式梯度截断。

---

## 问题定位

GRPO 和 GSPO 都用**硬裁剪（hard clipping）**控制 off-policy 更新：
- **GRPO**：token-level clip，r 超出 [1±ε] 梯度归零 → 断崖式，clip 宽度两难（太窄丢样本，太宽引入噪声）
- **GSPO**：sequence-level clip，几个 off-policy token 拉低整个序列的 s_i，序列整体梯度归零 → 近 on-policy token 的有效信号被废弃，样本效率低

两者的共同缺陷：**梯度非连续性**导致有效梯度样本量小，且在 MoE 架构中因路由异质性被放大。

---

## 核心方法

### SAPO 目标函数

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta))\,\hat{A}_{i,t}\right]$$

其中 gating 函数（**不对称温度设计**）：

$$f_{i,t}(x) = \sigma\!\left(\tau_{i,t}(x-1)\right) \cdot \frac{4}{\tau_{i,t}}, \quad \tau_{i,t} = \begin{cases}\tau_{\text{pos}} & \hat{A}_i > 0 \\ \tau_{\text{neg}} & \hat{A}_i \leq 0\end{cases}$$

### 梯度权重：sech² 形式

对目标函数求导得：

$$w_{i,t}(\theta) = 4\,p_{i,t}(1-p_{i,t}) = \mathrm{sech}^2\!\left(\frac{\tau_{i,t}}{2}(r_{i,t}(\theta)-1)\right)$$

**关键性质**：
- $r=1$（on-policy）时 $w=1$，梯度满权重
- $r$ 偏离 1 时平滑指数衰减，无截断点
- $\tau$ 越大，衰减越快（更紧的信任域）

### 不对称温度的动机

负 advantage 的梯度更新增加所有 unsampled tokens 的 logit → 影响 |V| 个词汇（通常 >10 万）。正 advantage 只更新 1 个 sampled token。

∴ 负 token 扰动幅度远大于正 token，应更快衰减：$\tau_\text{neg} > \tau_\text{pos}$。

---

## 数学分析：与 GSPO 的等价条件

### 两个假设

**(A1)** 小步长/近 on-policy：$r_{i,t}(\theta) \approx 1$，故 $\log r_{i,t} \approx r_{i,t}-1$

**(A2)** 低序列内分散度：$\mathrm{Var}_i(\theta) = \frac{1}{|y_i|}\sum_t(z_{i,t}-\mu_i)^2$ 较小

### 退化定理

在 (A1)+(A2) 下，通过二阶 Taylor 展开可证：

$$\frac{1}{|y_i|}\sum_t g_{\tau_i}(z_{i,t}) \approx g_{\tau_i}(\mu_i) = \mathrm{sech}^2\!\left(\frac{\tau_i}{2}\log s_i(\theta)\right)$$

逼近误差上界：$D_i(\theta) \leq \frac{\tau_i^2}{4}\mathrm{Var}_i(\theta)$

**结论**：SAPO 在低方差序列上退化为以 sech² 为 gating 的 GSPO，但在高方差序列（少数 outlier tokens）上保持 token-level 适应性，**不废弃近 on-policy token 的梯度信号**。

---

## 方法关系总结

| 方法 | 层级 | 机制 | 信任域形状 |
|------|------|------|------------|
| GRPO | token-level | 硬裁剪 | 矩形窗口，截断外归零 |
| GSPO | sequence-level | 硬裁剪（几何均值） | 矩形窗口，整序列截断 |
| **SAPO** | token-level（seq-coherent） | sigmoid 软门控 | **连续钟形，sech²衰减** |
| VESPO | sequence-level | 变分最优化 | W^α·exp(-λW)，推导自原理 |

---

## 实验结果

### 数学推理 (Qwen3-4B / Qwen3-30B-A3B)
- SAPO 在相同训练 budget 下比 GRPO/GSPO **Pass@1 更高**
- SAPO 比 GRPO/GSPO **在崩溃前保持更久的稳定学习**（更晚发散）
- 温度 ablation：必须 τ_neg > τ_pos；对称温度导致早期崩溃

### Qwen3-VL 生产训练
- SAPO 用于训练 **Qwen3-VL 全系列**（文字+多模态混合任务）
- 跨越不同模型尺寸和架构，均有一致性能提升

### 与 VESPO 的对比（from VESPO 论文）
| Staleness | GRPO | SAPO | VESPO |
|-----------|------|------|-------|
| N=16 | ~57% | ~52% | ~58% |
| N=64 | ~44.7% | **~18.4%（崩溃）** | **~58.5%（稳定）** |

⚠️ **SAPO 在高 staleness 下崩溃**：token-level 设计缺乏序列级 IS 方差控制理论支撑，极度 off-policy 时衰减不足。

---

## 我的分析

### 技术价值：★★★★☆

**优点**：
1. **sech² 形式非常优雅**：与 tanh 激活函数同族，连续可微无截断点，解析 IS 权重为 logistics 函数的自然推导
2. **数学证明严格**：(A1)+(A2)→GSPO 的等价条件有误差上界，不是经验主义
3. **不对称温度设计有理论依据**：从 logit 梯度角度分析正/负 advantage 的不同稳定性
4. **生产验证**：Qwen3-VL 实际使用，是工业级方法

**局限**：
1. **VESPO 指出的根本弱点**：SAPO 本质上是 token-level 方法，处理 sequence-level IS 权重时仍是"近似+启发式"，缺乏统一理论框架
2. **高 staleness 下崩溃**：N=64 时仅 18.4%，比 GRPO（44.7%）还差——说明在极度 off-policy 场景下 token-level 软衰减反而不如硬裁剪
3. **τ 超参数调优**：两个温度参数 + 不对称设计需要调，实际工程成本高于 GRPO

### SAPO vs VESPO：两条路线的深层分歧

| 维度 | SAPO | VESPO |
|------|------|-------|
| 基础 | 启发式软门控，经验驱动 | 变分原理，数学推导 |
| 层级 | token-level（经 Taylor 近似退化到 seq-level） | 原生 sequence-level |
| 高 staleness | 崩溃（18.4%） | 稳定（58.5%） |
| 实用性 | ✅ 已生产（Qwen3-VL） | 代码开源，但生产验证不详 |
| 适用场景 | 同步 RL，staleness 可控 | 异步 RL，高 staleness |

**我的判断**：SAPO 是 on-policy/近 on-policy 同步训练的优质改进，但不是 off-policy 稳定性的解决方案；VESPO 在理论上更严格，但 SAPO 的生产验证更强。两者不是竞争而是适用域不同。

### 与 GRPO 全景的关系

- **TrustRegion 维度**：SAPO 是 soft trust region 方法（连续），GRPO/GSPO 是 hard clip（离散）
- **Token 维度**：SAPO 的不对称温度设计可以看作对 token-level 差异化处理的一种方式（vs STAPO 的 spurious token mask）
- MASPO（2602.17xxx，mass-adaptive soft policy）名字相似，但机制不同——MASPO 用自适应 soft trust region，SAPO 用固定 sech² 形状

---

## 引用关系

- **被 VESPO 引用**：作为 token-level IS 方法的代表，VESPO 的关键对比基线
- **被 GSPO 作者写**：Chang Gao 是 SAPO 一作，Chujie Zheng 是 GSPO 一作，同为 Qwen 团队——SAPO 是同一团队在 GSPO 之后继续改进的工作
- **SAPO → GSPO 时序**：SAPO (2025-11-25) 早于 GSPO (2025-07-24)？错，GSPO 是 2025-07-24，SAPO 是 2025-11-25。SAPO 晚于 GSPO，是对 GSPO hard clipping 的改进反思

---

## 总结

SAPO 的核心贡献是**把 hard clip 的 "梯度截断" 变成 sech² 的 "梯度软衰减"**，从数学上证明了 token-level soft gate 在温和条件下等价于 sequence-level soft gate（GSPO 的软化版本），并给出不对称温度的理论动机。它是 Qwen 团队内部的 GRPO 改进链条（GRPO → GSPO → SAPO），最终被用于 Qwen3-VL 生产训练。但在极度 off-policy 场景下，VESPO 的变分方法理论上和实验上都更优。

---

## 关键词连接

- [[AI/LLM/RL/Other-Algorithms/GSPO-Group-Sequence-Policy-Optimization|GSPO（Qwen3正式版）]] — **同源关系**：Qwen 团队内部改进链条，GSPO (2025-07) → SAPO (2025-11)；SAPO 是对 GSPO hard clip 的软化改进
- [[AI/LLM/RL/Other-Algorithms/VESPO-Variational-Sequence-Policy-Optimization|VESPO]] — **理论对比**：SAPO 是启发式 sech² 软门控；VESPO 从变分原理给出最优 IS kernel；同步训练 SAPO 足够，高 staleness 异步训练 VESPO 更优
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 名字相似机制不同：MASPO = 自适应 soft trust region + 概率质量校正；SAPO = 固定 sech² 形状；均属 soft clipping 家族
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — SAPO 属于 TrustRegion 维度的 soft clip 改进路线
