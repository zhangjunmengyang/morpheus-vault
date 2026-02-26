---
title: "lc4 · Llama 系列专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc4_llama"
tags: [moc, ma-rlhf, llama, rope, gqa, rmsnorm, swiGLU, lc4]
---

# lc4 · Llama 系列专题地图

> **目标**：掌握 Llama 相对 GPT 的四大改进组件，理解现代 LLM 的标准架构选型。  
> **核心定位**：Llama 是开源 LLM 的里程碑——它不是发明了新组件，而是**选对了组合**。几乎所有后续开源模型（Qwen / DeepSeek / Mistral）都沿用了这套架构。

---

## 带着这三个问题学

1. **Llama 相对 GPT-2 改了哪四个组件？每个改了的动机是什么？**
2. **RoPE 如何编码相对位置？NTK-aware RoPE 如何把 4K context 扩展到 32K？**
3. **GQA 为什么能在不显著损失精度的情况下大幅减少 KV Cache？**

---

## 学习顺序

```
Step 1  RMSNorm                   ← 替代 LayerNorm，更快更简
   ↓
Step 2  SwiGLU 激活               ← 替代 GELU，门控 FFN
   ↓
Step 3  RoPE 旋转位置编码          ← 替代 Absolute PE，相对位置
   ↓
Step 4  GQA 分组查询注意力         ← 替代 MHA，KV Cache 效率
   ↓
Step 5  Llama 完整模型集成         ← 四大组件组装（🌟核心）
   ↓
Step 6  NTK-aware RoPE            ← 长文本扩展：4K → 32K+
   ↓
Step 7  Benchmark 评测            ← 选择题评测预训练模型
```

---

## 笔记清单

### Step 1：RMSNorm

**[[AI/LLM/Architecture/Llama-手撕实操|Llama 手撕实操]]**（RMSNorm 部分）

- **vs LayerNorm**：RMSNorm 去掉了 mean centering（减均值），只做 RMS 归一化：`y = x / RMS(x) * γ`
- **RMS(x)** = `√(mean(x²))`，不需要计算均值 → 少一次 reduce 操作 → 速度快 ~10-15%
- **为什么有效**：实验表明 re-centering（减均值）对性能贡献极小，去掉后精度几乎无损
- **反向梯度**：推导 `∂L/∂x` 需要考虑 RMS 对 x 的依赖（链式法则较 LN 更简洁）

课程代码：`RMSNorm.ipynb`（Normalization 系统介绍 + RMSNorm/LayerNorm 几何对比可视化 + 梯度推导）

---

### Step 2：SwiGLU 激活

⏳ 待入库：**SwiGLU 原理笔记**

- **GLU（Gated Linear Unit）**：`GLU(x) = (xW₁) ⊙ σ(xW₂)`，门控机制提供精细化特征选择
- **SwiGLU**：`SwiGLU(x) = (xW₁ ⊙ SiLU(xW₂)) · W₃`，其中 SiLU(x) = x·σ(x)
- **vs GELU FFN**：标准 FFN 是 `max(0, xW₁ + b₁)W₂ + b₂`；SwiGLU 多一个门控矩阵 → 3 个权重矩阵（参数量增加 ~50%），但实验性能更好
- **为什么更好**：门控让网络可以**选择性通过**信息，比无条件变换更灵活

课程代码：`SwiGLU.ipynb`（门控特征选择的实现与分析）

---

### Step 3：RoPE 旋转位置编码（🌟核心）

**[[Transformer 位置编码|位置编码]]** · **[[AI/LLM/MA-RLHF课程/lc8-RoPE全家桶手撕实操|RoPE 全家桶手撕实操 ✅]]**

- **核心思想**：将位置信息编码为旋转角度，对 Q/K 向量做旋转变换
- **数学**：将 d 维向量看成 d/2 个 2D 向量，每个施加旋转矩阵 `R(mθ_i)`，θ_i = 10000^{-2i/d}
- **关键性质**：`<R(mθ)q, R(nθ)k> = <q, R((n-m)θ)k>` → 内积只依赖相对位置 m-n
- **为什么外推性好**：不需要学习每个位置的固定向量，旋转角度可以平滑外推到未见过的位置

课程代码：`RoPE.ipynb`（🌟 推导 + 理想位置编码性质验证 + 完整实现）

手撕实操包含：标准 RoPE → PI（位置插值）→ NTK-RoPE → YaRN 三策略完整对比，面试级公式推导。

深入阅读：[[Transformer 位置编码|位置编码全家族]]

---

### Step 4：GQA 分组查询注意力

**[[GQA-MQA|GQA-MQA 理论]]** · **[[AI/LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操|GQA + KV Cache 手撕实操 ✅]]**

MHA → MQA → GQA 演进：

| 方法 | Q 头数 | K/V 头数 | KV Cache 大小 | 精度 |
|------|--------|---------|-------------|------|
| MHA | h | h | 100% | 最高 |
| MQA | h | 1 | 1/h | 有损 |
| GQA | h | g（如 h/4） | g/h（如 25%） | 接近 MHA |

- **GQA 核心**：将 h 个 Q 头分成 g 组，每组共享一对 K/V → KV Cache 减少 h/g 倍
- **为什么有效**：实验发现不同 Attention 头的 K/V 表示高度冗余，共享 K/V 几乎不损失精度
- **Llama-2 70B**：h=64, g=8 → KV Cache 只有 MHA 的 1/8

课程代码：`GroupedQueryAttention.ipynb`（🌟 MQA/GQA 实现 + 注意力头冗余性分析）

手撕实操包含：GQA 完整 PyTorch 实现 + KV Cache 增量解码（past_key_values + position 追踪）。

---

### Step 5：Llama 完整模型（🌟核心）

**[[AI/LLM/Architecture/Llama-手撕实操|Llama 手撕实操]]**

```
Token Embedding
  → N × [RMSNorm → GQA(RoPE) → Add → RMSNorm → SwiGLU → Add]
  → Final RMSNorm → LM Head
```

vs GPT-2 的变化总结：LayerNorm → RMSNorm，GELU FFN → SwiGLU，Absolute PE → RoPE，MHA → GQA

课程代码：`Llama.ipynb`（🌟 聚合四大组件实现完整前向计算）

---

### Step 6：NTK-aware RoPE — 长文本扩展

⏳ 待入库：**NTK-RoPE 长文本扩展笔记**

- **问题**：RoPE 在训练长度（如 4K）内有效，超出后注意力 score 衰减 → 生成质量崩塌
- **Position Interpolation（PI）**：将位置除以扩展比例 `m' = m * L_train / L_target` → 缺陷：高频信息被压缩，损失局部分辨率
- **NTK-aware RoPE**：从频率域视角，**高频外推、低频内插**
  - 高频维度（局部位置）：保持原始频率，不插值 → 保留局部精度
  - 低频维度（全局位置）：做 NTK 插值 → 扩展全局范围
  - 实现：修改 base frequency `θ' = θ · α^{d/(d-2)}`，α 为扩展比例
- **效果**：4K → 32K 几乎无损，无需微调（或极少量微调）

课程代码：`NTK-aware-RoPE.ipynb`（PI 推导 → PI 缺陷分析 → NTK-RoPE 动态插值实现）

---

## 面试高频场景题

**Q：GQA 和 MQA 的参数量和 KV Cache 对比？**  
A：假设 d_model=4096, h=32。MHA：K/V 各 32 头，KV Cache ∝ 32；MQA：K/V 各 1 头，KV Cache ∝ 1（1/32）；GQA-8：K/V 各 8 头（每 4 个 Q 头共享 1 个 KV），KV Cache ∝ 8（1/4）。参数量上 GQA 的 W_K/W_V 是 MHA 的 g/h，额外节省。

**Q：NTK-RoPE 的核心思路是什么？**  
A：在频率域将 RoPE 的频率分为高频（编码局部位置）和低频（编码全局位置）。对低频做插值以扩展全局范围，对高频保持外推以保留局部分辨率。通过修改 base frequency 实现 `θ' = θ · α^{d/(d-2)}`，一个参数 α 控制扩展比例。

**Q：RMSNorm 比 LayerNorm 快多少？为什么可以去掉 mean centering？**  
A：快 10-15%（少一次 mean reduce + 减法）。实验（Zhang & Sennrich, 2019）证明 LayerNorm 的 re-centering 对 Transformer 性能贡献极小——主要贡献来自 re-scaling（除以标准差 / RMS），re-centering 的效果被后续的可学习仿射变换（γ, β）吸收了。

**Q：Llama 的架构选型为什么成为开源标准？**  
A：RoPE 提供最好的长文本泛化，GQA 大幅降低推理成本，SwiGLU 在相同参数量下性能最好，RMSNorm 训练更稳定且更快。Meta 开源了完整权重 + 训练细节，后续模型（Qwen / DeepSeek / Mistral）都验证了这套组合的有效性。
