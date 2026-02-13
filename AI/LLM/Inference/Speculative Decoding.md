---
title: "Speculative Decoding 推测解码"
date: 2026-02-13
tags:
  - ai/llm/inference
  - ai/llm/optimization
  - type/technique
  - interview/hot
status: active
---

# Speculative Decoding 推测解码

> 用小模型猜、大模型验——无损加速 LLM 推理的核心技术

## 1. 核心问题：为什么 LLM 推理慢？

LLM 自回归解码是 **memory-bound** 操作：每生成一个 token 都需要加载全部模型权重，但只做极少量计算。GPU 算力大量闲置。

```
传统自回归解码（target model T）:
Step 1: T(prompt) → t1        [加载全部权重]
Step 2: T(prompt+t1) → t2     [再次加载全部权重]
Step 3: T(prompt+t1+t2) → t3  [又加载一次...]
...
每步延迟 ≈ 模型权重加载时间 >> 计算时间
```

核心瓶颈：**每步只产出 1 个 token，但付出了加载整个模型的代价**。

参见 [[KV Cache 优化]] 和 [[推理优化]] 中的内存带宽分析。

## 2. 核心原理

### 基本思想

Leviathan et al. (2022) & Chen et al. (2023) 同时独立提出：

1. **Draft（起草）**：用小而快的 **draft model** D 连续生成 γ 个候选 token
2. **Verify（验证）**：用大的 **target model** T **一次 forward pass** 并行验证这 γ 个 token
3. **Accept/Reject（接受/拒绝）**：按概率校验，保证输出分布与纯 T 解码 **完全一致**

```
Draft model D (小、快):
  d1, d2, d3, d4, d5 ← 5个候选 token（γ=5）

Target model T (大、慢):
  一次 forward pass 验证: ✓d1, ✓d2, ✓d3, ✗d4 → 替换 d4 为 t4

结果: 一次 T 的 forward pass 产出了 3+1=4 个确认 token
加速比 ≈ 4x / (1 + D的开销)
```

### 数学保证：拒绝采样

关键创新：验证过程使用 **modified rejection sampling**，保证最终输出分布与直接用 T 解码 **完全相同**（无损）。

对于第 i 个候选 token $x_i$：
- 以概率 $\min\left(1, \frac{p_T(x_i)}{p_D(x_i)}\right)$ 接受
- 若拒绝，从调整分布 $\text{norm}\left(\max(0, p_T(x) - p_D(x))\right)$ 中重新采样

```python
def speculative_verify(draft_tokens, draft_probs, target_probs):
    """验证草稿 token，保证与 target 分布一致"""
    accepted = []
    for i, token in enumerate(draft_tokens):
        # 接受概率
        accept_prob = min(1.0, target_probs[i][token] / draft_probs[i][token])
        
        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # 拒绝：从修正分布重新采样
            adjusted = np.maximum(0, target_probs[i] - draft_probs[i])
            adjusted /= adjusted.sum()
            new_token = np.random.choice(len(adjusted), p=adjusted)
            accepted.append(new_token)
            break  # 后续 token 全部丢弃
    
    # 额外：T 的最后一个 logit 可以直接采样一个 bonus token
    if len(accepted) == len(draft_tokens):
        bonus = sample(target_probs[len(draft_tokens)])
        accepted.append(bonus)
    
    return accepted
```

## 3. 关键概念

### 接受率（Acceptance Rate）

$\alpha$ = 平均每个草稿 token 的接受概率。取决于 D 与 T 的分布匹配度。

- α 高 → D 和 T 对齐好 → 加速比高
- α 低 → D 预测偏差大 → 频繁拒绝，加速比差

### 加速比分析

理论加速比（忽略 D 的开销）：

$$\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha) \cdot c}$$

其中 $c = \frac{T_D \cdot \gamma + T_T}{T_T}$（D 和 T 的时间比）。

实际场景中：
- **α ≈ 0.7~0.9**：好的 draft model 选择
- **γ ≈ 3~7**：草稿长度（太长接受率下降）
- **实际加速：1.5x~3x**（取决于模型对和硬件）

### 草稿长度 γ 的选择

```
γ 太小 → 每轮验证 token 少 → 加速不明显
γ 太大 → 后面的 token 接受率越来越低 → 浪费
最优 γ* ≈ -1 / ln(α)
实践中 γ=5 是常用默认值
```

## 4. Draft Model 选择策略

### 方案对比

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **独立小模型** | 同系列小号（如 Llama-70B + Llama-8B） | 简单直接，接受率高 | 需要额外显存 |
| **Self-Drafting** | 用 target 的前几层/浅层做 draft | 零额外模型开销 | 接受率可能偏低 |
| **Medusa** | 在 target 模型上加多个预测 head | 不需独立 draft model | 需要微调额外 head |
| **Eagle** | 特征级预测（用隐藏层特征预测） | 接受率高（85%+） | 实现复杂 |
| **n-gram / Retrieval** | 基于已有文本的 n-gram 匹配 | 零计算成本 | 仅对重复性文本有效 |

### 代码示例：vLLM 中使用 Speculative Decoding

```python
from vllm import LLM, SamplingParams

# 方式 1: 独立 draft model
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.1-8B-Instruct",  # draft
        "num_speculative_tokens": 5,  # γ
    }
)

# 方式 2: n-gram based (无需额外模型)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 4,
    }
)

# 方式 3: Eagle (需要 Eagle head)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "path/to/eagle-head",
        "num_speculative_tokens": 5,
    }
)

params = SamplingParams(temperature=0.0, max_tokens=512)
outputs = llm.generate(["Explain quantum computing"], params)
```

参见 [[vLLM]] 的推理引擎配置。

## 5. Tree-based Speculative Decoding

### 从链到树

传统 spec decoding 是线性草稿（一条链），SpecInfer (Miao et al. 2024) 扩展为树状草稿：

```
         d1
        / | \
      d2a d2b d2c
      /     |
    d3a   d3b

验证时: T 对整棵树做一次 forward pass（用 tree attention mask）
接受最长的有效路径
```

### 优势

- 探索更多候选路径 → 更高的有效 token 产出
- 适合采样温度高（多样性大）的场景
- 单次验证成本与 token 数近似线性（tree attention）

### 性能数据（2025 benchmarks）

| 方法 | 模型对 | 加速比 | 接受率 |
|------|--------|--------|--------|
| Vanilla SD | Llama-70B + 8B | 2.0x | ~0.75 |
| Eagle-2 | Llama-70B + Eagle Head | 2.8x | ~0.85 |
| Medusa-2 | Llama-70B + 5 Heads | 2.3x | ~0.78 |
| SpecInfer (tree) | Llama-70B + 8B | 2.5x | ~0.80 |
| SWIFT (SwiftSpec) | Llama-70B + 8B | 3.1x | ~0.82 |

## 6. 适用场景与限制

### 最适合

- 大模型 + 好的小模型对（同系列效果最佳）
- **低 batch size**（batch=1 加速最明显）
- 高精度要求（无损，输出分布完全一致）
- 延迟敏感的交互式应用

### 限制

- **高 batch size** 时加速比下降（计算变成 compute-bound，而非 memory-bound）
- Draft model 需额外显存（或计算资源）
- 对采样参数敏感（temperature 越高接受率越低）
- 在线服务中 batch 动态变化，效果不稳定

## 7. 面试常见问题

1. **Q: Speculative Decoding 为什么是"无损"的？**
   A: 通过 modified rejection sampling，数学上证明了最终输出的 token 分布与直接用 target model 解码完全一致。被拒绝时会从修正分布重新采样，保证概率不变。

2. **Q: 加速的本质来源是什么？**
   A: LLM 解码是 memory-bound，一次 forward pass 加载权重的成本几乎不随 token 数线性增长（因为用了 KV Cache + 并行验证）。所以一次验证 5 个 token 的成本 ≈ 生成 1 个 token 的成本，但产出了多个。

3. **Q: 为什么 batch size 大时效果差？**
   A: 大 batch 下计算从 memory-bound 变成 compute-bound，GPU 算力已充分利用，并行验证的"免费午餐"消失。

4. **Q: Draft model 和 target model 不匹配怎么办？**
   A: 接受率会很低，大部分草稿被拒绝，加速比 < 1x（反而更慢）。实践中同系列模型（如 Llama-70B + Llama-8B）效果最好。

5. **Q: Medusa 和传统 Speculative Decoding 的区别？**
   A: 传统 SD 用独立 draft model，Medusa 在 target model 上加额外的预测 head（每个 head 预测未来第 k 个 token）。Medusa 不需要额外模型但需要微调额外 head。

## 相关笔记

- [[KV Cache 优化]] — KV Cache 机制与优化
- [[推理优化]] — LLM 推理优化全景
- [[vLLM]] — vLLM 推理引擎
- [[Transformer]] — Transformer 架构
- [[Attention 详解]] — 注意力机制
