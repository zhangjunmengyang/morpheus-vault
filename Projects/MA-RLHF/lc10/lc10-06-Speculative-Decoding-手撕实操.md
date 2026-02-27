---
title: Speculative Decoding 手撕实操
brief: Speculative Decoding 手撕实现：Draft Model 连续预测 N token（并行，快）→ Target Model 一次验证（并行拒绝采样）→ 加速比 2-4x 无质量损失。面试高频必考。MA-RLHF lc10 推理系统专题。
date: 2026-02-26
type: code-practice
source: MA-RLHF lc10 推理系统 / Speculative_Decoding.ipynb
tags:
  - code-practice
  - inference
  - speculative-decoding
  - draft-model
  - rejection-sampling
related:
  - "[[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-02-vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]]"
  - "[[AI/3-LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow-Video-LLM-Speculative-Decoding]]"
  - "LLM推理优化2026全景"
---

# Speculative Decoding 手撕实操

> **来源**：MA-RLHF lc10 推理系统 / Speculative_Decoding.ipynb  
> **难度**：★★★★☆  
> **面试频率**：★★★★★（推理加速必考，Google/Meta/ByteDance 高频）  
> **关联**：[[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]] [[Projects/MA-RLHF/lc10/lc10-02-vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] [[AI/3-LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow-Video-LLM-Speculative-Decoding]]

---

## 核心问题

**LLM Decode 是串行的**：每一步都要完整运行目标模型（Target Model）生成 1 个 token → GPU 严重 underutilized（memory-bound，带宽瓶颈）。

**Speculative Decoding 思路**：
1. 用一个**小草稿模型**（Draft Model）快速连续预测 N 个 token（N=5~10）
2. 目标模型**一次**并行验证所有 N 个 token
3. 接受其中前 K 个正确的，从第 K+1 个开始用目标模型纠正

**关键洞察**：
- Draft Model 的 N 次串行生成成本 << Target Model N 次串行生成
- Target Model 验证是**并行的**（N 个 token 同时过模型），相当于批处理
- 期望加速比 = `accepted_tokens + 1`（最差接受 0 个也至少多了 target 的 1 个）

---

## 双模型架构

```
草稿模型（Draft Model）：
  - 参数量小（7B → 1B，或同模型的 early exit layer）
  - 速度快，可以串行生成多个 token
  - 输出分布 p(x) 与目标模型 q(x) 近似但不完全相同

目标模型（Target Model）：
  - 参数量大，输出分布即"真实"分布 q(x)
  - 每次 forward 验证所有草稿 token（并行）
  - 最终确保生成的 token 分布 ≡ 纯用目标模型生成
```

**分布距离度量**（KL Divergence）：

```python
def KL(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    kl = F.kl_div(log_p, q, reduction='sum', log_target=False)
    return kl

# target vs draft（相似度越高，接受率越高）
# target vs random（KL 最大，接受率最低）
# target vs target（KL=0，接受率 100%）
```

---

## 核心算法

### 1. Greedy Speculative Decoding（基础版）

```python
class SPDecoding:
    def __init__(self, model_target, model_draft, spec_n):
        self.spec_n = spec_n  # 每次草稿生成几个 token
    
    def generate_draft(self, spec_n, x):
        """草稿模型串行生成 spec_n 个 token"""
        logits_y = []
        for i in range(spec_n):
            logits = self.model_draft(x)[:, [-1], :]  # [bsz, 1, vocab]
            logits_y.append(logits)
            next_token = torch.argmax(logits, dim=-1)   # greedy
            x = torch.cat([x, next_token], dim=1)
        return x, torch.cat(logits_y, dim=1)  # x: [bsz, L+spec_n], logits: [bsz, spec_n, vocab]
    
    def generate(self, x, max_new_tokens=30):
        generated = 0
        accept_count = []
        
        while generated < max_new_tokens:
            # 1. 草稿模型生成 spec_n 个候选 token
            x_with_draft, logits_draft = self.generate_draft(self.spec_n, x)
            y_draft = x_with_draft[:, x.shape[1]:]  # [bsz, spec_n]
            
            # 2. 目标模型一次性验证（并行！）
            # 输入: x + 所有草稿 token，获得 spec_n+1 个 logits
            logits_target = self.model_target(x_with_draft)[:, -self.spec_n-1:, :]
            y_target = torch.argmax(logits_target, dim=-1)  # [bsz, spec_n+1]
            
            # 3. 找第一个不一致的位置
            verify = (y_draft == y_target[:, :self.spec_n])   # [bsz, spec_n]
            mismatch = torch.where(verify == False)
            
            if len(mismatch[1]) == 0:
                # 全部接受！+spec_n+1 个 token（最后那个 target 自生成的）
                accept_len = self.spec_n + 1
            else:
                # 部分接受：前 k 个接受，用 target 纠正第 k+1 个
                accept_len = mismatch[1][0].item() + 1  # 包含 target 纠正的那个
            
            # 4. 拼接被接受的 token
            x = torch.cat([x, y_target[:, :accept_len]], dim=1)
            generated += accept_len
            accept_count.append(accept_len)
        
        return x, sum(accept_count) / len(accept_count)
```

**关键细节**：
```
输入 x（长度 L）+ 草稿 [a1,a2,a3,a4,a5]
↓
目标模型 forward（长度 L+5）
↓
得到 6 个 logits（位置 L, L+1, L+2, L+3, L+4, L+5）
↓
y_target = [b1,b2,b3,b4,b5,b6]  ← 目标模型预测的每个位置的下一个token

验证：a1==b1? a2==b2? a3==b3?
若前2个接受（a1==b1, a2==b2, a3≠b3）：
  接受 [b1,b2]（等价于草稿的a1,a2），
  再取 b3（目标模型在位置L+2的预测，替换不匹配的a3）
  共 3 个新 token
```

### 2. Speculative Sampling（支持随机采样）

当草稿模型用 nucleus/temperature 采样（而非 greedy）时，验证逻辑需要改变：

```python
class SPSamplingDecoding:
    def _sampling(self, prob):
        return torch.multinomial(prob, num_samples=1)
    
    def generate_draft(self, spec_n, x):
        logits_y = []
        for i in range(spec_n):
            logits = self.model_draft(x)[:, [-1], :]
            logits_y.append(logits)
            prob = F.softmax(logits, dim=-1)
            next_token = self._sampling(prob)  # 随机采样！
            x = torch.cat([x, next_token], dim=1)
        return x, torch.cat(logits_y, dim=1)
    
    def verify_with_sampling(self, logits_draft, logits_target, y_draft):
        """
        基于分布差异的接受/拒绝（论文中的核心公式）
        acceptance ratio r = min(1, q(x) / p(x))
        """
        p = F.softmax(logits_draft, dim=-1)  # 草稿分布
        q = F.softmax(logits_target, dim=-1)  # 目标分布
        
        for i, token in enumerate(y_draft):
            r = q[token] / p[token]  # 接受比例
            if torch.rand(1) < r:
                # 接受
                continue
            else:
                # 拒绝：从修正分布 max(0, q-p) 中重新采样
                corrected = torch.clamp(q - p, min=0)
                corrected /= corrected.sum()
                return i, torch.multinomial(corrected, 1)
        
        # 全接受：从目标模型采样下一个 token
        return len(y_draft), torch.multinomial(q[-1], 1)
```

**为什么这个采样方式保证分布正确？**
- 接受的 token 是从草稿模型采样的，但通过接受率 `q(x)/p(x)` 修正
- 拒绝时从修正分布 `max(0, q-p)` 重采样
- 数学上可以证明：最终生成的序列分布 = 纯用目标模型 q 生成（无偏）

---

## 关键数字与超参

| 参数 | 典型值 | 含义 |
|------|--------|------|
| spec_n | 5~10 | 草稿模型预测几个 token |
| 期望接受率 | 60%~90% | 取决于草稿/目标模型的分布差异 |
| 理论加速比 | 2~4x | accepted_tokens + 1 的期望值 |
| 草稿模型比例 | ~1/7 | 7B target → 1B draft |

**接受率的影响**：
```
spec_n=5 时：
  全接受（100%）→ 每步 +6 tokens
  全拒绝（0%）  → 每步 +1 token（退化为标准 decode）
  50% 接受       → 平均每步 +3 tokens → ~3x 加速
```

---

## KVCache 的处理

```
Speculative Decoding 的 KVCache 管理（比普通 decode 复杂）：

Step 1: 对初始 x 做一次 Prefill → 填充 KVCache

Step k（循环）:
  草稿生成：每次只追加 1 个 token 的 KV
  目标验证：输入 spec_n 个草稿 token，追加 spec_n 个 KV
  接受 k 个：KVCache 有效长度 += k+1
  拒绝处理：需要截断 KVCache 到接受的位置（回滚）
  
核心技巧：目标模型 forward 时输入 y_draft，不是单 token，
          模式 = "带 KVCache 的多 token 输入"（介于 Prefill 和 Decode 之间）
```

---

## Self-Speculative Decoding（自推测）

不需要独立草稿模型，用**同一模型的浅层作为草稿**：

```
草稿阶段：只运行前 8 层（Early Exit）→ 快速生成候选 token
验证阶段：运行全 32 层，并行验证草稿 token
```

优势：
- 无需额外 Draft Model 显存
- Draft/Target 分布天然接近（同模型的表示空间）
- 适合资源受限场景

---

## 面试考点

**Q1: Speculative Decoding 如何保证生成质量不变？**
> A: 数学上通过接受/拒绝采样（acceptance-rejection sampling）保证：对每个草稿 token x，以概率 min(1, q(x)/p(x)) 接受；拒绝时从修正分布 max(0, q-p) 重采样。可以证明最终序列分布 = 目标模型 q 的分布，即无损加速。

**Q2: 为什么目标模型验证是"并行"的？**
> A: 给定当前序列 x 和草稿 token [a1,a2,a3,a4,a5]，目标模型做一次 forward：输入长度 L+5，输出 L+5 个 logits，同时得到了 a1...a5 每个位置的预测结果 b1...b5。这是 Transformer 的 causal attention 特性——一次 forward 同时计算所有位置的输出。

**Q3: spec_n 设多少合适？**
> A: 取决于期望接受率。如果草稿模型和目标模型分布很接近（接受率 80%+），spec_n=8~10 很合适；如果接受率只有 50%，spec_n=3~5 更好（太多草稿容易全部被拒绝，浪费计算）。实践中通过 profile 实际吞吐量来选。

**Q4: Speculative Decoding 和 Chunked-Prefill 的核心区别？**
> A: Chunked-Prefill 解决的是 Prefill 阻塞 Decode 的延迟问题（TTFT/TPOT 矛盾）；Speculative Decoding 解决的是 Decode 阶段吞吐量低的问题（GPU memory-bound）。两者面向不同瓶颈，可以同时使用。

**Q5: 为什么拒绝时要从 max(0, q-p) 中采样？**
> A: 当草稿 token x 被拒绝，意味着 p(x) > q(x)（草稿模型过度预测了这个 token）。修正分布 q - p 中那些 p > q 的位置被 clamp 到 0，剩余的是 q 比 p 更偏好的那些 token。从这个分布采样，保证了最终分布的无偏性。

---

## 与 PageAttention 的关系

vLLM 将 Speculative Decoding 与 PageKVCache 结合：
- 草稿阶段：使用 Page KVCache 管理草稿 token 的 KV
- 验证成功：保留对应 page
- 验证失败回滚：释放被拒绝 token 的 page（类似于 Copy-on-Write 的逆操作）

---

## 延伸阅读

- [[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]] — 推理服务基础
- [[Projects/MA-RLHF/lc10/lc10-02-vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] — KV 内存管理
- [[Projects/MA-RLHF/lc10/lc10-05-Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] — Prefill/Decode 混合调度
- [[AI/3-LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow-Video-LLM-Speculative-Decoding]] — 视频 LLM 的推测解码变体

---

*笔记来源：MA-RLHF lc10 / Speculative_Decoding.ipynb — 2026-02-26*
