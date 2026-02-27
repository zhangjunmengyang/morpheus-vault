---
brief: "GSPO（Group Sequence Policy Optimization）——将 importance sampling 从 token 级提升到 sequence 级，解决 GRPO/PPO token 级 IS ratio 噪声问题；Qwen3 MoE 训练采用，序列级约束更适合长链推理的策略更新。"
title: "GSPO: Group Sequence Policy Optimization"
type: paper
domain: ai/llm/rl
tags:
  - rl
  - GRPO
  - importance-sampling
  - sequence-level
  - MoE
  - Qwen3
  - off-policy
created: 2026-02-21
status: v1
---

# GSPO: Group Sequence Policy Optimization

**论文**: Group Sequence Policy Optimization  
**arXiv**: 2507.18071 (cs.LG)  
**提交**: 2025-07-24（v2: 2025-07-28）  
**作者**: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin  
**机构**: Alibaba / Qwen 团队  
**应用**: Qwen3 模型的 RL post-training  
**评分**: ★★★★☆

---

## 背景：为什么 GRPO 的 Token-Level IS 有问题？

GRPO 的目标函数：
```
J_GRPO(θ) = E[1/G Σ_i 1/|y_i| Σ_t min(w_i,t(θ)·Â_i, clip(w_i,t(θ), 1-ε, 1+ε)·Â_i)]
```

其中 `w_i,t(θ) = π_θ(y_i,t|·) / π_θ_old(y_i,t|·)` 是**token-level** 重要性比率。

**核心矛盾**：奖励 r(x, y_i) 是**序列级别**的（整个输出对不对），但重要性采样比率是 **token 级别**的。这造成了**对齐错误**：

- 序列级奖励意味着每个 token 对最终结果的贡献是联合的，不是独立的
- Token-level IS 把每个 token 的更新独立处理，隐式假设 token 独立——这与序列奖励矛盾
- 在 MoE 场景下，路由波动进一步放大 token-level variance，导致训练不稳定

---

## 核心方法：序列级重要性比率

GSPO 把重要性采样比率从 token 级别提升到**序列级别**：

```
s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
        = exp(1/|y_i| · Σ_{t=1}^{|y_i|} log[π_θ(y_i,t|x,y_i,<t) / π_θ_old(y_i,t|x,y_i,<t)])
```

**数学等价性**：这是所有 token log IS ratio 的**几何平均**（arithmetic mean of logs → geometric mean of ratios）。

GSPO 目标函数：
```
J_GSPO(θ) = E[1/G Σ_i min(s_i(θ)·Â_i, clip(s_i(θ), 1-ε, 1+ε)·Â_i)]
```

关键变化：
1. 用 `s_i(θ)` 替代 token-level `w_i,t(θ)`
2. Clipping 在**序列层面**执行，而非 token 层面
3. 没有 `1/|y_i|` 的内层 token 求和（已被折叠进 `s_i`）

---

## 为什么 Geometric Mean 优于 Arithmetic Mean？

GRPO 实际上在用：
```
GRPO IS ≈ (1/T) Σ_t w_t  [算术平均，通过 1/|y_i| 归一化实现]
```

GSPO 用的是：
```
GSPO IS = (Π_t w_t)^(1/T)  [几何平均]
```

**几何平均的优点**：
- 对极端 token-level ratio 更鲁棒（单个 token 的 exploding ratio 在乘积里被压缩，在求和里会主导）
- 更忠实地反映**序列整体策略变化**
- 数学上与序列级奖励 r(x,y) 的对齐更一致

**与 GMPO 的对比**（GMPO = Geometric-Mean Policy Optimization）：
- GSPO：序列级 IS ratio + 序列级 clipping
- GMPO：保留 token-level 分解，但用几何平均替代算术平均聚合
- 效果相似，GSPO 更简洁；GMPO 保留 token-level 粒度

---

## MoE 场景下的额外收益

GRPO 在 MoE 上特别不稳定，原因：
1. **路由波动**：同一个 token 在新旧策略中可能走不同 expert，~10% 的 expert 会变
2. 路由变化直接导致 token-level log-prob 大幅波动 → IS ratio 极端 → clip 频繁触发
3. 序列级 IS ratio 把路由波动"平均掉"，减少了单个 token 路由变化的影响

GSPO 文中提到的另一个缓解手段：**routing replay**（复用旧策略的 expert assignment 计算新策略的 log-prob）——但这会限制 router 更新，内存/通信开销大，是 GSPO 的辅助手段而非核心。

---

## 实验结果

- 比 GRPO 更高效、更稳定
- **特别对 MoE 训练效果显著**：稳定了 Qwen3 系列 MoE 模型的 RL 训练
- GSPO 作为 Qwen3 的核心 RL 算法角色已被官方确认："GSPO has contributed to the remarkable improvements in the latest Qwen3 models"

---

## 批判性分析

### 亮点

1. **问题定位准确**：token-level IS ratio 与 sequence-level reward 的不对齐是真实问题，几何平均是自然的修正
2. **工业验证**：Qwen3 的成功部分归功于 GSPO，这是强有力的 production 背书
3. **MoE 特化价值**：路由波动问题在 MoE 上真实存在，GSPO 的序列聚合天然缓解这个问题

### VESPO 对 GSPO 的批评：是否成立？

VESPO (arXiv 2602.10693) 指出：
> GSPO 的 1/T 归一化（即 s_i 的指数 1/|y_i|）引入了 length bias：更长的序列的 IS weight 被系统性地压缩，导致短序列更容易 satisfy clip constraint，而长序列逐渐被 ignore → 训练偏向短序列 → 最终 collapse。

**我的判断：VESPO 的批评部分成立，但被过度简化了。**

VESPO 的实验显示"无归一化最稳定"——但这与 GSPO 的工业成功形成张力。真实情况可能是：
- GSPO 的 1/T 归一化是好的（减少 token-level noise），但可能确实引入了 length bias
- Length bias 在 **on-policy 同步训练**（GSPO 的典型场景）中影响较小
- 在**高 staleness 异步训练**（VESPO 的测试场景）中，1/T 归一化的 length bias 被放大

换言之：**GSPO 在 on-policy 场景是优秀的；VESPO 在 off-policy/async 场景做出了更完整的修正。**

两者并非竞争关系，而是不同适用域下的最优解。

---

## 与 Vault 其他论文的关系

| 论文 | 与 GSPO 的关系 |
|------|--------------|
| **GRPO** | GSPO 的直接改进：把 token-level IS → sequence-level IS |
| **VESPO** | 批评 GSPO 的 1/T 归一化引入 length bias；VESPO 的变分推导提供了更一般的框架 |
| **DAPO** | ByteDance 的 GRPO 改进，专注 clip 策略和 entropy 控制，不改动 IS 层 |
| **MASPO** | Token-level 概率质量分析，与 GSPO 的序列级改动正交 |
| **Jet-RL** | 从系统层解决 off-policy；GSPO 从算法层解决序列对齐；两者正交 |

---

## 技术谱系

```
GRPO（token-level IS + token-level clip）
  ↓ 改进 IS 层
GSPO（sequence-level IS = geometric mean，sequence-level clip）
GMPO（保留 token-level 分解，geometric mean 聚合，类似 GSPO）
  ↓ 更严格的理论
VESPO（变分推导 optimal IS kernel，1/T 归一化被证伪为 length bias 来源）
```

---

## 元数据

- **Tags**: #GRPO #importance-sampling #sequence-level #MoE #Qwen3 #off-policy
- **关联笔记**: [[AI/3-LLM/RL/算法/VESPO-Variational-Sequence-Policy-Optimization|VESPO]] ⭐ — VESPO 是 GSPO 的理论上界：GSPO 发现 1/T 归一化的 length bias 问题，VESPO 变分推导给出最优 IS kernel，证明为什么 GSPO 在高 staleness 时 collapse | [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景 2026]] | [[AI/3-LLM/RL/算法/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] | [[AI/3-LLM/RL/算法/SAPO-Soft-Adaptive-Policy-Optimization|SAPO]] — **GSPO 的软门控继承版**：Qwen 团队同一路线上的后续工作，把 sequence-level hard clip 替换为 token-level sigmoid 软衰减；理论证明在 A1+A2 假设下 SAPO ≡ GSPO（连续版），在 MoE 异构 token 场景更鲁棒
- **写于**: 2026-02-21
