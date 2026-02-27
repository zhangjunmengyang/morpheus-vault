---
title: "GRPO KL 散度三种近似手撕（MA-RLHF lc8）"
brief: "KL 散度三种数值近似方式的完整手撕：① k1（1st order Taylor，加法形式）② k2（squared difference，偏差较大）③ k3（Schulman 博客推荐，最小方差无偏估计）；GRPO/PPO 的 KL 惩罚项正确实现是理解对齐训练数学基础的关键。来源：MA-RLHF GRPO_KL.ipynb。"
type: code-practice
date: 2026-02-26
source: "MA-RLHF notebook/GRPO/GRPO_KL.ipynb"
tags:
  - GRPO
  - KL散度
  - 数值近似
  - PPO
  - 手撕实操
  - MA-RLHF-lc8
related:
  - [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]]
  - [[AI/3-LLM/RL/实践/RLHF-工程全栈|RLHF-DPO-2026-技术全景]]
  - [[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]]
---

# GRPO KL 散度三种近似手撕（MA-RLHF lc8）

**来源**：MA-RLHF `notebook/GRPO/GRPO_KL.ipynb`  
**参考**：Schulman 博客 http://joschu.net/blog/kl-approx.html  
**难度**：★★★★☆（数学基础，理解 GRPO/PPO KL 惩罚的正确实现）  
**关联**：[[AI/3-LLM/MA-RLHF课程/lc8-GRPO-手撕实操|lc8-GRPO-手撕实操]] | [[Projects/MA-RLHF/lc8-PPO/lc8-01-PPO-手撕实操|lc8-RLHF-PPO-手撕实操]]

---

## 一、为什么需要近似？

KL 散度的精确定义：
$$\text{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

**问题**：在 LLM 训练中，我们只有 token 级别的 log prob，没有完整分布 $p$ 和 $q$。需要用**蒙特卡洛近似**：采样 $x \sim q$（当前 policy），估计 $\text{KL}(p \| q)$。

令 $r = \frac{p(x)}{q(x)} = e^{\log p - \log q}$（probability ratio），三种近似：

---

## 二、三种 KL 近似

```python
# 采样 x ~ q（当前 policy）
x = q.sample(sample_shape=(100_000,))
# log ratio: log(p/q)
logr = p.log_prob(x) - q.log_prob(x)
r = logr.exp()   # r = p(x)/q(x)

# K1: 最简近似（有偏，可为负）
k1 = -logr                   # = -log(p/q) = log(q/p)

# K2: 二阶泰勒展开近似（无偏，永远非负）
k2 = logr ** 2 / 2           # ≈ KL via (log r)²/2

# K3: Schulman 推荐（无偏，永远非负，方差最小）
k3 = (r - 1) - logr          # = exp(logr) - 1 - logr
```

### 数值验证（p = N(0,1), q = N(0.5,1)，true KL ≈ 0.125）

| 近似 | 均值（应≈0.125）| 标准差 | 特性 |
|------|---------|--------|------|
| K1 = -log(p/q) | ✓（接近）| 大 | **可能为负**，方差大 |
| K2 = (log r)²/2 | ✓（接近）| 中 | 永远非负，稍有偏差 |
| K3 = r-1-log(r) | ✓（最准）| **最小** | 永远非负，方差最小 |

**结论**：K3 是最优近似，GRPO 源码用的正是 K3。

---

## 三、K3 的数学基础：为什么永远非负？

关键不等式：$\log x \leq x - 1$（对任意 $x > 0$ 成立，等号仅在 $x=1$ 时取到）

因此：
$$k_3 = r - 1 - \log r \geq r - 1 - (r - 1) = 0$$

等号条件：$r = 1$，即 $p = q$（两个分布完全相同时 KL = 0）。✓

**几何直觉**：$\log x$ 曲线永远在直线 $y = x - 1$ 的下方，两者之差 $(x-1) - \log x$ 就是 K3。

---

## 四、在 GRPO/PPO 中的实际使用

```python
# GRPO 代码中（lc8-GRPO-手撕实操）
def grpo_kl(pi_logprob, pi_ref_logprob):
    # pi_logprob:     log π_θ(a|s)
    # pi_ref_logprob: log π_ref(a|s)
    # r = π_ref / π_θ = exp(log_ref - log_theta)
    r = (pi_ref_logprob - pi_logprob).exp()
    return r - (pi_ref_logprob - pi_logprob) - 1
    # = exp(log_ref - log_theta) - (log_ref - log_theta) - 1
    # = K3 形式的 KL(π_ref || π_θ)

# PPO 代码中（lc8-RLHF-PPO-手撕实操）
def compute_rewards_kl(reward, ref_logprobs, old_logprobs, kl_ctl):
    kl = old_logprobs - ref_logprobs   # log(π_old/π_ref) ≈ K1 形式
    kl_reward = -kl_ctl * kl           # token-level KL penalty
    kl_reward[:, -1] += reward[:, 0]   # 最后 token 叠加 scalar reward
```

**注意 GRPO vs PPO 的差异**：
- GRPO 用 K3（Schulman 推荐，方差最小）
- PPO 的 `compute_rewards_kl` 用 K1（`logprobs_old - logprobs_ref`），实现更简单但可为负

---

## 五、三种近似的适用场景

| 近似 | 代码 | 适用 | 缺陷 |
|------|------|------|------|
| K1 = -log(p/q) | `-logr` | 快速实现 | 可为负，高方差，导致 KL reward 不稳定 |
| K2 = (log r)²/2 | `logr**2/2` | 学术对比 | 非标准，实现少 |
| K3 = r-1-log(r) | `r.exp()-logr-1` | **生产推荐** | 略复杂但最准确 |

**实践建议**：GRPO/verl 等框架用 K3，PPO 的 reward shaping 常用 K1（因为 K1 可负，符合"过分偏离则负惩罚"的直觉）。

---

## 六、面试必备问题

**Q1：为什么 KL 散度要近似？**  
A：LLM 训练中我们无法计算精确的 KL（需要完整分布），只能用采样的方式用 Monte Carlo 估计。

**Q2：GRPO 用的哪种 KL 近似，为什么？**  
A：K3 = `exp(log_ref - log_theta) - (log_ref - log_theta) - 1`。因为 K3 永远非负（保证 KL 惩罚方向正确），且方差最小（训练最稳定）。

**Q3：K3 为什么永远非负？**  
A：`log x ≤ x-1` 对所有正 x 成立，因此 `(x-1) - log x ≥ 0`，令 `x = r = p/q` 即得 K3 ≥ 0。

---

*入库时间：2026-02-26*  
*来源：MA-RLHF notebook/GRPO/GRPO_KL.ipynb*  
*状态：Batch B ✅*
