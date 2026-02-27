---
title: GRPO KL 散度三种 Schulman 近似
brief: 实现并对比 KL 散度的三种 Schulman 近似（k1/k2/k3）：一阶泰勒展开（k1）、IS-ratio 估计（k2）、正则化比率（k3）。理解 GRPO/PPO 中 KL 惩罚项的计算精度与计算成本 tradeoff。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl
  - grpo
  - kl-divergence
  - policy-gradient
related:
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-02-GRPO-完整Notebook实现]]"
  - "lc8-RL×LLM-MOC"
---

# GRPO KL 散度：三种 Schulman 近似

> 来源：`ma-rlhf/notebook/GRPO/GRPO_KL.ipynb`（16 cells）
> 参考：[Schulman (2020) - Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

## 1. 为什么要近似 KL？

在 GRPO/PPO 中，KL 散度 $D_{KL}[\pi_\theta \| \pi_{ref}]$ 用于惩罚策略偏离 reference model。但标准 KL 散度的计算需要**对整个词表求和**：

$$D_{KL}[p \| q] = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

在 LLM 场景下：
- 词表大小 ~128K（LLaMA 3），每个 token position 都要对 128K 维做 softmax + 求和
- 序列长度可达数千 token
- 每个训练 step 有 batch_size × group_size 条序列

**直接计算 KL 的代价太高**。实际上我们只需要**已采样 token** 的 log-prob（一个标量），不需要整个分布。Schulman 提出了三种基于采样的 KL 近似，只需要 $\log \pi_\theta(a|s)$ 和 $\log \pi_{ref}(a|s)$。

---

## 2. 三种近似公式

设 $r = \frac{p(x)}{q(x)}$（即 $\log r = \log p(x) - \log q(x)$），从 $q$ 中采样 $x \sim q$：

### k1：负 log ratio

$$k_1 = -\log r = \log q(x) - \log p(x)$$

- **含义**：最直接的估计，$\mathbb{E}_{x \sim q}[-\log r] = D_{KL}[q \| p]$
- **问题**：**有正有负**，单个样本可以是负值，导致**方差大**
- **均值为 0 轴对称分布**，需要大量样本才能稳定

### k2：平方 log ratio

$$k_2 = \frac{(\log r)^2}{2} = \frac{(\log p(x) - \log q(x))^2}{2}$$

- **含义**：二阶 Taylor 近似
- **优点**：**恒为正值**（平方保证），方差比 k1 小
- **缺点**：当 $p$ 和 $q$ 差距大时，**高估** KL

### k3：Schulman 近似（GRPO 实际使用）

$$k_3 = r - \log r - 1 = \frac{p(x)}{q(x)} - \log\frac{p(x)}{q(x)} - 1$$

- **含义**：基于不等式 $x - 1 \geq \ln x$，恒为非负
- **优点**：**恒为正值 + 方差小 + 均值准确**
- **数学保证**：$f(r) = r - \ln r - 1 \geq 0$，等号当且仅当 $r=1$（即 $p=q$）

---

## 3. 代码实现与数值实验

### 3.1 实验设置

用两个高斯分布 $p = \mathcal{N}(0,1)$，$q = \mathcal{N}(0.5,1)$，采样 100K 个点：

```python
import torch.distributions as dis

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.5, scale=1)
x = q.sample(sample_shape=(100_000,))
truekl = dis.kl_divergence(p, q)  # 解析解

logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr
```

### 3.2 均值与方差对比

| 近似 | 均值 | 标准差 | 偏差/真实KL | 标准差/真实KL |
|------|------|--------|-------------|---------------|
| k1 | ≈ 真实 KL | **大**（有正有负振荡） | ~0 | ~2.0 |
| k2 | 略高估 | 中 | 正偏差 | ~1.4 |
| k3 | ≈ 真实 KL | **小** | ~0 | ~1.0 |

**结论**：k3 是最佳选择——均值无偏、方差最小、恒为正值。

### 3.3 k1 的问题：正负振荡

```python
print(k1[:10])
# tensor([-0.2338,  0.7869, -0.1310,  0.4817, -0.5332, ...])
```

k1 值围绕 0 轴对称分布，单个样本可能给出负的"KL"——这在物理上没有意义，需要大量采样取平均才有意义。

### 3.4 k2 vs k3：都为正值，但 k3 更紧

```python
print(k2.mean())  # 略高于真实 KL
print(k3.mean())  # 接近真实 KL
```

k2（绿色）整体比 k3（蓝色）分布更高，因为平方对极端值放大严重。

### 3.5 三种近似随 $r = q(x)/p(x)$ 的变化曲线

```python
x = torch.arange(0, 10.001, 0.001)
y1 = -x.log()           # k1：对数函数，可为负
y2 = x.log()**2 / 2     # k2：平方对数，增长快
y3 = x - 1 - x.log()    # k3：线性 - 对数，温和增长
```

关键观察：
- 当 $r \approx 1$（$p \approx q$）时，三者都接近 0
- 当 $r$ 偏离 1 时，k2 增长最快（过度惩罚），k3 增长温和
- k1 在 $r < 1$ 时为负值

### 3.6 k3 恒为正的数学证明

$k_3 = r - 1 - \ln r$，令 $f(r) = r - 1 - \ln r$：

- $f'(r) = 1 - 1/r = 0 \Rightarrow r = 1$
- $f''(r) = 1/r^2 > 0$（凸函数）
- $f(1) = 0$

所以 $f(r) \geq f(1) = 0$，等号当且仅当 $r = 1$。

这就是 **$x - 1 \geq \ln x$** 不等式的直接推论。

---

## 4. 在 GRPO/PPO 中 KL 惩罚的实际作用

### 防止策略崩溃

没有 KL 惩罚时，策略会 exploit reward function 的漏洞（reward hacking），生成高 reward 但低质量的输出。KL 惩罚把策略锚定在 reference model 附近。

### 控制更新步长

KL 惩罚系数 $\beta$ 控制探索 vs 保守的 trade-off：
- $\beta$ 大 → 策略保守，接近 reference，训练稳定但学习慢
- $\beta$ 小 → 策略激进，可能发散但学习快

### GRPO 中的具体实现

在 GRPO Loss 中，KL 作为 **per-token** 的惩罚项：

$$\mathcal{L} = -\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t \left[ \text{policy\_gradient}_t - \beta \cdot D_{KL,t} \right]$$

注意这里 KL 是 token-level 的，不是 sequence-level 的——这更精细地控制每个 token 的偏移程度。

---

## 5. GRPO 中 KL 的完整代码

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    """
    Schulman k3 近似：r - log(r) - 1
    其中 r = π_ref / π_θ = exp(log π_ref - log π_θ)
    """
    return (pi_ref_logprob.exp() / pi_logprob.exp()
            - (pi_ref_logprob - pi_logprob) - 1)
```

等价写法（更清晰）：

```python
def grpo_kl_v2(pi_logprob, pi_ref_logprob):
    log_ratio = pi_ref_logprob - pi_logprob  # log(π_ref / π_θ)
    return torch.exp(log_ratio) - log_ratio - 1
```

---

## 6. 面试考点

### Q1：为什么 GRPO 用 Schulman k3 而不是直接计算 KL？

**答**：两个原因。(1) **计算效率**：直接计算 KL 需要对整个词表（~128K）求和 $\sum_v p(v)\log(p(v)/q(v))$，而 k3 近似只需要已采样 token 的 log-prob——一个标量操作，计算量从 $O(V)$ 降到 $O(1)$。(2) **数值性质**：k3 恒为正值（由 $x - 1 \geq \ln x$ 保证），均值无偏，方差比 k1、k2 小。相比之下，k1 有负值导致方差大，k2 在分布差异大时严重高估。

### Q2：三种 KL 近似的数值稳定性排序？各自的适用场景？

**答**：稳定性排序 **k3 > k2 > k1**。
- **k1**（$-\log r$）：最简单但方差最大，有负值，适合理论分析但不适合训练
- **k2**（$\frac{(\log r)^2}{2}$）：恒正但高估 KL，适合需要保守估计的场景（宁可惩罚过重也不要策略偏移）
- **k3**（$r - \log r - 1$）：恒正、无偏、方差小，是 GRPO/PPO 的最佳选择。DeepSeek-R1、TRL GRPOTrainer 都使用 k3
