---
title: KL散度：K1-K3 估计器 & Forward vs. Reverse
brief: KL 散度的蒙特卡洛估计（K1 无偏高方差、K2 有偏低方差、K3 无偏低方差）及 Forward/Reverse KL 在 RL 对齐中的应用——GRPO 使用 Reverse KL + K3 估计器防止策略偏移。
type: concept
domain: ai/llm/rl/fundamentals
created: 2026-02-13
updated: 2026-02-22
tags:
  - ai/llm/rl/fundamentals
  - type/concept
  - interview/hot
status: complete
sources:
  - Schulman, Approximating KL Divergence — http://joschu.net/blog/kl-approx.html
  - Trust Region Policy Optimization (TRPO) — arXiv:1502.05477
  - DeepSeekMath (GRPO) — arXiv:2402.03300
  - Cover & Thomas, Elements of Information Theory, 2nd ed., Wiley, 2006
related:
  - "[[GRPO 深度理解|GRPO 深度理解]]"
  - "[[PPO 原理|PPO 原理]]"
  - "[[信息论|信息论]]"
---
#  K1-K3 & Forward vs. Reverse KL散度

# 一、KL 散度

https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/r1-k1.5/grpo_kl.ipynb

## 基本概念

用于衡量两个概率分布 $P$ 和 $Q$ 之间的差异，对于离散随机变量，公式定义为：

$$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

- $P(x)$：真实分布的概率质量函数
- $Q(x)$：近似分布的概率质量函数

每一项 $P(x) \log \frac{P(x)}{Q(x)}$ 表示在事件 $x$ 上，真实分布 $P$ 与近似分布 $Q$ 的差异加权后的信息损失。

> 来源：Cover & Thomas, *Elements of Information Theory*, 2nd ed., Ch. 2

## 关键性质

- **非对称性**：KL 散度不是距离度量——$D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$，也不满足三角不等式
- **直觉理解**：KL 散度本质是用最优编码的信息论代理量，衡量两个分布有多大区别（[霍夫曼编码角度](https://www.zhihu.com/question/345907033/answer/3072696582)）
- **非负性**：$D_{\text{KL}}(P \| Q) \geq 0$，等号当且仅当 $P = Q$
- **与交叉熵关系**：$D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)$，最小化交叉熵等价于最小化 KL 散度（当 $P$ 固定时）
在实际应用中，精确计算KL散度通常不可行，因为：

1. **成本过高**：计算所有 ( xx*x* ) 的概率和需要过多的计算或内存。
1. **无法计算**：分布可能没有闭合表达式，无法解析求解。
蒙特卡洛方法成为估计 KL 散度的常用策略。假设我们从分布 q 中采样，

如何构造一个**无偏、低方差的估计器**是关键问题。

## K1 K2 K3

### K1 无偏高方差

> 来源：Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

最直接的蒙特卡洛估计器，令 $r = \frac{p(x)}{q(x)}$：

$$K_1 = \mathbb{E}_{x \sim q}\left[-\log r\right] = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$$

- **特点**：这个估计器是**无偏的**（$\mathbb{E}[K_1] = D_{\text{KL}}(q \| p)$）。然而，由于 $\log r$ 在正负之间变化（$r > 1$ 时为正，$r < 1$ 时为负），**方差很高**，而 KL 散度本身始终为正。这种高方差使得 K1 在实际应用中表现不佳。
### K2 有偏低方差

$$K_2 = \mathbb{E}_{x \sim q}\left[\frac{(\log r)^2}{2}\right]$$

- **特点 1**：有偏，但方差显著低于 K1——每个样本 $\frac{(\log r)^2}{2} \geq 0$，与 KL 的非负性一致
- **特点 2**：$K_2$ 的期望是一个 f-散度：$\mathbb{E}[K_2] = D_f(q \| p)$，其中 $f(r) = \frac{(\log r)^2}{2}$

当 $p$ 和 $q$ 接近时，所有可微的 f-散度在二阶近似下与 KL 散度等价。

### K3 无偏低方差

为了兼顾无偏和低方差，Schulman 引入了控制变量（control variate）方法。利用 $\mathbb{E}_{x \sim q}[r - 1] = 0$，构造：

$$K_3 = \mathbb{E}_{x \sim q}\left[(r - 1) - \log r\right]$$

由于 $\log$ 的凹性：$r - 1 \geq \log r$（Jensen 不等式），所以 **$K_3$ 恒非负、无偏且低方差**——这是 GRPO 的默认选择。

## 代码验证

```
*import* torch.distributions *as* dis
p = dis.Normal(*loc*=0, *scale*=1)
q = dis.Normal(*loc*=0.1, *scale*=1)
x = q.sample(*sample_shape*=(10_000_000,))
truekl = dis.kl_divergence(q, p)
print("true", truekl)
logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr
*for* k *in* (k1, k2, k3):
    print(k.mean(), (k.mean() - truekl) / truekl, k.std() / truekl)
```

# **二、正向KL vs. 反向KL**

- **正向KL**：$D_{\text{KL}}(P \| Q)$ — 真实分布 $P$ 相对于模型分布 $Q$
- **反向KL**：$D_{\text{KL}}(Q \| P)$ — 模型分布 $Q$ 相对于真实分布 $P$
## **正向KL散度：Mean-seeking**

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = -\sum_x P(x) \log Q(x) + \text{const}$$

由于 $H(P)$ 不依赖于 $Q$，最小化 $D_{\text{KL}}(P \| Q)$ 等价于最大化 $\mathbb{E}_{x \sim P}[\log Q(x)]$。

**关键特征**：期望在真实分布 $P$ 下计算。

当 $P$ 是多模态分布时：

- 优化目标要求 $Q$ 在 $P$ 的**所有**高概率区域都要有合理概率密度
- 如果 $Q$ 忽略 $P$ 的某个模态，该区域的 $\log Q(x) \to -\infty$，导致目标函数变差

**结果**：$Q$ 倾向于"覆盖"$P$ 的所有模态（mean-seeking），即使要在模态之间分配概率质量，导致模糊的平均化效果。

## **反向KL散度：Mode-seeking**

$$D_{\text{KL}}(Q \| P) = \sum_x Q(x) \log \frac{Q(x)}{P(x)} = -\sum_x Q(x) \log P(x) + \sum_x Q(x) \log Q(x)$$

等价于最大化 $\mathbb{E}_{x \sim Q}[\log P(x)] + H(Q)$。

**关键特征**：期望在模型分布 $Q$ 下计算。

当 $P$ 是多模态分布时：

- 优化目标鼓励 $Q$ 将概率质量集中在 $P$ 的**高概率**区域
- 如果 $Q$ 在 $P$ 的低概率区域分配质量，$\log P(x)$ 很小，拖低目标函数
- 熵项 $H(Q)$ 鼓励分散，但第一项影响更强

**结果**：$Q$ 倾向于"选择"$P$ 的一个或少数几个主要模态（mode-seeking），忽略其他模态。

> 来源：TRPO arXiv:1502.05477 中使用 Reverse KL 约束策略更新，正是利用了 mode-seeking 特性
## 对比

- 可视化：[csc413-2020.github.io](https%3A%2F%2Fcsc413-2020.github.io%2Fassets%2Ftutorials%2Ftut09_infotheory.pdf)
![image](LSJNd9ocfoVOOpxOHXVcpxHfn3e.png)

### 应用对比

| 方向 | 应用 | 原因 |
|------|------|------|
| Forward KL | GAN 训练 | 希望生成样本覆盖数据所有变化 |
| Forward KL | VAE 正则化 | 约束 $Q(z)$ 接近先验 $P(z) = \mathcal{N}(0, I)$ |
| Reverse KL | 模型蒸馏 | 学生集中学教师的主要模式 |
| Reverse KL | RL 对齐（[[GRPO 深度理解\|GRPO]]） | $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 防止新策略偏离参考策略太远 |

> 延伸阅读：[Reverse vs Forward KL](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
# 三、GRPO 应用

GRPO 中使用的是**反向 KL + K3 估计器**：

> 来源：Shao et al., "DeepSeekMath" arXiv:2402.03300, Sec. 3.2

- $\pi_\theta$ 是新模型，$\pi_{\text{ref}}$ 是参考模型
- $r = \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}$
- KL 惩罚项：$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \approx \mathbb{E}_{a \sim \pi_\theta}[(r - 1) - \log r]$（K3 估计器）

**代码验证：**

- 无偏
```
*import* torch.distributions *as* dis
*import* torch
p = dis.Normal(*loc*=0, *scale*=1)
q = dis.Normal(*loc*=0.1, *scale*=1)
x = q.sample(*sample_shape*=(10_000_000,))
torch.sum((q.log_prob(x) - p.log_prob(x)) < 0)
```

- 非负
```
*import* matplotlib.pyplot *as* plt
*import* numpy *as* np
xs = np.arange(0.01, 5, 0.01)
plt.plot(np.log(xs), *label*=r'$\log x$')
plt.plot(xs-1, *label*='$x-1$')
plt.legend()
```

## 📚 推荐阅读

### 原始论文
- [Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) — K1/K2/K3 估计器的原始推导
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) — KL 约束在 RL 策略优化中的经典应用
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO 中 K3 估计器的实际使用

### 深度解读
- [KL 散度的霍夫曼编码直觉（知乎）](https://www.zhihu.com/question/345907033/answer/3072696582) — 信息论角度理解 KL 散度 ⭐⭐⭐⭐
- [Reverse vs Forward KL Visualization](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html) — Mean-seeking vs Mode-seeking 可视化

### 实践资源
- [chunhuizhang/llm_rl: GRPO KL notebook](https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/r1-k1.5/grpo_kl.ipynb) — K1/K2/K3 代码验证
- [CSC413 Info Theory Tutorial (PDF)](https://csc413-2020.github.io/assets/tutorials/tut09_infotheory.pdf) — Forward/Reverse KL 可视化教程

## 🔧 落地应用

### 直接可用场景
- **GRPO/PPO 的 KL 惩罚实现**：K3 估计器是 TRL `GRPOTrainer` 的默认 KL penalty（`kl_type="k3"`）
- **模型蒸馏 loss 设计**：Reverse KL 让学生模型集中学教师的高置信输出

### 工程实现要点
- **K3 公式**：`kl = (ratio - 1) - torch.log(ratio)`，其中 `ratio = exp(logprob_new - logprob_ref)`
- **β 系数调优**：GRPO 中 KL 惩罚系数 $\beta$ 通常取 0.01-0.1，太大限制探索，太小导致策略漂移
- **数值稳定性**：`ratio` 接近 0 时 `log(ratio)` 会溢出，实践中做 clamp

### 面试高频问法
- Q: Forward KL 和 Reverse KL 的直觉区别？为什么 RLHF 用 Reverse？
  A: Forward KL（$D_{\text{KL}}(P \| Q)$）是 mean-seeking——$Q$ 必须覆盖 $P$ 的所有模态；Reverse KL（$D_{\text{KL}}(Q \| P)$）是 mode-seeking——$Q$ 倾向集中在 $P$ 的主模态。RLHF 用 Reverse 是因为我们希望新策略 $\pi_\theta$ 集中在参考策略 $\pi_{\text{ref}}$ 的高概率区域，而非强制覆盖所有可能输出。

## 💡 启发与思考

### So What？对老板意味着什么
- KL 散度是 RL 对齐的"安全阀"——理解 K1/K2/K3 的偏差-方差 trade-off 是调优 GRPO/PPO 的基础
- Forward/Reverse 的选择不是随意的——它决定了模型更新的"保守程度"

### 未解问题与局限
- K3 虽然无偏低方差，但在 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 差异很大时，单点估计仍可能不准
- 是否存在比 K3 更优的 KL 估计器？自适应方差的估计器（如 RLOO）是否更好？

### 脑暴：如果往下延伸
- 如果把 [[PPO 原理|PPO]] 的 clip ratio 和 KL 惩罚统一到一个框架，能否得到更优的策略约束？
- Jensen-Shannon 散度（Forward + Reverse 的对称平均）在 RLHF 中是否有应用潜力？

## 相关

> 🔗 See also: [[GRPO 深度理解|GRPO 深度理解]] — 使用 Reverse KL + K3 的主流 RL 对齐方法
> 🔗 See also: [[PPO 原理|PPO 原理]] — KL 约束的另一种实现（clip ratio）
> 🔗 See also: [[信息论|信息论]] — KL 散度的数学基础

- [[概率与分布|概率与分布]]
- [[DPO-TRL实践|DPO]]
