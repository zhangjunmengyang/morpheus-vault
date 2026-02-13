---
title: "K1-K3 & Forward vs. Reverse KL散度"
type: concept
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/concept
---
#  K1-K3 & Forward vs. Reverse KL散度

# 一、KL 散度

https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/r1-k1.5/grpo_kl.ipynb

## 基本概念

用于衡量两个概率分布 p 和 q 之间的差异，对于离散随机变量，公式定义为：

- ：真实分布的概率质量函数。
- ：近似分布的概率质量函数。
每一项  表示在事件 x 上，真实分布 P 与近似分布 Q 的差异加权后的信息损失。

## 关键性质

- **非对称性**​：首先 KL 散度并不是距离，最主要是非对称性（也不满足三角不等式），。
- **KL 散度是否代表分布之间的距离？**不代表距离，因为不满足对称性和三角不等式
- 其他角度理解本质：[霍夫曼编码角度](https%3A%2F%2Fwww.zhihu.com%2Fquestion%2F345907033%2Fanswer%2F3072696582)，KL散度其实是利用最优编码这个代理量，去衡量两个分布之间有多大的区别。
- **非负性**​：
- **与交叉熵关系**​：最小化交叉熵等价于最小化 KL 散度（当 P 固定时），KL 散度可分解为：
在实际应用中，精确计算KL散度通常不可行，因为：

1. **成本过高**：计算所有 ( xx*x* ) 的概率和需要过多的计算或内存。
1. **无法计算**：分布可能没有闭合表达式，无法解析求解。
蒙特卡洛方法成为估计 KL 散度的常用策略。假设我们从分布 q 中采样，

如何构造一个**无偏、低方差的估计器**是关键问题。

## K1 K2 K3

### K1 无偏高方差

最直接的蒙特卡洛估计器是基于KL散度的定义：

![image](assets/YrsudpfnPoiBuAxmzXucrB8Gn5e.png)

- **特点**：这个估计器是**无偏的**，即其期望等于真实的KL散度。然而，由于 log(r) 的值在正负之间变化（当 r > 1 时为正，当 r < 1时为负），其**方差较高**，而 KL 散度本身始终为正。这种高方差使得 K1 在实际应用中表现不佳。
### K2 有偏低方差

Schulman提出了一种替代估计器 ([Approximating KL Divergence](http%3A%2F%2Fjoschu.net%2Fblog%2Fkl-approx.html))：

![image](assets/JfGydQHT4o8w68xY9pXcPBlNner.png)

- 特点 1：有偏，但是方差显著低于 K1（都是正数），与 KL 性质一致，每个样本都反映了 p 和 q 之间的差异。
- 特点 2：**低偏倚，K2 的期望是一个f-散度，其形式为：**
![image](assets/DxEUd3ZR5oYa24xde7Fcg5clnkc.png)

当 p 和 q 接近时，所有可微的f-散度在二阶近似下与KL散度等价。

### K3 无偏低方差

为了兼顾无偏和低方差，Schulman引入了控制变量（control variate）方法。利用 

可以构造一个新的估计器：

*λ*=1 时，估计器变为：

由于对数的凹性（或者基于泰勒展开），K3 是恒为正的，无偏且低方差

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

- **正向KL**:  - 真实分布  相对于模型分布 
- **反向KL**:  - 模型分布  相对于真实分布 
## **正向KL散度：Mean-seeking**

展开：

由于  不依赖于 ，可以去掉：

等价于：

**关键特征**：期望是在真实分布P下计算的

当P是多模态分布时：

- 优化目标要求 在  的**所有**高概率区域都要有合理的概率密度
- 如果  忽略  的某个模态，该区域的  会变得很小（负无穷），导致整体目标函数值变差
**结果**：  倾向于"覆盖"P的所有模态，即使这意味着要在模态之间分配概率质量，导致模糊的平均化效果。

## **反向KL散度：Mode-seeking**

展开：

等价于：

**关键特征**：期望是在模型分布  下计算的

当  是多模态分布时：

- 优化目标鼓励 将概率质量集中在P的**高概率**区域

- 如果 在 的低概率区域分配质量， 很小，降低目标函数值
- 熵项  鼓励分散，但第一项的影响通常更强
**结果**： 倾向于"选择"P的一个或少数几个主要模态，忽略其他模态，导致尖锐的模式搜索效果。

的样本主要来自其高概率区域。如果 试图覆盖 的多个模态：

- 在模态之间的低概率区域， 很小， 接近负无穷
- 这些区域的贡献会拖累整体目标函数值
- 因此 更倾向于集中在  的某个单一模态上
## 对比

- 可视化：[csc413-2020.github.io](https%3A%2F%2Fcsc413-2020.github.io%2Fassets%2Ftutorials%2Ftut09_infotheory.pdf)
![image](assets/LSJNd9ocfoVOOpxOHXVcpxHfn3e.png)

- 应用
- **forward**
- **生成模型：比如 GAN 的训练，**希望生成样本覆盖数据的所有变化，
- **VAE**​：KL 散度约束隐变量分布 Q(z) 接近先验分布（如标准正态分布）。
- **reverse**
- **模型蒸馏**：学生模型集中学习教师模型的主要行为模式
- 
- **强化学习**：
- GRPO： ，防止新的模型分布相较原来偏离太远。
- 其他材料：[Reverse vs Forward KL](https%3A%2F%2Fwww.tuananhle.co.uk%2Fnotes%2Freverse-forward-kl.html)
# 三、GRPO 应用

GRPO 中使用的是** 反向、K3 散度**

-  是新模型，  是原始模型
- 
- 
K3 项

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
