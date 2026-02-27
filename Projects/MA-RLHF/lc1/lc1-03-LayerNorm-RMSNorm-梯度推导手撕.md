---
title: "LayerNorm & RMSNorm 梯度推导手撕"
brief: "MA-RLHF 深化：LayerNorm / RMSNorm 手动梯度推导完整实现（含 Jacobian 矩阵推导）+ RMSNorm 不变性分析（前向尺度不变性 + 反向梯度不变性），代码验证与 PyTorch autograd 结果一致。面试被问'手写归一化层反向传播'时的完整答案。"
type: code-practice
date: 2026-02-28
tags:
  - ma-rlhf
  - layernorm
  - rmsnorm
  - backprop
  - gradient
  - code-practice
source: "MA-RLHF notebook/LayerNorm_and_RMSNorm_analysis.ipynb (https://github.com/dhcode-cpp/MA-RLHF)"
related:
  - "[[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|lc1-02 基础数学组件手撕]]"
  - "[[AI/1-Foundations/DL-Basics/Layer Normalization|Layer Normalization 详解]]"
---

# LayerNorm & RMSNorm 梯度推导手撕

> 本笔记是 [[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|lc1-02]] 的深化专项。
> lc1-02 覆盖了实现，这里专攻**手动梯度推导**——面试级别的反向传播推导能力。

---

## 核心问题

给定 LayerNorm 层：$y = \text{LN}(W_1 x)$，损失 $L$，求 $\frac{\partial L}{\partial W_1}$。

链式法则分解：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x'} \cdot \frac{\partial x'}{\partial W_1}$$

其中 $x' = W_1 x$（线性变换输出），$y = \text{LN}(x')$（归一化输出）。

难点在第二项：LayerNorm 的 $\frac{\partial y}{\partial x'}$ 是一个满矩阵 Jacobian，不是对角矩阵。

---

## 一、LayerNorm 梯度推导

### 1.1 前向公式

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：$\mu = \frac{1}{d}\sum x_i$，$\sigma^2 = \frac{1}{d}\sum(x_i-\mu)^2$，$\gamma, \beta \in \mathbb{R}^d$ 是逐特征可学习参数。

### 1.2 损失对参数的梯度

**对 $\gamma$ 的梯度**：

$$\frac{\partial L}{\partial \gamma} = \sum_{\text{tokens}} \frac{\partial L}{\partial y} \odot \hat{x}, \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

```python
dgamma = (out_mean_var * de_dy).sum(dim=1)
```

**对 $\beta$ 的梯度**：

$$\frac{\partial L}{\partial \beta} = \sum_{\text{tokens}} \frac{\partial L}{\partial y}$$

```python
dbeta = de_dy.sum(dim=1)
```

### 1.3 对输入 x 的梯度（Jacobian 推导）

LayerNorm 的 Jacobian $J \in \mathbb{R}^{d \times d}$（逐 token 独立）：

$$J_{ij} = \frac{\gamma_i}{\sigma} \left[\delta_{ij} - \frac{1}{d} - \frac{(x_i - \mu)(x_j - \mu)}{d\sigma^2}\right]$$

矩阵形式（element-wise $\gamma$ 乘法）：

$$J = \frac{1}{\sigma}\left[I - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{(x-\mu)(x-\mu)^T}{d\sigma^2}\right] \odot \gamma$$

**三项含义**：
1. $I/\sigma$：直接通路（$x_i$ 对自身 $y_i$ 的影响）
2. $-\frac{1}{d\sigma}\mathbf{1}\mathbf{1}^T$：均值项（均值被所有维度共享）
3. $-\frac{(x-\mu)(x-\mu)^T}{d\sigma^3}$：方差项（方差被所有维度共享）

```python
I = torch.ones(d_model, d_model)
diag_I = torch.eye(d_model)

grad_x_ln = torch.zeros_like(x_prime)
for i in range(batch_size):
    for j in range(seq_len):
        # Jacobian 矩阵（逐 token 计算）
        J = (
            (diag_I - 1/d_model * I) / x_std[i,j,0]
            - x_centered[i,j,:].outer(x_centered[i,j,:]) / (x_std[i,j,0]**3 * d_model)
        ) * ln.gamma
        # 上游梯度左乘 Jacobian
        grad_x_ln[i,j,:] = de_dy[i,j,:] @ J
```

**为什么 Jacobian 是满矩阵**：因为 $\mu$ 和 $\sigma^2$ 是所有维度共享的全局统计量，改变任意 $x_j$ 都会改变所有 $y_i$。

### 1.4 验证结果

```python
# 手动梯度 vs PyTorch autograd 结果一致 ✅
print(grad_x_ln)   # 手动推导
print(x_prime.grad)  # autograd
```

---

## 二、RMSNorm 梯度推导

### 2.1 前向公式

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$$

与 LN 相比：**去掉了减均值（re-centering）**，只保留 re-scaling。

### 2.2 Jacobian 矩阵

$$J_{ij} = \frac{\gamma_i}{\text{RMS}(x)}\left[\delta_{ij} - \frac{x_i x_j}{d \cdot \text{RMS}(x)^2}\right]$$

只有两项（对比 LN 的三项）：直接通路 + RMS 共享项。

```python
I_diag = torch.eye(d_model)

grad_x_rms = torch.zeros_like(x_prime)
for i in range(batch_size):
    for j in range(seq_len):
        J = (
            I_diag / rms[i,j,0]
            - x_prime[i,j,:].outer(x_prime[i,j,:]) / (rms[i,j,0]**3 * d_model)
        ) * rms_norm.gamma
        grad_x_rms[i,j,:] = de_dy[i,j,:] @ J
```

---

## 三、RMSNorm 不变性分析

### 3.1 前向：尺度不变性

$$\text{RMSNorm}(\alpha x) = \text{RMSNorm}(x), \quad \forall \alpha > 0$$

证明：$\text{RMS}(\alpha x) = \alpha \cdot \text{RMS}(x)$，分子分母的 $\alpha$ 约掉。

```python
# 实验验证
out_10 = rmsnorm(x * 10)
out_1  = rmsnorm(x)
assert torch.allclose(out_10, out_1)  # True ✅
```

### 3.2 反向：梯度不变性

对于输入 $\alpha x$，关于原始 $x$ 的梯度与 $\alpha$ 无关。

直觉：前向不变 → 上游梯度不变 → 反传时 $\alpha$ 在链式法则中被约掉。

```python
# alpha=10 和 alpha=1 时，关于原始 x 的梯度相同 ✅
assert torch.allclose(grad_x_rms_10, grad_x_rms_1)
```

**实践意义**：前层激活值的尺度扰动不会放大到梯度中，训练对初始化和学习率更鲁棒。

### 3.3 梯度大小对比

```python
print(rms_w_gradient.norm())  # ~= ln_w_gradient.norm()
print(ln_w_gradient.norm())
```

两者梯度范数相近，但 RMSNorm 的计算路径更简洁，GPU 上更友好。

---

## 四、去 Center 化的深层意义

RMSNorm 去掉 $-\mu$ 的理论依据（苏剑林）：

> Center 操作（减均值）等价于储存预训练任务的先验分布信息在模型参数里，这反而降低了模型的迁移能力。

推论链：
- $\mu$ 是对当前输入分布的 hard-coding → 特定任务偏置
- 去掉 $\mu$ → 模型学到更通用的 scaling → 迁移能力更强
- T5 因此同时去掉了 center 和所有层的 bias

这也解释了为什么 RMSNorm 在大规模预训练 + 微调场景下（即现代 LLM 的标准范式）优于 LayerNorm。

---

## 五、面试一句话答案

**Q：手写 LayerNorm 反向传播，给出关于输入 $x$ 的梯度。**

> **A**：LayerNorm 的 Jacobian 是满矩阵（非对角），因为均值和方差是所有维度共享的全局统计量。$J = \frac{\gamma}{\sigma}[I - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{(x-\mu)(x-\mu)^T}{d\sigma^2}]$，三项分别对应直接通路、均值共享、方差共享。梯度 $= \frac{\partial L}{\partial y} \cdot J$，逐 token 独立计算。

**Q：RMSNorm 和 LayerNorm 的梯度有什么区别？**

> **A**：RMSNorm 的 Jacobian 少了均值项，只有两项：$J = \frac{\gamma}{\text{RMS}}[I - \frac{xx^T}{d\cdot\text{RMS}^2}]$，计算更简洁，且具有前向尺度不变性和反向梯度不变性（输入乘以常数不影响梯度），训练更稳定。

---

## See Also

- [[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|lc1-02：基础数学组件手撕]] — 上游：LayerNorm/RMSNorm 正向实现
- [[AI/1-Foundations/DL-Basics/Layer Normalization|Layer Normalization 详解]] — 横向理论：BN vs LN vs RMSNorm vs DeepNorm，Pre/Post-Norm 对比
- [[AI/3-LLM/Architecture/Attention 变体综述|Attention 变体综述]] — 下游：各模型架构中 Norm 的使用位置

*写作时间：2026-02-28 05:15 | MA-RLHF Batch D 收尾，notebook/LayerNorm_and_RMSNorm_analysis.ipynb（53 cells）*
