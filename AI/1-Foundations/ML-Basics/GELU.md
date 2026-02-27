---
brief: "GELU（Gaussian Error Linear Unit）——平滑近似 ReLU 的激活函数；BERT/GPT 系列的标准激活函数选择；SwiGLU（Swish+GLU）是现代 LLM 的进一步改进，FFN 参数减少 2/3 但性能持平；激活函数选型的面试参考。"
title: "GELU"
type: concept
domain: ai/foundations/ml-basics
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/foundations/ml-basics
  - type/concept
---
# GELU

# 一、原理

## Relu 问题

一般的激活函数使用 ReLU，但其为分段函数，在 0 点处不可微。

GELU(Gaussian Error Linear Unit) 是 ReLU 的近似。

1. 曲线在趋近  , 
1. 定义 , 找到一个  有上述特性的函数
**高斯概率密度函数 (PDF):**

**高斯累积分布函数 (CDF):**

其中：

-  是均值
-  是标准差
-  是误差函数
# 二、手撕实现

## 2.1 误差函数 erf 实现

### 高斯分布版本实现

```
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import math

mu = 0.0      # 均值
sigma = 1.0   # 标准差

plot_x = torch.linspace(-10, 10, 1000)
def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 累积分布函数 (CDF) - 使用误差函数
def gaussian_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(torch.tensor(2.0)))
    return 0.5 * (1 + torch.erf(z))

pdf_values = gaussian_pdf(plot_x, mu, sigma)
cdf_values = gaussian_cdf(plot_x, mu, sigma)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制PDF
plt.subplot(2, 1, 1)
plt.plot(plot_x.numpy(), pdf_values.numpy(), 'b-', linewidth=2, label='PDF')
plt.plot(plot_x.numpy(), cdf_values.numpy(), 'r-', linewidth=2, label='CDF')
# plt.plot(x.numpy(), pdf_values_torch.numpy(), 'r--', linewidth=1, alpha=0.7, label='PDF (Torch Distribution)')
plt.title('Norm.Distribute, PDF/CDF')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()

plt.legend()

y_cdf = gaussian_cdf(x, mu, sigma)

x_list = x.tolist()
y_relu_list = y_relu.tolist()
y_gelu_list = y_gelu.tolist()
y_cdf_list = (x * y_cdf).tolist()

plt.figure(figsize=(10, 4))

plt.plot(x_list, y_relu_list, label='ReLU', color='blue', linewidth=2)
plt.plot(x_list, y_gelu_list, label='GELU', color='red', linewidth=2)
plt.plot(x_list, y_cdf_list, label='CDF', color='green', linewidth=2)

plt.title('ReLU vs GELU vs CDF', fontsize=14)
plt.xlabel('x ', fontsize=12)
plt.ylabel('y ', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### 标准正态分布实现（GELU 简化版本）

```
import torch
import math

def ReLU(x):
    return (x + torch.abs(x)) * 0.5

def GELU(x):
    '''
    先不分析此代码, 近用于绘图展示 GELU 的激活曲线
    '''
    # cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf

x = (torch.arange(200) - 100) / 10
y_relu = ReLU(x)
y_gelu = GELU(x)

import matplotlib.pyplot as plt
import numpy as np

x_list = x.tolist()
y_relu_list = y_relu.tolist()
y_gelu_list = y_gelu.tolist()

plt.figure(figsize=(10, 4))

plt.plot(x_list, y_relu_list, label='ReLU', color='blue', linewidth=2)
plt.plot(x_list, y_gelu_list, label='GELU', color='red', linewidth=2)

plt.title('ReLU vs GELU', fontsize=14)
plt.xlabel('x ', fontsize=12)
plt.ylabel('y ', fontsize=12)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

## 2.2 tanh 近似

```python
def GELU_approx_1(x):
    cdf = 0.5 * (1.0 + torch.tanh( math.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf
```

## 2.3 sigmoid 近似

```python
def GELU_approx_2(x):
    cdf = torch.sigmoid(1.702 * x)
    return x * cdf
    
approx_1 = GELU_approx_1(x)
approx_2 = GELU_approx_2(x)

y_cdf = gaussian_cdf(x, mu, sigma)

x_list = x.tolist()
y_relu_list = y_relu.tolist()
y_gelu_list = y_gelu.tolist()
y_cdf_list = (x * y_cdf).tolist()
y_approx_1_list = approx_1.tolist()
y_approx_2_list = approx_2.tolist()

plt.figure(figsize=(10, 4))

plt.plot(x_list, y_relu_list, label='ReLU', color='blue', linewidth=1, alpha = 0.5)
plt.plot(x_list, y_gelu_list, label='GELU', color='red', linewidth=1, alpha = 0.5)
plt.plot(x_list, y_cdf_list, label='GELU CDF', color='green', linewidth=1, alpha = 0.5)
plt.plot(x_list, y_approx_1_list, label='GELU approx1', color='pink', linewidth=1, alpha = 0.5)
plt.plot(x_list, y_approx_2_list, label='GELU approx2', color='black', linewidth=2, alpha = 0.5)

plt.title('ReLU vs GELU vs CDF', fontsize=14)
plt.xlabel('x ', fontsize=12)
plt.ylabel('y ', fontsize=12)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

# 三、CDF 推导

待补充

---

## See Also

- [[AI/1-Foundations/ML-Basics/激活函数|激活函数]] — GELU 所属的激活函数家族：ReLU/GELU/SiLU 对比
- [[AI/1-Foundations/ML-Basics/SwiGLU|SwiGLU]] — 进化版：SwiGLU = Swish(xW) × (xV)，实验上比 GELU 提升 ~1%（LLaMA/PaLM 均采用）
- [[AI/3-LLM/SFT/LoRA|LoRA]] — 共同话题：激活函数选择对低秩适配的影响
- [[AI/1-Foundations/DL-Basics/Layer Normalization|Layer Normalization]] — 同为 Transformer FFN 基础组件（激活函数 × 归一化共同设计）
