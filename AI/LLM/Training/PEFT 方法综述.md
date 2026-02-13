---
title: "PEFT 方法综述：参数高效微调技术全景"
date: 2026-02-13
tags:
  - ai/llm/training
  - ai/fine-tuning
  - ai/peft
  - type/survey
  - interview/hot
status: active
---

# PEFT 方法综述：参数高效微调技术全景

> Parameter-Efficient Fine-Tuning (PEFT)——用 <1% 的可训练参数达到全量微调 95%+ 的效果

## 1. 为什么需要 PEFT

```
全量微调 LLaMA 3 70B:
  可训练参数: 70B → 显存需求 ~560 GB (FP16 + Adam)
  单次实验: 8× A100 80GB, 数天训练

PEFT (LoRA r=16):
  可训练参数: ~100M (0.14%) → 显存需求 ~160 GB (FP16 冻结 + adapter)
  单次实验: 2× A100 80GB, 数小时训练
```

**PEFT 的核心假设**：任务适配所需的权重变化位于一个**低维子空间**（Intrinsic Dimensionality）。

## 2. PEFT 方法分类

```
PEFT
├── Addition-based（增加参数）
│   ├── Adapter（串联模块）
│   ├── Prefix Tuning（前缀软提示）
│   └── Prompt Tuning（输入软提示）
├── Reparameterization-based（重参数化）
│   ├── LoRA（低秩适配）
│   ├── QLoRA（量化 + LoRA）
│   ├── DoRA（方向 + 幅度分解）
│   ├── AdaLoRA（自适应秩分配）
│   └── VeRA（共享随机矩阵）
└── Selective（选择性微调）
    └── IA³（抑制与放大适配器）
```

## 3. LoRA：基石方法

### 核心思想

冻结预训练权重 $W_0$，用低秩矩阵 $BA$ 近似权重更新：

$$W = W_0 + \frac{\alpha}{r} BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # 冻结原权重

        self.lora_A = nn.Linear(in_features, r, bias=False)   # 下投影
        self.lora_B = nn.Linear(r, out_features, bias=False)   # 上投影
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)  # 初始化为 0，训练开始时不改变输出

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merge(self):
        """推理时合并权重，零额外延迟"""
        self.linear.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scaling
```

### 关键超参数

```
r (rank):     4-64, 越大表达力越强，参数越多
alpha:        通常 = 2r, 控制适配强度
target:       通常 q_proj, v_proj; 全部投影效果更好
dropout:      0.05-0.1, 防过拟合
```

## 4. QLoRA：量化驱动的显存革命

### 核心创新

在 LoRA 基础上将冻结的基座模型量化到 4-bit，大幅降低显存：

```
QLoRA = 4-bit NormalFloat 量化 + Double Quantization + Paged Optimizers + LoRA

显存对比 (LLaMA 65B):
  全量 FP16:    ~780 GB (不可能)
  LoRA FP16:    ~130 GB (2× A100)
  QLoRA 4-bit:   ~33 GB (单张 A100!) ← 突破性节省
```

```python
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (正态分布优化)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # 二次量化，进一步节省 ~0.4 bit/param
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B", quantization_config=bnb_config
)

# 在量化模型上加 LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)
```

### NormalFloat4 (NF4)

QLoRA 的关键创新——专门为正态分布权重设计的量化数据类型：

```
FP4:  均匀量化 → 对正态分布权重有偏
NF4:  按正态分布分位点划分 → 每个量化 bin 包含等概率密度的权重
      理论上是信息论最优的 4-bit 正态量化
```

## 5. DoRA：方向与幅度的解耦

### 核心洞察

全量微调改变权重的**方向**和**幅度**，但 LoRA 主要改变方向。DoRA 显式分解两者：

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

- $m$：可训练的幅度向量（per-column 标量）
- $\frac{W_0 + BA}{\|W_0 + BA\|_c}$：方向分量（通过 LoRA 更新）

```python
class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA 分量（方向更新）
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

        # 可训练幅度向量
        self.magnitude = nn.Parameter(self.weight.norm(dim=1))  # (out_features,)

    def forward(self, x):
        adapted_weight = self.weight + self.lora_B.weight @ self.lora_A.weight
        # 列归一化
        column_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / column_norm
        # 幅度 × 方向
        final_weight = self.magnitude.unsqueeze(1) * direction
        return x @ final_weight.T
```

**效果**：在多数任务上比 LoRA 提升 1-3%，尤其是常识推理和指令遵循任务。

## 6. AdaLoRA：自适应秩分配

### 核心思想

不同层对任务适配的**重要性不同**，应该自适应分配秩：

```
标准 LoRA:  所有层 r=16（均匀分配）
AdaLoRA:   重要层 r=32, 不重要层 r=4（自适应分配）

总参数预算相同，但效果更好
```

### 实现机制

```
1. SVD 参数化: ΔW = P Λ Q (代替 BA)
   P: 左奇异向量
   Λ: 奇异值（对角矩阵）
   Q: 右奇异向量

2. 重要性评分:
   importance(i) = |λ_i| × sensitivity(layer)
   sensitivity = 梯度大小的移动平均

3. 动态剪枝:
   训练过程中逐步将不重要的奇异值置零
   高重要性层保留更多秩 → 高重要性层 r ↑, 低重要性层 r ↓
```

## 7. Prefix Tuning & Prompt Tuning

### Prefix Tuning

在每层 attention 的 K/V 前拼接可训练的「虚拟 token」：

```python
class PrefixTuning(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, prefix_len=20):
        super().__init__()
        # 每层 attention 的 prefix K/V
        self.prefix_k = nn.Parameter(torch.randn(n_layers, prefix_len, n_heads, d_k))
        self.prefix_v = nn.Parameter(torch.randn(n_layers, prefix_len, n_heads, d_k))
        # 通过 MLP 重参数化以稳定训练
        self.reparameterize = nn.Sequential(
            nn.Linear(n_heads * d_k, 512),
            nn.Tanh(),
            nn.Linear(512, n_heads * d_k)
        )
```

### Prompt Tuning

更简单——只在输入 embedding 前拼接可训练向量：

```
[trainable_soft_prompt | actual_input_tokens] → LLM

可训练参数: prefix_len × d_model
例: 20 × 4096 = 81,920 参数 (~0.001% of 7B model)
```

**局限**：模型规模 < 10B 时效果不佳，且不如 LoRA 灵活。

## 8. Adapter

在 Transformer 的每个子层后插入小型瓶颈网络：

```
     ┌─────────────┐
x →  │ Self-Attn    │ → + → LayerNorm → ┌────────┐ → + → ...
     └─────────────┘                     │Adapter │
                                         │Down(d→r)│
                                         │ReLU     │
                                         │Up(r→d)  │
                                         └────────┘
```

可训练参数：每个 Adapter = $2 \times d \times r + r$（bias）。

**缺点**：增加推理延迟（串联在主路径上，无法合并）。

## 9. IA³：抑制与放大

### 核心思想

不新增矩阵，而是在 K/V/FFN 上乘以可训练的缩放向量：

$$\text{IA}^3: \quad k = l_k \odot W_k x, \quad v = l_v \odot W_v x, \quad \text{FFN}_{out} = l_{ff} \odot W_{ff} x$$

```python
# IA3 极简实现
class IA3Linear(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        # 只有一个缩放向量！
        self.ia3_scaling = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, x):
        return self.linear(x) * self.ia3_scaling  # 逐元素缩放
```

**参数量极少**（仅 $d$ 个标量/层），但适配能力有限，适合 few-shot 场景。

## 10. 综合对比

| 方法 | 可训练参数 | 显存占用 | 推理延迟 | 质量(vs 全量) | 典型场景 |
|------|-----------|---------|---------|-------------|---------|
| 全量微调 | 100% | ★★★★★ | 无额外 | 基准 | 充足资源 |
| **LoRA** | 0.1-1% | ★★★☆☆ | 可合并=0 | 95-98% | **默认首选** |
| **QLoRA** | 0.1-1% | ★★☆☆☆ | 可合并=0 | 94-97% | **显存受限** |
| DoRA | 0.1-1%+α | ★★★☆☆ | 可合并=0 | 96-99% | 追求质量 |
| AdaLoRA | 0.1-1% | ★★★☆☆ | 可合并=0 | 96-99% | 异质任务 |
| Prefix Tuning | <0.1% | ★★☆☆☆ | 有（前缀拼接） | 90-95% | NLU 任务 |
| Adapter | 1-5% | ★★★☆☆ | 有（串联） | 95-98% | 多任务切换 |
| IA³ | <0.01% | ★☆☆☆☆ | 可合并=0 | 88-93% | Few-shot |

## 11. 选型决策树

```
显存够做 LoRA?
├── 是 → 需要极致质量?
│        ├── 是 → DoRA 或 AdaLoRA
│        └── 否 → LoRA (r=16-64, target=all-linear)
└── 否 → QLoRA (4-bit NF4)
          需要超低参数?
          ├── 是 → IA³ 或 Prompt Tuning
          └── 否 → QLoRA + r=16
```

**2025 年实践共识**：
- 大多数场景用 **LoRA** 或 **QLoRA**，target_modules 选 all-linear
- 追求质量用 **DoRA**（额外成本很小）
- 异质多任务场景试 **AdaLoRA**
- Prefix/Prompt Tuning 已逐渐边缘化

## 面试常见问题

### Q1: LoRA 为什么有效？底层数学直觉是什么？

LoRA 基于 **Intrinsic Dimensionality** 假说——预训练模型在下游任务适配时，权重变化 $\Delta W$ 的有效秩远低于其维度。Aghajanyan et al. (2021) 实验证明，即使将更新限制在 ~200 维子空间，仍能达到全量微调 90% 的效果。LoRA 用 $BA$ 的低秩分解显式利用了这一特性。$B$ 初始化为零保证训练开始时不改变模型输出，$\alpha/r$ 缩放控制适配强度。推理时 $BA$ 可合并进原权重，**零额外延迟**。

### Q2: QLoRA 的 NF4 为什么比 FP4 好？

预训练后的模型权重近似服从**正态分布** $\mathcal{N}(0, \sigma^2)$。FP4 均匀划分量化区间，导致靠近零的高概率区域精度不够、尾部低概率区域精度浪费。NF4 按正态分布的**等概率分位点**划分 16 个量化 bin，使每个 bin 内的权重数量均匀，达到信息论最优的量化精度。配合 Double Quantization（对量化常数再做量化），每个参数平均只需 ~4.5 bits。

### Q3: DoRA 相比 LoRA 的改进体现在哪里？

DoRA 的核心发现是：全量微调倾向于**大幅改变权重方向但小幅改变幅度**，而 LoRA 将方向和幅度变化耦合在一起，导致方向更新不够自由。DoRA 显式解耦为：幅度向量 $m$（少量标量参数）+ 方向更新（通过 LoRA）。这让方向更新可以更激进而不受幅度约束，在常识推理 (CommonsenseQA) 和指令遵循任务上提升 1-3%。额外参数仅为每层一个 $d$-维向量。

### Q4: 如何选择 LoRA 的 rank 和 target modules？

**Rank**：r=16 是稳健默认值。简单任务（风格迁移、格式对齐）r=4-8 足够；复杂任务（领域知识注入、多语言）r=32-64 更好。验证方法：若训练 loss 下降但验证 loss 饱和，说明 r 已足够。**Target modules**：2025 年共识是 `target_modules="all-linear"`（Q/K/V/O + FFN），比只选 q_proj/v_proj 效果显著更好。HuggingFace PEFT 库支持自动识别。

### Q5: PEFT 方法能否用于预训练（而非仅 SFT）？

可以但场景有限。**Continual Pre-training (CPT)** 场景下 LoRA 常用于领域适配（如法律/医学语料续训），但需要更高的 rank (r=64-256)。核心问题是：预训练的权重变化不一定是低秩的——新知识注入可能需要高秩更新。实践中 CPT 用 LoRA 的效果通常不如全量微调（差距 3-10%），但在资源受限时是可行折中。LoRA 更适合 **SFT/对齐** 阶段，因为格式/风格适配确实是低秩的。
