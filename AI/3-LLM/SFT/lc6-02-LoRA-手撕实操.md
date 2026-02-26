---
title: LoRA 手撕实操 · MA-RLHF lc6 Batch E
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - ma-rlhf
  - lc6
  - lora
  - peft
  - sft
  - fine-tuning
brief: LoRA 从零完整推导与实现：冻结预训练权重 W，旁路低秩分解 ΔW=W_A·W_B（r≪d）；初始化原则（W_A 高斯随机初始化而非全零，保证 W_B 梯度不消失）；rank/alpha/dropout 三超参调优；与 QLoRA/DoRA/LoRA+ 的对比；在 SFT/RLHF/DPO 流程中的应用位置。
related:
  - "[[LoRA]]"
  - "[[AI/LLM/SFT/SFT-手撕实操]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-GPTLoss-Muon优化器-手撕实操]]"
  - "[[AI/LLM/MA-RLHF课程/lc6-SFT全链路-MOC]]"
---

# LoRA 手撕实操（MA-RLHF Batch E）

> **来源**：`lecture/lc6_sft/LoRA.ipynb`（41 cells）
> **评级**：★★★★★
> **字数**：~8500

---

## TL;DR

LoRA（Low-Rank Adapter）是参数高效微调（PEFT）的核心方法：冻结原模型权重 W，在旁路加入低秩分解 $\Delta W = W_A W_B$（$r \ll d$），只训练 W_A 和 W_B。理论基础是：大模型参数矩阵本身具有低秩性，梯度矩阵在充分训练后低秩性更强。关键 insight：**WA 初始为 0（WB 随机），而非 WA/WB 全零**——全零初始化导致 WB 的梯度消失。

---

## 一、LoRA 的出发点

### 全参微调的代价

对 7B 模型全参微调：
- 参数：7B × 4 bytes = 28GB
- 梯度：28GB
- Adam 优化器状态（m + v）：56GB
- **总计 > 100GB，4×A100 才能放下**

### PEFT 的思路

冻结大部分参数，只训练少量附加参数。LoRA 选择的附加结构：**低秩矩阵旁路**。

为什么是低秩？两个实验支撑：
1. **参数矩阵低秩性**：对 512×512 矩阵做 SVD，仅前 4 个奇异值显著大，rank=4 重建误差极小
2. **梯度矩阵低秩性**：模型训练越充分，`Wv.weight.grad` 的奇异值分布越集中（前几个奇异值极大，其余接近 0）→ 梯度本身就"低维"

---

## 二、LoRA 数学推导

### 前向计算

原始线性层：$h = XW$

引入低秩旁路：$h' = XW + \alpha \cdot X W_A W_B$

其中：
- $W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$（冻结，不更新）
- $W_A \in \mathbb{R}^{d_\text{in} \times r}$（可训练，下投影）
- $W_B \in \mathbb{R}^{r \times d_\text{out}}$（可训练，上投影）
- $r \ll d$，通常 r=4~16
- $\alpha$：缩放超参数，控制旁路强度

### 优化目标

原来：$\arg\min_W \mathcal{L}(W)$

转化为：$\arg\min_{W_A, W_B} \mathcal{L}(W + \alpha W_A W_B)$，原参数 W 冻结

**参数量对比**：
- 原线性层：$d_\text{in} \times d_\text{out}$（如 512×512 = 262144）
- LoRA 旁路：$r \times (d_\text{in} + d_\text{out})$（如 4×1024 = 4096）
- 压缩比：~64x（r=4 时）

---

## 三、完整代码实现

### LoRALinear 层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.dim_in = original_linear.in_features
        self.dim_out = original_linear.out_features
        self.r = rank

        # 冻结原始权重（requires_grad=False）
        self.weight = nn.Parameter(
            original_linear.weight.data.clone(), requires_grad=False
        )
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

        # 可训练的低秩分解参数
        self.WA = nn.Linear(self.dim_in, self.r, bias=False)
        self.WB = nn.Linear(self.r, self.dim_out, bias=False)

        # ⚠️ 关键初始化：WA=0，WB 随机
        nn.init.constant_(self.WA.weight.data, 0.0)  # WA 全零
        # WB 默认 kaiming_uniform 初始化（随机）

    def forward(self, X):
        bsz, seq_len, dim_in = X.shape
        # 原始路径（冻结）
        h = X.reshape(bsz * seq_len, dim_in) @ self.weight.T
        h = h.reshape(bsz, seq_len, self.dim_out)
        if self.bias is not None:
            h = h + self.bias

        # LoRA 旁路（可训练）
        lora_out = self.alpha * self.WB(self.WA(X))
        return h + lora_out
```

### 自动替换模型中所有 Linear 层

```python
def apply_lora_adapter(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = LoRALinear(module, rank=rank)
            setattr(model, name, new_layer)
        else:
            apply_lora_adapter(module)  # 递归处理子模块
    return model

# 使用示例
model = Attention(dim=512, dim_out=512)  # 原模型
model = apply_lora_adapter(model, rank=4)  # 注入 LoRA

# 验证：只有 WA、WB 有梯度
for n, p in model.named_parameters():
    if p.grad is not None:
        print(n, p.grad.shape)
# 输出：
# Wq.WA.weight [4, 512]
# Wq.WB.weight [512, 4]
# Wk.WA.weight [4, 512]
# ...
```

### LoRA Merge（部署时融合）

```python
# 推理时可以把 LoRA 参数融合回原参数，消除旁路 overhead
W_merged = W + alpha * (WA @ WB)
# 等价于把旁路直接加进原参数矩阵
# forward 从 O(d + r*2) 变回 O(d)，latency 降低
```

---

## 四、初始化策略深度分析

### 三种初始化方案对比

以 X[2,4], WA[4,3], WB[3,4] 为例手推梯度：

**方案 0：WA=0, WB=0（全零）**
```python
YA = X @ WA = 0           # 全零
Y = YA @ WB = 0           # 全零
dWB = YA.T @ dY = 0       # YA 全零 → WB 梯度为 0！
dYA = dY @ WB.T = 0       # WB 全零 → dYA 为 0
dWA = X.T @ dYA = 0       # dYA 为 0 → WA 梯度也为 0！
```
**结论：WA=WB=0 → 梯度消失，永远无法更新！**

**方案 1：WA=0, WB=随机**
```python
YA = X @ WA = 0           # 全零（旁路输出为 0 ✓）
Y = YA @ WB = 0           # 旁路不影响原始输出 ✓
dWB = YA.T @ dY = 0       # WB 梯度为 0（WB 暂时无法更新）
dYA = dY @ WB.T ≠ 0      # WB 随机 → dYA 非零
dWA = X.T @ dYA ≠ 0      # dYA 非零 → WA 有梯度！
```
**结论：WA=0 初始化：旁路输出为 0 ✓，WA 有梯度 ✓，WB 初始无梯度（下一步 WA 非零后 WB 才有梯度）**

**方案 2：WA=随机, WB=0**
```python
YA = X @ WA ≠ 0           # YA 非零
Y = YA @ WB = 0           # 旁路输出为 0 ✓
dWB = YA.T @ dY ≠ 0      # YA 非零 → WB 立即有梯度！
dYA = dY @ WB.T = 0      # WB=0 → dYA=0
dWA = X.T @ dYA = 0      # dYA=0 → WA 梯度为 0
```
**结论：WB=0 初始化：旁路输出为 0 ✓，WB 立即有梯度（YA 非零作为 WB 的输入）**

### 哪种方案更优？

**官方 LoRA 选择方案 1（WA=0, WB 随机）**，原因：

从 WB 的视角看：
- 方案 1（WA=0, WB 随机）：WB 的输入 YA 初始为 0，WB 第一步无梯度 → **WB 的学习从第二步才开始**
- 方案 2（WA 随机, WB=0）：WB 的输入 YA 非零，WB 立即有梯度 → **WB 能立即开始学习**

从梯度流稳定性看：方案 2 中 WA 初始无梯度但 WB 立即学习，更快破坏零初始化约束，训练更不稳定。

**实践结论**：两种方案都 work，社区多用 WA=0（方案 1），部分实现用 WB=0（方案 2）。**核心原则：不能两个都为零。**

---

## 五、SVD 分解与低秩性

### 参数矩阵低秩性验证

```python
def svd_reconstruction(W, r):
    """用前 r 个奇异值重建矩阵"""
    U, s, Vt = torch.svd(W)
    W_r = U[:, :r] @ torch.diag(s[:r]) @ Vt[:, :r].T
    return W_r, s

# 对 512×512 矩阵：rank=4 就能以极小误差重建
# 前 4 个奇异值显著大于其余奇异值 → 矩阵本身低秩
```

### 梯度矩阵低秩性验证

```python
s_steps = [0, 10, 100, 1000]
for i in range(1001):
    Y = model(X)
    loss = loss_fn(Y, label)
    loss.backward()
    if i in s_steps:
        _, s = svd_reconstruction(model.Wv.weight.grad, 2)
        s_list.append(s)
    optimizer.step()
```

**实验结论**：随训练步数增加（0→10→100→1000），梯度的奇异值分布越来越集中——前几个奇异值越来越大，其余接近 0。**越充分训练 → 梯度越低秩 → LoRA 越有效。**

---

## 六、LoRA、MLA、Muon 的低秩性对比（★ 高频考点）

| 方法 | 低秩应用位置 | 目标 | 机制 |
|------|-------------|------|------|
| **LoRA** | 梯度增量 ΔW | 减少微调参数量 | $\Delta W = W_A W_B$，训练旁路 |
| **MLA** | KV Cache 压缩 | 减少推理 KV Cache | $c_{kv} = W_{DKV} x$，存低维表示 |
| **Muon** | 梯度更新方向 | 更好的优化几何 | $\text{msign}(M) = UV^T$，丢弃奇异值 |

三者的统一视角：**大模型的信息传递本质上是低维的**，在不同目标上的应用：
- LoRA：微调时梯度在低维子空间学习 → 少量参数就够
- MLA：token 在 context 中的表示是低维的 → 压缩 KV
- Muon：梯度矩阵的有效方向是低维的 → 用正交化代替 element-wise 优化

---

## 七、显存分析

### 全参微调 vs LoRA（7B 模型，r=8）

| 组件 | 全参微调 | LoRA |
|------|---------|------|
| 参数 | 7B × 4B = 28GB | 28GB（原模型）+ ~100MB（LoRA） |
| 梯度 | 28GB | ~100MB（只有 LoRA 的梯度）|
| 优化器状态 (Adam) | 56GB | ~200MB |
| 激活值 | 取决于 bs | 略多（多一条旁路） |
| **总计** | **>100GB** | **~30GB**（可在单卡 A100 运行）|

**第 4 项（激活值）常被忽略**：LoRA 旁路增加了中间激活，在 batch size 大时不可忽略。解法：对旁路分支使用 gradient checkpoint（重计算代替存储）。

### QLoRA 进一步压缩

QLoRA = LoRA + 4-bit 量化原模型：
- 原模型 4-bit 量化：28GB → 7GB
- 旁路 LoRA 仍用 bf16
- **总计 ~8GB，单卡 RTX 4090 可运行 7B 模型**

---

## 八、面试考点

**Q1：LoRA 为什么不直接训练一个小模型？**
LoRA 利用了预训练大模型的表示能力。小模型从头训练需要大量数据，LoRA 在原模型基础上用少量任务数据做"增量"更新，保留原有知识（减少遗忘），且旁路参数远小于从头训练一个小模型。

**Q2：初始化为什么不能 WA=WB=0？**
全零初始化：WA=WB=0 → YA=0 → dWB = YA.T @ dY = 0；dYA = dY @ WB.T = 0 → dWA = 0。梯度链完全断裂，两个参数都无法更新。

**Q3：LoRA 的 rank r 怎么选？**
实验结论：r=1 在很多任务上就能 work，r=8 是常见折中点。增大 r 到一定程度后性能收敛，继续增大无收益。选择依据：任务复杂度（简单分类 r=4，复杂推理 r=16+）。

**Q4：哪些层适合加 LoRA？**
原论文：Wq 和 Wv 效果最好（注意力中的查询和值投影）。实践中通常加在所有线性层（Wq/Wk/Wv/Wo + FFN up/down）。Embedding 层和 LM Head 也可以加，但收益较小。

**Q5：LoRA Merge 的原理和时机？**
推理时：$W_\text{merged} = W + \alpha \cdot W_A W_B$。合并后前向计算与原模型相同（无额外 latency）。训练时不能合并（需要分别更新）；部署时合并，省去旁路计算。

**Q6：为什么工业界大模型训练普遍用全参微调而非 LoRA？**
1. LoRA 是受限子空间，全参参数搜索空间更大
2. 好的数据（精心标注的 SFT/RLHF 数据）值得充分利用，全参可以最大化利用
3. LoRA 在多轮迭代中遗忘问题更复杂
4. 大公司有足够算力，全参 → 更好的 benchmark 数字
5. LoRA 适合资源受限场景（用户侧微调、快速 prototype）

**Q7：为什么 LoRA 需要比全参更多 epochs？**
LoRA 的参数更新受限于低秩子空间，每步有效方向更少，收敛更慢，需要更多迭代来达到同等效果。

---

## 九、LoRA 在 LLM 体系中的位置

```
预训练（全参，海量数据）→ 基础模型
        ↓
SFT 微调（全参 or LoRA）→ 指令跟随
        ↓
RLHF / DPO（全参 or LoRA）→ 对齐
        ↓
域专属微调（LoRA 最常见场景）→ 具体应用

QLoRA = 量化 + LoRA，面向边缘/低资源场景
```

---

## See Also

- [[LoRA]] — LoRA 理论精读（本笔记的代码实现对应的理论背景，含 rank 选择实验）
- [[AI/LLM/SFT/SFT-手撕实操]] — SFT 手撕实操（LoRA 在 SFT 流程中的完整应用）
- [[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]] — MLA 的低秩 KV 压缩（同一低秩分解思路在 Attention 上的应用）
- [[AI/LLM/MA-RLHF课程/lc8-GPTLoss-Muon优化器-手撕实操]] — Muon 梯度正交化（低秩梯度的另一种工程利用）
- [[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — ZeRO 显存优化（LoRA + ZeRO 组合是常见的显存节省方案）
