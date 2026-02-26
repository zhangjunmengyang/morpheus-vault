---
brief: "PEFT 参数高效微调方法（面试版）——LoRA/QLoRA/Prefix Tuning/Adapter 的原理、显存占用、训练速度对比；何时选 LoRA vs 全量微调的决策框架；面试 AI 工程师必问的微调方法论。"
title: "PEFT 参数高效微调方法"
date: 2026-02-14
tags: [training, peft, lora, qlora, interview]
type: note
---

> [!info] 📖 版本说明
> 本篇为**面试速查版**（简洁直接）。深度工程版：[[AI/LLM/SFT/PEFT 方法对比|PEFT 方法综述工程深度版]]

# PEFT 参数高效微调方法

## 1. 为什么需要 PEFT

全量微调（Full Fine-Tuning）的核心问题：

| 模型规模 | 参数量 | FP16 显存 | Adam 优化器额外开销 | 总训练显存估算 |
|---------|-------|----------|-------------------|-------------|
| 7B | 70 亿 | ~14 GB | ~28 GB（梯度+动量） | ~56 GB+ |
| 13B | 130 亿 | ~26 GB | ~52 GB | ~104 GB+ |
| 70B | 700 亿 | ~140 GB | ~280 GB | ~560 GB+ |

全量微调的痛点：
- **显存爆炸**：每个参数需存储梯度 + 两份优化器状态（Adam），总显存 ≈ 4× 模型本体
- **多任务存储**：每个下游任务都保存一份完整模型副本，70B 模型一个 checkpoint 就 140GB
- **灾难性遗忘**：全量更新容易破坏预训练学到的通用知识
- **训练不稳定**：大模型全量微调对学习率、batch size 非常敏感

**PEFT 核心思想**：冻结预训练权重的绝大部分，只训练少量新增参数（通常 < 1% 原始参数），实现接近全量微调的效果。

---

## 2. LoRA（Low-Rank Adaptation）

### 2.1 核心原理

**关键假设**：微调过程中权重的变化量 ΔW 是低秩的，即可以用两个小矩阵的乘积近似。

对于预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将更新分解为：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$
- $A$ 用高斯随机初始化，$B$ 初始化为零 → 训练开始时 $\Delta W = 0$
- 前向传播：$h = W_0 x + \frac{\alpha}{r} BAx$

### 2.2 关键超参数

**秩 r**：
- 控制 LoRA 的表达能力，通常取 4、8、16、32、64
- r 越大参数越多，效果越好但效率下降
- 经验值：对话/指令微调 r=8~16 通常足够；复杂领域适配可能需要 r=32~64

**缩放因子 alpha（α）**：
- 实际缩放比为 α/r，控制 LoRA 更新相对于原始权重的幅度
- 常见设置：α = 2r（即缩放比为 2）或 α = r（缩放比为 1）
- α 越大，LoRA 的更新对输出影响越大

**target_modules（加在哪些层）**：
- **必加**：Q、V 投影矩阵（原始论文验证效果最好）
- **推荐**：Q、K、V、O 全加（实践中效果更好）
- **激进**：加上 MLP 的 gate_proj / up_proj / down_proj（参数多但效果更好）
- HuggingFace PEFT 中可设 `target_modules="all-linear"` 全部线性层都加

### 2.3 LoRA 的优势

- **训练高效**：只训练 ~0.1%-1% 的参数，显存降低 60%+
- **推理零开销**：训练完后可将 BA 合并回 $W_0$，推理时没有额外延迟
- **即插即用**：可以快速切换不同任务的 LoRA adapter
- **与量化兼容**：配合 QLoRA 可在消费级 GPU 上微调大模型

---

## 3. QLoRA

### 3.1 核心创新

QLoRA = **4-bit NF4 量化的基座模型** + **LoRA adapter（BF16/FP16）**

三大技术贡献：

1. **NF4（NormalFloat 4-bit）量化**：
   - 假设预训练权重近似正态分布
   - 将正态分布的分位点作为 4-bit 的 16 个量化级别
   - 比普通 INT4 量化信息损失更小

2. **Double Quantization（双重量化）**：
   - 量化常数（scale factor）本身也被量化
   - 每个量化块（64 个参数）的 FP32 scale → 再用 FP8 量化
   - 额外节省约 0.37 bit/参数

3. **Paged Optimizers**：
   - 利用 NVIDIA 统一内存，将优化器状态在 GPU↔CPU 间分页
   - 防止长序列训练时 OOM

### 3.2 训练流程

```
基座模型权重 (NF4, 冻结)
    ↓ 反量化为 BF16
    → 前向计算
    → + LoRA adapter (BF16) 的输出
    → 反向传播只更新 LoRA 参数
```

### 3.3 效果

- 65B 模型可在单张 48GB GPU（A6000）上微调
- 在 MMLU 等 benchmark 上与 16-bit 全量微调几乎无差（< 0.5% 差距）
- 7B 模型可在 24GB 消费级 GPU（RTX 3090/4090）上训练

---

## 4. DoRA（Weight-Decomposed Low-Rank Adaptation）

### 4.1 动机

分析发现 LoRA 和全量微调的学习模式不同：
- 全量微调：**方向**变化大，**幅度**变化相对小
- LoRA：方向和幅度耦合在一起，难以独立调节

### 4.2 原理

将权重矩阵分解为**幅度（magnitude）**和**方向（direction）**：

$$W = m \cdot \frac{V}{||V||_c}$$

- $m \in \mathbb{R}^{1 \times k}$：每列的幅度向量（可训练标量）
- $V / ||V||_c$：归一化后的方向矩阵

然后只对方向矩阵 $V$ 施加 LoRA 更新：

$$W' = m \cdot \frac{V + \Delta V}{||V + \Delta V||_c} \quad \text{其中} \quad \Delta V = BA$$

### 4.3 优势

- 方向和幅度解耦，更接近全量微调的学习行为
- 在相同 r 下通常优于 LoRA 0.5-1%
- 训练开销仅比 LoRA 多一个幅度向量（几乎可忽略）

---

## 5. AdaLoRA（Adaptive Low-Rank Adaptation）

### 5.1 核心思想

**不同层/模块的重要性不同**，应该分配不同的秩：
- 重要的权重矩阵 → 分配更高的秩 r
- 不重要的权重矩阵 → 分配更低的秩，甚至裁剪掉

### 5.2 实现方式

将 ΔW 参数化为 SVD 形式：$\Delta W = P \Lambda Q$

- $P$, $Q$ 为左右奇异向量矩阵
- $\Lambda$ 为对角奇异值矩阵

训练过程中通过**重要性评分**（基于奇异值大小和梯度信息）动态裁剪不重要的奇异值分量，实现全局秩的自适应分配。

### 5.3 效果

- 在相同参数预算下优于固定秩的 LoRA
- 自动发现：注意力层通常需要更高的秩，MLP 层相对需求较低
- 缺点：训练过程更复杂，SVD 参数化引入额外开销

---

## 6. LoRA 变体改进

### 6.1 LoRA+

**问题**：标准 LoRA 中 A 和 B 使用相同的学习率，但理论分析表明这不是最优的。

**改进**：
- 矩阵 B 的学习率应该远大于矩阵 A 的学习率
- 推荐比例：$\eta_B / \eta_A \approx 16$
- 原因：A 负责特征提取（需要稳定），B 负责投影到输出空间（需要快速适应）

**效果**：训练速度提升 ~2×，最终效果提升 1-2%。

### 6.2 rsLoRA（Rank-Stabilized LoRA）

**问题**：标准 LoRA 的缩放因子 α/r 在 r 变大时效果不稳定。

**改进**：将缩放因子从 $\alpha / r$ 改为 $\alpha / \sqrt{r}$

**效果**：
- 在大秩（r=64, 128）时效果显著更好
- 允许使用更大的 r 而不会导致训练不稳定

---

## 7. Soft Prompt 方法对比

### 7.1 Prefix Tuning

- 在每一层的 Key 和 Value 前面拼接可训练的 prefix 向量
- prefix 长度通常 10-100 个 token
- 每层都有独立的 prefix 参数
- 总参数量 = prefix_length × num_layers × 2 × hidden_dim

### 7.2 Prompt Tuning

- **只在输入 embedding 层**前面添加可训练的 soft token
- 比 Prefix Tuning 更简单，参数更少
- 在模型足够大时（>10B）效果接近全量微调
- Google 提出，验证了"规模足够大时，简单的方法就足够好"

### 7.3 P-Tuning v2

- 在**每一层**添加可训练的 prefix（类似 Prefix Tuning）
- 增加了重参数化技巧（MLP 编码器生成 prefix）
- 专门优化了 NLU 任务（分类、NER、QA）的性能

### 7.4 对比

| 方法 | 可训练位置 | 参数量 | 适用场景 | 推理开销 |
|------|----------|-------|---------|---------|
| Prompt Tuning | 输入 embedding 层 | 极少 | 超大模型、多任务 | 极小 |
| Prefix Tuning | 每层的 K, V | 较少 | 生成任务 | 有（额外 KV） |
| P-Tuning v2 | 每层 prefix | 中等 | NLU 任务 | 有 |

**共同缺点**：
- 占用序列长度，减少有效上下文窗口
- 效果通常不如 LoRA 系列
- 推理时无法合并，有持续开销

---

## 8. Adapter 方法

### 8.1 Houlsby Adapter（2019）

在 Transformer 的每个子层（Self-Attention 和 FFN）后面插入一个 bottleneck 结构：

```
输入 → LayerNorm → Down-projection (d→r) → 非线性激活 → Up-projection (r→d) → 残差连接
```

- 每层插入 2 个 adapter（attention 后 + FFN 后）
- 参数量 = num_layers × 2 × (d×r + r×d) ≈ 4 × L × d × r
- 缺点：**增加推理延迟**（串行在主干上，无法合并）

### 8.2 LLaMA-Adapter

- 只在**高层**（如最后 L 层）的注意力中插入可训练的 prefix
- 引入**零初始化注意力（zero-init attention）**：用一个初始为零的门控因子控制 adapter 输出
- 防止训练初期随机的 adapter 输出破坏预训练模型
- 参数量极小（1.2M for LLaMA-7B），适合指令微调

---

## 9. 各方法性能-效率 Trade-off 对比

| 方法 | 可训练参数占比 | 训练显存节省 | 推理开销 | 效果（vs 全量微调） | 实现复杂度 |
|------|-------------|------------|---------|-------------------|----------|
| LoRA | 0.1%~1% | ~60% | **零**（可合并） | ≈95-99% | ★☆☆ 简单 |
| QLoRA | 0.1%~1% | ~75% | 零（合并后） | ≈94-98% | ★★☆ 中等 |
| DoRA | 0.1%~1% | ~60% | 零（可合并） | ≈96-99.5% | ★★☆ 中等 |
| AdaLoRA | 自适应 | ~60% | 零（可合并） | ≈96-99% | ★★★ 复杂 |
| Prefix Tuning | 0.01%~0.1% | ~70% | 有（额外 KV） | ≈90-95% | ★☆☆ 简单 |
| Prompt Tuning | <0.01% | ~75% | 极小 | ≈88-95%（依赖规模） | ★☆☆ 简单 |
| Adapter | 0.5%~3% | ~50% | 有（串行延迟） | ≈95-98% | ★★☆ 中等 |
| 全量微调 | 100% | 0% | 零 | 100%（基准） | ★☆☆ 简单 |

---

## 10. 实践建议：什么场景用什么方法

### 场景 → 方法映射

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 消费级 GPU（24GB）微调 7B-13B | **QLoRA** | 显存友好，效果好 |
| 生产部署、推理延迟敏感 | **LoRA / DoRA** | 可合并，零推理开销 |
| 多任务快速切换 | **LoRA** | 每个任务只存几 MB adapter |
| 资源充足，追求最佳效果 | **DoRA + 大 r** | 接近全量微调效果 |
| 超大模型（100B+）简单适配 | **Prompt Tuning** | 参数极少，适合超大规模 |
| NLU 任务（分类、NER） | **P-Tuning v2 / LoRA** | 对 NLU 有专门优化 |
| 研究/实验快速迭代 | **LoRA (r=8, all-linear)** | 简单高效，效果稳定 |

### 通用实践 Tips

1. **从 LoRA 开始**：r=16, alpha=32, target_modules="all-linear" 是一个很强的 baseline
2. **学习率**：LoRA 的学习率通常比全量微调高 5-10×（如 2e-4 vs 2e-5）
3. **训练轮数**：PEFT 通常需要更多 epoch（3-5 epochs vs 全量微调 1-3 epochs）
4. **数据质量 > 方法选择**：高质量数据 + 简单 LoRA > 低质量数据 + 花式方法
5. **合并后评估**：始终在 LoRA 合并后的模型上做最终评估

---

## 11. 面试常见问题及回答要点

### Q1: LoRA 的原理是什么？为什么有效？

**回答要点**：
- 预训练模型的权重更新 ΔW 具有低秩特性（Aghajanyan et al. 的 intrinsic dimensionality 研究证实）
- 将 ΔW 分解为两个低秩矩阵 B×A 的乘积，大幅减少可训练参数
- A 高斯初始化、B 零初始化，保证训练起点等价于预训练模型
- 缩放因子 α/r 控制更新幅度
- 推理时可合并回原始权重，不增加延迟

### Q2: LoRA 的 r 和 alpha 怎么选？有什么影响？

**回答要点**：
- r 控制表达能力：r 越大能拟合越复杂的更新，但参数也越多
- alpha/r 是实际的缩放比例，控制 LoRA 更新对输出的影响大小
- 常见设置：r=8~64, alpha=2r
- 固定 alpha=16 调 r，或固定 alpha/r=2 等比缩放，都是常见策略
- r 不是越大越好，过大可能过拟合且效率下降

### Q3: QLoRA 和 LoRA 的核心区别是什么？

**回答要点**：
- 核心区别在**基座模型的精度**：QLoRA 将基座量化到 4-bit NF4 格式
- LoRA adapter 本身仍然是 BF16/FP16
- NF4 利用权重的正态分布特性设计量化级别，信息损失更小
- 双重量化进一步压缩量化常数
- 反向传播时将 NF4 权重反量化为 BF16 计算梯度
- 本质是用量化压缩基座模型的显存，用 LoRA 保证微调效果

### Q4: Prefix Tuning 和 LoRA 的本质区别是什么？

**回答要点**：
- **修改位置不同**：Prefix Tuning 修改注意力的 K/V 输入；LoRA 修改权重矩阵本身
- **推理开销不同**：Prefix Tuning 有持续开销（额外 KV 序列）；LoRA 可合并为零开销
- **影响方式不同**：Prefix Tuning 通过额外的"虚拟 token"影响注意力分布；LoRA 直接改变线性变换
- **上下文影响**：Prefix Tuning 占用序列长度（减少有效上下文）；LoRA 不影响
- **实际效果**：LoRA 通常在各种任务上更稳定，Prefix Tuning 在极端低参数场景有优势

### Q5: 为什么 LoRA 要把 B 初始化为零、A 用高斯初始化，而不是反过来？

**回答要点**：
- 目的是让初始的 ΔW = BA = 0，训练起点就是预训练模型本身
- 只需要 A 或 B 中的一个为零即可，论文选择 B=0
- A 用高斯初始化是为了打破对称性，让不同 rank 维度有不同的初始特征提取方向
- 如果 A 也是零，梯度更新会导致 A 的所有行保持相同（对称性问题）
- 这个设计类似于 residual learning 的思想：学习残差而非直接学习目标

### Q6: 如果线上服务需要同时服务多个 LoRA 适配的任务，怎么高效部署？

**回答要点**：
- **方案一：LoRA 动态加载**：基座模型共享，按请求加载不同的 LoRA 权重（S-LoRA / Punica 框架）
- **方案二：Batched LoRA**：将不同 LoRA 的 BA 矩阵组织成 batch，利用 CUDA kernel 并行计算
- **方案三：合并导出**：为每个任务合并一个独立模型，用路由层分发请求（简单但显存倍增）
- **关键技术**：S-LoRA 使用统一内存管理 + 自定义 CUDA kernel，可在单 GPU 上服务数千个 LoRA
- **vLLM 支持**：vLLM 已原生支持多 LoRA serving，动态加载切换

---

## See Also

- [[AI/LLM/RL/Fundamentals/强化学习的数学原理|强化学习数学原理]] — PEFT 与 RL 结合：LoRA + GRPO 的参数高效 post-training
- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth]] — PEFT 实践工具：LoRA/QLoRA 的高效训练框架
- [[AI/Foundations/Training/Scaling Laws|Scaling Laws]] — PEFT 的动机：为什么小参数微调能保留大模型能力
- [[AI/Foundations/目录|Foundations MOC]] — 训练基础全图谱
