---
title: Flow Matching 手撕实操 · MA-RLHF Batch D
type: code-practice
date: 2026-02-27
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - flow-matching
  - generative-model
  - ode
  - code-practice
  - ma-rlhf
  - diffusion
brief: Flow Matching 从零手撕：学习噪声分布到目标分布的向量场（Vector Field），用欧拉法数值求解 ODE 完成生成；进阶 Conditional Flow Matching 通过 tag 条件控制生成目标。理解 Flow Matching 与 Diffusion 的核心区别：直接学线性插值路径而非去噪过程，训练更简单、生成轨迹更直。
rating: ★★★☆
related:
  - "[[AI/3-LLM/Architecture/Mamba-SSM]]"
  - "[[Projects/MA-RLHF/lc5/lc5-03-Flow-Attention-架构手撕实操]]"
  - "[[Projects/MA-RLHF/课程索引]]"
---

# Flow Matching 手撕实操（MA-RLHF Batch D）

> **来源**：`notebook/flow-matching/flow_matching.ipynb`（10 cells，191KB）  
> **评级**：★★★☆  
> **位置**：MA-RLHF 课程 notebook/ 目录，Batch D 扩展内容

---

## 为什么学 Flow Matching

在 LLM 生态中，Flow Matching 的意义正在超越图像生成：

- **多模态生成**（图像/视频/语音 token）：LaViDa-R1、Stable Diffusion 3 等用 Flow Matching 替代 DDPM
- **RL for LLM 扩展**：FlowRL（GFlowNet trajectory balance）把生成过程看作流，理解 Flow Matching 是理解这类工作的前提
- **连续扩散 LLM**：MDLM / Plaid 等把 LLM token 生成建模为连续流

MA-RLHF 课程把它放在 notebook/ 扩展内容，是理论框架补充。

---

## 一、核心思想：学向量场，不学去噪

### Diffusion vs. Flow Matching

| 维度 | Diffusion（DDPM）| Flow Matching |
|------|-----------------|---------------|
| 训练目标 | 预测噪声 ε（去噪） | 预测向量场 v(x,t)（流速） |
| 路径 | 随机游走（Markov chain）| 确定性 ODE 轨迹 |
| 插值 | 复杂 noise schedule | **线性插值** $x_t = (1-t)x_0 + t x_1$ |
| 生成 | DDPM 反向采样（多步，随机）| ODE 求解（欧拉法，确定性）|
| 训练复杂度 | 高（需要 ELBO / diffusion schedule）| **低**（MSE loss 直接） |

**Flow Matching 的关键 insight**：与其学复杂的噪声去除过程，不如直接学"把噪声点搬运到目标点的速度场"。训练样本是 $(x_0, x_1)$ 对（噪声点和目标点），在 $t \in [0,1]$ 的线性插值路径上的速度就是 $v = x_1 - x_0$（常量）。

### 线性插值路径

```python
# 训练时的插值：从噪声 x0 到目标 x1 的直线路径
t = torch.rand(batch_size, 1)         # 随机时间步 t ∈ [0,1]
xt = (1 - t) * x0 + t * x1           # 线性插值：t=0 是噪声，t=1 是目标
v_target = x1 - x0                   # 目标速度：常量（直线路径的切向量）

# 模型预测向量场
v_pred = model(xt, t)

# 损失：直接 MSE
loss = F.mse_loss(v_pred, v_target)
```

这比 DDPM 的 noise schedule + ELBO 优化简单得多。

---

## 二、向量场网络（VectorField）

```python
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),   # 输入：x (dim) + t (1)
            nn.ReLU(),
            nn.Linear(64, dim)         # 输出：速度向量 v (dim)
        )

    def forward(self, x, t):
        # t 必须是 (batch_size, 1)，与 x 的 (batch_size, dim) 拼接
        return self.net(torch.cat([x, t], dim=1))
```

**设计要点**：
- 输入维度 = 数据维度 + 1（时间 t）——t 是连续的，不是离散步骤
- 输出维度 = 数据维度（速度向量，与数据同维）
- 模型在每个 $(x, t)$ 处输出一个速度 $v(x,t)$，指示如何移动

---

## 三、生成（ODE 求解）

训练完成后，生成过程是**数值求解 ODE**：

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) = x_0 \sim \mathcal{N}(0, I)$$

用欧拉法（最简单的 ODE 求解器）：

```python
x = noise_sample        # 从噪声分布出发
dt = 1.0 / num_steps    # 步长

with torch.no_grad():
    for i in range(num_steps):
        t_tensor = torch.tensor(i * dt, dtype=torch.float32)
        vt = model(x, t_tensor)      # 查询当前速度
        x = x + vt * dt              # 欧拉步：x(t+Δt) = x(t) + v(t)·Δt

# x 现在应该在目标分布附近
```

**生成步数**：课程代码用 50 步（`num_steps=50`）。步数越多，ODE 解越精确，但越慢。更高阶的 ODE solver（RK4、DPM-Solver）可以用更少步数达到同等精度。

---

## 四、Conditional Flow Matching（条件控制）

标准 Flow Matching 是无条件生成（噪声→目标，不控制生成哪类样本）。Conditional FM 加入条件 tag：

```python
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, 64),  # 输入：x (2) + t (1) + tag (1) = 4
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t, tag):
        return self.net(torch.cat([x, t, tag], dim=1))
```

**课程 Demo 设计**：目标分布是正弦曲线 $y = \sin(x)$，$x \in [0, 4\pi]$，按 x 值把目标点分成 10 段（tag = 0~9）。训练 Conditional FM 后，给定 tag 就能生成落在特定 x 段的点。

训练时 tag 是真实标签；生成时 tag 是我们想要生成的类别。

---

## 五、与 RLHF 框架的关系

Flow Matching 在当前 LLM RL 生态中的连接点：

**FlowRL / GFlowNet 连接**：
- GFlowNet 把生成过程建模为 flow（从初始状态到终止状态的流量平衡）
- LAD（2602.20132）把 RL for LLM 建模为 f-divergence 匹配（最一般的 distribution matching 框架）
- Flow Matching 是这类"分布匹配"思想在连续空间的实现

**连续扩散 LLM**：
- MDLM / Plaid 等用连续 FM 替代离散 token 生成
- 理解 Flow Matching 是理解这类架构的前提

**Diffusion Policy for Agent**：
- 机器人控制中已有 Diffusion Policy（用 DDPM 输出动作分布）
- Flow Matching 的确定性 ODE 轨迹比 Diffusion 更适合实时推理

---

## 六、关键数字与实验

课程 Demo 的超参：

| 参数 | Unconditional FM | Conditional FM |
|------|-----------------|----------------|
| 数据维度 | 2D | 2D |
| 目标分布 | sin(x) 曲线 | sin(x) + 10 类 tag |
| 样本量 | 200 | 1000 |
| 训练 epochs | 100,000 | 5,000 |
| ODE 步数 | 50 | 50 |
| 学习率 | 1e-2 | 1e-3 |
| 网络 | 2层 MLP (64 hidden) | 2层 MLP (64 hidden) |

Conditional FM 用更多样本（1000 vs 200）但更少 epoch（5000 vs 100000）——条件信息大幅降低了学习难度。

---

## 七、面试问法

**Q: Flow Matching 和 Diffusion 的核心区别？**
> Flow Matching 直接学线性插值路径的速度场（v = x₁ - x₀，常量），用 MSE loss 训练；生成是确定性 ODE。Diffusion（DDPM）学噪声预测（ε），路径是随机 Markov chain，生成是随机采样。Flow Matching 训练更简单（不需要复杂 noise schedule），生成轨迹更直（线性插值），步数更少，适合实时推理。

**Q: Conditional Flow Matching 怎么加条件？**
> 把条件 tag 直接 concat 到输入（x + t + tag），模型在有条件约束的子空间内学向量场。训练时每个 (x₀, x₁) 对带有真实 tag；生成时指定想要的 tag，ODE 求解就会把噪声点引导到该条件的目标分布区域。

**Q: 为什么 LLM 社区开始关注 Flow Matching？**
> 三个方向：① 多模态生成（图像/视频 token 生成比 DDPM 快）；② 连续扩散 LLM（MDLM 等把 token 嵌入视为连续流）；③ FlowRL / GFlowNet（把 RL 生成轨迹建模为流，Flow Matching 是其连续空间基础）。

---

## See Also

- [[AI/3-LLM/Architecture/LaViDa-R1-Diffusion-LLM-Reasoning]] — Flow Matching 在 Diffusion LLM 中的应用
- [[Projects/MA-RLHF/课程索引]] — MA-RLHF 课程全局导航

