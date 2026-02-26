---
title: "Bradley-Terry 模型实现"
brief: "Bradley-Terry 偏好模型完整实现：从成对比较数据学习 item strength，理解 DPO 隐式 Reward 的数学基础（log σ(r_w - r_l) = BT 似然）。DPO 手撕的理论入口。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, rl, dpo, reward-modeling, preference-learning]
related:
  - "[[AI/LLM/RL/DPO/DPO-手撕实操]]"
  - "[[AI/LLM/RL/DPO/DPO-完整Notebook实现]]"
  - "[[AI/LLM/RL/PPO/LLaMA2-Reward-Model实现]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# Bradley-Terry 模型实现

> 来源：`ma-rlhf/notebook/DPO/BT_model.ipynb`（11 cells）
> 定位：DPO 的数学前置——理解偏好建模的概率框架

## 1. BT 模型定义

Bradley-Terry（BT）模型是一种**成对比较的概率模型**，通过两两比较来推断个体的潜在实力（分数）。

$$P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma(s_i - s_j)$$

其中 $s_i$ 是个体 $i$ 的潜在实力分数，$\sigma$ 是 sigmoid 函数。

**映射到 RLHF 场景**：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

- $x$：prompt
- $y_w$：人类偏好的 response（chosen）
- $y_l$：被拒绝的 response（rejected）
- $r(x, y)$：Reward Model 的打分

**核心假设**：偏好概率只取决于两个 response 的 **reward 差值**，而不取决于绝对值。

---

## 2. 为什么用 BT 模型建模偏好

### 问题背景

在 RLHF 中，我们有人类标注的偏好数据 $(x, y_w, y_l)$，但人类无法给出精确分数——只能说"A 比 B 好"。如何从这种**成对比较**中学出一个 reward function？

### BT 模型的优势

1. **概率框架**：把"A比B好"转化为概率 $P(A \succ B)$，可以用最大似然估计
2. **只需要序关系**：不需要绝对分数，只需要偏好对
3. **可导**：sigmoid 函数可微，可以端到端训练
4. **理论优雅**：满足 IIA（Independence of Irrelevant Alternatives）——加入第三个选项不影响 A、B 之间的比较

### 竞技赛类比

BT 模型最初用于体育竞赛排名——无法直接测量选手的"绝对实力"，但可以通过比赛结果（成对比较）反推每个选手的潜在分数。RLHF 中的 response 偏好对就是"比赛结果"。

---

## 3. BT Loss 推导 + 代码实现

### 3.1 两种等价形式

**指数形式（Maximum Likelihood Estimation）**：

$$P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)}$$

Loss:
$$\mathcal{L} = -\log P(i \succ j) = -\log\frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)}$$

**Sigmoid 形式（Logistic Regression）**：

$$P(i \succ j) = \sigma(s_i - s_j)$$

Loss:
$$\mathcal{L} = -\log\sigma(s_i - s_j)$$

两者数学等价（因为 $\frac{e^a}{e^a + e^b} = \sigma(a - b)$）。

### 3.2 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BTModel(nn.Module):
    def __init__(self, N):
        super(BTModel, self).__init__()
        self.reward = nn.Parameter(torch.ones(N))  # 每个个体的潜在分数
        self.BCE_loss = nn.BCELoss()

    def forward_exp(self, chosen_id, rejected_id):
        """指数形式：P = exp(s_i) / (exp(s_i) + exp(s_j))"""
        reward_chosen = torch.exp(self.reward[chosen_id])
        reward_rejected = torch.exp(self.reward[rejected_id])
        return reward_chosen / (reward_chosen + reward_rejected)

    def forward_sigmoid(self, chosen_id, rejected_id):
        """Sigmoid 形式：P = σ(s_i - s_j)"""
        reward_chosen = self.reward[chosen_id]
        reward_rejected = self.reward[rejected_id]
        return torch.sigmoid(reward_chosen - reward_rejected)

    def loss_exp(self, pred, label):
        """MLE Loss"""
        return -torch.log(pred) if label == 1 else -torch.log(1 - pred)

    def loss_sigmoid(self, pred, label):
        """BCE Loss"""
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        loss = -(label * torch.log(pred) + (1 - label) * torch.log(1 - pred))
        return loss
```

### 3.3 训练示例

```python
N = 4  # 4 个选手
model = BTModel(N)
datas = [(0, 1, 1), (2, 3, 1), (1, 3, 1)]
# 含义：选手0>选手1，选手2>选手3，选手1>选手3

optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(100):
    total_loss = 0
    for data in datas:
        id_i, id_j, label = data
        optimizer.zero_grad()
        pred = model.forward_sigmoid(id_i, id_j)
        loss = model.loss_sigmoid(pred, torch.tensor(label, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
```

训练结果：胜率高的选手获得更高分数，胜率低的选手获得更低分数。**epoch 越多，分数差距越大**（过拟合时趋于 $\pm\infty$）。

---

## 4. BT → RL → DPO 推导链

BT 模型是 DPO 的数学前置。整条推导链：

### Step 1：BT 模型定义偏好

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

用 BT Loss 训练 Reward Model $r_\phi(x, y)$。

### Step 2：RL 优化（PPO/GRPO）

有了 $r_\phi$ 后，用 RL 优化策略：

$$\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta}[r_\phi(x, y)] - \beta \cdot D_{KL}[\pi_\theta \| \pi_{ref}]$$

这个 KL-constrained optimization 有闭式解：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \cdot \exp\left(\frac{1}{\beta} r(x, y)\right)$$

### Step 3：从闭式解反推 reward

$$r(x, y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

### Step 4：DPO——代入 BT 模型

将 reward 代回 BT 模型：

$$P(y_w \succ y_l) = \sigma\left(\beta \log\frac{\pi^*(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi^*(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

$Z(x)$ 项在差值中消掉！最终 DPO Loss：

$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

**关键洞察**：DPO 把 BT + RM Training + RL 三步合并为一步，直接用偏好数据优化策略，但数学基础仍然是 BT 模型。

---

## 5. BT 模型的关键性质

### 过拟合行为

Notebook 中用 100K epoch 训练，观察到**分数无限增长**：

```python
# Epoch 0:    reward = [1., 1., 1., 1.]
# Epoch 10K:  reward = [28., -27., 28., -53.]
# Epoch 90K:  reward = [257., -256., 257., -510.]
```

因为 $\sigma(s_i - s_j)$ 在 $|s_i - s_j| \to \infty$ 时趋于 0 或 1，loss 趋于 0。没有正则化时，BT 模型会让分数差距无限扩大。

**RLHF 启示**：这就是为什么 Reward Model 训练需要正则化（weight decay、early stopping），否则 reward 的绝对值会 blow up。

### 分数的相对性

BT 模型只关心 **分数差值** $s_i - s_j$，绝对值无意义。给所有分数加一个常数 $c$，预测概率不变。这和 DPO 中的 $Z(x)$ 消除是同一个道理。

---

## 6. 面试考点

### Q0：为什么 RLHF 要用 BT 模型而不是直接打分？

**答**：人类偏好标注成对比较比绝对打分更可靠（人类擅长"哪个好"，不擅长"好多少分"）。BT 模型把这些相对比较转为绝对 scalar reward，供 PPO 使用。

### Q1：DPO 为什么不需要单独训练 Reward Model？数学上是怎么绕过去的？

**答**：DPO 的推导链是 BT → KL-constrained RL → 闭式解 → 代回 BT。关键步骤是 KL-constrained RL 有闭式最优策略 $\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$，从中可以反解出 $r(x,y) = \beta \log(\pi^*(y|x)/\pi_{ref}(y|x)) + \beta\log Z(x)$。代入 BT 模型后，$Z(x)$ 在 chosen 和 rejected 的差值中消掉，最终 loss 只依赖策略的 log-prob ratio，不再需要显式的 reward。BT 模型是这条推导链的**起点**——它定义了偏好和 reward 的关系。

### Q2：BT 模型的核心假设是什么？什么场景下会失效？

**答**：BT 模型的核心假设是 **IIA**（Independence of Irrelevant Alternatives）——A 和 B 的偏好关系不受第三者 C 的影响。等价地，偏好概率只取决于两者的 reward 差值，不取决于 context。**失效场景**：(1) 偏好不可传递：人类可能 A>B, B>C, 但 C>A；(2) 偏好依赖上下文：同一个回答在不同文化背景下评价不同；(3) 多维偏好：安全性、有用性、流畅度可能无法用单一标量 reward 表达。这些局限催生了 Contextual BT、多目标 RLHF 等改进。
