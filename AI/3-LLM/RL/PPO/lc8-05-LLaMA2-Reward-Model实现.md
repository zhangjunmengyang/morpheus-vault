---
title: LLaMA2 Reward Model 实现
brief: 基于 LLaMA2 的 Reward Model 完整实现：Bradley-Terry 偏好建模、序列级 reward head（接 last token）、偏好对数据构建和 RM training loop。RLHF 四模型体系中 Reward Model 工程实现参考。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl
  - reward-model
  - rlhf
  - llama2
related:
  - "[[AI/3-LLM/RL/DPO/Bradley-Terry模型实现]]"
  - "[[AI/3-LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]]"
  - "[[PPO 原理]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# LLaMA2 Reward Model 实现

> 来源：`ma-rlhf/notebook/reward/LLaMA2-reward.ipynb`（25 cells）
> 参考论文：Llama 2: Open Foundation and Fine-Tuned Chat Models (§3.2.2)
> 整理日期：2026-02-25

---

## 1. Reward Model 结构

### 核心思想

Reward Model = **SFT 模型（LLM backbone）** + **Value Head（线性回归头）**

将预训练语言模型的 next-token prediction head（`lm_head`，输出 vocab_size 维）替换为一个回归头（输出 1 维标量 reward）。

> "The model architecture and hyper-parameters are identical to those of the pretrained language models, except that the classification head for next-token prediction is replaced with a regression head for outputting a scalar reward." — LLaMA2 论文 §3.2.2

### 结构图

```
┌────────────────────────────────────────────┐
│             Reward Model                    │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │     LLaMA Transformer Backbone       │  │
│  │  (与 SFT 模型完全相同的架构和参数)      │  │
│  │  hidden_size = 4096 (以 LLaMA2-7B 为例)│ │
│  └──────────────┬───────────────────────┘  │
│                 │                           │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │   Value Head: nn.Linear(4096, 1)     │  │
│  │   输出: 标量 reward score              │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  输入: [prompt, response] 的 token ids     │
│  输出: 一个标量 r ∈ ℝ，表示 response 质量   │
└────────────────────────────────────────────┘
```

### 代码实现

```python
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification

torch.manual_seed(1)

# 配置（玩具模型参数，实际用 LLaMA2-7B/13B）
config = LlamaConfig(
    vocab_size=100,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)

# Step 1: 先训练/加载一个 SFT 模型
model = LlamaForCausalLM(config)
model.save_pretrained('./lm_pretrained')

# Step 2: 加载 SFT 权重，将 lm_head 替换为 regression head
# num_labels=1 → 输出标量
rm_model = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained', num_labels=1
)
```

`LlamaForSequenceClassification` 的关键改动：
- 去掉 `lm_head`（vocab_size 维的线性层）
- 加上 `score`（1 维的线性层）：`nn.Linear(hidden_size, 1, bias=False)`
- Forward 时取最后一个非 padding token 的 hidden state → score 层 → 输出标量

---

## 2. 训练目标：Bradley-Terry Loss

### 核心思想

给定同一个 prompt，人类标注 chosen（更好的 response）和 rejected（更差的 response），Reward Model 应该给 chosen 更高的分数。

### Bradley-Terry 偏好模型

人类偏好 chosen 的概率建模为：

$$P(y_c \succ y_r | x) = \sigma(r_\theta(x, y_c) - r_\theta(x, y_r))$$

### 训练 Loss

$$\mathcal{L} = -\log \sigma\left(r_\theta(x, y_c) - r_\theta(x, y_r)\right)$$

### LLaMA2 的 Margin Loss 变体

LLaMA2 引入了 **margin 项**，根据标注者的偏好强度调整 loss：

$$\mathcal{L}_{\text{margin}} = -\log \sigma\left(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r)\right)$$

其中 $m(r)$ 是离散函数，取决于偏好等级：

| 偏好标注 | margin |
|---------|--------|
| Significantly Better | 大（如 3.0） |
| Better | 中（如 1.5） |
| Slightly Better | 小（如 0.5） |
| Negligibly Better | 0 |

**直觉**：当标注者认为 chosen 明显更好时，margin 更大 → 要求 RM 的分差更大才能使 loss 变小。

### 代码实现

```python
X_chosen = torch.randint(0, 100, (1, 10))       # 好的 response 的 token ids
X_rejected = torch.randint(0, 100, (1, 10))      # 差的 response 的 token ids
margin = 3.0                                      # "Significantly Better" 级别的 margin

# 分别对 chosen 和 rejected 打分
rm_chosen = rm_model(input_ids=X_chosen).logits     # r_θ(x, y_c)：标量
rm_rejected = rm_model(input_ids=X_rejected).logits  # r_θ(x, y_r)：标量

# 标准 Bradley-Terry Loss
loss = -torch.sigmoid(rm_chosen - rm_rejected).log()

# 带 Margin 的 Loss（LLaMA2）
loss_with_margin = -torch.sigmoid(rm_chosen - rm_rejected - margin).log()

print(f'chosen reward:  {rm_chosen.item():.4f}')
print(f'rejected reward: {rm_rejected.item():.4f}')
print(f'BT loss:        {loss.item():.4f}')
print(f'BT+margin loss: {loss_with_margin.item():.4f}')
```

### 完整训练循环（伪代码）

```python
optimizer = torch.optim.AdamW(rm_model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        # batch 包含: prompt + chosen_response, prompt + rejected_response, margin
        chosen_ids, rejected_ids, margins = batch

        r_chosen = rm_model(input_ids=chosen_ids).logits      # [B, 1]
        r_rejected = rm_model(input_ids=rejected_ids).logits   # [B, 1]

        # Bradley-Terry Loss with Margin
        loss = -F.logsigmoid(r_chosen - r_rejected - margins).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 3. 推理时如何给 Response 打分

```python
rm_model.eval()                                    # 切换到推理模式

x = torch.randint(0, 100, (1, 10))                # [prompt, response] 的 token ids
with torch.no_grad():
    rm_score = rm_model(input_ids=x).logits        # 输出标量 reward score
print(f'reward score: {rm_score.item():.4f}')
```

### LLaMA2 的双 Reward 选择机制

LLaMA2 同时训练两个 RM：**Safety RM** 和 **Helpfulness RM**，推理时动态选择：

$$R_c(g|p) = \begin{cases} R_s(g|p) & \text{if is\_safety}(p) \text{ or } R_s(g|p) < 0.15 \\ R_h(g|p) & \text{otherwise} \end{cases}$$

**逻辑**：如果 Safety RM 认为回复不安全（分数 < 0.15），就用 Safety 分数（压低该回复），否则用 Helpfulness 分数。安全优先。

```python
def llama2_reward_select(reward_safety, reward_helpfulness):
    """安全优先的 reward 选择"""
    return reward_safety if reward_safety < 0.15 else reward_helpfulness

# 不安全的回复 → 使用 safety reward（负分）
rc = llama2_reward_select(reward_safety=-0.3, reward_helpfulness=0.7)
# rc = -0.3  ← 被安全 RM 否决

# 安全的回复 → 使用 helpfulness reward
rc = llama2_reward_select(reward_safety=1.3, reward_helpfulness=0.4)
# rc = 0.4   ← 正常使用 helpfulness 分数
```

### Reward 后处理：Logit + Whiten

LLaMA2 在使用 reward 之前做两步处理：

$$\hat{R} = \text{WHITEN}(\text{LOGIT}(R_c(g|p)))$$

**Step 1: 逆 Sigmoid（Logit 函数）**

将 [0,1] 范围的概率映射回实数域：

```python
def inverse_sigmoid(x):
    """logit 函数：sigmoid 的逆"""
    return torch.log(x / (1 - x))

# 0.9 → 2.197,  0.5 → 0.0,  0.01 → -4.595
```

**Step 2: Whiten（标准化）**

增加稳定性，使 reward 分布标准化，便于与 KL penalty 项平衡：

```python
def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """将 reward 标准化为零均值单位方差"""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean                         # 可选：保留均值
    return whitened

# 示例：[0.83, 1.20, 3.30, 4.60] → [-1.03, -0.80, 0.51, 1.32]
# 示例：[100.83, 101.20, 103.30, 104.60] → 同样的标准化结果
```

> "We also find it important to whiten the final linear scores in order to increase stability and balance properly with the KL penalty term (β)." — LLaMA2 论文

### 最终 Reward（送入 PPO）

$$R(g|p) = \hat{R}_c(g|p) - \beta \cdot D_{\text{KL}}(\pi_\theta(g|p) \| \pi_0(g|p))$$

---

## 4. 完整流程总结

```
训练阶段:
  人类偏好数据 (chosen, rejected) → Bradley-Terry Loss → 训练 RM

推理/RLHF阶段:
  Actor 生成 response
       ↓
  [prompt, response] → Safety RM → r_s
  [prompt, response] → Helpfulness RM → r_h
       ↓
  reward_select(r_s, r_h) → r_c
       ↓
  inverse_sigmoid(r_c) → logit(r_c)
       ↓
  whiten(logit(r_c)) → r̂
       ↓
  r̂ - β·KL → 送入 PPO 训练
```

---

## 5. RM 的常见问题

### 5.1 Reward Hacking

**定义**：Actor 学会 exploit Reward Model 的弱点，生成 RM 给高分但人类评价低的 response。

**表现**：
- 训练后期 RM reward 持续上升，但实际生成质量下降或出现退化模式
- 重复某些高分短语、生成过长/过短的回复、使用讨好式语气

**成因**：
- RM 是在有限数据上训练的，存在泛化盲区
- RM 的打分是可微的（与 PPO 梯度对齐），Actor 可以沿梯度方向精确 exploit
- RM 可能学到 spurious correlation（如长回复 = 好回复）

**缓解方法**：
- KL penalty（最重要）：限制 Actor 偏离 SFT 模型的程度
- 定期用新数据重新训练 RM（Iterative RLHF）
- 使用多个 RM 做 ensemble
- 设置 reward 上限（reward clipping）
- Best-of-N 采样替代 PPO（不对 RM 做梯度优化）

### 5.2 分布外（OOD）泛化

**问题**：RM 训练数据是 SFT 模型生成的 response，但 PPO 训练过程中 Actor 的分布会漂移。Actor 生成的 response 逐渐偏离 RM 的训练分布，RM 的打分变得不可靠。

**表现**：
- 训练初期 RM 打分准确，后期打分失真
- Actor 生成的文本风格/长度/内容与训练数据差异越来越大

**缓解方法**：
- KL penalty 限制分布漂移
- Iterative RLHF：用当前 Actor 生成新数据，重新训练 RM
- 使用 DPO 等方法，不显式训练 RM

### 5.3 标注噪声和不一致性

**问题**：人类标注者之间存在分歧，同一个标注者在不同时间也可能给出不同标注。

**影响**：
- RM 学到的是 "平均偏好"，可能忽略少数群体的偏好
- 噪声标注会限制 RM 的性能上界

**LLaMA2 的做法**：
- 使用多级偏好标注（4 个等级 + margin）而非简单的 binary
- 训练多个 RM（safety + helpfulness）分别建模不同维度

---

## 6. 面试考点

### Q1: Reward Model 和 Critic Model 有什么区别？

**参考答案**：

| 维度 | Reward Model | Critic Model |
|------|-------------|-------------|
| **输出** | 对完整序列的标量评分 | 每个 token 位置的状态价值 V(s) |
| **输入粒度** | 整个 [prompt, response] | 每个 token 位置的 hidden state |
| **训练方式** | 用人类偏好数据（BT Loss） | 用 PPO 中的 GAE returns（MSE Loss） |
| **训练时机** | RLHF 之前，独立训练 | RLHF 过程中，和 Actor 一起训练 |
| **是否冻结** | 冻结（PPO 中不更新） | 可训练（PPO 中更新） |
| **作用** | 提供外部奖励信号 | 降低 advantage 估计的方差 |

**联系**：Critic 的 backbone 可以从 RM 初始化（共享预训练权重），但训练目标完全不同。

### Q2: 为什么 LLaMA2 用 Margin Loss 而不是标准 BT Loss？

**参考答案**：

标准 BT Loss 只区分 chosen/rejected 二分类，而人类偏好其实有**强度差异**：

- "A 远好于 B" 和 "A 略好于 B" 在标准 BT Loss 中被等同对待
- Margin Loss 引入 $m(r)$ 项，强偏好要求更大的分差才能使 loss 变小

**好处**：
1. **更丰富的监督信号**：利用了偏好强度信息，不浪费标注信息
2. **更好的校准**：RM 的分差更接近真实的偏好强度
3. **数据效率更高**：在标注数据有限时，margin 提供了更强的约束

**注意**：margin 需要多级偏好标注数据支持，如果只有 binary 标注则退化为标准 BT Loss（margin=0）。

### Q3: 如何判断 Reward Model 训练好了？有哪些评估指标？

**参考答案**：

**主要指标**：
1. **Accuracy**：在测试集上，chosen 的 reward 是否高于 rejected。LLaMA2 报告的 RM accuracy ~70-75%（不同分类级别有差异）
2. **分级准确率**：不同偏好强度级别分别计算。"Significantly Better" 级别应该 accuracy 更高

**辅助指标**：
3. **Score 分布**：chosen 和 rejected 的 reward 分布是否有足够区分度（直方图不重叠）
4. **Calibration**：RM 的 score 差异是否和真实偏好强度正相关
5. **OOD 鲁棒性**：在 Actor 生成的新 response 上打分是否合理

**实践经验**：
- RM accuracy > 70% 通常够用（因为人类标注者之间的一致性也只有 ~75%）
- 如果 RM accuracy 很高（>85%）但 RLHF 效果差，可能是 overfitting 或 reward hacking
- 最终评估还是看 RLHF 训练后的 LLM 质量（win rate against SFT）
