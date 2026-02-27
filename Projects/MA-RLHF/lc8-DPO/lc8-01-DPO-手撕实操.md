---
title: "DPO 手撕实操"
brief: "DPO（Direct Preference Optimization）完整推导与PyTorch实现：KL-constrained reward→隐式reward→Bradley-Terry偏好→DPO loss，与PPO的范式对比，SimPO/IPO/ORPO等变体差异总结，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, dpo, preference-optimization, rlhf, pytorch]
related:
  - "[[Projects/MA-RLHF/lc8-PPO/lc8-01-PPO-手撕实操|PPO-手撕实操-MA-RLHF]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-KTO/lc8-01-KTO-手撕实操|KTO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc7/lc7-01-RL基础算法手撕实操|RL基础算法手撕实操]]"
---

# DPO 手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

DPO（Direct Preference Optimization）跳过 Reward Model 训练，直接用偏好数据优化策略模型。核心推导：将 RLHF 的 KL-constrained reward maximization 转化为闭合形式。

**DPO 损失函数**：

$$\mathcal{L}_{\text{DPO}}(\pi;\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**本质**：隐式地将 reward 定义为 $r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，然后用 Bradley-Terry 模型做偏好学习。

## 二、核心实现

### 2.1 Bradley-Terry (BT) 模型

**原理**：BT 模型通过成对比较建模个体强度分数。两种形式等价：

指数形式：$P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)}$

Sigmoid 形式：$P(i \succ j) = \sigma(s_i - s_j)$

**代码**：

```python
class BTModel(nn.Module):
    def __init__(self, N):
        super(BTModel, self).__init__()
        self.reward = nn.Parameter(torch.ones(N))
    
    def forward_sigmoid(self, chosen_id, rejected_id):
        reward_chosen = self.reward[chosen_id]
        reward_rejected = self.reward[rejected_id]
        return torch.sigmoid(reward_chosen - reward_rejected)
    
    def loss_sigmoid(self, pred, label):
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        return -(label * torch.log(pred) + (1 - label) * torch.log(1 - pred))

# 训练
model = BTModel(4)
datas = [(0, 1, 1), (2, 3, 1), (1, 3, 1)]  # 偏好数据
for epoch in range(100):
    for id_i, id_j, label in datas:
        pred = model.forward_sigmoid(id_i, id_j)
        loss = model.loss_sigmoid(pred, torch.tensor(label, dtype=torch.float32))
        loss.backward()
        optimizer.step()
# 训练后 model.reward 反映各选手的相对强度
```

**关键洞察**：更多的 epoch 会导致分数差距越大——过拟合的前兆。

### 2.2 DPO Token-Level Policy 计算

**代码**：

```python
def get_probs(logits, labels):
    """从 logits 获取每个 token 位置对应 label 的 log probability"""
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(2)
    return per_token_logps

# 分别获取 ref/model, chosen/rejected 的 log probs
logits_chosen_ref = ref_model(**x_chosen).logits
logits_chosen = model(**x_chosen).logits
probs_chosen_ref = get_probs(logits_chosen_ref, prompt_chosen)
probs_chosen = get_probs(logits_chosen, prompt_chosen)
# ... 同理获取 rejected

# DPO Loss
beta = 0.1
pi_logratios = probs_chosen - probs_rejected
ref_logratios = probs_chosen_ref - probs_rejected_ref
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits) * label  # label mask 仅计算 response 部分
loss = losses.sum(-1) / attention_mask.sum()
```

### 2.3 DPO 过拟合问题

**IPO 论文指出的问题**：当 $p^*(y \succ y') = 1$ 时，BT 模型要求 $(r(y) - r(y')) \to +\infty$，导致 $\pi^*(y') = 0$——KL 正则化形同虚设。

**代码演示**：

```python
# DPO 训练：rejected policy 概率趋近于 0
for epoch in range(100):
    logits, probs_rejected = get_logits(model, ref_model, x_chosen, x_rejected)
    losses = -F.logsigmoid(beta * logits) * label
    loss.backward(); optimizer.step()
# 观察：probs_rejected → 0（过拟合），且 β 的取值不影响最终结果
```

### 2.4 IPO：DPO 的修正

**IPO 损失函数**：

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(h_\pi(y_w, y_l, x) - \frac{\tau^{-1}}{2}\right)^2\right]$$

其中 $h_\pi$ 与 DPO 的 logits 计算相同（log ratio 差），但用**二次回归**代替 log-sigmoid。

**代码**：

```python
def train_XPO(model, beta, loss_type, epochs, lr):
    for epoch in range(epochs):
        logits, probs_rejected = get_logits(model, ref_model, x_chosen, x_rejected)
        if loss_type == 'DPO':
            losses = -F.logsigmoid(beta * logits) * label
        elif loss_type == 'IPO':
            constant = 1.0 / (beta * 2.0)
            losses = torch.square(logits - constant) * label
        loss.backward(); optimizer.step()
```

**实验结果**：
- DPO：$\beta$ 不影响 $\pi(y')$ 的收敛结果，rejected policy 始终趋向 0
- IPO：$\beta$ 可控制 rejected policy 的收敛值，避免过拟合到 0
- IPO 用较大学习率容易训飞，需要比 DPO 更小的 lr

## 三、工程实践（配套代码）

> 完整代码见：`/tmp/ma-rlhf/ma-rlhf/dpo.py`

### dpo.py 关键架构

```python
def train():
    model, tokenizer = create_model_tokenizer(model_name)
    train_datasets, _ = create_dpo_datasets(dataset_name, None, tokenizer)
    peft_config = create_peft(is_peft)
    
    training_args = DPOConfig(
        loss_type='sigmoid',  # 标准 DPO；也支持 'ipo', 'kto_pair', 'hinge', 'robust'
        max_completion_length=output_max_length,
        max_prompt_length=output_max_length,
        max_length=seq_length,
    )
    trainer = DPOTrainer(
        model, None,  # ref_model=None 时自动从 model 创建
        args=training_args,
        train_dataset=train_datasets,
        peft_config=peft_config,
    )
    trainer.train()
```

**数据预处理**：将各来源（hh-rlhf、CValues 等）统一为 `{prompt, chosen, rejected}` 三列格式。

## 四、关键洞察与总结

1. **DPO 的优雅**：不需要 RM + Critic + Ref 四模型，只需要 model + ref_model（可通过 LoRA 共享）
2. **BT 模型是桥梁**：RLHF 和 DPO 都基于 BT 偏好模型，区别在于显式/隐式学习 reward
3. **DPO 的过拟合是真实问题**：特别是数据稀疏时，DPO 会让 rejected 策略概率趋近 0
4. **IPO 更稳健**：二次损失天然提供正则化，$\beta$ 可控制收敛程度
5. **DPO 在 TRL 中的使用**：`loss_type` 参数支持多种变种（sigmoid/ipo/hinge/kto_pair 等）
6. **DPO vs PPO**：DPO 更简单高效但天花板可能更低；PPO 更灵活但工程复杂度高

> 完整代码见：`/tmp/ma-rlhf/notebook/DPO/BT_model.ipynb`、`/tmp/ma-rlhf/notebook/DPO/DPO.ipynb`、`/tmp/ma-rlhf/ma-rlhf/dpo.py`

## See Also

- [[Projects/MA-RLHF/lc8-DPO/lc8-02-Bradley-Terry模型实现|Bradley-Terry 模型实现]] — BT 偏好建模理论入口：`P(y_w>y_l) = σ(r_w - r_l)` 完整实现（DPO 数学基础）
- [[Projects/MA-RLHF/lc8-DPO/lc8-03-DPO-完整Notebook实现|DPO 完整 Notebook 实现]] — 端到端 Notebook：偏好对处理 + log-ratio loss + β 温度控制，配合本文查工程细节
- [[Projects/MA-RLHF/lc8-KTO/lc8-02-KTO-完整Notebook实现|KTO 完整 Notebook 实现]] — DPO 替代方案（无需 paired data），对比学习
- [[Projects/MA-RLHF/lc8-PPO/lc8-03-RLHF-PPO-完整Pytorch实现|RLHF-PPO 完整实现]] — PPO 四模型架构对照：理解 DPO 为什么能去掉 RM 和 Critic
- RLHF-DPO 2026 技术全景 — 偏好优化算法完整谱系（DPO/IPO/SimPO/ORPO/KTO/REBEL）
