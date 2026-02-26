---
title: "KTO 手撕实操"
brief: "KTO（Kahneman-Tversky Optimization）完整实现：从前景理论出发的偏好学习，无需pair-wise偏好数据（只需单样本好/坏标签），KTO loss推导与实现，与DPO数据效率对比，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, kto, preference-optimization, kahneman-tversky, pytorch]
related:
  - "[[AI/3-LLM/RL/DPO/DPO-手撕实操|DPO-手撕实操]]"
  - "[[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/3-LLM/RL/Fundamentals/RL基础算法手撕实操|RL基础算法手撕实操]]"
---

# KTO 手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

KTO（Kahneman-Tversky Optimization）基于前景理论（Prospect Theory），不需要成对偏好数据——只需要每条数据标记 desirable/undesirable 即可训练。

**KTO 损失函数**：

$$L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim D}[w(y)(1 - v_{\text{KTO}}(x,y;\beta))]$$

其中：

$$r_{\text{KTO}}(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

$$z_{\text{ref}} = \mathbb{E}_{x' \sim D}\left[\beta \text{KL}(\pi_\theta(y'|x') \| \pi_{\text{ref}}(y'|x'))\right]$$

$$v_{\text{KTO}}(x,y;\beta) = \begin{cases} \sigma(r_{\text{KTO}} - z_{\text{ref}}) & \text{if } y \sim y_{\text{desirable}} \\ \sigma(z_{\text{ref}} - r_{\text{KTO}}) & \text{if } y \sim y_{\text{undesirable}} \end{cases}$$

$$w(y) = \begin{cases} \lambda_D & \text{if desirable} \\ \lambda_U & \text{if undesirable} \end{cases}$$

**与 DPO 的核心区别**：
- DPO 需要成对偏好数据 $(y_w, y_l)$
- KTO 只需要单条数据 + 二元标签（好/坏）
- KTO 通过 $z_{\text{ref}}$（batch 级别的 KL 基线）实现相对性评估

## 二、核心实现

### 2.1 数据准备

**KTO 数据格式**：prompt + completion + label（True/False）

```python
kto_dataset_dict = {
    "prompt": ["Hey, hello", "How are you", "What is your name?", ...],
    "completion": ["hi nice to meet you", "leave me alone", "My name is Mary", ...],
    "label": [True, False, True, ...],  # desirable / undesirable
}
```

**Token 级别处理**：

```python
def process_kto_dataset(example):
    prompt_id = tokenizer('USER:' + example['prompt'] + 'ASSISTANT:',
                          add_special_tokens=False)['input_ids']
    completion_id = tokenizer(example['completion'],
                              add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    example['input_ids'] = prompt_id + completion_id
    example['label_mask'] = [0] * len(prompt_id) + [1] * len(completion_id)
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example
```

### 2.2 KTO Loss 计算

```python
def get_probs(logits, labels, mask):
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps *= mask / mask.sum()  # 归一化
    return per_token_logps

# 获取 model 和 ref_model 的 log probs
probs = get_probs(logits, batch['input_ids'], batch['label_mask'])
ref_probs = get_probs(ref_logits, batch['input_ids'], batch['label_mask'])

# KL 基线（batch 级别）
kl = (probs - ref_probs).mean().detach()

# 分离 chosen 和 rejected
chosen_id = [k for k, v in enumerate(batch['label']) if v == True]
rejected_id = [k for k, v in enumerate(batch['label']) if v == False]

chosen_ratio = probs[chosen_id] - ref_probs[chosen_id]
rejected_ratio = probs[rejected_id] - ref_probs[rejected_id]

# KTO Loss
beta = 0.1
desirable_weight = 1.33
undesirable_weight = 1.0

chosen_losses = 1 - F.sigmoid(beta * (chosen_ratio - kl))
rejected_losses = 1 - F.sigmoid(beta * (kl - rejected_ratio))  # 注意方向反转

losses = torch.cat(
    (desirable_weight * chosen_losses,
     undesirable_weight * rejected_losses), 0)
kto_loss = losses.nanmean()
```

### 2.3 KTO 的前景理论直觉

- **desirable 样本**：当 $r_{\text{KTO}} > z_{\text{ref}}$ 时，$v$ 接近 1，loss 接近 0——已经足够好的回答不需要过多优化
- **undesirable 样本**：当 $r_{\text{KTO}} < z_{\text{ref}}$ 时，$v$ 接近 1，loss 接近 0——已经足够差的回答也不需要继续惩罚
- **$z_{\text{ref}}$ 作为参考点**：类比前景理论中的"参考价格"，区分 gain 和 loss
- **$\lambda_D > \lambda_U$**：loss aversion——人们对损失比收益更敏感

## 三、工程实践

> 完整代码见：`/tmp/ma-rlhf/notebook/KTO/KTO.ipynb`

**使用 TRL 的 KTO Trainer**：
```python
from trl import KTOTrainer, KTOConfig
# KTO 数据格式：{prompt, completion, label}
# 不需要成对偏好数据，降低数据收集成本
```

## 四、关键洞察与总结

1. **数据效率**：KTO 不需要成对比较，只需要好/坏标签——数据收集成本远低于 DPO
2. **$z_{\text{ref}}$ 是关键**：它提供了相对性评估的基线，没有它 KTO 退化为简单的二分类
3. **前景理论的优雅映射**：desirable/undesirable 对应 gain/loss，$z_{\text{ref}}$ 对应参考点，$\lambda$ 对应 loss aversion
4. **与 DPO 的关系**：当 KTO 的 desirable 和 undesirable 样本恰好配对时，接近 DPO 的效果
5. **实践中的 $\beta$ 选择**：控制偏离 ref_model 的强度，太大导致策略不动，太小导致偏离过远

| 方法 | 数据需求 | 模型数量 | 复杂度 |
|------|---------|---------|--------|
| PPO  | prompt + RM | 4个 | 高 |
| DPO  | (chosen, rejected) 对 | 2个 | 中 |
| KTO  | 单条 + 好/坏标签 | 2个 | 低 |

## See Also

- [[AI/3-LLM/RL/KTO/KTO-完整Notebook实现|KTO 完整 Notebook 实现]] — 端到端完整实现（前景理论偏好建模 + loss asymmetry + z_ref 参考期望），与本文互补：本文讲核心原理，Notebook 讲完整训练流程
- [[AI/3-LLM/RL/DPO/DPO-手撕实操|DPO 手撕实操]] — KTO 的前置算法：理解 BT 偏好模型和 DPO loss 后再看 KTO 的数据效率优势更清晰
- [[AI/3-LLM/RL/DPO/DPO-完整Notebook实现|DPO 完整 Notebook 实现]] — 与 KTO-Notebook 对比：DPO 需要 paired data vs KTO 只需单条 + 标签
- [[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC|lc8 RL×LLM 专题地图]] — 课程 MOC 入口，KTO 在偏好优化章节（Step 4）

## 推荐阅读

- [KTO 原论文（arXiv:2402.01306）](https://arxiv.org/abs/2402.01406) — Kahneman-Tversky Optimization 理论原文
- [前景理论（Kahneman & Tversky 1979）](https://www.jstor.org/stable/1914185) — KTO 数学设计的行为经济学基础
- [[RLHF-DPO-2026-技术全景|RLHF-DPO 2026 技术全景]] — 偏好优化算法谱系（含 KTO 与 DPO/IPO/SimPO 的系统对比）
