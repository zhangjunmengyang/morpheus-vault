---
title: "KTO 手撕实操（MA-RLHF lc8）"
brief: "KTO（Kahneman-Tversky Optimization）从零手撕：去除配对偏好数据假设，用单条 desirable/undesirable 样本训练；KL 参考点改为全局期望而非 per-pair；与 DPO 的关键对比——KTO 更适合非配对真实数据场景。MA-RLHF lc8 实操笔记。"
type: code-practice
date: 2026-02-26
source: "MA-RLHF notebook/KTO/KTO.ipynb"
tags:
  - KTO
  - 偏好优化
  - RLHF
  - 手撕实操
  - MA-RLHF-lc8
related:
  - [[Projects/MA-RLHF/lc8-DPO/lc8-05-DPO-IPO-手撕实操|lc8-DPO-IPO-手撕实操]]
  - [[Projects/MA-RLHF/lc8-PPO/lc8-09-LLaMA2-Reward-Model手撕|lc8-LLaMA2-Reward-Model手撕]]
  - [[AI/3-LLM/RL/实践/RLHF-工程全栈|RLHF-DPO-2026-技术全景]]
  - [[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]]
---

# KTO 手撕实操（MA-RLHF lc8）

**来源**：MA-RLHF `notebook/KTO/KTO.ipynb`  
**论文**：KTO: Model Alignment as Prospect Theoretic Optimization  
**难度**：★★★★☆（偏好学习重要变体，无需 pair 数据，工程部署更简单）  
**关联**：[[Projects/MA-RLHF/lc8-DPO/lc8-05-DPO-IPO-手撕实操|lc8-DPO-IPO-手撕实操]] | [[Projects/MA-RLHF/lc8-DPO/lc8-02-Bradley-Terry模型实现|lc8-Bradley-Terry-偏好建模手撕]]

---

## 一、KTO 的核心动机：为什么不需要 Pair？

**DPO 的数据需求**：每个样本必须是 `(prompt, chosen, rejected)` 三元组。问题：
1. 人类标注成对数据成本高（每对需要两条回答 + 比较判断）
2. 许多场景只有 **单标签数据**（这条回答好/这条回答不好），没有对应的反例

**KTO 的创新**：来自**前景理论**（Prospect Theory，行为经济学）——人类对损失的感知比对等量收益更强烈（损失厌恶）。不需要成对比较，只需要 `(prompt, completion, label: good/bad)` 单标签数据。

---

## 二、KTO Loss 公式理解

$$\mathcal{L}_{KTO} = \mathbb{E}_{x,y \sim D}\left[w(y)\left(1 - v_{KTO}(x,y;\beta)\right)\right]$$

其中：

$$r_{KTO}(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \quad \text{（implicit reward，与 DPO 相同）}$$

$$z_{ref} = \mathbb{E}_{x'}\left[\beta \cdot \text{KL}(\pi_\theta(y'|x') \| \pi_{ref}(y'|x'))\right] \quad \text{（KL baseline，batch 估计）}$$

$$v_{KTO}(x,y;\beta) = \begin{cases}
\sigma(r_{KTO}(x,y) - z_{ref}) & \text{if desirable（好）} \\
\sigma(z_{ref} - r_{KTO}(x,y)) & \text{if undesirable（差）}
\end{cases}$$

$$w(y) = \begin{cases}
\lambda_D & \text{if desirable} \\
\lambda_U & \text{if undesirable}
\end{cases}$$

**直觉解读**：
- 好回答（desirable）：想让 $r_{KTO} - z_{ref}$ 更大 → sigmoid 趋近 1 → loss 趋近 0 ✓
- 差回答（undesirable）：想让 $z_{ref} - r_{KTO}$ 更大（即 $r_{KTO}$ 更小）→ loss 趋近 0 ✓
- $z_{ref}$ = KL 散度期望值，作为 baseline，防止 policy 乱漂（类似 GRPO 的 advantage baseline）

---

## 三、数据格式：单标签，不需要 Pair

```python
kto_dataset_dict = {
    "prompt": [
        "Hey, hello",        # 同一 prompt 可以对应多条 completion
        "How are you",
        "What is your name?",
        "What is your name?",  # 同 prompt，两条不同的 completion
        "How to kill a man?",
    ],
    "completion": [
        "hi nice to meet you",     # label=True（好）
        "leave me alone",           # label=False（差）
        "I don't have a name",      # label=False
        "My name is Mary",          # label=True
        "Use gun shoot his head",   # label=False（安全问题）
    ],
    "label": [True, False, False, True, False]
}
```

**关键**：`label` 是布尔值（好/差），不是 pair（chosen/rejected）。好回答和差回答可以来自完全不同的 prompt，**不需要配对**。

---

## 四、数据处理：Padding Collator

```python
def process_kto_dataset(example):
    prompt_id = tokenizer('USER:' + example['prompt'] + 'ASSISTANT:',
                          add_special_tokens=False)['input_ids']
    completion_id = tokenizer(example['completion'])['input_ids'] + [eos_token_id]

    example['input_ids']   = prompt_id + completion_id
    example['label_mask']  = [0] * len(prompt_id) + [1] * len(completion_id)  # response 部分为 1
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example
```

**Padding 策略**：实现了 `KTOPaddingCollator`，对不同长度的样本做动态 padding，与 SFT 的 `DataCollatorForTokenClassification` 类似。

---

## 五、KTO Loss 完整实现

```python
# Step 1: Forward 两个模型（model + ref）
ref_logits = ref_model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask']).logits
logits = model(input_ids=batch['input_ids'],
               attention_mask=batch['attention_mask']).logits

# Step 2: 计算 token-level log prob（response 部分加权）
def get_probs(logits, labels, mask):
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2,
        index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps *= mask / mask.sum()  # ← mask 归一化，response 部分有效
    return per_token_logps

probs     = get_probs(logits,     batch['input_ids'], batch['label_mask'])
ref_probs = get_probs(ref_logits, batch['input_ids'], batch['label_mask'])

# Step 3: 估计 KL baseline（batch 内均值）
kl = (probs - ref_probs).mean().detach()  # ← detach，不对 z_ref 求梯度

# Step 4: 分离 chosen/rejected
chosen_id   = [k for k, v in enumerate(batch['label']) if v == True]
rejected_id = [k for k, v in enumerate(batch['label']) if v == False]

chosen_ratio   = probs[chosen_id]   - ref_probs[chosen_id]    # r_KTO(chosen)
rejected_ratio = probs[rejected_id] - ref_probs[rejected_id]  # r_KTO(rejected)

# Step 5: KTO Loss
beta = 0.1
desirable_weight   = 1.33  # λ_D（通常略大于 λ_U，强调好回答）
undesirable_weight = 1.0   # λ_U

# desirable：sigmoid(r - z_ref)，想让 r > z_ref
chosen_losses   = 1 - F.sigmoid(beta * (chosen_ratio   - kl))
# undesirable：sigmoid(z_ref - r)，想让 r < z_ref
rejected_losses = 1 - F.sigmoid(beta * (rejected_ratio - kl))
# ← 注意符号相反！
# DPO：sigmoid(r_w - r_l)，同时处理 chosen 和 rejected 的对比
# KTO：分别对 chosen 和 rejected 做独立的 sigmoid

losses = torch.cat([
    desirable_weight   * chosen_losses,
    undesirable_weight * rejected_losses
], dim=0)
kto_loss = losses.nanmean()
```

---

## 六、KTO 关键设计解析

### 6.1 z_ref 的作用

`z_ref = E[β * KL(π_θ || π_ref)]` 是 KL 散度的期望，作为 reward 的 baseline：
- 如果 model 与 ref 完全一致（未训练），`z_ref ≈ 0`，`r_KTO ≈ 0`
- 训练后 model 偏离 ref，`z_ref` 增大，threshold 提高——防止 policy 一直提高 chosen 的 r 而不惩罚 rejected
- `detach()`：z_ref 是估计值，不参与梯度（与 GRPO advantage 中的 baseline 处理一致）

### 6.2 权重不对称（λ_D > λ_U）

前景理论：人类对"损失"（不好的回答）的感知强于对"收益"（好的回答）。KTO 用 `λ_D = 1.33 > λ_U = 1.0` 模拟这种不对称性——稍微更强调奖励好回答。TRL 的实现中 λ_D 通常在 1.0-2.0 之间调整。

### 6.3 对比 DPO 的符号差异

```
DPO chosen:   -logsigmoid(beta * h_π)  # h_π = r_w - r_l（差值）
DPO rejected: 同一个 loss 项（不分开）

KTO desirable:   1 - sigmoid(beta * (r - z_ref))  # r 越大越好
KTO undesirable: 1 - sigmoid(beta * (z_ref - r))  # r 越小越好（注意符号反转）
```

**KTO 把偏好问题分解成两个独立的"满意度"目标**，而 DPO 是一个统一的对比目标。

---

## 七、KTO vs DPO vs PPO 三角对比

| 维度 | PPO | DPO | KTO |
|------|-----|-----|-----|
| 数据格式 | 在线生成 | (prompt, chosen, rejected) | (prompt, completion, label) |
| 数据配对需求 | 无需 pair | 必须 pair | **无需 pair** |
| 数据采集成本 | 最高（RL rollout）| 中等（需标注 pair）| **最低（单标签即可）** |
| 训练稳定性 | 不稳定（PPO 方差大）| 中等 | **高（无 RL rollout）** |
| 过拟合风险 | 低（KL 约束有效）| 中（π_l → 0）| 低（z_ref 防collapse）|
| 使用场景 | 复杂 reasoning | 有偏好对比数据 | **海量单标签数据** |
| 代表应用 | R1/o1 RLHF | Llama-2 Chat | 工业数据（只有点赞/点踩）|

---

## 八、面试必备问题

**Q1：KTO 为什么不需要成对偏好数据？**  
A：KTO 的 loss 对 desirable 和 undesirable 分别计算独立的 sigmoid，不需要 chosen-rejected 的对比差。z_ref（KL baseline）替代了 rejected 样本在 DPO 中的"参照物"角色。

**Q2：z_ref 在 KTO 中的作用？**  
A：相当于 reward 的动态 baseline。好回答的 reward 需要超过 z_ref 才算"真的好"，差回答的 reward 需要低于 z_ref 才算"真的差"。随训练 KL 增大，z_ref 也增大，自动调整难度。

**Q3：λ_D > λ_U 的理论依据？**  
A：前景理论（Kahneman & Tversky）——人类对损失的感知比对等量收益更强烈（loss aversion）。KTO 模拟这种人类行为，适当放大对好回答的奖励权重。

**Q4：什么场景下 KTO 比 DPO 更好？**  
A：①只有点赞/踩（单标签）没有明确偏好对比的工业数据；②数据量大但标注成本低的场景；③已有 SFT 数据想直接转换（每条 SFT 数据天然是 desirable）。

---

## 九、知识连接

- **前驱**：[[Projects/MA-RLHF/lc8-DPO/lc8-02-Bradley-Terry模型实现|lc8-Bradley-Terry-偏好建模手撕]] — BT Model 是 DPO 的根基
- **对比**：[[Projects/MA-RLHF/lc8-DPO/lc8-05-DPO-IPO-手撕实操|lc8-DPO-IPO-手撕实操]] — 需要成对数据的主流方法
- **理论**：Prospect Theory (Kahneman & Tversky 1979) — KTO 的行为经济学基础
- **工程**：TRL `KTOTrainer` — 工业级 KTO 实现，API 与 `DPOTrainer` 基本一致

---

*入库时间：2026-02-26*  
*来源：MA-RLHF notebook/KTO/KTO.ipynb*  
*状态：Batch B ✅*
