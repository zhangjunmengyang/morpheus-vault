---
title: "KTO 完整 Notebook 实现"
brief: "KTO（Kahneman-Tversky Optimization）完整 Notebook 实现：前景理论偏好建模（loss asymmetry：win loss 权重不对称）、单样本训练（不需要成对数据）、z_ref 参考期望计算。DPO 替代方案，适合无对比数据场景。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, rl, kto, notebook-implementation, preference-learning]
related:
  - "[[AI/3-LLM/RL/KTO/KTO-手撕实操]]"
  - "[[AI/3-LLM/RL/DPO/DPO-完整Notebook实现]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# KTO 完整 Notebook 实现

> 来源：`ma-rlhf/notebook/KTO/KTO.ipynb` — KTO: Model Alignment as Prospect Theoretic Optimization

---

## 1. 前景理论（Prospect Theory）与 KTO 的动机

### 1.1 为什么选择前景理论？

DPO 需要 **paired preference data**（同一 prompt 下的 chosen/rejected 对），标注成本高。现实中更容易获得的是 **binary feedback**：对单个回答标记"好"或"坏"（👍/👎）。

KTO 的核心思想：借鉴行为经济学中 Kahneman & Tversky 的前景理论（Prospect Theory），人类对损失的敏感度远高于对等量收益的敏感度（**损失厌恶**）。

### 1.2 损失厌恶系数 λ

前景理论的关键参数：
- $\lambda_D$（desirable weight）= 1.33 — 好回答的权重
- $\lambda_U$（undesirable weight）= 1.0 — 坏回答的权重

$\lambda_D > \lambda_U$ 意味着模型对"失去好回答"的惩罚大于"保留坏回答"的惩罚。这个不对称性与人类决策中"损失比收益感受更强"的心理一致。

### 1.3 为什么这个框架有效？

- 不需要配对数据，每条样本独立标注 binary label
- 通过非对称权重引入人类偏好的心理学先验
- Loss 设计让模型在 desirable 样本上的 reward 相对 baseline 为正，在 undesirable 上为负

---

## 2. KTO Loss 公式

### 2.1 完整公式

$$\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}} \left[ w(y) \cdot (1 - v_{\text{KTO}}(x, y; \beta)) \right]$$

其中：

$$r_{\text{KTO}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

$$z_{\text{ref}} = \mathbb{E}_{x' \sim \mathcal{D}} \left[ \beta \, \text{KL}\left(\pi_\theta(y'|x') \| \pi_{\text{ref}}(y'|x')\right) \right]$$

$$v_{\text{KTO}}(x, y; \beta) = \begin{cases} \sigma(r_{\text{KTO}}(x,y) - z_{\text{ref}}) & \text{if } y \sim y_{\text{desirable}} \\ \sigma(z_{\text{ref}} - r_{\text{KTO}}(x,y)) & \text{if } y \sim y_{\text{undesirable}} \end{cases}$$

$$w(y) = \begin{cases} \lambda_D & \text{if } y \sim y_{\text{desirable}} \\ \lambda_U & \text{if } y \sim y_{\text{undesirable}} \end{cases}$$

### 2.2 直觉解读

| 分支 | 信号含义 | 优化方向 |
|------|---------|---------|
| Desirable | $r_{\text{KTO}} - z_{\text{ref}}$ 越大越好 | 让好回答的 reward 远高于 baseline |
| Undesirable | $z_{\text{ref}} - r_{\text{KTO}}$ 越大越好 | 让坏回答的 reward 远低于 baseline |

$z_{\text{ref}}$ 是 batch 级别的 KL baseline，起到"参考点"作用 — 这正是前景理论中"参考点"概念的体现。

---

## 3. 完整代码实现

### 3.1 模型加载

```python
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
torch.manual_seed(42)

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)
ref_model = LlamaForCausalLM(config)
ref_model.eval()
model = LlamaForCausalLM(config)

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceM4/tiny-random-LlamaForCausalLM')
```

### 3.2 数据准备：单样本 + Binary Label

**核心区别**：不需要 (chosen, rejected) 配对，只需要 (prompt, completion, label)。

```python
kto_dataset_dict = {
    "prompt": [
        "Hey, hello",
        "How are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "How to kill a man?",
    ],
    "completion": [
        "hi nice to meet you",        # ✅ True
        "leave me alone",              # ❌ False
        "I don't have a name",         # ❌ False
        "My name is Mary",             # ✅ True
        "Python",                      # ✅ True
        "C++",                         # ❌ False
        "Java",                        # ❌ False
        "Use gun shoot his head",      # ❌ False
    ],
    "label": [True, False, False, True, True, False, False, False],
}
```

### 3.3 Tokenization：构建 prompt + completion + label_mask

```python
from datasets import Dataset

def process_kto_dataset(example):
    prompt_id = tokenizer(
        'USER:' + example['prompt'] + 'ASSISTANT:',
        add_special_tokens=False
    )['input_ids']
    completion_id = tokenizer(
        example['completion'],
        add_special_tokens=False
    )['input_ids'] + [tokenizer.eos_token_id]

    example['input_ids'] = prompt_id + completion_id
    example['label_mask'] = [0] * len(prompt_id) + [1] * len(completion_id)
    # label_mask: 0=prompt区域不参与loss, 1=completion区域参与loss
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example

dataset = Dataset.from_dict(kto_dataset_dict).map(process_kto_dataset)
```

### 3.4 自定义 Padding Collator

```python
from transformers import DataCollatorWithPadding

class KTOPaddingCollator(DataCollatorWithPadding):
    def collate_batch(self, examples):
        input_ids = [ids for ids in examples['input_ids']]
        attention_mask = [mask for mask in examples['attention_mask']]
        label_mask = [lm for lm in examples['label_mask']]
        label = [l for l in examples['label']]

        max_length = max(len(ids) for ids in input_ids)

        # 动态 padding 到 batch 内最大长度
        padded_input_ids, padded_attention_mask, padded_label_mask = [], [], []
        for ids, mask, lab in zip(input_ids, attention_mask, label_mask):
            pad_len = max_length - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_label_mask.append(lab + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "label_mask": torch.tensor(padded_label_mask, dtype=torch.long),
            "label": label,
        }
```

### 3.5 Token-Level Log Prob（KTO 版）

```python
def get_probs(logits, labels, mask):
    """与 DPO 的区别：加了 mask 归一化"""
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(2)
    per_token_logps *= mask / mask.sum()  # label_mask 加权归一化
    return per_token_logps
```

### 3.6 KTO Loss 计算

```python
# Forward
ref_logits = ref_model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask']).logits
logits = model(input_ids=batch['input_ids'],
               attention_mask=batch['attention_mask']).logits

probs = get_probs(logits, batch['input_ids'], batch['label_mask'])
ref_probs = get_probs(ref_logits, batch['input_ids'], batch['label_mask'])

# z_ref: batch 级别的 KL baseline
kl = (probs - ref_probs).mean().detach()

# 按 label 分组
chosen_id = [k for k, v in enumerate(batch['label']) if v == True]
rejected_id = [k for k, v in enumerate(batch['label']) if v == False]

# 计算 log ratio: r_KTO = β * log(π/π_ref)
chosen_ratio = probs[chosen_id, :] - ref_probs[chosen_id, :]
rejected_ratio = probs[rejected_id, :] - ref_probs[rejected_id, :]

# KTO Loss 两个分支
beta = 0.1
desirable_weight = 1.33   # λ_D — 损失厌恶：好样本权重更大
undesirable_weight = 1.0  # λ_U

# Desirable: σ(r_KTO - z_ref) → 越大越好 → loss = 1 - σ(...)
chosen_losses = 1 - F.sigmoid(beta * (chosen_ratio - kl))
# Undesirable: σ(z_ref - r_KTO) → 越大越好 → loss = 1 - σ(...)
rejected_losses = 1 - F.sigmoid(beta * (kl - rejected_ratio))

# 加权合并
losses = torch.cat(
    (desirable_weight * chosen_losses,
     undesirable_weight * rejected_losses), 0)
kto_loss = losses.nanmean()
```

---

## 4. KTO vs DPO：数据效率对比

| 维度 | DPO | KTO |
|------|-----|-----|
| **数据格式** | $(x, y_w, y_l)$ 三元组 | $(x, y, \text{label})$ 二元组 |
| **标注需求** | 同一 prompt 下的 paired comparison | 单个回答的 binary feedback（👍/👎） |
| **标注成本** | 高（需要人工比较两个回答） | 低（只需判断单个回答好坏） |
| **数据利用** | 每条数据必须配对 | 可以利用所有有标签的数据 |
| **不平衡容忍度** | 必须 1:1 配对 | 天然支持正负样本不平衡（通过 λ 调节） |
| **实际场景** | 适合有专业标注团队的场景 | 适合用户反馈收集（点赞/点踩） |

**核心优势**：KTO 将偏好学习的数据门槛从 "paired comparison" 降低到 "binary label"，在标注资源有限时显著更实用。

---

## 5. 面试考点

### Q1：KTO 的 $z_{\text{ref}}$ 为什么用 batch 级别的 KL？

$z_{\text{ref}} = \mathbb{E}[\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})]$ 是当前策略与参考策略的平均偏离程度，作为前景理论中的"参考点"。好回答的 reward 必须超过这个 baseline 才算"收益"，坏回答的 reward 必须低于 baseline 才算"损失"。没有这个 baseline，模型无法区分"绝对好坏"和"相对于当前策略的好坏"。实现上用 `(probs - ref_probs).mean().detach()` 估计。

### Q2：KTO 的 $\lambda_D = 1.33, \lambda_U = 1.0$ 怎么来的？

来自前景理论的实验研究：人类对损失的感受强度约为等量收益的 1.5-2.5 倍。KTO 论文取 $\lambda_D / \lambda_U \approx 1.33$，让 desirable 样本的 loss 权重略高于 undesirable。直觉：如果模型"丢掉"一个好回答的惩罚应该比"保留"一个坏回答的惩罚更大。具体值可以作为超参数调节。

### Q3：KTO 不需要 paired data，那它怎么知道什么是"好"？

DPO 通过同一 prompt 下的 A>B 比较学习相对偏好。KTO 的机制不同：它通过 $z_{\text{ref}}$（KL baseline）建立参考点，desirable 样本的 reward 应高于参考点，undesirable 的应低于参考点。本质上是把"相对比较"替换成了"相对于策略均值的绝对判断"。这意味着 KTO 学到的是"这个回答比平均水平好/差"，而非"A 比 B 好"。
