---
title: "DPO 完整 Notebook 实现"
brief: "DPO 端到端 Notebook 实现：偏好对数据处理、log-ratio loss 计算、reference model 冻结策略、β 温度控制。含 DPO vs PPO 关键工程差异（无 Reward Model，无 value function）对比代码。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, rl, dpo, notebook-implementation]
related:
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-01-DPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-02-Bradley-Terry模型实现]]"
  - "[[Projects/MA-RLHF/lc8-PPO/lc8-03-RLHF-PPO-完整Pytorch实现]]"
  - "lc8-RL×LLM-MOC"
---

# DPO 完整 Notebook 实现

> 来源：`ma-rlhf/notebook/DPO/DPO.ipynb` — 手撕 DPO + IPO 对比实验

---

## 1. DPO 数学推导：从 RLHF 到闭式 Loss

### 1.1 起点：KL-正则化 RL 目标

标准 RLHF 目标是在 reward 最大化的同时，约束策略不要偏离参考策略太远：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \, \text{KL}\left[\pi(y|x) \| \pi_{\text{ref}}(y|x)\right]$$

### 1.2 最优解的闭式形式

对上述目标求解，最优策略满足：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ 是配分函数。

### 1.3 反解 reward

从最优策略中反解出 reward：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

### 1.4 代入 Bradley-Terry 偏好模型

BT 模型假设人类偏好概率为：

$$p(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

将反解的 reward 代入，$Z(x)$ 项对消：

$$p(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

### 1.5 DPO Loss（最终形式）

最大似然 → 取负对数 → 得到 DPO 损失函数：

$$\boxed{\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}$$

**关键洞察**：DPO 完全绕过了显式 reward model 和 RL 训练，直接用偏好数据优化策略。

---

## 2. 完整代码实现

### 2.1 模型与数据准备

```python
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
torch.manual_seed(42)

# 小型 Llama 模型用于演示
config = LlamaConfig(
    vocab_size=32,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)
ref_model = LlamaForCausalLM(config)
ref_model.eval()  # 参考模型冻结
model = LlamaForCausalLM(config)  # 待训练的策略模型
```

### 2.2 构造偏好数据

```python
# Chosen:   [Prompt Token | Response Chosen Token]
# Rejected: [Prompt Token | Response Rejected Token]
prompt_length = 6
answer_length = 4

prompt_chosen   = torch.tensor(5, 8, 9, 10, 5, 3,  16, 29, 18, 17, dtype=torch.int64)
prompt_rejected = torch.tensor(5, 8, 9, 10, 5, 3,  26, 14, 31, 0,  dtype=torch.int64)
attention_mask  = torch.tensor(1, 1, 1, 1,  1, 1,  1,  1,  1,  1,  dtype=torch.bool)
label           = torch.tensor(0, 0, 0, 0,  0, 0,  1,  1,  1,  1,  dtype=torch.bool)
# label=1 表示 response 区域，只在这些位置计算 loss

x_chosen = {'input_ids': prompt_chosen, 'attention_mask': attention_mask}
x_rejected = {'input_ids': prompt_rejected, 'attention_mask': attention_mask}
```

### 2.3 Token-Level Log Probability 计算

```python
def get_probs(logits, labels):
    """从 logits 中提取每个 token 对应的 log probability"""
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len] — 要提取概率的 token id
    per_token_logps = torch.gather(
        logits.log_softmax(-1),    # 先做 log_softmax
        dim=2,
        index=labels.unsqueeze(2)  # 用 token id 作为索引
    ).squeeze(2)
    return per_token_logps  # [batch, seq_len]
```

**原理**：对于位置 $t$，模型输出 logits $\to$ log_softmax 得到所有 token 的 log prob $\to$ gather 取出实际 token $y_t$ 对应的 $\log \pi(y_t | y_{<t}, x)$。

### 2.4 DPO Loss 计算

```python
# 分别计算 ref/model 在 chosen/rejected 上的 log prob
logits_chosen_ref   = ref_model(**x_chosen).logits
logits_rejected_ref = ref_model(**x_rejected).logits
logits_chosen       = model(**x_chosen).logits
logits_rejected     = model(**x_rejected).logits

probs_chosen_ref   = get_probs(logits_chosen_ref, prompt_chosen)
probs_chosen       = get_probs(logits_chosen, prompt_chosen)
probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
probs_rejected     = get_probs(logits_rejected, prompt_rejected)

# DPO Loss
beta = 0.1
pi_logratios  = probs_chosen - probs_rejected        # log(π(y_w)) - log(π(y_l))
ref_logratios = probs_chosen_ref - probs_rejected_ref # log(π_ref(y_w)) - log(π_ref(y_l))
logits_dpo    = pi_logratios - ref_logratios           # 相当于 log-ratio 的差
losses = -F.logsigmoid(beta * logits_dpo) * label     # 只在 response 区域计 loss
loss = losses.sum(-1) / attention_mask.sum()
```

### 2.5 完整训练循环

```python
import torch.optim as optim

model = LlamaForCausalLM(config)
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 100

neg_policy_prob = []  # π(y_l) 追踪：rejected 样本的策略概率
logistic_prob = []    # σ(β·logits) 追踪：DPO sigmoid 值
loss_record = []      # Loss 追踪

for i in range(epochs):
    optimizer.zero_grad()

    # 1. Forward: ref 模型不计算梯度
    with torch.no_grad():
        logits_chosen_ref   = ref_model(**x_chosen).logits
        logits_rejected_ref = ref_model(**x_rejected).logits
    logits_chosen   = model(**x_chosen).logits
    logits_rejected = model(**x_rejected).logits

    # 2. Logits → Log Prob
    probs_chosen_ref   = get_probs(logits_chosen_ref, prompt_chosen)
    probs_chosen       = get_probs(logits_chosen, prompt_chosen)
    probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
    probs_rejected     = get_probs(logits_rejected, prompt_rejected)

    # 3. DPO Loss
    beta = 0.1
    pi_logratios  = probs_chosen - probs_rejected
    ref_logratios = probs_chosen_ref - probs_rejected_ref
    logits_dpo    = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits_dpo) * label
    loss = losses.sum(-1) / attention_mask.sum()

    # 4. Backward
    loss.backward()
    optimizer.step()

    # 5. 记录指标
    loss_record.append(loss.item())
    neg_policy_prob.append(torch.exp(probs_rejected[:, -1]).item())
    logistic_prob.append(torch.sigmoid(beta * logits_dpo)[:, -1].item())

    if i % 10 == 0:
        print(f'step {i}, loss:{loss.item():.4f}, '
              f'π_rej:{neg_policy_prob[-1]:.4f}, '
              f'σ_prob:{logistic_prob[-1]:.4f}')
```

**训练观察**：
- `π_rej`（rejected 的策略概率）随训练趋近于 0 — 这正是 DPO 的过拟合问题
- `σ_prob`（DPO sigmoid 值）趋近于 1 — 模型越来越自信 chosen > rejected
- Loss 单调递减

---

## 3. DPO 的两个已知问题

### 3.1 问题一：需要 Paired Preference Data

DPO 的每条训练数据必须是 $(x, y_w, y_l)$ 三元组 — 同一个 prompt 下的 chosen 和 rejected 配对。实际上标注成本高，配对关系难以保证质量。

### 3.2 问题二：过拟合 / OOD 泛化差

**IPO 论文指出的核心问题**：当 $p^*(y_w \succ y_l) = 1$（绝对偏好）时，BT 模型要求 reward 差趋向 $+\infty$，代入最优策略就有 $\pi(y_l) \to 0$。

> 即使真实偏好只有 $p^* = 0.8$，有限数据下经验估计可能为 $\hat{p} = 1$，此时对任意 KL 系数 $\beta$，策略都会让 $\pi(y_l) = 0$。

**后果**：KL 正则化失效 → 策略偏离参考模型太远 → 在分布外（OOD）输入上表现差。

Notebook 实验验证：训练 100 epoch 后，DPO 的 `π_rej` 无论 $\beta$ 取 0.1 还是 0.5，都收敛到 0。

---

## 4. IPO：修正 DPO 过拟合

**一句话**：IPO 将 DPO 的 log-sigmoid loss 替换为二次回归损失 $(\text{logits} - \frac{1}{2\tau})^2$，使得 log-ratio 差收敛到有限值而非 $\pm\infty$，从而避免 $\pi(y_l) \to 0$ 的过拟合。

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(y_w, y_l, x) \sim \mathcal{D}} \left[\left(h_\pi(y_w, y_l, x) - \frac{1}{2\tau}\right)^2\right]$$

其中 $h_\pi$ 与 DPO 的 logits 计算公式相同：$\log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$。

**Notebook 实验对比**：
- DPO：$\beta$ 不影响 $\pi(y_l)$ 最终收敛结果，都趋向 0
- IPO：$\beta$ 可控地调节 $\pi(y_l)$ 的收敛值，避免策略坍缩

```python
# IPO loss（对比 DPO）
constant = 1.0 / (beta * 2.0)  # = 1/(2τ)
if loss_type == 'DPO':
    losses = -F.logsigmoid(beta * logits) * label
elif loss_type == 'IPO':
    losses = torch.square(logits - constant) * label  # 二次回归
```

---

## 5. 面试考点

### Q1：DPO 为什么不需要训练 Reward Model？

DPO 的核心思路：KL-正则化 RL 的最优策略有闭式解 → 反解出 reward 用策略表示 → 代入 BT 偏好模型后配分函数 $Z(x)$ 对消 → 得到只依赖 $\pi_\theta / \pi_{\text{ref}}$ 的闭式 loss。整个过程不需要显式的 reward model，也不需要 PPO 等 RL 算法。

### Q2：DPO Loss 中 β 的物理意义是什么？

$\beta$ 是 KL 正则化系数，控制策略偏离参考模型的程度。$\beta$ 越大，偏离惩罚越重，策略越保守。在 loss 中它乘以 log-ratio 差，相当于缩放偏好信号的强度。但 IPO 论文指出，在 DPO 的 BT 框架下，$\beta$ 无法真正阻止 $\pi(y_l) \to 0$。

### Q3：DPO 训练时为什么需要一个 frozen 的 reference model？

参考模型 $\pi_{\text{ref}}$ 的作用：(1) 提供 KL 锚点，防止策略过度偏离预训练分布；(2) 在 loss 计算中，$\log \pi_{\text{ref}}$ 项与 $\log \pi_\theta$ 形成 log-ratio，这个比值才是 DPO 真正优化的对象。如果没有 ref model，loss 无法正确定义。

### Q4：从 DPO 到 IPO 解决了什么问题？核心改动是什么？

DPO 的 BT 模型要求 reward 差趋向无穷来拟合确定性偏好，导致 $\pi(y_l) \to 0$（过拟合）。IPO 把 log-sigmoid loss 换成二次损失 $(h - 1/2\tau)^2$，让 log-ratio 差回归到有限目标值 $1/2\tau$，而非无穷大。核心改动就一行：sigmoid → square。
