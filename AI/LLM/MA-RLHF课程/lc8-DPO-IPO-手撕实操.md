---
title: "DPO + IPO 手撕实操（MA-RLHF lc8）"
brief: "DPO（Direct Preference Optimization）从 BT 模型推导到代码的完整手撕：LogSigmoid 损失、chosen/rejected 对构造、temperature 参数；IPO（Identity Preference Optimization）去 BT 假设的改进；MA-RLHF lc8 实操笔记，面试必备偏好优化基础。"
type: code-practice
date: 2026-02-26
source: "MA-RLHF notebook/DPO/DPO.ipynb"
tags:
  - DPO
  - IPO
  - 偏好优化
  - RLHF
  - 手撕实操
  - MA-RLHF-lc8
related:
  - [[AI/LLM/MA-RLHF课程/lc8-LLaMA2-Reward-Model手撕|lc8-LLaMA2-Reward-Model手撕]]
  - [[AI/LLM/MA-RLHF课程/lc8-KTO-手撕实操|lc8-KTO-手撕实操]]
  - [[AI/LLM/RL/RLHF-DPO-2026-技术全景|RLHF-DPO-2026-技术全景]]
  - [[AI/LLM/MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]]
---

# DPO + IPO 手撕实操（MA-RLHF lc8）

**来源**：MA-RLHF `notebook/DPO/DPO.ipynb`  
**难度**：★★★★★（面试高频，偏好对齐主流方法，RLHF 替代方案）  
**关联**：[[AI/LLM/MA-RLHF课程/lc8-Bradley-Terry-偏好建模手撕|lc8-Bradley-Terry-偏好建模手撕]] | [[AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-手撕实操|lc8-RLHF-PPO-手撕实操]] | [[AI/LLM/MA-RLHF课程/lc8-KTO-手撕实操|lc8-KTO-手撕实操]]

---

## 一、DPO 的核心推导

### BT Model → DPO：三步消去 Reward Model

**Step 1**：最优 RL policy 在 KL 约束下有解析解：
$$\pi^*(y|x) = \frac{\pi_{ref}(y|x) \exp(r(x,y)/\beta)}{Z(x)}$$

**Step 2**：反解出 reward：
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Step 3**：代入 BT Model loss（$Z(x)$ 在 chosen-rejected 对中相消）：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**关键**：不需要训 Reward Model，不需要 PPO rollout，直接用偏好数据对做对比学习。两个模型（policy + ref），一次 forward 各算 logprob，做减法。

---

## 二、Token-Level Log Probability 提取

```python
def get_probs(logits, labels):
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    # 从 vocab_size 维度取出 label token 的 log prob
    per_token_logps = torch.gather(
        logits.log_softmax(-1),      # [batch, seq, vocab] → log_softmax
        dim=2,
        index=labels.unsqueeze(2)    # [batch, seq, 1]
    ).squeeze(2)                     # [batch, seq]
    return per_token_logps
```

**数据格式**：
```python
prompt_length = 6
# prompt + response（chosen/rejected 共享相同 prompt）
prompt_chosen   = [[5, 8, 9, 10, 5, 3,  16, 29, 18, 17]]  # prompt(6) + chosen(4)
prompt_rejected = [[5, 8, 9, 10, 5, 3,  26, 14, 31,  0]]  # prompt(6) + rejected(4)
label = [[0, 0, 0, 0, 0, 0,  1, 1, 1, 1]]  # response 部分为 1，prompt 为 0
```

---

## 三、DPO Loss 完整实现

```python
beta = 0.1

# 4 次 forward：model/ref × chosen/rejected
logits_chosen_ref  = ref_model(**x_chosen).logits     # π_ref(y_w|x)
logits_rejected_ref = ref_model(**x_rejected).logits  # π_ref(y_l|x)
logits_chosen      = model(**x_chosen).logits          # π_θ(y_w|x)
logits_rejected    = model(**x_rejected).logits        # π_θ(y_l|x)

probs_chosen_ref  = get_probs(logits_chosen_ref,  prompt_chosen)
probs_chosen      = get_probs(logits_chosen,      prompt_chosen)
probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
probs_rejected    = get_probs(logits_rejected,    prompt_rejected)

# DPO loss：log ratio 之差
pi_logratios  = probs_chosen  - probs_rejected       # log[π_θ(y_w)/π_θ(y_l)]
ref_logratios = probs_chosen_ref - probs_rejected_ref # log[π_ref(y_w)/π_ref(y_l)]

logits_dpo = pi_logratios - ref_logratios             # implicit reward gap
losses = -F.logsigmoid(beta * logits_dpo) * label    # 只在 response 部分算 loss
loss = losses.sum(-1) / attention_mask.sum()
```

**逐步理解**：
- `pi_logratios = log π_θ(y_w) - log π_θ(y_l)`：policy 对 chosen/rejected 的相对偏好
- `ref_logratios`：ref policy 对 chosen/rejected 的相对偏好（归一化基准）
- `logits_dpo = pi_logratios - ref_logratios`：policy 相对于 ref 的改进量
- `logsigmoid(beta * logits_dpo)`：BT 模型中 "chosen 比 rejected 好" 的概率的 log

---

## 四、DPO 的过拟合问题与 IPO 的修复

### DPO 过拟合根因

来自论文 IPO（"A General Theoretical Paradigm to Understand Learning from Human Preferences"）：

**Case**：假设 $P^*(y \succ y') = 1$（y 绝对优于 y'）。BT Model 要求 $r(y) - r(y') \to +\infty$。代入最优 policy：$\pi^*(y') / \pi^*(y) = 0$，即 $\pi^*(y') = 0$。

**结果**：π_θ(y_l) → 0，KL 正则化失效（KL 强度随 policy 确定性增强而减弱），loss 可以被"hack"到无穷小。实验验证：DPO 训练到最后 rejected 概率真的趋向 0。

### IPO Loss：二次目标防 collapse

```python
# IPO 使用二次损失（不是对数 sigmoid）
# h_π = log[π(y_w)/π_ref(y_w)] - log[π(y_l)/π_ref(y_l)]  (与 DPO 相同)
# 目标：h_π → 1/(2β)（有限值，不发散）

def compute_loss(logits, probs_rejected, beta, loss_type):
    if loss_type == 'DPO':
        losses = -F.logsigmoid(beta * logits) * label   # 对数目标，可趋近 0
    elif loss_type == 'IPO':
        constant = 1.0 / (beta * 2.0)                   # 目标值：1/(2β)
        losses = (logits - constant) ** 2 * label        # 二次目标，有限收敛
    return losses.sum(-1) / attention_mask.sum()
```

**IPO vs DPO 数值实验结论**（来自 notebook 实验）：
- DPO：β 取什么值，π(y_l) 最终都收敛到 0（β 不控制收敛目标）
- IPO：β 控制最终 π(y_l) 收敛到的**有限值**（β 越大，收敛的 π_l 越大）
- IPO 对学习率更敏感：SGD 大 lr 容易训飞，需要 Adam + 小 lr

**关键图示理解**：
```
DPO:  π_l → 0（无论 β）
IPO:  π_l → finite_value(β)
      β=0.1 → 较小值
      β=0.5 → 中等值
      β=1.0 → 较大值
```

---

## 五、DPO vs PPO 对比

| 维度 | PPO | DPO |
|------|-----|-----|
| 数据需求 | 在线生成（实时 rollout）| 离线 preference pair |
| 模型数 | 4（actor/ref/rm/critic）| 2（policy/ref）|
| 计算开销 | 高（rollout + 多模型）| 低（只需 forward）|
| 核心问题 | reward hacking | distribution shift（OOD）|
| 适合场景 | 动态 reward，复杂任务 | 静态 preference 数据集 |
| 数学推理 | GRPO 专用 | 不适合（无 verifiable reward）|

**DPO 的根本局限**：训练数据是离线的 preference pair，推断的是在 SFT 分布下的 reward。但 policy 会随训练偏移，新的 policy 生成的数据可能与训练数据分布差异大（OOD），导致 reward 信号不可靠。这是 Online DPO / Iterative DPO 试图解决的问题。

---

## 六、面试必备问题

**Q1：DPO 为什么不需要 Reward Model？**  
A：BT Model 的最优 RL policy 有解析解，反解出 reward 后代入 preference loss，Z(x) 在 chosen-rejected 对中相消。Policy 的 log-ratio 对 ref 的偏移量隐式地充当了 reward。

**Q2：DPO 的 β 控制什么？**  
A：β 控制偏离 ref policy 的 KL 惩罚强度。β 大 → 保守（贴近 ref）；β 小 → 激进（深度改变偏好）。但在 DPO 中 β 不控制收敛目标（π_l 总趋向 0）——这是 DPO 过拟合的核心。IPO 的 β 则真正控制收敛到的有限值。

**Q3：IPO 如何修复 DPO 的过拟合？**  
A：用二次损失 `(h_π - 1/2β)²` 替代对数 sigmoid。二次目标强制 h_π 收敛到有限值 `1/(2β)` 而不是无穷，KL 正则化不失效。

**Q4：DPO 和 SFT 的关系？**  
A：DPO = 在 SFT checkpoint 基础上做偏好对齐，不改变 SFT 打下的语言格式基础，只调整 chosen vs rejected 的相对概率。SFT 初始化对 DPO 训练质量有决定性影响（OOD 问题）。

---

## 七、知识连接

- **前驱**：[[AI/LLM/MA-RLHF课程/lc8-Bradley-Terry-偏好建模手撕|lc8-Bradley-Terry-偏好建模手撕]] — DPO 的数学基础
- **比较**：[[AI/LLM/MA-RLHF课程/lc8-KTO-手撕实操|lc8-KTO-手撕实操]] — 不需要成对数据的偏好学习替代方案
- **更新**：Online DPO / RLOO — 解决 OOD 问题的在线版本
- **上游**：[[lc6-SFT全链路-PyTorch手撕实操]] — SFT checkpoint 是 DPO 的起点

---

*入库时间：2026-02-26*  
*来源：MA-RLHF notebook/DPO/DPO.ipynb*  
*状态：Batch B ✅*
