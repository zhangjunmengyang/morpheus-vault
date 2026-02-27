---
title: "LLaMA2 Reward Model 手撕（MA-RLHF lc8）"
brief: "LLaMA2 Reward Model 从零手撕：在 LLaMA2 backbone 上加线性层输出 scalar reward；BT 模型 loss（-log σ(r_chosen - r_rejected)）训练；与 PPO Actor-Critic 中 Critic head 的区别；这是 RLHF pipeline 的关键中间件，面试必考。MA-RLHF lc8 实操笔记。"
type: code-practice
date: 2026-02-26
source: "MA-RLHF notebook/reward/LLaMA2-reward.ipynb"
tags:
  - RewardModel
  - RLHF
  - LLaMA2
  - BradleyTerry
  - 手撕实操
  - MA-RLHF-lc8
related:
  - [[Projects/MA-RLHF/lc8-DPO/lc8-05-DPO-IPO-手撕实操|lc8-DPO-IPO-手撕实操]]
  - [[AI/3-LLM/MA-RLHF课程/lc8-KTO-手撕实操|lc8-KTO-手撕实操]]
  - [[AI/3-LLM/RL/实践/RLHF-工程全栈|RLHF-DPO-2026-技术全景]]
  - [[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]]
---

# LLaMA2 Reward Model 手撕（MA-RLHF lc8）

**来源**：MA-RLHF `notebook/reward/LLaMA2-reward.ipynb`  
**参考**：LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al. 2023)  
**难度**：★★★★☆（工业级 RM 设计，理解 RLHF 生产细节）  
**关联**：[[Projects/MA-RLHF/lc8-PPO/lc8-01-PPO-手撕实操|lc8-RLHF-PPO-手撕实操]] | [[Projects/MA-RLHF/lc8-DPO/lc8-02-Bradley-Terry模型实现|lc8-Bradley-Terry-偏好建模手撕]]

---

## 一、LLaMA2 RM 架构：从 LM Head 到 Regression Head

```python
# 基础：LLaMA 预训练模型（LM Head = next-token 预测）
model = LlamaForCausalLM(config)
model.save_pretrained('./lm_pretrained')

# RM：把 LM Head（词汇表大小 output）替换成 Regression Head（scalar output）
rm_model = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained',
    num_labels=1     # ← scalar reward，不是分类
)
```

**论文原文**：
> The model architecture and hyper-parameters are identical to those of the pretrained language models, **except that the classification head for next-token prediction is replaced with a regression head for outputting a scalar reward**.

**工程含义**：RM 复用了 LLM 的所有权重（embedding + transformer layers），只替换最后一层。这保留了语言理解能力，同时把输出从 vocab_size 维压缩为 1 维 scalar。

---

## 二、带 Margin 的 RM Loss

$$\mathcal{L} = -\log\sigma(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r))$$

其中 $m(r)$ 是 **margin**，是标注者置信度的离散函数：
- 标注者认为 chosen 明显好于 rejected（高置信）→ margin 大
- 标注者不确定 → margin 小或为 0

```python
margin = 3.0  # 高置信度的标注对，要求 rm_chosen - rm_rejected 至少大 3

rm_chosen   = rm_model(input_ids=X_chosen).logits   # scalar
rm_rejected = rm_model(input_ids=X_rejected).logits  # scalar

loss             = -torch.sigmoid(rm_chosen - rm_rejected).log()
loss_with_margin = -torch.sigmoid(rm_chosen - rm_rejected - margin).log()
```

**Margin 的作用**：
- 无 margin 的 BT loss：`rm_chosen - rm_rejected > 0` 即为正确
- 有 margin：`rm_chosen - rm_rejected > margin` 才算足够好
- 对高置信度样本施加更强的"间距"要求，迫使 RM 学到更有区分度的 reward

**类比 SVM**：BT loss without margin = logistic loss，BT loss with margin = margin-based logistic loss（类似 SVM 的 hinge loss 思想）。

---

## 三、LLaMA2 双 RM 架构：Safety + Helpfulness

$$R_c(g|p) = \begin{cases} R_s(g|p) & \text{if is\_safety}(p) \text{ or } R_s(g|p) < 0.15 \\ R_h(g|p) & \text{otherwise} \end{cases}$$

```python
def llama2_reward_select(reward_safety, reward_helpfulness):
    # 安全优先：如果安全分低（<0.15），用安全 RM 的分
    # 否则：用 helpfulness RM 的分
    return reward_safety if reward_safety < 0.15 else reward_helpfulness

# 示例
rc = llama2_reward_select(reward_safety=-0.3, reward_helpfulness=0.7)
print(rc)   # -0.3 (安全分过低，强制用安全 RM)

rc = llama2_reward_select(reward_safety=1.3, reward_helpfulness=0.4)
print(rc)   # 0.4 (安全过关，用 helpfulness 分)
```

**设计哲学**：
- 两个独立 RM：一个专注安全（`R_s`），一个专注有用性（`R_h`）
- 安全第一：`R_s < 0.15` = 回答有安全风险，无论多 helpful 都用安全分惩罚
- 正常情况：用 helpfulness 分优化对话质量
- **这是 Constitutional AI / RLHF safety 的工程实现**：不是"把安全和有用混合"，而是"安全 gate + helpfulness optimize"

---

## 四、Reward 白化（Whitening）

论文原文：
> We also find it important to **whiten** the final linear scores (shown here by reversing the sigmoid with the **logit function**) in order to increase stability and balance properly with the KL penalty term (β).

$$\hat{R} = \text{WHITEN}(\text{LOGIT}(R_c(g|p)))$$

**两步操作**：

**Step 1: Logit（逆 Sigmoid）**
```python
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

inverse_sigmoid(0.9)  # → 2.197  (把概率空间映射回 logit 空间)
inverse_sigmoid(0.5)  # → 0.0
inverse_sigmoid(0.01) # → -4.595
```

**Step 2: Whitening（零均值、单位方差）**
```python
def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)  # rsqrt = 1/sqrt
    if not shift_mean:
        whitened += mean  # 只标准化方差，保留均值（某些场景用）
    return whitened

# 效果：不同量级的 reward 经过 whiten 后可以直接与 KL 惩罚相加
# [0.83, 1.20, 3.30, 4.60] → [-1.27, -0.97, 0.64, 1.60]
# [100.83, 101.20, 103.30, 104.60] → 同样的结果（量级无关）
```

**为什么需要 Whitening？**
- RM 的 scalar reward 量级不稳定（可能是 0~10，也可能是 -100~100）
- KL penalty 的量级固定（β * KL，通常 0.01~1.0）
- 如果 reward >> KL，KL 约束失效；如果 reward << KL，policy 不优化
- Whitening 把 reward 标准化到固定量级，保证两者平衡

---

## 五、完整 Reward 计算（LLaMA2 风格）

$$R(g|p) = \hat{R}_c(g|p) - \beta \cdot \text{KL}(\pi_\theta(g|p) \| \pi_0(g|p))$$

```python
# Step 1: RM 打分（safety/helpfulness 二选一）
rm_score = rm_model(X).logits.item()   # scalar

# Step 2: 白化（logit + whiten）
rm_logit = inverse_sigmoid(torch.tensor(rm_score))
rm_whitened = whiten(rm_logit)

# Step 3: KL 惩罚
output_new = model(X).logits[:,-1,:].sigmoid()   # 新 policy 的 prob
prob = torch.gather(output_new, dim=1, index=next_token_id)
kl = F.kl_div(torch.log(prob), prob_old)

# Step 4: 最终 reward
final_reward = rm_whitened - beta * kl
```

---

## 六、LLaMA2 RM vs 标准 RLHF RM 的差异

| 特性 | 标准 RLHF RM | LLaMA2 RM |
|------|-------------|-----------|
| 模型数 | 1 个 RM | **2 个**（safety + helpfulness）|
| Loss | BT（无 margin）| BT + **confidence margin** |
| Reward 处理 | 直接用 scalar | **logit + whiten** 后用 |
| 安全机制 | 混合打分 | **独立 safety gate**（0.15 阈值）|
| 理论基础 | Bradley-Terry | Bradley-Terry + SVM margin |

---

## 七、面试必备问题

**Q1：LLaMA2 为什么要训两个 Reward Model？**  
A：Safety 和 Helpfulness 存在 tension（有用但不安全 vs 安全但无用）。混合一个 RM 难以精确控制两者的权衡。用 safety gate（阈值 0.15）硬性隔离安全问题，用 helpfulness RM 在安全范围内优化。

**Q2：Reward Whitening 的作用？**  
A：标准化 RM 输出到固定量级（零均值单位方差），保证 reward 和 KL penalty 在 PPO 训练中量级匹配。先 logit（映射到无界实数域），再 whiten（标准化）。

**Q3：Margin 在 RM Loss 中的作用？**  
A：对高置信度的偏好标注施加更强约束，要求 chosen - rejected > margin（而非 > 0）。迫使 RM 在确定的偏好上学到更大的间距，提高 reward 区分度。

---

*入库时间：2026-02-26*  
*来源：MA-RLHF notebook/reward/LLaMA2-reward.ipynb*  
*状态：Batch B ✅*
