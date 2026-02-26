---
title: lc8 — GRPO Pytorch 从零手写
brief: 从零实现 GRPO（Group Relative Policy Optimization）完整 Pytorch 训练循环：group sampling、advantage 归一化、ratio clip、KL 约束。与 MA-RLHF Batch A 版的对比：Batch B 更侧重 group rollout 结构和优势函数计算的实现细节。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl-alignment
  - grpo
  - group-sampling
  - lc8
related:
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操]]"
  - "[[AI/LLM/RL/GRPO/GRPO-完整Notebook实现]]"
  - "[[GRPO 深度理解]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
  - "[[GRPO-Improvement-Panorama-2026]]"
---

# GRPO Pytorch 从零手写

> MA-RLHF Batch B / lc8-GRPO
> Source: `notebook/GRPO/GRPO.ipynb`
> Author: xiaodongguaAIGC / dhcode-cpp
> 评分: ★★★★★

---

## TL;DR

用 Llama 玩具模型实现 GRPO 完整 pipeline。与 PPO notebook 最大的区别：**去掉了 critic**。用 group-level reward normalization 产生 advantage，用近似 KL 替代 KL 惩罚加在 loss 中。代码逻辑：**格式化 prompt → 批量采样 → 规则 reward → Group Advantage → GRPO Loss（clip + KL）→ 训练**。

---

## GRPO vs PPO 核心差异

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic | ✅ 需要（ModelValueHead）| ❌ 去掉 |
| Advantage | GAE（γ, λ bootstrap）| Group reward normalization |
| KL 处理 | KL 作为 reward penalty（per-token）| KL 作为 loss 项（β 系数）|
| 采样模式 | 单条 rollout | Group of G 条 rollouts per prompt |
| 模型数量 | 4（actor/ref/rm/critic）| 2（actor/ref）+ rule-based reward |

**DeepSeek R1 的关键洞察**：LLM 推理训练中，rule-based verifiable reward（答案对/错）足以产生高质量 advantage 信号，critic 反而引入额外噪声和成本。GRPO 的成功证明了这一点。

---

## 格式化 Prompt 设计

```python
DEFINE_THINK_START = 25
DEFINE_THINK_END   = 26
DEFINE_ANSWER_START = 27
DEFINE_ANSWER_END   = 28

def format_prompt(question_token_ids):
    # 在 prompt 前加 <think> 示例，强制引导模型 CoT
    example = [DEFINE_THINK_START, 2,3,4, DEFINE_THINK_END,
               DEFINE_ANSWER_START, 7, DEFINE_ANSWER_END]
    # 末尾开启 <think> token，引导生成
    format_question = example + question_token_ids + [DEFINE_THINK_START]
    return format_question
```

**设计意图**：
- few-shot example 演示 `<think>...<\think><answer>...<\answer>` 格式
- 末尾 `[THINK_START]` 强制模型先进入思考模式
- 这是 o1-style CoT 的 prompt 工程基础

---

## 批量采样（Rejection Sampling）

```python
def GRPO_batch_rejection_sample(inputs, nums, max_new_tokens=10):
    grpo_xy_batch = []
    grpo_x_len = []
    for input in inputs:
        format_inputs = [format_prompt(input)] * nums    # 同一 prompt 复制 G 次
        format_input_len = len(format_inputs[0])
        grpo_x_len.append(format_input_len)
        input_x_tensor = torch.tensor(format_inputs, dtype=torch.long)
        grpo_xy = grpo_rejection_sampling(model, input_x_tensor, max_new_tokens)
        grpo_xy_batch.append(grpo_xy)   # [G, T] 的 tensor
    return grpo_xy_batch, grpo_x_len
```

**关键**：每个 prompt 采样 G=3 条 rollout（`grpo_samples_nums=3`）。同一 batch 内不同 prompt 长度不同，所以用 list of tensors，不直接 stack（生产中需要 padding + masking）。

---

## 规则 Reward 函数

```python
def rule_reward(response, label_id):
    # 检查 response 中是否有 <ANSWER_START> label <ANSWER_END>
    for i in range(len(response) - 2):
        if (response[i] == DEFINE_ANSWER_START and 
            response[i+1] == label_id and 
            response[i+2] == DEFINE_ANSWER_END):
            return True
    return False

def think_reward(response):
    # 检查是否有完整的 <think>...</think> 结构
    found_one = False
    for num in response:
        if num == DEFINE_THINK_START:
            found_one = True
        elif num == DEFINE_THINK_END:
            if found_one:
                return True
    return False
```

**两种 reward 的意义**：
- `rule_reward`：**结果 reward** — 答案是否正确（hard verifiable）
- `think_reward`：**格式 reward** — 是否有 CoT 结构（soft format check）

这是 RLVR 的典型模式：结果 reward + 格式 reward，两者分离、可组合。

---

## Group Advantage（GRPO 核心）

```python
def grpo_advantage(rewards):
    epsilon = 0.0001
    rewards = torch.tensor(rewards, dtype=torch.float)
    A = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return A
```

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

**作者观察（重要）**：
```
grpo_advantage([0,0,0,0,0,0]) → all NaN/0  # 全错：advantage=0，无梯度信号
grpo_advantage([1,0,0,0,0,0]) → [5.0, -0.45, -0.45, ...]  # 1对5错：对的 A 极大
grpo_advantage([1,0,0,0,1,0]) → [1.38, -0.55, ..., 1.38]  # 2对4错：A 中等
grpo_advantage([1,1,1,1,1,1]) → all NaN/0  # 全对：advantage=0，无梯度信号
```

**关键性质**：
1. **全对/全错 = 梯度消失**：这就是 NoRD 论文（Dr. GRPO）的出发点——中等难度样本 std 大，advantage 被归一化压小
2. **越少正例，advantage 越大**：激励模型在困难任务上突破
3. **G 越大，advantage 估计越准确**（统计意义）
4. **advantage 是 group-relative 的**：同一 prompt 的 G 条 rollout 互相比较

---

## GRPO KL（近似 KL）

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    # Schulman (2020) 近似 KL
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1
```

数学展开（Schulman 2020 KL 近似）：
$$D_{KL}[\pi_\theta \| \pi_{ref}] \approx \frac{\pi_{ref}(a)}{\pi_\theta(a)} - \log\frac{\pi_{ref}(a)}{\pi_\theta(a)} - 1$$

这是对真实 KL 的**二阶近似**：
- $D_{KL}[P\|Q] = \mathbb{E}_P[\log P/Q]$
- 当 $r = P/Q$ 时，$\log r \approx r-1$（一阶）；KL ≈ $r - \log r - 1$（二阶更精确）

**为什么用近似**：避免在计算图中显式求和，数值稳定，可以 per-token 计算并加入 loss。

---

## GRPO Loss

$$\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\min\left(r_t \hat{A}_{i,t},\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta D_{KL}[\pi_\theta\|\pi_{ref}]\right]$$

```python
def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len):
    epsilon = 0.2
    beta = 0.01
    bs, seq_len = pi_logprob.shape

    advantage = advantage.unsqueeze(dim=1)    # [G] → [G, 1]（广播到每个 token）

    # Clipped policy gradient
    ratio = torch.exp(pi_logprob - pi_old_logprob)     # r_t = π_θ / π_old
    ratio_clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)

    # KL penalty
    kl = grpo_kl(pi_logprob, pi_ref_logprob)

    # Mask: 只对 response 部分算 loss
    len_oi = seq_len - input_len
    len_oi = torch.tensor([len_oi] * bs, dtype=torch.long)
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1

    loss = (policy_gradient - beta * kl) * mask
    loss = (-1 / bs) * loss / len_oi.unsqueeze(dim=1)   # per-token 归一化
    loss = loss.sum()
    return loss
```

**关键细节**：
- `advantage.unsqueeze(1)`：同一条 rollout 的所有 token 共享同一个 advantage（trajectory-level advantage，逐 token 广播）
- `len_oi = seq_len - input_len`：只计算生成部分的 token 数
- mask prompt 部分：`mask[:, :input_len] = 0`
- KL penalty 直接加在 loss 里（不同于 PPO 中加在 reward 里）

---

## 完整训练 Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # === On-policy 采样阶段 ===
        # 1. G 次 rollout per prompt
        grpo_xy_batch, grpo_x_len = GRPO_batch_rejection_sample(input, G, max_new_tokens)
        # 2. Rule reward 打分
        batch_rewards = GRPO_batch_reward(input, grpo_xy_batch, label)
        # 3. Group advantage 归一化
        batch_advantage = [grpo_advantage(r) for r in batch_rewards]
        
        # 4. 存 old policy logprob（no_grad）和 ref logprob（no_grad）
        pi_old_logprob_list = []
        pi_ref_logprob_list = []
        for grpo_xy in grpo_xy_batch:
            with torch.no_grad():
                old_logits = model(grpo_xy).logits
                ref_logits = model_ref(grpo_xy).logits
            # gather per-token log prob
            pi_old_logprob = log_softmax + gather(grpo_xy tokens)
            pi_ref_logprob = log_softmax + gather(grpo_xy tokens)

        # === 训练阶段（inner grpo_epochs 次）===
        for k in range(grpo_epochs):
            total_loss = 0
            for pi_old, pi_ref, adv, x_len, grpo_xy in zip(...):
                # 当前 policy forward（有梯度）
                pi_grpo = model(grpo_xy).logits → log_softmax → gather
                loss = grpo_loss(pi_grpo, pi_old, pi_ref, adv, x_len)
                total_loss += loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

**Note**：`grpo_epochs=10`（inner epochs），相当于同一批 rollout 数据复用 10 次训练。对 RLVR 来说比 PPO 的 3 inner epochs 更激进，可能因为 rule reward 更稳定。

---

## Advantage 广播：trajectory-level 还是 token-level？

本 notebook 实现的是 **trajectory-level advantage**：
- G 条 rollout 各得到一个 scalar reward `r_i`
- advantage `A_i = (r_i - mean) / std`
- 每条 rollout 内的**所有 token 共享相同的 A_i**（`advantage.unsqueeze(1)` 广播）

**这是 GRPO 原始论文的做法。**

对比改进：
- **GiGPO**：anchor state grouping，找到重复经过的中间状态，产生 step-level advantage（更精细）
- **DAPO**：去掉 std 归一化（NoRD/Dr. GRPO 的核心主张），避免中等难度样本梯度被压制

---

## 与 GRPO_KL.ipynb 的关系

`GRPO_KL.ipynb`（单独文件）详细分析了 Schulman 近似 KL 的数学推导和数值行为。本 notebook 直接引用结论：

$$D_{KL}[\pi_\theta \| \pi_{ref}] \approx \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$$

完整推导见独立分析文件。

---

## 面试高频考点

**Q: GRPO 怎么去掉 critic 还能正常训练？**
A: 用 G 条 rollout 的 reward 做 within-group normalization，产生相对 advantage。本质是用 Monte Carlo 采样代替 value function 估计——G 条采样的均值就是 V(s) 的估计，reward - mean 就是 advantage 的估计。代价是需要更多采样。

**Q: GRPO 的 advantage 为什么是 trajectory-level 的？**
A: 原始 GRPO 中，reward 是 sequence-level 的（整条 response 得一个分），所以每个 token 的 advantage 相同。改进版（GiGPO）用 anchor state grouping 实现 step-level advantage，不需要额外 rollout。

**Q: GRPO 中 KL 为什么加在 loss 里而不是 reward 里？**
A: 两种方式等价（数学上都是约束 policy 偏离 ref 的惩罚），但加在 loss 里实现更简洁，不需要 mask 处理 KL 的位置，且与 clip 机制共存更自然。

**Q: GRPO 的训练数据是 on-policy 还是 off-policy？**
A: 部分 on-policy。采样时用当前模型（on-policy），但在 `grpo_epochs` 内重复使用同一批数据（off-policy 近似）。PPO-clip 控制 ratio 不超过 1±ε，防止 off-policy 偏差过大。

**Q: GRPO 与 DPO 的本质区别？**
A: DPO 离线（需要 preference pair 数据集），无 rollout；GRPO 在线（需要实时采样），有 rollout。GRPO 可以利用 verifiable reward，DPO 依赖人工标注的偏好。

**Q: 全对/全错时 advantage = 0，梯度消失怎么办？**
A: 这是 GRPO 的已知问题。常见解决方案：
1. **Dr. GRPO/NoRD**：去掉 std 归一化，改用固定 scaling
2. **课程学习**：先用中等难度样本，再逐渐加难
3. **增大 G**：更大 group size 降低全对/全错概率
4. **Online difficulty sampling**：实时估计 pass@k，过滤极端样本

---

## 关联笔记

- `AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写.md` — PPO 对比（同批次）
- `AI/LLM/RL/Other-Algorithms/GRPO-Improvement-Panorama-2026.md` — GRPO 改进全景（八维框架）
- `AI/LLM/RL/Other-Algorithms/NoRD-Dr-GRPO.md` — std 归一化问题的专题分析
- `AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization.md` — step-level advantage 改进

## See Also

- [[AI/LLM/RL/GRPO/GRPO-手撕实操]] — 同算法手撕实操版（MA-RLHF lc8 Batch A）
- [[AI/LLM/RL/GRPO/GRPO-完整Notebook实现]] — 同算法 Notebook 端到端版
- [[GRPO 深度理解]] — GRPO 理论深度解析
- [[AI/LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写]] — 同系列：DPO/IPO/BT 实现
- [[GRPO-Improvement-Panorama-2026]] — GRPO 七维改进全景
