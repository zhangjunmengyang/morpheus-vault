---
title: lc8 — RLHF-PPO Pytorch 从零手写
brief: 从零实现 RLHF 四模型（Actor/Critic/RM/Ref）PPO 训练完整流程：rollout 生成、reward 计算、advantage 估计（GAE）、PPO clip 更新、KL 惩罚。Batch B 相比 Batch A 更注重四模型 GPU 调度和显存管理的工程细节。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl-alignment
  - ppo
  - rlhf
  - four-model
  - gae
  - lc8
related:
  - "[[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF]]"
  - "[[AI/3-LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-LLaMA2-Reward-Model-从零手写]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
  - "[[PPO 原理]]"
---

# RLHF-PPO Pytorch 从零手写

> MA-RLHF Batch B / lc8-RLHF-PPO
> Source: `notebook/RLHF_PPO_Pytorch.ipynb`
> Author: xiaodongguaAIGC / dhcode-cpp
> 评分: ★★★★★

---

## TL;DR

用 Llama 玩具模型（vocab=32, hidden=256, 2层）实现 RLHF-PPO 完整 pipeline。代码极简但结构完整：**四模型架构 → Experience Collection → GAE → Critic Loss（clipped）→ Policy Loss（clipped）→ Entropy → 合并训练**。是理解 PPO-from-scratch 最清晰的范例之一。

---

## 四模型架构

```
SFT model
    ├── model_ref      (frozen, 提供 KL 基准)
    └── model_actor    (训练中, policy)

Reward Model
    └── model_critic   (从 RM 初始化, 多加 ValueHead, 估计状态价值)
```

```python
class PPOModels():
    def __init__(self, model_actor, model_ref, model_rm, model_critic):
        self.actor = model_actor
        self.ref = model_ref    # frozen eval mode
        self.rm = model_rm      # frozen eval mode
        self.critic = model_critic  # trainable ValueHead
```

**关键关系**：
- `model_ref` ← SFT checkpoint（不训练，用于 KL 约束）
- `model_rm` ← SFT/RM checkpoint（不训练，打分器）
- `model_actor` ← SFT checkpoint（训练目标）
- `model_critic` ← RM checkpoint + `ModelValueHead`（训练目标，估 V(s)）

---

## Critic 设计：ModelValueHead

```python
class ModelValueHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.05)
        self.summary = torch.nn.Linear(model.config.hidden_size, 1)

    def forward(self, xy):
        hidden_states = model(**xy, output_hidden_states=True).hidden_states
        last_hidden_states = hidden_states[-1]    # [B, T, H]
        output = self.dropout(last_hidden_states)
        output = self.summary(output)[:, :, 0]   # [B, T]
        return output
```

每个 token 位置都输出一个 scalar value，因此 `values.shape == [B, T]`。这不同于 DPO 等只看最后一个 token 的 reward model。

---

## PPO Pipeline 完整流程

### Phase 1: Experience Collection（每轮 outer epoch）

```
x (prompt)
   ↓ actor.generate()
xy (prompt + response)
   ↓ rm.forward()
rm_scores [B, 1]
   ↓ compute_rewards_kl()
rewards_kl [B, T]   ← KL penalty 加在每个 token，RM score 加在最后一个 token
```

```
batch_forward(ref, None, xy)   → logprobs_ref [B, T]
batch_forward(actor, critic, xy) → logprobs_old [B, T], values_old [B, T]
```

**注意**：Experience collection 阶段的 logprobs/values 存为 `old`，训练阶段的叫 `logprobs/values`（新的 forward pass），两者差异产生 clipping。

### Phase 2: KL Reward 设计

```python
def compute_rewards_kl(reward, ref_logprobs, old_logprobs, kl_ctl):
    kl = logprobs_old[:, :] - logprobs_ref[:, :]   # per-token KL (approximate)
    kl_reward = -kl_ctl * kl                        # KL penalty per token
    kl_reward[:, -1] += reward[:, 0]                # RM score 加在最后一个 token
    return kl_reward
```

**关键设计**：
- per-token KL 惩罚：每步偏离 ref 都有代价
- RM 分数只加在最后一个生成 token（sequence level → token level 的桥接）
- `kl_ctl=0.01` 控制 KL 强度，是最重要的超参之一

**Mask 设计（作者注释）**：
```
x1 : prompt tokens    → mask=0（不参与 loss）
x1 : response tokens  → mask=1
pad tokens            → mask=0（变长 response 的 padding）

KL+rewards 叠加位置：
  x1: kl[0..T-2], kl[-1] + rm_score  (最后一个真实 token 处)
  x2: 同上，但 pad 处 mask=0 所以不影响
```

### Phase 3: PPO 训练 Loop

```python
def ppo_train_step(models, ppo_batchs, ppo_config, compute_loss):
    for epoch in range(ppo_config.ppo_epochs):   # inner PPO epochs (e.g. 3)
        minibatches = get_minibatch(ppo_batchs, batch_size, mini_batch_size)
        for mini_batch in minibatches:
            # 重新 forward（得到当前 policy 的 logprobs/values）
            logprobs, values, logits = batch_forward(
                actor, critic, mini_batch['sequence'])
            # compute PPO loss
            loss, pg_loss, vl_loss = compute_loss(models, mini_batch, ppo_config)
            loss.backward()
```

**inner PPO epochs 的意义**：同一批 experience 数据重复训练多次，提高样本效率（类比 minibatch SGD）。但更新次数多了会导致 policy 偏离太远 → 这就是为什么需要 clipping。

---

## GAE（Generalized Advantage Estimation）

```python
def get_GAE(rewards, mask, values, gamma, lam):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]

    values = values * mask     # mask out padding
    rewards = rewards * mask

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t+1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
    return advantages
```

**GAE 推导**：
$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

递推公式（从后往前）：
$$A_t = \delta_t + \gamma\lambda A_{t+1}$$

**超参含义**：
- `gamma=0.9`：折扣因子，控制未来 reward 的衰减速度
- `lam=0.9`：GAE λ，控制 bias-variance tradeoff（λ→0: 高 bias 低 variance；λ→1: 低 bias 高 variance）

**观察实验**（作者提供）：
```python
# 只在最后 token 有 reward=10，其他 token reward=0，value=0
# GAE 输出：从最后往前逐步衰减（gamma*lam 折扣）
# value=1 时：GAE 折扣更快（critic 已"预期"了部分价值）
```

---

## Critic Loss（Clipped Value Loss）

```python
def get_value_loss(advantages, values, values_old, mask, cliprange_value):
    returns = advantages + values_old           # target return = GAE advantage + old value
    advantages = advantages.detach()            # stop gradient from GAE through values

    # Clipped value prediction（防止 value 函数更新太剧烈）
    vpredclipped = clip_by_value(
        values,
        values_old - cliprange_value,
        values_old + cliprange_value,
    )

    vf_losses1 = (values - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
    return vf_loss
```

**注意**：`returns = advantages + values_old`，这是 GAE 的经典 target 公式：$V_{target} = A_t + V_{old}(s_t)$。

**Clipped Value Loss 的意义**：和 policy clipping 类比，防止 critic 更新步长过大。取 `max(loss1, loss2)` 等价于"如果预测值偏离太远，用 clip 版本的损失"。

---

## Policy Loss（PPO-Clip）

```python
def get_policy_loss(logprobs, logprobs_old, advantages, mask, cliprange):
    ratio = torch.exp(logprobs - logprobs_old)    # 重要性权重 r_t(θ)

    pg_losses  = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
    return pg_loss
```

**PPO-Clip 数学**：
$$L^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{old}(a_t|s_t))$

**直觉**：
- $A_t > 0$（好动作）：鼓励 $r_t$ 变大，但 clip 在 $1+\epsilon$（不能过度强化）
- $A_t < 0$（坏动作）：鼓励 $r_t$ 变小，但 clip 在 $1-\epsilon$（不能过度惩罚）
- `torch.max` 选取"保守"的那个梯度方向

---

## Entropy Loss

```python
def get_entropy_loss(logits, mask):
    pd = F.softmax(logits, dim=-1)
    # log_sum_exp(x) = log Z，softmax 的分母
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy
```

**数学推导**：
$$H(\pi) = -\sum_a \pi(a) \log \pi(a) = \log Z - \sum_a \pi(a) x_a = \text{logsumexp}(x) - \mathbb{E}[x]$$

这是一个数值稳定的 entropy 计算方式（不需要先 log_softmax 再 entropy）。熵损失通常以负号加入总 loss（鼓励探索）。

---

## Final Loss 汇总

```python
def compute_loss(mini_batchs, ppo_config):
    GAE = get_GAE(...)
    vl_loss = get_value_loss(GAE, ...)
    pg_loss = get_policy_loss(..., GAE, ...)
    loss = pg_loss + ppo_config.vl_coef * vl_loss    # vl_coef=0.01
    return loss, pg_loss, vl_loss
```

$$\mathcal{L}_{PPO} = L^{CLIP} + c_1 L^{VF} + c_2 H$$

这里实现了前两项（entropy 未显式加入总 loss，单独计算供监控）。

---

## 端到端 Pipeline 伪代码

```python
# Outer Loop
for epoch in range(epochs):
    # 1. Generate with actor
    xy = actor.generate(x)
    # 2. Score with RM
    rm_scores = rm.forward(xy)
    # 3. Collect experience
    logprobs_ref = ref.forward(xy)          # no_grad
    logprobs_old, values_old = actor+critic.forward(xy)
    rewards_kl = compute_rewards_kl(rm_scores, logprobs_ref, logprobs_old, kl_ctl)

    # Inner PPO Loop
    for ppo_epoch in range(ppo_epochs):
        for mini_batch in minibatches(experience):
            # 4. Current policy forward
            logprobs, values = actor+critic.forward(mini_batch.sequence)
            # 5. Compute losses
            GAE = get_GAE(rewards_kl, mask, values, gamma, lam)
            vl_loss = get_value_loss(GAE, values, values_old, ...)
            pg_loss = get_policy_loss(logprobs, logprobs_old, GAE, ...)
            loss = pg_loss + vl_coef * vl_loss
            # 6. Update
            optimizer.step()
```

---

## 与生产 RLHF 的差异

| 维度 | 本 notebook | 生产 RLHF（TRL/OpenRLHF/verl） |
|------|-------------|-------------------------------|
| 模型规模 | vocab=32，2层 | 7B-70B+ |
| 分布式 | 无 | FSDP/Megatron+vLLM |
| 生成 | model.generate() | vLLM/SGLang 异步生成 |
| KL 实现 | per-token 近似 KL | 精确 KL 或 token-level |
| Reward | 随机化 RM | 真实 RM 或 verifiable reward |
| 异步 | 同步 rollout+train | 异步（VerlTool/slime） |
| 长度 | fixed `max_new_tokens` | 变长 padding + 动态 batch |

---

## 面试高频考点

**Q: PPO 为什么需要四个模型？**
A: actor（被训练的 policy）、ref（KL 约束基准）、rm（提供外部 reward signal）、critic（估计 state value，计算 advantage）。少一个都会破坏某个核心机制。

**Q: GAE 的 lambda 参数控制什么？**
A: bias-variance tradeoff。λ=0 等价于单步 TD，低 variance 高 bias；λ=1 等价于 MC return，低 bias 高 variance。PPO 通常用 λ=0.95-0.99。

**Q: RM reward 为什么只加在最后一个 token？**
A: RM 给的是 sequence-level reward（对整个 response 打分），而 PPO 需要 token-level reward。惯用做法是把 scalar reward 加在 EOS/最后 token 位置，其他位置只有 KL penalty（已经是 token-level 的）。

**Q: Policy loss 和 Critic loss 为什么都有 clipping？**
A: 两者都是为了防止更新步长过大破坏稳定性。Policy clip 通过限制 importance ratio；Critic clip 通过限制 value 函数偏离 old value 的范围。

**Q: `advantages.detach()` 在 Critic loss 里的作用？**
A: GAE 计算用到了 values（以估计 bootstrap target），但 critic loss 的梯度不应该通过 GAE 路径流回来（那是 actor 的事）。detach 确保梯度只来自直接的 `(values - returns)^2` 路径。

**Q: PPO vs GRPO：Critic 的优劣？**
A: Critic 优势：dense reward 场景下 GAE 比 MC 更稳定；能正确处理时序折扣（γ）。Critic 劣势：需要额外训练一个模型，与 Actor 训练耦合；对 sparse/verifiable reward 场景效果不好（reward 几乎全在最后，Critic 难以学习中间状态）。GRPO 在 reasoning 任务（sparse verifiable reward）上去掉 Critic 是合理的简化。

---

## 代码陷阱

1. **mask 设计**：prompt 部分 mask=0，生成部分 mask=1。`masked_mean` 确保只在有效 token 上算均值——如果忘了 mask，pad token 会污染 loss。

2. **compute_rewards_kl 里有个 BUG**（原始代码）：函数参数命名混乱，内部使用了外部作用域的 `logprobs_old` 而不是参数 `old_logprobs`。生产代码需要修正。

3. **values 的 forward 两次**：experience collection 阶段一次（存为 `values_old`），训练阶段重新 forward 一次（存为 `values`）。两次都必须做，不能直接用 old values 训练 critic（那样梯度无法流动）。

4. **inner PPO epoch 数不能太多**：此 notebook 用了 3 epochs。每次 inner epoch 后 policy 偏离，importance ratio 偏离 1，clip 效果降低。生产中通常配合 early stopping（KL 超阈值时停止）。

---

## 关联笔记

- `AI/LLM/RL/RLHF/` — RLHF 理论体系
- `AI/LLM/MA-RLHF课程/lc8-GRPO-notebook.md` — 同批次对比
- `AI/LLM/RL/Other-Algorithms/GRPO-Improvement-Panorama-2026.md` — GRPO vs PPO 全景
- `AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写.md` — 分布式训练配套
- `AI/LLM/MA-RLHF课程/lc10-推理系统-MOC.md` — 课程体系全貌

## See Also

- [[AI/3-LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]] — 同内容 Batch A 版（Notebook 端到端）
- [[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF]] — 同算法手撕实操版
- [[AI/3-LLM/MA-RLHF课程/lc8-LLaMA2-Reward-Model-从零手写]] — 同系列：RM 实现（PPO 前置）
- [[AI/3-LLM/MA-RLHF课程/lc8-GRPO-notebook-Pytorch从零手写]] — 同系列：GRPO（无 Critic 的轻量版）
- [[PPO 原理]] — PPO 算法理论深度解析
