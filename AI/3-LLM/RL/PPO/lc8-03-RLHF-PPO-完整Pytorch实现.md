---
title: RLHF-PPO 完整 PyTorch 实现
brief: RLHF-PPO 四模型架构完整实现（Actor/Critic/Reward/Reference）：PPO clip loss、GAE advantage 估计、value function 更新、KL 约束。56 cells 覆盖从 tokenize 到参数更新完整链路。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl
  - ppo
  - rlhf
  - notebook-implementation
related:
  - "[[PPO 原理]]"
  - "[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF]]"
  - "[[AI/LLM/RL/PPO/MA-RLHF-核心代码注解]]"
  - "[[AI/LLM/RL/GRPO/GRPO-完整Notebook实现]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# RLHF-PPO 完整 PyTorch 实现

> 来源：`ma-rlhf/notebook/RLHF_PPO_Pytorch.ipynb`（56 cells）
> 整理日期：2026-02-25

---

## 1. RLHF 四模型架构

RLHF-PPO 需要 **4 个模型**同时参与训练，这是与标准 RL 最大的工程区别：

```
┌─────────────────────────────────────────────────────────────┐
│                    RLHF-PPO 四模型架构                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Actor π_θ  │  Critic V_φ  │ Reference π_ref │ Reward Model │
│  (可训练)     │   (可训练)    │   (冻结)        │   (冻结)      │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ 输入: prompt │ 输入: [x,y]  │ 输入: [x,y]  │ 输入: [x,y]   │
│ 输出: y的     │ 输出: 每个    │ 输出: 每个    │ 输出: 标量     │
│ token概率分布 │ token的V值   │ token的logprob│ reward score  │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ 来源: SFT模型│ 来源: SFT +  │ 来源: SFT模型│ 来源: SFT +   │
│              │ Value Head   │ 的冻结副本    │ 分类头训练     │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### 模型关系

```
SFT Model ──┬──→ Actor (可训练，生成 response)
            ├──→ Reference (冻结，计算 KL 散度基准)
            └──→ Reward Model base (冻结，+ regression head)
                     └──→ Critic base (+ Value Head，可训练)
```

### 代码定义

```python
# --- Actor：直接用 LlamaForCausalLM，输出 token logits ---
model_actor = LlamaForCausalLM(config)

# --- Reference：SFT 模型的冻结副本，不更新参数 ---
model_ref = LlamaForCausalLM(config)
model_ref.eval()  # 冻结

# --- Reward Model：LlamaForSequenceClassification，num_labels=1 输出标量 ---
model_rm = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained', num_labels=1
)
model_rm.eval()  # 冻结

# --- Critic：基础 LM + Value Head，输出每个 token 位置的 V 值 ---
class ModelValueHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model                                    # LLM backbone
        self.dropout = torch.nn.Dropout(0.05)
        self.summary = torch.nn.Linear(model.config.hidden_size, 1)  # 线性层 → 标量

    def forward(self, xy):
        hidden_states = self.model(**xy, output_hidden_states=True).hidden_states
        last_hidden_states = hidden_states[-1]                # 取最后一层 hidden
        output = self.dropout(last_hidden_states)
        output = self.summary(output)[:, :, 0]               # [batch, seq_len]
        return output

model_critic = ModelValueHead(model_base)

# --- 四模型打包 ---
class PPOModels():
    def __init__(self, model_actor, model_ref, model_rm, model_critic):
        self.actor = model_actor
        self.ref = model_ref
        self.rm = model_rm
        self.critic = model_critic
```

---

## 2. 完整训练循环（带行级中文注释）

### 2.1 PPO 超参数配置

```python
class PPOConfig():
    def __init__(self):
        self.ppo_epochs = 3           # 每批经验重复训练几个 epoch
        self.mini_batch_size = 1      # mini-batch 大小
        self.epochs = 2               # 外层采样轮数
        self.kl_ctl = 0.01            # KL penalty 系数 β
        self.vl_coef = 0.01           # value loss 权重
        self.lam = 0.9                # GAE 的 λ
        self.gamma = 0.9              # 折扣因子 γ
        self.cliprange_value = 0.2    # value function clip 范围
        self.cliprange = 0.2          # policy ratio clip 范围 ε
```

### 2.2 核心辅助函数

```python
def logprobs_from_logits(logits, labels, gather=True):
    """从 logits 中提取指定 token 的 log probability"""
    logp = F.log_softmax(logits, dim=2)                      # [B, T, V] → log 概率分布
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)  # 取出对应 token 的 logprob
    return logpy, logp                                       # logpy: [B, T]

def get_logits(model, xy):
    """前向传播获取 logits"""
    return model(input_ids=xy).logits                        # [B, T, vocab_size]

def get_values(model, xy):
    """通过 Critic 获取每个位置的 V 值"""
    return model({'input_ids': xy})                          # [B, T]

def get_reward(rm_model, xy):
    """通过 Reward Model 获取 scalar reward"""
    return rm_model(input_ids=xy).logits                     # [B, 1]

def get_generate(model, x, max_new_tokens):
    """Actor 自回归生成 response"""
    return model.generate(input_ids=x, max_new_tokens=max_new_tokens)  # [B, T+max_new]
```

### 2.3 主训练循环：Rollout → Reward → PPO Step

```python
def ppo_train(models, ppo_config, ppo_step, x):
    """
    外层循环：采样 → 评分 → 优化
    """
    for epoch in range(ppo_config.epochs):
        # ===== Phase 1: Rollout =====
        # Actor 根据 prompt x 生成 response，拼接成 xy = [x, y]
        xy = get_generate(models.actor, x, max_new_tokens)

        # ===== Phase 2: Reward 计算 =====
        # Reward Model 对完整序列 [x, y] 打分
        rewards = get_reward(models.rm, xy)

        # ===== Phase 3: PPO Step（详见下方） =====
        loss, reward = ppo_step(models, ppo_config, x, xy, rewards)

    return loss, reward
```

### 2.4 PPO Step 详细流程

```python
def ppo_step(models, ppo_config, x, xy, rm_rewards):
    """
    完整 PPO Step：
    1. Forward 收集 experience
    2. 计算 KL-penalized reward
    3. 多 epoch mini-batch 优化
    """
    # ===== Step 1: 收集经验数据 =====
    mask = build_response_mask(x, xy)                        # 只对 response 部分计算 loss

    with torch.no_grad():
        logprobs_ref, _, _ = batch_forward(models.ref, None, xy)  # Reference 的 logprob（不可训练）

    logprobs_old, values_old, _ = batch_forward(              # Actor 当前的 logprob + Critic V值
        models.actor, models.critic, xy
    )

    # ===== Step 2: 计算 KL-penalized reward =====
    reward_kl = compute_rewards_kl(rm_rewards, logprobs_ref, logprobs_old, ppo_config.kl_ctl)

    # ===== Step 3: PPO 多 epoch 训练 =====
    ppo_batchs = {
        'sequence': xy, 'mask': mask,
        'logprobs_ref': logprobs_ref, 'logprobs_old': logprobs_old,
        'values_old': values_old, 'rewards_kl': reward_kl,
    }

    losses = []
    for _ in range(ppo_config.ppo_epochs):                   # 对同一批经验重复训练
        for mini_batch in get_minibatch(ppo_batchs):         # 拆分 mini-batch
            # 获取当前策略的 logprob 和 V值
            logprobs, values, logits = batch_forward(
                models.actor, models.critic, mini_batch['sequence']
            )
            mini_batch['logprobs'] = logprobs
            mini_batch['values'] = values
            mini_batch['logits'] = logits

            # 计算总 loss = policy loss + coef * value loss
            loss, pg_loss, vl_loss = compute_loss(mini_batch, ppo_config)
            loss.backward()                                  # 反向传播
            # optimizer.step() ...                           # 更新 Actor + Critic
            losses.append(loss.item())

    return losses
```

### 2.5 Batch Forward 函数

```python
def batch_forward(model_policy, model_value, xy):
    """同时获取 policy logprob 和 critic value"""
    logits = get_logits(model_policy, xy)                    # Actor forward → logits
    logprobs, _ = logprobs_from_logits(logits, xy, True)     # logits → 每个 token 的 logprob
    values = None
    if model_value is not None:
        values = get_values(model_value, xy)                 # Critic forward → V(s)
    return logprobs, values, logits
```

---

## 3. KL-Penalized Reward

### 公式

$$r_t = -\beta \cdot \text{KL}_t + \begin{cases} r_{\text{RM}} & \text{if } t = T \text{ (最后一个 token)} \\ 0 & \text{otherwise} \end{cases}$$

其中 token-level KL 散度：

$$\text{KL}_t = \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t})$$

等价地，最终 reward 可以写成：

$$r = r_{\text{RM}} - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

### 直觉

- **RM reward**：鼓励生成高质量 response
- **KL penalty**：惩罚偏离 SFT 模型太远，防止 reward hacking
- **β 控制平衡**：β 越大越保守，β 越小越激进

### 代码实现

```python
def compute_rewards_kl(reward, ref_logprobs, old_logprobs, kl_ctl):
    """
    计算 KL-penalized reward
    - 每个 token 位置: r_t = -β * (logprob_old - logprob_ref)
    - 最后一个 token 额外加上 RM reward
    """
    kl = old_logprobs[:, :] - ref_logprobs[:, :]             # token-level KL 近似
    kl_reward = -kl_ctl * kl                                 # 每个位置的 KL penalty
    kl_reward[:, -1] += reward[:, 0]                         # 末尾位置叠加 RM reward
    return kl_reward                                         # [B, T]，每个 token 的总 reward
```

### Reward 分配示意（含 padding）

```
pad_token_id = 1

x1: 3, 4, 6, 8, 20, 29, 30, 2         ← 无 padding
x2: 8, 2, 6, 9, 13, 2,  1,  1         ← 后两位是 pad

x1 KL+reward: kl, kl, kl, kl, kl, kl, kl, [kl + r_RM]
x2 KL+reward: kl, kl, kl, kl, kl, [kl + r_RM], 0, 0    ← pad 位置置零
```

---

## 4. GAE（Generalized Advantage Estimation）

### 推导

GAE 是对 advantage 的指数加权移动平均，在 bias 和 variance 之间取平衡：

**TD 残差（单步 advantage 估计）**：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**GAE 递推公式**：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

等价递推形式（从后往前算）：

$$\hat{A}_T = \delta_T$$
$$\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}$$

**特殊情况**：
- λ=0 → 单步 TD：$\hat{A}_t = \delta_t$（低方差高偏差）
- λ=1 → Monte Carlo：$\hat{A}_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t)$（高方差低偏差）
- 0 < λ < 1 → 折中

### 代码实现

```python
def get_GAE(rewards, mask, values, gamma, lam):
    """
    从后往前递推计算 GAE
    rewards: [B, T]  KL-penalized reward
    mask:    [B, T]  response mask（prompt 位置为 0）
    values:  [B, T]  Critic 预测的 V 值
    gamma:   折扣因子
    lam:     GAE 的 λ 参数
    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]                              # 序列长度 T

    values = values * mask                                   # mask 掉 prompt 部分
    rewards = rewards * mask

    for t in reversed(range(gen_len)):                       # 从最后一个 token 倒推
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0  # V(s_{t+1})
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]   # TD 残差 δ_t
        lastgaelam = delta + gamma * lam * lastgaelam               # GAE 递推
        advantages_reversed.append(lastgaelam)

    # 翻转回正序
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
    return advantages                                        # [B, T]
```

---

## 5. PPO Loss 三部分

### 5.1 Policy Loss（Clipped Surrogate Objective）

$$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}} \hat{A}, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}\right)\right]$$

```python
def get_policy_loss(logprobs, logprobs_old, advantages, mask, cliprange):
    """
    PPO Clipped Policy Loss
    """
    ratio = torch.exp(logprobs - logprobs_old)               # π_θ / π_θ_old（概率比）

    pg_losses = -advantages * ratio                          # 无 clip 的 surrogate loss
    pg_losses2 = -advantages * torch.clamp(                  # clip 后的 surrogate loss
        ratio, 1.0 - cliprange, 1.0 + cliprange
    )

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)  # 取较大值（更悲观的估计）
    return pg_loss
```

### 5.2 Value Loss（Clipped Value Function）

$$L^{V} = \frac{1}{2}\mathbb{E}\left[\max\left((V_\theta - R)^2, (V_{\text{clip}} - R)^2\right)\right]$$

其中 $R = \hat{A} + V_{\text{old}}$（GAE returns），$V_{\text{clip}} = \text{clip}(V_\theta, V_{\text{old}} \pm \epsilon_v)$

```python
def get_value_loss(advantages, values, values_old, mask, cliprange_value):
    """
    Clipped Value Loss
    """
    returns = advantages + values_old                        # GAE returns = A + V_old
    advantages = advantages.detach()                         # advantage 不参与 value 梯度

    vpredclipped = clip_by_value(                            # clip V 值防止更新过大
        values,
        values_old - cliprange_value,
        values_old + cliprange_value,
    )

    vf_losses1 = (values - returns) ** 2                     # 无 clip 的 MSE
    vf_losses2 = (vpredclipped - returns) ** 2               # clip 后的 MSE
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
    return vf_loss
```

### 5.3 Entropy Bonus

$$L^{\text{entropy}} = -\mathbb{E}\left[H(\pi_\theta)\right] = -\mathbb{E}\left[-\sum_a \pi_\theta(a) \log \pi_\theta(a)\right]$$

```python
def get_entropy_loss(logits, mask):
    """
    计算策略的 entropy（鼓励探索）
    """
    pd = F.softmax(logits, dim=-1)                           # 概率分布
    entropy = torch.logsumexp(logits, axis=-1) - \
        torch.sum(pd * logits, axis=-1)                      # H = logsumexp - E[logits]
    return entropy                                           # 可加负号作为 loss（鼓励高 entropy）
```

### 5.4 总 Loss

```python
def compute_loss(mini_batchs, ppo_config):
    # Step 1: 计算 GAE
    GAE = get_GAE(mini_batchs['rewards_kl'], mini_batchs['mask'],
                  mini_batchs['values'], ppo_config.gamma, ppo_config.lam)

    # Step 2: Value Loss
    vl_loss = get_value_loss(GAE, mini_batchs['values'], mini_batchs['values_old'],
                             mini_batchs['mask'], ppo_config.cliprange_value)

    # Step 3: Policy Loss
    pg_loss = get_policy_loss(mini_batchs['logprobs'], mini_batchs['logprobs_old'],
                              GAE, mini_batchs['mask'], ppo_config.cliprange)

    # Step 4: 总 loss = policy loss + 系数 × value loss
    # 注：原始 notebook 未显式加 entropy bonus，实际工程中会加
    loss = pg_loss + ppo_config.vl_coef * vl_loss
    return loss, pg_loss, vl_loss
```

---

## 6. 与标准 PPO 的区别（LLM Token-Level 的特殊性）

| 维度 | 标准 PPO（游戏/机器人） | RLHF-PPO（LLM） |
|------|------------------------|-----------------|
| **状态空间** | 连续/离散低维状态 | token 序列（高维离散） |
| **动作空间** | 几个到几十个 | vocab_size（32000+） |
| **时间步** | 环境的物理帧 | 每个生成的 token |
| **Reward 来源** | 环境自然给出 | Reward Model 打分（仅在序列末尾） |
| **KL 约束** | 无（或 trust region） | 必须有 KL penalty 防止偏离 SFT |
| **Reference Policy** | 无 | 冻结的 SFT 模型 |
| **模型规模** | 小网络（MLP） | 数十亿参数的 Transformer |
| **Reward 密度** | 每步有 reward | 稀疏（只有 EOS 有 RM reward，靠 KL 填充中间） |
| **Critic** | 独立小网络 | 与 Actor 共享 backbone + Value Head |
| **训练稳定性** | 相对稳定 | 极不稳定，需要大量 tricks |

### 关键区别详解

1. **Token-level MDP**：在 RLHF 中，每生成一个 token 都是一个 action。状态 = prompt + 已生成的 tokens，动作 = 下一个 token。这导致 horizon 等于 response 长度。

2. **稀疏 reward + KL 填充**：RM 只在序列末尾给一个标量 reward，中间 token 位置的 reward 全靠 KL penalty 提供信号。这使得 credit assignment 非常困难，GAE 的作用尤为关键。

3. **必须有 Reference Model**：标准 PPO 的 trust region 靠 clip 实现。RLHF 除了 clip，还需要 KL penalty 来约束策略不偏离 SFT 模型——否则模型会 "reward hack"（生成得分高但质量差的 response）。

4. **Mask 机制**：prompt 部分不计算 loss，只对 response 部分（mask=1）计算 policy loss、value loss 和 GAE。

---

## 7. 面试考点

### Q1: RLHF 中为什么需要 4 个模型？能否减少？

**参考答案**：

4 个模型各有不可替代的作用：
- **Actor**：生成 response 的策略模型，是唯一被优化的生成模型
- **Critic**：估计状态价值 V(s)，用于计算 advantage。不能省略，否则只能用 Monte Carlo 估计（方差极大）
- **Reference**：冻结的 SFT 模型，提供 KL penalty 的基准。省略会导致 reward hacking
- **Reward Model**：提供训练信号，替代人工标注

**能否减少？**
- Actor 和 Critic 可以共享 backbone（节省显存），只需不同的 head
- Reference 和 Actor 可以用 LoRA——Reference 是基座，Actor 是基座 + LoRA adapter
- DPO 方法彻底去掉了 Critic 和 RM，只需 Actor + Reference 两个模型

### Q2: KL penalty 的 β 如何选择？太大/太小会怎样？

**参考答案**：

- **β 太大**：策略被锁死在 SFT 附近，几乎无法学习新偏好。表现为 reward 不上升，KL 接近 0。
- **β 太小**：策略偏离 SFT 太远，发生 reward hacking（模型学会 exploit RM 的弱点，生成高分但低质的文本）。
- **自适应 β**：实践中常用 adaptive KL controller——设定 KL 目标值 $\text{KL}_{\text{target}}$，如果实际 KL > target 就增大 β，反之减小。
- **典型值**：β ∈ [0.01, 0.2]，KL target 通常 6.0 左右（InstructGPT）。

### Q3: GAE 中的 λ 和 γ 分别控制什么？RLHF 中如何设置？

**参考答案**：

- **γ（折扣因子）**：控制未来 reward 的权重。γ=1 完全不折扣（平等看待未来），γ=0 只看当前。
- **λ（GAE 参数）**：控制 bias-variance 平衡。λ=0 纯 TD（低方差高偏差），λ=1 纯 MC（高方差低偏差）。
- **RLHF 常用设置**：γ=1.0, λ=0.95。因为 LLM 的 response 长度有限（几百 token），不需要强折扣；λ 接近 1 因为稀疏 reward 需要长距离传播。
- **Notebook 中用的是** γ=0.9, λ=0.9，在玩具模型上这样设置是合理的。

### Q4: PPO 的 clip 机制为什么用 max(L1, L2) 而不是 min？

**参考答案**：

这是一个容易混淆的细节。原始 PPO 论文用 `min`，但那是因为 objective 前面有负号：

```
L^CLIP = E[min(r_t * A_t, clip(r_t) * A_t)]   ← 论文写法（最大化目标）
loss  = -L^CLIP                                 ← 实际代码（最小化 loss）
```

在这个 notebook 的实现中，loss 定义时直接用了 `-A * ratio`（前面带负号），所以用 `max` 来选择更悲观的估计，效果等价：

```python
pg_losses  = -advantages * ratio
pg_losses2 = -advantages * clamp(ratio, 1-ε, 1+ε)
pg_loss = max(pg_losses, pg_losses2)  # 选择更大的 loss（更悲观）
```

本质上都是在限制策略更新步长：当 ratio 偏离 1 太远时，使用 clip 版本截断梯度。

### Q5: RLHF-PPO 的 Reward 为什么只在最后一个 token 位置给，而不是每个 token 都给？

**参考答案**：

1. **RM 的训练方式决定了输出粒度**：Reward Model 是对完整 response 打分（sequence classification），它看到整个 `[prompt, response]` 后输出一个标量。它没有能力也没有训练数据来对每个 token 打分。

2. **Token-level reward 的构造**：虽然 RM 只给末尾一个分数，但通过 KL penalty 每个 token 位置都有信号：
   ```
   r_t = -β * KL_t           (t < T，中间 token)
   r_T = -β * KL_T + r_RM    (t = T，最后一个 token)
   ```
   这样 GAE 就能从末尾的 RM reward 逐步反向传播到每个 token。

3. **为什么不训练 token-level RM**：人工标注时，标注者是对完整 response 做偏好排序，而不是对每个 token 标注好坏。获取 token-level 标注在实践中几乎不可能。

4. **Process Reward Model（PRM）** 是一种尝试给中间步骤打分的方法（如 OpenAI 的 Let's Verify Step by Step），但目前主要用于数学推理场景，不是通用方案。
