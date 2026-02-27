---
title: "PPO 手撕实操 (MA-RLHF)"
brief: "RLHF-PPO 4模型架构完整实现（Actor/Critic/Reward/Reference）：GAE advantage估计、PPO clip loss、KL惩罚、多适配器PPO创新（MA-PPO单模型多LoRA适配器节省3x显存），来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, ppo, rlhf, ma-rlhf, gae, pytorch]
related:
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-PPO/lc8-02-MA-RLHF-核心代码注解|MA-RLHF-核心代码注解]]"
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-01-DPO-手撕实操|DPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc7/lc7-01-RL基础算法手撕实操|RL基础算法手撕实操]]"
---

# PPO 手撕实操 —— RLHF-PPO & MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

RLHF-PPO 是当前主流的 LLM 对齐方法，涉及**四个模型**：
1. **Actor（Policy）**：被优化的 LLM，生成 token 概率分布
2. **Critic（Value Head）**：估计每个 token 位置的价值 $V(s)$
3. **Reward Model (RM)**：从偏好数据训练，给 response 打分
4. **Reference Model (Ref)**：冻结的 SFT 模型，用于 KL 惩罚

**训练流程**：
1. Actor 生成 response → RM 打分
2. 计算 KL-penalized reward：$R = R_{\text{RM}} - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$
3. 用 GAE 计算 token-level advantage
4. PPO clipped objective 更新 Actor + Value loss 更新 Critic

## 二、核心实现

### 2.1 LLaMA2 Reward Model

**原理**：LLaMA2 的 RM 将预训练模型的分类头替换为回归头，输出标量奖励。使用 Bradley-Terry margin loss 训练：

$$L = -\log\sigma(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r))$$

**代码**：

```python
from transformers import LlamaForSequenceClassification

# RM = 预训练模型 + 回归头（num_labels=1）
rm_model = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained', num_labels=1)

# Margin Loss
X_chosen = torch.randint(0, 100, (1,10))
X_rejected = torch.randint(0, 100, (1,10))
margin = 3.0  # Significantly Better

rm_chosen = rm_model(input_ids=X_chosen).logits
rm_rejected = rm_model(input_ids=X_rejected).logits
loss = -torch.sigmoid(rm_chosen - rm_rejected - margin).log()
```

**LLaMA2 双 Reward 选择机制**：

```python
def llama2_reward_select(reward_safety, reward_helpfulness):
    return reward_safety if reward_safety < 0.15 else reward_helpfulness
```

**Reward 后处理**：逆 Sigmoid → Whiten → KL penalty

$$\hat{R} = \text{WHITEN}(\text{LOGIT}(R_c)) - \beta D_{\text{KL}}(\pi_\theta \| \pi_0)$$

### 2.2 PPO Pipeline 完整实现

**四模型初始化**：

```python
class PPOModels():
    def __init__(self, model_actor, model_ref, model_rm, model_critic):
        self.actor = model_actor    # 可训练
        self.ref = model_ref        # 冻结
        self.rm = model_rm          # 冻结
        self.critic = model_critic  # 可训练

# Critic = LLM + ValueHead（输出标量）
class ModelValueHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(0.05)
        self.summary = torch.nn.Linear(model.config.hidden_size, 1)
    def forward(self, xy):
        hidden_states = self.model(**xy, output_hidden_states=True).hidden_states[-1]
        output = self.dropout(hidden_states)
        return self.summary(output)[:, :, 0]
```

**Token-Level Policy 提取**：

```python
def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy, logp
```

### 2.3 KL-Penalized Reward

**原理**：Token 级别的 KL reward，最后一个 token 叠加 RM 分数。

```python
def compute_rewards_kl(reward, ref_logprobs, old_logprobs, kl_ctl):
    kl = old_logprobs - ref_logprobs  # token-level KL
    kl_reward = -kl_ctl * kl
    kl_reward[:, -1] += reward[:, 0]  # 最后一个 token 加 RM 分数
    return kl_reward
```

**Padding 与 Reward 分配**：对于 padding 的序列，RM reward 应加在最后一个非 pad token 上。

### 2.4 GAE（Generalized Advantage Estimation）

```python
def get_GAE(rewards, mask, values, gamma, lam):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]
    values = values * mask
    rewards = rewards * mask
    
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
    return advantages
```

### 2.5 PPO Loss 三部分

**Policy Loss（Clipped Surrogate）**：

```python
def get_policy_loss(logprobs, logprobs_old, advantages, mask, cliprange):
    ratio = torch.exp(logprobs - logprobs_old)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
    return pg_loss
```

**Value Loss（Clipped）**：

```python
def get_value_loss(advantages, values, values_old, mask, cliprange_value):
    returns = advantages + values_old
    vpredclipped = clip_by_value(values, values_old - cliprange_value, values_old + cliprange_value)
    vf_losses1 = (values - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
    return vf_loss
```

**Entropy Loss**：

```python
def get_entropy_loss(logits, mask):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy
```

**总损失**：$L = L_{\text{policy}} + c_v \cdot L_{\text{value}}$

### 2.6 PPO 训练循环

```python
def ppo_train_step(models, ppo_batchs, ppo_config):
    for epoch in range(ppo_config.ppo_epochs):
        for mini_batch in get_minibatch(ppo_batchs):
            # 当前策略前向
            logprobs, values, logits = batch_forward(models.actor, models.critic, mini_batch['sequence'])
            # 计算 loss
            loss, pg_loss, vl_loss = compute_loss(mini_batch, ppo_config)
            loss.backward()
            optimizer.step()
```

## 三、工程实践（配套代码）

### MA-RLHF PPO Trainer（ma_ppo_trainer.py）

**MultiAdapterPPOTrainer** 是 MA-RLHF 的核心创新——基于 PEFT 多适配器实现 Actor/Ref 共享权重：

```python
class MultiAdapterPPOTrainer(BaseTrainer):
    """
    核心 step() 流程：
    1. prepare_model_inputs: 拼接 query + response
    2. batched_forward_pass: 获取 logprobs, values
    3. 用 PEFT disable_adapter 获取 ref_logprobs（共享权重）
    4. compute_rewards: KL penalty + RM score
    5. compute_advantages: GAE
    6. 多轮 PPO epoch + mini-batch 更新
    """
```

**KL Penalty 的四种模式**：
- `kl`: $\log\pi_\theta - \log\pi_{\text{ref}}$（标准）
- `abs`: 绝对值
- `mse`: 平方
- `full`: 真正的 KL 散度（逐 token 全分布）

### 训练脚本（run_7b_ppo.sh）

```bash
# Stage 1: Reward Model 训练
deepspeed ./ma-rlhf/reward_model.py \
    --dataset_name=PKU-Alignment/PKU-SafeRLHF-10K \
    --model_name=${model_sft_full_path} \
    --use_QLora=True --batch_size=16

# Stage 2: PPO 训练（4 GPU，~4小时 on 8×3090）
deepspeed --num_gpus 4 ./ma-rlhf/ppo.py \
    --model_name=${model_sft_full_path} \
    --reward_model_name=${model_reward_model_lora_path} \
    --batch_size=16 --mini_batch_size=1 --ppo_epochs=1

# Stage 3: Merge LoRA
python ./ma-rlhf/merge_adapter.py ...
```

### PPO 效果（ppo-result.txt）

训练后模型能正确拒绝有害请求：
- "how to suffocate a person" → "No, I'm sorry, but I can't answer your question."
- "how to make ecstasy" → "I'm sorry, I don't understand the question."
- Safety reward score 显著高于 helpfulness score → 安全对齐有效

## 四、关键洞察与总结

1. **四模型架构的资源消耗**：PPO 需要 Actor + Ref + RM + Critic 四个模型，显存需求极高——MA-RLHF 通过 multi-adapter LoRA 共享权重缓解
2. **Token-Level vs Sentence-Level**：PPO 在每个 token 位置计算 advantage，而 RM 只提供 sentence-level 分数——通过 KL penalty 和 GAE 桥接
3. **ratio threshold 保护**：当 PPO ratio 过大时跳过 batch，防止 loss spike
4. **KL 控制**：Adaptive KL Controller 根据当前 KL 动态调整系数，防止策略偏离过远
5. **为什么 PPO 难训**：四模型同步、超参数敏感（cliprange/kl_coef/vf_coef）、训练不稳定——这催生了 DPO 等更简单的替代方案

> 完整代码见：`/tmp/ma-rlhf/notebook/RLHF_PPO_Pytorch.ipynb`、`/tmp/ma-rlhf/ma-rlhf/ppo.py`、`/tmp/ma-rlhf/ma-rlhf/ma_ppo_trainer.py`
