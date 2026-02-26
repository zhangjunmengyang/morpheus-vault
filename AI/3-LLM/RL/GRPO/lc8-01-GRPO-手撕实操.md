---
title: GRPO 手撕实操
brief: GRPO（Group Relative Policy Optimization）完整PyTorch实现：loss函数（clip版/简化版）、Group Relative Advantage计算、KL惩罚、与PPO关键差异（无Critic），含完整公式LaTeX推导，来源 MA-RLHF 教学项目。
date: 2026-02-25
type: code-practice
source: MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
tags:
  - code-practice
  - grpo
  - rl
  - policy-optimization
  - pytorch
related:
  - "[[GRPO 深度理解|GRPO深度理解]]"
  - "[[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/3-LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]]"
  - "[[OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]]"
---

# GRPO 手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 采用的核心训练算法，**去掉了 Critic 模型**，用同一 prompt 的多次采样构建 group-relative advantage。

**GRPO 损失函数**（带 clip 版）：

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}\hat{A}_{i,t},\;\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]$$

**Group Relative Advantage**：

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

**简化版（TRL 默认，单次更新）**：无需 clip，因为 $\pi_\theta = \pi_{\theta_{\text{old}}}$。

## 二、核心实现

### 2.1 GRPO 采样与格式化

**关键思想**：对每个 prompt 采样 G 次，用规则 reward 评估，构建 group advantage。

```python
# 格式化：<think>示例</think><answer>示例</answer>question<think>
def format_prompt(question_token_ids):
    example = [THINK_START, 2,3,4, THINK_END, ANSWER_START, 7, ANSWER_END]
    return example + question_token_ids + [THINK_START]

# GRPO 采样
def grpo_rejection_sampling(model, x, max_new_tokens=10):
    return model.generate(input_ids=x, max_new_tokens=max_new_tokens, do_sample=True)

# 每个问题采样 G 次
input_x = [format_prompt(X[0])] * grpo_samples_nums  # 重复 G 次
input_x_tensor = torch.tensor(input_x, dtype=torch.long)
grpo_xy = grpo_rejection_sampling(model, input_x_tensor, max_new_tokens)
```

### 2.2 Rule-Based Reward

```python
def rule_reward(response, label_id):
    """检查 response 中是否有 <answer>label_id</answer> 格式的正确答案"""
    for i in range(len(response) - 2):
        if (response[i] == ANSWER_START and
            response[i+1] == label_id and
            response[i+2] == ANSWER_END):
            return True
    return False

def think_reward(response):
    """检查是否出现 <think>...</think> 格式"""
    found_start = False
    for num in response:
        if num == THINK_START: found_start = True
        elif num == THINK_END and found_start: return True
    return False
```

### 2.3 GRPO Advantage

```python
def grpo_advantage(rewards):
    epsilon = 0.0001
    rewards = torch.tensor(rewards, dtype=torch.float)
    A = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return A

# 关键特性演示
grpo_advantage([0,0,0,0,0,0])  # 全错 → 全0（skip）
grpo_advantage([1,0,0,0,0,0])  # 越少正例，advantage 越大
grpo_advantage([1,1,1,1,1,1])  # 全对 → 全0（skip）
grpo_advantage([1,0,0,0,1,0])  # 正常情况：正例 > 0，负例 < 0
```

**关键洞察**：
1. 全对或全错时 advantage 为 0，优化 skip——这意味着需要足够的采样多样性
2. 越少正例，其 advantage 越大——稀有正确答案获得更大梯度
3. 采样越多，advantage 估计越准确

### 2.4 GRPO KL 散度（Schulman 近似）

**三种 KL 近似**（参考 [Schulman 2020](http://joschu.net/blog/kl-approx.html)）：

$$k_1 = -\log r, \quad k_2 = \frac{(\log r)^2}{2}, \quad k_3 = r - 1 - \log r$$

其中 $r = \pi_{\text{ref}} / \pi_\theta$。GRPO 使用 $k_3$（恒正，方差小）：

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    """K3: r - 1 - log(r)，其中 r = pi_ref/pi_theta"""
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1
```

**为什么选 K3**：
- K1 有负值，方差大
- K2 和 K3 恒正，K3 均值更小
- K3 = (r-1) - log(r) 恒正（因为 $x - 1 \geq \log x$）

### 2.5 GRPO Loss 完整实现

```python
def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len):
    epsilon = 0.2
    beta = 0.01
    bs, seq_len = pi_logprob.shape
    
    advantage = advantage.unsqueeze(dim=1)  # broadcast
    
    # PPO-style clipped ratio
    ratio = torch.exp(pi_logprob - pi_old_logprob)
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)
    
    # KL penalty
    kl = grpo_kl(pi_logprob, pi_ref_logprob)
    
    # Response mask（仅对生成部分计算 loss）
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1
    
    group_num = bs
    len_oi = seq_len - input_len
    
    loss = (policy_gradient - beta * kl) * mask
    loss = (-1 / group_num) * loss / len_oi
    return loss.sum()
```

### 2.6 GRPO 完整训练循环

```python
for epoch in range(epochs):
    for batch in data_loader:
        # STEP 1: On-policy 采样
        grpo_xy_batch, grpo_x_len = GRPO_batch_rejection_sample(
            batch['input'], grpo_samples_nums, max_new_tokens)
        
        # STEP 2: 计算 reward 和 advantage
        batch_rewards = GRPO_batch_reward(batch['input'], grpo_xy_batch, batch['label'])
        batch_advantage = [grpo_advantage(rewards) for rewards in batch_rewards]
        
        # STEP 3: 记录 old policy 和 ref policy（frozen）
        with torch.no_grad():
            pi_old_logprob = model(grpo_xy).logits → log_softmax → gather
            pi_ref_logprob = model_ref(grpo_xy).logits → log_softmax → gather
        
        # STEP 4: 多轮训练
        for k in range(grpo_epochs):
            total_loss = 0
            for group in zip(pi_old, pi_ref, advantage, x_len, grpo_xy):
                pi_grpo_logprob = model(grpo_xy).logits → log_softmax → gather
                loss = grpo_loss(pi_grpo_logprob, pi_old, pi_ref, advantage, x_len)
                total_loss += loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## 三、工程实践

### verl 框架实践

> 完整代码见：`/tmp/ma-rlhf/r1/verl/tutorial.ipynb`

verl（Volcano Engine RL）是字节跳动开源的 RL 训练框架，支持 GRPO。核心概念：

```python
import ray
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup

# 资源池：4 GPU
resource_pool = RayResourcePool([4], use_gpu=True)

# GPU Worker
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self):
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

# Worker Group：统一管理多 GPU
worker_group = RayWorkerGroup(resource_pool, class_with_args)
worker_group.execute_all_sync("add", x=[1, 1, 1, 1])

# 支持 GPU 共享：多个 WorkerGroup 可映射到同一 ResourcePool
# 支持 Megatron 并行：TP/PP/DP 对用户透明
```

### R1 项目结构

> 完整代码见：`/tmp/ma-rlhf/r1/`

支持：Dense/MoE Base Model + DeepSpeed ZeRO-1/2/3 + DPO + GRPO RLVR + Agentic-RL

## 四、关键洞察与总结

1. **去掉 Critic 是 GRPO 的核心创新**：用 group relative advantage 替代 GAE，大幅减少训练资源
2. **Rule-Based Reward 的优势**：数学/代码任务可以用规则验证正确性，不需要 RM——这是 R1 能成功的关键
3. **采样多样性是关键**：全对/全错都无法学习，需要混合正负样本
4. **KL 近似的选择**：K3（Schulman）恒正且方差小，是 GRPO 的标准选择
5. **GRPO vs PPO**：
   - PPO 需要 4 个模型（Actor + Critic + RM + Ref），GRPO 只需要 2 个（Actor + Ref）
   - PPO 用 Critic 估计 advantage，GRPO 用 group sampling
   - GRPO 对 reward 质量更敏感——rule reward 必须准确
6. **verl 的工程意义**：将 Ray + Megatron 的分布式能力封装，让 GRPO 训练可以无缝扩展到数百 GPU

> 完整代码见：`/tmp/ma-rlhf/notebook/GRPO/GRPO.ipynb`、`GRPO_KL.ipynb`、`/tmp/ma-rlhf/r1/grpo.py`
