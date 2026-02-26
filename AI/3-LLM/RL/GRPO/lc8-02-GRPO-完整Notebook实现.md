---
title: GRPO 完整 Notebook 实现
brief: GRPO 端到端 Notebook 实现：组采样、advantage 归一化（含 group-level std）、clip ratio、KL 惩罚项。含 GRPO vs PPO 关键差异对比代码和消融实验。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl
  - grpo
  - notebook-implementation
related:
  - "[[GRPO 深度理解]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操]]"
  - "[[AI/LLM/RL/GRPO/GRPO-KL散度三种近似]]"
  - "[[AI/LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# GRPO 完整 Notebook 实现

> 来源：`ma-rlhf/notebook/GRPO/GRPO.ipynb`（48 cells）
> 原作者：xiaodongguaAIGC (dhcode-cpp)

## 1. GRPO vs PPO：核心差异

| 维度 | PPO | GRPO (DeepSeek) |
|------|-----|-----------------|
| Critic | 需要训练一个 Value Network 估计 V(s) | **无 Critic**，用组内归一化替代 |
| Advantage 计算 | GAE：$A_t = \delta_t + \gamma\lambda\delta_{t+1} + \cdots$ | **组内归一化**：$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ |
| 采样方式 | 单次采样 | 同一 prompt 采样 G 条（Group Rejection Sampling）|
| 训练开销 | Policy Model + Value Model | 仅 Policy Model（+ frozen Reference Model）|
| Reward 来源 | 通用 Reward Model | 天然适配 **Rule-based Reward**（verifiable tasks）|

**核心洞察**：GRPO 去掉了 Critic 网络，用「同一问题多次采样 → 组内比较」来估计 advantage。这在推理任务中特别有效——因为推理任务有确定性答案（verifiable reward），不需要学一个复杂的 value function。

---

## 2. 完整实现流程

### 2.1 模型配置

用一个小型 LLaMA 做演示：

```python
vocab_size = 32
hidden_size = 256
intermediate_size = 512
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 4
batch_size = 2
length_x = 10
max_new_tokens = 10
grpo_samples_nums = 3  # 每个 prompt 采样 3 条

config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
)
model = LlamaForCausalLM(config)
model_ref = LlamaForCausalLM(config)  # frozen reference model
```

### 2.2 格式化 Prompt（模拟 CoT 格式）

定义特殊 token 来模拟 `<think>...</think><answer>...</answer>` 格式：

```python
DEFINE_THINK_START = 25
DEFINE_THINK_END = 26
DEFINE_ANSWER_START = 27
DEFINE_ANSWER_END = 28

def format_prompt(question_token_ids):
    """
    格式：example + question + <think>
    强制模型以 CoT 格式输出
    """
    example = [DEFINE_THINK_START, 2, 3, 4, DEFINE_THINK_END,
               DEFINE_ANSWER_START, 7, DEFINE_ANSWER_END]
    format_question = example + question_token_ids + [DEFINE_THINK_START]
    return format_question
```

### 2.3 GRPO Rejection Sampling

同一 prompt 复制 G 份，一次性采样：

```python
def grpo_rejection_sampling(model, x, max_new_tokens=10):
    idx = {'input_ids': x}
    y = model.generate(**idx, max_new_tokens=max_new_tokens, do_sample=True)
    return y

def GRPO_batch_rejection_sample(inputs, nums, max_new_tokens=10):
    grpo_xy_batch = []
    grpo_x_len = []
    for input in inputs:
        format_inputs = [format_prompt(input)] * nums  # 复制 G 份
        format_input_len = len(format_inputs[0])
        grpo_x_len.append(format_input_len)
        input_x_tensor = torch.tensor(format_inputs, dtype=torch.long)
        grpo_xy = grpo_rejection_sampling(model, input_x_tensor, max_new_tokens)
        grpo_xy_batch.append(grpo_xy)
    return grpo_xy_batch, grpo_x_len
```

### 2.4 Rule-based Reward

两种 reward 信号：

```python
def rule_reward(response, label_id):
    """准确率 reward：答案是否被 <answer>...</answer> 正确包裹且匹配"""
    for i in range(len(response) - 2):
        if (response[i] == DEFINE_ANSWER_START and
            response[i + 1] == label_id and
            response[i + 2] == DEFINE_ANSWER_END):
            return True
    return False

def think_reward(response):
    """格式 reward：是否出现完整的 <think>...</think> 结构"""
    found_one = False
    for num in response:
        if num == DEFINE_THINK_START:
            found_one = True
        elif num == DEFINE_THINK_END:
            if found_one:
                return True
    return False
```

**Rule-based Reward 的设计哲学**：
- **格式 reward**：引导模型遵循 CoT 格式（`<think>` 标签完整）
- **准确率 reward**：验证最终答案是否正确（`<answer>` 包裹的值是否匹配 label）
- 两者可以组合加权，DeepSeek-R1 中 format reward 确保格式合规，accuracy reward 驱动推理能力

### 2.5 Advantage 计算（组内归一化）

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

```python
def grpo_advantage(rewards):
    epsilon = 0.0001
    rewards = torch.tensor(rewards, dtype=torch.float)
    A = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return A
```

**关键性质**：
1. **相对性**：全对或全错时 advantage ≈ 0，训练自动 skip（无信号可学）
2. **稀有正例高 advantage**：`[1,0,0,0,0,0]` → 正例 advantage ≈ 2.24，远高于 `[1,1,1,0,0,0]` 的 ≈ 1.0
3. **正负兼有**：正例为正 advantage，负例为负 advantage
4. **采样越多越准**：G 越大，advantage 估计越稳定

```python
# 示例
grpo_advantage([1,0,0,0,0,0])  # tensor([ 2.2361, -0.4472, -0.4472, -0.4472, -0.4472, -0.4472])
grpo_advantage([1,0,0,0,1,0])  # tensor([ 1.2247, -0.8165, -0.8165, -0.8165,  1.2247, -0.8165])
grpo_advantage([0,0,0,0,0,0])  # tensor([0., 0., 0., 0., 0., 0.])  ← 全错，无梯度
grpo_advantage([1,1,1,1,1,1])  # tensor([0., 0., 0., 0., 0., 0.])  ← 全对，无梯度
```

### 2.6 KL 散度

采用 Schulman (2020) 的 k3 近似：

$$D_{KL}[\pi_\theta \| \pi_{ref}] = \frac{\pi_{ref}(o_{i,t})}{\pi_\theta(o_{i,t})} - \log\frac{\pi_{ref}(o_{i,t})}{\pi_\theta(o_{i,t})} - 1$$

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    # k3 近似：r - log(r) - 1，其中 r = π_ref / π_θ
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1
```

### 2.7 GRPO Loss

**Clip 版本**（完整版，对应 PPO-style clipping）：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( r_t(\theta) \hat{A}_{i,t},\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) - \beta \, D_{KL}[\pi_\theta \| \pi_{ref}] \right]$$

**简化版本**（TRL 实现，因为每次生成只更新一次，$\pi_\theta = \pi_{old}$，ratio=1）：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t})}{\left[\pi_\theta(o_{i,t})\right]_{\text{no\_grad}}} \hat{A}_{i,t} - \beta \, D_{KL}[\pi_\theta \| \pi_{ref}] \right]$$

```python
def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len):
    epsilon = 0.2
    beta = 0.01
    bs, seq_len = pi_logprob.shape

    advantage = advantage.unsqueeze(dim=1)  # broadcast 到每个 token

    # PPO-style clipping
    ratio = torch.exp(pi_logprob - pi_old_logprob)
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)

    # KL penalty
    kl = grpo_kl(pi_logprob, pi_ref_logprob)

    # Mask：只对 response 部分计算 loss
    group_num, len_oi = pi_logprob.shape
    len_oi = len_oi - input_len
    len_oi = torch.tensor([len_oi] * group_num, dtype=torch.long)

    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1

    loss = (policy_gradient - beta * kl) * mask
    loss = (-1 / group_num) * loss / len_oi.unsqueeze(dim=1)
    loss = loss.sum()
    return loss
```

### 2.8 完整训练循环

```python
optimizer = optim.Adam(model.parameters(), lr=1e-6)

epochs = 10
grpo_epochs = 10  # 每批采样数据上的更新轮数

for i in range(epochs):
    for batch in data_loader:
        input = batch['input'].tolist()
        label = batch['label']

        # ===== STEP 1: On-policy Sampling =====
        grpo_xy_batch, grpo_x_len = GRPO_batch_rejection_sample(
            input, grpo_samples_nums, max_new_tokens=max_new_tokens
        )

        # ===== STEP 2: Compute Reward =====
        batch_rewards = GRPO_batch_reward(input, grpo_xy_batch, label)

        # ===== STEP 3: Compute Advantage =====
        batch_advantage = [grpo_advantage(r) for r in batch_rewards]

        # ===== STEP 4: Cache old & ref log-probs (no grad) =====
        pi_old_logprob_list, pi_ref_logprob_list = [], []
        for grpo_xy in grpo_xy_batch:
            with torch.no_grad():
                old_logits = model(grpo_xy).logits
                ref_logits = model_ref(grpo_xy).logits

            pi_old = F.log_softmax(old_logits, dim=-1)
            pi_old = torch.gather(pi_old, dim=-1, index=grpo_xy.unsqueeze(-1)).squeeze(-1)
            pi_old_logprob_list.append(pi_old)

            pi_ref = F.log_softmax(ref_logits, dim=-1)
            pi_ref = torch.gather(pi_ref, dim=-1, index=grpo_xy.unsqueeze(-1)).squeeze(-1)
            pi_ref_logprob_list.append(pi_ref)

        # ===== STEP 5: Policy Gradient Updates =====
        for k in range(grpo_epochs):
            total_loss = 0
            for pi_old, pi_ref, adv, x_len, grpo_xy in zip(
                pi_old_logprob_list, pi_ref_logprob_list,
                batch_advantage, grpo_x_len, grpo_xy_batch
            ):
                logits = model(grpo_xy).logits
                pi = F.log_softmax(logits, dim=-1)
                pi = torch.gather(pi, dim=-1, index=grpo_xy.unsqueeze(-1)).squeeze(-1)

                loss = grpo_loss(pi, pi_old, pi_ref, adv, x_len - 1)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

---

## 3. 为什么 GRPO 特别适合推理任务

**Verifiable Reward 的本质**：

推理任务（数学、代码、逻辑）具有**可验证的确定性答案**，这意味着：

1. **不需要 Reward Model**：用规则（rule-based）直接判定对错，避免 RM 的 bias 和 reward hacking
2. **二值 reward 足够**：对/错（0/1），GRPO 的组内归一化天然处理二值信号
3. **无需 Critic**：传统 RL 用 Critic 减小 variance，但推理任务的 reward 噪声小（答案确定），组内比较已经足够
4. **CoT 天然适配**：格式 reward 引导模型输出结构化推理过程，准确率 reward 驱动正确性
5. **计算效率**：省掉 Value Network 的训练和推理，对大模型场景节省 ~50% 显存

---

## 4. 面试考点

### Q1：GRPO 为什么不需要 Critic？去掉 Critic 之后 variance 怎么控制？

**答**：GRPO 用组内归一化（Group Normalization）替代 Critic 来估计 baseline。对同一 prompt 采样 G 条 response，用 `(r_i - mean(r)) / std(r)` 作为 advantage。这本质上是用 Monte Carlo 均值作为 baseline（类似 REINFORCE with baseline），当 G 足够大时方差可控。在推理任务中 reward 是确定性的（0/1），不需要复杂的 value function 来减小 variance。

### Q2：GRPO 的 advantage 全对/全错时为 0，这意味着什么？

**答**：全对/全错时所有 reward 相同，归一化后 advantage = 0，梯度为 0，训练自动 skip。这是合理的——组内无差异说明当前采样无法区分好坏，强行更新会引入噪声。这也说明 GRPO 的训练效率依赖于**采样的多样性**：如果一个 prompt 太简单（全对）或太难（全错），对训练没有贡献。实践中需要通过课程学习或难度采样来保证有足够的「mixed」样本。

### Q3：GRPO Loss 的两种形式有什么区别？实践中用哪个？

**答**：
- **Clip 版本**：保留 PPO 的 ratio clipping，$\min(r_t A, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A)$，适用于在同一批采样数据上多次更新（multi-epoch）
- **简化版本**：ratio 固定为 1（因为 $\pi_\theta = \pi_{old}$），简化为 $\frac{\pi_\theta}{\pi_\theta^{\text{no\_grad}}} \hat{A}$，仅做一次更新

TRL 实现默认用简化版本。但如果需要 multi-epoch 更新（如 notebook 中 `grpo_epochs=10`），应使用 clip 版本防止策略偏移过大。

### Q4：GRPO 中 KL 散度的作用是什么？为什么用 Schulman k3 近似？

**答**：KL 惩罚 $\beta \cdot D_{KL}[\pi_\theta \| \pi_{ref}]$ 防止策略偏离 reference model 太远（避免 reward hacking、模式坍塌）。用 k3 近似 $r - \log r - 1$（其中 $r = \pi_{ref}/\pi_\theta$）因为：(1) 恒为正值（$x - 1 \geq \ln x$），不会出现负 KL；(2) 数值稳定，不像 k1 有正负振荡导致高 variance；(3) 仅需 log-prob，计算高效。

### Q5：GRPO 的 mask 策略是什么？为什么只对 response 部分计算 loss？

**答**：GRPO loss 中 `mask[:, input_len:] = 1`，即只对生成的 response token 计算 policy gradient 和 KL。原因：(1) prompt 部分是给定的，不是策略生成的，对 prompt token 计算梯度没有意义；(2) 每条 response 长度不同，需要用 `1/|o_i|` 做长度归一化，避免长 response 主导梯度；(3) 这和标准 RLHF 的做法一致——只优化 action（生成的 token），不优化 state（输入的 token）。
