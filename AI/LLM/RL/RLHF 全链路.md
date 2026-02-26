---
brief: "RLHF 全链路深度笔记——从 Reward Model 训练到 PPO/GRPO/DPO 对齐算法的完整工程链；覆盖数据收集/RM 训练/policy 优化/稳定性调参全阶段；interview/hot 标注，面试深度参考。"
title: "RLHF 全链路 — 从 Reward Model 到 PPO/GRPO/DPO"
date: 2026-02-13
tags:
  - ai/llm/rl
  - ai/llm/alignment
  - type/concept
  - interview/hot
status: active
---

# RLHF 全链路

> Reinforcement Learning from Human Feedback — 从偏好数据到模型对齐的完整工程 Pipeline

## 1. RLHF 全景图

```
                    ┌─────────────────────────────────────────┐
                    │           RLHF Full Pipeline            │
                    └─────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   Stage 1: SFT               Stage 2: RM                Stage 3: RL
   监督微调                    奖励模型训练                策略优化
   ─────────                   ──────────                  ────────
   Human demos →               Human prefs →               RM scores →
   Instruction tuning          Bradley-Terry               PPO / GRPO
                               Pairwise ranking            Policy gradient

                    ┌─────────────────────────────────────────┐
                    │        Direct Alignment (跳过 RM)       │
                    └─────────────────────────────────────────┘
                         DPO / KTO / ORPO / SimPO
                      直接从偏好数据优化策略，无需 RM
```

## 2. Stage 1: SFT（监督微调）

起点是一个预训练 base model，通过高质量指令-回答对进行微调。

参见 [[AI/LLM/SFT/SFT 原理|SFT 原理]] 和 [[AI/LLM/SFT/SFT-TRL实践|SFT-TRL 实践]]。

关键注意点：
- SFT 数据质量 >> 数量（千条高质量 > 万条噪声数据）
- 过度 SFT 会导致 **alignment tax**——损害 base model 的通用能力
- 最佳实践：1-3 epochs，learning rate 1e-5 ~ 5e-6

## 3. Stage 2: Reward Model 训练

### 3.1 数据格式：偏好对

```json
{
  "prompt": "解释什么是量子纠缠",
  "chosen": "量子纠缠是量子力学中的一种现象，当两个粒子...(详细准确的回答)",
  "rejected": "量子纠缠就是两个东西连在一起...(模糊不准确的回答)"
}
```

### 3.2 Bradley-Terry 模型

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

损失函数：
$$\mathcal{L}_{RM} = -\mathbb{E}[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$

```python
# Reward Model 训练核心
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # 取最后一个 token 的隐藏状态作为序列表示
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward

# 训练损失
def reward_loss(chosen_reward, rejected_reward):
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
```

### 3.3 工程实践要点

- RM 通常比 policy model **小 1-2 个量级**（更快推理）
- 过拟合风险高 → 需要正则化、early stopping
- **Reward Hacking**：RM 的漏洞会被 RL 算法利用（如产生冗长但空洞的回答）
- 解决方案：多个 RM 集成、定期更新 RM、加入长度惩罚

## 4. Stage 3: RL 策略优化

### 4.1 PPO (Proximal Policy Optimization)

经典 RLHF 算法，OpenAI InstructGPT 使用。

参见 [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] 和 [[AI/LLM/RL/PPO/PPO-TRL实践|PPO-TRL 实践]]。

```
PPO Architecture:
┌──────────────┐    ┌──────────────┐
│ Policy Model │    │ Value Model  │  ← Actor-Critic 架构
│   (Actor)    │    │   (Critic)   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
  Generate response    Estimate V(s)
       │                   │
       ▼                   ▼
  ┌──────────┐      ┌──────────────┐
  │ Reward   │      │  Advantage   │
  │ Model    │─────→│  Estimation  │
  └──────────┘      └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Clipped PPO  │
                    │    Loss      │
                    └──────────────┘
```

核心公式：

$$\mathcal{L}^{PPO} = -\mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**PPO 的问题**：
- 需要 4 个模型：Policy + Value + Reward + Reference → **显存需求巨大**
- 训练不稳定，超参敏感
- 工程复杂度高（需要精心调参 KL coefficient, clip range 等）

### 4.2 GRPO (Group Relative Policy Optimization)

DeepSeek 在 2024 年提出，R1 的核心训练算法。

参见 [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]。

```
GRPO vs PPO 的关键区别：

PPO:  需要 Value Model 估计基线
GRPO: 用同一 prompt 的多个采样的平均奖励作为基线

PPO:  4 models (Policy + Value + RM + Ref)
GRPO: 3 models (Policy + RM + Ref)  ← 省掉 Value Model!
```

GRPO 核心流程：
```python
# GRPO 简化伪代码
def grpo_step(policy, ref_policy, reward_model, prompt, G=16):
    # 1. 对同一 prompt 生成 G 个回答
    responses = [policy.generate(prompt) for _ in range(G)]
    
    # 2. 计算每个回答的奖励
    rewards = [reward_model(prompt, r) for r in responses]
    
    # 3. 组内归一化作为 advantage（关键！）
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
    
    # 4. 策略梯度更新 + KL 约束
    for response, advantage in zip(responses, advantages):
        ratio = policy.log_prob(response) - ref_policy.log_prob(response)
        loss = -advantage * clipped_ratio(ratio) + β * kl_penalty(ratio)
    
    return loss
```

### 4.3 DPO (Direct Preference Optimization)

跳过 RM 训练，直接从偏好数据优化策略。

参见 [[AI/LLM/RL/DPO/DPO-TRL实践|DPO-TRL 实践]]。

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

直觉：**提高 chosen 的概率，降低 rejected 的概率，同时不要偏离 reference 太远**。

```python
# DPO 训练核心（TRL 简化版）
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    beta=0.1,           # KL 惩罚强度
    loss_type="sigmoid", # 标准 DPO loss
    learning_rate=5e-7,
    max_length=1024,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 或 None（用 LoRA 时自动构建）
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 5. 算法对比

### 5.1 全面对比表

| 维度 | PPO | GRPO | DPO | DAPO |
|------|-----|------|-----|------|
| 需要 RM | ✅ | ✅ | ❌ | ✅ |
| 需要 Value Model | ✅ | ❌ | ❌ | ❌ |
| 在线/离线 | 在线 | 在线 | 离线 | 在线 |
| 显存需求 | 极高(4模型) | 高(3模型) | 中(2模型) | 高(3模型) |
| 训练稳定性 | 低 | 中高 | 高 | 高 |
| 性能上限 | 最高 | 很高 | 中高 | 很高 |
| 工程复杂度 | 极高 | 中 | 低 | 中 |
| 代表作 | InstructGPT | DeepSeek-R1 | Zephyr | 开源社区 |

### 5.2 何时选什么？

```
决策树：
├── 有高质量偏好数据，想快速迭代？
│   → DPO（最简单，效果不错）
├── 需要推理能力（数学/代码）？
│   → GRPO + Rule-based Reward（DeepSeek-R1 路线）
├── 追求极致对齐效果？
│   → PPO（如果工程能力支撑得住）
└── 想用开源框架快速上手？
    → TRL DPO/GRPO → verl PPO/GRPO
```

参见 [[AI/LLM/Frameworks/TRL/TRL 概述|TRL 概述]] 和 [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]。

## 6. 工程 Pipeline 实操

### 6.1 完整流程

```bash
# Step 1: SFT
python sft_train.py --model meta-llama/Llama-3.1-8B \
    --dataset alpaca_gpt4 --epochs 3

# Step 2: RM Training (如果用 PPO/GRPO)
python rm_train.py --model sft_checkpoint \
    --dataset ultrafeedback --epochs 1

# Step 3a: PPO/GRPO Training
python rl_train.py --algo grpo --policy sft_checkpoint \
    --reward_model rm_checkpoint --group_size 16

# Step 3b: DPO Training (无需 RM)
python dpo_train.py --model sft_checkpoint \
    --dataset ultrafeedback --beta 0.1
```

### 6.2 verl GRPO 实践要点

```yaml
# verl GRPO 配置关键参数
algorithm:
  name: grpo
  group_size: 16        # 每个 prompt 采样数
  clip_ratio: 0.2       # PPO clip
  kl_coef: 0.01         # KL 惩罚系数
  
reward:
  type: rule_based      # 规则奖励（推理任务）
  # 或 type: model_based + reward_model_path
```

参见 [[AI/LLM/Frameworks/verl/GRPO-verl实践|GRPO-verl 实践]] 和 [[AI/LLM/RL/GRPO/GRPO-TRL实践|GRPO-TRL 实践]]。

## 7. 前沿趋势 (2025-2026)

### 7.1 Rule-based Reward 的崛起

DeepSeek-R1 证明：对于数学/代码等可验证任务，**规则奖励比 RM 更稳定**：
- 正确性验证：代码能不能跑通？数学答案对不对？
- 格式奖励：有没有 `<think>` 标签？
- 避免 Reward Hacking

### 7.2 DAPO (Direct Alignment from Preferences Online)

在 GRPO 基础上改进：
- 动态采样：根据 advantage 分布调整采样策略
- 过滤 trivial examples（全对或全错的 prompt）
- 更长的 rollout 以获取更多信号

### 7.3 GSPO (Group Score Policy Optimization)

2025 年提出，在 GRPO 和 DPO 之间取折中：
- 使用二元信号（好/坏）而非连续奖励
- 更稳定的训练
- 适合不确定性高的场景

## 8. 面试常见问题

**Q1: DPO 为什么可以跳过 Reward Model？**
A: DPO 论文证明 RLHF 的 RL 目标在数学上等价于一个 classification 目标。通过重参数化，把隐式的 reward 函数嵌入到 policy 本身中，直接优化偏好对的 log-likelihood ratio。

**Q2: GRPO 的 "Group Relative" 具体指什么？**
A: 对同一个 prompt 生成一组（Group）回答，计算组内奖励的均值和标准差，用 `(r - mean) / std` 归一化作为 advantage。"Relative" 指的是优势是相对于组内其他回答计算的。

**Q3: PPO 的 KL 惩罚为什么重要？**
A: 防止 policy 偏离 reference model 太远导致 Reward Hacking。没有 KL 约束，模型会学到欺骗 RM 的 pattern（如生成又长又冗余的回答来刷高分）。

**Q4: 在线（on-policy）和离线（off-policy）方法的核心区别？**
A: 在线方法（PPO/GRPO）用当前策略实时生成数据训练，数据分布与策略一致；离线方法（DPO）用预先收集的固定数据集训练，存在分布偏移。在线方法性能上限更高但工程更复杂。

**Q5: 如何检测和缓解 Reward Hacking？**
A: 检测：监控 reward 上升但 win rate 下降。缓解：KL 惩罚、长度惩罚、多 RM 集成、定期人工评估、约束 reward model 的泛化范围。

## 相关链接

- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] / [[AI/LLM/RL/PPO/PPO-TRL实践|PPO-TRL 实践]]
- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] / [[AI/LLM/RL/GRPO/GRPO-TRL实践|GRPO-TRL 实践]]
- [[AI/LLM/RL/DPO/DPO-TRL实践|DPO-TRL 实践]]
- [[AI/LLM/RL/DAPO/|DAPO]]
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]] / [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]]
- [[AI/LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/LLM/RL/Fundamentals/RL 概览|RL 概览]]
