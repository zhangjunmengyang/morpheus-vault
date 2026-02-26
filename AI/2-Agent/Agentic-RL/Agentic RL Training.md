---
title: "Agentic RL Training"
brief: "用 RL 训练 Agent 多步交互能力（工具调用/环境反馈/任务规划）的工程笔记；基于 verl 框架（agentic_rl/agent_loop/reward_loop）；对比单轮 GRPO vs 多轮 Agentic RL 的核心差异与实现要点"
type: concept
domain: ai/agent/agentic-rl
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/agent/agentic-rl
  - type/concept
sources:
  - "verl 文档: https://verl.readthedocs.io/en/latest/start/agentic_rl.html"
---
# Agentic RL Training

> 用 RL 训练 Agent 的多步交互能力：工具调用、环境反馈、任务规划。这是当前 post-training 的前沿方向。

verl 文档：
- https://verl.readthedocs.io/en/latest/start/agentic_rl.html
- https://verl.readthedocs.io/en/latest/advance/agent_loop.html
- https://verl.readthedocs.io/en/latest/advance/reward_loop.html
- https://verl.readthedocs.io/en/latest/advance/fully_async.html

## 为什么需要 Agentic RL

传统的 RLHF/GRPO 训练的是 **单轮生成**：给一个 prompt，生成一个 response，打个分。但真实的 Agent 场景是 **多轮交互**：

```
用户: 帮我查一下上季度销售数据，做个分析
Agent: [调用 SQL 查询] → 得到原始数据
Agent: [调用 Python] → 数据清洗和分析  
Agent: [调用画图工具] → 生成图表
Agent: 这是分析报告...
```

这种场景下，reward 是 **稀疏且延迟的** — 只有整个任务完成后才知道好不好。中间每一步的决策质量很难单独评估。

## 核心挑战

### 1. Credit Assignment

多步交互中，最终的成功/失败应该归因于哪一步？

```
Step 1: 选对了工具 ✓
Step 2: 参数传错了 ✗ ← 这里出了问题
Step 3: 看到错误，正确 retry ✓
Step 4: 最终完成任务 ✓

# 传统 RL: 整条 trajectory 共享一个 reward
# 问题: Step 2 的错误被 Step 3 的 retry 掩盖了
```

### 2. 长 Trajectory

Agent 的交互可能有 10+ 轮，每轮生成 hundreds of tokens。整条 trajectory 可能有几万 tokens。

```python
# 显存和计算的挑战
# 假设 trajectory 长度 T=10, 每步 500 tokens
# 总 tokens = 5000
# 需要存储整条 trajectory 的 log_probs 和 advantages
# 对于 7B 模型，一条 trajectory 的 KV cache 就要几个 GB
```

### 3. 环境交互延迟

Agent 需要和真实环境交互（API 调用、代码执行、搜索），每次交互都有延迟。训练效率的瓶颈从 GPU 计算变成了环境交互。

## verl 的 Agentic RL 架构

verl 为 agentic RL 设计了专门的训练循环：

```python
# 伪代码：verl 的 agent loop
class AgenticRLTrainer:
    def rollout(self, prompts):
        trajectories = []
        for prompt in prompts:
            trajectory = []
            obs = prompt
            
            for step in range(max_steps):
                # Actor 生成 action（可能是文本、工具调用等）
                action = self.actor.generate(obs)
                
                # 环境执行 action，返回 observation
                obs, reward, done = self.env.step(action)
                trajectory.append((obs, action, reward))
                
                if done:
                    break
            
            trajectories.append(trajectory)
        return trajectories
    
    def train_step(self, trajectories):
        # 计算每步的 advantage
        advantages = self.compute_advantages(trajectories)
        # 用 PPO/GRPO 更新策略
        self.actor.update(trajectories, advantages)
```

### Reward Loop

reward 的计算可以发生在多个时机：

```python
# 1. 每步 reward（最理想但很难设计）
step_reward = reward_model(state, action)

# 2. 终局 reward（最常见）
final_reward = judge_task_completion(final_state)

# 3. 混合 reward
total_reward = final_reward + α * sum(step_rewards)

# verl 的 reward loop 支持自定义 reward function
# 可以接入沙箱执行、代码评测、人工标注等
```

### Fully Async Training

为了解决环境交互延迟，verl 支持完全异步训练：

```
Rollout Workers: 持续生成 trajectories
                 ↓ (异步推入 buffer)
Training Worker: 从 buffer 取数据训练
                 ↓ (周期性同步权重)
Rollout Workers: 拿到新权重继续 rollout
```

这种设计的 trade-off：**牺牲一点 on-policy 纯度换取训练效率**。通过 importance sampling 修正分布偏移。

## Reward 设计实践

Agentic RL 的 reward 设计是最难的部分：

```python
def agent_reward(trajectory):
    reward = 0.0
    
    # 1. 任务完成度（最重要）
    if task_completed(trajectory):
        reward += 5.0
    elif partially_completed(trajectory):
        reward += 2.0
    
    # 2. 效率奖励（鼓励用更少步骤完成）
    reward -= 0.1 * len(trajectory.steps)
    
    # 3. 工具使用质量
    for step in trajectory.steps:
        if unnecessary_tool_call(step):
            reward -= 0.3
        if correct_error_recovery(step):
            reward += 0.5
    
    # 4. 安全性惩罚
    if unsafe_action(trajectory):
        reward -= 10.0
    
    return reward
```

## 与传统 RL 的区别

| 维度 | 传统 RLHF | Agentic RL |
|------|-----------|------------|
| 交互轮数 | 1 轮 | N 轮 |
| Action 空间 | token sequence | tool call + reasoning |
| Reward | 即时 | 延迟/稀疏 |
| 环境 | 无 | 真实环境（API/沙箱） |
| 训练难度 | 中等 | 高 |

## 相关

- [[Tool Use|Tool Use]] — Agent 工具调用基础
- [[记忆模块|记忆模块]] — Agent 记忆系统
- [[多轮 RL 训练交互|多轮 RL 训练交互]] — verl 的多轮实现
- [[Sandbox Fusion 沙箱|Sandbox Fusion 沙箱]] — 代码执行沙箱
- [[Off Policy 异步训练器|Off Policy 异步训练器]] — 异步训练
- [[On-Policy vs Off-Policy|On-Policy vs Off-Policy]] — 策略分类基础
- [[GRPO 深度理解|GRPO 深度理解]]
- [[PPO 原理|PPO 原理]]
- [[verl 概述|verl 概述]]
- [[DeepSeek-R1|DeepSeek-R1]]
