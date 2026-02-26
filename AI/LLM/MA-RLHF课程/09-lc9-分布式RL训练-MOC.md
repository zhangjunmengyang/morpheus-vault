---
title: "lc9 · 分布式 RL 训练专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc9_training"
tags: [moc, ma-rlhf, distributed-rl, ray, verl, grpo, r1, lc9]
---

# lc9 · 分布式 RL 训练专题地图

> **目标**：理解 LLM RL 训练的核心架构挑战——训推分离，掌握 Ray 三角架构和 verl 框架实战。  
> **核心问题**：为什么 LLM 的 RL 训练不能像 SFT 一样简单 forward-backward？为什么 generation 和 training 必须分离？

---

## 带着这三个问题学

1. **为什么分布式 RL 训练需要训推分离？** SFT 训练不需要，RL 训练需要——本质差异在哪？
2. **异步 GRPO 的 off-policy 问题是什么？** 用旧策略生成的数据训练新策略，会出什么问题？怎么缓解？
3. **从 SFT cold start 到 GRPO RLVR，推理能力是怎么「涌现」的？** R1 的训练流程是什么？

---

## 学习顺序

```
Step 1  理解 RL 训练 vs SFT 训练的差异    ← 为什么需要 rollout
   ↓
Step 2  Ray 三角架构                       ← Generator / Coordinator / Trainer
   ↓
Step 3  参数同步机制                       ← 训练完 → 同步到推理引擎
   ↓
Step 4  verl 框架实战                      ← GRPO/PPO 完整配置
   ↓
Step 5  R1 复现：SFT → GRPO → 推理涌现    ← 完整训练流水线
```

---

## 笔记清单

### Step 1：RL 训练 vs SFT 训练的本质差异

⏳ 待入库：**RL 训推分离原理笔记**

SFT 训练循环：
```
for batch in dataset:
    loss = model(batch)  # 前向
    loss.backward()      # 反向
    optimizer.step()     # 更新
```

RL 训练循环（多了 rollout 步骤）：
```
for prompt in dataset:
    responses = generate(model, prompt)     # 🔑 rollout（推理）
    rewards = reward_fn(prompt, responses)  # 打分
    loss = rl_loss(model, responses, rewards)  # RL loss
    loss.backward()
    optimizer.step()
```

**关键差异**：RL 训练每步需要先用**当前策略**生成 response（rollout），再基于 reward 计算 RL loss。Generation 是 autoregressive 的，无法用 teacher forcing 加速 → 推理开销远大于训练 → 必须用专门的推理引擎（vLLM）加速 rollout。

---

### Step 2：Ray 三角架构

**[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray 分布式 RL 训练实操]]**

```
                  Coordinator
                 /           \
                /             \
    Generator Actor        Trainer Actor
    (vLLM rollout)        (梯度更新)
```

- **Generator Actor**：封装 vLLM 推理引擎，接收 prompt → 生成 response → 返回 (response, log_probs)
- **Trainer Actor**：接收 (prompt, response, reward) → 计算 RL loss → 梯度更新
- **Coordinator**：调度 Generator 和 Trainer，管理参数同步，控制训练节奏

**参数同步**：Trainer 更新权重后 → broadcast 到 Generator 的 vLLM 引擎 → 下一轮 rollout 使用新策略

课程代码：
- `coordinator.py` — 训练调度核心
- `generator_actor.py` — vLLM 推理 Actor
- `trainer_actor.py` — 梯度更新 Actor
- `main.py` — 入口与配置
- `config.py` / `models.py` / `data_utils.py` — 辅助模块

⚠️ 课程代码状态：AI 生成，调试中。核心价值在于理解架构设计。

---

### Step 3：参数同步与 Off-Policy 问题

⏳ 待入库：**异步 RL 训练与 Off-Policy 处理笔记**

- **同步训练**：每轮 rollout 用最新参数 → on-policy → 数据质量好，但 GPU 利用率低（Generator 等 Trainer，Trainer 等 Generator）
- **异步训练**：Generator 和 Trainer 并行，不互等 → 高 GPU 利用率，但 Generator 用的是旧参数 → **off-policy 问题**
- **Off-policy 的后果**：旧策略生成的数据分布与当前策略不匹配 → RL 梯度估计有偏 → 训练不稳定
- **缓解方法**：
  - **Importance Sampling 修正**：`ratio = π_new(a|s) / π_old(a|s)`，加权修正分布偏移
  - **PPO Clipping**：限制 ratio 在 [1-ε, 1+ε] → 即使 off-policy 也不会步子太大
  - **控制 staleness**：限制参数版本差距（如最多落后 1-2 步）

---

### Step 4：verl 框架实战

**[[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]**

- **verl**（Volcano Engine RL）：字节跳动开源的 LLM RL 训练框架
- **支持算法**：GRPO / PPO / REINFORCE
- **核心抽象**：HybridFlow（训推混合调度）+ Ray 编排 + FSDP/Megatron 训练后端

关键配置：
- `reward_function`：可以是 rule-based（GSM8K 答案匹配）或 model-based（RM 打分）
- `num_generations`：GRPO 每个 prompt 生成多少条 response（G 值）
- `kl_penalty`：KL 散度惩罚系数 β

深入阅读：[[AI/LLM/Frameworks/verl/算法概述|verl 算法概述]] · [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]] · [[AI/LLM/Frameworks/verl/verl 训练参数|训练参数]]

---

### Step 5：R1 复现 — 推理能力涌现

**[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray 分布式 RL 训练实操]]**（R1 部分）

R1 训练流程：
```
Stage 1: SFT Cold Start
  ← 用少量高质量 CoT 数据做 SFT → 模型学会输出 <think>...</think> 格式

Stage 2: GRPO RLVR（RL with Verifiable Reward）
  ← 在数学/代码任务上用 verifiable reward（答案匹配）做 GRPO
  ← 模型自动学会更长、更深的推理链
  ← 推理能力「涌现」：模型开始自发 self-correct、backtrack

Stage 3: 拒绝采样 + 二次 SFT（可选）
  ← 用 Stage 2 的模型生成大量 response → 过滤高 reward 样本
  ← 二次 SFT 提炼推理能力
```

**涌现时刻**：GRPO 训练到某个阶段，模型的 response 会突然从「直接给答案」变成「先推理再给答案」→ CoT 能力并非显式教会的，而是 RL 奖励信号驱动的自然涌现。

课程代码：`lc9_training/r1/` 目录（R1 复现相关脚本）

深入阅读：[[AI/LLM/Architecture/DeepSeek-R1|DeepSeek R1]]

---

## 与其他课时的关系

- **前置**：lc7（RL 基础：策略梯度 / GAE） + lc8（GRPO / PPO 原理） → 本课时把它们放到分布式环境
- **后续**：xtrain（分布式并行手写）→ 理解 Trainer Actor 内部的 DP/TP/PP
- **关联**：lc10（推理系统）→ Generator Actor 内部就是 vLLM

---

## 面试高频场景题

**Q：为什么分布式 RL 训练需要训推分离？**  
A：RL 训练每步需要用当前策略做 rollout（生成 response），这是 autoregressive 推理，不能 teacher forcing。推理计算量大且内存需求不同（需要 KV Cache），和训练（前向+反向+优化器状态）共享 GPU 会互相干扰。分离后推理用 vLLM（高吞吐），训练用 FSDP/TP（高效并行），各自最优。

**Q：异步 GRPO 带来的 off-policy 问题怎么处理？**  
A：异步时 Generator 用旧参数 π_old 生成数据，Trainer 用当前参数 π_θ 计算 loss → 分布不匹配。处理方式：1）Importance Sampling ratio = π_θ/π_old 修正；2）PPO clipping 限制更新幅度；3）控制参数版本差距（最多落后 k 步）；4）μ-GRPO 等变体通过 replay buffer + 多轮复用缓解。

**Q：verl 和 OpenRLHF 的区别？**  
A：verl 基于 HybridFlow（混合 Ray + FSDP/Megatron），支持大规模 MoE 模型；OpenRLHF 基于纯 Ray + DeepSpeed，更成熟但灵活性略低。两者都支持 PPO/GRPO，verl 在异步训练和大模型支持上更前沿。

**Q：R1 的推理能力是怎么训练出来的？**  
A：三阶段：1）SFT cold start 教会模型输出 `<think>` 格式；2）GRPO 用 verifiable reward（数学答案匹配）做强化学习 → 模型发现「多想几步再答」能获得更高 reward → 自发产生更长推理链；3）可选的拒绝采样+二次 SFT 提炼。关键是 RL 奖励信号驱动了推理能力的涌现，而非人工标注的 CoT 数据。
