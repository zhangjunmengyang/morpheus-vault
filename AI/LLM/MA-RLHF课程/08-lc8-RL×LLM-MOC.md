---
title: "lc8 · RL×LLM 专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc8_rl_llm"
tags: [moc, ma-rlhf, rlhf, ppo, dpo, grpo, kto, prm, lc8]
---

# lc8 · RL×LLM 专题地图

> **核心问题链**：RLHF 为什么需要 RL？PPO 的四模型架构如何工作？DPO 为什么能绕过 RM？GRPO 为什么比 PPO 更高效？O1 的 PRM 搜索是怎么实现的？

---

## 带着这三个问题学

1. RLHF 范式中，各个模型的输入输出分别是什么？
2. 偏好学习（PPO/DPO/KTO）各自有哪些致命缺点？
3. 为什么 GRPO 那么有效，为什么我们要提升模型的推理能力？

---

## 学习顺序（严格按序）

```
Step 1  Reward Model 训练        ← 先知道奖励从哪来
   ↓
Step 2  RLHF-PPO 完整实现        ← 四模型架构，完整 PyTorch 实现
   ↓
Step 3  Bradley-Terry 模型       ← DPO 的数学基础
   ↓
Step 4  DPO 原理与实现            ← 绕过 RM 的优雅解法
   ↓
Step 5  KTO 原理与实现            ← 前景理论，单样本偏好
   ↓
Step 6  GRPO 原理与实现           ← 无 Critic，Group 归一化
   ↓
Step 7  Process Reward Model     ← 步骤级奖励
   ↓
Step 8  O1 搜索实现（MCTS/PRM）   ← Test-time compute scaling
```

---

## 笔记清单

### Step 1：Reward Model 训练

**[[AI/LLM/RL/PPO/MA-RLHF-核心代码注解|MA-RLHF 核心代码注解]]** — 含 `reward_model.py` 完整注解

关键点：
- Reward Model = SFT 模型 + Value Head（输出标量奖励）
- 训练目标：chosen response 得分 > rejected response（Bradley-Terry Loss）
- **[[AI/LLM/RL/PPO/LLaMA2-Reward-Model实现|LLaMA2 RM 完整实现]]** ✅ — LLaMA2 + Bradley-Terry Loss 完整 Notebook
- **[[AI/LLM/RL/DPO/Bradley-Terry模型实现|Bradley-Terry 模型实现]]** ✅ — BT 偏好建模理论入口（DPO 数学基础）

---

### Step 2：RLHF-PPO 完整实现

**[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO 手撕实操（MA-RLHF 版）]]**

四模型架构：
```
Actor（SFT初始化）    ← 被训练的策略
Critic（SFT初始化）   ← 估计 V(s)，只更新参数不生成
Reference（SFT冻结）  ← KL 惩罚基准，防止跑偏
Reward Model（RM冻结）← 给出奖励信号
```

关键公式：
- `r_KL = r_RM - β * KL(π_θ || π_ref)`
- `GAE advantage = Σ (γλ)^t * δ_t`，δ_t = r_t + γV(s_{t+1}) - V(s_t)
- `PPO Loss = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`

MA-PPO（Multi-Adapter PPO）：Actor + Critic 共享一个 backbone，双 LoRA adapter → 节省 3x 显存

**[[AI/LLM/RL/PPO/RLHF-PPO-完整Pytorch实现|RLHF-PPO 完整 Pytorch 实现]]** ✅ — 56-cell 四模型架构完整实现

深入阅读：[[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] · [[AI/LLM/RL/RLHF 全链路|RLHF 全链路]]

---

### Step 3-4：Bradley-Terry + DPO

**[[AI/LLM/RL/DPO/DPO-手撕实操|DPO 手撕实操]]**

**[[AI/LLM/RL/DPO/Bradley-Terry模型实现|Bradley-Terry 模型实现]]** ✅ · **[[AI/LLM/RL/DPO/DPO-完整Notebook实现|DPO 完整 Notebook]]** ✅
- `P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))`
- DPO 的推导：把 r(x,y) 用 KL-reg RL 的最优解代入 BT → 得到 DPO Loss

DPO Loss：
```python
loss = -log_sigmoid(β * (log_ratio_chosen - log_ratio_rejected))
```

致命缺点：需要 paired data（chosen + rejected），数据收集成本高

深入阅读：[[AI/LLM/RL/Preference-Optimization/IPO-Identity-Preference-Optimization|IPO]] · [[AI/LLM/RL/Preference-Optimization/ORPO-Odds-Ratio-Preference-Optimization|ORPO]]

---

### Step 5：KTO

**[[AI/LLM/RL/KTO/KTO-手撕实操|KTO 手撕实操]]**

前景理论视角：人类对损失的厌恶 > 对收益的喜爱  
不需要 paired data，只需要单样本 + binary label（好/坏）  
KTO Loss：分别用 desirable 和 undesirable response 各自训练

**[[AI/LLM/RL/KTO/KTO-完整Notebook实现|KTO 完整 Notebook]]** ✅ — 前景理论偏好建模完整实现

---

### Step 6：GRPO

**[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO 手撕实操]]**

vs PPO 的关键差异：
- 无 Critic → 节省一个模型的显存和计算
- Advantage = (reward - group_mean) / group_std（同一 prompt 的 G 条回答组内归一化）
- 无 GAE，不需要 V(s_t)，只需要最终 reward

```python
# GRPO Advantage
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
# GRPO Loss（简化版）
loss = -(advantages * log_probs).mean() + β * kl_penalty
```

为什么有效：Group 归一化相当于自动构建了相对奖励基准，无需 critic 就有低方差的 advantage 估计

**[[AI/LLM/RL/GRPO/GRPO-完整Notebook实现|GRPO 完整 Notebook]]** ✅ · **[[AI/LLM/RL/GRPO/GRPO-KL散度三种近似|GRPO KL 三种近似]]** ✅

深入阅读：[[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] · [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]]

---

### Step 7-8：PRM + O1 搜索

**[[AI/LLM/RL/PPO/PRM-O1-Search-手撕实操|PRM + O1 搜索手撕实操]]**

Process Reward Model（步骤级奖励）：
- Outcome RM：只给最终结果打分（稀疏）
- Process RM：每个推理步骤打分（密集）→ 更好的 credit assignment

**[[AI/LLM/RL/PPO/O1-PRM搜索完整实现|O1 PRM 搜索完整实现]]** ✅ — Beam Search + MCTS 完整实现

Notebook 覆盖：
- `o1_prm_search.ipynb`：Beam Search + PRM 打分
- `gomoku-mcts-pytorch.ipynb`：MCTS 完整实现（在游戏环境中验证逻辑）
- `mc-example.ipynb`：Monte Carlo 采样估计步骤价值

深入阅读：[[AI/LLM/RL/Fundamentals/MCTS|MCTS 深度理解]] · [[AI/LLM/Inference/TTC-Test-Time-Compute-Efficiency-2026-综合分析|TTC 综合分析 2026]]

---

## 算法横向对比

| 算法 | 数据需求 | Critic | 优势 | 缺点 |
|------|---------|--------|------|------|
| PPO | prompt + RM | 需要 | 理论扎实，稳定 | 4 模型，成本高 |
| DPO | paired (chosen, rejected) | 不需要 | 简单，无 RM 训练 | 需 paired data，分布外泛化差 |
| KTO | unpaired + label | 不需要 | 只需 binary label | 理论基础较弱 |
| GRPO | prompt + verifiable reward | 不需要 | 无 Critic，适合推理任务 | 需要 group sampling（G 条） |
| PRM+Search | prompt + step labels | 不需要 | Test-time 可扩展 | 数据标注成本极高 |

---

## 面试高频场景题

**Q：RLHF 流水线各个模型的输入输出分别是什么？**  
A：SFT Model 生成 response → RM 对 response 打 reward 分 → PPO Actor 接受 reward 更新策略（被 Critic 的 GAE 加速），Reference Model 提供 KL 惩罚基准保证不跑偏。

**Q：DPO 和 PPO 怎么选？**  
A：有 paired preference data + 追求简单 → DPO；需要 online RL + 推理任务 + verifiable reward → PPO/GRPO。GRPO 是当前推理对齐的事实标准。

**Q：GRPO 为什么不需要 Critic？**  
A：Critic 的作用是估计 baseline V(s) 来降低 advantage 方差。GRPO 用同一 prompt 的 G 条 rollout 的 mean reward 作为 baseline，Group 归一化天然起到低方差 baseline 的作用。

**Q：PRM 和 ORM 的区别？什么时候用 PRM？**  
A：ORM 只在最终结果给 reward（稀疏），信用分配困难；PRM 在每步给 reward（密集），适合长链推理。代价是标注成本高（需要步骤级正确性标注）。

---

## Batch B — 从零手写系列（2026-02-26 新增）

> Batch B 与 Batch A 的区别：Batch A 侧重算法核心（手撕实操 + 独立 Notebook），Batch B 侧重**完整工程流程**（显存管理、四模型调度、Batch B 更接近生产级实现）。两批互为补充，面试前两个都要过一遍。

| 步骤 | Batch B 笔记 | 对应 Batch A |
|------|-------------|-------------|
| RM 训练 | [[AI/LLM/MA-RLHF课程/lc8-LLaMA2-Reward-Model手撕\|lc8-LLaMA2 RM]] | [[AI/LLM/RL/PPO/LLaMA2-Reward-Model实现\|Batch A RM]] |
| RLHF-PPO | [[AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-手撕实操\|lc8-RLHF-PPO]] | [[AI/LLM/RL/PPO/RLHF-PPO-完整Pytorch实现\|Batch A PPO]] |
| BT + DPO + IPO | [[AI/LLM/MA-RLHF课程/lc8-DPO-IPO-手撕实操\|lc8-DPO/IPO/BT]] | [[AI/LLM/RL/DPO/DPO-完整Notebook实现\|Batch A DPO]] |
| Bradley-Terry 偏好建模 | [[AI/LLM/MA-RLHF课程/lc8-Bradley-Terry-偏好建模手撕\|lc8-BT偏好建模]] | — |
| KTO + PRM Search | [[AI/LLM/MA-RLHF课程/lc8-KTO-手撕实操\|lc8-KTO/PRM]] | [[AI/LLM/RL/KTO/KTO-完整Notebook实现\|Batch A KTO]] |
| GRPO Pytorch | [[AI/LLM/MA-RLHF课程/lc8-GRPO-notebook-Pytorch从零手写\|lc8-GRPO]] | [[AI/LLM/RL/GRPO/GRPO-完整Notebook实现\|Batch A GRPO]] |
| GRPO KL 散度三种近似 | [[AI/LLM/MA-RLHF课程/lc8-GRPO-KL-三种近似手撕\|lc8-GRPO-KL]] | — |
