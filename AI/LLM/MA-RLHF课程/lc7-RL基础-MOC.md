---
title: "lc7 · RL 基础专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc7_rl"
tags: [moc, ma-rlhf, rl, reinforcement-learning, lc7]
---

# lc7 · RL 基础专题地图

> **目标**：建立 RL 数学直觉，为 lc8（RL×LLM）打地基。  
> **核心问题链**：Bellman 方程如何推导出 Q-Learning？策略梯度为什么比 Q-Learning 更适合 LLM？GAE 解决了什么问题？

---

## 学习顺序

```
MDP 建模（状态/动作/奖励/转移）
   ↓
Bellman 方程（值函数递推基础）
   ↓
动态规划（GPI 广义策略迭代）
   ↓
蒙特卡洛方法（MC V / MC Q）
   ↓
TD 学习（Q-Learning / SARSA）
   ↓
DQN（深度 Q 网络）
   ↓
Policy Gradient（REINFORCE）
   ↓
Actor-Critic / A2C / GAE    ← PPO 的直接前置
```

---

## 核心笔记

### [[AI/LLM/RL/Fundamentals/RL基础算法手撕实操|RL 基础算法手撕实操]]
覆盖：Cross Entropy / 蒙特卡洛 / Bellman / GPI / MC V/Q / Q-Learning / SARSA / DQN / Policy Gradient

---

## 关键概念对 LLM 的映射

| RL 概念 | LLM 对应 |
|--------|---------|
| 状态 s | 当前对话上下文（token 序列） |
| 动作 a | 下一个 token |
| 奖励 r | Reward Model 评分 / Rule-based reward |
| 策略 π | LLM 参数 θ |
| 价值函数 V(s) | Critic 网络（PPO 中） |
| Episode | 一次完整生成（prompt → response） |

---

## 深入阅读

- [[AI/LLM/RL/Fundamentals/强化学习的数学原理|强化学习数学原理]]
- [[AI/LLM/RL/Fundamentals/策略梯度方法|策略梯度方法]]
- [[AI/LLM/RL/Fundamentals/贝尔曼方程|贝尔曼方程]]
- [[AI/LLM/RL/Fundamentals/RL 概览|RL 概览]]
- 课程配套代码：`/tmp/ma-rlhf/lecture/lc7_rl/`（GPI.py / dqn.py / policy_gradient.py 等）
