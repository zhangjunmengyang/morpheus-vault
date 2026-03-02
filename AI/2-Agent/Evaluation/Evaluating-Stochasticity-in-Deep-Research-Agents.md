---
title: "Evaluating Stochasticity in Deep Research Agents"
date: 2026-03-01
updated: 2026-03-01
arxiv: 2602.23271
tags:
  - AI/Agent
  - deep-research
  - evaluation
  - stochasticity
  - harness
---

## TL;DR

Deep Research Agent（DRA）落地时一个常被忽略的工程障碍：**stochasticity（同一 query 重跑结果差异巨大）**。
这篇把 DRA 形式化成 information acquisition MDP，并提出一个衡量“方差”的评估框架，把方差来源拆成三类：
- information acquisition（检索/浏览/工具调用）
- information compression（摘要/笔记/上下文管理）
- inference（模型生成本身）

关键结论（摘要）：
- 降低 stochasticity 往往能提升 research output quality
- **inference** 与 **early-stage stochasticity** 对输出方差贡献最大
- 给出两类 mitigation：structured output + ensemble-based query generation
- 在 DeepSearchQA 上：平均 stochasticity ↓ **22%**，同时保持高质量

## 论文信息
- Paper: *Evaluating Stochasticity in Deep Research Agents*
- arXiv:2602.23271 (2026-02-26)
- Authors: Haotian Zhai, Elias Stengel-Eskin, Pratik Patil, Liu Leqi

## 我认为最重要的“可迁移机制”

### 1) 把“agent 不稳定”当一等公民指标
现有很多 DRA 只看一次运行的 accuracy / judge score。
但真实部署要问的是：
- 同一输入，系统输出的**方差**多大？
- 方差来自哪里？（检索、压缩、推理）

这篇的贡献是把 stochasticity 作为 evaluation 轴，和质量并列。

### 2) 早期随机性更致命（路径依赖）
摘要指出 early-stage stochasticity 贡献最大。
直觉原因：
- 早期检索/选择的分叉会改变后续可见证据集合 → 后续推理是条件化的 → 方差被放大。

这和我对 SMTL / Search-P1 的理解一致：长任务的关键不是“更深想”，而是“证据路径结构”。

### 3) mitigation 策略的工程含义
- structured output：约束中间产物形态（类似把 agent 的 latent state 外显、减少自由度）
- ensemble-based query generation：把搜索 query 的随机性做成 ensemble，等价于对 acquisition policy 做 variance reduction

**我的判断**：这类方法本质是在做“控制变量实验”的工程化：你不控制中间过程，评测一次是没有意义的。

## 与 Vault 主线的连接

- Harness Engineering：stochasticity 是 harness 的系统性质，不只是模型温度。
- Agent RL/credit assignment：early-stage variance 大 → 训练信号（reward）方差也会大，间接影响 RL 稳定性。

## Open Questions（待后续读 PDF）

- stochasticity metric 具体定义是什么？（输出差异度、citation overlap、结论一致性？）
- 信息压缩模块如何建模？是否与 context compression / KV cache 管理（SideQuest）关联？
- ensemble query generation 的成本—收益曲线：多少 ensemble 才够？是否可自适应？
