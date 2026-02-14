---
title: "Slime: 清华/智谱 LLM RL 后训练框架"
date: 2026-02-14
tags: [rl, post-training, framework, glm, open-source, agentic-rl]
type: note
---

# Slime: LLM Post-Training RL Scaling Framework

> GitHub: [THUDM/slime](https://github.com/THUDM/slime)
> 出品: 清华 THUDM + 智谱 Z.AI
> 定位: SGLang-native 的 LLM 后训练 RL 框架，GLM-4.5/4.6/4.7/5 的 RL 训练基础设施

## 核心价值

解决大规模 RL 训练的两个关键瓶颈：
1. **长尾生成等待**：传统 RL 训练中 >90% 时间在等模型生成 rollout
2. **训练-生成耦合**：生成和训练同步进行，资源利用率低

## 架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Training   │◄───│  Data Buffer │◄───│   Rollout    │
│  (Megatron)  │    │   (桥接层)    │    │ (SGLang+Router)│
└──────┬───────┘    └──────────────┘    └──────▲───────┘
       │          参数同步                       │
       └────────────────────────────────────────┘
```

- **Training**: 基于 Megatron，从 Data Buffer 读数据训练
- **Rollout**: 基于 SGLang + Router，生成新数据（含 reward/verifier 输出）
- **Data Buffer**: 桥接模块，管理 prompt 初始化、自定义数据、rollout 生成

## 关键技术

### 异步解耦
训练和数据生成完全解耦，允许独立生成训练轨迹，不互相阻塞。

### Active Partial Rollouts (APRIL)
- 解决长尾生成瓶颈
- 不等待所有 rollout 完成，部分完成即可开始训练
- 显著提升训练吞吐量

### Agent-Oriented Design
- 专门为 Agentic RL 设计的异步框架
- 支持多步骤、多工具调用的 agent 训练轨迹
- [设计文档](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL)

## 支持的模型

| 模型系列 | 具体版本 |
|----------|----------|
| GLM | 4.5, 4.6, 4.7, 5 |
| Qwen | Qwen3 (含 MoE), Qwen2.5 |
| DeepSeek | V3, V3.1, R1 |
| LLaMA | 3 |

## 实战案例

### P1: 物理奥赛推理模型
纯 RL 训练的物理推理模型，使用多阶段 RL + 自适应可学习性调整

### GLM-5 训练
- 744B MoE 模型的 RL 阶段使用 slime
- 在华为昇腾上完成训练
- 幻觉率从 -36 提升至 -1（Omniscience Index）

## 与其他 RL 框架对比

| 框架 | 核心特点 | 训练后端 | 生成后端 |
|------|----------|----------|----------|
| **slime** | 异步解耦、APRIL、Agentic RL | Megatron | SGLang |
| verl | 统一调度、HybridEngine | verl/FSDP | vLLM |
| OpenRLHF | 简洁、Ray 调度 | DeepSpeed | vLLM |
| TRL | HuggingFace 生态、易上手 | Accelerate | 内置 |

## 对我们的意义

1. **RL 方向核心工具**：老板做 RL 研究，slime 是目前最强的开源 RL 后训练框架之一
2. **支持主流模型**：Qwen/DeepSeek/LLaMA 全覆盖，可直接用于实验
3. **Agentic RL**：专门的 agent 训练设计，跟 Agent 算法工程师方向完全对口
4. **面试谈资**：能讲清楚 slime 的架构和 APRIL 的原理，展示对 RLHF/RL 工程的深度理解

## 参考
- [GitHub](https://github.com/THUDM/slime)
- [LMSYS Blog: slime vision](https://lmsys.org/blog/2025-07-09-slime/)
- [DeepWiki 文档](https://deepwiki.com/THUDM/slime)
- [v0.1.0 Release Note](https://thudm.github.io/slime/blogs/release_v0.1.0.html)
