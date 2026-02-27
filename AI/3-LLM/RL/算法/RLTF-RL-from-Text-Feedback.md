---
brief: "RLTF（arXiv:2602.02482）——从自然语言文本反馈中提取 reward signal，无需打分模型；LLM 直接消费 human/AI 的文字评价生成密集奖励，适用于 non-verifiable 域（创意写作/开放问答）的 RL 对齐。"
title: "RLTF: RL from Text Feedback"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - type/paper
  - rl/reward-design
created: 2026-02-16
arxiv: "2602.02482"
---

# RLTF — RL from Text Feedback

## 基本信息

- **论文**: Expanding the Capabilities of Reinforcement Learning via Text Feedback
- **arXiv**: [2602.02482](https://arxiv.org/abs/2602.02482)
- **提交**: 2026-02-02 (v2: 2026-02-11)
- **方向**: RL post-training, reward signal design

---

## 核心 Insight

RL for LLM post-training 的 reward 信号太稀疏（1 bit per rollout），distillation 又需要昂贵的 demonstrations。

**RLTF 提出中间路线**：text feedback 作为 intermediate supervision signal。
- 比 scalar reward 信息更丰富
- 比完整 demonstration 更便宜、更易获取

---

## 两种方法

### RLTF-SD (Self Distillation)
- 训练阶段：给模型 text feedback（critique），让它做 second-turn 修正
- 目标：训练 single-turn policy 去匹配自己 feedback-conditioned 的 second-turn 输出
- 推理阶段：无需 feedback（已内化）

### RLTF-FM (Feedback Modeling)
- 预测 feedback 作为辅助目标
- 理论上可以改善 representation，帮助模型理解自己的错误模式

---

## 我的评价

### 优点
- 概念 elegant：feedback 内化的 self-distillation 思路干净
- 有理论分析支撑两种方法
- 跨任务验证（推理谜题 + 竞赛数学 + 创意写作）

### 疑点
- **关键问题**：text feedback 从哪来？如果需要人工标注，成本并不低；如果用 LLM 自动生成，质量如何保证？论文似乎假设 feedback 已存在，但这在实践中往往是最大的瓶颈。
- 与 critic model / process reward model 的关系没有充分讨论
- 实验 margin 未见，需要读完整论文才能判断效果量级

### 定位
**Incremental but solid**。在 reward signal design 这条线上有贡献，但不是 paradigm shift。

---

## 与已有工作的关系

- RLHF 用 scalar preference → RLTF 用 text critique
- 与 Constitutional AI 的 critique-revision loop 有相似之处，但 RLTF 强调 internalization
- Process reward model (PRM) 提供 step-level 信号；RLTF 提供 natural language 信号

---

## Tags

`#RLHF` `#reward-design` `#text-feedback` `#self-distillation` `#post-training`

---

## See Also

- [[AI/3-LLM/RL/Theory/RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]] — 同为 non-verifiable 对齐方向：RLTF 用文本反馈，RLRR 用 reference + judge
- [[AI/3-LLM/RL/Theory/MARS-Margin-Aware-Reward-Modeling-Self-Refinement|MARS]] — reward model 训练的前置：更好的 RM 意味着更好的 text feedback
-  — LLM 强化学习全图谱
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 2026 全景]] — RLTF 中 reward shaping 与 GRPO reward 设计的关系
