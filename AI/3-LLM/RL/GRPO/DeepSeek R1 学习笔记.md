---
brief: "DeepSeek R1 论文学习笔记——聚焦 R1 的 Aha Moment（RL 自发涌现自我反思推理）和 Cold Start 训练策略；GRPO 如何驱动 R1 的推理能力突破，以及 R1-Zero vs R1 训练流程对比。"
title: "2. DeepSeek R1 Paper"
type: tutorial
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/tutorial
---
# 2. DeepSeek R1 Paper

## The Breakthrough ‘Aha’ Moment

这一章节是一篇快速阅读论文的课程。我们将用简单的语言来解读这篇论文，然后我们将分解其中的关键概念和收获。

DeepSeek R1 在语言模型训练方面代表了一个重要的进步，特别是在通过强化学习发展推理能力方面。论文介绍了一种新的强化学习算法，称为组相对策略优化（GRPO）。

在下一章节中，我们将在此基础上进行实践，实现 GRPO。

初始目标是探索是否可以通过纯强化学习来发展推理能力，而**无需监督微调**。在此之前，所有流行的 LLM 都需要一些**监督微调**，我们在第 11 章中探讨了这一点。

在 R1-Zero 的训练中最令人瞩目的发现之一是 ‘Aha’ Moment 的出现。这种现象在解决问题时与人类突然的顿悟有些相似。它是这样运作的：

1. **Initial Attempt**: The model makes an initial attempt at solving a problem
初始尝试：模型最初尝试解决一个问题
1. **Recognition**: It recognizes potential errors or inconsistencies
识别：它识别潜在的错误或不一致
1. **Self-Correction**: It adjusts its approach based on this recognition
自我修正：它会根据这种识别调整其方法
1. **Explanation**: It can explain why the new approach is better
解释：它可以解释为什么新的方法更好
这项突破与学习者产生了共鸣，感觉就像是“恍然大悟”的时刻。它展示了学习而不是简单的记忆，所以让我们花一点时间想象一下“恍然大悟”的感觉。

这种能力是自然地从强化学习训练中产生的，没有被明确编程，这展示了学习而不是仅仅对训练数据中的过程的记忆。

## The Training Process  训练过程

Training R1 was a multi-phase process. Let’s break down the phases and the key innovations in each phase.

训练 R1 是一个多阶段的过程。让我们分解每个阶段及其关键创新。

The final process results in two models:

最终过程产生了两个模型：

- DeepSeek-R1-Zero: A model trained purely using reinforcement learning.
DeepSeek-R1-Zero：一个仅使用强化学习训练的模型。
- DeepSeek-R1: A model that builds on the foundation of DeepSeek-R1-Zero and adds supervised fine-tuning.
DeepSeek-R1：一个基于 DeepSeek-R1-Zero 的基础并添加了监督微调的模型。
Feature

DeepSeek-R1-Zero

DeepSeek-R1

Training Approach  训练方法

Pure RL  纯 RL

Multi-phase (SFT + RL)  多阶段（SFT + RL）

Fine-tuning  微调

None  无

Supervised fine-tuning  监督微调

Reasoning Capability  推理能力

Emergent  自发的

Enhanced  提升

AIME Performance  AIME 性能

71.0%

79.8%

Key Characteristics  关键特征

Strong reasoning but readability issues
强大的推理但可读性问题

Better language consistency and readability
更好的语言一致性与可读性

虽然 DeepSeek-R1-Zero 展示了纯强化学习在开发推理能力方面的潜力，但 DeepSeek-R1 在此基础上采取了更为平衡的方法，既重视推理性能，也重视易用性。

训练过程包括四个阶段：

1. Cold Start Phase  冷启动阶段
1. 这个阶段旨在为模型的可读性和响应质量奠定坚实的基础。它使用**来自 R1-Zero 的高质量样本小数据集来微调 V3-Base 模型**。从 DeepSeek-V3-Base 模型开始，团队使用了数千个经过验证的高质量样本来自 R1-Zero 进行**监督微调**。这种创新的方法使用**少量但高质量的数据集**来建立强大的基线可读性和响应质量。
1. Reasoning RL Phase  推理强化学习阶段
1. 推理 RL 阶段专注于在数学、编程、科学和逻辑等领域开发核心**推理能力**。该阶段采用**基于规则的强化学习**，**奖励直接与解题正确性挂钩**。
1. crucially，这一阶段的所有任务都是“可验证”的，因此我们可以检查模型的答案是否正确。例如，在数学的情况下，我们可以使用数学求解器来检查模型的答案是否正确。
1. 这一阶段特别具有创新性的是其直接优化方法，消去了单独的奖励模型的需要，简化了训练过程。
1. Rejection Sampling Phase 拒绝采样阶段
1. 在拒绝采样阶段，模型生成样本，然后通过质量控制过程进行筛选。**DeepSeek-V3 作为质量评判者**，评估输出范围广泛，不仅限于纯粹的推理任务。筛选后的数据用于监督微调。这一阶段的创新之处在于能够结合多种质量信号以确保高标准的输出。
1. Diverse RL Phase  多样化强化学习阶段
1. 最终的多样化 RL 阶段使用一种复杂的混合方法处理多种任务类型。对于确定性任务，它采用基于规则的奖励，而主观任务则通过LLM反馈进行评估。该阶段旨在通过其创新的混合奖励方法实现人类偏好对齐，结合基于规则系统的精确性和语言模型评估的灵活性。
## The Algorithm: Group Relative Policy Optimization (GRPO)

现在我们已经对训练过程有了很好的理解，让我们来看看用于训练模型的算法。

作者将 GRPO 描述为模型微调的一个突破：

![image](Uk1Kd2i9Io78VhxRghfco1DdnAe.png)

GRPO 的新颖之处在于其能够“**直接优化偏好矫正**”。这表示了一种更直接和高效的途径来使模型与期望的输出对齐，与传统的强化学习算法（如 PPO）相比更为不同。让我们通过其三个主要组成部分来分解 GRPO 的工作原理。

### 1、Group Formation: Creating Multiple Solutions
组建：创建多个解决方案

GRPO 的第一步非常直观——它类似于学生通过尝试多种方法来解决一个难题。当给定一个提示时，模型不只是生成一个回应；相反，它会创建多个尝试来解决同一个问题（通常是 4、8 或 16 种不同的尝试）。

想象你正在训练一个模型来解决数学问题。对于关于农场里**数非下蛋母鸡的问题**，模型可能会生成几种不同的解答：

- 一种解决方案可能是逐步分解问题：首先统计总鸡数，然后减去公鸡，最后计算非下蛋母鸡
- 另一个可能使用不同的但同样有效的办法
- 一些尝试可能会包含错误或效率较低的解决方案
所有这些尝试都被一起保留下来，就像有多个学生的答案可以比较和学习一样。

![image](V5AcdPtv8ocDT7x3SH3cWNl7n1f.png)

### 2、Preference Learning: Understanding What Makes a Good Solution
偏好学习：理解什么是好的解决方案

这里 GRPO 真的在简洁性上表现出色。与其他需要额外奖励模型来预测解决方案可能有多好的 RLHF 方法不同，GRPO 可以使用任何函数或模型来评估解决方案的质量。例如，我们可以使用长度函数来奖励较短的回答，或者使用数学求解器来奖励准确的数学解决方案。

评估过程会从多个方面来审视每个解决方案：

- 最终答案是否正确？
- 解决方案是否遵循了正确的格式（例如使用了正确的 XML 标签）？
- 逻辑是否与提供的答案相符？
这种方法特别巧妙的地方在于它处理评分的方式。它不是仅仅给出绝对评分，而是对每个组内的奖励进行了标准化。它使用了一个简单但有效的公式来进行组间相对优势估计：

`Advantage = (reward - mean(group_rewards)) / std(group_rewards)`

这种归一化就像曲线 grading，但适用于 AI。**它帮助模型理解组内哪些解决方案比 peers 更好或更差，而不是仅仅看绝对分数。**

### 3、Optimization: Learning from Experience
优化：从经验中学习

最后一步是让模型根据评估一组解决方案所学到的内容来改进。

这个过程既强大又稳定，主要基于两个原则：

- 它鼓励模型产生更多像成功的解决方案那样的结果，同时远离不太有效的做法
- 它包含一个安全机制（称为 KL 散度惩罚），防止模型一次性发生太大的变化
这种方法比传统方法更稳定，因为它：

- 它会同时考虑多个解决方案，而不是一次只比较两个
- 基于组的归一化有助于防止奖励缩放问题
- KL 惩罚起到了安全网的作用，确保模型在学习新事物的同时不会忘记已经知道的内容
GRPO’s key innovations are:

GRPO 的关键创新是：

- Learning directly from any function or model, eliminating the reliance on a separate reward model.
直接从任何函数或模型中学习，消除对独立奖励模型的依赖。
- Group-based learning, which is more stable and efficient than traditional methods like pairwise comparisons.
基于组的学习比传统的成对比较等方法更稳定和高效。
### GRPO 算法伪代码

```
Input: 
- initial_policy: Starting model to be trained
- reward_function: Function that evaluates outputs
- training_prompts: Set of training examples
- group_size: Number of outputs per prompt (typically 4-16)

Algorithm GRPO:
1. For each training iteration:
   a. Set reference_policy = initial_policy (snapshot current policy)
   b. For each prompt in batch:
      i. Generate group_size different outputs using initial_policy
      ii. Compute rewards for each output using reward_function
      iii. Normalize rewards within group:
           normalized_advantage = (reward - mean(rewards)) / std(rewards)
      iv. Update policy by maximizing the clipped ratio:
          min(prob_ratio * normalized_advantage, 
              clip(prob_ratio, 1-epsilon, 1+epsilon) * normalized_advantage)
          - kl_weight * KL(initial_policy || reference_policy)
          
          where prob_ratio is current_prob / reference_prob

Output: Optimized policy model
```

该算法展示了 GRPO 如何通过结合基于组的优势估计与策略优化，在通过裁剪和 KL 散度约束保持稳定性的情况下实现这一目标。

## Limitations and Challenges of GRPO

虽然 GRPO 在语言模型的强化学习方面代表了一个重要进步，但了解其局限性和挑战也很重要：

- 生成成本：与只生成一两个完成的方法相比，为每个提示生成多个完成（4-16 个）会增加计算需求。
- 批量大小限制：需要一起处理一组完成会限制有效的批量大小，增加训练过程的复杂性，甚至可能减慢训练速度。
- 奖励函数设计：训练质量在很大程度上取决于精心设计的奖励函数。设计不良的奖励可能导致意外行为或优化错误的目标。
- 组大小权衡：选择最优的组大小需要在解决方案的多样性与计算成本之间进行平衡。样本太少可能无法提供足够的多样性，而样本太多则会增加训练时间和资源需求。
- KL 散度调优：找到 KL 散度惩罚的最佳平衡点需要仔细调优——太高的话模型将无法有效学习，太低的话它可能会远离其初始能力。
## Conclusion

The DeepSeek R1 paper represents a significant milestone in language model development. The Group Relative Policy Optimization (GRPO) algorithm has demonstrated that pure reinforcement learning can indeed develop strong reasoning capabilities, challenging previous assumptions about the necessity of supervised fine-tuning.

Perhaps most importantly, DeepSeek R1 has shown that it’s possible to balance high performance with practical considerations like cost-effectiveness and accessibility. The successful distillation of the model’s capabilities across different sizes, from 1.5B to 70B parameters, demonstrates a path forward for making advanced AI capabilities more widely available.

DeepSeek R1 论文代表了语言模型开发中的一个重要里程碑。组相对策略优化（GRPO）算法表明，纯粹的强化学习确实可以发展出强大的推理能力，这挑战了之前关于监督微调必要性的假设。

或许最重要的是，DeepSeek R1已经表明，在高性能与成本效益和可及性等实际考量之间实现平衡是可行的。该模型能力在不同参数规模（从 15亿 1.5B 到 700亿 70B参数）下的成功蒸馏，为更广泛地提供先进的人工智能能力指明了方向。

## 相关

- [[GRPO 深度理解|GRPO 深度理解]]
- [[PPO 原理|PPO 原理]]
- [[DeepSeek-R1|DeepSeek-R1]]
- [[verl 概述|verl 概述]]
- [[DeepSeek-Math|DeepSeek-Math]]
