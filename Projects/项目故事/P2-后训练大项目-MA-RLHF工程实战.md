---
title: P2：后训练工程实战——SFT→DPO→GRPO 完整 Pipeline
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, post-training, SFT, DPO, GRPO, verl, RLHF, DeepSeek-R1]
brief: 系统实现了完整的后训练 pipeline：SFT（全参+LoRA）→ DPO 偏好对齐 → GRPO 强化学习推理增强，用 verl+Ray 跑分布式实验，复现了 DeepSeek-R1 训练流程。故事线：从业务 Agent 的 prompt 天花板出发，一路深入到模型训练层。
related:
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解]]"
  - "[[AI/3-LLM/RL/RLHF-DPO-2026-技术全景]]"
  - "[[Projects/MA-RLHF/lc9/lc9-03-verl-R1复现全流程]]"
  - "[[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解]]"
---

# P2：后训练工程实战——SFT→DPO→GRPO 完整 Pipeline

---

## 故事线（面试完整版，5-8 分钟）

> 这是一个娓娓道来的版本。技术点在故事里，不是独立罗列。

---

### 第一幕：发现问题

在美团做了两年业务 Agent，有一件事一直困扰我。

我们的分析 Agent——一个挂着取数工具、能自主写分析报告的系统——在大多数情况下表现不错，但每隔一段时间就会在某个场景上突然变差。不是崩溃，是"变傻"了——该追问的时候不追问，该归因的时候停在表层，给出的建议逻辑对不上。

我们的第一反应是 prompt 工程：加细节、加例子、加约束。确实有用，但每次能修一个问题，下次又冒出一个新的。我慢慢意识到：**prompt 在修症状，不在修根因。** 真正的问题是模型在多步推理上有系统性的短板，这不是 prompt 能弥补的。

就在这个时候，DeepSeek-R1 发布了。我当时看论文看到一半就停下来想：**GRPO + 可验证 reward，这是从模型层解决多步推理问题的路**。我决定系统搞清楚这整条后训练的链路。

---

### 第二幕：进场——先把基础打扎实

第一个问题就很实际：从哪里开始？

我没有直接跳到 GRPO，而是从头来。我需要搞清楚整条 pipeline——SFT 是干什么的，DPO 解决什么问题，GRPO 在哪个位置发力——如果只懂最后一步，前面出了问题根本不知道怎么查。

**先跑 SFT。**

选了全参 SFT，用 Accelerate + DeepSpeed ZeRO-1，跑 7B 的模型。数据是 CoT 格式的数学推理数据，目的是让模型学会 `<think>...</think>` 这个格式——先思考，再回答。

这步踩了一个很蠢但很难发现的坑：**Chat Template 不对齐**。训练时用的 system prompt 格式，和推理时 tokenizer 的 `apply_chat_template` 默认格式不一样。模型训出来之后，拿去推理，结果一塌糊涂——它根本不知道在什么格式下工作。问题不报错，只会让输出看起来奇怪。查了很久，才发现是这里。

这个坑让我意识到一件事：后训练的问题很多不是算法问题，是工程对齐问题。训练和推理的环境必须一致，细到 tokenizer 的每个参数。

---

### 第三幕：DPO——以为是解法，发现是过渡

SFT 之后，模型会说话了，但说得还不够好——格式对了，但质量参差不齐。下一步是 DPO。

DPO 的思路很直觉：给模型看配对数据——同一个问题，一个好回答、一个差回答——让模型学会偏向好的。数学上，DPO 等价于在 actor 和 reference model 之间做比值，让 chosen 的 log prob 相对 ref 升高，让 rejected 的下降，不需要显式训练 reward model。

跑下来效果有，GSM8K 从 62% 涨到 65%。

但有个问题一直让我不舒服：**DPO 是离线的**。数据在训练开始前就定死了，模型学的是"历史上哪种回答更好"，而不是"在当前的推理过程中，哪条路能走通"。这个区别在静态任务上不明显，但我知道，如果要把这套东西用在 Agent 工具调用上——模型需要在真实环境里探索——DPO 是不够的。

这让我有了一个认知：DPO 是从 SFT 到 RL 的过渡，不是终点。

---

### 第四幕：GRPO——遇到真正的对手

接下来是 GRPO，也是这个项目里最烧脑的部分。

**为什么不用 PPO？**

不是 PPO 不好，是 PPO 的工程成本在这里不合算。PPO 需要同时维护四个模型：actor（被训练的策略）、reference（基线）、critic（价值估计）、reward model（打分）。光 critic 的训练就是一个独立的难题——它需要估计每个状态的价值，训练不稳定，调起来很费劲。对于数学推理这种有精确验证器的任务，这些成本都是不必要的。

GRPO 的思路更干净：**同一个问题，采样一组（group）回答，用这组回答的相对 reward 来估计每个回答有多好，省掉 critic。** 去掉了两个模型，显存省了将近三成，训练也更稳定。

选定了算法，下一个问题是框架：用 verl。

**verl 解决了什么问题？**

在 RL 训练里，有一个天然的矛盾：训练需要用 DeepSpeed 这类框架（对梯度计算高度优化），采样需要用 vLLM（PagedAttention、continuous batching，吞吐量比训练框架高好几倍）。如果两者用同一套进程，要么牺牲采样速度，要么牺牲训练效率。

verl 把这两件事解耦了：Actor 进程负责训练，Rollout 进程（vLLM）负责采样，两者通过 weight sync 保持一致。

这个设计在论文里看起来很优雅。但实际跑起来，我碰到了第一个严重的问题：**verl 0.7.0 和 vLLM 0.11.0 不兼容**。vLLM 的 API 更新很快，经常有破坏性变更。weight sync 时 tensor shape 对不上，worker 启动直接失败，报错信息非常不友好，完全看不出是版本问题。

花了很长时间定位，最后解法说起来简单：锁版本，写了一个启动前的 version check 脚本，强制验证环境。但这次经历让我明白，分布式 RL 训练的工程噪音非常高，很多时间花在和框架本身搏斗，不是在做算法。

---

### 第五幕：Reward——这才是真正的核心

环境搭起来，框架跑通，以为可以开始正式训练了。但接下来遇到了整个项目里最重要的问题：**reward 设计。**

最初的版本非常简单：答案对了给 1，错了给 0。跑了一段时间，loss 在下降，但效果没什么变化。

问题很明显：**reward 太稀疏了**。模型生成了一大段推理，错了，reward 0，它不知道是哪步出了问题。训练信号几乎没有。

然后做了第一个改进：**加 format reward**。

```
r_format = 0.2   # 只要有 <think>...</think> 结构
r_answer = 1.0   # 答案精确匹配
r_total = r_format + r_answer
```

这个改动看起来微小，但效果立竿见影——训练早期的 loss 曲线稳定多了。原因是：format reward 给了模型一个更早的正反馈信号，让它先学会"用 CoT 格式思考"，再去学"CoT 里怎么推理"。顺序对了，学习就顺了。

有了这个体会之后，我开始理解为什么 reward 设计比算法本身更重要——算法的框架是固定的，reward 决定了模型往哪个方向学。

训练的最终结果：GSM8K 从 SFT 的 62%，经过 DPO 涨到 65%，再经过 GRPO 完整训练涨到 78%。GRPO 在 500 steps 后效果就明显超过 DPO，说明 RL 的 online 探索在这类有明确验证器的任务上有本质优势。

---

### 第六幕：跑完之后，我在想什么

这个项目让我对后训练有了一个系统认知，但更重要的是让我想清楚了一件事：**后训练的价值不只是提升模型在 benchmark 上的分数，而是让模型能习得你想要的行为模式**。

数学推理是最简单的情况，因为有精确验证器。但如果要训练业务 Agent——比如我们的分析 Agent，"好分析"很难精确定义——reward 设计会复杂得多，RL 的挑战会更大。

我开始思考：分析 Agent 的 reward 应该怎么设计？怎么把"这个分析有没有找到根因"这种模糊判断，转化成可训练的 reward signal？这是我下一步想深入的方向。

---

## 技术路径深化（面试追问完整版）

### a. SFT 阶段：全参 vs LoRA vs QLoRA 选型逻辑

**三种方案的核心差异：**

| 方案 | 显存占用 | 效果上限 | 适用场景 |
|------|---------|---------|---------|
| 全参 SFT | 最高（约模型大小×18，混精度+梯度+优化器） | 最高 | 最终版本、有足够 GPU |
| LoRA | 中（冻结基座，只训练 adapter） | 略低于全参（差距通常 1-2%） | 快速验证、资源受限 |
| QLoRA | 最低（4bit 量化基座 + LoRA） | 再低一点（量化误差） | 单卡实验、消费级 GPU |

**我的选择逻辑：**
- 第一轮用 QLoRA：验证数据格式/chat template/loss 曲线——单卡跑通，省掉 90% 调试时间
- 确认 pipeline 通了再切全参，这时候才是在比算法，不是在比工程

**LoRA rank/alpha 怎么调：**
- 起点：rank=16, alpha=32（alpha/rank=2）
- rank 太小（4-8）：adapter 容量不够，复杂任务学不动
- rank 太大（128-256）：慢，且数据量有限时过拟合
- alpha/rank 控制 adapter 更新的 scaling，影响有效学习率量级
- 面试官追问"为什么不用 DoRA"：DoRA 分解方向+大小更新，少量数据场景比 LoRA 好一点，但工程复杂，项目里标准 LoRA 已够用

**面试官追问"为什么不用 QLoRA 做最终版本"：**
量化误差在最终性能上是可测量的损失。7B 全参和 QLoRA 的 GSM8K 差距约 1-2%，乘上 DPO/GRPO 的复合效应，最终差距会被放大。实验预算够时，全参是更稳的选择。

---

### b. DPO 阶段：DPO vs SimPO vs IPO vs ORPO 区别与选型

**各方案的核心机制差异：**

| 方案 | 是否需要 reference model | 核心优化目标 | 主要问题 |
|------|------------------------|------------|---------|
| DPO | 是 | 最大化 chosen - rejected 的 logprob 差 | length bias（偏好更长的 chosen） |
| SimPO | 否（用 sequence 平均 logprob 作 reward） | 直接最大化 margin | 不适合有明确 ref 的场景 |
| IPO | 是 | 加 KL 约束避免 reward hacking | 超参数更多，调参复杂 |
| ORPO | 否（在 SFT loss 上直接叠 odds ratio） | SFT + 对齐一步完成 | 不适合已有 SFT checkpoint 的场景 |

**为什么选 DPO：**
我的场景是 SFT 之后接 DPO——reference model 就是 SFT 后的 checkpoint，pipeline 自然衔接。SimPO 更适合没有明确 ref 的场景。

**DPO 的 length bias 问题：**
DPO 会偷学捷径：chosen 往往比 rejected 更长，模型发现"说得越多越容易被偏好"。应对方案：
1. 构造偏好数据时控制 chosen/rejected 长度比（不要系统性让 chosen 更长）
2. 加 length penalty：`r_total = r_dpo - β × len(output)`
3. 用 SimPO（sequence-level 平均 logprob 天然规避 length 问题）

**DPO 的 offline 限制：**
DPO 用固定偏好数据集，不做 online 探索。模型学到一定程度后，它生成的新回答和数据集分布越来越远，梯度信号有效性下降。所以 DPO 通常只做 1-2 个 epoch，多了效果反降。这是 DPO 的根本局限，GRPO 的 online 采样解决了这个问题。

---

### c. GRPO 阶段：reward 设计迭代路径

**为什么不用 PPO（补充详版）：**
PPO 需要 4 个模型（actor/ref/critic/reward_model）。Critic 训练是独立难题——需要估计每个状态的期望 return，本身需要大量样本和调参；对于有精确验证器的任务，成本完全没必要。GRPO 用 group 内相对 reward 替代 critic 估计——Monte Carlo 近似，同一 prompt 多个采样，reward 均值即期望 return 的无偏估计。

**Reward 设计三个阶段：**

**阶段一（稀疏，失败）：**
```python
reward = 1.0 if answer_correct else 0.0
```
问题：200 个 token 的推理错了，reward=0，模型不知道哪步出了问题。训练信号几乎为 0，loss 下降但效果不变。

**阶段二（格式+结果，成功）：**
```python
r_format = 0.2  # 有 <think>...</think> 格式就给
r_answer = 0.8  # 答案精确匹配
reward = r_format + r_answer
```
关键洞察：format reward 提供比 answer reward 更早的正反馈——模型先学会"用 CoT 格式思考"，再学"CoT 里怎么推理"。学习有了支撑点，曲线立刻稳定。

**阶段三（进阶）：step-level reward：**
更进一步是给推理链每一步单独打分（process reward），用 PRIME 或 AgentPRM——正确推理步骤给部分 credit，不必等到最后才知道对错。但实现复杂度高（需要 step 级别标注或隐式 PRM），数学推理任务上性价比不如 format+answer 两级奖励。

**面试官追问"为什么不用 DAPO/REINFORCE++/Dr-GRPO"：**
DAPO 解决 entropy collapse（训练后期策略过于确定性）；REINFORCE++ 去掉 baseline 估计方差；Dr-GRPO 修 std 归一化的 difficulty bias。三个都是 GRPO 的工程改进，效果差异通常 1% 以内。学习项目标准 GRPO 足够；生产系统才值得逐一 ablation。

---

### d. 分布式：verl Actor-Rollout 解耦的通信开销分析

**weight sync 的具体机制：**

verl 里 Actor（DeepSpeed ZeRO-3）和 Rollout（vLLM）是两套独立进程。每完成一个 rollout batch，需要把 Actor 最新参数同步给 Rollout worker：

```
1. Actor 训练完一个 batch，参数更新
2. DeepSpeed ZeRO-3 参数分散在多个 GPU
   → All-Gather 把完整参数汇聚到各 GPU
3. 通过 NCCL broadcast 发给 vLLM workers
4. vLLM 加载新参数，继续采样
```

**通信量估算：**
7B 参数，bf16 精度，每次 weight sync 约传输 **14GB** 数据。InfiniBand 100Gb/s 机器上，单次 sync 约 **1-2 秒**。

**什么时候通信是瓶颈：**
rollout batch 太小（如每 prompt 只采 4 条），rollout 很快做完，weight sync 相对开销就大。建议每 prompt 至少采 8 条（G=8），让 vLLM 批量吞吐优势充分发挥，稀释同步开销。

**版本兼容问题（真实踩坑）：**
verl 0.7.0 + vLLM 0.11.0 不兼容——vLLM weight loading API 做了破坏性变更，weight sync 时 tensor shape 对不上，worker 启动直接失败。错误信息完全不指向版本问题。解法：锁版本，写启动前的 environment check 脚本强制验证环境。

**面试官追问"为什么不用 Ray RLlib 或 OpenRLHF"：**
Ray RLlib 是通用 RL 框架，没有专门针对 LLM 优化，没有 Actor/Rollout 解耦设计，采样和训练混在一起，吞吐量低。OpenRLHF 设计类似 verl，但生态和文档没 verl 成熟；两者都可以用，选 verl 主要是 DeepSeek 官方用这套，有更多实战参考。

---

## 快速技术速查（面试追问备用）

**"GRPO 的 advantage 是怎么算的？"**
同一个 prompt 采样 G 个回答，`advantage_i = (r_i - mean(r)) / std(r)`，group 内归一化，省掉 critic。

**"verl 的 Actor 和 Rollout 解耦是什么意思？"**
Actor 用 DeepSpeed 跑训练（梯度计算），Rollout 用 vLLM 跑采样（高吞吐），两者是独立进程，通过 weight sync 同步权重。切换时有通信开销，但采样速度提升 3-5x，整体更合算。

**"DPO 和 GRPO 在什么情况下各自适合？"**
DPO 适合有高质量偏好标注数据、任务相对静态的场景（对话对齐）。GRPO 适合有可验证 reward、需要 online 探索的场景（数学、代码、工具调用）。两者不互斥，SFT→DPO 是常见的预训练前置，GRPO 接在后面继续优化。

**"SFT 全参和 LoRA 怎么选？"**
快速实验用 QLoRA（资源省，迭代快）；最终版本用全参（效果上限高）。LoRA rank 的选择是个坑——rank 太小学不到，rank 太大跑慢了又有过拟合风险，通常从 rank=16 开始调。

---

## See Also

- [[Projects/项目故事/P1-xtrain-分布式预训练工程]] — 基础设施支撑
- [[Projects/项目故事/P5-分析Agent-从ReAct到RL训练闭环]] — 把 GRPO 用在 Agent
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解]]
- [[AI/3-LLM/RL/RLHF-DPO-2026-技术全景]]
- [[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解]]
- [[AI/3-LLM/Frameworks/verl/训练后端]]
