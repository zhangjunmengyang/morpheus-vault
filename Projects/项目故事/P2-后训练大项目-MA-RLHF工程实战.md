---
title: P2：后训练工程实战——SFT→DPO→GRPO 完整 Pipeline
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, post-training, SFT, DPO, GRPO, verl, RLHF, DeepSeek-R1]
brief: 系统实现了完整的后训练 pipeline：SFT（全参+LoRA）→ DPO 偏好对齐 → GRPO 强化学习推理增强，用 verl+Ray 跑分布式实验，复现了 DeepSeek-R1 训练流程，并深入理解了每个阶段的工程 trade-off。
related:
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解]]"
  - "[[AI/3-LLM/RL/RLHF-DPO-2026-技术全景]]"
  - "[[Projects/MA-RLHF/lc9/lc9-03-verl-R1复现全流程]]"
  - "[[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解]]"
---

# P2：后训练工程实战——SFT→DPO→GRPO 完整 Pipeline

> **一句话定位**：这不是读论文，是系统地把 SFT、DPO、GRPO 都跑通了，理解了每个阶段的工程难点，用 verl 在真实 GPU 集群上复现了 DeepSeek-R1 的训练流程。

---

## 背景故事（面试口径）

> 怎么引入：

"在业务里做了两年 Agent 应用，我越来越感受到一件事：prompt 工程有天花板。你可以把 prompt 写得多精妙，但如果底层模型在某类任务上有系统性的缺陷——比如多步推理、工具调用的格式稳定性——prompt 是救不了的。

真正的解法在模型端：通过后训练让模型学会你想要的行为。DeepSeek-R1 出来之后，我觉得这个方向彻底清晰了——GRPO 加上可验证的 reward，就能让模型学会深度推理。

所以我系统地学习并实现了完整的后训练 pipeline。这个项目从 SFT 开始，一路到 DPO 再到 GRPO，用 verl 框架在 8-GPU 环境跑了分布式实验。"

---

## 项目技术架构

```
完整后训练 Pipeline：

┌─────────────────────────────────────────────────────────┐
│ Phase 1：Continue Pretraining                           │
│   基础模型 → 领域推理语料继续预训练（可选）             │
│   目的：给模型打上 CoT 推理的语料基础                   │
└─────────────┬───────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2：SFT（Supervised Fine-Tuning）                  │
│   sft.py：全参 SFT（Accelerate + DeepSpeed ZeRO-1）    │
│   sft_qlora.py：QLoRA 低资源版（4bit bitsandbytes）    │
│   目的：让模型学会 CoT 格式，<think>...</think> 结构   │
└─────────────┬───────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3：DPO（Direct Preference Optimization）          │
│   dpo.py：TRL DPOTrainer                               │
│   目的：偏好对齐，从 chosen/rejected 对学习人类偏好    │
└─────────────┬───────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4：GRPO（Group Relative Policy Optimization）     │
│   verl/main_ppo.py（verl 0.7.0 + Ray + vLLM 0.11.0）  │
│   Reward：GSM8K 数学答案匹配（可验证 reward）          │
│   目的：强化推理能力，让模型学会 think → verify → 答  │
└─────────────────────────────────────────────────────────┘
```

---

## 技术深度（面试展开）

### SFT 阶段的关键工程决策

**全参 vs LoRA 怎么选？**

```
场景分析：

全参 SFT（Accelerate + DeepSpeed ZeRO-1）：
  优点：效果上限高，不损失模型能力
  缺点：显存开销大（7B 需要 ~60GB GPU 内存）
  适用：有足够 GPU、想要最优效果

QLoRA（4bit 量化 + LoRA）：
  优点：单卡 A100 80GB 可以跑 13B 模型
  缺点：量化本身引入精度损失，LoRA rank 选不好效果差
  适用：资源受限场景，快速实验

我的选择依据：
  - 探索实验 → QLoRA（省资源，快速迭代）
  - 最终版本 → 全参（效果最优）
```

**踩坑：Chat Template 必须对齐**

"SFT 里踩过一个坑——Chat Template 不对齐。训练用的模板和推理时用的模板不一样，结果模型在 fine-tuning 后变傻了。深查之后发现是 tokenizer 的 apply_chat_template 在训练和推理时用了不同的系统 prompt 格式，导致模型学到了错误的格式。这种问题很隐蔽，不会报错，只会让结果变差。"

### DPO 阶段的理解

**DPO 的数学本质是什么？**

```
DPO Loss：
  L = -E[ log σ( β × log(π(y_w|x)/π_ref(y_w|x)) - β × log(π(y_l|x)/π_ref(y_l|x)) ) ]

等价于：
  让 chosen 轨迹的 log prob 相对 ref 升高
  让 rejected 轨迹的 log prob 相对 ref 降低

为什么不需要 reward model？
  DPO 把 reward 隐式地定义在了 π/π_ref 的比值里
  省掉了 PPO 里 Critic 网络的训练
```

**什么时候 DPO 会失效？**

"DPO 的假设是你有高质量的 chosen/rejected 配对数据。如果数据里 chosen 和 rejected 质量差异不大，DPO 的训练信号就很弱。另外，DPO 的分布 shift 问题——训练分布和推理分布的差异——在 off-policy 设置下会更严重。这也是为什么后来有 SimPO、ORPO 这些改进。"

### GRPO 阶段（核心）

**GRPO vs PPO 的工程 trade-off**

```
PPO：
  需要 4 个模型同时在显存里：actor + critic + ref + reward
  Critic 训练需要价值估计，不稳定
  适合：有明确 reward model 的场景

GRPO（Group Relative Policy Optimization）：
  去掉 Critic，用 Group 内的 relative reward 估计 advantage
  advantage_i = (r_i - mean(r)) / std(r)
  只需要 2 个模型：actor + ref
  适合：有可验证 reward 的场景（数学、代码）

工程优势：
  显存节省 ~30%（去掉 Critic 网络）
  训练更稳定（不依赖 Critic 收敛）
```

**为什么 GRPO 在数学任务上效果好？**

"数学任务有可验证 reward——答案对不对是明确的，不需要 reward model 的主观判断。这消除了 Reward Hacking 的主要来源：你没办法骗一个精确匹配的验证器。GRPO 通过 group 内对比，让模型自己发现哪种推理路径能得到正确答案，这比 DPO 的 offline 偏好学习更直接。"

**verl 框架的工程实现**

```
verl 三层架构：
  Layer 1: Ray（分布式执行引擎）
    → RayResourcePool 管理 GPU 资源
    → Worker Actor 封装 GPU 进程

  Layer 2: Worker 抽象
    → ActorWorker（策略模型，负责生成和训练）
    → RolloutWorker（vLLM，负责高速采样）
    → 二者可以分离到不同 GPU 组

  Layer 3: 算法实现
    → PPOTrainer / GRPOTrainer
    → 管理 rollout → compute reward → compute advantage → update 的循环

关键设计：Actor 和 Rollout 解耦
  → 训练时用 Megatron/DeepSpeed（高效梯度计算）
  → 采样时用 vLLM（高效 inference，PagedAttention）
  → 通过 weight sync 保持两者一致
```

**实验中遇到的坑：vLLM 版本兼容**

"verl 0.7.0 和 vLLM 0.11.0 的兼容性问题——vLLM 更新很快，API 经常破坏性变更。在对齐版本之前，遇到过 worker 启动失败、weight sync 时 tensor shape 不匹配等问题。解决方案是锁版本，并且写了一个 version check 脚本在训练开始前验证环境。"

### Reward 设计的关键决策

**数学任务的 reward 怎么设计**

```
简单匹配（baseline）：
  r = 1 if answer == gt else 0

问题：稀疏 reward，模型很难学

改进：Format reward + Answer reward
  r_format = 0.2 if has_thinking_tag else 0  # 鼓励 CoT 格式
  r_answer = 1.0 if correct else 0
  r_total = r_format + r_answer

进一步改进（针对推理质量）：
  对 <think> 内容长度有 soft reward
  对明显的 copy-paste 有惩罚
```

"Reward 设计是 GRPO 效果差异最大的地方，比算法本身的影响还大。我们发现 format reward 对训练早期稳定性很关键——如果模型一开始不学 CoT 格式，后面答案 reward 很难拉回来。"

---

## 实验结果

**基准实验（GSM8K 数学推理）**

| 方法 | GSM8K Accuracy |
|------|---------------|
| Base 模型（无后训练）| ~45% |
| + SFT CoT 数据 | ~62% |
| + DPO 偏好对齐 | ~65% |
| + GRPO（500 steps）| ~73% |
| + GRPO（完整训练）| ~78% |

**关键观察**：GRPO 在 500 steps 后效果就明显超过 DPO，说明 RL 的 online learning 比 offline 偏好学习更适合这类有明确验证器的任务。

---

## 深度认知（面试加分点）

**后训练的本质是什么？**

"后训练不是教模型知识——预训练已经把知识装进去了。后训练是教模型什么时候说什么，怎么组织输出，以及面对困难问题时要坚持推理而不是直接猜。从这个视角看，SFT 是格式化，DPO 是校正，GRPO 是内化行为。"

**为什么后训练比 prompt 更根本？**

"prompt 是运行时的，模型的权重没变。后训练是把你想要的行为写进权重。当这个行为足够复杂（比如多步推理、工具调用的稳定性），只靠 prompt 是不够的——模型会在新的分布上退化。"

---

## See Also

- [[Projects/项目故事/P1-xtrain-分布式预训练工程]] — 基础设施支撑
- [[Projects/项目故事/P5-分析Agent-从ReAct到RL训练闭环]] — 把 GRPO 用在 Agent
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解]]
- [[AI/3-LLM/RL/RLHF-DPO-2026-技术全景]]
- [[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解]]
- [[AI/3-LLM/RL/SFT/SFT-手撕实操]]
- [[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF]]
- [[AI/3-LLM/Frameworks/verl/训练后端]]
