---
brief: "RL 概览——B站 Reasoning Model 系列课程入口笔记；从基础 RL 概念过渡到 LLM 推理模型（o1/R1 类）的训练原理，是理解 RLVR 如何驱动推理能力涌现的视频学习入口。"
title: "001"
type: concept
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/concept
---
# 001

https://www.bilibili.com/video/BV15yA3eWE5b?vd_source=8f7f521ad9c1872425eb493f20f42f57

# Reasoning model

Reasoning model：本质上 think 还是模型的输出，有一些工作证明了：

- think aloud 的效果是更好的
- 直觉上来看回答复杂问题不应该和简单问题同样的计算量
训练设计到构建的数据形式：

- COT（reasoning process）
- Questions
- Answers
#  pre-train vs.  SFT vs. RL

![image](GJcad3R4Eoi5WGxMc3Uc09ZxnTg.png)

- Pre-training 本质上也是一种激励，不过是一种弱激励
# 幻觉

![image](RaUPdNrGVov1J7xfLFbc7XUGnXc.png)

- 幻觉怎么解决：用 reward 的方式，如果 unhedged（非含糊其辞，说地信誓旦旦） and wrong，就狠狠惩罚
# R1 训练

![image](ESHWdiucSoS4JCxP9YicMpvjnCO.png)

整体流程：

1. 先用 v3 通过 GRPO，来得到 R1-zero，目的是产生高质量的 CoT Data，奖励 Accuracy、Formatting
1. 然后基于 R1-zero 进行 cold start 对 v3 重新进行 RL，奖励 Accuracy、Formatting、语言一致性
# RL 一些基本概念

![image](W9r8dcAwHo3y7Qxiebsc2qLEnqd.png)

# GRPO

![image](WNyvdm39LolcXExNZizcJk7qnNb.png)

- R1 和 k1.5 都舍弃了 value function（PPO 的 GAE），都舍弃了过程奖励模型。
# PG

![image](CKrEdLLwBogTDhxIHRCcMoqRnAd.png)

# 开源复现

![image](PrYQdYnkIoirP6xZxKcceHn2nph.png)

# 材料

https://www.bilibili.com/video/BV1pXA5eyEEg?vd_source=8f7f521ad9c1872425eb493f20f42f57

https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/r1-k1.5/trl_grpo.ipynb

一个 demo：https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

trl 的例子（没什么意思，不如上面的 demo）：https://huggingface.co/docs/trl/grpo_trainer

# 学习路线

![image](Ipird4TnxoCKI6x2uSxcPYrOnD4.png)

# GRPO

- 在 24 年 deepseek math 刚开始的时候这个 reward 还是 learn 出来的，到 R1 被推出的时候就变成了一个简单的 reward function
- 相比 PPO 又多了 KL 散度，是 response 级别的，在 reward 之外
# 材料

- https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/r1-k1.5/reasoning_model_chat_template_and_inference.ipynb
- https://ollama.com/library/deepseek-r1:1.5b/blobs/c5ad996bda6e
# Deepseek API

https://api-docs.deepseek.com/guides/reasoning_model

![image](U7Emd4EkIoul0sx6yB2cYLUTn3d.png)

- loss 为 0 不代表不能再优化
- Loss = -Advantage
008 核心强化学习算法，GRPO、RLOO、REINFORCE++、REINFORCE++ baseline

https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/objectives_adv/core_algo_adv.ipynb

---

## See Also

- [[强化学习的数学原理|强化学习数学原理]] — RL 概览的数学深化版
- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — 概览之后的前沿：GRPO 七维改进体系
- [[AI/3-LLM/RL/目录|RL MOC]] — LLM 强化学习全图谱
- [[机器学习|机器学习]] — 监督学习到 RL 的演化
