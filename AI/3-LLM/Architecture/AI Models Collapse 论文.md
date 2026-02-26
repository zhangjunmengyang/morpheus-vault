---
title: "AI Models Collapse when Trained on Recursively Generated Data"
brief: "Nature 2024 论文证明：LLM 在自身合成数据上递归训练会导致不可逆的模型崩溃（Model Collapse）——输出分布退化，低频尾部信息丢失，性能持续下降。对数据飞轮依赖合成数据的工程团队有重要警示意义。"
type: paper
domain: ai/llm/architecture
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/architecture
  - type/paper
---
# AI Models Collapse when Trained on Recursively Generated Data

- https://www.nature.com/articles/s41586-024-07566-y
- https://www.aminer.cn/pub/66a140d401d2a3fbfc159ab8/ai-models-collapse-when-trained-on-recursively-generated-data
paper：https://www.aminer.cn/pub/66a140d401d2a3fbfc159ab8/ai-models-collapse-when-trained-on-recursively-generated-data

解读 code：https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/garbage_out.ipynb

研究指出大型语言模型（LLMs）在不断生成数据训练时会出现“模型崩溃”现象，即数据分布的尾部信息消失，这一问题在LLMs、变分自编码器（VAEs）和高斯混合模型（GMMs）中均会出现，作者构建了理论基础并展示了该现象在所有学习生成模型中的普遍性。

背景：合成数据已经大量出现，不自知地会去利用这样的数据，因为现实的互联网数据已大量地混入 aigc 的数据，真假难辨，尤其是2023年3月，GPT4 发布之后；

模型误差来源：

核心对比实验：

- 控制变量：no data preserved vs. 10% data preserved
- metrics：PPL
Model Collapse refers to a degenerative learning process where models start forgetting **improbable events** over time, as the model becomes poisoned with its own projection of reality.

影响：

- 均值在变低，高概率事件被高估，低概率事件被低估。
- 尾部变高更长，后期代的模型开始生成原始模型永远不会生成的样本。
更好的变化：

- 均值在变低
- 尾部在变低
---

## See Also

- [[Attention 变体综述|Attention 变体综述]] — 模型架构与 model collapse 风险的关系
- [[预训练流程|预训练流程]] — 合成数据在预训练中的影响
- [[AI/3-LLM/Pretraining/LLM预训练与分布式训练2026全景|LLM 预训练 2026 全景]] — 数据工程：合成数据与 model collapse 的工业实践
-  — 大语言模型知识全图谱
