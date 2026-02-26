---
brief: "Karpathy nanochat——$72 在 8×H100 节点 3 小时训练出 GPT-2 级别模型（~1000 行代码）；展示现代预训练基础设施的极致精简；逆向验证大模型训练的核心要素（tokenizer/数据pipeline/attention/优化器）。"
title: "Karpathy nanochat: $72 训练 GPT-2"
date: 2026-02-14
tags: [llm, training, efficiency, karpathy, open-source]
type: note
---

# Karpathy nanochat: $72 训练 GPT-2

## 核心价值
Karpathy 的 nanochat 用 ~1000 行代码，在单个 8×H100 节点上 3 小时训练出 GPT-2 级别模型，成本 **$72**（spot instance ~$20）。

对比：2019 年 OpenAI 训练 GPT-2 花了 **$43,000**、168 小时。7 年后成本降低 **600 倍**，速度提升 **60 倍**。

## 设计哲学
- **一个旋钮**：`--depth`（transformer 层数）决定模型规模，其他超参数自动计算
- **全流程覆盖**：tokenization → pretraining → finetuning → evaluation → inference → chat UI
- **极简可 hack**：~1000 行代码，适合教学和实验

## 技术亮点
- 计算最优模型配置（自动调宽度、heads、LR、训练步数）
- GPT-2 级别 ≈ depth 26
- DCLM CORE score 作为评测标准
- 社区驱动的 Time-to-GPT-2 Leaderboard（当前记录：2.76 小时）

## 对我们的启示
1. **训练成本雪崩**：$43K → $72，意味着小团队/个人也能训练有意义的模型
2. **教学价值**：理解 LLM 全流程的最佳入口
3. **Benchmark 基线**：可以用 nanochat 快速验证新训练技巧的效果
4. **与 Agentic Engineering 呼应**：Karpathy 同时提出了 Agentic Engineering 概念，nanochat 是底层能力民主化，agentic 是上层应用范式转移

## Leaderboard（截至 2026-02）

| # | Time | CORE | Description | Date |
|---|------|------|-------------|------|
| 0 | 168h | 0.2565 | Original GPT-2 | 2019 |
| 1 | 3.04h | 0.2585 | d24 baseline | Jan 29 |
| 2 | 2.91h | 0.2578 | d26 +fp8 | Feb 2 |
| 3 | 2.76h | 0.2602 | batch size 1M | Feb 5 |

## 参考
- [GitHub: karpathy/nanochat](https://github.com/karpathy/nanochat)
- [DeepWiki 文档](https://deepwiki.com/karpathy/nanochat)
- [B站解读](https://www.bilibili.com/video/BV171cwzWE5C/)

---

## See Also

- [[AI/LLM/Pretraining/LLM预训练与分布式训练2026全景|LLM 预训练 2026 全景]] — nanochat 实验与工业级预训练的对比
- [[AI/Foundations/Training/预训练流程|预训练流程]] — 预训练的基础概念
- [[AI/LLM/Frameworks/verl/verl 概述|verl 框架]] — 对照：verl 是工业级 RL 训练框架
- [[AI/LLM/目录|LLM MOC]] — 大语言模型知识全图谱
