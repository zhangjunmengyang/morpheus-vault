---
title: "Synthetic Data"
type: concept
domain: ai/llm/application/synthetic-data
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/application/synthetic-data
  - type/concept
---
# Synthetic Data 综述

合成数据是 2024-2025 年 LLM 领域最重要的趋势之一。从 Alpaca 到 Phi 系列再到 DeepSeek-R1 的蒸馏数据，合成数据已经从"权宜之计"变成了"主流方法论"。

## 为什么合成数据成为主流

### 数据墙（Data Wall）

互联网高质量文本数据已经被主流模型训练了好几遍。Epoch AI 的研究表明，到 2026 年高质量自然语言数据将基本耗尽。合成数据是突破这个瓶颈的关键路径。

### 成本考量

| 方式 | 单条成本 | 质量 | 规模 |
|------|----------|------|------|
| 人工标注 | $2-10 | 高但方差大 | 千级 |
| 众包 | $0.5-2 | 中 | 万级 |
| LLM 生成 | $0.001-0.05 | 中高 | 百万级 |

## 主要方法

### 1. 蒸馏（Distillation）

用强模型的输出训练弱模型，这是最简单有效的合成数据方法：

```python
# 用 GPT-4o 生成 → 训练 Qwen-7B
for question in questions:
    response = gpt4o.generate(question)
    dataset.append({"input": question, "output": response})
```

代表：Alpaca (GPT-3.5 → LLaMA)、Orca (GPT-4 → 小模型)

### 2. Self-Instruct

模型自己生成指令和回答：

```
种子指令(175条) → LLM生成新指令 → LLM生成回答 → 过滤 → 循环
```

核心在于**多样性控制**：每次生成时从种子池随机采样 few-shot examples，并用 ROUGE 去重。

### 3. Evol-Instruct

WizardLM 的方法，通过"进化"增加指令复杂度：

- **深度进化**：增加推理步骤、约束条件
- **广度进化**：改变场景、任务类型
- **淘汰机制**：去掉进化失败的样本

### 4. 推理数据合成

DeepSeek-R1 的做法启发了一波新范式——先用 RL 训练出强推理模型，再蒸馏其 CoT 过程：

```
强模型(带 CoT) → 收集 <think>...</think> 输出 → 训练弱模型模仿推理过程
```

关键发现：小模型学会了推理**格式**后，也能在一定程度上获得推理**能力**。

### 5. 对齐数据合成

RLHF/DPO 需要偏好对数据。合成方法：

```python
# Constitutional AI 风格
response_a = model.generate(prompt)  # 原始回答
critique = model.generate(f"找出以下回答的问题：{response_a}")
response_b = model.generate(f"基于以下反馈改进回答：{critique}")
# (response_b, response_a) 构成偏好对
```

## 质量保障

合成数据最大的风险是 **garbage in, garbage out**：

1. **多级过滤**：规则过滤 → Reward Model 打分 → 人工抽检
2. **数据配比**：合成数据和真实数据混合，通常 7:3 到 9:1
3. **去污染**：确保测试集中的问题没有出现在合成训练集中
4. **多样性指标**：监控生成数据的 n-gram 分布、主题覆盖度

## 开源工具链

- **[DataFlow](https://github.com/opendcai/DataFlow)**：字节出的数据质量评估框架
- **[Magpie](https://github.com/magpie-align/magpie-align)**：对齐数据生成
- **[Distilabel](https://github.com/argilla-io/distilabel)**：Argilla 的合成数据 pipeline

## 相关

- [[AI/LLM/Application/Synthetic-Data/DataFlow|DataFlow 框架]]
- [[AI/LLM/Application/数据合成|数据合成（Prompt 视角）]]
- [[AI/LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[AI/LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/LLM/RL/DPO/DPO-TRL实践|DPO 实践]]
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]
