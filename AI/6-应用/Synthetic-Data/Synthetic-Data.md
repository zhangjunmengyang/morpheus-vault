---
brief: "Synthetic Data——合成数据基础概念和主流生成方法；Self-Play/Backtranslation/Persona-based 生成策略；质量 vs 多样性的核心权衡；LLM 训练数据工程的基础参考。"
title: "Synthetic Data"
type: concept
domain: ai/llm/application/synthetic-data
created: "2026-02-13"
updated: "2026-02-13"
sources:
  - "Phi-1: arXiv:2306.11644 (Microsoft)"
  - "WizardLM: arXiv:2304.12244"
  - "FineWeb-Edu: HuggingFace 2024"
  - "DeepSeek-R1: arXiv:2501.12948"
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

## See Also

- [[DataFlow|DataFlow 框架]]
- [[AI/6-应用/Synthetic-Data/数据合成|数据合成（Prompt 视角）]]
- [[AI/3-LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/3-LLM/RL/实践/DPO-TRL实践|DPO 实践]]
- [[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek-R1]]

---

## Prompt 视角：合成数据工程

# 数据合成（Prompt 视角）

用 LLM 生成训练数据是当前最热的 Prompt Engineering 应用场景之一。核心挑战不是"能不能生成"，而是"生成的数据质量够不够"。

## 为什么要用 Prompt 合成数据

1. **人工标注成本高**：一条高质量的 SFT 数据人工标注成本可以到几美元
2. **领域数据稀缺**：医疗、法律等垂直领域的标注数据极难获取
3. **数据多样性**：人工标注容易陷入模式固化，LLM 反而能覆盖更多变体

## 核心 Pipeline

```
种子数据 → Prompt 模板 → LLM 生成 → 质量过滤 → 去重 → 训练集
```

### Step 1: 种子数据准备

种子数据不需要多，但要**高质量且有代表性**：

```python
seeds = [
    {"instruction": "解释什么是 MapReduce", "response": "MapReduce 是一种..."},
    {"instruction": "Spark 和 Flink 的区别", "response": "两者都是..."},
    # 10-50 条即可
]
```

### Step 2: 进化式生成（Evol-Instruct）

WizardLM 提出的 Evol-Instruct 是最经典的方法——通过 prompt 让 LLM 对种子指令做"进化"：

```python
evolve_prompt = """我有一条指令，请用以下方式之一改写，使其更复杂：

1. 增加约束条件
2. 增加推理步骤  
3. 要求处理 edge case
4. 将简单任务变为复合任务

原始指令：{instruction}

改写后的指令（只输出改写结果）：
"""
```

然后用更强的模型（如 GPT-4o / Claude）生成 response。

### Step 3: 质量过滤

生成容易，过滤才是关键。常用策略：

```python
def quality_filter(item):
    # 1. 长度检查
    if len(item["response"]) < 50:
        return False
    
    # 2. 格式检查（是否遵循指令要求的格式）
    if not format_check(item):
        return False
    
    # 3. LLM-as-Judge 打分
    score = judge_llm(f"""
    对以下回答打分（1-5分）：
    问题：{item["instruction"]}
    回答：{item["response"]}
    评分标准：准确性、完整性、有用性
    只输出数字。
    """)
    return int(score) >= 4

    # 4. Reward Model 打分（更准但更贵）
```

### Step 4: 去重

语义级去重比字面去重更重要：

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode([d["instruction"] for d in dataset])

# 余弦相似度 > 0.85 视为重复
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
duplicates = set()
for i in range(len(sim_matrix)):
    for j in range(i+1, len(sim_matrix)):
        if sim_matrix[i][j] > 0.85:
            duplicates.add(j)
```

## 实用技巧

- **多模型交叉生成**：用 A 模型生成，B 模型打分，效果好于单模型
- **拒绝采样**：生成 N 条，只取得分最高的 K 条（N/K 通常 3-5x）
- **Persona-driven**：在 prompt 里加入不同角色（学生/工程师/研究员），增加多样性
- **批判式生成**：先生成一版，再让模型自我批评并修改

## 注意事项

⚠️ **模型坍缩**：如果用模型 A 的输出训练模型 A，多轮后会质量退化（Model Collapse）。解决方案是用更强的模型生成，或掺入真实数据。

⚠️ **License 风险**：用 GPT-4 生成的数据训练竞品模型可能违反 ToS，需要关注。

## 相关

- [[AI/6-应用/Synthetic-Data/Synthetic Data|Synthetic Data 综述]]
- [[DataFlow|DataFlow 框架]]
- [[AI/3-LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]]
- [[Prompt-Engineering-概述|Prompt Engineering 概述]]