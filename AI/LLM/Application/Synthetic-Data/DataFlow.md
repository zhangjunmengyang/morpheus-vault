---
title: "DataFlow"
type: concept
domain: ai/llm/application/synthetic-data
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/application/synthetic-data
  - type/concept
---
# DataFlow

> 文档：https://opendcai.github.io/DataFlow-Doc/zh/guide/basicinfo/framework/

DataFlow 是字节跳动开源的数据质量评估与筛选框架，专门为 LLM 训练数据而设计。核心定位是"数据的 lint 工具"——帮你在训练之前发现数据中的问题。

## 为什么需要 DataFlow

搞过 fine-tuning 的都知道：**数据质量比数据数量重要 10 倍**。Phi-1.5 用 1B 条精选数据干翻了用 10x 数据训练的模型。但"精选"这件事，靠人看是不现实的——百万级数据集你看到猴年马月？

DataFlow 把数据评估自动化了。

## 架构

```
┌──────────────────────────────────────┐
│             DataFlow Pipeline         │
├──────────┬───────────┬───────────────┤
│ Operator │ Operator  │   Operator    │ ← 可插拔的算子
│ (统计类) │ (模型类)  │  (规则类)     │
├──────────┴───────────┴───────────────┤
│           Core Engine                 │
│   数据加载 / 并行处理 / 缓存管理      │
├──────────────────────────────────────┤
│   Input: HF Dataset / JSON / Parquet │
└──────────────────────────────────────┘
```

## 核心概念

### Operator（算子）

DataFlow 的基本单元。每个 operator 负责一个维度的质量评估：

```python
from dataflow.operators import TextLengthFilter, LanguageDetector, PPLFilter

# 组合多个 operator
pipeline = [
    TextLengthFilter(min_len=50, max_len=5000),
    LanguageDetector(target_lang="zh"),
    PPLFilter(model="gpt2", threshold=500),  # 困惑度过滤
]
```

### 内置算子分类

| 类型 | 算子 | 用途 |
|------|------|------|
| 统计类 | TextLength, WordCount | 基础长度过滤 |
| 语言类 | LangDetect, CharsetFilter | 语言/字符集过滤 |
| 质量类 | PPLFilter, QualityClassifier | 基于模型的质量评分 |
| 去重类 | MinHash, SimHash | 近似/精确去重 |
| 安全类 | ToxicFilter, PIIDetector | 有害内容/PII 过滤 |

### 困惑度过滤（PPL Filter）

这是最有用的算子之一。原理很简单：用一个已训练好的 LM 计算每条数据的困惑度，PPL 太高说明文本不通顺，PPL 太低说明是模板/重复文本。

```python
# 好的范围通常在 50-500 之间（取决于模型和语言）
ppl_filter = PPLFilter(
    model="gpt2",
    min_ppl=30,    # 过滤掉太"死板"的文本
    max_ppl=500,   # 过滤掉太"混乱"的文本
)
```

## 实用 Pipeline

一个完整的数据清洗流程：

```python
from dataflow import Pipeline

pipe = Pipeline(
    input_path="raw_data.jsonl",
    output_path="clean_data.jsonl",
    operators=[
        # 1. 基础过滤
        TextLengthFilter(min_len=100, max_len=10000),
        
        # 2. 语言过滤
        LanguageDetector(target_lang="zh", threshold=0.8),
        
        # 3. 质量打分
        PPLFilter(model="gpt2", max_ppl=500),
        
        # 4. 去重
        MinHashDedup(threshold=0.7, num_perm=128),
        
        # 5. 安全过滤
        ToxicFilter(threshold=0.5),
    ],
    num_workers=8,
)

stats = pipe.run()
print(f"原始: {stats.input_count}, 保留: {stats.output_count}, "
      f"过滤率: {stats.filter_rate:.1%}")
```

## 与其他工具的对比

- **Data-Juicer（阿里）**：功能类似，算子更多，但配置更复杂
- **NeMo Curator（NVIDIA）**：GPU 加速的大规模去重，适合预训练数据
- **Dolma（AI2）**：面向预训练数据，工具链完整但偏重英文

DataFlow 的优势在于轻量和易用，适合 SFT/对齐数据的场景。

## 相关

- [[AI/LLM/Application/Synthetic-Data/Synthetic Data|Synthetic Data 综述]]
- [[AI/LLM/Application/数据合成|数据合成（Prompt 视角）]]
- [[AI/LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[AI/LLM/SFT/SFT 原理|SFT 原理]]
