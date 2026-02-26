---
brief: "PPL（Perplexity）困惑度——语言模型质量的核心评估指标；PPL = exp(CrossEntropyLoss)；为什么 PPL 下降不等于下游任务提升；量化模型 PPL vs 全精度 PPL 的对比意义；LLM 训练调参的常用诊断指标。"
title: "PPL 困惑度"
type: concept
domain: ai/foundations/ml-basics
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/foundations/ml-basics
  - type/concept
---
# PPL 困惑度

# 定义

PPL是Perplexity的缩写，中文称为"困惑度"。它是一种衡量语言模型性能的指标，常用于评估语言模型在给定测试集上的表现。PPL越低，说明语言模型对测试数据的预测越准确，性能越好。

PPL的计算基于语言模型对测试集的概率估计。给定一个测试集 ，其中N是测试集的长度(单词数)，语言模型的PPL定义为:

其中， 是语言模型对整个测试集的概率估计， 是语言模型在给定前i-1个单词的情况下，对第i个单词的概率估计。

由于连乘的概率值很小，为了避免下溢出(underflow)，通常使用对数概率进行计算。因此，PPL的计算公式可以改写为:

---

## See Also

- [[AI/3-LLM/Evaluation/LLM 评测体系|LLM 评测体系]] — PPL 作为语言模型评估指标的位置
- [[损失函数|损失函数]] — 交叉熵损失与 PPL 的关系：PPL = exp(cross-entropy)
- [[AI/1-Foundations/Training/Scaling Laws|Scaling Laws]] — PPL 与模型规模/数据量的幂律关系
- [[AI/1-Foundations/目录|Foundations MOC]] — ML 基础全图谱
