---
brief: "Transformer 原始架构——Attention is All You Need（Vaswani 2017）核心设计解析；Multi-Head Attention/FFN/残差连接/LayerNorm 的数学推导；从论文到实现的代码级理解；现代 LLM 架构理解的起点。"
title: "Transformer"
type: concept
domain: ai/foundations/dl-basics
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/foundations/dl-basics
  - type/concept
---
# Transformer

https://www.mianshiya.com/bank/1821834692534505473?current=1&pageSize=20

https://colab.research.google.com/drive/1x2JDTCEUNc1pzJ72exguxt23IdEaXBWq?usp=sharing

Transformer 架构整理： https://huggingface.co/learn/llm-course/zh-CN/chapter1/4?fw=pt

手撕个 transformer：https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/Transformer%E7%BB%84%E4%BB%B6%E5%AE%9E%E7%8E%B0.ipynb

---

## See Also

- [[AI/Foundations/DL-Basics/Layer Normalization|Layer Normalization]] — Pre-Norm / RMSNorm 在 Transformer 中的关键角色
- [[AI/Foundations/Math/线性代数|线性代数]] — QKV 矩阵乘法、注意力分数计算的数学基础
- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]] — 从标准 MHA 到 MQA/GQA/FlashAttention 的演化
- [[AI/Foundations/目录|Foundations MOC]] — 深度学习基础全图谱
