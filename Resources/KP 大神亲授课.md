---
title: "KP 大神亲授课"
type: reference
domain: resources
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - resources
  - type/reference
---
# KP 大神亲授课

## 概述

KP（Andrej Karpathy）是 AI 领域最好的教育者之一。前 Tesla AI Director、OpenAI 创始成员，他的教学特点是 **从第一性原理出发，手写实现，让你真正理解而不是调库**。

## 核心课程资源

### Neural Networks: Zero to Hero（YouTube 系列）

最推荐的入门系列，从零搭建神经网络到 GPT：

1. **The spelled-out intro to neural networks and backpropagation**
   - 手写 micrograd：一个极简的自动微分引擎
   - 理解反向传播的本质：链式法则 + 计算图
   - 关键洞察：每个节点存 `data` 和 `grad`，backward 时从输出反向传播

2. **The spelled-out intro to language modeling**
   - 从 bigram 到 MLP 语言模型
   - 构建 makemore：字符级语言模型

3. **Building makemore Part 2-4**
   - Batch Normalization 的直觉和实现
   - Wavenet 架构中的因果卷积
   - 手写 backprop（不用 PyTorch autograd）

4. **Let's build GPT**
   - 从零实现 GPT-2：Attention、LayerNorm、残差连接
   - 训练一个能生成莎士比亚文本的小 GPT
   - 这一期最有价值，值得反复看

5. **Let's build the GPT Tokenizer**
   - BPE（Byte Pair Encoding）从零实现
   - 理解为什么 tokenizer 这么重要

6. **Let's reproduce GPT-2 (124M)**
   - 工程实现的精华：数据加载、分布式训练、混合精度
   - 实际复现 GPT-2 124M 的完整流程

### nn.zero-to-hero GitHub

- 仓库地址：https://github.com/karpathy/nn-zero-to-hero
- 每个视频对应的 Jupyter Notebook
- 建议跟着视频敲代码，而不是只看

### 其他重要项目

- **minGPT / nanoGPT**：极简的 GPT 训练代码，理解 Transformer 训练的最佳起点
- **micrograd**：~100 行代码的自动微分引擎，理解反向传播的精华
- **llm.c**：纯 C 实现的 LLM 训练，极致性能优化

## 核心学习收获

### 1. 反向传播的直觉

KP 讲反向传播不用矩阵公式，而是用 **计算图 + 局部导数**：

```python
# micrograd 核心思想
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad   # 链式法则
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

**每个运算只需要知道局部导数，反向传播自动完成全局梯度计算。** 这个直觉比公式推导有用 100 倍。

### 2. 从头构建的价值

KP 的理念：不要从调库开始学。先理解 `nn.Linear` 背后是矩阵乘法 + 偏置，`nn.CrossEntropyLoss` 背后是 softmax + 负对数似然。**用过才知道库帮你做了什么**。

### 3. Transformer 的核心

从 Let's build GPT 中提炼的核心理解：

```python
# Self-Attention 的本质：加权聚合
# Q @ K^T → 注意力分数（谁跟谁相关）
# softmax → 归一化为概率
# × V → 按概率加权聚合 value

# 因果注意力：只看前面的 token
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
```

## 学习路径建议

```
Week 1-2: micrograd + backprop 视频
    → 能手写简单的自动微分
Week 3-4: makemore 系列
    → 理解语言模型训练全流程
Week 5-6: Let's build GPT
    → 从零实现 Transformer
Week 7-8: GPT-2 复现
    → 工程层面的分布式训练
```

这条路径走完，对 LLM 的理解会比只看论文深得多。

## 相关

- [[AI/Foundations/DL-Basics/Transformer|Transformer]]
- [[AI/Foundations/DL-Basics/Transformer 通识|Transformer 通识]]
- [[AI/LLM/Architecture/GPT|GPT]]
- [[AI/Foundations/DL-Basics/Attention 详解|Attention 详解]]
- [[AI/LLM/SFT/SFT 原理|SFT 原理]]
- [[Projects/Training-Experiments/实现一个 LLM|实现一个 LLM]]
- [[AI/Foundations/DL-Basics/深度学习|深度学习]]
- [[AI/LLM/RL/Fundamentals/RL & LLMs 入门|RL & LLMs 入门]]
