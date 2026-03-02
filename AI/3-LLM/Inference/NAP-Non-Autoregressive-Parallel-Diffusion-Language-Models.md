---
title: NAP：让 Diffusion Language Model 真正并行解码（数据×解码 co-design）
arxiv: 2602.23225
venue: arXiv (ICML submission)
date: 2026-03-01
source: https://arxiv.org/abs/2602.23225
tags:
  - inference
  - diffusion-lm
  - dlm
  - non-autoregressive
  - parallel-decoding
  - cot
rating: ★★★★☆
brief: 论文解释了为什么很多“快速”扩散语言模型最终仍呈现 AR-like（左到右）解码：主因是训练监督与语料本身高度顺序化（尤其 long CoT）造成 objective mismatch。作者提出 NAP：用“并行对齐”的数据（多条独立推理轨迹作为一个样本）+ parallel-forced decoding 强制多 token/多轨并行更新；并提出 SeqDep 与 ARness 指标分析训练语料的顺序依赖。结论是：要得到真正的 non-AR 并行生成，必须 data-decoding co-design，而不是只改 sampling schedule。
related:
  - "[[AI/3-LLM/Inference/SNDT-Stitching-Noisy-Diffusion-Thoughts-Test-Time-Scaling]]"
  - "Rejection-Mixing（Fast Semantic Propagation for DLLM inference）"
  - "[[AI/3-LLM/Architecture/Flow-Matching-手撕实操]]"
---

# Why Diffusion Language Models Struggle with Truly Parallel Decoding?（arXiv:2602.23225）

## 1) 他们观察到的“反直觉现象”
DLM 常被宣传为“并行生成 token”，但很多 practical fast DLM 最终会**收敛到类似 AR 的左到右解码动力学**。

论文的立场很明确：
- “并行”不是有没有 mask/unmask 的形式问题
- 而是：训练监督 + 数据分布**把模型推回顺序化临界路径**

## 2) 机制假设：objective mismatch + 数据高度 sequential
核心一句话：AR-like decoding 的主因，是 **DLM objectives 与训练数据的高度顺序结构不匹配**。

特别点名两类数据：
- 标准预训练语料（天然是顺序文本）
- long chain-of-thought（更强的左到右推理依赖）

=> 你越用“单条 canonical CoT”监督，模型越学会“必须等前面确定了才能写后面”。

## 3) 他们怎么量化“顺序性”与“AR 化”？
### 3.1 ARness（autoregressive bias）
沿用/扩展已有 ARness 指标：把解码过程看成每一步 commit/unmask 的位置序列 p=(p1,...,pL)，区分：
- 全局左到右偏好（优先填最左 token）
- 局部连续性（相邻位置连续填充）

意义：
- 很多 DLM 表面上是“随机顺序/并行”，但 ARness 仍然很高（本质还是 AR）。

### 3.2 SeqDep（Sequential Dependence）
他们提出/使用 SeqDep 来量化语料本身的顺序依赖：
- token 在某位置对其 preceding context 的决定性有多强

结论（从 Figure 3 的描述）：
- 常用训练语料（例：FineWeb、OpenR1-Math）SeqDep 持续高且上升
- => 训练数据本身强制模型内化 AR 依赖

## 4) NAP：Non-Autoregressive Parallel DLMs
作者给的解法是**数据×解码 co-design**，不是单点 patch。

### 4.1 Data Curation：把“并行监督”做成样本内结构
问题：标准 CoT 只有一条推理序列（canonical L2R order），天然不适合并行 DLM。

NAP 的做法：对同一个 query x，采样 P 条**独立推理轨迹** {r(1),...,r(P)}，并把它们作为同一个训练样本的并行监督结构（而不是简单当作 P 个独立样本）。
- 用高温采样 τ=1.0 增加轨迹多样性（不同解题路径/不同逻辑顺序）

直觉：
- 让“同一个答案”可以由多条局部独立子链路支撑
- 模型在训练时就学到：不同位置/子结构可以同时被更新，而不是强依赖前缀

### 4.2 Parallel-Forced Decoding：强制多 token 并行更新
他们强调：即使有并行数据，如果解码策略不强制并行，模型仍可能 collapse 回顺序化。

所以 NAP 在推理时用 parallel-forced decoding，显式鼓励 multi-token parallel updates。

### 4.3 Ablation 的关键洞察：没有训练支持，强行并行会更差
他们在 GSM8K 上做 ablation（Dream-7B）：
- 仅把 parallel-forced decoding 套在“没用并行数据训练”的 base model 上，会比 AO（arbitrary order）掉得更厉害
- => 并行解码不是“解码器技巧”，而是需要模型在训练中适配“碎片化并行上下文”

一句话总结：**并行是能力，不是开关。**

## 5) 我对这篇论文的判断
### 5.1 这篇的增量在哪里？
它不是又一个 schedule / sampler，而是给了一个很硬的因果链：
- 语料/监督高度 sequential → DLM 学出 AR-like 解码（即便形式上并行）
- 要破局必须：训练监督本身提供“可并行的因果结构” + 解码阶段强制并行

这和我们在其他地方看到的共性一致：
- SNDT 是 test-time stitching，利用 dLLM 的非因果性做拼接
- NAP 是 training-time 把“非顺序化监督结构”灌进去，减少 AR collapse

### 5.2 边界
- 它主要解决“数学推理”类数据上的并行性；是否能迁移到开放域长文本生成，还要看 SeqDep 的本质是否能被这种多轨迹 supervision 改写。
- 但至少它给出一个非常清晰的诊断工具（ARness/SeqDep）来判断：你所谓的 DLM 并行到底是不是伪并行。

## Links
- arXiv:2602.23225 — https://arxiv.org/abs/2602.23225
- Code — https://github.com/pixeli99/NAP
