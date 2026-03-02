---
title: Scale Can’t Overcome Pragmatics：Reporting Bias 如何压制 VLM 推理能力
arxiv: 2602.23351
venue: arXiv
date: 2026-03-01
source: https://arxiv.org/abs/2602.23351
tags:
  - mllm
  - vlm
  - data
  - pragmatics
  - evaluation
  - counting
  - spatial-reasoning
rating: ★★★★
brief: 论文从语用学(pragmatics)视角提出：VLM 推理弱不是“模型不够大”，而是训练图文数据存在 reporting bias——人类描述图片时会省略对推理至关重要的隐含信息，导致四类推理技能（空间/时间/否定/计数）在 web-scale 或合成数据中仍系统性缺失。核心结论：scale（数据、模型、多语言）不自动带来这些能力；更有效的路径是用明确的标注指令收集 tacit information。
related:
  - "[[AI/3-LLM/MLLM/MLLM 概述]]"
---

# Scale Can’t Overcome Pragmatics（arXiv:2602.23351）

## 0) 这篇论文在“反对”什么？
社区常见直觉：VLM 推理差（counting/spatial/negation）→ 继续 scale 数据/模型/语言，能力会 emergent。

论文的反直觉结论：**scale 并不能克服数据的语用学偏置**。推理监督缺失是结构性问题，不是样本不够多。

## 1) 核心概念：Reporting Bias（报告偏置）
定义（用他们的例子直观理解）：
- 人类在给图片写 caption 时更可能写“at the game today!”
- 而不是写“a photo of 37 people standing behind a field”

也就是说：
- caption/alt-text 更倾向表达“显著/有趣/社交语境”
- 而推理任务需要的 **tacit information（默认被省略的细节）** 没被说出来

=> 数据里缺的不是 token，而是 **监督信号的类型**。

## 2) 他们提出的四类“被压制的推理技能”
报告偏置会系统性压制四种 reasoning：
1. Spatial reasoning（空间：左右/前后/相对位置）
2. Temporal reasoning（时间：先后/变化）
3. Negation（否定：不在/没有/不是）
4. Counting（计数：人数/物体数量）

这四类的共同点：
- 日常叙述里常被省略（除非有必要）
- 但 benchmark/推理任务需要把它说清楚

## 3) 证据链（他们怎么证明的）
### 3.1 数据侧：看主流开源 VLM 训练语料
他们检查了 popular VLMs 的数据（摘要点名）：
- OpenCLIP
- LLaVA-1.5
- Molmo

结论：即使语料 web-scale 或含合成数据，四类推理信息仍“代表性不足”。

### 3.2 评测侧：构建 curated benchmarks
他们用一组 curated benchmarks 来专门测上述能力，并得到：
- (i) VLM 在这些被 reporting bias 压制的推理任务上表现差
- (ii) scale（数据量/模型量/多语言）不带来自动涌现
- (iii) 但如果加入“专门为补 tacit info 而收集的标注”，能有效缓解

## 4) 最可迁移的洞察：为什么“更多数据”没有用？
**因为缺的是“会被写出来的标签空间”**。

把它写成一个可操作的训练数据结论：
- 如果 data collection 不改变，新增样本只是在重复同一种语用分布
- 于是模型学到的是“人类通常怎么说”，而不是“完成推理需要说什么”

这和我们在 Agent 系统里看到的现象非常像：
- 如果中间产物接口（agent 输出）不包含关键 latent state，后面的模块再强也无法补救

## 5) 方法论启示（对我们做 MLLM/评测/数据工程）
1. **数据收集要写清楚 instruction**：让 annotator 把 tacit info 显式化（counting、空间关系、否定等）
2. 不要把“emergence”当默认：在特定能力上，scale 可能只是更强的“偏置拟合器”
3. Benchmark 设计要与数据偏置成对：
   - 不是泛测“综合能力”
   - 而是测“数据里缺什么监督信号”

## 6) 我对这篇论文的边界判断
- ✅ 价值：给了一个**解释框架**（pragmatics → reporting bias），能解释“为什么大模型仍数不清/分不清方位”。
- ⚠️ 边界：它主要论证“数据偏置导致能力缺失”，但不意味着“模型结构/训练范式不重要”；更像是告诉你：结构再好也别指望用 web caption 自动学到这些推理监督。

## See Also

- [[AI/3-LLM/MLLM/MLLM 概述]] — MLLM 入口与总体图景。
- [[AI/3-LLM/MLLM/Universal Multimodal Retrieval]] — 多模态检索：当 caption 缺 tacit info 时，检索层也会继承 bias。
- [[AI/3-LLM/MLLM/Multimodal-Perception-RL-综合分析]] — 感知/推理能力提升的另一条路径：从数据偏置之外看 RLVR 与训练范式。
- [[AI/3-LLM/Pretraining/预训练数据工程]] — 对应“数据分布决定能力上限”的工程视角（如果该笔记不存在则需后续补链/改名）。

## Links
- arXiv:2602.23351 — https://arxiv.org/abs/2602.23351
- HTML — https://arxiv.org/html/2602.23351v1
