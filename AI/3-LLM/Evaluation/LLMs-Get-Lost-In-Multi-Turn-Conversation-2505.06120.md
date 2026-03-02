---
title: "LLMs Get Lost In Multi-Turn Conversation"
arxiv: "2505.06120"
date: 2026-03-01
authors:
  - Philippe Laban
  - Hiroaki Hayashi
  - Yingbo Zhou
  - Jennifer Neville
venue: arXiv
area:
  - LLM
  - Evaluation
  - Multi-turn
status: skim
source:
  - https://arxiv.org/abs/2505.06120
  - https://arxiv.org/pdf/2505.06120.pdf
---

## TL;DR（我的判断）
这篇工作把一个我们“体感很强”的现象定量化：**同一模型在 multi-turn 对话里显著更差**，平均性能下降 **39%**（作者在 6 类 generation task 上测得）。关键不是“能力变弱”，而是 **unreliability（不稳定/不可恢复）显著上升**：一旦前几轮走错方向，模型会过早押注某个假设、过早产出最终解，并在后续轮次里对自己的错误路径过度依赖，**很难自我纠错回正**。

> 价值点：它提示 multi-turn evaluation 的主要瓶颈不是单轮能力（aptitude），而是“**恢复力/可纠错性**”——这与 Agent RL 的 long-horizon credit assignment、以及 tool-use 里的“错误早期承诺”是同一类 failure mode。

## 论文要点（来自 arXiv 摘要 + 页面信息；PDF 未完整解析）
- 任务设定：对比 single-turn（指令 fully specified）vs multi-turn（通过多轮对话逐步澄清/推进）
- 方法：large-scale simulation experiments
- 规模：**200,000+** simulated conversations
- 结论：所有测试的 top open-/closed-weight LLM 在 multi-turn 都显著更差；跨 6 个 generation tasks，平均下降 **39%**
- 分解：performance degradation =
  - minor loss in aptitude
  - significant increase in unreliability
- 机制观察：
  - early turns 里容易“做假设”
  - 过早尝试生成 final solution
  - 后续轮次对早期产出的解/假设过度依赖
  - 结果是“走错一步就迷路且不恢复”

## 我关心的技术含义（对 Vault 主线的映射）
1) **Multi-turn 的核心指标应从 accuracy 转向 recoverability**
   - 单轮 benchmark 让模型只需要“一次性答对”；multi-turn 更像 control system：关键是能否在噪声/误解下回到正确轨道。

2) 这更像“推理时策略失稳”而非知识缺失
   - 作者把 degradation 分为 aptitude vs unreliability，本质上是在说：multi-turn 场景里，对话上下文导致策略处在更强的 distribution shift 下，错误被放大。

3) 对 Agent/Tool-Use：错误早期承诺（premature commitment）= 最常见死法
   - 这与我们在工具调用/长链路任务里看到的现象一致：过早写代码/过早下结论 → 后面所有动作都在“验证自己”而不是“寻找真相”。

## 可以立刻用的工程启发（可转成 checklist）
- 强制 clarification 预算：前 N 轮必须问清楚关键信息（减少 early assumption）
- 显式维护 competing hypotheses：不要把早期假设写成“事实”，在上下文里保留备选解释
- 允许/鼓励 backtracking：把“撤销前提/重来”作为一个合法动作（而非丢脸）
- 评测新增维度：
  - wrong-turn injection（故意在早期给一个模棱两可/误导信息）
  - recovery score（后续是否能恢复到正确轨道）

## 关联
- Multi-turn RL/自进化：RAGEN/StarPO、SCoRe（Echo Trap 与策略失稳）
- Credit Assignment：长程任务中早期错误对后续回报的放大

## TODO
- [ ] 解析 PDF：看 6 个 generation tasks 分别是什么、multi-turn 是如何模拟的、unreliability 的度量定义（calibration? variance? error persistence?）
- [ ] 把“recoverability”加入到 Agent evaluation 设计笔记里（SWE/OSWorld/WebArena 类）
