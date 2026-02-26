---
title: "INBOX"
type: inbox
tags:
  - type/inbox
updated: 2026-02-26
---

# 📥 INBOX — 待处理区

> 新内容落地处。三类内容：待读论文、待炼化笔记、知识缺口。
> 处理原则：读完 → 写独立笔记 → 从这里删掉。不处理不等于没看到。

---

## 待读论文

### LLM / RL

- [ ] REINFORCE++：高效 RLHF，对 prompt/reward 鲁棒 — [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)
- [ ] GRPO / DeepSeekMath — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)（基础必读，GRPO 出处）
- [ ] IOPO：复杂指令跟随，输入输出偏好优化 — [arXiv:2411.06208](https://arxiv.org/abs/2411.06208)
- [ ] LLMs Get Lost In Multi-Turn Conversation — [arXiv:2505.06120](https://arxiv.org/abs/2505.06120)
- [ ] GRPO 完整流程实践（SWIFT 文档）— https://swift.readthedocs.io/zh-cn/latest/BestPractices/GRPO完整流程.html
- [ ] 知乎 GRPO 分析 — https://zhuanlan.zhihu.com/p/20021693569
- [ ] 知乎 GRPO 分析 2 — https://zhuanlan.zhihu.com/p/21046265072

### Prompt Engineering

- [ ] Chain-of-Thought Prompting — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- [ ] Self-Consistency — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- [ ] Least-to-Most — [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
- [ ] ReAct — [arXiv:2210.03493](https://arxiv.org/abs/2210.03493)
- [ ] Tree of Thoughts — [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- [ ] ART: Automatic multi-step reasoning — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

### 论文阅读资源

- 李沐论文精读笔记：https://github.com/Tramac/paper-reading-note
- 选读列表（LLMs）：https://github.com/km1994/llms_paper
- 李沐视频精读：https://github.com/mli/paper-reading
- ML Papers of the Week：https://github.com/dair-ai/ML-Papers-of-the-Week

---

## 知识缺口（待入库）

> 来源：全景类文件扫描后提炼的缺口。有对应全景文件的先从全景中提取，全部覆盖后删全景文件。

### 高优先级（与核心方向直接相关）

- [ ] **LLM 数据工程** — 预训练数据清洗/合成/配比策略（Dedup/质量过滤/domain mix）
  - 来源：`AI/3-LLM/Pretraining/LLM-数据工程-2026-技术全景.md`（3794行，待提炼）
- [ ] **LLM Evaluation 体系** — MMLU/HumanEval/LiveBench/AlpacaEval 评估框架对比
  - 来源：`AI/3-LLM/Evaluation/LLM评估与Benchmark-2026技术全景.md`（1949行，待提炼）
- [ ] **合成数据与数据飞轮** — Self-play/Evol-Instruct/数据配比，与 RLHF 数据管线的关系
  - 来源：`AI/3-LLM/Application/Synthetic-Data/合成数据与数据飞轮-2026技术全景.md`（1740行）
- [ ] **知识蒸馏与模型压缩** — Pruning/Distillation/Quantization 系统性对比（目前只有1篇笔记）
  - 来源：`AI/3-LLM/Efficiency/知识蒸馏与模型压缩-2026技术全景.md`（2168行，待提炼）

### 中优先级

- [ ] **Tool Use / Function Calling 深度** — MCP 协议层、ACT 格式、工具调用稳定性
  - 来源：`AI/2-Agent/Fundamentals/LLM工具调用与Function-Calling-2026技术全景.md`（1774行）
- [ ] **自监督学习 / 对比学习** — SSL/SimCLR/CLIP 等预训练范式（与 LLM 预训练的关系）
  - 来源：`AI/3-LLM/Pretraining/自监督学习与对比学习-2026技术全景.md`（2427行）

### 低优先级（非核心方向，按需处理）

- [ ] **Crypto 量化交易** — 资金费率/波动率/多空策略（Quant 方向，非 AI 核心）
  - 来源：`AI/6-应用/Quant/Crypto-量化交易-2026-技术全景.md`
- [ ] **AI 搜索与推荐系统** — 向量检索/召回精排/在线学习
  - 来源：`AI/6-应用/AI搜索与推荐系统-2026技术全景.md`
- [ ] **LLM 应用部署与工程化** — vLLM/Triton/监控/灰度发布
  - 来源：`AI/6-应用/LLM-应用部署与工程化-2026技术全景.md`

---

## 待炼化笔记

> Scholar/哨兵写入但尚未系统炼化的内容。

_（当前为空，新内容写入后在此登记）_

---

## 待转「思考」的全景文件

> 这些领域独立深度笔记已充分，全景文件的价值只剩"串联视角"——提炼成一篇能发出去的思考文章后，原文件删除。

- [ ] Transformer 架构深度解析（32篇独立笔记覆盖）→ 写「我理解的 Transformer 架构演化」
- [ ] LLM 推理优化全景（36篇覆盖）→ 写「推理优化的本质是什么」
- [ ] Prompt Engineering 全景（14篇覆盖）→ 写「Prompt 工程的边界在哪里」
- [ ] LLM 微调实战全景（12篇覆盖）→ 写「微调选型决策框架」
- [ ] 多智能体系统全景（16篇覆盖）→ 写「多 Agent 协作的核心设计问题」
- [ ] AI 安全与对齐全景（18篇覆盖）→ 写「对齐问题的本质」
- [ ] RAG 全景（11篇覆盖）→ 写「RAG 的适用边界」
- [ ] 多模态大模型全景（22篇覆盖）→ 写「多模态融合的几种路线」
- [ ] RLHF/DPO 全景（RL 有大量手撕覆盖）→ 写「从 RLHF 到 DPO 的设计演化」
