---
title: "SMTL: Search More, Think Less — Long-Horizon Agentic Search 的效率/泛化范式"
date: 2026-03-01
updated: 2026-03-01
arxiv: 2602.22675
doi: 10.48550/arXiv.2602.22675
authors: ["OPPO AI Agent Team", "Qianben Chen", "Tianrui Qin", "Wangchunshu Zhou", "et al."]
tags:
  - AI/Agent
  - agentic-search
  - long-horizon
  - efficiency
  - context-management
  - data-synthesis
  - reinforcement-learning
---

## TL;DR

SMTL 的核心不是“更会想”，而是**更会找**：
- 把 long-horizon deep research agent 的主瓶颈，从 *sequential reasoning 深度* 改写为 *evidence acquisition 的并行化 + 上下文预算管理*。
- 用统一的数据合成管线覆盖 deterministic QA 与 open-ended research，从训练阶段就逼迫“跨任务指标/目标”的泛化。
- 训练 recipe：SFT 初始化稳定行为 + RL（改造版 RLOO）做 outcome 优化。

## 论文信息
- Paper: *Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization*
- arXiv:2602.22675 (v1, 2026-02-26)
- Team: OPPO AI Agent Team
- Context window：128K（论文实验设置）

## 我认为真正“可迁移”的贡献

### 1) 结构性替换：线性推理 → 并行证据获取（parallel task decomposition + concurrent tool execution）
他们直接把“线性 reasoning 轨迹”视为 scalability bottleneck，并提出：
- parallel task decomposition
- concurrent tool execution
- plan-driven 的 context 管理

**我的判断**：这是在重写推理-检索拓扑：从 chain 变成 star/graph。
收益可能来自：
- 降低早期错误假设的路径依赖（先铺证据，再汇总）
- 把 token budget 从冗长 CoT 转移到外部信息增益

### 2) Context management 是第一等公民（不是提示词技巧）
论文明确指出：长任务（如 BrowseComp）会超过 vanilla agent 的“有效上下文容量”，并且 SMTL 因为每步产生更多 tool observation 更容易爆 context。

他们的 inference 侧策略（来自文中实验设置段落）：
- **periodic plan refinement**：每 N=5 steps refine 一次计划
- **overflow-triggered compression**：当历史逼近 128K context budget 时触发压缩

**我的判断**：这块本质是 harness/系统能力（压缩、去重、pack）对 agent performance 的直接贡献；“模型聪明”无法替代。

### 3) 泛化来自“任务类型 + 指标形态”的统一训练闭环
他们把 agentic search 任务拆成两类：
- deterministic QA（BrowseComp / GAIA / WebWalker 等，主指标 accuracy）
- open-ended research（DeepResearch Bench/Gym 等，强调 coverage/coherence/synthesis）

并用 unified data synthesis pipeline 构造 multi-type search tasks + task-appropriate evaluation metrics。

**我的判断**：这是把 reward/metric 的异质性纳入数据/训练设计；否则 RL 容易在单一指标上学捷径，跨 setting 直接崩。

## Data synthesis pipeline（从 PDF 抓到的关键细节）

论文有一套高多样性/高密度的数据合成流程（Fig.3），核心步骤包括：
- raw corpus collection
- graph construction
- subgraph extraction
- QA generation + verification

Deep Search tasks：
- hop depth 约 2~5
- 对同一 subgraph 保留多个 hierarchical question variants
- 通过“answer frequency”之类的策略避免高频答案过度采样（文本里提到会做频率控制）

他们还强调并行证据路径之间的 **verifiable conditions / cross-validation**：
- 当多个 (i+1)-hop 节点存在语义关系时，显式编码依赖关系
- 用 LLM-based verification 防止信息泄露：如果能“提前推断出答案”，就重构问题或做信息 obfuscation；最多迭代 5 次

## Training recipe（关键：RL 用的不是 GRPO）

### 5.1 SFT
目的：先把 agent 行为空间拉到“稳定且高效”的搜索行为，再进入 RL。
他们还提到两点我很认同的过滤：
- active information acquisition（偏好主动拉证据的轨迹）
- trajectory efficiency optimization：多条成功轨迹只保留**正确且最短**的（interaction length 最短）

### 5.2 RL：改造版 RLOO + 序列级重要性采样修正
- RL 算法：**modified REINFORCE Leave-One-Out (RLOO)**（对比 GRPO，RLOO advantage 无偏）
- 修改点（文中明确写的 3 条）：
  1) 参考 DAPO，用 token-level loss
  2) 为缓解 training–inference mismatch（logprob 计算差异），做 **sequence-level importance sampling** 的 rollout correction
  3) 过滤部分 negative trajectories，避免环境不稳定诱导的 spurious behavior（稳定训练）

### Reward
- outcome-based reward
- LLM-as-a-judge 判 final answer 对错：对=1，错=0
- tool call 格式违规：立即终止并给 0（显式鼓励正确工具调用）

## 结果与现象（从 PDF 摘要/正文片段摘）

- BrowseComp：
  - SMTL-100：≈43.6%（文中对比 30B 级别时给的数字）
  - SMTL-300：48.6%（+5.0）
  - 相对 Mirothinker-v1.0：在 max 100 interaction steps 下 BrowseComp 平均 reasoning steps ↓70.7%（摘要）
  - latency：提到可到 2.6× 改善（正文贡献点段落）

- 短任务收益更小：GAIA 74.8% → 75.7%，WebWalker 74.9% → 76.5%（正文）
  - **解释**：额外 interaction budget 主要帮助“深多步证据聚合”，而不是弥补系统性推理错误。

- Ablation（BrowseComp）：
  - max steps 50→300：成功轨迹的 median steps 不明显上升 → 更多预算主要救“真正难例”，不是让简单题也变啰嗦。
  - retrieval top-k 4→8：性能显著上升（SMTL-300: 43.8→47.0；SMTL-100: 36.6→>41.8），说明**检索宽度**是关键杠杆。

## 机制审查（我建议继续追问的点）

1) **reasoning steps 的定义**：到底是内部思考 token、还是 agent step？不同定义对“效率”结论影响很大。
2) **并行的真实成本**：并行证据获取可能把成本从思考 token 转移到更多并发请求/网页解析/摘要压缩。
3) **泛化的来源**：要看 ablation（去掉 unified synthesis / 去掉 context management / 去掉并行工具执行）哪个掉得最厉害。

## 与 Vault 主线的映射（我的结论）

- 与 Agent RL/信号设计：SMTL 在做“信息增益优先”的结构化交互，这和 MIG / Search-P1 同一母题：**把交互结构当训练信号，而非只依赖最终答案 reward**。
- 与 Harness Engineering：并行 fetch、缓存、去重、压缩、pack，直接决定可用交互步数与证据密度。

## 下一步 TODO
- 读开源代码（论文标注 Open Source: Code）：确认其并行工具执行、context 压缩、plan refinement 的具体实现。
- 抽象成可复用 harness pattern：parallel fetch → dedup → compress → pack → judge。
