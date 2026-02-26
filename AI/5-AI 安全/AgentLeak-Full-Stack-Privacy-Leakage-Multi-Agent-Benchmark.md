---
title: "AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems"
brief: "Polytechnique Montréal：首个全栈 multi-agent 隐私泄漏 benchmark；inter-agent 内部通信泄漏率 68.8%（比用户可见输出高 2.5×）；output-only 审计漏掉 41.7% 隐私违规——多 Agent 系统不能只审计最终输出（arXiv:2602.11510）"
date: 2026-02-21
updated: 2026-02-22
tags:
  - ai/safety
  - multi-agent
  - privacy
  - benchmark
  - data-leakage
  - inter-agent-communication
  - information-flow-control
domain: ai/safety
arxiv: "2602.11510"
rating: 4
status: active
---

# AgentLeak: Multi-Agent 系统的全栈隐私泄漏 Benchmark

**arXiv**: 2602.11510  
**机构**: Polytechnique Montréal（蒙特利尔理工学院）  
**作者**: Faouzi El Yagoubi, Godwin Badu-Marfo, Ranwa Al Mallah  
**提交**: 2026-02-11  
**评分**: ★★★★☆  
**一句话**: 第一个覆盖 multi-agent 内部通信渠道的全栈隐私 benchmark——发现 output-only 审计会漏掉 **41.7%** 的隐私违规，inter-agent 消息泄漏率 **68.8%**，比用户输出高 2.5×。

## See Also

- [[OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]] — 姊妹论文，攻击端：单次 injection 通过 orchestrator 链泄漏；AgentLeak 是测量端：系统性量化哪个渠道泄漏多少——两篇殊途同归：output-only 审计是系统性盲区
- [[AutoInject-RL-Prompt-Injection-Attack|AutoInject]] — RL 生成 universal adversarial suffix（单 agent 攻击）；AgentLeak 的 A2 adversary 级别正是这类攻击，multi-agent 环境放大了攻击面
- [[CowCorpus-Human-Intervention-Modeling-Web-Agents|CowCorpus]] — 人类干预时机建模；AgentLeak 发现 C2 内部通信是最大泄漏渠道，指向在 C2 层引入 CowCorpus 式干预节点的高价值
- [[AI Agent 集体行为与安全漂移|AI Agent 集体行为与安全漂移]] — 多 Agent 安全漂移机制；AgentLeak 的 68.8% 内部泄漏率是"安全漂移"在隐私维度的量化证据
- [[AI安全与对齐-2026技术全景|AI 安全与对齐 2026 全景]] — 全栈安全框架；AgentLeak 提供的 channel-level 泄漏分类补充了全景中 multi-agent 安全的实证数据

---

## 核心发现（先看数字）

| 渠道 | 类型 | 泄漏率 |
|------|------|--------|
| **C1** 最终输出（用户可见） | 外部 | 27.2%（multi-agent）vs 43.2%（single-agent）|
| **C2** inter-agent 消息 | **内部** | **68.8%** ← 最高 |
| **C5** shared memory | **内部** | 46.7% |
| 系统总暴露（C1∨C2∨C5） | 综合 | **68.9%**（vs single-agent ~43%）|

**关键悖论**：多 Agent 系统的*最终输出*比单 Agent 更安全（27.2% vs 43.2%），但*系统整体风险*更高（68.9% vs ~43%）。漏的地方从用户可见渠道转移到了不可见的内部渠道。

---

## 问题定义

### 被遗漏的攻击面

现有安全审计的盲区：**Agent 之间说了什么，没有人检查**。

真实案例（论文引用的实际医疗系统观察）：
> 一个调度 agent 返回了干净的预约确认（通过了所有 output audit），  
> 但它委托给验证 agent 时传递的消息，包含了病人的完整病历。  
> 最终输出合规，隐私违规完全未被发现。

这个例子精准地说明了为什么"只看输出"是根本错误的。

### 七条泄漏渠道（C1-C7）

```
外部渠道（现有防御大多覆盖）：
  C1: 最终输出（用户看到的）
  C3: 工具调用参数（传给外部 API 的）
  C4: 工具返回值（从外部 API 收到的）
  C6: 系统日志 / telemetry
  C7: 持久化文件、生成的 artifacts

内部渠道（现有防御几乎不覆盖）：
  C2: inter-agent 消息（agent 之间的委托/协作消息）← 最危险
  C5: shared memory（跨 session 持久化记忆）← 第二危险
```

**核心洞察**：C2 和 C5 是真正的防御空白。LangChain/CrewAI/AutoGPT/AutoGen/MetaGPT **没有一个**对 C2 做默认的隐私控制。

### 数据最小化原则的可测量化

论文把 GDPR Art. 25 的"数据最小化"原则变成可测试的 empirical criterion：

```
vault V = {所有敏感字段}        # 29个字段/场景（中位数）
allowed set A ⊆ V               # 3个字段/场景（中位数）
vault-to-allowed ratio = 9.7:1  # 绝大多数字段不应该被传播

leakage event: ∃v ∈ (V \ A), ∃C_j ∈ {C1..C7}: sim(v, content(C_j)) > τ
```

一个场景中，agent 有权接触 3 个字段，但 vault 里有 29 个。它是否只传递了需要的 3 个？

---

## Benchmark 设计

### 规模

- **1,000 个场景**（250/领域 × 4 领域：healthcare, finance, legal, corporate）
- **4,979 条执行 traces**（5 个模型 × 多轮）
- **32 类攻击**（横跨 A0/A1/A2 三级对手）
- **7 个渠道**，每个渠道独立测量

### 三级对手模型

- **A0（无对手）**：测量系统默认行为下的 baseline 泄漏
- **A1（弱对手）**：控制外部信息源（恶意网页、污染的 API 响应、文档注入）
- **A2（强对手）**：控制系统组件（工具后门、agent 角色植入、复杂 prompt injection 链）

### 三层检测 Pipeline

1. **Canary 匹配**：精确字符串匹配，检测已知敏感字段是否出现在各渠道
2. **Pattern 提取**：正则 + NER（SSN、信用卡号、诊断代码等结构化 PII）
3. **LLM-as-Judge**：对 canary + pattern 无法捕捉的语义泄漏（比如：模型重述而非直接引用敏感数据）

---

## 实验结果

### 模型间对比

| 模型 | C1 外部泄漏 | C2 内部泄漏 | 评价 |
|------|------------|------------|------|
| **Claude 3.5 Sonnet** | **3.3%** | **28.1%** | 外部最低，内部也最低 |
| GPT-4o | ~27% | ~69% | 典型表现 |
| GPT-4o-mini | 更高 | 更高 | 成本低但安全性差 |
| Mistral Large | 中等 | 中等 | — |
| Llama 3.3 70B | 中等 | 中等 | — |

**Claude Sonnet 的 C2 虽然最低，但 28.1% 仍然很高**——说明即使最安全的模型，内部渠道泄漏仍是严重问题，不是靠模型对齐就能完全解决的。

### 跨域稳定模式

**规律**：C2 ≥ C1（inter-agent 消息泄漏率 ≥ 输出泄漏率）在所有 5 个模型 × 4 个领域的组合中**一致成立**。

这不是某个模型或领域的特例，而是 multi-agent 架构的结构性问题。

### Privacy-Utility Tradeoff

论文做了 Pareto 分析，结论：**现有防御无法同时维持任务完成率和内部渠道隐私保护**。

提高任务完成率需要 agent 传递更多上下文 → 内部泄漏增加。  
减少内部泄漏需要截断 agent 消息 → 任务失败率上升。

这是当前架构下的 fundamental tradeoff，不是 tuning 能解决的。

---

## 与现有工作的对比

| Benchmark | 多 Agent | 内部渠道 | 攻击类别 | 场景数 |
|-----------|---------|--------|---------|------|
| AgentDojo | ✗ | ✗ | 15 | 97 |
| AgentDAM | ✗ | ✗ | — | 246 |
| PrivacyLens | ✗ | ✗ | — | 493 |
| TOP-Bench | ✗ | ✗ | — | 180 |
| **AgentLeak** | **✓** | **✓** | **32** | **1000** |

AgentLeak 填补了一个明显的空白。没有竞品。

---

## 我的分析

### 这篇论文好在哪儿

**好问题 + 好数字**。

"Agent 之间说了什么"这个问题，在 OMNI-LEAK 之前几乎没有人系统研究。OMNI-LEAK 证明了攻击可行，AgentLeak 证明了问题的普遍性——不只是某个 orchestrator 模式，而是在所有主流框架、所有主流模型上，C2 泄漏率都比 C1 高 2.5×。

4,979 条 traces，1,000 个场景，这个数据量够说话。

### 理解"output-only 审计漏掉 41.7%"的精确含义

这个数字是这样算的：

- output-only 审计只看 C1（27.2% 泄漏）
- full-stack 审计看所有渠道（68.9% 系统暴露）
- 差值 41.7% = 多 agent 模式下新引入的、output audit 发现不了的泄漏

用 GDPR/HIPAA 的语言说：这些违规在监管层面是真实存在的，即使用户从来没有看到泄漏的数据。原因：inter-agent 消息和 shared memory 在大多数云部署中都是被基础设施提供商可以访问的。

### "Claude 更安全"的机制假设

论文的解释是 Constitutional AI 风格的安全训练可能 transfer 到内部渠道保护。

我更倾向于另一个假设：Claude 的训练更明确地惩罚"过度分享信息"（信息 minimization 作为对齐目标），而不仅仅是"拒绝有害请求"（这是 GPT 系列对齐的主要目标）。如果这个假设成立，那么对 LLM 的 RLHF 目标设计本身就是隐私防御的一部分。

### 对盾卫项目的启示

**直接影响**：memory_guard 目前只保护 MEMORY.md 的写入操作（C5 的一个子集）。AgentLeak 的框架提示我们 C2（inter-agent 消息）完全没有保护。

**盾卫 Phase 2.3 的新方向**：
1. 当一个 agent 调用工具时，工具返回的内容应该被扫描（C4）
2. 当一个 agent 向另一个 agent 传递消息时（如果有多 agent 场景），消息内容应该被扫描（C2）
3. 目前的 rate-based 检测对 internal channel 完全无效，因为 C2 的信息流速度通常很快

**更重要的洞察**：data minimization 作为一个设计原则，比 injection detection 更根本。与其检测"这条消息是否是攻击"，不如问"这个 agent 为什么需要传递这么多字段？"

### 局限

1. **场景生成方式**：论文没有详细说明 1,000 个场景如何生成，是否有 distribution bias
2. **检测 pipeline 的精确率**：LLM-as-Judge 在语义泄漏检测上的 precision/recall 数字没有给出
3. **对防御的建议**：论文证明了问题但防御方案部分偏薄——提了 data minimization prompting 但效果有限
4. **机构背景**：Polytechnique Montréal，安全研究质量一般不如 Oxford/Torr lab，论文严谨度可能略低于 OMNI-LEAK

---

## 关键数字汇总

```
单 Agent C1 输出泄漏率:    43.2%
多 Agent C1 输出泄漏率:    27.2%（反而更好，但是幻觉）
多 Agent C2 内部泄漏率:    68.8%（比 C1 高 2.5×）
多 Agent C5 内存泄漏率:    46.7%
系统总暴露（C1∨C2∨C5）:   68.9%（比 single-agent 高 1.6×）
output-only 审计漏掉:       41.7% 的违规

Claude 3.5 Sonnet 外部:   3.3%（最佳）
Claude 3.5 Sonnet 内部:   28.1%（最佳，但仍然很高）

数据集规模:
  - 1,000 场景 × 4 领域
  - 4,979 execution traces
  - 32 类攻击
  - 7 个渠道
  - vault-to-allowed ratio 中位数: 9.7:1
```

---

## 对 OMNI-LEAK + AgentLeak 的综合认识

两篇论文放在一起，给出了 multi-agent 安全的完整图景：

| 视角 | OMNI-LEAK | AgentLeak |
|------|-----------|-----------|
| 目标 | 攻击端：证明漏洞可以被利用 | 测量端：量化漏洞有多普遍 |
| 关注点 | orchestrator 模式的 exfiltration chain | 所有框架的所有 channel |
| 关键发现 | 4/5 模型可被攻陷 | C2 在所有模型/领域普遍泄漏 |
| 防御指向 | 需要跨 agent 信息流控制 | 需要 internal-channel 访问控制 |
| 共同结论 | **output-only 审计是系统性盲区** | **output-only 审计是系统性盲区** |

两篇都得出了同一个结论，用的是完全不同的方法。这种"殊途同归"的收敛是高置信度的信号。

---

## See Also

- [[PI-Landscape-SoK-Prompt-Injection-Taxonomy-Defense|PI-Landscape SoK]] ⭐ — SoK 将 AgentPI benchmark 定位为专门覆盖 context-dependent 场景的评测；AgentLeak 从隐私泄漏角度互补，两篇共同描绘多Agent安全评测的全景
- [[OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]] — 殊途同归：两篇均发现"output-only 审计是系统性盲区"，但方法完全不同——OMNI-LEAK 是攻击演示，AgentLeak 是量化测量

## Tags
#MultiAgent #PrivacyLeakage #InternalChannels #InterAgentCommunication #Benchmark #DataMinimization #FullStack #AgentSafety #InformationFlowControl #盾卫 #GDPR
