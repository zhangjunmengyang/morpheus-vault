---
title: AI 工程实践模式（LLMOps）
tags: [design-pattern, llmops, prompt-engineering, evaluation, observability, guardrails, production]
date: 2026-02-28
---

# AI 工程实践模式（LLMOps）

> 这是 AI 应用工程师和"会用 ChatGPT"之间最大的鸿沟。
> 原型跑通是起点，能在生产环境稳定运行才是真正的工程能力。
> 这篇讲的不是"怎么设计对象"，而是**"AI 系统怎么活在生产环境里"**。

---

## 一、Prompt 工程模式

### 1. 少样本提示（Few-Shot Prompting）

**问题**：直接描述任务效果不好，但给几个例子，LLM 立刻懂了。

````mermaid
flowchart TD
  n1["Zero-shot（无示例）："]
  n2["\"把这段话翻译成英文：xxx\"  → 效果一般"]
  n3["Few-shot（有示例）："]
  n4["\"翻译示例："]
  n5["中文：你好  英文：Hello"]
  n6["中文：谢谢  英文：Thank you"]
  n7["中文：再见  英文：Goodbye"]
  n8["请翻译：美好的一天\"  → 效果更好"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
````

**关键设计**：
- 示例要有代表性，覆盖边界情况
- 示例顺序影响结果（最新的示例权重更高）
- 过多示例消耗 context，需要权衡

**AI 映射**：NL2SQL 中用 few-shot 示例（经典 SQL 模式），显著提升准确率。

---

### 2. 思维链提示（Chain-of-Thought，CoT）

**问题**：复杂推理任务，直接要答案效果差——强制 LLM 展示推理过程，质量显著提升。

````mermaid
flowchart TD
  n1["直接问："]
  n2["\"苹果5个，送掉2个，又买了3个，还有几个？\" → 可能直接说6（错误）"]
  n3["CoT："]
  n4["\"苹果5个，送掉2个，又买了3个，还有几个？请一步一步思考。\""]
  n5["→ \"先算送掉：5-2=3，再算买入：3+3=6，所以还有6个。\" → 正确"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
````

**变体**：
- **Zero-shot CoT**：只加"Let's think step by step"
- **Few-shot CoT**：给带推理过程的示例
- **Tree of Thoughts**：多个推理路径并行，选最优（Beam Search 的 prompt 版）

---

### 3. 提示词模板与版本管理

**问题**：Prompt 是 AI 系统的"代码"，但散落在各处、没有版本控制。

````mermaid
flowchart TD
  n1["Prompt 版本管理："]
  n2["v1.0: \"你是一个助手，请回答用户问题\""]
  n3["v1.1: \"你是一个专业的数据分析师，...\"（加角色）"]
  n4["v2.0: \"你是一个专业的数据分析师，请遵循以下规范...\"（加规范）"]
  n5["变更必须："]
  n6["1. 有版本号"]
  n7["2. 记录变更原因"]
  n8["3. 有回滚能力"]
  n9["4. 有 A/B 测试数据支撑"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
````

**Prompt 组成结构**：
````mermaid
flowchart TD
  n1["[系统提示 System Prompt]"]
  n2["├─ 角色定义（你是谁）"]
  n3["├─ 能力边界（你能做什么/不能做什么）"]
  n4["├─ 输出格式（JSON/Markdown/自然语言）"]
  n5["└─ 关键约束（安全规则、语气要求）"]
  n6["[Few-shot 示例]（可选）"]
  n7["[上下文注入]"]
  n8["├─ 检索到的文档（RAG）"]
  n9["└─ 历史对话摘要"]
  n10["[用户输入]"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
````

**AI 映射**：你们的 HEARTBEAT.md / SOUL.md 就是系统提示的版本化管理。

---

### 4. 结构化输出（Structured Output）

**问题**：LLM 输出是自然语言，下游系统需要结构化数据，如何可靠地解析？

````mermaid
flowchart TD
  n1["不可靠方式："]
  n2["→ LLM 输出自由格式 → 正则解析（容易失败）"]
  n3["可靠方式1：JSON Mode"]
  n4["→ 系统提示要求 JSON → 模型保证输出合法 JSON"]
  n5["可靠方式2：Function Calling / Tool Use"]
  n6["→ 定义 schema → 模型直接输出符合 schema 的结构"]
  n7["可靠方式3：Instructor / Pydantic"]
  n8["→ 代码层校验 → 不符合就重试（带错误信息）"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
````

**AI 映射**：
- Agent 的工具调用参数解析（必须结构化）
- NL2SQL 的中间表示（NL2Param 的 param 就是结构化输出）

---

## 二、RAG 工程模式

### 5. 朴素 RAG vs 高级 RAG

````mermaid
flowchart TD
  n1["朴素 RAG（Naive RAG）："]
  n2["查询 → 向量检索 → 拼接上下文 → 生成"]
  n3["问题：检索质量差、上下文过长、答案幻觉"]
  n4["高级 RAG（Advanced RAG）三个阶段的优化："]
  n5["Pre-Retrieval（检索前）："]
  n6["查询重写（Query Rewriting）     ← 用户意图模糊时"]
  n7["查询扩展（Query Expansion）     ← 生成多个变体"]
  n8["HyDE（假设文档嵌入）            ← 先生成假设答案再检索"]
  n9["Retrieval（检索）："]
  n10["混合检索（向量 + BM25）         ← 语义 + 关键词互补"]
  n11["多粒度检索（句子/段落/文档）"]
  n12["元数据过滤（时间/来源/类型）"]
  n13["Post-Retrieval（检索后）："]
  n14["重排序（Reranker）              ← 精排，提升相关性"]
  n15["上下文压缩（Context Compression）← 去掉不相关片段"]
  n16["答案融合（Answer Fusion）       ← 多文档答案合并"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
  n10 --> n11
  n11 --> n12
  n12 --> n13
  n13 --> n14
  n14 --> n15
  n15 --> n16
````

---

### 6. 分块策略（Chunking Strategy）

**问题**：文档怎么切块直接影响检索质量——切太大上下文溢出，切太小丢失语义。

````mermaid
flowchart TD
  n1["固定大小切块："]
  n2["按字符数切（简单，语义不完整）"]
  n3["语义切块（Semantic Chunking）："]
  n4["按段落/标题切（语义完整）"]
  n5["滑动窗口："]
  n6["块A = [1,2,3]，块B = [2,3,4]，块C = [3,4,5]（重叠，保上下文）"]
  n7["层级切块（Hierarchical Chunking）："]
  n8["大块（文档摘要）+ 小块（具体内容）"]
  n9["检索时两级联动"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
````

---

### 7. 检索质量评测（RAG Evaluation）

````mermaid
flowchart TD
  n1["五个维度："]
  n2["Context Relevance：检索到的内容和问题相关吗？"]
  n3["Answer Faithfulness：回答基于检索内容，没有幻觉吗？"]
  n4["Answer Relevance：回答和问题相关吗？"]
  n5["Context Recall：应该检索到的关键信息，检索到了吗？"]
  n6["Groundedness：每个声明都有来源支撑吗？"]
  n7["工具：RAGAS 框架（自动化 RAG 评测）"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
````

---

## 三、评测模式（Evaluation Patterns）⭐

### 8. LLM-as-Judge

**问题**：AI 系统的输出质量难以自动评测（不是代码，没有标准答案），人工评测成本高。

````mermaid
flowchart TD
  n1["传统评测：人工打分 → 慢、贵、不一致"]
  n2["LLM-as-Judge："]
  n3["用强模型（GPT-4/Claude）评测弱模型的输出"]
  n4["评测维度：准确性、相关性、完整性、格式合规"]
  n5["Pipeline："]
  n6["[测试集] → [被评测 Agent] → [输出] → [Judge LLM] → [评分 + 理由]"]
  n7["↓"]
  n8["汇总报告"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
````

**关键注意**：
- Judge 模型不能是被测模型（避免自我评分偏差）
- 评测 prompt 要明确评分标准
- 要定期用人工标注校准 Judge 模型的评分一致性

**AI 映射**：
- Stanford Agentic Reviewer（FARS 评测）就是 LLM-as-Judge
- 你们的分析 Agent 输出质量评测

---

### 9. 在线评测 vs 离线评测

````mermaid
flowchart TD
  n1["离线评测（Offline Evaluation）："]
  n2["准备测试集 → 批量跑 → 统计指标"]
  n3["优点：可重复，成本低"]
  n4["缺点：不代表真实用户分布"]
  n5["在线评测（Online Evaluation）："]
  n6["真实流量 → 采样 → 实时/准实时评分"]
  n7["A/B 测试：新旧版本对比"]
  n8["Golden Test Set（黄金测试集）："]
  n9["精心选择的代表性测试用例，每次发布前必跑"]
  n10["覆盖：正常 case + 边界 case + 故意刁难的 case"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
````

---

### 10. 回归测试套件（Regression Test Suite）

**问题**：每次改 Prompt / 换模型，怎么确认没有破坏已有能力？

````mermaid
flowchart TD
  n1["构建回归测试套件："]
  n2["场景1: 普通查询（应该正常回答）"]
  n3["场景2: 边界情况（应该优雅处理）"]
  n4["场景3: 对抗性输入（应该拒绝）"]
  n5["场景4: 历史 bug（不应该复现）"]
  n6["每次变更：自动跑全套 → 输出对比报告"]
  n7["如果某个 case 质量下降 → 阻断发布"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
````

---

## 四、护栏模式（Guardrails）⭐

### 11. 输入护栏（Input Guardrails）

**问题**：用户输入可能包含有害内容、注入攻击、越权操作——在进入 LLM 前过滤。

````mermaid
flowchart TD
  n1["用户输入"]
  n2["↓"]
  n3["[输入护栏层]"]
  n4["├─ Prompt Injection 检测（\"忽略之前的指令...\"）"]
  n5["├─ 有害内容过滤（暴力/色情/违法）"]
  n6["├─ PII 检测（手机号/身份证号脱敏）"]
  n7["├─ 主题相关性检查（超出业务范围的问题）"]
  n8["└─ 权限校验（这个用户有权限做这件事吗？）"]
  n9["↓"]
  n10["LLM / Agent"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
````

### 12. 输出护栏（Output Guardrails）

````mermaid
flowchart TD
  n1["LLM 输出"]
  n2["↓"]
  n3["[输出护栏层]"]
  n4["├─ 幻觉检测（声明是否有来源支撑？）"]
  n5["├─ 事实核查（关键数字/日期/名称校验）"]
  n6["├─ 格式校验（JSON schema 是否合法？）"]
  n7["├─ 有害内容过滤（二次审核）"]
  n8["└─ 品牌安全（不能说竞争对手的好话？）"]
  n9["↓"]
  n10["用户"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
````

**护栏工具**：Guardrails AI、NVIDIA NeMo Guardrails、自定义规则引擎。

**AI 映射**：
- 你们美团分析 Agent 的输出：数字/结论需要来源核实
- NL2SQL：生成的 SQL 要校验语法 + 执行权限

---

## 五、可观测性模式（Observability）⭐

### 13. 追踪（Tracing）

**问题**：一个用户请求经过了哪些 Agent、调用了哪些工具、每步耗时多少——出问题时怎么排查？

````mermaid
flowchart TD
  n1["请求 ID: req_abc123"]
  n2["│"]
  n3["├─ [00:00] 用户请求到达"]
  n4["├─ [00:01] 意图识别（耗时 500ms）"]
  n5["├─ [00:02] 检索相关文档（耗时 300ms）"]
  n6["│    └─ 检索到 3 个文档（相关性分数: 0.87, 0.79, 0.65）"]
  n7["├─ [00:03] 调用工具：web_search（耗时 1200ms）"]
  n8["│    └─ 返回 5 条结果"]
  n9["├─ [00:04] LLM 推理（耗时 2000ms，tokens: 1847/512）"]
  n10["└─ [00:05] 输出护栏检查（耗时 200ms）"]
  n11["总耗时：4200ms  总 token：2359  总成本：$0.0047"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
  n10 --> n11
````

**必须记录的指标**：
- 每个步骤的延迟
- Token 消耗（输入/输出）
- 工具调用成功/失败
- 最终输出质量评分

**工具**：LangSmith、Langfuse、Arize Phoenix、自研日志系统

---

### 14. 成本监控模式

````mermaid
flowchart TD
  n1["Token 成本三维监控："]
  n2["1. 每请求成本（Cost per Request）"]
  n3["→ 找出异常高成本的请求（无限循环/超长上下文）"]
  n4["2. 每功能成本（Cost per Feature）"]
  n5["→ 哪个功能最烧钱？值得优化吗？"]
  n6["3. 成本趋势（Cost Trend）"]
  n7["→ 新模型/新Prompt 上线后成本变化"]
  n8["降成本三板斧："]
  n9["缓存（Semantic Cache）：相似问题复用答案"]
  n10["提示词压缩（Prompt Compression）：去掉冗余"]
  n11["模型降级路由：简单任务用小模型，复杂任务用大模型"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
  n10 --> n11
````

---

### 15. 降级策略（Graceful Degradation）

**问题**：LLM 服务不可用时，系统应该优雅降级而不是直接崩溃。

````mermaid
flowchart TD
  n1["降级层次："]
  n2["Level 1（正常）：GPT-4 / Claude Opus"]
  n3["↓（超时/失败）"]
  n4["Level 2（降级）：GPT-3.5 / Claude Haiku"]
  n5["↓（全挂了）"]
  n6["Level 3（兜底）：规则引擎 / 缓存答案"]
  n7["↓（什么都没有）"]
  n8["Level 4（最低限度）：返回友好错误信息，告知用户稍后重试"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
````

---

## LLMOps 完整流水线

````mermaid
flowchart TD
  n1["开发阶段："]
  n2["Prompt 设计 → 少样本构建 → 本地测试 → 离线评测"]
  n3["发布阶段："]
  n4["金丝雀发布（1%）→ 特性开关控制 → 蓝绿切换"]
  n5["生产阶段："]
  n6["输入护栏 → Agent 执行 → 输出护栏"]
  n7["↓"]
  n8["Tracing（全链路追踪）"]
  n9["成本监控"]
  n10["质量监控（LLM-as-Judge 采样评测）"]
  n11["迭代阶段："]
  n12["线上数据 → 发现问题 → Prompt 优化 → 回归测试 → 再发布"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
  n10 --> n11
  n11 --> n12
````

---

## AI 应用工程师核心能力图谱

````mermaid
flowchart TD
  n1["初级（会用）"]
  n2["└─ 会写 Prompt，会调 API，会用 LangChain"]
  n3["中级（会建）"]
  n4["├─ RAG 系统设计（分块/检索/评测）"]
  n5["├─ Agent 工具设计（六边形架构）"]
  n6["├─ 结构化输出（Instructor / Function Calling）"]
  n7["└─ 基础评测（LLM-as-Judge）"]
  n8["高级（会治）"]
  n9["├─ Prompt 版本管理 + A/B 测试"]
  n10["├─ 护栏设计（输入/输出双层）"]
  n11["├─ 全链路可观测性（Tracing + 成本监控）"]
  n12["├─ 降级策略 + 熔断器"]
  n13["└─ 回归测试套件"]
  n14["架构师（会设计系统）"]
  n15["├─ 六边形架构（可测试/可替换）"]
  n16["├─ 事件驱动 Multi-Agent"]
  n17["├─ 绞杀者模式（AI 替换传统系统）"]
  n18["└─ 领域驱动设计划定 Agent 边界"]
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
  n5 --> n6
  n6 --> n7
  n7 --> n8
  n8 --> n9
  n9 --> n10
  n10 --> n11
  n11 --> n12
  n12 --> n13
  n13 --> n14
  n14 --> n15
  n15 --> n16
  n16 --> n17
  n17 --> n18
````
