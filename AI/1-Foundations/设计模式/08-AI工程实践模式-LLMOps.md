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

```
Zero-shot（无示例）：
  "把这段话翻译成英文：xxx"  → 效果一般

Few-shot（有示例）：
  "翻译示例：
   中文：你好  英文：Hello
   中文：谢谢  英文：Thank you
   中文：再见  英文：Goodbye
   请翻译：美好的一天"  → 效果更好
```

**关键设计**：
- 示例要有代表性，覆盖边界情况
- 示例顺序影响结果（最新的示例权重更高）
- 过多示例消耗 context，需要权衡

**AI 映射**：NL2SQL 中用 few-shot 示例（经典 SQL 模式），显著提升准确率。

---

### 2. 思维链提示（Chain-of-Thought，CoT）

**问题**：复杂推理任务，直接要答案效果差——强制 LLM 展示推理过程，质量显著提升。

```
直接问：
  "苹果5个，送掉2个，又买了3个，还有几个？" → 可能直接说6（错误）

CoT：
  "苹果5个，送掉2个，又买了3个，还有几个？请一步一步思考。"
  → "先算送掉：5-2=3，再算买入：3+3=6，所以还有6个。" → 正确
```

**变体**：
- **Zero-shot CoT**：只加"Let's think step by step"
- **Few-shot CoT**：给带推理过程的示例
- **Tree of Thoughts**：多个推理路径并行，选最优（Beam Search 的 prompt 版）

---

### 3. 提示词模板与版本管理

**问题**：Prompt 是 AI 系统的"代码"，但散落在各处、没有版本控制。

```
Prompt 版本管理：
  v1.0: "你是一个助手，请回答用户问题"
  v1.1: "你是一个专业的数据分析师，..."（加角色）
  v2.0: "你是一个专业的数据分析师，请遵循以下规范..."（加规范）

变更必须：
  1. 有版本号
  2. 记录变更原因
  3. 有回滚能力
  4. 有 A/B 测试数据支撑
```

**Prompt 组成结构**：
```
[系统提示 System Prompt]
  ├─ 角色定义（你是谁）
  ├─ 能力边界（你能做什么/不能做什么）
  ├─ 输出格式（JSON/Markdown/自然语言）
  └─ 关键约束（安全规则、语气要求）

[Few-shot 示例]（可选）

[上下文注入]
  ├─ 检索到的文档（RAG）
  └─ 历史对话摘要

[用户输入]
```

**AI 映射**：你们的 HEARTBEAT.md / SOUL.md 就是系统提示的版本化管理。

---

### 4. 结构化输出（Structured Output）

**问题**：LLM 输出是自然语言，下游系统需要结构化数据，如何可靠地解析？

```
不可靠方式：
  → LLM 输出自由格式 → 正则解析（容易失败）

可靠方式1：JSON Mode
  → 系统提示要求 JSON → 模型保证输出合法 JSON

可靠方式2：Function Calling / Tool Use
  → 定义 schema → 模型直接输出符合 schema 的结构

可靠方式3：Instructor / Pydantic
  → 代码层校验 → 不符合就重试（带错误信息）
```

**AI 映射**：
- Agent 的工具调用参数解析（必须结构化）
- NL2SQL 的中间表示（NL2Param 的 param 就是结构化输出）

---

## 二、RAG 工程模式

### 5. 朴素 RAG vs 高级 RAG

```
朴素 RAG（Naive RAG）：
  查询 → 向量检索 → 拼接上下文 → 生成
  问题：检索质量差、上下文过长、答案幻觉

高级 RAG（Advanced RAG）三个阶段的优化：

Pre-Retrieval（检索前）：
  查询重写（Query Rewriting）     ← 用户意图模糊时
  查询扩展（Query Expansion）     ← 生成多个变体
  HyDE（假设文档嵌入）            ← 先生成假设答案再检索

Retrieval（检索）：
  混合检索（向量 + BM25）         ← 语义 + 关键词互补
  多粒度检索（句子/段落/文档）
  元数据过滤（时间/来源/类型）

Post-Retrieval（检索后）：
  重排序（Reranker）              ← 精排，提升相关性
  上下文压缩（Context Compression）← 去掉不相关片段
  答案融合（Answer Fusion）       ← 多文档答案合并
```

---

### 6. 分块策略（Chunking Strategy）

**问题**：文档怎么切块直接影响检索质量——切太大上下文溢出，切太小丢失语义。

```
固定大小切块：
  按字符数切（简单，语义不完整）

语义切块（Semantic Chunking）：
  按段落/标题切（语义完整）

滑动窗口：
  块A = [1,2,3]，块B = [2,3,4]，块C = [3,4,5]（重叠，保上下文）

层级切块（Hierarchical Chunking）：
  大块（文档摘要）+ 小块（具体内容）
  检索时两级联动
```

---

### 7. 检索质量评测（RAG Evaluation）

```
五个维度：
  Context Relevance：检索到的内容和问题相关吗？
  Answer Faithfulness：回答基于检索内容，没有幻觉吗？
  Answer Relevance：回答和问题相关吗？
  Context Recall：应该检索到的关键信息，检索到了吗？
  Groundedness：每个声明都有来源支撑吗？

工具：RAGAS 框架（自动化 RAG 评测）
```

---

## 三、评测模式（Evaluation Patterns）⭐

### 8. LLM-as-Judge

**问题**：AI 系统的输出质量难以自动评测（不是代码，没有标准答案），人工评测成本高。

```
传统评测：人工打分 → 慢、贵、不一致

LLM-as-Judge：
  用强模型（GPT-4/Claude）评测弱模型的输出
  评测维度：准确性、相关性、完整性、格式合规

Pipeline：
  [测试集] → [被评测 Agent] → [输出] → [Judge LLM] → [评分 + 理由]
                                                            ↓
                                                     汇总报告
```

**关键注意**：
- Judge 模型不能是被测模型（避免自我评分偏差）
- 评测 prompt 要明确评分标准
- 要定期用人工标注校准 Judge 模型的评分一致性

**AI 映射**：
- Stanford Agentic Reviewer（FARS 评测）就是 LLM-as-Judge
- 你们的分析 Agent 输出质量评测

---

### 9. 在线评测 vs 离线评测

```
离线评测（Offline Evaluation）：
  准备测试集 → 批量跑 → 统计指标
  优点：可重复，成本低
  缺点：不代表真实用户分布

在线评测（Online Evaluation）：
  真实流量 → 采样 → 实时/准实时评分
  A/B 测试：新旧版本对比
  
Golden Test Set（黄金测试集）：
  精心选择的代表性测试用例，每次发布前必跑
  覆盖：正常 case + 边界 case + 故意刁难的 case
```

---

### 10. 回归测试套件（Regression Test Suite）

**问题**：每次改 Prompt / 换模型，怎么确认没有破坏已有能力？

```
构建回归测试套件：
  场景1: 普通查询（应该正常回答）
  场景2: 边界情况（应该优雅处理）
  场景3: 对抗性输入（应该拒绝）
  场景4: 历史 bug（不应该复现）

每次变更：自动跑全套 → 输出对比报告
如果某个 case 质量下降 → 阻断发布
```

---

## 四、护栏模式（Guardrails）⭐

### 11. 输入护栏（Input Guardrails）

**问题**：用户输入可能包含有害内容、注入攻击、越权操作——在进入 LLM 前过滤。

```
用户输入
    ↓
[输入护栏层]
  ├─ Prompt Injection 检测（"忽略之前的指令..."）
  ├─ 有害内容过滤（暴力/色情/违法）
  ├─ PII 检测（手机号/身份证号脱敏）
  ├─ 主题相关性检查（超出业务范围的问题）
  └─ 权限校验（这个用户有权限做这件事吗？）
    ↓
LLM / Agent
```

### 12. 输出护栏（Output Guardrails）

```
LLM 输出
    ↓
[输出护栏层]
  ├─ 幻觉检测（声明是否有来源支撑？）
  ├─ 事实核查（关键数字/日期/名称校验）
  ├─ 格式校验（JSON schema 是否合法？）
  ├─ 有害内容过滤（二次审核）
  └─ 品牌安全（不能说竞争对手的好话？）
    ↓
用户
```

**护栏工具**：Guardrails AI、NVIDIA NeMo Guardrails、自定义规则引擎。

**AI 映射**：
- 你们美团分析 Agent 的输出：数字/结论需要来源核实
- NL2SQL：生成的 SQL 要校验语法 + 执行权限

---

## 五、可观测性模式（Observability）⭐

### 13. 追踪（Tracing）

**问题**：一个用户请求经过了哪些 Agent、调用了哪些工具、每步耗时多少——出问题时怎么排查？

```
请求 ID: req_abc123
    │
    ├─ [00:00] 用户请求到达
    ├─ [00:01] 意图识别（耗时 500ms）
    ├─ [00:02] 检索相关文档（耗时 300ms）
    │    └─ 检索到 3 个文档（相关性分数: 0.87, 0.79, 0.65）
    ├─ [00:03] 调用工具：web_search（耗时 1200ms）
    │    └─ 返回 5 条结果
    ├─ [00:04] LLM 推理（耗时 2000ms，tokens: 1847/512）
    └─ [00:05] 输出护栏检查（耗时 200ms）
    
总耗时：4200ms  总 token：2359  总成本：$0.0047
```

**必须记录的指标**：
- 每个步骤的延迟
- Token 消耗（输入/输出）
- 工具调用成功/失败
- 最终输出质量评分

**工具**：LangSmith、Langfuse、Arize Phoenix、自研日志系统

---

### 14. 成本监控模式

```
Token 成本三维监控：

1. 每请求成本（Cost per Request）
   → 找出异常高成本的请求（无限循环/超长上下文）

2. 每功能成本（Cost per Feature）
   → 哪个功能最烧钱？值得优化吗？

3. 成本趋势（Cost Trend）
   → 新模型/新Prompt 上线后成本变化
   
降成本三板斧：
  缓存（Semantic Cache）：相似问题复用答案
  提示词压缩（Prompt Compression）：去掉冗余
  模型降级路由：简单任务用小模型，复杂任务用大模型
```

---

### 15. 降级策略（Graceful Degradation）

**问题**：LLM 服务不可用时，系统应该优雅降级而不是直接崩溃。

```
降级层次：
  Level 1（正常）：GPT-4 / Claude Opus
      ↓（超时/失败）
  Level 2（降级）：GPT-3.5 / Claude Haiku
      ↓（全挂了）
  Level 3（兜底）：规则引擎 / 缓存答案
      ↓（什么都没有）
  Level 4（最低限度）：返回友好错误信息，告知用户稍后重试
```

---

## LLMOps 完整流水线

```
开发阶段：
  Prompt 设计 → 少样本构建 → 本地测试 → 离线评测

发布阶段：
  金丝雀发布（1%）→ 特性开关控制 → 蓝绿切换

生产阶段：
  输入护栏 → Agent 执行 → 输出护栏
      ↓
  Tracing（全链路追踪）
  成本监控
  质量监控（LLM-as-Judge 采样评测）

迭代阶段：
  线上数据 → 发现问题 → Prompt 优化 → 回归测试 → 再发布
```

---

## AI 应用工程师核心能力图谱

```
初级（会用）
  └─ 会写 Prompt，会调 API，会用 LangChain

中级（会建）
  ├─ RAG 系统设计（分块/检索/评测）
  ├─ Agent 工具设计（六边形架构）
  ├─ 结构化输出（Instructor / Function Calling）
  └─ 基础评测（LLM-as-Judge）

高级（会治）
  ├─ Prompt 版本管理 + A/B 测试
  ├─ 护栏设计（输入/输出双层）
  ├─ 全链路可观测性（Tracing + 成本监控）
  ├─ 降级策略 + 熔断器
  └─ 回归测试套件

架构师（会设计系统）
  ├─ 六边形架构（可测试/可替换）
  ├─ 事件驱动 Multi-Agent
  ├─ 绞杀者模式（AI 替换传统系统）
  └─ 领域驱动设计划定 Agent 边界
```
