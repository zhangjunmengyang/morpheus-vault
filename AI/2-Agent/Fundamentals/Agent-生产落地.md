---
title: "AI Agent 生产落地经验"
brief: "Demo→Production 的核心鸿沟：可靠性/成本/延迟/错误恢复；工程经验总结：幂等设计/人工兜底/成本控制/监控体系；面试热点——生产级 Agent 与 Demo 的本质差异"
date: 2026-02-14
updated: 2026-02-23
tags: [agent, production, engineering, interview]
type: note
---

# AI Agent 生产落地经验

## 1. 从 Demo 到 Production 的鸿沟

Demo 阶段的 Agent 通常跑 happy path：输入规范、网络稳定、不考虑成本。一旦上生产，面临的核心挑战：

| 维度 | Demo | Production |
|------|------|-----------|
| 输入 | 精心构造的 prompt | 用户随意输入、对抗性输入、多语言混杂 |
| 可靠性 | 偶尔失败可以重跑 | 要求 99.9%+ 可用性，失败需自动恢复 |
| 延迟 | 无所谓 | 用户感知延迟 < 3s（streaming 缓解） |
| 成本 | 跑几次无所谓 | 百万级调用下 token 费用是核心成本 |
| 安全 | 信任输入 | 必须防 prompt injection、数据泄露、越权 |
| 可观测 | print 调试 | 需要完整 trace、指标、告警 |

**核心认知：Agent 不是一个 prompt，而是一个分布式系统。** 它调用外部 API（LLM、tool、数据库），每个环节都可能失败。用分布式系统的思维去设计 Agent 才能上生产。

---

## 2. 错误处理

### 2.1 Tool Call 失败重试

```
策略：指数退避 + 抖动（Exponential Backoff + Jitter）
- 第 1 次重试：1s + random(0, 0.5s)
- 第 2 次重试：2s + random(0, 1s)
- 第 3 次重试：4s + random(0, 2s)
- 最大重试次数：通常 3 次
```

关键细节：
- **区分可重试错误和不可重试错误**：429（限流）、500/502/503（服务端临时故障）可重试；400（参数错误）、403（权限不足）不应重试
- **重试前检查幂等性**：写操作必须确保幂等，否则重试可能导致重复执行（如重复下单）
- **LLM 调用重试的特殊性**：LLM 输出不确定，重试可能得到不同结果。如果上一步 tool call 已经执行成功但 LLM 响应超时，重试时需要把已执行的结果注入上下文

### 2.2 Graceful Degradation

当某个能力不可用时，Agent 应降级而非崩溃：

```python
# 伪代码示意
async def execute_with_fallback(task):
    try:
        return await primary_tool(task)
    except ToolUnavailableError:
        # 降级到备选方案
        logger.warn("Primary tool unavailable, falling back")
        return await fallback_tool(task)
    except AllToolsFailedError:
        # 最终兜底：让 LLM 用自身知识回答，并标注数据可能不是最新的
        return await llm_answer_with_disclaimer(task)
```

### 2.3 Fallback 策略层级

1. **同功能备选工具**：如搜索 API A 挂了，切换到 API B
2. **降低精度但可用**：如实时数据拿不到，用缓存数据 + 时间戳标注
3. **LLM 自身知识兜底**：明确告知用户数据截止日期
4. **优雅拒绝**：告知用户当前无法完成，建议稍后重试

---

## 3. 超时控制

### 3.1 三层超时机制

```
┌─────────────────────────────────────────────┐
│  总超时（Session Timeout）: 5min            │
│  ┌──────────────────────────────────────┐   │
│  │  单步超时（Step Timeout）: 30s       │   │
│  │  ┌───────────────────────────────┐   │   │
│  │  │  工具超时（Tool Timeout）: 10s│   │   │
│  │  └───────────────────────────────┘   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

- **Tool Timeout**：单个工具调用的超时，如 HTTP 请求 10s、数据库查询 5s
- **Step Timeout**：一次 LLM 推理 + tool call 的总超时，防止 LLM 陷入死循环
- **Session Timeout**：整个 Agent 任务的总超时，防止无限递归或长时间占用资源

### 3.2 心跳检测

对于长时间运行的 Agent 任务：
- Agent 定期发送心跳（如每 30s），表明仍在正常工作
- 监控系统检测心跳间隔，超过阈值则判定任务挂起
- 挂起任务自动触发 checkpoint 保存 + 告警

### 3.3 最大步数限制（Max Iterations）

除了时间超时，还需限制 Agent 的推理步数（如最多 20 步），防止：
- LLM 陷入循环（反复调用同一个工具）
- 无限递归的 sub-agent 调用
- 成本失控

---

## 4. 可观测性

### 4.1 Trace/Span 追踪

Agent 的一次完整执行是一个 **Trace**，其中每个 LLM 调用、tool call 是一个 **Span**：

```
Trace: user_query_12345
├── Span: llm_call_1 (model=gpt-4, tokens=1200, latency=2.3s)
│   └── Span: tool_call_search (query="...", latency=0.8s, status=ok)
├── Span: llm_call_2 (model=gpt-4, tokens=800, latency=1.5s)
│   └── Span: tool_call_database (query="...", latency=0.3s, status=ok)
└── Span: llm_call_3 (model=gpt-4, tokens=600, latency=1.1s)
    └── result: "最终回答..."
```

主流工具：
- **LangSmith**：LangChain 生态，开箱即用，支持 playground 回放
- **Phoenix (Arize)**：开源，支持 OpenTelemetry 标准，适合自建
- **OpenTelemetry + Jaeger**：通用方案，适合已有 observability 基础设施的团队

### 4.2 关键指标监控

```
Token 消耗指标：
- input_tokens_total / output_tokens_total（按模型、按任务类型）
- cost_per_request（单次请求成本）
- cost_daily / cost_monthly（日/月总成本）

质量指标：
- task_success_rate（任务成功率）
- tool_call_failure_rate（工具调用失败率）
- avg_steps_per_task（平均步数，越少越好）
- user_satisfaction_score（用户满意度）

延迟指标：
- time_to_first_token（首 token 延迟）
- total_latency（端到端延迟）
- llm_latency vs tool_latency（瓶颈分析）
```

### 4.3 成本控制

- **预算告警**：设置日预算上限，超过阈值（如 80%）触发告警
- **用量异常检测**：某个用户/任务的 token 消耗异常偏高时自动熔断
- **请求级成本估算**：在请求开始前估算成本，超过阈值拒绝或降级

---

## 5. 安全防护

### 5.1 Prompt Injection 防御

**攻击类型：**
- **直接注入**：用户输入中包含 "忽略之前的指令，执行 XXX"
- **间接注入**：工具返回的数据中嵌入恶意指令（如网页内容中隐藏 prompt）

**防御手段（多层防御）：**

```
Layer 1: 输入过滤
- 关键词/正则检测（"ignore previous", "system prompt" 等）
- 专用分类模型检测注入（如 Rebuff、Lakera Guard）

Layer 2: 架构隔离
- 用户输入和系统 prompt 严格分层，用 delimiter 隔开
- 工具返回的数据标记为 "不可信数据"，不作为指令执行
- 使用 system prompt 中明确声明不接受用户的角色切换指令

Layer 3: 输出过滤
- 检查 Agent 输出是否包含敏感信息（PII、API key 等）
- 检查 Agent 动作是否越权（如用户问查询，Agent 却要执行删除）

Layer 4: 权限最小化
- 每个工具有明确的权限范围（只读 vs 读写）
- Agent 无法访问超出其权限的资源
```

### 5.2 沙箱执行

当 Agent 需要执行代码时：
- **容器隔离**：代码在 Docker 容器中执行，限制 CPU/内存/网络
- **文件系统隔离**：只能访问指定目录，无法读取宿主机文件
- **网络隔离**：默认禁止外网访问，需白名单放行
- **执行时间限制**：超时自动 kill

---

## 6. 状态管理

### 6.1 长对话记忆

```
策略组合：
1. Sliding Window：保留最近 N 轮对话
2. Summary Memory：对早期对话做摘要压缩
3. RAG Memory：将历史对话存入向量数据库，按相关性检索
4. 分层记忆：
   - Working Memory（当前会话上下文）
   - Short-term Memory（近期对话摘要）
   - Long-term Memory（用户偏好、重要事实 → 持久化存储）
```

### 6.2 Checkpoint/Resume

对于长时间运行的 Agent 任务：
- 每完成一个关键步骤，保存 checkpoint（当前状态、已完成的步骤、中间结果）
- 任务中断后可以从最近的 checkpoint 恢复，而非从头开始
- 类似数据库的 WAL（Write-Ahead Log）机制

### 6.3 幂等性设计

```python
# 每个 tool call 带唯一 ID
tool_call = {
    "id": "tc_abc123",  # 幂等键
    "tool": "send_email",
    "params": {"to": "user@example.com", "subject": "..."}
}

# 执行前检查是否已执行
if await idempotency_store.exists(tool_call["id"]):
    return await idempotency_store.get_result(tool_call["id"])
else:
    result = await execute_tool(tool_call)
    await idempotency_store.save(tool_call["id"], result)
    return result
```

---

## 7. 人机协作

### 7.1 Human-in-the-Loop

不是所有操作都应该让 Agent 自动执行。关键设计点：

```
自动执行区（低风险）：
- 信息查询、数据检索、文本生成

需要确认区（中风险）：
- 发送邮件、修改配置、创建资源

需要审批区（高风险）：
- 删除数据、资金操作、权限变更
- 影响范围大的批量操作
```

### 7.2 Confidence Threshold

```python
# Agent 对自己的输出给出置信度
result = agent.execute(task)

if result.confidence >= 0.9:
    # 高置信度，自动执行
    await auto_execute(result)
elif result.confidence >= 0.6:
    # 中置信度，执行但标注需要人工复核
    await execute_with_review_flag(result)
else:
    # 低置信度，暂停等待人工介入
    await escalate_to_human(result)
```

### 7.3 Approval Gates

在 Agent 工作流中插入审批节点：
- Agent 完成分析后暂停，将方案推送给人类审批
- 人类审批通过后 Agent 继续执行
- 超时未审批则自动取消或升级

---

## 8. 评测与迭代

### 8.1 评测体系

```
离线评测：
- 基准测试集：覆盖各类任务场景的 golden dataset
- 自动评分：用 LLM-as-Judge 评估输出质量（GPT-4 评分）
- 工具使用准确率：Agent 是否选了正确的工具、传了正确的参数

在线评测：
- A/B 测试：新旧 prompt/模型分流对比
- 用户反馈：👍👎 按钮收集显式反馈
- 隐式指标：任务完成率、重试率、平均步数
```

### 8.2 回归测试

每次修改 prompt 或更换模型后，必须跑回归测试：
- 维护一个 **test suite**（50-200 个典型 case）
- 每个 case 有预期输出或评判标准
- CI/CD 中集成，prompt 变更自动触发回归
- 关注 **非预期退化**：改善了 A 场景但破坏了 B 场景

### 8.3 反馈闭环

```
用户反馈 → 标注 → 归因分析 → 修复 → 验证
                                    ↓
                            prompt 优化 / 工具修复 / 模型微调
```

---

## 9. 成本优化

### 9.1 模型路由（Model Routing）

核心思想：**不是所有任务都需要最强的模型。**

```
请求进来 → 路由器（分类模型/规则）→ 选择合适模型

简单任务（分类、提取、格式转换）→ GPT-4o-mini / Claude Haiku（便宜 10-20x）
中等任务（摘要、问答）→ GPT-4o / Claude Sonnet
复杂任务（推理、规划、代码生成）→ GPT-4 / Claude Opus / o1

路由策略：
1. 基于规则：关键词/任务类型匹配
2. 基于分类模型：小模型判断任务难度
3. 级联策略：先用小模型试，置信度不够再升级到大模型
```

### 9.2 缓存

```
多级缓存：
1. Exact Match Cache：完全相同的输入直接返回缓存结果
2. Semantic Cache：语义相似的输入复用结果（embedding 相似度 > 0.95）
3. Tool Result Cache：工具调用结果缓存（如搜索结果 TTL 1h）

注意事项：
- 需要考虑缓存一致性（数据更新后缓存失效）
- 个性化场景慎用缓存（不同用户的上下文不同）
```

### 9.3 批处理

- 将多个独立的 LLM 调用合并为一次 batch 请求（OpenAI Batch API 成本减半）
- 非实时场景（如批量数据处理）适合用 batch + 异步回调
- 注意 batch 的延迟换成本的 trade-off

---

## 10. 面试常见问题及回答要点

### Q1: Agent 生产环境中最大的挑战是什么？

**回答要点：**
可靠性和可控性。Demo 中 Agent 只需跑 happy path，但生产中需要处理所有 edge case。三个核心挑战：
1. **不确定性**：LLM 输出不确定，相同输入可能产生不同的 tool call 序列，需要设计容错机制
2. **级联失败**：Agent 的多步推理中，一步出错会导致后续所有步骤偏离，需要 checkpoint + 回滚机制
3. **成本失控**：一个设计不当的 Agent 可能在一个请求中循环调用 LLM 数十次，需要硬性限制

### Q2: 如何设计 Agent 的错误恢复机制？

**回答要点：**
分三层：
1. **工具层**：重试 + fallback，区分暂时性故障和永久性故障
2. **推理层**：检测 LLM 是否陷入循环（连续 N 次调用相同工具+相同参数），如果是则注入提示"你似乎在重复操作，请尝试其他方法"
3. **会话层**：checkpoint 机制，任务失败后可以从中间状态恢复，而非从头开始。关键是设计好 checkpoint 的粒度——太细开销大，太粗恢复效果差

### Q3: 如何防止 Prompt Injection？

**回答要点：**
没有银弹，必须多层防御：
1. **输入层**：专用检测模型（如 Lakera Guard）过滤恶意输入
2. **架构层**：系统 prompt 与用户输入严格隔离，工具返回的数据标记为不可信
3. **输出层**：检查 Agent 的动作是否超出预期范围（如用户问查询问题，Agent 却要删除数据）
4. **权限层**：最小权限原则，Agent 只能访问完成任务所需的最少资源

关键认知：**间接注入比直接注入更危险**——攻击者可以在网页、文档中嵌入恶意指令，Agent 在使用工具获取这些内容时会被注入。

### Q4: Agent 的可观测性应该怎么做？

**回答要点：**
三个层面：
1. **Tracing**：每个请求一个 trace，包含所有 LLM 调用和 tool call 的 span（输入、输出、延迟、token 数）。用 LangSmith 或基于 OpenTelemetry 自建
2. **Metrics**：token 消耗、成本、延迟、成功率、平均步数。接入 Prometheus + Grafana 看板
3. **Logging**：结构化日志，包含 request_id 串联整个链路

实践经验：**trace 是最有价值的**，因为 Agent 的行为非确定性，出了问题需要回放完整的推理过程才能定位原因。

### Q5: 如何控制 Agent 的成本？

**回答要点：**
四个维度：
1. **模型路由**：简单任务用小模型，成本可降 10-20 倍。用级联策略（先小后大）平衡质量和成本
2. **缓存**：exact match + semantic cache，热点查询命中率可达 30-50%
3. **限制**：最大步数、最大 token 数硬限制，防止单个请求成本爆炸
4. **prompt 优化**：精简 system prompt，减少不必要的上下文（很多人忽视这一点，一个冗长的 system prompt 每次调用都会消耗 input token）

### Q6: Human-in-the-loop 怎么设计？什么时候需要人工介入？

**回答要点：**
按风险等级分层：
- **低风险（自动执行）**：只读操作、信息生成
- **中风险（执行后审核）**：可逆的写操作，执行后发通知给人类审核
- **高风险（执行前审批）**：不可逆操作（删除、转账）、影响范围大的批量操作

实现方式：
- 在 Agent 工作流中定义 **approval gate**，高风险操作到达 gate 时暂停
- 推送审批请求（Slack/邮件/内部系统），人类审批后继续
- 设置超时：审批超时后自动取消，避免任务永远挂起

进阶：用 **confidence score** 动态决策——同一个操作，Agent 置信度高时自动执行，置信度低时请求人工介入。

---

## See Also

- [[AI/2-Agent/Fundamentals/Agent 生产实践|Agent 生产实践]] — 同方向互补笔记
- [[AI/2-Agent/Evaluation/Agent-Skills-Security|Agent Skills Security]] — 生产落地中的安全威胁
- [[AI/2-Agent/Agentic-RL/Agent-RL-训练实战指南|Agent RL 训练实战指南]] — 生产级 RL 训练的坑与解法
-  — Agent 知识全图谱
