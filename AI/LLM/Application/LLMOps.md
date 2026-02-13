---
title: "LLMOps 生产实践"
date: 2026-02-13
tags:
  - ai/llm/application
  - ai/llm/serving
  - ai/ops
  - type/practice
  - interview/hot
status: active
---

# LLMOps 生产实践

> 从 demo 到生产的全流程——模型服务化、监控、A/B 测试、版本管理、成本优化

## 1. LLMOps 概述

LLMOps 是 MLOps 在 LLM 时代的演进。与传统 MLOps 相比，LLM 带来了全新挑战：

```
传统 MLOps:                      LLMOps:
┌───────────────────┐            ┌───────────────────────────┐
│ 数据 → 训练 → 部署│            │ 数据 → 训练/微调 → 部署   │
│ 模型小 (MB-GB)    │            │ 模型巨大 (GB-TB)          │
│ 推理快 (<10ms)    │            │ 推理慢 (100ms-10s)        │
│ 确定性输出        │            │ 随机性输出 (temperature)   │
│ 标准指标          │            │ 质量难量化 (幻觉、安全)   │
│ 计算成本低        │            │ GPU 成本极高 ($1-10/1K req)│
└───────────────────┘            └───────────────────────────┘
```

### LLMOps 全景图

```
┌─────────────── LLMOps Lifecycle ───────────────────────┐
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 开发     │→ │ 评估     │→ │ 部署     │→ │ 运维   │ │
│  │          │  │          │  │          │  │        │ │
│  │ Prompt   │  │ Evals    │  │ Serving  │  │ Monitor│ │
│  │ RAG      │  │ A/B Test │  │ Gateway  │  │ Alert  │ │
│  │ Fine-tune│  │ Red Team │  │ Scaling  │  │ Cost   │ │
│  │ Agent    │  │ Benchmark│  │ Version  │  │ Feedback│ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│         ↑                                     │         │
│         └──────── 持续改进 ←──────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## 2. 模型服务化

### 推理框架选型

| 框架 | 适用场景 | 特点 |
|------|---------|------|
| [[vLLM]] | 通用部署 | PagedAttention, 易用 |
| [[TensorRT-LLM]] | 极致性能 | NVIDIA 优化, 编译引擎 |
| SGLang | 结构化输出, Agent | RadixAttention, 快 |
| [[Ollama]] | 本地/Edge | 一键部署, 简单 |
| Triton Server | 企业级 | 多模型, 版本管理 |

### API Gateway 层

生产环境不直接暴露推理框架，需要 Gateway 层：

```python
# LLM Gateway 核心功能
class LLMGateway:
    def __init__(self):
        self.router = ModelRouter()      # 模型路由
        self.rate_limiter = RateLimiter() # 限流
        self.cache = SemanticCache()      # 语义缓存
        self.logger = RequestLogger()     # 日志

    async def handle_request(self, request):
        # 1. 认证 + 限流
        self.rate_limiter.check(request.api_key)

        # 2. 语义缓存命中检查
        cached = await self.cache.lookup(request.messages)
        if cached:
            return cached  # 命中缓存，省钱！

        # 3. 模型路由 (A/B 测试、降级、成本优化)
        model_endpoint = self.router.route(request)

        # 4. 转发请求
        response = await model_endpoint.generate(request)

        # 5. 日志 + 缓存写入
        self.logger.log(request, response, latency, tokens)
        await self.cache.store(request.messages, response)

        return response
```

### 常用 Gateway 工具

- **LiteLLM**：统一 100+ LLM 提供商 API，负载均衡，fallback
- **Portkey**：AI Gateway + 可观测性
- **OpenRouter**：模型市场 + 智能路由

## 3. 监控体系

### 核心监控指标

```
┌─── 性能指标 ───────────────────────────────────────┐
│  TTFT (Time to First Token): P50 < 200ms, P99 < 1s │
│  TPOT (Time per Output Token): < 50ms               │
│  Throughput: tokens/s per GPU                        │
│  Request Latency E2E: P99 < 5s                       │
└──────────────────────────────────────────────────────┘

┌─── 质量指标 ───────────────────────────────────────┐
│  Hallucination Rate: 幻觉率 < 5%                    │
│  Toxicity Score: 有害内容分数                        │
│  Groundedness: RAG 回答的接地率                      │
│  User Satisfaction: 👍/👎 比例                      │
└──────────────────────────────────────────────────────┘

┌─── 成本指标 ───────────────────────────────────────┐
│  Cost per Request: $/request                         │
│  Token Usage: input/output tokens per request        │
│  GPU Utilization: > 70% 目标                         │
│  Cache Hit Rate: 语义缓存命中率                      │
└──────────────────────────────────────────────────────┘

┌─── 安全指标 ───────────────────────────────────────┐
│  Prompt Injection Rate: 注入攻击检测率               │
│  PII Leak Rate: 个人信息泄露率                       │
│  Guardrail Trigger Rate: 护栏触发比例               │
└──────────────────────────────────────────────────────┘
```

### 可观测性 Stack

```python
# 使用 OpenTelemetry + LangSmith 监控
from opentelemetry import trace
from langsmith import traceable

tracer = trace.get_tracer("llm-service")

@traceable(run_type="llm")  # LangSmith trace
async def generate(prompt: str, model: str):
    with tracer.start_as_current_span("llm_generate") as span:
        span.set_attribute("model", model)
        span.set_attribute("input_tokens", count_tokens(prompt))

        start = time.time()
        response = await llm.generate(prompt)
        latency = time.time() - start

        span.set_attribute("output_tokens", count_tokens(response))
        span.set_attribute("latency_ms", latency * 1000)
        span.set_attribute("cost_usd", calculate_cost(model, tokens))

        # 异步质量评估
        asyncio.create_task(evaluate_quality(prompt, response))

        return response
```

### 常用监控工具

| 工具 | 类型 | 功能 |
|------|------|------|
| **LangSmith** | Trace + Eval | LangChain 生态，prompt 追踪 |
| **LangFuse** | 开源 Trace | 自托管，成本追踪 |
| **Weights & Biases** | 实验管理 | 训练+推理全流程 |
| **Prometheus + Grafana** | 基础设施 | 延迟、吞吐、GPU 监控 |
| **Arize Phoenix** | LLM 可观测 | 幻觉检测、Embedding 漂移 |

## 4. Evaluation（评估）

### 离线评估

```python
# 使用 RAGAS 评估 RAG 质量
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：回答是否基于检索内容
    answer_relevancy,    # 相关性：回答是否切题
    context_precision,   # 精确度：检索内容是否精确
    context_recall       # 召回率：是否检索到所有相关内容
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy,
             context_precision, context_recall]
)
print(results)
# {'faithfulness': 0.89, 'answer_relevancy': 0.92, ...}
```

### LLM-as-Judge

```python
# 用强模型评估弱模型的输出
JUDGE_PROMPT = """
评估以下回答的质量 (1-5分):
问题: {question}
回答: {answer}

评分标准:
- 准确性: 事实是否正确
- 完整性: 是否覆盖关键点
- 清晰度: 表达是否清楚

输出 JSON: {"score": X, "reason": "..."}
"""

async def llm_judge(question, answer):
    result = await gpt4.generate(
        JUDGE_PROMPT.format(question=question, answer=answer)
    )
    return json.loads(result)
```

## 5. A/B 测试

### 模型 A/B 测试策略

```python
# 基于 hash 的稳定分流
import hashlib

class ABRouter:
    def __init__(self, experiments):
        self.experiments = experiments
        # 例: {"control": ("gpt-4o", 0.5), "treatment": ("gpt-4o-mini", 0.5)}

    def route(self, user_id: str, experiment_id: str) -> str:
        # 相同 user 始终路由到同一分组
        hash_val = hashlib.md5(
            f"{user_id}:{experiment_id}".encode()
        ).hexdigest()
        bucket = int(hash_val[:8], 16) / 0xFFFFFFFF

        cumulative = 0
        for variant, (model, weight) in self.experiments.items():
            cumulative += weight
            if bucket < cumulative:
                return model
        return list(self.experiments.values())[-1][0]
```

### 评估维度

```
A/B 测试关键指标:
├── 质量: LLM-as-Judge 分数、用户评价
├── 性能: 延迟 P50/P99
├── 成本: $/1000 requests
└── 安全: 拒绝率、幻觉率

统计显著性: 至少 1000+ 请求/组，使用 Bootstrap 或 t-test
```

## 6. 模型版本管理

### 版本策略

```yaml
# model-registry.yaml
models:
  chat-v1:
    base: meta-llama/Llama-3.1-70B-Instruct
    adapter: s3://models/chat-lora-v1/
    quantization: fp8
    serving:
      framework: vllm
      tp_size: 4
      max_model_len: 8192

  chat-v2:  # 新版本，灰度中
    base: meta-llama/Llama-3.1-70B-Instruct
    adapter: s3://models/chat-lora-v2/
    quantization: fp8
    serving:
      framework: vllm
      tp_size: 4
      max_model_len: 16384

routing:
  default: chat-v1
  canary:
    model: chat-v2
    percentage: 10  # 10% 流量
```

### 回滚机制

```
蓝绿部署:
  ┌─── Blue (当前版本) ─── 90% 流量 ──→ 用户
  │
  └─── Green (新版本) ─── 10% 流量 ──→ 用户
                                ↓
                        质量指标异常?
                        ├── Yes → 0% 流量, 回滚
                        └── No  → 逐步增加到 100%
```

## 7. 成本优化

### 成本结构

```
LLM 服务成本构成:
├── GPU 计算 (60-80%)
│   ├── 推理 GPU 租赁/折旧
│   ├── 训练/微调 GPU
│   └── GPU 利用率低造成的浪费
├── API 调用 (10-25%)  ← 使用第三方 API 时
│   ├── Input tokens 费用
│   └── Output tokens 费用
├── 存储 (5-10%)
│   ├── 模型权重
│   ├── 向量数据库
│   └── 日志/trace
└── 网络 + 其他 (5%)
```

### 核心优化策略

#### (1) 语义缓存

```python
# 语义缓存：相似问题返回缓存结果
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.95):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = {}  # embedding → response
        self.threshold = threshold

    def lookup(self, query: str):
        query_emb = self.encoder.encode(query)
        for cached_emb, response in self.cache.items():
            similarity = np.dot(query_emb, cached_emb)
            if similarity > self.threshold:
                return response  # Cache hit!
        return None

    def store(self, query: str, response: str):
        emb = tuple(self.encoder.encode(query))
        self.cache[emb] = response

# 效果: 高频问题缓存命中率 30-60%, 成本直降
```

#### (2) 智能路由 (Model Cascade)

```python
# 简单问题用小模型，复杂问题用大模型
class ModelCascade:
    def __init__(self):
        self.fast_model = "gpt-4o-mini"    # $0.15/1M input
        self.strong_model = "gpt-4o"       # $2.50/1M input

    async def generate(self, prompt):
        # Step 1: 用小模型尝试
        response = await call_llm(self.fast_model, prompt)

        # Step 2: 自检是否有把握
        confidence = await self.check_confidence(response)

        if confidence > 0.8:
            return response  # 小模型足够好
        else:
            return await call_llm(self.strong_model, prompt)

# 效果: 70% 请求用小模型处理, 成本降低 ~60%
```

#### (3) Prompt 优化

```
成本与 token 数直接相关:
├── 精简 system prompt (去除冗余说明)
├── 使用结构化输出减少 output tokens
├── Few-shot → Zero-shot (减少 input tokens)
└── 压缩上下文 (LLMLingua 等工具)
```

### 成本对比

```
自建 vs API 成本 (月均 100M tokens):

API (GPT-4o):     ~$250/M tokens × 100 = $25,000/月
API (GPT-4o-mini): ~$15/M tokens × 100 = $1,500/月

自建 (4×H100, LLaMA-70B):
  硬件: $12,000/月 (云租赁)
  运维: $3,000/月 (人力)
  合计: ~$15,000/月, ~$0.15/M tokens

自建 (4×A100, LLaMA-70B INT4):
  硬件: $8,000/月
  运维: $3,000/月
  合计: ~$11,000/月, ~$0.11/M tokens

结论: > 50M tokens/月 → 自建更划算
```

## 8. 生产环境最佳实践

### Guardrails（护栏）

```python
# 输入/输出安全检查
from guardrails import Guard, OnFail

guard = Guard().use_many(
    # 输入检查
    PromptInjectionDetector(on_fail=OnFail.EXCEPTION),
    PIIDetector(on_fail=OnFail.FIX),  # 自动脱敏

    # 输出检查
    ToxicityCheck(threshold=0.8, on_fail=OnFail.REASK),
    HallucinationCheck(on_fail=OnFail.REASK),
)

response = guard(
    llm_api=openai.chat.completions.create,
    prompt=user_input,
    max_reasks=2
)
```

### 生产 Checklist

```
□ 限流 + API Key 管理
□ 输入/输出安全护栏
□ 请求/响应日志 (含 token 计数)
□ 延迟 + 错误率报警
□ 成本预算报警
□ 模型版本管理 + 回滚能力
□ 灾备: API fallback (自建 → 第三方)
□ 负载测试: 峰值 2x 的容量
□ PII 处理: 日志脱敏
□ 合规审计: 可追溯的输入/输出
```

## 9. 与其他主题的关系

- **[[vLLM]]** / **[[TensorRT-LLM]]**：推理框架是 LLMOps serving 层的核心
- **[[RAG 工程实践]]**：RAG 的质量监控和评估是 LLMOps 的重要组成
- **[[推理优化]]**：性能优化直接影响成本和延迟 SLA
- **[[量化综述|量化]]**：量化是降低 serving 成本的核心手段
- **[[Continuous Batching]]**：批处理策略影响吞吐和延迟 trade-off

## 面试常见问题

### Q1: LLMOps 和传统 MLOps 的核心区别是什么？

四大区别：(1) **成本结构**——LLM 推理成本极高（GPU 或 API 费用），需要精细的 token 级成本管控；(2) **质量评估**——生成式输出无法用简单指标评估，需要 LLM-as-Judge、人类评估等多维度方法；(3) **安全挑战**——prompt injection、幻觉、PII 泄露等新风险；(4) **Prompt 管理**——prompt 是核心"代码"，需要版本管理、A/B 测试。

### Q2: 如何监控 LLM 服务的质量？

分层监控：(1) **系统层**——延迟（TTFT/TPOT）、吞吐、错误率、GPU 利用率，用 Prometheus+Grafana；(2) **应用层**——token 用量、成本、缓存命中率，用 LangFuse/LangSmith；(3) **质量层**——定期 LLM-as-Judge 评估、用户反馈（thumbs up/down）、幻觉检测、RAG 的 faithfulness 指标；(4) **安全层**——prompt injection 检测率、PII 泄露率、guardrail 触发率。

### Q3: 如何优化 LLM 服务成本？

四个层次：(1) **基础设施层**——量化（FP8/INT4 降低显存→小卡部署）、Continuous Batching（提高 GPU 利用率）；(2) **缓存层**——语义缓存（30-60% 命中率）、KV Cache 复用（Prefix Caching）；(3) **路由层**——Model Cascade（简单问题用小模型）、自建+API 混合（峰值用 API 兜底）；(4) **Prompt 层**——精简 prompt、结构化输出减少 tokens。

### Q4: 如何做 LLM 的 A/B 测试？

关键点：(1) **稳定分流**——基于 user_id hash 确保同一用户始终在同一组；(2) **多维度评估**——质量（LLM-as-Judge）、延迟、成本、安全四个维度综合评判；(3) **样本量**——至少 1000+ 请求/组确保统计显著性；(4) **灰度发布**——10% → 50% → 100% 逐步放量，每步都检查指标。

### Q5: 自建推理 vs 调用 API，如何决策？

核心公式：**月 token 量 × API 单价 vs GPU 租赁+运维成本**。一般来说，月均 > 50M tokens 时自建更划算。但还需考虑：(1) 数据隐私要求（必须自建）；(2) 定制化需求（微调模型必须自建）；(3) 团队能力（运维 GPU 集群需要专业人员）；(4) 弹性需求（流量波动大时 API 更灵活）。推荐：混合架构，自建处理基线流量，API 处理峰值。
