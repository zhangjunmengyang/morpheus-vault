---
brief: "RAG 评测体系——RAGAS 框架（Faithfulness/Answer Relevancy/Context Precision/Recall）的完整指标体系；端到端评测 vs 组件级评测的方法论；生产 RAG 系统 QA 的落地实践。"
tags: [RAG, 模型评测, RAGAS, Faithfulness, Relevancy, LLM评测, RAG质量评估]
created: 2026-02-14
status: draft
---

# RAG 评测

[[AI/RAG/RAG-2026-技术全景|RAG]] 系统评测是确保生产质量的关键环节。与传统 NLP 任务不同，RAG 涉及检索和生成两个阶段，需要多维度、端到端的评估体系。

## RAG 评测三个维度

### 1. 检索质量 (Retrieval Quality)

评估检索模块能否找到相关文档片段。

**核心指标：**
- **Hit Rate@K：** 前 K 个结果中是否包含正确答案
- **MRR (Mean Reciprocal Rank)：** 第一个相关结果的平均倒数排名
- **NDCG@K：** 考虑排序质量的归一化折扣累计增益

```python
def calculate_hit_rate(retrieved_docs, relevant_docs, k):
    """计算 Hit Rate@K"""
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    return len(retrieved_set & relevant_set) > 0

def calculate_mrr(retrieved_docs, relevant_docs):
    """计算 Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0
```

### 2. 生成质量 (Generation Quality)

评估基于检索内容生成答案的质量。

**核心指标：**
- **Factual Accuracy：** 事实准确性
- **Faithfulness：** 对检索内容的忠实性
- **Completeness：** 答案完整性
- **Fluency：** 语言流畅性

### 3. 端到端质量 (End-to-End Quality)

评估整个 RAG 系统的综合表现。

**核心指标：**
- **Answer Relevancy：** 答案与问题的相关性
- **Context Precision：** 检索上下文的精确性
- **Context Recall：** 检索上下文的召回率
- **Overall Quality Score：** 综合质量分数

## RAGAS 框架详解

RAGAS (RAG Assessment) 是目前最流行的 RAG 评测框架，提供了标准化的评测指标。

### 核心指标

#### 1. Faithfulness (忠实性)

衡量生成答案是否忠实于检索到的上下文，避免幻觉。

$$\text{Faithfulness} = \frac{\text{Number of claims supported by context}}{\text{Total number of claims in answer}}$$

```python
from ragas.metrics import faithfulness
from ragas import evaluate

def evaluate_faithfulness(dataset):
    """
    dataset 需包含 question, answer, contexts 字段
    """
    result = evaluate(
        dataset, 
        metrics=[faithfulness]
    )
    return result['faithfulness']

# 示例数据格式
sample_data = {
    'question': '什么是 Transformer 架构？',
    'answer': 'Transformer 是基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。',
    'contexts': ['Transformer 架构在论文《Attention Is All You Need》中首次提出...']
}
```

#### 2. Answer Relevancy (答案相关性)

评估生成答案与用户问题的匹配度。

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_q, E_{g_i})$$

其中 $E_q$ 是原始问题的嵌入，$E_{g_i}$ 是由答案生成的问题嵌入。

#### 3. Context Precision (上下文精确性)

衡量检索到的上下文中有用信息的比例。

$$\text{Context Precision@K} = \frac{1}{K} \sum_{i=1}^{K} \frac{\text{Relevant chunks before rank i}}{\text{Total chunks before rank i}}$$

#### 4. Context Recall (上下文召回率)

评估检索到的上下文是否包含回答问题所需的所有信息。

$$\text{Context Recall} = \frac{\text{Sentences in answer attributed to context}}{\text{Total sentences in answer}}$$

### RAGAS 完整评测流程

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 构建评测数据集
eval_dataset = Dataset.from_dict({
    'question': questions,
    'answer': generated_answers,
    'contexts': retrieved_contexts,
    'ground_truth': reference_answers  # 可选
})

# 执行评测
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall
    ]
)

print(f"Faithfulness: {result['faithfulness']:.3f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.3f}")
print(f"Context Precision: {result['context_precision']:.3f}")
print(f"Context Recall: {result['context_recall']:.3f}")
```

## 其他评测方法

### ARES (Automated RAG Evaluation System)

**特点：**
- 无需人工标注，自动生成评测数据
- 使用 LLM 作为评判者 (LLM-as-Judge)
- 支持三个维度：Context Relevance, Answer Faithfulness, Answer Relevance

```python
from ares import ARES

# 初始化 ARES
ares_evaluator = ARES(
    model_name="gpt-4",
    metrics=["context_relevance", "answer_faithfulness", "answer_relevance"]
)

# 评测 RAG 系统
scores = ares_evaluator.evaluate(
    questions=questions,
    contexts=contexts, 
    answers=answers
)
```

### TruLens

**特点：**
- 实时监控和评测
- 丰富的可视化界面
- 支持多种 LLM 框架

```python
from trulens_eval import TruChain, Feedback
from trulens_eval.feedback import Groundedness

# 定义反馈函数
grounded = Groundedness(groundedness_provider=openai)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(TruChain.select_context())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# 包装 RAG 应用
tru_rag = TruChain(
    rag_chain,
    app_id='RAG_v1',
    feedbacks=[f_groundedness]
)
```

### DeepEval

**特点：**
- 支持单元测试风格的评测
- 内置多种评测指标
- 与 Pytest 集成

```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, RelevancyMetric

def test_rag_system():
    # 定义评测指标
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)
    relevancy_metric = RelevancyMetric(threshold=0.5)
    
    # 执行评测
    test_case = {
        'input': 'What is machine learning?',
        'actual_output': generated_answer,
        'retrieval_context': retrieved_docs
    }
    
    evaluate([test_case], [faithfulness_metric, relevancy_metric])
```

## LLM-as-Judge 在 RAG 评测中的应用

### 核心思想

使用强大的 LLM（如 GPT-4）作为评判者，评估 RAG 系统输出质量。

### 评判 Prompt 设计

```python
FAITHFULNESS_PROMPT = """
你是一个专业的事实检查员。请评估给定答案是否忠实于提供的上下文。

上下文：
{context}

答案：
{answer}

评估标准：
1. 答案中的每个事实声明是否都能在上下文中找到支持
2. 答案是否包含上下文中不存在的信息
3. 答案是否歪曲了上下文的含义

请给出 1-5 分的评分，并简要说明理由。

评分：
理由：
"""

def llm_judge_faithfulness(context, answer, llm_client):
    prompt = FAITHFULNESS_PROMPT.format(
        context=context, 
        answer=answer
    )
    
    response = llm_client.generate(prompt)
    # 解析评分和理由
    return parse_score_and_reason(response)
```

### 多维度评判框架

```python
class RAGJudge:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.criteria = {
            'faithfulness': self._evaluate_faithfulness,
            'relevancy': self._evaluate_relevancy,
            'completeness': self._evaluate_completeness,
            'clarity': self._evaluate_clarity
        }
    
    def comprehensive_evaluate(self, question, answer, context):
        scores = {}
        for criterion, evaluator in self.criteria.items():
            scores[criterion] = evaluator(question, answer, context)
        
        # 加权平均
        weights = {'faithfulness': 0.4, 'relevancy': 0.3, 
                   'completeness': 0.2, 'clarity': 0.1}
        
        overall_score = sum(scores[k] * weights[k] for k in weights)
        return overall_score, scores
```

## 面试常见问题

### 1. RAGAS 的 Faithfulness 和 Answer Relevancy 有什么区别？

**答案：**
- **Faithfulness（忠实性）：** 评估答案是否忠实于检索到的上下文，主要防止幻觉。计算答案中有多少声明能被上下文支持
- **Answer Relevancy（答案相关性）：** 评估答案是否回答了用户的问题，关注答案与问题的匹配度

**核心区别：**
- Faithfulness: 答案 vs 上下文（垂直维度）
- Answer Relevancy: 答案 vs 问题（水平维度）

**实际案例：**
```
问题：Python 的 GIL 是什么？
上下文：GIL 是全局解释器锁...
答案：Java 的 JVM 有垃圾回收机制...

Faithfulness: 低（答案内容与上下文无关）
Answer Relevancy: 低（答案没回答 Python 问题）
```

### 2. 如何设计 RAG 系统的 A/B 测试评测？

**答案：**
**多层指标体系：**
1. **在线指标：** 用户满意度、点击率、停留时间
2. **离线指标：** RAGAS 指标、专家评分
3. **系统指标：** 响应延迟、吞吐量、成本

**A/B 测试设计：**
```python
class RAGABTest:
    def __init__(self):
        self.control_group = RAGSystemV1()
        self.test_group = RAGSystemV2()
        
    def evaluate_groups(self, test_queries):
        results = {}
        
        for query in test_queries:
            # 随机分组
            if hash(query.user_id) % 2 == 0:
                response = self.control_group.query(query)
                group = 'control'
            else:
                response = self.test_group.query(query)
                group = 'test'
                
            # 记录指标
            self.log_metrics(query, response, group)
        
        return self.analyze_significance()
```

**关键原则：**
- 分层抽样确保用户分布均匀
- 设置合理的实验周期和样本量
- 综合考虑统计显著性和业务意义

### 3. LLM-as-Judge 评测方法有哪些局限性？

**答案：**
**主要局限性：**
1. **评判者偏见：** LLM 可能有特定的评分偏好
2. **一致性问题：** 同样的输入可能得到不同评分
3. **成本高昂：** 大模型调用成本高，难以大规模使用
4. **评判能力边界：** 某些专业领域判断可能不准确

**改进策略：**
```python
class RobustLLMJudge:
    def __init__(self, judges=['gpt-4', 'claude-3', 'gemini-pro']):
        self.judges = judges
        
    def multi_judge_evaluation(self, question, answer, context):
        scores = []
        for judge in self.judges:
            score = self.single_judge_eval(judge, question, answer, context)
            scores.append(score)
        
        # 多评委投票
        final_score = self.aggregate_scores(scores)
        confidence = self.calculate_agreement(scores)
        
        return final_score, confidence
    
    def aggregate_scores(self, scores):
        # 去掉极值，计算中位数
        trimmed_scores = self.remove_outliers(scores)
        return np.median(trimmed_scores)
```

### 4. 如何评测 RAG 系统在多轮对话中的表现？

**答案：**
**多轮对话特有挑战：**
1. **上下文连贯性：** 是否理解对话历史
2. **信息累积：** 能否利用历史检索信息
3. **指代消解：** "它"、"这个"等指代词理解

**评测框架：**
```python
class MultiTurnRAGEvaluator:
    def __init__(self):
        self.conversation_metrics = [
            'context_coherence',
            'information_accumulation', 
            'reference_resolution',
            'response_consistency'
        ]
    
    def evaluate_conversation(self, conversation_history):
        scores = {}
        
        for i, turn in enumerate(conversation_history[1:], 1):
            # 评估当前轮次
            turn_score = self.evaluate_single_turn(
                history=conversation_history[:i],
                current_turn=turn
            )
            
            # 评估整体连贯性
            coherence_score = self.evaluate_coherence(
                conversation_history[:i+1]
            )
            
            scores[f'turn_{i}'] = {
                'turn_quality': turn_score,
                'coherence': coherence_score
            }
        
        return self.aggregate_conversation_scores(scores)
```

**关键评测点：**
- 每轮检索是否考虑了对话历史
- 答案是否与前面的回复保持一致
- 是否正确处理了指代关系

### 5. 如何处理 RAG 评测中的数据隐私问题？

**答案：**
**隐私保护策略：**

1. **数据脱敏：** 
```python
def anonymize_evaluation_data(raw_data):
    """数据脱敏处理"""
    anonymized = []
    for item in raw_data:
        # 替换敏感信息
        clean_question = replace_personal_info(item['question'])
        clean_context = replace_sensitive_content(item['context'])
        
        anonymized.append({
            'question': clean_question,
            'context': clean_context,
            'answer': item['answer']
        })
    return anonymized
```

2. **本地评测：** 使用开源模型在内网环境评测
3. **差分隐私：** 在评测结果中加入噪声
4. **合成数据：** 生成与真实数据分布相似的合成评测集

**实践方案：**
- 敏感行业（金融、医疗）：本地部署评测系统
- 一般企业：数据脱敏 + 云端评测
- 开发测试：合成数据评测

**关键原则：** 在保证评测有效性前提下，最小化数据暴露风险。

---

相关笔记：[[AI/LLM/Application/RAG/RAG 原理与架构|RAG 系统架构设计]] | [[AI/LLM/Application/RAG/向量数据库选型|向量数据库选型]] | [[AI/LLM/Evaluation/LLM 评测体系|LLM 评测体系]]