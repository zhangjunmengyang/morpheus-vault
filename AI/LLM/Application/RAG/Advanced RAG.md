---
brief: "Advanced RAG 进阶技术——GraphRAG/Self-RAG/CRAG/Adaptive-RAG 的核心机制对比；多跳推理和动态知识场景的检索增强方案；RAG 工程师进阶面试的深度参考。"
tags: [RAG, GraphRAG, Self-RAG, CRAG, Adaptive-RAG, Advanced-Retrieval]
created: 2026-02-14
status: draft
---

# Advanced RAG

传统 [[AI/RAG/RAG-2026-技术全景|RAG]] 系统在面对复杂推理、多跳查询和动态知识时存在局限性。Advanced RAG 技术通过引入自适应机制、知识图谱增强、质量评估和路由策略等方法，显著提升了检索增强生成的鲁棒性和准确性。

## Self-RAG：自适应检索与自我反思

Self-RAG 通过在生成过程中动态决定何时检索，并对检索结果进行自我评估，实现了更智能的检索策略。

### 核心机制

**Retrieval Token**：模型学会生成特殊的 token 来控制检索行为
- `[Retrieve]`：触发检索操作
- `[No Retrieval]`：继续基于已有信息生成
- `[Irrelevant]`：检索内容不相关
- `[Relevant]`：检索内容有用

```python
class SelfRAG:
    def __init__(self, llm, retriever, critique_model):
        self.llm = llm
        self.retriever = retriever
        self.critique_model = critique_model
        
        # 特殊控制 token
        self.RETRIEVE_TOKEN = "[Retrieve]"
        self.NO_RETRIEVE_TOKEN = "[No Retrieve]"
        self.RELEVANT_TOKEN = "[Relevant]"
        self.IRRELEVANT_TOKEN = "[Irrelevant]"
    
    def generate_with_self_rag(self, query, max_steps=5):
        context = ""
        current_generation = ""
        
        for step in range(max_steps):
            # 1. 决定是否需要检索
            retrieve_decision = self._should_retrieve(query, current_generation)
            
            if retrieve_decision == self.RETRIEVE_TOKEN:
                # 2. 执行检索
                retrieved_docs = self.retriever.retrieve(
                    query + " " + current_generation, top_k=3
                )
                
                # 3. 评估检索质量
                relevance_scores = []
                for doc in retrieved_docs:
                    relevance = self._assess_relevance(query, doc, current_generation)
                    relevance_scores.append(relevance)
                
                # 4. 筛选相关文档
                relevant_docs = [doc for doc, score in zip(retrieved_docs, relevance_scores) 
                                if score > 0.7]
                
                if relevant_docs:
                    context += "\n".join(relevant_docs)
            
            # 5. 基于当前上下文继续生成
            next_segment = self._generate_segment(query, context, current_generation)
            current_generation += next_segment
            
            # 6. 检查是否完成
            if self._is_complete(current_generation):
                break
        
        return current_generation
    
    def _should_retrieve(self, query, current_text):
        prompt = f"""
        Query: {query}
        Current generation: {current_text}
        
        Should retrieve more information? Respond with {self.RETRIEVE_TOKEN} or {self.NO_RETRIEVE_TOKEN}
        """
        return self.critique_model.generate(prompt).strip()
    
    def _assess_relevance(self, query, document, current_generation):
        prompt = f"""
        Query: {query}
        Current generation: {current_generation}
        Retrieved document: {document}
        
        Rate relevance (0-1): """
        score = float(self.critique_model.generate(prompt).strip())
        return score
```

### 训练策略

Self-RAG 使用 **强化学习** 优化检索决策：

```python
def compute_self_rag_reward(generated_text, ground_truth, retrieved_docs):
    """Self-RAG 奖励函数"""
    # 1. 生成质量奖励
    generation_reward = compute_rouge_score(generated_text, ground_truth)
    
    # 2. 检索效率奖励（避免过度检索）
    efficiency_penalty = -0.1 * len(retrieved_docs)
    
    # 3. 相关性奖励
    relevance_reward = sum([assess_relevance(doc, ground_truth) for doc in retrieved_docs])
    
    total_reward = generation_reward + efficiency_penalty + relevance_reward
    return total_reward
```

## Corrective RAG (CRAG)：检索质量评估与修正

CRAG 通过引入 **检索评估器** 和 **知识修正策略**，动态调整检索策略以提高答案质量。

### 三级评估机制

```python
class CRAG:
    def __init__(self, retriever, web_search, llm, evaluator):
        self.retriever = retriever
        self.web_search = web_search
        self.llm = llm
        self.evaluator = evaluator
    
    def corrective_retrieve_generate(self, query):
        # 1. 初始检索
        retrieved_docs = self.retriever.retrieve(query, top_k=5)
        
        # 2. 评估检索质量
        evaluation_scores = []
        for doc in retrieved_docs:
            score = self.evaluator.evaluate_relevance(query, doc)
            evaluation_scores.append(score)
        
        avg_score = np.mean(evaluation_scores)
        
        if avg_score >= 0.8:
            # 高质量：直接使用
            strategy = "direct_use"
            final_docs = retrieved_docs
            
        elif avg_score >= 0.4:
            # 中等质量：知识精炼
            strategy = "refine_knowledge"
            final_docs = self._refine_knowledge(query, retrieved_docs, evaluation_scores)
            
        else:
            # 低质量：网络搜索补充
            strategy = "web_search"
            web_results = self.web_search.search(query, num_results=3)
            final_docs = retrieved_docs + web_results
        
        # 3. 生成答案
        context = "\n".join(final_docs)
        answer = self.llm.generate(f"Context: {context}\nQuestion: {query}")
        
        return {
            "answer": answer,
            "strategy": strategy,
            "confidence": avg_score,
            "sources": final_docs
        }
    
    def _refine_knowledge(self, query, docs, scores):
        """知识精炼：移除不相关部分，突出相关信息"""
        refined_docs = []
        
        for doc, score in zip(docs, scores):
            if score > 0.5:  # 保留中等以上相关性的文档
                # 使用 LLM 提取与查询最相关的部分
                refined_content = self._extract_relevant_parts(query, doc)
                refined_docs.append(refined_content)
        
        return refined_docs
    
    def _extract_relevant_parts(self, query, document):
        prompt = f"""
        Extract the most relevant parts from the following document for answering the query.
        Keep only information directly related to the query.
        
        Query: {query}
        Document: {document}
        
        Relevant excerpts:
        """
        return self.llm.generate(prompt)
```

## GraphRAG：知识图谱增强检索

GraphRAG 利用知识图谱的结构化信息和关系推理能力，支持多跳查询和复杂推理任务。

### 图结构检索

```python
class GraphRAG:
    def __init__(self, knowledge_graph, entity_linker, path_finder, llm):
        self.kg = knowledge_graph  # Neo4j / NetworkX 图
        self.entity_linker = entity_linker
        self.path_finder = path_finder
        self.llm = llm
    
    def graph_enhanced_retrieve(self, query, max_hops=2):
        # 1. 实体识别与链接
        entities = self.entity_linker.extract_entities(query)
        
        # 2. 子图检索
        subgraph = self._extract_subgraph(entities, max_hops)
        
        # 3. 路径推理
        reasoning_paths = self.path_finder.find_reasoning_paths(
            subgraph, entities, max_length=3
        )
        
        # 4. 构建图上下文
        graph_context = self._serialize_graph_info(subgraph, reasoning_paths)
        
        # 5. 传统文档检索（补充）
        doc_context = self._traditional_retrieve(query)
        
        # 6. 融合生成
        combined_context = f"""
        Knowledge Graph Information:
        {graph_context}
        
        Document Information:
        {doc_context}
        """
        
        answer = self.llm.generate(f"{combined_context}\nQuestion: {query}")
        return answer
    
    def _extract_subgraph(self, entities, max_hops):
        """提取以给定实体为中心的子图"""
        subgraph_nodes = set(entities)
        current_nodes = set(entities)
        
        for hop in range(max_hops):
            next_nodes = set()
            for node in current_nodes:
                # 获取邻居节点
                neighbors = self.kg.neighbors(node)
                next_nodes.update(neighbors)
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            if not next_nodes:  # 没有更多邻居
                break
        
        return self.kg.subgraph(subgraph_nodes)
    
    def _serialize_graph_info(self, subgraph, reasoning_paths):
        """将图信息序列化为文本"""
        context_parts = []
        
        # 实体信息
        for node in subgraph.nodes():
            node_info = f"Entity: {node}"
            if 'description' in subgraph.nodes[node]:
                node_info += f" - {subgraph.nodes[node]['description']}"
            context_parts.append(node_info)
        
        # 关系信息
        for edge in subgraph.edges(data=True):
            relation_info = f"Relationship: {edge[0]} -> {edge[2].get('relation', 'related_to')} -> {edge[1]}"
            context_parts.append(relation_info)
        
        # 推理路径
        for path in reasoning_paths[:3]:  # 限制路径数量
            path_str = " -> ".join(path)
            context_parts.append(f"Reasoning Path: {path_str}")
        
        return "\n".join(context_parts)
```

### Cypher 查询生成

```python
def generate_cypher_query(natural_query, schema):
    """将自然语言转换为 Cypher 查询"""
    prompt = f"""
    Database Schema:
    {schema}
    
    Natural Language Query: {natural_query}
    
    Generate Cypher query:
    """
    cypher = llm.generate(prompt)
    return cypher.strip()

# 示例使用
query = "找到与张三相关的所有公司"
cypher = generate_cypher_query(query, kg_schema)
# 生成：MATCH (p:Person {{name: "张三"}})-[:WORKS_AT]->(c:Company) RETURN c.name
```

## Adaptive RAG：路由策略

Adaptive RAG 根据查询类型和复杂度，动态选择最适合的检索策略。

```python
class AdaptiveRAG:
    def __init__(self, strategies):
        self.strategies = {
            'simple_rag': SimpleRAG(),
            'self_rag': SelfRAG(),
            'graph_rag': GraphRAG(),
            'crag': CRAG(),
            'web_search': WebSearchRAG()
        }
        self.query_classifier = QueryClassifier()
    
    def adaptive_generate(self, query):
        # 1. 查询分类
        query_type = self.query_classifier.classify(query)
        
        # 2. 策略路由
        if query_type['complexity'] == 'simple' and query_type['domain'] == 'general':
            strategy = 'simple_rag'
        elif query_type['requires_reasoning']:
            strategy = 'graph_rag'
        elif query_type['requires_fresh_info']:
            strategy = 'web_search'
        elif query_type['uncertainty_high']:
            strategy = 'crag'
        else:
            strategy = 'self_rag'
        
        # 3. 执行相应策略
        result = self.strategies[strategy].generate(query)
        
        return {
            'answer': result,
            'strategy_used': strategy,
            'query_analysis': query_type
        }

class QueryClassifier:
    def classify(self, query):
        features = {
            'length': len(query.split()),
            'has_time_constraint': any(word in query.lower() for word in ['recent', 'latest', 'current']),
            'has_comparison': any(word in query.lower() for word in ['compare', 'versus', 'difference']),
            'has_reasoning_words': any(word in query.lower() for word in ['why', 'how', 'cause', 'reason']),
            'has_entities': self._extract_entities(query),
            'domain': self._classify_domain(query)
        }
        
        complexity = 'complex' if features['length'] > 15 or features['has_reasoning_words'] else 'simple'
        
        return {
            'complexity': complexity,
            'requires_reasoning': features['has_reasoning_words'],
            'requires_fresh_info': features['has_time_constraint'],
            'uncertainty_high': features['has_comparison'],
            'domain': features['domain']
        }
```

## 方案对比与适用场景

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| Self-RAG | 智能检索决策，效率高 | 训练复杂，需要标注数据 | 通用问答，动态知识需求 |
| CRAG | 自动质量评估，鲁棒性强 | 计算开销大 | 知识质量不稳定的场景 |
| GraphRAG | 支持复杂推理，关系丰富 | 图构建成本高 | 多跳推理，实体关系查询 |
| Adaptive RAG | 策略灵活，覆盖面广 | 路由逻辑复杂 | 多样化查询场景 |

## 评测指标与基准

```python
def evaluate_advanced_rag(system, test_dataset):
    metrics = {
        'accuracy': [],
        'relevance': [],
        'completeness': [],
        'efficiency': [],
        'coherence': []
    }
    
    for sample in test_dataset:
        query = sample['query']
        ground_truth = sample['answer']
        
        start_time = time.time()
        result = system.generate(query)
        end_time = time.time()
        
        # 准确性评估
        accuracy = compute_exact_match(result['answer'], ground_truth)
        metrics['accuracy'].append(accuracy)
        
        # 相关性评估
        relevance = compute_semantic_similarity(result['answer'], ground_truth)
        metrics['relevance'].append(relevance)
        
        # 完整性评估（是否回答了所有子问题）
        completeness = assess_completeness(query, result['answer'])
        metrics['completeness'].append(completeness)
        
        # 效率评估
        efficiency = 1.0 / (end_time - start_time)  # 简化指标
        metrics['efficiency'].append(efficiency)
        
        # 连贯性评估
        coherence = assess_coherence(result['answer'])
        metrics['coherence'].append(coherence)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

## 面试常见问题

**Q1：Self-RAG 中的自我反思机制具体是如何实现的？**

A：Self-RAG 通过训练模型生成特殊的控制 token 实现自我反思：1）在训练时，为每个生成步骤添加检索决策标签（[Retrieve]/[No Retrieve]）和相关性评估标签（[Relevant]/[Irrelevant]）；2）使用强化学习优化，奖励函数结合生成质量、检索效率和相关性；3）推理时，模型会在生成过程中主动判断是否需要检索更多信息，并评估检索内容的质量。这种机制让模型具备了动态调整检索策略的能力。

**Q2：GraphRAG 相比传统 RAG 在处理多跳推理时有什么优势？**

A：GraphRAG 的优势主要体现在：1）**结构化推理**：通过图的路径遍历自然支持多跳推理，而传统 RAG 需要多次检索；2）**关系建模**：显式建模实体间关系，避免了向量检索中关系信息的丢失；3）**全局视角**：子图提供了实体的完整关系网络，而不是孤立的文档片段；4）**可解释性**：推理路径清晰可追溯。但 GraphRAG 需要高质量的知识图谱，构建和维护成本较高。

**Q3：CRAG 的三级评估机制是如何设计的？为什么这样分级？**

A：CRAG 的三级评估基于检索质量的置信度：1）**高置信度（>0.8）**：直接使用，因为检索结果高度相关；2）**中置信度（0.4-0.8）**：知识精炼，通过 LLM 过滤无关信息，保留相关部分；3）**低置信度（<0.4）**：触发网络搜索，补充外部知识源。这种设计平衡了效率和准确性：避免了对高质量检索的过度处理，同时为低质量检索提供了补救机制。

**Q4：Adaptive RAG 的查询分类策略有哪些关键特征？**

A：Adaptive RAG 的查询分类通常包含：1）**复杂度判断**：基于查询长度、语法结构、推理词汇等；2）**时效性需求**：是否包含"最新"、"当前"等时间约束词；3）**推理需求**：是否包含"为什么"、"如何"等因果推理词；4）**领域识别**：通过关键词或实体识别判断专业领域；5）**不确定性评估**：查询的歧义性和多义性。基于这些特征，系统可以路由到最适合的策略：简单问答用标准 RAG，复杂推理用 GraphRAG，时效性问题用网络搜索等。

**Q5：在生产环境中部署 Advanced RAG 需要考虑哪些工程问题？**

A：主要工程问题包括：1）**延迟控制**：复杂策略增加计算开销，需要合理设置超时和并发；2）**缓存策略**：缓存图查询结果、评估分数等中间结果；3）**降级方案**：当高级策略失败时，回退到简单 RAG；4）**成本管理**：平衡策略复杂度和计算资源；5）**监控体系**：跟踪各策略的成功率、延迟、准确性等指标；6）**A/B 测试**：对比不同策略在真实场景中的表现。建议采用微服务架构，将不同策略模块化，支持独立扩缩容和版本管理。

相关链接：[[AI/RAG/RAG-2026-技术全景|RAG]], [[AI/LLM/Application/RAG/Reranker|Reranker]], [[AI/LLM/Application/RAG/文本分块策略|文本分块策略]], [[AI/LLM/LLM 评测体系|LLM 评测体系]], [[AI/LLM/Application/RAG/Advanced RAG|Advanced RAG — 知识图谱部分]], [[AI/RAG/向量数据库选型|向量数据库]]