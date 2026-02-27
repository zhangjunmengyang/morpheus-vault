---
title: "Open-Source AI Cookbook"
type: reference
domain: resources
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - resources
  - type/reference
---
# Open-Source AI Cookbook

> https://huggingface.co/learn/cookbook/index

## 概述

Hugging Face 官方出品的 AI 实践手册，特点是 **全部基于开源工具和模型，附带可运行的 Notebook**。不同于论文或教程的"讲概念"，Cookbook 更像是"给你一个能跑的 recipe，你照着改就行"。

对于工程师来说，这是从"了解原理"到"动手做"之间最好的桥梁之一。

## 核心内容模块

### 1. RAG（检索增强生成）

Cookbook 里 RAG 相关的 recipe 最丰富，覆盖了从简单到生产级的完整链路：

```python
# 基础 RAG 流程示意
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 1. Embedding + 向量化
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# 2. 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.get_relevant_documents(query)

# 3. 生成
generator = pipeline("text-generation", model="meta-llama/Llama-3-8B-Instruct")
context = "\n".join([doc.page_content for doc in relevant_docs])
response = generator(f"Context: {context}\n\nQuestion: {query}")
```

关键 recipe：
- **Simple RAG**：最基础的向量检索 + 生成
- **Advanced RAG**：加入 reranking、query expansion、hybrid search
- **RAG Evaluation**：用 RAGAS 框架评估 RAG 系统质量
- **Multimodal RAG**：图文混合的 RAG

### 2. Fine-tuning 实战

```python
# SFT 微调的典型流程
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
```

覆盖的方向：
- **SFT on custom data**：在自己的数据上微调
- **DPO/RLHF**：偏好对齐训练
- **Vision model fine-tuning**：多模态模型微调（BLIP-2、LLaVA）

### 3. Agent 构建

Cookbook 紧跟 Hugging Face 的 `smolagents` 和 `transformers.agents`：

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(
    tools=[],  # 可以添加自定义工具
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
)

result = agent.run("下载这个 CSV 文件并画出销售趋势图")
```

关键 recipe：
- **Build an Agent with tool use**：从零构建能使用工具的 Agent
- **Multi-agent systems**：多 Agent 协作
- **RAG + Agent**：Agent 调用 RAG 系统作为工具

### 4. 多模态

- **Image captioning with BLIP-2**
- **Visual QA with InstructBLIP**
- **Document understanding with LayoutLM**
- **Audio processing with Whisper**

### 5. 推理优化

```python
# 量化推理
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B-Instruct",
    quantization_config=bnb_config,
)
```

## 使用建议

1. **直接在 Colab / Spaces 运行**：大部分 recipe 提供一键运行按钮
2. **从 Simple RAG 开始**：如果是第一次做 AI 应用，RAG 是最实用的起点
3. **关注版本更新**：Cookbook 持续更新，新模型发布后通常很快有对应 recipe
4. **结合 HF Course**：Cookbook 偏实操，HF NLP Course 偏原理，两者互补

## 与其他学习资源的对比

| 资源 | 侧重 | 适合 |
|------|------|------|
| HF Cookbook | 实操 recipe | 想快速上手做项目 |
| HF NLP Course | NLP 原理 + 实操 | 系统学习 NLP |
| KP 课程 | 从零实现 | 深入理解原理 |
| Fast.ai | 自顶向下 | 快速出活 |
| Stanford CS224N | 学术严谨 | 科研方向 |

## 相关

- [[Projects/HF Agent Course|HF Agent Course]]
- [[Projects/HF-MCP-Course|HF MCP Course]]
- [[Projects/HF LLM + Agent|HF LLM + Agent]]
- [[Projects/KP 大神亲授课]]
- [[AI/3-LLM/Frameworks/TRL/TRL 概述|TRL 概述]]
- [[AI/3-LLM/SFT/LoRA|LoRA]]
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/3-LLM/Application/Embedding/Embedding|Embedding]]
- [[Projects/RAG-System/企业 RAG 系统|企业 RAG 系统]]
- [[AI/3-LLM/Frameworks/Unsloth/notebook 合集|notebook 合集]]
