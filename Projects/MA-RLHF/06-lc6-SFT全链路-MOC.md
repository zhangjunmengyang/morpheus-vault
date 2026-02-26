---
title: "lc6 · SFT 全链路专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc6_sft"
tags: [moc, ma-rlhf, sft, lora, rag, react, llm-judge, lc6]
---

# lc6 · SFT 全链路专题地图

> **目标**：掌握从数据准备到模型微调到评估部署的完整 SFT 工程链路，同时理解 RAG/ReAct 等外部能力集成。  
> **核心问题**：SFT 到底改了模型的什么？LoRA 为什么用极低 rank 就能工作？什么时候该微调，什么时候该 RAG？

---

## 带着这三个问题学

1. **SFT 的本质是什么？** 和预训练的 Loss 有什么不同？Chat Template 的 special token 为什么重要？
2. **LoRA 的 rank 为什么可以很低（4-16）？** 这说明了预训练模型的什么性质？
3. **CoT、微调、RAG、Agent、AgenticRL——哪些是内部能力，哪些借用外部能力？**

---

## 学习顺序

```
Step 1  SFT 数据格式                ← Chat Template / Messages 格式 / Loss Mask
   ↓
Step 2  SFT 全参微调（PyTorch）     ← 手动训练循环，理解底层
   ↓
Step 3  SFT 框架微调（Transformers） ← Trainer / SFTTrainer，工程最佳实践
   ↓
Step 4  LoRA 低秩微调              ← 原理推导 + 实操对比
   ↓
Step 5  Prompt Engineering         ← 不改模型，改输入
   ↓
Step 6  Embedding & RAG            ← 向量检索增强生成
   ↓
Step 7  ReAct Agent                ← 推理 + 行动交替，工具调用
   ↓
Step 8  LLM-as-Judge & Eval        ← 模型评估模型
```

---

## 笔记清单

### Step 1：SFT 数据格式

**[[AI/LLM/SFT/SFT-手撕实操|SFT 手撕实操]]**（数据部分）

- **Chat Template**：`<|system|>...<|user|>...<|assistant|>...` 格式，special token 标记角色边界
- **Loss Mask 关键**：只对 **assistant** 的 token 计算 Loss，system/user 部分 mask 掉 → 不让模型「学会提问」，只学回答
- **DataCollate**：padding + attention_mask + labels（-100 mask 非目标 token）

课程代码：`Supervised_Finetuning_Dataset.ipynb`（手撕 SFT Dataset + Collate + DataLoader）

---

### Step 2-3：SFT 训练实现

**[[AI/LLM/SFT/SFT-手撕实操|SFT 手撕实操]]**

两种实现路径：

| 方式 | 特点 | 适合 |
|------|------|------|
| PyTorch 手写 | 手动 train loop，完全控制 | 理解底层，面试手撕 |
| Transformers Trainer | 封装好的训练器 + TRL SFTTrainer | 工程实践，快速上线 |

关键训练细节：
- Qwen3 预训练模型 + Alpaca 数据集
- 学习率调度：cosine decay + warmup
- 梯度累积：有效 batch size = micro_batch × accumulation_steps
- 评估指标：Loss 曲线 + 生成质量人工检查

课程代码：`Supervised_Finetuning_PyTorch.ipynb`（手动版） · `Supervised_FineTuning_transformers_Qwen3.ipynb`（🌟 Trainer + SFTTrainer 三版本对比）

**[[AI/LLM/MA-RLHF课程/lc6-SFT全链路-PyTorch手撕实操|lc6-SFT全链路 PyTorch 手撕实操 ✅]]** ★★★★★ — Chat Template 格式化 + Loss Mask 逻辑 + 纯 PyTorch 训练循环 + Transformers SFTTrainer 三路径完整对比（2026-02-26）

---

### Step 4：LoRA 低秩微调

**[[LoRA|LoRA]]**

- **核心思想**：冻结预训练权重 W₀，只训练低秩增量 ΔW = BA，其中 B∈R^{d×r}, A∈R^{r×k}
- **前向**：`y = (W₀ + BA)x = W₀x + BAx`
- **为什么低 rank 就够**：预训练模型的 task-specific adaptation 存在于低维子空间（intrinsic dimensionality 假说）→ rank 4-16 就能捕获 adaptation 所需的信息
- **初始化策略**：A 用 Kaiming 初始化，B 用零初始化 → 训练开始时 ΔW = 0，不破坏预训练知识
- **rank 选择**：简单任务 r=4-8，复杂任务 r=16-64；通常先试 r=16，再根据效果调整
- **梯度**：`∂L/∂A = B^T · ∂L/∂(W₀x+BAx)·x^T`，只更新 A 和 B → 可训练参数 << 全参

课程代码：`LoRA.ipynb`（原理推导 + 梯度推导 + LoRA vs 全参微调对比实验）

**[[AI/LLM/MA-RLHF课程/lc6-LoRA-手撕实操|lc6-LoRA 手撕实操 ✅]]** — 41 cells 完整实现，含 QLoRA/DoRA/LoRA+ 对比、三超参调优实战

深入阅读：[[PEFT 方法对比|PEFT 方法对比]] · [[PEFT 方法对比|PEFT 方法综述]]

---

### Step 5：Prompt Engineering

⏳ 待入库：**Prompt Engineering 实战笔记**

- Zero-shot / Few-shot / Chain-of-Thought 三种范式
- Prompt 结构化设计：角色设定 → 任务描述 → 输出格式约束

课程代码：`Prompt_Enginerring.ipynb`（TODO，课程待完善）

深入阅读：[[Prompt-Engineering-基础|Prompt Engineering 基础]] · [[Prompt Engineering 高级|Prompt Engineering 高级]]

---

### Step 6：Embedding & RAG

**[[RAG 原理与架构|RAG 原理]]**

- **RAG 完整流程**：文档分块 → Embedding 向量化 → 向量数据库存储 → Query Embedding → Top-K 检索 → 注入 Context → LLM 生成
- **Embedding 模型**：将文本映射到稠密向量空间，语义相似的文本向量距离近
- **vs 微调**：RAG 借助外部知识（实时更新、可溯源），微调将知识内化到参数中（更新成本高）

课程代码：`Embedding.ipynb` + `RAG.ipynb`（TODO，课程待完善）

深入阅读：[[RAG vs Fine-tuning|RAG vs Fine-tuning]] · [[Embedding 与向量检索|Embedding 与向量检索]]

---

### Step 7：ReAct Agent

⏳ 待入库：**ReAct Agent 实现笔记**

- **ReAct 模式**：Thought（推理当前状态）→ Action（调用工具/API）→ Observation（获取结果）→ 循环
- **工具调用**：LLM 生成结构化 Action（函数名 + 参数）→ 系统执行 → 结果注入 prompt → 继续推理
- **vs 纯 CoT**：CoT 只推理不执行；ReAct 推理 + 执行 → 能利用外部工具和实时信息

课程代码：`ReAct.ipynb`（TODO，课程待完善）

---

### Step 8：LLM-as-Judge & 评估

⏳ 待入库：**LLM-as-Judge 评估笔记**

- **核心思想**：用强 LLM（如 GPT-4）给弱 LLM 的输出打分
- **Positional Bias 问题**：LLM 倾向给放在前面的答案更高分 → 需要双向评估（A-B 和 B-A）取平均
- **SimpleEval**：标准化评估框架，输出结构化分数

课程代码：`LLM_as_a_Judge.ipynb` + `SimpleEval.ipynb`（TODO，课程待完善）

---

## 内部能力 vs 外部能力对照

| 方法 | 能力来源 | 修改参数？ | 实时性 |
|------|---------|-----------|--------|
| SFT/LoRA 微调 | 内部（参数内化） | ✅ | 需要重训 |
| CoT Prompt | 内部（推理引导） | ❌ | 即时 |
| RAG | 外部（检索增强） | ❌ | 实时更新 |
| ReAct Agent | 外部（工具调用） | ❌ | 实时交互 |
| Agentic RL | 内部+外部 | ✅ | 需要训练 |

---

## 面试高频场景题

**Q：LoRA 的 rank 怎么选？rank 太高或太低会怎样？**  
A：rank 太低（1-2）→ 低秩子空间不足以表达 adaptation，欠拟合；rank 太高（>64）→ 接近全参微调，失去 LoRA 的效率优势，可能过拟合。经验法则：简单任务（格式对齐、风格迁移）r=4-8 足够；复杂任务（知识注入、多领域）r=16-64。先用 r=16 baseline，根据 val loss 调整。

**Q：RAG 和 Fine-tuning 什么时候选哪个？**  
A：RAG 适合：知识频繁更新、需要可溯源、数据量小（<1000 条）、不想改模型；Fine-tuning 适合：知识稳定、需要内化能力（如特定推理模式）、有足够标注数据、追求低延迟（不需要检索步骤）。两者也可结合：先 RAG 召回 → 微调模型学会利用 context。

**Q：SFT 的 Loss Mask 为什么只对 assistant 部分计算？**  
A：SFT 的目标是让模型学会「给定问题如何回答」，不是学会「如何提问」。如果对 user 部分也计算 Loss，模型会学习复读用户输入的分布 → 影响 assistant 回复质量。Loss Mask 确保模型只在 assistant token 上优化 next-token prediction。

**Q：Chat Template 的 special token 为什么重要？**  
A：Special token（如 `<|im_start|>`, `<|im_end|>`）标记角色边界和对话轮次。推理时模型依赖这些 token 判断「当前是谁在说话」和「什么时候该停止生成」。如果 SFT 和推理使用不同的 template → 模型行为异常（输出格式错乱、不停止生成）。
