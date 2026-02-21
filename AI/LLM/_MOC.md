---
title: "大语言模型 LLM"
type: moc
domain: ai/llm
tags:
  - ai/llm
  - type/moc
updated: 2026-02-22
---

# 🧠 大语言模型 LLM — 学习路线图

> 从基础概念到前沿研究的 LLM 全栈知识体系，按学习路径编排。

---

## 第一章 基础概念（Foundations）

> 前置知识：[[AI/Foundations/_MOC|数学 + ML + DL 基础]]

- [[AI/LLM/Architecture/Tokenizer|Tokenizer]] — 分词基础
- [[AI/LLM/Architecture/Tokenizer 深度理解|Tokenizer 深度理解]]
- [[AI/LLM/Inference/采样策略|采样策略]] — Temperature / Top-p / Top-k
- [[AI/LLM/Application/幻觉问题|幻觉问题]]
- [[AI/LLM/幻觉问题与缓解|幻觉问题与缓解]]
- [[AI/LLM/小规模训练手册|小规模训练手册]] — 构建世界级 LLM 的秘密

---

## 第二章 模型架构（Architecture）

> 从 Vanilla Transformer 到 MoE/SSM，理解 LLM 的骨架

### 核心架构

- [[AI/LLM/Architecture/Transformer架构深度解析-2026技术全景|🔥 Transformer 架构深度解析 2026]] ⭐ — 面试终极武器，1617行，从数学第一性原理到 MoE/SSM/2026前沿全覆盖 ★★★★★
- [[AI/LLM/Architecture/架构范式对比|架构范式对比]] — Encoder / Decoder / Encoder-Decoder

### 经典模型系列

| 模型 | 类型 | 说明 |
|------|------|------|
| [[AI/LLM/Architecture/BERT\|BERT]] | Encoder | 双向编码器 |
| [[AI/LLM/Architecture/GPT\|GPT]] | Decoder | 自回归生成 |
| [[AI/LLM/Architecture/T5\|T5]] | Enc-Dec | Encoder-Decoder |
| [[AI/LLM/Architecture/LLaMA\|LLaMA]] | Decoder | Meta 开源系列 |
| [[AI/LLM/Architecture/Qwen\|Qwen]] | Decoder | 阿里通义系列 |
| [[AI/LLM/Architecture/DeepSeek-R1\|DeepSeek-R1]] | Decoder | 推理能力突破 |
| [[AI/Models/Qwen3.5-Plus\|Qwen3.5-Plus]] | MoE | 397B-A17B + Linear Attention |

### Attention 机制

- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]]
- [[AI/LLM/Architecture/FlashAttention|FlashAttention]] — IO-aware 高效注意力
- [[AI/LLM/Architecture/GQA-MQA|GQA / MQA]] — Grouped / Multi-Query Attention
- [[AI/LLM/Architecture/Multi-Head Latent Attention|Multi-Head Latent Attention]]
- [[AI/LLM/Architecture/Transformer 位置编码|位置编码]] — RoPE / ALiBi 等

### 高级架构

- [[AI/LLM/Architecture/MoE 深度解析|MoE 深度解析]] — 混合专家架构
- [[AI/LLM/Architecture/Mamba-SSM|Mamba-SSM]] — 状态空间模型
- [[AI/LLM/Architecture/MiniCPM-SALA|MiniCPM-SALA]] — Sparse + Linear Attention 混合架构
- [[AI/LLM/Architecture/SLA2-Learnable-Router|SLA2]] — 可学习路由器动态选 sparse/linear 分支
- [[AI/LLM/Architecture/长上下文处理|长上下文处理]]
- [[AI/LLM/Architecture/长上下文技术|长上下文技术]]

### 前沿架构研究

- [[AI/LLM/Architecture/Engram-Conditional-Memory-DeepSeek-V4|Engram（DeepSeek V4 架构）]] — 记忆稀疏第二轴 ★★★★★
- [[AI/LLM/Architecture/mHC-Manifold-Constrained-Hyper-Connections-DeepSeek|mHC（DeepSeek V4 架构）]] — 流形约束超连接 ★★★★☆
- [[AI/LLM/Architecture/Manifold-Constrained Hyper-Connections|Manifold-Constrained Hyper-Connections（早期版）]]
- [[AI/LLM/Architecture/ReFINE-Fast-Weight-RL-Next-Sequence-Prediction|ReFINE]] — Fast Weight + GRPO ★★★★☆
- [[AI/LLM/Architecture/Growing-to-Looping-Iterative-Computation-Unification|Growing to Looping]] — 迭代计算统一理论 ★★★★☆
- [[AI/LLM/Architecture/LaViDa-R1-Diffusion-LLM-Reasoning|LaViDa-R1]] — 扩散语言模型推理 ★★★★☆
- [[AI/LLM/Architecture/GLM-5 Agentic Engineering|GLM-5]] — Agentic Engineering
- [[AI/LLM/Architecture/AI Models Collapse 论文|AI Models Collapse]] — 递归训练坍塌

---

## 第三章 预训练（Pretraining）

> 从零开始训练一个 LLM：数据、并行、Scaling Law

- [[AI/LLM/Pretraining/预训练原理|预训练原理]] — 自回归预训练基础
- [[AI/LLM/Pretraining/LLM-预训练与分布式训练-2026-全景|🔥 预训练与分布式训练 2026 全景]] ⭐ — 2183行全覆盖 ★★★★★
- [[AI/LLM/Pretraining/LLM-数据工程-2026-技术全景|🔥 数据工程 2026 全景]] ⭐ — 3793行深度专项 ★★★★★
- [[AI/LLM/Pretraining/Karpathy-nanochat|Karpathy nanochat]] — $72 训练 GPT-2

### 训练基础设施 → [[#附录 A 训练基础设施（Infra）]]

---

## 第四章 微调训练（SFT → RL）

> 预训练后的能力对齐：从 SFT 到 RLHF/DPO/GRPO

### 4.1 监督微调 SFT

- [[AI/LLM/SFT/SFT 原理|SFT 原理]] — 监督微调基础
- [[AI/LLM/SFT/LLM微调实战-2026技术全景|🔥 LLM 微调实战 2026 全景]] ⭐ — 1860行全链路 ★★★★★
- [[AI/LLM/SFT/SFT-TRL实践|SFT-TRL 实践]]
- [[AI/LLM/SFT/SFT-实战指南|SFT 实战指南]]
- [[AI/LLM/SFT/训练数据构建|训练数据构建]]
- [[AI/LLM/SFT/Post-Training Unified View 论文|Post-Training 统一视角]]

### 4.2 参数高效微调 PEFT

- [[AI/LLM/SFT/LoRA|LoRA]] — 低秩适应
- [[AI/LLM/SFT/PEFT 方法对比|PEFT 方法对比]]（530行，正式版）
- [[AI/LLM/SFT/EWC-LoRA-Continual-Learning-Low-Rank|EWC-LoRA]] ⭐ — 持续学习 + 低秩正则，ICLR 2026 ★★★★☆

### 4.3 强化学习 RL → [[AI/LLM/RL/_MOC|RL 详细 MOC]]

- PPO / GRPO / DPO / DAPO / KTO / RLOO 及更多算法
- TRL / verl / Unsloth / OpenRLHF 框架实践

---

## 第五章 推理部署（Inference & Deployment）

> 把训练好的模型高效上线

### 5.1 推理优化总览

- [[AI/LLM/Inference/LLM-推理优化-2026-全景|🔥 推理优化 2026 全景]] — 941行全覆盖
- [[AI/LLM/Inference/推理优化|推理优化综述]]
- [[AI/LLM/Inference/推理服务架构|推理服务架构]]
- [[AI/LLM/Inference/模型部署实践|模型部署实践]]

### 5.2 推理引擎

| 引擎 | 说明 |
|------|------|
| [[AI/LLM/Inference/vLLM\|vLLM]] | PagedAttention 高性能推理 |
| [[AI/LLM/Inference/TensorRT-LLM\|TensorRT-LLM]] | NVIDIA 推理优化 |
| [[AI/LLM/Inference/Ollama\|Ollama]] | 本地部署 |

### 5.3 KV Cache

- [[AI/LLM/Inference/KV Cache|KV Cache]]（830行，正式版）
- [[AI/LLM/Inference/DMS KV Cache压缩|DMS KV Cache 压缩]]
- [[AI/LLM/Inference/Continuous Batching|Continuous Batching]]

### 5.4 解码加速

- [[AI/LLM/Inference/Speculative Decoding|Speculative Decoding]] — 推测解码
- [[AI/LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow]] — Video LLM 推测解码 ★★★★☆
- [[AI/LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention|MAGE]] — Block Diffusion 稀疏注意力 ★★★★☆
- [[AI/LLM/Inference/Sink-Aware-Pruning-Diffusion-LLM|Sink-Aware Pruning]] — Diffusion LLM 剪枝 ★★★★☆

### 5.5 量化

- [[AI/LLM/Inference/量化综述|量化综述]]（正式版） — GPTQ / AWQ / GGUF
- [[AI/LLM/Inference/剪枝与蒸馏|剪枝与蒸馏]]
- [[AI/LLM/Inference/端侧推理量化精度陷阱-跨骁龙芯片精度失真|端侧量化精度陷阱]] ★★★★☆

### 5.6 Test-Time Compute (TTC) — 推理时扩展

- [[AI/LLM/Inference/Test-Time-Compute|TTC 综述]] — CoT / PRM / Best-of-N / Budget Forcing
- [[AI/LLM/Inference/TTC-Test-Time-Compute-Efficiency-2026-综合分析|🔥 TTC 效率 2026 综合分析]] ⭐ ★★★★★
- [[AI/LLM/Inference/Gemini-3-Deep-Think|Gemini 3 Deep Think]] — ARC-AGI-2 84.6%
- [[AI/LLM/Inference/Deep-Thinking-Ratio-DTR|DTR]] — 推翻"CoT 越长越好" ★★★★☆
- [[AI/LLM/Inference/Deep-Thinking-Ratio-DTR-v2-Think-At-N|DTR v2 + Think@N]] ⭐ — "推理深度在开头50 token已决定" ★★★★★
- [[AI/LLM/Inference/Progressive-Thought-Encoding-Cache-Efficient-RL|PTE]] ⭐ — KV cache 满时先学习再 evict，ICLR 2026 ★★★★★
- [[AI/LLM/Inference/Accordion-Thinking-Self-Regulated-Step-Summaries|Accordion-Thinking]] — RL 学会主动压缩 ★★★★☆
- [[AI/LLM/Inference/ConformalThinking-Risk-Control-Test-Time-Compute|ConformalThinking]] ⭐ — 统计风险控制停止策略，ICML 2026 ★★★★★

---

## 第六章 应用层（Application: RAG / Prompt / Code）

> 用 LLM 构建实际产品

### 6.1 Prompt Engineering

- [[AI/LLM/Prompt-Engineering-2026实战全景|🔥 Prompt Engineering 2026 实战全景]] ⭐ — 2784行 ★★★★★
- [[AI/LLM/Application/Prompt-Engineering-基础|Prompt Engineering 基础]]
- [[AI/LLM/Application/Prompt-Engineering-概述|Prompt 概述]]
- [[AI/LLM/Application/Prompt Engineering 高级|Prompt Engineering 高级]]
- [[AI/LLM/Application/高级-Prompt-技巧|高级 Prompt 技巧]]
- [[AI/LLM/Application/Prompt-攻击|Prompt 攻击]]
- [[AI/LLM/Application/Prompt-Tools|Prompt 工具]]
- [[AI/LLM/Application/数据合成|数据合成]]

### 6.2 RAG → 另见 [[AI/RAG/_MOC|RAG 详细 MOC]]

- [[AI/LLM/Application/RAG/RAG 原理与架构|RAG 原理与架构]]
- [[AI/LLM/Application/RAG/Advanced RAG|Advanced RAG]]
- [[AI/LLM/Application/RAG 工程实践|RAG 工程实践]]
- [[AI/LLM/Application/RAG/RAG vs Fine-tuning|RAG vs Fine-tuning]]
- [[AI/LLM/Application/RAG/RAG 评测|RAG 评测]]
- [[AI/LLM/Application/RAG/Reranker|Reranker]]
- [[AI/LLM/Application/RAG/向量数据库选型|向量数据库选型]]
- [[AI/LLM/Application/RAG/文本分块策略|文本分块策略]]
- [[AI/LLM/Application/RAG/文档解析|文档解析]]
- [[AI/LLM/Application/RAG/检索策略|检索策略]]

### 6.3 Embedding & 向量检索

- [[AI/LLM/Application/Embedding/Embedding|Embedding]]
- [[AI/LLM/Application/Embedding/Embedding 选型|Embedding 选型]]
- [[AI/LLM/Application/Embedding 与向量检索|Embedding 与向量检索]]
- [[AI/LLM/Application/Embedding/大模型线上排查 SOP|线上排查 SOP]]

### 6.4 代码生成

- [[AI/LLM/Application/LLM代码生成-2026技术全景|🔥 LLM 代码生成 2026 全景]] ⭐ — 1083行 ★★★★★

### 6.5 合成数据

- [[AI/LLM/Application/Synthetic-Data/合成数据与数据飞轮-2026技术全景|🔥 合成数据与数据飞轮 2026 全景]] ⭐ — 1738行 ★★★★★
- [[AI/LLM/Application/Synthetic-Data/Synthetic Data|合成数据]]
- [[AI/LLM/Application/Synthetic-Data/DataFlow|DataFlow]]

### 6.6 其他应用

- [[AI/LLM/Application/LLMOps|LLMOps]]
- [[AI/LLM/RolePlaying/OpenCharacter-Large-Scale-Synthetic-Persona-Training|OpenCharacter]] — 合成 Persona 角色扮演训练 ★★★

---

## 第七章 前沿进展（Latest Research）

### 效率与压缩

- [[AI/LLM/Efficiency/知识蒸馏与模型压缩-2026技术全景|🔥 知识蒸馏与模型压缩 2026 全景]] ⭐ — 2061行 ★★★★★
- [[AI/LLM/Efficiency/模型蒸馏|模型蒸馏]]

### 评估与趋势

- [[AI/LLM/LLM评估与Benchmark-2026技术全景|🔥 LLM 评估与 Benchmark 2026 全景]] ⭐ — 1854行 ★★★★★
- [[AI/LLM/Evaluation/LLM 评测体系|LLM 评测体系]]
- [[AI/LLM/Evaluation/ICLR-2026-趋势分析|ICLR 2026 趋势分析]] — 5357 篇论文趋势
- [[AI/LLM/Evaluation/PERSIST-LLM-Personality-Stability-Benchmark|PERSIST]] ⭐ — LLM 人格稳定性基准，AAAI 2026 ★★★★★

### 前沿模型 → [[AI/Frontiers/_MOC|前沿详细 MOC]]

---

## 附录 A 训练基础设施（Infra）

- [[AI/LLM/Infra/DeepSpeed|DeepSpeed]]
- [[AI/LLM/Infra/FSDP|FSDP]] — PyTorch 原生分布式
- [[AI/LLM/Infra/Megatron-LM|Megatron-LM]]
- [[AI/LLM/Infra/Ray|Ray]]
- [[AI/LLM/Infra/分布式训练|分布式训练综述]]
- [[AI/LLM/Infra/GPU 显存计算指南|GPU 显存计算指南]]
- [[AI/LLM/Infra/混合精度训练|混合精度训练]]
- [[AI/LLM/Infra/模型并行策略|模型并行策略]]

## 附录 B 工具框架（Frameworks）

### TRL
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL]] — HuggingFace 训练框架

### OpenRLHF
- [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]]

### Slime-RL
- [[AI/LLM/Frameworks/Slime-RL-Framework|Slime-RL]] — THUDM 异步 RL 框架

### Unsloth
- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]] — 低资源微调
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|训练示例]] / [[AI/LLM/Frameworks/Unsloth/CPT|CPT]] / [[AI/LLM/Frameworks/Unsloth/Chat Templates|Templates]] / [[AI/LLM/Frameworks/Unsloth/Checkpoint|Checkpoint]]
- [[AI/LLM/Frameworks/Unsloth/运行 & 保存模型|运行保存]] / [[AI/LLM/Frameworks/Unsloth/量化|量化]] / [[AI/LLM/Frameworks/Unsloth/量化 & 显存预估|显存预估]] / [[AI/LLM/Frameworks/Unsloth/多卡并行|多卡并行]]
- [[AI/LLM/Frameworks/Unsloth/数据合成|数据合成]] / [[AI/LLM/Frameworks/Unsloth/notebook 合集|notebook 合集]]
- [[AI/LLM/Frameworks/Unsloth/Gemma 3 训练|Gemma 3]] / [[AI/LLM/Frameworks/Unsloth/Qwen3 训练|Qwen3]] / [[AI/LLM/Frameworks/Unsloth/gpt-oss 训练|gpt-oss]] / [[AI/LLM/Frameworks/Unsloth/TTS 训练|TTS]]

### verl
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]] — 字节 RL 框架
- [[AI/LLM/Frameworks/verl/算法概述|算法]] / [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] / [[AI/LLM/Frameworks/verl/verl 训练参数|参数]] / [[AI/LLM/Frameworks/verl/配置文件|配置]]
- [[AI/LLM/Frameworks/verl/训练后端|后端]] / [[AI/LLM/Frameworks/verl/Reward Function|Reward]] / [[AI/LLM/Frameworks/verl/Post-Training 数据准备|数据准备]]
- [[AI/LLM/Frameworks/verl/RL with Lora|RL+LoRA]] / [[AI/LLM/Frameworks/verl/Off Policy 异步训练器|Off-Policy]] / [[AI/LLM/Frameworks/verl/多轮 RL 训练交互|多轮交互]] / [[AI/LLM/Frameworks/verl/实现其他 RL 方法|扩展算法]]
- [[AI/LLM/Frameworks/verl/性能调优|性能调优]] / [[AI/LLM/Frameworks/verl/硬件资源预估|硬件预估]] / [[AI/LLM/Frameworks/verl/Sandbox Fusion 沙箱|沙箱]] / [[AI/LLM/Frameworks/verl/grafana 看板|Grafana]]

---

## 导航

- ↑ 上级：[[AI/_MOC]]
- ← 前置：[[AI/Foundations/_MOC]]
- → 相关：[[AI/MLLM/_MOC]] · [[AI/Agent/_MOC]] · [[AI/RAG/_MOC]]
