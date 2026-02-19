---
title: "大语言模型 LLM"
type: moc
domain: ai/llm
tags:
  - ai/llm
  - type/reference
---

# 🧠 大语言模型 LLM

> 从模型架构到训练部署的 LLM 全栈知识

## 模型架构 (Architecture)
- [[AI/LLM/Architecture/BERT|BERT]] — 双向编码器
- [[AI/LLM/Architecture/GPT|GPT]] — 自回归生成
- [[AI/LLM/Architecture/T5|T5]] — Encoder-Decoder
- [[AI/LLM/Architecture/LLaMA|LLaMA]] — Meta 开源系列
- [[AI/LLM/Architecture/Qwen|Qwen]] — 阿里通义系列
- [[AI/Models/Qwen3.5-Plus|Qwen3.5-Plus]] — 397B-A17B MoE + Linear Attention
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — 推理能力突破
- [[AI/LLM/Architecture/DeepSeek Engram|DeepSeek Engram]]
- [[AI/LLM/Architecture/MoE 深度解析|MoE 深度解析]] — 混合专家架构
- [[AI/LLM/Architecture/Mamba-SSM|Mamba-SSM]] — 状态空间模型
- [[AI/LLM/Architecture/架构范式对比|架构范式对比]]
- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]]
- [[AI/LLM/Architecture/FlashAttention|FlashAttention]] — 高效注意力
- [[AI/LLM/Architecture/GQA-MQA|GQA-MQA]] — Grouped/Multi-Query Attention
- [[AI/LLM/Architecture/Multi-Head Latent Attention|Multi-Head Latent Attention]]
- [[AI/LLM/Architecture/Manifold-Constrained Hyper-Connections|Manifold-Constrained Hyper-Connections]]
- [[AI/LLM/Architecture/Transformer 位置编码|Transformer 位置编码]] — RoPE 等
- [[AI/LLM/Architecture/Tokenizer|Tokenizer]]
- [[AI/LLM/Architecture/Tokenizer 深度理解|Tokenizer 深度理解]]
- [[AI/LLM/Architecture/长上下文处理|长上下文处理]]
- [[AI/LLM/Architecture/长上下文技术|长上下文技术]]
- [[AI/LLM/Architecture/AI Models Collapse 论文|AI Models Collapse]] — 递归训练坍塌
- [[AI/LLM/Architecture/GLM-5 Agentic Engineering|GLM-5]] — 从 Vibe Coding 到 Agentic Engineering

## Prompt Engineering
- [[AI/LLM/Prompt-Engineering/Prompt Engineering|Prompt Engineering]] — 提示工程
- [[AI/LLM/Prompt-Engineering/Prompt engineering 概述|Prompt 概述]]
- [[AI/LLM/Prompt-Engineering/高级 Prompt 技巧|高级 Prompt 技巧]]
- [[AI/LLM/Prompt-Engineering/prompt 攻击|Prompt 攻击]] — 安全对抗
- [[AI/LLM/Prompt-Engineering/Tools|Prompt 工具]]
- [[AI/LLM/Prompt-Engineering/数据合成|数据合成]]

## 监督微调 SFT
- [[AI/LLM/SFT/SFT 原理|SFT 原理]] — 监督微调基础
- [[AI/LLM/SFT/SFT-TRL实践|SFT-TRL实践]]
- [[AI/LLM/SFT/LoRA|LoRA]] — 低秩适应
- [[AI/LLM/SFT/PEFT 方法对比|PEFT 方法对比]]
- [[AI/LLM/SFT/训练数据构建|训练数据构建]]
- [[AI/LLM/SFT/Post-Training Unified View 论文|Post-Training 统一视角]]

## ⭐ 强化学习 RL → [[AI/LLM/RL/_MOC|RL 详细 MOC]]
- PPO / GRPO / DPO / DAPO / KTO / RLOO 及更多算法
- TRL / verl / Unsloth / OpenRLHF 框架实践

## 推理部署 (Inference)
- [[AI/LLM/Inference/LLM-推理优化-2026-全景|LLM 推理优化 2026 全景]] — 面试武器版，941行，vLLM/TRT-LLM/KV Cache/Speculative Decoding 全覆盖
- [[AI/LLM/Inference/vLLM|vLLM]] — 高性能推理
- [[AI/LLM/Inference/TensorRT-LLM|TensorRT-LLM]] — NVIDIA 推理优化
- [[AI/LLM/Inference/Ollama|Ollama]] — 本地部署
- [[AI/LLM/Inference/Test-Time-Compute|Test-Time Compute (TTC)]] — 推理时扩展综述：CoT/PRM/Best-of-N/Budget Forcing
- [[AI/LLM/Inference/Gemini-3-Deep-Think|Gemini 3 Deep Think]] — ARC-AGI-2 84.6%, TTC scaling
- [[AI/LLM/Inference/KV Cache|KV Cache]] — 推理核心机制
- [[AI/LLM/Inference/KV Cache 优化|KV Cache 优化]]
- [[AI/LLM/Inference/DMS KV Cache压缩|DMS KV Cache 压缩]]
- [[AI/LLM/Inference/Continuous Batching|Continuous Batching]] — 动态批处理
- [[AI/LLM/Inference/Speculative Decoding|Speculative Decoding]] — 推测解码
- [[AI/LLM/Inference/推理优化|推理优化]] — 综述
- [[AI/LLM/Inference/推理服务架构|推理服务架构]]
- [[AI/LLM/Inference/模型部署实践|模型部署实践]]
- [[AI/LLM/Inference/采样策略|采样策略]]
- [[AI/LLM/Inference/量化技术综述|量化技术综述]]
- [[AI/LLM/Inference/量化综述|量化综述]]
- [[AI/LLM/Inference/剪枝与蒸馏|剪枝与蒸馏]]

## 训练基础设施 (Infra)
- [[AI/LLM/Infra/DeepSpeed|DeepSpeed]] — 分布式训练优化
- [[AI/LLM/Infra/FSDP|FSDP]] — PyTorch 原生分布式
- [[AI/LLM/Infra/Megatron-LM|Megatron-LM]] — 大规模并行
- [[AI/LLM/Infra/Ray|Ray]] — 分布式计算框架
- [[AI/LLM/Infra/分布式训练|分布式训练]] — 综述
- [[AI/LLM/Infra/GPU 显存计算指南|GPU 显存计算指南]]
- [[AI/LLM/Infra/混合精度训练|混合精度训练]]

## 训练框架 (Frameworks)
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL]] — HuggingFace 训练框架
- [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]]
- [[AI/LLM/Frameworks/Slime-RL-Framework|Slime-RL]] — THUDM 异步 RL Post-Training 框架

### Unsloth
- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]] — 低资源微调
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/LLM/Frameworks/Unsloth/CPT|CPT]] — Continued Pretraining
- [[AI/LLM/Frameworks/Unsloth/Chat Templates|Chat Templates]]
- [[AI/LLM/Frameworks/Unsloth/Checkpoint|Checkpoint]]
- [[AI/LLM/Frameworks/Unsloth/运行 & 保存模型|运行 & 保存模型]]
- [[AI/LLM/Frameworks/Unsloth/量化|量化]]
- [[AI/LLM/Frameworks/Unsloth/量化 & 显存预估|量化 & 显存预估]]
- [[AI/LLM/Frameworks/Unsloth/多卡并行|多卡并行]]
- [[AI/LLM/Frameworks/Unsloth/数据合成|数据合成]]
- [[AI/LLM/Frameworks/Unsloth/notebook 合集|notebook 合集]]
- [[AI/LLM/Frameworks/Unsloth/Gemma 3 训练|Gemma 3 训练]]
- [[AI/LLM/Frameworks/Unsloth/Qwen3 训练|Qwen3 训练]]
- [[AI/LLM/Frameworks/Unsloth/gpt-oss 训练|gpt-oss 训练]]
- [[AI/LLM/Frameworks/Unsloth/TTS 训练|TTS 训练]]

### verl
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]] — 字节 RL 框架
- [[AI/LLM/Frameworks/verl/算法概述|算法概述]]
- [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] — 核心架构
- [[AI/LLM/Frameworks/verl/verl 训练参数|训练参数]]
- [[AI/LLM/Frameworks/verl/配置文件|配置文件]]
- [[AI/LLM/Frameworks/verl/训练后端|训练后端]]
- [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]]
- [[AI/LLM/Frameworks/verl/Post-Training 数据准备|Post-Training 数据准备]]
- [[AI/LLM/Frameworks/verl/RL with Lora|RL with LoRA]]
- [[AI/LLM/Frameworks/verl/Off Policy 异步训练器|Off Policy 异步训练器]]
- [[AI/LLM/Frameworks/verl/多轮 RL 训练交互|多轮 RL 训练交互]]
- [[AI/LLM/Frameworks/verl/实现其他 RL 方法|实现其他 RL 方法]]
- [[AI/LLM/Frameworks/verl/性能调优|性能调优]]
- [[AI/LLM/Frameworks/verl/硬件资源预估|硬件资源预估]]
- [[AI/LLM/Frameworks/verl/Sandbox Fusion 沙箱|Sandbox Fusion 沙箱]]
- [[AI/LLM/Frameworks/verl/grafana 看板|Grafana 看板]]

## 应用层 (Application)

### Embedding & 向量检索
- [[AI/LLM/Application/Embedding/Embedding|Embedding]] — 向量化
- [[AI/LLM/Application/Embedding/Embedding 选型|Embedding 选型]]
- [[AI/LLM/Application/Embedding 与向量检索|Embedding 与向量检索]]
- [[AI/LLM/Application/Embedding/大模型线上排查 SOP|线上排查 SOP]]

### RAG
- [[AI/LLM/Application/RAG/RAG 原理与架构|RAG 原理与架构]]
- [[AI/LLM/Application/RAG/Advanced RAG|Advanced RAG]]
- [[AI/LLM/Application/Advanced RAG|Advanced RAG (旧)]]
- [[AI/LLM/Application/RAG 工程实践|RAG 工程实践]]
- [[AI/LLM/Application/RAG/RAG vs Fine-tuning|RAG vs Fine-tuning]]
- [[AI/LLM/Application/RAG/RAG 评测|RAG 评测]]
- [[AI/LLM/Application/RAG/Reranker|Reranker]]
- [[AI/LLM/Application/RAG/向量数据库选型|向量数据库选型]]
- [[AI/LLM/Application/RAG/文本分块策略|文本分块策略]]
- [[AI/LLM/Application/RAG/文档解析|文档解析]]
- [[AI/LLM/Application/RAG/检索策略|检索策略]]

### 合成数据
- [[AI/LLM/Application/Synthetic-Data/Synthetic Data|合成数据]]
- [[AI/LLM/Application/Synthetic-Data/DataFlow|DataFlow]]

### 其他应用
- [[AI/LLM/Application/LLMOps|LLMOps]]
- [[AI/LLM/Application/Prompt Engineering 高级|Prompt Engineering 高级]]
- [[AI/LLM/Application/幻觉问题|幻觉问题]]

## 模型系列 (Models)
- [[AI/Models/Qwen 系列架构|Qwen 系列架构]] — Qwen 系列模型架构详解

## 预训练 (Pretraining)
- [[AI/LLM/Pretraining/预训练原理|预训练原理]]

## 训练技术 (Training)
- [[AI/LLM/Training/SFT 实战指南|SFT 实战指南]]
- [[AI/LLM/Training/PEFT 方法综述|PEFT 方法综述]]
- [[AI/LLM/Training/数据工程 for LLM|数据工程 for LLM]]
- [[AI/LLM/Training/模型并行策略|模型并行策略]]
- [[AI/LLM/Training/模型蒸馏|模型蒸馏]]
- [[AI/LLM/Training/Karpathy nanochat|Karpathy nanochat]] — $72 训练 GPT-2

## 评估与趋势 (Evaluation)
- [[AI/LLM/Evaluation/LLM 评测体系|LLM 评测体系]]
- [[AI/LLM/Evaluation/ICLR-2026-趋势分析|ICLR 2026 趋势分析]] — 5357 篇 accepted papers 趋势

## 其他
- [[AI/LLM/小规模训练手册|小规模训练手册]] — 构建世界级 LLM 的秘密
- [[AI/LLM/幻觉问题与缓解|幻觉问题与缓解]]
- [[AI/LLM/LLM 评测体系|LLM 评测体系 (旧)]]

## 相关 MOC
- ↑ 上级：[[AI/_MOC]]
- ← 前置：[[AI/Foundations/_MOC]]
- → 相关：[[AI/MLLM/_MOC]]、[[AI/Agent/_MOC]]
