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
- [[AI/LLM/Architecture/Transformer架构深度解析-2026技术全景|Transformer 架构深度解析 2026]] ⭐ — 面试终极武器，1617行，从数学第一性原理到 MoE/SSM/2026前沿全覆盖，附 15+道难度递进面试题+必背公式表 ★★★★★
- [[AI/LLM/Architecture/BERT|BERT]] — 双向编码器
- [[AI/LLM/Architecture/GPT|GPT]] — 自回归生成
- [[AI/LLM/Architecture/T5|T5]] — Encoder-Decoder
- [[AI/LLM/Architecture/LLaMA|LLaMA]] — Meta 开源系列
- [[AI/LLM/Architecture/Qwen|Qwen]] — 阿里通义系列
- [[AI/Models/Qwen3.5-Plus|Qwen3.5-Plus]] — 397B-A17B MoE + Linear Attention
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — 推理能力突破
- [[AI/LLM/Architecture/Engram-Conditional-Memory-DeepSeek-V4|Engram（DeepSeek V4 架构）]] — 记忆稀疏第二轴：N-gram 嵌入 O(1) 查找 + MoE 计算稀疏互补，100B 参数表卸载 <3% overhead（arXiv:2601.07372，★★★★★）
- [[AI/LLM/Architecture/MoE 深度解析|MoE 深度解析]] — 混合专家架构
- [[AI/LLM/Architecture/Mamba-SSM|Mamba-SSM]] — 状态空间模型
- [[AI/LLM/Architecture/ReFINE-Fast-Weight-RL-Next-Sequence-Prediction|ReFINE]] — Fast Weight + GRPO：NSP 目标解决 NTP 与长程记忆的结构性 mismatch，LaCT-760M RULER +8.5~15%，Princeton ICML（★★★★☆）
- [[AI/LLM/Architecture/Growing-to-Looping-Iterative-Computation-Unification|Growing to Looping]] — Depth Growing 与 Looping 统一理论：两者都是迭代计算的变体，先 grow 再 loop 可推理时免训练获最高 2x 提升，TU Munich + Google（arXiv:2602.16490，★★★★☆）
- [[Architecture/Transformer 架构演进 2026|Transformer 架构演进 2026（面试武器版）]] — 从 Vanilla Transformer → MoE → SSM → 2026 前沿，817行，面试场景驱动（路径待迁移至 AI/LLM/Architecture/）
- [[AI/LLM/Architecture/架构范式对比|架构范式对比]]
- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]]
- [[AI/LLM/Architecture/MiniCPM-SALA|MiniCPM-SALA]] — Sparse + Linear Attention 混合架构：256K 上下文 3.5× 加速，1M token 支持（arXiv 2602.11761）
- [[AI/LLM/Architecture/SLA2-Learnable-Router|SLA2]] — 可学习路由器动态选 sparse/linear 分支：视频 diffusion 97% 稀疏度 + 18.6× attention 加速（arXiv 2602.12675）
- [[AI/LLM/Architecture/FlashAttention|FlashAttention]] — 高效注意力
- [[AI/LLM/Architecture/GQA-MQA|GQA-MQA]] — Grouped/Multi-Query Attention
- [[AI/LLM/Architecture/Multi-Head Latent Attention|Multi-Head Latent Attention]]
- [[AI/LLM/Architecture/Manifold-Constrained Hyper-Connections|Manifold-Constrained Hyper-Connections]] — 早期面试版（305行，2026-02-14）
- [[AI/LLM/Architecture/mHC-Manifold-Constrained-Hyper-Connections-DeepSeek|mHC（DeepSeek V4 架构）]] — 流形约束超连接深度版：多流残差拓扑替代单路残差，训练稳定性++，DeepSeek-AI（arXiv:2512.24880，★★★★☆）
- [[AI/LLM/Architecture/Transformer 位置编码|Transformer 位置编码]] — RoPE 等
- [[AI/LLM/Architecture/Tokenizer|Tokenizer]]
- [[AI/LLM/Architecture/Tokenizer 深度理解|Tokenizer 深度理解]]
- [[AI/LLM/Architecture/长上下文处理|长上下文处理]]
- [[AI/LLM/Architecture/长上下文技术|长上下文技术]]
- [[AI/LLM/Architecture/AI Models Collapse 论文|AI Models Collapse]] — 递归训练坍塌
- [[AI/LLM/Architecture/GLM-5 Agentic Engineering|GLM-5]] — 从 Vibe Coding 到 Agentic Engineering
- [[AI/LLM/Architecture/LaViDa-R1-Diffusion-LLM-Reasoning|LaViDa-R1]] — 扩散语言模型推理：Answer-Forcing + Tree Search + GRPO，Adobe/UCLA/GaTech（★★★★☆）

## Prompt Engineering
- [[AI/LLM/Prompt-Engineering-2026实战全景|Prompt Engineering 2026 实战全景]] ⭐ — 2784行深度全景：CoT·Few-shot·System Prompt设计·自动优化·对抗防护，含大量代码示例（2026-02-20）★★★★★
- [[AI/LLM/Prompt-Engineering/Prompt Engineering|Prompt Engineering]] — 提示工程
- [[AI/LLM/Prompt-Engineering/Prompt engineering 概述|Prompt 概述]]
- [[AI/LLM/Prompt-Engineering/高级 Prompt 技巧|高级 Prompt 技巧]]
- [[AI/LLM/Prompt-Engineering/prompt 攻击|Prompt 攻击]] — 安全对抗
- [[AI/LLM/Prompt-Engineering/Tools|Prompt 工具]]
- [[AI/LLM/Prompt-Engineering/数据合成|数据合成]]

## 监督微调 SFT
- [[AI/LLM/Training/LLM微调实战-2026技术全景|LLM 微调实战 2026 全景]] ⭐ — 面试武器版，1850行，SFT→LoRA→QLoRA→RLHF→DPO→GRPO全链路，含实战代码+常见坑 ★★★★★
- [[AI/LLM/SFT/SFT 原理|SFT 原理]] — 监督微调基础
- [[AI/LLM/SFT/SFT-TRL实践|SFT-TRL实践]]
- [[AI/LLM/SFT/LoRA|LoRA]] — 低秩适应
- [[AI/LLM/SFT/PEFT 方法对比|PEFT 方法对比]]
- [[AI/LLM/SFT/训练数据构建|训练数据构建]]
- [[AI/LLM/SFT/Post-Training Unified View 论文|Post-Training 统一视角]]
- [[AI/LLM/SFT/EWC-LoRA-Continual-Learning-Low-Rank|EWC-LoRA（持续学习+低秩正则）]] ⭐ — 证明独立Fisher估计A/B在bilinear结构下数学不完整；全维Fisher投影到LoRA空间；存储恒定+λ连续可调；ICLR 2026（西安交通大学）★★★★☆

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
- [[AI/LLM/Inference/Deep-Thinking-Ratio-DTR|Deep-Thinking Ratio (DTR)]] — 质量 > 数量：深层 token 占比 r=0.828 准确率，推翻"CoT 越长越好"，UVA+Google（★★★★☆）
- [[AI/LLM/Inference/Deep-Thinking-Ratio-DTR-v2-Think-At-N|DTR v2 + Think@N]] ⭐ — 精读完整版：50-token prefix DTR 比全序列更准；Think@N 在 AIME25 准确率+2%同时成本减半；"推理深度在开头50 token已决定"（★★★★★）
- [[AI/LLM/Inference/KV Cache|KV Cache]] — 推理核心机制
- [[AI/LLM/Inference/KV Cache 优化|KV Cache 优化]]
- [[AI/LLM/Inference/DMS KV Cache压缩|DMS KV Cache 压缩]]
- [[AI/LLM/Inference/Continuous Batching|Continuous Batching]] — 动态批处理
- [[AI/LLM/Inference/Speculative Decoding|Speculative Decoding]] — 推测解码
- [[AI/LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow]] — Video LLM 推测解码：Visual Semantic Internalization，25k visual tokens 下 2.82x 加速，NUDT（★★★★☆）
- [[AI/LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention|MAGE]] — Block Diffusion LLM 稀疏注意力：All-[MASK]第一步预测全局重要KV，128K下后续步6.3x加速，near-lossless（★★★★☆）
- [[AI/LLM/Inference/Sink-Aware-Pruning-Diffusion-LLM|Sink-Aware Pruning]] — Diffusion LLM 注意力 sink 感知剪枝：LLaDA 上 40% 冗余层可裁剪，MMLU/GSM8K 仅降 <0.5%，MBZUAI（arXiv:2602.17664，★★★★☆）
- [[AI/LLM/Inference/Progressive-Thought-Encoding-Cache-Efficient-RL|PTE（Progressive Thought Encoding）]] ⭐ — KV cache 满时先学习再 evict：cross-attention 压缩 evicted token 到 LoRA ΔW，online self-distillation；AIME +33%，内存 -40%；ICLR 2026，微软研究院（arXiv:2602.16839）★★★★★
- [[AI/LLM/Inference/Accordion-Thinking-Self-Regulated-Step-Summaries|Accordion-Thinking]] — 让模型用 RL 学会主动压缩：每步生成 summary 后 fold（丢弃原始 CoT），RL 强制 summary 无损；Gap-Vanishing 现象证明压缩=等价；4× throughput 零精度损失；ICML 2026（arXiv:2602.03249）★★★★☆
- [[AI/LLM/Inference/推理优化|推理优化]] — 综述
- [[AI/LLM/Inference/端侧推理量化精度陷阱-跨骁龙芯片精度失真|端侧量化精度陷阱]] — 同一 INT8 模型跨 5 款骁龙 SoC 精度差 20%；云端 benchmark 完全失真；NPU INT8 算子实现差异根因；PTQ vs QAT 端侧选型建议（馆长工程笔记，2026-02-20）★★★★☆
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

### 代码生成
- [[AI/LLM/Application/LLM代码生成-2026技术全景|LLM 代码生成 2026 全景]] ⭐ — 面试武器库 #17，1083行：预训练数据工程→代码模型架构→核心模型对比→代码 RL 训练→安全合规（2026-02-20）★★★★★

### 其他应用
- [[AI/LLM/Application/LLMOps|LLMOps]]
- [[AI/LLM/Application/Prompt Engineering 高级|Prompt Engineering 高级]]
- [[AI/LLM/Application/幻觉问题|幻觉问题]]

## 模型系列 (Models)
- [[AI/Models/Qwen 系列架构|Qwen 系列架构]] — Qwen 系列模型架构详解

## 预训练 (Pretraining)
- [[AI/LLM/Pretraining/预训练原理|预训练原理]]
- [[AI/LLM/Pretraining/LLM-预训练与分布式训练-2026-全景|LLM 预训练与分布式训练 2026 全景]] — 2183行，覆盖数据工程→分布式训练→MoE→长上下文→面试考点（面试武器版）
- [[AI/LLM/Training/LLM数据工程2026技术全景|LLM 数据工程 2026 技术全景]] — 3778行深度专项：预训练管线·合成数据·SFT构建·质量评估·合规安全，含代码示例 + 12道面试题（互补上方全景版）

## 训练技术 (Training)
- [[AI/LLM/Training/SFT 实战指南|SFT 实战指南]]
- [[AI/LLM/Training/PEFT 方法综述|PEFT 方法综述]]
- [[AI/LLM/Training/数据工程 for LLM|数据工程 for LLM]]
- [[AI/LLM/Training/模型并行策略|模型并行策略]]
- [[AI/LLM/Training/模型蒸馏|模型蒸馏]]
- [[AI/LLM/Training/Karpathy nanochat|Karpathy nanochat]] — $72 训练 GPT-2

## 效率与压缩 (Efficiency & Compression)
- [[AI/LLM/Efficiency/知识蒸馏与模型压缩-2026技术全景|知识蒸馏与模型压缩 2026 全景]] ⭐ — 面试武器库 #18，2061行：KD/量化/剪枝/低秩分解/架构效率/端侧部署全覆盖（2026-02-21）★★★★★

## 评估与趋势 (Evaluation)
- [[AI/LLM/LLM评估与Benchmark-2026技术全景|LLM 评估与 Benchmark 2026 技术全景]] ⭐ — 1854行全景：Benchmark 设计·主流评测集·自动化评估·前沿趋势（2026-02-20）★★★★★
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
