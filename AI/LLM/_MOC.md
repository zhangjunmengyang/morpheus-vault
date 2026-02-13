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
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — 推理能力突破
- [[AI/LLM/Architecture/AI Models Collapse 论文|AI Models Collapse]] — 递归训练坍塌

## Prompt Engineering
- [[AI/LLM/Prompt-Engineering/Prompt Engineering|Prompt Engineering]] — 提示工程
- [[AI/LLM/Prompt-Engineering/Prompt engineering 概述|Prompt 概述]]
- [[AI/LLM/Prompt-Engineering/prompt 攻击|Prompt 攻击]] — 安全对抗
- [[AI/LLM/Prompt-Engineering/Tools|Prompt 工具]]
- [[AI/LLM/Prompt-Engineering/数据合成|数据合成]]

## 监督微调 SFT
- [[AI/LLM/SFT/SFT 原理|SFT 原理]] — 监督微调基础
- [[AI/LLM/SFT/SFT-TRL实践|SFT-TRL实践]]
- [[AI/LLM/SFT/LoRA|LoRA]] — 低秩适应
- [[AI/LLM/SFT/Post-Training Unified View 论文|Post-Training 统一视角]]

## ⭐ 强化学习 RL → [[AI/LLM/RL/_MOC|RL 详细 MOC]]
- PPO / GRPO / DPO / DAPO / KTO / RLOO 及更多算法
- TRL / verl / Unsloth / OpenRLHF 框架实践

## 推理部署 (Inference)
- [[AI/LLM/Inference/vLLM|vLLM]] — 高性能推理
- [[AI/LLM/Inference/Ollama|Ollama]] — 本地部署

## 训练基础设施 (Infra)
- [[AI/LLM/Infra/DeepSpeed|DeepSpeed]] — 分布式训练优化
- [[AI/LLM/Infra/FSDP|FSDP]] — PyTorch 原生分布式
- [[AI/LLM/Infra/Megatron-LM|Megatron-LM]] — 大规模并行
- [[AI/LLM/Infra/Ray|Ray]] — 分布式计算框架
- [[AI/LLM/Infra/分布式训练|分布式训练]] — 综述

## 训练框架 (Frameworks)
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL]] — HuggingFace 训练框架
- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth]] — 低资源微调
- [[AI/LLM/Frameworks/verl/verl 概述|verl]] — 字节 RL 框架
- [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]]

## 应用层 (Application)
- [[AI/LLM/Application/Embedding/Embedding|Embedding]] — 向量化
- [[AI/LLM/Application/Embedding/Embedding 选型|Embedding 选型]]
- [[AI/LLM/Application/Synthetic-Data/Synthetic Data|合成数据]]
- [[AI/LLM/Application/Synthetic-Data/DataFlow|DataFlow]]
- [[AI/LLM/Application/Embedding/大模型线上排查 SOP|线上排查 SOP]]

## 其他
- [[AI/LLM/小规模训练手册|小规模训练手册]] — 构建世界级 LLM 的秘密

## 相关 MOC
- ↑ 上级：[[AI/_MOC]]
- ← 前置：[[AI/Foundations/_MOC]]
- → 相关：[[AI/MLLM/_MOC]]、[[AI/Agent/_MOC]]
