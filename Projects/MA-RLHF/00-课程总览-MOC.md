---
title: "MA-RLHF 课程总览 · 地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [moc, ma-rlhf, llm-engineering, course, index]
---

# MA-RLHF 课程总览 · 地图

> **来源**：[MA-RLHF](https://github.com/dhcode-cpp/MA-RLHF)（dhcode-cpp）  
> **定位**：LLM 算法工程师从零到可面试的完整工程课程。10 个专题，从 Transformer 手写到 RLHF 全链路，从单卡到分布式，从推理框架到 R1 复现。  
> **使用方式**：按专题顺序学习，每个专题有独立 MOC。跟着「推荐学习路径」走，不要跳跃。

---

## 课程结构全景

```
MA-RLHF 课程
├── lc1  基础组件          ← Tokenizer / Embedding / 位置编码
├── lc2  Transformer       ← 完整 Encoder-Decoder 手写
├── lc3  GPT 系列          ← Decoder-only / KV Cache / BPE
├── lc4  Llama 系列        ← RoPE / GQA / SwiGLU / RMSNorm
├── lc5  DeepSeek V3       ← MoE / MLA / MTP / YaRN / mHC
├── lc6  SFT 全链路        ← 数据 / LoRA / 完整训练 / RAG / ReAct
├── lc7  RL 基础           ← REINFORCE / DQN / 策略梯度 / GAE
├── lc8  RL×LLM            ← RLHF-PPO / DPO / KTO / GRPO / PRM / O1
├── lc9  分布式 RL 训练     ← Ray 三角架构 / 异步 GRPO / verl 实战
└── lc10 推理系统          ← Continue Batching / PageAttention / vLLM V0/V1 / PD分离
    └── xtrain 专题        ← 从零手写 DP/TP/PP/CP/EP 五大并行
```

---

## 推荐学习路径

### 路径 A：架构面试速通（1.5 天）
```
lc1 → lc2 → lc3 → lc4 → lc5
重点：Transformer 手写 → GPT/Llama 演进 → DeepSeek MLA/MoE
```

### 路径 B：RL 对齐面试速通（1.5 天）
```
lc7（RL基础）→ lc6（SFT）→ lc8（RLHF-PPO / DPO / GRPO）→ lc9（分布式训练）
重点：RLHF 四模型架构 → GRPO vs PPO → R1/verl 实战
```

### 路径 C：推理系统面试速通（1 天）
```
lc10（Continue Batching → PageKVCache → PageAttention → vLLM V0/V1 → Chunked Prefill → PD 分离）
重点：vLLM 完整架构 + Speculative Decoding
```

### 路径 D：分布式训练硬核（2 天）
```
xtrain-lc1（通信原语）→ lc2（DP/DDP）→ lc3（ZeRO）→ lc4（TP）→ lc5（PP/DualPipe）→ lc6（CP/Ring）→ lc7（MoE/EP）
重点：从零手写，不依赖 DeepSpeed/Megatron
```

### 路径 E：全栈 LLM 工程师（5 天）
```
路径A → 路径B → 路径C → 路径D → 简历项目包装
```

---

## 专题地图（点进去学）

| 专题 | 核心内容 | 状态 | MOC 入口 |
|------|---------|------|---------|
| **lc1** 基础组件 | BPE Tokenizer / Embedding / 位置编码全家族 | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc1-基础组件-MOC]] |
| **lc2** Transformer | Encoder-Decoder 完整实现 / 数据集 / 训练推理全流程 | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc2-Transformer-MOC]] |
| **lc3** GPT 系列 | GELU / PreNorm / BPE / KV Cache / Perplexity / ICL | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc3-GPT系列-MOC]] |
| **lc4** Llama 系列 | RoPE / NTK-RoPE / GQA / RMSNorm / SwiGLU / Benchmark | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc4-Llama系列-MOC]] |
| **lc5** DeepSeek V3 | MoE / MLA / MTP / YaRN / TopK梯度 / mHC / Load Balance | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]] |
| **lc6** SFT 全链路 | 数据处理 / LoRA / SFT完整训练 / RAG / ReAct / LLM-Judge | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc6-SFT全链路-MOC]] |
| **lc7** RL 基础 | MC/TD/Q-Learning/DQN/PolicyGradient/GAE | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc7-RL基础-MOC]] |
| **lc8** RL×LLM | RLHF-PPO / Bradley-Terry / DPO / KTO / GRPO / PRM / MCTS-O1 | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]] |
| **lc9** 分布式 RL 训练 | Ray三角架构 / 异步GRPO / verl实战 / R1复现 | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc9-分布式RL训练-MOC]] |
| **lc10** 推理系统 | Continue Batching / PageKV / PageAttention / vLLM V0→V1 / Chunked Prefill / PD分离 / SD | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/lc10-推理系统-MOC]] |
| **xtrain** 分布式手写 | DP/ZeRO/TP/PP(DualPipe)/CP(RingAttn)/EP(MoE) 从零手写 | ✅ MOC完成 | [[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]] |

---

## 与 Vault 研究笔记的关系

课程笔记 = **工程实现视角**（怎么写代码，怎么调通，面试怎么答）  
Vault 研究笔记 = **论文原理视角**（理论推导，实验证据，前沿进展）

两者互补：
- 课程笔记内嵌「→ 深入阅读」链接，指向对应 Vault 论文笔记
- 例：lc8 的 GRPO 手撕 → 深入阅读 [[GRPO 深度理解]] + [[GRPO-Improvement-Panorama-2026]]

---

## 入库进度跟踪

> 更新时间：2026-02-25

- **已完成（原始批次）**：25 篇（架构/推理/分布式/RL对齐/多模态基础）
- **Batch A（lc10推理系统）**：进行中，6/8 篇
- **待入库**：约 55 篇（按心跳逐批完成）
- **完整清单**：见 `memory/ma-rlhf-ingest-plan.md`
