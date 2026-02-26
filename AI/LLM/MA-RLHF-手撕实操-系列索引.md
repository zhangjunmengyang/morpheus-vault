---
title: "MA-RLHF 手撕实操系列索引"
brief: "基于 MA-RLHF 开源项目（github.com/dhcode-cpp/MA-RLHF）的完整代码实操笔记系列：架构/推理/分布式训练/RL对齐/xtrain从零手写/RL Notebook实现/多模态 52 篇（含 lc10 推理系统 6 篇、xtrain lc1-lc7 从零手写 7 篇、RL Notebook 8 篇、lc8 Architecture Batch D 6 篇），总计约 21,000 行代码精读，LLM 算法工程师面试代码题核心武器库。"
date: 2026-02-25
type: index
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, index, ma-rlhf, interview, pytorch, llm-engineering]
related:
  - "[[AI/LLM/目录|LLM MOC]]"
  - "[[AI/LLM/RL/目录|RL MOC]]"
---

# MA-RLHF 手撕实操系列索引

> **来源**：[MA-RLHF](https://github.com/dhcode-cpp/MA-RLHF)（dhcode-cpp）
> **入库日期**：2026-02-25
> **总规模**：52 篇，约 21,000 行代码实操（2026-02-26 Architecture Batch D 持续扩容）
> **定位**：LLM 算法工程师 / AI 研究工程师面试代码实现题的核心武器库

---

## 为什么要手撕这些？

面试中**会要求手写代码**的方向：
- Transformer Self-Attention 的完整实现
- PPO / GRPO 的 loss 计算
- FlashAttention 的 Online Softmax 推导
- ZeRO 的显存分片逻辑
- Ring-AllReduce 的通信原语

这个系列提供了**原理 → 公式 → PyTorch 代码**的完整路径，每篇都来自真实可运行的开源实现。

---

## 系列全景

### 🏗️ 架构基础（12 篇）

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/Architecture/基础数学组件手撕|基础数学组件手撕]] | MHA/MQA/GQA/LayerNorm/RMSNorm/SwiGLU/LoRA | 各 Attention 变体参数量对比 |
| [[AI/LLM/Architecture/Transformer-手撕实操|Transformer-手撕实操]] | Encoder-Decoder 完整实现 | 训练并行 vs 推理串行 |
| [[AI/LLM/Architecture/GPT2-手撕实操|GPT2-手撕实操]] | Decoder-only + GPT-1→2→3 演进 | GPT 系列关键创新递进 |
| [[AI/LLM/Architecture/Llama-手撕实操|Llama-手撕实操]] | RoPE/RMSNorm/SwiGLU/GQA/KV Cache | MHA→GQA 效率提升路径 |
| [[AI/LLM/Architecture/DeepSeek-V3-手撕实操|DeepSeek-V3-手撕实操]] | MLA（KV cache降16x）+ MoE + mHC | 当前最高效 MoE 架构 |
| [[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操|Tokenizer-Embedding-手撕实操]] | BPE/WP/SP + 位置编码谱系 | RoPE 和 ALiBi 对比 |
| [[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写\|lc8-MLA 从零手写]] | MHA→MLA：低秩压缩/矩阵吸收/RoPE 分离/c-cache | MLA vs MQA 本质区别（低秩表示 vs 共享头） |
| [[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写\|lc8-TPA+YaRN 从零手写]] | TPA 张量积低秩分解 + YaRN NTK-by-parts 分段外推 | 长上下文外推三策略对比（PI/NTK/YaRN） |
| [[AI/LLM/MA-RLHF课程/lc8-mHC-流形超连接从零手写\|lc8-mHC 从零手写]] | 残差→HC→mHC：doubly stochastic + Sinkhorn-Knopp | DeepSeek V4 预研，residual 系统性重设计 |
| [[AI/LLM/MA-RLHF课程/lc8-RoPE全家桶手撕实操\|lc8-RoPE 全家桶]] | 标准 RoPE 推导 + 衰减分析 + PI/NTK/YaRN 三策略对比 | RoPE 外推失效原因 + YaRN 为何更优 |
| [[AI/LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操\|lc8-GQA+KV Cache]] | GQA repeat_kv + KV Cache 增量解码 position 追踪 | GQA 如何实现：Q 头复用 vs 共享 K/V |
| [[AI/LLM/MA-RLHF课程/lc8-GPTLoss-Muon优化器-手撕实操\|lc8-GPT Loss+Muon]] | GPT 训练 Loss 格式 + Muon Newton-Schulz 正交化 | Muon 比 Adam 好在哪里（梯度方向偏差问题） |

### ⚡ 推理优化（8 篇）

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]] | Online Softmax + 分块前向 + 反向传播 | IO 复杂度从 O(N²) 降至 O(N) |
| [[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]] | PagedAttention + Continuous Batching 整体架构 | 物理/逻辑 KV 块映射 |
| [[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]] | 动态 Batch 调度器（TokenSlot + Dynamic Scheduler） | GPU 利用率 30% → 90%+ |
| [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] | 分页 KV Cache 管理（BlockTable + 逻辑→物理映射） | Prefix Caching + CoW 机制 |
| [[AI/LLM/Inference/vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]] | PagedAttention Kernel（非连续块 Attention 计算） | Triton/CUDA Kernel 接口 |
| [[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] | SARATHI: Prefill 分块 + Decode piggybacking | TTFT vs TPOT 两难 |
| [[AI/LLM/Inference/Speculative-Decoding-手撕实操|Speculative-Decoding-手撕实操]] | Draft-Target 对 + 拒绝采样（无偏推断） | γ=4 猜测窗口加速比估算 |
| [[AI/LLM/Inference/PD-Disaggregation-手撕实操|PD-Disaggregation-手撕实操]] | Prefill-Decode 物理分离 + Ray Actor + KV 异步传输 | compute-bound vs memory-bound 分离设计 |

### 🔧 分布式训练（7 篇）

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/Infra/分布式训练通信原语-手撕实操|分布式训练通信原语-手撕实操]] | NCCL 8 大原语 + Ring-AllReduce 推导 | 各并行范式的通信选型 |
| [[AI/LLM/Infra/ZeRO-手撕实操|ZeRO-手撕实操]] | ZeRO-1/2/3 + 分布式 Adam | 三阶段显存节省量化对比 |
| [[AI/LLM/Infra/Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]] | 列/行并行 + 序列并行 | all-reduce 通信位置选择 |
| [[AI/LLM/Infra/Pipeline-Parallel-手撕实操|Pipeline-Parallel-手撕实操]] | GPipe → 1F1B → 交错式 1F1B | Bubble 率公式推导 |
| [[AI/LLM/Infra/MoE-Context-Parallel-手撕实操|MoE-Context-Parallel-手撕实操]] | Expert 并行 + Ring Attention | MoE all-to-all vs TP all-reduce |
| [[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]] | Generator-Coordinator-Trainer 三角 | 训推分离 + 异步 GRPO |
| [[AI/LLM/Infra/Ray-推理系统实操|Ray-推理系统实操]] | Ray Actor vLLM 封装 + 负载均衡 | Rollout 生成侧设计 |

### 🎯 RL 对齐（16 篇）

**手撕实操（理解原理，面试速答）**

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/RL/Fundamentals/RL基础算法手撕实操|RL基础算法手撕实操]] | REINFORCE/A2C/PPO/DQN | LLM 对齐中各算法选择 |
| [[AI/LLM/SFT/SFT-手撕实操|SFT-手撕实操]] | Chat Template + Loss Mask + LoRA | RLHF 流水线第一阶段 |
| [[AI/LLM/MA-RLHF课程/lc6-SFT全链路-PyTorch手撕实操|lc6-SFT全链路 PyTorch 手撕实操]] ★★★★★ | SFT 全链路：Dataset→Collate→Loss Mask→纯PyTorch→SFTTrainer，三路径完整对比 | SFT Loss vs 预训练 Loss 本质区别；Chat Template 为何必要 |
| [[AI/LLM/MA-RLHF课程/lc6-LoRA-手撕实操|lc6-LoRA 手撕实操]] ★★★★★ | LoRA 原理推导+梯度推导+QLoRA/DoRA/LoRA+三变体实操，41 cells | rank 为什么可以极低；QLoRA 的 NF4 量化原理 |
| [[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]] | 四模型架构 + GAE + MA-PPO | 多适配器节省 3x 显存 |
| [[AI/LLM/RL/PPO/MA-RLHF-核心代码注解|MA-RLHF-核心代码注解]] | 完整项目代码注解 | MA-PPO Trainer 逐行解析 |
| [[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]] | Clip 版 + 简化版 loss + GAE 替换 | GRPO vs PPO 无 Critic |
| [[AI/LLM/RL/DPO/DPO-手撕实操|DPO-手撕实操]] | Bradley-Terry + 隐式 reward 推导 | DPO 如何绕过 RM 训练 |
| [[AI/LLM/RL/KTO/KTO-手撕实操|KTO-手撕实操]] | 前景理论 + 单样本偏好 | KTO vs DPO 数据效率 |
| [[AI/LLM/RL/PPO/PRM-O1-Search-手撕实操|PRM-O1-Search-手撕实操]] | Process Reward + Beam Search + MCTS | Test-time compute scaling |

**完整 Notebook 实现（端到端，深入理解）**

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/RL/DPO/Bradley-Terry模型实现|Bradley-Terry模型实现]] | BT 偏好建模（DPO 数学基础） | `P(y_w>y_l) = σ(r_w - r_l)` 推导 |
| [[AI/LLM/RL/PPO/LLaMA2-Reward-Model实现|LLaMA2-Reward-Model实现]] | LLaMA2 + RM Head + BT Loss 完整实现 | RM 训练 vs 直接 DPO 的工程 tradeoff |
| [[AI/LLM/RL/PPO/RLHF-PPO-完整Pytorch实现|RLHF-PPO-完整Pytorch实现]] | 四模型 56-cell 完整实现（GAE + KL 约束全链路） | PPO 与 GRPO 的 Critic 有无差异 |
| [[AI/LLM/RL/PPO/O1-PRM搜索完整实现|O1-PRM搜索完整实现]] | PRM + MCTS/Beam Search 端到端搜索 | UCT 选择公式 + 树回溯 |
| [[AI/LLM/RL/GRPO/GRPO-完整Notebook实现|GRPO-完整Notebook实现]] | 组采样 + advantage 归一化 + KL 项完整实现 | group_std 归一化 vs GAE |
| [[AI/LLM/RL/GRPO/GRPO-KL散度三种近似|GRPO-KL散度三种近似]] | k1/k2/k3 Schulman 近似实现对比 | 精度 vs 计算成本 tradeoff |
| [[AI/LLM/RL/DPO/DPO-完整Notebook实现|DPO-完整Notebook实现]] | 偏好对处理 + log-ratio loss 完整实现 | β 温度对策略偏离的控制 |
| [[AI/LLM/RL/KTO/KTO-完整Notebook实现|KTO-完整Notebook实现]] | 前景理论偏好建模，无需成对数据 | z_ref 参考期望的计算方式 |

### 🔩 xtrain 从零手写（7 篇）——不依赖框架，深度强化版

> 难度最高：纯 `torch.distributed` + P2P 通信实现，理解底层才能理解框架

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/LLM/Infra/xtrain-lc1-分布式通信原语从零手写|xtrain-lc1-通信原语]] | Ring-AllReduce + NCCL 8原语从零实现 | 各操作通信量公式 |
| [[AI/LLM/Infra/xtrain-lc2-数据并行从零手写|xtrain-lc2-数据并行]] | DP/DDP 从零实现，gradient sync 机制 | 为什么 reduce 梯度而非 loss |
| [[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写|xtrain-lc3-ZeRO]] | ZeRO-1/2/3 三阶段分片从零实现 | flatten 化切分 vs 按层切分 |
| [[AI/LLM/Infra/xtrain-lc4-张量并行从零手写|xtrain-lc4-张量并行]] | 列/行并行 Linear + 序列并行从零实现 | MLP 并行只需 1 次 AllReduce 的原因 |
| [[AI/LLM/Infra/xtrain-lc5-流水线并行从零手写|xtrain-lc5-流水线并行]] | 1F1B + DualPipe 从零实现（Bubble率推导） | DualPipe 如何将 bubble 降至接近零 |
| [[AI/LLM/Infra/xtrain-lc6-Context并行RingAttention手写|xtrain-lc6-Context并行]] | Ring Attention 从零实现，超长上下文训练 | Ring Attention 通信量与 GPU 数无关 |
| [[AI/LLM/Infra/xtrain-lc7-MoE专家并行从零手写|xtrain-lc7-MoE专家并行]] | Expert Parallelism + AllToAll 路由从零实现 | dispatch-combine 路径的通信计算重叠 |

### 🖼️ 多模态（1 篇）

| 笔记 | 核心内容 | 面试重点 |
|------|----------|----------|
| [[AI/MLLM/CLIP-ViT-LLaVA-手撕实操|CLIP-ViT-LLaVA-手撕实操]] | ViT + CLIP InfoNCE + LLaVA 投影 | 视觉 token 如何进 LLM |

---

## 学习路径建议

### 路径 A：架构面试速通（1天）
```
基础数学组件手撕 → Transformer → GPT2 → Llama → FlashAttention → vLLM
```

### 路径 B：RL 对齐面试速通（1天）
```
RL基础算法 → SFT → PPO → GRPO → DPO → (KTO/PRM 选读)
```

### 路径 C：分布式系统面试速通（1天）
```
通信原语 → ZeRO → Tensor Parallel → Pipeline Parallel → Ray分布式RL
```

### 路径 D：全栈 LLM 工程师（3天）
```
路径A + 路径B + 路径C + DeepSeek-V3 + MoE-Context-Parallel
```

### 路径 E：推理系统深度（1.5天）
```
Continue-Batching → vLLM-PageKVCache → vLLM-PageAttention → Chunked-Prefill → Speculative-Decoding → PD-Disaggregation
→ 深入阅读：LLM-推理优化-2026-全景
```

### 路径 F：分布式训练硬核 xtrain（2天）
```
xtrain-lc1（通信原语）→ lc2（DP/DDP）→ lc3（ZeRO）→ lc4（TP）→ lc5（PP/DualPipe）→ lc6（CP/RingAttn）→ lc7（MoE/EP）
→ 配合：分布式训练.md（理论） + xtrain-分布式并行手写-MOC（课程地图）
```

### 路径 G：RL 对齐完整链路（2天）
```
BT模型实现 → DPO-手撕 → DPO-Notebook → KTO-手撕 → KTO-Notebook
↓
RL基础算法手撕 → RLHF-PPO完整实现 → LLaMA2-RM实现
↓
GRPO-手撕 → GRPO-Notebook → GRPO-KL三种近似 → PRM-O1搜索实现
→ 深入阅读：GRPO深度理解 + RLHF-DPO-2026全景
```

---

## 关联论文

本系列笔记对应的理论深度版：

- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO深度理解]] — GRPO 理论完整推导
- [[AI/LLM/RL/RLHF-DPO-2026-技术全景|RLHF-DPO-2026全景]] — 偏好优化算法全谱
- [[AI/LLM/Architecture/Transformer架构深度解析-2026技术全景|Transformer架构深度解析]] — 架构演进理论视角
- [[AI/LLM/RL/Other-Algorithms/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] — off-policy RL 最新进展（与 Ray 分布式训练直接相关）

---

## 来源说明

所有代码来自 [MA-RLHF](https://github.com/dhcode-cpp/MA-RLHF)（MIT License），**教学级代码，不是生产代码**。用于理解核心算法原理，实际部署请参考 verl/OpenRLHF/TRL 等成熟框架。
