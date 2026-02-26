---
title: "vLLM"
brief: "UC Berkeley 开源高吞吐 LLM 推理引擎，核心创新 PagedAttention：将 KV Cache 分页管理（类操作系统虚拟内存），消除内存碎片，吞吐量比 HuggingFace Transformers 高 24×。支持 Continuous Batching、前缀缓存、多 GPU 张量并行，是工业 LLM 服务的事实标准。"
type: concept
domain: ai/llm/inference
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/inference
  - type/concept
---
# vLLM

## 概述

vLLM 是 UC Berkeley 在 2023 年发布的高性能 LLM 推理引擎，核心创新是 **PagedAttention**——借鉴操作系统虚拟内存的分页思想来管理 KV Cache。

在 LLM 推理中，KV Cache 的显存管理是核心瓶颈。传统实现为每个请求预分配最大长度的连续显存，浪费率高达 60-80%。vLLM 通过分页机制将显存碎片化问题降到最低，吞吐量提升 2-4x。

## PagedAttention 核心原理

### 问题：KV Cache 的显存浪费

LLM 自回归生成时，每个 token 需要缓存所有前面 token 的 Key 和 Value。传统做法：

```
请求 1: [K/V预分配 2048 tokens] → 实际只用了 500 tokens → 75% 浪费
请求 2: [K/V预分配 2048 tokens] → 实际只用了 1200 tokens → 41% 浪费
```

而且内存必须连续分配，导致碎片化——即使总显存够，也可能因为找不到足够大的连续块而无法服务新请求。

### 解法：分页

借鉴 OS 虚拟内存的 paging：

```
物理显存被分成固定大小的 Block（如 16 tokens）
每个请求的 KV Cache 由多个不连续的 Block 组成
用 Block Table 维护逻辑位置 → 物理位置的映射

请求 1: [Block 5][Block 12][Block 3]... → 逻辑连续，物理不连续
请求 2: [Block 7][Block 1][Block 9]...
```

核心优势：
1. **几乎零碎片**：按需分配 block，最后一个 block 内部浪费平均只有半个 block
2. **灵活的内存共享**：beam search、parallel sampling 可以共享 prefix 的 KV block（copy-on-write）
3. **动态增长**：生成过程中按需分配新 block，不需要预估最大长度

### PagedAttention 的计算

Attention 计算需要适配分页：

```python
# 传统 Attention
attention = softmax(Q @ K.T / sqrt(d)) @ V  # K, V 连续存储

# PagedAttention
# K, V 分散在不同的 block 中
for block_idx in block_table[seq_id]:
    k_block = physical_blocks[block_idx].key    # [block_size, num_heads, head_dim]
    v_block = physical_blocks[block_idx].value
    # 逐 block 计算 attention score，最后合并
```

vLLM 用定制的 CUDA kernel 实现了高效的分页 attention，overhead 很小。

## 架构设计

```
Client → API Server (OpenAI-compatible)
              ↓
         Scheduler (调度器)
              ↓
         Worker(s) → GPU(s)
              ↓
         Model Executor (模型执行)
              ↓
         PagedAttention Engine (KV Cache 管理)
```

### Scheduler

vLLM 的调度策略是 **continuous batching**（也叫 iteration-level batching）：

- 不等一个 batch 全部生成完再处理下一个 batch
- 每次 iteration（每生成一个 token）都可以加入新请求
- 已完成的请求立即释放，腾出显存给排队请求

这比 static batching 的吞吐量高很多，因为：
- 短请求不需要等长请求
- GPU 利用率接近 100%

### Tensor Parallelism

vLLM 原生支持多 GPU 推理：
- **Tensor Parallel**：把模型的每一层切分到多个 GPU
- **Pipeline Parallel**（较新版本）：把不同层分到不同 GPU
- 支持 NCCL 通信

```python
# 使用示例
from vllm import LLM

model = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # 4 GPU
    gpu_memory_utilization=0.9,
)
```

## 使用方式

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
)

prompts = ["Explain quantum computing in simple terms."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible Server

```bash
# 启动服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85

# 调用（完全兼容 OpenAI API）
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 在 RL 训练中的角色

vLLM 在 RLHF/GRPO 训练中扮演关键角色——用于 **rollout 阶段的快速采样**：

```
GRPO 训练循环:
1. Policy model 生成 response → 用 vLLM 加速这一步
2. Reward model 打分
3. 计算 advantage
4. 更新 policy

vLLM 的 continuous batching 在这里特别有用，
因为不同 prompt 的 response 长度差异很大
```

TRL 和 verl 都集成了 vLLM 作为 generation backend。

## 性能对比

与其他推理框架的对比（大致量级，具体取决于模型和硬件）：

| 框架 | 吞吐量 | 特点 |
|------|--------|------|
| HuggingFace Transformers | 1x（基线） | 最简单、最灵活 |
| vLLM | 3-5x | PagedAttention、continuous batching |
| TensorRT-LLM | 3-6x | NVIDIA 优化、编译优化 |
| SGLang | 3-5x | RadixAttention、结构化生成 |

vLLM 的优势是 **易用性和生态**——OpenAI API 兼容、支持模型广泛、社区活跃。

## 关键配置参数

```python
LLM(
    model="...",
    tensor_parallel_size=1,      # GPU 数量
    gpu_memory_utilization=0.9,  # GPU 显存利用率（预留 10% 给其他）
    max_model_len=8192,          # 最大序列长度（影响 KV Cache 大小）
    enforce_eager=False,         # True 关闭 CUDA graph（调试用）
    quantization="awq",          # 量化方式（awq/gptq/fp8 等）
    dtype="auto",                # 数据类型
    enable_prefix_caching=True,  # 前缀缓存（对相同系统提示有用）
)
```

## 相关

- [[Ollama|Ollama]] — 本地推理的另一个选择（面向个人用户）
- [[DeepSeek-R1|DeepSeek-R1]] — vLLM 用于 R1 的推理部署
- [[GRPO 深度理解|GRPO 深度理解]] — vLLM 在 RL 训练中的角色
- [[TRL 概述|TRL 概述]] — 集成 vLLM 的训练框架
- [[verl 概述|verl 概述]] — 另一个集成 vLLM 的框架
- [[分布式训练|分布式训练]] — Tensor Parallel 基础
- [[LLaMA|LLaMA]]
- [[AI/3-LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]] — **代码路径**：从零实现 PagedAttention + Continuous Batching 完整代码注解，理解 vLLM 内部机制 ⭐⭐⭐⭐⭐
- [[AI/3-LLM/Inference/vLLM-V0-V1-完整系统实操|vLLM V0/V1 完整系统实操]] — **系统级深度**：V0 page-level Prefill 转换 + last-token 提取；V1 SchedulerInfo + merge_prompt + KV.split 统一调度；完整 10 文件代码结构 ⭐⭐⭐⭐⭐
- [[LLM-推理优化-2026-全景|LLM 推理优化 2026 全景]] — vLLM 在整个推理优化生态中的位置
