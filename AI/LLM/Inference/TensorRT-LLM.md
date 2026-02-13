---
title: "TensorRT-LLM 深度解析"
date: 2026-02-13
tags:
  - ai/llm/inference
  - ai/nvidia
  - ai/llm/serving
  - type/concept
  - interview/hot
status: active
---

# TensorRT-LLM 深度解析

> NVIDIA 官方 LLM 推理优化框架——Kernel Fusion、量化、In-flight Batching 的极致优化

## 1. 概述

TensorRT-LLM 是 NVIDIA 在 2023 年底开源的 LLM 推理框架，基于 TensorRT 深度学习推理引擎构建，专为 NVIDIA GPU 优化。它的核心定位是 **在 NVIDIA 硬件上提供最极致的推理性能**。

### 架构全景

```
                    ┌──────────────────────────┐
                    │    Triton Inference Server │  ← 服务层（可选）
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    TensorRT-LLM Runtime   │  ← 调度 + 内存管理
                    │  ┌─────────────────────┐  │
                    │  │ In-flight Batching   │  │
                    │  │ Paged KV Cache       │  │
                    │  │ Beam Search / Top-K  │  │
                    │  └─────────────────────┘  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    TensorRT Engine        │  ← 优化后的计算图
                    │  ┌─────────────────────┐  │
                    │  │ Kernel Fusion        │  │
                    │  │ FP8/INT4 Quantization│  │
                    │  │ Custom CUDA Kernels  │  │
                    │  └─────────────────────┘  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    NVIDIA GPU (A100/H100) │
                    └──────────────────────────┘
```

## 2. 核心优化技术

### (1) Kernel Fusion

将多个小的 GPU 操作合并为一个大 kernel，减少 kernel launch 开销和中间结果的 HBM 读写：

```
未融合:                          融合后:
┌──────┐  ┌──────┐  ┌──────┐    ┌───────────────────────┐
│MatMul│→ │ Bias │→ │ GELU │    │  MatMul+Bias+GELU     │
└──────┘  └──────┘  └──────┘    │  (一个 kernel 完成)    │
  ↑写HBM   ↑读写HBM  ↑读写HBM    └───────────────────────┘
                                   只读写 HBM 一次
```

TensorRT-LLM 的典型 fusion 模式：

| Fusion 类型 | 合并操作 | 节省 |
|-------------|---------|------|
| **MHA Fusion** | QKV 投影 + Attention + Output 投影 | 3-4 个 kernel → 1 个 |
| **MLP Fusion** | Gate + Up + SiLU + Down | 4 → 1 |
| **LayerNorm Fusion** | Residual Add + LayerNorm | 2 → 1 |
| **Quantize Fusion** | Dequant + MatMul + Quant | 3 → 1 |

### (2) 量化支持

TensorRT-LLM 提供业界最全的量化选项：

```python
# TensorRT-LLM 量化配置
from tensorrt_llm.quantization import QuantConfig

# FP8 (Hopper+, 推荐)
fp8_config = QuantConfig(
    quant_mode="fp8",
    kv_cache_quant_mode="fp8"  # KV Cache 也量化
)

# W4A16 (GPTQ/AWQ 风格)
w4a16_config = QuantConfig(
    quant_mode="w4a16_gptq",
    group_size=128
)

# SmoothQuant (W8A8)
sq_config = QuantConfig(
    quant_mode="smoothquant",
    smoothquant_val=0.5
)
```

| 量化方式 | 权重精度 | 激活精度 | 速度提升 | 质量影响 |
|---------|---------|---------|---------|---------|
| FP16 (baseline) | FP16 | FP16 | 1x | 无 |
| FP8 | FP8 | FP8 | **1.5-2x** | < 0.5% |
| W4A16 (AWQ) | INT4 | FP16 | **1.3-1.8x** | 1-2% |
| SmoothQuant | INT8 | INT8 | **1.3-1.5x** | < 1% |
| W4A8 | INT4 | INT8 | **2-2.5x** | 1-3% |

### (3) In-flight Batching

TensorRT-LLM 的 [[Continuous Batching]] 实现：

```
In-flight Batching 特点:
1. Iteration-level 调度 (同 continuous batching)
2. Chunked Context (分块 prefill)
3. Paged KV Cache (类似 PagedAttention)
4. 支持 Speculative Decoding

与 vLLM 的区别:
- vLLM: Python 调度器 + CUDA kernel
- TRT-LLM: C++ 调度器 + TensorRT 优化引擎
  → 调度本身开销更低，适合超高并发
```

### (4) Tensor Parallelism + Pipeline Parallelism

```python
# 多卡推理配置
import tensorrt_llm

# 4 卡 Tensor Parallel
build_config = tensorrt_llm.BuildConfig(
    max_batch_size=64,
    max_input_len=4096,
    max_seq_len=8192,
)
build_config.plugin_config.set_nccl_plugin()

# TP=4: 每层权重切分到 4 卡
# PP=2: 前 40 层在 GPU 0-3，后 40 层在 GPU 4-7
parallel_config = tensorrt_llm.Mapping(
    world_size=8,
    tp_size=4,
    pp_size=2
)
```

## 3. 使用流程

### 完整部署流程

```bash
# Step 1: 安装
pip install tensorrt-llm -U --extra-index-url https://pypi.nvidia.com

# Step 2: 下载模型并转换
python convert_checkpoint.py \
    --model_dir ./Llama-3.1-70B-Instruct \
    --output_dir ./trt_ckpt/llama-70b-tp4 \
    --dtype float16 \
    --tp_size 4

# Step 3: 构建 TensorRT 引擎
trtllm-build \
    --checkpoint_dir ./trt_ckpt/llama-70b-tp4 \
    --output_dir ./trt_engines/llama-70b-tp4 \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --paged_kv_cache enable \
    --use_fused_mlp enable

# Step 4: 运行推理
mpirun -n 4 python run.py \
    --engine_dir ./trt_engines/llama-70b-tp4 \
    --tokenizer_dir ./Llama-3.1-70B-Instruct \
    --max_output_len 512
```

### Python API

```python
import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams

# 加载引擎
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    quantization="fp8",      # 自动量化
    kv_cache_config={
        "enable_block_reuse": True,  # Prefix Caching
    }
)

# 推理
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

outputs = llm.generate(
    ["What is FlashAttention?", "Explain GQA."],
    sampling_params=sampling_params
)
for output in outputs:
    print(output.text)
```

## 4. TensorRT-LLM vs vLLM 对比

| 维度 | TensorRT-LLM | [[vLLM]] |
|------|-------------|---------|
| **厂商** | NVIDIA 官方 | UC Berkeley → 社区 |
| **硬件** | 仅 NVIDIA GPU | NVIDIA + AMD ROCm + CPU |
| **优化方式** | 编译优化 (TensorRT engine) | 运行时优化 (Python + CUDA) |
| **Kernel** | TRT 自动优化 + 手写 kernel | FlashAttention + FlashInfer |
| **量化** | FP8/INT4/INT8 原生 | AWQ/GPTQ/FP8 + 第三方 |
| **易用性** | 需要编译引擎（分钟级） | 直接加载权重（秒级） |
| **模型支持** | 主流模型 | 更广泛 |
| **API 兼容** | OpenAI 兼容 (via Triton) | OpenAI 原生兼容 |
| **社区** | NVIDIA 主导 | 活跃社区 |
| **适合场景** | 极致性能、NVIDIA 硬件 | 快速原型、多硬件、灵活性 |

### 性能对比 (2025 Benchmark)

```
模型: LLaMA 3.1 70B, 硬件: 4× H100

Throughput (tokens/s, batch=64):
  TensorRT-LLM FP8:  ~12,000
  vLLM FP16:         ~6,500
  vLLM FP8:          ~9,000
  TRT-LLM 优势:      ~33-85%

TTFT (ms, input=2048 tokens):
  TensorRT-LLM:  ~45ms
  vLLM:           ~65ms

TPOT (ms, batch=32):
  TensorRT-LLM:  ~18ms
  vLLM:           ~22ms

注: B200 GPU 上 TRT-LLM 优势更大 (深度优化新架构)
    短对话场景差距较小
```

## 5. 高级特性

### Speculative Decoding

```python
# Draft model 投机解码
build_config = {
    "speculative_decoding_mode": "draft_tokens",
    "max_draft_len": 5,
}

# 主模型 + 小模型协同
# Draft: LLaMA-3.1-1B → 快速猜 5 个 token
# Target: LLaMA-3.1-70B → 一次验证
# 期望: 每步接受 3-4 个 token → 加速 2-3x
```

### Prefix Caching (KV Cache Reuse)

```python
# 共享系统 prompt 的 KV Cache
kv_cache_config = {
    "enable_block_reuse": True,   # 自动检测共享 prefix
    "max_tokens_in_paged_kv_cache": 1000000
}

# 效果: 多请求共享相同 system prompt → prefill 跳过
# 适合: 大量请求使用相同 system prompt 的场景
```

## 6. 部署最佳实践

### 选型决策树

```
开始
  ├── 需要最极致性能 且 只用 NVIDIA GPU?
  │   ├── Yes → TensorRT-LLM
  │   │   ├── H100/B200 → FP8 量化
  │   │   └── A100 → FP16 或 W4A16
  │   └── No → 继续
  │
  ├── 需要快速部署 或 多硬件支持?
  │   └── Yes → vLLM
  │
  ├── 单机简单推理?
  │   └── Yes → [[Ollama]] / llama.cpp
  │
  └── 前沿功能 (RadixAttention, 结构化输出)?
      └── SGLang
```

### 生产环境配置建议

```bash
# 生产部署推荐配置 (4× H100)
trtllm-build \
    --tp_size 4 \
    --max_batch_size 128 \            # 高并发
    --max_input_len 8192 \            # 支持长 prompt
    --max_seq_len 16384 \             # 输入+输出总长
    --paged_kv_cache enable \         # 必开
    --use_paged_context_fmha enable \ # Paged Flash Attention
    --gemm_plugin float16 \           # 或 fp8
    --remove_input_padding enable \   # 去除 padding
    --use_fused_mlp enable \          # 融合 MLP
    --multi_block_mode enable         # 长序列优化
```

## 7. 与其他优化技术的关系

- **[[FlashAttention]]**：TRT-LLM 使用 FlashAttention 的思想实现 fused attention kernel
- **[[Continuous Batching]]**：TRT-LLM 的 in-flight batching 是 continuous batching 的 C++ 高性能实现
- **[[量化综述|量化]]**：TRT-LLM 提供最完整的量化支持，尤其是 FP8 量化
- **[[KV Cache 优化]]**：Paged KV Cache + KV Cache 量化
- **[[Speculative Decoding]]**：TRT-LLM 原生支持 draft model speculative decoding
- **[[推理优化]]**：TRT-LLM 是推理优化技术的集大成者

## 面试常见问题

### Q1: TensorRT-LLM 相比 vLLM 的核心优势是什么？

**编译优化**：TRT-LLM 将模型编译为 TensorRT 引擎，进行图优化（算子融合、常量折叠、内存规划），生成针对特定 GPU 架构优化的 CUDA kernel。vLLM 是运行时优化，灵活但无法做到相同深度的优化。在 NVIDIA GPU 上，TRT-LLM 吞吐通常高 30-80%，尤其在 H100/B200 + FP8 场景下优势更大。

### Q2: Kernel Fusion 具体融合了哪些操作？为什么能加速？

典型融合：(1) QKV 投影+Attention+Output 投影；(2) Gate+Up+SiLU+Down MLP；(3) Residual Add+LayerNorm。加速原因：减少 kernel launch 开销（每次 launch ~5μs），更重要的是 **减少 HBM 读写**——未融合时中间结果需写入 HBM 再读回，融合后中间结果留在 register/shared memory 中。

### Q3: TRT-LLM 的量化流程是怎样的？

两步：(1) **Calibration**——用校准数据跑前向，收集激活分布统计（min/max/histogram）；(2) **Engine Build**——根据统计信息选择量化参数，编译引擎。FP8 量化最简单（per-tensor scaling），INT4 (AWQ/GPTQ) 需要预先用工具做权重量化。推荐路线：H100+ → FP8（质量好+速度快），A100 → W4A16 AWQ。

### Q4: 什么时候应该选 TensorRT-LLM 而不是 vLLM？

三个判断条件：(1) **纯 NVIDIA GPU 部署** + 不需要多硬件支持；(2) **对延迟/吞吐有极致要求**，愿意接受编译引擎的复杂度；(3) **大规模生产环境**，几十上百卡的集群部署，需要 Triton Inference Server 的企业级功能（模型版本管理、A/B 测试、多模型调度）。如果是快速原型或需要频繁更换模型，vLLM 更合适。

### Q5: TensorRT-LLM 的 in-flight batching 和 vLLM 的 continuous batching 有什么区别？

**原理相同**，都是 iteration-level 调度 + paged KV cache。区别在于实现：TRT-LLM 用 **C++ runtime** 做调度，开销更低（μs 级），适合超高并发（数百并发请求）；vLLM 用 **Python 调度器**，灵活性更好但有 GIL 和解释器开销。实际差异在低并发时不明显，高并发（100+）时 TRT-LLM 的调度器优势会体现出来。
