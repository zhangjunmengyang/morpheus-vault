---
brief: "Qwen3.5 技术分析——阿里 Qwen3.5 的架构改进和能力提升分析；Linear Attention 机制/GSPO 训练/MoE 扩展的技术路线；在推理/多模态/Agent 上的能力格局；国内开源 LLM 旗舰的全面评测。"
title: "Qwen3.5 技术分析"
type: model
domain: ai/frontiers
tags:
  - ai/frontiers
  - type/model
  - model/qwen
  - arch/moe
created: 2026-02-16
---

# Qwen3.5：阿里春节 AI 攻势旗舰

> 发布: 2026-02-16 | 阿里云 / 通义千问团队
> 标签: #MoE #Multimodal #AgentAI #OpenSource #Apache2.0

## TL;DR

Qwen3.5 是阿里春节前 AI 攻势的核心发布，**397B MoE / 17B active**，原生多模态（文本+图像+视频），主打 Agentic AI。关键数字：BrowseComp 78.6（open weights 最高）、SWE-bench 76.4、比 Qwen2.5 便宜 60%。

---

## 1. 模型系列

| 模型 | 规格 | 获取方式 |
|------|------|---------|
| **Qwen3.5-397B-A17B** | 397B total / 17B active，开源 | HuggingFace，Apache 2.0 |
| **Qwen3.5-Plus** | 规格未公开，hosted | 阿里云 Model Studio，1M token context |

两款均可通过 Qwen Chat 访问。

---

## 2. 核心架构

### MoE 设计
- **397B total / 17B active per forward pass** — 稀疏激活，激活内存比同能力 dense 模型减少 **95%**
- Sparse MoE + **Hybrid Linear Attention**（hybrid attention 和稀疏 expert routing 并行计算）
- 比上一代便宜 **60%**；吞吐 **8x** 提升；32K context 下 8.6x、256K context 下 **19x** decoding 加速
- 8xH100 实测：约 **45 tokens/s**

### 词表
- **250K tokens**（Qwen3 的 152K → 250K，+65%）
- 非英语文本 token 数节省 10-60%，201 语言成本直接降低

### 原生多模态（Early Fusion）
- 文本 + 图像 + 视频 从**预训练第一阶段**就 joint training，非 adapter 拼接
- 视觉规格：图片最高 1344×1344；视频 60 秒 @ 8 FPS
- UI 截图元素检测，支持跨 app 自主操作

### 训练 Infrastructure
- 原生 **FP8** pipeline：激活内存减少 ~50%
- 异步 RL 框架：训练/推理 workload 解耦（类 GLM-5 Slime 思路）
- **Speculative decoding + rollout replay + multi-turn rollout locking** 三合一推理优化

### 语言支持
- Qwen2.5 (29) → Qwen3 (119) → **Qwen3.5 (201 语言)** — 覆盖范围行业领先

---

## 3. Benchmark 数据

### 推理 & 数学
| Benchmark | Qwen3.5 | 说明 |
|-----------|---------|------|
| **AIME26** | **91.3** | 奥数级，2026 竞争榜 top tier |
| GPQA Diamond | **88.4** | 研究生级推理，frontier 水平 |
| MathVista | 90.3 | 数学图文推理 |
| MMLU-Pro | 87.8 | 多语言知识 |
| MMLU | 88.5 | 知识广度 |
| IFBench | 76.5 | 指令跟随 |

### 代码 & Agentic（核心亮点）
| Benchmark | Qwen3.5 | 对比 |
|-----------|---------|------|
| **LiveCodeBench v6** | **83.6** | 竞赛级编程，极强信号 |
| SWE-bench Verified | 76.4 | 真实代码工作流 |
| Terminal-Bench 2 | 52.5 | 终端 coding agent |
| **BrowseComp** | **78.6** | open weights SOTA；Claude Opus 4.6: 84.0；Gemini 3 Pro: 59.2 |
| BFCL v4 | 72.9 | Agentic tool use |

BrowseComp 78.6 是 open weights 中的 SOTA；LiveCodeBench v6 83.6 在竞赛编程上极具竞争力。

### 多模态
| Benchmark | Qwen3.5 |
|-----------|---------|
| **OmniDocBench v1.5** | **90.8** |
| MMMU | 85.0 |
| MMMU-Pro | 79.0 |
| Video-MME | 87.5 |
| VITA-Bench | 49.7 |
| ERQA | 67.5 |

---

## 4. 我的批判性评估

### ✅ 真正值得关注的

1. **BrowseComp 78.6 = open weights SOTA**：这个 benchmark 测的是需要多跳搜索的复杂事实问题，难以 cherry-pick。意味着 Qwen3.5 的 web agent 能力在 open weights 中领先。

2. **397B/17B 的成本效率**：17B active 意味着实际推理成本接近 17B dense，但能力接近 400B 级别。60% 价格降低对企业部署很有吸引力。

3. **原生多模态**：不是事后打补丁，而是 joint training。Video-MME 87.5 说明视频理解能力扎实。

4. **201 语言**：对全球化部署有意义，比大多数竞品覆盖范围广。

### ⚠️ 需要怀疑的地方

1. **"5x faster agent execution"**：相对于谁？baseline 是什么？Alibaba 没有给出清晰的 apples-to-apples 对比。

2. **Qwen3.5-Plus 细节缺失**：1M context 是怎么做到的？是 native 还是某种 retrieval augmentation？没有披露。

3. **SWE-bench 76.4**：这个数字介于 GLM-5（接近 Opus 4.5 的 ~80）和 Doubao（有差距）之间，表现中规中矩。

4. **VITA-Bench 49.7 偏低**：agentic multimodal interaction 这个维度相对薄弱，说明多模态 agent 还有提升空间。

### 与同期竞品对比

| 模型 | BrowseComp | SWE-bench | 价格 |
|------|-----------|-----------|------|
| Claude Opus 4.6 | 84.0 | ~80 | 高 |
| **Qwen3.5** | **78.6** | 76.4 | 中低 |
| GLM-5 | ~= Opus 4.5 | ~= Opus 4.5 | $0.80/M in |
| Gemini 3 Pro | 59.2 | - | - |

Qwen3.5 在 BrowseComp 的位置很显眼，但整体 ARC 能力（Agentic/Reasoning/Coding）三维综合来看略弱于 GLM-5。

---

## 5. 背景：阿里春节 AI 攻势

Qwen3.5 是阿里 2026 年春节前一系列发布的顶点：
- Qwen3-Coder-Next（代码专项）
- Qwen Image 2.0（图像生成）
- **Qwen3.5**（旗舰多模态 + agent）

战略意图明显：在 ByteDance (Doubao 2.0/Seedance 2.0)、智谱 (GLM-5)、MiniMax (M2.5) 的围攻下，阿里用 Apache 2.0 开源 + 低价 + 多模态综合能力来争夺市场。

---

## 6. 对老板的意义

1. **选型参考**：开源 multimodal agent 任务，Qwen3.5 是目前性价比最高的选择之一
2. **面试素材**：MoE sparse activation 的效率优势（17B active，400B 能力）是经典题
3. **BrowseComp 作为 benchmark 的价值**：测复杂多跳搜索，比 MMLU 更难作弊，值得关注

---

## 相关笔记

- [[AI/Frontiers/GLM-5-技术报告精读|GLM-5 技术报告]] — 同期竞品，ARC 综合更强
- [[AI/Frontiers/Doubao-Seed-2.0-技术分析|Doubao Seed 2.0]] — ByteDance 竞品，混合专家架构
- [[AI/Frontiers/2026年2月模型潮|2026年2月模型潮]] — 整体竞争格局背景
- [[AI/Frontiers/目录|Frontiers MOC]] — 前沿模型全景索引
- [[AI/LLM/Architecture/Attention变体综述|Attention 变体综述]] — Qwen3.5 采用的架构技术基础
- [[AI/LLM/目录|LLM MOC]] — 上级知识域

---
## 7. 部署参考

### Qwen3.5-Plus API（托管）
```python
from openai import OpenAI
client = OpenAI(
    api_key="your-dashscope-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[{"role": "user", "content": "..."}],
    stream=True
)
```
- **OpenAI SDK 兼容**，迁移成本极低
- 1M token context，内置 adaptive tool use + web search
- 定价：~$0.18/M tokens

### 开源自托管
- HuggingFace Apache 2.0 下载
- 最低：8xH100 GPUs（45 tok/s）
- 支持 vLLM / TGI serving + full fine-tuning

---

*Created: 2026-02-16 | Updated: 2026-02-19 | Source: digitalapplied.com + latestly.com + Reuters 交叉验证 | Confidence: High*
