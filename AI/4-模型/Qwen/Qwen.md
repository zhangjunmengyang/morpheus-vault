---
title: Qwen 系列：阿里通义千问的架构演进与技术突破
brief: Qwen 是阿里通义实验室的 LLM 系列，从 Qwen1（7B-72B）到 Qwen2.5 持续迭代。核心架构特点：GQA（KV Cache 减少 75%）+ SwiGLU 激活 + RoPE 位置编码 + YaRN 长上下文扩展。Qwen2.5-72B 在代码（HumanEval 89.5%）和数学（GSM8K 95.2%）上达到顶尖水平，29 种语言支持使其成为最强多语言开源模型之一。
type: survey
domain: ai/llm/architecture
created: 2026-02-14
updated: 2026-02-22
tags:
  - ai/llm/architecture
  - ai/llm/qwen
  - type/survey
status: complete
sources:
  - Bai et al. *Qwen Technical Report* arXiv:2309.16609
  - Yang et al. *Qwen2 Technical Report* arXiv:2407.10671
  - "Qwen Team. *Qwen2.5: A Party of Foundation Models* 技术报告 2024"
related:
  - "[[LLaMA|LLaMA]]"
  - "[[Transformer 通识|Transformer 通识]]"
  - "[[AI/3-LLM/Architecture/长上下文处理|长上下文处理]]"
  - "[[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]"
---

# Qwen 系列架构解析

Qwen（通义千问）是阿里巴巴通义实验室开发的大规模语言模型系列，从 2023 年首个版本发布至今经历了持续的技术演进，在多语言、代码生成、数学推理等多个维度取得了显著突破。

> 来源：Bai et al. *Qwen Technical Report* arXiv:2309.16609; Yang et al. *Qwen2 Technical Report* arXiv:2407.10671

## 1. Qwen 系列演进历程

### 1.1 Qwen 1.0（2023年）
Qwen 1.0 作为系列的起点，建立了基本的 Transformer 架构框架，支持中英双语，模型规模从7B到72B。关键特点：
- 基于标准 Transformer 架构，采用 Pre-RMSNorm
- 支持32K上下文长度
- 引入[[AI/3-LLM/Architecture/Attention 变体综述|分组查询注意力]]（GQA）优化推理效率
- 预训练数据覆盖多语言、代码、数学等领域

### 1.2 Qwen2（2024年初）
Qwen2 在架构和训练方面进行了全面升级：
- **扩展模型规模**：从0.5B到72B的完整尺寸矩阵
- **优化注意力机制**：全面采用 [[AI/3-LLM/Architecture/Attention 变体综述|分组查询注意力]]（GQA），显著提升推理速度
- **激活函数升级**：使用 [[AI/1-Foundations/ML-Basics/SwiGLU|SwiGLU]] 替代传统 ReLU，提升表达能力
- **位置编码改进**：采用 [[AI/3-LLM/Architecture/Transformer 位置编码|RoPE]] 旋转位置编码，支持更长序列
- **多语言增强**：预训练数据包含29种语言，总量达7万亿tokens

### 1.3 Qwen2.5（2024年中）
Qwen2.5 在保持架构稳定性的基础上，重点提升了特定能力：
- **代码能力突破**：在 HumanEval 等代码评测中达到SOTA水平
- **数学推理增强**：在 GSM8K、MATH 等数学推理任务上大幅提升
- **指令跟随优化**：通过大规模指令微调和[[RLHF-DPO-2026-技术全景|强化学习人类反馈]]（RLHF）提升对话质量
- **长上下文支持**：部分模型支持高达128K的上下文长度

### 1.4 Qwen3（2025年预期）
虽然尚未正式发布，但从技术路线图来看，Qwen3 可能会：
- 进一步扩大模型规模，可能达到千亿参数级别
- 引入更先进的[[AI/3-LLM/Architecture/MoE 深度解析|混合专家模型]]（MoE）架构
- 集成多模态能力，支持图像、音频等模态
- 优化推理效率，支持更高效的部署

## 2. 核心架构特点

### 2.1 分组查询注意力（GQA）
Qwen 系列全面采用 GQA，这是一种介于多头注意力（MHA）和多查询注意力（MQA）之间的设计：

```python
# GQA 核心原理
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups):
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.group_size = n_heads // n_kv_groups
        
    def forward(self, x):
        # Query 保持多头
        Q = self.project_q(x)  # [batch, seq, n_heads * head_dim]
        
        # Key, Value 使用分组
        K = self.project_k(x)  # [batch, seq, n_kv_groups * head_dim]
        V = self.project_v(x)  # [batch, seq, n_kv_groups * head_dim]
        
        # 通过重复扩展 K, V 到与 Q 相同的头数
        K = K.repeat_interleave(self.group_size, dim=-1)
        V = V.repeat_interleave(self.group_size, dim=-1)
```

**优势**（KV Cache 内存缩减公式）：

$$\text{Memory}_{\text{GQA}} = \text{Memory}_{\text{MHA}} \times \frac{n_{\text{kv\_groups}}}{n_{\text{heads}}}$$

例如 Qwen2.5-72B：32 头 → 8 组，KV Cache 内存减少 75%。
- 相比 MHA，减少了 KV Cache 内存占用
- 相比 MQA，保持了更好的模型质量
- 在大规模推理时显著提升吞吐量 20-40%

### 2.2 SwiGLU 激活函数
Qwen2 开始采用 SwiGLU（Swish-Gated Linear Unit），这是一种结合了 Swish 激活和门控机制的激活函数：

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV), \quad \text{Swish}(x) = x \cdot \sigma(x)$$

```python
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return swish(gate) * x

def swish(x):
    return x * torch.sigmoid(x)
```

**特点**：
- 相比 ReLU，具有更平滑的梯度特性
- 门控机制允许网络选择性地激活信息
- 在语言建模任务中表现优于 GELU 和 ReLU

### 2.3 RoPE 位置编码
Qwen 系列使用 [[AI/3-LLM/Architecture/Transformer 位置编码|RoPE]]（Rotary Position Embedding）进行位置编码：

```python
def apply_rope(q, k, cos, sin, position_ids):
    # 旋转变换
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**优势**：
- 相对位置编码，泛化能力更强
- 支持外推到更长序列
- 计算效率高，不增加参数量

### 2.4 YaRN 长上下文扩展
对于长上下文版本，Qwen 采用 YaRN（Yet another RoPE extensioN）技术：

```python
# YaRN 核心：动态缩放因子
def yarn_get_mscale(scale, mscale):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
```

**特点**：
- 动态调整 RoPE 的缩放因子
- 在不同位置范围使用不同的插值策略
- 有效支持从32K扩展到128K上下文

## 3. Qwen2.5 技术亮点

### 3.1 多语言能力
- **29种语言支持**：包括中、英、日、韩、法、德、西、阿拉伯等主要语言
- **跨语言一致性**：在不同语言上保持相似的性能水平
- **多语言代码注释**：支持多种语言的代码注释和文档生成

### 3.2 代码生成能力
- **HumanEval 89.5%**：在代码生成基准测试中达到顶尖水平
- **多语言编程**：支持 Python、Java、C++、JavaScript 等20+编程语言
- **代码理解与调试**：具备代码解释、错误检测和修复能力

### 3.3 数学推理能力
- **GSM8K 95.2%**：在小学数学应用题上接近人类水平
- **MATH 75.8%**：在竞赛级数学问题上表现突出
- **多步推理**：支持复杂的数学证明和推导过程

### 3.4 指令跟随与安全对齐
- **大规模 SFT**：使用超过100万条高质量指令数据进行监督微调
- **RLHF 优化**：通过人类反馈强化学习提升对话质量
- **安全过滤**：内置多层安全机制，减少有害输出

## 4. 与其他模型对比

### 4.1 vs LLaMA 3.1
| 维度 | Qwen2.5-72B | LLaMA3.1-70B |
|------|-------------|--------------|
| 多语言能力 | **优势**（29种语言） | 英语为主 |
| 代码生成 | **89.5%**（HumanEval） | 80.5% |
| 数学推理 | **95.2%**（GSM8K） | 92.3% |
| 长上下文 | 128K | 128K |
| 开源程度 | 权重+微调代码 | 权重+部分代码 |

### 4.2 vs Mistral Large
| 维度 | Qwen2.5-72B | Mistral Large |
|------|-------------|---------------|
| 模型规模 | 72B | ~140B（推测） |
| 推理效率 | **优势**（GQA优化） | 标准MHA |
| 多语言 | **29种语言** | 主要欧洲语言 |
| 部署友好性 | **开源，易部署** | API服务为主 |

### 4.3 vs DeepSeek Coder V2
| 维度 | Qwen2.5-Coder-32B | DeepSeek Coder V2-16B |
|------|------------------|---------------------|
| 代码生成 | 89.5%（HumanEval） | **92.5%** |
| 模型规模 | 32B | 16B |
| 多语言编程 | 20+语言 | **25+语言** |
| 通用能力 | **平衡较好** | 专注代码 |

## 面试常见问题

### Q1: Qwen2.5 相比 Qwen2 有哪些主要改进？

**答案**：
1. **数据质量提升**：预训练数据从3万亿扩展到18万亿tokens，质量过滤更严格
2. **特定能力强化**：代码生成（HumanEval 89.5%）、数学推理（GSM8K 95.2%）大幅提升
3. **指令跟随优化**：SFT数据规模扩大10倍，RLHF 流程优化
4. **长上下文支持**：部分模型支持128K上下文，采用YaRN技术
5. **多语言均衡**：29种语言的性能差距显著缩小

### Q2: GQA相比传统MHA有什么优势？如何影响推理性能？

**答案**：
1. **内存优化**：KV Cache 大小减少，从 `n_heads` 降低到 `n_kv_groups`
2. **推理加速**：减少 Key/Value 的计算和存储开销，提升吞吐量 20-40%
3. **质量保持**：相比 MQA，GQA 保持更多的注意力多样性，性能下降最小
4. **扩展友好**：在大模型部署时，内存节省效果更明显

**计算公式**：
```
Memory_GQA = Memory_MHA × (n_kv_groups / n_heads)
例如：32头 → 8组，内存减少75%
```

### Q3: Qwen系列为什么选择SwiGLU而不是ReLU或GELU？

**答案**：
1. **梯度特性**：SwiGLU 具有平滑的梯度，缓解梯度消失问题
2. **门控机制**：允许网络学习选择性激活，提高表达能力
3. **实验表现**：在大规模语言建模中，SwiGLU 相比 GELU 提升 2-3% 困惑度
4. **计算效率**：虽然增加一定计算开销，但在现代硬件上可以很好优化

**公式对比**：
```python
ReLU(x) = max(0, x)                    # 硬截断
GELU(x) = x * Φ(x)                     # 软截断
SwiGLU(x) = Swish(Wx) ⊙ (Ux)         # 门控+平滑
```

### Q4: Qwen2.5在代码生成方面为何表现突出？

**答案**：
1. **数据组成**：预训练数据中代码占比达到30%，覆盖20+编程语言
2. **数据质量**：使用静态分析工具过滤语法错误，保留高质量代码
3. **多任务训练**：同时训练代码生成、代码解释、调试等多个任务
4. **指令微调**：大量编程相关指令数据，包含不同难度级别
5. **评估驱动**：针对 HumanEval、MBPP 等基准专门优化

### Q5: 如何评估Qwen2.5的多语言能力？存在哪些挑战？

**答案**：

**评估方法**：
1. **机器翻译**：WMT系列基准，BLEU/COMET指标
2. **多语言理解**：XNLI、XCOPA等跨语言推理任务
3. **生成质量**：人工评估不同语言的流畅度和准确性
4. **领域适应**：法律、医学、技术等专业领域的多语言表现

**挑战与解决**：
1. **数据不均衡**：英文数据占优，通过采样策略平衡
2. **语言干扰**：多语言混杂影响生成质量，使用语言标识符
3. **文化差异**：不同文化背景的表达习惯，增加多样化训练数据
4. **评估标准**：缺乏统一的多语言评估框架，采用多维度评估

## 📚 推荐阅读

### 原始论文
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) — Qwen 1.0 架构和训练细节
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Qwen2 全面升级，GQA/SwiGLU/RoPE 的详细实验
- [Qwen2.5: A Party of Foundation Models](https://qwenlm.github.io/blog/qwen2.5/) — Qwen2.5 技术博客，代码/数学能力突破

### 深度解读
- [Qwen 系列技术解读（知乎）](https://zhuanlan.zhihu.com/p/664785124) — 中文社区优质解读 ⭐⭐⭐⭐
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — GQA 原始论文

### 实践资源
- [Qwen 官方 GitHub](https://github.com/QwenLM/Qwen2.5) — 模型代码和推理示例
- [Qwen HuggingFace Collection](https://huggingface.co/Qwen) — 全系列模型权重

## 🔧 落地应用

### 直接可用场景
- **中英双语对话**：Qwen2.5 在 29 种语言上性能均衡，是中文场景最强的开源模型之一
- **代码助手**：Qwen2.5-Coder-32B 在 HumanEval 89.5%，可直接替代部分 Copilot 场景
- **数学推理**：GSM8K 95.2%，适合教育和科研场景的数学问题求解
- **RAG 底座**：Qwen2.5 支持 128K 上下文（YaRN），适合长文档 RAG 场景

### 工程实现要点
- **模型选型**：7B 适合边缘/低成本，72B 适合高质量需求；14B 是性价比最优的"甜点"
- **YaRN 长上下文**：从 32K 扩展到 128K 时，注意 attention entropy 的变化，建议先在目标长度上测试质量
- **vLLM 部署**：Qwen2.5 的 GQA 在 vLLM 中自动优化 KV Cache，72B 模型在 4×A100-80G 上可流畅推理

### 面试高频问法
- Q: Qwen2.5 相比 LLaMA 3.1 的核心优势？
  A: 多语言（29 种 vs 英语为主）、代码（89.5% vs 80.5% HumanEval）、中文场景碾压。架构上都用了 GQA + RoPE，但 Qwen 的预训练数据更多语言均衡。

## 💡 启发与思考

### So What？对老板意味着什么
- **Qwen 是中文场景的首选开源底座**。对于中国团队，Qwen2.5 > LLaMA 3.1 的理由很充分：更好的中文、更好的代码、更好的数学，而且开源程度高（权重+微调代码）
- **GQA + SwiGLU + RoPE 是 2024-2025 的"标准配置"**：Qwen、LLaMA、DeepSeek 都在用，理解这三个组件就理解了当代 LLM 的架构骨架

### 未解问题与局限
- Qwen2.5 的预训练数据配比细节未完全公开（18T tokens 的语言/领域分布不透明）
- 128K 长上下文的实际使用质量仍受 Lost in the Middle 问题影响（参见 [[AI/3-LLM/Architecture/长上下文处理|长上下文处理]]）
- Qwen3 是否会引入 MoE 架构？如果是，与 [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] 的 DeepSeekMoE 会有什么差异？

### 脑暴：如果往下延伸
- Qwen2.5 + [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] 的 GRPO 训练：用 R1 的 RL 方法在 Qwen2.5 上训练推理能力，事实上 R1-Distill-Qwen 系列已经验证了这条路线
- 如果把 Qwen 的多语言能力和 [[AI/3-LLM/Architecture/Mamba-SSM|Mamba]] 的长序列效率结合，能否做出多语言长文档理解的"最优解"？

> 🔗 See also: [[AI/3-LLM/Architecture/架构范式对比|架构范式对比]] — Qwen 的 Decoder-Only 架构在全景中的定位