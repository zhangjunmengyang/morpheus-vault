---
title: "Decoder-Only vs Encoder-Decoder vs Prefix LM"
date: 2026-02-14
tags: [architecture, transformer, gpt, t5, interview]
type: note
---

# Decoder-Only vs Encoder-Decoder vs Prefix LM

## 1. 三种架构定义与核心区别

三种架构的**本质区别在于注意力掩码（attention mask）**：

### 架构概览

```
Encoder-Decoder:
  Encoder: [x1, x2, x3] → 双向注意力（所有 token 互相看见）
  Decoder: [y1, y2, y3] → 因果注意力 + cross-attention 到 encoder 输出

Decoder-Only:
  [x1, x2, x3, y1, y2, y3] → 严格因果注意力（每个 token 只看左边）

Prefix LM:
  [x1, x2, x3 | y1, y2, y3] → prefix 部分双向 + 后续部分因果
       ^双向^    ^因果^
```

### Attention Mask 对比（核心！）

```
Encoder-Decoder (Encoder 部分):    Decoder-Only:              Prefix LM:
  x1 x2 x3                          x1 x2 x3 y1 y2 y3         x1 x2 x3 y1 y2 y3
x1 ✓  ✓  ✓                       x1 ✓  ✗  ✗  ✗  ✗  ✗       x1 ✓  ✓  ✓  ✗  ✗  ✗
x2 ✓  ✓  ✓                       x2 ✓  ✓  ✗  ✗  ✗  ✗       x2 ✓  ✓  ✓  ✗  ✗  ✗
x3 ✓  ✓  ✓                       x3 ✓  ✓  ✓  ✗  ✗  ✗       x3 ✓  ✓  ✓  ✗  ✗  ✗
                                  y1 ✓  ✓  ✓  ✓  ✗  ✗       y1 ✓  ✓  ✓  ✓  ✗  ✗
                                  y2 ✓  ✓  ✓  ✓  ✓  ✗       y2 ✓  ✓  ✓  ✓  ✓  ✗
                                  y3 ✓  ✓  ✓  ✓  ✓  ✓       y3 ✓  ✓  ✓  ✓  ✓  ✓
```

关键观察：
- **Encoder-Decoder**：输入端完全双向，输出端因果，中间通过 cross-attention 桥接
- **Decoder-Only**：全程因果掩码，输入和输出没有结构性区分
- **Prefix LM**：Decoder-Only 的变体，只是 prefix 部分放开为双向注意力

---

## 2. Encoder-Decoder 架构

### 2.1 结构特点

- **两个独立的 Transformer stack**：Encoder 和 Decoder 参数不共享
- **Encoder**：双向自注意力，每个 token 可以看到全部输入
- **Decoder**：因果自注意力 + cross-attention（attend to encoder 输出）
- **信息流**：输入 → Encoder → 隐状态序列 → Decoder cross-attend → 输出

### 2.2 代表模型

**T5（Text-to-Text Transfer Transformer）**：
- Google, 2019。统一所有 NLP 任务为 text-to-text 格式
- 预训练目标：Span Corruption（随机遮盖连续 span，预测被遮盖的内容）
- 规模：T5-Small (60M) → T5-XXL (11B)
- 关键洞见：通过统一的输入输出格式，一个模型解决所有任务

**BART（Bidirectional and Auto-Regressive Transformer）**：
- Meta, 2019。结合 BERT 的双向编码和 GPT 的自回归解码
- 预训练目标：去噪自编码（文本填充、句子排列、token 删除等多种噪声）
- 特别擅长文本生成和摘要任务

**Flan-T5**：
- Google, 2022。T5 经过大规模指令微调
- 在 1800+ 任务上进行 instruction tuning
- 证明了指令微调可以大幅提升 Encoder-Decoder 模型的零样本能力

### 2.3 优势与局限

**优势**：
- 输入端双向编码，对输入的理解更充分
- 天然适合 sequence-to-sequence 任务（翻译、摘要）
- 对输入和输出有明确的结构划分

**局限**：
- 参数量分布在两个 stack，同等参数量下不如 Decoder-Only 的 scaling 效率
- 预训练目标（span corruption）与下游生成任务有 gap
- Cross-attention 增加了架构复杂度，KV cache 管理更复杂

---

## 3. Decoder-Only 架构

### 3.1 结构特点

- **单一 Transformer stack**，使用严格的因果注意力掩码
- 输入和输出在同一个序列中，没有显式的编码器-解码器边界
- 训练目标：Next Token Prediction（自回归语言建模）
- 简洁统一：所有任务都转化为"给定前文，生成后文"

### 3.2 代表模型

**GPT 系列（OpenAI）**：
- GPT-1 (2018): 117M，验证了"预训练 + 微调"范式
- GPT-2 (2019): 1.5B，展示了零样本能力
- GPT-3 (2020): 175B，in-context learning 的里程碑
- GPT-4 (2023): 据传 MoE 架构，多模态

**LLaMA 系列（Meta）**：
- LLaMA-1 (2023): 7B-65B，开源，验证了"更多数据 + 较小模型"的优越性
- LLaMA-2 (2023): 7B-70B，加入 RLHF 对齐
- LLaMA-3 (2024): 8B-405B，15T tokens 训练，性能飞跃

**其他重要模型**：
- Chinchilla (DeepMind): 70B，提出了最优的数据-参数比例（Chinchilla Scaling Law）
- Mistral / Mixtral: 高效的 MoE Decoder-Only 架构
- Qwen、DeepSeek、Yi: 中国开源 LLM，全部采用 Decoder-Only

### 3.3 为什么 Decoder-Only 成为主流

这是面试高频问题，需要从多个角度解释：

**① Scaling Law 更优**：
- Decoder-Only 的所有参数都在同一个 stack 中，参数利用效率更高
- Encoder-Decoder 中 encoder 和 decoder 的参数分配比例是额外的超参数
- 实验证明：同等计算预算下，Decoder-Only 的 scaling 更 smooth、更可预测

**② 训练目标简洁统一**：
- Next Token Prediction 是最简单、最通用的目标
- 不需要设计复杂的预训练任务（T5 的 span corruption 需要调 span 长度、遮盖比例等）
- 一个目标函数覆盖所有能力

**③ 工程实现简单**：
- 只有一种注意力模式（因果掩码），KV cache 管理简单
- 不需要 cross-attention，推理 pipeline 更直接
- 并行化和分布式训练更容易实现和调优

**④ In-Context Learning 能力**：
- GPT-3 发现的 emergent ability，Decoder-Only 架构在这方面表现最好
- 输入和输出在同一序列中，few-shot 示例可以自然拼接
- Encoder-Decoder 的 in-context learning 能力相对较弱

**⑤ 涌现能力（Emergent Abilities）**：
- Chain-of-Thought、指令遵循等能力在大规模 Decoder-Only 中更显著
- 可能与因果建模天然适配自回归推理有关

**⑥ 生态效应**：
- GPT-3 的成功带来了大量后续工作，形成正反馈
- 开源生态（LLaMA 系列）进一步巩固了 Decoder-Only 的主导地位
- 工具链（vLLM、TGI）高度优化了 Decoder-Only 推理

---

## 4. Prefix LM

### 4.1 结构特点

- 形式上是 Decoder-Only（单一 stack）
- 但将输入序列分为 **prefix**（双向注意力）和 **generation**（因果注意力）两部分
- Prefix 部分的 token 可以互相看见，类似 Encoder
- Generation 部分保持因果掩码，类似 Decoder

### 4.2 代表模型

**U-PaLM**：
- Google, 2022。PaLM 的 Prefix LM 变体
- 在预训练后期加入 UL2（Unifying Language Learning）混合去噪目标
- 结合了 Decoder-Only 的 scaling 优势和双向编码的理解能力

**GLM（General Language Model）**：
- 清华, 2022。早期版本使用自回归空白填充 + 双向注意力
- GLM-130B 部分采用了 Prefix LM 的思路
- 后续的 ChatGLM 系列逐渐转向标准 Decoder-Only

**UniLM**：
- Microsoft, 2019。同时支持双向、单向和 seq2seq 注意力
- 通过不同的 attention mask 实现多种模式
- 可看作 Prefix LM 的早期探索

### 4.3 优势与局限

**优势**：
- 保留了 Decoder-Only 的架构简洁性（单一 stack）
- Prefix 部分的双向注意力对输入理解更好
- 在 NLU 任务和条件生成任务上可能有优势

**局限**：
- 需要预先确定 prefix 的边界（训练时需要标注哪些是输入、哪些是输出）
- KV cache 中 prefix 部分不能增量计算（需要一次性全部算完）
- 预训练时需要混合因果和双向目标，增加了训练复杂度
- 实验证明，在足够大的规模下，Decoder-Only 的因果掩码并不会劣于双向注意力

---

## 5. 各架构适用场景

| 任务类型 | 最适合架构 | 理由 |
|---------|----------|------|
| 开放式文本生成 | Decoder-Only | 因果建模天然适配自回归生成 |
| 对话 / Chat | Decoder-Only | 多轮对话是连续的序列延伸 |
| 机器翻译 | Encoder-Decoder | 输入输出长度差异大，双向编码理解源语言更好 |
| 文本摘要 | Encoder-Decoder / Decoder-Only | 传统上 Enc-Dec 更好，但大规模 Dec-Only 也很强 |
| 文本分类 / NLU | Encoder-Only (BERT) / Prefix LM | 双向编码对理解任务有天然优势 |
| 代码生成 | Decoder-Only | 代码本质是序列生成，因果建模更合适 |
| 通用指令遵循 | Decoder-Only | 最灵活，一个模型覆盖所有任务 |
| 语音识别 (ASR) | Encoder-Decoder (Whisper) | 输入（音频）和输出（文本）模态不同 |

---

## 6. 代表模型对比表

| 模型 | 架构 | 参数量 | 预训练目标 | 关键特点 |
|------|------|-------|----------|---------|
| GPT-3 | Decoder-Only | 175B | Next Token Prediction | 开创 In-Context Learning |
| GPT-4 | Decoder-Only (MoE?) | 未公开 | NTP + RLHF | 多模态、最强通用能力 |
| LLaMA-3 | Decoder-Only | 8B/70B/405B | NTP (15T tokens) | 开源最强、高质量数据 |
| T5 | Encoder-Decoder | 60M~11B | Span Corruption | 统一 text-to-text 框架 |
| Flan-T5 | Encoder-Decoder | 80M~11B | Span Corruption + Instruction FT | 1800+ 任务指令微调 |
| BART | Encoder-Decoder | 140M/400M | 去噪自编码 | 擅长摘要和文本生成 |
| mBART | Encoder-Decoder | 680M | 多语言去噪 | 多语言翻译 |
| PaLM | Decoder-Only | 540B | NTP | Google 最大单体 LLM |
| U-PaLM | Prefix LM | 540B | NTP + UL2 混合 | 结合双向和因果优势 |
| GLM-130B | Prefix LM → Dec-Only | 130B | 自回归空白填充 | 中英双语 |
| Whisper | Encoder-Decoder | 1.5B | 音频→文本 | 语音识别 SOTA |
| Mistral 7B | Decoder-Only | 7B | NTP | Sliding Window Attention |
| Mixtral 8x7B | Decoder-Only (MoE) | 46.7B (12.9B active) | NTP | 稀疏 MoE，高效推理 |
| DeepSeek-V3 | Decoder-Only (MoE) | 671B (37B active) | NTP | MLA + DeepSeekMoE |
| Qwen-2.5 | Decoder-Only | 0.5B~72B | NTP | 中文最强开源系列之一 |

---

## 7. 面试常见问题及回答要点

### Q1: Decoder-Only 和 Encoder-Decoder 的核心区别是什么？

**回答要点**：
- **最本质的区别是 attention mask**：
  - Encoder-Decoder 的 encoder 用双向注意力，decoder 用因果注意力 + cross-attention
  - Decoder-Only 全程使用因果注意力掩码
- **参数组织**：Encoder-Decoder 有两个独立的参数 stack；Decoder-Only 只有一个
- **输入输出关系**：Encoder-Decoder 显式分离输入和输出；Decoder-Only 将两者拼接在同一序列中
- **信息传递**：Encoder-Decoder 通过 cross-attention 传递；Decoder-Only 通过序列内的因果注意力隐式传递

### Q2: 为什么 Decoder-Only 架构成为了当前 LLM 的主流？

**回答要点**（从重要到次要）：
1. **Scaling 效率更高**：所有参数集中在一个 stack，避免了 encoder/decoder 参数分配的问题
2. **训练目标简洁**：Next Token Prediction 一个损失函数覆盖所有，不需要设计复杂的预训练目标
3. **工程简单**：KV cache 管理、分布式并行、推理优化都更容易
4. **In-Context Learning**：因果建模天然适配 few-shot 场景
5. **生态正反馈**：GPT-3 成功 → 后续研究都跟进 → 工具链成熟 → 更多人采用

补充：并不是说 Encoder-Decoder 不好，而是在 LLM 的超大规模下，Decoder-Only 的优势被放大了。

### Q3: Encoder-Decoder 的双向编码不是更好吗？为什么 Decoder-Only 不需要？

**回答要点**：
- 直觉上，双向编码应该对输入理解更好（BERT 在 NLU 上确实优于 GPT）
- 但在**足够大的规模**下，因果注意力的"劣势"几乎消失：
  - 虽然每个 token 只能看左边，但前面的 token 已经编码了之前的全部信息
  - 大模型有足够的容量来补偿单向注意力的信息损失
- **Wang et al. (2022)** 的研究表明，在相同规模和数据下，Decoder-Only 和 Prefix LM 的效果差异很小
- 实际上，因果掩码还有好处：训练时每个 token 都产生有效的梯度信号（而双向模型只有被遮盖的 token 产生损失）

### Q4: Prefix LM 有什么优势？为什么没有成为主流？

**回答要点**：
- **优势**：兼顾了双向编码（对输入的理解）和自回归生成（输出的质量），同时保持单一 stack 的简洁性
- **没有成为主流的原因**：
  1. 需要预先定义 prefix 边界，在预训练时不方便（纯文本语料没有天然的输入/输出分界）
  2. 实验表明在大规模下，相比标准 Decoder-Only 的提升不显著
  3. 工程上增加了复杂度（prefix 部分的 KV cache 不能增量计算）
  4. 社区和生态已经围绕 Decoder-Only 构建，切换成本高

### Q5: 如果现在要训练一个专门做翻译的模型，你会选什么架构？

**回答要点**：
- **如果数据量有限、模型不大（<10B）**：选 Encoder-Decoder
  - 翻译是典型的 seq2seq 任务，encoder 的双向编码对源语言理解更好
  - Cross-attention 显式对齐源语言和目标语言
  - T5、mBART 在翻译上的效果已充分验证
- **如果追求极致性能且有足够资源**：大规模 Decoder-Only + 大量平行语料
  - GPT-4 级别的模型翻译能力已超过专门的翻译模型
  - 但这是靠规模堆出来的，性价比不如 Encoder-Decoder
- **工业实践**：Google Translate 从 LSTM-based Encoder-Decoder → Transformer Encoder-Decoder，仍然是 Encoder-Decoder 架构
- 总结：**任务特化选 Encoder-Decoder，通用能力选 Decoder-Only**
