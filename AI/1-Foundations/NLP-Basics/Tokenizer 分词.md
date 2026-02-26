---
brief: "Tokenizer 分词——BPE/WordPiece/SentencePiece 三大分词算法对比；tokenization 对模型能力的影响（中文 token 效率/数学符号覆盖）；自定义词表扩充的工程实践；面试被问 tokenization 的标准参考。"
title: "Tokenizer 与分词"
date: 2026-02-14
tags:
  - nlp
  - tokenizer
  - bpe
  - interview
type: note
---

# Tokenizer 与分词

## 1. 为什么需要 Tokenizer

神经网络只能处理数字，不能直接处理文本。Tokenizer 是 **文本 → 数字** 的桥梁：

```
"Hello world" → tokenizer → [15496, 995] → embedding → [0.12, -0.34, ...], [0.56, 0.78, ...]
```

### 分词粒度的三种选择

| 粒度 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| **Word-level** | `["Hello", "world"]` | 语义完整 | 词表爆炸，OOV 问题严重 |
| **Character-level** | `["H","e","l","l","o"," ","w","o","r","l","d"]` | 无 OOV，词表小 | 序列太长，丢失语义 |
| **Subword-level** | `["He", "llo", " world"]` | 平衡词表大小与序列长度 | 需要训练分词模型 |

**现代 LLM 全部使用 subword 分词**——在词表大小和序列长度之间取最优折中。

---

## 2. BPE (Byte Pair Encoding)

### 2.1 原理

BPE 的核心是 **自底向上合并**：从最小单元（字符或字节）出发，反复合并出现频率最高的相邻 pair，直到达到目标词表大小。

### 2.2 训练过程

```
语料: "low low low low low lowest lowest newer newer newer wider wider wider"

初始词表: {'l','o','w','e','s','t','n','r','i','d', ...}

Step 1: 统计所有相邻 pair 频率
        ('l','o') → 7, ('o','w') → 7, ('e','r') → 6, ...
Step 2: 合并频率最高的 pair: 'l'+'o' → 'lo'
Step 3: 更新语料，重新统计
Step 4: 合并 'lo'+'w' → 'low'
...
重复直到词表达到目标大小（如 50257）
```

### 2.3 推理（编码）过程

给定一个新词，按照训练时的合并顺序（merge rules）从左到右贪心合并：
```
"lowest" → ['l','o','w','e','s','t'] → ['lo','w','e','s','t'] → ['low','e','s','t'] → ['low','est']
```

### 2.4 GPT 系列的使用

| 模型 | Tokenizer | 词表大小 | 特点 |
|------|-----------|----------|------|
| GPT-2 | BPE | 50,257 | Byte-level BPE，开创性工作 |
| GPT-3 | BPE | 50,257 | 沿用 GPT-2 |
| GPT-4 | cl100k_base | 100,256 | tiktoken 实现，更大词表 |

---

## 3. WordPiece

### 3.1 原理

WordPiece 与 BPE 思路类似，但合并策略不同：

- **BPE**：合并 **出现频率最高** 的 pair
- **WordPiece**：合并使 **语言模型似然增加最大** 的 pair

合并评分公式：
$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

这本质上是 **互信息**（Pointwise Mutual Information）的一种近似——优先合并那些"总是一起出现"的 pair，而非仅仅频率高的 pair。

### 3.2 与 BPE 的关键区别

| 特性 | BPE | WordPiece |
|------|-----|-----------|
| **合并标准** | 频率最高 | 似然增益最大（互信息） |
| **子词前缀** | 无特殊标记 | 非首子词加 `##` 前缀 |
| **编码方向** | 自底向上合并 | 最长匹配优先（贪心） |
| **代表模型** | GPT 系列 | BERT, DistilBERT |

### 3.3 WordPiece 编码示例

```
"unaffable" → ["un", "##aff", "##able"]

# "##" 表示该子词不是词的开头，用于还原完整单词
```

---

## 4. SentencePiece

### 4.1 定位

SentencePiece 不是一种新的分词算法，而是一个 **与语言无关的分词框架**，特点：

1. **直接处理原始文本**：不需要预先分词（pre-tokenization），将空格视为普通字符（用 `▁` 表示）
2. **支持多种算法**：BPE 和 Unigram LM 都可以作为后端
3. **完全可逆**：`decode(encode(text)) == text`，无信息丢失

### 4.2 Unigram LM 算法

与 BPE 的自底向上合并相反，Unigram 采用 **自顶向下裁剪**：

```
训练过程：
1. 初始化一个很大的候选词表（所有子串 + 字符）
2. 为每个子词赋予概率（基于 EM 算法）
3. 计算去掉每个子词后对语料似然的损失
4. 移除损失最小的子词（保留损失大的 = 重要的）
5. 重复直到达到目标词表大小
```

编码时使用 **Viterbi 算法** 找到使总概率最大的分词方式：
$$\arg\max_{(x_1,...,x_n)} \prod_{i=1}^{n} P(x_i)$$

### 4.3 BPE vs Unigram 对比

| 特性 | BPE | Unigram LM |
|------|-----|------------|
| **构建方向** | 自底向上合并 | 自顶向下裁剪 |
| **编码确定性** | 确定性（贪心合并） | 可采样（概率分词） |
| **训练速度** | 较快 | 较慢（需要 EM 迭代） |
| **可采样性** | ❌ | ✅ subword regularization |

### 4.4 使用 SentencePiece 的模型

| 模型 | 后端算法 | 词表大小 | 特点 |
|------|----------|----------|------|
| LLaMA / LLaMA 2 | BPE (SentencePiece) | 32,000 | 中文支持有限 |
| LLaMA 3 | BPE (tiktoken) | 128,256 | 切换到 tiktoken，大幅扩充词表 |
| Qwen / Qwen2 | BPE (tiktoken) | 151,643 | 大词表，中英文均衡 |
| T5 | Unigram (SentencePiece) | 32,000 | Unigram LM 的经典应用 |
| XLNet | Unigram (SentencePiece) | 32,000 | — |

---

## 5. Byte-level BPE vs Character-level

### 5.1 Character-level BPE

- 初始词表 = 所有 Unicode 字符
- **问题**：Unicode 有 14 万+ 字符，基础词表太大；且稀有字符（如 emoji、罕见汉字）可能不在训练集中 → OOV

### 5.2 Byte-level BPE（GPT-2 开创）

- 初始词表 = **256 个字节**（0x00 - 0xFF）
- 任何 UTF-8 文本都可以分解为字节序列 → **彻底消除 OOV**
- GPT-2 将字节映射为可打印字符（如 `Ġ` 表示空格开头的字节 `0x20`）

### 5.3 多语言处理的影响

| 方面 | Character-level | Byte-level |
|------|----------------|------------|
| **OOV** | 可能出现 | 完全不可能 |
| **中文** | 每个字 = 1 token（基础词表大） | 每个字 = 2-3 bytes → 需要 BPE 合并 |
| **Emoji/特殊符号** | 可能 OOV | 总是可编码 |
| **词表效率** | 基础词表大 | 基础词表仅 256，高效 |

> **面试要点**：Byte-level BPE 是目前最通用的方案，牺牲了一点中文等多字节语言的编码效率（一个汉字需要多个 byte token 才能合并），但换来了零 OOV 的鲁棒性。

---

## 6. Vocabulary Size 的选择与影响

### 6.1 常见选择

| 词表大小 | 代表模型 | 特点 |
|----------|----------|------|
| ~32K | LLaMA 1/2, T5, BERT | 经典选择，英文为主 |
| ~50K | GPT-2/3 | 英文 Byte-level BPE |
| ~64K | BLOOM, Falcon | 多语言场景 |
| ~100K | GPT-4 (cl100k) | 更好的多语言 + 代码支持 |
| ~128K | LLaMA 3 | 大词表趋势 |
| ~152K | Qwen 2 | 中英文均衡优化 |
| ~256K | Gemini | 极大词表，覆盖更多语言 |

### 6.2 词表大小的 Trade-off

**词表越大：**
- ✅ 每个 token 承载更多信息 → 序列更短 → 推理更快（fewer decoding steps）
- ✅ 常见词/短语被完整编码 → 减少碎片化
- ✅ 多语言覆盖更好（中文、日文等不需要拆成字节）
- ❌ Embedding 层参数量 = `vocab_size × d_model`，词表翻倍 → embedding 参数翻倍
- ❌ Softmax 计算量 = `O(vocab_size)`，每步解码更慢
- ❌ 低频 token 训练不充分，embedding 质量差

**词表越小：**
- ✅ Embedding 层参数少，模型更紧凑
- ✅ 每个 token 都能充分训练
- ❌ 序列更长 → 更多 decoding steps → 推理更慢
- ❌ 多语言场景下非英文语言效率低

### 6.3 经验法则

> **关键洞察**：对于 7B+ 的大模型，embedding 层参数占比很小（<5%），词表从 32K 扩到 128K 对总参数量影响不大，但对 **推理速度**（序列更短）和 **多语言能力** 有显著提升。这就是为什么 2024 年后的模型纷纷采用大词表（100K+）。

---

## 7. tiktoken 等现代实现

### 7.1 tiktoken（OpenAI）

- 用 **Rust** 实现核心编码逻辑，Python 绑定
- 速度比 HuggingFace tokenizers 快 3-6x
- 支持 GPT 系列的所有编码方案（`gpt2`, `r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`）

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, 世界!")
# [9906, 11, 220, 33176, 6447, 0]
print(enc.decode(tokens))
# "Hello, 世界!"
```

### 7.2 HuggingFace Tokenizers

- Rust 核心 + Python/Node 绑定
- 支持 BPE、WordPiece、Unigram 所有算法
- 提供完整的训练 + 编码 pipeline
- 与 `transformers` 库深度集成

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokens = tokenizer.encode("Hello, 世界!")
```

### 7.3 SentencePiece（Google）

- C++ 实现，Python 绑定
- 直接操作原始文本（无需 pre-tokenization）
- 训练和推理使用同一个 `.model` 文件

### 7.4 性能对比

| 实现 | 语言 | 速度（相对） | 特点 |
|------|------|-------------|------|
| tiktoken | Rust + Python | ⭐⭐⭐⭐⭐ | 最快，仅编码 |
| HF Tokenizers | Rust + Python | ⭐⭐⭐⭐ | 功能全面，训练+编码 |
| SentencePiece | C++ + Python | ⭐⭐⭐ | 语言无关，支持 Unigram |

---

## 8. 分词对模型训练和推理的影响

### 8.1 Tokenization Fertility（生育率）

**Fertility** = 平均每个词被分成多少个 token。不同语言差异巨大：

| 语言 | 英文 LLaMA-2 (32K) | Qwen (152K) |
|------|-------------------|-------------|
| 英文 | ~1.3 | ~1.2 |
| 中文 | ~2.5（拆字节） | ~1.1（大量中文 token） |
| 日文 | ~3.0 | ~1.5 |

> 这意味着同样的中文文本，用 LLaMA-2 tokenizer 编码后序列长度是 Qwen 的 2 倍以上 → 推理成本翻倍。

### 8.2 对上下文窗口的影响

模型的上下文窗口是按 **token 数** 计的（如 4K, 8K, 128K）。分词效率直接影响同一窗口能容纳多少"真实信息"。词表大、分词效率高的模型在实际使用中等效上下文更长。

---

## 9. 面试常见问题及回答要点

### Q1: BPE 的核心思想是什么？
> **答**：BPE 从最小单元（字符或字节）出发，统计语料中所有相邻 pair 的频率，反复合并频率最高的 pair，直到达到目标词表大小。编码时按训练得到的合并规则从左到右贪心合并。核心优点是自动在字符和单词之间找到最优的 subword 粒度。

### Q2: BPE 和 WordPiece 有什么区别？
> **答**：合并策略不同。BPE 合并出现频率最高的 pair；WordPiece 合并使语言模型似然增益最大的 pair（近似互信息）。WordPiece 更倾向于合并"专属搭配"而非仅仅高频 pair。实际效果差异不大，BPE 更主流。

### Q3: 为什么 LLaMA 3 把词表从 32K 扩到 128K？
> **答**：三个原因。（1）更好的多语言支持——中文等多字节语言不再需要拆成字节级别，编码效率大幅提升；（2）序列更短——同样的文本用更少的 token 表示，推理时 decoding steps 更少，速度更快；（3）对 7B+ 模型来说，embedding 参数占比很小，词表扩大的参数开销可以忽略。

### Q4: 什么是 Byte-level BPE？为什么能消除 OOV？
> **答**：Byte-level BPE 以 256 个字节（而非 Unicode 字符）作为初始词表。由于任何文本都可以被编码为 UTF-8 字节序列，所以不存在 OOV 的可能。GPT-2 首先采用这种方案，后来成为主流。

### Q5: SentencePiece 的 Unigram 模型和 BPE 有什么本质区别？
> **答**：构建方向相反。BPE 自底向上合并——从小单元开始不断合并成大单元；Unigram 自顶向下裁剪——从大词表开始不断删除不重要的子词。编码时 BPE 是确定性的（贪心合并），Unigram 可以采样多种分词方式（subword regularization），这可以作为一种数据增强手段。

### Q6: 词表大小如何选择？有什么 trade-off？
> **答**：核心 trade-off 是 **序列长度 vs embedding 参数量**。大词表使序列更短（推理更快、等效上下文更长），但 embedding 层更大。对于大模型（7B+），embedding 占比很小，所以趋势是用更大的词表（100K+）。对于小模型（<1B），大词表会使 embedding 参数占比过高，反而不划算。

### Q7: tokenizer 对模型效果有多大影响？
> **答**：影响非常大但容易被忽视。（1）分词粒度决定了模型"看"文本的方式——拆得太碎会丢失语义，拆得太粗会导致 OOV；（2）分词效率决定了有效上下文长度；（3）不同语言的分词公平性直接影响多语言能力。一个好的 tokenizer 能让中文的 token 效率从 2.5 token/字提升到 1.1 token/字，等效于上下文窗口翻倍。

---

## See Also

- [[Tokenizer 深度理解|Tokenizer 深度理解]] — 深度版：BPE/WordPiece/SentencePiece 内部机制
- [[Transformer|Transformer 通识]] — Tokenizer 在 Transformer pipeline 中的位置
- [[数据预处理|数据预处理]] — Tokenization 是文本数据预处理的核心步骤
- [[AI/1-Foundations/目录|Foundations MOC]] — NLP 基础全图谱
