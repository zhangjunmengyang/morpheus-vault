---
title: "Tokenizer 深度理解：BPE / WordPiece / SentencePiece / Tiktoken"
brief: "深度拆解主流 Tokenizer 算法：BPE（GPT系列，频率驱动合并）/ WordPiece（BERT，最大化序列概率）/ SentencePiece（语言无关，字符级预分词）/ Tiktoken（GPT-4高效实现）。面试热点：词表大小对模型的影响、多语言 tokenizer 设计、OOV 处理策略。"
date: 2026-02-13
tags:
  - ai/llm/architecture
  - ai/nlp
  - ai/tokenizer
  - type/concept
  - interview/hot
status: active
---

# Tokenizer 深度理解：BPE / WordPiece / SentencePiece / Tiktoken

> Tokenizer 是 LLM 的"感官器官"——它决定了模型看到什么、怎么看、能看多远

## 1. 为什么子词切分是主流？

```
切分粒度的困境:
  字符级 → 序列太长，计算爆炸，难以捕捉语义
  词级   → 词表太大 (英语 50 万+词)，OOV 问题严重
  子词级 → 平衡：常见词保持完整，罕见词拆为子词片段

示例 ("unhappiness"):
  词级:     ["unhappiness"]          → 低频词，embedding 学不好
  字符级:   ["u","n","h","a","p",...] → 13 个 token，太长
  子词(BPE): ["un", "happiness"]     → 2 个常见子词，语义丰富
```

## 2. BPE（Byte Pair Encoding）

### 原理

最初是数据压缩算法，Sennrich et al. (2016) 引入 NLP。

**训练过程**：
1. 初始化词表为所有单字符（+ 特殊 token）
2. 统计所有相邻 token pair 的频率
3. 将频率最高的 pair 合并为新 token，加入词表
4. 重复步骤 2-3，直到词表大小达到目标

```python
# BPE 训练简化实现
def train_bpe(corpus: list[str], vocab_size: int):
    """BPE 训练过程"""
    # 初始化：字符级 token
    # 假设 corpus = ["low", "lower", "newest", "widest"]
    # 初始词表：{'l','o','w','e','r','n','s','t','i','d'}
    
    vocab = set(char for word in corpus for char in word)
    # 将每个词表示为字符序列 + 词尾标记
    words = {tuple(word) + ('</w>',): freq for word, freq in word_counts.items()}
    
    while len(vocab) < vocab_size:
        # 统计所有相邻 pair 频率
        pairs = count_pairs(words)
        if not pairs:
            break
        
        # 找到最高频 pair
        best_pair = max(pairs, key=pairs.get)
        
        # 合并这个 pair
        words = merge_pair(words, best_pair)
        vocab.add(''.join(best_pair))
    
    return vocab

# 合并过程示例:
# 迭代1: ('e','s') → "es"  (频率最高)
# 迭代2: ('es','t') → "est"
# 迭代3: ('l','o') → "lo"
# 迭代4: ('lo','w') → "low"
# ...
```

**推理（编码）过程**：
1. 将输入文本 pre-tokenize（按空格/标点分词）
2. 将每个词拆为字符序列
3. 按训练时的合并顺序，贪心地合并 pair

### GPT 系列的 BPE

GPT-2/3/4 使用 **Byte-level BPE**：
- 初始词表 = 256 个 byte value（而非 Unicode 字符）
- 优点：**零 OOV**——任何 byte 序列都能编码
- 缺点：非英语文本每个字符可能被拆为多个 byte token

## 3. WordPiece

### 原理

Google 为 BERT 开发（Schuster & Nakajima, 2012）。

与 BPE 的关键区别——**合并策略不同**：

```
BPE:       合并频率最高的 pair
WordPiece: 合并使语言模型似然提升最大的 pair

Score(a, b) = freq(ab) / (freq(a) × freq(b))

直觉：不只看共现频率，还考虑 pair 的"互信息"
  → 倾向于合并语义紧密的组合
```

### BERT 的 WordPiece

```python
# BERT tokenizer 示例
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unhappiness")
# → ['un', '##happiness']
# "##" 前缀表示这不是词首 (continuation token)
```

特点：
- 使用 `##` 前缀标记非词首子词
- 词表大小 30,522（BERT-base）
- 先做 lowercasing + accent stripping 等预处理

## 4. SentencePiece

### 解决的问题

BPE 和 WordPiece 都依赖 **pre-tokenization**（先用空格/规则分词），这导致：
- 语言依赖：中文、日文没有空格，需要专门分词器
- 不可逆：分词后丢失了空格信息

### 核心创新

Kudo & Richardson, 2018：

```
SentencePiece 的关键设计:
  1. 将空格替换为特殊符号 ▁ (Unicode 0x2581)
  2. 将整个句子（含"空格"）视为一个连续字符流
  3. 在这个字符流上训练 BPE 或 Unigram 模型

示例:
  输入: "New York is big"
  → "▁New▁York▁is▁big"  (空格变为▁前缀)
  → ["▁New", "▁York", "▁is", "▁big"]  或被进一步切分

优点：
  - 语言无关：中英日韩统一处理
  - 完全可逆：token → 原始文本 无损
  - 直接从原始文本训练，不需要预处理
```

### Unigram 模型（SentencePiece 的另一种算法）

```
与 BPE 相反的方向:
  BPE:     从小词表开始，逐步合并 (bottom-up)
  Unigram: 从大词表开始，逐步删除 (top-down)

过程:
  1. 初始化一个很大的候选词表
  2. 为每个子词计算 unigram 概率
  3. 找到 Viterbi 最优切分
  4. 删除使 likelihood 下降最小的子词
  5. 重复 3-4 直到词表缩小到目标大小

优点: 支持多种切分方式的概率采样
     → 可以做 subword regularization（数据增强）
```

代表模型：T5、LLaMA 1/2、ALBERT

## 5. Tiktoken

### OpenAI 的高性能 BPE 实现

```python
import tiktoken

# GPT-4 / GPT-4o 的 tokenizer
enc = tiktoken.encoding_for_model("gpt-4o")

# 编码
tokens = enc.encode("Hello, world! 你好世界")
print(tokens)       # [13225, 11, 2375, 0, 220, 57668, 53901, 3574]
print(len(tokens))  # 8 tokens

# 解码
text = enc.decode(tokens)
print(text)          # "Hello, world! 你好世界"

# 查看各编码的词表大小
for model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
    enc = tiktoken.encoding_for_model(model)
    print(f"{model}: vocab_size = {enc.n_vocab}")
# gpt-3.5-turbo: 100,277 (cl100k_base)
# gpt-4:         100,277 (cl100k_base)
# gpt-4o:        200,019 (o200k_base)
```

### Tiktoken vs SentencePiece

```
维度            Tiktoken            SentencePiece
──────────────────────────────────────────────────
算法            Byte-level BPE      BPE 或 Unigram
实现语言        Rust (Python绑定)   C++ (Python绑定)
速度            极快 (3-6x faster)  快
pre-tokenize    正则表达式 split    无（直接处理原始文本）
空格处理        编码为 byte         替换为 ▁
可训练          ❌ 只提供编码/解码   ✅ 可自定义训练
代表模型        GPT-4, LLaMA 3     T5, LLaMA 1/2
──────────────────────────────────────────────────
```

LLaMA 3 从 SentencePiece 切换到 Tiktoken，词表从 32K 扩大到 128K。

## 6. Vocab Size 选择

```
模型          Vocab Size    特点
───────────────────────────────────
GPT-2         50,257       英语为主
BERT          30,522       英语，WordPiece
T5            32,000       SentencePiece Unigram
LLaMA 1/2     32,000       SentencePiece BPE
LLaMA 3       128,256      Tiktoken，大幅增加多语言
GPT-4o        200,019      最大词表
Qwen2.5       152,064      含大量中文 token
DeepSeek-V3   129,280      平衡中英
───────────────────────────────────

Trade-off:
  大词表: ✅ 更短序列 → 推理快、上下文利用率高
          ✅ 更好的多语言支持
          ❌ Embedding 层参数量大 (vocab × hidden_dim)
          ❌ Softmax 计算量增加
          
  小词表: ✅ Embedding 参数少
          ❌ 序列更长 → 推理慢
          ❌ 非英语效率低（中文一个字可能 3-4 个 token）

经验法则:
  - 英语为主: 32K-50K 足够
  - 多语言: 100K-200K
  - 词表大小通常取 128 的倍数 (GPU 对齐优化)
```

## 7. 多语言 Tokenizer 的挑战

```
问题: 中文 "机器学习" 在不同 tokenizer 下的 token 数

GPT-2 (50K, 英语BPE):     ~8 tokens  (每个汉字拆为 3 个 UTF-8 bytes)
LLaMA-2 (32K):             ~4 tokens  (部分常见汉字是独立 token)
Qwen2.5 (152K):            ~2 tokens  ("机器" + "学习" 各为一个 token)
GPT-4o (200K):             ~2 tokens  (大词表优化)

影响:
  - 同样 4K 上下文，Qwen 能处理的中文内容是 GPT-2 的 4 倍
  - Token 效率直接影响推理成本和速度
  - 这也是为什么中文模型需要扩词表的原因
```

### 词表扩展实践

```python
# 在已有模型基础上扩展中文 token
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
print(f"Original vocab: {len(tokenizer)}")  # 32000

# 添加新 token
chinese_tokens = ["机器学习", "深度学习", "自然语言处理", ...]
tokenizer.add_tokens(chinese_tokens)
print(f"Extended vocab: {len(tokenizer)}")  # 32000 + len(chinese_tokens)

# 注意：扩词表后需要 resize model embedding
model.resize_token_embeddings(len(tokenizer))
# 新 token 的 embedding 随机初始化，需要继续预训练 (CPT)
```

## 8. 面试高频题

### Q1: BPE 和 WordPiece 的核心区别？
**答**：合并策略不同。BPE 合并**频率最高**的相邻 pair，是纯统计方法；WordPiece 合并**使语言模型似然提升最大**的 pair，等价于选择互信息最高的 pair（score = freq(ab) / freq(a)×freq(b)）。直觉上，BPE 倾向合并共现频繁的组合，WordPiece 倾向合并语义关联紧密的组合。实际效果差异不大。

### Q2: SentencePiece 解决了什么问题？
**答**：传统 BPE/WordPiece 依赖 pre-tokenization（用空格/标点先分词），这对中日韩等无空格语言不友好，且分词结果不可逆。SentencePiece 将空格视为普通字符（替换为 ▁），直接在原始文本上训练子词模型，实现了**语言无关 + 完全可逆**的 tokenization。它支持 BPE 和 Unigram 两种算法。

### Q3: 为什么 LLaMA 3 从 SentencePiece 切换到 Tiktoken？
**答**：主要原因：(1) Tiktoken 基于 Rust 实现，编码速度快 3-6 倍；(2) LLaMA 3 将词表从 32K 扩大到 128K 以支持多语言，Tiktoken 的 Byte-level BPE 配合正则表达式 pre-tokenization 在大词表下效率更高；(3) Tiktoken 的 byte-level 设计保证零 OOV，与大词表的多语言需求匹配。

### Q4: Vocab size 对模型性能和效率有什么影响？
**答**：大词表：(1) 序列更短 → 推理速度快、上下文窗口能装更多内容；(2) Embedding 层参数增加（vocab_size × hidden_dim），对小模型影响显著（如 7B 模型，128K 词表的 embedding 占 ~1.7GB）；(3) 输出层 softmax 计算量增加。小词表反之。经验上，100K-200K 是多语言 LLM 的甜点区间。词表大小通常取 128 的倍数以利用 GPU tensor core 对齐。

### Q5: 给定一个英文为主的 LLM，如何高效扩展中文能力？
**答**：两步走：(1) **扩词表**——在原有 tokenizer 基础上训练中文子词并添加，将中文 token 效率提升 2-4 倍。关键是合理选择添加的 token 数量（通常 10K-30K 中文 token）。(2) **继续预训练 (CPT)**——新 token 的 embedding 是随机初始化的，必须在中文语料上做 CPT 让模型学会新 token 的表示。数据配比建议中文:英文 = 7:3 到 8:2，防止英文能力退化。之后可以做中文 SFT 进一步对齐。

---

**相关笔记**：[[Transformer 位置编码|Transformer 位置编码]] | [[BERT|BERT]] | [[GPT|GPT]] | [[LLaMA|LLaMA]] | [[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操|Tokenizer-Embedding-手撕实操]] — 手写 BPE 合并 + token embedding lookup 完整实现
