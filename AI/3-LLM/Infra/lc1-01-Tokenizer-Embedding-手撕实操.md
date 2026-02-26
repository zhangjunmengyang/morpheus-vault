---
title: "Tokenizer & Embedding 手撕实操"
brief: "BPE/WordPiece/SentencePiece Tokenizer原理与实现，Word2Vec/GloVe→Contextual Embedding演进，Positional Encoding（绝对/相对/RoPE/ALiBi）完整代码对比，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, tokenizer, embedding, bpe, rope, pytorch]
related:
  - "[[AI/3-LLM/Architecture/Transformer-手撕实操|Transformer-手撕实操]]"
  - "[[AI/3-LLM/Architecture/基础数学组件手撕|基础数学组件手撕]]"
  - "[[AI/3-LLM/Architecture/Llama-手撕实操|Llama-手撕实操]]"
---

# Tokenizer & Embedding 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

语言模型理解文本的两个核心步骤：

1. **离散化（Tokenization）**：将连续文本切分为 token 序列——可以是字符级、词级或子词级
2. **连续化（Embedding）**：将离散 token 映射为连续向量，使模型可以进行梯度优化

三种表示方法的对比：
- **One-hot**：离散→离散向量，高维稀疏（词表 128K 则向量 128K 维），计算低效
- **Embedding**：离散→连续向量，低维稠密，特征由训练任务自动学习
- **Vector Quantization**：连续→离散（CV 常用，NLP 不涉及）

**分词器 = 分词规则 + 词表**。规则决定切分方式，词表存储合法 token 集合。规则唯一 → 编码结果唯一。

---

## 二、核心实现

### 2.1 One-hot 编码

**原理**：词表大小为 V，每个 token 用 V 维向量表示，仅对应位置为 1，其余为 0。问题：V 很大时极度稀疏。

**代码**：

```python
# 来自 lecture/lc1_base/embedding.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 7
seq_len = 4
x = torch.randint(0, vocab_size, (1, seq_len))

# PyTorch 内置
one_hot = F.one_hot(x, num_classes=vocab_size)

# 手动实现
one_hot_manual = torch.zeros(1, seq_len, vocab_size, dtype=torch.long)
one_hot_manual[:, torch.arange(seq_len).tolist(), x[0,:]] = 1
```

### 2.2 Embedding 层

**原理**：本质是一个 `vocab_size × dim` 的查找表矩阵。给定 token_id，取对应行向量作为该 token 的连续表示。随模型训练，Embedding 参数通过反向传播自动调整——表征什么完全由训练任务决定。

**代码**：

```python
# 来自 lecture/lc1_base/embedding.ipynb
dim = 5
E = torch.randn(vocab_size, dim)  # 随机初始化 embedding 矩阵

# 手动实现：直接索引
input_embd = E[x[0,:], :]

# PyTorch 实现
embedding_layer = nn.Embedding(vocab_size, dim)
embedding_layer.weight.data = E.clone()  # 可用自定义矩阵初始化
output = embedding_layer(x)  # 与手动索引结果一致
```

**关键洞察**：`nn.Embedding` 不是"计算"，而是"查表"。它的梯度只更新被索引到的行，这就是为什么 embedding 层的梯度是稀疏的。

### 2.3 Word2Vec：无监督词向量学习

**原理**：假设"相邻词之间有关联"。给定中心词，预测其上下文窗口内的词（Skip-gram 思想）。通过大量语料训练，Embedding 自然获得语义——相似词的向量距离更近。

**代码**：

```python
# 来自 lecture/lc1_base/embedding.ipynb
# 构建数据集：中心词 -> 上下文词
gram = 2  # 窗口大小
dataset_x, dataset_y = [], []
for i in range(w2v_seq_len):
    left_id = max(0, i - gram)
    right_id = min(w2v_seq_len, i + gram)
    for j in range(left_id, right_id + 1):
        if j != i and j < w2v_seq_len:
            dataset_x.append(x_c[i].item())
            dataset_y.append(x_c[j].item())

# 训练：中心词 embedding 与整个 embedding 矩阵做内积 → logits → 交叉熵
def train(center_words, target_words, model):
    logits = model[center_words, :] @ model.t()  # n × vocab_size
    label = torch.tensor(target_words, dtype=torch.long)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    model = model + lr * model.grad
    model.grad = torch.zeros_like(model)
    return loss.item(), model
```

**关键洞察**：Word2Vec 的核心是 `E[center] @ E.T` 产生 logits——本质是用向量内积度量词间相似度。训练后的 embedding 是静态的（同一词在不同上下文中表示相同），这正是后来 Transformer 要解决的问题。

### 2.4 序列语义表征：从词级到上下文级

**原理**：单个 token 的 embedding 是 word-level 表示，不含上下文信息。要获得 context-level 表示，需要对序列 embedding 进行组合：

| 方法 | 公式 | 特点 |
|------|------|------|
| 均值归并 | $S = \frac{1}{N}\sum_j X_j$ | 权重均等，最简单 |
| 加权组合 | $S_i = \sum_j w_{ij} X_j$ | 不同 token 贡献不同，$w_{ij}$ 可学习 → **注意力机制的雏形** |
| RNN | $h_t = x_t + w \cdot h_{t-1}$ | 递归累积，串行计算慢 |

**代码**：

```python
# 来自 lecture/lc1_base/embedding.ipynb
# 加权组合（每个 token 有独立权重向量）
weight = torch.randn(seq_len, seq_len)
weight = F.softmax(weight, dim=1)  # 行归一化
S = weight @ X_0  # [seq_len, dim] — 每个 token 都有 context 表示
```

**关键洞察**：加权组合中，每个 token 对应一行权重 $w_i \in \mathbb{R}^N$，这正是 Attention 权重矩阵的原型。当 $w_{ij} = Q_i K_j^T$ 时，就是自注意力机制。

### 2.5 基于 Embedding 的文本分类模型

**代码**：

```python
# 来自 lecture/lc1_base/embedding.ipynb
class SimplesLanguageModel(nn.Module):
    def __init__(self, dim=512, vocab_size=100, class_num=2):
        super().__init__()
        self.E = nn.Embedding(vocab_size, dim)
        self.w_feat = nn.Linear(dim, dim, bias=False)
        self.weight = nn.Linear(dim, 1)       # 学习每个 token 的权重
        self.head = nn.Linear(dim, class_num)

    def forward(self, X):
        bs, seq_len = X.shape
        X = self.E(X)
        feat = self.w_feat(X)
        weight = self.weight(X) / seq_len     # [bs, seq_len, 1]
        h = weight.transpose(2, 1) @ feat     # 加权聚合 → [bs, 1, dim]
        Y = self.head(h)
        return Y[:, 0, :]  # logits
```

---

## 三、字符级分词器手撕

### 3.1 分词规则设计

**原理**：设计正则表达式将文本切分为最小 token 单元。规则优先级：特殊 token（`<SOS>`等）> 标点符号 > 数字（逐字符）> 中文字符（逐字符）> 连续英文字符串。

**代码**：

```python
# 来自 lecture/lc1_base/tokenizer_basic.ipynb
import re, string

zh_symbols = '，。！？；：""''【】（）《》、'
en_symbols = re.escape(string.punctuation)
all_symbols = zh_symbols + en_symbols + ' '
special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']

pattern = (
    r'(?:' + '|'.join(special_tokens) + ')'
    r'|[' + re.escape(all_symbols) + ']'
    r'|\d'
    r'|[\u4e00-\u9fa5]'
    r'|[^' + re.escape(all_symbols) + r'\d\u4e00-\u9fa5<>]+'
)

text = "<SOS>我唱跳和rap有 2 年半。<EOS>"
token_list = re.findall(pattern, text)
# ['<SOS>', '我', '唱', '跳', '和', 'rap', '有', ' ', '2', ' ', '年', '半', '。', '<EOS>']
```

### 3.2 词表构建（Train）

```python
# 来自 lecture/lc1_base/tokenizer_basic.ipynb
from typing import Dict

token_all = token_init_list + token_corpus_list
vocab: Dict[str, int] = {}
vocab_reverse: Dict[int, str] = {}
idx = 0
for value in token_all:
    if value not in vocab:
        vocab[value] = idx
        vocab_reverse[idx] = value
        idx += 1
```

### 3.3 编码（Encode）与解码（Decode）

```python
# 来自 lecture/lc1_base/tokenizer_basic.ipynb
def encode_anything(vocab, pattern, text):
    tokens = re.findall(pattern, text)
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            if len(token) == 1:
                token_ids.append(vocab['<UNK>'])
            else:
                for t in token:  # 未知多字符 token 按字符回退编码
                    token_ids.append(vocab.get(t, vocab['<UNK>']))
    return tokens, token_ids

def decode(vocab_reverse, token_ids):
    return [vocab_reverse[idx] for idx in token_ids]
```

**关键洞察**：`encode_anything` 的回退策略——当整词不在词表中时，逐字符编码。这保证了只要基础字符在词表中，就不会丢失信息。但如果某个字符（如"我"）也不在词表中，则永久丢失为 `<UNK>`。

### 3.4 Padding 与 Truncation

```python
# 来自 lecture/lc1_base/tokenizer_basic.ipynb
import torch

def padding_max_length(input_ids, max_len=32, pad_token_id=None,
                       padding_side='RIGHT', truction_side='RIGHT'):
    tokens_lens = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
    tokens_max_len = min(torch.max(tokens_lens).item(), max_len)

    # Truncation
    if truction_side == 'RIGHT':
        input_ids = [ids[:min(len(ids), tokens_max_len)] for ids in input_ids]
    else:
        input_ids = [ids[-min(len(ids), tokens_max_len):] for ids in input_ids]

    # Padding
    padded = torch.ones(len(input_ids), tokens_max_len, dtype=torch.long) * pad_token_id
    for i in range(len(input_ids)):
        L = len(input_ids[i])
        if padding_side == 'RIGHT':
            padded[i, :L] = torch.tensor(input_ids[i], dtype=torch.long)
        else:
            padded[i, -L:] = torch.tensor(input_ids[i], dtype=torch.long)
    return padded
```

---

## 四、BPE 分词器手撕

### 4.1 BPE 算法原理

**BPE（Byte-Pair Encoding）** 是 GPT 系列采用的子词分词方法。核心思想：

1. 初始化：将文本转为 UTF-8 字节序列（256 个基础 token）
2. 迭代合并：统计所有相邻 byte-pair 出现频次 → 取最高频 pair → 合并为新 token → 更新序列
3. 重复直到达到目标词表大小

**优势**：子词粒度兼顾了字符级的完备性和词级的效率。基础字符词表保证能编码任意文本，高频组合被合并为单独 token 压缩序列长度。

**缺陷**：对数字比较和字符计数任务天然不友好（如 "9.11" vs "9.8"、"strawberry" 有几个 "r"）。

### 4.2 核心工具函数

```python
# 来自 notebook/common/BPE-Tokenizer.ipynb & lecture/lc3_gpt/BPE-Tokenizer.ipynb
def get_stats(ids, counts=None):
    """统计相邻 token pair 出现频次"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """将序列中所有匹配的 pair 替换为新 token idx"""
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

### 4.3 BPE 训练

```python
# 来自 lecture/lc3_gpt/BPE-Tokenizer.ipynb
INITIAL_VOCAB_SIZE = 256

class BasicTokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(INITIAL_VOCAB_SIZE)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= INITIAL_VOCAB_SIZE
        num_merges = vocab_size - INITIAL_VOCAB_SIZE

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(INITIAL_VOCAB_SIZE)}

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)  # 取最高频 pair
            idx = 256 + i
            ids = merge(ids, pair, idx)       # 合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # 新 token = 两个子 token 拼接

        self.merges = merges
        self.vocab = vocab
```

### 4.4 BPE 编码

**原理**：编码不是"子串匹配"，而是重放训练过程——先将文本转为最小粒度 byte，然后按 merges 表中的 **id 从小到大** 依次合并（id 越小 = 训练时出现频次越高 = 优先合并）。

```python
# 来自 lecture/lc3_gpt/BPE-Tokenizer.ipynb
def encode(self, text):
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2:
        stats = get_stats(ids)
        # 取 merges 表中 idx 最小的 pair（频次最高的先合并）
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        if pair not in self.merges:
            break
        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
    return ids
```

**关键洞察**：编码时用 `min` 而非 `max`——因为 `merges[pair]` 的值是新 token 的 id，id 越小说明在训练中越早被合并（频次越高），应该优先处理。这保证了编码结果与训练规则完全一致。

### 4.5 BPE 解码

```python
# 来自 lecture/lc3_gpt/BPE-Tokenizer.ipynb
def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")
```

### 4.6 中英文 BPE 扩展

```python
# 来自 lecture/lc3_gpt/BPE-Tokenizer.ipynb
from collections import Counter
import copy

class BPETokenizer:
    def __init__(self, text):
        self.merges = {}
        # 以字符频次排序构建初始词表（而非固定 256 byte）
        char_counter = Counter(text)
        sorted_chars = sorted(char_counter.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {i: k for i, (k, v) in enumerate(sorted_chars)}

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - len(self.vocab)
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        ids = [reverse_vocab[c] for c in text]
        merges = {}
        vocab = copy.deepcopy(self.vocab)
        cur_vocab_size = len(self.vocab)

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = cur_vocab_size + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
```

---

## 五、Embedding 反向传播手撕

```python
# 来自 lecture/lc1_base/embedding.ipynb
# 前向
E = torch.randn(vocab_size, dim, requires_grad=True)
x = torch.randint(1, vocab_size, (1, seq_len))[0]
W = torch.randn(dim, dim_out)
label = torch.randn(1, dim_out)

X = E[x, :]      # 查表取 embedding
Y = X @ W         # 线性变换
loss = (1/Y.numel()) * ((Y - label) ** 2).sum()  # MSE

# 反向手撕
dL = (1/Y.numel()) * 2 * (Y - label)
dW = X.t() @ dL
dX = dL @ W.t()

# Embedding 梯度：scatter_add 回词表
dE = torch.zeros(vocab_size, dim)
for i in range(seq_len):
    dE[x[i], :] += dX[i, :]  # 同一 token 出现多次，梯度累加
```

**关键洞察**：Embedding 的反向传播本质是 scatter_add——只有被前向索引到的行会收到梯度，且同一 token 出现多次时梯度叠加。这解释了为什么高频词的 embedding 训练更快。

---

## 六、配套实操

> 完整代码见：
> - `/tmp/ma-rlhf/lecture/lc1_base/` — embedding.ipynb, tokenizer_basic.ipynb
> - `/tmp/ma-rlhf/notebook/common/BPE-Tokenizer.ipynb` — BPE 通用实现
> - `/tmp/ma-rlhf/lecture/lc3_gpt/BPE-Tokenizer.ipynb` — BPE 详细讲解版

---

## 七、关键洞察与总结

1. **文本处理流水线**：Raw Text → Tokenizer（分词+词表映射）→ token_ids → Embedding 查表 → 连续向量 → 模型

2. **Embedding 的本质**：一个可学习的查找表。不做计算，只做索引。梯度通过 scatter_add 回传。

3. **Word2Vec 的局限**：静态词向量——"刷子"在"两把刷子"和"用刷子画画"中表示相同。这催生了需要上下文的动态表征（→ Attention）。

4. **加权组合是注意力的原型**：$S_i = \sum_j w_{ij} X_j$ 中，当权重 $w_{ij}$ 从固定变为数据依赖（$w_{ij} = f(X_i, X_j)$），就是注意力机制。

5. **BPE 的精髓**：
   - 训练 = 贪心合并最高频 pair
   - 编码 = 按训练顺序（id 从小到大）重放合并
   - 词表 = 基础 byte + 合并产生的新 token
   - `merges` 表和 `vocab` 同等重要，缺一不可

6. **分词器的根本取舍**：词表大 → 序列短、embedding/lm_head 参数多；词表小 → 序列长、计算量大。主流 LLM 词表约 32K~128K，是效率和参数量的平衡点。
