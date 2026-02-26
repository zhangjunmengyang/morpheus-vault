---
title: Tokenizer 与分词：现代 LLM 的语言切分艺术
brief: 分词是 NLP 的第一步也是最被低估的环节。三大主流算法：BPE（贪心合并最频繁字符对，GPT/LLaMA 采用）、WordPiece（基于语言模型概率选择合并，BERT 采用）、Unigram（概率模型全局最优，SentencePiece 默认）。词表大小的黄金法则：英文 30-50K，多语言 50-100K。Byte-level BPE 彻底解决 OOV 问题，是现代 LLM 的主流选择。
type: concept
domain: ai/llm/architecture
created: 2026-02-14
updated: 2026-02-22
tags:
  - ai/llm/tokenizer
  - ai/llm/architecture
  - type/concept
status: complete
sources:
  - Sennrich et al. *Neural Machine Translation of Rare Words with Subword Units (BPE)* arXiv:1508.07909
  - "Kudo & Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing* arXiv:1808.06226"
  - "Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* arXiv:1804.10959 (Unigram LM)"
  - OpenAI tiktoken 文档 — https://github.com/openai/tiktoken
related:
  - "[[BERT|BERT]]"
  - "[[GPT|GPT]]"
  - "[[Qwen|Qwen]]"
  - "[[LLaMA|LLaMA]]"
---

# Tokenizer 与分词：现代 LLM 的语言切分艺术

分词（Tokenization）是 NLP 的第一步，也是最容易被忽视的关键环节。从词级别到子词再到字节级编码，分词技术的演进直接影响了模型的性能上限。本文将深入解析现代 LLM 中的分词技术，重点关注 BPE、WordPiece、Unigram 等主流算法。

## 分词技术演进史

### 传统方法的局限

```python
# 传统空格分词的问题
text = "I'm loving machine-learning!"

# 简单空格分词
simple_split = text.split()
print(simple_split)  
# ['I'm', 'loving', 'machine-learning!']

# 问题：
# 1. I'm -> 应该分成 I + 'm
# 2. machine-learning -> 连字符处理
# 3. 标点符号处理
# 4. OOV (Out-of-Vocabulary) 问题
```

### 子词分词的优势

子词（Subword）分词解决了传统方法的核心问题：

1. **开放词表**：处理未见过的词
2. **形态学感知**：捕获词缀信息
3. **词表大小可控**：平衡表达能力和效率
4. **多语言友好**：统一处理不同语言

## 主流算法深度解析

### 1. BPE (Byte Pair Encoding)

BPE（arXiv:1508.07909）是最广泛使用的子词算法，核心思想是贪心地合并最频繁的字符对。

#### 算法原理

```python
from collections import defaultdict, Counter
import re

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = []
        
    def train(self, corpus):
        # 1. 统计词频
        self.word_freqs = Counter()
        for text in corpus:
            words = re.findall(r'\w+', text.lower())
            self.word_freqs.update(words)
        
        # 2. 初始化：每个词分解为字符
        alphabet = set()
        for word in self.word_freqs:
            alphabet.update(word)
        
        # 初始词表：单字符 + </w>
        vocab = list(alphabet) + ['</w>']
        
        # 3. 初始分割：添加词尾标记
        self.splits = {
            word: [c for c in word[:-1]] + [word[-1] + '</w>']
            for word in self.word_freqs
        }
        
        # 4. 迭代合并
        while len(vocab) < self.vocab_size:
            # 计算所有相邻pair的频率
            pairs = defaultdict(int)
            for word, freq in self.word_freqs.items():
                split = self.splits[word]
                for i in range(len(split) - 1):
                    pairs[(split[i], split[i+1])] += freq
            
            if not pairs:
                break
                
            # 找到最频繁的pair
            best_pair = max(pairs, key=pairs.get)
            
            # 合并这个pair
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab.append(new_token)
            
            # 更新所有splits
            self.merge_vocab(best_pair)
    
    def merge_vocab(self, pair):
        """合并指定的字符对"""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in self.word_freqs:
            w_out = self.splits[word]
            # 查找并合并
            new_split = []
            i = 0
            while i < len(w_out):
                if i < len(w_out) - 1 and (w_out[i], w_out[i+1]) == pair:
                    new_split.append(w_out[i] + w_out[i+1])
                    i += 2
                else:
                    new_split.append(w_out[i])
                    i += 1
            self.splits[word] = new_split
    
    def tokenize(self, text):
        """对新文本进行分词"""
        words = re.findall(r'\w+', text.lower())
        result = []
        
        for word in words:
            # 应用学到的merges
            split = [c for c in word[:-1]] + [word[-1] + '</w>']
            for pair in self.merges:
                # 依次应用每个merge规则
                split = self.apply_merge(split, pair)
            result.extend(split)
        return result
    
    def apply_merge(self, split, pair):
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                new_split.append(split[i] + split[i+1])
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        return new_split

# 使用示例
corpus = [
    "the quick brown fox jumps",
    "the fox is quick",
    "brown fox jumps high"
]

tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(corpus)
tokens = tokenizer.tokenize("the quickest fox")
print(f"分词结果: {tokens}")
# 可能输出: ['th', 'e</w>', 'qui', 'ck', 'est</w>', 'fox</w>']
```

#### BPE 变体：Byte-level BPE

GPT-2/LLaMA 使用的改进版本：

```python
import json

class ByteLevelBPE:
    """GPT-2 风格的字节级 BPE"""
    
    def __init__(self):
        # 字节到Unicode的映射（处理所有可能字节）
        self.bytes_to_unicode = self._bytes_to_unicode()
        self.unicode_to_bytes = {v: k for k, v in self.bytes_to_unicode.items()}
        
    def _bytes_to_unicode(self):
        """创建字节到Unicode字符的映射"""
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))
    
    def encode_text(self, text):
        """文本转换为字节序列"""
        byte_encoded = text.encode('utf-8')
        return ''.join([self.bytes_to_unicode[b] for b in byte_encoded])
    
    def decode_text(self, tokens):
        """字节序列转换回文本"""
        byte_string = ''.join(tokens)
        byte_array = bytes([self.unicode_to_bytes[c] for c in byte_string])
        return byte_array.decode('utf-8', errors='replace')

# 使用示例：处理多语言文本
byte_bpe = ByteLevelBPE()
text = "Hello 世界! 🤖"
encoded = byte_bpe.encode_text(text)
print(f"字节编码: {encoded}")
decoded = byte_bpe.decode_text(encoded)
print(f"解码结果: {decoded}")
```

### 2. WordPiece (BERT 使用)

WordPiece 与 BPE 类似，但使用不同的合并策略。

```python
import math
from collections import defaultdict

class WordPieceTokenizer:
    def __init__(self, vocab_size=1000, unk_token='[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        
    def train(self, corpus):
        # 1. 收集词频
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.split()
            for word in words:
                word_freqs[word] += 1
        
        # 2. 初始词表
        alphabet = set()
        for word in word_freqs:
            alphabet.update(word)
        
        vocab = {char: i for i, char in enumerate(sorted(alphabet))}
        vocab[self.unk_token] = len(vocab)
        
        # 3. 准备训练数据
        word_splits = {}
        for word, freq in word_freqs.items():
            word_splits[word] = [char for char in word]
        
        # 4. 迭代添加词片段
        while len(vocab) < self.vocab_size:
            scores = {}
            
            # 计算每个可能合并的分数
            for word, freq in word_freqs.items():
                split = word_splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    if pair not in scores:
                        scores[pair] = 0
                    scores[pair] += freq
            
            # 选择分数最高的pair
            if not scores:
                break
                
            best_pair = max(scores.items(), key=lambda x: x[1])
            pair, score = best_pair
            
            # 创建新token
            new_token = pair[0] + pair[1]
            vocab[new_token] = len(vocab)
            
            # 更新splits
            for word in word_freqs:
                new_split = []
                i = 0
                while i < len(word_splits[word]):
                    if (i < len(word_splits[word]) - 1 and 
                        word_splits[word][i] == pair[0] and 
                        word_splits[word][i+1] == pair[1]):
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(word_splits[word][i])
                        i += 1
                word_splits[word] = new_split
        
        self.vocab = vocab
        
    def tokenize(self, text):
        """使用贪心最长匹配"""
        words = text.split()
        result = []
        
        for word in words:
            tokens = self._tokenize_word(word)
            result.extend(tokens)
        return result
    
    def _tokenize_word(self, word):
        """对单个词进行WordPiece分词"""
        if word in self.vocab:
            return [word]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # 贪心找最长子串
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr  # WordPiece的子词前缀
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return [self.unk_token]
            
            tokens.append(cur_substr)
            start = end
        
        return tokens

# 使用示例
wp = WordPieceTokenizer(vocab_size=100)
corpus = ["playing played player", "walking walked walker"]
wp.train(corpus)
tokens = wp.tokenize("walking player")
print(f"WordPiece分词: {tokens}")
# 可能输出: ['walk', '##ing', 'play', '##er']
```

### 3. Unigram Language Model

SentencePiece 的默认算法（Kudo. *Subword Regularization* arXiv:1804.10959），使用概率模型。

```python
import math
from collections import defaultdict, Counter

class UnigramTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.log_probs = {}
        
    def train(self, corpus, num_iterations=10):
        # 1. 收集所有可能的子串
        substrings = set()
        word_freqs = Counter()
        
        for text in corpus:
            words = text.split()
            for word in words:
                word_freqs[word] += 1
                # 添加所有可能的子串
                for i in range(len(word)):
                    for j in range(i + 1, len(word) + 1):
                        substrings.add(word[i:j])
        
        # 2. 初始化大词表（保留高频子串）
        substr_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for substr in substrings:
                if substr in word:
                    substr_counts[substr] += freq * word.count(substr)
        
        # 选择初始词表
        sorted_substrs = sorted(substr_counts.items(), 
                              key=lambda x: x[1], reverse=True)
        initial_vocab = dict(sorted_substrs[:self.vocab_size * 3])
        
        # 3. EM算法迭代优化
        current_vocab = initial_vocab
        
        for iteration in range(num_iterations):
            # E-step: 计算最佳分割
            word_splits = {}
            for word in word_freqs:
                word_splits[word] = self._best_split(word, current_vocab)
            
            # M-step: 更新概率
            token_counts = defaultdict(int)
            total_tokens = 0
            
            for word, freq in word_freqs.items():
                for token in word_splits[word]:
                    token_counts[token] += freq
                    total_tokens += freq
            
            # 计算对数概率
            new_log_probs = {}
            for token, count in token_counts.items():
                new_log_probs[token] = math.log(count / total_tokens)
            
            # 修剪词表
            if len(token_counts) > self.vocab_size:
                sorted_tokens = sorted(token_counts.items(), 
                                     key=lambda x: x[1], reverse=True)
                current_vocab = dict(sorted_tokens[:self.vocab_size])
                self.log_probs = {k: new_log_probs[k] for k in current_vocab}
            else:
                current_vocab = token_counts
                self.log_probs = new_log_probs
        
        self.vocab = current_vocab
    
    def _best_split(self, word, vocab):
        """使用动态规划找最佳分割"""
        n = len(word)
        # dp[i] 存储 word[:i] 的最佳分割的负对数概率
        dp = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)
        dp[0] = 0.0
        
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, n + 1):
                substr = word[i:j]
                if substr in vocab:
                    prob = self.log_probs.get(substr, -20.0)  # 默认低概率
                    if dp[i] - prob < dp[j]:  # 负对数概率，越小越好
                        dp[j] = dp[i] - prob
                        parent[j] = i
        
        # 回溯构建分割
        result = []
        pos = n
        while pos > 0:
            start = parent[pos]
            result.append(word[start:pos])
            pos = start
        
        return result[::-1]
    
    def tokenize(self, text):
        words = text.split()
        result = []
        for word in words:
            tokens = self._best_split(word, self.vocab)
            result.extend(tokens)
        return result

# 使用示例
unigram = UnigramTokenizer(vocab_size=50)
corpus = ["machine learning", "deep learning", "neural network"]
unigram.train(corpus)
tokens = unigram.tokenize("deep neural")
print(f"Unigram分词: {tokens}")
```

### 4. SentencePiece：统一框架

Google 的 SentencePiece（arXiv:1808.06226）提供了统一的接口：

```python
# 安装: pip install sentencepiece
import sentencepiece as spm

def train_sentencepiece(input_file, vocab_size=8000):
    """训练 SentencePiece 模型"""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix='sp_model',
        vocab_size=vocab_size,
        model_type='bpe',  # 'bpe', 'unigram', 'char', 'word'
        max_sentence_length=4192,
        shuffle_input_sentence=True,
        character_coverage=0.9995,
        # 特殊token
        pad_id=0,
        unk_id=1, 
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<mask>']
    )

# 加载和使用
sp = spm.SentencePieceProcessor()
sp.load('sp_model.model')

# 编码
text = "SentencePiece is a great tokenizer!"
tokens = sp.encode(text, out_type=str)
print(f"Tokens: {tokens}")

ids = sp.encode(text, out_type=int)
print(f"IDs: {ids}")

# 解码
decoded = sp.decode(ids)
print(f"Decoded: {decoded}")

# 词表信息
print(f"Vocab size: {sp.vocab_size()}")
print(f"UNK token: {sp.id_to_piece(sp.unk_id())}")
```

## 分词对模型性能的影响

### 1. 词表大小的权衡

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_vocab_size_impact():
    """分析词表大小对各项指标的影响"""
    vocab_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    
    # 模拟数据（基于经验）
    compression_ratio = [0.3, 0.45, 0.6, 0.7, 0.75, 0.8]  # 压缩比
    training_speed = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]       # 训练速度
    downstream_performance = [0.7, 0.8, 0.85, 0.9, 0.92, 0.93]  # 下游任务性能
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 压缩比
    ax1.plot(vocab_sizes, compression_ratio, 'b-o')
    ax1.set_xlabel('词表大小')
    ax1.set_ylabel('压缩比')
    ax1.set_title('压缩比 vs 词表大小')
    ax1.grid(True)
    
    # 训练速度
    ax2.plot(vocab_sizes, training_speed, 'r-s')
    ax2.set_xlabel('词表大小')
    ax2.set_ylabel('相对训练速度')
    ax2.set_title('训练速度 vs 词表大小')
    ax2.grid(True)
    
    # 下游性能
    ax3.plot(vocab_sizes, downstream_performance, 'g-^')
    ax3.set_xlabel('词表大小')
    ax3.set_ylabel('下游任务性能')
    ax3.set_title('模型性能 vs 词表大小')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return vocab_sizes, compression_ratio, training_speed, downstream_performance

# 运行分析
analyze_vocab_size_impact()
```

### 2. 不同算法的特性对比

| 算法 | 压缩效率 | 训练速度 | OOV处理 | 多语言 | 实现复杂度 |
|------|----------|----------|---------|--------|------------|
| **BPE** | 高 | 快 | 优秀 | 好 | 简单 |
| **WordPiece** | 中 | 中 | 优秀 | 中 | 中等 |
| **Unigram** | 中 | 慢 | 最优 | 最好 | 复杂 |
| **SentencePiece** | 高 | 快 | 优秀 | 最好 | 简单（库） |

### 3. 真实案例：模型对比

```python
def compare_tokenizers_on_text():
    """对比不同分词器的效果"""
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "机器学习是人工智能的一个重要分支。",
        "I'm loving this new iPhone! 😍",
        "COVID-19 has significantly impacted the world.",
        "She said, 'Hello!' very enthusiastically."
    ]
    
    # 模拟不同分词器的结果
    results = {
        'GPT-2 (BPE)': {
            0: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.'],
            1: ['æ', 'ģ', 'Ń', 'å', 'Ļ', 'Ģ', 'å', 'ĸ', 'Ń', 'ä', 'ł', 'Ģ', 'æ', 'Ĺ', '¯', 'äºº', 'å·¥'],
            2: ["I'm", 'Ġloving', 'Ġthis', 'Ġnew', 'ĠiPhone', '!', 'Ġ😍'],
        },
        'BERT (WordPiece)': {
            0: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'],
            1: ['机', '##器', '##学', '##习', '##是', '##人', '##工', '##智', '##能', '##的'],
            2: ['i', "'", 'm', 'loving', 'this', 'new', 'iphone', '!', '[UNK]'],  # emoji OOV
        },
        'T5 (SentencePiece)': {
            0: ['▁The', '▁quick', '▁brown', '▁fox', '▁jumps', '▁over', '▁the', '▁lazy', '▁dog', '.'],
            1: ['▁机器', '学习', '是', '人工', '智能', '的', '一个', '重要', '分支', '。'],
            2: ['▁I', "'", 'm', '▁loving', '▁this', '▁new', '▁iPhone', '!', '▁😍'],
        }
    }
    
    for i, text in enumerate(test_texts[:3]):
        print(f"\n文本 {i+1}: {text}")
        for tokenizer, tokenizations in results.items():
            if i in tokenizations:
                tokens = tokenizations[i]
                print(f"{tokenizer:20}: {tokens} ({len(tokens)} tokens)")

compare_tokenizers_on_text()
```

## 多语言分词挑战

### 中文分词特殊性

```python
class ChineseTokenizationAnalysis:
    def __init__(self):
        self.examples = {
            '词边界模糊': ['研究生命的起源', '研究/生命/的/起源 vs 研究生/命/的/起源'],
            '词语长度变化': ['人工智能技术发展', '词长度分布不均匀'],
            '新词出现': ['ChatGPT很厉害', '新词需要及时处理'],
        }
    
    def analyze_segmentation_ambiguity(self):
        """分析中文分词歧义"""
        text = "研究生命的起源"
        
        segmentations = [
            ['研究', '生命', '的', '起源'],      # 正确
            ['研究生', '命', '的', '起源'],      # 错误
            ['研究', '生', '命', '的', '起源'],  # 过度分割
        ]
        
        print("中文分词歧义示例:")
        for i, seg in enumerate(segmentations):
            print(f"方案{i+1}: {' / '.join(seg)}")
        
        return segmentations
    
    def subword_advantages_for_chinese(self):
        """子词分词对中文的优势"""
        examples = [
            {
                'word': '人工智能',
                'char_level': ['人', '工', '智', '能'],
                'subword': ['人工', '智能'],
                'advantage': '保留语义单元'
            },
            {
                'word': 'ChatGPT',
                'char_level': ['C', 'h', 'a', 't', 'G', 'P', 'T'],
                'subword': ['Chat', 'GPT'],
                'advantage': '处理英文混合'
            }
        ]
        
        for ex in examples:
            print(f"\n词语: {ex['word']}")
            print(f"字符级: {ex['char_level']}")
            print(f"子词级: {ex['subword']}")
            print(f"优势: {ex['advantage']}")

# 分析中文分词
chinese_analysis = ChineseTokenizationAnalysis()
chinese_analysis.analyze_segmentation_ambiguity()
chinese_analysis.subword_advantages_for_chinese()
```

### 跨语言统一处理

```python
def multilingual_tokenization_strategy():
    """多语言分词统一策略"""
    
    # 不同语言的特点
    language_features = {
        '英文': {
            '特点': ['空格分隔', '形态变化丰富', '大小写敏感'],
            '挑战': ['缩写处理', '复合词', '新词'],
            '策略': 'BPE with byte-level encoding'
        },
        '中文': {
            '特点': ['无明显分隔', '字符密集', '语义组合'],
            '挑战': ['分词歧义', '新词识别', '古汉语'],
            '策略': 'SentencePiece Unigram'
        },
        '日文': {
            '特点': ['多种文字混合', '无空格', '助词丰富'],
            '挑战': ['假名汉字混合', '语言变体'],
            '策略': 'SentencePiece with character coverage adjustment'
        },
        '阿拉伯文': {
            '特点': ['从右到左', '连写', '变音符号'],
            '挑战': ['字形变化', '方向性'],
            '策略': 'Byte-level BPE with normalization'
        }
    }
    
    print("多语言分词策略:")
    for lang, info in language_features.items():
        print(f"\n{lang}:")
        print(f"  特点: {', '.join(info['特点'])}")
        print(f"  挑战: {', '.join(info['挑战'])}")
        print(f"  推荐策略: {info['策略']}")
    
    # 统一处理方案
    unified_approach = {
        'character_coverage': 0.9995,  # 覆盖99.95%的字符
        'vocab_size': 32000,           # 平衡各语言需求
        'model_type': 'unigram',       # 最适合多语言
        'normalization': True,         # 规范化输入
        'byte_fallback': True,         # 字节级后备
    }
    
    print(f"\n统一配置: {unified_approach}")

multilingual_tokenization_strategy()
```

## 现代 LLM 的分词选择

### 主流模型对比

| 模型 | 分词算法 | 词表大小 | 特殊设计 |
|------|----------|----------|----------|
| **GPT-2/3** | Byte-level BPE | 50,257 | 字节级编码 |
| **GPT-4** | BPE (改进版) | ~100K | 多语言优化 |
| **BERT** | WordPiece | 30,522 | 中文字符级 |
| **T5** | SentencePiece | 32,128 | 多语言统一 |
| **LLaMA** | SentencePiece BPE | 32,000 | 效率优化 |
| **Claude** | 未知 (推测BPE) | ~100K | 专有算法 |

### 选择策略

```python
def choose_tokenization_strategy(use_case):
    """根据应用场景选择分词策略"""
    
    strategies = {
        'english_only': {
            'algorithm': 'BPE',
            'vocab_size': 50000,
            'features': ['简单高效', '成熟稳定'],
            'examples': ['GPT-2', 'RoBERTa']
        },
        'multilingual': {
            'algorithm': 'SentencePiece Unigram',
            'vocab_size': 32000,
            'features': ['多语言友好', '统一处理'],
            'examples': ['T5', 'mT5', 'XLM-R']
        },
        'chinese_focused': {
            'algorithm': 'SentencePiece BPE',
            'vocab_size': 21128,
            'features': ['中文优化', '字词平衡'],
            'examples': ['BERT-Chinese', 'ERNIE']
        },
        'code_generation': {
            'algorithm': 'Byte-level BPE',
            'vocab_size': 50000,
            'features': ['代码友好', '符号处理'],
            'examples': ['CodeGPT', 'GitHub Copilot']
        },
        'domain_specific': {
            'algorithm': '自定义BPE',
            'vocab_size': 30000,
            'features': ['领域词汇', '定制优化'],
            'examples': ['BioBERT', 'FinBERT']
        }
    }
    
    if use_case in strategies:
        strategy = strategies[use_case]
        print(f"推荐策略 ({use_case}):")
        print(f"  算法: {strategy['algorithm']}")
        print(f"  词表大小: {strategy['vocab_size']}")
        print(f"  特点: {', '.join(strategy['features'])}")
        print(f"  案例: {', '.join(strategy['examples'])}")
        return strategy
    else:
        print(f"未知用例: {use_case}")
        return None

# 测试不同场景
for case in ['english_only', 'multilingual', 'chinese_focused', 'code_generation']:
    choose_tokenization_strategy(case)
    print()
```

## 面试常见问题

### Q1：BPE、WordPiece、Unigram 三种算法的核心区别是什么？

**答案**：

**BPE (Byte Pair Encoding)**：
- 原理：贪心合并最频繁的字符对
- 优势：简单高效，压缩效果好
- 劣势：纯统计方法，缺乏语言学考虑
- 适用：通用场景，特别是英文

**WordPiece**：
- 原理：基于语言模型概率选择合并
- 优势：考虑上下文信息，更有语言学意义
- 劣势：训练复杂度高
- 适用：理解任务，BERT系列

**Unigram**：
- 原理：概率语言模型，EM算法优化
- 优势：全局最优解，多语言友好
- 劣势：计算复杂，训练慢
- 适用：多语言模型，SentencePiece默认

### Q2：为什么现代 LLM 普遍选择较大的词表（32K-100K）？

**答案**：
1. **表达能力**：大词表减少序列长度，提高模型效率
2. **多语言支持**：覆盖更多语言的常用词汇
3. **领域适应**：包含专业术语和新词
4. **计算平衡**：嵌入层增大 vs 序列长度减少的权衡
5. **硬件发展**：现代GPU内存充足，支持大词表

**经验规律**：
- 英文：30-50K 足够
- 多语言：50-100K
- 代码生成：50-100K
- 专业领域：根据语料调整

### Q3：字节级 BPE 相比传统 BPE 有什么优势？

**答案**：
**传统BPE问题**：
1. 词表会遗漏一些Unicode字符
2. 处理多语言时需要大词表
3. 新语言/emoji 处理困难

**字节级BPE优势**：
1. **完全覆盖**：任何文本都能编码，无OOV
2. **多语言友好**：统一处理所有语言
3. **新内容兼容**：emoji、特殊符号、新语言
4. **压缩效率**：字节级编码更紧凑

**代价**：
- 序列变长（中文等）
- 编码更复杂
- 解码需要额外处理

### Q4：如何评估分词质量？分词对下游任务有什么影响？

**答案**：
**评估指标**：
1. **压缩率**：原始字符数 / token数
2. **词边界准确率**：与人工分词对比
3. **OOV率**：未见词汇比例
4. **下游任务性能**：最终评价标准

**影响分析**：
- **过度分割**：信息密度低，序列变长，训练慢
- **分割不足**：OOV问题，泛化能力差
- **不一致分割**：同一概念多种表示，学习困难

**最佳实践**：
```python
def evaluate_tokenization_quality(tokenizer, test_data):
    metrics = {
        'compression_ratio': [],
        'avg_token_length': [],
        'oov_rate': []
    }
    
    for text in test_data:
        tokens = tokenizer.tokenize(text)
        char_count = len(text.replace(' ', ''))
        token_count = len(tokens)
        
        metrics['compression_ratio'].append(char_count / token_count)
        metrics['avg_token_length'].append(
            sum(len(t.replace('##', '').replace('▁', '')) for t in tokens) / len(tokens)
        )
        # 计算OOV率等
    
    return metrics
```

### Q5：在实际项目中如何选择和优化分词策略？

**答案**：
**选择流程**：

1. **需求分析**：
   - 语言类型（单语言/多语言）
   - 领域特点（通用/专业）
   - 计算资源约束

2. **baseline建立**：
   - 使用现有分词器（如GPT-2的BPE）
   - 在验证集上测试表现

3. **自定义优化**：
```python
# 领域适应示例
def adapt_tokenizer_for_domain(base_tokenizer, domain_corpus):
    # 1. 分析领域特有词汇
    domain_vocab = extract_domain_terms(domain_corpus)
    
    # 2. 调整词表
    extended_vocab = base_tokenizer.vocab.copy()
    extended_vocab.update(domain_vocab)
    
    # 3. 重新训练（或微调）
    adapted_tokenizer = train_tokenizer(
        corpus=domain_corpus,
        base_vocab=extended_vocab,
        vocab_size=target_size
    )
    
    return adapted_tokenizer
```

4. **A/B测试**：
   - 对比不同分词策略
   - 在下游任务上验证效果

5. **持续优化**：
   - 监控新词出现
   - 定期更新词表

**工程建议**：
- 优先使用成熟方案（SentencePiece）
- 保留扩展性（易于更新）
- 版本管理（分词器版本与模型绑定）
- 向后兼容（新版本兼容旧数据）

## 📚 推荐阅读

### 原始论文
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — BPE 原文，子词分词的开山之作
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 统一分词框架，支持 BPE/Unigram
- [Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959) — Unigram 分词算法，概率模型方法

### 深度解读
- [HuggingFace Tokenizers 教程](https://huggingface.co/docs/tokenizers/) — 分词器训练和使用的最佳实践 ⭐⭐⭐⭐⭐
- [Let's build the GPT Tokenizer (Karpathy)](https://www.youtube.com/watch?v=zduSFxRajkE) — Karpathy 从零实现 BPE 的视频教程 ⭐⭐⭐⭐⭐

### 实践资源
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI 的高性能 BPE 实现，GPT-4 使用
- [sentencepiece](https://github.com/google/sentencepiece) — Google 官方实现，LLaMA/T5/Qwen 使用
- [tokenizers](https://github.com/huggingface/tokenizers) — HuggingFace 的 Rust 高性能分词库

## 🔧 落地应用

### 直接可用场景
- **模型选型时的 Tokenizer 评估**：中文场景下，tiktoken（GPT-4）平均每个汉字 ~1.5 token，而 LLaMA 的 SentencePiece 每个汉字 ~2.5 token——直接影响有效上下文长度
- **领域自定义 Tokenizer**：医学/法律等专业领域的术语如果被过度分割，会浪费上下文窗口。用领域语料训练自定义 BPE 可提升 15-30% 的压缩率
- **多语言部署**：选择 Byte-level BPE 或 SentencePiece Unigram，确保零 OOV

### 工程实现要点
- **词表大小经验值**：英文 30-50K，多语言 50-100K，代码场景 50-100K
- **Byte-level BPE 的代价**：中文每字符需要 3 个 UTF-8 字节，压缩前序列更长，需要更大词表补偿
- **版本绑定**：Tokenizer 版本必须与模型版本严格绑定，更换分词器等于换了模型

### 面试高频问法
- Q: 为什么现代 LLM 不用字符级或词级分词？
  A: 字符级序列太长（$O(5\times)$），词级有 OOV 问题且无法泛化到新词。子词分词是最优平衡——词表可控、无 OOV、保留形态学信息。BPE/Unigram 的压缩率在 3-5 characters/token 之间。

## 💡 启发与思考

### So What？对老板意味着什么
- **Tokenizer 决定了模型的"视力"**：分词不好，模型看到的就是碎片化的字符而非有意义的语义单元。选模型时不只看参数量，还要看它的 tokenizer 对目标语言的效率
- **中文场景的隐藏成本**：很多英文优先的模型（如早期 LLaMA）对中文分词效率低，同样 4K 上下文窗口的"有效中文容量"可能只有 GPT-4 的 60%

### 未解问题与局限
- 子词分词是否已经是最优方案？最近有 byte-level 模型（如 ByT5）直接在字节上建模，跳过分词步骤，但训练成本更高
- 分词对下游任务的影响量化研究仍不充分——同一个模型，换分词器会造成多大的性能差异？
- 代码分词的特殊挑战：缩进、括号、运算符的处理没有统一最优方案

### 脑暴：如果往下延伸
- 如果 [[Mamba-SSM|Mamba]] 的线性复杂度让超长序列变得廉价，字符级/字节级模型是否会卷土重来？（不再需要压缩序列长度）
- [[Qwen|Qwen]] 的多语言分词策略 vs [[GPT|GPT-4]] 的 tiktoken：哪种对中文更友好？量化对比是一个有价值的实验

---

## See Also

- [[BERT|BERT]] — 使用 WordPiece 分词（子词级）
- [[GPT|GPT]] — 使用 Byte-level BPE (tiktoken)，字节级 BPE
- [[LLaMA|LLaMA]] — 使用 SentencePiece BPE，多语言支持
- [[Qwen|Qwen]] — 使用 SentencePiece，多语言优化，中文 token 效率高
- [[AI/LLM/Architecture/Tokenizer深度理解]] — 同主题深度版（BPE/WordPiece/SentencePiece 原理对比 + 面试题）
- [[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操]] — 手撕实操版（BPE 算法 + Embedding 完整实现）