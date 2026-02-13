---
title: "数据工程 for LLM：从预训练到对齐"
date: 2026-02-13
tags:
  - ai/llm/training
  - ai/data-engineering
  - ai/synthetic-data
  - type/practice
  - interview/hot
status: active
---

# 数据工程 for LLM：从预训练到对齐

> Data is the new moat——LLM 的能力上限由数据质量决定，而非模型架构

## 1. 数据工程全景

```
Pre-training Data Pipeline          Post-training Data Pipeline
┌──────────────────────┐         ┌──────────────────────────┐
│ 原始语料 (Common Crawl等)      │ SFT 数据合成             │
│  ↓ 语言检测                    │  ↓ 指令构造               │
│  ↓ 启发式清洗                  │  ↓ 回答生成               │
│  ↓ 质量过滤 (Model-based)      │  ↓ 质量筛选               │
│  ↓ 去重 (MinHash/SimHash)     │                          │
│  ↓ PII 脱敏                   │ DPO/RLHF 偏好数据         │
│  ↓ 数据配比                    │  ↓ 采样 chosen/rejected   │
│  ↓ Tokenization               │  ↓ 奖励模型打分            │
│  ↓ 数据打包                    │  ↓ 偏好对构造             │
└──────────────────────┘         └──────────────────────────┘
```

## 2. 预训练数据管线

### 2.1 数据源

```
源                 规模              质量     用途
───────────────────────────────────────────────
Common Crawl       PB 级            低-中    通用知识
Wikipedia          ~20B tokens      高       事实知识
Books (PG/L-Book)  ~5B tokens       高       长文本/推理
ArXiv/PubMed       ~10B tokens      高       学术/STEM
GitHub/StackOverflow ~100B tokens   中-高    代码
专有数据            变化             变化     竞争壁垒

2025 数据量级:
  LLaMA 3: 15T tokens
  DeepSeek-V3: 14.8T tokens
  Qwen 2.5: ~18T tokens
```

### 2.2 启发式清洗

```python
def heuristic_filter(doc: str) -> bool:
    """常用启发式规则 (参考 C4/RedPajama/FineWeb)"""
    lines = doc.split('\n')
    words = doc.split()

    # 长度过滤
    if len(words) < 50 or len(words) > 100_000:
        return False

    # 重复行比例 (去除模板/导航栏)
    unique_lines = set(lines)
    if len(unique_lines) / max(len(lines), 1) < 0.3:
        return False

    # 脏词/色情/暴力内容过滤
    if contains_blocklist_words(doc):
        return False

    # 特殊字符比例 (去除乱码/代码垃圾)
    alpha_ratio = sum(c.isalpha() for c in doc) / max(len(doc), 1)
    if alpha_ratio < 0.4:
        return False

    # 句子完整性（以标点结尾的行占比）
    ended_lines = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?。！？')
    if ended_lines / max(len(lines), 1) < 0.1:
        return False

    return True
```

### 2.3 Model-based 质量过滤

**2025 年最佳实践**：用小模型 (如 fasttext / 小型 BERT) 打分，替代纯规则过滤：

```python
# FineWeb-Edu 方法: 用 LLM 标注教育价值分数，训练 fasttext 分类器
# 1. 用 GPT-4/Claude 对 ~100K 样本打分 (0-5 分)
# 2. 训练 fasttext 分类器预测分数
# 3. 用分类器过滤整个语料库

import fasttext

# 训练质量分类器
model = fasttext.train_supervised(
    input="quality_labeled.txt",  # 格式: __label__high <text>
    lr=0.1, epoch=5, wordNgrams=2
)

# 批量过滤
for doc in corpus:
    label, score = model.predict(doc)
    if score > 0.8 and label == "__label__high":
        keep(doc)
```

```
效果 (FineWeb-Edu 论文):
  无过滤:     MMLU 40.2
  启发式过滤:  MMLU 45.1
  模型过滤:    MMLU 51.3  (+11.1 points!)
```

### 2.4 去重

去重是**单项收益最大的数据处理步骤**：

```
去重层级:
  1. Exact dedup:  哈希去重（URL/SHA256）→ 去除完全相同文档
  2. Fuzzy dedup:  MinHash LSH → 去除近似重复文档
  3. Paragraph dedup: 段落级去重 → 去除跨文档的重复段落

MinHash 去重原理:
  1. 文档 → n-gram 集合
  2. 多个哈希函数取最小值 → MinHash 签名
  3. LSH (Locality-Sensitive Hashing) 将相似签名分到同一桶
  4. 桶内文档两两比较 Jaccard 相似度
  5. 相似度 > 阈值 (通常 0.8) → 去除
```

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for ngram in ngrams(text.split(), 5):  # 5-gram
        m.update(' '.join(ngram).encode('utf8'))
    return m

# 创建 LSH 索引
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc_id, text in enumerate(corpus):
    mh = create_minhash(text)
    lsh.insert(doc_id, mh)

# 查找近似重复
duplicates = set()
for doc_id, text in enumerate(corpus):
    mh = create_minhash(text)
    result = lsh.query(mh)
    if len(result) > 1:
        duplicates.update(result[1:])  # 保留第一个
```

```
去重对训练的影响 (Llama 技术报告):
  去重前: ~5T tokens, 训练出现过拟合
  去重后: ~1.4T tokens, 质量提升 2-5%
  关键: 去重后的 1.4T 效果优于去重前的 5T
```

### 2.5 数据配比

```
典型预训练数据配比 (参考 LLaMA 3):
  Web 数据:    ~50% (Common Crawl 清洗后)
  代码:        ~15% (GitHub + StackOverflow)
  学术论文:    ~10% (ArXiv + PubMed)
  书籍:         ~8%
  Wikipedia:    ~5%
  数学:         ~5%
  多语言:       ~5%
  对话/论坛:     ~2%

配比原则:
  1. 代码数据提升推理能力 (Codex 论文已证明)
  2. 高质量数据上采样 (Wikipedia 重复 2-5 epochs)
  3. 数学/科学数据提升 STEM 能力
  4. 合成数据补充弱势领域
```

## 3. SFT 数据合成

### 3.1 指令构造方法

```
方法              描述                    质量    成本
───────────────────────────────────────────────
Self-Instruct    LLM 自动生成指令+回答    中     低
Evol-Instruct    渐进增加指令复杂度       高     中
Backtranslation  从回答反推指令           高     中
Seed-driven      人工种子 + LLM 扩展     高     中
GPT-4 蒸馏       直接用 GPT-4 生成       高     高
```

### 3.2 Evol-Instruct (WizardLM 方法)

```python
EVOLUTION_PROMPT = """
请将以下指令进化为更复杂的版本。可以通过以下方式:
1. 添加约束条件
2. 深化问题
3. 增加推理步骤
4. 要求多角度分析
5. 加入具体化场景

原始指令: {instruction}
进化后的指令:
"""

def evolve_instruction(instruction, model, depth=3):
    """多轮进化，逐步增加复杂度"""
    current = instruction
    for _ in range(depth):
        evolved = model.generate(EVOLUTION_PROMPT.format(instruction=current))
        # 质量检查: 确保进化后的指令仍然可解
        if is_answerable(evolved, model):
            current = evolved
    return current
```

### 3.3 数据质量验证

```python
def validate_sft_sample(instruction, response, validator_model):
    """多维度验证 SFT 数据质量"""
    checks = {
        "accuracy": "回答是否事实准确?",
        "completeness": "是否完整回答了指令?",
        "format": "格式是否符合要求?",
        "safety": "是否包含有害内容?",
        "difficulty": "指令难度评级 (1-5)?",
    }
    scores = {}
    for dim, question in checks.items():
        prompt = f"指令: {instruction}\n回答: {response}\n\n{question}\n请给出 1-5 分评分和理由。"
        scores[dim] = validator_model.score(prompt)

    # 综合分数 > 4 才保留
    return sum(scores.values()) / len(scores) > 4.0
```

**2025 年经验**：参见 [[SFT 实战指南]]，**1000 条高质量 > 100K 条低质量**（LIMA 论文）。

## 4. DPO 偏好数据构造

### 4.1 标准流程

```
Step 1: 准备 Prompt 集合 (来自 SFT 数据的指令部分)

Step 2: 采样多个回答
  对每个 prompt，用 SFT 模型生成 K 个回答 (K=4-8)
  温度 > 0 以确保多样性

Step 3: 打分/排序
  方法 A: 奖励模型打分 (RM-based)
  方法 B: LLM-as-Judge (GPT-4/Claude 评分)
  方法 C: 规则判断 (数学/代码题可验证正确性)

Step 4: 构造偏好对
  chosen: 最高分回答
  rejected: 最低分回答 (或随机选一个低分回答)
```

```python
from vllm import LLM, SamplingParams

def generate_preference_pairs(prompts, sft_model_path, reward_model):
    """生成 DPO 偏好数据"""
    llm = LLM(model=sft_model_path)
    params = SamplingParams(temperature=0.8, top_p=0.95, n=8)  # 每个 prompt 采样 8 个

    dataset = []
    for prompt in prompts:
        outputs = llm.generate([prompt], params)[0]
        responses = [o.text for o in outputs.outputs]

        # 奖励模型打分
        scores = reward_model.score(prompt, responses)

        # 选择最高/最低分
        best_idx = scores.argmax()
        worst_idx = scores.argmin()

        # 确保分差足够大 (margin-based 过滤)
        if scores[best_idx] - scores[worst_idx] > 1.0:
            dataset.append({
                "prompt": prompt,
                "chosen": responses[best_idx],
                "rejected": responses[worst_idx],
            })

    return dataset
```

### 4.2 偏好数据的关键技巧

```
✅ 最佳实践:
  1. On-policy 采样: 用当前 SFT 模型生成，而非外部模型
     (off-policy 数据会导致分布偏移，DPO 效果下降)

  2. Margin 过滤: chosen 和 rejected 的分差要足够大
     太小 → 信号弱，学不到东西
     太大 → 可能是 trivial case（如 rejected 是乱码）

  3. Prompt 多样性: 覆盖各种任务类型和难度
     安全类 ~20%, 推理类 ~30%, 创作类 ~20%, 指令遵循 ~30%

  4. 避免长度偏差: 长回答容易被偏好 → 控制 chosen/rejected 长度接近

  5. Iterative DPO: 每轮 DPO 后用新模型重新采样 → 多轮迭代
     (参考 Self-Play Fine-Tuning / SPIN)

❌ 常见陷阱:
  1. 用 GPT-4 作为 chosen + 自家模型作为 rejected → 学到的是风格而非质量
  2. 偏好标注不一致 → 模型学到矛盾信号
  3. 数据量过少 (< 5K) → DPO 不稳定
```

### 4.3 2025 新趋势：规则驱动的偏好数据

```python
# 数学/代码任务: 规则验证代替人工标注
def math_preference_pair(prompt, sft_model, ground_truth):
    """基于正确性构造偏好对"""
    responses = sft_model.generate(prompt, n=8, temperature=0.8)

    correct = [r for r in responses if verify_math(r, ground_truth)]
    incorrect = [r for r in responses if not verify_math(r, ground_truth)]

    if correct and incorrect:
        return {
            "prompt": prompt,
            "chosen": random.choice(correct),
            "rejected": random.choice(incorrect),
        }
    return None  # 全对或全错则跳过
```

这种方法在 DeepSeek-R1、Qwen 2.5-Math 等模型中大量使用。参见 [[GRPO 深度解析]]。

## 5. 数据配比的实验方法论

```
Scaling Law for Data Mixing (Doremi, 2023):
  1. 训练一个小型代理模型 (proxy model)
  2. 在多个配比下测试
  3. 找到最优配比
  4. 将最优配比应用到大模型训练

实践经验:
  - 小模型 (1B) 的最优配比通常可迁移到大模型 (70B)
  - 但需要在 10B scale 上验证一次
  - 配比在训练过程中可动态调整 (curriculum learning)
```

## 面试常见问题

### Q1: 预训练数据清洗管线的关键步骤有哪些？每步的目标是什么？

标准管线：(1) **语言检测**（fasttext，去除非目标语言）→ (2) **URL/格式过滤**（去除成人网站、低质量域名）→ (3) **启发式清洗**（长度/重复行/特殊字符比例等规则）→ (4) **Model-based 质量过滤**（用小模型评分，FineWeb-Edu 方法可提升 MMLU 10+ 点）→ (5) **去重**（Exact→MinHash Fuzzy→段落级，去重是单项收益最大的步骤）→ (6) **PII 脱敏**（正则+NER 去除个人信息）。核心原则：**质量 > 数量**，去重后 1.4T tokens 优于未去重的 5T。

### Q2: MinHash 去重的原理是什么？为什么用 LSH？

MinHash 利用哈希函数估计两个集合的 **Jaccard 相似度**——对文档的 n-gram 集合，多个哈希函数各取最小值构成签名，两个签名中相同位置值相等的概率等于 Jaccard 相似度。但直接两两比较仍是 $O(N^2)$，所以用 **LSH** 将签名分段，任一段完全匹配就分入同一桶，只比较桶内文档，将复杂度降至近似 $O(N)$。阈值通常 0.8（太低误删太多，太高漏掉近似重复）。

### Q3: SFT 数据合成有哪些方法？如何保证质量？

主要方法：**Self-Instruct**（LLM 从种子指令自动扩展）、**Evol-Instruct**（WizardLM，渐进增加复杂度）、**Backtranslation**（从高质量回答反推指令）、**GPT-4 蒸馏**。质量保证：(1) 多维度 LLM-as-Judge 验证（准确性/完整性/格式/安全）；(2) 去重和多样性过滤；(3) 难度分层确保覆盖简单到复杂；(4) **LIMA 原则**：宁少勿多，1000 条高质量 > 100K 低质量。

### Q4: DPO 偏好数据为什么要 on-policy 采样？

因为 DPO 的损失函数隐式假设 chosen/rejected 来自**当前策略模型的分布**。如果用外部模型（如 GPT-4）作 chosen、当前模型作 rejected，模型学到的是模仿 GPT-4 的风格（分布偏移），而非改善自身的决策质量。On-policy 采样（用当前 SFT 模型生成多个回答再打分选对）确保训练信号在正确的分布上，DPO 的理论保证才成立。Iterative DPO 每轮更新模型后重新采样，进一步缓解分布偏移。

### Q5: 如何决定预训练数据的配比？

经验配比参考 LLaMA 3：Web~50%、代码~15%、学术~10%、书籍~8%、Wiki~5%、数学~5%。**科学方法**是用 **DoReMi** 框架——先在小模型上多配比实验，找 Pareto 最优，再迁移到大模型。关键发现：(1) 代码数据即使不做代码任务也能提升推理；(2) 高质量数据（Wiki/Books）可上采样 2-5 epochs；(3) 配比可动态调整（curriculum），如训练后期提高数学/代码比例增强 STEM。
