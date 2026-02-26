# Continue Pretraining（CPT）实操

> 基于 MA-RLHF 仓库整理，使用 HuggingFace + DeepSpeed 进行领域适配的继续预训练。
> 来源：`ma-rlhf/pretrained.py`, `ma-rlhf/text_book.py`, `scripts/run_7b_cpt.sh`, `scripts/prepare_dataset.sh`

---

## 1. CPT vs SFT 的核心区别

### 1.1 Loss 设计差异

```
CPT (Continue Pretraining):
  Input:  [领域文本全序列]
  Label:  [领域文本全序列]（向右移一位）
  Loss:   对所有 token 计算 CLM loss（无 mask）
  
SFT (Supervised Fine-Tuning):
  Input:  [System] [User Prompt] [Assistant Response]
  Label:  [-100]   [-100]        [Response Tokens]
  Loss:   只对 Response 部分计算 CLM loss（prompt 部分 mask 为 -100）
```

| 维度 | CPT | SFT |
|------|-----|-----|
| **Loss 目标** | 全序列 Causal LM Loss | Masked Loss（只算 response） |
| **数据格式** | 纯文本（无对话结构） | instruction-response 对 |
| **学习目标** | 吸收领域知识和分布 | 学习指令跟随和格式对齐 |
| **学习率** | 较小（2e-5~5e-5） | 中等（1e-5~5e-5） |
| **Epoch** | 1~3（避免遗忘） | 1~5 |
| **数据规模** | 通常较大（GB 级文本） | 较小（千~万条对话） |

### 1.2 为什么 CPT 不做 Mask？

CPT 的目标是让模型**理解领域文本的分布**，每个 token 都包含领域信息：
- 专业术语的上下文关系
- 领域特定的表达模式
- 文档结构和逻辑关系

如果做 mask，等于丢弃了大部分领域知识信号。

---

## 2. 数据准备

### 2.1 `prepare_dataset.sh`

```bash
# 调用 text_book.py 将原始文本转为 HuggingFace Dataset 格式
python ./../ma-rlhf/text_book.py
```

### 2.2 `text_book.py` 完整注解：文本 → Dataset

```python
from datasets import Dataset
import os

# === 配置 ===
folder_path = "./../dataset/med_qa_textbook"   # 原始文本文件夹
output_path = './../dataset/second_pretrained_datasets'  # 输出路径
chunk_size = 512     # 每个文本块的字符数
chunk_mode = True    # 是否启用分块

def split_string(text, chunk_size):
    """将长文本按固定字符数切分为多个块"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

texts = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

            if chunk_mode:
                # 分块模式：将长文档切分为 512 字符的块
                # 每个块作为一个训练样本
                text_chunk = split_string(text, chunk_size)
                texts.extend(text_chunk)       # extend 而非 append
                print(file_name, len(text), len(text_chunk))
            else:
                # 整篇模式：每个文件作为一个训练样本
                texts.append(text)

print('total:', len(texts))

# 转为 HuggingFace Dataset 格式并保存到磁盘
dataset = Dataset.from_dict({"text": texts})
dataset.save_to_disk(output_path)
```

### 2.3 分块策略详解

| 策略 | 方式 | 优点 | 缺点 |
|------|------|------|------|
| **固定字符分块** | 每 512 字符切一刀 | 简单、数据量充足 | 可能在句子中间截断 |
| 固定 token 分块 | 先 tokenize 再按 token 数切 | 与模型上下文对齐 | 需要 tokenizer |
| 句子边界分块 | 按句号/段落切分 | 语义完整 | 长度不均匀 |
| Packing | 多个短文本拼接到 `max_length` | GPU 利用率最高 | 需要注意 attention mask |

**本项目使用固定字符分块（512 字符）**，适合医学教材等结构化文本。

> ⚠️ **注意**：512 字符 ≠ 512 tokens。中文约 1 字符 ≈ 1~2 tokens，英文约 4~5 字符 ≈ 1 token。实际训练时 `seq_length` 参数控制 token 数上限。

---

## 3. 训练代码

### 3.1 `pretrained.py` 完整注解

```python
# === ma-rlhf/pretrained.py（CPT 核心代码）===
# 与 SFT 的关键区别：使用 packing=True + 纯文本，无 response mask

from trl import SFTTrainer, SFTConfig

def create_model_tokenizer(name):
    """QLoRA 量化加载——与 SFT 相同"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        use_flash_attention_2=True,    # CPT 用 Flash Attention 加速长序列
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def create_datasets(dataset_name, tokenizer):
    """加载纯文本数据集"""
    dataset = load_dataset(dataset_name, split="train", num_proc=32)
    # 注意：这里直接返回原始文本数据，不做 messages 格式化
    return dataset, None

def formatting_func(example):
    """CPT 的格式化极其简单——直接返回原始文本"""
    return example['text']
    # 对比 SFT：需要构造 "###System: ...\n###Question: ...\n###Answer: ..." 格式

def train():
    model, tokenizer = create_model_tokenizer(model_name)
    torch.distributed.barrier()  # 多进程同步

    tokenizer.pad_token = tokenizer.eos_token  # CPT 需要设置 pad token
    train_dataset, _ = create_datasets(dataset_name, tokenizer)
    torch.distributed.barrier()

    peft_config = create_peft(is_peft)  # QLoRA: r=32, alpha=8

    training_args = SFTConfig(
        output_dir=output_name,
        logging_steps=1,
        num_train_epochs=1,              # CPT 通常只跑 1 epoch 防止遗忘
        gradient_checkpointing=True,     # 节省显存
        bf16=True,                       # BF16 混合精度
        learning_rate=2e-5,              # 较小的学习率，保护预训练知识
        warmup_ratio=0.1,               # 10% warmup 防止初始震荡
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        deepspeed=deepspeed_config_name, # DeepSpeed ZeRO-1
        report_to='wandb',
        lr_scheduler_type='cosine',      # Cosine 衰减
        packing=True,                    # ← 关键！将多个短文本拼接为一个长序列
        max_steps=10,                    # 调试用（正式训练去掉）
        dataset_num_proc=16,             # 数据预处理并行度
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,  # 纯文本格式化
        # 注意：没有 data_collator！不使用 CompletionOnlyLM
        # → 全序列都计算 loss，这就是 CPT 和 SFT 的核心区别
    )

    trainer.train()
    trainer.save_model(output_name)
```

### 3.2 关键代码对比：CPT vs SFT

```python
# === CPT ===
SFTConfig(
    packing=True,           # 拼接文本，最大化 GPU 利用率
)
SFTTrainer(
    formatting_func=lambda x: x['text'],  # 纯文本
    # 无 data_collator → 全序列 loss
)

# === SFT ===
SFTConfig(
    packing=False,          # 不拼接，保持对话边界
)
collator = DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)
SFTTrainer(
    formatting_func=alpaca_format,  # 结构化格式
    data_collator=collator,         # response mask
)
```

---

## 4. 启动配置

### 4.1 `run_7b_cpt.sh` 参数解析

```bash
# === 路径配置 ===
base_model_path='meta-llama/Meta-Llama-3-8B'    # 基础模型（Llama-3 8B）
deepspeed_config_name=./config/ds.json           # DeepSpeed ZeRO-1 配置
output_path='./output'
model_pretrained_lora_path=${output_path}'/pretrained_lora'   # CPT LoRA 适配器输出
model_pretrained_full_path=${output_path}'/pretrained_full'   # 合并后的完整模型

# === Stage 1: Continue Pretraining ===
pt_dataset_name='stanfordnlp/imdb'    # 示例用 IMDB 数据集（实际替换为领域数据）
deepspeed ./ma-rlhf/pretrained.py \
    --dataset_name=${pt_dataset_name} \           # 数据集名称
    --model_name=${base_model_path} \             # 基础模型
    --seq_length=512 \                            # 最大序列长度（token 数）
    --batch_size=4 \                              # per-device batch size
    --output_name=${model_pretrained_lora_path} \ # LoRA 输出路径
    --use_QLora=True \                            # 使用 QLoRA（4-bit 量化 + LoRA）
    --use_flash_attention_2=True \                # Flash Attention 2 加速
    --deepspeed_config_name=${deepspeed_config_name} \  # DeepSpeed 配置
    --num_train_epochs=1                          # 1 个 epoch

# === Stage 2: 合并 LoRA 适配器 ===
# 将 Base Model + LoRA 适配器合并为完整模型
# 合并后可以直接用 from_pretrained 加载，无需 PEFT
python ./ma-rlhf/merge_adapter.py \
    --base_model_name=${base_model_path} \        # 原始 Base Model
    --model_name=${model_pretrained_lora_path} \  # LoRA 适配器路径
    --merged_model_name=${model_pretrained_full_path}  # 合并输出路径
```

### 4.2 参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset_name` | stanfordnlp/imdb | HuggingFace 数据集名（或本地路径） |
| `--model_name` | meta-llama/Meta-Llama-3-8B | 基础预训练模型 |
| `--seq_length` | 512 | token 级别的最大序列长度 |
| `--batch_size` | 4 | 每 GPU 的 batch size |
| `--use_QLora` | True | 使用 QLoRA 量化训练 |
| `--use_flash_attention_2` | True | 使用 Flash Attention 2 |
| `--num_train_epochs` | 1 | 训练轮数（CPT 建议 1~3） |

---

## 5. CPT 的适用场景

### 5.1 领域适配

| 场景 | 数据来源 | 数据量 | 效果 |
|------|----------|--------|------|
| **医疗** | 医学教材、论文、病历 | 1~10 GB | 提升医学术语理解和推理 |
| **法律** | 法律法规、判例文书 | 1~5 GB | 理解法律条文和逻辑 |
| **代码** | GitHub 代码、文档 | 10~100 GB | 提升编程能力（如 CodeLlama） |
| **金融** | 财报、研报、新闻 | 1~10 GB | 理解金融术语和逻辑 |

### 5.2 CPT vs 从头预训练

| 维度 | CPT | 从头预训练 |
|------|-----|-----------|
| **成本** | 少量 GPU 数小时~数天 | 数千 GPU 数月 |
| **数据需求** | 1~100 GB 领域文本 | TB 级通用文本 |
| **通用能力** | 保留大部分通用能力 | 从零开始 |
| **领域专业度** | 中高（取决于数据质量） | 高（但需要领域数据混入） |
| **风险** | 灾难性遗忘 | 无（但可能领域不足） |

### 5.3 CPT 的最佳实践

1. **小学习率**：2e-5 ~ 5e-5，过大会导致灾难性遗忘
2. **少 epoch**：1~3 epoch，多跑容易过拟合到领域数据
3. **混合数据**：将通用数据混入领域数据（如 70% 领域 + 30% 通用），减少遗忘
4. **Cosine 衰减**：逐步降低学习率，后期不要太激进
5. **QLoRA**：对于消费级 GPU，QLoRA 是 CPT 的实用选择（如 4090 跑 7B）
6. **LoRA 合并后再 SFT**：CPT(LoRA) → merge → SFT → GRPO 是推荐流水线

---

## 6. 面试考点

### Q1: CPT 和 SFT 的 Loss 区别？为什么 CPT 不做 mask？

**参考答案**：

**Loss 区别**：
- CPT 使用标准 Causal Language Modeling (CLM) loss，对序列中的**每个 token** 都计算交叉熵损失
- SFT 使用 Masked CLM loss，只对 **response 部分的 token** 计算损失，prompt 部分通过 `label=-100` 忽略

**为什么 CPT 不做 mask**：
CPT 的目标是让模型理解**整个领域文本的分布**——包括术语的搭配、段落的逻辑结构、专业表达习惯。每个 token 都承载领域信息，如果只选取部分 token 计算 loss（如只算段落最后一句），模型无法充分吸收领域知识。

**对比**：SFT 做 mask 是因为 prompt 是给定的输入条件，不是模型需要"学会生成"的内容。模型只需学会在给定 prompt 下生成正确的 response。

### Q2: CPT 中如何避免灾难性遗忘？

**参考答案**：

灾难性遗忘（Catastrophic Forgetting）是 CPT 的核心挑战——模型在学习领域知识的同时可能丢失通用能力。

**缓解策略**：

1. **小学习率**（2e-5）：减小参数更新幅度，降低对原始权重的扰动
2. **少 epoch**（1~3）：避免在领域数据上过度训练
3. **混合数据**：将通用语料（如 Wikipedia、常用问答）混入领域数据，保持通用知识
4. **LoRA / QLoRA**：只更新少量参数（<1% 总参数），原始权重冻结不变——这是最有效的防遗忘方法
5. **Cosine 学习率衰减**：后期学习率接近 0，减少训练末期的参数漂移
6. **Replay 缓冲区**：在训练过程中周期性回放通用数据
7. **正则化**：L2 正则或 EWC（Elastic Weight Consolidation）限制参数偏移

**工程实践**：本项目采用 QLoRA + 小学习率 + 1 epoch 的组合，在消费级 GPU 上实现低成本领域适配。

---

## 附录

### A. 完整训练流水线

```bash
# Step 1: 数据准备
python text_book.py                    # 原始文本 → HuggingFace Dataset

# Step 2: CPT
deepspeed pretrained.py \
    --dataset_name=领域数据 \
    --model_name=meta-llama/Meta-Llama-3-8B \
    --use_QLora=True \
    --num_train_epochs=1

# Step 3: 合并 LoRA
python merge_adapter.py \
    --base_model_name=meta-llama/Meta-Llama-3-8B \
    --model_name=./output/pretrained_lora \
    --merged_model_name=./output/pretrained_full

# Step 4: SFT（在合并后的模型上）
deepspeed sft.py \
    --model_name=./output/pretrained_full \
    --dataset_name=指令数据 \
    ...

# Step 5: GRPO（可选，需 verl）
# 见 verl-GRPO-实战指南.md
```

### B. 文件路径速查

| 文件 | 作用 |
|------|------|
| `ma-rlhf/pretrained.py` | CPT 训练代码（QLoRA + TRL SFTTrainer） |
| `ma-rlhf/text_book.py` | 原始文本 → HuggingFace Dataset 转换 |
| `scripts/run_7b_cpt.sh` | Llama-3 8B CPT 启动脚本 |
| `scripts/prepare_dataset.sh` | 数据准备入口 |
| `ma-rlhf/merge_adapter.py` | LoRA 适配器合并工具 |
| `config/ds.json` | DeepSpeed ZeRO-1 配置 |
