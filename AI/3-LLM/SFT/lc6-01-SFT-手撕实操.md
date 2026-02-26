---
title: SFT 手撕实操
brief: 监督微调（SFT）完整实现：Chat Template格式/数据Collator/Loss Mask（只计算response部分）/LoRA微调配置（QLoRA 4-bit），Alpaca/ShareGPT两种数据格式处理，是RLHF全流水线的第一阶段，来源 MA-RLHF 教学项目。
date: 2026-02-25
type: code-practice
source: MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
tags:
  - code-practice
  - sft
  - supervised-finetuning
  - lora
  - qlora
  - pytorch
related:
  - "[[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/3-LLM/SFT/LoRA|LoRA]]"
  - "[[AI/3-LLM/RL/PPO/MA-RLHF-核心代码注解|MA-RLHF-核心代码注解]]"
---

# SFT 手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

SFT（Supervised Fine-Tuning）是 LLM Post-Training 的第一步，通过在指令数据上有监督微调，使预训练模型获得**指令跟随能力**——从"复读机"变为能正常对话、预测 EOS 的模型。

核心流程：
1. **数据构建**：将任务数据转化为 QA/Messages 格式，构造 `input_ids` + `labels`（仅拟合 Assistant 回答部分）
2. **Chat Template**：通过对话模板格式化多轮对话，标记哪些 token 是 label
3. **训练**：使用 Cross-Entropy Loss，`ignore_index=-100` 跳过非回答部分
4. **参数高效微调**：LoRA 在原模型各层附加低秩旁路，大幅减少可训练参数

## 二、核心实现

### 2.1 SFT 数据构建与对话模板

**原理**：SFT 数据为 QA 对（支持多轮对话），使用 Chat Template 将数据格式化为 token 序列。关键是**先对各 message 做 tokenize，再拼接 token id**，避免 special token 被错误编码。

**代码**：

```python
DEFINED_EOS_TOKEN = '<|im_end|>'
DEFINED_SOS_TOKEN = '<|im_start|>'

def ChatTemplateToken(example, tokenizer):
    sos_token_id = tokenizer(DEFINED_SOS_TOKEN).input_ids[0]
    eos_token_id = tokenizer(DEFINED_EOS_TOKEN).input_ids[0]
    
    input_ids = [sos_token_id]
    is_labels = [0]
    
    for i, item in enumerate(example):
        if item['role'] == 'ASSISTANT':
            prompt = '\n#' + item['role'] + ':'
            content_prompt = item['content']
            prompt_token_ids = tokenizer(prompt).input_ids
            content_prompt = tokenizer(content_prompt).input_ids
            # ASSISTANT 的 role tag 不算 label，content + EOS 才是
            is_labels += [0]*len(prompt_token_ids) + [1]*len(content_prompt) + [1]
            input_ids += prompt_token_ids + content_prompt + [eos_token_id]
        else:
            prompt = '\n#' + item['role'] + ':' + item['content']
            prompt_token_ids = tokenizer(prompt).input_ids
            input_ids += prompt_token_ids
            is_labels += [0]*len(prompt_token_ids)
    return input_ids, is_labels
```

**关键洞察**：
- **方式1（先拼接字符串再 tokenize）不安全**：special token 可能被拆分编码（如 `<|box_start|>` 不编码成单个 token id）
- **方式2（先 tokenize 再拼接 id）是正确做法**
- EOS 一定要标记为 label，否则模型生成时无法停止

### 2.2 Collate Function 与 Label 构造

**原理**：不同长度的序列需要 padding 到同一长度。Labels 的构造遵循 next-token prediction：label 左移一位，非回答部分填 `-100`。

**代码**：

```python
def paddding_collate_fn(batch_data, pad_token_id=None, ignore_index=-100):
    input_lens = []
    bs = len(batch_data)
    for data in batch_data:
        input_lens.append(data['input_ids'].shape[0])
    max_input_len = torch.max(torch.tensor(input_lens, dtype=torch.long))
    
    # Right Padding
    input_ids = torch.ones(bs, max_input_len, dtype=torch.long) * pad_token_id
    attention_masks = torch.zeros(bs, max_input_len, dtype=torch.long)
    labels = torch.ones(bs, max_input_len, dtype=torch.long) * ignore_index

    for i in range(bs):
        input_ids[i, :input_lens[i]] = batch_data[i]['input_ids']
        attention_masks[i, :input_lens[i]] = 1
        # label 左移：is_label[k]=1 的位置，对应的 label 是 input_ids[k]
        idx = torch.where(batch_data[i]['is_label'] != 0)[0]
        labels[i, idx-1] = batch_data[i]['input_ids'][idx]
    return {'input_ids': input_ids, 'attention_masks': attention_masks, 'labels': labels}
```

**关键洞察**：
- Label 的本质：`labels[i, idx-1] = input_ids[i, idx]`——位置 k 的 logits 要预测位置 k+1 的 token
- `CrossEntropyLoss(ignore_index=-100)` 自动跳过非回答部分

### 2.3 PyTorch 手撕 SFT 训练

**代码**：

```python
learning_rate = 1e-5
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(epochs):
    for batch in dataloader:
        optim.zero_grad()
        bsz, seq_len = batch['input_ids'].shape
        output = model(input_ids=batch['input_ids'])
        logits = output.logits
        loss = loss_fn(logits.view(bsz*seq_len, vocab_size),
                       batch['labels'].view(bsz*seq_len))
        loss.backward()
        optim.step()
```

### 2.4 TRL SFTTrainer 调包训练

**原理**：HuggingFace TRL 库封装了 SFT 全流程，支持 messages 格式数据直接训练。

**代码**：

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

config = SFTConfig(
    output_dir="output/qwen3_sft",
    per_device_train_batch_size=2,
    max_length=256,
    max_steps=10
)
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=config,
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

**关键洞察**：
- TRL 的 `DataCollatorForCompletionOnlyLM` 自动定位 response 部分构造 labels
- 使用 `formatting_func` 将自定义数据集转换为训练格式
- 注意 `response_template` 的 token id 必须精确匹配，否则 label 构造错误

### 2.5 LoRA 原理与实现

**原理**：将模型参数更新 $\Delta W$ 参数化为低秩分解 $W_A W_B$，其中 $W_A \in \mathbb{R}^{d_{in} \times r}$，$W_B \in \mathbb{R}^{r \times d_{out}}$，$r \ll d$。前向计算变为：

$$h' = XW + \alpha X W_A W_B$$

**为什么 LoRA 能 work**：
1. 大模型参数矩阵具有**低秩性**——SVD 分解后前几个奇异值显著大于其余
2. 越充分训练的模型，其梯度矩阵的低秩性越强
3. $r=1$ 仍能训练出效果，$r$ 增长到一定程度性能收敛

**代码**：

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.dim_in = original_linear.in_features
        self.dim_out = original_linear.out_features
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        
        self.WA = nn.Linear(self.dim_in, rank, bias=False)
        self.WB = nn.Linear(rank, self.dim_out, bias=False)
        # 初始化：WB=0 保证旁路初始输出为零
        nn.init.constant_(self.WB.weight.data, 0.0)

    def forward(self, X):
        bsz, seq_len, dim_in = X.shape
        h = X.reshape(bsz*seq_len, dim_in) @ self.weight
        h = h.reshape(bsz, seq_len, self.dim_out)
        h_lora = self.alpha * self.WB(self.WA(X))
        return h + h_lora
```

**LoRA 初始化策略**：
- 全零初始化 (WA=0, WB=0)：无梯度，不可行
- WA=0, WB≠0 或 WA≠0, WB=0：均可保证旁路输出为零
- **WB=0 更优**：因为 WB 的输入 $X W_A$ 不为零，更容易学习

**LoRA Merge**：部署时将 LoRA 参数融合到原参数：$W' = W + W_A W_B$

### 2.6 SFT 数据集构建实践

**原理**：混合多来源数据集（Alpaca、ChatAlpaca、hh-rlhf、选择题、数学、代码），统一为 Alpaca 格式 `{instruction, input, output}`。

**代码**（数据格式转换示例）：

```python
# hh-rlhf → Alpaca 格式
def format_hhrlhf_to_alpaca(example):
    text = example["chosen"]
    text = re.sub(r'\n\nHuman:', '\n###Question:', text)
    text = re.sub(r'\n\nAssistant:', '\n###Answer:', text)
    instruction = text.rsplit('\n###Answer:',1)[0].split('###Question: ',1)[1]
    output = text.rsplit('\n###Answer: ',1)[1]
    example['instruction'] = instruction
    example['input'] = ''
    example['output'] = output
    return example

# 合并与保存
dataset = concatenate_datasets([dataset1['train'], dataset2['train'], ...])
dataset = DatasetDict({'train': dataset}).shuffle(seed=42)
dataset.save_to_disk('./output/awesome-sft')
```

**偏好数据集构建**（用于 DPO/RM）：
- CValues_DPO、hh-rlhf、PKU-SafeRLHF、Chinese-dpo-pairs
- 对 SafeRLHF 需处理 safety 标签：safe > unsafe，both safe 选 better

## 三、工程实践（配套代码）

> 完整代码见：`/tmp/ma-rlhf/ma-rlhf/sft.py` 和 `sft_pack.py`

### sft.py 关键架构

```python
def train():
    model, tokenizer = create_model_tokenizer(model_name)  # 支持 QLoRA 量化加载
    datasets, _ = create_datasets(dataset_name, dataset_sub_name)
    collator = create_collator(tokenizer)  # DataCollatorForCompletionOnlyLM
    peft_config = create_peft(is_peft)     # LoRA 配置
    
    trainer = SFTTrainer(
        model, args=training_args,
        train_dataset=train_datasets,
        peft_config=peft_config,
        data_collator=collator,
        formatting_func=format_fun,  # Alpaca 格式化
    )
    trainer.train()
```

### sft_pack.py（Packing 策略）

使用 `ConstantLengthDataset` 将多条短序列打包到固定长度，减少 padding 浪费：

```python
train_dataset = ConstantLengthDataset(
    tokenizer, train_dataset,
    formatting_func=format_func,
    seq_length=seq_length, shuffle=True,
)
```

## 四、关键洞察与总结

1. **数据是 SFT 的核心**：Chat Template 构造、label 标记、padding 策略都需要精确处理
2. **Tokenizer 有坑**：special token 可能不编码成单个 id；先 tokenize 再拼接是安全做法
3. **LoRA 的本质**：在全参空间中找到一个低秩子空间做梯度搜索，保留原模型能力的同时适配新任务
4. **LoRA vs 全参微调**：LoRA 适合差数据/少数据，遗忘少；全参适合好数据，搜索空间大但易遗忘
5. **工程实践**：实际 SFT 需要混合多数据源、规则过滤（长度/质量）、Packing 减少 padding
6. **预训练模型不能对话 ≠ 模型差**：预训练模型只是没学过对话格式和 EOS 预测，SFT 赋予了指令跟随能力

### 思考题

- 数据长短方差大时如何减少 padding？→ Packing 策略
- TRL SFTTrainer 对多轮对话是 fitting 每轮回答还是最后一轮？→ 每轮 Assistant 回答
- 如何不用 transformers 的 generate，手写带 KVCache 的 generate？
- Trainer 源码流程是什么？→ 建议阅读源码画流程图
