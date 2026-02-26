---
title: SFT 全链路 PyTorch 手撕实操 · MA-RLHF lc6
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
brief: SFT 全链路手撕实操：Chat Template 格式化 → Loss Mask → CrossEntropyLoss 实现 → HuggingFace TRL SFTTrainer；对比预训练 Loss（无 Mask）和 SFT Loss（只计算 ASSISTANT token）；PyTorch 纯手写 + transformers 两路径，Qwen3 为示例模型。
tags:
  - sft
  - fine-tuning
  - loss-mask
  - chat-template
  - pytorch
  - transformers
  - trl
  - qwen
  - code-practice
  - interview-core
rating: ★★★★★
related:
  - "[[LoRA|LoRA]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写|lc8-DPO-IPO-BT]]"
  - "[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[AI/LLM/SFT/SFT-手撕实操|SFT-手撕实操]]"
---

# SFT 全链路 PyTorch 手撕实操

> MA-RLHF 课程 lc6_sft | 覆盖 notebook: Supervised_Finetuning_PyTorch + Supervised_FineTuning_transformers_Qwen3
> 评级: ★★★★★ | 面试核心 | 2026-02-26

---

## 核心目标

把预训练模型（"复读机"）变成能正确跟随指令的 SFT 模型。

**为什么预训练模型不能对话？**
- 预训练数据不是每条都有 `<EOS>`，模型不学停止
- 没有角色格式——不知道什么是"用户问"，什么是"AI答"
- 不代表预训练模型弱（它是一切能力的来源）

---

## 一、数据流水线（手撕版 PyTorch）

### 1. 数据格式：Alpaca 三字段

```python
{
  'instruction': '如何学习python?',
  'input': '',   # 可空
  'output': '学Python先...'
}
```

### 2. Chat Template Token 化

SFT 的关键：**只对 ASSISTANT 回复部分计算 loss**，其他位置用 `ignore_index=-100` 屏蔽。

```python
DEFINED_EOS_TOKEN = '<|im_end|>'
DEFINED_SOS_TOKEN = '<|im_start|>'
DEFINED_PAD_TOKEN = '<|endoftext|>'

def ChatTemplateToken(example, tokenizer):
    sos_token_id = tokenizer(DEFINED_SOS_TOKEN).input_ids[0]
    eos_token_id = tokenizer(DEFINED_EOS_TOKEN).input_ids[0]
    
    input_ids = [sos_token_id]
    is_labels = [0]  # 0=不计算loss, 1=计算loss
    
    for item in example:
        if item['role'] == 'ASSISTANT':
            prompt = '\n#ASSISTANT:'
            content = item['content']
            prompt_ids = tokenizer(prompt).input_ids
            content_ids = tokenizer(content).input_ids
            
            # 角色标记不计loss，内容+EOS计loss
            is_labels += [0]*len(prompt_ids) + [1]*len(content_ids) + [1]
            input_ids += prompt_ids + content_ids + [eos_token_id]
        else:
            prompt = '\n#' + item['role'] + ':' + item['content']
            prompt_ids = tokenizer(prompt).input_ids
            input_ids += prompt_ids
            is_labels += [0]*len(prompt_ids)
    
    return input_ids, is_labels
```

**关键设计点**：
- `is_labels` 是 binary mask，1 表示这个 token 要计算 loss
- EOS token 也标记为 1：教模型学会"何时停止"
- SOS 只加一个，不重复

### 3. Dataset 包装

```python
class TokenSFTDataset(Dataset):
    def __init__(self, messages_list, tokenizer):
        data_list = [ChatTemplateToken(msg, tokenizer) for msg in messages_list]
        self.data = [
            [torch.tensor(d[0], dtype=torch.long),   # input_ids
             torch.tensor(d[1], dtype=torch.long)]    # is_label
            for d in data_list
        ]
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx][0], 'is_label': self.data[idx][1]}
```

### 4. Collate 函数：Right Padding + Label 映射

```python
def padding_collate_fn(batch_data, pad_token_id=None, ignore_index=-100):
    bs = len(batch_data)
    max_input_len = max(d['input_ids'].shape[0] for d in batch_data)
    
    # 初始化：input_ids 填 pad，labels 填 ignore_index
    input_ids = torch.ones(bs, max_input_len, dtype=torch.long) * pad_token_id
    attention_masks = torch.zeros(bs, max_input_len, dtype=torch.long)
    labels = torch.ones(bs, max_input_len, dtype=torch.long) * ignore_index
    
    for i in range(bs):
        l = batch_data[i]['input_ids'].shape[0]
        input_ids[i, :l] = batch_data[i]['input_ids']
        attention_masks[i, :l] = 1
        
        # is_label==1 的位置就是 label token 的位置（input_ids 里的值）
        idx = torch.where(batch_data[i]['is_label'] != 0)[0]
        labels[i, idx-1] = batch_data[i]['input_ids'][idx]
        # ⚠️ 注意：idx-1 是因为 labels 是 input_ids 的左移版本
        # input_ids[t] 是 token t，labels[t-1] = input_ids[t] 意味着"预测下一个"
    
    return {'input_ids': input_ids, 'attention_masks': attention_masks, 'labels': labels}
```

**为什么 `labels[i, idx-1] = input_ids[idx]`？**
- LM 是自回归：给 token[0..t-1]，预测 token[t]
- labels[t] 是 input_ids[t+1] 的 ground truth
- 所以 ASSISTANT 的第一个 token（位置 idx[0]）的 label 在 idx[0]-1 处

---

## 二、训练循环

```python
learning_rate = 1e-5
epochs = 1
grad_accumulative = 10

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # 自动忽略 -100 位置

for epoch in range(epochs):
    for k, batch in enumerate(dataloader):
        optimizer.zero_grad()
        bsz, seq_len = batch['input_ids'].shape
        
        output = model(input_ids=batch['input_ids'],
                       attention_masks=batch['attention_masks'])
        logits = output.logits  # [bsz, seq_len, vocab_size]
        
        # reshape for cross entropy
        loss = loss_fn(logits.view(bsz*seq_len, vocab_size),
                       batch['labels'].view(bsz*seq_len))
        
        loss.backward()
        optimizer.step()
```

**注意事项**：
- `CrossEntropyLoss(ignore_index=-100)`：不对 padding 和 prompt token 计算 loss
- lr=1e-5 较小：全参微调防止遗忘
- 实际生产中需加梯度裁剪 (`clip_grad_norm_`) 和 warmup

---

## 三、推理：带 Chat Template 的 Generate

```python
def generate(model, tokenizer, prompt, max_new_tokens=128):
    messages_inst = [
        {'role': 'SYSTEM', 'content': SYSTEM_PROMPT},
        {'role': 'USER', 'content': prompt},
        {'role': 'ASSISTANT', 'content': ''},  # 空，触发生成
    ]
    input_ids, is_label = ChatTemplateToken(messages_inst, tokenizer)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids[:, :-1]  # 去掉 EOS，让模型从这里继续生成
    
    output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

---

## 四、进阶版：Qwen3 官方 Chat Template 流水线

### 数据处理差异

官方模板更规范，使用 `tokenizer.apply_chat_template()`：

```python
def map_apply_chat_template(example):
    tmp_messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': example['prompt']},
        {'role': 'assistant', 'content': example['completion']},
    ]
    example['text'] = tokenizer.apply_chat_template(
        tmp_messages, tokenize=False
    )
    return example
```

Qwen3 特殊 token：
- `<|im_start|>` = 对话开始标记
- `<|im_end|>` = 151645（EOS）
- `<|endoftext|>` = 151643（PAD）
- `<think>` = 151667（推理开始标记，用于区分 completion 起始）

### Label 定位（Qwen3 关键）

```python
def find_completion_start_end(token_ids):
    """找到 assistant 回复的 <think> token 位置作为 label 起始"""
    start = -1
    for i in range(len(token_ids)-1, -1, -1):
        if token_ids[i] == 151667:  # <think> token id
            start = i
            break
    end = len(token_ids) - 1
    return start, end
```

**设计哲学**：以 `<think>` 为 completion 起始，不是 `<|im_start|>assistant`，因为 Qwen3 在 assistant 角色标记后立即有一个思考结构 `<think>\n\n</think>\n\n`，这是推理 token，属于 ASSISTANT 的输出内容。

### 解决 labels padding 问题

`DataCollatorWithPadding` 不处理 `labels`，三种解法：
1. **去掉 labels，重构 loss**（麻烦）
2. **继承 `DataCollatorWithPadding` 加 labels padding**（推荐）
3. **手写 Collator**（最灵活）

```python
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [feature.pop('labels') for feature in features]
        batch = super().__call__(features)  # 处理 input_ids, attention_mask
        
        bsz, seq_len = batch['input_ids'].shape
        padding_labels = torch.ones(bsz, seq_len, dtype=torch.long) * -100
        for i in range(bsz):
            tmp_len = len(labels[i])
            if self.tokenizer.padding_side == 'right':
                padding_labels[i, :tmp_len] = torch.tensor(labels[i])
        
        batch['labels'] = padding_labels
        return batch
```

### TRL SFTTrainer（最省力）

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="output/qwen3_sft",
    per_device_train_batch_size=2,
    max_length=256,
    max_steps=10
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=config,
    train_dataset=dataset_alpaca['train']  # messages 格式
)
trainer.train()
```

数据必须是 `messages` 格式（含 role/content 字典列表），SFTTrainer 自动处理 chat template 和 label masking。

---

## 五、工程细节 & 面试要点

### Label Shift 原理

```
input_ids:  [SOS, t1, t2, ..., tm, EOS, PAD]
labels:     [-100, -100, ..., tm, EOS, -100, -100]
            
cross_entropy: 位置 i 的 logits 预测 labels[i]
= 用 input_ids[0..i] 预测 input_ids[i+1]
```

### Right Padding vs Left Padding

- 训练用 **Right Padding**（标准）：序列靠左，padding 在右
- 推理（batch generate）用 **Left Padding**：padding 在左，EOS 对齐，generation 从同一位置开始
- KV Cache 的 `attention_mask` 必须与 padding_side 一致

### 数据过滤

```python
dataset_sft_filter = dataset_sft.filter(lambda x: len(x["input_ids"]) < config_max_len)
```

过滤超长序列：防止显存 OOM + 减少 padding 浪费

### Gradient Accumulation

`grad_accumulative = 10` 时：每 10 步才 `optimizer.step()`，等效 batch size × 10，但不调用 `zero_grad()`。

### 思考题答案（课程精华）

**Q: 预训练模型很小（0.6B），数据很少，SFT 为什么通用性好？**
→ 预训练已经学会语言知识，SFT 只是"激活"指令跟随能力，不需要从头学语义。数据量决定**行为形式**，不决定**知识量**。

**Q: 如何减少 padding？**
→ 数据 packing：把多条短序列拼接成一条长序列（用特殊 token 分隔），减少 padding 浪费；同时需要修改 attention mask 为块对角结构防止跨序列 attention。

**Q: SFTTrainer 对多轮对话，fit 每轮还是最后一轮？**
→ 默认 fit 所有 ASSISTANT 轮次（完整对话 label masking）。

---

## 六、知识连接

| 概念 | SFT 版本 | RL 版本（PPO/GRPO）|
|------|---------|---------|
| Label | ASSISTANT 回复 token | 由 reward 决定哪些 token 有正 advantage |
| Loss mask | is_label / ignore_index | 同，但 advantage 加权 |
| 数据来源 | 人工标注示例 | 模型自身 rollout |
| 目标 | 模仿行为 | 最大化期望奖励 |

SFT → DPO（偏好对比）→ PPO/GRPO（RL）：能力边界逐渐拓宽，但数据要求和工程复杂度也在上升。

---

## 七、文件索引

| 类型 | 路径 |
|------|------|
| 手撕 PyTorch SFT | `lecture/lc6_sft/Supervised_Finetuning_PyTorch.ipynb` |
| Qwen3 + TRL SFT | `lecture/lc6_sft/Supervised_FineTuning_transformers_Qwen3.ipynb` |
| 数据集构造 | `lecture/lc6_sft/Supervised_Finetuning_Dataset.ipynb` |
| LoRA 微调 | `lecture/lc6_sft/LoRA.ipynb` → 见 `lc6-LoRA-手撕实操.md` |

---

## See Also

- [[AI/LLM/MA-RLHF课程/lc6-LoRA-手撕实操|lc6-LoRA 手撕实操]] — SFT 全链路之后的 LoRA 低秩微调完整实现（本笔记 Step 2 → 该笔记 Step 4）
- [[AI/LLM/MA-RLHF课程/lc6-SFT全链路-MOC|lc6 SFT 全链路专题地图]] — 本笔记所属的课程 MOC，含 Step 1-8 完整学习路径
- [[LoRA|LoRA 深度理解]] — LoRA 原理 + 数学推导 + rank 选择理论依据（本实操的理论对应）
- [[SFT-实战指南|SFT 实战指南]] — 工程最佳实践（超参 / 数据质量 / 过拟合诊断）
- [[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO 手撕实操]] — SFT 之后的 RLHF 阶段手撕（Loss Mask 概念延续到 advantage mask）
- [[MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]] — 完整系列入口

## 推荐阅读

- 原课程代码：[MA-RLHF/lecture/lc6_sft](https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc6_sft)
- [[SFT 原理|SFT 原理]] — Chat Template / Loss Mask 的理论背景
- [[AI/LLM/Frameworks/TRL/SFT实践|TRL SFT 实践]] — HuggingFace TRL SFTTrainer 生产用法
