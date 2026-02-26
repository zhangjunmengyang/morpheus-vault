---
title: MA-RLHF 核心代码注解
brief: MA-RLHF开源项目（github.com/dhcode-cpp/MA-RLHF）完整代码注解：SFT→Reward Model→PPO/MA-PPO全流水线，多适配器PPO Trainer逐行解析（与trl PPOTrainer的关键差异），是本批次手撕实操笔记的项目总览，来源 MA-RLHF 教学项目。
date: 2026-02-25
type: code-practice
source: MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
tags:
  - code-practice
  - ma-rlhf
  - ppo
  - lora
  - multi-adapter
  - code-walkthrough
related:
  - "[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[LoRA|LoRA]]"
  - "[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]]"
---

# MA-RLHF 核心训练代码注解

> 来源：MA-RLHF (<https://github.com/dhcode-cpp/MA-RLHF>)
> 入库日期：2026-02-25

---

## 一、项目结构速览

```
MA-RLHF/
├── ma-rlhf/
│   ├── utils.py                 # 公共工具：QLoRA 模型构建、LoRA 配置工厂、prompt 模板、数据 collator
│   ├── sft.py                   # Stage 1: SFT 监督微调
│   ├── reward_model.py          # Stage 2: Reward Model 训练（序列分类头）
│   ├── ppo.py                   # Stage 3a: 标准 PPO（trl>=0.11，不支持多适配器）
│   ├── ppo_multi_adapter.py     # Stage 3b: 多适配器 PPO（核心创新，基于 AutoModelForCausalLMWithValueHead）
│   ├── ma_ppo_config.py         # 多适配器 PPO 的配置 dataclass
│   ├── ma_ppo_trainer.py        # 多适配器 PPO Trainer 完整实现（从 trl PPOTrainer 重写）
│   ├── dpo.py                   # 替代路线：DPO 对齐训练
│   ├── generate.py              # 推理/生成脚本
│   └── merge_adapter.py         # LoRA 适配器合并到基座模型
├── scripts/
│   ├── run_7b_ppo.sh            # 标准 PPO 全流程脚本
│   └── run_7b_ma_ppo.sh         # 多适配器 PPO 全流程脚本
└── config/
    └── ds.json                  # DeepSpeed ZeRO 配置
```

**核心流水线**：SFT → DPO（可选预对齐）→ Reward Model → PPO / MA-PPO → Merge → Generate

**关键设计**：所有阶段均支持 QLoRA（4-bit NF4 量化 + LoRA），使得 7B 模型可在 4×3090（24GB）上完成全流程 RLHF。

---

## 二、SFT 训练实现（sft.py）

### 2.1 模型初始化：QLoRA + Flash Attention

```python
def create_model_tokenizer(name):
    # QLoRA 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = {"": Accelerator().local_process_index}  # 每个进程映射到对应 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if is_peft else None,
        device_map=device_map,
        use_flash_attention_2=use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # gradient checkpointing 要求关闭 KV cache
    )
    model.gradient_checkpointing_enable()  # 用时间换显存
    # ...
```

**注解**：
- `device_map = {"": local_process_index}` 是多 GPU QLoRA 的标准写法——将整个模型放到当前进程对应的 GPU
- `use_cache=False` 是 gradient checkpointing 的必要条件（否则激活缓存冲突）
- 非 QLoRA 时 `quantization_config=None`，走全精度/bf16 路径

### 2.2 数据处理：Completion-Only Loss

```python
def create_collator(tokenizer):
    """只对 Answer 部分计算 loss，Question 部分 mask 掉"""
    response_template = "###Answer:"
    response_template_id = tokenizer.encode(response_template, add_special_tokens=False)[1:]
    return DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)
```

**注解**：使用 trl 的 `DataCollatorForCompletionOnlyLM`，自动将 `###Answer:` 之前的 token 标签设为 `-100`，只在回答部分计算交叉熵损失。`[1:]` 跳过首 token 是因为 tokenizer 会在开头添加特殊前缀。

### 2.3 训练配置

```python
training_args = SFTConfig(
    gradient_checkpointing=True,  # 节省~50%显存
    bf16=True,                    # 混合精度
    lr_scheduler_type='cosine',   # 余弦退火
    warmup_ratio=0.1,
    deepspeed=deepspeed_config_name,
    # ...
)
trainer = SFTTrainer(
    model, args=training_args,
    train_dataset=train_datasets,
    peft_config=peft_config,       # LoRA r=64, alpha=8
    data_collator=collator,
    formatting_func=format_fun,    # 批量格式化 alpaca 格式
)
```

**注解**：SFT 阶段使用 alpaca 格式 `###System: ... ###Question: ... ###Answer: ...`，配合 `DataCollatorForCompletionOnlyLM` 实现 completion-only loss。

---

## 三、Reward Model（reward_model.py）

### 3.1 模型结构

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # QLoRA 4-bit
    num_labels=1,                     # 单标量奖励输出
    use_cache=False,
)
```

**注解**：基于 SFT 后的模型（非原始基座），加一个序列分类头（`num_labels=1`）输出标量奖励分数。模型结构 = LLM backbone + linear head (hidden_size → 1)。

### 3.2 LoRA 配置：SEQ_CLS 任务类型

```python
def create_peft_reward_model(peft_flag=False):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 序列分类任务（非 CAUSAL_LM）
        r=32,                         # rank 32（比 SFT 的 64 小）
        lora_alpha=8,
        lora_dropout=0.05,
        modules_to_save=["scores"],   # 关键：分类头 scores 层必须全量训练
    )
```

**注解**：`modules_to_save=["scores"]` 确保新增的分类头权重被完整保存和训练（非 LoRA），否则随机初始化的 head 无法收敛。

### 3.3 偏好数据处理（多数据集适配）

```python
# PKU-SafeRLHF 数据集：综合安全性和质量偏好
if safe_0 == True and safe_1 == True:
    response_chosen = response_0 if better_id == 0 else response_1
    response_rejected = response_1 if better_id == 0 else response_0
elif safe_0 == True and safe_1 == False:
    response_chosen = response_0     # 安全的一定优于不安全的
    response_rejected = response_1
# ...
```

**注解**：支持三种偏好数据格式：
1. **通用格式**：`question / response_chosen / response_rejected`
2. **Anthropic hh-rlhf**：`chosen / rejected`（需正则替换 `\n\nHuman:` → `\n###Question:`）
3. **PKU-SafeRLHF**：双维度偏好（安全性优先于质量）

### 3.4 训练

```python
trainer = RewardTrainer(
    model, args=reward_config,
    train_dataset=train_datasets,
    processing_class=tokenizer,
    peft_config=peft_config,  # QLoRA
)
```

**注解**：trl 的 `RewardTrainer` 内部自动计算 pairwise ranking loss：`loss = -log(σ(r_chosen - r_rejected))`。

---

## 四、标准 PPO 实现（ppo.py）

> **注意**：此文件使用 trl>=0.11 的新版 `PPOTrainer`，**不支持多适配器**。四模型分离加载。

### 4.1 四模型架构

```python
# Policy（Actor）: 生成模型
policy_model = AutoModelForCausalLM.from_pretrained(name)
# Value（Critic）: 奖励预测（与 RM 相同结构）
value_model = AutoModelForSequenceClassification.from_pretrained(rm_model_name, num_labels=1)
# Reward Model: 冻结，提供奖励信号
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_model_name, num_labels=1)
# Reference Model: 冻结，提供 KL 基线
if peft_config is None:
    ref_model = AutoModelForCausalLM.from_pretrained(name)
else:
    ref_model = None  # LoRA 模式下共享基座参数，disable adapter 即为 ref
```

**注解**：
- **标准 RLHF 需要 4 个模型同时在显存中**——这是 RLHF 显存瓶颈的根源
- LoRA 模式下 ref_model=None，因为基座参数冻结，关闭 adapter 等价于 ref model
- Value Model 和 Reward Model 从同一个 RM checkpoint 初始化（但 value 会被训练更新）

### 4.2 PPO 配置

```python
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=batch_size,         # 16
    mini_batch_size=mini_batch_size, # 1（显存不足时逐样本更新）
    num_ppo_epochs=ppo_epochs,     # 1（每个 batch 只过一次 PPO）
    max_grad_norm=1.0,             # 梯度裁剪防止 NaN
    # ...
)
```

### 4.3 保存模型的 hack

```python
# 训练时需修改 trl `ppo_trainer.py` 的 save_model 函数：
# self.model = self.model.policy         # ← 原始（单 GPU）
# self.model = self.model.module.policy  # ← 修改后（多 GPU，需 unwrap DDP）
trainer.save_model(output_name)
```

**注解**：这是 trl 新版 PPOTrainer 在 DeepSpeed/DDP 下的已知问题，需要手动 unwrap `module`。

---

## 五、MA-PPO：多适配器 PPO（核心创新）

> 文件：`ma_ppo_config.py` + `ma_ppo_trainer.py` + `ppo_multi_adapter.py`

### 5.1 设计思路

标准 PPO 需要 4 个独立模型（Actor/Critic/Ref/Reward），7B 参数量 × 4 = 显存灾难。

**MA-PPO 的核心创新**：一个基座模型 + 多个 LoRA 适配器

```
┌─────────────────────────────────┐
│      共享基座模型（QLoRA 4-bit冻结）  │
├─────────┬──────────┬────────────┤
│ LoRA    │ LoRA     │ LoRA       │
│ (Policy)│ (Value)  │ (Reward)   │
│ 可训练  │ 可训练   │ 冻结       │
└─────────┴──────────┴────────────┘
  Ref = 关闭所有 adapter 的基座模型
```

### 5.2 模型初始化（ppo_multi_adapter.py）

```python
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    name,
    quantization_config=bnb_config,    # QLoRA 4-bit
    peft_config=peft_config,           # Policy LoRA adapter
    reward_adapter=rm_model_name,      # 自动加载 Reward adapter
    device_map=device_map,
    use_flash_attention_2=is_use_flash_attention2,
)
```

**注解**：
- `AutoModelForCausalLMWithValueHead` 是 trl 提供的包装器，在 CausalLM 基础上加 Value Head
- `peft_config` 创建 Policy adapter（可训练）
- `reward_adapter` 自动加载预训练的 RM LoRA 权重作为第三个 adapter（冻结）
- **显存节省**：基座参数只存一份（4-bit），三个 adapter 各自只有几 MB

### 5.3 奖励计算

```python
# 在训练循环中，切换到 reward adapter 计算奖励
rm_model = trainer.accelerator.unwrap_model(trainer.model)
for text in texts:
    inputs = tokenizer(text, return_tensors='pt').to(device)
    # compute_reward_score 内部切换到 reward adapter
    score = rm_model.compute_reward_score(**inputs)[0, -1, 0] - reward_baseline
    raw_rewards.append(score)
```

**注解**：`compute_reward_score` 方法在 `AutoModelForCausalLMWithValueHead` 内部实现——它临时切换活跃 adapter 到 reward adapter，前向传播获取奖励分数，再切回 policy adapter。

### 5.4 Reference Model = disable_adapter

```python
# ma_ppo_trainer.py 中
self.optional_peft_ctx = (
    self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
    if self.is_peft_model
    else nullcontext
)

# step() 中计算 ref logprobs：
with self.optional_peft_ctx():
    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
        self.model,  # 同一个模型，但 adapter 被禁用
        queries, responses, model_inputs,
    )
```

**注解**：PEFT 模型关闭所有 adapter 后，输出等价于原始基座模型的输出，无需额外的 ref model。

### 5.5 PPO Loss 实现

```python
def loss(self, old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns):
    # === Policy Loss (PPO-Clip) ===
    ratio = torch.exp(logprobs - old_logprobs)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange,
                                            1.0 + self.config.cliprange)
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

    # === Value Loss (Clipped) ===
    vpredclipped = clip_by_value(vpreds,
                                  values - self.config.cliprange_value,
                                  values + self.config.cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

    loss = pg_loss + self.config.vf_coef * vf_loss  # vf_coef=0.1

    # Ratio 过大时跳过 batch（防止 loss spike）
    avg_ratio = masked_mean(ratio, mask).item()
    if avg_ratio > self.config.ratio_threshold:  # 默认 10.0
        pg_loss = pg_loss * 0.0
        vf_loss = vf_loss * 0.0
```

**注解**：
- **PPO-Clip**：`cliprange=0.2`，限制策略更新幅度
- **Value Clip**：`cliprange_value=0.2`，稳定 value function 训练
- **Ratio Threshold**：当 `exp(logp_new - logp_old)` 均值超过 10，整个 batch 的 loss 归零，防止训练崩溃
- `masked_mean` 只在 response token 上计算，query 部分被 mask 掉

### 5.6 GAE 优势估计

```python
def compute_advantages(self, values, rewards, mask):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
    returns = advantages + values
    advantages = masked_whiten(advantages, mask)  # 白化
```

**注解**：标准 GAE（Generalized Advantage Estimation），`gamma=1.0, lam=0.95`。白化 advantage 稳定训练。

### 5.7 KL 惩罚与自适应控制

```python
def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
    for score, logprob, ref_logprob, mask in zip(...):
        kl = logprob - ref_logprob                    # per-token KL
        non_score_reward = -self.kl_ctl.value * kl     # KL 惩罚
        reward = non_score_reward.clone()
        last_non_masked_index = mask.nonzero()[-1]
        reward[last_non_masked_index] += score         # RM 奖励加在最后一个 token 上

class AdaptiveKLController:
    def update(self, current, n_steps):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # 动态调整 KL 系数
```

**注解**：
- RM 奖励 score 只加在 response 最后一个 token 上
- 每个 token 位置施加 KL 惩罚 `-β * (logp_policy - logp_ref)`
- `AdaptiveKLController` 根据实际 KL 距离动态调整 β（init=0.2, target=6.0）

### 5.8 训练主循环（ppo_multi_adapter.py）

```python
for epoch, batch in enumerate(trainer.dataloader):
    if epoch >= config.total_ppo_epochs:
        break

    # 1. 生成 response
    response_tensors = trainer.generate(question_tensors, return_prompt=False,
                                         **generation_kwargs)

    # 2. 计算 reward（切换到 reward adapter）
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rm_model = trainer.accelerator.unwrap_model(trainer.model)
    rewards = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        score = rm_model.compute_reward_score(**inputs)[0, -1, 0]
        rewards.append(score)

    # 3. PPO 更新
    stats = trainer.step(question_tensors, response_tensors, rewards)
    trainer.log_stats(stats, batch, rewards)

    # 4. 定期保存
    if save_freq and epoch % save_freq == 0:
        trainer.save_pretrained(f'{output_name}_{epoch}')
```

---

## 六、DPO 实现（dpo.py）

### 6.1 DPO 核心配置

```python
training_args = DPOConfig(
    loss_type='sigmoid',              # 标准 DPO loss: σ(β * (logp_w - logp_l))
    max_completion_length=output_max_length,
    max_prompt_length=output_max_length,
    max_length=seq_length,
    dataset_num_proc=64,              # 数据预处理并行度
    # ...
)
trainer = DPOTrainer(
    model,
    None,                              # ref_model=None → 自动用 LoRA 基座做 ref
    args=training_args,
    peft_config=peft_config,
)
```

**注解**：
- `loss_type='sigmoid'` 即经典 DPO：`L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))`，β=0.1
- `ref_model=None` 且使用 LoRA 时，trl 自动用 disable_adapter 模式作为 ref model
- DPO **不需要 Reward Model**，直接从偏好数据学习，比 PPO 流水线更简单

### 6.2 与 PPO 对比

| 维度 | DPO | PPO |
|------|-----|-----|
| 是否需要 RM | ❌ | ✅ |
| 是否需要在线生成 | ❌（离线偏好数据） | ✅（rollout） |
| 训练复杂度 | 低（一个 Trainer） | 高（4 模型 + 生成循环） |
| 显存需求 | 较低 | 较高 |
| MA-RLHF 中的角色 | 作为 PPO 前的预对齐 | 最终对齐方法 |

在 MA-RLHF 流水线中，DPO 先做一轮预对齐（DPO model 作为 SFT 基座），再进入 PPO/MA-PPO。

---

## 七、生成推理（generate.py）

### 7.1 基本推理

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
)
# 支持多种终止符
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
]
```

### 7.2 两种生成模式

```python
if step_generate:
    # 分步推理模式（for o1-style chain-of-thought）
    input = format_prompt(STEP_INSTRUCTION + instruction)
    output = model.generate(
        ..., do_sample=True, temperature=0.6, top_p=0.95,
        eos_token_id=terminators
    )
    output = output.replace(DEFINE_SEP_TOKEN, " [SEP]\n")  # 每步用分隔符显示
else:
    # 标准贪婪生成
    output = model.generate(..., do_sample=False, eos_token_id=terminators)
```

**注解**：
- **Step Generate** 模式：用特殊 step token（`<|reserved_special_token_1|>`）分隔推理步骤，类似 o1 的 chain-of-thought。需要 SFT 阶段训练过 step token
- **标准模式**：`do_sample=False` 贪婪解码
- 使用 HuggingFace 原生 generate，未集成 vLLM（但项目中有 vLLM 相关注释提及未来计划）

---

## 八、训练脚本解析

### 8.1 标准 PPO（run_7b_ppo.sh）

```bash
# Stage 1: 前置条件——DPO 模型已训练好
sft_path='./output/dpo_full'

# Stage 2: Reward Model（QLoRA）
deepspeed ./ma-rlhf/reward_model.py \
    --model_name=${model_sft_full_path} \     # 基于 DPO 后的模型
    --seq_length=512 \
    --batch_size=16 \
    --use_QLora=True \
    --use_flash_attention_2=True \
    --num_train_epochs=2 \
    --learning_rate=2e-5

# Stage 3: PPO（标准，4 GPU）
deepspeed --num_gpus 4  ./ma-rlhf/ppo.py \
    --model_name=${model_sft_full_path} \
    --reward_model_name=${model_reward_model_lora_path} \
    --batch_size=16 \
    --mini_batch_size=1 \            # 逐样本更新（显存受限）
    --ppo_epochs=1 \                 # 每 batch 只过 1 次 PPO
    --output_max_length=256 \
    --seq_length=64 \                # query 最大长度（较短）
    --gradient_accumulation_steps=2

# Stage 4: 合并 LoRA → 全量模型
python ./ma-rlhf/merge_adapter.py \
    --base_model_name=${model_sft_full_path} \
    --model_name=${model_ppo_lora_path} \
    --merged_model_name=${model_ppo_full_path}
```

### 8.2 多适配器 PPO（run_7b_ma_ppo.sh）

```bash
# 唯一区别：调用 ppo_multi_adapter.py 而非 ppo.py
deepspeed --num_gpus 4  ./ma-rlhf/ppo_multi_adapter.py \
    --dataset_name=${rm_dataset_name} \
    --model_name=${model_sft_full_path} \
    --reward_model_name=${model_reward_model_lora_path} \
    # ... 其余参数完全相同
```

### 8.3 关键超参解读

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 16 | 每步 rollout 采样 16 条 |
| `mini_batch_size` | 1 | PPO 更新时逐条处理（省显存，效率换空间） |
| `ppo_epochs` | 1 | 每个 batch 只训练一个 epoch（防止过拟合） |
| `seq_length` | 64 | Query 最大长度（PPO 阶段不需要长 prompt） |
| `output_max_length` | 256 | 生成 response 最大长度 |
| `gradient_accumulation_steps` | 2 | 有效 batch = 16 × 2 = 32 |
| `num_gpus` | 4 | PPO 阶段用 4 卡（RM 用全部卡） |
| `learning_rate` (RM) | 2e-5 | RM 学习率比 PPO 的 1e-5 更大 |

### 8.4 DeepSpeed ZeRO 配置

脚本使用统一的 `./config/ds.json`。根据 `ma_ppo_trainer.py` 中的逻辑：
- ZeRO Stage 3：ref model 和 active model 都做参数分片
- 非 Stage 3：ref model 以 Stage 0 加载（不分片），假设能放进单卡

---

## 九、工程要点总结

### 9.1 低成本 RLHF 的关键设计决策

1. **QLoRA（4-bit NF4）**：基座参数量化到 4-bit，只训练 LoRA 适配器
   - SFT/DPO: `r=64, alpha=8`（参数量 ~50MB for 7B）
   - RM: `r=32, alpha=8` + `modules_to_save=["scores"]`
   - PPO Policy: `r=64, alpha=8`

2. **多适配器共享基座**（MA-PPO 核心创新）：
   - Policy adapter（可训练）+ Value Head（可训练）+ Reward adapter（冻结）共享同一个 4-bit 基座
   - Ref Model = 关闭所有 adapter
   - 相比标准 PPO 节省约 3× 基座模型显存

3. **DeepSpeed ZeRO**：参数/梯度/优化器状态分片到多卡

4. **Gradient Checkpointing**：`use_cache=False` + `gradient_checkpointing_enable()`

5. **Flash Attention 2**：O(n) 显存的注意力计算

### 9.2 训练稳定性措施

- **Adaptive KL Controller**：动态调整 KL 惩罚系数，防止 policy 偏离过远
- **Ratio Threshold**（10.0）：PPO ratio 过大时跳过整个 batch
- **Gradient Clipping**（`max_grad_norm=1.0`）：防止梯度爆炸导致 NaN
- **Early Stopping**：KL > 1.5 × target_kl 时提前终止当前 epoch

### 9.3 流水线最佳实践

```
SFT (alpaca data)
    ↓ 合并 LoRA → full model
DPO (偏好数据预对齐，可选)
    ↓ 合并 LoRA → full model (dpo_full)
Reward Model (从 dpo_full 初始化，QLoRA)
    ↓ 保存 LoRA adapter
MA-PPO (从 dpo_full + RM adapter，QLoRA)
    ↓ 合并 LoRA → full model (ppo_full)
Generate & Evaluate
```

### 9.4 代码局限与已知问题

1. **trl 版本敏感**：`ppo.py` 需要 trl>=0.11；`ppo_multi_adapter.py` 需要 trl==0.11（特定版本）
2. **save_model hack**：标准 PPO 的 `save_model` 在多 GPU 下需手动修改 trl 源码
3. **奖励计算串行**：MA-PPO 中 reward 逐条计算（`for text in texts`），未做批量推理优化
4. **生成未用 vLLM**：rollout 使用 HF generate，比 vLLM 慢数倍
