---
brief: "GRPO Demo——最小可运行的 GRPO 训练示例；包含完整的 reward function 定义/tokenization/训练循环；适合快速验证 GRPO 流程和调试奖励设计，新手入门首选。"
title: "GRPO-demo"
type: project
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/project
---
# GRPO-demo

github：https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

# 代码解读

## prompt 准备

```
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
```

系统提示词定义了结构化的推理格式，这对于数学推理任务至关重要。GRPO特别适合需要复杂推理链的任务，因为它可以通过多个奖励函数来评估不同方面的质量。

## Message 格式化

```
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore
```

`get_gsm8k_questions` 函数将GSM8K数据集转换为对话格式，这是GRPO训练所需的标准输入格式。

## Reward 设计

### 正确性奖励 

`correctness_reward_func` 给予正确答案最高奖励(2.0)，错误答案零奖励。这直接对应GRPO中的任务特定奖励。

```
def correctness_reward_func(prompts, 
completions, answer, **kwargs) -> list
[float]:
    # 提取答案并与标准答案比较
    return [2.0 if r == a else 0.0 for r, 
    a in zip(extracted_responses, answer)]
```

### 格式奖励函数

这些函数实现了"粒度化格式奖励"的概念，通过多层次的格式检查来引导模型生成结构化的推理过程。这正是论文标题中"Granular Format Rewards"的体现。

- `strict_format_reward_func` ：严格的XML格式检查
- `soft_format_reward_func` ：宽松的格式检查
- `xmlcount_reward_func` ：细粒度的XML标签计数
```
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

### 数值类型奖励 

`int_reward_func` 鼓励模型输出数字答案，对数学问题特别重要。

```
def int_reward_func(completions, 
**kwargs) -> list[float]:
    return [0.5 if r.isdigit() else 0.0 
    for r in extracted_responses]
```

## 训练部分

```
#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()
```

# 实践

训练配置

```
*import* re
*import* torch
*from* datasets *import* load_dataset, Dataset
*from* transformers *import* AutoTokenizer, AutoModelForCausalLM
*from* peft *import* LoraConfig
*from* trl *import* GRPOConfig, GRPOTrainer
*import* swanlab
*import* os

*# 命令*
*# python /home/hadoop-kg-llm-ddpt/grpo_demo.py*
*# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/hadoop-kg-llm-ddpt/grpo_demo.py*

*# A100 优化：开启 TF32 以提升吞吐（对 FP32/混精有加速；与 bf16 不冲突）*
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

swanlab.login(*api_key*='pou4vcomXMay52Pg2Fqdj', *save*=False)

*# 设备选择*
*# 单卡最大化：固定到 0 卡（A100 80G）*
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
*# 双卡建议：改为 "0,1" 并用 torchrun 启动*
*# 运行示例：torchrun --nproc_per_node=2 grpo_demo.py*
local_rank = int(os.getenv("LOCAL_RANK", "0"))  *# DDP 进程就绪后每个进程用各自的 GPU*

*# Load and prep dataset*
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def **extract_xml_answer**(*text*: str) -> str:
    answer = *text*.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    *return* answer.strip()

def **extract_hash_answer**(*text*: str) -> str | None:
    *if* "####" not in *text*:
        *return* None
    *return* *text*.split("####")[1].strip().replace(",", "").replace("$", "")

*# uncomment middle messages for 1-shot prompting*
def **get_gsm8k_questions**(*split* = "train") -> Dataset:
    data = load_dataset('/home/hadoop-kg-llm-ddpt/huggingface.co/datasets/openai/gsm8k', 'main')[*split*] *# type: ignore*
    data = data.map(lambda *x*: { *# type: ignore*
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *#{'role': 'user', 'content': 'What is the largest single-digit prime number?'},*
            *#{'role': 'assistant', 'content': XML_COT_FORMAT.format(*
            *#    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",*
            *#    answer="7"*
            *#)},*
            {'role': 'user', 'content': *x*['question']}
        ],
        'answer': extract_hash_answer(*x*['answer'])
    }) *# type: ignore*
    *return* data *# type: ignore*

dataset = get_gsm8k_questions()

*# Reward functions*
def **correctness_reward_func**(*prompts*, *completions*, *answer*, ***kwargs*) -> list[float]:
    responses = [completion[0]['content'] *for* completion *in* *completions*]
    q = *prompts*[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) *for* r *in* responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{*answer*[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    *return* [2.0 *if* r == a *else* 0.0 *for* r, a *in* zip(extracted_responses, *answer*)]

def **int_reward_func**(*completions*, ***kwargs*) -> list[float]:
    responses = [completion[0]['content'] *for* completion *in* *completions*]
    extracted_responses = [extract_xml_answer(r) *for* r *in* responses]
    *return* [0.5 *if* r.isdigit() *else* 0.0 *for* r *in* extracted_responses]

def **strict_format_reward_func**(*completions*, ***kwargs*) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] *for* completion *in* *completions*]
    matches = [re.match(pattern, r, *flags*=re.DOTALL) *for* r *in* responses] 
    *return* [0.5 *if* match *else* 0.0 *for* match *in* matches]

def **soft_format_reward_func**(*completions*, ***kwargs*) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] *for* completion *in* *completions*]
    matches = [re.match(pattern, r, *flags*=re.DOTALL) *for* r *in* responses] 
    *return* [0.5 *if* match *else* 0.0 *for* match *in* matches]

def **count_xml**(*text*) -> float:
    count = 0.0
    *if* *text*.count("<reasoning>\n") == 1:
        count += 0.125
    *if* *text*.count("\n</reasoning>\n") == 1:
        count += 0.125
    *if* *text*.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(*text*.split("\n</answer>\n")[-1])*0.001
    *if* *text*.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(*text*.split("\n</answer>")[-1]) - 1)*0.001
    *return* count

def **xmlcount_reward_func**(*completions*, ***kwargs*) -> list[float]:
    contents = [completion[0]["content"] *for* completion *in* *completions*]
    *return* [count_xml(c) *for* c *in* contents]

*# model_name = "meta-llama/Llama-3.2-1B-Instruct"*
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

*if* "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
*else*:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"
    
training_args = GRPOConfig(
    *output_dir*=output_dir,
    *run_name*=run_name,
    *learning_rate*=5e-6,
    *adam_beta1* = 0.9,
    *adam_beta2* = 0.99,
    *weight_decay* = 0.1,
    *warmup_ratio* = 0.1,
    *lr_scheduler_type*='cosine',
    *logging_steps*=1,
    *bf16*=True,
    *per_device_train_batch_size*=32,
    *generation_batch_size* = 64,
    *# gradient_checkpointing=True,*
    *gradient_accumulation_steps*=1,
    *num_generations*=16,
    *max_prompt_length*=256,
    *max_completion_length*=786,
    *num_train_epochs*=1,
    *save_steps*=500,
    *max_grad_norm*=0.1,
    *report_to*="swanlab",
    *log_on_each_node*=False,
    *# loss_type="dapo",*
    
    *# 性能相关*
    *optim*="adamw_torch_fused",
    *dataloader_num_workers*=8,
    *tf32*=True,
    *ddp_find_unused_parameters*=False,
)
peft_config = LoraConfig(
    *r*=16,
    *lora_alpha*=64,
    *target_modules*=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    *task_type*="CAUSAL_LM",
    *lora_dropout*=0.05,
)

local_model_path = "/home/hadoop-kg-llm-ddpt/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    *torch_dtype*=torch.bfloat16,
    *attn_implementation*="sdpa",
    *device_map*=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token

*# 左填充*
tokenizer.padding_side = "left"

*# use peft at your own risk; not working for me with multi-GPU training*
trainer = GRPOTrainer(
    *model*=model,
    *processing_class*=tokenizer,
    *reward_funcs*=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    *args*=training_args,
    *train_dataset*=dataset,
    *#peft_config=peft_config*
)
trainer.train()
```

## 图表

![image](assets/AL4fd5lZQotrvMxs50Kc1tTKngg.png)

- loss：先上升后下降，符合预期
- `num_tokens`: The total number of tokens processed so far, including both prompts and completions. 线性增长，正常，暂时不知道有什么用。
- 值得关注的是学习率，先线性上升，再缓慢衰减，跟学习率的机制有关，详见 
![image](assets/SYYSdktVjopwBuxRSMmcVdw1nId.png)

![image](assets/VvsYdVgi4og0ShxCoZDcb4LFnIf.png)

![image](assets/WwUCdGc9AoFP4rxiByHci2KanQc.png)

![image](assets/DKyVdo7CIoizjaxto4OcftKKnep.png)

- reward mean 持续上升逐渐接近奖励最大值，同时 std 下降，说明逐渐到了学习格式。
- 最短长度也在上升，应该是由 reasoning 部分影响，可能代表着推理能力正在逐渐上升。
- correct 趋于收敛。
- entropy 也降低收敛，说明输出多样性变少了，趋于稳定。
---

## See Also

- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — GRPO 算法原理
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 2026 全景]] — GRPO 的七维改进框架
- [[AI/3-LLM/RL/GRPO/TRL 中实现 GRPO|TRL 中实现 GRPO]] — 实现细节
-  — LLM 强化学习全图谱
