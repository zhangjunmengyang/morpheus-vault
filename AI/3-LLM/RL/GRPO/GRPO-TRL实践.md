---
brief: "GRPO TRL 实践——基于 HuggingFace TRL 库的 GRPO 训练工程指南；GRPOTrainer 配置/reward function 设计/数据格式规范/常见报错；单卡/多卡训练参数参考，面向实际工程上手。"
title: "GRPO"
type: project
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/project
---
# GRPO

# 一、文档

https://huggingface.co/docs/trl/grpo_trainer

# 二、步骤

GRPO 是一种在线学习算法，意味着它通过在训练过程中使用训练模型自身生成的数据来迭代改进。GRPO 目标背后的直观想法是**最大化生成补全的收益**，**同时确保模型保持接近参考策略**。要理解 GRPO 的工作原理，可以将其分解为四个主要步骤：生成补全、计算收益、估计 KL 散度以及计算损失。

![image](Up5udoOBnonndPxHHrEceotpn9b.png)

## 

1. 生成补全：在每个训练步骤中，采样一批提示，并为每个提示生成一组 G，或者叫 output 记作 oi。
1. 计算优势：对于每个 *G* 序列，使用奖励模型计算奖励。为了与奖励模型的比较性质相一致——通常在相同问题的输出比较数据集上训练——优势被计算出来以反映这些相对比较。它被如下标准化（Group Relative Policy Optimization (GRPO).）
![image](VzltdnL17o3MAMx5lQecxvoen5k.png)

1. KL 散度：来自 schulman，基于 K3（后续其他研究证明这是非必要的）
1. loss：见论文
论文《理解 R1-Zero-Like 训练：一个批判性视角》表明，按 std(r)std(**r**) 进行缩放可能会导致问题级别的难度偏差。您可以通过在 GRPOConfig 中设置 `scale_rewards=False` 来禁用这种缩放。

请注意，与 DeepSeekMath 中原始的表述《在开放语言模型中推动数学推理的极限》相比，我们不会按    1 / ∣oi∣ 进行缩放，因为在论文《理解 R1-Zero-like 训练：一个关键视角》中表明，这会引入响应级别的长度偏差。更多细节请参考损失类型。

请注意，与 DeepSeekMath 中原始的表述《在开放语言模型中推动数学推理的极限》相比，我们默认使用 β=0.0*β*=0.0 ，这意味着 KL 散度项不会被使用。这一选择是由最近几项研究（例如《Open-Reasoner-Zero：一种扩展基础模型强化学习的方法》）所驱动的，这些研究表明 KL 散度项对于使用 GRPO 进行训练并非必需。因此，排除它已成为一种普遍做法（例如《理解 R1-Zero-like 训练：一个关键视角》、DAPO：一种大规模开源 LLM 强化学习系统）。如果您希望包含 KL 散度项，可以将 GRPOConfig 中的 `beta` 设置为非零值。

# 三、指南

## 日志指标

- `num_tokens`：迄今为止处理的令牌总数，包括提示和完成。
- `completions/mean_length`：生成的完成的平均长度。
- `completions/min_length`：生成的完成的最小长度。
- `completions/max_length`：生成的完成的最大长度。
- `completions/mean_terminated_length`：以 EOS 终止的生成完成的平均长度。
- `completions/min_terminated_length`：以 EOS 终止的生成完成的最小长度。
- `completions/max_terminated_length`：以 EOS 终止的生成完成的最大长度。
- `completions/clipped_ratio`：截断比例。
- `reward/{reward_func_name}/mean`：来自特定奖励函数的平均奖励。
- `reward/{reward_func_name}/std`：奖励与特定奖励函数的标准差。
- `reward`：应用奖励权重后的整体平均奖励。
- `reward_std`：应用奖励权重后，每个批次内整体奖励的标准差。
- `frac_reward_zero_std`：生成批次中奖励标准差为零的样本比例，意味着该提示的多样性很小（所有答案都是正确或不正确的）。
- `entropy`：生成的补全中标记预测的平均熵。（如果`mask_truncated_completions=True`，则屏蔽序列标记被排除。）
- `kl`：模型与参考模型之间的平均 KL 散度，根据生成的完成情况计算得出。仅当`beta`非零时才记录。
- `clip_ratio/region_mean`：标记（或序列，如果 ）概率的比率，`importance_sampling_level="sequence"`其中 GRPO 目标被剪辑以保持在信任区域内：
- `clip_ratio/low_meanimportance_sampling_level="sequence"`：在信任区域下限上剪切的标记
- `clip_ratio/low_minimportance_sampling_level="sequence"`：在信任区域下限上剪切的标记
- `clip_ratio/high_meanimportance_sampling_level="sequence"`：在信任区域上限上剪裁的标记
- `clip_ratio/high_maximportance_sampling_level="sequence"`：在信任区域上限上剪裁的标记
## vLLM

我们在训练期间支持使用 vLLM 的两种方式：server mode、colocate mode.

- sever：在此模式下，vLLM 在单独的进程中运行（并使用独立的 GPU），并通过 HTTP 与训练器通信。如果你有专门用于推理的 GPU，这将是最理想的选择。
- Colocate：在此模式下，vLLM 在训练器进程中运行，并与训练模型共享 GPU 内存。这避免了启动单独的服务器，可以提高 GPU 利用率，但可能会导致训练 GPU 上的内存争用。
## 并行训练

几点关键优化：

- **DeepSpeed ZeRO Stage 3**: ZeRO 利用数据并行性将模型状态（权重、梯度、优化器状态）分布在多个 GPU 和 CPU 上，减少每个设备的内存和计算需求。由于大型模型无法适应单个 GPU，因此训练此类模型需要使用 ZeRO 阶段 3。更多详情，请参阅 DeepSpeed 集成。
- **Accelerate**: 加速是一个简化多 GPU 和节点上分布式训练的库。它提供了一个简单的 API 来启动分布式训练，并处理分布式训练的复杂性，如数据并行、梯度累积和分布式数据加载。更多详情，请参阅分布式训练。
- **vLLM**: 加速生成步骤
## 自定义奖励函数

### 入参出参规范

1. Input arguments:
- The function must accept the following as keyword arguments:
- `prompts` (contains the prompts),
- `completions` (contains the generated completions),
- `completions_ids` (contains the tokenized completions),
- `trainer_state` (`TrainerState`): The current state of the trainer. This can be used to implement dynamic reward functions, such as **curriculum learning,** where the reward is adjusted based on the training progress.
- All columns names (but `prompt`) that the dataset may have. For example, if the dataset contains a column named `ground_truth`, the function will be called with `ground_truth` as a keyword argument.
- The easiest way to comply with this requirement is to use `**kwargs` in the function signature.
- Depending on the dataset format, the input will vary:
- For [standard format](https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftrl%2Fdataset_formats%23standard), `prompts` and `completions` will be lists of strings.
- For [conversational format](https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftrl%2Fdataset_formats%23conversational), `prompts` and `completions` will be lists of message dictionaries.
1. Return value: The function must return a list of floats. Each float represents the reward corresponding to a single completion.
### 常用奖励

长度奖励

```
# 1 统计 token 个数
def reward_func(completions_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of token count)."""
    return [float(len(ids)) for ids in completions_ids]
    
# 2 统计字符串长度
def reward_func(completions, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of character count)."""
    return [float(len(completion)) for completion in completions]
```

格式奖励

```
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

答案正确性奖励

```
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

多任务奖励

```
from datasets import Dataset
from trl import GRPOTrainer

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(
    [
        {"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
    ]
)

# Math-specific reward function
def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards

# Coding-specific reward function
def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

在这个示例中， `math_reward_func` 和 `coding_reward_func` 被设计为与包含数学和编程问题的混合数据集一起工作。数据集中的 `task` 列用于确定要应用于每个问题的奖励函数。如果数据集中的某个样本没有相关的奖励函数，奖励函数将返回 `None` ，GRPOTrainer 将继续使用有效的函数和任务。这允许 GRPOTrainer 处理多个具有不同适用性的奖励函数。

请注意，GRPOTrainer 会忽略奖励函数返回的 `None` 奖励，而只考虑相关函数返回的奖励。这确保了模型仅在相关任务上进行训练，并忽略没有相关奖励函数的任务。

有多个奖励函数可以将它们作为一个列表传递，励将计算为每个函数的奖励之和，或者在配置中提供 `reward_weights` 时为加权之和。

```
from trl import GRPOTrainer

trainer = GRPOTrainer(
    reward_funcs=[reward_func1, reward_func2],
    ...,
)
```

## VLM 支持

支持 Gemma3、LLaVA-NeXT、Qwen2-VL、Qwen2.5-VL、SmolVLM2 等

示例命令：

```
accelerate launch \
  --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
  examples/scripts/grpo_vlm.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir grpo-Qwen2.5-VL-3B-Instruct \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --lora_target_modules "q_proj", "v_proj" \
  --log_completions
```

如果图像标记被截断，VLM 训练可能会失败。我们强烈建议通过将`max_prompt_length`设置为`None`来禁用截断。

## 相关

- [[PPO 原理|PPO 原理]]
- [[DPO-TRL实践|DPO]]
- [[TRL 概述|TRL 概述]]
- [[DeepSeek-R1|DeepSeek-R1]]
