---
brief: "DPO TRL 实践——HuggingFace TRL DPOTrainer 工程指南；偏好数据格式（chosen/rejected）/损失函数参数（beta）/常见收敛问题；无需 RM 的轻量对齐方案工程上手。"
title: "DPO"
type: project
domain: ai/llm/rl/dpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/dpo
  - type/project
---
# DPO

# 一、文档

https://huggingface.co/docs/trl/v0.21.0/en/dpo_trainer

# 二、步骤

第一步是训练一个 SFT 模型，以确保我们训练的数据符合 DPO 算法的分布。

通过 DPO 微调语言模型包括两个步骤，比 PPO 更容易：

- **Data collection**: 给定一个提示，收集包含正负样本对的生成偏好数据集。
- **Optimization:** 直接最大化 DPO 损失的似然对数。
# 三、指南

训练示例

```
*# train_dpo.py*from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

直接执行 `accelerate launch train_dpo.py`

## 数据准备

DPO 需要一个偏好数据集。DPOTrainer 支持对话式和标准数据集格式。当提供对话式数据集时，训练器会自动将聊天模板应用于数据集。

尽管 DPOTrainer 支持显式和隐式提示，但我们推荐使用显式提示。如果提供隐式提示数据集，训练器将自动从 `"chosen"` 和 `"rejected"` 列中提取提示。有关更多信息，请参阅偏好样式部分。

## 日志指标

- `rewards/chosen` ：所选响应的政策模型和参考模型的 log 概率之间的平均差值，乘以 beta
- `rewards/rejected` ：被拒绝响应的政策模型和参考模型的 log 概率之间的平均差值，乘以 beta
- `rewards/accuracies` : 所选奖励平均出现频率高于对应拒绝奖励的次数
- `rewards/margins` : 所选奖励与对应拒绝奖励之间的平均差异
## 损失函数

- sigmoid (默认)：根据偏好数据，我们可以根据 Bradley-Terry 模型拟合一个二元分类器，实际上 DPO 作者通过 `logsigmoid` 在归一化似然上提出使用 Sigmoid 损失来拟合逻辑回归。
- hinge：RSO 作者提议使用 SLiC 论文中归一化似然上的 hinge 损失。在这种情况下， `beta` 是边距的倒数。https://arxiv.org/abs/2309.06657
- ipo：IPO 作者对 DPO 算法提供了更深入的理论理解，并指出了一个过拟合问题，同时提出了一个替代损失函数。在这种情况下， `beta` 是所选完成对与被拒绝完成对的对数似然比之间的差距的倒数，因此 `beta` 越小，这个差距就越大。根据论文，损失函数是对完成的对数似然进行平均的（与 DPO 只进行求和不同）。https://arxiv.org/abs/2310.12036
- MPO：DPO 训练器支持将多个具有不同权重的损失函数组合起来，从而实现更复杂的优化策略。这对于实现 MPO（混合偏好优化）等算法特别有用。MPO 是一种结合多个优化目标的训练方法，https://arxiv.org/abs/2411.10442。
- RPO、WPO、LD-DPO...
损失函数改动这里太多了，各有各的优劣，详细看文档。

**对于 MOE 模型，可以启用辅助损失**：

- 案例：[昆仑万维-SkyworkMoE](https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F701854564)
- 当负载在专家之间大致平均分配时，MOE（专家混合）最为高效。为确保在偏好微调期间**类似地**训练 MOE，将负载均衡器中的辅助损失添加到最终 loss 中是有益的。
- 此选项通过在模型配置中设置 `output_router_logits=True` （例如 `MixtralConfig` ）来启用。要调整辅助损失对总损失的贡献程度，请在模型配置中使用超参数 `router_aux_loss_coef=...` （默认值： `0.001` ）。
## DPO 前 PEFT 注意事项

假设想要用 DPO 进一步增强的模型是通过(Q)LoRA 进行微调的。

在使用 PEFT 时，参考模型的工作方式有三种方式：

1. 只需创建**两个模型实例**，每个实例加载你的适配器——这可以正常工作，但效率非常低。
1. 将适配器合并到基础模型中，再在上面创建另一个适配器，然后将 `ref_model` 参数设为空，此时 DPOTrainer 会卸载适配器用于参考推理——高效，但存在以下潜在缺点。
- 首先 qlora 是建议量化后合并的。
As suggested by [Benjamin Marie](https%3A%2F%2Fmedium.com%2F%40bnjmn_marie%2Fdont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997), the best option for merging QLoRA adapters is to first dequantize the base model, then merge the adapter. Something similar to [this script](https%3A%2F%2Fgithub.com%2Fjondurbin%2Fqlora%2Fblob%2Fmain%2Fqmerge.py).

- 然而，使用这种方法后，**你将得到一个未量化的基础模型**。因此，要使用 QLoRA 进行 DPO 后**需要重新量化合并后的模型**，或者使用未量化的合并（这将导致更高的内存需求）。
1. 使用不同名称加载适配器两次，然后在训练过程中使用 `set_adapter` 在被 DPO 处理的适配器和参考适配器之间切换——与方案 2（~适配器大小 VRAM 开销）相比效率略低，但避免了潜在问题。
代码案例：

```
# Load the base model.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mixtral-8x7b-v0.1",
    load_in_4bit=True,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

# Load the adapter.
model = PeftModel.from_pretrained(
    model,
    "/path/to/peft",
    is_trainable=True,
    adapter_name="train",
)
# Load the adapter a second time, with a different name, which will be our reference model.
model.load_adapter("/path/to/peft", adapter_name="reference")

# Initialize the trainer, without a ref_model param.
training_args = DPOConfig(
    model_adapter_name="train",
    ref_adapter_name="reference",
)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    ...
)
```

## 相关

- [[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/3-LLM/Frameworks/TRL/TRL 概述|TRL 概述]]
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/3-LLM/SFT/LoRA|LoRA]]
