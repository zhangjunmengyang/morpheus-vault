---
brief: "Unsloth notebook 合集——官方 Colab/Kaggle notebook 汇总；覆盖 Llama/Qwen/Gemma/Mistral 等主流模型的 SFT/DPO/GRPO 训练模板；快速上手不同任务的参考入口。"
title: "notebook 合集"
type: concept
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/concept
---
# Unsloth Notebook 合集

> 官方 Notebooks：https://docs.unsloth.ai/get-started/unsloth-notebooks

Unsloth 提供了一系列开箱即用的 Colab/Jupyter Notebook，覆盖了从 SFT 到 RL 的主要训练场景。这是上手 Unsloth 最快的方式——直接打开 notebook，改几个参数就能跑。

## Notebook 分类

### SFT（监督微调）

| Notebook | 模型 | 说明 |
|----------|------|------|
| Llama 3.1 SFT | Llama-3.1-8B | 最基础的 SFT 示例 |
| Qwen 2.5 SFT | Qwen2.5-7B | ChatML 格式 |
| Gemma 3 SFT | Gemma-3-4B | Google 模型 |
| Phi-4 SFT | Phi-4-14B | 微软模型 |
| Mistral SFT | Mistral-7B | Mistral 格式 |

### RL（强化学习）

| Notebook | 算法 | 说明 |
|----------|------|------|
| GRPO Training | GRPO | DeepSeek 提出的 Group RL |
| DPO Training | DPO | 直接偏好优化 |
| GSPO Training | GSPO | 改进版偏好优化 |
| ORPO Training | ORPO | 无需 SFT 的对齐方法 |

### 特殊场景

| Notebook | 场景 | 说明 |
|----------|------|------|
| CPT | 持续预训练 | 注入领域知识 |
| Vision Fine-tuning | 多模态 | VLM 微调 |
| TTS Training | 语音 | 文本转语音模型 |

## 使用建议

### 1. 直接从 Colab 开始

Colab 免费版有 T4 GPU（16GB 显存），对于 7B 模型的 4bit LoRA 微调足够了。

```
Colab 免费版: T4 16GB → 可跑 7B 4bit LoRA
Colab Pro: A100 40GB → 可跑 14B+ 模型
自有服务器: 按需选择
```

### 2. 修改数据部分

Notebook 自带的数据集是示例用的。实际使用时只需要改数据加载部分：

```python
# 原始 notebook 中的示例数据
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# 替换为你自己的数据
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")
```

### 3. 调参顺序

建议按以下顺序调整参数：

```
1. 先跑通默认参数 → 确认 pipeline 没问题
2. 调 learning_rate → 最影响训练效果的参数
3. 调 num_epochs → 看 loss 曲线决定
4. 调 LoRA rank → 影响模型容量
5. 调 batch_size → 影响训练稳定性
```

### 4. 关注 Loss 曲线

```python
# 训练结束后绘制 loss 曲线
import matplotlib.pyplot as plt

# 从 trainer_state.json 中提取
losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
```

正常的 loss 曲线应该是先快速下降，然后逐渐趋平。如果：
- **一直下降不收敛** → epoch 太少或 lr 太高
- **震荡剧烈** → lr 太高或 batch 太小
- **迅速到 0** → 过拟合，数据可能有泄露

## 我的使用心得

Unsloth 的 notebook 最大的价值不在于"复制粘贴就能跑"，而在于它给了你一个**可验证的 baseline**。先确认默认配置能正常训练，再一步步改成你需要的样子。不要一上来就魔改 10 个参数——出了问题你根本不知道是哪里改错了。

## 相关

- [[AI/3-LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/3-LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/3-LLM/Frameworks/Unsloth/Chat Templates|Chat Templates]]
- [[AI/3-LLM/Frameworks/Unsloth/Checkpoint|Checkpoint 管理]]
- [[AI/3-LLM/Pretraining/小规模训练手册|小规模训练手册]]
