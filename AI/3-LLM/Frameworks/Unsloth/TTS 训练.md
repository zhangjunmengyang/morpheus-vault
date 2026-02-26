---
brief: "Unsloth TTS 训练——Text-to-Speech 模型的 Unsloth 微调指南；语音合成模型（如 CSM/Kokoro）的训练数据格式、audio tokenizer 集成、多模态训练的 Unsloth 配置。"
title: "TTS"
type: project
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/project
---
# TTS

## 概述

Unsloth 支持 TTS（Text-to-Speech）模型的高效微调，主要针对的是基于 LLM 架构的语音模型——如 **Orpheus-TTS**、**Sesame CSM** 等。这些模型把语音合成任务转化为 token 生成任务，所以可以用 LLM 的训练方法来微调。

这是一个相对新的方向：用 LLM 的训练基础设施（LoRA、量化、Unsloth 加速）来做语音模型的 fine-tuning。

## 核心思路：语音即 Token

现代 TTS 模型的范式转变：

```
传统 TTS: Text → Acoustic Model → Vocoder → Audio
LLM-based TTS: Text → LLM (生成 audio tokens) → Decoder → Audio
```

关键组件：
1. **Audio Tokenizer**（如 Mimi、Encodec）：把连续的音频波形编码成离散 token
2. **LLM**：把文本 token 和音频 token 统一在一个序列中，自回归生成
3. **Audio Decoder**：把生成的 audio token 解码回波形

因为核心就是一个 LLM 在做 next-token prediction，所以 Unsloth 的所有优化（LoRA、QLoRA、内存优化）都可以直接用。

## Orpheus-TTS 训练

Orpheus 是目前 Unsloth 支持较好的 TTS 模型。

### 数据准备

```python
# 数据格式：文本 + 对应的 audio tokens
# audio tokens 通常由 SNAC/Mimi tokenizer 预处理得到

dataset = {
    "text": "你好，欢迎使用语音合成系统。",
    "audio_tokens": [1023, 456, 789, ...],  # 离散 audio token 序列
    "speaker_id": "speaker_001",  # 可选：说话人 ID
}
```

音频预处理流程：

```python
# 1. 加载音频
import torchaudio
waveform, sr = torchaudio.load("sample.wav")

# 2. 重采样到模型要求的采样率（通常 24kHz）
resampler = torchaudio.transforms.Resample(sr, 24000)
waveform = resampler(waveform)

# 3. 用 audio tokenizer 编码
from snac import SNAC
tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
audio_tokens = tokenizer.encode(waveform)
```

### 训练配置

```python
from unsloth import FastLanguageModel

# 加载 TTS base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="canopylabs/orpheus-3b-0.1-pretrained",
    max_seq_length=2048,
    load_in_4bit=True,  # QLoRA
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)

# 用 SFTTrainer 训练
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./tts-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        bf16=True,
        max_seq_length=2048,
    ),
    train_dataset=tts_dataset,
)
trainer.train()
```

### 推理

```python
# 生成语音
inputs = tokenizer("Hello, how are you today?", return_tensors="pt")
audio_token_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
)

# 解码回音频
audio = audio_decoder.decode(audio_token_ids)
torchaudio.save("output.wav", audio, 24000)
```

## 应用场景

### 1. 个性化语音克隆

用少量目标说话人的音频（几分钟到几十分钟）微调模型：
- 企业级客服语音
- 个人语音助手
- 有声读物定制

### 2. 情感 / 风格控制

训练数据中加入情感标签，让模型学会不同情感的语音生成：

```python
# 带情感标签的数据
{
    "text": "<emotion:happy> 太棒了，我们成功了！",
    "audio_tokens": [...]
}
```

### 3. 多语言 TTS

基于多语言 LLM（如 Qwen）的 TTS 模型天然支持多语言。

## Unsloth 的优势

用 Unsloth 做 TTS 训练的好处：
1. **2-5x 训练加速**：Unsloth 的 kernel 优化对 TTS 同样有效
2. **70% 显存节省**：QLoRA 让单卡 24GB 就能训练 3B TTS 模型
3. **统一的训练工具链**：和文本 LLM 用同一套代码

## 局限性

1. **音频质量**：LLM-based TTS 的音质还不如专门的 TTS 系统（如 VITS、CosyVoice 的某些配置）
2. **延迟**：自回归生成天然慢于并行生成（如 non-autoregressive TTS）
3. **Tokenizer 质量**：audio tokenizer 的质量直接决定上限，这部分 Unsloth 管不了

## 相关

- [[Unsloth 概述]] — Unsloth 框架总览
- [[量化]] — QLoRA 量化技术
- [[LoRA|LoRA]] — LoRA 技术详解
- [[SFT-TRL实践|SFT-TRL实践]] — SFT 训练实践
- [[非文本的模态对齐|非文本的模态对齐]] — 多模态对齐
