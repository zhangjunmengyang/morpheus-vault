---
brief: "DeepSeek-OCR Unsloth 实践——DeepSeek-VL OCR 能力的 Unsloth 微调工程指南；OCR 数据格式/标注规范/自定义 OCR 任务微调的完整流程；面向文档理解场景的 VLM 工程实践。"
title: "DeepSeek-OCR"
type: project
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/project
---
# DeepSeek-OCR Unsloth 实践

> 用 Unsloth 微调 DeepSeek-VL 做领域 OCR 的实战指南。
> 参考：https://docs.unsloth.ai/new/deepseek-ocr-how-to-run-and-fine-tune

## 环境准备

```bash
pip install unsloth
# Unsloth 会自动安装兼容版本的 transformers, peft, trl 等
```

硬件要求：
- **推理**：单卡 24GB（RTX 4090 / A5000）
- **微调（LoRA）**：单卡 24GB，batch_size=1-2
- **微调（QLoRA）**：单卡 16GB，INT4 量化 + LoRA

## 加载模型

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/deepseek-vl2-tiny",  # 3B 版本，适合微调
    max_seq_length=4096,
    load_in_4bit=True,  # QLoRA
    dtype=None,  # 自动选择
)

# 添加 LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 省显存
)
```

## 数据准备

OCR 微调的数据格式是关键。Unsloth 支持 ShareGPT 格式：

```python
# 单条训练数据示例
sample = {
    "conversations": [
        {
            "role": "user",
            "content": "<image>\n请将文档内容转换为 Markdown 格式，表格用 Markdown 表格，公式用 LaTeX。"
        },
        {
            "role": "assistant", 
            "content": "# 季度财务报告\n\n## 营收概况\n\n| 季度 | 营收(万元) | 同比增长 |\n|------|-----------|--------|\n| Q1 | 1,234 | +15% |\n| Q2 | 1,456 | +18% |\n\n总营收公式：$R_{total} = \\sum_{i=1}^{4} R_i$"
        }
    ],
    "images": ["path/to/financial_report.png"]
}
```

**数据构建技巧**：

```python
# 用 PDF 批量生成训练数据
import fitz  # PyMuPDF

def pdf_to_training_data(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    samples = []
    for page in doc:
        # 渲染页面为图片
        pix = page.get_pixmap(dpi=dpi)
        img_path = f"page_{page.number}.png"
        pix.save(img_path)
        
        # 提取文本作为 ground truth
        text = page.get_text("text")
        # 提取表格
        tables = page.find_tables()
        
        # 组合成 Markdown ground truth
        markdown = convert_to_markdown(text, tables)
        
        samples.append({
            "image": img_path,
            "text": markdown
        })
    return samples
```

## 训练配置

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./deepseek-ocr-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 等效 batch_size=8
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",  # 省显存
    ),
)

trainer.train()
```

## 训练注意事项

1. **图片分辨率**：OCR 任务对分辨率敏感，不要过度压缩。建议保持 1024×1024 以上
2. **序列长度**：文档 OCR 输出很长，`max_seq_length` 至少 4096，长文档需要 8192+
3. **数据质量 > 数量**：100 条高质量标注 > 1000 条噪声数据
4. **过拟合风险**：领域数据量少时，`num_train_epochs` 控制在 2-3，配合 `lora_dropout`

## 推理和评估

```python
# 加载微调后的模型
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./deepseek-ocr-finetuned",
    max_seq_length=4096,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# OCR 推理
from PIL import Image

image = Image.open("test_document.png")
messages = [{"role": "user", "content": [image, "请将文档内容转换为Markdown格式"]}]
response = model.chat(messages, tokenizer=tokenizer)
print(response)
```

**评估指标**：
- **字符准确率（CER）**：逐字符比较
- **BLEU/ROUGE**：整体文本质量
- **表格 F1**：表格结构还原准确率
- **人工抽检**：最终还是得人看

## 我的经验

1. 微调效果主要取决于**领域数据的代表性**，不是数据量
2. QLoRA (4-bit) 微调效果略差于 full LoRA，但显存节省 50%+，trade-off 合理
3. 中文文档 OCR 场景，DeepSeek-VL2-tiny (3B) 微调后效果已经很不错，不需要上大模型
4. 生产环境建议 LoRA merge 后用 vLLM 部署，比 Unsloth 推理快很多

## 相关

- [[DeepSeek-OCR 原理|DeepSeek-OCR 原理]] — 底层原理
- [[DeepSeek-VL|DeepSeek-VL]] — 基座模型
- [[Qwen 2.5 VL-Unsloth训练|Qwen 2.5 VL Unsloth 训练]] — 同类实践
- [[训练示例概述|Unsloth 训练示例]] — Unsloth 通用指南
- [[运行 & 保存模型|Unsloth 运行保存]]
