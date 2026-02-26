---
brief: "DeepSeek-OCR 原理——DeepSeek 视觉模型在 OCR 任务上的技术路线；高分辨率输入处理/文字区域感知/多语言 OCR 能力；理解为什么 DeepSeek-VL 在文档/发票/截图理解上表现优异的技术原因。"
title: "DeepSeek-OCR"
type: concept
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/concept
---
# DeepSeek-OCR

> 一图胜千言 — 基于 VLM 的端到端 OCR 新范式。
> 论文：https://arxiv.org/abs/2510.18234

## 核心思路

传统 OCR pipeline 分多个阶段：文本检测 → 文本识别 → 版面分析 → 结构化输出。DeepSeek-OCR 的核心观点是：**VLM 本身就能做端到端的 OCR**，不需要那么多独立模块，直接让模型「看图说字」。

这里的关键不是训练一个新模型，而是通过精心设计的 prompt 和微调策略，释放 DeepSeek-VL 系列模型在 OCR 任务上的潜力。

## 技术架构

DeepSeek-OCR 基于 DeepSeek-VL2 的多模态架构：

```
输入图像 → Vision Encoder (SigLIP) → MLP Projector → LLM Backbone
                                                          ↓
                                                     结构化文本输出
```

关键改进点：

1. **高分辨率处理**：OCR 对分辨率极度敏感。采用 dynamic resolution 策略，将图片分割成多个 tile，每个 tile 独立编码后拼接
2. **结构化输出训练**：不只是识别文字，还要输出 Markdown/LaTeX 格式的结构化内容
3. **位置感知**：通过 bounding box 标注训练模型理解文字的空间位置关系

## 数据构建

这是 DeepSeek-OCR 最有价值的部分——高质量 OCR 训练数据的构建方法：

```python
# 数据构建的核心流程
data_pipeline = {
    "文档类": {
        "来源": ["arXiv papers", "教科书", "政府公文"],
        "标注": "PDF 渲染 → 截图 → 自动对齐文本",
        "格式": "Markdown with LaTeX equations"
    },
    "表格类": {
        "来源": ["HTML tables", "Excel screenshots"],
        "标注": "HTML/CSV → 截图 → 结构化表格文本",
        "格式": "Markdown table or HTML"
    },
    "自然场景": {
        "来源": ["街景", "产品图", "手写体"],
        "标注": "人工标注 + 模型辅助校验",
        "格式": "纯文本 + 位置信息"
    }
}
```

**数据量级**：论文中使用了约 500 万张图文对进行训练，其中文档类数据占大头。

## 与传统 OCR 的对比

| 维度 | 传统 Pipeline | DeepSeek-OCR |
|------|-------------|-------------|
| 架构复杂度 | 多个模型串联 | 单一 VLM |
| 版面理解 | 需要额外模块 | 隐式学习 |
| 表格识别 | 专门的表格模型 | 统一处理 |
| 公式识别 | LaTeX OCR | 统一处理 |
| 多语言 | 各语言独立模型 | 天然支持 |
| 上下文理解 | 无 | 有语义理解能力 |
| 速度 | 快（专用模型） | 较慢（大模型推理） |
| 长文档 | 分页处理 | token 长度限制 |

## 实践要点

**Prompt 设计对效果影响极大**：

```
# 通用文档 OCR
请将图片中的所有文字内容转换为 Markdown 格式输出，保持原始排版结构。
表格使用 Markdown 表格格式，数学公式使用 LaTeX 格式。

# 发票/票据 OCR（结构化提取）
请从图片中提取以下信息并以 JSON 格式输出：
- 发票号码
- 开票日期
- 金额（税前、税额、税后）
- 购买方/销售方信息
```

**微调策略**：

```python
# 使用 LoRA 微调 DeepSeek-VL2 做领域 OCR
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 训练数据格式
training_sample = {
    "image": "path/to/document.png",
    "conversations": [
        {"role": "user", "content": "<image>\n请将文档内容转换为Markdown格式"},
        {"role": "assistant", "content": "# 文档标题\n\n正文内容..."}
    ]
}
```

## 我的观点

VLM-based OCR 的优势在「理解」而非「识别」。传统 OCR 在标准印刷体上识别率已经很高了，DeepSeek-OCR 真正的价值在于：

1. **复杂版面的结构化输出** — 论文、合同、技术文档里图文表混排的场景
2. **语义级别的内容提取** — 不只是识别文字，还能理解上下文关系
3. **降低工程复杂度** — 一个模型搞定原来需要 5-6 个模块的 pipeline

但也要清醒：对于简单场景（标准印刷体、固定模板），用 VLM 是杀鸡用牛刀。推理成本和速度是实际部署时必须考虑的。

## 相关

- [[DeepSeek-VL|DeepSeek-VL]] — 底层多模态架构
- [[DeepSeek-OCR-Unsloth实践|DeepSeek-OCR Unsloth 实践]] — 微调实战
- [[Qwen 2.5 VL-Unsloth训练|Qwen 2.5 VL 训练]] — 另一种 VLM OCR 方案
- [[InternVL3|InternVL3]] — 对比方案
