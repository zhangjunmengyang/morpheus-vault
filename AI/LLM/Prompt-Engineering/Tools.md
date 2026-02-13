---
title: "Tools"
type: concept
domain: ai/llm/prompt-engineering
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/prompt-engineering
  - type/concept
---
# Prompt 工具集

> System prompts 收集：https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools/tree/main

搞 Prompt Engineering 不能闭门造车。这个领域迭代极快，工具和资源是核心生产力。

## System Prompt 泄露与学习

学 prompt 最好的方式就是看别人写的好 prompt。上面的 repo 收集了各大 AI 产品的 system prompt，包括 ChatGPT、Claude、Perplexity 等。

**看这些 prompt 要关注什么：**

1. **结构**：好的 system prompt 都有清晰的分层——角色、能力边界、输出格式、edge case 处理
2. **约束表达**：怎么写"不要做什么"比"要做什么"更难
3. **工具定义**：如何向模型描述可用的工具和调用格式

```markdown
# 典型的好 prompt 结构
## Identity — 你是谁
## Capabilities — 你能做什么
## Constraints — 你不能做什么
## Tools — 可用工具及调用格式
## Output Format — 输出规范
## Examples — 示例（可选）
```

## 常用工具

### Prompt 开发与测试

| 工具 | 特点 |
|------|------|
| [LangSmith](https://smith.langchain.com/) | LangChain 官方，追踪 + 评估一体 |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | 开源，支持批量测试和对比 |
| [Anthropic Workbench](https://console.anthropic.com/) | 直接在控制台迭代 prompt |

### Promptfoo 快速上手

```yaml
# promptfooconfig.yaml
prompts:
  - "将以下文本分类为正面或负面：{{text}}"
  - "分析以下评论的情感倾向（正面/负面）：{{text}}"

providers:
  - openai:gpt-4o
  - anthropic:claude-sonnet-4-20250514

tests:
  - vars:
      text: "这个产品太好用了！"
    assert:
      - type: contains
        value: "正面"
  - vars:
      text: "垃圾，退货了"
    assert:
      - type: contains
        value: "负面"
```

```bash
npx promptfoo eval
npx promptfoo view  # 打开 Web UI 查看结果
```

### Prompt 版本管理

Prompt 跟代码一样需要版本管理。我的做法：

```python
# prompts/v2/classifier.py
CLASSIFIER_PROMPT = """
Version: 2.1
Last Updated: 2026-02-10
Change: 增加了 neutral 类别

将以下文本分类为：正面(positive)、负面(negative)、中性(neutral)
输出 JSON 格式：{"label": "...", "confidence": 0.0-1.0}

文本：{text}
"""
```

### Prompt 自动优化

DSPy 的思路很有意思——把 prompt 当作可学习的参数，通过少量标注数据自动优化：

```python
import dspy

class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought("text -> label")
    
    def forward(self, text):
        return self.classify(text=text)

# 自动优化 prompt
optimizer = dspy.BootstrapFewShot(metric=accuracy)
optimized = optimizer.compile(Classifier(), trainset=train_data)
```

## 实用资源

- **[Prompt Engineering Guide](https://www.promptingguide.ai/)**：最全的 prompt 技巧百科
- **[OpenAI Cookbook](https://cookbook.openai.com/)**：官方示例集
- **[Anthropic Docs](https://docs.anthropic.com/)**：Claude 的 prompt 最佳实践
- **[LangChain Hub](https://smith.langchain.com/hub)**：社区共享的 prompt 模板

## 我的观点

工具只是辅助。Prompt Engineering 最终还是要靠对模型能力边界的理解和对业务问题的拆解能力。最好的 prompt 工程师往往不是写 prompt 最花哨的人，而是最了解问题本身的人。

## 相关

- [[AI/LLM/Prompt-Engineering/Prompt engineering 概述|Prompt Engineering 概述]]
- [[AI/LLM/Prompt-Engineering/prompt 攻击|Prompt 攻击]]
- [[AI/LLM/Prompt-Engineering/数据合成|数据合成]]
- [[AI/LLM/Prompt-Engineering/Prompt Engineering|Prompt Engineering 实践]]
