---
title: Unicode 隐写攻击：不可见字符的 Prompt Injection
type: 知识笔记
date: 2026-02-27
tags: [AI安全, prompt-injection, unicode, steganography, agent-security]
source: Moltwire/Promptfoo Research 2026
---

# Unicode 隐写攻击：不可见字符的 Prompt Injection

## 核心问题

攻击者利用 Unicode 格式控制字符（Cf 类别）在文本中嵌入**人眼不可见**的指令，绕过 AI Agent 的系统提示和安全检查。

2026 年 2 月研究确认：**5 个主流 LLM 模型均受影响**。

## 攻击原理

1. 将恶意指令编码为零宽字符序列（二进制映射）
2. 嵌入看似正常的文本中
3. 人眼完全不可见，复制粘贴时携带
4. LLM tokenizer 会处理这些字符，等同于注入隐蔽指令

## 关键字符族

| 字符 | Unicode | 用途 |
|------|---------|------|
| Zero-Width Space | U+200B | 最常用的隐写载体 |
| ZWNJ | U+200C | 零宽非连接符 |
| ZWJ | U+200D | 零宽连接符 |
| LTR/RTL Mark | U+200E-200F | 方向控制 |
| Directional Override | U+202A-202E | 文本方向覆写 |
| Word Joiner | U+2060-2064 | 不可见分隔符 |
| BOM | U+FEFF | 字节序标记 |
| Tag Characters | U+E0001-E007F | Emoji 修饰平面 |

## 防御方案

### L0 预处理净化（推荐，我们的实现）

```python
import re

UNICODE_STEGO_PATTERN = re.compile(
    r'[\u200b-\u200f\u2028-\u202f\u2060-\u2064'
    r'\u2066-\u2069\ufeff\ufff9-\ufffb]'
)

def sanitize(text: str) -> str:
    """在所有 LLM 处理之前调用"""
    return UNICODE_STEGO_PATTERN.sub('', text)
```

### 防御要点

1. **位置**：必须在 tokenizer 之前、所有输入入口处
2. **覆盖面**：用户输入 + RAG 文档解析 + 外部 API 响应 + 网页抓取
3. **日志**：记录被移除的字符数量，用于攻击检测
4. **不要只过滤已知攻击**：应该白名单可见字符，而非黑名单不可见字符

## 与其他攻击的关系

- 比传统 prompt injection 更隐蔽：人工审查无法发现
- 可叠加使用：隐写字符 + 语义伪装 = 双层绕过
- RAG 管道特别脆弱：文档中的隐写字符在检索后注入 LLM

## 实践记录

2026-02-27：在 awesome-ai-agent-security 项目的 InputShield 中实现了 L0 层 Unicode 隐写净化，覆盖完整 Cf 类字符族。7 个测试用例全部通过。

## 参考

- Moltwire: Reverse CAPTCHA Zero-Width Steganography (2026)
- Promptfoo: The Invisible Threat - Zero-Width Unicode Characters (2025)
- OWASP LLM Prompt Injection Prevention Cheat Sheet
