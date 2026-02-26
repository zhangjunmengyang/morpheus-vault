---
title: "关于 AI 好玩的"
type: reference
domain: resources
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - resources
  - type/reference
---
# 关于 AI 好玩的

## 为什么记录这个

技术不能只有严肃的论文和调参，AI 领域最吸引人的一面是它不断涌现的 **创造性应用和令人意外的能力**。收集这些"好玩的"东西，一方面保持好奇心，另一方面很多今天好玩的 demo 就是明天的产品。

## 创意应用

### AI 艺术与生成

- **Midjourney / DALL-E / Stable Diffusion**：文生图已经从"能用"到"能用得好"。关键不是技术本身，而是 prompt engineering 变成了一种新的"创作技能"
- **Suno / Udio**：文生音乐。输入一段歌词或风格描述，直接生成完整歌曲。音乐民主化的开始
- **Runway Gen-3 / Sora**：文生视频。从几秒的模糊片段到分钟级的高质量视频，进化速度惊人
- **ControlNet**：给 Stable Diffusion 加精确控制——边缘图、深度图、姿态骨架。从"随机生成"到"可控创作"的关键一步

### AI + 编程

- **Cursor / GitHub Copilot / Claude Code**：AI 辅助编程已经不是"补全几行代码"了，而是能理解项目上下文、重构代码、写测试。实际体感：日常开发效率提升 30-50%
- **v0.dev / bolt.new**：描述你想要的 UI，直接生成前端代码。适合快速原型
- **Devin / OpenHands**：AI 软件工程师的早期形态。虽然目前还不能完全替代人，但能处理明确的、边界清晰的开发任务

### AI 分身与数字人

- **HeyGen**：上传一段视频，AI 克隆你的外观和声音。跨语言的"数字替身"
- **ElevenLabs**：语音克隆。几分钟的样本就能生成极逼真的声音
- **Character.AI**：角色扮演式对话。把 LLM 包装成各种性格的角色，社交娱乐场景

## 有趣的研究发现

### 涌现能力（Emergent Abilities）

模型规模突破某个阈值后，突然出现之前没有的能力：
- 思维链推理（Chain of Thought）：加一句"Let's think step by step"就能显著提升推理准确率
- 少样本学习（Few-shot Learning）：给几个例子就能学会新任务
- 这到底是真正的"涌现"还是评测方式导致的幻觉？目前还有争议

### 大模型的"心智理论"

GPT-4 在部分 Theory of Mind 测试中表现接近成年人水平。但这到底是真正理解了"他人的想法"，还是模式匹配的结果？这个问题很有意思，可能永远无法完全回答。

### 模型坍缩（Model Collapse）

用 AI 生成的数据训练 AI → 模型质量逐代下降。这个现象在理论上已经被证明，实际中 Stable Diffusion 社区已经观察到：大量 AI 生成的图片被用作训练数据后，模型的多样性显著下降。

### AI 的"性格"

不同模型有明显不同的"性格"：
- **Claude**：谨慎、会说"我不确定"、更少幻觉
- **GPT-4**：自信、全面、偶尔编造看起来很真的东西
- **Gemini**：擅长多模态但指令遵循有时不稳定

这些"性格"是 RLHF/RLAIF 训练出来的，不是模型真的"有性格"。但用户体验上确实有差异。

## 工具和平台收集

### 日常好用的

- **Perplexity**：AI 搜索引擎。比传统搜索好的地方在于会帮你综合多个来源给出回答，带引用
- **NotebookLM**：Google 的"AI 笔记本"。上传文档后可以对话、生成摘要、甚至生成播客（!）
- **Napkin.ai**：把文本自动转成图表/信息图。写技术文档配图特别方便

### 开发者工具

- **Hugging Face Spaces**：免费部署 AI Demo 的最好平台
- **Replicate**：API 方式调用各种开源模型，不用自己搞 GPU
- **Ollama**：本地跑开源 LLM，一行命令 `ollama run llama3`

### 玩具但有启发

- **AI Dungeon**：AI 驱动的文字冒险游戏，最早的 LLM 消费级产品之一
- **This Person Does Not Exist**：GAN 生成的逼真人脸。虽然技术已经不新了，但当年第一次看到还是很震撼
- **Infinite Craft**：AI 生成的无限合成游戏。两个概念组合生成新概念，永远玩不完

## 值得关注的趋势

1. **多模态统一**：文本、图像、音频、视频在同一个模型中处理。GPT-4V / Gemini 是开始
2. **Agent 化**：从对话式 AI 到能自主行动的 Agent。MCP（Model Context Protocol）等协议正在标准化
3. **端侧 AI**：模型越来越小但越来越能干。Apple Intelligence、Gemini Nano 等
4. **AI 硬件军备竞赛**：NVIDIA H200 → B200 → GB200。每一代训练速度翻倍
5. **开源追赶闭源**：Llama 3 / Qwen / DeepSeek 等开源模型与 GPT-4 的差距在快速缩小

## 相关

- [[AI 综合|AI 综合]]
- [[AI 思考|AI 思考]]
- [[Projects/提效-Agent/AI 分身|AI 分身]]
- [[关于 AI 学习提效思考|关于 AI 学习提效思考]]
- [[AI/MLLM/MLLM 概述|MLLM 概述]]
- [[AI/LLM/Architecture/AI Models Collapse 论文|AI Models Collapse 论文]]
- [[AI/LLM/Inference/Ollama|Ollama]]
- [[AI/MLLM/ControlNet|ControlNet]]
- [[AI/Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]]
