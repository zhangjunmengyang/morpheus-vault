---
title: "LLM + Agent"
type: concept
domain: ai/agent/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/fundamentals
  - type/concept
---
# LLM + Agent

课程：

- LLM：https://huggingface.co/learn/llm-course/zh-CN/chapter1/1
- Agent：https://huggingface.co/learn/agents-course/unit0/introduction
# LLM

![image](assets/H1F8dhpwGowWhTxIqh0cX3wqnUf.png)

Transformer 有三种类型：

1. 编码器（Encoders）
基于编码器的 Transformer 接收文本（或其他数据）作为输入，并输出该文本的密集表示（或嵌入）。
- 示例：Google 的 BERT
- 用例：文本分类、语义搜索、命名实体识别
- 典型规模：数百万个参数
1. 解码器（Decoders）
基于解码器的 Transformer 专注于逐个生成新令牌以完成序列。
- 示例：Meta 的 Llama
- 用例：文本生成、聊天机器人、代码生成
- 典型规模：数十亿（按美国用法，即 10^9）个参数
1. 序列到序列（编码器-解码器，Seq2Seq（Encoder–Decoder））
序列到序列的 Transformer *结合*了编码器和解码器。编码器首先将输入序列处理成上下文表示，然后解码器生成输出序列。
- 示例：T5、BART
- 用例：翻译、摘要、改写
- 典型规模：数百万个参数
虽然大语言模型 (LLMs) 有多种形式，但它们通常是基于**解码器**的模型。

- **token**
- 大语言模型 (LLM) 的基本原理简单却极其有效：其目标是在给定一系列前一个令牌的情况下，预测下一个令牌。这里的“令牌”是 LLM 处理信息的基本单位。出于效率考虑，LLM 并不直接使用整个单词。
- 例如，虽然英语估计有 60 万个单词，但一个 LLM 的词汇表可能只有大约 32,000 个令牌（如 Llama 2 的情况）。令牌化通常作用于可以组合的子词单元。举个例子，考虑如何将令牌“interest”和“ing”组合成“interesting”，或者添加“ed”形成“interested”。
- **EOS**
- 每个大语言模型 (LLM) 都有一些特定于该模型的特殊令牌。LLM 使用这些令牌来开启和关闭其生成过程中的结构化组件。例如，用于指示序列、消息或响应的开始或结束。此外，我们传递给模型的输入提示也使用特殊令牌进行结构化。其中最重要的是序列结束令牌 (EOS，End of Sequence token)。
- **自回归**
- 大语言模型 (LLM) 被认为是自回归的，这意味着一次通过的输出成为下一次的输入。这个循环持续进行，直到模型预测下一个词元为 EOS（结束符）词元，此时模型可以停止。
- 一旦输入文本被词元化，模型就会计算序列的表示，该表示捕获输入序列中每个词元的意义和位置信息。
- 这个表示被输入到模型中，模型输出分数，这些分数对词汇表中每个词元作为序列中下一个词元的可能性进行排名。
- **token 预测**
- 最简单的解码策略是总是选择分数最高的词元。
- 更先进的解码策略如***束搜索（beam search）*** 会探索多个候选序列，以找到总分数最高的序列——即使其中一些单个词元的分数较低。（但实际用起来不好用，当时发现框架就是效果很差，然后开始跟工程一点点排查，最后发现是有个参数 beam = 2，相当于看接下来两个词）
- **attention**
- 尽管自 GPT-2 以来，大语言模型（LLM）的基本原理——预测下一个词元——一直保持不变，但在扩展神经网络以及使注意力机制能够处理越来越长的序列方面已经取得了显著进展。你可能对*上下文长度*这个术语很熟悉，它指的是大语言模型能够处理的最大词元数，以及其最大的*注意力跨度*。
- **prompt**
- **为什么提示词很重要：**考虑到大语言模型（LLM）的唯一工作是通过查看每个输入词元来预测下一个词元，并选择哪些词元是“重要的”，因此你提供的输入序列的措辞非常重要。你提供给大语言模型的输入序列被称为*提示*。精心设计提示可以更容易地引导大语言模型的生成朝着期望的输出方向进行。
- 每个模型会有**聊天模版**，作用是指导消息交换如何格式化为单个提示。（最后聊天模板的替换还是在输入到大模型之前被解析为包含特殊 token 的消息，比如开始结束符，最后给大模型输入）
```
格式化前：
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
    {"role": "user", "content": "How do I use it ?"},
]

格式化后：
<|im_start|>system
You are a helpful assistant focused on technical topics.<|im_end|>
<|im_start|>user
Can you explain what a chat template is?<|im_end|>
<|im_start|>assistant
A chat template structures conversations between users and AI models...<|im_end|>
<|im_start|>user
How do I use it ?<|im_end|>
```

- System prompt：系统消息（也称为系统提示）定义了模型应该如何表现。它们作为持久性指令，指导每个后续交互。
- User and Assistant Messages)：对话由用户和LLM之间的交替消息组成。（扩展：正常来讲是需要交替的，但是实际是个工程问题，比如是否允许出现连续的 user message，之前在 openai 的 API 上测试是可以的，会将 user message 拼接成一个消息作为上下文。）
- **Tools**
- 优秀工具应能补充 LLM 的核心能力。LLM 内部知识仅包含训练截止前的信息。因此，若智能体需要最新数据，必须通过工具获取。若直接询问 LLM（无搜索工具）今日天气，LLM 可能会产生随机幻觉。
- 工具组成：描述、参数、操作对象
- 原理：工具调用的输出是对话中的另一种消息类型。工具调用步骤通常对用户不可见：**智能体检索对话、调用工具、获取输出、将其作为新消息添加，并将更新后的对话再次发送给 LLM**。从用户视角看，仿佛 LLM 直接使用了工具，但实际执行的是我们的应用代码（智能体）。
- MCP：模型上下文协议（MCP）是一种开放式协议，它规范了应用程序向 LLM 提工具的方式。MCP 提供：
- 不断增加的预构建集成列表，您的 LLM 可以直接接入这些集成
- 在 LLM 提供商和供应商之间灵活切换的能力
- 在基础设施内保护数据安全的最佳实践
这意味着任何实施 MCP 的框架都可以利用协议中定义的工具，从而无需为每个框架重新实现相同的工具接口。

# Agent

学术界和工业界对术语“智能体”提出了各种定义。大致来说，一个智能体应具备类似人类的思考和规划能力，拥有记忆甚至情感，并具备一定的技能以便与环境、智能体和人类进行交互。

可以将智能体想象成环境中的数字人，其中

智能体 = 大语言模型（LLM） + 观察 + 思考 + 行动 + 记忆

这个公式概括了智能体的功能本质。为了理解每个组成部分，让我们将其与人类进行类比：

1. 大语言模型（LLM）：LLM作为智能体的“大脑”部分，使其能够处理信息，从交互中学习，做出决策并执行行动。
1. 观察：这是智能体的感知机制，使其能够感知其环境。智能体可能会接收来自另一个智能体的文本消息、来自监视摄像头的视觉数据或来自客户服务录音的音频等一系列信号。这些观察构成了所有后续行动的基础。
1. 思考：思考过程涉及分析观察结果和记忆内容并考虑可能的行动。这是智能体内部的决策过程，其可能由LLM进行驱动。
1. 行动：这些是智能体对其思考和观察的显式响应。行动可以是利用 LLM 生成代码，或是手动预定义的操作，如阅读本地文件。此外，智能体还可以执行使用工具的操作，包括在互联网上搜索天气，使用计算器进行数学计算等。
1. 记忆：智能体的记忆存储过去的经验。这对学习至关重要，因为它允许智能体参考先前的结果并据此调整未来的行动。
https://huggingface.co/learn/agents-course/unit0/introduction

*智能体是一个系统，它利用 AI 模型与环境交互，以实现用户定义的目标。它结合观察、推理、规划和动作执行（通常通过外部工具）来完成任务。*

DeepMind团队发表的一篇论文「Position: Levels of AGI for Operationalizing Progress on the Path to AGI」中，详细定义了AGI的不同级别。

- 智能体的交互层次
- L1：Agent 输出不影响整体流程，处理器或者作为工具（Agent as Tool）
- L2：Agent 输出决定了基本控制流（Planner）
- L3：Agent 输出决定函数调用（子 Agent）
- L4：Agent 输出控制迭代及程序延续（子 Agent）
- L5：Agent 输出启动其他 Agent 行动，就是智能体通信
- DeepMind 划分智能化层次论文：https://arxiv.org/abs/2311.02462
![image](assets/YrzSdvkkNoCo1SxcvfJcidMkneh.png)

- AI stages by OpenAI
![image](assets/O4aVdLymfoYHTDxcyhwcgqXwnjb.png)

- 思考
- 思维（Thought）代表着智能体解决任务的内部推理与规划过程。可将其视为智能体的内部对话，在此过程中它会考量当前任务并制定应对策略。智能体的思维负责获取当前观察结果，并决定下一步应采取的行动。
- 思维类型：
- ReAct
- **Zero-shot**：最早期ReAct 是一种提示技术，在让 LLM 解码后续 token 前添加“Let’s think step by step”（让我们逐步思考）的提示。通过提示模型”逐步思考”，可以引导解码过程生成计划而非直接输出最终解决方案，因为模型被鼓励将问题分解为子任务。（最早出现在 zero-shot 中，发现引导模型思考可以获得更好的答案，更偏向于隐式反思）
- **训练CoT**：在 Deepseek R1 或 OpenAI 的 o1 等模型的开发中。这些模型经过微调，被训练为"先思考再回答"。它们通过特殊标记（`<thought>` 和 `</thought>`）来界定 *思考* 部分。这不仅是类似 ReAct 的提示技巧，更是通过分析数千个示范案例，让模型学习生成这些思考段的训练方法。
- 显式反思：显式输出，并且用下一轮对话来阅读思考的内容，这是MAS中的方式，是有效果的。
- 行动
- Agent 常见的行动：
- 工具使用 (Tool Usage)
- 信息收集 (Information Gathering)
- 环境交互 (Environment Interaction)
- 通信 (Communication)
-  LLM 输出内容以实现通信
- 处理输出
- 以结构化格式生成 (Generation in a Structured Format)：智能体以清晰、预定义的格式（JSON或代码）输出其预期动作。
- 停止进一步生成 (Halting Further Generation)：一旦动作完成，智能体停止生成额外的 token
- 解析输出 (Parsing the Output)：外部解析器读取格式化的动作，确定要调用哪个工具，并提取所需的参数。（一般也就是工程里面**正则匹配**，确保用一些特殊标识能解析就行）
- 一点感悟：如果一个动作是确定的且定制化的，那么尽可能用 Workflow 的方式封装成 Tool，避免把任何事情都交给 Agent 来做。
- 观察
- Observations（观察）是智能体感知其行动结果的方式。Agent 观察来自环境的信号（API 返回的数据、错误信息还是系统日志）引导 LLM 进行下一轮的思考-行动。常见观察类型
- 在观察阶段，智能体会：
- 收集反馈：接收数据或确认其行动**是否成功**
- 附加结果：将**新信息整合到现有上下文**中，有效**更新记忆**
- 调整策略：使用更新后的上下文来**优化后续思考和行动**
在通过 API 与模型进行的”典型”对话中，对话将在用户和助手消息之间交替进行，如下所示：

```
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

函数调用为对话带来了新的角色！

1. 一个用于 行动(Action) 的新角色
1. 一个用于 观察(Observation) 的新角色
在带有 function call 的框架中，会是这样

```
conversation = [
    {
        "role": "user",
        "content": "What's the status of my transaction T1001?"
    },
    {
        "role": "assistant",
        "content": "",
        "function_call": {
            "name": "retrieve_payment_status",
            "arguments": "{\"transaction_id\": \"T1001\"}"
        }
    },
    {
        "role": "tool",
        "name": "retrieve_payment_status",
        "content": "{\"status\": \"Paid\"}"
    },
    {
        "role": "assistant",
        "content": "Your transaction T1001 has been successfully paid."
    }
]
```

在这种情况下和许多其他API中，模型将要采取的行动格式化为”助手”消息。聊天模板然后将此表示为函数调用的特殊词元 (special tokens)。

- `[AVAILABLE_TOOLS]` – 开始可用工具列表
- `[/AVAILABLE_TOOLS]` – 结束可用工具列表
- `[TOOL_CALLS]` – 调用工具（即采取”行动”）
- `[TOOL_RESULTS]` – “观察”行动的结果
- `[/TOOL_RESULTS]` – 观察结束（即模型可以再次解码）
详细了解：https://docs.mistral.ai/capabilities/function_calling/

训练模型的 function call 能力

https://huggingface.co/learn/agents-course/zh-CN/bonus-unit1/fine-tuning

涉及到产品化的时候，会有观察和评估系统：

- 但其实对可量化的问题还算有效，但比如分析场景，每次迭代如何用可量化的指标来衡量分析的“好与坏”，非常困难
- https://huggingface.co/learn/agents-course/zh-CN/bonus_unit2/monitoring-and-evaluating-agents-notebook
---

## See Also

- [[AI/Agent/Fundamentals/HF Agent Course|HF Agent Course]] — HF Agent 学习系列的另一篇
- [[AI/Agent/Fundamentals/Tool Use|Tool Use]] — HF LLM+Agent 的核心能力：tool use
- [[AI/Agent/_MOC|Agent MOC]] — Agent 知识全图谱
- [[AI/Foundations/ML-Basics/机器学习|机器学习]] — LLM 的 ML 基础
