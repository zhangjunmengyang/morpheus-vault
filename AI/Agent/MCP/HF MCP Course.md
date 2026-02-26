---
brief: "HuggingFace MCP 课程——Model Context Protocol 官方入门课程；MCP Server/Client 设计原理、工具注册机制、context 传递协议；Anthropic 标准化 AI 工具接口的学习入口，Claude/smolagents 生态的关键协议。"
title: "MCP Course"
type: tutorial
domain: ai/agent/mcp
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/mcp
  - type/tutorial
---
# MCP Course

https://huggingface.co/learn/mcp-course/unit0/introduction

## MCP 的重要性

随着大型语言模型（LLMs）和其他 AI 系统的不断发展，这些模型的能力越来越强。然而，这些模型往往受限于其训练数据，并且缺乏访问实时信息或专业工具的能力。这种限制阻碍了 AI 系统在许多场景中提供真正相关、准确和有帮助的响应的潜力。

MCP 使 AI 模型能够连接到外部数据源、工具和环境。

MCP 常被描述为“AI 应用的 USB-C”。正如 USB-C 提供了用于将各种外设连接到计算设备的标准化物理和逻辑接口一样，MCP 也提供了用于将 AI 模型连接到外部功能的一致协议。这种标准化将使整个生态系统受益：

- 用户在 AI 应用程序中享受更简单、更一致的体验
- 人工智能应用程序开发人员可以轻松与不断增长的工具和数据源生态系统集成
- 工具和数据提供商只需创建一个可与多个 AI 应用程序配合使用的单一实现
- 更广泛的生态系统受益于增强的互操作性、创新和减少的碎片化
## 如果没有 mcp 会怎么样——集成问题

M×N 集成问题指的是在没有标准化方法的情况下，将 M 种不同的 AI 应用连接到 N 种不同的外部工具或数据源的挑战

### 没有 MCP（M×N 问题）

没有像 MCP 这样的协议，开发者需要为每种可能的 AI 应用与外部能力的组合创建 M×N 个自定义集成。每个 AI 应用都需要单独与每个工具/数据源进行集成。这是一个非常复杂且昂贵的过程，会给开发者带来很多摩擦，并增加维护成本。一旦我们有了多个模型和多个工具，集成的数量变得难以管理，每个集成都有其独特的界面。

![image](assets/W1MvdVoWKoBq0oxD6N2ch2OMnhg.png)

### 使用 MCP（M+N 解决方案）

MCP 通过提供标准接口，将此转换为 M+N 问题：每个 AI 应用程序只需实现 MCP 的客户端一次，每个工具/数据源只需实现 MCP 的服务器端一次。这极大地减少了集成复杂性和维护负担。

![image](assets/C3kRd2Q3coIYkExO16LcIed4n7f.png)

每个 AI 应用程序只需实现 MCP 的客户端一次，每个工具/数据源只需实现 MCP 的服务器端一次。

# 组成部分

- Host：面向用户的 AI 应用程序，最终用户可以直接与其交互。例如，Anthropic 的 Claude 桌面版、增强型 IDE 如 Cursor、推理库如 Hugging Face Python SDK，或在 LangChain 或 smolagents 等库中构建的自定义应用程序。Hosts 会连接到 MCP 服务器，并协调用户请求、LLM 处理和外部工具之间的整体流程。
- Client：Host 应用程序中的一个组件，负责与特定的 MCP 服务器进行通信。每个 Client 都与一个单一的 Server 保持 1:1 的连接，处理 MCP 通信的协议细节，并作为 Host 逻辑与外部 Server 之间的中介。
- Server：外部程序或服务，通过 MCP 协议暴露功能（工具、资源、提示）。
很多内容会将“Client”和“Host”互换使用。从技术上讲，Host 是面向用户的应用程序，而 Client 是 Host 应用程序中负责与特定 MCP Server 通信的组件。

# CS 架构

![image](assets/AkBtdrZProV4eSxH8ElcgQlunXg.png)

模型上下文协议（MCP）建立在客户端-服务器架构之上，这种架构使得 AI 模型与外部系统之间的结构化通信成为可能。

MCP 架构由三个主要组件组成，每个组件都有明确的职责和任务：Host、Client 和 Server。我们在上一节中提到了这些组件，但现在让我们更深入地探讨每个组件及其职责。

## 问题 MCP 基于 http 链接，不会把 sever 耗尽吗？

HTTP+SSE 传输用于远程通信，其中客户端和服务器可能在不同的机器上：

通信通过 HTTP 进行，服务器使用服务器发送事件（SSE）通过持久连接向客户端推送更新。

这种传输的主要优点是可以在网络之间工作，能够与 Web 服务集成，并且与无服务器环境兼容。

MCP 标准的最近更新引入或完善了“可流式 HTTP”，这提供了更多的灵活性，允许服务器在需要时动态升级到 SSE 进行流式传输，同时保持与无服务器环境的兼容性。

### 🔄 什么是 Streamable HTTP？

Streamable HTTP 是一种基于标准 HTTP 的通信机制，允许服务器在需要时将响应升级为 SSE 流式传输，同时保持与无服务器（Serverless）环境的兼容性。它的核心特点包括：[博客园](https%3A%2F%2Fwww.cnblogs.com%2Fxiao987334176%2Fp%2F18845151%3Futm_source%3Dchatgpt.com)

1. **统一端点**：所有通信通过单一的 `/message` 端点进行，简化了客户端与服务器的交互。[博客园+7李乾坤的博客](https%3A%2F%2Fqiankunli.github.io%2F2025%2F04%2F06%2Fmcp.html%3Futm_source%3Dchatgpt.com)[+7fastclass.cn](https%3A%2F%2Fqiankunli.github.io%2F2025%2F04%2F06%2Fmcp.html%3Futm_source%3Dchatgpt.com)[+7](https%3A%2F%2Fqiankunli.github.io%2F2025%2F04%2F06%2Fmcp.html%3Futm_source%3Dchatgpt.com)
1. **按需流式传输**：服务器可以根据具体需求选择返回普通 HTTP 响应或升级为 SSE 流，提供更大的灵活性。[博客园+6阿里云开发者社区+6墨天轮+6](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
1. **会话管理**：引入会话 ID 机制，支持状态管理和连接恢复，增强了通信的可靠性。[博客园+4阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)[+4fastclass.cn](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)[+4](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
1. **灵活初始化**：客户端可通过空的 GET 请求主动初始化 SSE 流，适应不同的应用场景。[李乾坤的博客+4博客园+4阿里云开发者社区+4](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18842223%3Futm_source%3Dchatgpt.com)
---

### ⚙️ 技术原理与工作流程

Streamable HTTP 的工作流程如下：[腾讯云+8墨天轮](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)[+8fastclass.cn](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)[+8](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)

1. **会话初始化**：客户端发送初始化请求到 `/message` 端点，服务器可选择生成会话 ID 并返回。[博客园](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+7fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+7墨天轮+7](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
1. **消息传递**：客户端通过 HTTP POST 请求发送消息到 `/message` 端点，服务器根据需要返回普通响应或升级为 SSE 流。[李乾坤的博客](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+5fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+5墨天轮+5](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
1. **主动建立 SSE 流**：客户端可发送 GET 请求到 `/message` 端点，服务器通过该流推送通知或请求。[oschina.net+3fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+3墨天轮+3](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
1. **连接恢复**：连接中断时，客户端可使用之前的会话 ID 重新连接，服务器恢复会话状态继续交互。[博客园+4阿里云开发者社区+4墨天轮+4](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
---

### ✅ 优势与应用场景

Streamable HTTP 相较于传统的 HTTP + SSE 机制，具有以下优势：[oschina.net](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18851650%3Futm_source%3Dchatgpt.com)[+4博客园+4阿里云开发者社区+4](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18851650%3Futm_source%3Dchatgpt.com)

- **资源效率**：服务器无需为每个客户端维持长时间的 SSE 连接，降低了资源消耗。[博客园+2墨天轮](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)[+2fastclass.cn](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)[+2](https%3A%2F%2Fwww.modb.pro%2Fdb%2F1910178965881368576%3Futm_source%3Dchatgpt.com)
- **兼容性强**：与现有的 HTTP 基础设施（如 CDN、负载均衡器、API 网关等）无缝集成，适用于各种部署环境。[博客园](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+2fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+2墨天轮+2](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
- **灵活性高**：支持无状态和有状态模式，适应不同的应用需求。[博客园](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+6fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+6墨天轮+6](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
- **可靠性提升**：支持断线重连和会话恢复，增强了通信的稳定性。[博客园](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+2fastclass.cn](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)[+2墨天轮+2](https%3A%2F%2Ffastclass.cn%2Fvideo%2Fmcp-streamable-http%3Futm_source%3Dchatgpt.com)
适用场景包括：

- **无状态 API 服务**：如数学计算、文本处理等简单工具 API 服务。[阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)[+1oschina.net](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)[+1](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
- **长时间运行的任务**：如大文件处理、复杂 AI 生成等需要实时反馈的任务。[阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
- **复杂 AI 会话**：如 AI 助手、聊天机器人等需要维护会话状态的应用。[阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1661704%3Futm_source%3Dchatgpt.com)
---

### 🧪 实际应用与支持

阿里云的 Spring AI Alibaba 与 Higress 已联合发布了业界首个 Streamable HTTP 实现方案，提供了完整的 Java 实现和可运行示例，展示了该协议在实际应用中的可行性和优势。 [fastclass.cn](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18842223%3Futm_source%3Dchatgpt.com)[+5博客园+5阿里云开发者社区+5](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18842223%3Futm_source%3Dchatgpt.com)

---

### 📌 总结

Streamable HTTP 是 MCP 协议的一次重要升级，通过引入按需流式传输、统一端点和会话管理机制，解决了传统 HTTP + SSE 机制在资源消耗和连接稳定性方面的不足，提升了通信的灵活性和可靠性。对于需要高并发、实时反馈和状态管理的 AI 应用，Streamable HTTP 提供了更优的解决方案。[阿里云开发者社区+4博客园](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18851650%3Futm_source%3Dchatgpt.com)[+4fastclass.cn](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18851650%3Futm_source%3Dchatgpt.com)[+4](https%3A%2F%2Fwww.cnblogs.com%2Falisystemsoftware%2Fp%2F18851650%3Futm_source%3Dchatgpt.com)

如果您对 Streamable HTTP 的具体实现或在特定场景中的应用有进一步的兴趣，欢迎继续提问。

---

## See Also

- [[AI/Agent/MCP/如何给人深度科普 MCP|如何深度科普 MCP]] — MCP 的概念深度版
- [[AI/Agent/Fundamentals/Tool Use|Tool Use]] — MCP 是 Tool Use 的标准化协议
- [[AI/Agent/目录|Agent MOC]] — Agent 知识全图谱
- [[AI/LLM/目录|LLM MOC]] — MCP 在 LLM 生态中的位置
