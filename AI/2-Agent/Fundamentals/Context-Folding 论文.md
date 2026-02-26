---
brief: "Context-Folding——通过压缩/折叠上下文扩展 LLM Agent 的有效 context window；将历史交互摘要为紧凑表示，使长 horizon Agent 任务在有限 context 下仍能保持关键信息；长时域 Agent 的工程方案之一。"
title: "Scaling Long-Horizon LLM Agent via Context-Folding"
type: paper
domain: ai/agent/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/fundamentals
  - type/paper
---
# Scaling Long-Horizon LLM Agent via Context-Folding

Context-Folding 记忆折叠方案 by Seed

针对上下文在记忆中无法卸载的问题，之前有过初步探讨，希望通过Agent 主动上下文管理来实现【记忆裁剪】，最近发现 seed 有论文研究【记忆折叠】工作，也是聚焦于 Agent 主动管理上下文，可以深入探查一下，用以解决 Agent 上下文长度问题。

# 核心

**核心观点**：**上下文管理** 应当从 Engineering driven --> Agent driven，**都受智能体控制。**

- Context-Folding 记忆折叠方案 by Seed
- 链接：[https://arxiv.org/abs/2510.11967](https%3A%2F%2Farxiv.org%2Fabs%2F2510.11967)
![image](CZq0dctBZoUjFBxRaZncPJZInQd.png)

![image](QjJPdh71uow8wTx6zekcUqAhnA3.png)

# 精读

Existing approaches to scale long-horizon LLM agents largely fall into two classes: (1) Summary-based methods, which trigger a post-hoc summarization stage when the working context is full [1, 19, 24, 34, 38, 43]. While this compresses the context, it can abruptly disrupt the agent’s working context and reasoning flow, which may lead to sub-optimal results. (2) Multi-agent systems, which distribute tasks across specialized agents to manage context length [2, 33, 40, 41]. Yet, these systems typically depend on handcrafted, problem-specific workflows that are difficult to generalize and resist end-to-end optimization.

论文指出，基于目前研究，目前解决长上下文普遍有两种思路：

1. **总结**：当上下文多的时候对上下文进行压缩总结，缺点是突然中断推理流程，在任务执行一半的时候去压缩上下文。
1. **多 agent**：把任务解耦，以避免长上下文，但都是依赖于人工设计、针对特定问题的工作流程，难以泛化和端到端优化。
1. **补充**：目前落地还尝试过的方法包括特定区域的裁剪，比如来自某个 agent、tool 等 tag 的上下文进行裁剪，甚至对场景定制化（如只放入某些 meta data），同时可能为了避免丢失细节，又采取一些局部 summary，这些都会导致整个系统通用性变差。
**结论：合理的方式应当为，允许 Agent 主动管理上下文。**

Context-Folding机制允许代理主动管理其工作上下文。它引入了两个特殊动作：

- branch(description, prompt)：用于创建一个临时子任务（sub-task）的子轨迹（sub-trajectory），并使用独立的上下文。description是子任务的简要概括，prompt是详细指令。
- return(message)：在子任务完成后，将子轨迹的中间步骤“折叠”（folded），即从上下文中移除，只保留一个简明的摘要message，然后返回主线程。
通过这两个工具，代理可以将token密集型操作（如网页搜索或代码库探索）卸载到分支中，只保留关键发现和洞察力用于高层推理。这种方法实现了代理对上下文的主动管理，使得短期上下文不受干扰，长期上下文自动得到管理。

[https://km.sankuai.com/api/file/cdn/2732598934/202834466590?contentType=1&isNewContent=false](https://km.sankuai.com/api/file/cdn/2732598934/202834466590?contentType=1&isNewContent=false)

以深度研究任务和智能体编码任务为例，展示了这一过程：在这些任务中，智能体会将消耗 token 较多的操作（例如网页搜索或代码库探索）转移到分支中执行，仅保留关键发现与核心见解，用于后续的高层推理。

与现有方法相比，上下文折叠为主动上下文管理提供了一种 “智能体驱动式” 方案：在此方案下，智能体的短期上下文始终保持完整无中断，而长期上下文则会被自动管理。

Specifically, our RL algorithm teaches the model how to effectively decompose a problem into localized sub-tasks for branching, guided by an **Unfolded Token Penalty** that discourages token-heavy operations in the main context. Furthermore, it learns to maintain focus within a sub-task via an **Out-of-Scope Penalty**, and to preserve crucial information in its summaries to aid the final objective. By mastering these skills, the agent can handle vastly longer interaction histories, allowing our framework to scale the agent’s effective horizon and improve overall system efficiency

**FoldGRPO**—— 它通过整合（i）动态折叠的 LLM 上下文，以及（ii）密集的、token 级别的过程奖励（这些奖励直接引导上下文折叠行为），对标准 GRPO 算法进行了增强。引入两种惩罚：

- Unfolded Token Penalty
- Out-of-Scope Penalty
核心贡献：

In summary, our contributions are threefold: 

(i) We introduce Context Folding, a mechanism that enables agents to actively manage their context and mitigate the challenge of linear history growth. 

(ii) We present FoldGRPO, a reinforcement learning framework with dynamic folded LLM contexts and dense process rewards that trains agents to effectively acquire the capability of context folding. 

(iii) We demonstrate promising performance on long-horizon benchmarks, highlighting our approach as a scalable and extensible path toward stronger LLM agents.

## Context-Folding vs. MAS

上下文折叠（Context Folding）可被理解为通用多智能体系统的一种特定实现形式：在该形式中，主智能体（main agent）将子任务委派给子智能体（sub-agents）。与主流多智能体系统 [9] 相比，我们的设计在以下方面存在差异：

（i）上下文折叠不采用预定义子智能体（predefined sub-agents）；相反，子智能体由主智能体**动态创建**（on the fly）；

（ii）所有智能体共享相同的上下文前缀（context prefix），这一设计对键值缓存（KV-cache）更友好；（如果用工具实现，将没有这步的优势）

（iii）主智能体与子智能体采用**交替运行**（interleave）模式，而非并行运行（operating in parallel）。（也可能是一个缺点，对于有强依赖节点的任务无影响

## Context-Folding vs. summary-based

与基于启发式摘要的上下文管理（这类方法会在任意节点丢弃细节）不同，上下文折叠（Context Folding）可被视为一种**与子任务边界对齐的可学习摘要机制**。这确保了推理过程在执行期间得以完整保留，且仅在其效用已实现后才进行压缩。

- **heuristic summarization（启发式摘要）**：指基于预设规则（如固定保留前 N 句、删除低频词等）进行上下文压缩的方式，无需模型学习，灵活性和适配性较低。
## 落地拓展

Whether the folding agent can benefit from parallel branching — i.e., creating multiple sub-branches that run simultaneously — remains an open question. We experimented on BrowseComp-Plus by training an agent that utilizes parallel branching under the same setup as the single-branch agent. The parallel-branch version achieved a 0.6133 Pass@1 on BrowseComp-Plus, outperforming the baseline but performing similarly to the single-branch version. Moreover, after training, the parallel-branch agent created about 2.3 parallel branches on average and read more web pages (110 vs. 80 for the single-branch version). However, it did not achieve a higher score, possibly because the task characteristics are more depth-first in nature. Other tasks with a breadth-first structure (eg WideSearch [33]) may be more promising for studying parallelism in LLM agents.

结论：论文中说明并行没有带来提升，也说明了落地时候采用并行分支不会带来效果**下降**。

# 动手实验

原始论文是通过“**在推理时，通过KV-cache回滚实现高效的上下文管理。**”，但不从推理层面，直接基于工具管理上下文也能实现类似的效果，值得从应用层进行试验。

---

## See Also

- [[AI/2-Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong]] — 极长 horizon 下 context 压缩的另一路线
- [[AI/3-LLM/Inference/KV Cache|KV Cache 优化]] — Context 压缩的底层机制
-  — Agent 知识全图谱
-  — Context 管理在 LLM 层面的全景
