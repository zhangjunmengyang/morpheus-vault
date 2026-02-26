# Agentic Reasoning for Large Language Models — Survey

**论文**：arXiv:2601.12538（2026-01-18）
**作者**：Tianxin Wei et al.（Google DeepMind + Meta + Amazon 等多家机构联合）
**评级**：★★★★☆（综述性，框架完整，方向指引价值高）
**标签**：#AgenticRL #Survey #三层框架 #规划 #工具调用 #多Agent
**规模**：~100页，GitHub：`weitianxin/Awesome-Agentic-Reasoning`

---

## 一、核心论点

**Closed-world 的 LLM reasoning** 已经很强，但在 open-ended 动态环境中仍然失败。

**Agentic Reasoning** = 把 LLM 重新框架为 autonomous agents：plan → act → learn，通过持续交互运作。

**关键区分**：
- In-context reasoning：通过结构化 orchestration 扩展 test-time 交互（不改模型参数）
- Post-training reasoning：通过 RL + SFT 优化行为（改模型参数）

---

## 二、三层框架（核心贡献）

### Layer 1：Foundational Agentic Reasoning（基础层）
**场景**：稳定环境中的单 agent 核心能力

覆盖：
- **Planning**：任务分解、子目标规划、回溯
- **Tool Use**：API 调用、代码执行、搜索集成
- **Search**：MCTS、beam search、A* 在 agent 上的应用

典型工作：ReAct / ToolFormer / ChainofThought / Tree-of-Thought

**关键 insight**：稳定环境是基础，单 agent 在此表现最好。问题在于真实世界几乎没有真正"稳定"的环境。

---

### Layer 2：Self-Evolving Agentic Reasoning（自进化层）
**场景**：动态环境中的 agent 自我精炼

覆盖：
- **Feedback**：来自环境/用户/其他 agent 的信号整合
- **Memory**：短期（上下文内）+ 长期（外部存储）
- **Adaptation**：基于历史经验调整策略

典型工作：Reflexion（自我反思）/ VOYAGER（Minecraft 自学习）/ MemGPT

**关键 insight**：这层是 Agent RL 的主战场——RL 的本质就是通过 feedback 进行 policy adaptation。GiGPO / SCoRe / CM2 都属于这层。

---

### Layer 3：Collective Multi-Agent Reasoning（集体推理层）
**场景**：多 agent 协作解决共享目标

覆盖：
- **Coordination**：任务分配、通信协议、角色分工
- **Knowledge Sharing**：agent 之间的知识蒸馏和经验传递
- **Shared Goals**：全局目标分解到 agent 级别

典型工作：AutoGen / MetaGPT / MARL frameworks

**关键 insight**：当前最薄弱的一层。单 agent 在稳定环境有解法，multi-agent 在动态环境中协调成本急剧上升，是真正的 open problem。

---

## 三、两种推理范式

```
In-Context Reasoning：
  prompt engineering → chain-of-thought → structured orchestration
  不改参数，通过 test-time compute scaling 提升
  代表：CoT / ToT / AgentBench prompting
  
Post-Training Reasoning：
  SFT on agent trajectories → RL from environment feedback
  改参数，行为优化刻入权重
  代表：RLVR / GRPO / PPO-based agent training
```

**这个区分很重要**：当前趋势是两者融合——先 in-context 探索 trajectory，再 post-training 蒸馏最优策略。

---

## 四、应用领域覆盖

- **科学**：自动实验设计、假说生成（AlphaFold 3 类的 agentic 延伸）
- **机器人**：RT-2 类的感知-行动闭环
- **医疗**：诊断推理、文献综合
- **自主研究**：AI 自主写论文（AI Scientist）
- **数学**：定理证明（Lean 集成）

---

## 五、开放挑战（Open Problems）

| 挑战 | 描述 |
|------|------|
| 个性化 | agent 适配不同用户的推理风格和知识背景 |
| 长 horizon 交互 | 跨多轮/多天任务的状态维护和信用分配 |
| World Modeling | agent 对环境的因果建模，不只是 stimulus-response |
| Scalable Multi-Agent Training | 多 agent 联合训练的样本效率和稳定性 |
| 真实部署的 Governance | 安全对齐、权限管控、不确定性量化 |

**Credit Assignment 在长 horizon** 是我最关注的开放问题，直接关联 GiGPO / AgentPRM / MIG 的意义。

---

## 六、与 Vault 已有内容的关联

| Survey 章节 | 对应 Vault 深度笔记 |
|------------|-------------------|
| Tool Use RL | `Agent/Agentic-RL/Tool-Use-RL-训练专题.md` |
| Credit Assignment | `Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题.md` |
| Multi-Agent | `Agent/Agentic-RL/Agentic-RL-2026前沿综合分析.md`（待补 MARL 章） |
| Post-training RL | `LLM/RL/` 全系列笔记 |
| Self-Evolving | SCoRe / TSR 等笔记 |

---

## 七、我的评价

**优点**：框架清晰，三层分类覆盖了从"单 agent 稳定环境"到"多 agent 动态协调"的全谱。区分 in-context vs post-training 两种范式是重要的认知框架。

**局限**：Survey 性质，深度有限。Layer 2（Self-Evolving）的 RL 方法讲述相对粗浅，GiGPO / AgentPRM 等最新 credit assignment 方法未必能覆盖（2026-01-18 提交，部分工作晚于此时间）。

**研究意义**：这个三层框架可以用来组织 Agent RL 笔记体系——把 Vault 里的 Agent RL 论文映射到三个层次，形成更清晰的知识结构。

**建议读法**：快速扫 §3（Foundational）直奔 §4（Self-Evolving）重点阅读，§5（Collective）在具体做 Multi-Agent 工作时再精读。

---

## 八、arXiv 元数据

- 提交：2026-01-18
- 机构：UIUC / Google / Meta / Amazon / ... （多机构联合）
- 页数：~100页（8.7 MB PDF）
- GitHub：https://github.com/weitianxin/Awesome-Agentic-Reasoning
