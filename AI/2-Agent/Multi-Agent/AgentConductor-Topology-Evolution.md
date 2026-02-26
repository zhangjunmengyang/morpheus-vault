---
title: "AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation"
brief: "上海交通大学，ICML 2026：用 RL 动态进化 Multi-Agent 拓扑结构（非固定架构），在竞赛级代码生成任务（Codeforces/LeetCode）上自适应调整 agent 角色和连接；GRPO 驱动拓扑优化（arXiv:2602.17100）"
date: 2026-02-19
updated: 2026-02-23
arxiv: "2602.17100"
authors: ["Siyu Wang", "Ruotian Lu", "Zhihao Yang", "Yuchao Wang", "Yanzhou Zhang", "Lei Xu", "Qimin Xu", "Guojun Yin", "Cailian Chen", "Xinping Guan"]
venue: "ICML 2026 (投稿)"
institution: "上海交通大学"
domain: AI/Agent
tags:
  - multi-agent
  - code-generation
  - topology
  - RL
  - GRPO
  - workflow
  - compound-ai
  - type/paper
rating: 4
status: permanent
---

# AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation

> arXiv: 2602.17100 | ICML 投稿
> 作者: Siyu Wang, Ruotian Lu, Zhihao Yang, Yuchao Wang, Yanzhou Zhang, Lei Xu, Qimin Xu, Guojun Yin, Cailian Chen, Xinping Guan
> 机构: 上海交通大学（主要）
> 提交: 2026-02-19

## 评分: ★★★★☆

## 一句话

[[Multi-Agent 概述|Multi-agent]] 竞赛级代码生成的瓶颈不是 LLM 能力，而是 **interaction topology 的设计**：easy 题用 full mesh 是浪费，hard 题用 chain 是瓶颈。AgentConductor 用 [[GRPO-Improvement-Panorama-2026|RL（GRPO）]] 训练一个轻量 orchestrator（3B）动态生成 YAML 表示的 DAG topology，随 task difficulty 和 execution feedback 演化，同时引入 density function 把"效率"纳入 reward。

---

## 问题：固定 topology 的三个死穴

现有 MAS 方法分三类：
1. **Graph pruning**（GPTSwarm 等）：离线剪枝，每个 dataset 用同一个拓扑，不适应单题难度
2. **Graph generation**（AFlow 等）：per-instance 生成拓扑，但一旦生成就冻结，不响应 execution feedback
3. **Workflow-centric RL**（FlowSteer 等）：multi-turn RL，但拓扑局限于 chain/tree 结构，表达能力弱

三者共同缺陷：
- 不能根据题目难度调整 topology **密度**（easy 题不需要 N 个 expert 相互引用）
- 不能在单题求解过程中根据执行结果**迭代进化**拓扑
- 导致：easy 题冗余 communication（token 浪费），hard 题表达能力不足（性能瓶颈）

---

## 方法：AgentConductor

### 拓扑表示：Layered DAG（YAML 编码）

拓扑 = 有层结构的 DAG，两种连接：
- **Intra-layer parallelism**：同层 agent 并行执行（解决 chain 的串行慢问题）
- **Cross-layer connections**：可跨层引用（比纯分层 DAG 更灵活，比全连接 mesh 更稀疏）

用 YAML 表示：step = layer，ref = edge，LLM 直接生成 YAML token sequence，再 decode 成 DAG。

```yaml
# 示例结构（示意）
step_1:
  - agent: Planner
step_2:
  - agent: Coder_A
    ref: [step_1/Planner]
  - agent: Coder_B
    ref: [step_1/Planner]
step_3:
  - agent: Reviewer
    ref: [step_2/Coder_A, step_2/Coder_B, step_1/Planner]  # cross-layer
```

为什么 YAML？Human-readable + LLM 直接生成，不需要特殊 parser。

### 三阶段框架

**Stage 1：SFT（拓扑先验注入）**
- 用 GPT-4o 为 4500 道题（450 竞赛 + 300 基础，各三个难度档）生成 YAML 拓扑
- 包括 first-turn（生成拓扑）和 second-turn（根据失败 feedback 迭代修改）
- 在 Qwen-2.5-Instruct-3B 上 SFT → 给 orchestrator 基础拓扑知识

**Stage 2：RL（GRPO）**
- Orchestrator 学习 task-difficulty-aware 的动态拓扑生成
- Multi-turn 轨迹：失败 → 接收 execution feedback → 重新生成拓扑 → 再执行
- Reward = Execution Reward + Graph Density Reward

**Stage 3：Inference**
- Orchestrator 冻结，zero-shot transfer 到新 dataset
- 每道题：infer difficulty → 生成初始拓扑 → 执行 → 如果失败则重新生成改进拓扑

---

## 核心设计一：Graph Density Evaluation Function

这是论文最数学的部分，也是区别于其他工作的核心。

**三个维度量化 topology 复杂度**：

```
S_node = exp(-|V| / N_max(l))        # 节点数 vs 难度上限
S_edge = exp(-|E| / (|V|(|V|-0.5)))  # 边密度 vs 完全图
S_depth = 1 - s/|V|                  # 层深 vs 节点总数（并行度）
```

**综合密度分数**：
```
S_complex = exp(S_node + 2·S_edge + S_depth)
```

权重 2 在 S_edge：边是最直接的 token cost 来源（每条边 = 一次引用 = 更多 context）。

**难度分档（Difficulty Interval Partitioning）**：
- 每个难度级别 l 有对应的 N_max(l)
- 这防止了"过度剪枝"：easy 题 N_max 小，hard 题 N_max 大，不会把 hard 题的拓扑剪得和 easy 题一样稀疏
- 实现精细的"难度 → 密度上限"映射

论文有 Theorem 1 证明 S_complex 的数学性质（density 单调性、归一化等）。

---

## 核心设计二：Multi-Objective Reward

```
r_φ(G^(k), z^code_k) = r_e(G^(k), z^code_k) + r_g(G^(k))
```

**Execution Reward r_e**（两部分）：

YAML 格式正确性：
| Error | Reward |
|-------|--------|
| NO_YAML_FOUND | -2.0 |
| YAML_PARSE_ERROR | -1.5 |
| YAML_SCHEMA_INVALID | -1.0 |
| YAML_LOGIC_INVALID | -0.5 |

Code 执行结果：
| Result | Reward |
|--------|--------|
| WRONG_ANSWER | +1.0 |
| TIME_LIMIT_EXCEEDED | +0.9 |
| MEMORY_LIMIT_EXCEEDED | +0.8 |
| RUNTIME_ERROR | +0.7 |
| COMPILATION_ERROR | +0.6 |

注意：即使答案错，只要能执行（WRONG_ANSWER）也比编译错误分高 → 鼓励先让代码跑起来。

**Graph Density Reward r_g**：基于 S_complex 的密度评分，在满足准确率的前提下鼓励稀疏拓扑。

---

## 实验结果

**Benchmark**：3 个竞赛级（CodeForces/AIZU/HackerEarth 类型）+ 2 个基础代码 dataset

**主要结果**（vs 最强 baseline）：
- Pass@1 准确率：**+14.6%**（竞赛级最难题型上的提升）
- 平均 topology 密度：**-13%**（更稀疏的拓扑）
- Token 开销：**-68%**（大幅降低 API 成本）

三个指标同向改善：准确率更高 + 拓扑更稀疏 + 成本更低。

**Zero-shot transfer**：orchestrator 冻结后迁移到新 dataset 效果稳定（Stage 3 设计的验证）。

**Transfer to new task types**：在 operator library 里加入新 role 后只需少量额外训练，泛化能力强。

---

## 我的分析

### 为什么 +14.6% 同时 -68% token cost？

直觉上这两个方向是矛盾的（更多 agent 通信 = 更高准确率 vs 更少通信 = 更低开销）。

AgentConductor 的 insight：**问题本身的难度决定了需要多少 agent 协作**。Easy 题用 2 个 agent 就能解决，用 6 个反而引入噪声和错误传播；Hard 题用 6 个 agent 的 dense cross-layer topology 才能发挥优势。

固定 topology 只能选一个点（要么为 hard 题配置 → easy 题过度消耗，要么为 easy 题配置 → hard 题能力不足）。动态 difficulty-aware topology 同时优化了两端。

### 和 FlowSteer 的对比

两篇论文有相似性（都是 RL + multi-turn + workflow），但本质不同：

| | FlowSteer | AgentConductor |
|--|-----------|----------------|
| 核心 | 自动化 workflow 结构 | 动态 MAS 拓扑（agent间通信图） |
| 任务 | QA/数学/代码（通用） | 竞赛级代码（专精） |
| 拓扑 | DAG of operators | DAG of agent roles |
| 密度控制 | diversity reward（heuristic） | 数学 density function（理论） |
| Difficulty-aware | 无 | 有（核心设计） |

AgentConductor 的密度函数更 principled，FlowSteer 的 operator 抽象更通用。

### 技术上有多新颖？

**Novel**：
1. Difficulty-aware topology density 是真正的贡献——之前方法没人把"难度→允许通信密度"做成连续的量化函数
2. YAML 表示 DAG + LLM 直接生成，比 JSON 或 code 更 lightweight，对 3B 小模型友好
3. 分层 DAG 同时支持 intra-layer parallel 和 cross-layer，比 chain/tree 表达力强，比 mesh 更可控

**不那么新的**：
- SFT + GRPO 两阶段 pipeline 已是 agentic RL 标配
- Multi-objective reward 设计是工程调参，理论依据有限
- "Execution feedback → 重新生成" 的 multi-turn 思路和 FlowSteer/REMuL 等同期工作类似

### 局限性

1. **Orchestrator 本身的质量**：整个系统依赖 3B orchestrator 正确 infer difficulty 和生成合法 YAML，这里的错误代价很高（-2.0 reward）
2. **Role pool 是预定义的**：灵活性受限，加新 role 需要重新训练
3. **竞赛代码场景**：有 ground-truth test case 才能用 execution feedback，不能直接迁移到 open-ended 代码任务

---

## See Also

- [[FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer CWRPO]] — 同为 GRPO + workflow multi-turn RL，但拓扑表达力弱于 AgentConductor（chain/tree vs DAG）；FlowSteer 更通用，AgentConductor 更专精
- [[IMAGINE-多Agent蒸馏到单模型|IMAGINE]] — MAS 反方向：把多 Agent 集体能力蒸馏进单模型（能力集中）vs AgentConductor（动态组合）
- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — AgentConductor 使用 GRPO 训练 orchestrator 的算法基础
- [[LLM代码生成-2026技术全景|LLM 代码生成 2026 全景]] — 代码生成赛道全局视图，AgentConductor 在竞赛级代码的位置
-  — Agent 研究全图谱

## 连接

- 前驱/竞争：FlowSteer（2602.01664）、AFlow、GPTSwarm、LATS
- RL 基础：GRPO（Shao 2024）
- 任务：竞赛级代码（CodeForces 类型问题）
- 更大图景：MAS 拓扑优化是 compound AI 的核心问题
