---
title: P5：分析 Agent——从 ReAct 到 RL 训练闭环
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, agent, RL, GRPO, reward-design, analysis-agent, meituan]
brief: 基于美团节假日分析 Agent 的真实业务经验，延伸设计了把 GRPO 用于 Agent 训练的完整闭环：reward 设计、数据收集、训练范式选择，回答了"业务 Agent 如何从 prompt 工程走向后训练"的核心问题。
related:
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析]]"
  - "[[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization]]"
  - "[[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL]]"
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解]]"
  - "[[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]]"
---

# P5：分析 Agent——从 ReAct 到 RL 训练闭环

> **一句话定位**：从美团节假日分析 Agent 的真实业务出发，设计了把 GRPO 用于分析 Agent 训练的完整方案——这是业务 Agent 从 prompt 工程走向后训练的完整思路链。

---

## 背景故事（面试口径）

> 怎么引入：

"我们在美团做节假日分析 Agent，一个挂着取数工具的 Agent，能自主取数、写出节假日分析报告。这个 Agent 用的是 ReAct 框架，效果还不错——但有一个明显的天花板：它的分析深度取决于 prompt 里有没有写'要对比同比数据'、'要归因到流量/转化率'。

换句话说，分析质量完全依赖 prompt 工程。你能写多好的 prompt，Agent 就能做多好的分析。但人不可能把所有分析套路都写进 prompt——有些认知是隐性的、情境化的。

这让我开始思考：能不能用 RL 训练这个 Agent，让它从历史分析案例里学会'好分析'应该是什么样的？"

---

## 系统设计：ReAct 分析 Agent 基线

**工具集**：
```
节假日分析 Agent 工具：
  - get_metrics(date_range, dimensions)：取核心指标（GMV/订单/流量/转化率）
  - get_peer_benchmark(date_range)：取竞品对标数据
  - get_holiday_calendar()：节假日日历（触发分析时机）
  - query_historical_patterns(holiday_type)：历史同类节假日表现

输出目标：结构化分析报告（异动识别 + 根因归因 + 策略建议）
```

**ReAct 流程**：
```
Thought: 用户想了解这次五一假期的经营表现
Action: get_metrics(date_range="2025-05-01:2025-05-05", dimensions=["GMV","订单","流量"])
Observation: GMV 下滑 15%，流量持平，转化率下滑 12%
Thought: 转化率下滑是主因，需要看套餐/价格维度
Action: get_metrics(..., dimensions=["转化率", "套餐点击率", "价格分布"])
Observation: ...
...
Final Answer: 分析报告
```

**现有问题**：
1. 分析深度不够——不总会想到要看竞品对标
2. 归因不系统——有时候只说"转化率下滑"，不深挖是哪个套餐、哪个价格段
3. 建议不具体——提完问题不给行动方案
4. Token 浪费——很多 Thought 步骤是无效的重复推理

---

## 关键设计：把 GRPO 用于分析 Agent

### 为什么选 GRPO 不选其他方法？

**方案对比**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 更好的 prompt 工程 | 零成本 | 天花板低，不能迁移到新场景 |
| SFT（专家示范） | 简单、有效 | 需要大量高质量标注，成本高 |
| DPO（偏好对比） | 不需要 reward model | 需要 chosen/rejected 配对，offline |
| GRPO | 在线学习，从实际执行中学 | 需要 reward 设计，计算开销大 |

**为什么 GRPO 适合分析 Agent**：

"分析 Agent 有一个特性：虽然没有数学题那样的精确验证器，但'好分析'和'差分析'是可以被评估的——我们有历史的人工评审记录、有商家的后续行动数据（接受建议 vs 忽略建议）。这些都可以构成 reward signal。"

### Reward 设计（最关键的部分）

**多维度 reward 函数**：

```python
def compute_analysis_reward(trajectory, metadata):
    reward = 0.0
    
    # 1. 结构完整性 reward（可验证，权重 0.2）
    report = extract_final_report(trajectory)
    if has_section(report, "异动识别"): reward += 0.1
    if has_section(report, "根因归因"): reward += 0.1
    if has_section(report, "策略建议"): reward += 0.1
    
    # 2. 工具调用效率 reward（可验证，权重 0.2）
    tool_calls = extract_tool_calls(trajectory)
    unique_dimensions = len(set([t.dimension for t in tool_calls]))
    redundant_calls = count_redundant_calls(tool_calls)
    reward += 0.2 * (unique_dimensions / MAX_DIMS) - 0.1 * redundant_calls
    
    # 3. 归因深度 reward（半自动，权重 0.3）
    # 判断是否做了多层下钻（流量下滑 → 哪个渠道 → 哪个时段）
    attribution_depth = measure_attribution_depth(report)
    reward += 0.3 * min(attribution_depth / 3, 1.0)  # 最多 3 层
    
    # 4. 建议可操作性 reward（需要 LLM judge，权重 0.3）
    # 用另一个 LLM 评判建议是否具体、是否可立即执行
    actionability_score = llm_judge_actionability(report)
    reward += 0.3 * actionability_score
    
    return reward
```

**关键设计决策：为什么 reward 4 用 LLM judge 而不是规则？**

"'建议是否可操作'本质上是主观判断，规则写不完。但我们可以用另一个更强的 LLM（GPT-4 或 Claude Opus）当裁判，让它评判分析质量。这个思路来自 CM2 论文——把不可验证的 reward 转化为 LLM judge 的 checklist 评分，每一项 checklist 是明确的（'建议中是否包含具体的时间节点？'），可以量化。"

**Reward Shaping（解决稀疏 reward 问题）**：

```
Step-level reward（参考 GiGPO）：
  - 每次工具调用后，不只看最终报告质量
  - 中间步骤也给 reward：调用竞品数据 +0.05，归因到第二层 +0.1
  - 这样避免了'前 9 步做得好但最后一步失败就全没 reward'

Format reward（稳定训练）：
  - 要求 Agent 在 Thought 里明确写出归因假设
  - 有明确假设的 Thought 比无结构的 Thought reward 高
```

### 训练范式选择

**为什么不直接用 GRPO，而是参考 SeeUPO？**

"标准 GRPO 在 multi-turn agent 场景下有理论问题——SeeUPO 证明了 GRPO 的 variance normalization 在多轮场景下会破坏收敛性。分析 Agent 是典型的 multi-turn 场景（取数 → 分析 → 再取数 → 报告，可能 10+ 轮）。

所以我们的训练方案参考 SeeUPO 的设计：
1. 去掉 variance normalization（用 group 内的 raw reward 差异，不除以 std）
2. 用逆序更新（先更新最后一步，再更新前面的步骤）
3. 配合 GiGPO 的 step-level credit assignment：不同轮次的工具调用有不同 advantage"

### 数据收集方案

**怎么收集训练数据？**

```
来源 1：历史分析记录（offline）
  - 几年的节假日分析报告，有人工质量标注
  - 问题：分布和当前 Agent 行为差异大（人写的 vs Agent 写的）
  
来源 2：在线自动收集（online）
  - Agent 运行时记录完整 trajectory
  - 用 reward function 自动打分
  - 低质量 trajectory 过滤掉（reward < 0.3 的不进训练集）
  
来源 3：对比采样（group sampling）
  - 同一个分析任务，Agent 采样 8 个不同 trajectory
  - Group 内对比，用 GRPO advantage 计算
  - 这就是 GRPO 的核心机制
```

---

## 与业务结合的洞察（面试亮点）

**Agent RL 的 reward 设计比算法本身更重要**

"在这个项目里，我深刻体会到 Agentic RL 元问题笔记里说的：当前 Agentic RL 的瓶颈不是算法（GRPO 已经够用），而是 Reward Signal Quality。分析 Agent 的困难不在于用什么 RL 算法，而在于怎么定义'好分析'。

具体来说，我们遇到了三个 reward 设计难题：
1. **短期 reward 和长期价值的错位**：Agent 写了很长的分析，reward 高；但商家读了之后没有行动，说明分析没用。但这个信息要两周后才能收集到。
2. **Goodhart's Law**：reward function 定义好之后，Agent 会找到'刷分'方法——比如在报告里堆砌大量的'建议'，每条建议都很具体，但彼此矛盾。
3. **分布 shift**：训练时的分析场景（历史节假日）和推理时的场景（新的节假日模式）不完全一样，模型会过拟合历史模式。"

**解法：从 CM2 论文学到的 Checklist reward**

"CM2 的核心思想：把不可验证的 reward 转化为可验证的 checklist。'分析质量高'这个模糊评价，可以拆解成：
- ✅ 是否识别了主要异动指标？
- ✅ 是否追溯到了至少两级根因？
- ✅ 每条建议是否包含具体行动？
- ✅ 建议是否和根因匹配（因流量下滑建议活动，不因转化率下滑建议活动）？

每个 checklist item 是可验证的，加权求和得到 reward。这把主观评估变成了可计算的指标。"

---

## 一句话总结（面试结尾）

"这个项目最重要的产出是一套思路：如何把一个业务 Agent 从 prompt-only 系统升级为有训练闭环的系统。关键路径是 reward 设计——你得先想清楚什么是'好行为'，才能告诉模型去学。这比算法本身难多了。"

---

## See Also

- [[Projects/项目故事/P2-后训练大项目-MA-RLHF工程实战]] — 技术基础
- [[Projects/项目故事/P4-商家诊断Agent-安全防御层]] — 同一业务背景
- [[AI/2-Agent/Agentic-RL/Agentic-RL-元问题-瓶颈与突破方向]] — 深度认知来源
- [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization]] — Step-level credit
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL]] — Checklist reward
- [[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]] — Multi-turn RL 理论
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解]]
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization]] — 关键步骤优化
