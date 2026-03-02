---
title: Search-R1++ — Deep Research RL 训练三要素消融
brief: "对 Deep Research 场景做系统性 RL 消融：Slow Thinking 会被 <think> 捷径 exploit、F1 reward 诱发 answer avoidance、GRPO 稳定性最差；最终推荐 Fast Thinking + F1+ action penalty + REINFORCE 作为稳健配置。"
arxiv: "2602.19526"
date: 2026-02-23
rating: ★★★★☆
tags:
  - agent-rl
  - tool-use-rl
  - search-agent
  - reward-design
  - policy-optimization
  - deep-research
related:
  - "[[AI/2-Agent/Agentic-RL/Search-R1-Reasoning-Search-Engine-RL]]"
  - "[[AI/2-Agent/Agentic-RL/Tool-Use-RL-训练专题]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操]]"
---

# Search-R1++ — Deep Research RL 训练三要素消融

> **一句话**：Search-R1 框架的系统性 RL 训练消融，发现 Slow Thinking 崩溃根因、F1 reward 的 answer avoidance 陷阱、GRPO 稳定性最差，最终搭出 Search-R1++ 将基线从 0.403 → 0.442。

**作者**：Yinuo Xu, Shuo Lu, Jianjie Cheng, Meng Wang 等（CASIA + Meituan Inc.）
**发表**：arXiv:2602.19526（2026-02-23）
**Venue**：arXiv preprint

---

## 核心价值

这篇论文的贡献不是提出新算法，而是**做了别人该做但没做的消融实验**：把 Deep Research RL 训练拆成三个维度，每个维度系统对比。

研究框架的三个正交维度：
- **Prompt Template**：Fast Thinking vs Slow Thinking
- **Reward Function**：EM vs F1 vs F1+（action penalty augmented）
- **Policy Optimization**：REINFORCE vs PPO vs GRPO

评估指标三维度：预测精度 / 训练稳定性 / 推理代价（search actions 数量）

---

## 发现一：Slow Thinking 崩溃根因

### 现象
Slow Thinking 模板（Search-R1 原版）训练中频繁出现 collapse：score 先爬升后骤跌。

### 崩溃机制（关键！）
1. Policy 学会了一个**捷径**：`<think>` tag 数量与 reward 有正相关（Pearson r=0.43，崩溃前 10 步内）
2. 稳定训练时 `<think>` 数量与 reward 几乎不相关（r=-0.0465）
3. 在 PPO 的稀疏 reward 结构下，policy 把"堆叠 `<think>` tag"当成优化方向
4. 最终演化成不断输出空的 `<think></think>` block，挤占正常决策空间

### Fast Thinking 模板（Search-R1++ 采用）

```
Template 1 (Fast Thinking):
Answer the given question. If you need external knowledge, 
call the search engine using <search> query </search>...
Use the returned information directly to produce the final answer.
When you have enough information, provide the answer inside <answer>...</answer>.
```

vs

```
Template 2 (Slow Thinking):  
Answer the given question. You must conduct reasoning inside <think> and </think> 
first every time you get new information. After reasoning, if you lack some 
knowledge, you can call a search engine...
```

关键区别：Fast Thinking **完全去掉 `<think>` 约束**，让 policy 直接输出 search/answer 决策。

### 实验结果
Fast Thinking 比 Slow Thinking：
- Qwen2.5-7B：0.403 → 0.422（+4.7%）
- Qwen2.5-3B：0.289 → 0.297（+2.8%）
- 训练稳定性显著提升，collapse 消失

### 洞察
**"更多思考 ≠ 更好性能"**。统计分析发现，reasoning tokens 和 information tokens 的增加与精度**负相关**。这与数学/代码任务的 Chain-of-Thought 规律相反——在 Deep Research 的 retrieve-rethink 循环中，显式 `<think>` 约束反而引入了可被 exploit 的捷径。

---

## 发现二：F1 Reward 的 Answer Avoidance 陷阱

### 问题
社区从 EM reward 迁移到 F1 reward 的动机：F1 更软，避免精确匹配过严。
但 Search-R1++ 发现：**F1 反而更不稳定，最终性能比 EM 差**。

### 根因：Answer Avoidance

```
F1 reward 下的 policy 学到的捷径：
  - 答错 → 得 0
  - 不答 → 得 0
  → 理性选择：不答（避免复杂推理的代价）
```

证据：collapse 时 overall score 骤降，但 answered-only accuracy 保持稳定 → 失败根因是**拒绝回答率上升**，不是回答变差。

### 修复：F1+（Action-Level Penalty Augmentation）

$$R_{F1+} = R_{F1} - \alpha \cdot \mathbb{I}[a_s = 0] - \beta \cdot \mathbb{I}[a_a = 0]$$

其中：
- $a_s$：本步执行的 search action 数量
- $a_a$：本步生成的 answer 数量
- $\alpha = \beta = 0.1$（轻量级 penalty）
- $\mathbb{I}[\cdot]$：indicator function

**效果**：F1+ 不仅消除 collapse，最终性能还**超过 EM baseline**。

### 对比结果（Qwen2.5-7B）

| Reward | Avg EM Score | Stability |
|--------|-------------|-----------|
| EM | 0.422 | 高 |
| F1 | 低于 EM | 经常 collapse |
| **F1+** | **>EM（最高）** | **高** |

---

## 发现三：REINFORCE > PPO > GRPO（稳定性逆序）

### 结论
这是整篇论文最反直觉的发现之一：

- **GRPO**：稳定性最差（三者中最低），Deep Research 场景下不推荐
- **PPO**：中等，有 Critic 开销
- **REINFORCE**：稳定性最好，最终精度最高，且 search actions 数量最少（推理代价最低）

### 为什么 GRPO 稳定性差？

GRPO 的 group normalization 在 Deep Research 中有问题：
- group 内同质性导致 advantage variance 过低 → 梯度消失
- 或 group 内异质性过高 → 梯度 spike

这与 RAGEN/StarPO（Echo Trap 论文 2504.20073）的 variance collapse 观察一致：**multi-turn tool use RL 中，group-based advantage 估计比 single-sample 的 REINFORCE 更脆弱**。

### 为什么 REINFORCE 更好？

REINFORCE 的 advantage 估计：$A_t = G_t - b$ （直接用 return 减 baseline）

在 Deep Research 的稀疏 reward 结构下：
- 每个 rollout 的 return 差异直接体现问题难度
- 不依赖 group 内部的相对比较
- 没有 PPO 的 clip 截断导致的梯度偏差

---

## Search-R1++ 配置汇总

| 组件 | 选择 | 原因 |
|------|------|------|
| Prompt Template | Fast Thinking | 消除 `<think>` exploitation 捷径 |
| Reward Function | F1 + action penalties | F1 精度 + 防 answer avoidance |
| Policy Optimizer | REINFORCE | 最高稳定性 + 最低推理代价 |

**最终性能**：
- Qwen2.5-7B：0.403 → **0.442**（+9.7%）
- Qwen2.5-3B：0.289 → **0.331**（+14.5%）

**7 个评测集**：NQ / TriviaQA / PopQA（General QA）+ HotpotQA / 2WikiMultiHop / Musique / Bamboogle（Multi-Hop QA）

---

## 与 Search-R1 原版的对比分析

| 维度 | Search-R1（原版） | Search-R1++ |
|------|------|------|
| Prompt | Slow Thinking（`<think>`强制） | Fast Thinking（无`<think>`约束） |
| Reward | EM 或 F1（单独） | F1 + action-level penalty |
| Optimizer | PPO | REINFORCE |
| 稳定性问题 | 频繁 collapse | 消除 |
| 性能 | 0.403（7B）/ 0.289（3B） | 0.442 / 0.331 |

---

## 落地应用价值

### 训练 Deep Research Agent 的工程 checklist
1. **Prompt 设计**：不要强制 `<think>` tag，让 policy 自由决定推理深度
2. **Reward 设计**：纯 outcome reward（EM/F1）在 tool use 场景下容易被 exploit → 加 action-level penalty
3. **Optimizer 选择**：Deep Research 用 REINFORCE；数学推理用 GRPO（不同任务结构不同选择）
4. **Collapse 诊断**：score 骤降 → 先看 answer rate，不是看 accuracy → 可能是 avoidance 而非理解问题

### 对 Token Masking 统一框架的补充
- Search-R1++ 的 token masking（retrieved information 不参与梯度）在这篇论文保持不变
- 新增的是 action-level penalty，本质是"强制某些动作必须发生"
- 这补充了 Tool Use RL 的**奖励完备性**问题：outcome-only reward 不足以约束 intermediate actions

### GRPO vs REINFORCE 适用场景总结
| 场景 | 推荐 Optimizer | 原因 |
|------|-------------|------|
| 数学/代码（可验证，单轮） | GRPO | group 内对比有意义，advantage 估计稳定 |
| Deep Research（multi-turn，稀疏 reward） | REINFORCE | group 内异质性高，直接 return 更稳定 |
| 一般 agent（tool use） | PPO 或 REINFORCE | 取决于 reward 密度 |

---

## 批判性评估

**值得认可**：
- 系统性消融，每个变量单独隔离，实验设计干净
- 三个发现都有明确的机制解释（不只是"我们试了，更好"）
- F1+ 的 action-level penalty 设计简洁优雅，α=β=0.1 轻量

**需要谨慎的地方**：
- 实验只在 Qwen2.5-3B/7B 上做，泛化性待验证
- REINFORCE > PPO 可能是超参数调优不到位，PPO 有更多需要调的地方
- Fast Thinking "去掉 `<think>`" 可能对更复杂的多跳推理有代价（longer chain 在 multi-hop 里仍然可能有效）
- action-level penalty 的 reward hacking 风险：policy 可能学会频繁发出无意义 search actions

---

## 与相关工作的交叉链接

- [[AI/2-Agent/Agentic-RL/Search-R1-Reasoning-Search-Engine-RL]] — 原版，token masking + EM reward
- [[AI/2-Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution]] — Echo Trap 分析（GRPO 在 multi-turn 崩溃机制）
- [[AI/2-Agent/Agentic-RL/Tool-Use-RL-训练专题]] — ASTRA / ToolRL / Search-R1 综合

---

> **vault_gap（已解决）**：本笔记解决馆长 2026-02-27 标注的 Search-R1++ 独立精读缺失
