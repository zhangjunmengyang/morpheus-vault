---
title: "Aletheia — Gemini Deep Think 数学科研 Agent"
brief: "Google DeepMind：从竞赛解题跨越到全自主科研论文生成的数学研究 Agent；在 Erdős 猜想数据库自主解决 4 个开放问题；三组件 Agentic Loop（生成/验证/探索分离）；formal math 的 AI frontier"
type: research
domain: ai/agent
updated: 2026-02-22
tags:
  - ai/agent
  - ai/llm/reasoning
  - ai/llm/inference
  - type/research
  - model/gemini
  - topic/math-ai
  - topic/scientific-discovery
created: 2026-02-19
---

# Aletheia — Gemini Deep Think 数学科研 Agent

> 发布：2026-02-11 (Google DeepMind)
> 论文 1：[arXiv 2602.10177](https://arxiv.org/abs/2602.10177) — 数学科研 agent 综述（Aletheia）
> 论文 2：[arXiv 2602.03837](https://arxiv.org/abs/2602.03837) — 跨学科科研结果（CS/物理）
> 代码/Prompts：[github.com/google-deepmind/superhuman](https://github.com/google-deepmind/superhuman/tree/main/aletheia)
> 关键词：math AI, agentic reasoning, scientific discovery, TTC scaling, NL verifier

## 为什么这篇重要

这不是 benchmark 提升，这是从 **"AI 做题"** 到 **"AI 做研究"** 的真正跨越。

历史坐标：
- AlphaGo/AlphaFold → 有明确反馈信号的封闭域，AI 超越人类
- Aletheia → **开放式数学/科学研究**，几乎无明确反馈信号，AI 开始产出 publishable 成果

## 架构设计

Aletheia 是 Gemini Deep Think 的 agentic wrapper，核心设计：

```
问题输入
    ↓
[生成候选解] ← Gemini Deep Think (TTC scaling)
    ↓
[NL Verifier] — 识别逻辑缺陷、数学错误
    ↓ (发现问题)
[迭代修订] ← 重新生成 or 局部修正
    ↓ (多轮后)
[失败承认] ← 关键设计：宣告无法解决，返回研究者
    ↓ (解决) 
输出解答
```

### 关键设计决策分析

**1. Natural Language Verifier（非形式化证明系统）**
- 牺牲形式化保证，换取适用范围的指数级扩展
- 形式化方法（Lean/Coq）仅适用于可被完全形式化的数学，NL verifier 可以处理物理、CS 理论等半结构化领域
- 代价：verifier 本身可能犯错 → 需要人在循环兜底

**2. 失败承认机制（Failure Admission）**
- 大多数 agent 系统会无限重试或生成垃圾输出
- Aletheia 可以 "承认失败"，将问题标记为当前能力范围外
- 实践意义：显著提升人类研究者效率（不浪费时间等待无效输出）
- 理论意义：这是 calibration 的一种形式，模型知道自己不知道什么

**3. Google Search Grounding**
- 防止 spurious citations（数学 hallucination 的常见形态）
- 可以主动检索相关文献，避免重复已知结论

**4. 迭代生成-验证-修订循环**
- 不是 one-shot，而是多轮对话式精化
- 每轮都有 NL verifier 给出具体反馈
- 收敛条件：verifier 通过 OR 达到失败阈值

## 科研成果（已产出）

### 数学领域（论文 2602.10177）

| 成果 | 自主程度 | 状态 | 说明 |
|------|----------|------|------|
| Feng26 (arXiv 2601.23245) | **完全自主** (Level 2) | 投顶刊 | Arithmetic geometry eigenweights 计算，零人工干预 |
| LeeSeo26 (arXiv 2602.02450) | 人机协作 (Level 2) | 投顶刊 | Independent sets 粒子交互界的证明 |
| Erdős 猜想评估 | 半自主 | 发表 | 700 个开放问题，自主解决 4 个 |
| Erdős-1051 推广 | AI 辅助 | BKKKZ26 (arXiv 2601.21442) | AI 解决并触发人类的进一步推广 |

**TTC Scaling 数据：**
- IMO-ProofBench Advanced：90%（随推理算力增加持续提升）
- Scaling law 在 PhD 级问题（FutureMath Basic benchmark）仍然成立
- Aletheia 实现了"更高推理质量 + 更低推理算力"的 Pareto 改进

### CS 理论 + 物理（论文 2602.03837）

**跨域迁移（最令人震惊的能力）：**
- **Max-Cut / Steiner Tree**：用连续数学工具（Kirszbraun 定理、测度论、Stone-Weierstrass）破解多年僵局的离散算法问题
- 这是真正的 "跨越数学边界"——不是在同一领域内优化，而是从完全不同的分支引入工具

**推翻旧直觉：**
- 2015 年 submodular 优化领域一个被接受十年的"显然"直觉（stream item copy 的价值不如 move original）
- Aletheia 构造三元反例，严格证明其为假
- 这类 counterexample 构造在历史上需要人类数年才能完成

**其他亮点：**
- 自动调参解释：证明某 ML 正则化技术之所以成功，是因为它在训练中隐式生成了自适应 penalty
- 经济理论扩展：用拓扑学和序理论将 AI token 拍卖的 Revelation Principle 从有理数域扩展到实数域
- 宇宙弦物理：用 Gegenbauer 多项式找到包含奇点的积分的解析解
- STOC'26 peer review：AI 辅助 CS 理论论文审稿，已在实际学术会议部署

### AI 辅助数学分类体系（新提出）

| Level | 名称 | 定义 | Aletheia 当前 |
|-------|------|------|---------------|
| L1 | Verification | 验证人类提出的结论 | ✅ |
| L2 | Publishable Quality | 独立产出可发表成果 | ✅ |
| L3 | Major Advance | 解决重要开放问题 | ❌（未声明） |
| L4 | Landmark Breakthrough | 领域里程碑 | ❌（未声明） |

Google 明确声明当前仅达到 L2，未声明 L3/L4——这种 epistemic humility 本身也是一个重要信号。

## 协作模式（"Advisor" Model）

论文 2602.03837 提出了有效的人机协作 "recipe"：

- **Vibe-Proving 循环**：人类提供直觉，AI 验证或反驳，迭代精化
- **Balanced Prompting**：同时要求 proof AND refutation，防止 confirmation bias
- **Code-Assisted Verification**：用代码验证数学 claim，减少纯语言推理的错误

## 我的批判性分析

### 真正 novel 的是什么

1. **跨域工具迁移**：Max-Cut 用连续数学工具这件事，在历史上通常需要人类数年偶然发现。AI 可以系统性地探索这种跨越，因为它的"知识库"没有学科边界
2. **NL verifier 取代形式化系统**：这个设计决策奠定了实用性，代价是可靠性上限，但覆盖面无限扩展
3. **失败承认**：calibrated AI 比 always-confident AI 在科研场景中更有价值，这是对 RLHF "越自信越被选择" 偏差的有意矫正

### 需要保持怀疑的地方

1. **NL verifier 的可靠性**：谁来 verify verifier？在更硬的数学问题上，verifier 自身的错误率是关键风险
2. **复现性**：Aletheia 产出的数学证明有多大比例经过了严格的独立人工验证？L2 投顶刊不等于已通过 peer review
3. **Cost**：每次 Aletheia 解决一个研究问题消耗多少推理计算？scalability 如何？
4. **数据污染**：训练数据中是否包含相关数学论文？"novel" 结论是否真的是推理出来的而非检索出来的？

### 对 Agentic RL 的启示

这是老板最相关的连接点：

```
Aletheia 的架构                    Agentic RL 的对应
─────────────────────────────────────────────────────
NL verifier                   →   reward model / process reward model
迭代生成-验证-修订             →   RL training loop / MCTS-style search
失败承认                       →   uncertainty estimation / abstention
Google Search grounding        →   RAG / tool use in agent
人机协作 "Advisor" model       →   HITL (human-in-the-loop) RL
```

**关键 insight**：Aletheia 的 NL verifier 本质上是一个 dense process reward signal——它不只说 "对/错"，而是指出具体哪步有问题。这正是 PRM (Process Reward Model) 的研究方向，但 Aletheia 用 LLM-as-verifier 而非训练出来的 reward model，避免了 verifier 本身的训练成本。

## See Also

**Aletheia 系列（前作 → 续集）**
- [[AI/2-Agent/Evaluation/Aletheia-Gemini3-DeepThink-FirstProof|Aletheia FirstProof（arXiv:2602.21201，2026-02-25 续集）]] ⭐ — **同一系统的下一级挑战**：Erdős 4题 → FirstProof 6/10 研究级数学题（8天限时，真实 open problems）；Best-of-2 策略 + 自我过滤设计；P7 解决 Weinberger 书中开放问题——AI 数学研究能力的最新 frontier

**基础模型与推理体系**
- [[AI/3-LLM/Inference/Gemini-3-Deep-Think|Gemini-3-Deep-Think]] — TTC scaling 基础；Aletheia A 基础模型；FirstProof 是其迄今最高难度的真实任务测试
- [[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]] — RL 训练动力学边界；Aletheia 用纯 TTC scaling 而非 RL，说明有验证器的任务 TTC 可能更快达到 frontier

**Agent 架构设计同族**
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（Tencent AI Lab）]] — 反事实验证 credit assignment；Aletheia 的 Generator-Verifier 架构与 CSO"关键步骤验证"设计哲学同源
- [[AI/2-Agent/Fundamentals/GitHub-Agentic-Workflows|GitHub-Agentic-Workflows]] — Agentic workflow 设计模式
- [[AI/3-LLM/RL/算法/MEL-Meta-Experience-Learning|MEL-Meta-Experience-Learning]] — 另一种 agent 自改进路线

---
*Created: 2026-02-19 by Scholar heartbeat | See Also updated: 2026-02-25 by Librarian*
