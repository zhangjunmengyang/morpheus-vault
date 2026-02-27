> [!warning] ⚠️ 路径错误 — 已迁移
> 本文件写入了旧路径 `AI/LLM/RL/Other-Algorithms/`（大重构后已弃用）。
> 正式版已迁移至：[[AI/3-LLM/RL/Other-Algorithms/ExpLang-Multilingual-Thinking-RL|ExpLang 正式版]]
> 本文件不再维护。

# ExpLang：多语言思维提升 LLM 推理的 RL 训练

**论文**：arXiv:2602.21887（ICML 2026 投稿）
**评级**：★★★☆☆
**标签**：#RLVR #多语言推理 #探索 #思维语言 #test-time-scaling
**代码**：https://github.com/RiverGao/ExpLang

---

## 一、核心问题

**现象**：多语言 CoT 的 Pass@k 显著高于英语 CoT——说明多语言思维可以提供更大的推理探索空间。

**传统做法**：LLM 推理研究几乎全部基于英语 CoT（"英语最强"的刻板印象）。

**ExpLang 的问题**：能否把多语言的探索优势系统化地引入 RL 训练？

---

## 二、方法：On-Policy Thinking Language Selection

**核心思路**：把"使用哪种语言思考"作为 RL 的一个**动作维度**，让模型在训练中自主选择思维语言。

**两阶段**：
1. **Supervised Cold-Starting**：用多语言 CoT 数据做 SFT 预热，教模型用不同语言思考
2. **RLVR with On-Policy Selection**：在 RL 阶段，模型可以在不同语言之间切换思维，按照 reward 信号学习哪种语言对哪种问题最有效

**关键结论**：
- 探索阶段：模型自发地使用多种语言思考（多样性高）
- 利用阶段：模型收敛到特定语言组合（找到最优思维语言策略）
- OOD 泛化：对训练中没见过的语言，模型也能 generalize 思维语言选择

---

## 三、为什么多语言有效？

信息论视角：不同语言对同一概念有不同的词汇和语法结构，这提供了：
- **表示多样性**：同一推理路径在不同语言中的 token 序列不同 → 不同的"局部最优解"被激活
- **先验知识差异**：某些语言（如中文、日文）在特定数学/编程概念上有不同的预训练分布 → 不同的推理"先验"

**类比**：这有点像集成学习——多语言 CoT 是多个"弱推理器"，各有优劣，组合后 Pass@k 更高。

---

## 四、与 GRPO/DAPO 的关系

ExpLang 是**正交于** GRPO 的改进：

```
GRPO：group-level advantage estimation（如何利用轨迹）
ExpLang：扩展探索空间（如何生成轨迹）

可以直接组合：GRPO + ExpLang = 多语言并行轨迹的 GRPO
```

**实验**：ExpLang + GRPO 在相同训练预算下稳定超过 English-only GRPO。

---

## 五、我的评价

**有意思，但不是 paradigm shift**。

有价值的 insight：多语言 CoT 确实扩展了探索空间（Pass@k 的提升是真实的）。把语言选择 on-policy 化是一个合理的工程做法。

**局限**：
1. 对中文用户的实际工程意义不大——中文模型已经在用中文思考
2. 不知道什么时候用中文比英文好有多少，论文的 analysis 不够深
3. ICML 投稿，结果还未确认

**作为知识点价值**：了解"多语言提升 Pass@k"这个现象，以及"思维语言 = 另一种 action 维度"的框架思路。不需要深读。
