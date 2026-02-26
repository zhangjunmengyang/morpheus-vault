> [!warning] ⚠️ 路径错误 — 旧结构副本
> 本文件写入了旧路径 `AI/LLM/RL/Other-Algorithms/`（大重构后已弃用）。
> 正式版请见：[[AI/2-Agent/Agentic-RL/MIG-Step-Marginal-Information-Gain-Credit-Assignment|MIG 正式版]]
> 本文件保留作参考，内容无独特洞察（正式版已更完整）。

# MIG：Step-wise Marginal Information Gain for Multi-Step LLM Reasoning

**论文**：arXiv:2602.01034（2026-02-01）
**标题**：Discovering Process–Outcome Credit in Multi-Step LLM Reasoning
**作者**：Xiangwei Wang, Wei Wang, Ken Chen, Nanduni Nimalsiri, Saman Halgamuge（墨尔本大学）
**评级**：★★★★☆
**标签**：#credit-assignment #dense-reward #PRM #reasoning-RL #MIG #信息论
**关键词**：Step-wise reward / Monotonic Historical Watermark / Decoupled Masking / Dual-Gated SFT

---

## 一、问题定位

**标准 RLVR 的两个核心缺陷**：

1. **Reward Sparsity**：binary {0,1} 的 outcome reward 对长链推理无中间信号——一个 10 步推理链，前 9 步完全正确但最后一步算错，得到和全错相同的 0 奖励
2. **Credit Assignment**：不知道哪一步贡献了正确答案，哪一步在做无效冗余

**现有解法的局限**：
- **PRM**：需要人工标注 step-level 正确性，昂贵且难以扩展
- **Handcrafted rules**：容易 reward hacking，且无法捕捉语义层面的"是否做出了推理突破"
- **VeriFree / LaTRO**：用模型对答案的条件概率作为 trace-level 信号，但整体轨迹粒度太粗——无法区分"哪一步是真正的逻辑突破"

**MIG 的核心问题**：如何在**无需外部标注**的情况下，从模型内部概率中计算出每个推理步骤的**内在语义价值**？

---

## 二、核心机制：Step-wise MIG + Monotonic Historical Watermark

### 2.1 MIG 的信息论直觉

**Marginal Information Gain（MIG）** = 这一步在多大程度上"推进了"对答案的预测？

形式化：设已有上下文为 `h_{k-1}`（历史推理步骤），第 k 步为 `s_k`，答案为 `a`：

```
MIG_k = H(a | h_{k-1}) - H(a | h_{k-1}, s_k)
      = 信息熵下降量（增加 s_k 后，对答案的不确定性减少了多少）
```

**实践近似**：用语言模型的条件概率来估计：
- `h_{k-1}` 状态下模型对答案 `a` 的预测 log-prob → 代表当前"推理进度"
- 加入 `s_k` 后对 `a` 的预测 log-prob → 代表这一步之后的"推理进度"
- **差值** = MIG_k（这一步带来的推理进展量）

### 2.2 Monotonic Historical Watermark

**关键创新**：不是用绝对的 MIG 值作为奖励，而是对比**历史最高水位线**：

```
watermark_{k-1} = max(MIG_1, MIG_2, ..., MIG_{k-1})  # 历史最高推理进展
reward_k = ReLU(MIG_k - watermark_{k-1})              # 只奖励超越历史最高的突破
```

**为什么需要 Watermark**？
- 没有 Watermark：每一步只要有正向 MIG 就得奖励，冗余的"重复已有推理"也得奖励 → 模型学会生成冗长废话
- 有 Watermark：只有真正"超越之前所有步骤"的认知突破才得奖励 → 鼓励真正的推理进步
- Monotonic 设计：watermark 只增不减 → 惩罚循环推理（模型绕回到之前说过的话）

**Rectified**：`reward_k = max(0, MIG_k - watermark_{k-1})` 确保负奖励被截断为 0，防止错误信号。

### 2.3 三阶段框架

```
Phase 1: Structured Rollout
  prompt → 生成 reasoning chain
  用结构化 tag（如 <step>...</step>）解析成离散的 step 序列
  
Phase 2: Step-Aware Probe（MIG 计算）
  对每个 step s_k 计算 P(answer | h_{k-1}, s_k)
  算 MIG_k = log P(a|h_{<k},s_k) - log P(a|h_{<k})
  对比 watermark → 计算 rectified reward g_k = ReLU(MIG_k - h_{k-1})
  更新 watermark: h_k = max(h_{k-1}, MIG_k)

Phase 3: Decoupled Hybrid Optimization
  Process reward (g_k) → 只作用于 CoT tokens
  Outcome reward (binary {0,1}) → 作用于全部 completion
  Dual-Gated SFT → 只在 outcome 正确 且 MIG 有突破时做 SFT
```

---

## 三、Decoupled Masking Strategy

**问题**：process reward（鼓励推理多样性）和 outcome reward（要求答案正确）是**对立的优化目标**：
- Process：希望探索更广的推理路径（发散）
- Outcome：希望约束在正确答案的流形上（收敛）

**解法**：用 mask 解耦两个目标的作用域：

```
CoT tokens:    只受 MIG-based dense reward 影响
Answer tokens: 只受 binary outcome reward 影响

总损失：
L_total = L_process(CoT | MIG reward) + λ * L_outcome(answer | {0,1})
```

**好处**：
- CoT 部分可以自由探索不同推理路径（MIG 鼓励多样性）
- Answer 部分严格对齐正确性（不受 exploration 噪声污染）

---

## 四、Dual-Gated SFT

**目标**：在 RL 训练中引入高质量 SFT 信号来稳定训练

**双门控条件**：SFT 只在同时满足两个条件时触发：

```
Gate 1（Outcome Gate）: outcome 正确（binary reward = 1）
Gate 2（MIG Gate）:    推理链有实质性的语义突破（max(g_k) > threshold）

→ 只有"正确答案 + 高质量推理过程"的轨迹才成为 SFT 正样本
```

**意义**：
- 纯 outcome-filtered SFT（如 STaR）会把"猜对的答案"也当正样本，污染训练数据
- 双门控排除了"瞎猫碰死鼠"型正确：结果对但推理过程是乱的
- 保留了"真正推理正确"的高密度示例作为 SFT 数据

---

## 五、与已有方法的对比

| 方法 | 信号类型 | 粒度 | 是否需要标注 | Reward Hacking 风险 |
|------|---------|------|------------|-------------------|
| GRPO | Binary outcome | Trajectory | 否 | 低（verifiable） |
| PRM | Human-labeled step reward | Step | **是** | 中 |
| VeriFree/LaTRO | 条件概率（P(a\|trace)） | Trace | 否 | 低 |
| **MIG** | 信息增益 vs watermark | **Step** | **否** | **很低（watermark防hack）** |

**MIG vs VeriFree**：
- VeriFree 看的是整条 trace 对答案的预测概率 → trace-level，粒度粗
- MIG 看的是每一步相对历史水位的边际增益 → step-level，粒度细
- MIG 额外解决了"冗余步骤"问题（冗余步 MIG 低于 watermark → 零奖励）

**MIG vs GiGPO**（Agent RL 的 credit assignment）：
- GiGPO：通过重访相同状态的不同 action 来估计 step value（需要状态 overlap）
- MIG：通过信息论度量每步对答案预测的贡献（不需要状态 overlap，只需要 verifiable answer）
- **适用场景不同**：MIG 适合数学/推理（有 verifiable answer）；GiGPO 适合 agent 任务（长 horizon，无明确 answer oracle）

---

## 六、实验结果

**Benchmarks**：
- 文本推理：MATH（数学），AIME，AMC
- 多模态：Super-CLEVR（视觉推理），MathVista

**主要结论**：
- 在所有 benchmark 上超过 GRPO baseline
- **Sample Efficiency 显著提升**：相同训练步数下达到更高准确率（MIG 的 dense reward 比 sparse outcome 更有效传递梯度）
- **OOD 泛化**：在 zero-shot transfer 到未见过的推理任务上表现更好
- **多模态泛化**：框架可直接扩展到视觉推理（MIG 的信息论视角是模态无关的）

**消融实验**（ablation）关键发现：
- 去掉 Monotonic Watermark → 模型学会生成冗余废话（MIG 总是正的，没有watermark过滤）
- 去掉 Decoupled Masking → outcome reward 污染 CoT 探索，多样性下降
- 去掉 Dual-Gated SFT → 训练不稳定，最终准确率下降

---

## 七、我的评价

### 亮点

**信息论视角的 credit assignment** 是真正 novel 的贡献。相比 GiGPO 依赖"轨迹重访相同状态"的结构假设，MIG 只依赖"有 verifiable answer"这一弱假设，适用范围更广。

**Monotonic Watermark** 这个设计非常优雅：
- 不是绝对值，而是边际增量（marginal gain）
- Monotonic 确保只奖励"突破历史最高"，自然淘汰冗余和循环
- ReLU 截断防止负奖励导致训练不稳定

### 质疑点

1. **计算开销**：每个 step 都需要额外 forward（计算 P(a|h_{<k}, s_k)），相比 GRPO 计算量显著增加。论文没有详细报告额外 FLOPs。

2. **Answer oracle 依赖**：MIG 需要 `P(a|context)` 作为探针，意味着需要知道正确答案。在 open-ended 任务（SOTOPIA、对话等）中无法直接用。

3. **Step 分割的鲁棒性**：用结构化 tag 解析 step 需要模型可控地输出格式化推理链。如果模型不遵循格式，step 分割失败。

4. **Watermark 的超参**：watermark 的 threshold 怎么设？论文提到 `threshold > 0` 才触发 SFT，但没有详细讨论这个超参的敏感性。

### 与其他方法的关系

MIG 填补了 Verifier-Free + Step-level 这个象限的空白：

```
           | Trace-level | Step-level |
-----------|-------------|------------|
需要标注   | —           | PRM ✓      |
无需标注   | VeriFree ✓  | MIG ✓      |
```

它不是在替代 GRPO，而是在 GRPO 的 outcome reward 基础上增加了 dense process guidance。

### 实用评价

对老板面试的直接价值：**理解 credit assignment 的信息论视角**。面试中 RLHF/GRPO 的 credit assignment 问题是高频考点，能说出"MIG = 每步对答案条件概率的边际增益，用 monotonic watermark 过滤冗余"是明显的加分项。

---

## 八、与 Vault 已有内容的关联

```
Credit Assignment 谱系（更新）：

Outcome-level：
  GRPO → group-relative advantage（trajectory-level average）
  
Step-level（显式）：
  PRM → MC rollout 估计 step value（GiGPO 是 PRM 的 rollout-free 版本）
  GiGPO → anchor state grouping（依赖 state revisit）
  AgentPRM → MC rollout → 适合 agent 任务
  
Step-level（隐式/无标注）：
  iStar → trajectory DPO = implicit step reward（适合 preference 数据）
  MIG → 信息论边际增益 + monotonic watermark（适合 verifiable answer）
  
两条路线正交，可组合：
  MIG（dense process）+ GRPO（sparse outcome）= 本文方案
  iStar + GiGPO = 无 overlap 约束 + anchor 精化
```

- `AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题.md`（需要更新加入 MIG）
- `AI/LLM/RL/Other-Algorithms/iStar*.md`
- `AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026.md`

---

## 九、arXiv 元数据

- arXiv ID：2602.01034
- 提交：2026-02-01
- 机构：墨尔本大学（University of Melbourne）
- 作者：Xiangwei Wang 等
