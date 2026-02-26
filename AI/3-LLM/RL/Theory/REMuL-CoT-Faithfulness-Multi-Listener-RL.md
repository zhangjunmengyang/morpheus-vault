---
brief: REMuL（arXiv:2602.16154，ICML 2026）——通过多 Listener RL 提升 CoT 忠实度；核心问题：模型推理过程与最终答案不一致（deceptive reasoning），多 listener 投票机制强制 CoT 与 outcome 对齐。
title: "REMuL: CoT Faithfulness via Multi-Listener RL"
date: 2026-02-18
tags:
  - faithfulness
  - CoT
  - interpretability
  - RL
  - GRPO
  - multi-agent
  - reasoning
  - ICML2026
domain: AI/LLM/RL/Theory
arxiv: "2602.16154"
rating: 5
status: permanent
see-also:
  - "[[GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]]"
  - "[[RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]]"
  - "[[对齐技术总结|对齐技术总结]]"
---

# REMuL: Balancing Faithfulness and Performance in Reasoning via Multi-Listener Soft Execution

> arXiv: 2602.16154 | ICML 投稿
> 作者: Nithin Sivakumaran, Shoubin Yu, Hyunji Lee, Yue Zhang, Ali Payani, Mohit Bansal, Elias Stengel-Eskin
> 机构: UNC Chapel Hill + Cisco AI Research
> 代码: https://github.com/nsivaku/remul

## 评分: ★★★★★

## 一句话

CoT 推理链并不一定真实反映模型的内部计算。REMuL 定义了一个可操作的 faithfulness 概念：**推理链可以被其他模型"继续执行"并到达相同结论**。然后用 GRPO 训练 speaker 生成对 listener 们更 faithful 的推理，同时用 masked SFT 维持正确率。

---

## 为什么这个问题重要

**CoT faithfulness 问题**是 LLM 可解释性领域的核心困境：

模型输出了一段推理链，但这段推理链真的是它"想"的过程吗？还是事后合理化（post-hoc rationalization）？

证据显示往往是后者：
- 模型经常在推理完毕后才真正确定答案，推理链只是结果的包装
- 用 RL 只优化最终答案正确率，进一步恶化 faithfulness（推理链和答案更加解耦）
- 在需要解释的高风险场景（医疗、法律、科研）里，不忠实的推理链有实质危害

**已知 trade-off**：提升 faithfulness 往往损害准确率——表达"如何想"和"答对题"是两个不同目标。

---

## 核心 idea：Faithfulness as Executability

**定义**：一条推理链是 faithful 的，当且仅当另一个能力相当的模型（listener）在只看到这条推理链的前缀时，能推断出和 speaker 相同的答案。

形式化：
```
faithfulness_reward(trace t, answer a) = I[L(x, t_{1:m}) = a]
```

其中：
- `t_{1:m}`：推理链截取到 m% 的前缀（25%、50%、75% 三个截断点）
- `L(·)`：listener 模型基于前缀继续推理并给出答案
- `I[...]`：listener 答案是否和 speaker 一致

**多 listener 的作用**：
- 使用 3 个不同模型（Ministral-3-14B-Reasoning, Phi-4-reasoning-plus, Qwen3-14B）
- 每个 listener 在 3 个截断点各做一次执行，总计 9 个信号
- Aggregated reward：`r_match = Σ I[a_j^i = a]`（各 listener 在各截断点的匹配数）

直觉：如果多个独立模型从推理链的前缀都能推断出相同答案，说明这条推理链逻辑清晰、自成一体，而不是靠后验合理化。

---

## 方法：REMuL

### 两阶段训练

**阶段 1：Faithfulness RL（GRPO + multi-listener reward）**
- Full-rank 更新
- Speaker 用 r_match 作为 reward，GRPO 训练
- 目标：让推理链对 listener 更可执行

**阶段 2：Correctness SFT（masked 只监督 answer token）**
- LoRA adapter 训练（与阶段 1 参数不共享训练信号）
- Loss **只作用于最终 answer token**，推理链 token 都 mask 掉
- 直觉：纠正答案的决策边界，但不强制推理路径

```python
L_ans(θ) = -Σ_{i ∈ A} log p_θ(y_i | x, y_{<i})
# A 只包含 answer token 的位置，推理链 token 对应 loss = 0
```

**两阶段串联**：RL 先学忠实推理 → SFT 再补正确率。合并 LoRA adapter 后只留一个 speaker 模型，推理时**不需要 listener**。

### 为什么不联合优化？

实验发现：把 faithfulness reward + correctness reward 放进一个 GRPO 目标（balanced reward `r_bal = r_match + λ I[a=g]`）会导致两个信号竞争，效果比分开训练差。

根本原因：faithfulness 奖励让模型明确表达推理过程，correctness 奖励让模型专注答对最后结果——两个目标在 token 空间里的梯度方向相互干扰。

---

## 实验结果

### 训练：BBH 5 个任务，1250 QA pairs（Qwen3-14B + Phi4）

### 评估：4 个推理 benchmark（OOD）

**Faithfulness 指标**：
1. **Hint Usage**：注入提示后，模型改了答案，是否在推理链里承认了提示的影响？
2. **Early Answering AOC**：在不同截断点强迫模型答题，AOC 越高说明推理链越关键
3. **Adding Mistakes AOC**：在推理链中注入错误，AOC 越高说明模型真的在跟着推理链走

### 主要结论

| 方法 | Faithfulness（Hint Usage↑） | Accuracy |
|------|--------------------------|---------|
| Original | baseline | baseline |
| Faithfulness Only | **+4.1%（Qwen3）/ +3.7%（Phi4）** | **-3.2% / -2.2%** |
| Correctness Only | -1.8% | +2.9% |
| MAT-Steer | +1.3% | ~ |
| **REMuL** | **+2.5% / +2.5%** | **+1.8%（Qwen3）/ +1.5%（Phi4）** |

**AOC（Qwen3-14B）**：
- Faithfulness Only: Truncated CoT AOC +0.054
- REMuL: +0.033（第二好，但保住了 accuracy）
- Adding Mistakes AOC：同样 REMuL 第二好，平均 +0.030

关键：**REMuL 是唯一同时提升 faithfulness 和 accuracy 的方法**。

---

## 我的分析

### 这篇论文哪里真正 novel？

**1. Faithfulness 的操作化定义是最大贡献**

之前的 faithfulness 研究要么用人工评分，要么用间接指标（hint usage、AOC）。REMuL 把 faithfulness 转化为一个可直接优化的 reward signal："其他模型能不能跟着你的推理链到达相同结论"。

这个定义的精妙之处：
- 不需要 ground-truth "正确推理链"（没有这种东西）
- 完全可计算（listener 推理是自动的）
- 语义上真实——如果你的推理对别人也说得通，大概率是真实的

**2. 发现了 faithfulness-correctness 冲突的来源**

实验明确证明：放进同一个 GRPO 目标 → 竞争 → 两者都变差。解法（先 RL 后 masked SFT）不是创意最强的，但是有实证依据的 principled 选择。

**3. 多 listener 的必要性**

Ablation 证明：单 listener 或同类型 listener 效果弱于多样化 listener pool。这和 ensemble 的直觉一致——不同模型的推理风格不同，能检验推理链在多个"认知角度"下的可执行性。

### 和 RL post-training 主流方向的关系

主流 RLVR（GRPO/ProGRPO 等）优化目标：**让答案正确**。

REMuL 优化目标：**让推理过程对其他模型可跟随**。

这是两个正交维度。论文证明只关注 correctness 会损害 faithfulness（RL 让模型找 shortcut，推理链越来越是包装）。

**对 alignment 的启示**：如果我们用 LLM 做高风险决策，我们不仅需要答案对，还需要知道"它为什么这么说"。REMuL 是朝这个方向的一步。

### 局限性和问题

1. **Listener pool 的选择**：用了 3 个 14B 模型——这已经是相当大的计算开销。生产环境里如何缩减？

2. **Faithfulness ≠ Truth**：推理链对 listener 可执行，不代表推理链在"物理上"反映了模型的内部计算（权重级别的因果关系）。这仍然是 functional faithfulness，不是 mechanistic faithfulness。

3. **领域迁移**：只训了 BBH 的 5 个任务，在 4 个 OOD benchmark 测，效果不错。但更复杂的任务（医疗/法律）待验证。

4. **LoRA 和 full-rank 混合**：两阶段分别用不同训练方式，参数合并是否总是稳定？

### 更大的框架：什么是好的推理？

以前的问题：如何让 LLM 推理**正确**？  
REMuL 追问：如何让 LLM 推理**真实**？

这两个问题的区别，恰恰是从「能力对齐」到「可信任 AI」的关键跨越。

---

## 连接

- 前驱：Lanham et al. 2023（AOC 指标），Chen et al. 2025（hint attribution），GRPO（训练框架）
- 竞争：MAT-Steer（training-free steering baseline）
- 应用：医疗/法律推理系统，科研助手，任何需要解释性的场景

## see-also

- [[GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — REMuL 在 faithfulness 维度扩展了 GRPO 的优化目标（correctness → correctness + faithfulness）
- [[RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]] — 同方向：两篇都在解决"RLVR 只优化结果正确性"的局限；RLRR 扩展到 non-verifiable，REMuL 扩展到 faithfulness
- [[对齐技术总结|对齐技术总结]] — REMuL 是 AI 可解释性与对齐的交叉点，faithful reasoning 是高风险场景对齐的前提
- [[ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]] — 同为 GRPO 改进；ProGRPO 改 advantage 结构，REMuL 改优化目标本身
- [[RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性 2026 统一分析]] — faithfulness-correctness 冲突的梯度干扰，与训练稳定性分析的 token-level 视角互补
- [[FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer (CWRPO)]] — 设计哲学同构：conditional release reward（structure quality 门控 answer reward）与 REMuL 的两阶段训练（faithfulness RL → masked SFT）都在用"过程质量"约束"结果优化"，殊途同归
- [[RICOL-Retrospective-In-Context-Online-Learning|RICOL]] — 方法论互补：REMuL 用多监听者 faithfulness 作为 RL reward（优化过程质量），RICOL 用 ICL 前后 log-prob 差作为 advantage（免 critic credit assignment）——两者都在挖掘 LLM 内部能力作为训练信号

## Tags
`#faithfulness` `#CoT` `#interpretability` `#RL` `#GRPO` `#multi-agent` `#reasoning` `#ICML` `#2026-02`
