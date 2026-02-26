---
title: "ConformalThinking: Risk Control for Test-Time Compute"
brief: "JHU + Google DeepMind（arXiv:2602.03814，ICML 2026）用 Conformal Prediction 给 Test-Time Compute 加风险控制：在指定错误率上界（如 10%）下最小化推理 token 开销，自适应提前停止。相比固定 budget 策略，可节省 40-60% token 同时保持覆盖率保证（统计意义上的 PAC 保证）。"
date: 2026-02-21
tags:
  - ai/llm/inference
  - test-time-compute
  - early-stopping
  - conformal-prediction
  - risk-control
  - icml2026
domain: ai/llm/inference
arxiv: "2602.03814"
rating: ★★★★★
status: active
---

# Conformal Thinking: Risk Control for Reasoning on a Compute Budget

**arXiv**: 2602.03814  
**机构**: JHU + Google DeepMind  
**作者**: Xi Wang*, Anushri Suresh*, Alvin Zhang*, Rishi More*, William Jurayj, Benjamin Van Durme, Mehrdad Farajtabar, Daniel Khashabi, Eric Nalisnick  
**提交**: 2026-02-03  
**投稿**: ICML  
**评分**: ★★★★★  
**一句话**: 不再用启发式阈值停止推理——把"何时停止思考"重新定义为统计风险控制问题，用 conformal prediction 框架给出**有分布无关保证**的双阈值停止机制。

## See Also

- [[AI/LLM/Inference/Deep-Thinking-Ratio-DTR-v2-Think-At-N|DTR v2 + Think@N]] — 同为推理 token 效率视角：DTR 发现深层 token 占比决定质量（推理深度），ConformalThinking 用统计框架决定"何时停"（推理宽度）——互补的两个效率维度
- [[AI/LLM/Inference/Accordion-Thinking-Self-Regulated-Step-Summaries|Accordion-Thinking]] — RL 主动压缩 CoT 的另一路径：Accordion fold 掉已处理 token，ConformalThinking 在统计边界处停止——两者都解决"过度思考"，机制不同（内容压缩 vs 边界判定）
- [[AI/LLM/Inference/Progressive-Thought-Encoding-Cache-Efficient-RL|PTE]] — 计算资源受限下的另一侧应对：KV cache 满时学习再 evict（PTE），置信度超阈值时早停（ConformalThinking）——从内存和计算两个角度共同构成 TTC 预算管理框架
- [[AI/LLM/Inference/Test-Time-Compute|Test-Time Compute 综述]] — ConformalThinking 是 TTC 领域的方法论补丁：现有 Best-of-N/PRM/Budget Forcing 都没解决"何时停止"的统计保证问题
- [[AI/LLM/RL/Other-Algorithms/IntroLLM-Introspective-Temperature-Policy-Hierarchical-RL|IntroLLM]] — 相反的置信度应用：IntroLLM 用内省温度提升生成多样性，ConformalThinking 用置信度信号触发双阈值停止——都依赖 token-level confidence 但目标方向相反（发散 vs 收敛）
- [[AI/Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（训练时搜索）]] — 对偶视角：ConformalThinking 在推理时控制 compute（"何时停"），TSR 在训练时用 search 提升 rollout 质量（"如何花"）——两者共同说明 compute 分配在推理/训练两个时间维度都有巨大优化空间

---

## 核心问题

Reasoning LLM 的"过度思考"问题众所周知：`<think>...</think>` 段落常常比需要的长得多，大量 token 被浪费在已经得到正确答案之后的继续推理上。

现有解法：监控 confidence/entropy 信号，超过阈值就停。

**已有解法的根本问题**：
1. 阈值没有解释性——"threshold=0.73" 意味着什么？具体错误率是多少？
2. 阈值选择是任意的——不同任务、不同模型需要不同阈值
3. 只有"停-on-confident"（上阈值），没有"停-on-hopeless"（下阈值）——困难问题上浪费大量 token 但从不停

**论文的重新定义**：

> 把"设置 token budget"的问题转化为"选择可接受的 error rate"。

用户不需要知道 threshold=0.73 是什么意思，但知道"我能接受 5% 的错误率"。给定用户指定的 ε（risk tolerance）和一个 validation set，用 distribution-free risk control 自动校准出对应的停止阈值，且有统计保证。

---

## 方法：双阈值停止机制

### 符号定义

- `y*`：真实答案  
- `ft(x)`：模型在 t 步推理后的预测
- `st`：步骤 t 的置信度信号（scalar）
- `s̃t = g(s1:t)`：平滑后的信号（如指数移动平均）
- `T`：最大推理预算（token 数）

### 四种 Loss 函数

**正确性损失**（控制 error）：

**False Positive Loss（上阈值）**：
```
ℓ_FP^upper(y*, ft(x), s̃t(x); λ+) = I[s̃t ≥ λ+] · I[ft(x) ≠ y*]
```
含义：模型认为答案正确（置信度超上阈值）但实际答案错了。

**False Negative Loss（下阈值）**：
```
ℓ_FN^lower(y*, ft:T(x), s̃t(x); λ-) = I[s̃t ≤ λ-] / (T-t+1) · Σ_{k≥t} I[fk(x) = y*]
```
含义：模型认为问题解不了（置信度低于下阈值）但实际上后面某步可以解对。注意：分母是 T-t+1，这样只有"大量未来步骤都能解对"才算真正的 false negative，随机猜对一步不计。

**效率损失**（控制 compute 浪费）：

**上阈值效率损失**：
```
ℓ_eff^upper(t) = (1/T) · max(0, t - t')
```
其中 `t' = min{t: ft(x) = y*}`，即第一次得到正确答案的时刻。含义：在已经得到正确答案后继续推理浪费的 token 占比。

**下阈值效率损失**：
```
ℓ_eff^lower(t) = (1/T) · Σ_{k≤t} I[y* ≠ fk(x)]
```
含义：在放弃之前已经浪费了多少 token 在错误答案上。

### 双阈值停止机制

**上阈值**（停-on-confident）：
```
τ+ = min{t : s̃t(x) ≥ λ+}
```
置信度超过 λ+ 即停止，认为已经找到正确答案。

**下阈值（新贡献）**（停-on-pessimism）：
```
τ- = min{t : s̃t(x) < λ-(t; c)}
```
其中下阈值是**参数化函数**（而非固定值）：
```
λ-(t; c) = σ(c · (ωt - B/2))
```
- `ωt`：已用 token 数
- `B`：总预算
- `c`：控制曲线陡度的参数
- σ：sigmoid 函数

**直觉**：随着 token 消耗增加，我们对模型的进展要求越来越高（曲线上升）。在推理早期，低置信度可以接受（刚开始，正常）；在推理中后期，置信度还是很低说明问题可能无解，触发下阈值停止。

### 阈值校准：Distribution-Free Risk Control

给定用户指定的 risk tolerance ε 和一个 calibration set：

1. 对 calibration set 计算每个实例在各 threshold 下的 loss
2. 用 Risk Control with P-values Structure（RCPS）框架找到**最小的 λ**（最激进的早停）使得期望损失 ≤ ε
3. 这个 λ 在 test set 上有有限样本统计保证：`E[ℓ(y*, ft(x); λ)] ≤ ε`

**关键**：这个保证是 **distribution-free** 的——不需要假设 calibration set 和 test set 同分布之外的任何东西。

当有多个 budget 控制标准时，引入 efficiency loss，从满足 risk bound 的所有阈值中选计算最高效的一个。

---

## 实验结果

**模型**：Qwen3-8B, DeepSeek-R1（推理模型）  
**任务**：AIME, MATH, LiveCodeBench 等多个推理 benchmark

**结论**：
- 上阈值 + 下阈值组合**一致**优于仅有上阈值
- 在给定 risk ε 的约束下，双阈值方法用**更少 token** 达到相同 accuracy
- 对困难问题（AIME），下阈值效果最显著：提前放弃不可解问题，节省大量 token

**Figure 2 的关键直觉**：
- 不可解问题：置信度信号在整个推理过程中**震荡**，永远不达上阈值 → 下阈值在预算中点前提前截停
- 可解问题：置信度**稳定上升**，最终触发上阈值

这两种行为的形状本身就是信号——"震荡"意味着模型在瞎猜，"稳定上升"意味着真正在收敛。

---

## 我的分析

### 真正 novel 的是什么？

**框架的认识论转换**（Epistemological Reframing）。

不是新的 confidence measure，不是新的神经网络模块——而是把一个"工程问题"（调阈值）转化成了一个"统计问题"（控制风险）。

这个转换的价值在于：
1. **可解释性**：用户能理解"5% error rate"，不能理解"threshold=0.73"
2. **可迁移性**：同样的框架适用于任意 confidence signal、任意任务、任意模型
3. **统计保证**：distribution-free，不是启发式的宣称

### 下阈值设计的精妙之处

大多数人想到 TTC 的 early stopping，只想到"stop when confident"。这篇论文问了一个新问题：**"什么时候我们应该放弃？"**

这两个问题是不对称的：
- 上阈值控制 false positive（错认为正确）
- 下阈值控制 false negative（错认为无法解决）

参数化的下阈值（sigmoid 曲线随 token 上升）是真正 novel 的设计——它实现了"越到后期越容忍不了停滞"的直觉，把推理过程的时序特性编码进了 threshold 的形状里。

### False Negative Loss 的设计选择值得深思

```
ℓ_FN = I[s̃t ≤ λ-] / (T-t+1) · Σ_{k≥t} I[fk(x) = y*]
```

为什么用**平均**而不是**存在性**（只要未来某步能解就算 FN）？

论文解释：如果只有 1000 个未来步骤中的 1 个是正确的（可能是随机猜对），用存在性定义会把这算作严重的 false negative——但实际上我们对这个"解"没有多少信心。平均定义更 conservative，只有当未来步骤**普遍**能解对时，才算是真正的 false negative。

这是统计上的 conservative 设计，代价是可能漏掉一些真正可解但只能偶尔解对的困难问题。这个 trade-off 是合理的。

### 局限

1. **需要 calibration set**：distribution-free 保证依赖于一个 held-out validation set，在 domain shift 严重时保证可能失效
2. **参数 c 的选择**：下阈值的 sigmoid 陡度 c 仍是需要调的参数，只是比原始 threshold 更有结构
3. **confidence signal 的选择仍是黑盒**：框架不解决"用什么作为 confidence signal"的问题，只解决"给定 signal 如何设阈值"

### 与现有工作的关系

| 工作 | 机制 | 保证 |
|------|------|------|
| Entropy-based（wang et al.） | upper threshold only | 无 |
| Dynamic Early Exit（yang et al.） | upper threshold only | 无 |
| Thought Calibration | hypothesis testing，similar FP loss | 有（但无 lower threshold） |
| **Conformal Thinking** | dual threshold + efficiency loss | distribution-free，dual risk |

### 对面试的价值

这篇论文提供了一个 elegant 的**框架级思维**：遇到"如何设超参"的工程问题，先问自己——"这个超参实际上在控制什么 risk？能不能把它变成一个可解释的 risk tolerance？"

这种思维方式在推理优化、Agent 决策（何时放弃任务）、自适应计算（何时停止 beam search）等场景都可以迁移。

---

## 关键公式

**双阈值停止规则**：
```
τ = min(τ+, τ-)    # 先触发哪个就停哪个
τ+ = min{t : s̃t ≥ λ+}                              # 停-on-confident
τ- = min{t : s̃t < σ(c · (ωt - B/2))}              # 停-on-hopeless
```

**Risk Control 目标**：
```
E[ℓ(y*, ft(x); λ)] ≤ ε    # distribution-free 保证
```

**False Negative Loss（设计精要）**：
```
ℓ_FN = I[s̃t ≤ λ-] / (T-t+1) · Σ_{k≥t} I[fk(x) = y*]
         ← 放弃标志    ← 归一化     ← 未来步骤的平均正确率
```

---

## Tags
#TTC #TestTimeCompute #EarlyStopping #ConformalPrediction #RiskControl #AdaptiveReasoning #ICML2026 #推理效率 #过度思考 #分布无关
