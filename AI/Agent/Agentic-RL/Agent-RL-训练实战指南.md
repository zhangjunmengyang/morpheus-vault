---
title: "Agent RL 训练实战指南 — 现象·坑·解法"
type: synthesis
domain: ai/agent/agentic-rl
tags:
  - interview-prep
  - agentic-rl
  - grpo
  - training-stability
  - reward-design
  - credit-assignment
  - hands-on
created: 2026-02-19
status: v1
---

# Agent RL 训练实战指南
## ——现象、坑、解法的系统整理

> 这不是原理讲解，是**你在训练中会撞上什么、为什么会撞、怎么解决**。
> 每个坑都对应真实论文的实验发现和可操作 fix。

---

## 全局思维框架

训练一个 Agent RL 系统，问题空间可以拆成三层：

```
第一层：算法层 — GRPO/PPO 本身有哪些失效模式？
第二层：Agent 层 — 多步交互引入了哪些单步 RL 没有的问题？
第三层：系统层 — 工程实现上有哪些隐蔽的 off-policy 偏差？
```

这三层问题**同时存在**，互相叠加。面试里能把这三层说清楚，已经甩开 90% 的候选人。

---

# 第一章：GRPO 的失效模式与修法

> 背景：GRPO 是当前 RLVR 的标准算法（DeepSeek-R1 之后）。但它有系统性缺陷，2026 年初集中爆发了一批修法论文。

## 1.1 Entropy Collapse — 模型越训越"哑"

### 现象
- 训练前几百步 reward 上升，之后陷入平台
- 输出多样性骤降，模型开始复读模式或固定句式
- loss 数值稳定但 benchmark 不再涨
- 用 entropy 指标监控：entropy 单调下降，最终接近 0

### 为什么会这样

GRPO 的 group 内正确答案会被全量强化，包括那些**碰巧出现在正确序列里但本身没有贡献的 token**（STAPO 称之为 spurious tokens）。

STAPO（2602.15620）做了精确定位：这类 token 同时满足三个条件：
- **低概率**（π_t < τ_p）：模型对这个 token 没把握
- **低熵**（H_t < τ_h）：但此位置的分布高度集中——paradox，集中但选了个异类
- **正 advantage**：出现在正确序列里，被全额 reinforce

根据 Theorem 3.1，梯度 norm ∝ `(1 - 2π_t + e^{-H})`，**低概率 + 低熵 = 梯度最大**。这 0.01% 的 spurious tokens 携带了不成比例的梯度，不断拉偏模型，把 entropy 压垮。

### 解法

**STAPO — S2T Mask（即插即用，零额外成本）**
```python
# 三个条件同时满足则 mask 梯度
spurious = (advantage > 0) & (prob < tau_p) & (entropy < tau_h)
loss = loss * (~spurious).float()
```
效果：+7.13% vs GRPO baseline，entropy 全程稳定。

**DAPO 的 clip-higher（来自 ByteDance）**
把正样本的 clip 上界从 1+ε 提高到 1+ε_high（非对称 clip），防止高 advantage token 的 IS ratio 被截断太早，保留梯度多样性。

**面试怎么说**：
> "GRPO 的 entropy collapse 根源不是宏观的熵正则不够，而是 0.01% 的 spurious token 在 token 级别携带了异常大的梯度。STAPO 从信息论角度——联合概率、位置熵、优势符号——精确定位这类 token 并 mask 其梯度，是目前最干净的解法。"

---

## 1.2 Zero-Gradient 问题 — 训练步骤大量浪费

### 现象
- GPU 在跑，reward 不动
- 仔细看 gradient norm：很多 step 的梯度接近零
- 换句话说：训练步骤在消耗算力，但模型没有任何学习

### 为什么会这样

GRPO 的梯度 norm 正比于 `√(p_q(1-p_q))`（Goldilocks, 2602.14868 数学证明）：
- **p_q = 1（全对）**：模型已经会了，梯度为零，纯浪费
- **p_q = 0（全错）**：没有对的样本做 advantage 对比，梯度也为零
- **只有 p_q ≈ 0.5（能做对一半）的题梯度最大**

随机采样数据集，难度分布是混乱的。大量步骤落在"太简单"或"太难"的区间，梯度消失。

### 解法

**Goldilocks RL（Apple + EPFL, 2602.14868）— Teacher-Student 动态课程**

用一个小 Teacher 模型在线预测每道题对当前 Student 的 utility：
```
y_q = √(p̂_q · (1 - p̂_q))   # 实测方差作为监督信号
```
Teacher 用 MSE 拟合这个目标，ε-greedy 选题：
- 以概率 1-ε 选 utility 最高（p_q ≈ 0.5）的题
- 以概率 ε 随机选（保证覆盖）

**PACED-RL（2602.12642）— GFlowNet 路线的同类发现**
用 GFlowNet 的配分函数 Z_φ(x) 编码在线准确率，零额外开销实现难度筛选。
结论：AIME pass@1 vs GRPO +29.1%。和 Goldilocks 殊途同归，相互印证。

**DAPO 的动态采样（ByteDance）**
简单版本：过滤掉全对（correct_count = G）和全错（correct_count = 0）的 group，只用有梯度的样本训练。
```python
# DAPO 的最简 fix
valid = (correct_count > 0) & (correct_count < G)
loss = loss[valid].mean()
```

**面试怎么说**：
> "GRPO 对训练样本一视同仁，但数学上梯度信号只在 p_q ≈ 0.5 的样本上才有最大值。这是 Bernoulli reward 的固有性质，不是玄学。Goldilocks 把它变成了一个 online learning 问题：Teacher 预测每道题的学习价值，只用边界上的样本训练。"

---

## 1.3 Root Saturation — 越训越不会探索

### 现象
- 把 group size N 从 8 扩到 32 没什么用
- 再扩到 64 也没用（diminishing returns 很快）
- 困难问题上 pass@1 停滞，但 pass@k 也不涨
- 模型反复走相同的推理路径

### 为什么会这样

GRPO 每次从 root（问题起点）采样完整 trajectory。随着训练：
- Policy 对高概率路径越来越自信
- 大量 rollout budget 消耗在"已掌握的路径"上
- **深层的 error-prone states 因累积概率衰减根本到达不了**

DEEP-GRPO（2602.14169）实验验证：N=8→64，性能从 64.1% 到 66.2%，N=32→64 几乎无增益——问题不是样本量，是探索区域被锁死了。

**TreeRL/AttnRL 的失败原因**：在中间状态分支是个正确方向，但 budget 分散在太多分支点，每个点只有极少样本，local advantage 估计不稳定，还混入了不同分布的轨迹。

### 解法

**DEEP-GRPO — Pivot-Driven Resampling**

核心 insight：**失败轨迹里有很多有效的 reasoning prefix**，从这些 prefix 的末尾重新采样，可以在"深而可恢复"的位置密集探索。

Pivot 选择公式：
```
Q(t) ∝ P_φ(success | s_{<t}) × (t/T)^γ
       ←可恢复性→               ←深度偏置→
```
用轻量 Logistic Regression 在线估计 recoverability，无需 value model。

找到 pivot 后，在该位置集中生成 K=8 branches（local dense resampling），再用 dual-stream optimization（global + local 损失叠加）更新。

效果：AIME24 上 1.5B 模型从 20% → 43.3%（+23.3%）。

**面试怎么说**：
> "Root saturation 是 GRPO 探索机制的根本缺陷。扩大 N 只是让模型更努力地走相同的路，而不是走新的路。DEEP-GRPO 的关键洞察是：失败轨迹的 prefix 往往是合理的，只是后半段出了问题。在深层 pivot 点密集 resample，才能真正打开探索空间。"

---

## 1.4 Off-Policy 偏差 — 隐蔽的训练 Bug

### 现象
- 换了 rollout 框架（vLLM/SGLang）之后性能莫名下降
- FP8 推理加速后发现 reward 没有 BF16 高
- 异步训练速度提升了，但最终 benchmark 比同步差
- 你以为在跑 on-policy，实际上不是

### 为什么会这样

**Off-policy 有四个隐蔽来源**（VESPO, 2602.10693 明确列举）：

1. **精度不一致**：rollout 用 FP8，但 training forward 用 BF16 → 两个不同的 policy
2. **Mini-batch 分割**：大 rollout → 多个 mini-batch 顺序更新，后面的 batch 已经在用过期参数
3. **异步训练**：rollout workers 和 training worker 解耦，rollout 永远滞后于当前 policy
4. **框架实现差异**：vLLM 和 PyTorch 对 MoE 路由的实现细节不同，输出分布有微小偏差

每一个都会引入 IS ratio 偏差。叠加起来，你的"on-policy GRPO"可能实际上是一个高度 stale 的 off-policy 系统。

### 解法

**Jet-RL（Song Han lab, 2601.14243）— 消除来源：统一精度 flow**

rollout 和 training forward 统一用 FP8，消除精度不一致：
```
G_infer ⊆ G_train_fwd  # rollout 是 training 前向图的子集
```
效果：rollout +33%，training +41%，E2E +16%；精度损失 <1%。

**Stable Asynchrony / VCPO（Song Han lab）— 适应 off-policy**

对于无法消除的异步引入的 staleness，用两个机制控制方差：
- **ESS-based LR Scaling**：计算 effective sample size = (Σwᵢ)²/Σwᵢ²，根据 rollout "新鲜度" 动态缩放 LR
- **Closed-form minimum-variance baseline**：off-policy 专用最优 baseline，无需 critic

**VESPO（2602.10693）— 算法层纠正**

变分推导 off-policy IS reshaping 的理论最优核：
```
φ(W) = W^α · exp(-λW)
```
在 N=64 staleness 下唯一稳定的方法（GRPO 直接 collapse）。

**面试怎么说**：
> "Off-policy 是实际训练里最被低估的问题。很多工程团队以为在跑 on-policy，实际上 FP8/BF16 精度差、mini-batch 分割、异步训练都在悄悄引入 off-policy 偏差。Jet-RL 的思路是从源头消除，VCPO 的思路是动态适应，VESPO 是理论最优的算法层纠正。三者正交，可以叠加。"

---

# 第二章：Agentic RL 特有的坑

> Agent RL ≠ 单步 RLVR。多步交互引入了全新的问题维度。

## 2.1 Credit Assignment — 功过如何归因

### 现象
- Agent 最终完成了任务，但训练后它学会的是"用更多 retry"而不是"第一步就对"
- 某个关键 tool call 失败了，模型后续 recover 了，但这次失败没有被"记住"
- Agent 在简单任务上 reward 高，在复杂任务上 reward 低，但两者的学习曲线形状相同
- 加密观察：模型学会了在 trajectory 末尾写"任务完成"来骗 reward

### 为什么会这样

**Sequence-level reward 的问题**：整条 trajectory 共享一个 reward，无法区分"哪一步决策质量高"。

```
Step 1: 选对了工具 ✓
Step 2: 参数传错了 ✗  ← 真正出了问题
Step 3: 看到错误，retry ✓  ← 掩盖了 Step 2
Step 4: 任务完成，reward = 1

GRPO 的 advantage 均匀分配给 Step 1-4
Step 2 的错误决策被正向强化了
```

### 解法

**HiPER — 分层 Advantage Estimation（2602.16165）**

把 policy 分为 Planner（subgoal 级）和 Executor（step 级），分层计算 advantage：
```
Planning advantage  = aggregate(executor returns within this subgoal)
Execution advantage = local GAE within subgoal execution window
```

Credit 不再从终局 reward 向前均匀传播，而是先在 subgoal 内聚合，再向上传到 planner。

效果：ALFWorld 97.4%（+6.6%），WebShop 83.3%（+8.3%）。

**Blockwise Advantage Estimation（2602.10231）**

不分层，但把 trajectory 切成固定长度的 block，每个 block 单独估计 advantage，防止远端 reward 对近端 token 的"污染"。

**CM2 — Checklist Reward（2602.12268）**

与其等待终局 reward，不如把每个 tool call 的质量分解为 binary criteria：
```
原始：这个 tool call 质量如何？（主观，难评估）
转化为：
  □ 是否在正确时机调用？
  □ 参数格式是否正确？
  □ 是否处理了 error case？
  □ 调用前是否说明了意图？
```
把 open-ended 评估 → classification，每个 step 有独立的 dense reward。

**面试怎么说**：
> "多步 agent 的 credit assignment 是单步 RLVR 没有的问题。GRPO 的 sequence-level advantage 会把 retry 的成功归功给之前的错误决策。HiPER 通过分层 advantage estimation，让 planner 和 executor 各自在自己的时间尺度上收到清晰的梯度信号，理论可证明方差更小。"

---

## 2.2 Reward 设计 — 如何衡量"做得好不好"

### 现象
- 用 LLM-as-judge 评分，同一个 trajectory 评两次得分不同
- Reward 很高但任务实际上没完成（reward hacking）
- 模型学会了写更长的回答来获取更高 reward
- 在训练分布上 reward 高，换个场景就崩

### 为什么会这样

**开放任务的根本困难**：不像数学题有 ground truth，agent 任务（工具调用、客服、研究）：
- 没有单一正确答案
- 中间步骤质量难以自动评估
- LLM-as-judge 有一致性问题（同一 judge 对同一输出评分不稳定）
- 训练环境质量低 → reward 容易 hack → agent 学到的是"在这个环境得高分"而不是"完成这类任务"

### 解法

**解法 A — Checklist Reward / CM2**

把 open-ended 评估转化为 binary classification 检查项，一致性从主观判断 → 可重复验证。

**解法 B — Rubric-based Reward / OpenRS（2602.14069）**

不把评分逻辑内化进 judge，而是**显式推导 rubric**，每次评分在 rubric 下推理：
- 可解释（能看到为什么这么评分）
- Rubric 可跨任务迁移
- 评分一致性 > 黑盒 judge

**解法 C — 环境内置 Expert Rubric（EnterpriseGym Corecraft, 2602.16179）**

Surge AI 构建了 2500+ 真实企业实体 + 23 种工具的高保真环境，专家手写 rubric 编码进环境本身，不依赖事后评估。

关键实验：在这个高保真环境上用 GRPO 训练 GLM 4.6，**单 epoch** 后在 3 个 OOD benchmark 上泛化（+4.5%/+7.4%/+6.8%）。

**核心洞察：环境质量决定泛化上限。** 在 toy 环境上，reward 太容易 hack，agent 学到的是环境特异性策略，不是通用能力。

**解法 D — Length Control / LACONIC（2602.14468）**

防止 reward hacking 的一种：模型学会用更长回答骗 reward。

LACONIC 把长度控制构造成约束优化：
```
maximize   E[task_reward]
subject to E[length] ≤ B
```
用 Primal-Dual 交替迭代，对偶变量 λ 自动适应当前长度是否超 budget：
- 超 budget → λ 升高 → 惩罚变强
- 不超 budget → λ → 0 → 退化为无约束 RL

**比固定 length penalty 好**：固定 penalty 优化的是 surrogate 目标（与真实目标错位），LACONIC 优化的就是真实约束。

**面试怎么说**：
> "Agentic RL 的 reward 设计是最难的部分。LLM-as-judge 的一致性太差，固定 length penalty 会偏移优化目标。我会先从环境质量入手——高保真环境 + expert rubric 是根本，然后在 reward function 设计上用 checklist 把 open-ended 评估结构化，最后用 primal-dual 约束来防止长度 hacking。"

---

## 2.3 Multi-Agent 训练 — 联合优化的不稳定性

### 现象
- Orchestrator + Subagent 联合训练时，两个都在动，最终都没有收敛
- 增加 subagent 数量，训练速度线性下降（串行 rollout）
- 一个 agent 的策略变了，其他 agent 的 reward 也变了，形成 moving target

### 为什么会这样

联合训练中，每个 agent 的 optimal policy 取决于其他 agent 的当前 policy。所有 agent 同时更新 → 优化目标在不断移动 → non-stationary 环境 → 收敛极难。

串行 rollout 的性能瓶颈：100 个 subagent 串行执行 → latency = Σ(每个 subagent 的执行时间)。

### 解法

**PARL（Kimi K2.5, 2602.02276）— 冻结 Subagent，只训 Orchestrator**

核心思路：**把联合优化问题转化为单 agent 优化问题**。

```
联合训练（有问题）：
  orchestrator ←→ subagent 互相影响，优化目标不稳定

PARL：
  subagent 固定（frozen）
  只训练 orchestrator 学习：
    - 如何分解任务
    - 如何创建新 subagent（via prompt/API）
    - 如何评估 subagent 结果
```

效果：
- Agent Swarm 最多支持 100 个 subagent
- Latency 降低 4.5x（subagent 并行化）
- 简化 credit assignment：只需要归因到 orchestrator 的决策

**面试怎么说**：
> "Multi-agent RL 的核心陷阱是联合优化。PARL 的解法很优雅：冻结 subagent 把问题规约到单 agent 优化，orchestrator 学的是任务分解和 subagent 管理，而不是 subagent 的具体执行策略。代价是 subagent 能力固定，但实践中用 prompt engineering 配置 subagent 已经足够灵活。"

---

## 2.4 长 Trajectory 的训练效率

### 现象
- 10 步以上的 trajectory，显存不够
- 环境交互（API/沙箱执行）成为瓶颈，GPU 大量空转
- KV cache 随 trajectory 长度线性增长，推理越来越慢

### 解法

**异步 RL 架构（Slime/APRIL, GLM-5 技术报告）**

解耦 rollout 和 training：
```
Rollout Workers: 持续与环境交互，生成 trajectories → buffer
Training Worker: 从 buffer 取数据异步更新，周期性同步权重
```

代价：引入 off-policy 偏差（前面 1.4 节的问题）。解法：VCPO 的 ESS-based LR scaling。

**HiPER 的内存优化**

分层执行 → 可以只保留当前 subgoal 的 KV cache，subgoal 完成后 flush，总内存占用 = O(subgoal_length) 而不是 O(trajectory_length)。

**rollout locking（verl 的 multi-turn rollout locking）**

多轮对话中，已经完成的轮次锁定不再重算梯度，只对新生成的部分计算。

---

# 第三章：Reward Hacking 的各种姿势

> 这是面试里的高分项：能说出模型是怎么"作弊"的，说明你真正训练过。

## 3.1 Length Exploitation

**现象**：模型输出越来越长，padding 了大量废话，reward 反而更高。

**根因**：reward 函数隐式或显式地奖励了"看起来内容丰富"的输出。

**检测**：对比 reward 和 output length 的相关性，如果 Pearson r > 0.5，已经被 hack 了。

**解法**：LACONIC 的 primal-dual 长度约束。或者更简单：在 reward 里显式扣除超出 budget 的长度惩罚（但要用 primal-dual，不要用固定系数）。

## 3.2 Format Gaming

**现象**：模型学会了在输出里写特定格式（"#### 答案是..."、"任务完成"、JSON 结构）来触发高 reward，内容却是错的。

**根因**：reward 函数用 pattern matching 检测"正确性"，模型找到了绕过检测的格式。

**检测**：抽样检查 high-reward 输出，用人工或更强 judge 验证实际正确率。

**解法**：reward 函数本身要更鲁棒（多重验证 + 随机化格式检测），或者直接用 sandbox 执行验证（代码题 run 一遍，结果真实）。

## 3.3 Sycophancy Reward Hacking

**现象**：用 LLM-as-judge，模型学会生成让 judge "喜欢"的输出而不是正确的输出（比如语气更自信、结论更简洁清晰）。

**根因**：LLM judge 本身有偏好（position bias、verbose bias、self-preference），模型通过 RL 发现并利用了这些偏好。

**解法**：
- 用多个 judge + majority vote
- 用 rubric-based reward（OpenRS）让判断可解释，把 sycophancy 模式暴露出来
- 对 judge 做 calibration：在 held-out 标注集上测试 judge 的系统偏差，做 bias correction

## 3.4 Memory Poisoning（Agentic 特有）

**现象**：Agent 在 trajectory 早期写入了错误信息到记忆/上下文，后续基于错误信息行动，但因为信息内部一致，reward 函数没有发现问题。

**检测**：审计 agent 的 memory write 操作，用独立验证器检查写入内容的正确性。

**解法**：memory write 也加入 reward 评估；或者用 read-only 记忆（agent 只能查询不能修改全局知识库）。

---

# 第四章：实战 Recipe — 2026 最佳组合

## 完整训练 Pipeline

```
数据准备
├── 过滤纯 verifiable reward 的任务（数学/代码 > 通用对话）
└── Goldilocks：用 Teacher LM 预标注样本 utility，过滤 p_q > 0.9 和 p_q < 0.1 的样本

算法
├── 基础：GRPO + DAPO 的 clip-higher（非对称 clip）
├── Token 层：STAPO S2T mask（mask 三条件 spurious tokens）
├── 探索层：DEEP-GRPO pivot resampling（困难任务上）
└── Trust Region：MASPO soft adaptive clip（替换固定 ε）

系统
├── 精度：Jet-RL 统一 FP8 flow（如果用 FP8 加速）
├── 异步：VCPO ESS-based LR scaling（如果用异步训练）
└── 框架：verl（支持 agentic RL loop + reward loop + async）

Reward
├── verifiable 任务：直接 pass/fail（最干净）
├── tool use 任务：CM2 checklist reward（step-level）
├── 开放任务：OpenRS rubric-based reward（可解释）
└── 长度控制：LACONIC primal-dual 约束

Agentic RL 特有
├── Credit assignment：HiPER 分层 advantage
├── Multi-agent：PARL 冻结 subagent
└── 环境：尽量用高保真环境（参考 EnterpriseGym Corecraft 原则）
```

## 各方法的成本-收益

| 方法 | 额外成本 | 预期收益 | 适用场景 | 优先级 |
|------|---------|---------|---------|--------|
| DAPO clip-higher | 零 | 防 entropy collapse | 所有 GRPO | ⭐⭐⭐⭐⭐ |
| STAPO S2T mask | 接近零 | +7% 稳定性 | 所有 GRPO | ⭐⭐⭐⭐⭐ |
| Goldilocks 样本筛选 | ~25% GPU（Teacher） | ~15% 数据效率 | 数据量大时 | ⭐⭐⭐⭐ |
| Jet-RL 统一精度 | 零（需重写 rollout） | +16% E2E | 用 FP8 时 | ⭐⭐⭐⭐⭐ |
| HiPER 分层 Advantage | 低（重组 rollout 结构） | +6-8% on agent tasks | 多步 agent | ⭐⭐⭐⭐ |
| DEEP-GRPO Pivot | ~30% rollout overhead | 困难题 +20%+ | 困难推理 | ⭐⭐⭐ |
| LACONIC 长度约束 | 接近零（dual variable） | 防 length hacking | 推理模型 | ⭐⭐⭐⭐ |
| PARL 冻结 subagent | 零（设计决策） | 训练收敛 | 多 agent | ⭐⭐⭐⭐⭐ |

---

# 第五章：面试高频问题直答

**Q: 你用过哪些 RL 框架，区别是什么？**

> PPO 需要 critic（value model），适合 reward 设计不完善时提供更稳定的 baseline。GRPO 去掉 critic，用 group 内均值替代，计算成本低，但 token-level credit assignment 能力弱。veRL（Volcano Engine Reinforcement Learning）是工程最完善的开源框架，支持 agentic RL loop、reward loop、异步训练，GLM-5 用的就是基于它的 slime 框架。

**Q: 为什么你们选 GRPO 而不是 PPO？**

> 去掉 critic 的计算成本（critic 和 actor 等大），在 verifiable reward 场景下（数学/代码，有 ground truth），group 内的 pass/fail 对比已经能提供足够的 advantage 信号。代价是 token-level credit assignment 较差，需要叠加 STAPO/HiPER 等补丁。

**Q: Reward Hacking 你遇到过哪些？怎么发现的？**

> 最常见的是 length exploitation——reward 和 output length 的相关性突然升高。发现方式：monitoring dashboard 上加 length vs reward 散点图，看到 r > 0.5 立刻报警。解法是 LACONIC 的 primal-dual 约束，比固定 length penalty 更 principled，因为 primal-dual 保证约束真正被满足，而固定 penalty 只是改了优化目标而不是加了约束。

**Q: 你们的 Agentic RL 训练环境是怎么设计的？**

> 环境质量是泛化的天花板。我们的做法是：先用真实业务场景定义 task taxonomy，用 expert rubric 而不是 LLM judge 来设计 reward，把 rubric 编码进环境而不是事后评估器。EnterpriseGym Corecraft 的实验说明，在高保真环境上单 epoch 训练就能在 OOD benchmark 上泛化，这说明环境质量 > 训练步数。

**Q: Off-Policy 问题在实践中如何处理？**

> 首先要承认它几乎无法避免——异步训练、FP8 加速、mini-batch 分割都是来源。我的策略是分层处理：系统层用 Jet-RL 统一精度 flow 消除最大的精度偏差；算法层用 VCPO 的 ESS-based LR scaling 动态适应 staleness；如果用 GFlowNet 路线，VESPO 有理论最优的 IS reshaping。不能消灭就要测量，在 training loop 里监控 average IS ratio 和 ESS，超阈值就触发 rollout 刷新。

**Q: GRPO 的 entropy collapse 和 explosion 分别是什么原因？**

> Collapse：spurious tokens 携带大梯度，反复强化，把高熵位置的概率压到几乎确定性，探索空间消失。Explosion：相反方向，某些 token 的梯度方向持续相反，策略在不同方向震荡，输出趋向随机。STAPO 从三维（概率 × 熵 × 优势符号）精确诊断 collapse 来源，DAPO 的非对称 clip 防止单方向的过度更新，两者互补。

---

## 关键数字备查

| 场景 | 数字 | 来源 |
|------|------|------|
| STAPO spurious token 比例 | 0.01% | STAPO 2602.15620 |
| STAPO 效果 | +7.13% vs GRPO | STAPO |
| Goldilocks Teacher GPU overhead | ~25% | Goldilocks 2602.14868 |
| DEEP-GRPO AIME24 1.5B | 20% → 43.3% (+23%) | DEEP-GRPO 2602.14169 |
| HiPER ALFWorld | 90.8% → 97.4% (+6.6%) | HiPER 2602.16165 |
| HiPER WebShop | 75% → 83.3% (+8.3%) | HiPER 2602.16165 |
| Jet-RL E2E 加速 | +16% | Jet-RL 2601.14243 |
| LACONIC 长度减少 | >50%（精度不降） | LACONIC 2602.14468 |
| EnterpriseGym OOD 泛化 | +4.5%/+7.4%/+6.8% | Corecraft 2602.16179 |
| PACED-RL vs GRPO | AIME pass@1 +29.1% | PACED-RL 2602.12642 |
| VESPO vs GRPO (N=64 staleness) | 58.5% vs 44.7% | VESPO 2602.10693 |
| PA-MoE vs GiGPO ALFWorld | +7.7% | PA-MoE 2602.17038 |
| PA-MoE 1.5B vs 7B baseline | 1.5B > 7B | PA-MoE |

---

## 深层反思：为什么所有问题指向同一个根因？

表面上 STAPO/Goldilocks/HiPER/DEEP-GRPO 解决的是不同问题，但有一个统一根因：

> **GRPO 用序列级奖励训练 token 级决策，假设所有 token 均匀等权，但实际上它们高度异构。**

- Token 层（STAPO）：不同 token 的梯度贡献不同
- 探索层（DEEP-GRPO）：不同位置 token 的探索价值不同
- 样本层（Goldilocks）：不同难度样本的梯度贡献不同
- Agent 层（HiPER）：不同时间尺度的 token 属于不同决策层次

**真正的解法是 token-level dense reward**，让每个 token 有自己的 credit assignment。但这需要完整的 critic——就是 PPO 为了去掉 critic 而建立 GRPO 试图避开的东西。

**这是个值得说出来的悖论**：GRPO 越改越复杂，最终可能绕回 actor-critic。面试里说出这个，是真正理解了问题的本质。

---

*写于 2026-02-19 | Scholar | 基于 Vault 50+ 篇笔记综合 | 面试核心备忘*

---

# 第六章：Tool Use RL 的特有坑

> Tool use 是 Agent RL 最核心的 skill，但也是最容易出问题的地方。它引入了一批单步 RLVR 完全没有的问题。

## 6.1 Tool Call 质量无法量化 — 没有 verifiable reward

### 现象
- 模型调用了工具，但参数错误 → 工具返回了结果（虽然是错的），reward 认为"调用成功"
- 模型在不该调用工具的时候调用，或该调用的时候没调用 → 任务最终失败，但 reward 传不到"这个决策"
- Reward 只看最终结果，模型学会了"用更多 retry 覆盖错误"而不是"第一次就对"

### 为什么单步 RLVR 的方法不够

数学题：answer 是 string，对错一眼看出。
Tool call 需要评估多个维度：
- 选对工具了吗？
- 参数格式正确吗？
- 调用时机对吗？
- 处理 error case 了吗？
- 调用前有没有说明意图？

这些维度**不是 binary**，不能用一个 0/1 reward 表示。用 LLM-as-judge 评一致性又差——同一个 tool call 评两次得分不同。

### 解法：CM2 Checklist Reward（2602.12268）

把"这个 tool call 好不好"分解成若干 binary criteria，每个单独评分：

```python
def tool_call_reward(step):
    checklist = {
        "right_tool":     check_tool_selection(step),   # 0/1
        "correct_format": check_param_format(step),     # 0/1
        "right_timing":   check_call_timing(step),      # 0/1
        "error_handled":  check_error_recovery(step),   # 0/1
        "stated_intent":  check_pre_call_reasoning(step), # 0/1
    }
    return sum(weights[k] * v for k, v in checklist.items())
```

这是 **dense reward**（每步都有信号），一致性高（binary 评估不依赖 judge 的主观性）。

**面试怎么说**：
> "Tool call 的 reward 设计是 agentic RL 里最难的部分之一。Binary 的 task success reward 太稀疏，LLM-as-judge 一致性又差。CM2 把'这个 tool call 好不好'拆解成若干 binary criteria，每个单独评估，组合成 dense reward。代价是需要提前设计 checklist，但收益是每一步都有清晰信号。"

---

## 6.2 Tool Use 的异步执行瓶颈

### 现象
- GPU 利用率 < 30%，大量时间在等工具返回结果
- 加更多 GPU，throughput 没有同比提升
- 训练速度被工具 latency 卡死，而不是计算

### 为什么会这样

Tool use 的 trajectory：
```
LLM 生成 → 调用工具（等待 API/沙箱，1-10s）→ 接收结果 → LLM 继续生成 → ...
```
LLM 前向时间（几十 ms）<< 工具执行时间（秒级）。串行执行时，GPU 大量空转。

### 解法：异步 Worker Pool（VerlTool / ARLT 范式）

```
Worker Pool:
    Agent A → 生成 action → 提交工具 → [等待]
    Agent B → [收到工具结果] → 继续生成 action
    Agent C → 生成 action → 提交工具 → [等待]

Training Worker:
    buffer 满 → 拉 completed trajectories → 更新参数
```

工具调用管理器维护 pending queue，GPU 始终处理有工具结果的 agent，不阻塞等待。

**代价**：引入 off-policy（第 1.4 节），需要 VCPO ESS-based LR scaling 配套。

**UI-TARS-2 的工程方案**（字节跳动）：
- 异步状态化环境：每个 agent instance 维护独立的环境状态快照，防止 rollout 互相污染
- 流式更新：工具结果实时推入，不等整条 trajectory 完成
- 统一沙盒平台：跨浏览器/虚拟机/模拟器，百万级 rollout 规模

**面试怎么说**：
> "Tool use 训练的效率瓶颈不是 GPU 算力，是工具执行延迟。串行执行时 GPU 大量空转。解法是异步化：维护 agent pool，有工具结果就继续，没有就切换其他 agent。这引入 off-policy 问题，需要配合 ESS-based LR scaling。"

---

## 6.3 多层 Reward Hacking — 选对了工具，用错了参数

### 现象
- Reward 函数检测"调用了正确工具"，模型学会了调用，但传错参数
- 或者反过来：模型直接猜答案绕过工具，偶尔猜对 reward 一样高

### 解法：多层分离 Reward

工具调用奖励分四层，分开计算，不合并：
- `r_selection`：工具选对了吗？（检查 tool_name）
- `r_params`：参数格式和语义是否正确？（沙箱执行或格式检查）
- `r_outcome`：工具执行结果是否符合预期？（结果验证）
- `r_task`：最终任务成功了吗？

不要合并为一个 proxy——模型只会 optimize 合并后最容易骗的部分。

---

# 第七章：Deep Research Agent 的特有问题

> "Search + Reason + Synthesize" 的循环，是比单步 tool call 复杂得多的 agentic 任务。

## 7.1 Open-ended Reward — 没有 ground truth

### 现象
- Agent 搜索了很多，但内容质量低，或和问题不相关
- Agent 学会了"多搜索"让 reward 认为它在认真找
- 最终 report 内容正确但观点陈旧，没发现真正 novel 信息

### 为什么极难

Deep research 的 reward 函数几乎无法自动化：
- 没有 ground truth answer（开放问题）
- 信息质量主观（"这个来源可靠吗？"）
- Novelty 需要领域知识才能判断
- Synthesis 质量难以 programmatic 评估

### 解法

**Aletheia 的 NL Verifier 路线**（Google DeepMind, arXiv 2602.10177）

用 LLM 作为 verifier，但不是打分——而是**指出具体问题**：
- 发现具体逻辑缺陷
- 指出数学/事实错误
- 生成 structured 反馈，触发迭代修订

Verifier 不打分，而是给修订方向。比 scalar reward 信息量高一个数量级，是 Process Reward Model 的一种轻量实现。

**失败承认机制**（Aletheia 的关键设计）：
Agent 可以"宣告放弃"——这比无限重试/生成垃圾更好：
- 防止模型在超出能力的问题上空转
- 训练信号：承认失败 < 错误答案的 negative reward
- 校准意义：模型知道自己的能力边界（这本身就是 alignment 目标）

**Rubric-based Reward（OpenRS，2602.14069）**

把"这篇 research report 好不好"转化为 rubric 下的结构化评估：
```
□ 是否覆盖了问题的所有子方面？
□ 来源是否多样（不全来自同一 domain）？
□ 是否区分了事实和观点？
□ 结论是否有文献支持？
□ 是否承认了不确定性？
```
Rubric 由领域专家编写，可跨任务迁移，一致性远高于黑盒 judge。

**面试怎么说**：
> "Deep research 的 reward 设计是 agentic RL 的最难问题，因为没有 ground truth。Aletheia 的做法是用 NL verifier 作为 process reward model——不打分，而是指出具体问题，让 agent 有明确修订方向。同时引入失败承认机制，防止模型无限生成垃圾。"

---

## 7.2 探索效率的 Cost-Aware Tradeoff

### 现象
- Agent 过度搜索：对每个问题都做大量查询，latency 极高但信息增益边际递减
- 或反向：agent 搜索不足，凭参数记忆回答，答案陈旧
- 没有机制让 agent 根据问题难度动态决定"搜多少"

### RL 为什么不能自然解决这个

**Calibrate-Then-Act（arXiv 2602.16699）的关键负结果**：

端到端 RL 训练的模型**无法学会正确的探索先验**，仍然采用静态策略（无论难易都搜相同次数）。

原因：RL 能优化"在固定策略下如何执行"，但"什么时候需要探索"这个 meta 决策依赖 cost-uncertainty 的量化输入，RL 梯度信号没有包含这个信息。

### 解法：显式注入 Prior

把 cost-uncertainty tradeoff 量化后注入 prompt：

```
这类问题的参数记忆准确率约 60%，检索能提升到 85%，
每次检索成本约 0.1 分（折扣），请决定是否先检索。
```

实验结果：
- 无 prior prompt → 随机探索，reward 低
- 有 prior prompt → 接近最优探索决策（**94% optimal match rate**）
- RL + 有 prior prompt → 进一步微调

**实践意义**：不要指望 RL 自动学会"何时搜索"，需要在 prompt 里给出量化 prior（模型 calibration score + 历史准确率）。

---

## 7.3 Multi-hop Search 的 Credit Assignment

### 现象
- 第一次搜索找到了关键信息，第二次基于第一次
- 最终答案正确，但 reward 无法区分是哪次搜索的贡献
- 模型学会了"多搜索"但没学会"精准搜索"

### 解法

**Step-level Search Reward**：在每次搜索后立即评估：

```python
def search_step_reward(query, results, context):
    relevance = compute_relevance(query, results, context)  # 和当前问题的相关性
    novelty   = compute_information_gain(results, context)  # 新信息量
    quality   = assess_source_quality(results)              # 来源可信度
    return relevance * novelty * quality
```

**BrowseComp 类 benchmark 的设计哲学**：
- 题目设计保证必须多跳搜索（一跳找不到答案）
- 每次搜索结果不可 hack（答案需要聚合多次检索）
- Qwen3.5 78.6、Kimi K2.5 SOTA——这个 benchmark 难以通过 shortcut 骗分

---

# 第八章：环境设计的深层学问

> 环境设计是整个 Agentic RL 最被低估的部分，但它决定了泛化能力的天花板。

## 8.1 Toy 环境的根本问题

### 现象
- ALFWorld/WebShop 上训练的 agent，换到真实应用大幅下降
- 训练 reward 高，真实任务成功率低
- 模型学会了环境特异性 shortcut，而不是通用能力

### 根因（EnterpriseGym Corecraft 2602.16179）

**关键 empirical finding**：
- Claude Opus 4.6 / GPT-5.2 在高保真企业任务上 pass rate **< 30%**
- 同样的模型在 toy benchmark 上 90%+
- 结论：toy benchmark 的 reward 被 hack 了，不是模型真的学会了 agent 能力

**为什么 toy 环境 reward 容易 hack？**
- 状态空间小 → 穷举策略也能高分
- 任务分布窄 → 记忆解法而非泛化
- Rubric 简单 → pattern matching 能骗过验证器

### 高保真环境设计三原则

**原则 1 — Task-Centric World Building，而非 Realism-Centric**

不是追求"看起来像真实世界"，而是"任务难度和多样性足够高"。

```
❌ 错误思路：构建一个逼真的 3D 厨房模拟器
✅ 正确思路：构建 2500 个不同企业实体，每个有独特属性和工具交互模式
```

Corecraft 关键：**任务多样性 > 环境真实感**。

**原则 2 — Expert-Authored Rubrics，而非 LLM Judge**

让领域专家写 rubric，编码进环境本身：
- 不受 LLM judge 一致性问题影响
- 精确描述"正确处理这类任务"是什么样的
- 细粒度区分部分完成 vs 完全完成

**原则 3 — 以 OOD 泛化为评估标准，不是 train performance**

在训练环境之外的独立 benchmark 评估泛化：
- Corecraft 训练 → BFCL/τ²-Bench/Toolathlon 评估
- 在训练集上 reward 高但 OOD 差 = 过拟合，不是能力

**Corecraft 关键数据**：
- 单 epoch GRPO 训练：25.37% → 36.76%（+11.4%）
- OOD benchmark：+4.5% / +7.4% / +6.8%

**面试怎么说**：
> "环境质量是 agentic RL 泛化的天花板，这是 Corecraft 的核心发现。Frontier 模型在 toy benchmark 上 90%，在高保真企业环境不到 30%——说明 toy benchmark 被 hack 了。高保真环境的关键不是逼真感，而是任务多样性 + expert rubric + OOD 泛化为评估标准。"

---

## 8.2 GUI / Web 环境的特有挑战

### POMDP 的 Partial Observability

GUI/Web 任务天然是 POMDP：
- 页面内容动态变化（agent 无法预测点击后的状态）
- 只能看到当前视窗（完整状态不可观测）
- 同一操作在不同页面效果不同

**WebPilot 的处理方式**：
- 用 accessibility tree（actree）表示页面结构，捕捉 UI 元素和交互性
- MCTS 做 forward simulation：提交 action 前先模拟若干步可能的后续状态
- **Maximal Value Backpropagation**：反传子节点最大值（vs 平均值），避免好路径被差路径平均稀释

### 环境 Non-determinism

Web 环境不是 deterministic：广告、弹窗、动态内容，同一 action 不同时间执行结果不同。

**UI-TARS-2 的工程解法**：
- 异步状态化环境：每个 agent instance 独立的环境状态快照，rollout 间互不污染
- 流式更新：结果实时推入训练 buffer，不等整条 trajectory 完成

### 多模态 Tool Output 的 Credit Assignment

GUI agent 的工具输出是截图（图片），不是文本：
- 图片 token 数量多，但有用信息集中在少数区域
- 均匀对待所有图片 token 会让视觉噪声干扰文本 reasoning 的梯度

**AT-RL（Anchor Token RL，arXiv 2602.11455）**：
识别视觉输入中的"锚定 token"（关键区域），credit assignment 时给锚定区域更高权重，减少视觉噪声的梯度污染。

---

## 8.3 数据飞轮 — 让训练数据随 Agent 成长

### 现象
- 训练一段时间后，agent 把训练任务都做会了，p_q → 1，梯度消失
- 新任务不知道怎么生成，或生成的太难（p_q → 0）
- 训练陷入瓶颈

### 解法：Data Flywheel（UI-TARS-2 的做法）

```
阶段 1：用已有数据训练模型 v1
         ↓
阶段 2：用 v1 跑环境，收集 trajectories（包括 v1 做对/做错的案例）
         ↓
阶段 3：清洗数据，边界任务（p_q ≈ 0.5）进入训练集
         ↓
阶段 4：扩充后的数据训练 v2
         ↓
        循环（持续预训练 → SFT → RL → 回到阶段 2）
```

关键：**让模型的训练数据随模型能力提升而进化**。
这和 Goldilocks Teacher 动态选题是同一思路：Goldilocks 是在线实时式，Data Flywheel 是离线批次式。

---

# 第九章：PA-MoE — 架构层面的 Credit Assignment

> 前面所有解法都在优化层（reward/gradient）做文章。PA-MoE 从架构层解决了 Agentic RL 的 simplicity bias。

## 9.1 问题：单一 Policy 的 Simplicity Bias

### 现象
- Agent 在简单子任务上很好，遇到多步工具交互的复杂子任务就崩
- 增加训练轮数，简单任务继续上升，复杂任务停滞
- Gradient monitoring：简单任务占 **75%** 参数梯度，复杂任务只有 **5%**

### 根因

Simple tasks 更频繁 → 更多梯度 → 参数偏向 simple behavior → complex tasks 只有 5% 参数表征容量。

这是**架构级**问题，梯度手术（PCGrad 等）只能缓解，不能根本解决——共享参数的争夺是结构性的。

## 9.2 PA-MoE 解法（arXiv 2602.17038）

**核心思路**：用 MoE 把不同行为阶段路由到不同 expert，让 simple 和 complex tasks 不再争同一组参数。

**为什么不是 token-level MoE？**

| 路由粒度 | 每 episode 切换次数 | 问题 |
|---------|-------------------|------|
| Token 级 | ~45 次等效 | 同一动作的不同 token 路由到不同 expert，语义碎片化 |
| Trajectory 级 | ~3 次 | 粒度太粗，episode 内行为变化无法适应 |
| **Phase 级** | **~8.4 次** | ✅ 对齐语义边界 |

**Phase-Aware Router 设计**：

```python
def phase_router(obs_t, goal, action_history):
    content = cross_attention(obs_t, goal)    # 当前观察 + 目标
    context = lstm(action_history)            # 动作历史
    expert_id = argmax(linear([content, context]))
    return expert_id
# 每个 environment step 路由一次（不是每个 token）
```

温度退火 + switching penalty：防止 router 频繁切换（过度碎片化）。

## 9.3 结果

| Benchmark | PA-MoE vs GiGPO |
|-----------|----------------|
| ALFWorld | +7.7% |
| WebShop | +14.9% |

**最震惊的 finding：1.5B PA-MoE > 7B GiGPO baseline。**

架构级的 credit assignment 改进，让小模型超越大模型。这是 scaling 之外的另一条路。

**面试怎么说**：
> "PA-MoE 解决的是 Agentic RL 的结构性问题：单一 policy 在 simplicity bias 下，简单任务会垄断参数容量，复杂任务的梯度被 overwhelm。用 MoE 在 phase 级别路由——不是 token 级（太细），不是 trajectory 级（太粗）——让不同行为阶段的参数分离。结果是 1.5B PA-MoE 超过 7B GiGPO，说明架构改进比 scale 更根本。"

---

# 第十章：补充面试题库

**Q: Tool use RL 和普通 RLVR 最大的区别是什么？**

> RLVR 是单步的，answer 有 ground truth。Tool use 是多步的，没有 step-level ground truth，只有终局 reward。连锁问题：credit assignment 变难（哪一步 tool call 导致了成功？）；reward 设计变难（tool call 质量怎么量化？）；训练效率变低（等工具返回时 GPU 空转）。解法分别是 HiPER/CM2、Checklist reward，以及异步 rollout。

**Q: 你怎么设计 deep research agent 的 reward？**

> Deep research 是 open-ended，没有 ground truth，所以不能用 verifiable reward。分层设计：Step level 用 checklist reward 评估每次搜索的相关性、新信息量、来源质量；Process level 用 NL verifier 指出具体推理错误（Aletheia 思路）；Task level 用 rubric-based reward 评估报告覆盖度、来源多样性、观点事实区分；最后加失败承认机制，让 agent 知道何时停止探索。

**Q: GUI agent 为什么用 MCTS？跟 GRPO 怎么配合？**

> GUI 任务是 POMDP，点击后状态不完全可预测。MCTS 在提交 action 前做 forward simulation，评估不同 action 的价值。MCTS 是 inference-time 的树搜索，用于生成高质量 rollout；GRPO 是 training-time 的策略优化，用 MCTS 生成的 rollout 计算 advantage 进行更新。两者可以结合——MCTS 生成多条 trajectory，GRPO 在这些 trajectory 上做 group relative optimization。

**Q: 为什么 PA-MoE 用 phase-level routing 而不是 token-level？**

> Agent 的决策单位是 environment step，不是 token。Token-level routing 会让同一个语义动作的不同 token 路由到不同 expert，expert 学到的是语言风格差异而不是行为阶段差异。Phase-level routing 在 step 边界切换，每个 expert 学到一类行为模式（探索、工具调用、结果综合等）。实验验证：phase-level 约 8.4 次/episode，是真正对齐了行为语义边界的粒度。

**Q: 为什么不能指望 RL 自动学会"何时探索"？**

> CTA 论文的关键负结果：在编码任务上，端到端 RL 训练的模型无法从环境交互中学会正确的探索先验，仍然采用静态策略。原因是：RL 能优化"在固定策略下如何执行"，但"何时需要探索"这个 meta 决策依赖 cost-uncertainty 的量化输入，而 RL 梯度信号没有包含这个信息。解法是在 prompt 里显式注入 prior（模型 calibration score + 历史准确率），实验显示有 prior vs 无 prior 差距是 94% vs 23% optimal match rate。

**Q: 环境设计里最容易踩的坑是什么？**

> 追求环境逼真感而不是任务多样性。Corecraft 的核心发现是：Frontier 模型在高保真企业 agent 任务上 pass rate < 30%，而同样的模型在 toy benchmark 上 90%——说明 toy benchmark 的 reward 被 hack 了，模型没有真正学到 agent 能力。正确思路是 task-centric world building：2500 个不同任务实体比一个逼真的 3D 模拟器更有价值。评估要看 OOD 泛化，而不是训练集 reward。

---

## 扩展关键数字

| 场景 | 数字 | 来源 |
|------|------|------|
| Frontier 模型企业 agent pass rate | <30% | Corecraft 2602.16179 |
| Corecraft 单 epoch OOD 泛化 | +4.5%~+7.4% | Corecraft |
| PA-MoE vs GiGPO ALFWorld | +7.7% | PA-MoE 2602.17038 |
| PA-MoE vs GiGPO WebShop | +14.9% | PA-MoE |
| PA-MoE 1.5B vs 7B baseline | 1.5B 超越 7B | PA-MoE |
| CTA 有 prior vs 无 prior | 94% vs 23% optimal match | CTA 2602.16699 |
| Kimi K2.5 latency 降低 | 4.5x | PARL 2602.02276 |
| UI-TARS-2 rollout 规模 | 百万级跨平台 | UI-TARS-2 |
| Simplicity bias：简单任务占参数容量 | 75% | PA-MoE |
| Simplicity bias：复杂任务占参数容量 | 5% | PA-MoE |

---

---

## 关键论文导航

> 本指南所有坑和解法均有论文支撑。按章节快速跳转：

### 第一章：GRPO 失效模式
- [[AI/LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO]] — Entropy Collapse 根治：S2T mask 精确定位 0.01% spurious token
- [[AI/LLM/RL/Other-Algorithms/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] — 梯度稀疏：Teacher LM 动态课程，选 p_q≈0.5 的样本
- [[AI/LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — Root Saturation：Pivot-Driven Resampling 打开探索空间
- [[AI/LLM/RL/Other-Algorithms/VESPO-Variational-Sequence-Policy-Optimization|VESPO]] — Off-policy：变分推导最优 IS kernel，staleness 64× 稳定
- [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 系统层 off-policy：统一 FP8 精度 flow

### 第二章：Agentic RL 特有问题
- [[AI/LLM/RL/Other-Algorithms/HiPER-Hierarchical-RL-Credit-Assignment|HiPER]] — 分层 Advantage Estimation，多步 agent credit assignment
- [[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — Simplicity Bias：Phase-level MoE expert 分离参数容量
- [[AI/LLM/RL/Other-Algorithms/CM2 — Checklist Rewards多轮Tool Use RL|CM2]] — Tool Use RL：Checklist reward 多轮工具交互奖励设计
- [[AI/Agent/EnterpriseGym-Corecraft|EnterpriseGym Corecraft]] — 环境设计：高保真企业 RL 环境 + OOD 泛化

### 综述与全景
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — 六维框架元分析，本指南的学术上位文档
- [[AI/LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性统一分析]] — Token/样本/探索/系统四维拓扑

---

*扩展 v2 写于 2026-02-19 | 新增：Tool Use RL / Deep Research / 环境设计 / PA-MoE / 探索 tradeoff | Scholar*
*链接补全：2026-02-21 | 馆长*
