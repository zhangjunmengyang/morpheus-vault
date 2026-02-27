---
title: P5：分析 Agent——从 ReAct 到 RL 训练闭环
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, agent, RL, GRPO, reward-design, analysis-agent, meituan]
brief: 节假日分析 Agent 是真实做过的业务，从 ReAct 单 Agent 开始，深挖 prompt 工程天花板后引出后训练思路：reward 怎么设计、多轮 Agent RL 和单轮推理 RL 有什么本质区别、CM2 的 checklist reward 如何落地。贯通业务经验和后训练认知。
related:
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析]]"
  - "[[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization]]"
  - "[[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL]]"
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解]]"
  - "[[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]]"
---

# P5：分析 Agent——从 ReAct 到 RL 训练闭环

---

## 故事线（面试完整版，5-8 分钟）

---

### 第一幕：一个做过的东西，和一个一直想做的东西

节假日分析 Agent 是我在美团真实做过的项目。

功能描述很简单：单 Agent，挂着取数工具，用户或者业务方说"帮我分析一下五一假期这次的经营表现"，Agent 自主取数、对比数据、写出一份分析报告。

在内部测试阶段，效果还不错。Agent 能取到对的数、识别出主要异动、写出逻辑通顺的报告。但上线后，有一件事一直让我觉得这个系统没做到位：

**分析的质量天花板，是 prompt 工程的天花板。**

我能在 prompt 里写"要对比同比数据""要归因到流量和转化率两个维度""要给出具体的行动建议"。但这些是我能想到的，我没想到的，Agent 就不会做。有一次假期分析，Agent 没有去看竞品数据——因为 prompt 里没写"要主动对标竞品"。那份报告数据是准的，但缺了对商家最有价值的那部分信息。

这不是 prompt 写错了，是 **prompt 无法覆盖所有好分析的隐性认知**。好的分析师去看竞品是本能，这个本能没法完全写成规则。

这让我开始想：如果不是靠 prompt 告诉 Agent 该做什么，而是靠训练让 Agent 从数据里学会"好分析是什么样的"，能解决这个问题吗？

---

### 第二幕：设计 RL 训练方案——第一个难题就是 reward

一旦开始认真想这个问题，第一个拦住我的就是 reward 设计。

数学推理任务的 reward 很好定义：答案对了 1，错了 0。但分析 Agent 的 reward 是什么？

"这份分析质量高"——这个判断本身就很主观。我试着列了一下什么叫"好分析"：识别出了关键异动、追溯到了根因而不是停在表层、给出的建议有针对性且可操作、和商家实际权限匹配……

这些我知道，但怎么计算？

第一个想法是找人工标注：让业务分析师评分。但有两个问题：一是成本高，每次 RL 训练要产生大量 rollout，人工评分根本跟不上；二是一致性差，不同分析师的标准不一样，训练信号会非常噪。

后来我从 CM2 这篇论文里找到了一个更实用的思路：**把主观判断拆成可验证的 checklist，每一项是明确的 0/1 判断，加权求和得到 reward**。

具体到分析 Agent：

- ✅ 是否识别出了主要异动指标（GMV/订单/流量/转化率）？
- ✅ 是否追溯到了至少二级归因（不只说"转化率下滑"，要说出哪个维度的转化率，哪个套餐，哪个渠道）？
- ✅ 每条建议是否包含具体行动（不是"考虑优化套餐"，是"建议把定价 X 的套餐调整为 Y"）？
- ✅ 建议是否与商家权限匹配（有锁客权限才推锁客活动）？
- ✅ 是否参考了竞品数据？

这五项，每一项都是 Agent 输出结束后可以自动判断的——前四项用规则，最后一项用一个轻量的 LLM 判断（"这份分析里有没有对标竞品的内容？"）。这把不可能人工规模化的评估变成了可自动化的流水线。

---

### 第三幕：多轮 Agent 的 RL——比单轮推理难在哪

有了 reward，下一个问题是：用什么算法训练？

最直觉的答案是 GRPO——我在后训练项目里跑通了它，对数学任务效果很好。但分析 Agent 和数学任务有一个本质区别：**多轮**。

数学推理是单轮的——模型生成一段文字，答案在最后，reward 算一次。分析 Agent 是多轮的——取数→分析→再取数→补充分析→最后报告，可能十几个来回。

多轮场景下，GRPO 有一个理论问题：它的 advantage 计算里有一个方差归一化步骤（除以 std），这个操作在多轮场景下会让训练梯度不稳定，极端情况下完全无法收敛。这是 SeeUPO 论文从数学上证明过的。

所以训练分析 Agent 不能直接照搬 GRPO，需要去掉方差归一化，用更适合多轮的设计。

另一个问题是 credit assignment：多轮里，"调用竞品数据接口"这步对最终报告质量贡献很大，但"格式化输出标题"这步贡献很小。如果只给最终 reward，模型不知道是哪步做得好，训练信号太稀疏。

参考 GiGPO 的思路，可以给每个 step 单独估计 advantage，而不是只用 trajectory 级别的 reward。对"关键取数步骤"和"报告生成步骤"给不同的权重，让训练信号更密集。

---

### 第四幕：数据收集——闭环从哪里来

训练要数据，分析 Agent 的训练数据从哪里来？

最自然的来源是**系统在线运行产生的 trajectory**——Agent 做了一次分析，记录下完整的工具调用序列、每步的输入输出、最终报告。用 reward function 自动打分，低质量的过滤掉（reward < 0.3 不进训练集），高质量的留下来。

这样系统越运行，训练数据越多，模型越好——理论上是个正向飞轮。

但有个实际问题：**分布 shift**。节假日分析是有季节性的，五一的数据和春节的数据分布差异很大。用历史节假日数据训练的模型，可能对新节假日的模式泛化不好。这个问题在单轮推理任务里也有，但在 Agent 场景里更严重——因为环境本身（接口数据）也在变。

解法是滑动窗口：训练集只保留最近 N 个假期的数据，让模型持续适应最新的数据分布。这是个工程上的权衡，不是完美解法，但比全量历史数据要好。

---

### 第五幕：回到原点——这件事让我理解了什么

这个设计方案做下来，让我对"Agent 后训练"有了一个更清晰的认知：

**从 prompt 工程到 RL 训练，最难的迁移不是算法，是 reward 的定义。**

数学推理能用 GRPO 做得很好，是因为 reward 是精确的（答案对不对）。分析 Agent 的困难不是缺少好的 RL 算法，是"什么是好分析"这个问题本身很难回答清楚。

checklist reward 是一个工程上的妥协——它不完美，但它可计算、可规模化、可迭代。这比等一个完美的 reward 函数，然后什么都做不了，要好得多。

这也让我理解了为什么学术界一直在研究 reward 设计：GRPO/PPO/DPO 这些算法框架已经很成熟了，真正卡住 Agentic RL 大规模落地的，是开放任务的 reward 信号质量问题。

---

## 技术路径深化（面试追问完整版）

### a. Reward 设计迭代：checklist 的具体每一项 + LLM judge 防 Goodhart

**Checklist reward 完整设计：**

```python
def compute_reward(output: str, context: dict) -> float:
    scores = {}
    
    # 1. 识别关键异动指标（规则检查）
    key_metrics = ["GMV", "订单", "流量", "转化率", "客单价"]
    scores["anomaly_id"] = 1.0 if any(m in output for m in key_metrics) else 0.0
    
    # 2. 二级归因深度（规则检查：有没有提到子维度）
    sub_dims = ["渠道", "套餐", "时段", "评分", "图片"]
    scores["attribution_depth"] = min(
        sum(d in output for d in sub_dims) / 2.0, 1.0
    )  # 至少提到 2 个子维度才满分
    
    # 3. 建议可操作性（LLM judge）
    judge_prompt = f"判断以下建议是否包含具体行动（是/否）：{extract_recommendations(output)}"
    scores["actionable"] = llm_binary_judge(judge_prompt)  # 0或1
    
    # 4. 权限匹配（规则检查）
    shop_perms = context["shop_permissions"]
    recommendations = extract_recommendations(output)
    scores["permission_match"] = check_permission_match(recommendations, shop_perms)
    
    # 5. 竞品参考（LLM judge）
    scores["peer_reference"] = llm_binary_judge(
        f"以下分析中是否提到了竞品或同行数据对比：{output[:500]}"
    )
    
    # 加权求和
    weights = [0.25, 0.25, 0.20, 0.20, 0.10]
    reward = sum(w * scores[k] for w, k in 
                 zip(weights, ["anomaly_id","attribution_depth","actionable","permission_match","peer_reference"]))
    return reward
```

**和业务分析师对比验证（Spearman ρ ≈ 0.78）：**
- 在 200 条历史分析上对比自动评分 vs 人工评分
- 0.78 说明主要维度抓对了，但有系统性偏差：分析师更看重"洞察的新颖性"（这点 checklist 没有覆盖），checklist 更看重格式合规性
- 结论：checklist 是训练信号，不是质量真理；指引方向，不能当成 KPI

**防 Goodhart's Law（重要）：**
Goodhart 定律：一个指标成为优化目标时，它就不再是好指标。具体风险：
- 模型学会"堆砌关键词"满足 checklist，但分析逻辑空洞
- 模型学会输出格式合规但无意义的子维度归因（如每次都提到"渠道/套餐/时段"三个词但不做实际分析）

应对方案：
1. **定期人工审查样本**：每 500 次 rollout 抽 10 条人工评，检查 checklist 高分的是不是真的质量高
2. **多元 reward**：不只优化 checklist，同时优化 KL 散度（防止过度 reward hacking）
3. **reward 版本管理**：每个 checklist 版本打 tag，结合线上转化率指标验证 reward 和真实效果的相关性——如果 reward 高但转化率没变，说明 Goodhart 了

---

### b. Multi-turn RL 方案选型：SeeUPO vs RAGEN vs 标准 GRPO

**三个方案的核心差异：**

| 方案 | 核心创新 | 适用条件 | 主要代价 |
|------|---------|---------|---------|
| 标准 GRPO | Group advantage + std 归一化 | 单轮推理（数学/代码） | 多轮场景梯度不稳定（SeeUPO 证明） |
| SeeUPO | 去掉 std 归一化，序列级别 advantage | 多轮 Agent，需要收敛保证 | 训练早期方差大，需要更仔细的 warmup |
| RAGEN | 把整个 Agent 轨迹（multi-turn）当作单一 "reasoning"，用 trajectory reward 端到端训 | 任务结构化程度高的场景（工具调用有明确分工） | 长轨迹梯度消失问题；显存占用高 |

**为什么分析 Agent 选 SeeUPO 方向：**

数学上，标准 GRPO 的 advantage：

```
a_i = (r_i - mean(r)) / std(r)
```

多轮场景下，每个 turn 的 group 内 std 差异很大（有些 turn 所有 rollout 都拿到一样的 reward，std → 0，除以 std → inf）。这会导致训练崩溃。

SeeUPO 把归一化去掉，改成：

```
a_i = r_i - mean(r)   # 只去中心，不归一化
```

代价是不同 turn 的 advantage 量级可能差很大，所以要配合 clipping（PPO-clip style）防止单个步骤的梯度过大。

**RAGEN 的适用场景说明：**
RAGEN 的逻辑是：既然整个 Agent 轨迹是连续决策，不如整体端到端训，让模型学习"什么样的工具调用序列"最终导致好结果，而不是在每个 step 单独归因。这对工具调用序列有固定结构（比如总是：取数→分析→输出）的任务更有效。但分析 Agent 的轨迹长度不固定（简单问题 2-3 轮，复杂问题 10+ 轮），RAGEN 的长轨迹梯度消失问题在这里比较严重，不选。

---

### c. Credit Assignment：GiGPO step-level vs trajectory-level

**为什么 trajectory-level reward 不够：**

```
多轮轨迹示例：
Turn 1: 理解商家问题，判断是流量还是转化率问题  ← 关键决策
Turn 2: 调用 get_core_metrics 取 GMV 数据       ← 重要
Turn 3: 调用 get_peer_data 取竞品数据           ← 重要
Turn 4: 发现异常，决定深挖时段维度              ← 关键决策
Turn 5: 调用 get_hourly_breakdown              ← 重要
Turn 6: 格式化输出报告标题                     ← 低贡献
Turn 7: 生成分析正文                           ← 高贡献
Turn 8: 生成建议                               ← 最高贡献

trajectory-level reward = 0.85（整体打分）
```

问题：Turn 6（格式化标题）和 Turn 8（建议）拿到同样的 reward。模型不知道 Turn 4 的关键归因决策是"功劳"最大的一步。

**GiGPO step-level advantage 计算：**

GiGPO 的思路是：对同一个 prompt，在同一个 step 采样多次，用这个 step 的局部 group reward 估计该步的 advantage：

```
# 每个 step i 的 advantage
# 从同一 step 起点出发，采样 G 条轨迹直到 done
step_rewards = [rollout_from_step_i(policy) for _ in range(G)]
a_i = mean(step_rewards) - global_baseline
```

好处：Turn 4 的多次 rollout 里，选择深挖时段的 rollout 最终 reward 更高，"深挖时段"这个 action 的 advantage 就更高。Turn 6（格式化标题）无论怎么变体 reward 差异不大，advantage ≈ 0，梯度几乎不更新。

**如何判断哪步是关键步（工程判断）：**
不总是能提前知道。实践上的做法：
1. 运行一批轨迹，对每个 step 的 advantage 方差做分布分析
2. 方差高的 step = 该 step 的决策对结果影响大（关键步）
3. 方差接近 0 的 step = 可以减少采样次数，节省计算
4. 用这个分布指导采样策略：关键 step 多采样（G=8），低贡献 step 少采样（G=4）

---

### d. 数据收集闭环：在线 rollout 过滤标准 + 冷启动

**过滤标准（三条线）：**

```python
def should_include_in_training(trajectory) -> bool:
    reward = compute_reward(trajectory.output, trajectory.context)
    
    # 1. 质量下限：reward 太低的不要（噪声太多）
    if reward < 0.3:
        return False
    
    # 2. 质量上限：reward 太高的也要小心（可能是"简单题"）
    # 过多高分样本会让训练集 easy case 过多，难案例学不到
    if reward > 0.95 and is_simple_case(trajectory):
        return random.random() < 0.3  # 只保留 30%
    
    # 3. 多样性过滤：避免同质化（相似 query 保留最多 5 个）
    similar_count = count_similar_in_buffer(trajectory.query)
    if similar_count >= 5:
        return False
    
    return True
```

**分布 shift 的处理（滑动窗口）：**

```
训练集 = 最近 6 个节假日 × 各类型商家样本
├── 覆盖时间跨度约 3 个月（避免太老的数据）
├── 每个节假日类型（节/周末/日常）保持大致平衡
└── 当新节假日到来，滑出最老的那批数据
```

滑动窗口的代价：冷启动期没有该类型数据（第一个五一来了，训练集里还没有五一的样本）。冷启动方案：
1. 用相似节假日的数据临时替代（国庆 → 五一，结构相似）
2. 手工构造 20-30 条该类型的 seed trajectory（成本可接受）
3. 上线时降低置信度阈值，允许更多探索

**在线 rollout vs 离线数据的取舍：**
在线 rollout（系统跑了给真实商家看的分析）有真实分布的优势，但有两个风险：① 收集速度受限于真实流量，冷启动慢；② 差的 rollout 已经展示给商家了，无法"撤回"。

折中方案：维护一个"shadow mode"——和真实 rollout 同时，以更高探索率（temperature=1.2）多跑一批 rollout，只用于训练，不展示给商家。这样可以获得更多样化的探索数据，同时不影响真实用户体验。

---

## 快速技术速查（追问备用）

**"分析 Agent 的 reward function 你有实际跑过实验吗？"**
checklist reward 函数本身做过验证——在一批历史分析案例上，用 checklist 自动评分，然后和业务分析师的人工评分对比，相关性在 0.78 左右（Spearman correlation），说明这个函数基本上抓住了人类判断的主要维度。完整的 GRPO 训练闭环还在推进中。

**"为什么 multi-turn RL 要去掉方差归一化？"**
SeeUPO 的证明是：GRPO 的 std 归一化会让不同 group 的梯度方向产生系统性偏差，在多轮设置下这个偏差会累积，破坏 Mirror Learning 框架的收敛保证。去掉归一化，用 raw reward 差异作为 advantage，可以在多轮场景下保持收敛性，代价是训练早期的稳定性稍差。

**"和你做的后训练项目（P2）有什么联系？"**
P2 是技术基础——搞清楚了 SFT/DPO/GRPO 的工程实现。P5 是往业务场景延伸——当 reward 不再是精确匹配，当任务不再是单轮，当数据来自线上系统而不是固定数据集，需要哪些额外的设计考量。这两个合在一起才是比较完整的后训练视角。

---

## See Also

- [[Projects/项目故事/P2-后训练大项目-MA-RLHF工程实战]]
- [[Projects/项目故事/P4-商家诊断Agent-安全防御层]]
- [[AI/2-Agent/Agentic-RL/Agentic-RL-元问题-瓶颈与突破方向]]
- [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization]]
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL]]
- [[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]]
- [[AI/3-LLM/RL/算法/GRPO 深度理解]]
