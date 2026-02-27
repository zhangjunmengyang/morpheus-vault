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
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解]]"
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
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解]]
