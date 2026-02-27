> [!warning] ⚠️ 路径错误 — 旧结构副本
> 本文件写入了旧路径 `AI/Agent/Agentic-RL/`（大重构后已弃用）。
> 正式版请见：[[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO 正式版]]（13711B，完整frontmatter+wikilinks）
> 本文件保留作参考，内容为正式版的子集。

# GiGPO：Group-in-Group Policy Optimization for LLM Agent Training

**论文**：arXiv:2505.10978（NeurIPS 2025）
**机构**：南洋理工大学 + Skywork AI
**作者**：Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An
**代码**：https://github.com/langfengQ/verl-agent
**评级**：★★★★★
**标签**：#AgentRL #credit-assignment #GRPO #anchor-state #多轮训练 #NeurIPS2025

---

## 一、核心问题

**GRPO 在 agent 任务上的根本局限**：

GRPO 的 advantage 估计是 trajectory-level 的——同一 episode 内的所有 step 共享一个 advantage 值（由 episode 总奖励相对组内其他 episode 的均值计算）：

```
A_GRPO(s, a) = R_episode - mean(R_group)

→ 无论你第3步做对还是做错，advantage 都一样
→ 无法区分"哪一步是关键转折"
```

**Agent 任务的特殊挑战**：
- 一个 ALFWorld episode 可以有 **50+ 步、20k+ tokens**
- 奖励通常在 episode 末尾才到（sparse reward）
- 一个 bad action 可能在 10 步后才体现为失败
- 用 episode-level advantage 训练 = 告诉模型"这整个 episode 都不好"，但不知道哪一步出了问题

**核心问题**：如何在 group-based RL 框架下（无需 critic，无需额外 rollout）实现 step-level credit assignment？

---

## 二、GiGPO 核心机制

### 2.1 两级 advantage 结构

```
Level 1（Episode-level）：Macro Advantage
  对每个 episode，计算相对于同组所有 episode 总奖励的 advantage
  = GRPO 的标准计算（baseline reduction）
  作用：判断"这个 episode 整体好不好"

Level 2（Step-level）：Micro Advantage via Anchor State Grouping
  关键创新：利用"同一任务内不同 trajectory 重访相同状态"这一自然现象
  构建 step-level group：把所有 trajectory 中处于相同 anchor state 的 action 分到一组
  在组内计算相对 advantage
  作用：判断"在同一状态下，这个 action 比其他 action 好多少"
```

### 2.2 Anchor State 的直觉

**关键 insight**：在 agent 任务中，**多条轨迹会自然地重复经过相同的中间状态**。

例如 ALFWorld（家庭导航任务）：
- Trajectory 1：进厨房 → 找杯子（成功） → 放到桌上
- Trajectory 2：进卧室 → 回到厨房 → 找杯子（成功） → 放到桌上
- Trajectory 3：进厨房 → 找错地方 → 回到厨房 → 找杯子（失败）

Trajectories 1 和 3 都经过了"厨房找杯子"这个 anchor state，但 1 找到了、3 没找到。把这两个 action 放在一起比较 → 可以精确评估"在这个状态下，这个 action 的价值"。

```
WebShop 的 anchor state：同一商品列表页面
ALFWorld 的 anchor state：同一房间同一状态
Web Agent：同一页面 + 同一任务状态
```

### 2.3 算法形式化

**Episode-level macro advantage**：
```
A_macro(τ_i) = (R_i - mean(R)) / std(R)  # 标准 GRPO normalization
```

**Step-level micro advantage**：
```
对于 anchor state s_a 和从该状态发出的 action a_k：
A_micro(s_a, a_k) = (Q(s_a, a_k) - mean(Q(s_a, *))) / std(Q(s_a, *))

其中 Q(s_a, a_k) = 从 (s_a, a_k) 出发到 episode 结束的累积奖励
```

**组合 advantage**：
```
A_GiGPO(s_t, a_t) = A_macro(τ) + α * A_micro(s_t, a_t)  if s_t is anchor state
                  = A_macro(τ)                             otherwise (no anchor)
```

其中 α 是平衡两级 advantage 的超参（通常 α=1）。

### 2.4 Anchor State 识别

如何判断两条轨迹经过了"相同状态"？

- **字符串匹配**：对环境 observation 做 hash 或文本比较（ALFWorld 的 observation 是文本）
- **语义相似度**：对 observation 做 embedding 后计算余弦相似度（更适合网页截图等高维输入）
- **结构化状态**：某些环境有显式状态表示（e.g., WebShop 的商品 URL + 购物车状态）

**实践细节**：retroactively 构建——先采集一批轨迹，再回头找 anchor states（而非 online 实时更新）。

---

## 三、计算成本分析

GiGPO 的优雅之处：**anchor state 复用了已有 rollout，无需额外计算**。

```
标准 GRPO：
  对每个 prompt，采集 G 条轨迹 → 计算 episode-level advantage → 更新
  
GiGPO：
  对每个 prompt，采集 G 条轨迹（完全相同的采集过程）
  → 计算 episode-level advantage（同 GRPO）
  → 额外：找 anchor states，在 anchor groups 内计算 micro advantage（只是查找 + 统计）
  → 更新
```

**额外时间开销：< 0.002%**（论文报告值）。

内存开销：和 GRPO 完全相同（不需要额外的 value network 或 extra rollout buffer）。

---

## 四、实验结果

| Benchmark | GiGPO vs GRPO | 模型 |
|-----------|--------------|------|
| ALFWorld（家庭导航）| **+12%** | Qwen2.5-1.5B/7B |
| WebShop（网购）| **+9%** | Qwen2.5-3B/7B |
| Search QA | 42.1%(3B), 47.2%(7B) | Qwen2.5-3B/7B |

**与其他 baseline 对比**：
- vs. Actor-Critic (PPO)：GiGPO 显著超过，且无需 critic network（更省内存）
- vs. ReAct + Reflexion（纯 prompt）：大幅超过（RL post-training 的优势）
- vs. GRPO：在所有任务上一致超过

**消融实验关键发现**：
- 去掉 micro advantage（只用 macro）→ 性能显著下降（回到 GRPO 水平）
- 只用 micro advantage（去掉 macro）→ 也下降（episode-level 全局信息不能丢）
- 两者组合 > 任意单独使用

**Scaling**：GiGPO 的优势随模型规模增大而更显著（1.5B → 3B → 7B 提升幅度增加）。

---

## 五、GiGPO vs 其他 Credit Assignment 方法

| 方法 | 需要额外 rollout | 需要 critic | 适用场景 | 粒度 |
|------|----------------|------------|---------|------|
| GRPO | 否 | 否 | 单/多轮 | episode |
| PPO | 否 | **是** | 单/多轮 | step（via V function）|
| AgentPRM | **是**（MC rollout）| 否 | agent | step |
| **GiGPO** | **否** | **否** | **多轮 agent** | **step** |
| iStar | 否 | 否 | preference 数据 | step（implicit）|
| MIG | 否（只需 P(a)）| 否 | verifiable tasks | step |

**GiGPO 的独特组合**：critic-free + rollout-free + step-level credit。这是真正的「既要又要还要」，代价是依赖环境状态可以被识别为"相同"。

---

## 六、适用条件与局限

### 何时 GiGPO 工作最好

1. **环境有可识别的重复状态**：同一任务不同轨迹会经过相同中间状态
2. **稀疏奖励的长 horizon 任务**：步骤越多、奖励越稀疏，GiGPO 相比 GRPO 的优势越大
3. **并行轨迹采样**：需要同一初始状态下多条轨迹（GRPO 的标准设置，满足）

### 局限

1. **Anchor State 依赖**：如果不同轨迹从不重访相同状态（例如高度随机化的环境），anchor 数量极少，micro advantage 无法有效估计
2. **状态识别精度**：两个"语义相同"的状态可能有微小的文本差异（environment observation 变动），需要鲁棒的状态比较方法
3. **单轮任务无效**：数学/代码等单轮任务不存在 step-revisit，无法构建 anchor groups（退化为 GRPO）
4. **Anchor 稀疏性**：早期训练时轨迹比较随机，anchor state 本来就少；随着训练进展，轨迹变得更规范，anchor 更多 → GiGPO 的效果随训练进展而增强

---

## 七、工程实现（verl-agent）

代码基于 verl 框架（RLHF 训练框架），在 `verl-agent` repo 实现：
- `algos/gigpo.py`：核心算法
- `envs/`：ALFWorld / WebShop 环境接口
- anchor state 识别：基于 observation 文本 hash

**训练超参**：
- Group size G = 8（每个 prompt 采集 8 条轨迹，同 GRPO）
- α = 1.0（macro 和 micro advantage 等权）
- anchor 识别阈值：exact match 或 cosine ≥ 0.95

---

## 八、我的评价

### 真正 Novel 的地方

**Anchor state grouping** 这个思路极其 elegant。它把一个看似困难的问题（step-level credit without extra rollout）转化为：利用并行轨迹中**已经发生**的状态重叠来构建 step-level group。

这是一种**数据重用**的思维：同样的 G 条轨迹，GRPO 只用来做 episode-level 比较，而 GiGPO 额外提取出 step-level 的比较信息。信息被更充分地利用了。

### 与 MIG 的关键对比

MIG 和 GiGPO 都是在 GRPO 框架上增加 step-level credit，但出发点完全不同：

```
MIG：从"信息论"出发 ← 这一步对预测答案的边际贡献
     需要：verifiable answer oracle
     适合：推理/数学任务（有明确答案）

GiGPO：从"比较"出发 ← 在相同状态下，这个 action 比其他 action 好多少
       需要：环境状态可以被识别为相同
       适合：agent 任务（有重复状态的多轮交互）
```

两者正交，理论上可以组合（对有 anchor 的 step 用 GiGPO micro advantage，对其他 step 用 MIG）。

### 面试价值

GiGPO 是 **NeurIPS 2025** 的 Agent RL 工作，面试谈到 credit assignment 时能准确说出：
- "GRPO 的 advantage 是 episode-level 的，step-level 信息丢失"
- "GiGPO 通过 anchor state grouping，利用并行轨迹中自然出现的状态重叠，构建 step-level group 估计 micro advantage"
- "零额外 rollout 和 memory 开销（< 0.002% 额外时间）"

---

## 九、与 Vault 其他笔记的关联

```
Credit Assignment 谱系（完整版）：

Trajectory-level：
  GRPO → group relative advantage
  
Step-level（无需标注，无需额外 rollout）：
  GiGPO  → anchor state grouping（依赖状态重叠，适合 agent）
  MIG    → 信息论边际增益（依赖 verifiable answer，适合推理）
  iStar  → trajectory DPO = implicit step reward（preference 数据）

Step-level（需要额外 rollout/标注）：
  AgentPRM → MC rollout 估计 step value
  PRM      → 人工标注
```

关联笔记：
- `Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题.md`（需更新 GiGPO 部分）
- `LLM/RL/Other-Algorithms/MIG-Step-Marginal-Information-Gain.md`
- `Agent/Agentic-RL/Agentic-RL-2026前沿综合分析.md`
