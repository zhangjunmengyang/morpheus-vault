---
brief: "GRPO 改进全景：七维框架（优势估计/奖励塑形/KL约束/长度规范/熵控制/多目标/Diversity）系统梳理 2026 年 GRPO 衍生算法；DAPO/VAPO/Dr. GRPO/ProGRPO/MASPO/SAPO/GSPO 等 15+ 变种的核心改进方向对比分析。关键边界：GRPO 在 multi-turn 场景无收敛保证（SeeUPO 不可能定理），单轮推理仍是首选。"
title: "GRPO 改进全景分析：2026 年七维框架"
type: synthesis
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - grpo
  - survey
  - interview-prep
  - type/synthesis
date: 2026-02-20
updated: 2026-02-24
sources:
  - "GRPO/DeepSeekMath: arXiv:2402.03300"
  - "DAPO: arXiv:2503.14476 (ByteDance/清华，NeurIPS 2025)"
  - "MASPO: arXiv:2602.17550 (Meituan+Fudan等)"
  - "SAPO: arXiv:2511.20347 (Qwen团队，Qwen3-VL生产)"
  - "GSPO: arXiv:2507.18071 (Qwen3团队)"
  - "STAPO: arXiv:2602.15620 (清华+滴滴)"
  - "DEEP-GRPO: arXiv:2602.14169 (ICML投稿)"
  - "Goldilocks RL: arXiv:2602.14868 (Apple+EPFL)"
  - "Jet-RL: arXiv:2601.14243 (MIT HAN Lab)"
  - "SeeUPO: arXiv:2602.06554 (Tongyi Lab) — multi-turn收敛边界定理"
  - "IntroLLM: arXiv:2602.13035 (Diversity维度，hierarchical温度policy)"
  - "ProGRPO: arXiv:2602.05281"
  - "RePO: arXiv:2602.10819"
---

# GRPO 改进全景分析：2026 年六维框架

**类型**: 综合分析 / 面试级元分析  
**覆盖时间**: 2025-10 ~ 2026-02  
**写作日期**: 2026-02-20  
**状态**: v1（覆盖 6 篇核心论文，持续更新）

---

## 为什么写这篇

GRPO（Group Relative Policy Optimization）在 DeepSeek-R1 之后成为 RLVR 的标准算法。在 2025 秋季到 2026 年初，出现了大量针对 GRPO 各种缺陷的改进工作。

这些工作散落在不同论文中，每篇都说自己解决了"GRPO 的问题"。但问题是——**GRPO 有很多不同层次的问题，这些论文实际上是在修不同的漏洞**。

这篇笔记的目标：把所有改进工作**定位到正确的层次**，建立一个统一的分析框架。

---

## GRPO 的核心公式（基准）

对于 query q，采样 G 个 response：

```
J(θ) = E[1/G Σᵢ 1/|oᵢ| Σₜ min(rᵢ,ₜ · Aᵢ,ₜ, clip(rᵢ,ₜ, 1-ε, 1+ε) · Aᵢ,ₜ)]
```

其中：
- rᵢ,ₜ = π_θ(oᵢ,ₜ|q) / π_θ_old(oᵢ,ₜ|q)（重要性比率）
- Aᵢ = (rᵢ - mean(r)) / std(r)（组内相对优势）
- clip 防止过大的策略更新

**GRPO vs PPO 的简化**：用组内均值替代 critic value model → 去掉 critic，但同时失去了 token 级别的 credit assignment。

---

## 七维改进框架（v2：新增 Diversity 维度）

> **v2 更新（2026-02-21）**：补充维度七"Diversity/Entropy"，收录 ProGRPO 和 RePO。

### 维度七：Diversity 层 — 如何保留多条正确路径？

**问题**：GRPO advantage 只看 reward，不看路径的生成概率。高概率的「主流解法」每次被采样都得正 advantage，低频但同样正确的路径概率越来越低。训练后 pass@1 尚可，pass@k 暴跌——多样性死了。

**ProGRPO**（arXiv 2602.05281，Pengyi Li et al.）
发现：entropy collapse 的根因在 advantage 本身，而非外部熵正则化不足
解法：ARM（Advantage Re-weighting Mechanism）
- `c_θ(q)`：prompt 置信度（模型对该问题的熟悉程度）
- `c_θ(o|q)`：answer 置信度（对该条路径的生成自信度）
- `Ã_i = A_i + α(c_θ(q) - c_θ(o|q))`
- 高置信路径（dominant solution）→ advantage 打折扣；低频正确路径 → advantage 加权
- 只对低概率 token（约 20%）做长度归一化，避免 trivial token 稀释信号
效果：Pass@1 +5.7%，Pass@32 +13.9%（Qwen2.5-7B），CodeForces rating +180

与 entropy regularization 的区别：entropy bonus 是外部强制多样性，ARM 是内部重塑 advantage 结构，更 principled，不破坏整体 objective。

**RePO**（arXiv 2602.10819，Linxuan Xia et al.）
视角：从 off-policy 知识利用角度解 hard sample 采不到的 diversity 问题
发现：LUFFY（off-policy RL）失败的根因是词表不一致导致 importance ratio 失控
解法：Rephrasing Policy Optimization
1. 让模型读懂 off-policy 专家解法，然后用自己的话重写 → on-policy 兼容轨迹
2. 只在 group 失败率 ≥ ρ 时才注入重写轨迹（替换最差 rollout）
3. 对正常问题保持纯 on-policy，不污染分布

RePO 和 ProGRPO 解决了 diversity 的两个不同来源：
- ProGRPO → 已采到的正确路径里，扶持低频路径
- RePO → 采不到的正确路径，通过知识内化引入

---

### 维度一：Token 层 — 哪些 token 不该学？

**问题**：序列级奖励均匀分配给所有 token，但某些 token 是"虚假信号源"——它们的梯度破坏训练稳定性。

**STAPO**（arXiv 2602.15620，Tsinghua + DiDi）  
发现：0.01% 的 token 满足三个条件（低概率 + 低熵 + 正优势）→ 这类 token 对训练贡献极端梯度 → 引发 entropy collapse  
解法：S2T mask（Spurious-to-Truncate）——把这类 token 的 clip 截断从 1+ε 提前到 1  
效果：+7.13% vs GRPO on MATH benchmarks；entropy 稳定

**MASPO**（arXiv 2602.17550 ✅，MSRA，Xiaoliang Fu/Xunliang Cai）  
发现：正/负样本的概率质量分布不平衡（正样本 token 概率 > 0.5 占多数，负样本相反）→ 固定 clip ε 对两类样本效果不同  
解法：Soft Adaptive Trust Region——根据每个 token 的概率质量动态调整 clip 范围  
效果：比固定 clip GRPO 提升 ~5%

**共同洞察**：token 级别的梯度分析比序列级别的奖励分析更重要。GRPO 把所有 token 一视同仁，是根本缺陷。

---

### 维度二：Exploration 层 — 探索从何处来？

**问题**：GRPO 的 root sampling 在生成开始时固定了起点，policy 只在高概率区域探索 → 在困难问题上 diversity 严重不足。

**DEEP-GRPO**（arXiv 2602.14169，ICML 投稿）  
发现：N=8→64 个采样几乎不提升性能（root 锁定了探索区域）  
解法：Pivot-driven Resampling + Logistic Regression Recoverability 估计  
- 找到推理链中的关键决策点（pivot）
- 在 pivot 处重新分支，强制探索 pivot 之后的分叉路径
- 用 Q(t) ∝ P(success|s_{<t}) × (t/T)^γ 选择最有价值的 pivot
效果：avg 54.0% vs 51.4% (Dr.GRPO)

**QeRL / Jet-RL**（arXiv 2510.11696 / 2601.14243，Song Han lab）  
发现的副作用：量化噪声 → policy entropy 增加 → 更好探索  
这不是目的设计，但 QeRL 把它系统化为 AQN（Adaptive Quantization Noise）  
效果：reward 在 200 步内快速上升（vs vanilla LoRA 需 500+ 步）

**IntroLLM**（arXiv 2602.13035，ICML 投稿，2026-02-13）  
发现：温度是 RLVR 最被忽视的控制变量——固定温度无法适应 token 位置、prompt 难度、训练阶段的变化  
解法：Hierarchical RL — 从 LLM **内部隐状态** hₜ 学习 temperature policy πϕ(τₜ|hₜ)  
- 轻量 MLP head（d/2 bottleneck）从最后一层 decoder 分支
- 混合离散-连续动作：Bernoulli gate（是否更新）+ Beta 分布（连续值采样）
- GRPO coordinate ascent 联合优化 token policy θ 和 temperature policy ϕ
效果：一致优于固定温度和启发式自适应；高温自然分配到推理转折点，低温到数值计算/答案合成  
关键洞察：这是三级探索控制精度的最高级 — trajectory-level（粗）→ token-level（中）→ **hidden-state-conditioned token-level（最细）**

**共同洞察**：GRPO 的探索依赖温度和 group size，这是非常粗糙的机制。Pivot 级别的 targeted exploration（DEEP-GRPO）、parameter-level entropy injection（QeRL）、internal-state-conditioned temperature（IntroLLM）是三种不同精度的路径，可以组合。

---

### 维度三：Sample 层 — 用什么样的题训？

**问题**：GRPO 对训练样本一视同仁，但不同难度的题对学习贡献不同：太简单（全对）= 零梯度信号；太难（全错）= 也是零梯度。最有价值的样本在"能力边界"附近。

**Goldilocks RL**（arXiv 2602.14868，Apple + EPFL）  
核心数学：`||∇L_PG|| = √(p_q(1-p_q))`  
当 p_q ≈ 0.5 时梯度最大，即"刚好能做对一半"的题梯度最丰富  
解法：用 Teacher LM（小而快）预测每道题的 utility，只选梯度丰富的题训练  
Teacher LM 判断：提前采样少量 responses，估算该题对当前模型的 pass rate  
效果：在"Edge of Competence"数据上训练比全量 random 样本高 ~15% 效率

**隐含的 curriculum**：这不只是难度筛选，是 adaptive curriculum learning 的隐式实现——随着模型能力提升，适合它的题集也在动态变化。

**PACED-RL**（arXiv 2602.12642，ICML 投稿，Dohyung Kim 等）— GFlowNet 框架下的 Sample 选择  
核心发现：GFlowNet 训练时必须学的可学习配分函数 Z_φ(x) 实际上编码了在线准确率：  
`p_old(x) = β·log Z*(x) - β·D_KL(π_old || π_θ)` → KL 项可近似忽略（实验最大值 < 4×10⁻³）  
`p_old(x) ≈ β·log Z_φ(x)`  
这意味着中间难度题的筛选可以**零额外开销**地做（代价已摊入 GFlowNet 训练）。  
组件 1：用 Z_φ 估计每题准确率，优先选 accuracy≈0.5 的题（与 Goldilocks 殊途同归！）  
组件 2：accuracy estimation error 优先的 replay（利用 GFlowNet off-policy 容忍度）  
效果：AIME pass@1 vs GRPO +29.1%，vs FlowRL +40.0%；pass@k（多样性）同样提升

**Sample 维度两条路**：
```
Goldilocks（GRPO 框架）：Teacher LM → utility 估计 → 中间难度  
PACED-RL（GFlowNet 框架）：Z_φ → 在线准确率估计 → 中间难度  
```
两者用不同数学工具发现了**同一个 empirical 规律**（中间难度最有效），相互印证。

**共同洞察**：数据选择比算法改进的天花板更低但更稳定。对 production 系统来说，Goldilocks/PACED-RL 类似的 sample filter 可能比复杂的算法改进更实用。GFlowNet 路线额外保持了输出多样性——这对 test-time scaling（pass@k）有直接价值。

---

### 维度四：Trust Region 层 — clip 该怎么设？

**问题**：固定的 ε（clip 范围）是 GRPO 最大的超参数之一，但对不同 token 应该不一样：

- 高概率 token（模型已经很确定）：小 ε，不需要大幅更新
- 低概率 token（模型不确定）：大 ε，允许更大幅度的策略更新
- 负样本 token 和正样本 token：概率质量分布不同，clip 效果不同

**MASPO** 已在维度一中讨论，但 Trust Region 视角更清晰：  
Soft Adaptive Trust Region 的本质是**把 clip 从全局超参数变成 token 级别的动态函数**。

**SAPO**（arXiv 2511.20347，2025-11-25，Qwen 团队 Chang Gao 等）  
核心洞察：hard clip 的梯度截断是非连续的，导致 clip 内外梯度断崖。  
解法：sigmoid 软门控——梯度权重 = **sech²(τ/2 · (r−1))**，在 r=1 时满权重，随偏差平滑指数衰减。  
不对称温度：τ_neg > τ_pos，因为负 advantage 梯度影响 |V| 个 unsampled token，更不稳定。  
理论分析：在 (A1) 小步长 + (A2) 低序列内方差 条件下，SAPO 退化为 GSPO 的连续版本（sech² 序列门控）。  
生产验证：Qwen3-VL 全系列用 SAPO 训练。  
与 GSPO 关系：同一 Qwen 团队，GSPO(2025-07) → SAPO(2025-11)，是对 GSPO 硬裁剪的直接改进。  
**弱点**：在高 staleness（N=64）下崩溃（18.4%），VESPO 同条件 58.5%——token-level 软化仍缺乏序列级 IS 方差理论。

**与 DAPO/VAPO 的关系**：
- DAPO（ByteDance）：提高 clip 上界防 entropy collapse + token-level loss
- VAPO（Bytedance）：Variance-Aware 优势估计，对高方差 token 保守更新
- MASPO：Probability-Mass Aware trust region
- **SAPO**：sech² 软衰减，连续信任域

四者都在解决固定 ε 的问题，但切入角度不同（hyper / variance / mass / **softness**）。

---

### 维度五：Off-Policy 层 — 训练-推理精度一致性

**问题**：RL 训练要求 rollout 和 logit evaluation 使用**相同**的策略。但当 rollout 用低精度（FP8）而 evaluation 用高精度（BF16）时，两者实际上是不同的策略 → 引入 off-policy 偏差。

Off-policy 有三个实践来源（VESPO 明确列举）：
1. mini-batch 分割（大 rollout → 多个 mini-batch 顺序更新，后面的 batch 使用过期参数）
2. 异步训练（rollout 和 training 并行，rollout 永远落后）
3. train-inference mismatch（FSDP/Megatron vs vLLM/SGLang 实现不同，MoE 路由差异放大）

**Jet-RL**（arXiv 2601.14243，Song Han lab）：**消除** off-policy 来源  
统一 rollout 和 training forward pass 到 FP8，消除精度差  
核心：G_infer ⊆ G_train_fwd → rollout 是训练前向图子集，不引入偏差  
效果：rollout +33%，training +41%，E2E +16%；精度损失 <1%

**Stable Asynchrony / VCPO**（Song Han lab，2/19 提交，ID 未确认）：**适应** off-policy  
ESS-based Learning Rate Scaling：根据有效样本量动态调整 LR  
Closed-form minimum-variance baseline 进一步稳定训练  

**VESPO**（arXiv 2602.10693，2/11）：**纠正** off-policy（算法层面）  
核心洞察：所有 importance weight reshaping（GRPO clip、GSPO 长度归一化）都隐式定义了一个 proposal distribution Q  
变分推导：min KL(Q‖μ) + KL(Q‖π) s.t. E_Q[W] ≤ C → 闭合形式 kernel：  
```
ϕ(W) = W^α · exp(-λW)
```
结果：N=64 staleness 下 avg=58.5%（GRPO 44.7%，SAPO 18.4% collapse）  
全异步训练下唯一稳定的方法  

**四种方法的 staleness 对比（VESPO 论文数据）**：

| 方法 | N=16 staleness | N=64 staleness |
|------|--------------|--------------|
| GRPO | ~57% | ~44.7% |
| SAPO | ~52% | **~18.4%（崩溃）** |
| VCPO (est.) | 稳定 | 稳定 |
| VESPO | ~58% | **~58.5%（稳定）** |

**OAPL**（arXiv 2602.19362，2/25，Cornell+Databricks+Harvard）：**放弃 IS，从理论出发**
核心洞察：与其用 IS 把 off-policy 数据伪装成 on-policy，不如直接从 KL-regularized RL closed-form 解推导一个原生 off-policy 算法。
推导：$\max_\pi E[r] - \beta \text{KL}(\pi \| \pi_{vllm})$ → closed-form $\pi^* \propto \pi_{vllm} \cdot e^{r/\beta}$ → 最优化目标等价 squared regression loss
效果：允许 400 步 policy lag；AIME25/HMMT25/BRUMO25 超越 GRPO+IS；LiveCodeBench v5 用 1/3 生成量匹配 DeepCoder
参见：[[OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] ★★★★★

**四种路径正交，可组合**：
- Jet-RL（系统层：消除来源）
- VCPO（优化层：动态适应 LR）
- VESPO（算法层：软纠正 IS）
- OAPL（算法层：放弃 IS，closed-form squared loss）
- SAPO（Token-level 软衰减，适合 on-policy/近 on-policy 场景）

**共同洞察**：off-policy 是 RL 实现中最隐蔽的 bug。很多工程团队不知道他们的"on-policy"系统实际上已经悄悄变成了 off-policy。量化、异步、重用 rollout、mini-batch 分割都是来源。GSPO 的长度归一化实际上引入了长度偏差（更长的序列更难被 clip → 正反馈 → collapse）。

---

### 维度六：System 层 — 量化与效率

**问题**：大模型 RL 训练的计算瓶颈主要在 rollout 阶段（占 60-80% 时间）。如何在保持 on-policy 假设的前提下加速？

**QeRL**（arXiv 2510.11696，ICLR 2026，Song Han lab 前作）  
路径：NVFP4 权重量化 + LoRA 参数高效微调 + AQN 探索增益  
意外发现：量化噪声在 RL 中是有益的（增加 entropy → 促进探索）  
效果：1.5× rollout 加速，单 H100 可训 32B；性能超越 BF16 LoRA  
限制：需要 Hopper/Blackwell GPU（H100+）；LoRA 限制了表达能力

**Jet-RL**（arXiv 2601.14243，ICLR 2026，Song Han lab）  
路径：统一 FP8 flow（rollout 和 training 同精度）  
不同于 QeRL：全参数而非 LoRA；关注 on-policy 保证而非量化探索增益  
效果：E2E +16%；精度损失 <1%；兼容 VeRL/SLIME/OpenRLHF

**互补关系**：QeRL 解决"单卡资源约束"，Jet-RL 解决"精度一致性"。未来的工作可能是 FP4 + on-policy 统一（QeRL 思路 + Jet-RL 原则）。

---

## 七维总结表（v2）

| 维度 | 核心问题 | 代表论文 | 解法本质 | 关键数字 |
|------|---------|---------|---------|---------|
| **Diversity** | 多条正确路径被压死 | ProGRPO / RePO | 概率置信度重调 advantage / off-policy 知识内化 | Pass@32 +13.9% / hard sample 利用率↑ |
| **Token** | 哪些 token 有毒 | STAPO / MASPO | 基于 token 属性的梯度 mask / adaptive clip | +7.13% / +5% |
| **Exploration** | 探索区域如何扩大 | DEEP-GRPO / QeRL | Pivot 分支采样 / 量化噪声增 entropy | avg +2.6% / reward 2.5× 更快 |
| **Sample** | 哪道题梯度最丰富 | Goldilocks | Teacher LM 预测 utility，选 edge-of-competence | ~15% 数据效率提升 |
| **Trust Region** | clip 该怎么适应 | MASPO / DAPO / VAPO / **SAPO** | Probability-mass adaptive / variance-aware / **sech²软衰减** | +5% |
| **Off-Policy** | 精度/异步引入偏差 | Jet-RL / VCPO | 统一 flow / ESS-based LR scaling | E2E +16% / 稳定性显著提升 |
| **System** | 计算效率 | QeRL / Jet-RL | FP4+LoRA / 统一 FP8 | 单 H100 训 32B / +16% E2E |

---

## 深层统一视角：为什么所有问题都指向同一根因？

表面上，六个维度是独立的。但它们都有一个共同的根因：

> **GRPO 用序列级奖励训练 token 级决策，假设所有 token 均匀可交换（i.i.d.），但实际上它们高度异构。**

- Token 层问题：不同 token 的梯度异构，不该均匀对待
- Exploration 层问题：不同位置的 token 对于探索价值异构（pivot vs 普通 token）
- Sample 层问题：不同难度的题对模型的梯度贡献异构
- Trust Region 问题：不同概率区间的 token 需要异构的 clip 范围
- Off-Policy 问题：rollout 和 evaluation 的 token 分布本应同质，但被各种因素破坏
- System 层问题：rollout 阶段的 token 生成和 training 阶段的梯度计算有异构的资源需求

**真正的解法**：token 级别的密集奖励（dense reward），让每个 token 有自己的 credit assignment。

但这需要一个可靠的 token-level value model——这恰好是 GRPO 为了去掉 critic 而避开的东西。  
**悖论**：GRPO 越改越复杂，最终可能绕回 PPO+critic 的路。

---

## 面试 FAQ

**Q: GRPO 和 PPO 的核心区别是什么？**  
A: GRPO 用 group 内均值替代 critic 的 value estimation，省掉了 critic model，但代价是无法做 token 级别的 credit assignment。所有同组的 token 共享同一个 advantage。

**Q: 为什么 GRPO 会导致 entropy collapse？**  
A: 因为某些 token（STAPO 称之为 spurious tokens）的梯度极大——低概率、低熵、正优势三者叠加，这类 token 的梯度在 clip 之前已经很大，训练会过度优化这类 token，导致其他 token 的概率被压缩，entropy 下降。

**Q: 这些改进里，什么最适合 production 系统？**  
A: 优先级：  
1. **Goldilocks（样本筛选）**：独立于算法，低成本高收益
2. **STAPO（S2T mask）**：单行代码修改，稳定性提升显著
3. **Jet-RL（统一精度）**：如果用 FP8，必须统一，否则有隐性 off-policy
4. **QeRL（量化加速）**：资源受限时首选
5. **DEEP-GRPO（Pivot resampling）**：困难问题上有显著收益，但实现复杂

**Q: 量化在 SFT 和 RL 中效果为什么相反？**  
A: SFT 目标是分布匹配（最小化 KL），量化噪声偏离目标 = 有害。RL 目标是 reward 最大化，量化噪声增加 policy entropy = 促进探索 = 有益。问题结构不同，噪声的效果就不同。

**Q: GRPO 的 off-policy 问题有多严重？**  
A: 比大多数人意识到的严重。任何 rollout 和 evaluation 精度不一致的系统（FP8 rollout + BF16 eval）都有 off-policy 偏差。Jet-RL 实验显示，naive FP8 rollout 的精度损失 >5%，统一 FP8 flow 后降到 <1%。在异步训练中更严重（VCPO 动机）。

---

## 维度八（补充）：Difficulty Debiasing — std 归一化的隐患

> **2026-02-25 补充**：NoRD (CVPR 2026) 在自动驾驶 VLA 领域首次实证了 difficulty bias 对弱 SFT 策略的系统性破坏，并验证了 Dr. GRPO 的有效性。

**问题**：GRPO 的 advantage 计算中 `std(r)` 归一化引入了隐性的难度偏差：
- 高方差 group（中等难度样本）→ std 大 → advantage 被压缩 → **有效梯度几乎为零**
- 低方差 group（极简单或极难样本）→ std 小 → advantage 被放大 → **梯度集中在无价值的样本上**

后果：GRPO 实际只从"简单样本"（全对 or 全错）中学习，忽视了能力边界处最有价值的"中等难度样本"。

**Dr. GRPO**（Liu et al., 2029，原发表于 LLM 数学推理）  
解法极其简单：**去掉 std 归一化**。  
`A_i = r(o_i|x) - mean_j(r(o_j|x))` （就这一行改动）  
加上 DAPO 风格非对称 clipping + 无 KL 正则，保持训练稳定。

**NoRD (2602.21172, CVPR 2026) — 跨域实证验证**  
- 任务：自动驾驶 VLA（Qwen-2.5VL-3B，Waymo/NAVSIM benchmark）  
- 发现：弱 SFT 策略（80k 样本，无推理标注）在 PDM score [0.2, 0.65] 范围内产生高 intra-group variance 的 majority 样本  
- GRPO：+0.67% PDM（无效）；Dr. GRPO：**+11.68% PDM**  
- 结论：difficulty bias **不只是 LLM 推理问题**，而是任何 reward 分布极化 + 弱 SFT 策略组合的普遍现象

**适用判断**：当出现以下情况时，Dr. GRPO > GRPO：
1. SFT 策略相对弱（数据少、无推理标注、cold start）
2. Reward 分布极化（bimodal：简单全对 + 困难全错）
3. 中等难度样本占多数（majority 落在高方差区间）

**与 Goldilocks/Sample 层的关系**：  
Goldilocks = 筛选掉 std=0 的极端样本（数据层）  
Dr. GRPO = 对 std 大的样本不惩罚（算法层）  
两者互补：一个前置过滤，一个后置不歧视。

---

## 开放问题

1. **Token 级别密集奖励**：能否设计轻量级的 per-token reward model，而不需要完整的 critic？
2. **Exploration 组合效应**：DEEP-GRPO（pivot resampling）+ QeRL（entropy injection）能否叠加而不相互干扰？
3. **Goldilocks 的 online 版本**：Teacher LM 预测 utility 是 offline 的，能否做到 online adaptive curriculum？
4. **系统层和算法层的 co-design**：Jet-RL 和 VCPO 解决了系统层 off-policy，但是否有算法层能够容忍一定程度的 off-policy？（off-policy RL 算法，如 V-trace/IMPALA）
5. **边界扩展：non-verifiable 任务**：RLRR（arXiv:2602.16802，ICLR 2026）用 reference-guided judge 为对齐任务造了软 verifier，把 RLVR 的能力边界向 non-verifiable 域推进。七维框架目前假设 verifiable reward 存在——non-verifiable 场景下，软 verifier 的误差率（~21%）如何影响这七个维度的改进效果？See: [[RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]]

---

## 引用论文

- ProGRPO: arXiv 2602.05281 (Pengyi Li et al.) — Diversity 维度，ARM 概率置信度重加权
- RePO: arXiv 2602.10819 (Linxuan Xia et al.) — Diversity 维度，off-policy 知识内化
- STAPO: arXiv 2602.15620 (Tsinghua + DiDi)
- MASPO: arXiv 2602.17xxx (MSRA, Xiaoliang Fu/Xunliang Cai)
- DEEP-GRPO: arXiv 2602.14169 (ICML submission)
- Goldilocks RL: arXiv 2602.14868 (Apple + EPFL)
- Jet-RL: arXiv 2601.14243 (Song Han lab, NVIDIA/MIT)
- QeRL: arXiv 2510.11696 (Song Han lab, NVIDIA/MIT, ICLR 2026)
- Stable Asynchrony / VCPO: ~arXiv 2602.1xxxx (Song Han lab, 2/19 提交，ID 待确认)
- VESPO: arXiv 2602.10693 (变分 IS reshaping，off-policy 理论最严格)
- SAPO: arXiv 2511.20347 (Qwen 团队，sech² 软门控，Qwen3-VL 生产使用)
- GSPO: arXiv 2507.18071 (Qwen 团队，sequence-level IS ratio，SAPO 前驱)
- AT-RL: arXiv 2602.11455 (多模态视觉锚点 credit assignment)
- Dr. GRPO: (Liu et al., 2029，原文见 GRPO/Dr-GRPO-Unbiased-Optimization.md) — difficulty bias 去 std 归一化
- NoRD: arXiv 2602.21172 (Applied Intuition + UC Berkeley, CVPR 2026) — 自动驾驶 VLA，首次在非 LLM 推理领域验证 Dr. GRPO，弱 SFT + Dr. GRPO 无推理数据达到 SOTA

---

## see-also

- [[STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO]] — Token 级别维度
- [[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 多维 GRPO 改进
- [[DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — 探索维度
- [[Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] — 样本维度
- [[Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 系统精度维度
- [[QeRL-Quantization-Enhanced-RL|QeRL]] — 量化探索维度
- [[Stable-Asynchrony-VCPO-Off-Policy-RL|VCPO]] — 系统异步 off-policy 维度
- [[VESPO-Variational-Sequence-Policy-Optimization|VESPO]] — 变分 off-policy 修正，理论最严格
- [[SAPO-Soft-Adaptive-Policy-Optimization|SAPO]] — sech² 软门控，Qwen3-VL 生产
- [[GSPO-Group-Sequence-Policy-Optimization|GSPO]] — 序列级 IS ratio（SAPO 前驱）
- [[Dr-GRPO-Unbiased-Optimization|Dr. GRPO]] — 去 std 归一化，difficulty debiasing
- [[NoRD-Dr-GRPO-Reasoning-Free-VLA-Autonomous-Driving|NoRD]] — 自动驾驶 VLA，Dr. GRPO 跨域实证 (CVPR 2026)
- [[AT-RL-Anchor-Token-Reinforcement-Learning-Multimodal|AT-RL]] — 多模态维度 credit assignment
- [[RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性 2026 统一分析]] — 与本文互补，聚焦稳定性而非分类框架
- [[OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] — 目标函数范式转移：KL-regularized closed-form → squared regression，放弃 IS
- [[LAD-Learning-Advantage-Distribution|LAD]] — 目标函数范式转移：advantage 诱导分布匹配（f-divergence），自然保留多模式轨迹；与 OAPL 正交可组合
- [[HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER（ICML 2026）]] — GRPO 的 Agent 扩展方向：StarPO → multi-turn trajectory-level GRPO → HiPER 的 segment-level HAE（三步演进）
- [[RAGEN-StarPO-Multi-Turn-RL-Self-Evolution|RAGEN & StarPO]] — GRPO 在 multi-turn agent 场景的稳定性挑战（Echo Trap），及 StarPO 框架
- [[ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]] — Diversity 维度：ARM 概率信号重调 advantage
- [[RePO-Rephrasing-Policy-Optimization|RePO]] — Diversity 维度：off-policy 知识内化到 on-policy 兼容轨迹
- [[SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees|SeeUPO（arXiv:2602.06554）]] ⚠️ — **GRPO 的理论边界**：不可能定理证明 GRAE+PPU（GRPO主体）在 multi-turn contextual bandit 中无收敛保证；单轮推理 GRPO 仍有效，多轮 Agent 训练需换 SeeUPO 逆序更新

---

## 🔧 落地应用

- **单轮推理 RLVR（数学/代码）**：GRPO 是标准起点 → 遇到 entropy collapse 加 DAPO 的 clip-higher + entropy bonus → trust region 过硬加 MASPO/SAPO 软化 → 探索死加 DEEP-GRPO pivot resampling → 课程学习加 Goldilocks RL 选 p≈0.5 题
- **大规模异步训练（70B+）**：generation/training 解耦后用 VCPO 方差控制 + Jet-RL FP8 精度一致性，防 off-policy drift
- **多轮 Agent 训练**：⚠️ 不要用 GRPO！variance normalization 破坏 multi-turn 收敛（SeeUPO 定理）→ 切换 SeeUPO 逆序更新
- **选哪个改进**：只有一个问题用 DAPO（生产验证最充分）；想统一修三个问题用 MASPO；生产环境用 Qwen 团队 SAPO/GSPO（Qwen3-VL 在跑）
- **面试话术**：七维框架是回答"GRPO 有什么问题/怎么改进"的标准结构，先分维度再说具体论文，比直接背论文名字清晰10倍

## 💡 启发与思考

- **"改进 GRPO"的论文已经多到需要元分析**：每篇都声称解决了"GRPO 的问题"，但实际上修的是不同层次的缺陷——七维框架的价值就在于此，让你一眼看清每篇论文的真正贡献在哪个维度
- **SeeUPO 定理改变了讨论框架**：之前说 GRPO 在 multi-turn 不稳定是经验观察，现在有了数学证明。这意味着七维框架的所有改进都是针对**单轮 GRPO**的——multi-turn 场景需要完全不同的算法族
- **Qwen 团队生产验证的算法是最值得相信的**：SAPO（sech² 软门控，Qwen3-VL）/ GSPO（序列级 IS，Qwen3）都是被大规模生产训练验证过的——学术论文好但没有工业级规模验证的算法，在实际部署时要谨慎
- **Diversity 维度是最被低估的**：大家都在改 trust region / KL 约束，但 rollout 多样性崩塌（within-state 探索死亡）才是 hard exploration 的根因。VAM/QeRL/IntroLLM 这条线在 2026 下半年可能会爆发

## 📚 推荐阅读

1. **DAPO**（arXiv:2503.14476）— GRPO 最重要的工业级改进，四项改进都有充分的 ablation，NeurIPS 2025
2. **MASPO**（arXiv:2602.17550）— 三维统一改进框架，是理解"trust region 改进全家桶"的好教材
3. **SAPO**（arXiv:2511.20347）— Qwen 团队，sech² 软门控，生产验证；与 GSPO 一起读理解 Qwen RL 栈
4. **SeeUPO**（arXiv:2602.06554）— 必读：理解 GRPO 的理论边界（multi-turn 无收敛保证），正确定位本全景的适用范围
