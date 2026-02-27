---
title: "PA-MoE: Phase-Aware Mixture of Experts for Agentic RL"
brief: "PA-MoE：在 GiGPO 基础上加 Phase-Aware MoE，用不同 LoRA Expert 处理不同 Agent 任务阶段；解决 Simplicity Bias（模型总走最短路径）；ALFWorld +7.7%、WebShop +14.9%（在 GiGPO 基础上）（arXiv:2602.17038）"
type: paper
domain: ai/agent/agentic-rl
created: 2026-02-20
updated: 2026-02-22
tags:
  - Agentic-RL
  - MoE
  - LoRA
  - credit-assignment
  - routing-granularity
  - simplicity-bias
status: v1
---

# PA-MoE: Phase-Aware Mixture of Experts for Agentic RL

**论文**: Phase-Aware Mixture of Experts for Agentic Reinforcement Learning  
**arXiv**: 2602.17038 (cs.AI)  
**提交**: 2026-02-19  
**作者**: Shengtian Yang, Yu Li, Shuo He, Yewen Li, Qingpeng Cai, Peng Jiang, Lei Feng  
**机构**: 未明确披露（cs.AI，16 pages）  
**代码**: https://anonymous.4open.science/r/PA-MoE-576C/  
**评分**: ★★★★☆

---

## 核心问题：Simplicity Bias in Agentic RL

现有 Agentic RL 方法（RLOO、GRPO、GiGPO）都用**单一 policy network** 处理所有行为阶段。这导致：

> **简单任务占主导**：pick_and_place 等简单任务占 **75% 参数容量**，而 heat/cool/clean 等需要多步工具交互的复杂任务仅获得 **5%** 的参数容量。

具体机制：
- 简单任务更频繁 → 更大梯度贡献 → 参数偏向简单行为
- 复杂任务梯度被 overshadow → 难以获得充足的表征容量
- 即使 GiGPO 用分层 relative advantages，底层单一网络架构仍有此问题

这是 **架构层面**的缺陷，梯度手术（PCGrad/CAGrad）只是优化级 patch，不能产生 persistent 的参数专化。

---

## 直觉：用 MoE 隔离行为阶段

**为什么 MoE？**  
MoE 允许不同参数（experts）专注不同任务，阻止简单任务垄断所有参数。这与 simplicity bias 的解决需求天然契合。

**为什么不直接用标准 MoE？**  
标准 MoE 的路由粒度是 **token-level**，但 Agentic RL 的决策粒度是 **environment step**。

| 路由粒度 | 切换频率（每 episode） | 问题 |
|----------|----------------------|------|
| Token-level | ~1200 token switches ≈ **45 step-level 等效切换** | 过度碎片化：同一个 "open fridge" 动作内不同 token 路由到不同 expert |
| Trajectory-level | **3 switches** | 过于粗糙：episode 内行为变化无法自适应 |
| **Phase-level (PA-MoE)** | **8.4 switches** | ✅ 平衡：在真实的语义 phase 边界切换 |

**核心洞察**：Agent 轨迹的语义结构不在 token 层，而在 environment-step 层，phase-level 路由才能正确对齐。

---

## 方法：PA-MoE 架构

### 整体结构

```
Agent Episode
  → Phase-Aware Router (轻量级，每 step 预测)
    → 选择 expert k*
      → LoRA Expert k* (on 冻结 backbone)
        → 生成 action a_t
```

每个 Expert = LoRA module（极少参数开销，共享冻结 base 模型）。

### Phase-Aware Router

路由器 π_r 整合两路信息：

**路径 1：Goal-Conditioned Observation Encoding**
```
o_t^align = CrossAttn(Q=Enc(o_t), K=Enc(g), V=Enc(g))
```
- Enc(·) = 冻结 base model 最终层 hidden states 的 mean-pooling
- CrossAttn 让路由器关注与目标相关的 observation 特征
- 区分"外观相似但需要不同行为模式"的状态

**路径 2：Temporal History Modeling**
```
h_t^enc = LSTM(Embed(h_t)) ∈ R^d
```
- h_t = [a_{t-L:t-1}, o_{t-L:t-1}]，L=5（历史窗口），d=256（LSTM 隐层）
- 3 层 LSTM，捕捉 phase transition 的时序信号
- LSTM 的 recurrent 结构天然适合追踪 long-horizon 中的状态变化

**Expert 概率分布**
```
p_t = softmax(MLP([o_t^align; h_t^enc]) / τ) ∈ Δ^K
k* = argmax_k p_t^k  （确定性选择，训练和推理均用 argmax）
```

### 时间一致性机制

**切换惩罚 (Switching Penalty)**
```
L_switch = (λ_s / (T-1)) * Σ_{t=1}^{T-1} 1[z_t ≠ z_{t+1}]
```
- λ_s = 0.05
- 注意：indicator 函数不可微，backward 用 soft surrogate：
  ```
  1[z_t ≠ z_{t+1}] → 1 - Σ_k p_t^k · p_{t+1}^k
  ```
  前向用 hard indicator（精确计算），后向用 soft disagreement（传播梯度）

**温度退火 (Temperature Annealing)**
```
τ(t) = max(τ_f, τ_0 - (τ_0 - τ_f) · t / T_anneal)
```
- τ_0 = 2.0（初始高温，鼓励探索不同路由模式）
- τ_f = 0.5（最终低温，产生更确定的分配）
- T_anneal = 3000 steps

效果：switches 从 45/episode → **8.4/episode**，保留在真实 phase 边界切换的灵活性。

### 训练目标

```
L_total = L_RL + L_div + L_bal + L_switch
```
- **L_RL**: RL 目标（与底层 RL 算法无关，PA-MoE agnostic to RL algorithm）
- **L_div**: 多样性损失（鼓励不同 expert 行为差异）
- **L_bal**: 负载均衡损失（保证 expert 利用率均衡，约 ~30% per expert）
- **L_switch**: 时间一致性惩罚（见上）

---

## 实验结果

### Benchmark

- **ALFWorld**：embodied navigation（具身导航）
- **WebShop**：web 交互（电商购物）

### 主要结果 vs GiGPO baseline

| 数据集 | GiGPO (7B) | PA-MoE (1.5B) | PA-MoE (7B) |
|--------|-----------|---------------|-------------|
| ALFWorld | baseline | **超过 7B baseline** | +7.7% |
| WebShop | baseline | — | +14.9% |

**关键数据**：
- PA-MoE 1.5B > GiGPO 7B → 参数效率大幅提升
- 均匀任务难度覆盖：pick_and_place 96% / heat/cool/clean 92% / complex 98%（PA-MoE）vs 单一网络的严重倾斜

### 参数分配对比

| 方法 | pick_and_place | 复杂多步任务 |
|------|---------------|-------------|
| 单一网络 | 75% 参数 | 5% 参数 |
| PA-MoE | ~30% per expert（均衡） | ~30% per expert |

---

## 批判性分析

### 亮点

1. **问题定义精准**：Simplicity Bias 是 Agentic RL 的真实痛点，用实验数据（75% vs 5%）具体量化，有说服力
2. **粒度分析清晰**：Token(45) → Phase(8) → Trajectory(3) 的对比给出了直觉，routing granularity 对 Agentic 任务的重要性首次被明确讨论
3. **可微分 surrogate 设计精巧**：switching penalty 的 forward/backward 不对称是训练 stable discrete routing 的工程技巧
4. **1.5B > 7B 的结果强**：说明架构创新确实解决了 simplicity bias，而不只是参数量的问题

### 疑问与局限

1. **Phase 边界的可解释性**：Router 学到的 phase boundary 是什么？论文提到"planning, acting, verifying"，但没有定量分析 expert 的语义专化程度
2. **K 的选择**：Expert 数量 K 怎么设置？ablation 是否测试了 K 的影响？（论文未明确）
3. **Router overhead**：Cross-attention + LSTM 的计算开销是否影响 throughput？对比没有明确
4. **与 HiPER 的关系**：HiPER 用层级 credit assignment，PA-MoE 用层级参数隔离；两者是否可叠加？

### 我的判断

这是一篇**想法清晰、实验扎实**的 Agentic RL 架构论文。Simplicity Bias 问题是真实的，routing granularity 分析有新意。

1.5B 打 7B 的结果说明问题是**架构缺陷**而非规模问题——这一点对 Agentic RL 的实践有直接影响。

局限：目前只在 ALFWorld/WebShop 测试，在更复杂的 long-horizon agentic 任务（SWE-bench/OSWorld）上的表现待验证。

---

## 与 Vault 其他论文的关系

| 论文 | 与 PA-MoE 的关系 |
|------|----------------|
| **HiPER** | 同为 Agentic RL 信用分配改进，但 HiPER 从 reward 层做分层，PA-MoE 从参数层做隔离；理论上可叠加 |
| **GiGPO** | PA-MoE 的主要 baseline，GiGPO 解决了 reward 分配问题但未解决 simplicity bias |
| **GRPO Panorama** | PA-MoE 可理解为 GRPO 系的**架构维度**改进：参数层面的隔离 vs 算法层面的优化 |
| **Goldilocks** | 都关注任务难度异质性，Goldilocks 从采样层解决，PA-MoE 从参数层解决 |

---

## 启发

- **RL 路由粒度问题**：token/step/trajectory 的粒度选择对 Agentic RL 是第一原则问题，没人明确讨论过，PA-MoE 开了一个好头
- **Hybrid 可能**：Phase-level LoRA expert + VESPO 的 variational IS + HiPER 的层级 credit assignment — 三者在不同层（参数/采样/奖励）各自解决一个独立问题，理论上可叠加
- **魂匣人格角度**：不同行为阶段用不同 expert → **人格模块化**？探索阶段/执行阶段/验证阶段分离参数……有启发

---

## 元数据

- **Tags**: #Agentic-RL #MoE #LoRA #credit-assignment #routing-granularity
- **关联笔记**: [[AI/2-Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment]] [[AI/3-LLM/RL/算法/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks-RL-Task-Difficulty-Curriculum]] [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO-Improvement-Panorama-2026]]
- **写于**: 2026-02-20
