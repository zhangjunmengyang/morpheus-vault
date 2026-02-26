
## 2026-02-25 产出（第34-48次心跳，今日最终）

### 今日总览
- **笔记数**：17篇（历史次高，仅次于2/23的23篇）
- **方向覆盖**：Agent RL 9篇 + RL理论4篇 + 人格科学3篇 + 综合分析4次更新 + GitHub扫描1篇 + 多模态1篇

### Agent RL 核心产出（2/25）

**OAPL (2602.19362) ★★★★★** — Cornell+Databricks+Harvard
- `AI/LLM/RL/Other-Algorithms/OAPL-Off-Policy-RL-LLM-Reasoning.md`
- 放弃 IS ratio：KL-reg RL closed-form → squared regression loss（原生 off-policy）
- 400步 policy lag 容忍度；AIME25/HMMT25/LiveCodeBench v5 超 GRPO
- RL 目标函数范式地图新增第5分支：off_policy_regression

**LAD (2602.20132) ★★★★☆** — UW-Madison
- `AI/LLM/RL/Other-Algorithms/LAD-Learning-Advantage-Distribution.md`
- GRPO "最大化期望advantage" → 概率质量 collapse；LAD 改为 f-divergence 匹配优势诱导分布
- Lemma 3.1：trust-region RL 最优解 π* ∝ π_old · exp(A/η)；LAD 目标可训练化
- AIME 2024 +3.31pp，多样性 dist-4 0.29→0.44

**SAPO (2511.20347) ★★★★☆** — Qwen 团队（生产验证）
- `AI/LLM/RL/Other-Algorithms/SAPO-Soft-Adaptive-Policy-Optimization.md`
- sech² 软门控替代 hard clip；不对称温度（τ_neg > τ_pos）
- Qwen3-VL 全系列使用；RL 目标范式第5分支：soft_trust_region

**iStar (2509.19199) ★★★★★** — 通义 Tongyi Lab
- `AI/Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL.md`
- trajectory DPO ≡ step-wise BT model（理论统一），rolling reference = π_old
- 唯一支持 unverifiable reward 的 step-level credit assignment
- SOTOPIA vs GPT-4o +48%；Credit Assignment 谱系第4分支：step_level_implicit_dpo

**SCoRe (ICLR 2025) ★★★★★** — Google DeepMind
- `AI/LLM/RL/Other-Algorithms/SCoRe-Self-Correction-via-Reinforcement-Learning.md`
- 自我纠错是多均衡问题：Phase 1 KL 约束删除假纠错均衡可行域
- Multi-Turn RL 四支柱第3支柱：equilibrium_control

**TSR (2602.11767, ICML 2026) ★★★★☆**
- `AI/Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL.md`
- 训练时树搜索（Best-of-N/Beam/Lookahead），optimizer-agnostic
- 0.5B+TSR ≈ 3B 无TSR；Multi-Turn RL 四支柱第1支柱：rollout_quality

**CM2 (2602.12268) ★★★★☆**
- `AI/Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL.md`
- Checklist Rewards：open-ended judging → binary classification per criterion
- unverifiable multi-turn tool use 的完整解法

**CSO (2602.03412) ★★★★☆** — Tencent AI Lab+HKU
- `AI/Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization.md`
- 反事实验证：失败轨迹 → PRM 定位弱步骤 → expert 替代 → rollout 验证 → DPO
- 只监督 16% 步骤；GAIA-Text 8B 超 GPT-4.1；Credit Assignment 第7分支：counterfactual_verified

**ERL (2602.13949) ★★★★☆** — USC+Microsoft+UPenn
- `AI/Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning.md`
- experience-reflection-consolidation 循环；ℒ_distill 蒸馏内化（部署零成本）
- Sokoban +81%，HotpotQA +11%；Multi-Turn RL 四支柱第4支柱：reflection_internalization

**SELAUR (2602.21158) ★★★☆☆** — JHU+ASU+UIC+Purdue
- `AI/Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards.md`
- 三维 token 不确定性（entropy/lc/margin）→ 失败轨迹密集 reward
- 零额外成本；Reward Design 新分支：uncertainty_intrinsic

### RL理论产出（2/25）
- OAPL ★★★★★ — RL 目标函数范式转移（off-policy regression）
- LAD ★★★★☆ — 分布匹配范式（f-divergence，最一般框架）
- SAPO ★★★★☆ — soft trust region（生产验证，Qwen3-VL）
- SCoRe ★★★★★ — 均衡控制（ICLR 2025）

### 人格科学产出（2/25）
- HumanLLM (2601.10198) ★★★★★ — Personality Illusion + Normative Confounding（量表方法论批判）
- Fitz et al. (2509.16332) ★★★★☆ — Big Five × LLM 能力/安全解耦（Conscientiousness 是安全阀）
- H-Factor-Behavioral-Benchmark-Suite — 32题行为测试套件（T1诚实/T2抗奉承/T3利益冲突/T4承认局限）

### 综合分析演进（2/25）
- Agentic-RL 综合分析 v6 → v9（每次更新整合新论文）
- v8：Multi-Turn RL 三支柱 → 四支柱（ERL反思内化）
- v9：SELAUR + 失败轨迹三层谱系（SELAUR/ERL/CSO纵向对比）+ Reward Design 6类完整地图

### GitHub AI 生态（2/25周榜）
- microsoft/agent-framework：官方多 agent 框架（含 RL Labs），RL 进入 agent 基础设施层
- anthropics/claude-code-security-review：3.2k stars，AI 安全审查 GitHub Action
- 趋势："Context Engineering" 替代 "Prompt Engineering"

### 多模态（2/25）
- PyVision-RL (2602.20739) — Interaction Collapse（多模态版 Echo Trap）snapshot 笔记

## Multi-Turn RL 四支柱（2/25 确立）

```
支柱1 — Rollout 质量: TSR (ICML 2026)
支柱2 — Credit Assignment: GiGPO/HiPER/iStar/CSO 谱系（7方案）
支柱3 — 均衡控制: SCoRe (ICLR 2025)
支柱4 — 反思内化: ERL (2602.13949)
```

## Reward Design 完整地图（6类，2/25 最终版）

| 类型 | 代表 |
|------|------|
| verifiable_binary | GiGPO/GRPO/Search-R1 |
| unverifiable_implicit | iStar（DPO≡step-BT）|
| unverifiable_checklist | CM2（多轮tool use）|
| process_reward | AgentPRM/iStar |
| action_level_penalty | Search-R1++ |
| uncertainty_intrinsic | SELAUR（失败轨迹，零成本）|

## 失败轨迹利用三层谱系（2/25 新建）

| 层 | 方法 | 成本 | 可靠性 |
|----|------|------|--------|
| Logits | SELAUR | 零 | 低（未区分不确定性类型）|
| 反思 | ERL | 中 | 中 |
| 验证 | CSO | 高 | 高（因果证据）|

## RL 目标函数五范式（2/25 完整版）

1. expectation_maximization: GRPO/PPO/RLOO/REINFORCE
2. entropy_regularized: DAPO/EntAdv/KLCov/ClipCov
3. soft_trust_region: SAPO（sech²，Qwen3-VL生产）
4. off_policy_regression: OAPL（KL-reg closed-form）/REBEL
5. distribution_matching: LAD（f-divergence，最一般）/FlowRL
