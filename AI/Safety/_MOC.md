---
title: "AI 安全"
type: moc
domain: ai/safety
tags:
  - ai/safety
  - type/reference
---

# 🛡️ AI 安全

> AI 安全从训练数据到自主 Agent 的完整威胁面与防御体系

## 威胁分析

- [[Agent 安全威胁全景 2026-02]] — 2026-02 Agent 安全 threat landscape 全景（prompt injection / 记忆投毒 / Agent 武器化 / OpenClaw 安全）
- [[Anthropic Claude Opus 4.6 蓄意破坏风险报告]] — Opus 4.6 蓄意破坏风险评估，ASL-3→4 灰区
- [[AI/Safety/AI Agent 集体行为与安全漂移|AI Agent 集体行为与安全漂移]] — 多 Agent 系统安全漂移
- [[AI/Agent/Papers/Collective-Behaviour-Hundreds-LLM-Agents-2026|Collective Behaviour（100+ Agents）]] — 更强模型→更差社会结果；文化进化→差均衡收敛；算法编码策略支持静态审计（arXiv:2602.16662）★★★★☆
- [[AI/Agent/Papers/Colosseum-Multi-Agent-Collusion-Audit-2026|Colosseum（勾结审计框架）]] — "纸上勾结"新现象；DCOP 形式化 + regret 指标；网络拓扑影响勾结倾向（arXiv:2602.15198）★★★★☆

## 真实攻击案例 & 能力基准

- [[AI/Safety/AutoInject-RL-Prompt-Injection-Attack|AutoInject（RL自动化Prompt Injection）]] ⭐ — ICML 2026：1.5B攻击模型用GRPO训练，77%+ ASR破Gemini-2.5-Flash；universal suffix可迁移通杀70个任务——盾卫项目必读（arXiv:2602.05746）★★★★★
- [[AI/Safety/Clinejection-AI-Coding-Agent-Supply-Chain-Attack|Clinejection]] ⭐ — 2026-02-17 真实事件：Cline AI triage bot 被 prompt injection 劫持，恶意代码推送给 5M+ 开发者（★★★★★）
- [[AI/Safety/EVMbench-AI-Agent-Smart-Contract-Exploit|EVMbench]] ⭐ — OpenAI+Paradigm：AI agent 自主利用智能合约漏洞；GPT-5.3-Codex 利用率 <20%→>70%（★★★★★）
- [[AI/Safety/OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]] ⭐ — Oxford/Torr+Gal lab，ICML 2026：单次 injection 通过 orchestrator 链污染整个多 Agent 网络；4/5 frontier 模型可被攻陷，15,000次实验；Claude Sonnet 4 唯一幸存（arXiv:2602.13477）★★★★★
- [[AI/Safety/AgentLeak-Full-Stack-Privacy-Leakage-Multi-Agent-Benchmark|AgentLeak]] — Polytechnique Montréal：第一个全栈 multi-agent 隐私泄漏 benchmark；inter-agent 内部通信泄漏率 68.8%（比用户输出高2.5×）；output-only 审计漏掉41.7%违规（arXiv:2602.11510）★★★★☆

## 基础知识

- [[AI 安全及隐私保护]] — 安全威胁分类、防御实践、隐私保护技术
- [[AI 伦理和治理]] — 偏见与公平、透明性、治理框架

## 对齐与防御

- [[AI/Safety/安全对齐与红队|安全对齐与红队]] — Jailbreak 分类、红队流程、防御策略、Anthropic/OpenAI 安全报告
- [[AI/Safety/对齐技术总结|对齐技术总结]] — 对齐技术综述：RLHF/DPO/Constitutional AI/Scalable Oversight 完整覆盖
- [[AI/Safety/Adaptive-Regularization-Safety-Degradation-Finetuning|Adaptive Regularization]] ⭐ — Harmful intent 在 pre-generation hidden state 中线性可分（AUROC>0.9）；safety critic 动态调整 KL 强度，ASR 从97%压回1-9%（arXiv:2602.17546）★★★★
- [[AI/LLM/RL/Theory/Rationale-Consistency-GenRM-Deceptive-Alignment|MetaJudge（GenRM欺骗性对齐）]] ⭐ — Reward Model 层的 deceptive alignment：outcome accuracy 无法区分正确推理 vs 表面猜对；RC 指标 + 乘法门控 R_final=R_rationale×R_outcome；RM-Bench SOTA 87.1%；o3 RC≈0.4，mini 模型系统性落入红区（arXiv:2602.04649）★★★★★

## 关联

- [[AI/Agent/Agent-Economy/_MOC|Agent 经济]] — Agent 经济安全风险（资金转移 / 身份冒用）
- [[AI/LLM/Application/Prompt-攻击|prompt 攻击]] — Prompt Injection 技术细节

## 综述与全景
- [[AI/Safety/AI安全与对齐-2026技术全景|AI 安全与对齐 2026 全景]] ⭐ — 面试武器库 #16，1130行：对齐技术栈→可扩展监督→红队测试→Agent安全→机械可解释性→模型评估→治理框架（2026-02-20）★★★★★

## 魂匣认知哲学（SoulBox 护城河研究）
- [[AI/Safety/SoulBox-认知哲学研究索引|SoulBox 认知哲学研究索引]] ⭐ — Vault 与 soulbox/research/ 的桥接入口；个人同一性（Locke/Hume/Parfit/王阳明）→ Agent 记忆继承的哲学论证；人格科学（Big Five/OCEAN/叙事认同）→ Agent 人格设计原则；人格漂移工程；魂匣伦理红线（2026-02-21，馆长）★★★★★

## 相关 MOC
- ↑ 上级：[[AI/_MOC]]
