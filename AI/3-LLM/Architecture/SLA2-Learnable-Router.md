---
title: "SLA2: Learnable Router for Sparse-Linear Attention"
brief: "（arXiv:2602.12675）在 Sparse-Linear Attention 框架中引入可学习路由器，动态决定每次 attention 走 sparse 还是 linear 分支；在视频 diffusion 模型上实现 97% 稀疏度、18.6× attention 加速，与原始 SLA 相比质量无损。"
date: 2026-02-20
tags: [论文, 架构, 注意力, 效率, diffusion]
source: "arXiv:2602.12675"
---

# SLA2: Sparse-Linear Attention with Learnable Routing

> **一句话**: 用可学习路由器动态决定每次 attention 走 sparse 还是 linear 分支，在视频 diffusion 模型上实现 97% 稀疏度和 18.6× attention 加速。

## 背景

SLA (Sparse-Linear Attention) 的原始想法：attention map P 可以分解为高稀疏部分 P₁ + 低秩部分 P₂。sparse attention 处理 P₁，linear attention 处理 P₂。

**SLA 的两个问题**：
1. **输出不匹配**: sparse attention map Pₛ 和理论分解的 P₁ 差了一个缩放因子 α，额外的投影层难以完全弥补
2. **启发式路由**: 按 attention weight 大小分配分支，不是最优——暴力搜索表明存在更好的划分方式

## SLA2 的三个改进

### (I) Learnable Router
- 训练一个 sparse-attention mask predictor R，支持梯度反传
- 目标函数：最小化 sparse+linear 组合与 full attention 之间的近似误差
- 不再依赖 attention weight 大小的启发式规则

### (II) 直接匹配的 Sparse-Linear 公式
- 直接学习比例因子 α 来组合 sparse 和 linear 分支
- 公式与原始 sparse+linear 分解严格对齐
- 不需要额外的投影层来弥补不匹配

### (III) Sparse + Low-bit Attention (QAT)
- 在 sparse attention 之上引入低比特量化
- Quantization-Aware Fine-tuning 减少量化误差
- 进一步加速 attention 计算

## 关键结果

| 指标 | SLA2 |
|------|------|
| Attention 稀疏度 | **97%** |
| Attention 加速 | **18.6×** |
| 生成质量 | 保持不变 |
| 应用场景 | 视频 Diffusion 模型 |

## 与 MiniCPM-SALA 的对比

| | MiniCPM-SALA | SLA2 |
|--|-------------|------|
| 应用 | LLM 长上下文 | Diffusion 视频生成 |
| 路由方式 | 层级静态分配 (1:3) | token 级动态学习 |
| 灵活度 | 固定比例 | 自适应 |
| 训练复杂度 | 低（迁移训练） | 高（需要 QAT） |
| 稀疏度 | ~75% linear | 97% sparse |

**共同趋势**: 都在从"全用 full attention"走向"智能混合 attention"。区别在于粒度——SALA 在层级决定，SLA2 在 token 级决定。

## 面试价值

**Q: Sparse attention 和 linear attention 如何混合使用？有哪些方案？**

要点：
- 层级静态混合 (MiniCPM-SALA): 选关键层用 sparse，其余 linear，1:3 比例
- Token 级动态路由 (SLA2): learnable router 每次计算决定走哪条路
- 核心洞察：attention map = 高稀疏分量 + 低秩分量，可以分别用最适合的机制处理
- 静态方案简单高效适合推理部署，动态方案灵活精确但训练成本高

---

*Source: arXiv:2602.12675, UC Berkeley + Tsinghua, 2026-02*

---

## See Also

- [[MiniCPM-SALA|MiniCPM SALA]] — 同期 MiniCPM 架构论文，互补
- [[Attention 变体综述|Attention 变体综述]] — SLA2 在 Attention 变体谱系中的位置
- [[Transformer|Transformer 通识]] — Learnable Router 的架构基础
- [[AI/3-LLM/目录|LLM MOC]] — 大语言模型知识全图谱
