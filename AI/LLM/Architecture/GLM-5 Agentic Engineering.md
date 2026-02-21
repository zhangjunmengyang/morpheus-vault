---
title: "GLM-5: 从 Vibe Coding 到 Agentic Engineering"
date: 2026-02-14
tags: [llm, open-source, glm, agentic]
type: note
---

# GLM-5: 从 Vibe Coding 到 Agentic Engineering

> 智谱 AI（Z.AI）于 2026 年 2 月 11 日发布 GLM-5，定位为面向复杂系统工程和长周期 Agent 任务的开源旗舰大模型。MIT 协议开源。

## 基本信息

| 项目 | 数值 |
|------|------|
| 总参数量 | 744B |
| 激活参数量 | 40B（每次推理） |
| 架构 | Sparse MoE（256 experts，每 token 激活 8 个） |
| 预训练数据 | 28.5T tokens（较 GLM-4.5 的 23T 增长 23.9%） |
| 上下文窗口 | 200K tokens |
| 最大输出 | 131K tokens |
| 注意力机制 | DeepSeek Sparse Attention (DSA) |
| 训练硬件 | 华为昇腾 910 系列（完全国产） |
| 训练框架 | MindSpore |
| 许可证 | MIT |
| 代号 | "Pony Alpha"（2 月初在 OpenRouter 上匿名测试） |

## 模型架构特点

### Sparse MoE 扩展
- 从 GLM-4.5 的 355B/32B active 扩展到 744B/40B active，参数规模翻倍
- 稀疏率约 5.9%，在保持推理效率的同时大幅提升模型能力
- 256 个专家中每 token 仅激活 8 个，计算开销可控

### DeepSeek Sparse Attention (DSA)
- 采用 DeepSeek 发明的稀疏注意力机制
- 两阶段处理：lightning indexer + token selector
- 将注意力计算复杂度从 O(n²) 降至近似线性
- 使 200K 上下文窗口在推理成本可控的前提下成为可能

### 强化学习基础设施 —— Slime
- 开发了名为 [slime](https://github.com/THUDM/slime) 的异步 RL 基础设施
- 将数据生成与模型训练解耦，允许独立生成训练轨迹
- 引入 Active Partial Rollouts (APRIL) 解决长尾生成瓶颈
- 显著提升训练吞吐量，使大规模 RL 训练稳定收敛

## "Agentic Engineering" 的具体含义

智谱将 GLM-5 的核心理念定义为从 **"Vibe Coding"（氛围编程）** 到 **"Agentic Engineering"（智能体工程）** 的范式跃迁：

### Vibe Coding（旧范式）
- AI 辅助写代码片段，人类主导架构和流程
- 类似 Copilot：补全、建议、解释
- 短周期、单次交互

### Agentic Engineering（新范式）
- AI **自主分解**高层目标为子任务并执行
- 跨越对话，进入**交付优先**（delivery-first）模式
- 自动编排工具、维护长执行周期状态
- 关键能力：
  - **任务分解**：将复杂系统工程拆解为可执行步骤
  - **工具编排**：自动调用搜索、代码执行、数据分析等工具
  - **长周期执行**：在多步骤工作流中保持上下文一致性
  - **自主决策**：低置信度时选择不回答而非编造（幻觉率大幅降低）

### Agent Mode (Beta) 具体能力
- **数据洞察**：上传数据 → 自动生成图表和分析 → 导出 xlsx/csv/png
- **智能写作**：大纲到终稿的全流程控制 → 直接导出 PDF/Word
- **全栈开发**：增强的指令理解 + 多步骤复杂工程任务执行

### Karpathy 的预言
> Andrej Karpathy（前 Tesla AI 总监）预测：如果说 2025 年是 AI 学会写代码的一年，那么 2026 年初我们可能正处于进入"Agentic Engineering"时代的前夜。

## Benchmark 表现

### 核心基准（GLM-5 vs 闭源前沿）

| Benchmark | GLM-5 | Claude Opus 4.5 | GPT-5.2 | Gemini 3 Pro |
|-----------|-------|-----------------|---------|--------------|
| HLE (w/ Tools) | **50.4** | 43.4 | 45.8 | 45.5 |
| SWE-bench Verified | 77.8% | **80.9%** | 76.2% | 80.0% |
| SWE-bench Multilingual | 73.3% | **77.5%** | 65.0% | 72.0% |
| Terminal-Bench 2.0 | 56.2% | **59.3%** | 54.0% | 54.2% |
| BrowseComp (w/ CM) | **75.9** 🥇 | 67.8 | 65.8 | 59.2 |
| MCP-Atlas | 67.8 | 65.2 | **68.0** | 66.6 |
| τ²-Bench | 89.7 | **91.6** | 85.5 | 90.7 |
| CyberGym | 43.2 | **50.6** | — | 39.9 |
| AIME 2026 I | 92.7 | **93.3** | — | 90.6 |

### 关键突破
- **Artificial Analysis Intelligence Index 得分 50**：首个突破此阈值的开源模型
- **幻觉率大幅降低**：Omniscience Index 从 -36 提升至 -1
- **输出效率提升**：完成评测所需 output tokens 比 GLM-4.7 减少 35.3%
- **BrowseComp 全场第一**（75.9），超越所有闭源模型

## 与其他开源模型对比

| 维度 | GLM-5 | DeepSeek-V3.2 | Kimi K2.5 | Qwen3-Coder (480B-35B) | Llama 4 Scout |
|------|-------|---------------|-----------|------------------------|---------------|
| 总参数 | 744B | ~671B | 未公开 | 480B | 250B（17B active） |
| 激活参数 | 40B | ~37B | 未公开 | 35B | 17B |
| 架构 | MoE | MoE | MoE | MoE | MoE |
| 上下文 | 200K | 128K | 128K | 128K | 10M |
| 许可证 | MIT | MIT-like | 自定义 | Apache 2.0 | Llama License |
| SWE-bench | **77.8%** | 73.1% | 76.8% | ~71% | ~58% |
| HLE (w/ Tools) | **50.4** | 40.8 | 51.8 | — | — |
| 训练硬件 | 华为昇腾 | NVIDIA | NVIDIA | NVIDIA | NVIDIA |

### 对比分析

- **vs DeepSeek-V3.2**：GLM-5 在 SWE-bench（+4.7%）、Terminal-Bench（+16.9%）、BrowseComp（+8.3）上全面领先。值得注意的是 GLM-5 采用了 DeepSeek 发明的 Sparse Attention 技术
- **vs Kimi K2.5**：两者在 HLE 上接近（GLM-5 50.4 vs K2.5 51.8），SWE-bench 上 GLM-5 略胜（77.8% vs 76.8%）。K2.5 在 BrowseComp 上也很强（74.9 vs 75.9）
- **vs Qwen3-Coder**：Qwen3-Coder 在算法密集型任务、调试和线程同步问题上有优势，但 GLM-5 在 Agentic 任务和系统工程上更强
- **vs Llama 4**：Llama 4 Scout 的 10M 上下文窗口是亮点，但在编码和 Agent 任务上与 GLM-5 差距明显

## 定价与部署

### API 定价
| 模型 | 输入价格/M tokens | 输出价格/M tokens |
|------|-------------------|-------------------|
| GLM-5 | ~$0.11 | TBD |
| GLM-5 (OpenRouter) | ~$0.80-1.00 | ~$2.56-3.20 |
| GPT-5.2 | $1.75 | $14.00 |
| Claude Opus 4.5 | $5.00 | $25.00 |

→ GLM-5 输入价格约为 GPT-5.2 的 1/16，Claude Opus 4.5 的 1/45

### 本地部署
- **BF16 全精度**：~1.65TB 存储，~1,490GB 推理显存，需多卡
- **FP8 量化**：8×GPU（vLLM/SGLang 均支持）
- **2-bit 量化**（Unsloth）：~241GB，可在 Mac Studio 等高端消费级设备运行
- 支持框架：vLLM、SGLang、KTransformers、xLLM
- 支持华为昇腾 NPU 部署

### GLM Coding Plan
Z.AI 提供订阅制编码方案，支持 Claude Code、Cursor、Cline、OpenClaw 等 20+ 编码工具：
- **Lite**：3× Claude Pro 用量，仅支持 GLM-4.7
- **Pro**：5× Lite 用量，含视觉分析、搜索、MCP
- **Max**：4× Pro 用量，支持 GLM-5，保证高峰时段性能

## 适用场景

### 最佳场景 ✅
- **复杂系统工程**：全栈开发、大型代码库重构、多步骤工程任务
- **长周期 Agent 任务**：自动化工作流、多工具编排、持续性任务执行
- **代码审查与安全**：CyberGym 43.2%，Terminal-Bench 56.2%
- **深度网页研究**：BrowseComp 全场第一（75.9）
- **需要低幻觉率的场景**：企业级应用、事实核查
- **成本敏感的大规模部署**：价格仅为 Claude/GPT 的几十分之一
- **中文场景**：BrowseComp-Zh 72.7%，中文理解有天然优势

### 非最佳场景 ⚠️
- **极限数学竞赛**：GPQA-Diamond 86.0 vs Gemini 3 Pro 91.9
- **需要最高编码精度**：SWE-bench 77.8% vs Claude Opus 4.5 80.9%
- **超长上下文**：200K vs Llama 4 Scout 10M / Claude 1M beta
- **本地轻量部署**：最小也需 ~241GB，不适合普通消费级硬件

## 战略意义

1. **硬件独立**：首个在非 NVIDIA 硬件上完成前沿规模训练的 MoE 模型，验证了中国 AI 算力自主路线的可行性
2. **开源前沿化**：Intelligence Index 得分 50，首次开源模型突破此门槛，挑战"开源模型永远落后闭源一个身位"的叙事
3. **定价颠覆**：以闭源模型 1/16 ~ 1/45 的价格提供接近的能力，推动行业重新评估成本结构
4. **Agent 范式推动**：从工具到自主体的转变，可能重新定义软件工程师与 AI 的协作模式

## 参考链接

- [HuggingFace 模型页](https://huggingface.co/zai-org/GLM-5)
- [Z.AI 技术博客](https://z.ai/blog/glm-5)
- [Z.AI Chat](https://chat.z.ai)
- [OpenRouter](https://openrouter.ai/z-ai/glm-5)
- [Slime RL 框架](https://github.com/THUDM/slime)
- [SCMP 报道](https://www.scmp.com/tech/article/3343239/chinas-zhipu-ai-launches-new-major-model-glm-5-challenge-its-rivals)
- [Reuters 报道](https://www.reuters.com/technology/chinas-ai-startup-zhipu-releases-new-flagship-model-glm-5-2026-02-11/)
- [llm-stats 分析](https://llm-stats.com/blog/research/glm-5-launch)
- [Digital Applied 分析](https://www.digitalapplied.com/blog/zhipu-ai-glm-5-release-744b-moe-model-analysis)

---

## See Also

- [[AI/Frontiers/GLM-5-技术报告精读|GLM-5 技术报告精读]] — GLM-5 完整技术分析（正式版）
- [[AI/LLM/Frameworks/Slime-RL-Framework|Slime RL Framework]] — GLM-5 使用的 RL 训练框架
- [[AI/Agent/_MOC|Agent MOC]] — Agentic Engineering 全图谱
- [[AI/LLM/_MOC|LLM MOC]] — 大语言模型知识全图谱
