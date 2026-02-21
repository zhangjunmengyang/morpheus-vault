---
title: "Gemini 3.1 Pro 速评"
date: 2026-02-20
type: frontier
domain: ai/frontiers
rating: ★★★★★
tags:
  - 模型追踪
  - Google
  - Gemini
  - benchmark
  - frontier
---

# Gemini 3.1 Pro 速评（2026-02-19 发布）

## 核心数据

| Benchmark | Gemini 3.1 Pro | Gemini 3 Pro | Claude Opus 4.6 | GPT-5.2 |
|-----------|---------------|-------------|-----------------|---------|
| **ARC-AGI-2** | **77.1%** | 31.1% | 68.8% | 52.9% |
| GPQA Diamond | **94.3%** | - | - | - |
| SWE-Bench Verified | 80.6% | - | **80.8%** | - |
| LiveCodeBench Pro (Elo) | **2,887** | 2,439 | - | 2,393 |
| MCP Atlas (Agent) | **69.2%** | - | - | - |
| BrowseComp | **85.9%** | - | - | - |
| MMMU Pro | 80.5% | **81.0%** | - | - |

## 关键要点

1. **ARC-AGI-2 翻倍**: 31.1% → 77.1%，超越 Opus 4.6 (68.8%) 和 GPT-5.2 (52.9%)
2. **价格**: 不到 Claude Opus 4.6 一半（具体定价待确认）
3. **Agentic 能力突出**: MCP Atlas 69.2% + BrowseComp 85.9%，Agent 场景强
4. **代码接近 Opus**: SWE-Bench 80.6% vs Opus 80.8%，几乎持平
5. **不足**: MMMU Pro 多模态上反而不如前代 3 Pro

## 对我们的影响

- **模型选型**: 如果价格确认是 Opus 半价，且 Agent/代码接近 Opus 水平，值得测试作为 Sentinel/Scholar 等子 Agent 的主力模型
- **面试话题**: "2026 年模型竞争格局" 的最新数据点——Google 正在追赶 Anthropic
- **量化方向**: Gemini API 的价格优势可能降低 AI+量化的推理成本

## 注意事项

- Benchmark ≠ 真实能力，SWE-Bench 独立跑分显示"当代模型真实水位普遍虚高"
- ARC-AGI-2 高分不代表 AGI 突破（历史上更高分的系统也没改变 AI 格局）
- 需要实际测试在我们的 pipeline 上表现如何

---

*Source: The Decoder, 2026-02-19*

## See Also
- [[AI/Frontiers/Claude-Opus-4.6|Claude Opus 4.6]] — ARC-AGI-2 68.8% vs Gemini 3.1 Pro 61.7%，推理旗舰直接对决；格局逆转标志性节点
- [[AI/Frontiers/Claude-Sonnet-4.6|Claude Sonnet 4.6]] — 性价比对标；Gemini 3.1 Flash 对标 Sonnet 4.6 价位
- [[AI/Frontiers/GLM-5-技术报告精读|GLM-5]] — 2026 模型竞争格局的另一重要数据点；中美模型同期发布潮
- [[AI/Frontiers/2026年2月模型潮|2026年2月模型潮]] — Gemini 3.1 Pro 是本次模型潮的核心角色；ARC-AGI-2 成为新的竞争标尺
- [[AI/LLM/Evaluation/PERSIST-LLM-Personality-Stability-Benchmark|PERSIST（LLM人格稳定性）]] — frontier模型（25个，1B-685B）的人格一致性评测；Gemini 3.1 Pro 这个规模段的模型在 PERSIST 中仍有 SD>0.3，说明模型能力提升与行为一致性是独立维度
