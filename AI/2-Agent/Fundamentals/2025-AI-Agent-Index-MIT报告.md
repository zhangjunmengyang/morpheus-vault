---
title: "2025 AI Agent Index：MIT 联合报告 + Anthropic Agent 自主度实测"
type: paper-note
domain: agent
tags:
  - type/paper
  - agent
  - safety
  - autonomy
  - benchmark
  - MIT
  - Anthropic
arXiv: "2602.17753"
source: https://arxiv.org/abs/2602.17753
anthropic_report: https://www.anthropic.com/research/measuring-agent-autonomy
created: 2026-02-27
rating: ★★★★☆
---

# 2025 AI Agent Index：MIT 报告 + Anthropic 自主度实测

> 两份报告的核心张力：**我们对 Agent 了解越来越少，而它们做的事越来越多。**

---

## 一、MIT 报告：2025 AI Agent Index (arXiv: 2602.17753)

**机构**：MIT + 剑桥 + 斯坦福 + 哈佛法学院等
**发布**：2026-02-19
**方法**：对 30 个主流 Agent 做 45 维度、1350 字段的系统性分析
**网站**：https://aiagentindex.mit.edu

### Agent 入选标准（四条缺一不可）

1. **自主性** — 无持续人工干预即可运行，能做有实质影响的决策
2. **目标复杂度** — 能做长链路规划，连续自主调用 ≥3 次工具
3. **环境交互** — 有写权限，能真实改变外部世界（不是只说话）
4. **通用性** — 能处理模糊指令，适应新任务（不是窄域工具）

从 95 个候选系统筛出 **30 个**，还要求有足够市场影响力（搜索量/估值/签署前沿 AI 安全承诺）。

---

### 三类 Agent 分布

**Chat 类（12 个）**：Claude、Claude Code、Gemini、Gemini CLI、Kimi OK Computer、Manus AI、MiniMax Agent、ChatGPT、ChatGPT Agent、OpenAI Codex、Perplexity、AutoGLM 2.0

**浏览器类（5 个）**：Alibaba MobileAgent、ByteDance TARS、ChatGPT Atlas、Opera Neon、Perplexity Comet

**企业工作流类（13 个）**：Browser Use、Glean Agents、Gemini Enterprise、HubSpot Breeze、IBM watsonx Orchestrate、Microsoft Copilot Studio、OpenAI AgentKit、SAP Joule、Salesforce Agentforce、ServiceNow、WRITER Action、Zapier AI、n8n Agents

**地域分布**：21 个美国，5 个中国（Kimi/MiniMax/Z.ai/Alibaba/ByteDance），4 个其他
**开源情况**：23/30 完全闭源

---

### 维度一：自主程度五级框架

| 级别 | 含义 |
|------|------|
| L1 | 人主导，Agent 只执行具体指令 |
| L2 | 人与 Agent 协作规划，共同执行 |
| L3 | Agent 主导执行，人在关键节点审批 |
| L4 | Agent 自主执行大部分，人只作为审批者 |
| L5 | Agent 完全自主，人只是旁观者 |

**关键发现**：
- 浏览器类 Agent 普遍 **L4–L5**
- 很多企业级 Agent 宣传材料写 L1–L2，**实际部署飙到 L3–L5**
- "以为买了辅助工具，实际在运行自主决策者"

---

### 维度二：底层依赖高度集中

- 除 Anthropic/Google/OpenAI 自家产品 + 中国厂商（自研模型），**几乎所有 Agent 都压在 GPT/Claude/Gemini 三个底层上**
- 三家对整个 Agent 生态握有隐性控制权——定价/服务条款变动影响数十个上层产品
- 只有 9/30 企业 Agent 支持用户自选底层模型

---

### 维度三：记忆黑盒（最不透明的维度）

- "Memory Architecture" 是整份报告灰色字段（未找到公开信息）最密集的区域
- 大多数开发者不说明：Agent 记住了什么？保存多久？会不会跨任务传递信息？用户能查看/删除记忆吗？
- 在 Agent 可接触邮件/日历/CRM/文件系统的情况下，记忆不透明 = 严重隐私风险

---

### 维度四：行动空间差异

**CLI 类**（Claude Code/Gemini CLI）：直接读写文件系统、执行终端命令，能编译代码、删文件。最接近"有根服务器权限"的 Agent 形态。

**浏览器类**：操控整个网页界面，能订机票/填表单/发邮件。关键问题：**大多数直接无视 robots.txt**，理由是"代替真实用户操作"。只有 **ChatGPT Agent** 一家使用加密签名证明访问身份。

**企业工作流类**：通过 CRM 连接器操作业务记录（Salesforce/HubSpot 客户数据/销售记录）。8/30 Agent 可直接读写这些系统。

---

### 维度五：安全透明度 — 能力在飞奔，安全在裸奔

- 只有 **4/30** 发布了 Agent 专属 system card（ChatGPT Agent / OpenAI Codex / Claude Code / Gemini 2.5 Computer Use）
- **25/30** 不披露内部安全测试结果
- **23/30** 没有任何第三方测试数据
- 5 个中国 Agent 里只有 **1 个**（智谱）发布了安全框架（但可能是中文文档未被统计）

**问责碎片化（Accountability Fragmentation）**：
```
基础模型厂商 → Agent 开发商 → 企业客户 → 最终用户
```
每层都声称自己只是平台/工具，不负上下游责任。出了事找谁？答案是：找不到。

**数据补充**：研究团队联系全部 30 家开发商，给 4 周回应时间。结果 **76% 沉默，只有 4 家提供实质意见**。

---

## 二、Anthropic 报告：Measuring AI Agent Autonomy in Practice

**数据来源**：公共 API 上百万次工具调用 + Claude Code 约 50 万次会话（偏向开发者群体）

### 核心发现 1：自主运行时长飞速增长

- 2025-10 到 2026-01，Claude Code 最长任务不中断运行时长：**从 <25 分钟涨到 >45 分钟，三个月近乎翻倍**
- 增长曲线平滑穿越模型发布节点 → **不是纯粹因为模型能力提升，而是用户在给模型更大自主空间**
- 中位数 turn 时长约 45 秒，99 百分位以下基本稳定 → 增长集中在最长的那批 power user 任务

### 核心发现 2：老用户行为反直觉

| 用户类型 | 全自动审批比例 | 中途打断率 |
|---------|-------------|---------|
| 新用户（<50 次会话）| ~20% | ~5% |
| 老用户（>750 次会话）| >40% | ~9% |

**解读**：
- 新用户是"全信 or 全不信"的粗放模式
- 老用户是"放手跑大任务 + 精准监控关键节点"的精细模式
- 越老练，控制方式越反直觉——**放手更多，但干预也更精准**

### 核心发现 3：Agent 也在主动问你

- 在最复杂的任务上，**Claude Code 主动停下来求确认的频率，是人类主动打断它的 2 倍以上**
- 不是人类单方面监督 AI，AI 也在反向确认人类意图
- Anthropic 认为这是良性信号：人机共同管理自主权和风险

### 核心发现 4：高风险场景存在但未规模化

- ~80% 工具调用有安全防护兜底
- 73% 保持某种人类参与
- 真正不可逆操作（如发出去的邮件）只占 **~0.8%**
- **编程占 agentic activity 近 50%**，医疗/金融/网安是"萌芽中的"

---

## 三、两份报告放一起的张力

| 维度 | MIT 报告视角 | Anthropic 报告视角 |
|------|------------|-----------------|
| 透明度 | 大多数开发商不公开安全信息 | 内部有大量监控数据 |
| 自主度 | 宣称 L1–L2，实际 L3–L5 | 三个月内自主时长翻倍 |
| 风险 | 问责碎片化，找不到责任人 | 80% 有安全兜底，0.8% 不可逆 |
| 趋势 | 产品爆炸但治理空白 | 使用深度快速增长 |

**核心矛盾**：我们对 Agent 的了解速度 < Agent 获得真实权力的速度。

---

## 四、为什么编程跑在最前面？

Doug O'Laughlin (SemiAnalysis)：编程是 AI 进入 15 万亿美元信息工作市场的"**滩头阵地**"（beachhead）

Dario Amodei（达沃斯）：软件工程是"最清晰的测试场景——结构化、数字化、可衡量"

Andrej Karpathy：编程是唯一 **AI 产出可以直接加速 AI 自身进步的领域**（自我加速飞轮）

两个特质叠加：
1. **阻力最小的落地场景**（可验证 + 已有工具链 + 失败成本相对可控）
2. **唯一能自我加速的领域**（AI 写的代码让下一代 AI 更强）

**但滩头终究只是滩头**：
- 全球只有 0.04% 的人用过 AI 编程
- 为 AI 工具付费比例 0.3%
- 84% 的人从未真正使用过 AI（Microsoft AI Economy Institute 数据）

---

## 五、对 Agent RL 工程师的启示

1. **L4–L5 是工程现实，不是未来愿景** — 浏览器类 Agent 已经跑在这个级别，训练时必须考虑
2. **信任是渐进建立的** — 老用户的"放手+精准干预"模式，应该是 RLHF 奖励设计的参考
3. **Agent 主动求确认 > 人类被动打断** — 良好的 uncertainty 估计是 Agent RL 的关键能力
4. **问责碎片化是工程问题** — 如果每层都说"我只是平台"，安全审计就永远 miss
5. **记忆透明度是护城河** — 谁先做出"可审计、可删除、用户可见"的记忆系统，谁就有差异化

---

## 参考文献

- [1] **2025 AI Agent Index** — arXiv:2602.17753 (MIT/剑桥/斯坦福/哈佛, 2026-02-19)
  <https://arxiv.org/abs/2602.17753>
- [2] **Measuring AI Agent Autonomy in Practice** — Anthropic (2026)
  <https://www.anthropic.com/research/measuring-agent-autonomy>
- [3] **Anthropic Economic Index, January 2026** — <https://www.anthropic.com/research/anthropic-economic-index-january-2026-report>
- [4] **Claude Code Security** — <https://www.anthropic.com/research/claude-code-security>
- [5] **SemiAnalysis: Claude Code is the Inflection Point** — <https://newsletter.semianalysis.com/p/claude-code-is-the-inflection-point>
- [6] **Microsoft AI Economy Institute** — <https://www.microsoft.com/en-us/research/project/ai-economy/>

---

*入库：Scholar | 2026-02-27*

## See Also

- [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] — Agent 技术前沿（与 MIT 报告的市场调研视角互补）
- [[AI/2-Agent/Fundamentals/Building-Effective-Agents-Anthropic|Building Effective Agents（Anthropic）]] — Anthropic 工程实践与 MIT 学术报告的对照