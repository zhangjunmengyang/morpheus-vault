---
title: "AI Agent 评测方法与 Benchmark"
date: 2026-02-14
tags: [agent, evaluation, benchmark, interview]
type: note
---

# AI Agent 评测方法与 Benchmark

## 1. 为什么 Agent 评测很难

传统 LLM 评测（如 MMLU、HellaSwag）只测单轮文本生成，但 Agent 评测面临根本性不同的挑战：

| 维度 | 传统 LLM 评测 | Agent 评测 |
|------|-------------|-----------|
| 交互轮次 | 单轮 | 多步骤、多轮决策链 |
| 环境 | 无 | 需要真实/模拟环境（浏览器、终端、API） |
| 确定性 | 有标准答案 | 同一目标可有多条正确路径 |
| 评判 | 精确匹配/BLEU | 需要功能性验证（代码能不能跑、网页操作是否达成目标） |
| 成本 | 低 | 高（环境搭建、运行时间、API 调用费用） |

**核心难点总结：**

- **路径多样性**：完成同一任务可能有截然不同的步骤序列，无法用固定 ground truth 比对
- **部分完成问题**：Agent 完成了 80% 的步骤但最终失败，如何给分？
- **环境非确定性**：网页内容变化、API 返回不同、文件系统状态差异
- **评测泄漏**：公开 benchmark 的测试用例可能被训练数据污染（data contamination）
- **安全与副作用**：Agent 可能在评测过程中产生真实世界的副作用（发邮件、删文件）

---

## 2. 主流 Benchmark 分类

### 2.1 代码类

#### SWE-bench / SWE-bench Verified

- **任务**：给定一个真实 GitHub repo 的 issue，Agent 需要定位代码、编写 patch、通过测试
- **数据来源**：12 个热门 Python 仓库（Django、scikit-learn、sympy 等）的 2294 个真实 issue
- **评测方式**：运行仓库的单元测试，pass@1
- **SWE-bench Verified**：人工审核后的 500 题子集，去除了歧义和不可解的问题
- **当前 SOTA**：~65%（Verified），全集 ~30%
- **意义**：最接近真实软件工程的 benchmark，区分度极高

#### HumanEval / MBPP

- **任务**：函数级代码生成，给 docstring 写实现
- **HumanEval**：164 题，OpenAI 出品；**MBPP**：974 题，Google 出品
- **评测方式**：pass@k（生成 k 个样本，至少一个通过测试）
- **局限**：题目简单，当前模型已接近饱和（>95%），区分度下降
- **变体**：HumanEval+（加强测试用例）、EvalPlus、MultiPL-E（多语言）

#### Terminal-Bench

- **任务**：在终端环境中完成系统管理、DevOps 类任务
- **特色**：测试 Agent 的 shell 命令生成、文件操作、环境配置能力
- **与 SWE-bench 的区别**：不限于代码修改，包含运维场景

### 2.2 网页类

#### WebArena

- **任务**：在真实网站的自托管镜像上完成复杂任务（如「在 GitLab 上 fork 一个仓库并提交 PR」）
- **环境**：包含 GitLab、Reddit、购物网站、CMS、地图 5 个网站的完整功能镜像
- **规模**：812 个任务，平均需要 5-10 步操作
- **评测方式**：功能性验证（检查最终状态是否满足条件）
- **意义**：目前最被认可的 web agent benchmark
- **变体**：VisualWebArena（加入视觉理解）、WorkArena（企业 SaaS 场景）

#### BrowseComp

- **来源**：OpenAI 发布
- **任务**：需要深度浏览和信息整合才能回答的问题（非简单搜索可解）
- **特色**：测试 Agent 的多跳信息检索与推理能力
- **难度**：人类平均准确率也不高，强调 browsing 深度而非广度

#### Mind2Web

- **任务**：给定自然语言指令，在真实网页快照上预测操作序列
- **数据**：2000+ 任务，覆盖 137 个真实网站
- **评测方式**：element accuracy、action F1、step success rate
- **特色**：离线评测（基于网页快照），成本低但不如 WebArena 真实

### 2.3 工具调用类

#### API-Bank

- **任务**：Agent 需要选择正确的 API、构造参数、处理返回值
- **规模**：73 个 API，314 个对话场景
- **评测维度**：API 选择准确率、参数填充正确率、多 API 编排能力
- **三级难度**：L1（单 API 调用）→ L2（多 API 串联）→ L3（需搜索 API 文档）

#### ToolBench

- **规模**：16000+ 真实 API（来自 RapidAPI Hub）
- **任务**：多工具、多步骤任务规划与执行
- **特色**：规模最大的工具调用 benchmark，覆盖 49 个类别
- **配套**：ToolLLaMA 模型 + DFSDT（深度优先搜索决策树）算法

#### τ-bench (Tau-bench)

- **来源**：Sierra（Bret Taylor 创办）
- **任务**：模拟客服场景，Agent 需要调用工具完成用户请求（退款、改订单等）
- **特色**：强调 policy compliance —— 不仅要完成任务，还要遵守业务规则
- **评测维度**：任务完成率 + 规则违反率
- **意义**：最贴近企业实际 Agent 部署场景

### 2.4 通用推理类

#### GAIA (General AI Assistants)

- **来源**：Meta 等机构
- **任务**：需要多步推理 + 工具使用才能回答的问题（如「某 PDF 论文中表 3 第二行第三列的数字是什么？」）
- **三级难度**：L1（~5步）→ L2（~10步）→ L3（需要专业领域知识）
- **评测方式**：精确匹配最终答案
- **意义**：强调 Agent 的实用性 —— 人类可以轻松完成但 AI 需要工具辅助

#### HLE (Humanity's Last Exam)

- **来源**：CAIS & Scale AI
- **任务**：来自数百位顶尖学者出题的超高难度问答
- **覆盖**：数学、物理、生物、历史、法律等多个专业领域
- **难度**：当前最强模型正确率 < 10%
- **意义**：作为 AI 能力的天花板测试，短期内不会饱和

### 2.5 安全类

#### AgentHarm

- **任务**：测试 Agent 是否会执行有害指令（如帮助制作恶意软件、社会工程攻击）
- **评测方式**：有害请求的拒绝率 + 被 jailbreak 的成功率
- **发现**：许多 Agent 框架的安全防护远弱于底层 LLM 本身

#### CyberGym

- **任务**：网络安全攻防场景，Agent 需要进行渗透测试或防御
- **特色**：测试 Agent 在安全关键领域的能力与边界

### 2.6 多模态

#### MCP-Atlas

- **定位**：针对使用 MCP（Model Context Protocol）工具的 Agent 评测
- **特色**：评测 Agent 在多模态工具调用场景下的表现（文本 + 图像 + 结构化数据）
- **覆盖**：工具发现、Schema 理解、多工具编排、错误恢复

---

## 3. 评测指标体系

| 指标 | 定义 | 适用场景 |
|------|------|---------|
| **Success Rate (SR)** | 任务完全完成的比例 | 所有 benchmark 的核心指标 |
| **步骤效率 (Step Efficiency)** | 完成任务的平均步骤数 vs 最优步骤数 | WebArena、GAIA |
| **Tool-call Accuracy** | 工具选择 × 参数正确 的准确率 | API-Bank、ToolBench |
| **Partial Credit** | 子目标完成比例（适用于多步骤任务） | SWE-bench、WebArena |
| **安全合规率** | 拒绝有害请求 / 遵守 policy 的比例 | AgentHarm、τ-bench |
| **Cost Efficiency** | 每个任务的 token 消耗 / API 调用成本 | 工程部署场景 |
| **Robustness** | 对 prompt 微扰 / 环境变化的稳定性 | 研究场景 |
| **Pass@k** | 生成 k 次至少成功一次的概率 | HumanEval、代码生成 |

**评测方法论要点：**

- **功能性验证 > 轨迹匹配**：检查最终状态而非中间步骤
- **多次运行取统计量**：Agent 的非确定性要求至少 3-5 次运行取平均
- **人工审核仍不可少**：自动评测可能有 false positive/negative
- **注意数据污染**：使用 held-out 测试集或定期更新题目

---

## 4. 面试常见问题及回答要点

### Q1: SWE-bench 为什么被认为是目前最好的 Agent Benchmark？

**回答要点：**
- 基于**真实** GitHub issue，不是人造题目，ecological validity 高
- 测试**端到端**能力：理解 issue → 定位代码 → 编写 patch → 通过测试
- 区分度好：当前 SOTA ~65%（Verified），远未饱和
- 有 Verified 子集解决了原始数据集中的噪声问题
- 可以反映模型在**实际软件工程**场景中的能力

### Q2: Agent 评测中如何处理「多条正确路径」的问题？

**回答要点：**
- **功能性验证**（outcome-based）：不检查路径，只检查最终状态是否满足条件（如 WebArena 检查数据库状态、页面内容）
- **单元测试**：SWE-bench 用原仓库的测试验证 patch 正确性
- **LLM-as-Judge**：用另一个 LLM 判断任务是否完成（但引入评判偏差）
- **人工评审**：金标准但不可扩展
- 注意区分 **过程正确性**（是否遵守约束）和 **结果正确性**（是否达成目标）

### Q3: τ-bench 和 WebArena 的核心区别是什么？

**回答要点：**
- **τ-bench** 强调 **policy compliance**（遵守业务规则），模拟企业客服场景；不仅要完成任务，还不能违反退款政策、隐私规则等
- **WebArena** 强调 **web navigation 能力**，测试 Agent 在复杂网页上的操作能力
- τ-bench 更贴近**企业部署**需求（安全合规是刚需）
- WebArena 更贴近**通用 web agent** 研究
- 两者互补：一个测「能不能做对」，一个测「做的时候有没有违规」

### Q4: 如何设计一个新的 Agent Benchmark？需要考虑哪些因素？

**回答要点：**
1. **任务真实性**：来自真实需求而非人造场景
2. **可复现性**：环境可控、可重复搭建（Docker 化）
3. **评测自动化**：功能性验证脚本，减少人工成本
4. **抗污染**：定期更新题目或使用 private held-out set
5. **分级难度**：L1/L2/L3 区分不同能力层次
6. **指标多维**：不只看成功率，还要看效率、安全性、成本
7. **规模足够**：至少数百个任务才有统计意义

### Q5: 当前 Agent 评测的最大局限是什么？未来方向？

**回答要点：**
- **局限**：
  - 静态 benchmark 会被刷榜和数据污染
  - 缺少**长期任务**评测（现有 benchmark 大多是分钟级任务）
  - **多 Agent 协作**场景几乎没有 benchmark
  - 安全评测覆盖不足（对抗性攻击、间接提示注入）
  - 成本高，难以大规模运行
- **未来方向**：
  - **动态 benchmark**：每次运行生成不同的任务实例
  - **长期任务**：跨天/跨周的项目级评测
  - **人机协作评测**：Agent 辅助人类的效率提升而非替代
  - **安全红队评测**：系统化的对抗性测试框架
  - **私有评测平台**：防污染，类似 Chatbot Arena 的 Elo 排名

---

## 参考资源

- SWE-bench: [swebench.com](https://swebench.com)
- WebArena: [webarena.dev](https://webarena.dev)
- GAIA: HuggingFace leaderboard
- τ-bench: Sierra AI 发布
- HLE: Scale AI & CAIS
