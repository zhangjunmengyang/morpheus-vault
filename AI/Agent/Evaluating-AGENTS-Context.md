---
title: "Evaluating AGENTS.md: Context Files 对 Coding Agent 真的有用吗？"
brief: "ETH Zurich，ICML 2026 投稿：系统评估 AGENTS.md 等上下文文件对 coding agent 的实际效果；发现结构良好的 context file 显著提升任务成功率，但不当设计反而干扰；上下文工程的实证研究（arXiv:2602.11988）"
date: 2026-02-19
updated: 2026-02-23
tags:
  - agent
  - coding-agent
  - context-engineering
  - benchmark
  - type/paper
arxiv: "2602.11988"
paper_url: https://arxiv.org/abs/2602.11988
authors: "Thibaud Gloaguen et al. (ETH Zurich)"
venue: ICML 2026 (submitted)
---

# Evaluating AGENTS.md: Context Files 对 Coding Agent 真的有用吗？

> **一句话总结**：在对 4 个主流 coding agent 和 2 个 benchmark 的大规模实验中，LLM 自动生成的 context files（如 AGENTS.md）**平均降低任务成功率 0.5%-2%，同时增加推理成本 20%+**；即使是开发者手写的 context files 也只带来约 4% 的微弱提升。论文的核心建议是：**只写最小必需内容，不写比写好。**

---

## 1. 为什么这篇论文重要

截至论文撰写时，超过 **60,000 个** GitHub 开源仓库已经包含 AGENTS.md 或 CLAUDE.md 等 context files。Anthropic、OpenAI、Qwen 等主要 agent 厂商都**官方推荐**用户为仓库创建这类文件，并提供了 `/init` 命令一键生成。

这种做法背后的直觉很合理：给 agent 一份关于仓库结构、工具链、代码风格的"入门指南"，应该能让它更高效地完成任务。但一个关键问题从未被严格验证过——

> **这些 context files 到底有没有用？它们真的提高了 agent 解决实际问题的能力吗？**

ETH Zurich 的研究团队（Mündler 等人此前构建了 SWT-Bench）首次对这个问题进行了系统性的实证研究，结论令人惊讶且实践意义重大。

---

## 2. 实验设计

论文采用了两种互补的评估设置，覆盖了 context files 的两种主要来源。

### 2.1 设置一：SWE-bench Lite + LLM 生成的 Context Files

**数据集**：SWE-bench Lite，包含 300 个任务，来自 11 个流行 Python 仓库（如 Django、scikit-learn、sympy 等）。这些仓库本身**没有** context files。

**实验流程**：
1. 对每个仓库，使用各 agent 自带的初始化命令（如 Claude Code 的 `/init`）自动生成 context file
2. 比较三种设置下的 agent 表现：
   - **None**：不提供任何 context file
   - **LLM**：使用 agent 推荐的方式自动生成 context file
3. 每个设置在 4 个 agent 上分别测试

**目标**：评估**自动生成**的 context files 在知名仓库上的效果。

### 2.2 设置二：AGENTbench（新构建）+ 开发者手写的 Context Files

这是论文最重要的方法论贡献。由于 context files 是 2025 年 8 月才被正式规范化的新实践，现有 benchmark 的仓库几乎都没有这类文件。论文因此构建了 **AGENTbench**——一个专门用于评估 context files 效果的新 benchmark。

**AGENTbench 的构建流程**（五个阶段）：

1. **仓库筛选**：通过 GitHub 搜索，找到根目录包含 AGENTS.md 或 CLAUDE.md 的 Python 仓库，要求有测试套件且至少 400 个 PR
2. **PR 过滤**：筛选引用了 issue、修改了 Python 文件、引入确定性可测试行为的 PR（使用规则 + LLM 组合过滤）
3. **环境搭建**：用 coding agent 为每个 PR 对应的仓库状态搭建可执行的测试环境
4. **任务描述标准化**：由于小众仓库的 PR 描述质量参差不齐，用 LLM agent 基于 PR 描述和 patch 生成标准化的 issue 描述（6 个部分：描述、复现步骤、预期行为、实际行为、规格说明、附加信息）
5. **单元测试生成**：由于多数 PR 不含测试修改，用 LLM 生成回归测试并人工校验

**最终数据集**：138 个实例，来自 12 个仓库，所有仓库都包含开发者手写的 context files。

**三种对比设置**：
- **None**：移除开发者提供的 context file
- **LLM**：用 agent 推荐方式自动生成 context file
- **Human**：使用开发者原始提交的 context file

**数据集关键统计**：

| 指标 | 均值 | 最小值 | 最大值 |
|------|------|--------|--------|
| Issue 描述词数 | 211.6 | 96 | 500 |
| 仓库文件数 | 3,337 | 151 | 26,602 |
| PR patch 行数 | 118.9 | 12 | 1,973 |
| 测试覆盖率 | 75% | 2.5% | 100% |
| Context file 词数 | 641.0 | 24 | 2,003 |
| Context file 章节数 | 9.7 | 1 | 29 |

### 2.3 评估的 Agent 和模型

| Agent | 模型 | 备注 |
|-------|------|------|
| Claude Code | Sonnet-4.5 | 温度 = 0 |
| Codex | GPT-5.2 | 温度 = 0 |
| Codex | GPT-5.1 mini | 温度 = 0 |
| Qwen Code | Qwen3-30b-coder | 温度 = 0.7, top-p = 0.8 |

### 2.4 评估指标

- **Success rate (𝒮)**：agent 生成的 patch 通过所有测试的比例
- **Steps**：agent 完成任务的交互步数（每次工具调用 = 1 步）
- **Cost**：完成任务的 LLM 推理成本（USD）

---

## 3. 核心发现

### 3.1 发现一：LLM 生成的 Context Files 降低成功率、增加成本

这是论文最核心也最反直觉的发现。

**成功率变化**：

在 8 个实验组合（4 个 agent × 2 个 benchmark）中，LLM 生成的 context files 在 **5 个** 组合中导致成功率下降：
- SWE-bench Lite：平均下降 **0.5%**
- AGENTbench：平均下降 **2%**

**成本变化**（对比 None 设置）：

| 维度 | SWE-bench Lite | AGENTbench |
|------|---------------|------------|
| 平均步数增加 | +2.45 步 | +3.92 步 |
| 平均成本增加 | **+20%** | **+23%** |

具体数据示例（AGENTbench）：
- Sonnet-4.5：None 设置 $1.15/实例 → LLM 设置 $1.33/实例（+15.7%）
- GPT-5.2：None 设置 $0.38/实例 → LLM 设置 $0.57/实例（**+50%**！）
- GPT-5.1 mini：None 设置 $0.18/实例 → LLM 设置 $0.20/实例（+11.1%）

**关键洞察**：成本增加在所有设置中都是**一致的**——即便是那些成功率有所提升的组合，成本也在增加。这意味着 context files 让 agent **做了更多工作，但不一定是更有效的工作**。

### 3.2 发现二：开发者手写的 Context Files 略好于 LLM 生成的，但提升幅度很小

在 AGENTbench 上，开发者手写的 context files：
- 在 4 个 agent 中的 **3 个** 上优于 None 设置（Claude Code 除外）
- 平均提升约 **4%**
- 但同样增加了成本：平均多 3.34 步，成本最多增加 19%

**对比 LLM 生成**：在所有 4 个 agent 上，开发者手写的文件都优于 LLM 自动生成的文件——尽管它们**并非针对特定 agent 定制**。

这暗示了一个重要信号：**人类的判断力——知道什么该写、什么不该写——比 LLM 的全面性更有价值。**

### 3.3 发现三：Context Files 不能有效充当仓库导览

Agent 厂商推荐在 context files 中包含仓库概览（codebase overview）。论文发现：

- 12 个 AGENTbench 仓库中的开发者 context files 中，8 个包含专门的概览章节
- LLM 生成的 context files 中，**100%** 的 Sonnet-4.5 生成文件包含概览，GPT-5.2 为 99%，Qwen3-30b-coder 为 95%
- 唯一的例外是 GPT-5.1 mini（仅 36% 包含概览）

但概览章节的**实际效果**如何？论文通过测量"agent 在多少步之后首次接触到 PR 修改的文件"来评估。结果：

> **无论有没有 context file，agent 定位到相关文件的速度几乎没有差别。**

更糟糕的是，对于 GPT-5.1 mini，context file 反而**显著增加**了定位时间。手动检查 trace 发现，这个模型会花多个步骤去**搜索 context file 的位置**，以及**反复重新阅读**已经在其上下文中的 context file。

### 3.4 发现四：Context Files 本质上是冗余文档

论文提出并验证了一个重要假说：

> LLM 生成的 context files 与仓库现有文档高度重复，而开发者手写的 context files 包含额外信息。

**验证实验**：在生成 context file 后，手动删除仓库中所有文档文件（.md 文件、示例代码、docs/ 目录），然后再评估 agent 表现。

**结果**：在这种"context file 是唯一文档来源"的设置下：
- LLM 生成的 context files **一致提升了表现**（平均 +2.7%）
- 而且 **LLM 生成的反而优于开发者手写的**

这解释了一个现象：为什么一些开发者会报告"加了 context file 后 agent 表现更好"——因为他们的仓库可能本身**缺乏文档**。在文档充足的仓库中，context file 只是噪音。

### 3.5 发现五：Agent 确实在遵循 Context File 的指令

一个重要的澄清：context files 的失败**不是因为 agent 忽略了它们**。

证据：
- 当 context file 提到使用 `uv` 工具时，agent 平均每实例使用 1.6 次；不提到时，使用 < 0.01 次
- 仓库特定工具（repo_tool）：提到时平均使用 2.5 次，不提到时 < 0.05 次
- 几乎所有工具的使用频率都与 context file 的提及高度相关

**这是一个关键发现**：agent 是**服从的**——它们会忠实执行 context file 中的指令。问题不在于"agent 不听话"，而在于**指令本身不帮忙、甚至有害**。

### 3.6 发现六：遵循 Context File 指令需要更多思考

论文分析了 GPT-5.2 和 GPT-5.1 mini 的 adaptive reasoning（自适应推理）token 用量——这些模型会根据任务难度自动调整 reasoning token 数量。

**结果**：

| 模型 | SWE-bench Lite reasoning token 增加 | AGENTbench reasoning token 增加 |
|------|--------------------------------------|--------------------------------|
| GPT-5.2 + LLM context | +22% | +14% |
| GPT-5.1 mini + LLM context | +14% | +10% |
| GPT-5.2 + Human context | +20% | — |
| GPT-5.1 mini + Human context | +2% | — |

**解读**：模型自身也"认为"有 context file 的任务更难了。额外的指令和约束增加了认知负担，agent 需要花更多 token 来思考如何同时满足原始任务要求和 context file 中的额外要求。

---

## 4. 具体的失败模式

论文的 trace analysis 揭示了多种具体的失败模式，值得逐一分析。

### 4.1 过度探索（Over-exploration）

Context files 导致 agent 进行**更广泛但不一定更有效**的探索：

- **更多 grep 搜索**：agent 会花更多时间搜索文件，试图理解 context file 中描述的仓库结构
- **更多文件读取**：读取更多与任务无关的文件
- **更多文件写入**：进行更多实验性的修改
- **更多测试运行**：反复运行测试来验证对 context file 要求的遵循

这些额外操作**消耗了 token 和步数**，但对核心任务的解决没有帮助，有时甚至产生干扰——agent 在探索中迷失了方向。

### 4.2 不必要的约束服从（Unnecessary Constraint Following）

这是最关键的失败模式。Context files 通常包含：

- **代码风格要求**（如 "使用 type hints"、"遵循 PEP 8"）
- **工具链要求**（如 "使用 uv 而非 pip"、"用 pytest 运行测试"）
- **架构约束**（如 "所有新功能必须放在 src/ 目录下"）

这些要求对于日常开发是合理的，但对于修复一个具体 bug 或实现一个小功能来说，它们是**不必要的额外约束**。Agent 试图同时满足原始任务和这些附加要求，导致：

1. **分散注意力**：agent 花时间确认自己的 patch 是否符合风格要求，而不是专注于让测试通过
2. **引入错误**：为了遵循某个 context file 中的约束，agent 可能修改了本不需要改的代码
3. **增加复杂度**：简单的修复方案被拒绝（因为不符合 context file 的要求），转而尝试更复杂的方案

### 4.3 Context File 寻址循环（Context File Seeking Loop）

在 GPT-5.1 mini 的 trace 中观察到一种特别的失败模式：

1. Agent 知道存在 context file（因为它已经在上下文中了）
2. 但 agent 仍然发起多次文件搜索命令来**定位** context file 的物理位置
3. 然后**重复阅读**已经在上下文中的内容

这种行为只在存在 context file 时出现，浪费了大量步数。

### 4.4 冗余信息噪音（Redundant Information Noise）

LLM 生成的 context files 的一个根本问题是：它们提取的信息（仓库结构、依赖列表、工具链说明）**已经可以从仓库本身获得**。

- Agent 在没有 context file 时，会按需探索仓库并发现这些信息
- 有了 context file，agent 反而需要**同时处理两个信息源**（context file + 仓库本身），并解决潜在的冲突或不一致

论文通过"删除所有文档后 context file 效果反转"的实验完美验证了这一点。

### 4.5 更强的模型不一定生成更好的 Context Files

论文对比了用 GPT-5.2（更强模型）生成 context file 和用各 agent 默认模型生成的效果：

- 在 SWE-bench Lite 上：GPT-5.2 生成的**平均提升 2%**（因为这些热门仓库的信息 GPT-5.2 的参数知识更丰富）
- 在 AGENTbench 上：GPT-5.2 生成的**平均下降 3%**（对于小众仓库，更强的模型反而生成了更多"自信但不准确"的内容）

这是一个值得深思的结论。直觉上我们会认为"用最好的模型生成 context file 一定更好"，但事实并非如此。更强的模型可能意味着：
- **更高的自信度**：GPT-5.2 可能会用更肯定的语气描述它并不完全了解的仓库细节
- **更详尽的内容**：更强的模型倾向于生成更长、更全面的文档——这恰恰增加了冗余
- **更多的"合理推断"**：模型可能基于常见模式推断仓库的特定配置，在小众项目中容易出错

这引出了一个更一般性的洞察：**在 context engineering 中，信息的准确性和相关性远比全面性重要。**

### 4.6 不同 Prompt 模板没有显著差异

论文对比了 Codex prompt 和 Claude Code prompt 生成的 context files：
- Claude Code 用 Codex prompt 生成的文件表现更好
- GPT 系列用哪个 prompt 在两个 benchmark 上表现不一致
- 总体结论：**prompt 的选择对 context file 质量影响不大**，问题不在 prompt 而在这种做法本身

这个发现尤其重要，因为它排除了"只需要更好的 prompt 就能解决问题"的可能性。问题是结构性的——自动生成 context file 这个行为本身，在当前的技术条件下，倾向于产生冗余信息。

值得注意的是，Claude Code 的官方 prompt 明确警告"不要列出容易自动发现的组件"，这比 Codex 和 Qwen Code 的 prompt 更保守。但即便如此，100% 的 Sonnet-4.5 生成的 context files 仍然包含了仓库概览——模型并没有很好地遵循自己 prompt 中的这条约束。这揭示了一个 meta 层面的问题：**指导 LLM 生成简洁内容本身就是一个未解决的挑战。**

---

## 5. 关键结论："最小化原则"

论文的结论直接且有力：

### 5.1 不推荐使用 LLM 自动生成的 Context Files

> "We therefore suggest omitting LLM-generated context files for the time being, contrary to agent developers' recommendations."

这与 Anthropic、OpenAI 等厂商的官方推荐**直接矛盾**。论文认为，至少在当前技术水平下，自动生成的 context files 弊大于利。

### 5.2 手写 Context Files 应只描述最小必需内容

如果确实需要 context file，应该遵循**最小化原则**：

> "Human-written context files should describe only minimal requirements."

具体来说：

**应该写的内容（minimal requirements）**：
- 仓库特有的、不可从代码/文档自动推断的工具链要求（如特殊的 build 命令、非标准的测试框架）
- 关键的安全约束或不可逆操作警告
- 仓库特有的非显而易见的约定（如特殊的文件命名规则）

**不应该写的内容**：
- 仓库结构概览（agent 可以自己探索，而且通常做得更好）
- 代码风格指南（对任务完成没有帮助）
- 依赖管理说明（通常已在 pyproject.toml 等标准文件中）
- 通用开发最佳实践（agent 的训练数据已经包含了）
- 可以从 README、docs/ 等现有文档推断的任何内容

### 5.3 Context Files 在缺乏文档的仓库中有价值

论文的"删除文档"实验表明：当仓库缺乏文档时，LLM 生成的 context files 确实有帮助（平均 +2.7%）。这为一个特定使用场景提供了指导：

> **如果你的仓库文档很少，context file 可以充当"最后手段"的文档补充。但如果文档已经完善，不要画蛇添足。**

---

## 6. 方法论的精妙之处

### 6.1 AGENTbench 的设计考量

论文构建 AGENTbench 时面临的核心挑战是：context files 是 2025 年下半年才流行起来的新事物，采用它的仓库往往是小众项目。这导致了几个级联问题：

1. **小众仓库的 PR 质量不稳定**：不像 Django、scikit-learn 那样有严格的 PR 模板
2. **多数 PR 不包含测试**：大仓库要求每个 PR 都带测试，小仓库没这个要求
3. **Issue 描述模糊**：有些 PR body 甚至是空的

论文通过 LLM agent 生成标准化的 issue 描述和单元测试来解决这些问题，并进行了 10% 抽样的人工验证（无 solution leaking），以及人工改进过度指定的测试用例。这种方法论上的严谨性值得学习。

### 6.2 用 Adaptive Reasoning 作为难度代理指标

利用 GPT-5.2 和 GPT-5.1 mini 的 adaptive reasoning 特性——模型会根据感知到的任务难度自动调整 reasoning token 用量——作为"任务是否变难了"的客观指标，这是一个巧妙的实验设计。它绕过了"成功率可能受随机性影响"的问题，从模型内部视角提供了额外证据。

### 6.3 控制变量的完整性

论文对以下维度进行了系统性的消融实验：
- Context file 来源（None / LLM / Human）
- 生成 context file 的模型（默认 vs. GPT-5.2）
- 生成 context file 的 prompt（Codex prompt vs. Claude Code prompt）
- Benchmark 类型（热门仓库 vs. 小众仓库）
- 文档可用性（保留文档 vs. 删除文档）

这种多维度的交叉验证使得结论的可靠性大大增强。

---

## 7. 论文的局限性

论文自身坦诚地讨论了几个局限：

### 7.1 仅评估了 Python

Python 是 LLM 训练数据中表示最充分的语言，模型对 Python 生态系统的工具链（pip、pytest、setuptools 等）已有丰富的参数知识。对于更小众的语言（如 Rust、Haskell、Zig），context files 可能更有价值，因为模型的先验知识不足。

### 7.2 仅评估了任务完成率

Context files 可能在其他维度上有价值：
- **代码安全性**：有研究表明，在 prompt 中要求安全编码可以显著提升代码安全性
- **代码效率**：context files 中的性能要求可能引导 agent 生成更高效的代码
- **代码风格一致性**：虽然不影响功能正确性，但对代码维护有价值

### 7.3 Benchmark 规模有限

AGENTbench 只有 138 个实例来自 12 个仓库。虽然论文的统计方法和跨 agent 一致性增强了结论的可信度，但更大规模的验证仍然需要。

---

## 8. 对我们的启示

### 8.1 重新审视我们的 AGENTS.md 实践

这篇论文直接挑战了当前的 best practice。对于我们自己的项目：

**立即行动**：
- 审查现有的 AGENTS.md / CLAUDE.md 文件，逐条评估：这条信息是 agent 无法自己发现的吗？
- 停止使用 `/init` 自动生成 context files——这些自动生成的内容弊大于利
- 如果仓库已有完善的 README 和 docs，考虑**完全移除** context file
- 对于已有的 context files，做一次"瘦身"：删除所有仓库结构描述、依赖列表、通用最佳实践

**保留什么（最小必需内容清单）**：
- 不可从代码推断的特殊工具链命令（如 `make proto` 编译 protobuf、自定义的 lint 命令）
- 安全红线（如"不要直接操作生产数据库"、"不要删除 migration 文件"）
- 仓库特有的 gotcha（如"测试必须在 Docker 中运行"、"CI 只在 Python 3.11 上跑"）
- 非标准的构建/部署流程（如果 `pip install -e .` 不够的话）
- 跨团队约定中不可从代码推断的部分（如"feature branch 必须基于 develop 分支"）

**坚决删除什么**：
- "本项目使用 Python 3.x"（pyproject.toml 里写着呢）
- "项目结构如下：src/ 包含源代码，tests/ 包含测试..."（agent 用 `ls` 就能看到）
- "请遵循 PEP 8 风格"（对任务完成没有帮助，反而增加约束）
- "使用 pytest 运行测试"（agent 看到 pytest.ini 或 pyproject.toml 就知道了）
- "提交前请运行 `pre-commit run --all-files`"（除非这是评估标准的一部分）

### 8.2 Context Engineering 的更深层启示

论文揭示的核心矛盾是：

> **信息 ≠ 有用信息。更多上下文 ≠ 更好的表现。**

这与 context engineering 领域的一个常见误区相关——人们倾向于"给 agent 更多信息总是好的"。但论文证明了：

1. **冗余信息有害**：重复已有文档的内容不仅无益，还增加了认知负担
2. **约束指令有成本**：每增加一条"你应该..."的指令，都在增加任务的复杂度
3. **Agent 会服从但不会判断**：agent 不会区分"对完成当前任务有帮助的指令"和"通用的好习惯但此刻无关的指令"

这对我们设计任何形式的 system prompt 或 context injection 都有启发：

> **每一条指令都有成本。只有当它的收益确定超过成本时才加入。**

### 8.3 Benchmark 方法论的借鉴

AGENTbench 的构建方法提供了一个模板：如何为新兴的 agent 实践创建评估 benchmark。特别是：

- 用 LLM 生成标准化 task description 的方法可以复用
- 用 LLM 生成单元测试 + 人工校验的 pipeline 是可扩展的
- "删除特定组件后再评估"的消融实验设计简单但强大

### 8.4 对 Agent 开发者的建议

如果你在构建 coding agent：

1. **不要默认推荐用户生成 context files**——至少在 context file 生成质量显著提升之前
2. **如果提供 `/init` 命令，应该生成极简的文件**——只提取真正非显而易见的信息
3. **考虑 context file 的选择性加载**——不是把整个文件注入上下文，而是根据当前任务动态选择相关章节
4. **训练模型区分"有用的上下文"和"噪音上下文"**——这可能是下一代 agent 的关键能力

### 8.5 我们的 AGENTS.md 反而是正面案例

回过头来看我们自己的 AGENTS.md（workspace 中的那份），它实际上已经**自觉地遵循了论文建议的最小化原则**：

- 没有冗长的仓库结构描述
- 核心内容是**流程和安全红线**，不是知识性描述
- 把详细信息分散到独立文件（TOOLS.md、HEARTBEAT.md），主文件只做索引
- 强调"上下文自律"——大任务丢子代理，回复简洁

这种设计哲学与论文的结论高度一致。但仍可以进一步精简：检查是否有任何内容是"agent 可以自己发现的"。

### 8.6 Context File 的未来演进方向

论文虽然对现状持悲观态度，但也指出了几个可能让 context files 真正有用的演进方向：

**方向一：任务感知的动态加载**
不把整个 context file 注入 agent 上下文，而是根据当前任务类型（bug fix、feature addition、refactoring）选择性地加载相关章节。这需要 agent harness 层面的支持，但可以显著减少冗余信息。

**方向二：从经验中学习的 context files**
类似 Dynamic Cheatsheet（Suzgun et al., 2025）的思路——agent 在完成任务后，将"对这个仓库有用的发现"写回 context file。这样 context file 就不再是静态的描述文档，而是**积累的经验知识库**。我们自己的 memory 系统（每日记录 → 长期记忆提炼）其实已经在做类似的事情。

**方向三：面向特定 agent 的定制**
论文发现开发者手写的通用 context file 在所有 agent 上都比 LLM 针对特定 agent 生成的文件表现更好。但未来可能有机会生成**真正针对特定 agent 弱点的补偿性指令**——不是告诉 agent "仓库长什么样"，而是告诉它"你在这类仓库中容易犯什么错"。

**方向四：负面指令（anti-patterns）**
论文没有探索，但一个有趣的假说是：**告诉 agent 不要做什么**可能比告诉它应该做什么更有效。例如："不要修改 migration 文件"、"不要运行完整测试套件（太慢了），只运行相关模块的测试"。这类指令直接减少了 agent 的搜索空间，而不是增加它需要遵循的约束。

---

## 9. 与其他相关工作的关系

### 9.1 Context Engineering 的实证化

这篇论文可以视为 context engineering 从"经验驱动"走向"实证驱动"的标志性工作。此前的文章（如 Chatlatanagulchai et al., 2025; Mohsenimofidi et al., 2025）主要是描述性研究（context files 里有什么），而这篇论文首次回答了"这些内容**有没有用**"。

### 9.2 与 Dynamic Cheatsheet 的对比

论文引用了 Suzgun et al. (2025) 的 Dynamic Cheatsheet 工作，提出了一个有趣的未来方向：context files 是否可以通过**持续学习**来自动优化？即 agent 在完成任务后，将学到的仓库知识写回 context file，形成正反馈循环。这与我们的"memory → 文件"设计理念相似。

### 9.3 AGENTS.md 生态的隐忧

论文没有直接说，但其结果暗示了一个更大的问题：如果 60,000+ 仓库的 context files 多数是用 LLM 自动生成的，那么这些文件不仅**没帮助**，还在**系统性地增加**使用这些仓库的 coding agent 的成本和错误率。这是一个值得整个社区关注的效率问题。

粗略估算影响规模：假设每个仓库平均每天有 10 次 agent 交互，成本增加 20%，基础成本 $0.50/次：
- 每天：60,000 × 10 × $0.50 × 20% = **$60,000/天的浪费**
- 每年：**~$22M 的行业级隐性成本**

这还不包括因成功率下降导致的生产力损失。

### 9.4 与 RAG 的类比

论文的发现与 RAG（Retrieval-Augmented Generation）领域的一些研究形成了有趣的呼应。RAG 研究中也发现了类似现象：

- 检索到的**不相关文档**会降低生成质量（类比：冗余的 context file 内容）
- **更多文档**不一定带来更好的结果（类比：更详尽的 context file 不一定更好）
- **精准检索**比**广泛检索**更有效（类比：最小化原则）

这暗示了一个更普遍的 LLM 使用原则：**上下文窗口是一种稀缺的注意力资源，每填入一个 token 都有机会成本。**

---

## 10. 总结

| 维度 | 核心发现 |
|------|---------|
| LLM 生成 context files 的效果 | 降低成功率 0.5-2%，增加成本 20%+ |
| 开发者手写 context files 的效果 | 微弱提升 ~4%，但也增加成本 |
| Context files 作为仓库导览 | 无效——不加速文件定位 |
| Agent 是否遵循指令 | 是——问题不在服从性，在指令本身 |
| 为什么有害 | 冗余信息 + 不必要约束 → 分散注意力 + 增加复杂度 |
| 何时有价值 | 仓库缺乏文档时，作为唯一文档来源 |
| 实践建议 | 停止自动生成；手写只写最小必需内容 |

**最深刻的教训**：

在 AI agent 的世界里，"多给信息"不等于"更好的结果"。Agent 是服从的——你给它指令，它就会遵循。但每一条指令都有机会成本。当 agent 把认知资源花在"遵循 context file 中的风格指南"上时，它就少了一些资源去"找到并修复真正的 bug"。

**最小化原则不是懒惰，而是智慧**——只告诉 agent 它无法自己发现的、对完成任务真正必要的信息。

**一个思维实验**：如果你要给一个新加入的**资深工程师**介绍你的仓库，你会花 30 分钟讲"我们用 Python，测试用 pytest，代码在 src/ 目录下"吗？不会——你只会说"注意，我们的 CI 在 Jenkins 上跑，不是 GitHub Actions"、"数据库 migration 不要用 auto-generate，要手写"这种**只有内部人才知道的、违反常识预期的**信息。

对 agent 也应该这样。它已经是"资深工程师"了——它见过几百万个 Python 项目。你只需要告诉它：你的项目**哪里不一样**。

---

*论文链接：[arXiv:2602.11988](https://arxiv.org/abs/2602.11988)*
*代码仓库：[eth-sri/agentbench](https://github.com/eth-sri/agentbench)*

---

## See Also

- [[AI/Agent/Agent 评测与 Benchmark|Agent 评测与 Benchmark]] — 评测方法全景
- [[AI/Agent/Gaia2-Dynamic-Async-Agent-Benchmark|Gaia2]] — 2026 标杆 benchmark：动态异步环境
- [[AI/Agent/Code Agent|Code Agent]] — AGENTS.md / context file 对 coding agent 的影响是本篇核心
- [[AI/Agent/目录|Agent MOC]] — Agent 知识全图谱
