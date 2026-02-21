---
title: "Code Agent 与 Computer Use"
date: 2026-02-14
tags: [agent, coding, computer-use, interview]
type: note
domain: AI/Agent
rating: 4
status: evergreen
---

# Code Agent 与 Computer Use Agent

## 1. Code Agent 定义

Code Agent 是一类以 **LLM 为核心推理引擎**，能够自主完成软件工程任务的 [[AI/Agent/_MOC|AI Agent]]。与传统代码补全（Copilot 行级补全）不同，Code Agent 具备：

- **端到端任务执行**：从需求理解 → 代码定位 → 编辑 → 测试 → 修复的完整闭环
- **工具使用**：调用文件系统、终端、搜索、浏览器等外部工具
- **多步推理**：将复杂任务分解为子步骤，根据中间结果动态调整策略
- **自我修正**：通过执行结果（编译错误、测试失败）反馈迭代

### Code Agent vs Code Completion

| 维度 | Code Completion | Code Agent |
|------|----------------|------------|
| 粒度 | 行/块级补全 | 任务级（issue → PR） |
| 交互 | 被动（等用户触发） | 主动（规划+执行） |
| 上下文 | 当前文件 ± 少量引用 | 全 repo + 文档 + 终端 |
| 反馈循环 | 无 | 执行→观察→调整 |
| 典型延迟 | 毫秒级 | 分钟级 |

---

## 2. 代表产品

### 2.1 GitHub Copilot (Agent Mode)

> 深度技术细节见 [[AI/Agent/GitHub-Agentic-Workflows|GitHub Agentic Workflows 深度分析]]

- **定位**：从行级补全进化为 IDE 内 Agent（2025 起 Agent Mode）
- **架构**：VS Code 集成，调用 GPT-4o / Claude 3.5 Sonnet
- **能力**：Copilot Workspace 支持 issue → plan → code → PR 流程
- **Copilot Agent**（2025）：自动修复 CI 失败、处理 issue、创建 PR
- **限制**：依赖 GitHub 生态，自定义工具有限

### 2.2 Cursor

- **定位**：AI-native IDE（VS Code fork）
- **核心功能**：
  - **Composer**：多文件编辑的 Agent 模式，理解全 repo 上下文
  - **Tab**：智能补全，理解编辑意图（预测下一个编辑位置）
  - **Cmd+K**：内联编辑，选中代码直接自然语言修改
  - **Chat**：带 codebase 索引的对话（@codebase 语义搜索）
- **技术亮点**：
  - 自研 codebase indexing（向量索引 + 关键词 + 代码图谱）
  - Speculative edits：预测用户下一步编辑
  - 支持切换模型（Claude/GPT-4o/自定义 API）
- **Bug Finder**（2025）：后台异步扫描 repo 发现潜在 bug

### 2.3 Devin (Cognition AI)

- **定位**：首个"AI 软件工程师"，完全自主的 Code Agent
- **架构**：
  - 独立沙盒环境（浏览器 + 终端 + 编辑器）
  - 基于 planner-executor 模式，长期任务管理
  - 支持 Slack 交互，异步执行
- **能力**：独立完成 issue、学习新框架、部署应用
- **SWE-bench 表现**：早期声称 13.86%（2024-03），引发行业关注
- **争议**：实际生产能力远低于 demo，幻觉问题严重

### 2.4 Claude Code (Anthropic)

- **定位**：终端原生 agentic coding 工具
- **架构**：
  - CLI 界面，直接运行在用户终端
  - 调用 Claude Sonnet/Opus 作为后端模型
  - 工具集：Read/Write/Edit/Bash/Search（可配置权限）
- **核心设计**：
  - **无 RAG**：依赖模型的长上下文（200k tokens）+ 按需读取文件
  - **权限模型**：allowedTools 白名单，用户显式授权危险操作
  - **CLAUDE.md**：项目级上下文文件，类似 .cursorrules
  - **子 Agent**：复杂任务自动拆分子任务（TodoWrite → subtask）
- **优势**：无需 IDE，可集成到 CI/CD pipeline，支持 headless 模式
- **SWE-bench Verified**：Claude Sonnet 在 agentic 设置下达 ~72%（2025）

### 2.5 其他重要产品

- **Windsurf (Codeium)**：AI IDE，Cascade agent 模式，深度 codebase 理解
- **Aider**：开源终端 Code Agent，支持 git 集成，多模型切换
- **OpenHands (原 OpenDevin)**：开源 Code Agent 平台，支持自定义 Agent
- **Amazon Q Developer**：AWS 生态 Code Agent，CLI agent 模式
- **Codex (OpenAI)**：云端异步 Code Agent（2025），沙盒执行

---

## 3. Code Agent 核心能力

### 3.1 代码理解与检索

- **Codebase Indexing**：对仓库建立语义索引（embedding + AST + 符号表）
- **上下文构建**：从 issue 描述 → 定位相关文件 → 构建最小充分上下文
- **技术**：向量搜索 + BM25 + 代码图谱（call graph / dependency graph）

### 3.2 代码编辑

- **Search-Replace 模式**：Claude Code 的 Edit 工具，精确匹配 + 替换
- **Diff 模式**：Cursor 的 apply model，生成 unified diff 后应用
- **挑战**：大文件编辑容易丢失上下文、缩进错误、合并冲突

### 3.3 测试与验证

- 运行现有测试套件验证修改正确性
- 自动生成测试用例覆盖新功能
- 通过测试结果反馈修正代码（test-fix loop）

### 3.4 规划与分解

- 将复杂任务分解为可执行步骤
- 动态调整计划（根据中间结果）
- 进度追踪与状态管理

### 3.5 环境交互

- 终端命令执行（安装依赖、构建项目、运行服务）
- 浏览器使用（查看文档、调试 web UI）
- API 调用（GitHub API、CI/CD 系统）

---

## 4. SWE-bench 评估流程

### 什么是 SWE-bench

> 代码生成能力的系统性全景见 [[AI/LLM/Application/LLM代码生成-2026技术全景|LLM 代码生成 2026 技术全景]]

- Princeton 发布的 Code Agent benchmark
- 从 12 个 Python 开源仓库的真实 GitHub issue 构建
- 每个实例：issue 描述 + 对应代码库快照 → Agent 生成 patch → 运行测试验证
- **SWE-bench Lite**：300 个精选实例；**SWE-bench Verified**：500 个人工验证实例

### 标准评估流程

```
1. 输入：issue 描述 + repo snapshot (git clone)
2. Agent 阶段：
   a. 阅读 issue，理解需求
   b. 探索代码库（搜索、阅读文件、理解结构）
   c. 定位相关代码（通常 1-5 个文件）
   d. 编写修复 patch
   e. 运行测试验证
   f. 迭代修正（如果测试失败）
3. 输出：git diff (patch)
4. 评估：应用 patch → 运行 gold test → pass/fail
```

### 典型 Agent 架构（以 SWE-agent 为例）

- **ACI (Agent-Computer Interface)**：为 LLM 设计的简化命令集（open/edit/search/submit）
- **思考-行动循环**：ReAct 式 thought → action → observation
- **上下文窗口管理**：滑动窗口显示文件内容，避免上下文溢出
- **限制**：通常设定 cost limit（如 $3/instance）或 step limit（如 50 步）

### 当前排行（2025-2026 截至写作时）

| 系统 | SWE-bench Verified |
|------|-------------------|
| Claude Code (Opus) | ~72% |
| Amazon Q Developer | ~68% |
| OpenHands + Claude | ~65% |
| Devin v2 | ~55% |
| SWE-agent + GPT-4o | ~33% |

> 数字持续变化，仅供参考量级。

---

## 5. Computer Use Agent

### 定义

Computer Use Agent 是能够通过 **GUI 操作**（鼠标点击、键盘输入、截图识别）来控制计算机的 AI Agent。与 Code Agent 通过 API/CLI 交互不同，Computer Use Agent 模拟人类的视觉-操作循环。

### 核心技术

1. **屏幕理解**：
   - 截图 → 视觉模型识别 UI 元素（按钮、输入框、菜单）
   - Set-of-Mark (SoM)：在截图上标注可交互元素的编号
   - OCR + 布局理解：识别文字内容和空间关系

2. **动作空间**：
   - 鼠标：移动、左/右/双击、拖拽、滚动
   - 键盘：输入文本、快捷键组合（Ctrl+C 等）
   - 高级：截图、等待、坐标定位

3. **推理循环**：
   ```
   截图 → 视觉理解 → 决策（下一步操作）→ 执行动作 → 截图 → ...
   ```

### 代表系统

| 系统 | 特点 |
|------|------|
| **Claude Computer Use** | Anthropic 官方，Claude 3.5 Sonnet 原生支持，tool-based API |
| **OpenAI Operator** | GPT-4o 驱动，Web 浏览 Agent，CUA (Computer-Using Agent) 模型 |
| **OS-Copilot** | 开源框架，支持 macOS/Linux/Windows GUI 操作 |
| **UFO (Microsoft)** | Windows UI 自动化，基于 UI Automation API + 视觉模型 |
| **Open Interpreter** | 开源，混合 CLI + GUI 操作 |

### Claude Computer Use 详解

- **工具定义**：`computer`（截图+鼠标+键盘）、`text_editor`（文件编辑）、`bash`（终端）
- **截图分辨率**：推荐 1280×800（XGA），缩放到模型可处理的分辨率
- **坐标系统**：像素级坐标定位，模型输出 (x, y) 坐标
- **典型流程**：
  1. 发送 `screenshot` 动作获取当前屏幕
  2. 模型分析截图，决定操作（如"点击搜索框"）
  3. 返回 `mouse_move` + `left_click` 到具体坐标
  4. 执行后再次截图验证结果
- **限制**：延迟较高（每步需截图+推理 ~2-5s）、坐标精度有限、不稳定

### Computer Use 的应用场景

- **RPA (Robotic Process Automation)**：自动化企业内部系统操作
- **测试自动化**：GUI 测试，替代 Selenium/Appium 脚本
- **遗留系统集成**：无 API 的老旧系统通过 GUI 操作桥接
- **辅助功能**：帮助残障用户操作电脑
- **数据录入**：跨系统数据迁移（无 API 时）

---

## 6. Agentic Engineering

### 概念

Agentic Engineering 是一种新的软件开发范式，核心理念是 **人类作为架构师和审核者，AI Agent 作为主要实现者**。不是简单的"AI 辅助编程"，而是围绕 Agent 能力重新设计工作流。

### 核心原则

1. **Specification-Driven**：花更多时间写清楚需求（spec/PRD），而非直接写代码
2. **Agent-Friendly Codebase**：
   - 完善的 CLAUDE.md / .cursorrules / AGENTS.md
   - 清晰的目录结构和命名规范
   - 充分的测试覆盖（Agent 依赖测试验证）
   - 类型标注（帮助 Agent 理解代码约束）
3. **Review over Write**：人类从"写代码"转向"审代码"
4. **Iterative Delegation**：先让 Agent 做，根据结果反馈调整，而非一次性完美指令
5. **Guardrails**：权限控制、沙盒执行、CI gate、human-in-the-loop

### 工作流模式

```
Human: 需求/Issue → Spec
  ↓
Agent: 探索 → 规划 → 实现 → 测试
  ↓
Human: Review → Approve/Feedback
  ↓
Agent: 修正 → 合并
  ↓
CI/CD: 自动部署
```

### 实践建议

- **小任务高频迭代** > 大任务一次性交付
- **维护上下文文件**：CLAUDE.md/AGENTS.md 是投资回报最高的文档
- **测试是 Agent 的眼睛**：没有测试 = Agent 盲人摸象
- **版本控制是安全网**：让 Agent 在 branch 上工作，merge 前 review
- **选择合适的 Agent 粒度**：行级用 Copilot，文件级用 Cursor，任务级用 Claude Code/Devin

---

## 7. 面试题

### Q1: Code Agent 的核心技术挑战是什么？如何应对？

**答**：主要挑战：

1. **上下文长度限制**：真实仓库可能有数百万行代码，远超模型上下文窗口。应对：① 建立代码索引（语义搜索 + AST 分析），按需检索相关片段；② 分层上下文（仓库结构 → 目录概览 → 相关文件 → 具体函数）；③ 上下文压缩（只保留最近 N 步的工具输出）。

2. **幻觉与代码正确性**：模型可能生成不存在的 API、错误的逻辑。应对：① 测试驱动（生成代码后立即运行测试）；② 类型检查和 lint 作为快速反馈；③ 限制修改范围（小 patch 比大重构可靠）。

3. **长期规划**：多步任务中容易偏离目标或陷入循环。应对：① 显式的任务分解和进度追踪（TodoWrite）；② 设置 step/cost 上限；③ human-in-the-loop checkpoint。

4. **环境状态管理**：文件系统、git 状态、运行中的服务等外部状态难以完整追踪。应对：① 每步后刷新状态（re-read 文件、git status）；② 沙盒隔离防止破坏；③ 快照/回滚机制。

### Q2: Computer Use Agent 与传统 RPA（如 UiPath）相比，优劣势是什么？

**答**：

**优势**：
- **无需预定义流程**：传统 RPA 需要人工录制/编写自动化脚本，Computer Use Agent 通过自然语言指令即可操作
- **自适应 UI 变化**：基于视觉理解而非 DOM/Accessibility API 的硬编码选择器，UI 布局变化后仍可能正常工作
- **泛化能力**：同一个 Agent 可以操作不同应用，无需为每个应用单独开发
- **处理异常情况**：遇到弹窗、错误提示等非预期情况可自主判断应对

**劣势**：
- **速度慢**：每步需截图+推理（2-5s），传统 RPA 毫秒级操作
- **不确定性**：LLM 推理不确定，同一任务可能有不同执行路径和结果
- **成本高**：每步调用多模态 LLM，API 成本远高于传统 RPA
- **精度问题**：坐标定位可能有偏差，小按钮/密集 UI 容易点错
- **无事务性**：传统 RPA 可以回滚，Computer Use Agent 操作不可逆

**结论**：Computer Use Agent 适合低频、非结构化、跨应用的任务；传统 RPA 适合高频、确定性、关键业务流程。未来趋势是融合——Agent 做规划和异常处理，RPA 做高速执行。

### Q3: SWE-bench 成绩从 ~5%（2024 初）提升到 ~70%（2025），核心突破是什么？

**答**：三个层面的突破：

1. **模型能力飞跃**：
   - 长上下文支持（128k→200k），可以一次性阅读多个大文件
   - 工具使用能力（tool use）大幅提升，模型能精确调用 search/edit/bash
   - 代码推理能力（尤其是 Claude Sonnet/Opus 和 GPT-4o 级别的模型）

2. **Agent 架构优化**：
   - 从简单的 ReAct loop → 多阶段流水线（定位 → 编辑 → 验证 → 修复）
   - 更好的上下文管理（按需读取 vs. 全部塞入）
   - 结构化工具设计（ACI - Agent-Computer Interface）
   - 测试驱动的反馈循环（编辑后立即 run test）

3. **基础设施成熟**：
   - 沙盒环境标准化（Docker 容器、安全隔离）
   - 成本降低使得多次 retry/ensemble 成为可能
   - 社区积累了大量 prompt engineering 和 tool design 经验

**关键 insight**：不是单一突破，而是模型+架构+工程的协同进化。最大的单一贡献者可能是模型的 tool use 能力——让 Agent 能可靠地调用工具是一切的基础。

### Q4: 如何设计一个 Code Agent 的权限和安全模型？

**答**：分层安全模型：

1. **工具级权限**：
   - 白名单模式：明确列出 Agent 可使用的工具（如 Claude Code 的 `allowedTools`）
   - 分级：只读工具（Read/Search）自动放行，写入工具（Write/Edit）需确认，危险工具（Bash/rm）需显式授权
   - 命令过滤：Bash 工具可限制正则（如禁止 `rm -rf /`、`curl | sh`）

2. **文件系统隔离**：
   - 工作目录白名单：Agent 只能读写项目目录
   - 敏感文件保护：.env、secrets、credentials 目录只读或禁止访问
   - 临时文件隔离：Agent 创建的文件在沙盒内

3. **网络隔离**：
   - 默认禁止外网访问（或仅允许特定域名）
   - 禁止 Agent 自行安装系统包或下载可执行文件

4. **执行隔离**：
   - Docker/虚拟机沙盒（Devin、Codex 的做法）
   - 进程级别的 seccomp/AppArmor 限制
   - 资源限制：CPU/内存/磁盘/网络带宽/执行时间

5. **审计与回滚**：
   - 所有操作日志记录（工具调用、文件修改、命令执行）
   - Git 作为天然回滚机制（每步 commit）
   - Human-in-the-loop：关键节点暂停等待人类审核

### Q5: Agentic Engineering 会如何改变软件工程师的角色？需要培养哪些新技能？

> 相关：[[AI/Agent/Fundamentals/Agent 生产实践|Agent 生产实践]] | [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 2026 全景]]（底层训练方法论）

**答**：

**角色转变**：
- **从实现者到架构师/审核者**：更多时间花在系统设计、需求定义、代码审查上
- **从写代码到写 Spec**：清晰的需求描述成为核心技能（prompt engineering 的延伸）
- **从手动操作到编排 Agent**：管理多个 Agent 的协作、分配任务、合并结果

**需要培养的新技能**：
1. **Spec Writing**：能把模糊需求转化为 Agent 可执行的清晰规格说明
2. **Codebase Curation**：维护对 Agent 友好的代码库（文档、测试、类型、结构）
3. **Agent Orchestration**：了解不同 Agent 的能力边界，知道何时用何种工具
4. **Review at Scale**：快速审查 AI 生成的大量代码，识别潜在问题
5. **系统思维**：在更高抽象层面设计系统，而非陷入实现细节
6. **测试策略**：设计更全面的测试作为 Agent 的质量保障
7. **安全意识**：理解 Agent 操作的安全风险，设计合适的 guardrails

**不会消失的**：系统设计能力、领域知识、调试复杂问题的能力、与人沟通协作。

**底线**：Agentic Engineering 提高了工程师的杠杆率，但也提高了对 judgment（判断力）的要求。写代码的门槛降低，但设计好系统、做出正确决策的门槛没有降低。
