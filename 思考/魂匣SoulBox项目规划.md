---
title: 魂匣 SoulBox 项目规划
type: project
date: 2026-02-15
updated: 2026-02-27
tags:
  - soulbox
  - agent-persona
  - project-plan
---
# 魂匣 — Agent Persona Platform

> 一个统一平台，售卖 Agent 人格包 + 安全工具。私有仓库，商业化运营。

---

## 战略

**不做散装开源，做统一平台产品。**

```
Phase 1: 基建 → 搭平台骨架、内容管线、人格标准
Phase 2: 内容 → 持续研发人格包、安全工具
Phase 3: 运营 → 小红书矩阵部署 + 运营策略
Phase 4: 商业化 → 定价上线、付费体系
```

### 定位

| 层级 | 代表 | 我们的位置 |
|------|------|-----------|
| 底层框架 | OpenClaw / SillyTavern / CrewAI | ↓ 下游 |
| **中层平台** | **魂匣** | **← 我们在这** |
| 顶层封闭 | Character.AI / Chub.ai | ↑ 上游 |

中层平台 = 内容可跨框架适配 + 可独立售卖 + 不被任何框架绑死

---

## 仓库结构（私有 monorepo）

```
soulbox/                          # 魂匣平台
├── README.md                     # 项目说明
├── packages/                     # 人格包（核心产品）
│   ├── sentinel/                 # 信息哨兵
│   │   ├── soul/                 # SOUL.md + AGENTS.md + HEARTBEAT.md + IDENTITY.md
│   │   ├── memory/               # bootstrap.md
│   │   ├── config/               # agent.json
│   │   └── metadata.yaml         # 包元数据（定价、标签、版本、兼容性）
│   ├── alfred/                   # 生活管家
│   ├── scholar/                  # 研究学者
│   ├── quant/                    # 量化分析师
│   ├── devops/                   # 运维守卫
│   └── ...                       # 持续新增
├── shield/                       # 安全工具（附加产品）
│   ├── memory_guard.py
│   ├── injection_detect.py
│   ├── rules/
│   └── research/                 # 安全研究（内部）
├── platform/                     # 平台基建
│   ├── schema/                   # 人格包标准格式定义
│   │   └── persona-pack.schema.yaml
│   ├── validator/                # 包格式校验工具
│   ├── installer/                # 一键安装到各框架
│   │   ├── openclaw.sh           # 安装到 OpenClaw
│   │   ├── sillytavern.sh        # 导出为酒馆角色卡
│   │   └── crewai.py             # 导出为 CrewAI YAML
│   └── catalog/                  # 包目录生成
│       └── build-catalog.py
├── research/                     # 研究资料（内部）
│   ├── market-analysis.md
│   ├── agent-personality-science.md
│   └── personality-philosophy.md
├── marketing/                    # 运营素材
│   ├── xiaohongshu/              # 小红书内容模板
│   ├── assets/                   # 图片、头像
│   └── copy/                     # 文案库
└── docs/                         # 内部文档
    ├── CONTRIBUTING.md            # 内容研发规范
    ├── PRICING.md                 # 定价策略
    └── ROADMAP.md                 # 路线图
```

---

## Phase 1: 基建（本周）

### 1.1 平台骨架
- [ ] 创建私有 GitHub 仓库 `soulbox`
- [ ] 搭建 monorepo 目录结构
- [ ] 定义 `persona-pack.schema.yaml`（人格包标准格式）
- [ ] 写 `validator`（校验包格式是否合规）

### 1.2 内容管线
- [ ] 迁移已有 sentinel-pack → `packages/sentinel/`
- [ ] 迁移已有 alfred-pack → `packages/alfred/`
- [ ] 迁移 shield 研究 → `shield/research/`
- [ ] 迁移 market-analysis → `research/`

### 1.3 人格标准
- [ ] 定义 `metadata.yaml` schema（名称、版本、定价、标签、兼容框架、依赖）
- [ ] 定义包目录结构规范（soul/ + memory/ + config/ 必须项）
- [ ] 定义质量标准（SOUL.md 最小深度、HEARTBEAT.md 必须有反空转）

### 1.4 跨框架适配
- [ ] OpenClaw 安装脚本（已有）
- [ ] SillyTavern 角色卡导出器（PNG V2 格式）
- [ ] CrewAI YAML 导出器

---

## Phase 2: 内容研发（持续）

### 人格包路线
| 包 | 定位 | 免费/付费 | 优先级 |
|----|------|----------|--------|
| sentinel | 信息哨兵 | 免费样品 | ✅ 已完成 |
| alfred | 生活管家 | 免费样品 | ✅ 已完成 |
| scholar | 研究学者 | 免费 | P1 |
| quant | 量化分析 | 付费 | P1 |
| devops | 运维守卫 | 免费 | P2 |
| writer | 内容创作 | 付费 | P2 |
| social | 社交运营 | 付费 | P3 |

### 安全工具路线
| 模块 | 功能 | 优先级 |
|------|------|--------|
| Memory Guard | 灵魂文件防篡改 | ✅ 原型完成 |
| Injection Detector | 注入检测 | P1 |
| Canary Traps | 金丝雀陷阱 | P2 |
| Behavior Auditor | 行为审计 | P2 |

---

## Phase 3: 运营（基建完成后）

### 小红书矩阵
- 账号定位：AI Agent 生产力 / AI 管家 / Agent 安全
- 内容类型：教程、人格展示、使用场景、对比评测
- 发布节奏：待定

### 运营策略
- 免费包引流 → 付费包转化
- 社区互动 → 需求收集 → 定制服务
- 技术博客 → SEO + 专业形象

---

## Phase 4: 商业化（运营验证后）

### 定价模型
| 层级 | 价格 | 内容 |
|------|------|------|
| 免费 | ¥0 | 2-3 个基础包（引流） |
| Pro | ¥29-69/个 | 垂直领域深度包 |
| Bundle | ¥99-199/套 | 5-8 个包 + 安全工具 |
| 定制 | ¥499+ | 企业/个人定制人格 |

---

## 待办

- [ ] ⚠️ 删除已创建的 3 个公开仓库（需 `gh auth refresh -s delete_repo`）
  - sentinel-pack / alfred-pack / agent-sentinel-shield
- [ ] 创建私有仓库 `soulbox`
- [ ] 搭建 monorepo + 迁移已有产出

---

_此文件是活的。每次推进后更新。_
