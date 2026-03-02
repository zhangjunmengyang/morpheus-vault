---
title: "Research Factory Protocol: File-as-API (Workspace as Shared Memory)"
brief: "用共享文件系统作为 multi-agent 异步协议：project workspace 目录、状态机、证据链与工件规范。适用于 FARS-style 自动科研与内部平台化研究。"
date: 2026-03-01
tags:
  - agent-system
  - multi-agent
  - workflow
  - protocol
  - ai4science
status: draft
related:
  - "[[AI/6-应用/FARS-Fully-Automated-Research-System]]"
---

# Research Factory Protocol: File-as-API

## 1. 核心思想
把 workspace 当成共享记忆与异步消息总线：Agent 之间不直接 message passing，而是读写约定文件（file-as-API）。

## 2. 目录与文件契约
```
projects/<project_id>/
  status.json          # 状态机
  hypothesis.md        # Ideation 输出
  plan.md              # Planning 输出
  run_specs/<run_id>.json
  artifacts/<run_id>/  # 产物目录（不可变）
  scorecards/<run_id>.json
  report.md            # Writing 输出
  links.json           # 证据链与实体关系
```

### status.json（最小字段）
- stage
- owners (agent ids)
- created_at / updated_at
- last_run_id
- blockers[]

### links.json（证据链）
- hypothesis -> plan -> runs -> artifacts -> scorecards -> experiences

## 3. 不可变性规则（最重要）
- artifacts/<run_id>/ 一旦生成禁止覆盖（只追加补丁文件并记录原因）
- run_specs/<run_id>.json 必须包含 data_version + code_version + config_hash

## 4. 并发规则
- 同一 project 允许并发多个 run，但必须在 status.json 中登记 run queue
- 写文件采用原子写（write tmp → rename）

## 5. 失败也要结构化
失败不等于异常退出：
- 失败必须产出 scorecard（失败原因 taxonomy）
- 失败必须能被后续 query/聚合（Failure Museum）
