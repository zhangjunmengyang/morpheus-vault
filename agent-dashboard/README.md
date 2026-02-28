# Agent 进度看板（方案 A）

你要的就是“像示例一样直接看 HTML”，不绑 Pages，最快部署为：

- 看板页面落盘到知识库仓库：`/Users/peterzhang/project/morpheus-vault/agent-dashboard/`
- 访问方式（公开仓库）：
  - `https://htmlpreview.github.io/?https://github.com/zhangjunmengyang/morpheus-vault/blob/main/agent-dashboard/index.html`

## 一次性生成 + 覆盖写入（推荐）

```bash
cd /Users/peterzhang/.openclaw/workspace/reports/agent-dashboard
AGENT_DASHBOARD_SITE_DIR=/Users/peterzhang/project/morpheus-vault/agent-dashboard \
AGENT_DASHBOARD_SITE_REPO_ROOT=/Users/peterzhang/project/morpheus-vault \
AGENT_DASHBOARD_SKIP_COMMIT=1 AGENT_DASHBOARD_SKIP_PUSH=1 \
AGENT_DASHBOARD_AUTOPUSH=0 python3 /Users/peterzhang/.openclaw/workspace/reports/agent-dashboard/sync_to_githubio.sh
```

说明：
- `SKIP_COMMIT=1`：本地直接同步文件，不打 commit
- `SKIP_PUSH=1`：不推送到任何仓库（你要手动提交）

## 自动更新（本机 5 分钟）

```bash
AGENT_DASHBOARD_SITE_DIR=/Users/peterzhang/project/morpheus-vault/agent-dashboard \
AGENT_DASHBOARD_SITE_REPO_ROOT=/Users/peterzhang/project/morpheus-vault \
bash /Users/peterzhang/.openclaw/workspace/reports/agent-dashboard/install_auto_update.sh
```

启动后会使用 LaunchAgent `ai.openclaw.agent-dashboard`：
- 默认每 5 分钟执行一次
- 仅更新 `agent-dashboard/index.html`、`agent-dashboard-data.json`、`README.md`
- 默认不推送（安全）

## 数据来源

- `shared/bulletin.md` 的任务区（当前看板以任务区主）
- 各 Agent workspace 的 `heartbeat-state.json` / `memory/heartbeat-state.json`

## 任务健康策略

- 最近 120 分钟内：在线
- 121-180 分钟：偏慢
- 180 分钟以上：告警
