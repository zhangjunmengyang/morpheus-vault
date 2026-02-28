# Agent 进度看板

- 运行 `python3 generate_dashboard.py` 重新生成数据与页面
- 打开 `agent-dashboard.html` 进行管理

说明：
- 数据来源：
  - `shared/bulletin.md` 的「活跃任务」区
  - 各 workspace 的 `heartbeat-state.json`
- 任务健康策略：
  - 最近 120 分钟内：在线
  - 121-180 分钟：偏慢
  - 180 分钟以上：告警
