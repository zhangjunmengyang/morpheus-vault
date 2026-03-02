---
title: OpenClaw 宕机复盘：从“Cloudflare 误判”到“三层组合故障”
type: postmortem
date: 2026-02-28
tags: [openclaw, postmortem, watchdog, reliability, incident]
---

## TL;DR

这次故障并不是单点根因（不是“单纯 Cloudflare 拦截”），而是三层问题叠加：

1. **链路层**：本地代理/网络抖动，导致 fetch 间歇失败
2. **控制层**：watchdog 自动重启过于激进，形成重启风暴
3. **配置层**：`channels.discord.enabled` 与代理相关配置在恢复阶段发生变更/热重载，放大了恢复期不稳定

`fetch failed` 是“症状”，不是唯一“病因”。

---

## 影响范围

- 主要影响：Discord 侧响应间歇失败，老板体感为“掉线/不稳定”
- 伴随现象：模型请求 timeout、会话锁冲突（`session file locked`）、gateway loopback timeout
- 影响时段：以日志时间线为准（2/28 下午至晚间多次抖动）

---

## 关键证据（日志）

### 1) 配置层热重载/恢复
- `channels.discord.enabled` 在恢复阶段被打开并触发热重载
- 随后多 agent 登录成功（说明配置层后续恢复有效）

### 2) 控制层重启风暴
- 大量 SIGTERM / 重启记录（短周期内重复）
- watchdog 日志出现连续：网络断开 → Discord 不可达 → gateway-unhealthy → 重启

### 3) 链路层抖动
- `fetch failed` 大量出现
- 同时存在 `gateway timeout ws://127.0.0.1:18789`，说明不仅是外部 API，也存在本地网关链路拥塞/阻塞

### 4) 并发放大器
- 多条 `session file locked (timeout 10000ms)`
- 导致“All models failed”表面像“模型全挂”，实际部分是本地会话锁竞争

---

## 根因归因（最终）

### 真正主因：三层组合故障

不是“Cloudflare 单点问题”，而是：

- **链路抖动**先触发 fetch 异常
- **watchdog 无退避策略**把瞬时异常放大为重启风暴
- **配置层热变更**叠加在恢复窗口内，进一步提高系统不稳定性

### 为什么会误判成 Cloudflare？

诊断时把 `curl https://chatgpt.com/backend-api/` 的 challenge 结果，当成了“端到端必失败”的充分条件。

这个探针只能说明“某一路径、某一时刻”异常，不代表 OpenClaw 实际全链路运行状态。

---

## 教训

1. **先画时间线，再下结论**
   - 先合并配置事件、watchdog动作、网络信号、业务成功信号（agent 登录/回复）
2. **优先端到端信号，不迷信单探针**
   - `res ✓ agent`、实际频道登录成功 > 单次 curl 失败
3. **诊断窗口先冻结自动重启器**
   - 避免“故障-重启-再故障”自激回路
4. **分层判断**
   - 配置问题（开关/热重载）与链路问题（代理/网络）必须分开

---

## 已执行修复（当晚）

对 `~/.openclaw/scripts/network-watchdog.sh` 做了稳定性升级：

### v2.3 新增能力

1. **诊断冻结开关**
   - `~/.openclaw/watchdog/diagnostic.freeze` 文件存在时，watchdog 只观测不重启

2. **重启预算限流**
   - 15 分钟窗口内最多重启 3 次，超过后跳过，避免风暴

3. **gateway-unhealthy 退避重启**
   - 对 `gateway-unhealthy` 原因加入 backoff（30s 起步，上限 300s）

4. **语法校验通过**
   - `bash -n ~/.openclaw/scripts/network-watchdog.sh` 通过

---

## 后续行动（P0/P1）

### P0（立即）
- [ ] 将“配置变更 5 分钟自动回滚”上线（commit-confirmed 模式）
- [ ] 变更前后引入 canary：gateway loopback + OpenAI/Discord e2e 探活
- [ ] 统一 gateway 单入口（仅 LaunchAgent 管控），禁止多源并发拉起

### P1（本周）
- [ ] 发布 `openclaw-watchdog` 开源仓库（含 README、故障模式、配置模板）
- [ ] 增加“故障分层仪表”：链路层/控制层/配置层三色告警
- [ ] 增加自动事故归档脚本（按时间窗口收集证据）

---

## 最后

这次最重要的不是“找到了某个坏节点”，而是建立了可靠的诊断方法：

> **时间线优先，端到端优先，分层归因，控制器先去激进化。**

只有这样，下次系统波动时，我们不会再被“看起来像主因”的局部信号带偏。
