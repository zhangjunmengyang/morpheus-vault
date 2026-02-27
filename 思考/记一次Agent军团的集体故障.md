---
title: 记一次 Agent 军团的集体故障
type: 思考
date: 2026-02-27
tags:
  - agent
  - 运维
  - 思考
  - openclaw
  - 故障复盘
---
# 记一次我的 Agent 军团的集体故障：当你的 AI 管家们集体失联 5 小时

> 2026年2月27日凌晨，我的 8 个 AI Agent 同时失联了 5 小时。没有任何报警。我是在上午打开 Discord 时才发现的——满屏的 "LLM request timed out."，整整 42 条。

## 先说背景：我有一支 AI 军团

我在一台 MacBook M1 Pro 上跑着一套完整的 AI Agent 系统——基于开源项目 OpenClaw 搭建。包括：

- **J.A.R.V.I.S.**（总管）—— 调度其他 Agent，处理我的日常任务
- **哨兵 Sentinel** —— 实时情报扫描，每天早上自动生成新闻日报
- **学者 Scholar** —— 论文精读，维护我的 AI 知识库
- **馆长 Librarian** —— 知识结构维护，笔记质量审计
- 还有数据科学家、量化分析师等若干专职 Agent

它们 24 小时在线，通过心跳机制（heartbeat）定期醒来干活——扫新闻、写笔记、检查系统状态、给我推送重要信息。

**直到今天凌晨，它们集体罢工了。**

---

## 故障现场：42 次 Timeout

早上打开 Discord，我看到的是这样的画面：

```
LLM request timed out.
LLM request timed out.
LLM request timed out.
...
```

连续 42 次。时间跨度从凌晨 01:16 到 05:51，整整接近 5 个小时。

每一个 Agent 的每一次心跳，都在尝试调用 LLM API——然后超时。没有一次成功。

**更令人不安的是：没有任何报警通知我。**

我是在上午主动打开 Discord 时才发现的。如果我没有检查，这些 Agent 可能会一直这样空转下去。

---

## 排查过程：从表象到根因

### 第一步：看日志时间线

先拉 Gateway 日志（OpenClaw 的核心进程）：

```
2026-02-27T00:57:37  Gateway PID 3308 启动
2026-02-27T00:57:44  收到 SIGTERM，进程被杀（启动仅 7 秒）
2026-02-27T00:57:51  PID 3600 启动
2026-02-27T00:59:47  收到 SIGINT，进程再次被杀
2026-02-27T01:01:38  PID 19501 启动，存活下来
2026-02-27T01:16:xx  第一次 LLM timeout 出现
...
2026-02-27T05:51:xx  最后一次 LLM timeout
2026-02-27T08:55:xx  网络看门狗恢复，重启 Gateway
```

两个关键信号：
1. Gateway 在凌晨 01:01 之后成功存活了，它本身没挂
2. 但从 01:16 开始，所有 LLM 请求都失败了

**Gateway 活着，但 LLM 调不通。** 这排除了 Gateway 崩溃的可能。

### 第二步：分析 Timeout 模式

看每次 timeout 的 `durationMs`（从发起请求到超时的时间）：

```
durationMs: 3127
durationMs: 5842
durationMs: 4213
durationMs: 7651
durationMs: 3891
```

**全部只有 3-8 秒。** 这太短了。

正常的 LLM 请求超时应该是 60-120 秒。3 秒就 timeout，说明不是"API 响应慢"，而是**连接本身就建不起来**。

TCP 握手失败 → 连接被拒或无响应 → 几秒后超时。

### 第三步：检查网络链路

我的 LLM API 请求链路是：

```
Agent → OpenClaw Gateway → 本地代理(:7897) → 互联网 → api.anthropic.com
```

在中国大陆访问 Anthropic API 需要走代理。我用的是本地 Clash/V2Ray 代理，监听在 `127.0.0.1:7897`。

快速验证当前连通性：

```bash
$ curl -s -o /dev/null -w "%{http_code} %{time_total}s" \
  --max-time 10 https://api.anthropic.com/v1/messages \
  -H "x-api-key: test" -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'

401 0.568306s
```

现在是通的（401 = API key 无效，但网络通了）。那凌晨为什么不通？

### 第四步：找到"沉默的看门狗"

我有一个自己写的 `network-watchdog.sh`，作为 LaunchAgent 常驻运行，负责：
- 检测网络中断 → 恢复后重启 Gateway
- 检测 Mac 休眠唤醒 → 重启 Gateway
- 检测 Gateway 进程死亡/僵死 → 重启

看看看门狗在凌晨那段时间干了什么：

```bash
$ grep "2026-02-27 0[1-6]:" ~/.openclaw/logs/network-watchdog.log
# （空）
```

**空的。** 凌晨 1 点到 6 点之间，看门狗一条日志都没有。

再看它什么时候恢复的：

```
[2026-02-27 08:55:12] ========== watchdog v2.0 启动 ==========
```

08:55 才启动。**看门狗本身也死了整整 7+ 小时。**

### 第五步：锁定根因

所有证据指向同一个答案：**Mac 深度睡眠。**

时间线重构：

1. **~00:57** — Mac 进入深度睡眠（合盖/闲置）
2. 睡眠导致代理进程（Clash）的网络栈中断
3. LaunchAgent 尝试启动新 Gateway，但与残留进程冲突 → 反复失败（03:00~04:23 的 lock 风暴，每 10 秒一次）
4. **看门狗自己也被冻结了** —— macOS 深度睡眠会暂停后台进程
5. **~08:55** — Mac 唤醒，看门狗恢复，发现 Gateway 不健康，执行重启
6. 代理随 Mac 唤醒自动恢复，Gateway 重启后 LLM 链路恢复

**核心漏洞：代理挂了，但没有任何组件能检测到。**

---

## 为什么没有报警？

这是最值得反思的部分。我的看门狗已经覆盖了 5 种故障场景：

| # | 场景 | 检测方式 |
|---|------|----------|
| 1 | 断网→恢复 | ping 8.8.8.8 |
| 2 | Gateway 进程死亡 | pgrep |
| 3 | Gateway 僵死 | HTTP health check |
| 4 | 休眠唤醒 | uptime 跳变 |
| 5 | Discord 不可达 | curl Discord API |

但没有覆盖第 6 种：

| 6 | **代理不通** | ❌ 没有检测 |

看门狗 ping `8.8.8.8` 能通（因为 Mac 唤醒后 WiFi 恢复了），Gateway health check 也能通（Gateway 进程本身是活的），所以它认为一切正常。

但 LLM API 请求走的是代理 → 代理没恢复 → 所有 LLM 调用全部 timeout。

**这是一个经典的"健康检查盲区"——你检查了你以为重要的东西，但漏掉了真正关键的链路。**

就像体检时查了血压、心率、血糖，但没查冠状动脉——直到心梗发作才知道。

更讽刺的是：**看门狗自己也被 Mac 睡眠搞死了**。守夜人也睡着了。

---

## 修复方案

### 短期修复：看门狗 v2.1

给 `network-watchdog.sh` 新增第 6 项检测——代理健康检测：

```bash
# 代理健康检测配置
PROXY_URL="http://127.0.0.1:7897"
LLM_API_CHECK_URL="https://api.anthropic.com"
PROXY_CHECK_INTERVAL=4    # 每 4 轮（60秒）检测一次
PROXY_FAIL_THRESHOLD=3    # 连续 3 次失败才触发

# 检测逻辑
proxy_healthy() {
    # 先检查代理进程是否在监听
    if ! curl -s --max-time 3 -o /dev/null "$PROXY_URL"; then
        return 1
    fi
    # 通过代理尝试连接 LLM API
    local code
    code=$(curl -s --max-time 8 --proxy "$PROXY_URL" \
           -o /dev/null -w "%{http_code}" "$LLM_API_CHECK_URL")
    # 任何 HTTP 响应（包括 401/403）都说明链路通
    [[ "$code" =~ ^[0-9]+$ ]] && (( code > 0 ))
}
```

逻辑很简单：通过代理 curl 一下 `api.anthropic.com`，只要能拿到任何 HTTP 响应（哪怕是 401），就说明代理 → 互联网 → API 的链路是通的。连续 3 次失败（约 3 分钟）才触发 Gateway 重启。

**为什么不直接重启代理？** 因为代理进程的管理不在看门狗职责范围内（它可能是 ClashX、V2Ray 客户端等 GUI 应用），而且历史数据显示代理通常随 Mac 唤醒自动恢复，问题出在 Gateway 没有重建连接。

### 长期思考：OpenClaw 应该原生支持断路器

看 OpenClaw 的源码，它**已经有**单次请求级别的 failover 逻辑：

```javascript
// 简化的 OpenClaw 内部逻辑
if (timedOut && !timedOutDuringCompaction) {
    // 尝试切换 auth profile
    if (await advanceAuthProfile()) continue;
    // 如果配置了 fallback 模型，触发 failover
    if (fallbackConfigured) {
        throw new FailoverError(message, { reason, provider, model });
    }
}
```

但这只处理"某次 API 调用偶尔超时"的场景。当底层网络/代理彻底断了时，每次心跳都会重复这个循环：发请求 → 等 timeout → failover 也 timeout → 放弃 → 下次心跳再来。42 次，次次如此。

**理想方案是断路器（Circuit Breaker）模式：**

```
正常状态 → 连续 N 次 timeout → 断路器打开 → 暂停心跳调度
                                     ↓
                              定期 probe（低频）
                                     ↓
                              probe 成功 → 断路器关闭 → 恢复正常调度
```

这样就不会在代理完全断掉的 5 小时里傻傻地尝试 42 次，而是快速进入"暂停+低频探测"模式，等链路恢复后自动切回正常。

这个功能不只对我有用——所有在中国大陆通过代理使用 OpenClaw 的用户都会遇到类似问题。

---

## 这不是第一次

回顾历史记录，**这是第 5 次同一模式的故障**：

| 日期 | 触发原因 | 影响时长 |
|------|----------|----------|
| 02-19 | Mac 睡眠 + 代理断 | ~2h |
| 02-22 | Mac 睡眠 + 代理断 | ~3h |
| 02-24 | Mac 睡眠 + 代理断 | ~2h |
| 02-25 | Mac 睡眠 + 代理断 | ~4h |
| **02-27** | **Mac 睡眠 + 代理断** | **~5h** |

每次都是同一个模式：Mac 睡眠 → 代理网络栈断 → LLM timeout → 无报警 → 手动发现。

**影响时间在递增。** 从 2 小时到 5 小时。如果我不修，下次可能是一整天。

5 次了才修，说实话有点丢人。但也正是因为"每次都不够严重、每次都自然恢复了"，导致一直没当回事。

**很多生产事故都是这样——小故障重复出现但每次都"没事"，直到某天它真的出事。**

---

## 几个值得记住的教训

### 1. 健康检查要覆盖完整链路

不要只检查你的服务本身，要检查**它依赖的每一个外部环节**。

我的 Gateway health check 通过了（进程活着），ping 也通了（网络正常），但 LLM 调用还是全部失败——因为中间的代理环节没有被检查。

这在微服务架构里叫做"深度健康检查（deep health check）"：不只检查自己的心跳，还要 probe 下游依赖。

### 2. 看门狗也会死

任何监控系统本身也是一个可能失败的系统。当 Mac 深度睡眠时，看门狗进程也被冻结了。

**谁来监控监控系统？** 这个问题没有完美答案，但至少可以：
- 用外部监控（云端 uptime checker）作为最后一层保底
- 看门狗恢复后检查"我睡了多久"，如果超过阈值立即执行全面健康检查

### 3. "每次都自然恢复"是最危险的信号

5 次同类故障，每次都"等一等就好了"。这种"假性自愈"会麻痹你的警觉性，直到某天它不再自愈。

**如果一个故障发生了 3 次，它不是偶发事件，它是一个系统性缺陷。**

### 4. 在中国跑 AI Agent 有额外的运维成本

代理是你的生命线。代理断了 = AI 断了。

如果你也在用代理访问 LLM API，请务必：
- 代理进程设为开机自启 + 崩溃自动重启
- 在看门狗里加上"通过代理探测 API 端点"的健康检查
- 考虑备用代理链路（代理 A 挂了自动切代理 B）
- Mac 用户注意 `caffeinate` 或 Energy Saver 设置，避免深度睡眠

---

## 后记

修复后的看门狗 v2.1 已经在运行了：

```
[2026-02-27 14:19:16] ========== watchdog v2.1 启动 ==========
[2026-02-27 14:19:16] 配置: interval=15s cooldown=120s tun_settle=8s
                      health_threshold=3 proxy_check=4轮 proxy_fail=3次
```

下次 Mac 再睡眠，代理恢复后最多 3 分钟内 Gateway 就会自动重建连接。不再需要我手动发现。

但说实话，这只是贴了个创可贴。真正的解决方案应该是 OpenClaw 原生支持 LLM 连通性探测和断路器模式——让框架本身具备"网不通就别傻等"的能力。

**毕竟，一个 7×24 运行的 AI Agent 系统，连"网断了"都检测不到，那它到底有多"智能"？**

---

*作者运行着一个基于 OpenClaw 的 8 Agent 军团，在 MacBook 上 24 小时运行。本文记录了一次真实的生产事故排查过程。*

*OpenClaw 是一个开源 AI Agent 框架，支持多 Agent 部署、心跳调度、多渠道通讯等功能。项目地址：https://github.com/openclaw/openclaw*
