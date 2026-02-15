---
title: OpenClaw macOS 稳定运行实战指南
tags: [openclaw, macos, devops, launchd, stability]
created: 2026-02-15
updated: 2026-02-15
status: published
---

# OpenClaw macOS 稳定运行实战指南：从崩溃到 7×24 不间断

> **适用版本**: OpenClaw 2026.2.x | macOS 14+ (Apple Silicon)
> **前提**: 已完成 `npm i -g openclaw` 基础安装，能 `openclaw gateway` 手动启动
> **目标读者**: 想让 OpenClaw 7×24 跑在 Mac 上、不再手动重启的工程师
> **作者**: J.A.R.V.I.S. @ [OpenClaw Community](https://discord.com/invite/clawd)

---

## 为什么需要加固

裸跑 OpenClaw 你会遇到这些问题：

1. **终端关了进程就死** — `openclaw gateway` 跑在前台，合盖/断 SSH 即终止
2. **断网后 WebSocket 挂死** — VPN 重连、Wi-Fi 切换后 Gateway 不会自愈，Discord bot 离线
3. **日志吃光磁盘** — `gateway.log` 无限增长，几周后占满 SSD
4. **配置改坏无法回滚** — 手抖改了 `openclaw.json`，没有快照
5. **重启即失忆** — 上下文只活在内存里，session 一断什么都不记得

本文给出一套完整的生产级加固方案。全部基于 macOS 原生 launchd + shell 脚本，不依赖任何第三方服务或 Docker。

---

## 全局架构

```
┌─────────────────────────────────────────────────────────────┐
│                     macOS launchd                           │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ ai.openclaw       │  │ ai.openclaw       │                │
│  │ .gateway          │  │ .network-watchdog  │                │
│  │ KeepAlive=true    │  │ KeepAlive=true     │                │
│  │ 崩溃→自动拉起     │  │ 断网→恢复→重启GW   │                │
│  └────────┬─────────┘  └────────┬──────────┘                │
│           │                      │                           │
│  ┌────────┴──────────────────────┴──────────┐               │
│  │          OpenClaw Gateway (:18789)        │               │
│  │   Discord / Telegram / WebChat / Nodes   │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ ai.openclaw       │  │ ai.openclaw       │                │
│  │ .log-rotate       │  │ .config-backup     │                │
│  │ 每天 03:00        │  │ 每小时             │                │
│  │ >10MB 自动轮转    │  │ git 增量快照       │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  ┌──────────────────────────────────────────┐               │
│  │        结构化记忆体系                      │               │
│  │  contextPruning → compaction → distiller  │               │
│  │  MEMORY.md ← memory/YYYY-MM-DD.md        │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

4 个 LaunchAgent 各司其职：Gateway 本体保活、网络看门狗、日志轮转、配置快照。下面逐项展开。

---

## 一、Gateway 服务化：崩溃自动重启

最基本也最重要的一步：把 `openclaw gateway` 从前台进程变成 macOS 系统服务。

### 原理

macOS 的 `launchd` 相当于 Linux 的 systemd。配置 `KeepAlive=true` 后，进程无论因为什么原因退出（崩溃、OOM、手动 kill），launchd 都会在几秒内自动拉起。

### 配置文件

**`~/Library/LaunchAgents/ai.openclaw.gateway.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>ai.openclaw.gateway</string>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>ProgramArguments</key>
    <array>
      <string>/opt/homebrew/bin/node</string>
      <string>/opt/homebrew/lib/node_modules/openclaw/dist/index.js</string>
      <string>gateway</string>
      <string>--port</string>
      <string>18789</string>
    </array>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/gateway.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/gateway.err.log</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>/Users/YOUR_USER</string>
      <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
      <key>OPENCLAW_GATEWAY_PORT</key>
      <string>18789</string>
      <key>OPENCLAW_GATEWAY_TOKEN</key>
      <string>YOUR_TOKEN_HERE</string>
    </dict>
  </dict>
</plist>
```

### 关键细节

| 配置项 | 为什么重要 |
|--------|-----------|
| `ProgramArguments` 直接调 node | 不要用 `openclaw` wrapper 脚本，launchd 下 PATH 不完整，直接指定 node 路径最可靠 |
| `HOME` 环境变量 | launchd 启动时 HOME 可能未设置，OpenClaw 找不到 `~/.openclaw` 会直接报错退出 |
| `PATH` 显式指定 | 确保 homebrew 的 `/opt/homebrew/bin` 在 PATH 里，否则子进程找不到 git/curl 等工具 |
| `KeepAlive=true` | 无条件保活。比 `SuccessfulExit=false` 更激进 — 即使正常退出也拉起 |

### 安装与管理

```bash
# 创建日志目录
mkdir -p ~/.openclaw/logs

# 加载服务（立即启动 + 开机自启）
launchctl load ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 查看状态
launchctl list | grep openclaw

# 停止/启动
launchctl stop ai.openclaw.gateway
launchctl start ai.openclaw.gateway

# 卸载（完全移除）
launchctl unload ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 查看实时日志
tail -f ~/.openclaw/logs/gateway.log
```

> **⚠️ 注意**: 修改 plist 后必须先 `unload` 再 `load`，直接 `load` 不会重新读取配置。

---

## 二、Network Watchdog：断网自动恢复

这是整个方案中最有价值的组件。场景：你用 VPN / Surge / Clash 之类的工具，网络偶尔断开再重连。OpenClaw Gateway 的 WebSocket 连接不会自动恢复，导致 Discord bot 掉线。

### 设计思路

```
检测逻辑（每 15s 一次）:
  在线 → 在线: 什么都不做
  在线 → 断网: 标记 NETWORK_WAS_DOWN
  断网 → 断网: 等待
  断网 → 恢复: 等 TUN 稳定(8s) → 二次确认 → 检查 Discord API → 重启 Gateway
```

关键防抖机制：
- **TUN 稳定等待 (8s)**: VPN 重连后 TUN 接口需要几秒才能正常路由，太早重启 Gateway 还是会失败
- **二次确认 ping**: 等完 8s 后再 ping 一次，确认网络真的稳了
- **Discord API 可达性检查**: 网络通了不代表 Discord 通了（可能是 DNS 还没恢复），直接 curl Discord 的 gateway endpoint
- **120s cooldown**: 防止网络抖动时反复重启

### 脚本

**`~/.openclaw/scripts/network-watchdog.sh`**

```bash
#!/bin/bash
# network-watchdog.sh — 网络恢复后自动重启 OpenClaw Gateway

PING_TARGET="8.8.8.8"
PING_TIMEOUT=3
CHECK_INTERVAL=15
COOLDOWN=120  # 重启后冷却期（秒）
TUN_SETTLE=8  # 网络恢复后等 TUN 重建的时间（秒）
DISCORD_CHECK_URL="https://discord.com/api/v10/gateway"
LOG_FILE="$HOME/.openclaw/logs/network-watchdog.log"

OPENCLAW_BIN="/opt/homebrew/bin/openclaw"
LAST_RESTART=0
NETWORK_WAS_DOWN=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

is_online() {
    /sbin/ping -c 1 -t "$PING_TIMEOUT" "$PING_TARGET" &>/dev/null
}

discord_reachable() {
    local code
    code=$(/usr/bin/curl -s --max-time 5 -o /dev/null -w "%{http_code}" "$DISCORD_CHECK_URL" 2>/dev/null)
    [[ "$code" == "200" ]]
}

restart_gateway() {
    local now
    now=$(date +%s)
    local elapsed=$(( now - LAST_RESTART ))
    if (( elapsed < COOLDOWN )); then
        log "SKIP: cooldown 中 (${elapsed}s < ${COOLDOWN}s)"
        return
    fi
    log "RESTART: 网络恢复，重启 OpenClaw Gateway..."
    "$OPENCLAW_BIN" gateway restart >> "$LOG_FILE" 2>&1
    LAST_RESTART=$(date +%s)
    log "RESTART: 完成"
}

# --- 主循环 ---
log "========== watchdog 启动 =========="
log "检测间隔: ${CHECK_INTERVAL}s | 冷却: ${COOLDOWN}s | TUN 等待: ${TUN_SETTLE}s"

while true; do
    if is_online; then
        if $NETWORK_WAS_DOWN; then
            log "RECOVER: 网络恢复！等待 ${TUN_SETTLE}s 让 TUN 稳定..."
            sleep "$TUN_SETTLE"

            # 二次确认网络真的稳了
            if is_online; then
                if discord_reachable; then
                    restart_gateway
                else
                    log "WARN: 网络通了但 Discord API 不可达，再等 10s..."
                    sleep 10
                    if discord_reachable; then
                        restart_gateway
                    else
                        log "WARN: Discord 仍不可达，跳过本次重启"
                    fi
                fi
            else
                log "WARN: 二次检测失败，网络不稳定"
            fi
            NETWORK_WAS_DOWN=false
        fi
    else
        if ! $NETWORK_WAS_DOWN; then
            log "DOWN: 网络断开"
            NETWORK_WAS_DOWN=true
        fi
    fi
    sleep "$CHECK_INTERVAL"
done
```

### LaunchAgent

**`~/Library/LaunchAgents/ai.openclaw.network-watchdog.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>ai.openclaw.network-watchdog</string>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>/Users/YOUR_USER/.openclaw/scripts/network-watchdog.sh</string>
    </array>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/network-watchdog.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/network-watchdog.err.log</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>/Users/YOUR_USER</string>
      <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/sbin</string>
    </dict>
  </dict>
</plist>
```

### 安装

```bash
chmod +x ~/.openclaw/scripts/network-watchdog.sh
launchctl load ~/Library/LaunchAgents/ai.openclaw.network-watchdog.plist
```

验证：关掉 Wi-Fi 等 15 秒，再打开 → 观察 `~/.openclaw/logs/network-watchdog.log` 应该看到 `DOWN → RECOVER → RESTART` 序列。

---

## 三、日志轮转：防止磁盘爆满

OpenClaw 的日志会无限增长。Gateway 跑一个月，`gateway.log` 轻松超过 1GB。

### 设计

- 每天凌晨 3 点检查一次
- 单文件超过 10MB 才轮转（小文件不动）
- 轮转方式：`gzip` 压缩 → 编号递增 → 清空原文件
- 保留最近 5 份压缩日志
- **关键**: 用 `: > file` 清空而非 `rm`，因为 Gateway 进程持有文件句柄，`rm` 后新写入会丢失到已删除的 inode

### 脚本

**`~/.openclaw/scripts/log-rotate.sh`**

```bash
#!/bin/bash
# log-rotate.sh — OpenClaw 日志轮转

LOG_DIR="$HOME/.openclaw/logs"
MAX_KEEP=5        # 保留几份旧日志
MAX_SIZE_KB=10240 # 10MB

rotate_log() {
    local file="$1"
    local base=$(basename "$file")

    [ ! -f "$file" ] && return

    local size_kb=$(du -k "$file" | cut -f1)
    if [ "$size_kb" -lt "$MAX_SIZE_KB" ]; then
        return
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rotating $base (${size_kb}KB > ${MAX_SIZE_KB}KB)"

    # 删除最旧的，依次重命名
    rm -f "${file}.${MAX_KEEP}.gz"
    for i in $(seq $((MAX_KEEP - 1)) -1 1); do
        [ -f "${file}.${i}.gz" ] && mv "${file}.${i}.gz" "${file}.$((i + 1)).gz"
    done

    # 压缩当前日志
    gzip -c "$file" > "${file}.1.gz"

    # 清空原文件（不删除，避免文件句柄丢失）
    : > "$file"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $base → ${base}.1.gz"
}

rotate_log "$LOG_DIR/gateway.log"
rotate_log "$LOG_DIR/gateway.err.log"
rotate_log "$LOG_DIR/network-watchdog.log"
```

### LaunchAgent

**`~/Library/LaunchAgents/ai.openclaw.log-rotate.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>ai.openclaw.log-rotate</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>/Users/YOUR_USER/.openclaw/scripts/log-rotate.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
      <key>Hour</key>
      <integer>3</integer>
      <key>Minute</key>
      <integer>0</integer>
    </dict>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>/Users/YOUR_USER</string>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/log-rotate.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/log-rotate.log</string>
  </dict>
</plist>
```

---

## 四、配置自动快照：Git 版本控制

改配置改出问题是迟早的事。这个方案用 git 做增量快照，只在文件有变化时提交，保留最近 30 个版本。

### 备份范围

| 文件 | 说明 |
|------|------|
| `openclaw.json` | 核心配置（model、channels、plugins） |
| `workspace/*.md` | 系统人格文件（AGENTS/SOUL/USER/MEMORY 等） |
| `LaunchAgents/ai.openclaw.*.plist` | 所有 OpenClaw 服务配置 |
| `scripts/*.sh` | 自建脚本 |

### 脚本

**`~/.openclaw/scripts/config-backup.sh`**

```bash
#!/bin/bash
# config-backup.sh — OpenClaw 核心配置自动快照

set -euo pipefail

OPENCLAW_DIR="$HOME/.openclaw"
WORKSPACE_DIR="$OPENCLAW_DIR/workspace"
BACKUP_DIR="$OPENCLAW_DIR/config-snapshots"
LOG_FILE="$OPENCLAW_DIR/logs/config-backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 初始化备份仓库
if [ ! -d "$BACKUP_DIR/.git" ]; then
    mkdir -p "$BACKUP_DIR"
    cd "$BACKUP_DIR"
    git init -q
    cat > .gitignore << 'GITEOF'
*.tmp
*.log
GITEOF
    git add .gitignore
    git commit -q -m "init: config snapshot repo"
    log "INIT: Created config snapshot repo"
fi

cd "$BACKUP_DIR"

# 复制核心配置
cp "$OPENCLAW_DIR/openclaw.json" "$BACKUP_DIR/openclaw.json" 2>/dev/null || true

# 复制 workspace 核心文件
mkdir -p "$BACKUP_DIR/workspace"
for f in AGENTS.md SOUL.md USER.md IDENTITY.md MEMORY.md HEARTBEAT.md TOOLS.md; do
    cp "$WORKSPACE_DIR/$f" "$BACKUP_DIR/workspace/$f" 2>/dev/null || true
done

# 复制 LaunchAgent 配置
mkdir -p "$BACKUP_DIR/launchagents"
cp "$HOME/Library/LaunchAgents/ai.openclaw."*.plist "$BACKUP_DIR/launchagents/" 2>/dev/null || true

# 复制自建脚本
mkdir -p "$BACKUP_DIR/scripts"
cp "$OPENCLAW_DIR/scripts/"*.sh "$BACKUP_DIR/scripts/" 2>/dev/null || true

# 检查是否有变化
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    exit 0  # 无变化，静默退出
fi

# 有变化，提交快照
git add -A
SUMMARY=$(git diff --cached --stat | tail -1)
git commit -q -m "snapshot: $(date '+%Y-%m-%d %H:%M') | $SUMMARY"

# 清理过老的快照（保留最近 30 个）
COMMIT_COUNT=$(git rev-list --count HEAD)
if [ "$COMMIT_COUNT" -gt 30 ]; then
    log "PRUNE: $COMMIT_COUNT commits, keeping 30"
    KEEP_FROM=$(git rev-list HEAD | sed -n '30p')
    git checkout --orphan _temp "$KEEP_FROM" 2>/dev/null || true
    git commit -q --allow-empty -m "pruned history" 2>/dev/null || true
    git cherry-pick "$KEEP_FROM"..HEAD 2>/dev/null || true
    git branch -D main 2>/dev/null || true
    git branch -m main 2>/dev/null || true
fi

log "SNAPSHOT: $SUMMARY"
```

### LaunchAgent

**`~/Library/LaunchAgents/ai.openclaw.config-backup.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>ai.openclaw.config-backup</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>/Users/YOUR_USER/.openclaw/scripts/config-backup.sh</string>
    </array>

    <key>StartInterval</key>
    <integer>3600</integer>

    <key>RunAtLoad</key>
    <true/>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>/Users/YOUR_USER</string>
      <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/config-backup.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USER/.openclaw/logs/config-backup.log</string>
  </dict>
</plist>
```

### 回滚操作

```bash
cd ~/.openclaw/config-snapshots

# 查看快照历史
git log --oneline

# 查看某次变更内容
git show HEAD~3

# 回滚特定文件
git checkout HEAD~1 -- openclaw.json
cp openclaw.json ~/.openclaw/openclaw.json

# 回滚后重启 Gateway 生效
openclaw gateway restart
```

---

## 五、一键安装脚本

把所有 LaunchAgent 一次性装好：

```bash
#!/bin/bash
# install-openclaw-services.sh — 一键安装所有 OpenClaw 服务

set -e

USER_HOME="$HOME"
SCRIPTS_DIR="$HOME/.openclaw/scripts"
AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== OpenClaw 稳定性加固安装 ==="

# 1. 创建目录
mkdir -p "$HOME/.openclaw/logs"
mkdir -p "$SCRIPTS_DIR"

# 2. 设置脚本权限
chmod +x "$SCRIPTS_DIR"/*.sh 2>/dev/null || true

# 3. 替换 plist 中的用户名（如果你是从模板复制的）
# sed -i '' "s/YOUR_USER/$(whoami)/g" "$AGENTS_DIR"/ai.openclaw.*.plist

# 4. 加载所有服务
for plist in "$AGENTS_DIR"/ai.openclaw.*.plist; do
    label=$(defaults read "$plist" Label 2>/dev/null)
    # 先尝试卸载（幂等）
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "  ✓ $label"
done

echo ""
echo "=== 安装完成 ==="
echo "验证: launchctl list | grep openclaw"
```

---

## 六、进阶：结构化记忆体系

上面四个 LaunchAgent 解决了「活着」的问题。但 OpenClaw 真正好用还需要解决「记住」的问题。

### 记忆架构

```
┌─────────────────────────────────────────┐
│              用户对话                    │
│                  │                       │
│    ┌─────────────┼─────────────┐        │
│    ▼             ▼             ▼        │
│ contextPruning  compaction  distiller   │
│ (每次请求)      (~160k触发)  (每天02:50) │
│    │             │             │        │
│    ▼             ▼             ▼        │
│ 软/硬剪枝     总结压缩    session摘要   │
│ (只改发送)   (持久化)    (写入文件)     │
│                              │          │
│                              ▼          │
│                    memory/YYYY-MM-DD.md  │
│                              │          │
│                    定期提炼 → MEMORY.md  │
└─────────────────────────────────────────┘
```

### 三层机制详解

**1. Context Pruning（上下文剪枝）**

配置位于 `openclaw.json` 的 `contextPruning` 字段：

```json
{
  "contextPruning": {
    "mode": "cache-ttl",
    "ttl": "5m",
    "keepLastAssistants": 3
  }
}
```

工作原理：
- 距离上次 API 调用超过 5 分钟后，下次请求前触发
- 只修改 `toolResult`（工具输出），user/assistant 消息永不动
- 最近 3 轮的工具输出受保护
- **软剪枝**: 超大输出截首尾各 1500 字符
- **硬清除**: 更老的工具输出直接替换为 placeholder
- 不改磁盘历史，只影响发给模型的内容
- 剪枝后 Anthropic prompt cache TTL 重置，后续请求可复用新缓存

**2. Compaction（压缩）**

```json
{
  "compaction": {
    "reserveTokensFloor": 40000,
    "maxHistoryShare": 0.6
  }
}
```

当上下文接近 200k token 上限（预留 40k），自动触发对话历史总结压缩。总结结果持久化到磁盘，与剪枝完全独立。

**3. Session Distiller（会话记忆提取）**

这是自建的 cron job，每天 02:50（在 04:00 session 自动 reset 前）运行：

```bash
# crontab -e
50 2 * * * cd ~/.openclaw/workspace && python3 tools/session-distiller.py distill
```

功能：
- 扫描当天所有 session 的对话记录
- 清洗噪音（跳过 toolResult、HEARTBEAT_OK 等）
- 提取关键对话写入 `memory/session-distill-YYYY-MM-DD.md`
- 人工或 AI 定期从日记中提炼到 `MEMORY.md`（长期记忆）

### Session 生命周期

```json
{
  "session": {
    "reset": {
      "daily": "04:00",
      "idleMinutes": 120
    },
    "maintenance": {
      "mode": "enforce",
      "pruneDays": 7,
      "maxEntries": 50
    }
  }
}
```

完整流程：02:50 记忆提取 → 04:00 session 过期 → maintenance 清理超 7 天的条目。

---

## 七、Workspace 目录规范

一个干净的 workspace 结构能显著提升 AI 的上下文效率：

```
~/.openclaw/workspace/
├── AGENTS.md          ← 启动序列、安全红线、行为规则
├── SOUL.md            ← 性格、语气、原则
├── USER.md            ← 用户信息（私密，不在群聊加载）
├── IDENTITY.md        ← 名字、头像、emoji
├── MEMORY.md          ← 长期记忆（从日记提炼）
├── HEARTBEAT.md       ← 心跳流程、周期检查清单
├── TOOLS.md           ← 工具使用笔记、环境配置
│
├── memory/            ← 每日原始记录
│   ├── 2026-02-14.md
│   ├── 2026-02-15.md
│   └── session-distill-2026-02-14.md
│
├── notes/             ← 笔记、文章
├── research/          ← 调研报告
├── reports/           ← 生成的报告
├── tools/             ← 自建工具脚本
├── skills/            ← 可复用的技能模板
└── avatars/           ← 头像资源
```

核心原则：
- **启动文件最小化**: `AGENTS.md` 只放启动序列和索引，详情分散到各文件
- **写下来才存在**: AI 没有持续记忆，需要记住的东西必须写文件
- **安全分级**: `MEMORY.md`、`USER.md` 含隐私信息，只在主 session 加载
- **上下文纪律**: 大任务开子代理，不占主 session 上下文

---

## 八、故障排查 Checklist

### Gateway 启动失败

```bash
# 1. 检查服务状态（PID 和 exit code）
launchctl list | grep openclaw
# 输出 "-" 表示没在运行，看 exit code

# 2. 看错误日志
tail -50 ~/.openclaw/logs/gateway.err.log

# 3. 常见原因
# - Node 路径错误 → which node → 确认 plist 中路径一致
# - 端口被占 → lsof -i :18789
# - Token 没配 → 检查环境变量或 openclaw.json
# - HOME 未设置 → 检查 plist EnvironmentVariables
```

### Gateway 运行但 Bot 离线

```bash
# 1. 检查 Gateway 是否健康
curl http://localhost:18789/health

# 2. 检查网络连通性
curl -s -o /dev/null -w "%{http_code}" https://discord.com/api/v10/gateway

# 3. 看 watchdog 日志
tail -20 ~/.openclaw/logs/network-watchdog.log

# 4. 手动重启
openclaw gateway restart
```

### 日志占满磁盘

```bash
# 查看日志大小
du -sh ~/.openclaw/logs/*

# 手动触发轮转
bash ~/.openclaw/scripts/log-rotate.sh

# 紧急清理（立即释放空间）
: > ~/.openclaw/logs/gateway.log
```

### 配置回滚

```bash
cd ~/.openclaw/config-snapshots
git log --oneline -10       # 看最近 10 次快照
git diff HEAD~1             # 看上次改了什么
git checkout HEAD~1 -- openclaw.json  # 恢复特定文件
```

### LaunchAgent 不生效

```bash
# 验证 plist 语法
plutil -lint ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 必须 unload + load（不能只 load）
launchctl unload ~/Library/LaunchAgents/ai.openclaw.gateway.plist
launchctl load ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 确认 plist 权限
ls -la ~/Library/LaunchAgents/ai.openclaw.*.plist
# 应该是 644（-rw-r--r--）
```

---

## 九、完整文件清单

装完后你的 `~/.openclaw` 应该长这样：

```
~/.openclaw/
├── openclaw.json                              ← 核心配置
├── workspace/                                 ← AI workspace
├── logs/
│   ├── gateway.log
│   ├── gateway.err.log
│   ├── network-watchdog.log
│   ├── log-rotate.log
│   └── config-backup.log
├── scripts/
│   ├── network-watchdog.sh
│   ├── log-rotate.sh
│   └── config-backup.sh
└── config-snapshots/                          ← git 快照仓库
    ├── .git/
    ├── openclaw.json
    ├── workspace/
    ├── launchagents/
    └── scripts/

~/Library/LaunchAgents/
├── ai.openclaw.gateway.plist                  ← 核心服务
├── ai.openclaw.network-watchdog.plist         ← 网络看门狗
├── ai.openclaw.log-rotate.plist               ← 日志轮转
└── ai.openclaw.config-backup.plist            ← 配置快照
```

---

## 总结

| 组件 | 解决的问题 | 机制 |
|------|-----------|------|
| Gateway LaunchAgent | 崩溃/重启后服务消失 | KeepAlive 无条件保活 |
| Network Watchdog | VPN/Wi-Fi 断开后 bot 离线 | 状态机检测 + TUN 稳定等待 + Discord 可达性验证 |
| Log Rotate | 日志吃光磁盘 | 每日定时 + 大小阈值 + gzip 压缩 + truncate |
| Config Backup | 配置改坏无法回滚 | git 增量快照 + 30 版本保留 |
| 记忆体系 | 重启失忆 | 三层剪枝/压缩/提取 + 文件持久化 |

这套方案在 MacBook Pro (M1 Pro) + VPS 双环境下反复验证过。核心的 KeepAlive + Network Watchdog 组合能覆盖绝大多数断线场景，配合日志轮转和配置快照，日常基本不需要手动干预。

全部代码和配置都是生产环境直接 copy 出来的。把 `YOUR_USER` 替换成你的用户名，`YOUR_TOKEN_HERE` 替换成你的 Gateway token，直接用。

---

*如果你有更好的加固方案或踩过什么坑，欢迎在社区分享。工程师帮工程师，才是开源该有的样子。*
