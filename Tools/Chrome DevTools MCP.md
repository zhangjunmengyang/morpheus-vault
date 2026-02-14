---
title: Chrome DevTools MCP
date: 2026-02-14
tags: [mcp, browser-automation, devtools, tools]
type: note
status: brief
---

# Chrome DevTools MCP

## 是什么
MCP (Model Context Protocol) server，让 AI coding agent 通过 Chrome DevTools Protocol 控制和检查浏览器。

## 核心能力
- 控制 Chrome 浏览器（导航、点击、截图）
- 检查 DOM / CSS / 网络请求
- 性能分析和调试
- 使用真实 Chrome profile（保留登录态）
- 语义搜索标签页内容

## 两个主要实现
1. **chrome-devtools-mcp** — 基础版，自动化 + 调试 + 性能分析
2. **chrome-devtools-advanced-mcp** — 高级版，支持真实 profile 交互

## 集成方式
- VS Code Copilot（Agent mode）
- Cursor
- Claude Code
- 任何支持 MCP 的 AI 工具

## 对我们的意义
- 我们已有 agent-browser + OpenClaw browser，功能重叠
- 如果未来需要在 Claude Code 中直接调试网页，可以考虑接入
- 当前优先级低，记录备查

## 参考
- [LobeHub MCP 目录](https://lobehub.com/mcp/chromedevtools-chrome-devtools-mcp)
- [DEV.to 教程: Chrome DevTools MCP + Copilot](https://dev.to/saloniagrawal/debugging-with-ai-chrome-devtools-mcp-copilot-1eoe)
