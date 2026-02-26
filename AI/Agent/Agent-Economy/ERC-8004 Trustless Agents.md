---
brief: "ERC-8004 Trustless Agents——以太坊 Agent 信任协议提案；无需信任第三方的 AI Agent 链上执行机制；智能合约验证 + 零知识证明的技术路径；链上 Agent 治理标准化的早期探索。"
title: "ERC-8004 Trustless Agents"
type: analysis
domain: ai/agent/agent-economy
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/agent-economy
  - crypto
  - ethereum
  - erc-8004
  - identity
  - status/active
status: active
---

# ERC-8004: Trustless Agents

> Agent 的链上身份证 + 大众点评 + 第三方验证。三个注册表解决发现、信誉、验证问题。

- EIP 原文：https://eips.ethereum.org/EIPS/eip-8004
- 作者：Marco De Rossi (MetaMask AI Lead), Davide Crapis (EF AI Lead), Jordan Ellis (Google), Erik Reppel (Coinbase)
- 状态：EIP Draft → Mainnet 已上线 (2026-01-29)
- 依赖标准：EIP-155, EIP-712, EIP-721, ERC-1271

## 三大注册表

### 1. Identity Registry（身份注册表）

本质是 ERC-721 NFT + URIStorage 扩展，每个 Agent 获得：
- **agentId**：自增 ERC-721 tokenId
- **agentRegistry**：`eip155:{chainId}:{contractAddress}` 全局唯一标识
- **agentURI**：指向注册文件（IPFS / HTTPS / base64 data URI）

注册文件支持声明 A2A、MCP、ENS、DID、x402 等服务端点。

**关键设计**：
- NFT 可转让 → Agent 身份可交易/转移（策略即资产）
- `agentWallet` 转移时自动清零，新所有者须重新验证
- 链上只存 agentId + agentURI 指针，注册文件在链下（灵活 + 省 gas）

### 2. Reputation Registry（信誉注册表）

**不是聚合分数，是原始反馈数据层。** 期望生态涌现多种评分服务商。

反馈结构：`value` (int128) + `valueDecimals` (uint8) 有符号定点数 + 自定义 tag 标签

示例：
| tag1 | 含义 | value | 人类可读 |
|------|------|-------|---------|
| starred | 质量评分 | 87 | 87/100 |
| uptime | 在线率 | 9977 (decimals=2) | 99.77% |
| tradingYield | 收益率 | -32 (decimals=1) | -3.2% |

**反 Spam 机制**：
- 反馈者不能是 Agent owner/approved operator
- `getSummary()` 强制传入 clientAddresses 过滤（无过滤 = Sybil 攻击）
- 支持反馈撤销 + Agent 回应追加

### 3. Validation Registry（验证注册表）

让 Agent 的工作可被独立第三方验证，结果上链（0-100 分）。

支持的验证模型（pluggable）：
- **Crypto-Economic** — 质押者重新执行推理，错误则 slash
- **zkML** — 零知识机器学习证明
- **TEE Oracle** — 可信执行环境验证（Intel SGX/TDX, AMD SEV-SNP）
- **Trusted Judges** — 人类或受信实体验证

## 部署状态

**统一地址**（CREATE2 部署，所有链相同）：
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432`
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63`

**已部署主网**：Ethereum, Base, Arbitrum, Optimism, Polygon, Linea, Scroll, Taiko, Avalanche, Celo, Gnosis, Abstract, MegaETH, Monad, BSC, Mantle (WIP)

**审计**：Cyfrin + Nethermind + EF Security 三方

**实际数据**（截至 2026-02-13）：
- ETH mainnet: ~5,003 holders
- 跨链总注册量可能已远超 13,000（20+ 条链部署）
- Testnet: 10,000+ agents + 20,000+ feedback entries

## 与现有标准的关系

| 标准 | 关系 |
|------|------|
| W3C DID | 互补——注册文件可包含 DID endpoint |
| Verifiable Credentials | ERC-8004 更偏公开评价，Validation Registry 类似 VC 功能 |
| EAS | EAS 通用证明层，ERC-8004 Agent 专用信任层，可组合 |
| A2A (Google) | 原生支持，Registration File 可包含 AgentCard URL |
| MCP (Anthropic) | 原生支持 MCP endpoint |
| x402 (Coinbase) | 支付协议互补，feedback 可包含 proofOfPayment |

## 对量化 Agent 的意义

1. **身份即资产** — 注册链上身份声明策略类型、回测收益；NFT 可转让 = 策略可交易
2. **可验证 Track Record** — `tradingYield` tag 已在规范示例中，客户可链上提交收益反馈，不可删除
3. **执行证明** — TEE 证明策略在隔离环境运行，zkML 证明推理正确性，Validation 结果可驱动智能合约自动信任决策
4. **合规基础** — KYA (Know Your Agent) 雏形，监管者可要求注册文件声明合规信息

## 关键判断

**真的部分**：
- EF 直推 + MetaMask/Google/Coinbase 背书，不是草台班子
- 架构精巧：最小化链上存储，最大化链下灵活性
- 填补了 A2A/MCP 未解决的跨组织信任空白
- 20+ 条链同时部署，生态野心大

**Hype 的部分**：
- 10k testnet 注册 ≠ 10k 真正在用的 Agent
- Reputation 系统目前没有成熟的反 Sybil 方案，靠"期望生态涌现"
- 量化场景的信誉验证远比 API 可用性检查复杂

### 时间线建议

- **短期 (0-6 月)** — 观察为主，关注 Reputation 聚合服务商出现
- **中期 (6-18 月)** — 若标准成熟，为量化 Agent 注册身份建立 on-chain track record
- **长期** — Agent 身份 = 链上品牌，Reputation = 可验证夏普比率，Validation = 密码学执行证明

## 关键链接

- [EIP 原文](https://eips.ethereum.org/EIPS/eip-8004) | [GitHub 合约](https://github.com/erc-8004/erc-8004-contracts)
- [Awesome ERC-8004](https://github.com/sudeepb02/awesome-erc8004) | [论坛](https://ethereum-magicians.org/t/erc-8004-trustless-agents/25098)
- [官网](https://8004.org) | [Telegram](https://t.me/ERC8004)
- [ETH Mainnet Contract](https://etherscan.io/address/0x8004A169FB4a3325136EB29fA0ceB6D2e539a432)

## 关联

- [[AI/Agent/Agent-Economy/Agent 经济基础设施|Agent 经济基础设施]] — 全景综述，ERC-8004 是身份层
- [[AI/Agent/Agent-Economy/Coinbase AgentKit 技术评估|Coinbase AgentKit 技术评估]] — AgentKit 钱包可注册 ERC-8004 身份
- [[AI/Agent/Agent-Economy/Virtuals Protocol|Virtuals Protocol]] — 商业网络层，ACP 的信任问题可被 ERC-8004 解决
- [[AI/Agent/Agent-Economy/Agentic Spring|Agentic Spring]] — 模型能力成熟 + 身份标准 = Agent 经济闭环
