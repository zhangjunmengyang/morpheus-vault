---
title: "Multi-Agent LLM Defense Pipeline Against Prompt Injection"
brief: "用多专门化 LLM agent 协同检测并中和 prompt injection 攻击：Chain-of-Agents（事后审查）和 Coordinator（事前拦截）两种架构，55 攻击 8 类别，实现 100% ASR 缓解（从 30% → 0%），同时维持对合法查询的正常响应。"
tags: [security, prompt-injection, agent-security, multi-agent, defense, pipeline]
sources:
  - "arXiv:2509.14285v4 (Hossain et al., Wichita State + Marshall + Aizu, 2025-09, v4 Dec 2025)"
rating: ★★★★☆
---

# Multi-Agent Defense Pipeline：用 LLM 检测 LLM 注入

## 核心思路

**传统防御的失败**：规则过滤、关键词检测 → 对语义混淆攻击无效  
**本文方案**：用专门化 LLM agent 作为"防御者"，其语义理解能力与攻击的语义复杂度匹配

> 用 LLM 来防御 LLM 被注入。

---

## 一、攻击数据集（HPI_ATTACK_DATASET）

55 个独立攻击，8 大类别，400 个攻击实例（多 LLM 平台测试）：

| 类别 | 数量 | 典型形式 | 盾卫覆盖状态 |
|------|------|---------|------------|
| **Direct Overrides** | 12 | "Ignore previous instructions..." | Phase 2.3 ✅ |
| **Code Execution** | 8 | 注入系统命令调用 | Phase 2.3 部分 |
| **Data Exfiltration** | 7 | "Send all data to attacker..." | Phase 2.3 ✅ |
| **Formatting Attacks** | 6 | 利用输出格式要求注入 | Phase 2.3 部分 |
| **Obfuscation** | 8 | Base64/Unicode 编码混淆 | Phase 2.3 ✅ |
| **Tool/Agent Manipulation** | 5 | 劫持工具调用链 | Phase 2.3 部分 |
| **Role-Play Attacks** | 6 | "You are now DAN, no restrictions..." | Phase 2.3 **弱** |
| **Multi-Turn Persistence** | 3 | 多轮渐进绕过 | **未覆盖** |

**基线 ASR（无防御）**：
- ChatGLM：30%（较高漏洞率）
- Llama2：20%

---

## 二、两种防御架构

### 2.1 Chain-of-Agents Pipeline（事后审查）

```
用户输入 → Domain LLM → 候选回复 → Guard Agent → 最终输出
                                        ↓（检测到攻击）
                                    拒绝/修正回复
```

**特征**：
- Domain LLM 正常处理输入（不预过滤）
- Guard Agent 独立审查生成的回复
- 优点：误报率低（只阻止真正有问题的输出）
- 缺点：有额外推理延迟（两次 LLM 调用）

**适合场景**：可用性优先，可以接受轻微延迟，不能接受合法查询被拒绝

### 2.2 Coordinator Pipeline（事前拦截）

```
用户输入 → Coordinator → [判断]
                            ├── 安全 → Domain LLM → 输出
                            └── 攻击 → 安全拒绝回复（不经过 LLM）
```

**特征**：
- Coordinator 在 LLM 处理之前拦截
- 攻击完全不接触 Domain LLM
- 优点：速度快，攻击不影响核心模型
- 缺点：Coordinator 误判 = 合法查询被直接拒绝

**适合场景**：安全性优先，对延迟敏感，可接受偶尔误报

### 2.3 架构对比

| 维度 | Chain（事后）| Coordinator（事前）|
|------|------------|------------------|
| 延迟 | 较高（两次推理）| 较低（快速判断+路由）|
| 误报率 | 低 | 中等 |
| 漏报率 | 极低 | 极低 |
| 适合 | 生产可用性敏感 | 高安全要求 |

---

## 三、实验结果

### 整体防御效果

| 平台 | 无防御 ASR | Chain ASR | Coordinator ASR |
|------|-----------|-----------|----------------|
| ChatGLM | 30% | **0%** | **0%** |
| Llama2 | 20% | **0%** | **0%** |

**三种架构（v1 Taxonomy / Phase2 Chain / Phase2 Coordinator）均达到 0% ASR**，尽管基线漏洞率不同。

### 按攻击类别的防御效果

特别有效（高基线 ASR 被完全消除）：
- **Tool/Agent Manipulation**：委托/工具劫持攻击
- **Role-Play Coercion**：角色扮演绕过
- **Reconnaissance/Environment Leakage**：环境探测
- **Data Exfiltration**：数据外泄指令

---

## 四、系统架构全貌

```
用户请求
    ↓
API Gateway（流量管控）
    ↓
Event Orchestrator（事件路由）
    ↓
Coordinator（安全决策）
   ├── 标记为攻击 → Safe Response（预定义）
   └── 标记为安全 → Domain LLM → Guard Agent（可选）→ 输出
```

Guard Agent 的职责：
1. **Policy Compliance Check**：回复是否符合系统策略
2. **Attack Indicator Detection**：回复中是否含有攻击后遗留的指示
3. **Format Compliance**：是否在预期输出格式内

---

## 五、关键洞察与评价

### 洞察 1：LLM-as-defender 的语义优势

规则过滤对编码混淆（Base64/Unicode）和语义伪装攻击效果差，因为攻击的恶意性在语义层。Guard Agent / Coordinator 本身就是 LLM，可以：
- 理解编码 → 解码后再判断
- 理解语义 → 识别"换个说法但目的相同"的攻击
- 理解上下文 → 识别 Role-play 绕过

### 洞察 2：防御的组合性

Chain + Coordinator 不互斥，高安全场景可以两者同时部署：
- Coordinator 在前：大多数攻击在这里被拦截
- Guard Agent 在后：绕过 Coordinator 的攻击在这里被审查

这正是 SoK (2602.10453) 建议的"defense-in-depth"策略。

### 洞察 3：论文的局限

- **测试集相对小**：55 种攻击，对优化型攻击（GCG 类）覆盖不足
- **单轮为主**：Multi-turn persistence 攻击只有 3 个实例，测试不充分
- **Context-dependent 盲区**：与 SoK 的发现一致，本文框架对 context-dependent 任务的处理也未专门讨论
- **模型较老**：ChatGLM 和 Llama2 不代表当前 frontier 模型的漏洞分布

### 我的评价

★★★★☆——方法论清晰，实验完整，但测试集规模和攻击多样性有限。最大价值是提供了**两种架构的对比框架**，而非具体数字。对盾卫的参考价值：**Coordinator 模式** = 盾卫可以在工具调用前先做 semantic scan，而不是只在返回内容时扫描。

---

## 六、对盾卫 Phase 2.4 的启示

### 当前盾卫架构类比

```
盾卫现状（Phase 2.3）：
    工具执行 → 工具返回内容 → scan_tool_return() → [正则+JSON+Base64] → 风险判断
               （只在事后审查工具返回内容，类似 Chain 架构的 Guard 角色）

Phase 2.4 目标：
    工具调用意图 → [Coordinator: semantic intent check] → 执行工具
                                    ↓
                              + 工具返回内容 → [Guard: scan_tool_return v2] → 输出
```

**Phase 2.4 应同时实现**：
1. **Pre-execution Coordinator**：在工具调用前，检查调用意图是否与当前 Agent 目标一致
2. **Semantic Guard 升级**：scan_tool_return 加语义层（LLM 判断），不只是正则

---

## 与 Vault 其他笔记的关联

- **[[AI/Safety/PI-Landscape-SoK-Prompt-Injection-Taxonomy-Defense|PI-Landscape SoK]]**：提供理论分类框架，本文是实践实现
- **[[AI/Safety/OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]]**：本文 Data Exfiltration 类别的真实攻击案例
- **[[AI/Safety/Agent 安全威胁全景 2026-02|Agent 安全威胁全景 2026-02]]**：宏观视角补充

---

## 推荐阅读

- **原论文**：[arXiv:2509.14285v4](https://arxiv.org/abs/2509.14285)（Wichita State + Marshall + Aizu，2025-09）
- **理论框架**：[[AI/Safety/PI-Landscape-SoK-Prompt-Injection-Taxonomy-Defense|PI-Landscape SoK]]（SoK，更全面的分类）
- **实攻击分析**：[[AI/Safety/OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]]（真实 context-dependent 攻击）
- **延伸**：IsolateGPT（Wu et al.）、ProgENT（Shi et al.）——执行层防御

---

## 启发思考

**So What**：Chain-of-Agents 防御模式的核心价值不是"多一个 LLM 就更安全"，而是**职责分离**——Domain LLM 专注任务，Guard Agent 专注安全审计。两者的优化目标不同，不相互干扰。这和人类系统中的"四眼原则"一致。

**盾卫未来方向**：把 memory_guard.py 的角色从"单一扫描器"升级为"安全 Agent Pipeline"——pre-execution Coordinator + post-execution Guard + multi-turn history analyzer（对抗渐进式攻击）。
