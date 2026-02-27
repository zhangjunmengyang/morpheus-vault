---
brief: "HybridFlow——verl 的核心架构设计；将 Actor（生成采样）和 Critic/RM（奖励计算）解耦到不同计算节点，支持异构资源调度；解决 RLHF 四模型同时运行的显存和通信瓶颈，实现训练吞吐量 3-4x 提升。"
title: "HybridFlow"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# HybridFlow

> verl 的核心编排机制：用 Ray 实现 **单控制器 (single-controller)** 的混合并行编排，让 RL 训练中的多个异构 worker 高效协作。

文档：https://verl.readthedocs.io/en/latest/hybrid_flow.html

## 为什么需要 HybridFlow

RL 训练（PPO/GRPO）涉及多个不同角色的模型：

```
Actor (Rollout) → 生成 response
Reward Model    → 给 response 打分
Critic          → 估计 value（PPO 需要）
Actor (Train)   → 用 RL loss 更新参数
Reference       → 计算 KL divergence
```

这些模型的计算特性完全不同：
- Rollout 是 **推理**（autoregressive, memory-bound）
- Train 是 **训练**（compute-bound, 需要梯度）
- Reward 可能是另一个模型或一个函数

传统方案的问题：

```
方案 1: 所有模型放一个进程
→ 显存爆炸，GPU 利用率低

方案 2: 每个角色独立的 GPU 组
→ GPU 数量翻倍，大部分时间在等

方案 3: verl 的 HybridFlow
→ 同一组 GPU 在不同阶段扮演不同角色
```

## HybridFlow 的核心设计

### 资源复用 (Resource Multiplexing)

```python
# 关键洞察：rollout 和 train 不会同时发生
# 同一组 GPU 可以：
# 阶段 1: 加载 actor weights → 做 rollout
# 阶段 2: 切换到 training mode → 做梯度更新
# 阶段 3: 加载 reward model → 计算 reward

# 省了什么？不需要为每个角色分配独立的 GPU
```

### Single-Controller 架构

```python
# 一个主进程（driver）编排所有操作
# 底层 worker 通过 Ray Actor 实现

@ray.remote
class ActorRolloutWorker:
    """同一个 worker 既做 rollout 又做 training"""
    
    def generate(self, prompts):
        """Rollout 阶段：推理模式"""
        with torch.no_grad():
            return self.model.generate(prompts)
    
    def train_step(self, batch):
        """Training 阶段：训练模式"""
        loss = self.compute_ppo_loss(batch)
        loss.backward()
        self.optimizer.step()

# Driver 编排流程
class PPODriver:
    def step(self):
        # 1. Rollout
        responses = ray.get([w.generate.remote(prompts) 
                           for w in self.workers])
        
        # 2. Reward
        rewards = ray.get([w.compute_reward.remote(responses) 
                         for w in self.reward_workers])
        
        # 3. Train
        ray.get([w.train_step.remote(batch) 
                for w in self.workers])
```

### 与 Multi-Controller 的对比

```
Multi-Controller (如 DeepSpeed-Chat):
- 每个角色独立的 torchrun 进程组
- 通过文件/Redis 同步状态
- 调试困难，状态管理复杂

Single-Controller (verl HybridFlow):
- 一个 Python 进程控制全局流程
- Ray Actor 管理远端 worker
- 像写单机代码一样写分布式逻辑
```

## 数据流

一个 PPO step 的完整数据流：

```
prompts (from dataset)
    │
    ▼
┌─────────────────┐
│  Actor Rollout   │ → responses, log_probs
│  (推理模式)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Reward Model    │ → rewards
│  (推理模式)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Critic Forward  │ → values
│  (推理模式)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  GAE Compute     │ → advantages, returns
│  (CPU)           │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Actor Update    │ → updated θ
│  (训练模式)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Critic Update   │ → updated φ
│  (训练模式)       │
└─────────────────┘
```

## 权重同步

Rollout 用的是当前 actor weights，但 training 会更新 weights。verl 的处理：

```python
# 每个 step 结束后同步权重
# 方案 1: 直接在 GPU 间做 broadcast（同节点）
# 方案 2: 通过 shared memory / object store（跨节点）

# 如果 rollout 和 train 在同一组 GPU 上（HybridFlow 的默认模式）
# 根本不需要同步 — 就是同一份权重
```

## 实践注意事项

1. **显存管理是关键**：rollout 阶段需要 KV cache，train 阶段需要梯度和 optimizer states，两者显存需求不同。verl 在阶段切换时会释放和重新分配显存
2. **micro-batch size 要分开配**：rollout 和 train 的 batch size 通常不同
3. **Placement Group 配置**：确保 TP 组在同节点

## See Also

- [[AI/3-LLM/Infra/Ray|Ray]] — HybridFlow 的底层编排框架
- [[AI/3-LLM/Infra/Megatron-LM|Megatron-LM]] — 训练后端之一
- [[AI/3-LLM/Frameworks/verl/训练后端|训练后端]] — Megatron vs FSDP
- [[AI/3-LLM/Frameworks/verl/硬件资源预估|硬件资源预估]] — 资源规划
- [[AI/3-LLM/Frameworks/verl/性能调优|性能调优]] — 训练性能优化
