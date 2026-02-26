---
title: "Ray 分布式 RL 训练系统实操"
brief: "基于Ray实现Generator-Coordinator-Trainer三角架构的分布式RL训练系统：训推分离设计、Remote Actor通信、GRPO reward计算、经验回放，是理解OpenRLHF/verl等框架核心设计的教学实现，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, ray, distributed-rl, rlhf, grpo, training-infra]
related:
  - "[[AI/3-LLM/Infra/Ray-推理系统实操|Ray-推理系统实操]]"
  - "[[AI/3-LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[AI/3-LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操|ZeRO-手撕实操]]"
---

# Ray 分布式 RL 训练系统手撕实操

> 来源：MA-RLHF (<https://github.com/dhcode-cpp/MA-RLHF>)  
> 路径：`lecture/lc9_training/ray-train/`  
> 入库日期：2026-02-25  
> 状态：教学级代码（AI 生成，调试中），用于理解 OpenRLHF / verl 等框架的核心设计

---

## 一、系统架构概览

本项目实现了一个 **Generator-Coordinator-Trainer 三角架构**的分布式 RL 训练系统，核心思路是**训推分离**（Training-Inference Separation）：

```
                    ┌──────────────────┐
                    │   Coordinator    │
                    │  (调度 & 监控)    │
                    └──┬───────────┬───┘
           启动/监控    │           │   启动/监控
              ┌────────┘           └────────┐
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Generator Actor │──────▶   │  Trainer Actor   │
    │  (rollout 推理)  │  发送    │  (策略更新)       │
    │  GPU:1 (vLLM)   │  样本    │  GPU:0 (PyTorch) │
    └──────────────────┘          └──────────────────┘
              ▲                             │
              │         参数同步             │
              └─────────────────────────────┘
                   (pull-based 拉取)
```

**数据流**：
1. **Coordinator** 启动训练循环和生成循环
2. **Generator** 用本地模型副本做 rollout 推理，生成 `(prompt, response)` 对
3. Generator 将样本 **push** 到 Trainer 的训练队列
4. **Trainer** 从队列采样，执行 PPO/SFT 训练步骤
5. Generator 定期从 Trainer **pull** 最新参数，更新本地模型
6. 循环往复，直到达到预设运行时间

**关键设计决策**：
- 多个 Generator 共享一个 Trainer（N:1 架构）
- Generator 和 Trainer 各占 0.5 GPU（`@ray.remote(num_gpus=0.5)`）
- 参数同步是 **pull-based**：Generator 主动从 Trainer 拉取，而非 Trainer 推送
- 训练队列有容量上限（deque maxlen），防止内存溢出

---

## 二、核心组件实现

### 2.1 配置（Config）

`config.py` 当前为空文件，但代码中引用了三个配置字典：

| 配置 | 用途 | 关键字段 |
|------|------|---------|
| `GENERATION_CONFIG` | 生成参数 | `max_tokens`, `temperature`, `generation_batch_size`, `update_frequency` |
| `TRAIN_CONFIG` | 训练参数 | `learning_rate`, `batch_size`, `queue_size`, `grad_clip`, `clean_threshold`, `clean_keep` |
| `SYSTEM_CONFIG` | 系统参数 | `num_generators`, `train_interval`, `generate_interval`, `runtime_seconds`, `status_interval` |

> 💡 实际框架（如 verl）通常用 Hydra/OmegaConf 管理配置，这里简化为 dict。

### 2.2 模型定义（Models）

`models.py` 当前为空，但代码中使用了 `SharedLanguageModel` 类，预期接口：

```python
class SharedLanguageModel(nn.Module):
    def forward(self, input_ids) -> logits       # 训练前向
    def generate(self, input_ids, max_length, temperature) -> token_ids  # 推理生成
    def state_dict() / load_state_dict()          # 参数同步接口
```

> 在真实系统中，这里会接入 HuggingFace model + vLLM engine，模型的 `forward` 和 `generate` 分别用于训练和推理。

### 2.3 数据工具（DataUtils）

两个核心类：

**`DataProcessor`** — 数据处理器：

```python
@dataclass
class TrainingSample:
    prompt: str
    generated: str
    input_ids: torch.Tensor    # 输入 token ids
    labels: torch.Tensor       # 目标 token ids（右移一位）
    generator_id: str          # 来自哪个 Generator
    timestamp: float

class DataProcessor:
    def simulate_tokenize(self, text, max_length=50) -> List[int]
    def create_training_samples(self, prompts, generated_texts, generator_id) -> List[TrainingSample]
    def batch_samples(self, samples, batch_size)  # yield 分批
```

关键逻辑——样本构建是经典 LM 方式：
```python
combined = prompt + " " + generated
token_ids = tokenize(combined)
input_ids = token_ids[:-1]   # 输入
labels    = token_ids[1:]    # 目标（右移一位）
```

**`PromptManager`** — 提示词管理器：从文件加载 prompts，找不到文件时回退到 20 条硬编码示例。

### 2.4 Generator Actor（rollout 生成）

Generator 是 Ray Actor，负责用当前策略生成 rollout 样本：

```python
@ray.remote(num_gpus=0.5)
class GeneratorActor:
    def __init__(self, generator_id, trainer_actor, device_id=1):
        self.local_model = SharedLanguageModel().to(device)
        self.trainer_actor = trainer_actor  # 持有 Trainer 的引用
        self._update_from_trainer()         # 初始化时拉取参数
```

**核心方法 `generate_and_send`**：

```python
def generate_and_send(self, prompts, batch_size):
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # 1. 用本地模型推理生成
        generated_texts = self.generate_with_vllm(batch_prompts)

        # 2. 构造训练样本
        batch_samples = self.prepare_training_data(batch_prompts, generated_texts)

        # 3. 推送到 Trainer 队列（同步 RPC）
        success = ray.get(
            self.trainer_actor.receive_generated_data.remote(batch_samples)
        )

        # 4. 定期从 Trainer 拉取最新参数
        if (i // batch_size) % self.config["update_frequency"] == 0:
            self._update_from_trainer()
```

**参数同步（pull-based）**：

```python
def _update_from_trainer(self):
    params_info = ray.get(self.trainer_actor.get_current_params.remote())
    self.local_model.load_state_dict(params_info["params"])
    self.params_version = params_info["version"]
```

> 🔑 参数通过 `state_dict()` 序列化为 CPU tensor 传输，Generator 收到后 `load_state_dict` + `.to(device)`。这是最直观的同步方式，但对大模型来说带宽开销巨大——实际框架会用 NCCL broadcast 或共享内存。

### 2.5 Trainer Actor（策略更新）

Trainer 是唯一的训练节点，维护模型权重的 ground truth：

```python
@ray.remote(num_gpus=0.5)
class TrainerActor:
    def __init__(self):
        self.model = SharedLanguageModel().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.training_queue = deque(maxlen=config["queue_size"])  # 有界队列
        self.params_version = 0   # 版本号，每次训练步 +1
```

**数据接收**——Generator push 过来的样本进入训练队列：

```python
def receive_generated_data(self, batch_data):
    for data in batch_data:
        sample = TrainingSample(**data)
        self.training_queue.append(sample)   # deque 自动淘汰最旧样本
    return True
```

**训练步骤** `train_step`——核心训练循环：

```python
def train_step(self, batch_size):
    if len(self.training_queue) < batch_size:
        return {"status": "waiting"}   # 队列不够，等待

    # 随机采样（非顺序消费！）
    indices = np.random.choice(len(self.training_queue), batch_size, replace=False)
    batch_samples = [self.training_queue[i] for i in indices]

    # padding 到统一长度
    max_len = max(t.size(0) for t in input_tensors)
    # ... pad with 0 for inputs, -100 for labels

    # 标准训练步骤
    logits = self.model(input_tensor)
    loss = CrossEntropyLoss(logits.view(-1, V), labels.view(-1))
    loss.backward()
    clip_grad_norm_(self.model.parameters(), grad_clip)
    self.optimizer.step()

    self.params_version += 1  # 版本号递增
```

**队列清理**——每 100 步触发一次：

```python
def _clean_queue(self):
    if len(self.training_queue) > clean_threshold:
        keep_indices = np.random.choice(current_size, clean_keep, replace=False)
        self.training_queue = deque([...keep...], maxlen=queue_size)
```

**参数导出**——供 Generator 拉取：

```python
def get_current_params(self):
    params_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    return {"params": params_cpu, "version": self.params_version}
```

### 2.6 Coordinator（调度器）

Coordinator 是系统的"大脑"，负责创建 Actor 并启动循环：

```python
@ray.remote
class SystemCoordinator:
    def __init__(self, num_generators):
        # 1. 创建唯一的 Trainer
        self.trainer = TrainerActor.remote()

        # 2. 创建 N 个 Generator，每个持有 Trainer 引用
        self.generators = []
        for i in range(num_generators):
            gen = GeneratorActor.remote(
                generator_id=f"Generator-{i+1}",
                trainer_actor=self.trainer,
                device_id=i % 2   # 轮流分配 GPU
            )
            self.generators.append(gen)
```

**训练循环**——启动一个 Ray remote function 作为后台 worker：

```python
def start_training_loop(self, interval):
    @ray.remote
    def training_worker(trainer, interval):
        while True:
            result = ray.get(trainer.train_step.remote())
            if result["status"] == "success" and result["step"] % 10 == 0:
                print(f"训练步骤 {result['step']}: loss={result['loss']:.4f}")
            time.sleep(interval)

    self.training_future = training_worker.remote(self.trainer, interval)
```

**生成循环**——每个 Generator 各启动一个后台 worker，独立轮询 prompts：

```python
def start_generation_loop(self, prompts_file, interval):
    prompts = PromptManager.load_prompts(prompts_file)

    @ray.remote
    def generation_worker(generator, prompts, interval, worker_id):
        idx = 0
        while True:
            batch_prompts = prompts[idx:idx+2]
            result = ray.get(generator.generate_and_send.remote(batch_prompts))
            idx = (idx + 2) % len(prompts)
            time.sleep(interval + random_jitter)

    for i, gen in enumerate(self.generators):
        self.generation_futures.append(
            generation_worker.remote(gen, prompts, interval, i)
        )
```

**状态监控**——`get_system_status` 聚合所有 Actor 状态；`stop_system` 优雅停机并打印最终统计。

### 2.7 主流程（Main）

`main.py` 是入口，流程非常清晰：

```python
def main():
    # 1. 初始化 Ray（声明 2 GPU）
    ray.init(num_gpus=2)

    # 2. 创建 Coordinator（会自动创建 Trainer + Generators）
    coordinator = SystemCoordinator.remote(num_generators=N)

    # 3. 启动训练循环
    ray.get(coordinator.start_training_loop.remote(interval=train_interval))

    # 4. 启动生成循环
    ray.get(coordinator.start_generation_loop.remote(prompts_file="prompts.txt"))

    # 5. 运行指定时间，定期打印状态
    for i in range(runtime_seconds):
        if i % status_interval == 0:
            status = ray.get(coordinator.get_system_status.remote())
            print(f"步骤={status['trainer']['training_step']}, "
                  f"队列={status['trainer']['queue_size']}, "
                  f"loss={status['trainer']['avg_loss']:.4f}")
        time.sleep(1)

    # 6. 停止 & 清理
    ray.get(coordinator.stop_system.remote())
    ray.shutdown()
```

> 整个系统的启动顺序：Ray init → Coordinator → Trainer → Generators → 训练循环 → 生成循环 → 监控循环 → 停机。

---

## 三、与 verl / OpenRLHF 框架对比

| 维度 | 本项目（ray-train 教学版） | verl (APRIL) | OpenRLHF |
|------|--------------------------|-------------|----------|
| **架构** | Generator-Coordinator-Trainer 三角 | Worker Group（Actor/Critic/Ref/Reward） | Ray-based Actor 多角色 |
| **训推分离** | ✅ Generator 和 Trainer 分离在不同 GPU | ✅ 通过 WorkerGroup 实现，支持 colocate/分离 | ✅ vLLM 推理 + 训练分离 |
| **参数同步** | Pull-based：Generator 主动拉取 `state_dict` | NCCL broadcast / 共享内存 | NCCL broadcast |
| **生成引擎** | 模拟 vLLM（实际是 PyTorch generate） | 真正集成 vLLM | 真正集成 vLLM |
| **RL 算法** | 纯 SFT loss（CrossEntropy） | GRPO / PPO / REINFORCE++ | PPO / DPO / GRPO 等 |
| **训练队列** | 有界 deque + 随机采样 | 同步批处理（无队列） | 同步批处理 |
| **异步性** | 半异步：Generator 和 Trainer 独立循环 | 默认同步，APRIL 支持异步 | 同步为主 |
| **规模** | 教学级（1 Trainer + N Generator） | 生产级（支持数千 GPU） | 生产级 |

**核心设计思路的共性**：

1. **Generator/Trainer 解耦是异步 RL 的基石**：本项目和 verl APRIL 都实现了推理和训练的独立运行。Generator 不需要等 Trainer 训完才能继续生成，反之亦然。
2. **参数版本管理**：本项目用 `params_version` 整数递增；verl 用更精细的版本追踪来处理 off-policy 修正。
3. **数据流方向一致**：都是 Generator → (样本) → Trainer → (参数) → Generator 的闭环。

**关键差异**：

- 本项目的 `state_dict` 传输方式在大模型场景下不可行（7B 模型约 14GB 参数），verl/OpenRLHF 用 NCCL 集合通信或 Ray 对象存储的零拷贝机制
- 本项目没有 Reward Model / Reference Model，缺少 PPO 中的 advantage 估计
- 本项目的训练队列 + 随机采样模式更像 off-policy replay buffer，而非标准 on-policy PPO

---

## 四、关键洞察

### 为什么 Generator 和 Trainer 要分离？

1. **硬件利用率**：推理是 memory-bound（大 batch、长序列），训练是 compute-bound（反向传播）。分离后可以针对各自特点分配资源——推理节点用 vLLM + PagedAttention 最大化吞吐，训练节点用 FSDP/DeepSpeed 最大化计算效率。

2. **异步流水线**：如果 Generator 和 Trainer 串行执行（先生成一批 → 再训练一步 → 再生成），GPU 利用率只有 ~50%。分离后两者可以并行运行，形成流水线。

3. **弹性扩展**：可以灵活调整 Generator 和 Trainer 的数量比例。如果生成是瓶颈（长序列解码），多加 Generator；如果训练是瓶颈（大模型反向传播），多加训练节点。

### 异步 RL 的稳定性问题怎么处理？

本项目暴露了一个重要问题：**Generator 用的模型参数可能落后于 Trainer 的当前版本**（off-policy gap）。

```
Generator 用 v3 参数生成 → 样本进入队列
Trainer 已经更新到 v7 → 用 v7 的模型去训练 v3 生成的样本
```

这就是经典的 **stale policy 问题**。处理方案：

| 方案 | 原理 | 代表框架 |
|------|------|---------|
| **重要性采样修正** | 用 `π_new(a|s) / π_old(a|s)` 加权，修正分布偏移 | PPO 的 clip ratio |
| **丢弃过旧样本** | 设置最大 staleness 阈值，超过就丢 | verl APRIL |
| **频繁同步** | 减小 `update_frequency`，每生成一个 batch 就同步 | 本项目的简单方案 |
| **完全同步** | 每个 iteration 先生成、后训练，强制 on-policy | verl 默认模式 |

本项目的 `deque(maxlen=...)` 有界队列是一种隐式的"丢弃旧样本"策略——当队列满时，最早的样本被自动淘汰。但更严格的做法应该检查 `params_version` 差距。

### 本项目的教学价值

这个项目虽然是教学代码（模拟 tokenizer、空 config/models），但完整展示了分布式 RL 训练的**骨架设计**：

1. ✅ Ray Actor 模型：每个角色是独立的 Actor，通过 remote call 通信
2. ✅ 训推分离：Generator 和 Trainer 在不同 GPU 上独立运行
3. ✅ 参数同步协议：pull-based，带版本号
4. ✅ 有界训练队列：防止内存溢出 + 隐式淘汰旧样本
5. ✅ Coordinator 模式：中央调度器管理生命周期

理解了这个骨架，再去看 verl 的 `WorkerGroup`、OpenRLHF 的 `ActorModel` / `CriticModel`，就能快速定位每个组件对应的角色。

---

## 附：文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `config.py` | 0 | 配置字典（待实现） |
| `models.py` | 0 | 模型定义（待实现） |
| `data_utils.py` | ~100 | 数据处理：tokenize、样本构建、prompt 管理 |
| `generator_actor.py` | ~160 | Generator Actor：rollout 生成 + 数据推送 + 参数拉取 |
| `trainer_actor.py` | ~170 | Trainer Actor：数据接收 + 训练 + 参数导出 |
| `coordinator.py` | ~180 | Coordinator：创建 Actor + 启动循环 + 监控 |
| `main.py` | ~80 | 入口：Ray init + 启动 + 监控 + 停机 |
