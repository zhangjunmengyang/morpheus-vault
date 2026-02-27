# verl-GRPO 实战 + R1 复现全流程（MA-RLHF Batch F）

**来源**：MA-RLHF `r1/` 目录 — verl 框架实战代码  
**难度**：★★★★★（面试项目级，分布式 GRPO 工程必备）  
**关联**：[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操]] | lc9-Ray分布式RL训练实操 | [[Projects/MA-RLHF/xtrain/xtrain-03b-ZeRO-手撕实操]]

---

## 一、R1 复现项目总览

**项目定位**：用开源工具栈复现 DeepSeek-R1-like 的完整训练 pipeline，不是学术研究，是**工程师的完整实战路线**。

```
R1 复现全流程（4阶段）：

Phase 1: Continue Pretraining
  └─ 基础模型 → 领域数据继续预训练（扩充推理语料）

Phase 2: SFT（Supervised Fine-Tuning）
  ├─ sft.py      — 全参 SFT（Accelerate + DeepSpeed ZeRO-1）
  ├─ sft_qlora.py — QLoRA 低资源 SFT（bitsandbytes 4bit）
  └─ 目标：给模型打上 COT 推理的格式基础

Phase 3: DPO（Direct Preference Optimization）
  └─ dpo.py — 用 TRL 的 DPO Trainer 做偏好对齐

Phase 4: GRPO（Group Relative Policy Optimization）
  └─ verl/main_ppo.py — 分布式 GRPO（verl + Ray + vLLM）
     ├─ gsm8k.py    — 数学推理基准（GSM8K 数据处理）
     └─ custom_data.py — 自定义数学数据集（X-R1-750，\boxed{} 匹配）

评测：
  └─ benchmark.py + generate.py — 效果验证
```

**技术栈选型**：
```
SFT:  Transformers + Accelerate + DeepSpeed ZeRO-1
DPO:  TRL DPOTrainer
GRPO: verl==0.7.0 + Ray + vLLM==0.11.0
分布式: 8 GPU 单机 / 多机（standard launcher）
```

---

## 二、verl 框架核心架构

### 2.1 三层抽象

```
verl 框架的三层设计：

Layer 1: Ray 基础（分布式执行引擎）
  └─ ray.remote actor → 分布式进程
  └─ RayResourcePool → GPU 资源分配
  └─ RayWorkerGroup → Worker 集群管理

Layer 2: verl Worker 抽象
  └─ Worker(base) → 单 GPU 工作单元（知道自己的 rank）
  └─ MegatronWorker → 支持 TP/PP 的 Megatron Worker
  └─ @register(Dispatch.XX) → 数据分发模式声明

Layer 3: PPO/GRPO Trainer
  └─ RayPPOTrainer → 协调 Actor/Critic/RM/Ref 的 Ray 版 Trainer
  └─ Role enum → ActorRollout / Critic / RewardModel / RefPolicy
  └─ ResourcePoolManager → GPU 资源分配管理
```

### 2.2 Worker 角色分工

```python
# verl 的 4 种 Role（对应 RLHF 四模型）
Role.ActorRollout  → Actor 模型（rollout + 训练）
Role.Critic        → Critic/Value 模型
Role.RewardModel   → Reward Model（可选，rule-based 时不需要）
Role.RefPolicy     → 参考策略（frozen SFT）
```

**关键设计**：verl 允许同一 GPU 上部署多个角色（通过 ResourcePool 共享）。GRPO 场景下：
- Actor + Ref 共享同一个 GPU 资源池
- 用 rule-based reward（gsm8k 精确匹配）完全去掉 RewardModel Worker
- 不需要 Critic（GRPO 无 Critic）

---

## 三、数据处理：GRPO 标准数据格式

### 3.1 GSM8K（数学推理基准）

```python
def process_fn(example, idx):
    question = example['question'] + " Let's think step by step and output the final answer after '####'."
    solution = extract_solution(answer_raw)  # 正则：#### (\-?[0-9\.]+)

    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {
            "style": "rule",            # ← 关键：rule-based，不需要 RM 模型
            "ground_truth": solution    # ← GRPO 的 verifiable reward
        },
        "extra_info": {"split": split, "index": idx, ...}
    }
```

**关键**：`reward_model.style = "rule"` → verl 在训练时用 `ground_truth` 做精确匹配打分，完全绕过 RM 模型。这是 GRPO/RLVR 的核心工程实现。

### 3.2 自定义数据集（X-R1-750，MATH 格式）

```python
# MATH 数据集用 \boxed{} 作为答案格式
def extract_solution(text):
    match = re.search(r'\\boxed{((?:[^{}]|\{[^{}]*\})*)}', text)
    return match.group(1) if match else None

# 关键：data_source 设置为 "lighteval/MATH"，触发 LaTeX 数学式规则匹配
data = {
    "data_source": "lighteval/MATH",  # ← 不是实际来源，而是触发 reward 逻辑的 key
    "prompt": [{"role": "user", "content": question}],
    "reward_model": {"style": "rule", "ground_truth": solution},
}
```

**工程技巧**：`data_source` 字段不只是来源标识，还控制 verl 用哪套 reward 函数。`"lighteval/MATH"` 触发 LaTeX 匹配（支持 `\frac{1}{2}` 等），`"openai/gsm8k"` 触发数字精确匹配。

**输出格式**：保存为 `.parquet`（Arrow 格式），比 JSON 快 5-10x 读取速度，是大规模 RL 训练的标准。

---

## 四、verl main_ppo.py：分布式训练入口

### 4.1 启动流程

```python
# 用 Hydra 管理配置（替代 argparse）
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)

def run_ppo(config):
    # Step 1: 初始化 Ray 集群
    ray.init(**ray_init_kwargs)

    # Step 2: 创建 TaskRunner（Ray Remote Actor，避免在 head 节点运行主任务）
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
```

**Hydra 配置的优势**：
- YAML 文件定义默认配置，命令行可覆盖任何字段
- `OmegaConf.resolve()` 支持 `${var}` 插值
- 天然支持多配置组合（不同实验只改 config 文件）

### 4.2 Worker 注册模式（核心设计）

```python
class TaskRunner:
    def run(self, config):
        # 1. 注册 Actor Rollout Worker（支持 fsdp/fsdp2/megatron 三种策略）
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        # 2. 注册 Critic Worker（GRPO 可关闭）
        self.add_critic_worker(config)

        # 3. 注册 Reward Model Worker（rule-based 时不需要）
        self.add_reward_model_worker(config)

        # 4. 注册 Ref Policy Worker（KL penalty 必须）
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 5. 资源池管理
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # 6. 创建数据集
        train_dataset = create_rl_dataset(config.data.train_files, ...)
        val_dataset   = create_rl_dataset(config.data.val_files, ...)

        # 7. 启动 RayPPOTrainer
        trainer = RayPPOTrainer(
            config=config,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            ...
        )
        trainer.init_workers()
        trainer.fit()
```

### 4.3 Actor 策略选择

```python
# config.actor_rollout_ref.actor.strategy 三选一：
if strategy in {"fsdp", "fsdp2"}:
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    # FSDP2 = PyTorch 原生分布式，支持 ZeRO-3 级别分片
elif strategy == "megatron":
    from verl.workers.megatron_workers import ActorRolloutRefWorker
    # Megatron = TP+PP，适合超大模型（70B+）
```

**选择原则**：
- `fsdp/fsdp2`：< 70B 模型，单机多卡或小规模多机，配置简单
- `megatron`：70B+ 模型，需要 TP+PP，需要 Megatron-LM 环境

### 4.4 Rollout 模式

```python
if config.actor_rollout_ref.rollout.mode == "async":
    actor_rollout_cls = AsyncActorRolloutRefWorker
    # async rollout: vLLM 做生成，与 training 解耦
    # 生成和训练可以 pipeline 并行，提升 GPU 利用率
```

**async rollout 的关键**：verl 在 GRPO 中用 vLLM 做 rollout（高效推理），用 DeepSpeed/FSDP 做 training（高效训练）。二者在不同进程上运行，通过 Ray 对象传递 rollout 结果。这正是 `slime` / verl 这类异步 RL 框架的核心创新——解决"生成是 inference bottleneck"的问题。

---

## 五、DeepSpeed ZeRO-1 配置（SFT 阶段）

```yaml
# deepspeed_zero1.yaml（Accelerate 格式）
compute_environment: LOCAL_MACHINE
deepspeed_config:
  zero_stage: 1          # ZeRO-1：只分片 optimizer state
  zero3_init_flag: false # ZeRO-3 初始化关闭（ZeRO-1 不需要）
  gradient_accumulation_steps: 1
distributed_type: DEEPSPEED
mixed_precision: bf16    # BF16 混合精度
num_processes: 8         # 8 GPU 单机
```

**ZeRO-1 的选择理由**：
- SFT 阶段模型不大（7B-13B），ZeRO-1 切 optimizer state 已够用
- ZeRO-3 切参数本体，通信开销大，SFT 不值得
- 真正的分布式 GRPO 训练才用 ZeRO-3 或 Megatron

---

## 六、Ray 分布式原语（verl tutorial 核心）

### 6.1 基础：RayResourcePool + RayWorkerGroup

```python
# 声明 4 GPU 资源池
resource_pool = RayResourcePool([4], use_gpu=True)

# Worker 类（每个实例对应一个 GPU 进程）
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self):
        self.value = torch.zeros(1, device="cuda") + self.rank  # self.rank 自动注入
    
    def add(self, x):
        self.value += x
        return self.value.cpu()

# 创建 WorkerGroup（在 4 GPU 上各起一个进程）
worker_group = RayWorkerGroup(resource_pool, RayClassWithInitArgs(GPUAccumulator))

# 并行执行，每个 rank 独立处理
print(worker_group.execute_all_sync("add", x=[1, 1, 1, 1]))  # rank 0→1, 1→2, 2→3, 3→4
```

### 6.2 数据分发模式（Dispatch 装饰器）

```python
# 三种分发模式：
@register(Dispatch.ONE_TO_ALL)    # 同一份数据广播给所有 worker（如模型初始化参数）
def init_model(self, config): ...

@register(Dispatch.ALL_TO_ALL)    # 每个 worker 接收不同的数据切片（标准数据并行）
def train_step(self, batch): ...

@register(Dispatch.ALL_TO_ALL, Execute.RANK_ZERO)  # 只在 rank 0 上执行（如 checkpoint 保存）
def save_model(self): ...
```

**工程意义**：Dispatch 装饰器让 verl 的 Worker 代码可以像单机代码一样写，分发/收集逻辑由框架自动处理。Driver（main 进程）看到的是统一接口。

### 6.3 ResourcePool 合并（共址部署）

```python
# Actor 和 Ref 在同一批 GPU 上（共址，节省显存通过切换而非常驻）
resource_pool_actor = RayResourcePool([4], name_prefix="actor")
resource_pool_ref   = RayResourcePool([4], name_prefix="ref")
resource_pool_merged = merge_resource_pool(resource_pool_actor, resource_pool_ref)
# → 8 GPU 分两批，actor 用前4，ref 用后4
```

**verl 的资源调度策略**：
- Actor + Rollout 同 GPU（避免 rollout 数据传输）
- Critic 可以与 Actor 共址，也可独立
- RM 可以独立资源池（`enable_resource_pool=True`）

---

## 七、Megatron TP Worker（高级）

```python
@ray.remote
class MLPLayerWorker(MegatronWorker):
    def __init__(self):
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=4,   # TP=4：4 GPU 并行一层
            pipeline_model_parallel_size=1,
            ...
        )
    
    @register(Dispatch.MEGATRON_COMPUTE)  # 特殊模式：DP维度切片输入，TP透明
    def run_layer(self, x): ...

# Driver 侧：数据按 DP 切分传入，TP 透明
output = layer_worker_group.run_layer([x])  # x: [seq, batch, hidden]
```

**`MEGATRON_COMPUTE` 分发模式**：输入按 DP 维度切分 → 广播给同 DP group 的所有 TP 进程 → TP 并行计算 → 只收集 tp=0 + last_pp 的输出。对 driver 来说，看不到 TP 的存在。

---

## 八、完整 R1 复现命令（面试可讲）

```bash
# ===== 环境配置 =====
conda create -n llm python=3.11
pip install verl==0.7.0 torch==2.8.0 vllm==0.11.0
# flash attention：从 github releases 下载对应版本 .whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-linux_x86_64.whl

# ===== Phase 1: 数据准备 =====
python gsm8k.py --local_save_dir ~/data/gsm8k
# 或自定义数据集：
python custom_data.py --local_save_dir ~/data/xr1-750

# ===== Phase 2: SFT =====
accelerate launch --config_file deepspeed_zero1.yaml \
    sft.py \
    --model_name_or_path Qwen/Qwen3-7B \
    --output_dir ./output/qwen3_sft

# ===== Phase 3: （可选）DPO =====
python dpo.py --model_path ./output/qwen3_sft ...

# ===== Phase 4: GRPO =====
python -m verl.trainer.main_ppo \
    algorithm=grpo \
    actor_rollout_ref.model.path=./output/qwen3_sft \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.mode=async \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1

# ===== Phase 5: 评测 =====
python generate.py --model ./output/qwen3_grpo
python benchmark.py --model ./output/qwen3_grpo
```

---

## 九、关键工程决策与权衡

### GRPO vs PPO 在工程上的差异

| 工程维度 | PPO | GRPO |
|----------|-----|------|
| Worker 数 | 4（Actor+Ref+RM+Critic）| 3 或 2（rule-based 去掉 RM）|
| 显存占用 | 高（4模型同时驻留）| 低（Actor+Ref 可轮换）|
| Critic 训练 | 必须，增加约 30% 计算 | 无，替换为采样统计 |
| Reward 计算 | 模型推理（RM forward）| 规则匹配（纯 CPU）|
| vLLM 集成 | 可选 | 强烈推荐（async rollout）|

### async rollout 的 2x 加速原理

```
传统同步 rollout：
  [generate] → [train] → [generate] → [train] ...
                GPU 在 generate 时 training 等待，反之亦然

verl async rollout：
  [generate_k] ─────────────────────┐
               [train_k-1] ──────── ┤ pipeline 并行
                             [gen_k+1]
  Actor（vLLM）和 Training（FSDP）在不同进程上，通过 Ray 对象传递数据
```

**本质**：把"推理 GPU"和"训练 GPU"解耦，允许二者 pipeline 执行，消除等待。verl 的 `AsyncActorRolloutRefWorker` 就是这个设计的实现。

### 数据格式选择：Parquet vs JSON

```
JSON:    行存储，解析开销大，不支持列式查询
Parquet: 列存储，Arrow 格式，5-10x 读取速度
         + 支持 filter pushdown（只读需要的列）
         + verl/HuggingFace Datasets 的标准输入格式
```

---

## 十、面试必备问题

**Q1：R1 复现的完整流程是什么？**  
A：Continue Pretraining（扩充推理语料）→ SFT（格式基础+CoT结构）→ 可选 DPO → GRPO（verifiable reward RL）→ Benchmark 评测。核心是 GRPO 阶段——用 rule-based reward 代替 RM，大幅降低训练复杂度。

**Q2：verl 框架的核心设计是什么？**  
A：Ray 多 Worker 异步协调 + Dispatch 装饰器抽象数据分发 + ResourcePool 管理 GPU 共享。关键创新是 async rollout：vLLM 做生成，FSDP/Megatron 做训练，两者 pipeline 并行，GPU 利用率从 ~50% 提升到 ~80%+。

**Q3：为什么用 rule-based reward 而不训 Reward Model？**  
A：①减少 4 模型到 3/2 模型，显存降低；②避免 RM 泛化性问题（RM 可能被 hacked）；③数学推理天然可验证（答案对错），不需要 RM 打分。代价是只适用于有明确答案的任务（math/code/logic）。

**Q4：verl 的 GRPO 具体是如何实现分布式的？**  
A：Actor 用 FSDP 做模型分片，vLLM 做 async rollout，Ray 连接两部分。每轮：①vLLM 并发生成 G 个 rollout per prompt；②rule-based reward 打分（纯 CPU）；③GRPO advantage 计算；④FSDP actor 做梯度更新；⑤同步 vLLM 权重（权重更新传播）。

**Q5：continue pretraining 的作用是什么？**  
A：基础模型可能缺乏足够的推理语料。Continue pretraining 在 SFT 之前用 CoT 数据（如数学解题过程）做无监督预训练，相当于给模型"打底"——让它在 SFT/GRPO 前就已经见过推理链的模式，降低冷启动难度。

---

## 十一、知识连接

- **算法基础**：[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操]] → verl 的分布式 GRPO 就是这个算法的工业实现
- **分布式基础**：[[Projects/MA-RLHF/xtrain/xtrain-03b-ZeRO-手撕实操]] → verl 用的 FSDP 与 ZeRO-3 原理相同
- **推理加速**：lc10-vLLM-PageKVCache-手撕实操 → verl async rollout 依赖 vLLM
- **Ray 基础**：[[Projects/MA-RLHF/lc9/lc9-ray-01-Ray-分布式RL训练实操]] → verl 的 RayWorkerGroup 就是这个的封装
- **生产框架对比**：verl vs slime(智谱) vs OpenRLHF vs TRL：verl 最模块化，slime 最异步，TRL 最易用

---

*入库时间：2026-02-26*  
*来源：MA-RLHF r1/ 目录（verl tutorial.ipynb + main_ppo.py + gsm8k.py + custom_data.py）*  
*状态：Batch F ✅（grpo.py/sft.py 等文件为空骨架，主要内容在 verl/ 子目录）*
