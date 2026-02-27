# verl + GRPO 实战指南

> 基于 MA-RLHF 仓库 `r1/verl/` 目录整理，使用 verl 框架在消费级 GPU (3090/4090) 上复现 R1-Zero 风格的 GRPO 训练。
> 来源：`tutorial.ipynb`, `main_ppo.py`, `gsm8k.py`, `custom_data.py`, `run_qwen3-0_6b.sh`, `run_qwen3_8b.sh`

---

## 1. verl 架构概述

### 1.1 核心设计：HybridFlow

verl（Volcano Engine RL）是字节跳动开源的 RLHF 框架，核心架构称为 **HybridFlow**：

```
┌────────────────────────────────────────────────┐
│                  Driver (单进程)                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│   │ DataLoad │  │ Reward   │  │ Trainer Loop │ │
│   └──────────┘  └──────────┘  └──────────────┘ │
│         │              │             │          │
│    ┌────▼────────────▼─────────────▼────┐      │
│    │      RayWorkerGroup 调度层          │      │
│    └────┬────────────┬─────────────┬────┘      │
│         │            │             │            │
│  ┌──────▼──┐  ┌──────▼──┐  ┌──────▼──┐        │
│  │ Actor   │  │ Rollout │  │   Ref   │        │
│  │(Megatron│  │ (sglang/│  │ Policy  │        │
│  │ /FSDP)  │  │  vLLM)  │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘        │
│    训练引擎      推理引擎      参考策略          │
└────────────────────────────────────────────────┘
```

**关键设计理念**：

| 特性 | 说明 |
|------|------|
| **训练-推理解耦** | Actor 用 Megatron/FSDP 做分布式训练，Rollout 用 sglang/vLLM 做高效推理，各用最适合的引擎 |
| **GPU 资源共享** | 同一组 GPU 上的 Actor 和 Rollout 通过 Ray 资源池切换，不需要独占两套卡 |
| **Single Controller** | Driver 进程统一调度所有 Worker，通过 `@register` 装饰器简化数据分发 |
| **异步 Rollout** | `mode=async` 支持训练和采样流水线化，提高 GPU 利用率 |

### 1.2 Ray 基础设施

verl 在 Ray 之上构建了三层抽象：

1. **RayResourcePool**：管理 GPU 资源，支持 `merge_resource_pool` 合并多组资源
2. **RayWorkerGroup**：将多个 Worker 封装为一组，提供 `execute_all_sync` 统一调度
3. **Dispatch 装饰器**：`ONE_TO_ALL`（广播）、`ALL_TO_ALL`（一一对应）、`MEGATRON_COMPUTE`（TP/PP 感知分发）

```python
# 示例：ONE_TO_ALL 模式，一个输入广播到所有 worker
@register(Dispatch.ONE_TO_ALL)
def add(self, x):
    self.value = self.value + x
    return self.value.cpu()

# 调用时自动广播
worker_group.add(x=10)  # x=10 发送给所有 worker
```

### 1.3 NVMegatronRayWorkerGroup

对于需要 Tensor Parallelism 的大模型，verl 提供了 `NVMegatronRayWorkerGroup`：
- 内部初始化 Megatron 的 `parallel_state`
- `MEGATRON_COMPUTE` 模式：按 DP 维度分发数据，TP/PP 内部自动通信
- 对 Driver 透明——用户只需按 DP 维度传数据

---

## 2. 环境配置

### 2.1 核心依赖

```bash
# 创建环境
conda create -n verl python=3.12
conda activate verl

# 核心包
pip install verl==0.7.0 torch==2.8.0 vllm==0.11.0

# Flash Attention（需匹配 CUDA 和 torch 版本）
# 从 https://github.com/Dao-AILab/flash-attention/releases 下载 whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### 2.2 关键版本矩阵

| 组件 | 版本 | 说明 |
|------|------|------|
| verl | 0.7.0 | RLHF 训练框架 |
| torch | 2.8.0 | PyTorch |
| vllm | 0.11.0 | 推理引擎（Rollout 后端之一） |
| sglang | 0.5.3 | 推理引擎（本实验实际使用） |
| flash-attn | 2.8.1 | Flash Attention 2 |
| transformers | 4.57.6 | HuggingFace Transformers |
| megatron-core | 0.13.1 | Megatron 分布式训练 |
| ray | 2.53.0 | 分布式调度 |
| accelerate | 1.12.0 | HuggingFace Accelerate |
| peft | 0.18.1 | LoRA 适配器 |
| datasets | 4.5.0 | 数据集处理 |

### 2.3 硬件要求

- **Qwen3-0.6B**：3090 (24GB) × 2，约 2 小时/epoch
- **Qwen3-8B**：4090 (48GB) × 4，约 2 小时/epoch，需 optimizer offload

---

## 3. 数据准备与 Reward Function

### 3.1 GSM8K 数据预处理（`gsm8k.py`）

```python
# === gsm8k.py 完整注解 ===

# 从 GSM8K 的标准答案格式中提取最终数字答案
# GSM8K 格式：多行推理过程 + "#### 最终答案"
def extract_solution(solution_str):
    # 正则匹配 "#### " 后面的数字（支持负数、小数、逗号分隔）
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    # 去掉 "#### " 前缀和逗号（如 "1,000" → "1000"）
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

# 数据集加载：支持本地路径或 HuggingFace Hub
dataset = datasets.load_dataset("openai/gsm8k", "main")

# 关键：添加 instruction_following 后缀引导模型输出格式
instruction_following = 'Let\'s think step by step and output the final answer after "####".'

# 数据转换函数：将原始数据转为 verl 的标准格式
def process_fn(example, idx):
    question = example["question"] + " " + instruction_following
    solution = extract_solution(example["answer"])

    data = {
        "data_source": "openai/gsm8k",      # 数据来源标识
        "prompt": [{                          # Chat 格式的 prompt
            "role": "user",
            "content": question,
        }],
        "ability": "math",                    # 能力标签
        "reward_model": {
            "style": "rule",                  # 规则奖励（非模型奖励）
            "ground_truth": solution           # 标准答案
        },
        "extra_info": { ... }                 # 元信息
    }
    return data

# 输出为 parquet 格式（verl 训练要求）
train_dataset.to_parquet("train.parquet")
test_dataset.to_parquet("test.parquet")
```

### 3.2 自定义数据：X-R1-750（`custom_data.py`）

实际训练使用的是 `xiaodongguaAIGC/X-R1-750` 数据集（750 道数学题），关键差异：

```python
# 答案格式不同：用 \boxed{} 包裹（LaTeX 数学格式）
def extract_solution(text):
    match = re.search(r'\\boxed{((?:[^{}]|\{[^{}]*\})*)}', text)
    return match.group(1) if match else None

# 重要技巧：data_source 设为 "lighteval/MATH" 而非实际来源
# 这样在训练时才能触发 verl 内置的 LaTeX 数学规则匹配 reward
"data_source": "lighteval/MATH",  # ← 触发 latex 规则匹配
```

### 3.3 Reward Function 设计要点

verl 的 reward 机制是 **基于规则的**（rule-based），不需要额外训练 Reward Model：

1. **数据侧**：在 `reward_model` 字段指定 `"style": "rule"` 和 `"ground_truth"`
2. **框架侧**：verl 根据 `data_source` 自动选择匹配策略
   - `openai/gsm8k` → 匹配 `####` 后的数字
   - `lighteval/MATH` → 匹配 `\boxed{}` 中的 LaTeX 表达式
3. **奖励值**：答案正确 → +1，答案错误 → 0（二元奖励）
4. **GRPO 归一化**：同一 prompt 的 n 个采样结果做 advantage 归一化

---

## 4. GRPO 训练配置

### 4.1 `main_ppo.py` 核心流程

`main_ppo.py` 是 verl 的标准入口（虽然文件名含 "ppo"，但通过配置可切换为 GRPO）：

```python
@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    run_ppo(config)  # 初始化 Ray → 创建 TaskRunner → 启动训练

class TaskRunner:
    def run(self, config):
        # 1. 创建 Actor + Rollout Worker（共享 GPU）
        self.add_actor_rollout_worker(config)
        # 2. 创建 Critic Worker（GRPO 不需要 Critic，但代码保留了接口）
        self.add_critic_worker(config)
        # 3. 加载 Reward Manager（rule-based）
        reward_fn = load_reward_manager(config, tokenizer)
        # 4. 初始化 RayPPOTrainer 并开始训练
        trainer = RayPPOTrainer(...)
        trainer.init_workers()
        trainer.fit()
```

### 4.2 关键配置参数

| 参数 | 含义 | Qwen3-0.6B 值 | Qwen3-8B 值 |
|------|------|----------------|-------------|
| `algorithm.adv_estimator` | 优势估计器 | `grpo` | `grpo` |
| `data.train_batch_size` | 每步训练的 prompt 数 | 4 | 8 |
| `data.max_prompt_length` | prompt 最大长度 | 512 | 256 |
| `data.max_response_length` | response 最大长度 | 1024 | 1024 |
| `actor.optim.lr` | Actor 学习率 | 1e-6 | 1e-6 |
| `actor.ppo_mini_batch_size` | mini-batch 大小 | 2 | 8 |
| `actor.ppo_micro_batch_size_per_gpu` | 每 GPU micro-batch | 2 | 2 |
| `actor.use_kl_loss` | 是否使用 KL 散度 loss | True | True |
| `actor.kl_loss_coef` | KL loss 系数 | 0.001 | 0.001 |
| `actor.kl_loss_type` | KL loss 类型 | `low_var_kl` | `low_var_kl` |
| `actor.entropy_coeff` | 熵正则系数 | 0 | 0 |
| `rollout.n` | 每个 prompt 采样数 | 8 | 8 |
| `rollout.name` | 推理引擎 | `sglang` | `sglang` |
| `rollout.gpu_memory_utilization` | 推理显存占比 | 0.5 | 0.5 |
| `rollout.mode` | Rollout 模式 | `async` | `async` |
| `rollout.tensor_model_parallel_size` | 推理 TP 大小 | 2 | 4 |
| `actor.strategy` | 训练策略 | `megatron` | `megatron` |
| `actor.megatron.tensor_model_parallel_size` | 训练 TP 大小 | 2 | 4 |
| `algorithm.use_kl_in_reward` | reward 中是否加 KL 惩罚 | False | False |
| `trainer.total_epochs` | 总 epoch 数 | 2 | 1 |

### 4.3 GRPO vs PPO 的关键差异

GRPO 通过 `algorithm.adv_estimator=grpo` 切换：

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic | 需要独立 Value Network | **不需要 Critic** |
| Advantage | GAE 估计 | 同一 prompt 的 n 个采样做组内归一化 |
| 采样 | 每 prompt 1 个 response | 每 prompt **n 个** response |
| 显存 | Critic 占额外显存 | 节省 Critic 显存，但 n 倍采样增加推理开销 |
| 适用场景 | 通用 RLHF | **数学推理等有明确规则奖励的场景** |

---

## 5. 启动命令完整解析

### 5.1 Qwen3-0.6B 配置（`run_qwen3-0_6b.sh`）

```bash
# 框架：verl + sglang + megatron
# 硬件：3090(24GB) × 2，约 2 小时 / epoch
# 结果：GSM8K test set 57.23%

set -x  # 打印执行的每条命令（调试用）

USE_MBRIDGE=FALSE  # 是否使用 MBridge（Megatron 桥接，实验性功能）

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \        # 使用 Megatron 训练策略的配置模板

    # === 算法配置 ===
    algorithm.adv_estimator=grpo \                      # 使用 GRPO 而非 PPO
    algorithm.use_kl_in_reward=False \                  # reward 中不加 KL 惩罚（KL 在 loss 中控制）

    # === 数据配置 ===
    data.train_files='./xr1-750/train.parquet' \        # 训练数据（750 道数学题）
    data.val_files='./xr1-750/test.parquet' \           # 验证数据
    data.train_batch_size=4 \                           # 每步 4 个 prompt
    data.max_prompt_length=512 \                        # prompt 最长 512 token
    data.max_response_length=1024 \                     # response 最长 1024 token
    data.filter_overlong_prompts=True \                 # 过滤超长 prompt
    data.truncation='error' \                           # 超长时报错（而非截断）

    # === Actor（训练引擎）配置 ===
    actor_rollout_ref.model.path="/data/Qwen3-0.6B-Base" \  # Base 模型路径
    actor_rollout_ref.actor.optim.lr=1e-6 \                  # 学习率（RL 阶段用小 lr）
    actor_rollout_ref.model.use_remove_padding=True \        # 去除 padding 加速
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \          # mini-batch = 2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \ # micro-batch = 2（梯度累积次数 = mini/micro = 1）
    actor_rollout_ref.actor.use_kl_loss=True \               # 在 loss 中加 KL 散度约束
    actor_rollout_ref.actor.kl_loss_coef=0.001 \             # KL 系数（小值，轻微约束）
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \        # 低方差 KL 估计
    actor_rollout_ref.actor.entropy_coeff=0 \                # 不使用熵正则
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \  # 梯度检查点节省显存

    # === Actor Megatron 并行配置 ===
    actor_rollout_ref.actor.strategy="megatron" \            # 使用 Megatron 做训练
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \  # TP=2（2 张卡做张量并行）
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \  # PP=1（不做流水线并行）
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \  # CP=1（不做上下文并行）
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \  # MBridge 开关

    # === Rollout（推理引擎）配置 ===
    actor_rollout_ref.rollout.name=sglang \                  # 使用 sglang 做推理
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \ # 推理也用 TP=2
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \   # 推理引擎只用 50% 显存（与训练共享 GPU）
    actor_rollout_ref.rollout.n=8 \                          # 每个 prompt 采样 8 个 response
    actor_rollout_ref.rollout.mode=async \                   # 异步模式：训练和推理流水线化
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \    # 验证时也做采样
    actor_rollout_ref.rollout.load_format=auto \             # 自动检测模型格式
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \  # log_prob 计算的 micro-batch

    # === Ref Policy 配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \  # 参考策略的 micro-batch

    # === Trainer 配置 ===
    trainer.critic_warmup=0 \                               # 无 Critic 预热（GRPO 无 Critic）
    trainer.logger='["console","wandb"]' \                  # 日志：控制台 + WandB
    trainer.project_name='verl_grpo_xr1750' \               # WandB 项目名
    trainer.experiment_name='qwen3_0dot6B_RL_Zero' \        # 实验名
    trainer.n_gpus_per_node=2 \                             # 每节点 2 GPU
    trainer.nnodes=1 \                                      # 单节点
    trainer.save_freq=50 \                                  # 每 50 步保存 checkpoint
    trainer.test_freq=20 \                                  # 每 20 步验证
    trainer.total_epochs=2                                  # 训练 2 个 epoch
```

### 5.2 Qwen3-8B 配置差异（`run_qwen3_8b.sh`）

```bash
# 硬件：4090(48GB) × 4，结果 91.87%
NGPUS=4
NBATCHS=8           # 大批量：8 个 prompt
NMICROBATCHS=2       # micro-batch=2 → 梯度累积 4 次
NSAMPLES=8           # 每 prompt 8 个采样

# 额外的显存优化
actor_rollout_ref.actor.megatron.optimizer_offload=True   # 优化器 offload 到 CPU
actor_rollout_ref.actor.megatron.param_offload=False      # 参数不 offload
actor_rollout_ref.actor.megatron.grad_offload=False       # 梯度不 offload
```

**关键对比**：8B 模型需要 4 卡 TP + optimizer offload 才能跑起来，但效果显著提升（57% → 92%）。

---

## 6. 调试技巧

### 6.1 常见报错与解决方案

| 报错 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA OOM` | 显存不足 | 1. 降低 `gpu_memory_utilization`（如 0.4）<br>2. 开启 `gradient_checkpointing`<br>3. 开启 `optimizer_offload` |
| `Ray actor died` | Worker 进程崩溃 | 检查 Ray dashboard 日志；可能是 NCCL 超时 |
| `data.truncation='error'` 报错 | prompt 超过 `max_prompt_length` | 改为 `'left'` 或增大 `max_prompt_length` |
| sglang 启动失败 | 端口冲突或版本不匹配 | 确保 sglang==0.5.3，检查端口占用 |
| reward 全为 0 | `data_source` 不匹配 reward 规则 | 确认 `data_source` 字段与 verl 内置规则一致 |
| Megatron TP 报错 | hidden_size 不能被 TP 整除 | TP 大小必须整除模型的 hidden_size 和 num_heads |

### 6.2 调试建议

1. **先跑小规模**：用 `trainer.total_epochs=1` + 小数据集验证流程
2. **清理输出**：脚本开头 `rm -rf ./outputs` 避免旧 checkpoint 干扰
3. **WandB 监控**：观察 reward 曲线是否在上升、KL 是否发散
4. **验证频率**：`test_freq=10~20`，频繁验证捕捉过拟合
5. **显存规划**：`gpu_memory_utilization=0.5` 意味着训练和推理各占一半显存

### 6.3 性能调优

- **增大 batch_size**：更稳定的 advantage 估计
- **调 n_samples**：n=8 是 GRPO 的常用值，太小方差大，太大推理开销高
- **KL 系数**：0.001 是经验值，过大会限制探索，过小会导致策略崩溃
- **学习率**：1e-6 是 RL 阶段的安全选择，不宜太大

---

## 7. 面试考点

### Q1: verl 和 OpenRLHF 的架构差异？

**参考答案**：

| 维度 | verl | OpenRLHF |
|------|------|----------|
| 调度框架 | Ray（Actor 模型） | Ray + vLLM |
| 训练引擎 | Megatron / FSDP（可选） | DeepSpeed |
| 推理引擎 | sglang / vLLM（解耦设计） | vLLM（内嵌） |
| GPU 共享 | 训练和推理**共享同一组 GPU**，通过 RayResourcePool 切换 | 训练和推理通常**分开部署**在不同 GPU 上 |
| 并行策略 | 原生支持 Megatron 的 TP/PP/CP | 主要依赖 DeepSpeed ZeRO |
| 核心抽象 | RayWorkerGroup + Dispatch 装饰器 | 传统 Trainer 模式 |

verl 的核心优势是 **HybridFlow 的 GPU 共享设计**——在消费级多卡环境中，一组 GPU 既做训练又做推理，无需额外推理集群。

### Q2: Reward Function 怎么设计？规则奖励 vs 模型奖励的取舍？

**参考答案**：

**规则奖励**（Rule-based）：
- 适用于有明确正确答案的任务（数学、代码、知识问答）
- 优点：无噪声、无额外训练成本、100% 可解释
- 缺点：只能判断最终答案对错，无法评估推理过程质量
- 实现：从模型输出提取答案（正则匹配 `####` 或 `\boxed{}`），与 ground_truth 比较

**模型奖励**（Model-based）：
- 适用于开放式任务（对话质量、创意写作）
- 优点：能评估过程和风格
- 缺点：有 reward hacking 风险、需要训练 RM

**GRPO 场景**：优先用规则奖励。因为 GRPO 依赖同一 prompt 多次采样的 advantage 归一化，规则奖励的无噪声特性让训练更稳定。

### Q3: verl 中 `rollout.n=8` 的含义及对训练的影响？

**参考答案**：

`n=8` 表示对每个 prompt 采样 8 个 response，这是 GRPO 算法的核心参数：

1. **Advantage 计算**：GRPO 不用 Critic，而是对同一 prompt 的 8 个 response 计算 reward，然后做组内归一化得到 advantage
2. **公式**：$A_i = \frac{r_i - \text{mean}(r_1,...,r_8)}{\text{std}(r_1,...,r_8)}$
3. **影响**：
   - n 太小（如 2）：advantage 估计方差大，训练不稳定
   - n 太大（如 32）：推理开销成倍增加，但 advantage 更准
   - n=8 是经验最优平衡点
4. **显存影响**：推理时需要同时生成 `batch_size × n` 个序列，是显存瓶颈之一
5. **与 PPO 对比**：PPO 每个 prompt 只需 1 个 response（Critic 负责估计 baseline），GRPO 用统计方法替代了 Critic

---

## 附录

### A. 训练结果对比

| 模型 | 硬件 | 数据 | Epoch | GSM8K Test |
|------|------|------|-------|------------|
| Qwen3-0.6B-Base | 3090×2 | X-R1-750 | 2 | 57.23% |
| Qwen3-8B-Base | 4090×4 | X-R1-750 | 1 | 91.87% |

### B. 文件路径速查

| 文件 | 作用 |
|------|------|
| `gsm8k.py` | GSM8K 数据预处理（`####` 格式） |
| `custom_data.py` | X-R1-750 数据预处理（`\boxed{}` 格式） |
| `main_ppo.py` | verl 训练入口（TaskRunner 编排） |
| `run_qwen3-0_6b.sh` | 0.6B 模型启动脚本 |
| `run_qwen3_8b.sh` | 8B 模型启动脚本 |
| `tutorial.ipynb` | Ray + verl 基础教程 |
| `environment.yml` | conda 环境完整依赖 |
